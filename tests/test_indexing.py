"""Incremental ingestion against a real temporary Chroma collection, offline."""

import json
import pytest
from chromadb import PersistentClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from market_research.indexing import identify_chunks
from market_research.rag import ingest_knowledge_base


class CountingEmbeddings(Embeddings):
    def __init__(self):
        self.texts = []

    def embed_documents(self, texts):
        self.texts.extend(texts)
        return [[1.0, float(len(text)), 0.5] for text in texts]

    def embed_query(self, text):
        return [1.0, float(len(text)), 0.5]


@pytest.fixture
def corpus(tmp_path):
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    index = tmp_path / "index"

    def write(name, text="Apple sells hardware and services.", metadata="ticker: AAPL\n"):
        path = knowledge / name
        path.write_text(f"---\ndocument_type: company_overview\n{metadata}---\n\n{text}\n", encoding="utf-8")
        return path

    def ingest(embeddings=None, **kwargs):
        return ingest_knowledge_base(
            knowledge, embeddings=embeddings or CountingEmbeddings(),
            persist_directory=index, collection_name="incremental_test", **kwargs,
        )

    def records():
        return PersistentClient(path=str(index)).get_collection(
            "incremental_test", embedding_function=None,
        ).get(include=["documents", "metadatas", "embeddings"])

    return knowledge, write, ingest, records


def test_identities_ignore_metadata_and_offsets_but_preserve_provenance():
    def chunk(source="apple.pdf", page=1, offset=0, ticker="AAPL"):
        return Document(page_content="Identical text", metadata={
            "source": source, "page_number": page, "start_index": offset, "ticker": ticker,
        })

    original = identify_chunks([chunk()])[0]
    changed = identify_chunks([chunk(offset=100, ticker="MSFT")])[0]
    copied = identify_chunks([chunk(source="copy.pdf")])[0]
    next_page = identify_chunks([chunk(page=2)])[0]
    repeated = identify_chunks([chunk(), chunk(offset=100)])
    assert original.id == changed.id
    assert original.metadata["document_id"] == changed.metadata["document_id"]
    assert len({original.id, copied.id, next_page.id}) == 3
    assert len({item.metadata["content_hash"] for item in [original, copied, next_page]}) == 1
    assert repeated[0].id != repeated[1].id


def test_add_change_delete_and_unchanged_runs(corpus):
    knowledge, write, ingest, records = corpus
    write("apple.md")
    first_model = CountingEmbeddings()
    first = ingest(first_model)
    original_id = records()["ids"][0]
    assert first.changes.embedded_texts == 1

    unchanged_model = CountingEmbeddings()
    unchanged = ingest(unchanged_model)
    assert not unchanged.changes.changed
    assert unchanged_model.texts == []
    assert unchanged.changes.unchanged_chunks == 1

    microsoft = write("microsoft.md", "Microsoft sells cloud services.", "ticker: MSFT\n")
    model = CountingEmbeddings()
    added = ingest(model)
    assert not added.rebuilt
    assert added.changes.added_chunks == 1
    assert model.texts == ["Microsoft sells cloud services."]
    assert original_id in records()["ids"]

    write("microsoft.md", "Microsoft operates Azure.", "ticker: MSFT\n")
    model = CountingEmbeddings()
    edited = ingest(model)
    assert edited.changes.added_chunks == edited.changes.deleted_chunks == 1
    assert model.texts == ["Microsoft operates Azure."]
    assert original_id in records()["ids"]
    assert "Microsoft sells cloud services." not in records()["documents"]

    microsoft.unlink()
    model = CountingEmbeddings()
    removed = ingest(model)
    assert removed.changes.deleted_chunks == 1
    assert model.texts == []
    (knowledge / "apple.md").unlink()
    empty = ingest(model)
    assert empty.chunk_count == 0
    assert empty.changes.deleted_chunks == 1
    assert records()["ids"] == []


def test_metadata_updates_remove_old_fields_without_embedding(corpus):
    _, write, ingest, records = corpus
    write("apple.md", metadata="ticker: AAPL\ncompany: Apple Inc.\n")
    ingest()
    before = records()
    write("apple.md", metadata="ticker: MSFT\nlast_reviewed: 2026-11-23\n")
    model = CountingEmbeddings()
    result = ingest(model)
    after = records()
    assert model.texts == []
    assert result.changes.metadata_updates == 1
    assert result.changes.added_chunks == result.changes.deleted_chunks == 0
    assert before["ids"] == after["ids"]
    assert before["embeddings"].tolist() == after["embeddings"].tolist()
    assert after["metadatas"][0]["ticker"] == "MSFT"
    assert "company" not in after["metadatas"][0]


def test_duplicate_sources_keep_citations_and_reuse_vectors_even_after_rename(corpus):
    knowledge, write, ingest, records = corpus
    write("apple.md")
    write("copy.md")
    model = CountingEmbeddings()
    result = ingest(model)
    assert len(model.texts) == 1
    assert result.chunk_count == 2
    assert result.changes.duplicate_chunks == result.changes.duplicate_sources == 1
    assert len(set(records()["ids"])) == 2
    assert {meta["source"] for meta in records()["metadatas"]} == {"apple.md", "copy.md"}

    (knowledge / "apple.md").unlink()
    (knowledge / "copy.md").rename(knowledge / "renamed.md")
    model = CountingEmbeddings()
    renamed = ingest(model)
    assert model.texts == []
    assert renamed.changes.added_chunks == 1
    assert renamed.changes.deleted_chunks == 2
    assert records()["metadatas"][0]["source"] == "renamed.md"


def test_rechunking_reuses_exact_text_despite_moved_offsets(corpus):
    _, write, ingest, records = corpus
    paragraphs = [letter * 35 for letter in "ABCD"]
    write("apple.md", "\n\n".join(paragraphs))
    ingest(chunk_size=40, chunk_overlap=0)
    before = dict(zip(records()["documents"], records()["ids"]))
    write("apple.md", "\n\n".join(["NEW" * 10, *paragraphs]))
    model = CountingEmbeddings()
    result = ingest(model, chunk_size=40, chunk_overlap=0)
    after = dict(zip(records()["documents"], records()["ids"]))
    assert model.texts == ["NEW" * 10]
    assert all(after[text] == record_id for text, record_id in before.items())
    assert result.changes.metadata_updates == 4  # Only start_index moved.


@pytest.mark.parametrize("invalidation", ["force", "model", "legacy_manifest"])
def test_incompatible_or_forced_index_rebuilds(corpus, invalidation):
    _, write, ingest, _ = corpus
    write("apple.md")
    first = ingest()
    model = CountingEmbeddings()
    kwargs = {}
    if invalidation == "force":
        kwargs["force_rebuild"] = True
    elif invalidation == "model":
        class DifferentEmbeddings(CountingEmbeddings):
            pass
        model = DifferentEmbeddings()
    else:
        first.manifest_path.write_text('{"index_version": 1}', encoding="utf-8")
    result = ingest(model, **kwargs)
    assert result.rebuilt
    assert len(model.texts) == 1
    assert json.loads(result.manifest_path.read_text())["index_version"] == 2
    assert not ingest(model).rebuilt


def test_embedding_failure_preserves_index_and_manifest(corpus):
    _, write, ingest, records = corpus
    write("apple.md")
    first = ingest()
    before = records()
    manifest = first.manifest_path.read_bytes()
    write("apple.md", "Changed report text.")

    class FailingEmbeddings(CountingEmbeddings):
        def embed_documents(self, texts):
            raise RuntimeError("Embedding service unavailable")

    with pytest.raises(RuntimeError, match="Embedding service unavailable"):
        ingest(FailingEmbeddings())
    assert records()["ids"] == before["ids"]
    assert records()["documents"] == before["documents"]
    assert first.manifest_path.read_bytes() == manifest
    assert ingest().changes.embedded_texts == 1


def test_invalid_source_does_not_delete_existing_records(corpus):
    _, write, ingest, records = corpus
    write("apple.md")
    ingest()
    before = records()["ids"]
    write("apple.md", metadata="fiscal_year: not-a-year\n")
    with pytest.raises(ValueError):
        ingest()
    assert records()["ids"] == before


def test_configured_model_change_invalidates_vectors(corpus):
    _, write, ingest, _ = corpus
    write("apple.md")
    first_model = CountingEmbeddings()
    first_model.model_name = "model-one"
    ingest(first_model)
    second_model = CountingEmbeddings()
    second_model.model_name = "model-two"
    result = ingest(second_model)
    assert result.rebuilt
    assert result.changes.embedded_texts == 1
