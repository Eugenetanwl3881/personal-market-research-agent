from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)
from pydantic import ValidationError

from market_research.rag import (
    KnowledgeDocumentMetadata,
    ScoreThresholdRetriever,
    build_knowledge_retriever,
    ingest_knowledge_base,
    load_knowledge_documents,
    search_knowledge,
    split_documents,
)


def test_hybrid_recovers_segment_answer_outside_vector_candidates():
    """Exact filing terms recover evidence even when vectors prefer certificates."""
    metadata = {
        "ticker_scope": "AAPL", "company": "Apple Inc.",
        "document_type": "annual_report", "fiscal_year": 2025, "document_version": "1",
        "source": "apple_2025.pdf", "source_format": "pdf", "page_number": 50,
    }
    answer = Document(
        id="answer", page_content="The reportable segments are Americas, Europe, "
        "Greater China, Japan and Rest of Asia Pacific.",
        metadata={**metadata, "start_index": 0},
    )
    certificate = Document(
        id="certificate", page_content="Apple Inc. 2025 Form 10-K certification.",
        metadata={**metadata, "page_number": 80, "start_index": 0},
    )

    class VectorStore:
        def similarity_search_with_relevance_scores(self, query, **kwargs):
            assert query == "reportable segments"
            assert kwargs["k"] == 20
            return [(certificate, 0.99)]

    retriever = ScoreThresholdRetriever(
        vector_store=VectorStore(), keyword_documents=[certificate, answer],
        tickers=["AAPL"], k=1,
    )
    result = retriever.invoke(
        "According to Apple's 2025 Form 10-K, what are its reportable segments?"
    )

    assert result[0].metadata["page_number"] == 50
    assert "Greater China, Japan and Rest of Asia Pacific" in result[0].page_content
    assert "keyword_score" in result[0].metadata
    assert "relevance_score" not in result[0].metadata  # No invented cosine score.


def test_hybrid_expands_split_pdf_passage_without_crossing_pages():
    metadata = {"source": "report.pdf", "source_format": "pdf",
                "ticker_scope": "AAPL", "page_number": 50}
    chunks = [
        Document(id="before", page_content="Americas, Europe, Greater China, Japan and Rest of Asia Pacific.",
                 metadata={**metadata, "start_index": 0}),
        Document(id="hit", page_content="Each of the reportable segments operates separately.",
                 metadata={**metadata, "start_index": 70}),
        Document(id="other-page", page_content="Unrelated page contents.",
                 metadata={**metadata, "page_number": 51, "start_index": 0}),
    ]

    class VectorStore:
        def similarity_search_with_relevance_scores(self, query, **kwargs):
            return [(chunks[1], 0.9)]

    result = ScoreThresholdRetriever(
        vector_store=VectorStore(), keyword_documents=chunks, tickers=["AAPL"], k=1,
    ).invoke("What are the reportable segments?")

    assert "Americas, Europe, Greater China, Japan and Rest of Asia Pacific" in result[0].page_content
    assert "Unrelated page contents" not in result[0].page_content
    assert result[0].metadata["page_number"] == 50
    assert result[0].metadata["matched_start_index"] == 70


@pytest.mark.parametrize("hit_at_start", [True, False])
def test_pdf_page_edge_hit_recovers_list_two_chunks_away_within_budget(hit_at_start):
    """An edge hit should not lose a list just because one neighbor is absent."""
    metadata = {"source": "report.pdf", "source_format": "pdf",
                "ticker_scope": "MSFT", "page_number": 7}
    hit_text = "Information about reportable segments."
    list_text = "The divisions are Alpha, Beta and Gamma."
    between_text = "The company manages each division separately."
    outside_text = "UNRELATED_PAGE_MATERIAL"
    texts = ([hit_text, between_text, list_text, outside_text] if hit_at_start
             else [outside_text, list_text, between_text, hit_text])
    chunks = []
    offset = 0
    for index, text in enumerate(texts):
        chunks.append(Document(id=f"chunk-{index}", page_content=text,
                               metadata={**metadata, "start_index": offset}))
        offset += len(text) + 1
    hit = chunks[0] if hit_at_start else chunks[-1]
    other_page = Document(id="another-page", page_content="OTHER_PAGE_EVIDENCE",
                          metadata={**metadata, "page_number": 8, "start_index": 0})

    class VectorStore:
        def similarity_search_with_relevance_scores(self, query, **kwargs):
            return [(hit, 0.9)]

    result = ScoreThresholdRetriever(
        vector_store=VectorStore(), keyword_documents=[*chunks, other_page],
        tickers=["MSFT"], k=1,
    ).invoke("What are the reportable segments?")[0]

    assert list_text in result.page_content
    assert hit_text in result.page_content
    assert between_text in result.page_content
    assert outside_text not in result.page_content
    assert "OTHER_PAGE_EVIDENCE" not in result.page_content
    assert result.metadata["page_number"] == 7
    assert result.metadata["matched_start_index"] == hit.metadata["start_index"]


def test_hybrid_filters_company_and_report_year_in_both_searches(tmp_path: Path):
    for ticker, year in [("AAPL", "2025"), ("AAPL", "2024"), ("MSFT", "2025")]:
        revision = "2025" if year == "2024" else "1"
        (tmp_path / f"{ticker}_{year}.md").write_text(
            f"---\nticker: {ticker}\ndocument_type: annual_report\n"
            f'fiscal_year: {year}\n'
            f'document_version: "{revision}"\n---\n'
            f"{ticker} reportable segments for {year}.", encoding="utf-8",
        )
    embeddings = CountingEmbeddings()
    retriever = build_knowledge_retriever(
        tmp_path, embeddings=embeddings, tickers=["AAPL"],
        score_threshold=0, persist_directory=tmp_path / "index",
    )
    results = retriever.invoke("What are AAPL's reportable segments in its 2025 Form 10-K?")

    assert [doc.metadata["source"] for doc in results] == ["AAPL_2025.md"]
    assert results[0].metadata["fiscal_year"] == 2025
    assert results[0].metadata["document_version"] == "1"
    assert retriever._report_scope("Apple's 2025 Form 10-K") == {
        "document_type": "annual_report", "fiscal_year": 2025,
    }
    assert embeddings.query_calls == 1
    assert retriever.invoke("What are AAPL's segments in its 2023 annual report?") == []


def test_annual_report_requires_fiscal_year_independent_of_revision():
    with pytest.raises(ValidationError, match="annual_report metadata requires fiscal_year"):
        KnowledgeDocumentMetadata(document_type="annual_report", document_version="2025")

    metadata = KnowledgeDocumentMetadata(
        document_type="annual_report", fiscal_year=2025, document_version="1.1",
    )
    assert metadata.model_dump(mode="json")["fiscal_year"] == 2025
    assert metadata.document_version == "1.1"
    assert KnowledgeDocumentMetadata(document_type="methodology").fiscal_year is None


@pytest.mark.parametrize("year", ["2025", True, 2025.5, 1899, 2100])
def test_fiscal_year_rejects_non_integer_or_unsupported_years(year):
    with pytest.raises(ValidationError, match="fiscal_year"):
        KnowledgeDocumentMetadata(document_type="annual_report", fiscal_year=year)


def test_pdf_loader_removes_repeated_margins_but_keeps_page_numbers(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "report.pdf"
    pdf_path.touch()
    pdf_path.with_suffix(".yaml").write_text(
        "document_type: annual_report\nfiscal_year: 2025\n", encoding="utf-8",
    )
    pages = [SimpleNamespace(extract_text=lambda n=n: f"Body for page {n}.\nApple Inc. | 2025 Form 10-K | {n}")
             for n in range(1, 5)]
    monkeypatch.setattr("pypdf.PdfReader", lambda path: SimpleNamespace(pages=pages))
    docs = load_knowledge_documents(tmp_path)
    assert [doc.page_content for doc in docs] == [f"Body for page {n}." for n in range(1, 5)]
    assert [doc.metadata["page_number"] for doc in docs] == [1, 2, 3, 4]


class KeywordEmbeddings(Embeddings):
    """Small deterministic embedding substitute for unit tests."""

    def _embed(self, text: str) -> list[float]:
        normalized = text.lower()
        return [
            float("aapl" in normalized or "apple" in normalized),
            float("msft" in normalized or "microsoft" in normalized),
            float("methodology" in normalized or "closing price" in normalized),
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class CountingEmbeddings(KeywordEmbeddings):
    """Keyword embeddings that record document-embedding work."""

    def __init__(self):
        self.document_calls = 0
        self.query_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return super().embed_query(text)


def write_company_document(path: Path, ticker: str, company: str, body: str):
    path.write_text(
        f"---\n"
        f"ticker: {ticker}\n"
        f"company: {company}\n"
        f"document_type: company_overview\n"
        f"source_url: https://example.com/{ticker.lower()}\n"
        f"---\n\n"
        f"{body}",
        encoding="utf-8",
    )


def write_methodology_document(path: Path):
    path.write_text(
        "---\n"
        "document_type: methodology\n"
        "source_name: Internal project methodology\n"
        "---\n\n"
        "The latest available closing price is not necessarily a live price.\n"
        "Share-cost calculations use price multiplied by shares.",
        encoding="utf-8",
    )


def write_text_pdf(path: Path, pages: list[str]):
    writer = PdfWriter()
    font = writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
    )

    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {NameObject("/F1"): font}
                )
            }
        )
        content = DecodedStreamObject()
        content.set_data(
            f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
        )
        page[NameObject("/Contents")] = writer._add_object(content)

    with path.open("wb") as file:
        writer.write(file)


def test_load_knowledge_documents_adds_source_metadata(tmp_path: Path):
    (tmp_path / "apple.md").write_text(
        "---\n"
        "ticker: AAPL\n"
        "company: Apple Inc.\n"
        "document_type: company_overview\n"
        "source_url: https://investor.apple.com/\n"
        "---\n\n"
        "# Apple\nApple develops hardware and services.",
        encoding="utf-8",
    )

    documents = load_knowledge_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].page_content == "# Apple\nApple develops hardware and services."
    assert documents[0].metadata == {
        "source": "apple.md",
        "ticker": "AAPL",
        "company": "Apple Inc.",
        "document_type": "company_overview",
        "source_url": "https://investor.apple.com/",
        "ticker_scope": "AAPL",
    }


def test_load_knowledge_documents_normalizes_date_metadata(tmp_path: Path):
    (tmp_path / "dated.md").write_text(
        "---\n"
        "document_type: company_overview\n"
        "published_date: 2026-01-15\n"
        "last_reviewed: 2026-09-26\n"
        "document_version: \"1.0\"\n"
        "---\n\n"
        "A dated reference document.",
        encoding="utf-8",
    )

    documents = load_knowledge_documents(tmp_path)

    assert documents[0].metadata["published_date"] == "2026-01-15"
    assert documents[0].metadata["last_reviewed"] == "2026-09-26"
    assert documents[0].metadata["document_version"] == "1.0"


def test_load_knowledge_documents_extracts_pdf_pages_with_sidecar_metadata(
    tmp_path: Path,
):
    pdf_path = tmp_path / "raw" / "apple_annual_report.pdf"
    pdf_path.parent.mkdir()
    write_text_pdf(
        pdf_path,
        ["Apple annual report page one.", "Apple annual report page two."],
    )
    pdf_path.with_suffix(".yaml").write_text(
        "ticker: AAPL\n"
        "company: Apple Inc.\n"
        "document_type: annual_report\n"
        "source_name: Apple Investor Relations\n"
        "source_url: https://example.com/apple-report.pdf\n"
        "published_date: 2026-01-15\n"
        "last_reviewed: 2026-09-26\n"
        "fiscal_year: 2025\n"
        "document_version: \"1\"\n",
        encoding="utf-8",
    )

    documents = load_knowledge_documents(tmp_path)

    assert [document.page_content for document in documents] == [
        "Apple annual report page one.",
        "Apple annual report page two.",
    ]
    assert [document.metadata["page_number"] for document in documents] == [1, 2]
    assert all(document.metadata["page_count"] == 2 for document in documents)
    assert all(document.metadata["source"] == "raw/apple_annual_report.pdf" for document in documents)
    assert all(document.metadata["source_format"] == "pdf" for document in documents)
    assert documents[0].metadata["ticker_scope"] == "AAPL"
    assert documents[0].metadata["published_date"] == "2026-01-15"
    assert all(document.metadata["fiscal_year"] == 2025 for document in documents)
    assert all(document.metadata["document_version"] == "1" for document in documents)


def test_load_knowledge_documents_requires_pdf_sidecar(tmp_path: Path):
    pdf_path = tmp_path / "report.pdf"
    write_text_pdf(pdf_path, ["A report without metadata."])

    with pytest.raises(FileNotFoundError, match="requires a YAML sidecar"):
        load_knowledge_documents(tmp_path)


def test_pdf_metadata_refresh_skips_extraction_but_content_changes_reprocess(tmp_path, monkeypatch):
    from market_research import rag

    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    pdf = knowledge / "report.pdf"
    write_text_pdf(pdf, ["Apple hardware business.", "Apple services business."])
    sidecar = pdf.with_suffix(".yaml")
    sidecar.write_text("ticker: AAPL\ndocument_type: annual_report\nfiscal_year: 2025\n")

    def ingest(model):
        return ingest_knowledge_base(knowledge, model, persist_directory=tmp_path / "index")

    first = ingest(CountingEmbeddings())
    calls = []
    loader = rag._load_pdf_documents

    def spy(path, source):
        calls.append(source)
        return loader(path, source)

    monkeypatch.setattr(rag, "_load_pdf_documents", spy)
    unchanged = ingest(CountingEmbeddings())
    assert calls == []
    assert unchanged.document_count == first.document_count == 2
    assert unchanged.changes.skipped_sources == 1

    sidecar.write_text(sidecar.read_text() + "last_reviewed: 2026-11-25\n")
    model = CountingEmbeddings()
    updated = ingest(model)
    assert calls == []
    assert updated.changes.processed_sources == 0
    assert updated.changes.metadata_only_sources == 1
    assert updated.changes.metadata_updates == 2
    assert model.document_calls == 0

    write_text_pdf(pdf, ["Apple updated hardware business.", "Apple services business."])
    model = CountingEmbeddings()
    changed = ingest(model)
    assert calls == ["report.pdf"]
    assert changed.changes.embedded_texts == 1

    sidecar.unlink()
    with pytest.raises(FileNotFoundError, match="requires a YAML sidecar"):
        ingest(CountingEmbeddings())


@pytest.mark.parametrize("source_format", ["pdf", "md"])
def test_metadata_only_refresh_preserves_vectors_and_updates_retrieval(tmp_path, monkeypatch, source_format):
    from chromadb import PersistentClient
    from market_research import rag

    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    path = knowledge / f"report.{source_format}"
    body = "Apple hardware business and services."
    if source_format == "pdf":
        write_text_pdf(path, [body])
        metadata_path = path.with_suffix(".yaml")
    else:
        metadata_path = path

    def write_metadata(text):
        metadata_path.write_text(text if source_format == "pdf" else f"---\n{text}---\n\n{body}")

    write_metadata("ticker: AAPL\ncompany: Old company\ndocument_type: annual_report\n"
                   "fiscal_year: 2025\nsource_url: https://example.com/old\n")
    first = ingest_knowledge_base(knowledge, CountingEmbeddings(), persist_directory=tmp_path / "index")
    collection = PersistentClient(path=str(first.persist_directory)).get_collection(
        first.collection_name, embedding_function=None,
    )
    before = collection.get(include=["embeddings", "metadatas", "documents"])
    write_metadata("ticker: MSFT\ndocument_type: annual_report\nfiscal_year: 2024\n"
                   "source_url: https://example.com/new\n")

    def forbidden(*args, **kwargs):
        pytest.fail("Metadata-only edit extracted or split source text")

    monkeypatch.setattr(rag, "_load_knowledge_source", forbidden)
    monkeypatch.setattr(rag, "split_documents", forbidden)
    model = CountingEmbeddings()
    refreshed = ingest_knowledge_base(knowledge, model, persist_directory=first.persist_directory)
    after = collection.get(include=["embeddings", "metadatas", "documents"])
    assert refreshed.changes.metadata_only_sources == 1
    assert refreshed.changes.processed_sources == refreshed.changes.skipped_sources == 0
    assert refreshed.changes.metadata_updates == 1
    assert before["ids"] == after["ids"]
    assert before["documents"] == after["documents"]
    assert before["embeddings"].tolist() == after["embeddings"].tolist()
    assert "company" not in after["metadatas"][0]
    assert after["metadatas"][0]["source_url"] == "https://example.com/new"
    assert collection.get(where={"ticker_scope": "AAPL"})["ids"] == []
    assert collection.get(where={"ticker_scope": "MSFT"})["ids"] == after["ids"]
    retriever = build_knowledge_retriever(
        knowledge, model, persist_directory=first.persist_directory, tickers=["MSFT"],
    )
    # Exercise vector filtering using the same unchanged vector with NEW metadata.
    assert retriever.vector_store.similarity_search(
        "Apple hardware", filter={"fiscal_year": 2024}, k=1,
    )[0].metadata["ticker"] == "MSFT"
    assert retriever.vector_store.similarity_search(
        "Apple hardware", filter={"fiscal_year": 2025}, k=1,
    ) == []
    # Then require keyword-only retrieval to prove BM25 received refreshed fields.
    monkeypatch.setattr(retriever.vector_store, "similarity_search_with_relevance_scores", lambda *a, **kw: [])
    result = retriever.invoke("2024 annual report hardware business")
    assert result[0].metadata["ticker"] == "MSFT"
    assert result[0].metadata["fiscal_year"] == 2024
    assert result[0].metadata["source_url"] == "https://example.com/new"
    assert retriever.invoke("2025 annual report hardware business") == []
    assert model.document_calls == 0
    again = ingest_knowledge_base(knowledge, model, persist_directory=first.persist_directory)
    assert again.changes.skipped_sources == 1
    assert again.changes.metadata_only_sources == 0

    # Omitting ticker/year/citation fields must remove stale filtering metadata.
    write_metadata("document_type: company_overview\n")
    generic = ingest_knowledge_base(knowledge, model, persist_directory=first.persist_directory)
    current = collection.get(include=["metadatas"])["metadatas"][0]
    assert generic.changes.metadata_only_sources == 1
    assert current["ticker_scope"] == "__generic__"
    assert not {"ticker", "fiscal_year", "source_url", "company"}.intersection(current)
    for key in ("start_index", "document_id", "content_hash", "page_number", "page_count"):
        if key in before["metadatas"][0]:
            assert current[key] == before["metadatas"][0][key]

    # Invalid replacement metadata must fail before any records/manifest change.
    manifest = first.manifest_path.read_bytes()
    write_metadata("document_type: annual_report\nfiscal_year: invalid\n")
    with pytest.raises(ValueError):
        ingest_knowledge_base(knowledge, model, persist_directory=first.persist_directory)
    assert first.manifest_path.read_bytes() == manifest
    assert collection.get(include=["metadatas"])["metadatas"][0] == current
    assert model.document_calls == 0


def test_split_documents_preserves_metadata_and_creates_chunks():
    documents = [Document(page_content="Apple business information " * 50, metadata={"source": "apple.md"})]

    chunks = split_documents(documents, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "apple.md" for chunk in chunks)
    assert all(len(chunk.page_content) <= 100 for chunk in chunks)


def test_search_knowledge_returns_relevant_chunk(tmp_path: Path):
    write_company_document(
        tmp_path / "apple.md",
        "AAPL",
        "Apple Inc.",
        "Apple develops hardware and services.",
    )
    write_company_document(
        tmp_path / "microsoft.md",
        "MSFT",
        "Microsoft Corporation",
        "Microsoft develops software and cloud services.",
    )

    retriever = build_knowledge_retriever(
        tmp_path,
        embeddings=KeywordEmbeddings(),
        chunk_size=200,
        chunk_overlap=0,
        k=1,
        persist_directory=tmp_path / "index",
    )
    results = search_knowledge("What does Apple do?", retriever=retriever)

    assert len(results) == 1
    assert results[0].metadata["source"] == "apple.md"


def test_search_knowledge_filters_low_score_chunks(tmp_path: Path):
    write_company_document(
        tmp_path / "apple.md",
        "AAPL",
        "Apple Inc.",
        "Apple develops hardware and services.",
    )
    write_company_document(
        tmp_path / "microsoft.md",
        "MSFT",
        "Microsoft Corporation",
        "Microsoft develops software and cloud services.",
    )

    retriever = build_knowledge_retriever(
        tmp_path,
        embeddings=KeywordEmbeddings(),
        chunk_size=200,
        chunk_overlap=0,
        k=2,
        persist_directory=tmp_path / "index",
    )
    results = search_knowledge("What does Apple do?", retriever=retriever)

    assert [result.metadata["source"] for result in results] == ["apple.md"]
    assert results[0].metadata["relevance_score"] == 1.0


def test_search_knowledge_filters_documents_by_requested_ticker(tmp_path: Path):
    write_company_document(
        tmp_path / "apple.md",
        "AAPL",
        "Apple Inc.",
        "Apple develops hardware and services.",
    )
    write_company_document(
        tmp_path / "microsoft.md",
        "MSFT",
        "Microsoft Corporation",
        "Microsoft develops software and cloud services.",
    )

    retriever = build_knowledge_retriever(
        tmp_path,
        embeddings=KeywordEmbeddings(),
        chunk_size=200,
        chunk_overlap=0,
        k=2,
        score_threshold=0.0,
        tickers=["aapl"],
        persist_directory=tmp_path / "index",
    )
    results = search_knowledge("What does Apple do?", retriever=retriever)

    assert [result.metadata["source"] for result in results] == ["apple.md"]


def test_build_knowledge_retriever_rejects_invalid_score_threshold(tmp_path: Path):
    write_company_document(
        tmp_path / "apple.md",
        "AAPL",
        "Apple Inc.",
        "Apple develops hardware and services.",
    )

    try:
        build_knowledge_retriever(
            tmp_path,
            embeddings=KeywordEmbeddings(),
            score_threshold=1.1,
        )
    except ValueError as exc:
        assert str(exc) == "score_threshold must be between zero and one."
    else:
        raise AssertionError("Expected invalid score threshold to be rejected")


def test_persistent_retriever_reuses_existing_index(tmp_path: Path):
    write_company_document(
        tmp_path / "apple.md",
        "AAPL",
        "Apple Inc.",
        "Apple develops hardware and services.",
    )
    persist_directory = tmp_path / "index"

    first_embeddings = CountingEmbeddings()
    first_retriever = build_knowledge_retriever(
        tmp_path,
        embeddings=first_embeddings,
        persist_directory=persist_directory,
        collection_name="test_collection",
        k=1,
    )
    first_results = search_knowledge("What does Apple do?", retriever=first_retriever)

    second_embeddings = CountingEmbeddings()
    second_retriever = build_knowledge_retriever(
        tmp_path,
        embeddings=second_embeddings,
        persist_directory=persist_directory,
        collection_name="test_collection",
        k=1,
    )
    second_results = search_knowledge("What does Apple do?", retriever=second_retriever)

    assert first_embeddings.document_calls == 1
    assert second_embeddings.document_calls == 0
    assert first_results[0].metadata["source"] == "apple.md"
    assert second_results[0].metadata["source"] == "apple.md"
    assert (persist_directory / "test_collection.manifest.json").exists()


def test_persistent_retriever_rebuilds_when_documents_change(tmp_path: Path):
    document_path = tmp_path / "apple.md"
    write_company_document(
        document_path,
        "AAPL",
        "Apple Inc.",
        "Apple develops hardware and services.",
    )
    persist_directory = tmp_path / "index"

    first_embeddings = CountingEmbeddings()
    build_knowledge_retriever(
        tmp_path,
        embeddings=first_embeddings,
        persist_directory=persist_directory,
        collection_name="test_collection",
        k=1,
    )

    write_company_document(
        document_path,
        "AAPL",
        "Apple Inc.",
        "Apple develops hardware, software, and services.",
    )
    second_embeddings = CountingEmbeddings()
    build_knowledge_retriever(
        tmp_path,
        embeddings=second_embeddings,
        persist_directory=persist_directory,
        collection_name="test_collection",
        k=1,
    )

    assert first_embeddings.document_calls == 1
    assert second_embeddings.document_calls == 1


def test_ingest_knowledge_base_reports_rebuild_and_reuse(tmp_path: Path):
    write_company_document(
        tmp_path / "apple.md",
        "AAPL",
        "Apple Inc.",
        "Apple develops hardware and services.",
    )
    persist_directory = tmp_path / "index"

    first_report = ingest_knowledge_base(
        tmp_path,
        embeddings=CountingEmbeddings(),
        persist_directory=persist_directory,
        collection_name="test_collection",
    )
    second_report = ingest_knowledge_base(
        tmp_path,
        embeddings=CountingEmbeddings(),
        persist_directory=persist_directory,
        collection_name="test_collection",
    )

    assert first_report.rebuilt is True
    assert second_report.rebuilt is False
    assert first_report.document_count == 1
    assert first_report.chunk_count == 1
    assert second_report.manifest_path == (
        persist_directory / "test_collection.manifest.json"
    )


def test_search_knowledge_rejects_empty_questions():
    try:
        search_knowledge("   ", retriever=object())
    except ValueError as exc:
        assert str(exc) == "Knowledge search question must not be empty."
    else:
        raise AssertionError("Expected empty knowledge search to fail")


@pytest.mark.parametrize(
    ("question", "tickers", "expected_source"),
    [
        (
            "What are Apple's major business segments?",
            ["AAPL"],
            "apple.md",
        ),
        (
            "What are Microsoft's major business segments?",
            ["MSFT"],
            "microsoft.md",
        ),
        (
            "What is the quote freshness methodology?",
            [],
            "methodology.md",
        ),
    ],
)
def test_retrieval_evaluation_cases(
    tmp_path: Path,
    question: str,
    tickers: list[str],
    expected_source: str,
):
    write_company_document(
        tmp_path / "apple.md",
        "AAPL",
        "Apple Inc.",
        "Apple's major business segments include iPhone, Mac, and Services.",
    )
    write_company_document(
        tmp_path / "microsoft.md",
        "MSFT",
        "Microsoft Corporation",
        "Microsoft's major business segments include Productivity, Cloud, and More Personal Computing.",
    )
    write_methodology_document(tmp_path / "methodology.md")

    retriever = build_knowledge_retriever(
        tmp_path,
        embeddings=KeywordEmbeddings(),
        k=2,
        score_threshold=0.45,
        tickers=tickers,
        persist_directory=tmp_path / "index",
        collection_name="evaluation",
    )
    results = search_knowledge(question, retriever=retriever)

    assert [result.metadata["source"] for result in results] == [expected_source]
    assert results[0].metadata["relevance_score"] >= 0.45


def test_retrieval_evaluation_prevents_cross_company_context(tmp_path: Path):
    write_company_document(
        tmp_path / "apple.md",
        "AAPL",
        "Apple Inc.",
        "Apple's major business segments include iPhone, Mac, and Services.",
    )
    write_company_document(
        tmp_path / "microsoft.md",
        "MSFT",
        "Microsoft Corporation",
        "Microsoft's major business segments include Productivity, Cloud, and More Personal Computing.",
    )

    retriever = build_knowledge_retriever(
        tmp_path,
        embeddings=KeywordEmbeddings(),
        k=2,
        score_threshold=0.0,
        tickers=["AAPL"],
        persist_directory=tmp_path / "index",
        collection_name="evaluation",
    )
    results = search_knowledge(
        "What are Microsoft's major business segments?",
        retriever=retriever,
    )

    assert "microsoft.md" not in {
        result.metadata["source"] for result in results
    }
