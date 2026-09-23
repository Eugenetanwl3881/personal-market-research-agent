from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from market_research.rag import (
    build_knowledge_retriever,
    ingest_knowledge_base,
    load_knowledge_documents,
    search_knowledge,
    split_documents,
)


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
