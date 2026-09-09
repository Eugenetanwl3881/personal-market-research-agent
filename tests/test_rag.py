from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from market_research.rag import (
    build_knowledge_retriever,
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


def test_load_knowledge_documents_adds_source_metadata(tmp_path: Path):
    (tmp_path / "apple.md").write_text("# Apple\nTicker: AAPL", encoding="utf-8")

    documents = load_knowledge_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].page_content == "# Apple\nTicker: AAPL"
    assert documents[0].metadata == {
        "source": "apple.md",
        "document_type": "markdown",
    }


def test_split_documents_preserves_metadata_and_creates_chunks():
    documents = [Document(page_content="Apple business information " * 50, metadata={"source": "apple.md"})]

    chunks = split_documents(documents, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "apple.md" for chunk in chunks)
    assert all(len(chunk.page_content) <= 100 for chunk in chunks)


def test_search_knowledge_returns_relevant_chunk(tmp_path: Path):
    (tmp_path / "apple.md").write_text(
        "Apple (AAPL) develops hardware and services.", encoding="utf-8"
    )
    (tmp_path / "microsoft.md").write_text(
        "Microsoft (MSFT) develops software and cloud services.", encoding="utf-8"
    )

    retriever = build_knowledge_retriever(
        tmp_path,
        embeddings=KeywordEmbeddings(),
        chunk_size=200,
        chunk_overlap=0,
        k=1,
    )
    results = search_knowledge("What does Apple do?", retriever=retriever)

    assert len(results) == 1
    assert results[0].metadata["source"] == "apple.md"


def test_search_knowledge_rejects_empty_questions():
    try:
        search_knowledge("   ", retriever=object())
    except ValueError as exc:
        assert str(exc) == "Knowledge search question must not be empty."
    else:
        raise AssertionError("Expected empty knowledge search to fail")
