import os
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import BaseModel, ConfigDict, field_validator


DEFAULT_KNOWLEDGE_DIR = (
    Path(__file__).resolve().parents[2] / "knowledge"
)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RETRIEVAL_K = 2
DEFAULT_RELEVANCE_THRESHOLD = 0.45


class KnowledgeDocumentMetadata(BaseModel):
    """Validated metadata declared in a knowledge document's frontmatter."""

    model_config = ConfigDict(extra="forbid")

    ticker: str | None = None
    company: str | None = None
    document_type: str
    source_url: str | None = None
    source_name: str | None = None

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str | None) -> str | None:
        if value is None:
            return None

        normalized = value.strip().upper()
        if not re.fullmatch(r"[A-Z]{1,5}", normalized):
            raise ValueError("ticker must contain one to five letters.")
        return normalized


class ScoreThresholdRetriever(BaseRetriever):
    """Retrieve only top-ranked chunks whose cosine score is high enough."""

    vector_store: InMemoryVectorStore
    k: int = DEFAULT_RETRIEVAL_K
    score_threshold: float = DEFAULT_RELEVANCE_THRESHOLD
    tickers: list[str] | None = None

    def _matches_requested_tickers(self, document: Document) -> bool:
        """Keep requested-company documents and untickered generic documents."""
        if not self.tickers:
            return True

        document_ticker = document.metadata.get("ticker")
        return document_ticker is None or document_ticker in self.tickers

    def _get_relevant_documents(self, query: str) -> list[Document]:
        """Search the vector store and discard weakly related chunks."""
        matches = self.vector_store.similarity_search_with_score(
            query,
            k=self.k,
            filter=self._matches_requested_tickers,
        )
        relevant_documents = []

        for document, score in matches:
            if score < self.score_threshold:
                continue

            relevant_documents.append(
                Document(
                    id=document.id,
                    page_content=document.page_content,
                    metadata={
                        **document.metadata,
                        "relevance_score": round(score, 4),
                    },
                )
            )

        return relevant_documents


def _parse_markdown_frontmatter(
    content: str,
    source_name: str,
) -> tuple[dict[str, Any], str]:
    """Parse and validate YAML frontmatter, returning metadata and document body."""
    lines = content.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise ValueError(
            f"Knowledge document must begin with YAML frontmatter: {source_name}"
        )

    closing_index = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"),
        None,
    )
    if closing_index is None:
        raise ValueError(
            f"Knowledge document frontmatter is not closed: {source_name}"
        )

    import yaml

    raw_metadata = yaml.safe_load("".join(lines[1:closing_index])) or {}
    if not isinstance(raw_metadata, dict):
        raise ValueError(
            f"Knowledge document metadata must be a mapping: {source_name}"
        )

    metadata = KnowledgeDocumentMetadata.model_validate(raw_metadata)
    document_metadata = metadata.model_dump(exclude_none=True)
    document_metadata["source"] = source_name

    body = "".join(lines[closing_index + 1:]).lstrip()
    return document_metadata, body


def load_knowledge_documents(
    knowledge_dir: str | Path | None = None,
) -> list[Document]:
    """Load Markdown files with validated YAML frontmatter metadata."""
    directory = Path(knowledge_dir) if knowledge_dir else DEFAULT_KNOWLEDGE_DIR

    if not directory.exists():
        raise FileNotFoundError(f"Knowledge directory does not exist: {directory}")

    paths = sorted(directory.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No Markdown documents found in: {directory}")

    documents = []
    for path in paths:
        content = path.read_text(encoding="utf-8")
        metadata, body = _parse_markdown_frontmatter(content, path.name)
        documents.append(Document(page_content=body, metadata=metadata))

    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> list[Document]:
    """Split documents into overlapping chunks suitable for retrieval."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between zero and chunk_size.")

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def create_embeddings(model_name: str | None = None) -> Embeddings:
    """Create local embeddings without requiring a separate embedding API key."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name
        or os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        encode_kwargs={"normalize_embeddings": True},
    )


def build_knowledge_retriever(
    knowledge_dir: str | Path | None = None,
    embeddings: Embeddings | None = None,
    *,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
    k: int = DEFAULT_RETRIEVAL_K,
    score_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    tickers: list[str] | None = None,
):
    """Build a relevance-filtered retriever from local Markdown documents."""
    if k <= 0:
        raise ValueError("k must be greater than zero.")
    if not 0 <= score_threshold <= 1:
        raise ValueError("score_threshold must be between zero and one.")

    requested_tickers = list(dict.fromkeys(
        ticker.strip().upper()
        for ticker in (tickers or [])
        if ticker.strip()
    ))

    documents = load_knowledge_documents(knowledge_dir)
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    vector_store = InMemoryVectorStore(
        embedding=embeddings or create_embeddings(),
    )
    vector_store.add_documents(chunks)
    return ScoreThresholdRetriever(
        vector_store=vector_store,
        k=k,
        score_threshold=score_threshold,
        tickers=requested_tickers or None,
    )


def search_knowledge(
    question: str,
    retriever: Any | None = None,
    **retriever_options: Any,
) -> list[Document]:
    """Return the most relevant knowledge chunks for a question."""
    if not question.strip():
        raise ValueError("Knowledge search question must not be empty.")

    active_retriever = retriever or build_knowledge_retriever(**retriever_options)
    return active_retriever.invoke(question)
