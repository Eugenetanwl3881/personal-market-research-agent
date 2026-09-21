import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, ConfigDict, field_validator


DEFAULT_KNOWLEDGE_DIR = (
    Path(__file__).resolve().parents[2] / "knowledge"
)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PERSIST_DIRECTORY = Path(__file__).resolve().parents[2] / ".rag_index"
DEFAULT_COLLECTION_NAME = "market_research_knowledge"
DEFAULT_RETRIEVAL_K = 2
DEFAULT_RELEVANCE_THRESHOLD = 0.45
INDEX_VERSION = 1
GENERIC_TICKER_SCOPE = "__generic__"


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

    vector_store: Any
    k: int = DEFAULT_RETRIEVAL_K
    score_threshold: float = DEFAULT_RELEVANCE_THRESHOLD
    tickers: list[str] | None = None

    def _metadata_filter(self) -> dict[str, Any] | None:
        """Build a Chroma filter for requested and generic documents."""
        if not self.tickers:
            return None

        return {
            "$or": [
                *({"ticker_scope": ticker} for ticker in self.tickers),
                {"ticker_scope": GENERIC_TICKER_SCOPE},
            ],
        }

    def _get_relevant_documents(self, query: str) -> list[Document]:
        """Search the vector store and discard weakly related chunks."""
        matches = self.vector_store.similarity_search_with_relevance_scores(
            query,
            k=self.k,
            filter=self._metadata_filter(),
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


def _index_manifest_path(
    persist_directory: Path,
    collection_name: str,
) -> Path:
    """Return the sidecar manifest path for a persisted collection."""
    return persist_directory / f"{collection_name}.manifest.json"


def _embedding_identity(embeddings: Embeddings | None) -> str:
    """Return a stable identity used to invalidate incompatible indexes."""
    if embeddings is None:
        return os.getenv("RAG_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL

    embedding_type = type(embeddings)
    return f"{embedding_type.__module__}.{embedding_type.__qualname__}"


def _index_signature(
    chunks: list[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
    embedding_identity: str,
) -> str:
    """Hash indexed content and configuration to detect stale persisted data."""
    signature_input = {
        "index_version": INDEX_VERSION,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_identity": embedding_identity,
        "chunks": [
            {
                "content": chunk.page_content,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ],
    }
    serialized = json.dumps(
        signature_input,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _read_index_signature(manifest_path: Path) -> str | None:
    """Read an existing index signature, returning None if unavailable."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    signature = manifest.get("index_signature")
    return signature if isinstance(signature, str) else None


def _has_expected_chunks(vector_store: Any, chunks: list[Document]) -> bool:
    """Check that a matching collection still contains every expected chunk."""
    expected_ids = {_stable_chunk_id(chunk) for chunk in chunks}
    try:
        stored_ids = set(vector_store.get(include=[])["ids"])
    except Exception:
        return False
    return stored_ids == expected_ids


def _write_index_manifest(
    manifest_path: Path,
    *,
    signature: str,
    chunk_count: int,
) -> None:
    """Write the metadata needed to decide whether an index can be reused."""
    manifest_path.write_text(
        json.dumps(
            {
                "index_version": INDEX_VERSION,
                "index_signature": signature,
                "chunk_count": chunk_count,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _create_persistent_vector_store(
    persist_directory: Path,
    collection_name: str,
    embeddings: Embeddings,
):
    """Create or open a local Chroma collection using cosine distance."""
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


def _stable_chunk_id(chunk: Document) -> str:
    """Create a deterministic ID so the persisted index is reproducible."""
    serialized = json.dumps(
        {
            "content": chunk.page_content,
            "metadata": chunk.metadata,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


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
    document_metadata["ticker_scope"] = document_metadata.get(
        "ticker",
        GENERIC_TICKER_SCOPE,
    )

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
    persist_directory: str | Path | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    force_rebuild: bool = False,
):
    """Build or reuse a persistent, relevance-filtered knowledge retriever."""
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
    embedding_function = embeddings or create_embeddings()
    embedding_identity = _embedding_identity(embeddings)
    signature = _index_signature(
        chunks,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_identity=embedding_identity,
    )

    index_directory = Path(
        persist_directory
        or os.getenv("RAG_INDEX_DIR")
        or DEFAULT_PERSIST_DIRECTORY
    )
    index_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = _index_manifest_path(index_directory, collection_name)
    vector_store = _create_persistent_vector_store(
        index_directory,
        collection_name,
        embedding_function,
    )

    index_is_current = (
        not force_rebuild
        and _read_index_signature(manifest_path) == signature
    )
    if index_is_current:
        index_is_current = _has_expected_chunks(vector_store, chunks)

    if not index_is_current:
        vector_store.delete_collection()
        vector_store = _create_persistent_vector_store(
            index_directory,
            collection_name,
            embedding_function,
        )
        vector_store.add_documents(
            chunks,
            ids=[_stable_chunk_id(chunk) for chunk in chunks],
        )
        _write_index_manifest(
            manifest_path,
            signature=signature,
            chunk_count=len(chunks),
        )

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
