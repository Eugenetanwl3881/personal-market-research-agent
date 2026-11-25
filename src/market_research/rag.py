import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from math import log
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .indexing import (
    INDEX_VERSION,
    IndexChanges,
    identify_chunks,
    stable_chunk_id as _stable_chunk_id,
    synchronize_index,
)


DEFAULT_KNOWLEDGE_DIR = (
    Path(__file__).resolve().parents[2] / "knowledge"
)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PERSIST_DIRECTORY = Path(__file__).resolve().parents[2] / ".rag_index"
DEFAULT_COLLECTION_NAME = "market_research_knowledge"
DEFAULT_RETRIEVAL_K = 2
DEFAULT_CANDIDATE_K = 20
DEFAULT_RELEVANCE_THRESHOLD = 0.45
GENERIC_TICKER_SCOPE = "__generic__"

# Common question words add noise to keyword matching. Keep content words,
# including numbers and filing terms, so this also works beyond annual reports.
KEYWORD_STOP_WORDS = frozenset(
    "a an and are as at be by can do does for from how i in is it its of on or "
    "s say says that the their this to was were what when where which who will "
    "with would according please tell me about".split()
)
ANNUAL_REPORT_PATTERN = r"\b(?:form\s+)?10[\s-]?k\b|\bannual\s+report\b"


def _keyword_tokens(text: str) -> list[str]:
    """Tokenize prose for local keyword retrieval without model calls."""
    return [
        token for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in KEYWORD_STOP_WORDS
    ]


def _keyword_search(
    query: str, documents: list[Document], k: int,
) -> list[tuple[Document, float]]:
    """Rank chunks with BM25, favoring rare terms over repeated boilerplate."""
    query_terms = sorted(set(_keyword_tokens(query)))
    if not documents or not query_terms:
        return []
    counts = [Counter(_keyword_tokens(doc.page_content)) for doc in documents]
    lengths = [sum(count.values()) for count in counts]
    average_length = sum(lengths) / len(lengths) or 1
    document_frequencies = Counter(
        term for count in counts for term in query_terms if term in count
    )
    matches = []
    for document, frequencies, length in zip(documents, counts, lengths):
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            document_frequency = document_frequencies[term]
            inverse_frequency = log(
                1 + (len(documents) - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            score += inverse_frequency * frequency * 2.5 / (
                frequency + 1.5 * (0.25 + 0.75 * length / average_length)
            )
        if score > 0:
            matches.append((document, score))
    return sorted(matches, key=lambda match: match[1], reverse=True)[:k]


@dataclass(frozen=True)
class KnowledgeIndexReport:
    """Summary of an explicit knowledge-indexing operation."""

    knowledge_directory: Path
    persist_directory: Path
    collection_name: str
    document_count: int
    chunk_count: int
    rebuilt: bool
    manifest_path: Path
    changes: IndexChanges = field(default_factory=IndexChanges)


@dataclass
class _PreparedKnowledgeIndex:
    """Internal result shared by indexing and retriever construction."""

    vector_store: Any
    knowledge_directory: Path
    persist_directory: Path
    collection_name: str
    documents: list[Document]
    chunks: list[Document]
    manifest_path: Path
    rebuilt: bool
    changes: IndexChanges


class KnowledgeDocumentMetadata(BaseModel):
    """Validated metadata declared in a knowledge document's frontmatter."""

    model_config = ConfigDict(extra="forbid")

    ticker: str | None = None
    company: str | None = None
    document_type: str
    source_url: str | None = None
    source_name: str | None = None
    published_date: date | None = None
    last_reviewed: date | None = None
    fiscal_year: int | None = Field(default=None, strict=True, ge=1900, le=2099)
    document_version: str | None = None

    @model_validator(mode="after")
    def require_annual_report_year(self):
        if self.document_type == "annual_report" and self.fiscal_year is None:
            raise ValueError(
                "annual_report metadata requires fiscal_year (an integer such as 2025); "
                "document_version is a separate revision label."
            )
        return self

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
    """Combine vector and keyword rankings over ticker-filtered chunks."""

    vector_store: Any
    k: int = DEFAULT_RETRIEVAL_K
    score_threshold: float = DEFAULT_RELEVANCE_THRESHOLD
    tickers: list[str] | None = None
    candidate_k: int = DEFAULT_CANDIDATE_K
    keyword_documents: list[Document] = Field(default_factory=list)

    def _report_scope(self, query: str) -> dict[str, str | int]:
        """Treat an explicit annual report and year as source constraints."""
        if not re.search(ANNUAL_REPORT_PATTERN, query, re.IGNORECASE):
            return {}
        scope: dict[str, str | int] = {"document_type": "annual_report"}
        years = set(re.findall(r"\b(?:19|20)\d{2}\b", query))
        if len(years) == 1:
            scope["fiscal_year"] = int(years.pop())
        return scope

    def _metadata_filter(self, query: str = "") -> dict[str, Any] | None:
        """Apply the same ticker/report constraints as keyword retrieval."""
        clauses: list[dict[str, Any]] = []
        if self.tickers:
            clauses.append({"$or": [
                *({"ticker_scope": ticker} for ticker in self.tickers),
                {"ticker_scope": GENERIC_TICKER_SCOPE},
            ]})
        clauses.extend({key: value} for key, value in self._report_scope(query).items())
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def _search_text(self, query: str, documents: list[Document]) -> str:
        """Search the topic after company/report identity is scoped by metadata."""
        scope = self._report_scope(query)
        if not scope:
            return query
        identity_terms = set()
        if self.tickers:
            for doc in documents:
                if doc.metadata.get("ticker_scope") in self.tickers:
                    identity_terms.update(_keyword_tokens(doc.metadata.get("company", "")))
            identity_terms.update(ticker.lower() for ticker in self.tickers)
        topic = query
        if scope:
            topic = re.sub(ANNUAL_REPORT_PATTERN, " ", topic, flags=re.IGNORECASE)
            identity_terms.add(str(scope.get("fiscal_year", "")))
        terms = [term for term in _keyword_tokens(topic) if term not in identity_terms]
        return " ".join(terms) or query

    def _expand_pdf_context(self, document: Document) -> Document:
        """Include adjacent chunks from the same PDF page around a search hit.

        A list or sentence can cross a chunk boundary. At most three original
        chunks are joined, keeping evidence bounded and page metadata intact.
        At page boundaries, shift the window toward available neighbors rather
        than dropping context because the hit has no preceding/following chunk.
        """
        metadata = document.metadata
        if metadata.get("source_format") != "pdf" or "start_index" not in metadata:
            return document
        page_chunks = sorted(
            (chunk for chunk in self.keyword_documents
             if chunk.metadata.get("source") == metadata.get("source")
             and chunk.metadata.get("page_number") == metadata.get("page_number")),
            key=lambda chunk: chunk.metadata["start_index"],
        )
        hit_index = next((i for i, chunk in enumerate(page_chunks)
                          if chunk.metadata["start_index"] == metadata["start_index"]), None)
        if hit_index is None:
            return document
        window_size = min(3, len(page_chunks))
        window_start = min(max(0, hit_index - 1), len(page_chunks) - window_size)
        neighbors = page_chunks[window_start:window_start + window_size]
        content = neighbors[0].page_content
        start = neighbors[0].metadata["start_index"]
        end = start + len(content)
        for neighbor in neighbors[1:]:
            offset = neighbor.metadata["start_index"]
            content += ("\n" if offset > end else "") + neighbor.page_content[max(0, end - offset):]
            end = offset + len(neighbor.page_content)
        return Document(
            id=document.id,
            page_content=content,
            metadata={**metadata, "start_index": start,
                      "matched_start_index": metadata["start_index"]},
        )

    def _get_relevant_documents(self, query: str) -> list[Document]:
        """Fuse BM25 and vector candidates, then return at most k chunks.

        Reciprocal rank fusion uses positions rather than comparing incompatible
        BM25 and cosine scores. Keyword ranks have twice the weight to help
        precise filing terms compete with semantically similar boilerplate.
        The cosine threshold applies to vector candidates only; keyword search
        can recover passages that vector search did not select.
        """
        allowed_scopes = {*self.tickers, GENERIC_TICKER_SCOPE} if self.tickers else None
        report_scope = self._report_scope(query)
        keyword_documents = [
            doc for doc in self.keyword_documents
            if (allowed_scopes is None
                or doc.metadata.get("ticker_scope", GENERIC_TICKER_SCOPE) in allowed_scopes)
            and all(doc.metadata.get(key) == value for key, value in report_scope.items())
        ]
        search_text = self._search_text(query, keyword_documents)
        matches = self.vector_store.similarity_search_with_relevance_scores(
            search_text,
            k=max(self.k, self.candidate_k),
            filter=self._metadata_filter(query),
        )
        keyword_matches = _keyword_search(
            search_text, keyword_documents, max(self.k, self.candidate_k),
        )
        candidates: dict[str, Document] = {}
        scores: dict[str, float] = {}
        evidence_scores: dict[str, dict[str, float]] = {}
        for ranking, weight, score_name in (
            ([(doc, score) for doc, score in matches if score >= self.score_threshold],
             1.0, "relevance_score"),
            (keyword_matches, 2.0, "keyword_score"),
        ):
            for rank, (document, score) in enumerate(ranking, start=1):
                chunk_id = document.id or _stable_chunk_id(document)
                candidates[chunk_id] = document
                scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (60 + rank)
                evidence_scores.setdefault(chunk_id, {})[score_name] = round(score, 4)

        best_ids = sorted(scores, key=scores.get, reverse=True)[:self.k]
        return [
            self._expand_pdf_context(Document(
                id=chunk_id,
                page_content=candidates[chunk_id].page_content,
                metadata={
                    **candidates[chunk_id].metadata,
                    **evidence_scores[chunk_id],
                    "retrieval_score": round(scores[chunk_id], 6),
                },
            ))
            for chunk_id in best_ids
        ]


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
    identity = f"{embedding_type.__module__}.{embedding_type.__qualname__}"
    # Include the local model/settings when exposed by an embedding wrapper.
    # Do not serialize the whole instance: it may contain credentials or counters.
    configuration = {
        key: getattr(embeddings, key)
        for key in ("model_name", "encode_kwargs")
        if hasattr(embeddings, key)
    }
    if configuration:
        identity += ":" + json.dumps(configuration, sort_keys=True, default=str)
    return identity


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


def _create_persistent_vector_store(
    persist_directory: Path,
    collection_name: str,
    embeddings: Embeddings | None,
    *,
    client: Any | None = None,
):
    """Create or open a local Chroma collection using cosine distance."""
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
        collection_configuration={"hnsw": {"space": "cosine"}},
        client=client,
    )


def _load_yaml_mapping(content: str, source_name: str) -> dict[str, Any]:
    """Load a YAML mapping from a metadata source."""
    import yaml

    raw_metadata = yaml.safe_load(content) or {}
    if not isinstance(raw_metadata, dict):
        raise ValueError(
            f"Knowledge document metadata must be a mapping: {source_name}"
        )
    return raw_metadata


def _build_document_metadata(
    raw_metadata: dict[str, Any],
    source_name: str,
) -> dict[str, Any]:
    """Validate declared metadata and add internal retrieval fields."""
    metadata = KnowledgeDocumentMetadata.model_validate(raw_metadata)
    document_metadata = metadata.model_dump(mode="json", exclude_none=True)
    document_metadata["source"] = source_name
    document_metadata["ticker_scope"] = document_metadata.get(
        "ticker",
        GENERIC_TICKER_SCOPE,
    )
    return document_metadata


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

    document_metadata = _build_document_metadata(
        _load_yaml_mapping(
            "".join(lines[1:closing_index]),
            source_name,
        ),
        source_name,
    )

    body = "".join(lines[closing_index + 1:]).lstrip()
    return document_metadata, body


def _pdf_metadata_path(pdf_path: Path) -> Path:
    """Find the required YAML sidecar for a PDF document."""
    candidates = [pdf_path.with_suffix(".yaml"), pdf_path.with_suffix(".yml")]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        raise FileNotFoundError(
            f"PDF knowledge document requires a YAML sidecar: {pdf_path} "
            f"(expected {candidates[0].name})"
        )
    if len(existing) > 1:
        raise ValueError(
            f"PDF knowledge document has multiple YAML sidecars: {pdf_path}"
        )
    return existing[0]


def _load_pdf_documents(pdf_path: Path, source_name: str) -> list[Document]:
    """Extract one Document per non-empty PDF page with page metadata."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF ingestion requires the pypdf package. Install the project "
            "dependencies before loading PDF knowledge documents."
        ) from exc

    metadata_path = _pdf_metadata_path(pdf_path)
    metadata = _build_document_metadata(
        _load_yaml_mapping(
            metadata_path.read_text(encoding="utf-8"),
            str(metadata_path),
        ),
        source_name,
    )
    reader = PdfReader(str(pdf_path))
    page_count = len(reader.pages)
    documents = []
    page_texts = [(page.extract_text() or "").strip() for page in reader.pages]
    # Repeated first/last lines are typically running titles or page footers.
    # Normalize page numbers for counting but preserve the original body text.
    def margin_key(line: str) -> str:
        normalized = " ".join(line.split())
        if normalized.isdecimal():
            return "#page"
        return re.sub(r"(?<=\|)\s*\d+\s*$", " #page", normalized)

    margin_counts: Counter[str] = Counter()
    for text in page_texts:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        margin_counts.update({margin_key(line) for line in lines[:1] + lines[-1:]})
    repeated_margins = {
        key for key, count in margin_counts.items()
        if count >= max(3, page_count * 0.2)
    }

    for page_number, text in enumerate(page_texts, start=1):
        lines = text.splitlines()
        while lines and (not lines[0].strip() or margin_key(lines[0]) in repeated_margins):
            lines.pop(0)
        while lines and (not lines[-1].strip() or margin_key(lines[-1]) in repeated_margins):
            lines.pop()
        page_content = "\n".join(lines).strip()
        if not page_content:
            continue

        documents.append(
            Document(
                page_content=page_content,
                metadata={
                    **metadata,
                    "source_format": "pdf",
                    "page_number": page_number,
                    "page_count": page_count,
                },
            )
        )

    if not documents:
        raise ValueError(f"PDF contains no extractable text: {pdf_path}")

    return documents


def load_knowledge_documents(
    knowledge_dir: str | Path | None = None,
    *,
    allow_empty: bool = False,
) -> list[Document]:
    """Load Markdown and PDF files with validated metadata."""
    directory = Path(knowledge_dir) if knowledge_dir else DEFAULT_KNOWLEDGE_DIR

    if not directory.exists():
        raise FileNotFoundError(f"Knowledge directory does not exist: {directory}")

    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".pdf"}
    )
    if not paths and not allow_empty:
        raise FileNotFoundError(
            f"No Markdown or PDF documents found in: {directory}"
        )

    documents = []
    for path in paths:
        source_name = path.relative_to(directory).as_posix()
        if path.suffix.lower() == ".pdf":
            documents.extend(_load_pdf_documents(path, source_name))
            continue

        content = path.read_text(encoding="utf-8")
        metadata, body = _parse_markdown_frontmatter(content, source_name)
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
    return identify_chunks(splitter.split_documents(documents))


def create_embeddings(model_name: str | None = None) -> Embeddings:
    """Create local embeddings without requiring a separate embedding API key."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name
        or os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        encode_kwargs={"normalize_embeddings": True},
    )


def _prepare_knowledge_index(
    knowledge_dir: str | Path | None = None,
    embeddings: Embeddings | None = None,
    *,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
    persist_directory: str | Path | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    force_rebuild: bool = False,
    require_embedding_function: bool = False,
) -> _PreparedKnowledgeIndex:
    """Load documents and build or reuse their persistent vector index."""
    knowledge_directory = (
        Path(knowledge_dir) if knowledge_dir else DEFAULT_KNOWLEDGE_DIR
    )
    documents = load_knowledge_documents(knowledge_directory, allow_empty=True)
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    embedding_function = embeddings
    if require_embedding_function and embedding_function is None:
        embedding_function = create_embeddings()
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
    from chromadb import PersistentClient

    client = PersistentClient(path=str(index_directory))
    changes = synchronize_index(
        client=client, collection_name=collection_name, documents=documents, chunks=chunks,
        manifest_path=manifest_path, signature=signature, embedding_identity=embedding_identity,
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, embeddings=embedding_function,
        embedding_factory=create_embeddings, force_rebuild=force_rebuild,
    )
    vector_store = _create_persistent_vector_store(
        index_directory, collection_name, embedding_function, client=client,
    )

    return _PreparedKnowledgeIndex(
        vector_store=vector_store,
        knowledge_directory=knowledge_directory,
        persist_directory=index_directory,
        collection_name=collection_name,
        documents=documents,
        chunks=chunks,
        manifest_path=manifest_path,
        rebuilt=changes.rebuilt,
        changes=changes,
    )


def ingest_knowledge_base(
    knowledge_dir: str | Path | None = None,
    embeddings: Embeddings | None = None,
    *,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
    persist_directory: str | Path | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    force_rebuild: bool = False,
) -> KnowledgeIndexReport:
    """Explicitly build or refresh the persistent knowledge index."""
    prepared = _prepare_knowledge_index(
        knowledge_dir,
        embeddings,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        persist_directory=persist_directory,
        collection_name=collection_name,
        force_rebuild=force_rebuild,
        require_embedding_function=False,
    )
    return KnowledgeIndexReport(
        knowledge_directory=prepared.knowledge_directory,
        persist_directory=prepared.persist_directory,
        collection_name=prepared.collection_name,
        document_count=len(prepared.documents),
        chunk_count=len(prepared.chunks),
        rebuilt=prepared.rebuilt,
        manifest_path=prepared.manifest_path,
        changes=prepared.changes,
    )


def build_knowledge_retriever(
    knowledge_dir: str | Path | None = None,
    embeddings: Embeddings | None = None,
    *,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
    k: int = DEFAULT_RETRIEVAL_K,
    score_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    candidate_k: int = DEFAULT_CANDIDATE_K,
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
    if candidate_k <= 0:
        raise ValueError("candidate_k must be greater than zero.")

    requested_tickers = list(dict.fromkeys(
        ticker.strip().upper()
        for ticker in (tickers or [])
        if ticker.strip()
    ))

    prepared = _prepare_knowledge_index(
        knowledge_dir,
        embeddings,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        persist_directory=persist_directory,
        collection_name=collection_name,
        force_rebuild=force_rebuild,
        require_embedding_function=True,
    )

    return ScoreThresholdRetriever(
        vector_store=prepared.vector_store,
        k=k,
        score_threshold=score_threshold,
        tickers=requested_tickers or None,
        candidate_k=candidate_k,
        keyword_documents=[
            Document(
                id=_stable_chunk_id(chunk),
                page_content=chunk.page_content,
                metadata=chunk.metadata,
            )
            for chunk in prepared.chunks
        ],
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
