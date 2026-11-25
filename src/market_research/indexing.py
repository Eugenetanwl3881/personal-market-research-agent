"""Source/chunk identity and incremental synchronization of a local Chroma index."""

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


INDEX_VERSION = 2
BATCH_SIZE = 128


def _digest(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def document_id(source: str) -> str:
    """Identify a source by its path relative to the selected knowledge root."""
    return "doc_" + _digest(source)


def content_hash(text: str) -> str:
    """Identify exact extracted text independently of source and metadata."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_chunk_id(chunk: Document) -> str:
    """Keep provenance distinct while excluding mutable offsets and metadata."""
    return "chunk_" + _digest({
        "document_id": document_id(chunk.metadata.get("source", "Unknown source")),
        "page_number": chunk.metadata.get("page_number"),
        "content_hash": content_hash(chunk.page_content),
        "occurrence": chunk.metadata.get("chunk_occurrence", 0),
    })


def identify_chunks(chunks: list[Document]) -> list[Document]:
    """Assign source/content identities and disambiguate repeated text per page."""
    occurrences = Counter()
    identified = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "Unknown source")
        text_hash = content_hash(chunk.page_content)
        key = (source, chunk.metadata.get("page_number"), text_hash)
        item = Document(page_content=chunk.page_content, metadata={
            **chunk.metadata,
            "document_id": document_id(source),
            "content_hash": text_hash,
            "chunk_occurrence": occurrences[key],
        })
        item.id = stable_chunk_id(item)
        occurrences[key] += 1
        identified.append(item)
    return identified


@dataclass(frozen=True)
class IndexChanges:
    """Counts describe this synchronization, not historical cumulative work."""

    changed: bool = False
    rebuilt: bool = False
    rebuild_reason: str | None = None
    source_count: int = 0
    added_chunks: int = 0
    metadata_updates: int = 0
    deleted_chunks: int = 0
    unchanged_chunks: int = 0
    embedded_texts: int = 0
    reused_embeddings: int = 0
    duplicate_chunks: int = 0
    duplicate_sources: int = 0


def _read_manifest(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_manifest(path: Path, manifest: dict) -> None:
    """Replace the manifest only after synchronization has succeeded."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(manifest, temporary, indent=2, ensure_ascii=False, sort_keys=True)
            temporary.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _source_records(documents: list[Document], chunks: list[Document]) -> dict:
    pages = defaultdict(list)
    chunk_ids = defaultdict(list)
    for doc in documents:
        pages[doc.metadata["source"]].append({
            "page_number": doc.metadata.get("page_number"), "text": doc.page_content,
        })
    for chunk in chunks:
        chunk_ids[chunk.metadata["source"]].append(chunk.id)
    return {
        document_id(source): {
            "source": source, "content_hash": _digest(source_pages),
            "chunk_ids": chunk_ids[source],
        }
        for source, source_pages in pages.items()
    }


def _batches(items: list):
    for start in range(0, len(items), BATCH_SIZE):
        yield items[start:start + BATCH_SIZE]


def synchronize_index(
    *,
    client: Any,
    collection_name: str,
    documents: list[Document],
    chunks: list[Document],
    manifest_path: Path,
    signature: str,
    embedding_identity: str,
    chunk_size: int,
    chunk_overlap: int,
    embeddings: Embeddings | None,
    embedding_factory: Callable[[], Embeddings],
    force_rebuild: bool,
) -> IndexChanges:
    """Upsert source records, reuse vectors by content hash, and delete stale IDs.

    Text records remain distinct for each source/page/occurrence. Embeddings are
    reused only for exact text under a known compatible embedding configuration.
    This is a single-writer operation; the manifest is not a multi-record DB
    transaction. A failed partial write is reconciled on the next run.
    """
    configuration = {"hnsw": {"space": "cosine"}}
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=None, configuration=configuration,
    )
    previous = _read_manifest(manifest_path)
    stored = collection.get(include=["documents", "metadatas"])
    existing = {
        record_id: {"text": text, "metadata": metadata or {}}
        for record_id, text, metadata in zip(stored["ids"], stored["documents"], stored["metadatas"])
    }
    desired = {chunk.id: chunk for chunk in chunks}
    sources = _source_records(documents, chunks)
    duplicate_sources = len(sources) - len({source["content_hash"] for source in sources.values()})
    duplicate_chunks = len(chunks) - len({chunk.metadata["content_hash"] for chunk in chunks})
    compatible = (
        previous.get("index_version") == INDEX_VERSION
        and previous.get("embedding_identity") == embedding_identity
    )
    rebuild = force_rebuild or not compatible
    reason = ("forced" if force_rebuild else "schema_or_embedding_change" if existing else "new_index") if rebuild else None
    added = list(desired) if rebuild else [key for key in desired if key not in existing]
    updates = [] if rebuild else [
        key for key, chunk in desired.items() if key in existing
        and existing[key]["metadata"] != chunk.metadata
    ]
    # A hash-derived ID should identify the same text. If stored data is damaged,
    # replace the record through the vector path rather than merely its metadata.
    corrupted = []
    if not rebuild:
        corrupted = [key for key in desired if key in existing and existing[key]["text"] != desired[key].page_content]
        added.extend(corrupted)
        updates = [key for key in updates if key not in corrupted]
    stale = list(existing) if rebuild else [key for key in existing if key not in desired]

    # Reuse vectors from any active old record, even if that source is about to
    # be removed. This also handles renames and exact copies without extra work.
    needed_hashes = {desired[key].metadata["content_hash"] for key in added}
    cached_ids = {}
    if not rebuild:
        for key, record in existing.items():
            if record["text"] is not None and key not in corrupted:
                text_hash = content_hash(record["text"])
                if text_hash in needed_hashes:
                    cached_ids.setdefault(text_hash, key)
    vectors = {}
    hash_for_id = {record_id: text_hash for text_hash, record_id in cached_ids.items()}
    for batch in _batches(list(hash_for_id)):
        cached = collection.get(ids=batch, include=["embeddings"])
        for record_id, vector in zip(cached["ids"], cached["embeddings"]):
            vectors[hash_for_id[record_id]] = vector.tolist() if hasattr(vector, "tolist") else list(vector)

    text_for_hash = {desired[key].metadata["content_hash"]: desired[key].page_content for key in added}
    missing_hashes = [text_hash for text_hash in text_for_hash if text_hash not in vectors]
    # Finish embedding work before resetting/deleting any existing source data.
    if missing_hashes:
        embedding_function = embeddings if embeddings is not None else embedding_factory()
        for batch in _batches(missing_hashes):
            encoded = embedding_function.embed_documents([text_for_hash[key] for key in batch])
            if len(encoded) != len(batch) or any(len(vector) == 0 for vector in encoded):
                raise RuntimeError("Embedding provider returned an invalid number of nonempty vectors.")
            vectors.update(zip(batch, encoded))

    if rebuild:
        client.delete_collection(collection_name)
        collection = client.get_or_create_collection(
            name=collection_name, embedding_function=None, configuration=configuration,
        )
    for batch in _batches(added):
        collection.upsert(
            ids=batch,
            documents=[desired[key].page_content for key in batch],
            metadatas=[desired[key].metadata for key in batch],
            embeddings=[vectors[desired[key].metadata["content_hash"]] for key in batch],
        )
    for batch in _batches(updates):
        # Chroma merges update metadata. Explicit None values remove old fields
        # that are no longer present (e.g. company/source_url/ticker).
        metadatas = []
        for key in batch:
            removed = {name: None for name in existing[key]["metadata"] if name not in desired[key].metadata}
            metadatas.append({**removed, **desired[key].metadata})
        collection.update(ids=batch, metadatas=metadatas)
    if not rebuild:
        for batch in _batches(stale):
            collection.delete(ids=batch)

    manifest = {
        "index_version": INDEX_VERSION, "index_signature": signature,
        "embedding_identity": embedding_identity,
        "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
        "chunk_count": len(chunks), "sources": sources,
        "chunks": {
            chunk.id: {"document_id": chunk.metadata["document_id"],
                       "content_hash": chunk.metadata["content_hash"],
                       "metadata_hash": _digest(chunk.metadata)} for chunk in chunks
        },
    }
    if manifest != previous:
        _write_manifest(manifest_path, manifest)
    changed = rebuild or bool(added or updates or stale) or previous.get("index_signature") != signature
    return IndexChanges(
        changed=changed, rebuilt=rebuild, rebuild_reason=reason,
        source_count=len(sources), added_chunks=len(added), metadata_updates=len(updates),
        deleted_chunks=len(stale), unchanged_chunks=len(chunks) - len(added) - len(updates),
        embedded_texts=len(missing_hashes), reused_embeddings=len(chunks) - len(missing_hashes),
        duplicate_chunks=duplicate_chunks, duplicate_sources=duplicate_sources,
    )
