"""Small, standalone retrieval pipeline for the local knowledge base."""

from pathlib import Path
import os
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore


DEFAULT_KNOWLEDGE_DIR = (
    Path(__file__).resolve().parents[2] / "knowledge"
)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_knowledge_documents(
    knowledge_dir: str | Path | None = None,
) -> list[Document]:
    """Load Markdown files as LangChain documents with stable source metadata."""
    directory = Path(knowledge_dir) if knowledge_dir else DEFAULT_KNOWLEDGE_DIR

    if not directory.exists():
        raise FileNotFoundError(f"Knowledge directory does not exist: {directory}")

    paths = sorted(directory.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No Markdown documents found in: {directory}")

    return [
        Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={
                "source": path.name,
                "document_type": "markdown",
            },
        )
        for path in paths
    ]


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
    k: int = 4,
):
    """Build an in-memory retriever from the local Markdown knowledge base."""
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    documents = load_knowledge_documents(knowledge_dir)
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    vector_store = InMemoryVectorStore(
        embedding=embeddings or create_embeddings(),
    )
    vector_store.add_documents(chunks)
    #k=4 means return the four most relevant chunks for each query.
    return vector_store.as_retriever(search_kwargs={"k": k})


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
