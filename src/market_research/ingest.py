import argparse
from pathlib import Path

from dotenv import load_dotenv

from .rag import (
    DEFAULT_COLLECTION_NAME,
    ingest_knowledge_base,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for knowledge-index ingestion."""
    parser = argparse.ArgumentParser(
        description="Build or refresh the local market-research knowledge index."
    )
    parser.add_argument(
        "--knowledge-dir",
        type=Path,
        help="Directory containing Markdown knowledge documents.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        help="Directory where the persistent Chroma index is stored.",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help=f"Chroma collection name (default: {DEFAULT_COLLECTION_NAME}).",
    )
    parser.add_argument(
        "--force-rebuild",
        "--force",
        dest="force_rebuild",
        action="store_true",
        help="Rebuild the index even when the existing index is current.",
    )
    return parser


def main() -> None:
    """Run the explicit knowledge-index ingestion command."""
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    try:
        report = ingest_knowledge_base(
            knowledge_dir=args.knowledge_dir,
            persist_directory=args.index_dir,
            collection_name=args.collection_name,
            force_rebuild=args.force_rebuild,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    action = "Rebuilt" if report.rebuilt else "Reused"
    print(f"{action} knowledge index.")
    print(f"- Knowledge directory: {report.knowledge_directory}")
    print(f"- Index directory: {report.persist_directory}")
    print(f"- Collection: {report.collection_name}")
    print(f"- Documents: {report.document_count}")
    print(f"- Chunks: {report.chunk_count}")
    print(f"- Manifest: {report.manifest_path}")


if __name__ == "__main__":
    main()
