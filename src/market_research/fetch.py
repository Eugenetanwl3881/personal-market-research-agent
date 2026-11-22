"""CLI for downloading official documents into the knowledge base."""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from .sources import fetch_apple_annual_report


def build_parser() -> argparse.ArgumentParser:
    """Create the document-fetching CLI parser."""
    parser = argparse.ArgumentParser(
        description="Download an official market-research knowledge document."
    )
    parser.add_argument(
        "--ticker",
        required=True,
        choices=["AAPL"],
        help="Ticker supported by the current source adapter.",
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Specific annual-report year. Defaults to automatic recent-year discovery.",
    )
    parser.add_argument(
        "--knowledge-dir",
        type=Path,
        help="Knowledge directory. Defaults to ./knowledge.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing report with the downloaded copy.",
    )
    return parser


def main() -> None:
    """Download a supported official document."""
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    try:
        report = fetch_apple_annual_report(
            report_year=args.year,
            knowledge_dir=args.knowledge_dir,
            force=args.force,
            timeout=args.timeout,
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    print("Downloaded knowledge document.")
    print(f"- Ticker: {report.ticker}")
    print(f"- Report year: {report.report_year}")
    print(f"- PDF: {report.pdf_path}")
    print(f"- Metadata: {report.metadata_path}")
    print(f"- Source: {report.source_url}")
    print("Run `market-research-index --force` to refresh the local vector index.")


if __name__ == "__main__":
    main()
