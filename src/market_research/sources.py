"""Download official company documents for the local knowledge base."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml

from .rag import KnowledgeDocumentMetadata


APPLE_TICKER = "AAPL"
APPLE_COMPANY = "Apple Inc."
APPLE_REPORT_URL_TEMPLATE = (
    "https://s2.q4cdn.com/470004039/files/doc_financials/"
    "{year}/ar/_10-K-{year}-As-Filed.pdf"
)
MICROSOFT_TICKER = "MSFT"
MICROSOFT_COMPANY = "Microsoft Corporation"
# Verified official investor-relations PDFs. Microsoft uses opaque asset URLs,
# so do not guess URLs or substitute a different fiscal year.
MICROSOFT_REPORT_URLS = {
    2025: "https://microsoft.gcs-web.com/static-files/9640e86d-a99f-4502-934d-1d3c883fa786",
}
DEFAULT_DOWNLOAD_TIMEOUT = 30.0
MAX_REPORT_SIZE_BYTES = 50 * 1024 * 1024


@dataclass(frozen=True)
class DownloadedReport:
    """Paths and source information for a downloaded knowledge document."""

    ticker: str
    report_year: int
    pdf_path: Path
    metadata_path: Path
    source_url: str


def apple_annual_report_url(report_year: int) -> str:
    """Return the official Apple investor-relations PDF URL for a year."""
    if report_year < 2000:
        raise ValueError("report_year must be 2000 or later.")

    return APPLE_REPORT_URL_TEMPLATE.format(year=report_year)


def _candidate_years(
    report_year: int | None,
    current_year: int | None = None,
    lookback: int = 3,
) -> list[int]:
    """Return explicit or recent report years to try during discovery."""
    if report_year is not None:
        return [report_year]

    if lookback <= 0:
        raise ValueError("lookback must be greater than zero.")

    year = current_year or date.today().year
    return list(range(year, year - lookback, -1))


def _download_pdf(
    url: str,
    *,
    timeout: float,
    opener: Callable[..., Any],
) -> bytes:
    """Download and validate one PDF response."""
    user_agent = os.getenv(
        "MARKET_RESEARCH_DOWNLOAD_USER_AGENT",
        "market-research-assistant/0.1.0",
    )
    request = Request(
        url,
        headers={
            "Accept": "application/pdf",
            "User-Agent": user_agent,
        },
    )

    try:
        with opener(request, timeout=timeout) as response:
            content = response.read(MAX_REPORT_SIZE_BYTES + 1)
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Could not download report from {url}: {exc}") from exc

    if len(content) > MAX_REPORT_SIZE_BYTES:
        raise ValueError(
            f"Report exceeds the {MAX_REPORT_SIZE_BYTES // (1024 * 1024)} MB limit: {url}"
        )

    if not content.startswith(b"%PDF"):
        raise ValueError(f"Downloaded content is not a PDF: {url}")

    return content


def _atomic_write(path: Path, content: bytes | str) -> None:
    """Write a file through a temporary sibling and replace it atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if isinstance(content, bytes) else "w"
    temp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode=mode,
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
            encoding=None if isinstance(content, bytes) else "utf-8",
        ) as temporary:
            temporary.write(content)
            temp_path = temporary.name
        os.replace(temp_path, path)
    finally:
        if temp_path and Path(temp_path).exists():
            Path(temp_path).unlink()


def _metadata_for_apple_report(report_year: int, source_url: str) -> dict[str, Any]:
    """Build and validate the metadata written beside an Apple report."""
    metadata = {
        "ticker": APPLE_TICKER,
        "company": APPLE_COMPANY,
        "document_type": "annual_report",
        "source_name": f"Apple {report_year} Form 10-K Annual Report",
        "source_url": source_url,
        "last_reviewed": date.today().isoformat(),
        "fiscal_year": report_year,
        "document_version": "1",
    }
    validated = KnowledgeDocumentMetadata(**metadata)
    return validated.model_dump(mode="json", exclude_none=True)


def fetch_apple_annual_report(
    *,
    report_year: int | None = None,
    knowledge_dir: str | Path | None = None,
    force: bool = False,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT,
    current_year: int | None = None,
    opener: Callable[..., Any] = urlopen,
) -> DownloadedReport:
    """Discover and download an Apple annual report plus its YAML sidecar.

    When ``report_year`` is omitted, recent years are tried newest first. The
    generated files match the layout expected by ``load_knowledge_documents``.
    """
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero.")

    directory = Path(knowledge_dir) if knowledge_dir else Path("knowledge")
    apple_directory = directory / "raw" / "apple"
    candidates = _candidate_years(report_year, current_year=current_year)
    failures: list[str] = []

    for candidate_year in candidates:
        source_url = apple_annual_report_url(candidate_year)
        try:
            content = _download_pdf(
                source_url,
                timeout=timeout,
                opener=opener,
            )
        except (RuntimeError, ValueError) as exc:
            failures.append(str(exc))
            continue

        pdf_path = apple_directory / f"annual_report_{candidate_year}.pdf"
        metadata_path = pdf_path.with_suffix(".yaml")
        if pdf_path.exists() and not force:
            raise FileExistsError(
                f"Report already exists: {pdf_path}. Use force=True to replace it."
            )

        metadata = _metadata_for_apple_report(candidate_year, source_url)
        _atomic_write(pdf_path, content)
        _atomic_write(
            metadata_path,
            yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True),
        )
        return DownloadedReport(
            ticker=APPLE_TICKER,
            report_year=candidate_year,
            pdf_path=pdf_path,
            metadata_path=metadata_path,
            source_url=source_url,
        )

    attempted = ", ".join(str(year) for year in candidates)
    details = "; ".join(failures)
    raise RuntimeError(
        f"Could not find an Apple annual report for years {attempted}. {details}"
    )


def fetch_microsoft_annual_report(
    *,
    report_year: int | None = None,
    knowledge_dir: str | Path | None = None,
    force: bool = False,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT,
    opener: Callable[..., Any] = urlopen,
) -> DownloadedReport:
    """Download a verified Microsoft Form 10-K PDF and validated sidecar.

    Without a year, use the newest configured PDF, not a claim about the latest
    report Microsoft has published. Currently the verified registry covers 2025.
    """
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero.")
    year = max(MICROSOFT_REPORT_URLS) if report_year is None else report_year
    if year not in MICROSOFT_REPORT_URLS:
        supported = ", ".join(str(value) for value in sorted(MICROSOFT_REPORT_URLS))
        raise ValueError(
            f"No official Microsoft PDF is configured for fiscal year {year}. "
            f"Configured years: {supported}."
        )

    directory = Path(knowledge_dir) if knowledge_dir else Path("knowledge")
    pdf_path = directory / "raw" / "microsoft" / f"annual_report_{year}.pdf"
    metadata_path = pdf_path.with_suffix(".yaml")
    if (pdf_path.exists() or metadata_path.exists()) and not force:
        raise FileExistsError(
            f"Report or metadata already exists: {pdf_path}. Use --force to replace it."
        )
    source_url = MICROSOFT_REPORT_URLS[year]
    metadata = KnowledgeDocumentMetadata(
        ticker=MICROSOFT_TICKER,
        company=MICROSOFT_COMPANY,
        document_type="annual_report",
        source_name=f"Microsoft {year} Form 10-K Annual Report",
        source_url=source_url,
        fiscal_year=year,
        document_version="1",
        last_reviewed=date.today(),
    ).model_dump(mode="json", exclude_none=True)
    content = _download_pdf(source_url, timeout=timeout, opener=opener)
    _atomic_write(pdf_path, content)
    _atomic_write(metadata_path, yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True))
    return DownloadedReport(
        ticker=MICROSOFT_TICKER,
        report_year=year,
        pdf_path=pdf_path,
        metadata_path=metadata_path,
        source_url=source_url,
    )
