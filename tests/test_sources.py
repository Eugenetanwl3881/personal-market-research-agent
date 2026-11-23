from datetime import date
from pathlib import Path
from urllib.error import HTTPError

import pytest
import yaml

from market_research.sources import (
    APPLE_REPORT_URL_TEMPLATE,
    MICROSOFT_REPORT_URLS,
    fetch_apple_annual_report,
    fetch_microsoft_annual_report,
)


class FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self, limit: int) -> bytes:
        return self.content[:limit]


def test_fetch_apple_report_writes_pdf_and_validated_sidecar(tmp_path: Path):
    captured = {}

    def fake_opener(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return FakeResponse(b"%PDF-1.7\nexample")

    report = fetch_apple_annual_report(
        report_year=2025,
        knowledge_dir=tmp_path,
        current_year=2026,
        opener=fake_opener,
    )

    assert captured == {
        "url": APPLE_REPORT_URL_TEMPLATE.format(year=2025),
        "timeout": 30.0,
    }
    assert report.pdf_path.read_bytes().startswith(b"%PDF")

    metadata = yaml.safe_load(report.metadata_path.read_text(encoding="utf-8"))
    assert metadata == {
        "ticker": "AAPL",
        "company": "Apple Inc.",
        "document_type": "annual_report",
        "source_name": "Apple 2025 Form 10-K Annual Report",
        "source_url": APPLE_REPORT_URL_TEMPLATE.format(year=2025),
        "last_reviewed": date.today().isoformat(),
        "fiscal_year": 2025,
        "document_version": "1",
    }


def test_fetch_apple_report_discovers_latest_available_year(tmp_path: Path):
    attempted_urls = []

    def fake_opener(request, timeout):
        attempted_urls.append(request.full_url)
        if request.full_url.endswith("2026-As-Filed.pdf"):
            raise HTTPError(request.full_url, 404, "not found", {}, None)
        return FakeResponse(b"%PDF-1.7\nexample")

    report = fetch_apple_annual_report(
        knowledge_dir=tmp_path,
        current_year=2026,
        opener=fake_opener,
    )

    assert report.report_year == 2025
    assert len(attempted_urls) == 2


def test_fetch_apple_report_rejects_non_pdf(tmp_path: Path):
    def fake_opener(request, timeout):
        return FakeResponse(b"<!doctype html>")

    with pytest.raises(RuntimeError, match="Could not find an Apple annual report"):
        fetch_apple_annual_report(
            report_year=2025,
            knowledge_dir=tmp_path,
            opener=fake_opener,
        )


def test_fetch_apple_report_does_not_overwrite_without_force(tmp_path: Path):
    def fake_opener(request, timeout):
        return FakeResponse(b"%PDF-1.7\nexample")

    fetch_apple_annual_report(
        report_year=2025,
        knowledge_dir=tmp_path,
        opener=fake_opener,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        fetch_apple_annual_report(
            report_year=2025,
            knowledge_dir=tmp_path,
            opener=fake_opener,
        )


def test_fetch_microsoft_report_writes_official_source_and_fiscal_metadata(tmp_path: Path):
    urls = []

    def fake_opener(request, timeout):
        urls.append(request.full_url)
        return FakeResponse(b"%PDF-1.7\nMicrosoft report")

    report = fetch_microsoft_annual_report(
        report_year=2025, knowledge_dir=tmp_path, opener=fake_opener,
    )
    metadata = yaml.safe_load(report.metadata_path.read_text(encoding="utf-8"))
    assert urls == [MICROSOFT_REPORT_URLS[2025]]
    assert report.pdf_path == tmp_path / "raw" / "microsoft" / "annual_report_2025.pdf"
    assert report.pdf_path.read_bytes() == b"%PDF-1.7\nMicrosoft report"
    assert report.ticker == "MSFT"
    assert metadata["ticker"] == "MSFT"
    assert metadata["company"] == "Microsoft Corporation"
    assert metadata["fiscal_year"] == 2025
    assert metadata["document_version"] == "1"
    assert metadata["document_type"] == "annual_report"
    assert metadata["source_url"] == MICROSOFT_REPORT_URLS[2025]


def test_fetch_microsoft_does_not_substitute_another_year_or_overwrite(tmp_path: Path):
    def forbidden_opener(*args, **kwargs):
        raise AssertionError("Invalid year or existing report must not download")

    with pytest.raises(ValueError, match="No official Microsoft PDF is configured"):
        fetch_microsoft_annual_report(report_year=2024, knowledge_dir=tmp_path, opener=forbidden_opener)
    directory = tmp_path / "raw" / "microsoft"
    directory.mkdir(parents=True)
    (directory / "annual_report_2025.yaml").write_text("Existing metadata", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        fetch_microsoft_annual_report(report_year=2025, knowledge_dir=tmp_path, opener=forbidden_opener)


def test_fetch_microsoft_rejects_non_pdf_without_writing_files(tmp_path: Path):
    with pytest.raises(ValueError, match="not a PDF"):
        fetch_microsoft_annual_report(
            report_year=2025, knowledge_dir=tmp_path,
            opener=lambda request, timeout: FakeResponse(b"<html>Not the report</html>"),
        )
    assert not (tmp_path / "raw").exists()


def test_fetch_microsoft_default_year_and_explicit_force_replace(tmp_path: Path):
    report = fetch_microsoft_annual_report(
        knowledge_dir=tmp_path,
        opener=lambda request, timeout: FakeResponse(b"%PDF-1.7\nfirst copy"),
    )
    replaced = fetch_microsoft_annual_report(
        knowledge_dir=tmp_path, force=True,
        opener=lambda request, timeout: FakeResponse(b"%PDF-1.7\nreplacement"),
    )
    assert report.report_year == replaced.report_year == max(MICROSOFT_REPORT_URLS)
    assert report.pdf_path == replaced.pdf_path
    assert replaced.pdf_path.read_bytes() == b"%PDF-1.7\nreplacement"
