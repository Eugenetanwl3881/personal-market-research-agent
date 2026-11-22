from datetime import date
from pathlib import Path
from urllib.error import HTTPError

import pytest
import yaml

from market_research.sources import (
    APPLE_REPORT_URL_TEMPLATE,
    fetch_apple_annual_report,
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
