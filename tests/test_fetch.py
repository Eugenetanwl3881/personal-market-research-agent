import sys
from types import SimpleNamespace

import pytest

from market_research import fetch


@pytest.mark.parametrize("ticker", ["AAPL", "MSFT"])
def test_fetch_cli_dispatches_to_requested_company(monkeypatch, capsys, tmp_path, ticker):
    calls = []

    def requested_fetcher(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(ticker=ticker, report_year=2025,
                               pdf_path=tmp_path / "report.pdf",
                               metadata_path=tmp_path / "report.yaml",
                               source_url="https://example.com/report.pdf")

    def wrong_fetcher(**kwargs):
        raise AssertionError("Wrong company adapter selected")

    monkeypatch.setattr(fetch, "fetch_apple_annual_report", requested_fetcher if ticker == "AAPL" else wrong_fetcher)
    monkeypatch.setattr(fetch, "fetch_microsoft_annual_report", requested_fetcher if ticker == "MSFT" else wrong_fetcher)
    monkeypatch.setattr(sys, "argv", ["fetch", "--ticker", ticker, "--year", "2025", "--knowledge-dir", str(tmp_path)])
    fetch.main()
    assert calls == [{"report_year": 2025, "knowledge_dir": tmp_path, "force": False, "timeout": 30.0}]
    assert f"Ticker: {ticker}" in capsys.readouterr().out
