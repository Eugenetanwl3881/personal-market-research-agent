from market_research.data import (
    classify_freshness,
    format_timestamp,
    format_trading_date,
    publisher_from_url,
)


def test_format_timestamp_is_readable():
    assert format_timestamp("2026-08-26T13:30:00-04:00") == (
        "August 26, 2026 at 1:30 PM UTC-04:00"
    )


def test_publisher_from_url():
    assert publisher_from_url("https://www.example.com/article") == "example.com"


def test_format_trading_date_does_not_imply_midnight_close():
    assert format_trading_date("2026-08-27T00:00:00-04:00") == "August 27, 2026"


def test_classify_same_trading_day():
    assert classify_freshness(
        "2026-08-26T13:30:00+00:00",
        "2026-08-26T16:00:00+00:00",
    ) == "same_trading_day"


def test_classify_prior_trading_session():
    assert classify_freshness(
        "2026-08-28T20:00:00+00:00",
        "2026-08-30T12:00:00+00:00",
    ) == "prior_trading_session"


def test_weekend_does_not_make_friday_close_stale():
    assert classify_freshness(
        "2026-08-28T00:00:00-04:00",
        "2026-08-30T12:00:00+00:00",
    ) == "prior_trading_session"


def test_classify_stale_price():
    assert classify_freshness(
        "2026-08-20T20:00:00+00:00",
        "2026-08-30T12:00:00+00:00",
    ) == "stale"
