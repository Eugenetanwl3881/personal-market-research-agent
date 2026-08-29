from market_research.data import format_timestamp, publisher_from_url


def test_format_timestamp_is_readable():
    assert format_timestamp("2026-08-26T13:30:00-04:00") == (
        "August 26, 2026 at 1:30 PM UTC-04:00"
    )


def test_publisher_from_url():
    assert publisher_from_url("https://www.example.com/article") == "example.com"
