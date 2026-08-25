from market_research.parser import parse_market_request


def test_parses_ticker():
    result = parse_market_request("What is the price of AAPL?")

    assert result["ticker"] == ["AAPL"]
    assert result["shares"] is None


def test_parses_ticker_and_shares():
    result = parse_market_request("How much would 15 shares of MSFT cost?")

    assert result["ticker"] == ["MSFT"]
    assert result["shares"] == 15


def test_parses_multiple_tickers():
    result = parse_market_request("Compare AAPL and MSFT prices.")

    assert result["ticker"] == ["AAPL", "MSFT"]


def test_does_not_treat_about_as_a_ticker():
    result = parse_market_request("What is the latest news about AAPL?")

    assert result["ticker"] == ["AAPL"]
