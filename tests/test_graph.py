from market_research.graph import (
    route_after_parse,
    route_after_quote,
    route_after_news,
    write_partial_answer,
)


def test_parse_routes_missing_ticker_to_partial_answer():
    assert route_after_parse({"intent": "error", "ticker": []}) == "partial_answer"


def test_quote_failure_routes_to_partial_answer():
    assert route_after_quote({"quote": {}, "needs_news": True}) == "partial_answer"


def test_successful_quote_can_continue_to_news():
    assert route_after_quote({
        "quote": {"AAPL": {"price": 100}},
        "needs_news": True,
        "shares": None,
    }) == "fetch_news"


def test_news_without_results_can_still_answer():
    assert route_after_news({"shares": None}) == "write_answer"


def test_partial_answer_includes_available_quote_and_errors():
    answer = write_partial_answer({
        "quote": {"AAPL": {"price": 100, "currency": "USD"}},
        "errors": ["News service unavailable."],
    })

    assert "AAPL" in answer
    assert "News service unavailable." in answer
