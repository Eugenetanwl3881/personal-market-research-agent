import market_research.graph as graph_module


def configure_graph_dependencies(monkeypatch, parsed_request, answer="Mock final answer"):
    """Replace external dependencies while preserving the real graph and routing."""
    monkeypatch.setattr(graph_module, "create_model", lambda: object())
    monkeypatch.setattr(
        graph_module,
        "parse_market_request",
        lambda question, model: parsed_request,
    )
    monkeypatch.setattr(graph_module, "write_market_answer", lambda state: answer)


def quote(ticker):
    return {
        "ticker": ticker,
        "price": 100.0,
        "currency": "USD",
        "price_type": "latest_available_close",
        "trading_date": "August 28, 2026",
        "retrieved_at_display": "August 30, 2026 at 11:00 AM UTC",
        "freshness_status": "prior_trading_session",
        "source": "Yahoo Finance",
    }


def test_valid_quote_news_and_calculation(monkeypatch):
    configure_graph_dependencies(monkeypatch, {
        "intent": "quote_and_cost",
        "ticker": ["AAPL"],
        "shares": 15,
        "needs_quote": True,
        "needs_news": True,
    })
    monkeypatch.setattr(graph_module, "get_stock_quote", quote)
    monkeypatch.setattr(graph_module, "search_market_news", lambda query: [{"title": "News"}])

    result = graph_module.build_graph().invoke({"question": "AAPL with 15 shares and news"})

    assert result["quote"]["AAPL"]["price"] == 100.0
    assert result["news_status"] == "ok"
    assert result["calculation"] == "AAPL: 15 shares × $100.00 = $1500.00"
    assert result["final_answer"] == "Mock final answer"


def test_valid_quote_with_no_news_still_returns_calculation(monkeypatch):
    configure_graph_dependencies(monkeypatch, {
        "intent": "quote_and_cost",
        "ticker": ["AAPL"],
        "shares": 15,
        "needs_quote": True,
        "needs_news": True,
    })
    monkeypatch.setattr(graph_module, "get_stock_quote", quote)
    monkeypatch.setattr(graph_module, "search_market_news", lambda query: [])

    result = graph_module.build_graph().invoke({"question": "AAPL with 15 shares and news"})

    assert result["news"] == []
    assert result["news_status"] == "no_results"
    assert "AAPL: 15 shares" in result["calculation"]
    assert result["final_answer"] == "Mock final answer"


def test_invalid_ticker_returns_partial_answer(monkeypatch):
    configure_graph_dependencies(monkeypatch, {
        "intent": "quote",
        "ticker": ["INVALID"],
        "shares": None,
        "needs_quote": True,
        "needs_news": False,
    })
    monkeypatch.setattr(
        graph_module,
        "get_stock_quote",
        lambda ticker: (_ for _ in ()).throw(ValueError("No quote data found.")),
    )

    result = graph_module.build_graph().invoke({"question": "What is the price of INVALID?"})

    assert result["quote"] == {}
    assert "Could not retrieve quote for INVALID" in result["errors"][0]
    assert "could not complete" in result["final_answer"].lower()
    assert result["steps"][-1] == "write_partial_answer completed"


def test_quote_api_failure_returns_partial_answer(monkeypatch):
    configure_graph_dependencies(monkeypatch, {
        "intent": "quote",
        "ticker": ["AAPL"],
        "shares": None,
        "needs_quote": True,
        "needs_news": False,
    })
    monkeypatch.setattr(
        graph_module,
        "get_stock_quote",
        lambda ticker: (_ for _ in ()).throw(RuntimeError("Quote service unavailable.")),
    )

    result = graph_module.build_graph().invoke({"question": "What is the price of AAPL?"})

    assert "Quote service unavailable." in result["errors"][0]
    assert result["steps"][-1] == "write_partial_answer completed"


def test_news_api_failure_preserves_quote_and_calculation(monkeypatch):
    configure_graph_dependencies(monkeypatch, {
        "intent": "quote_and_cost",
        "ticker": ["AAPL"],
        "shares": 15,
        "needs_quote": True,
        "needs_news": True,
    })
    monkeypatch.setattr(graph_module, "get_stock_quote", quote)
    monkeypatch.setattr(
        graph_module,
        "search_market_news",
        lambda query: (_ for _ in ()).throw(RuntimeError("Tavily unavailable.")),
    )

    result = graph_module.build_graph().invoke({"question": "AAPL with 15 shares and news"})

    assert result["news_status"] == "error"
    assert result["quote"]["AAPL"]["price"] == 100.0
    assert "AAPL: 15 shares" in result["calculation"]
    assert result["final_answer"] == "Mock final answer"


def test_missing_share_quantity_skips_calculation(monkeypatch):
    configure_graph_dependencies(monkeypatch, {
        "intent": "quote",
        "ticker": ["AAPL"],
        "shares": None,
        "needs_quote": True,
        "needs_news": False,
    })
    monkeypatch.setattr(graph_module, "get_stock_quote", quote)

    result = graph_module.build_graph().invoke({"question": "What is AAPL?"})

    assert "calculate_cost completed" not in result["steps"]
    assert "calculation" not in result


def test_multiple_tickers_return_multiple_quotes(monkeypatch):
    configure_graph_dependencies(monkeypatch, {
        "intent": "comparison",
        "ticker": ["AAPL", "MSFT"],
        "shares": None,
        "needs_quote": True,
        "needs_news": False,
    })
    monkeypatch.setattr(graph_module, "get_stock_quote", quote)

    result = graph_module.build_graph().invoke({"question": "Compare AAPL and MSFT prices."})

    assert set(result["quote"]) == {"AAPL", "MSFT"}


def test_refused_advice_request_does_not_call_external_dependencies(monkeypatch):
    monkeypatch.setattr(
        graph_module,
        "parse_market_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Should not parse")),
    )

    result = graph_module.build_graph().invoke({"question": "Should I buy Tesla?"})

    assert result["intent"] == "refused"
    assert result["steps"] == ["scope_check completed"]


def test_final_model_failure_uses_deterministic_fallback(monkeypatch):
    configure_graph_dependencies(monkeypatch, {
        "intent": "quote",
        "ticker": ["AAPL"],
        "shares": None,
        "needs_quote": True,
        "needs_news": False,
    })
    monkeypatch.setattr(graph_module, "get_stock_quote", quote)
    monkeypatch.setattr(
        graph_module,
        "write_market_answer",
        lambda state: (_ for _ in ()).throw(RuntimeError("Model unavailable.")),
    )

    result = graph_module.build_graph().invoke({"question": "What is the price of AAPL?"})

    assert "Model unavailable." in result["errors"][-1]
    assert "Latest available close for AAPL" in result["final_answer"]
