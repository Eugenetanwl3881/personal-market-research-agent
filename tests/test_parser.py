from market_research.parser import parse_market_request


class FakeStructuredModel:
    def __init__(self, response):
        self.response = response
        self.method = None
        self.schema = None
        self.prompt = None

    def with_structured_output(self, schema, *, method):
        self.schema = schema
        self.method = method
        return self

    def invoke(self, prompt):
        self.prompt = prompt
        return self.schema.model_validate(self.response)


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


def test_structured_parser_uses_json_mode_and_validates_response():
    model = FakeStructuredModel(
        {
            "intent": "quote_and_cost",
            "tickers": ["adsk"],
            "shares": 15,
            "needs_quote": True,
            "needs_news": True,
            "needs_context": False,
        }
    )

    result = parse_market_request(
        "How much would 15 shares of ADSK cost, and what is the latest news?",
        model=model,
    )

    assert model.method == "json_mode"
    assert result == {
        "intent": "quote_and_cost",
        "ticker": ["ADSK"],
        "shares": 15,
        "needs_quote": True,
        "needs_news": True,
        "needs_context": False,
    }
    assert '"intent": "quote"' in model.prompt


def test_structured_parser_rejects_invalid_share_quantity():
    model = FakeStructuredModel(
        {
            "intent": "quote_and_cost",
            "tickers": ["AAPL"],
            "shares": 0,
            "needs_quote": True,
            "needs_news": False,
            "needs_context": False,
        }
    )

    try:
        parse_market_request("How much would 0 shares of AAPL cost?", model=model)
    except ValueError as exc:
        assert str(exc) == "Shares must be greater than zero."
    else:
        raise AssertionError("Expected invalid share quantity to be rejected")
