TRADE_TERMS = {
    "buy",
    "sell",
    "trade",
    "place an order",
    "execute",
    "cancel order",
}

ADVICE_TERMS = {
    "should i buy",
    "should i sell",
    "recommend",
    "recommendation",
    "is it a good investment",
    "will it go up",
}

RESEARCH_TERMS = {
    "price",
    "quote",
    "shares",
    "cost",
    "news",
    "stock",
    "ticker",
    "market",
    "company",
}

REFERENCE_TERMS = {
    "business",
    "segment",
    "segments",
    "industry",
    "product",
    "products",
    "service",
    "services",
    "revenue",
    "fundamental",
    "fundamentals",
    "financial",
    "methodology",
    "freshness",
    "closing price",
    "data source",
    "annual report",
    "10-k",
    "10-q",
}


def check_scope(question: str) -> tuple[bool, str]:
    """Return whether a question is within the assistant's scope."""
    normalized = question.strip().lower()

    if not normalized:
        return False, "Please provide a stock-market research question."

    if any(term in normalized for term in TRADE_TERMS):
        return False, "I cannot place, execute, or manage trades."

    if any(term in normalized for term in ADVICE_TERMS):
        return False, "I can provide market information, but not personalized investment advice."

    if not any(term in normalized for term in RESEARCH_TERMS | REFERENCE_TERMS):
        return False, (
            "I can only help with stock prices, share calculations, recent market "
            "news, and related company or market-research background."
        )

    return True, "" 
