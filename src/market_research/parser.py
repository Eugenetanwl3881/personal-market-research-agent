import re
from typing import Any, Literal

from pydantic import BaseModel, Field


class MarketRequest(BaseModel):
    """Validated structured interpretation of a market-research question."""

    intent: Literal["quote", "quote_and_cost", "news", "comparison"]
    tickers: list[str] = Field(default_factory=list)
    shares: int | None = None
    needs_quote: bool = False
    needs_news: bool = False
    needs_context: bool = False


def _validate_request(request: MarketRequest) -> dict:
    """Apply deterministic checks after structured model extraction."""
    tickers = list(dict.fromkeys(ticker.strip().upper() for ticker in request.tickers))

    if not tickers:
        raise ValueError("No stock ticker was found in the question.")

    shares = request.shares
    if shares == 0 and request.intent != "quote_and_cost":
        # Some providers emit zero for an optional integer they did not use.
        shares = None

    if shares is not None and shares <= 0:
        raise ValueError("Shares must be greater than zero.")

    return {
        "intent": request.intent,
        "ticker": tickers,
        "shares": shares,
        "needs_quote": request.needs_quote,
        "needs_news": request.needs_news,
        "needs_context": request.needs_context,
    }


def _fallback_parse(question: str) -> dict:
    """Small offline fallback used by unit tests and parser experiments."""
    common_words = {
        "WHAT", "IS", "THE", "PRICE", "OF", "HOW", "MUCH", "WOULD",
        "SHARES", "COST", "LATEST", "NEWS", "ABOUT", "AND", "COMPARE",
    }
    words = re.findall(r"\b[A-Z]{1,5}\b", question.upper())
    tickers = [word for word in words if word not in common_words]
    share_match = re.search(r"\b(\d+)\s+shares?\b", question.lower())
    shares = int(share_match.group(1)) if share_match else None
    intent = "quote_and_cost" if shares is not None else "quote"
    return _validate_request(MarketRequest(
        intent=intent,
        tickers=tickers,
        shares=shares,
        needs_quote=True,
        needs_news="news" in question.lower(),
        needs_context=any(
            term in question.lower()
            for term in (
                "business",
                "segment",
                "industry",
                "product",
                "service",
                "revenue",
                "fundamental",
                "financial",
                "annual report",
                "10-k",
                "10-q",
            )
        ),
    ))


def parse_market_request(question: str, model: Any | None = None) -> dict:
    """Extract and validate a market request using structured model output."""
    if not question.strip():
        raise ValueError("Question must not be empty.")

    if model is None:
        return _fallback_parse(question)

    structured_model = model.with_structured_output(MarketRequest)
    request = structured_model.invoke(
        """Extract the stock-market request from the user's question.

Return only the requested schema. Use uppercase exchange tickers when present.
Set needs_quote for price or cost requests. Set needs_news for recent-news requests.
Set needs_context for stable company, industry, business-segment, product, or
financial-background questions that can be answered from reference documents.
Keep needs_context false for quote, cost, or recent-news questions that do not
ask for background information.
Use quote_and_cost when a positive share quantity is requested.

User question:
""" + question
    )
    return _validate_request(request)
