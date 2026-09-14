import re
from typing import Any, Literal

from pydantic import BaseModel, Field


class MarketRequest(BaseModel):
    """Validated structured interpretation of a market-research question."""

    intent: Literal["quote", "quote_and_cost", "news", "comparison", "context"]
    tickers: list[str] = Field(default_factory=list)
    shares: int | None = None
    needs_quote: bool = False
    needs_news: bool = False
    needs_context: bool = False


def _validate_request(request: MarketRequest) -> dict:
    """Apply deterministic checks after structured model extraction."""
    tickers = list(dict.fromkeys(ticker.strip().upper() for ticker in request.tickers))

    if not tickers and request.intent != "context":
        raise ValueError(
            "No stock ticker was found in the question. "
            "A ticker is required for quote or news requests."
        )

    if not tickers and (request.needs_quote or request.needs_news):
        raise ValueError("A stock ticker is required for quote or news requests.")

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
        "QUOTE", "FRESHNESS", "METHODOLOGY", "BUSINESS", "SEGMENT",
        "SEGMENTS", "INDUSTRY", "PRODUCT", "PRODUCTS", "SERVICE", "SERVICES",
        "REVENUE", "FUNDAMENTAL", "FUNDAMENTALS", "FINANCIAL", "ANNUAL", "REPORT",
        "MAJOR",
    }
    words = re.findall(r"\b[A-Z]{1,5}\b", question.upper())
    tickers = [word for word in words if word not in common_words]
    share_match = re.search(r"\b(\d+)\s+shares?\b", question.lower())
    shares = int(share_match.group(1)) if share_match else None
    context_terms = (
        "business",
        "segment",
        "industry",
        "product",
        "service",
        "revenue",
        "fundamental",
        "financial",
        "methodology",
        "freshness",
        "annual report",
        "10-k",
        "10-q",
    )
    needs_context = any(term in question.lower() for term in context_terms)
    intent = (
        "context"
        if needs_context and not tickers and shares is None
        else "quote_and_cost"
        if shares is not None
        else "quote"
    )
    return _validate_request(MarketRequest(
        intent=intent,
        tickers=tickers,
        shares=shares,
        needs_quote=intent != "context",
        needs_news="news" in question.lower(),
        needs_context=needs_context,
    ))


def parse_market_request(question: str, model: Any | None = None) -> dict:
    """Extract and validate a market request using structured model output."""
    if not question.strip():
        raise ValueError("Question must not be empty.")

    if model is None:
        return _fallback_parse(question)

    # OpenCode Go currently accepts JSON-object mode for this model, but its
    # compatibility layer may reject JSON Schema response formats or forced
    # tool choices used by other structured-output methods.
    structured_model = model.with_structured_output(
        MarketRequest,
        method="json_mode",
    )
    request = structured_model.invoke(
        """Extract the stock-market request from the user's question.

Return ONLY one valid JSON object with exactly these keys:
{
  "intent": "quote",
  "tickers": ["AAPL"],
  "shares": null,
  "needs_quote": true,
  "needs_news": false,
  "needs_context": false
}

Rules:
- intent must be exactly one of: quote, quote_and_cost, news, comparison, context.
- tickers must be an array of uppercase stock-exchange ticker symbols.
- Resolve a well-known company name to its ticker when the identification is
  unambiguous, such as Apple -> AAPL or Microsoft -> MSFT. If it is not
  unambiguous, return an empty tickers array.
- shares must be a positive integer when the user requests a share quantity;
  otherwise use null. Do not use zero as a placeholder.
- Set needs_quote to true for price or share-cost requests.
- Set needs_news to true for recent-news requests.
- Set needs_context to true for stable company, industry, business-segment,
  product, service, revenue, fundamental, financial-background, annual-report,
  10-K, or 10-Q questions that can be answered from reference documents.
- Keep needs_context false for quote, cost, or recent-news questions that do
  not ask for background information.
- Use context for stable background or methodology questions that do not
  require a ticker. Quote, quote_and_cost, news, and comparison requests must
  include at least one ticker.
- Use quote_and_cost when a positive share quantity is requested.
- Do not add company names, prices, business segments, notes, explanations,
  markdown, or any other keys.

User question:
""" + question
    )
    return _validate_request(request)
