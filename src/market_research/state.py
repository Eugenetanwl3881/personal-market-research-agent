from typing import Any, TypedDict


class MarketResearchState(TypedDict, total=False):
    question: str
    intent: str
    ticker: list[str]
    shares: int | None
    needs_quote: bool
    needs_news: bool
    quote: dict[str, Any]
    news: list[dict[str, Any]]
    calculation: str
    errors: list[str]
    final_answer: str
    steps: list[str]
