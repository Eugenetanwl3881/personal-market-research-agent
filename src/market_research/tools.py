from langchain_core.tools import tool

from .calculations import calculate_share_cost
from .data import get_stock_quote, search_market_news


@tool
def stock_quote(ticker: str) -> dict:
    """Get the latest available closing price for a publicly traded stock ticker."""
    return get_stock_quote(ticker)


@tool
def market_news(query: str) -> list[dict]:
    """Search for recent market news about a company, stock, or financial topic."""
    return search_market_news(query)


@tool
def share_cost(price: float, shares: int) -> str:
    """Calculate the estimated cost of buying a positive number of shares at a given price."""
    return str(calculate_share_cost(price, shares))


MARKET_RESEARCH_TOOLS = [stock_quote, market_news, share_cost]
