from datetime import datetime, timezone
import os

import yfinance as yf
from dotenv import load_dotenv
from tavily import TavilyClient


load_dotenv()


def format_timestamp(timestamp: str) -> str:
    """Convert an ISO timestamp into a concise, human-readable UTC/local timestamp."""
    parsed = datetime.fromisoformat(timestamp)
    timezone_name = parsed.tzname() or "UTC"
    hour = parsed.hour % 12 or 12
    meridiem = "AM" if parsed.hour < 12 else "PM"

    return (
        f"{parsed.strftime('%B')} {parsed.day}, {parsed.year} at "
        f"{hour}:{parsed.minute:02d} {meridiem} {timezone_name}"
    )


def get_stock_quote(ticker: str) -> dict:
    """Retrieve the latest available quote for a stock ticker."""
    symbol = ticker.strip().upper()

    if not symbol:
        raise ValueError("Ticker must not be empty.")

    stock = yf.Ticker(symbol)
    try:
        history = stock.history(period="5d")
    except Exception as exc:
        raise RuntimeError(f"Unable to retrieve quote data for {symbol}.") from exc

    if history.empty or "Close" not in history:
        raise ValueError(f"No quote data found for ticker: {symbol}")

    latest_close = history["Close"].dropna()

    if latest_close.empty:
        raise ValueError(f"No closing price found for ticker: {symbol}")

    try:
        currency = stock.fast_info.get("currency", "USD")
    except Exception:
        currency = "USD"

    retrieved_at = datetime.now(timezone.utc).isoformat()

    return {
        "ticker": symbol,
        "price": float(latest_close.iloc[-1]),
        "currency": currency,
        "price_timestamp": latest_close.index[-1].isoformat(),
        "price_timestamp_display": format_timestamp(
            latest_close.index[-1].isoformat()
        ),
        "retrieved_at": retrieved_at,
        "retrieved_at_display": format_timestamp(retrieved_at),
        "source": "Yahoo Finance",
    }


def search_market_news(query: str) -> list[dict]:
    """Search for recent market news and return structured results."""
    search_query = query.strip()

    if not search_query:
        raise ValueError("News search query must not be empty.")

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not configured.")

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=search_query,
        topic="news",
        search_depth="basic",
        max_results=5,
    )

    return [
        {
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "content": result.get("content", ""),
            "published_date": result.get("published_date"),
        }
        for result in response.get("results", [])
    ]
