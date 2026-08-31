from datetime import date, datetime, timedelta, timezone
import os
from urllib.parse import urlparse

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


def format_trading_date(timestamp: str) -> str:
    """Format a daily-bar timestamp as a trading date without implying a time."""
    parsed = datetime.fromisoformat(timestamp)
    return parsed.strftime("%B") + f" {parsed.day}, {parsed.year}"


def publisher_from_url(url: str) -> str:
    """Return a readable publisher fallback from a source URL."""
    hostname = urlparse(url).netloc.lower().removeprefix("www.")
    return hostname or "Unknown publisher"


def classify_freshness(price_timestamp: str, retrieved_at: str) -> str:
    """Classify a closing price using elapsed weekdays, excluding weekends."""
    price_time = datetime.fromisoformat(price_timestamp)
    retrieval_time = datetime.fromisoformat(retrieved_at)

    if price_time.tzinfo is None:
        price_time = price_time.replace(tzinfo=timezone.utc)
    if retrieval_time.tzinfo is None:
        retrieval_time = retrieval_time.replace(tzinfo=timezone.utc)

    price_time_local = price_time.astimezone(retrieval_time.tzinfo)
    price_date = price_time_local.date()
    retrieval_date = retrieval_time.date()

    if price_date == retrieval_date:
        return "same_trading_day"

    elapsed_weekdays = sum(
        (price_date + timedelta(days=offset)).weekday() < 5
        for offset in range(1, (retrieval_date - price_date).days + 1)
    )

    if elapsed_weekdays <= 1:
        return "prior_trading_session"
    return "stale"


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
    price_timestamp = latest_close.index[-1].isoformat()

    return {
        "ticker": symbol,
        "price": float(latest_close.iloc[-1]),
        "currency": currency,
        "price_type": "latest_available_close",
        "price_timestamp": price_timestamp,
        "market_timestamp": price_timestamp,
        "price_timestamp_display": format_trading_date(price_timestamp),
        "trading_date": format_trading_date(price_timestamp),
        "retrieved_at": retrieved_at,
        "retrieved_at_display": format_timestamp(retrieved_at),
        "freshness_status": classify_freshness(price_timestamp, retrieved_at),
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
        query=f"{search_query} latest",
        topic="finance",
        search_depth="basic",
        time_range="day",
        max_results=5,
    )

    results = response.get("results", [])
    if len(results) < 2:
        response = client.search(
            query=f"{search_query} latest",
            topic="finance",
            search_depth="basic",
            time_range="week",
            max_results=5,
        )
        results = response.get("results", [])

    news = []
    for result in results:
        url = result.get("url", "")
        published_timestamp = result.get("published_date")
        news.append(
            {
                "title": result.get("title", ""),
                "publisher": result.get("source") or publisher_from_url(url),
                "published_timestamp": published_timestamp,
                "url": url,
                "content": result.get("content", ""),
            }
        )

    return news
