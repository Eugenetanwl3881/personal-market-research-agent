import os

import yfinance as yf
from dotenv import load_dotenv
from tavily import TavilyClient


load_dotenv()


def get_stock_quote(ticker: str) -> dict:
    """Retrieve the latest available quote for a stock ticker."""
    symbol = ticker.strip().upper()

    if not symbol:
        raise ValueError("Ticker must not be empty.")

    stock = yf.Ticker(symbol)
    history = stock.history(period="5d")

    if history.empty or "Close" not in history:
        raise ValueError(f"No quote data found for ticker: {symbol}")

    latest_close = history["Close"].dropna()

    if latest_close.empty:
        raise ValueError(f"No closing price found for ticker: {symbol}")

    return {
        "ticker": symbol,
        "price": float(latest_close.iloc[-1]),
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
