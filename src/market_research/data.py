import yfinance as yf


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