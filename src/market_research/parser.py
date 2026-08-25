import re


COMMON_WORDS = {
    "WHAT",
    "IS",
    "THE",
    "PRICE",
    "OF",
    "HOW",
    "MUCH",
    "WOULD",
    "SHARES",
    "COST",
    "LATEST",
    "NEWS",
    "AND",
}


def parse_market_request(question: str) -> dict:
    """Extract likely stock tickers and a share quantity from a question."""
    words = re.findall(r"\b[A-Z]{1,5}\b", question.upper())
    tickers = [word for word in words if word not in COMMON_WORDS]

    share_match = re.search(r"\b(\d+)\s+shares?\b", question.lower())
    shares = int(share_match.group(1)) if share_match else None

    return {
        "ticker": tickers,
        "shares": shares,
    }
