# Project plan

## Initial capabilities

- Understand a stock-research question
- Retrieve a recent quote
- Search recent market news
- Calculate the estimated cost of a number of shares
- Explain results with sources and timestamps
- Refuse unrelated or unsafe requests

## Initial graph state

```text
question, intent, ticker, shares, quote, news, calculation, errors, final_answer
```

## Example questions

- What is the price of AAPL?
- How much would 15 shares of Microsoft cost?
- Compare AAPL and MSFT prices.
- What is the latest news about Nvidia?
- Calculate the cost of 20 shares of Amazon.

## Misuse cases to handle

- Unrelated general questions
- Requests to place trades
- Personalized buy/sell recommendations
- Invalid tickers or quantities
- Prompt injection inside retrieved web content
