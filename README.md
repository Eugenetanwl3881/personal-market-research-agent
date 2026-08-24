# Market Research Assistant

A conversational financial-market research assistant built with LangChain and LangGraph, designed as a focused foundation for future expansion.

## First version

The first version accepts a natural-language stock question, retrieves market data, searches recent news, performs deterministic calculations, and returns a sourced informational answer.

Example query:

> What is the price of AAPL, how much for 15 shares, and what is the latest news?

The system is designed to provide timely market context, recent news, and transparent calculations through a conversational interface.

## Planned workflow

```text
Question → scope check → parse request → fetch quote/news → calculate → answer
```

## Planned stack

- Python 3.11+
- LangGraph
- LangChain Core and one chat-model integration
- yfinance for educational market-data access
- Tavily for web/news search
- python-dotenv for local secrets

## Safety boundaries

The assistant is for informational market research and calculations. It will not place trades or provide personalized financial advice.

Market data may be delayed or incomplete. Nothing produced by this project is investment advice.

## Development milestones

1. Define and test plain Python data and calculation functions.
2. Wrap those functions as LangChain tools.
3. Build the basic LangGraph state and workflow.
4. Add news, calculations, guardrails, and error handling.
5. Add streaming execution and tests.
