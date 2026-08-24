# Market Research Assistant — Version-One Product Note

## Project goal

> A command-line market-research assistant that accepts an equity research question, retrieves current or recent pricing, searches recent market news, calculates share costs, and returns a sourced informational answer.

## First-version scope

Version one establishes a focused foundation for an extensible market-research product. It supports:

- One or more stock tickers
- Current or recent prices
- Number-of-shares cost calculations
- Recent market news
- Basic guardrails and refusal handling
- Streaming graph-step output

These capabilities are intentionally deferred from version one: portfolio management, trade execution, persistent databases, dashboards, and conversational memory.

## Proposed technology

- Python
- LangGraph
- LangChain Core
- One LLM provider initially
- `yfinance`
- Tavily
- `python-dotenv`

## Graph workflow

```text
START
  ↓
scope_check
  ↓
parse_request
  ↓
fetch_quote
  ↓
fetch_news
  ↓
calculate_cost
  ↓
write_answer
  ↓
END
```

Nodes perform work, edges determine the order of that work, and state carries shared information between nodes.

## Proposed graph state

```text
question
intent
ticker
shares
quote
news
calculation
errors
final_answer
```

The initial graph should avoid unnecessary conversation history while keeping the state structure easy to extend in later versions.

## Underlying Python functions

These functions must work independently of the LLM before they are exposed as LangChain tools:

```python
get_stock_quote(ticker)
search_market_news(query)
calculate_share_cost(price, shares)
```

The retrieval functions handle external data. The calculation function performs exact arithmetic and validates that the price and share count are usable. Each function should return understandable results or explicit errors.

## Example user questions

1. What is the current price of AAPL?
2. How much would 15 shares of Microsoft cost?
3. Compare the prices of AAPL and MSFT.
4. What is the latest news about Nvidia?
5. What is the price of Amazon, the cost of 20 shares, and the recent news?

## Misuse cases to reject

1. Unrelated general questions, such as asking for a recipe.
2. Requests to place, cancel, or execute trades.
3. Personalized buy/sell recommendations or instructions.
4. Invalid tickers, zero shares, negative shares, or otherwise invalid quantities.
5. Prompt injection or instructions embedded in retrieved web content.

## Learning checkpoints

1. **LangChain tools:** a typed, documented Python function the model is permitted to call; its arguments and result should be explicit.
2. **LangGraph state:** the shared structured information passed from one workflow step to the next.
3. **Nodes and edges:** nodes perform work, while edges control which node runs next.

The core workflow is therefore: validate the question, fetch relevant data, calculate deterministic values, and write a sourced answer with timestamps and an informational disclaimer.

## Milestones

1. Python data functions
2. LangChain tools
3. Basic LangGraph workflow
4. News and calculations
5. Guardrails and error handling
6. Streaming graph updates, then token-level answer streaming
7. Tests and refinement
