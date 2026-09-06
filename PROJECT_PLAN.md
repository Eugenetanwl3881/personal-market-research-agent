# Market Research Assistant — Status and Roadmap

## Implemented

- Command-line application installed through the `market-research` command.
- LangGraph workflow with conditional routing.
- Structured LLM request parsing with deterministic validation.
- Multi-ticker latest-close retrieval through Yahoo Finance.
- Freshness, trading-date, retrieval-time, currency, and source metadata for quotes.
- Finance news retrieval through Tavily with a day-to-week fallback.
- Deterministic share-cost calculation.
- Guardrails and partial-answer fallbacks.
- Grounded answer format with explicit source links and limitations.
- Prompt-injection protections for retrieved news content.
- Graph-step and token-level hybrid streaming in the CLI.
- Mocked tests for normal and failure scenarios.

## Next improvements

1. Improve news relevance and source-quality ranking.
2. Add clearer multi-ticker comparison output, including optional calculated differences.
3. Add retries, timeouts, and observability for external providers.
4. Add test coverage for CLI presentation and live-stream event ordering.
5. Add an optional web/API interface after the command-line workflow is stable.

## Deferred capabilities

- Portfolios and holdings
- Trade execution or order management
- Persistent database storage
- Dashboard/UI
- Long-term conversation memory
- Personalized investment advice or buy/sell recommendations

## Product boundaries

The application provides sourced informational market research. It does not execute trades or provide personalized investment advice. Quote and news data may be delayed, incomplete, or unavailable.
