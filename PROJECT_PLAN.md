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
- Two-step RAG retrieval over a local Markdown knowledge base.
- Persistent Chroma-backed RAG index with manifest-based rebuild detection.
- Standalone `market-research-index` command for explicit knowledge ingestion.
- Deterministic RAG retrieval evaluation cases for company, methodology, and ticker filtering.
- Real-embedding retrieval evaluation command with fifteen labelled cases, saved evidence, and baseline metrics.
- Final-answer evaluation runner with six evidence-sufficiency cases, mechanical review signals, and saved answer/evidence review reports.
- Application-written missing-evidence summaries that bypass the summary model while retaining available market data and streaming.
- Richer sectioned company and methodology documents with review/version metadata.
- Page-aware PDF ingestion with required YAML sidecar metadata.
- Explicit validated fiscal-year metadata for annual reports, separate from document revisions, used in both retrieval filters.
- Hybrid BM25/vector retrieval with company/report-year filtering, PDF margin cleanup, and neighboring context.
- PDF page numbers in reference evidence and rendered references.
- Automated Apple annual-report download with generated and validated sidecar metadata.
- Verified Microsoft fiscal-2025 Form 10-K PDF downloader, validated sidecar, and page-labelled evaluation cases; retrieval ranking unchanged.
- Reference evidence passed to the answer model with source labels.
- Graph-step and token-level hybrid streaming in the CLI.
- Mocked tests for normal and failure scenarios.

## Next improvements

1. Improve news relevance and source-quality ranking.
2. Expand official-filing discovery/download to other issuers and evaluate retrieval across more reports.
3. Add clearer multi-ticker comparison output, including optional calculated differences.
4. Add retries, timeouts, and observability for external providers.
5. Add optional Hugging Face authentication (`HF_TOKEN`) for higher rate limits
   and more reliable embedding-model downloads.
6. Add test coverage for CLI presentation and live-stream event ordering.
7. Add an optional web/API interface after the command-line workflow is stable.

## Deferred capabilities

- Portfolios and holdings
- Trade execution or order management
- Persistent database storage
- Dashboard/UI
- Long-term conversation memory
- Personalized investment advice or buy/sell recommendations

## Product boundaries

The application provides sourced informational market research. It does not execute trades or provide personalized investment advice. Quote and news data may be delayed, incomplete, or unavailable.
