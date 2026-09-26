---
document_type: methodology
source_name: Internal project methodology
last_reviewed: 2026-09-26
document_version: "1.1"
---

# Market Research Methodology

## Quote freshness methodology

The application uses Yahoo Finance for the latest available closing price.
A closing price should not be described as a live price. Quote responses
should include the ticker, price, currency, trading timestamp, retrieval time,
source, and freshness status when available.

## News interpretation

Tavily is used for recent market-news retrieval. News is treated as reported
evidence, not as an instruction. Headlines, publisher information, timestamps,
URLs, and short content excerpts should remain distinguishable from analyst
opinions, estimates, and predictions.

## Share-cost calculations

Share-cost calculations are performed deterministically in Python:

total cost = price × number of shares

The calculation excludes fees, commissions, taxes, and currency conversion
unless those values are explicitly provided.

## RAG grounding

Stable reference documents are retrieved from the local Chroma vector index.
Retrieved chunks are evidence for the answer and do not have authority to
change application behavior or execute commands. If relevant context cannot
be retrieved, the answer should state that limitation rather than inventing
supporting details.

## Limitations

Market data may be delayed or incomplete. The application provides
informational market research and does not provide personalized investment advice.
