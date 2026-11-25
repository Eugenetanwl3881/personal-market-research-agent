# Market Research Assistant

A command-line application for sourced, informational equity-market research. It retrieves the latest available closing prices, searches recent news, calculates estimated share costs, and presents the result with sources, timestamps, freshness information, and an investment-advice disclaimer.


## What it does

- Interprets one or more ticker symbols and an optional share quantity from a natural-language question.
- Retrieves the latest available closing price from Yahoo Finance through `yfinance`.
- Searches finance news through Tavily, preferring the past day and falling back to the past week when results are sparse.
- Retrieves relevant stable reference context from the local RAG knowledge base when a question asks for company or industry background.
- Calculates share costs deterministically in Python.
- Labels price type, trading date, retrieval time, source, and freshness.
- Separates verified quote data from reported news and model-written summaries.
- Refuses trade execution, unrelated requests, and personalized investment-advice requests.
- Streams workflow progress and answer text in the terminal.

## Architecture

```text
CLI question
  ↓
LangGraph workflow
  ↓
scope check → structured parsing → quote/news retrieval → calculation → answer
  ↓
terminal progress updates + streamed answer text
```

The graph coordinates the work. External retrieval, RAG retrieval, and arithmetic remain ordinary Python functions, independent of the language model.

The current RAG path is a predictable two-step flow:

```text
Question → retrieve relevant reference chunks → grounded answer
```

The knowledge base is stored in `knowledge/` as Markdown with YAML frontmatter and PDFs with YAML sidecars. The retriever uses local embeddings and a persistent Chroma vector store. By default, the index is stored in `.rag_index/`; set `RAG_INDEX_DIR` to use another local directory. Ingestion compares individual chunks with stored records, reuses embeddings for identical text, updates metadata, and deletes obsolete records. A manifest records source/chunk identities, content hashes, and embedding compatibility.

Retrieval combines up to 20 vector candidates with up to 20 BM25 keyword candidates, using reciprocal rank fusion to return at most two passages. Keyword ranks have twice the weight of vector ranks. The cosine threshold of 0.45 applies to vector candidates; keyword matches can recover passages missed by vector search. These scores rank evidence and are not probabilities that an answer is correct. Keyword scoring scans the loaded chunks locally, which suits the current small corpus.

Both searches filter by ticker, retaining generic untickered documents. An explicit annual-report/10-K reference adds a document-type filter; a single year in that request also filters the integer `fiscal_year` metadata field. `document_version` is a separate revision label, such as `"1"` or `"1.1"`, and is never used as the report year. Within that scope, the search focuses on the topic rather than repeating company and report identifiers. PDF loading strips repeated margin lines. Each selected PDF hit expands to at most three contiguous chunks on the same page, centered on the hit when possible. At a page boundary, the window shifts to use available chunks: two following chunks at the start or two preceding chunks at the end. References display the PDF page number, which can differ from the report's printed page number.

Company documents declare metadata explicitly:

```yaml
---
ticker: AAPL
company: Apple Inc.
document_type: company_overview
source_url: https://investor.apple.com/
source_name: Apple Investor Relations
last_reviewed: 2026-09-26
document_version: "1.1"
---
```

```mermaid
flowchart TD
    START([START]) --> S[scope_check]
    S -->|refused| END1([END])
    S -->|accepted| P[parse_request]

    P -->|parse failure or missing ticker for quote/news| PA[write_partial_answer]
    P -->|quote needed| Q[fetch_quote]
    P -->|news only| N[fetch_news]
    P -->|no retrieval needed| A[write_answer]

    Q -->|no valid quote| PA
    Q -->|news requested| N
    Q -->|shares supplied| C[calculate_cost]
    Q -->|otherwise| A

    N -->|shares supplied| C
    N -->|otherwise| A
    C --> A
    PA --> END2([END])
    A --> END3([END])
```

## Technology

- Python 3.11+
- LangGraph
- LangChain Core and `langchain-openai`
- `langchain-text-splitters`, `langchain-huggingface`, and `langchain-chroma`
- `pypdf` for page-level PDF text extraction
- `sentence-transformers` for local document embeddings
- `PyYAML` for knowledge-document frontmatter
- An OpenAI-compatible model endpoint (currently configured for OpenCode Go)
- `yfinance`
- Tavily
- `python-dotenv`

## Setup

Create and activate a virtual environment, then install the project in editable mode:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Create a `.env` file in the project root. Never commit this file.

```dotenv
TAVILY_API_KEY=your_tavily_key
OPENCODE_API_KEY=your_opencode_key
OPENCODE_BASE_URL=https://opencode.ai/zen/go/v1
OPENCODE_MODEL=deepseek-v4-flash
```

The model endpoint and model name are configurable so the application can later use another compatible provider without changing workflow code.

## Usage

After installation, run either form:

```bash
market-research "What was the latest available closing price of NVDA, how much would 8 shares cost, and what recent news was reported?"
```

```bash
python -m market_research.cli "Compare AAPL and MSFT prices."
```

Running `market-research` with no question starts an interactive prompt.

Prepare the local RAG index explicitly with:

```bash
market-research-index
```

The equivalent module command is:

```bash
python -m market_research.ingest
```

Download the latest available Apple annual report automatically:

```bash
market-research-fetch --ticker AAPL
```

The current Apple source adapter tries recent official investor-relations
annual-report PDFs and writes both the PDF and a validated YAML sidecar under
`knowledge/raw/apple/`. To fetch a specific year, use `--year`, and use
`--force` to replace an existing local copy:

```bash
market-research-fetch --ticker AAPL --year 2025
market-research-fetch --ticker AAPL --year 2025 --force
```

After downloading a report, refresh the persistent index:

```bash
market-research-index
```

Microsoft fiscal-2025 Form 10-K is also available through the downloader:

```bash
market-research-fetch --ticker MSFT --year 2025
```

It writes `knowledge/raw/microsoft/annual_report_2025.pdf` and its YAML sidecar.
The PDF is from Microsoft's official investor-relations host:
[Microsoft fiscal-2025 Form 10-K](https://microsoft.gcs-web.com/static-files/9640e86d-a99f-4502-934d-1d3c883fa786).
Microsoft's adapter uses a registry of verified PDF URLs; it currently supports
only fiscal 2025. With no `--year`, it selects the newest configured PDF, not
necessarily the latest report Microsoft has published. Unsupported years fail
explicitly. The existing local PDF/sidecar is protected unless `--force` is used.
Apple's recent-year discovery behavior is unchanged. Both adapters use the
existing PDF loader; no format conversion is required.

Use `--force` or `--force-rebuild` to rebuild even when the existing index is current. You can override the source and output locations with `--knowledge-dir` and `--index-dir`. Research queries still check the index automatically, so explicit ingestion is optional but useful when preparing a deployment or verifying the knowledge base.

### Incremental ingestion and identity

| Identity | Based on | Purpose |
| --- | --- | --- |
| Document ID | Path relative to the knowledge directory | Keeps a source's identity across content/metadata edits. |
| Chunk ID | Document ID, PDF page, exact text hash, repeated-text occurrence | Keeps source/page references distinct without depending on mutable offsets. |
| Content hash | Exact extracted chunk text | Reuses a vector when the same text appears in another source or location. |

An unchanged run embeds no document text. Adding a report embeds only text not
already represented in the compatible index. Editing a report reuses unchanged
chunk text, inserts new chunks, updates offsets/metadata, and removes old chunks.
A metadata-only edit updates Chroma metadata without embedding. Deleting sources
removes their records, including when the knowledge directory becomes empty.
Missing directories and invalid source metadata fail before synchronization.

Exact duplicate extracted text shares embedding work, while each source/page
keeps a separate Chroma record for filtering and citations. A rename changes
document/chunk IDs, but reuses matching vectors before deleting the old records.
Duplicate counts describe exact extracted text, not near-duplicate meaning or
byte-identical PDF files. Reuse draws from currently indexed records; vectors
for text deleted in an earlier run are not kept in a separate permanent cache.

The command reports added, metadata-updated, deleted, and unchanged chunks,
unique texts embedded, reused embedding assignments, and duplicate counts.
PDF pages are counted under `Documents`; `Source files` counts the source files.
The overall signature still records corpus/configuration changes for evaluations;
a changed signature alone no longer forces a full rebuild.

The old version-1 manifest needs one full migration rebuild. Changing the
embedding model, the index identity schema, or using `--force` also rebuilds.
Custom embedding implementations whose hidden configuration changes must use
`--force`; exposed `model_name` and `encode_kwargs` are included in compatibility.

Ingestion still reads, extracts, and splits all source files to find changes.
Only embedding and database writes are incremental. Use one ingestion writer
per collection. The manifest is replaced atomically after database writes;
the overall operation is not a database transaction. Interrupted writes can be
reconciled by rerunning ingestion. Keep the same knowledge root for a collection:
its relative paths define document identity.

PDF files can be placed anywhere below the knowledge directory. Each PDF requires a same-named YAML sidecar because PDFs do not contain the project’s frontmatter format:

```text
knowledge/raw/apple/annual_report_2025.pdf
knowledge/raw/apple/annual_report_2025.yaml
```

Example sidecar:

```yaml
ticker: AAPL
company: Apple Inc.
document_type: annual_report
source_name: Apple Investor Relations
source_url: https://example.com/apple-report.pdf
published_date: 2025-10-31
last_reviewed: 2026-09-26
fiscal_year: 2025
document_version: "1"
```

Annual reports require an explicit integer `fiscal_year` between 1900 and 2099
(the year range currently recognized in requests). This is the year covered
by the report, which may differ from its publication year. General company
overviews and methodology documents may omit it. Existing annual-report
frontmatter or sidecars using `document_version: "2025"` must add
`fiscal_year: 2025`; revise the version label independently. The loader does
not infer a fiscal year from the revision label, filename, or publication date.
Metadata changes update stored chunk metadata automatically without reembedding
unchanged text. The original PDF does not need to be downloaded again.

The loader creates one document per non-empty PDF page and preserves `page_number`, `page_count`, `source`, and `source_format` metadata. Run `market-research-index` after adding or changing a PDF. Scanned image-only PDFs are not supported yet because they require OCR.

## Output design

Python renders predictable, data-sensitive sections:

- Summary
- Quotes
- Estimated share cost, when shares are requested
- Recent news and source links
- Reference sources, when RAG context is used
- Data limitations
- Disclaimer

The language model writes only the narrative summary and news summaries. Quotes, sources, arithmetic, timestamps, and disclaimers are rendered by application code. News excerpts are treated as untrusted data and explicitly delimited in the model prompt; instructions inside articles are ignored.

If reference context is requested but no usable excerpts are retrieved, Python
writes a limitation summary directly and skips the summary-model call. It
distinguishes retrieval failure from no matching evidence. Available quotes,
calculations, and news still appear for mixed requests.

## Testing

Run the test suite with:

```bash
python -m pytest -q
```

External services are mocked in the normal tests, so they are fast, deterministic, and do not consume Yahoo, Tavily, or model API requests.

The RAG tests also include deterministic evaluation cases for Apple, Microsoft, methodology questions, keyword recovery when vector search misses the answer, company/year filtering, PDF margin cleanup, adjacent context, and PDF page references.

To measure retrieval with the real embedding model and current documents:

```bash
python -m market_research.evaluate
```

This evaluates fifteen labelled questions without calling the answer model or market
data APIs. Results and retrieved evidence are saved to the Git-ignored
`evaluations/results/latest.json`. See [the evaluation guide](evaluations/README.md)
for metrics and known baseline failures. It checks retrieval directly; it does
not evaluate request parsing or final-answer quality.

To generate final answers for the six evidence-sufficiency cases and prepare
a human review, run:

```bash
python -m market_research.evaluate_answers
```

This calls your configured OpenCode model. It saves actual answers, evidence,
and automated review signals to `evaluations/results/answers_latest.json`, plus
a readable `answers_latest.md` review checklist. Automated signals are not
correctness grades; review the full answers against their evidence. See the
[evaluation guide](evaluations/README.md#final-answer-evaluation) for details.

## Current limitations

- Prices are latest available closes, not guaranteed live prices.
- Yahoo Finance and Tavily data can be delayed, incomplete, or unavailable.
- The RAG corpus is small and local. Download adapters cover Apple and Microsoft's configured fiscal-2025 PDF; broader discovery and scheduled refresh are future work.
- Hugging Face authentication is currently optional; configure `HF_TOKEN` later
  if higher rate limits or more reliable embedding-model downloads are needed.
- The first version has no portfolios, trade execution, database, dashboard, or conversation memory.
- The assistant does not provide personalized buy/sell recommendations. (YET)
