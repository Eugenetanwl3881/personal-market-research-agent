# Retrieval evaluation

`retrieval_cases.json` is a small, manually labelled evaluation set for the
current knowledge base. It is outside `knowledge/` so questions and expected
answers cannot accidentally become retrieval evidence.

## Run

From the project root:

```bash
.venv/bin/python -m market_research.evaluate
```

If the embedding model is already cached, avoid Hugging Face network checks:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m market_research.evaluate
```

After reinstalling the editable package, `market-research-evaluate` is an
equivalent command. Options include `--cases`, `--output`, `--knowledge-dir`,
`--index-dir`, and `--k`.

This command runs the real local embedding model and retriever. It shares one
loaded model/index across questions and directly supplies each case's tickers.
It does not evaluate request parsing or call OpenCode, Tavily, or Yahoo Finance.
Normal index reuse/rebuild rules still apply.

## What a case means

```json
{
  "id": "apple-reportable-segments",
  "question": "According to Apple's 2025 Form 10-K, what are its reportable segments?",
  "tickers": ["AAPL"],
  "answerable": true,
  "expected_evidence": [{
    "source": "raw/apple/annual_report_2025.pdf",
    "pages": [5, 50],
    "required_terms": ["Americas", "Europe", "Greater China", "Japan", "Rest of Asia Pacific"]
  }],
  "notes": "Geographic segments, not product categories."
}
```

- `source` is the exact path relative to `knowledge/`.
- `pages` lists acceptable one-based PDF page numbers. Use an empty list for
  Markdown. These are viewer page numbers, not printed report page numbers.
- `required_terms` must all occur in a single returned passage from an acceptable
  source/page. Casing, whitespace/line wraps, and Unicode presentation differences
  are normalized. Multiple `expected_evidence` entries are alternative answers.
- Unanswerable cases have `answerable: false` and an empty evidence list. They
  currently test the strict expectation that retrieval returns no passages.

Check labels against the source text before interpreting failures. Update the
cases deliberately when the corpus changes; for example, adding Apple's 2023
report changes the missing-report case's premise.

## Read the metrics

For answerable questions:

- **Source hit rate at k:** fraction with an expected source/page anywhere among
  the returned passages. At k=2, this means in either of the two results.
- **Evidence hit rate at k:** fraction with both the expected location and all
  required terms in one passage. Finding a page alone is not enough.
- **Mean reciprocal rank:** average of `1 / first expected source/page rank`,
  with zero for a miss. A first-place hit contributes 1; second place contributes
  0.5. This metric checks location, while evidence hit rate checks required facts.

For unanswerable questions, **empty-result rate** is the fraction that returns
no passages without an error. A failed retrieval operation never counts as a
successful abstention.

These metrics evaluate retrieval against the labels. They do not measure answer
correctness, all returned passages' relevance, or whether a final answer cites
its claims correctly. Term matching is a simple evidence-presence check, not a
semantic judge. Related context may still allow an answer model to explain
that a question cannot be answered; returning it fails our strict empty-result
test without proving that the answer model would invent an answer.

## Saved results and initial baseline

The default output is `evaluations/results/latest.json`, ignored by Git. The
case data and evaluation code are intended to be versioned. Results include
each question, expected evidence, actual passages, retrieval latency, settings,
dataset hash, retriever-code hash, and the corpus/index signature. Use `--output`
with a new filename under `evaluations/results/` to keep comparison runs.

The initial run on the current corpus and default retrieval settings found:

- 8/8 answerable cases passed their evidence checks, all at the first result.
- 1/2 strict unanswerable cases returned no passages.
- Overall: 9/10 passed.
- The exact-future-share-price case returned related but non-answering financial
  text. Preserve this failure as a baseline for later abstention work.

The process exits with status 0 when every case passes, 1 when an evaluation
case fails, and 2 for setup/configuration errors. A status of 1 for the initial
baseline is expected. Ordinary unit tests separately check evaluation scoring
and verify that answerable labels exist in the original source pages, without
running embeddings or an answer model.
