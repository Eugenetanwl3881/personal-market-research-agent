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

## Final-answer evaluation

`answer_cases.json` defines expectations and review questions for three cases:
the supported Apple segments question, the missing 2023 report, and the
unsupported exact future-price forecast. Each references a question from
`retrieval_cases.json` so the wording and retrieval labels stay consistent.

Run:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m market_research.evaluate_answers
```

Use offline mode only when the embedding model is cached. It prevents Hugging
Face checks; the answer model still requires network access and a configured
`OPENCODE_API_KEY`. Each of the three default cases makes one streamed answer
request to the configured OpenCode model and may incur provider usage. No judge
model is called. After reinstalling the editable package,
`market-research-evaluate-answers` runs the same command.

This evaluation directly reuses the existing retriever and `write_market_answer`
with labelled tickers and reference-only request state. It isolates the final
answer stage: it does not exercise graph routing, scope guardrails, request
parsing, live quote/news retrieval, or graph fallback answers. Expectations and
review questions are never sent to the answer model.

The runner saves two Git-ignored files:

- `evaluations/results/answers_latest.json`: actual retrieved passages, the
  exact prepared reference excerpts, generated narrative and full answer,
  mechanical checks, model/settings, latency, hashes, and pending review fields.
- `evaluations/results/answers_latest.md`: a readable comparison of each
  question, expectation, actual answer, and supplied evidence, with a review
  checklist. Start your review here.

The default filenames are overwritten on the next run. Keep a baseline or
completed review with a different output name, for example:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m market_research.evaluate_answers \
  --output evaluations/results/answers_baseline.json
```

The `.md` companion uses the same output stem. The JSON and Markdown are review
artifacts, not a synchronized review application; edits in one do not update
the other automatically.

### Automated signals and human review

The automated checks look for:

- A nonempty model-written narrative.
- The required region names, supporting retrieved evidence, and a rendered
  expected reference for the supported case.
- Limitation language in the model-written narrative for unsupported cases.
  Python's static limitations section and disclaimer cannot satisfy this check.
- Currency amounts that deserve inspection in the forecast case.

Results are `checks_passed_review_required`, `flagged_review_required`, or
`error`. Neither of the first two is a final correctness verdict. A limitation
phrase can coexist with an invented claim; a spelled-out prediction can evade
the currency detector; and a valid paraphrase can fail a language pattern.
Review the whole answer and whether each claim is supported by its evidence.

The old retrieval empty-result check is recorded separately. An unanswerable
case can fail that strict retrieval check yet produce an appropriate answer
acknowledging that the retrieved text does not establish a prediction.

Exit code 0 means the automated signals passed with no execution errors;
human review is still required. Exit code 1 means at least one case was flagged
or could not run. Exit code 2 means setup/configuration failed. Provider errors
are recorded as errors, never as successful abstentions. Ordinary tests mock
answer generation and do not spend model credits.

### First live answer run

Two cases passed the automated checks and one was flagged; all generated
answers still require human review. Inspection of that run found:

- The supported answer listed all five geographic segments and displayed a
  reference to PDF page 50.
- The missing-2023-report answer claimed contents of the 2023 report despite
  receiving no reference excerpts. It did not acknowledge the evidence gap in
  its narrative. This is an actual unsupported-answer failure, even though
  Python appended a static "No relevant reference documents were found" line.
- The future-price answer provided no numerical prediction and passed the
  limitation-language check. Review its scope carefully: saying that the whole
  report contains no prediction is stronger than saying that the supplied
  excerpts do not establish one.

These observations establish a baseline, not a guarantee of how another model
run will behave. The next production improvement is to address unsupported
answers when reference evidence is empty, then rerun these same cases.
