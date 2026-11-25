# Retrieval evaluation

`retrieval_cases.json` is a manually labelled evaluation set of fifteen questions
for Apple, Microsoft, and methodology documents. It is outside `knowledge/` so questions and expected
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
Normal incremental-index compatibility rules still apply. Reports also record
`indexing_code_sha256` alongside the retriever-code hash to trace ingestion changes.

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

Annual-report source metadata uses `fiscal_year: 2025` (an integer) for year
filtering and a separate `document_version: "1"` for revision identity. Tests
include a 2024 report with revision `"2025"` to verify that a 2025 question
cannot select it by mistaking the revision for the fiscal year. Metadata changes
alter the index signature, so keep previous evaluation runs when migrating.

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

`answer_cases.json` defines expectations and review questions for six cases:
the original Apple segments, missing-2023-report, and future-price cases, plus
Microsoft segments, employee count, and a missing-2024-report case. Each references a question from
`retrieval_cases.json` so the wording and retrieval labels stay consistent.

Run:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m market_research.evaluate_answers
```

Use offline mode only when the embedding model is cached. It prevents Hugging
Face checks; the answer model still requires network access and a configured
`OPENCODE_API_KEY`. Model-written summaries make one streamed answer request
each and may incur provider usage. The current default run uses the model for
the two supported segment questions, Microsoft's headcount, and the forecast
case. The two missing-report cases return application-written limitations
without summary-model calls.
No judge model is called. After reinstalling the editable package,
`market-research-evaluate-answers` runs the same command.

This evaluation directly reuses the existing retriever and `write_market_answer`
with labelled tickers and reference-only request state. It isolates the final
answer stage: it does not exercise graph routing, scope guardrails, request
parsing, live quote/news retrieval, or graph fallback answers. Expectations and
review questions are never sent to the answer model.

The runner saves two Git-ignored files:

- `evaluations/results/answers_latest.json`: actual retrieved passages, the
  exact prepared reference excerpts, generated narrative and full answer,
  summary origin (`model` or `application`), mechanical checks, model/settings,
  latency, hashes, and pending review fields.
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

- A nonempty answer summary, written by the model or the application.
- Required facts, supporting retrieved evidence, and a rendered expected
  reference for supported cases.
- Limitation language in the answer summary for unsupported cases.
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
run will behave.

### After the empty-evidence fix

The original `answers_latest.*` files were preserved for comparison. A run
after the fix is saved in `evaluations/results/answers_after_empty_evidence.*`.
All three cases passed their automated checks, with one application-written
summary. The missing-report summary now states that relevant evidence could
not be retrieved and the requested reference information cannot be verified.
The forecast answer also limits its statement to the supplied excerpts.

The application now bypasses summary generation when reference context was
requested but no usable excerpts were found. Whitespace-only excerpts count
as absent. Retrieval errors produce a different limitation message. Mixed
requests still render available quotes/calculations and can summarize news.
No missing report contents are supplied by a model in this fallback path.

The initial three answer cases were a small starting set; version 2 adds three
Microsoft cases. Each model-written answer costs a provider request and human
review time. Broader coverage is
still needed for irrelevant or incomplete evidence, multiple reports, and
misleading claims inside otherwise plausible answers. Ordinary regression tests
already cover blank/error evidence, streaming, mixed requests, and the graph
path without using live services. Passing this set does not establish general
answer correctness.

## Microsoft report expansion (dataset version 2)

The corpus now includes Microsoft's fiscal-2025 Form 10-K from its official
investor-relations host, stored as
`knowledge/raw/microsoft/annual_report_2025.pdf` with a YAML sidecar. The PDF has
158 pages. Expectations were checked against its extracted source text:

- Segments: PDF pages 5, 36, and 82.
- Employee counts: PDF page 9.
- Intelligent Cloud product examples: PDF pages 6 and 83.

Five retrieval cases were added: segments, a paraphrase, headcount, Intelligent
Cloud, and a missing report year. The original ten questions and their expected
sources/pages/terms are retained; only the overview case's explanatory note was
updated to reflect the larger corpus. All original pass/fail outcomes remained
the same after the addition. The overview question still intentionally tests
retrieval of `microsoft.md`; report-specific questions target the PDF.

The verified-label run is saved separately as
`evaluations/results/retrieval_after_msft_verified_labels.json`:

- 13/15 cases passed, with no execution errors.
- 12/12 answerable cases found an acceptable source/page.
- 11/12 answerable cases found the required facts in a returned passage.
- Source/page MRR (still named `mean_reciprocal_rank`): approximately 0.958.
- Two of three strict unanswerable cases returned no passages.

The direct Microsoft segment question retrieves page 61's accounting guidance
and a page-36 excerpt that omits the full segment list. Its paraphrase retrieves
the list on page 5. Page 36 was added to the valid location labels after checking
its complete source text; the direct question still fails the required-facts
check. The other failure remains the original strict future-price empty-result
case. Preserve both failures rather than adjusting ranking to make this set pass.

Ranking code, weights, thresholds, candidate count, final k, and metric
definitions were kept unchanged. A new corpus and new questions measure a
different condition, so compare the original ten cases separately as well as
the expanded totals. All retrieved passages for company-scoped questions were
also checked for company-metadata consistency.

The six-case live answer run is saved in
`evaluations/results/answers_after_msft.json` with its readable `.md` companion.
Five cases passed automated signals, and the direct Microsoft segment question
was flagged for missing names and supporting retrieved facts. Its answer
appropriately acknowledged that the supplied excerpts do not establish the
complete list. This is an answer-completeness/retrieval failure under the case
expectation, not an invented segment list. The employee answer used the verified
historical total and date. Both missing-report summaries came from the
application. All six still have pending human review fields.

## PDF context window at page boundaries

The Microsoft segment failure exposed unused context budget at a page boundary.
The search hit was the final chunk on PDF page 36. The old expansion included
one preceding chunk and no following chunk, producing only two chunks. The
complete list was two chunks before the hit. The updated expansion shifts the
window to retain up to three contiguous chunks: two preceding an end hit, or
two following a start hit, while staying on the same page. Interior windows
still include one preceding and one following chunk.

This changes evidence assembly without changing the stored corpus, embeddings,
source-page order, ranking weights, thresholds, candidate count, final k, or
metric definitions. Dataset and index signatures match the pre-fix run; only
the retriever-code hash changes. Regression tests cover both page boundaries,
the three-chunk budget, and exclusion of other pages.

The unchanged fifteen-case dataset now passes 14/15 cases. All twelve answerable
questions contain the required facts, with no previous passing case regressing.
Both Microsoft segment phrasings pass: the direct question finds the complete
list in its second passage (page 36), and the paraphrase in its first (page 5).
The facts survive the answer prompt's evidence normalization and character cap.
The existing strict future-price empty-result case remains the sole failure.
Source/page MRR stays approximately 0.958, illustrating that evidence quality
can improve without changing the positions of the source pages.

The six-case live answer evaluation passes all automated checks. The direct
Microsoft answer now lists all three segments. Human review fields remain
pending; automated signals do not establish general correctness.

The previous reports are preserved. Compare them with:

- `evaluations/results/retrieval_after_page_edge_context.json`
- `evaluations/results/answers_after_page_edge_context.json`
- `evaluations/results/answers_after_page_edge_context.md`

## Incremental-ingestion verification

The version-2 manifest introduces document IDs, chunk IDs, and exact-text hashes.
The existing version-1 index therefore receives one migration rebuild. Keep its
older evaluation reports: the index signature changes even though source text,
ranking settings, and evaluation labels have not changed.

The real corpus migration produced 1,575 records from five source files and
embedded 1,536 unique chunk texts; 39 duplicate occurrences reused vectors.
An immediate second ingestion reported zero added/updated/deleted chunks,
1,575 unchanged chunks, and zero texts embedded.

`evaluations/results/retrieval_after_incremental_ingestion.json` preserves the
previous dataset, all returned passage text/source/page combinations, and all
metrics: 14/15 cases, all twelve answerable cases, source and evidence hit rates
of 1.0, and source-based MRR of approximately 0.958. The existing strict
future-price empty-result failure is unchanged. No new paid answer-model
evaluation was run for this ingestion change.

Reproduce the incremental checks with:

```bash
python -m pytest -q tests/test_indexing.py tests/test_ingest.py
HF_HUB_OFFLINE=1 market-research-index
HF_HUB_OFFLINE=1 market-research-index
HF_HUB_OFFLINE=1 python -m market_research.evaluate \
  --output evaluations/results/retrieval_after_incremental_ingestion.json
```

The first index command may migrate an older index; the second should reuse it.
Offline mode requires the local embedding model to be cached already.
