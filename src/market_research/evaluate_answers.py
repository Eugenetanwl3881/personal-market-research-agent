"""Generate final answers with real retrieved evidence and prepare a human review."""

import argparse
import hashlib
import json
import os
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from pydantic import Field, model_validator

from .answer import prepare_reference_evidence, write_market_answer
from .evaluate import (
    DEFAULT_CASES_PATH,
    PROJECT_ROOT,
    EvaluationModel,
    NonEmptyText,
    RetrievalDataset,
    normalize_evidence,
    score_case,
)
from .rag import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_PERSIST_DIRECTORY,
    DEFAULT_RETRIEVAL_K,
    ScoreThresholdRetriever,
    build_knowledge_retriever,
)


DEFAULT_ANSWER_CASES_PATH = PROJECT_ROOT / "evaluations" / "answer_cases.json"
DEFAULT_ANSWER_RESULTS_PATH = PROJECT_ROOT / "evaluations" / "results" / "answers_latest.json"

# These patterns detect review signals, not whether all claims are supported.
LIMITATION_PATTERNS = (
    r"\b(?:cannot|can't|could not|couldn't|unable to)\b.{0,100}\b(?:verify|determine|provide|confirm|answer|predict|forecast|access|retrieve)\b",
    r"\b(?:not|isn't|aren't|wasn't|weren't|no)\b.{0,100}\b(?:provided|available|included|present|given|forecast|prediction|evidence|information|access)\b",
    r"\b(?:does not|doesn't|do not|don't)\b.{0,100}\b(?:contain|provide|include|predict|forecast|specify|state)\b",
    r"\b(?:lack|lacks|missing|insufficient)\b.{0,70}\b(?:report|evidence|information|data)\b",
)
CURRENCY_AMOUNT_PATTERN = re.compile(
    r"(?:\$|\bUSD\s*)\s*\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s*(?:USD|dollars)\b",
    re.IGNORECASE,
)


class AnswerCase(EvaluationModel):
    retrieval_case_id: NonEmptyText
    expected_behavior: NonEmptyText
    required_terms: list[NonEmptyText] = Field(default_factory=list)
    require_limitation: bool = False
    flag_currency_amounts: bool = False
    review_questions: list[NonEmptyText] = Field(min_length=1)


class AnswerDataset(EvaluationModel):
    version: int = Field(gt=0)
    cases: list[AnswerCase] = Field(min_length=1)

    @model_validator(mode="after")
    def check_unique_cases(self):
        ids = [case.retrieval_case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("Answer evaluation case IDs must be unique.")
        return self


def check_answer_signals(case: AnswerCase, narrative: str) -> dict:
    """Check the answer summary, excluding appended limitations/disclaimers."""
    normalized = normalize_evidence(narrative).replace("’", "'")
    checks = {"nonempty_narrative": bool(normalized)}
    missing_terms = [term for term in case.required_terms
                     if normalize_evidence(term) not in normalized]
    if case.required_terms:
        checks["required_terms_present"] = not missing_terms
    if case.require_limitation:
        checks["limitation_language_detected"] = any(
            re.search(pattern, normalized) for pattern in LIMITATION_PATTERNS
        )
    amounts = CURRENCY_AMOUNT_PATTERN.findall(narrative) if case.flag_currency_amounts else []
    if case.flag_currency_amounts:
        checks["no_currency_amount_detected"] = not amounts
    return {"checks": checks, "missing_terms": missing_terms, "currency_amounts": amounts}


def evaluate_answers(
    dataset: AnswerDataset,
    retrieval_dataset: RetrievalDataset,
    retriever: ScoreThresholdRetriever,
    answer_writer: Callable = write_market_answer,
) -> dict:
    """Isolate retrieval + final-answer generation using labelled request inputs.

    Expectations are used only after generation; the model receives the normal
    question and retrieved context. Parsing, guardrails, live quotes, and news
    are outside this reference-answer evaluation.
    """
    retrieval_cases = {case.id: case for case in retrieval_dataset.cases}
    unknown_ids = {case.retrieval_case_id for case in dataset.cases} - retrieval_cases.keys()
    if unknown_ids:
        raise ValueError(f"Unknown retrieval case IDs: {', '.join(sorted(unknown_ids))}")

    results = []
    for case in dataset.cases:
        retrieval_case = retrieval_cases[case.retrieval_case_id]
        result = {
            "id": retrieval_case.id,
            "question": retrieval_case.question,
            "answerable": retrieval_case.answerable,
            "expectations": case.model_dump(),
            "status": "error",
            "human_review": {"verdict": "pending", "notes": ""},
        }
        started = perf_counter()
        try:
            scoped = retriever.model_copy(update={"tickers": retrieval_case.tickers or None})
            documents = scoped.invoke(retrieval_case.question)
            result["retrieval"] = score_case(retrieval_case, documents)
            result["retrieval_seconds"] = round(perf_counter() - started, 4)
            context = [{"content": doc.page_content, "source": doc.metadata.get("source", "Unknown source"),
                        "metadata": doc.metadata} for doc in documents]
            result["reference_evidence"] = prepare_reference_evidence(context)
            state = {
                "question": retrieval_case.question,
                "intent": "context", "ticker": retrieval_case.tickers,
                "needs_context": True, "needs_quote": False, "needs_news": False,
                "context": context, "context_status": "ok" if context else "no_results",
            }
            narrative_parts = []
            narrative_sources = set()

            def capture_narrative(event):
                if event.get("type") == "answer_token":
                    narrative_parts.append(event["text"])
                    narrative_sources.add("model")
                elif event.get("type") == "answer_section" and event.get("section") == "summary":
                    narrative_parts.append(event["text"])
                    narrative_sources.add(event.get("origin", "application"))

            started = perf_counter()
            result["final_answer"] = answer_writer(state, stream_writer=capture_narrative)
            result["answer_seconds"] = round(perf_counter() - started, 4)
            result["narrative"] = "".join(narrative_parts).strip()
            result["narrative_source"] = (
                next(iter(narrative_sources)) if len(narrative_sources) == 1
                else "mixed" if narrative_sources else "unknown"
            )
            signals = check_answer_signals(case, result["narrative"])
            if retrieval_case.answerable:
                signals["checks"]["expected_evidence_retrieved"] = (
                    result["retrieval"]["evidence_rank"] is not None
                )
                signals["checks"]["expected_reference_rendered"] = any(
                    expected.source in result["final_answer"] and (
                        not expected.pages or any(
                            f"{expected.source}, PDF page {page}" in result["final_answer"]
                            for page in expected.pages
                        )
                    ) for expected in retrieval_case.expected_evidence
                )
            result.update(signals)
            result["status"] = (
                "checks_passed_review_required" if all(signals["checks"].values())
                else "flagged_review_required"
            )
        except Exception as exc:
            result["error"] = str(exc)
            result["human_review"]["verdict"] = "not_generated"
        results.append(result)

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "retrieval and reference-answer generation; parsing, guardrails, quote/news APIs excluded",
        "settings": {
            "k": retriever.k, "candidate_k": retriever.candidate_k,
            "vector_score_threshold": retriever.score_threshold,
            "embedding_model": os.getenv("RAG_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL,
            "answer_model": os.getenv("OPENCODE_MODEL", "deepseek-v4-flash"),
            "answer_base_url": os.getenv("OPENCODE_BASE_URL", "https://opencode.ai/zen/go/v1"),
        },
        "summary": {
            "total": len(results),
            "checks_passed": sum(r["status"] == "checks_passed_review_required" for r in results),
            "flagged": sum(r["status"] == "flagged_review_required" for r in results),
            "errors": sum(r["status"] == "error" for r in results),
            "human_reviews_pending": sum(r["status"] != "error" for r in results),
            "application_summaries": sum(r.get("narrative_source") == "application" for r in results),
        },
        "results": results,
    }


def render_review(report: dict) -> str:
    """Create a readable companion report for inspecting claims and evidence."""
    lines = ["# Final-answer evaluation review", "",
             "Automated checks are signals, not correctness grades. Human review is pending.",
             "Model-written summaries receive the normal question and evidence, not expectations.",
             "Application-written summaries bypass the answer model when reference evidence is missing.", ""]
    for result in report["results"]:
        lines.extend([f"## {result['id']}", "", f"Question: {result['question']}", "",
                      f"Expected behavior: {result['expectations']['expected_behavior']}", "",
                      f"Status: `{result['status']}`", ""])
        if "error" in result:
            lines.extend([f"Error: {result['error']}", ""])
            continue
        else:
            lines.extend(["### Automated signals", ""])
            lines.extend(f"- {'OK' if value else 'FLAG'}: {name}" for name, value in result["checks"].items())
            lines.extend(["", "### Generated answer", "", result["final_answer"], "",
                          f"Summary source: `{result.get('narrative_source', 'unknown')}`", "",
                          "### Prepared reference evidence", ""])
            for evidence in result["reference_evidence"]:
                label = evidence["source"]
                if "page_number" in evidence:
                    label += f", PDF page {evidence['page_number']}"
                lines.extend([f"Source: {label}", "", "> " + evidence["content"], ""])
            if not result["reference_evidence"]:
                lines.extend(["No reference excerpts were available.", ""])
        lines.extend(["### Human review", "", "Verdict: pending", ""])
        lines.extend(f"- [ ] {question}" for question in result["expectations"]["review_questions"])
        lines.extend(["", "Notes:", ""])
    return "\n".join(lines)


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_ANSWER_CASES_PATH)
    parser.add_argument("--retrieval-cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_ANSWER_RESULTS_PATH)
    parser.add_argument("--knowledge-dir", type=Path)
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--k", type=int, default=DEFAULT_RETRIEVAL_K)
    args = parser.parse_args()
    try:
        if args.output.suffix.lower() != ".json":
            raise ValueError("--output must end in .json; its .md companion holds the review.")
        raw_cases = args.cases.read_bytes()
        raw_retrieval_cases = args.retrieval_cases.read_bytes()
        dataset = AnswerDataset.model_validate_json(raw_cases)
        retrieval_dataset = RetrievalDataset.model_validate_json(raw_retrieval_cases)
        if not os.getenv("OPENCODE_API_KEY"):
            raise ValueError("OPENCODE_API_KEY is not configured; final-answer evaluation calls your model.")
        retriever = build_knowledge_retriever(
            knowledge_dir=args.knowledge_dir, persist_directory=args.index_dir, k=args.k,
        )
        report = evaluate_answers(dataset, retrieval_dataset, retriever)
        index_dir = args.index_dir or Path(os.getenv("RAG_INDEX_DIR") or DEFAULT_PERSIST_DIRECTORY)
        manifest = json.loads((index_dir / f"{DEFAULT_COLLECTION_NAME}.manifest.json").read_text())
        report["index_signature"] = manifest["index_signature"]
        report["provenance"] = {
            "answer_cases_version": dataset.version,
            "answer_cases_sha256": hashlib.sha256(raw_cases).hexdigest(),
            "retrieval_cases_version": retrieval_dataset.version,
            "retrieval_cases_sha256": hashlib.sha256(raw_retrieval_cases).hexdigest(),
            "retriever_code_sha256": hashlib.sha256(Path(__file__).with_name("rag.py").read_bytes()).hexdigest(),
            "indexing_code_sha256": hashlib.sha256(Path(__file__).with_name("indexing.py").read_bytes()).hexdigest(),
            "answer_code_sha256": hashlib.sha256(Path(__file__).with_name("answer.py").read_bytes()).hexdigest(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        review_path = args.output.with_suffix(".md")
        review_path.write_text(render_review(report), encoding="utf-8")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    for result in report["results"]:
        print(f"[{result['status'].upper()}] {result['id']}")
        if "error" in result:
            print(f"  {result['error']}")
    print(f"\nSummary: {report['summary']}")
    print(f"Results: {args.output}")
    print(f"Review answers and evidence: {review_path}")
    return 1 if report["summary"]["errors"] or report["summary"]["flagged"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
