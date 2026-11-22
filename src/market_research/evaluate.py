"""Evaluate retrieval on labelled local documents without calling an answer LLM."""

import argparse
import hashlib
import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from .rag import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_PERSIST_DIRECTORY,
    DEFAULT_RETRIEVAL_K,
    ScoreThresholdRetriever,
    build_knowledge_retriever,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES_PATH = PROJECT_ROOT / "evaluations" / "retrieval_cases.json"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "evaluations" / "results" / "latest.json"
NonEmptyText = Annotated[str, Field(min_length=1)]


class EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ExpectedEvidence(EvaluationModel):
    source: NonEmptyText
    pages: list[PositiveInt] = Field(default_factory=list)
    required_terms: list[NonEmptyText] = Field(min_length=1)


class RetrievalCase(EvaluationModel):
    id: NonEmptyText
    question: NonEmptyText
    tickers: list[Annotated[str, Field(pattern=r"^[A-Z]{1,5}$")]]
    answerable: bool
    expected_evidence: list[ExpectedEvidence]
    notes: NonEmptyText

    @model_validator(mode="after")
    def check_expectations(self):
        if self.answerable != bool(self.expected_evidence):
            raise ValueError("Only answerable cases must declare expected evidence.")
        return self


class RetrievalDataset(EvaluationModel):
    version: PositiveInt
    cases: list[RetrievalCase] = Field(min_length=1)

    @model_validator(mode="after")
    def check_unique_ids(self):
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("Evaluation case IDs must be unique.")
        return self


def normalize_evidence(text: str) -> str:
    """Ignore PDF line wraps, Unicode presentation differences, and casing."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def score_case(case: RetrievalCase, documents: list[Document]) -> dict:
    """Check expected locations and facts independently on returned passages.

    Each expected-evidence entry is an acceptable alternative. All its terms
    must appear together in a passage at the specified source/page.
    """
    source_rank = None
    evidence_rank = None
    for rank, document in enumerate(documents, start=1):
        for expected in case.expected_evidence:
            if document.metadata.get("source") != expected.source:
                continue
            if expected.pages and document.metadata.get("page_number") not in expected.pages:
                continue
            if source_rank is None:
                source_rank = rank
            text = normalize_evidence(document.page_content)
            if all(normalize_evidence(term) in text for term in expected.required_terms):
                if evidence_rank is None:
                    evidence_rank = rank

    passed = evidence_rank is not None if case.answerable else not documents
    failure_reason = None
    if not passed:
        if not case.answerable:
            failure_reason = "Returned passages for an unanswerable case (strict empty-result check)."
        elif source_rank is None:
            failure_reason = "Expected source/page was not retrieved."
        else:
            failure_reason = "Expected source/page found, but its excerpts lack required terms."
    return {
        "source_rank": source_rank,
        "evidence_rank": evidence_rank,
        "passed": passed,
        "failure_reason": failure_reason,
        "retrieved": [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents],
    }


def evaluate_retrieval(dataset: RetrievalDataset, retriever: ScoreThresholdRetriever) -> dict:
    """Reuse the loaded embedding model/index and apply each case's tickers."""
    results = []
    for case in dataset.cases:
        scoped_retriever = retriever.model_copy(update={"tickers": case.tickers or None})
        started = perf_counter()
        try:
            result = score_case(case, scoped_retriever.invoke(case.question))
        except Exception as exc:
            # A retrieval error is a failure, never a successful abstention.
            result = {"source_rank": None, "evidence_rank": None,
                      "passed": False, "retrieved": [], "error": str(exc)}
        results.append({
            **case.model_dump(), **result,
            "retrieval_seconds": round(perf_counter() - started, 4),
        })

    answerable = [result for result in results if result["answerable"]]
    unanswerable = [result for result in results if not result["answerable"]]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "k": retriever.k, "candidate_k": retriever.candidate_k,
            "vector_score_threshold": retriever.score_threshold,
            "embedding_model": os.getenv("RAG_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL,
        },
        "metrics": {
            "total": len(results),
            "passed": sum(result["passed"] for result in results),
            "errors": sum("error" in result for result in results),
            "answerable_cases": len(answerable),
            "unanswerable_cases": len(unanswerable),
            "source_hit_rate_at_k": (
                sum(result["source_rank"] is not None for result in answerable) / len(answerable)
                if answerable else None
            ),
            "evidence_hit_rate_at_k": (
                sum(result["evidence_rank"] is not None for result in answerable) / len(answerable)
                if answerable else None
            ),
            "mean_reciprocal_rank": (
                sum(1 / result["source_rank"] if result["source_rank"] else 0 for result in answerable)
                / len(answerable) if answerable else None
            ),
            "empty_result_rate_on_unanswerable": (
                sum(result["passed"] for result in unanswerable) / len(unanswerable)
                if unanswerable else None
            ),
        },
        "results": results,
    }


def main() -> int:
    """Run real retrieval, print a baseline, and save evidence for inspection."""
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--knowledge-dir", type=Path)
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--k", type=int, default=DEFAULT_RETRIEVAL_K)
    args = parser.parse_args()
    try:
        raw_dataset = args.cases.read_bytes()
        dataset = RetrievalDataset.model_validate_json(raw_dataset)
        retriever = build_knowledge_retriever(
            knowledge_dir=args.knowledge_dir, persist_directory=args.index_dir, k=args.k,
        )
        report = evaluate_retrieval(dataset, retriever)
        index_dir = args.index_dir or Path(os.getenv("RAG_INDEX_DIR") or DEFAULT_PERSIST_DIRECTORY)
        manifest = json.loads((index_dir / f"{DEFAULT_COLLECTION_NAME}.manifest.json").read_text())
        report["dataset"] = {
            "path": str(args.cases.resolve()), "version": dataset.version,
            "sha256": hashlib.sha256(raw_dataset).hexdigest(),
        }
        report["index_signature"] = manifest["index_signature"]
        report["retriever_code_sha256"] = hashlib.sha256(
            Path(__file__).with_name("rag.py").read_bytes()
        ).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    for result in report["results"]:
        status = "ERROR" if "error" in result else "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {result['id']}")
        for passage in result["retrieved"]:
            metadata = passage["metadata"]
            page = metadata.get("page_number")
            suffix = f", PDF page {page}" if page is not None else ""
            print(f"  {metadata.get('source', 'Unknown source')}{suffix}")
        if "error" in result:
            print(f"  {result['error']}")
        elif result["failure_reason"]:
            print(f"  {result['failure_reason']}")
    print(f"\nPassed: {report['metrics']['passed']}/{report['metrics']['total']}")
    for metric in ("source_hit_rate_at_k", "evidence_hit_rate_at_k",
                   "mean_reciprocal_rank", "empty_result_rate_on_unanswerable"):
        value = report["metrics"][metric]
        print(f"{metric}: {value:.3f}" if value is not None else f"{metric}: n/a")
    print(f"Evidence and results: {args.output}")
    return 0 if all(result["passed"] for result in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
