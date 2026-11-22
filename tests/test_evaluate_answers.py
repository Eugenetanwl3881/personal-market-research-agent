from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from market_research.answer import write_market_answer
from market_research.evaluate import DEFAULT_CASES_PATH, RetrievalDataset
from market_research.evaluate_answers import (
    DEFAULT_ANSWER_CASES_PATH,
    AnswerDataset,
    check_answer_signals,
    evaluate_answers,
    render_review,
)


def datasets(case_id):
    dataset = AnswerDataset.model_validate_json(DEFAULT_ANSWER_CASES_PATH.read_bytes())
    case = next(case for case in dataset.cases if case.retrieval_case_id == case_id)
    return (
        AnswerDataset(version=dataset.version, cases=[case]),
        RetrievalDataset.model_validate_json(DEFAULT_CASES_PATH.read_bytes()),
    )


def fake_retriever(documents):
    return SimpleNamespace(
        k=2, candidate_k=20, score_threshold=0.45,
        model_copy=lambda update: SimpleNamespace(invoke=lambda question: documents),
    )


def mock_answer_model(monkeypatch, narrative):
    prompts = []

    class Model:
        def stream(self, prompt):
            prompts.append(prompt)
            yield SimpleNamespace(content=narrative)

    monkeypatch.setattr("market_research.answer.create_model", lambda: Model())
    return prompts


def test_supported_answer_uses_normal_evidence_and_remains_pending_review(monkeypatch):
    dataset, retrieval_dataset = datasets("apple-reportable-segments")
    # Unique expectations must never leak into the prompt given to the model.
    dataset.cases[0].expected_behavior = "EVALUATOR_ONLY_SENTINEL"
    text = "Apple's geographic segments are Americas, Europe, Greater China, Japan and Rest of Asia Pacific."
    prompts = mock_answer_model(monkeypatch, text)
    docs = [Document(page_content=text, metadata={
        "source": "raw/apple/annual_report_2025.pdf", "page_number": 50,
    })]

    report = evaluate_answers(dataset, retrieval_dataset, fake_retriever(docs), write_market_answer)
    result = report["results"][0]

    assert result["status"] == "checks_passed_review_required"
    assert result["human_review"]["verdict"] == "pending"
    assert result["checks"]["expected_reference_rendered"] is True
    assert result["checks"]["expected_evidence_retrieved"] is True
    assert len(prompts) == 1
    assert "EVALUATOR_ONLY_SENTINEL" not in prompts[0]
    assert text in prompts[0]
    assert result["reference_evidence"][0]["page_number"] == 50
    assert text == result["narrative"]
    assert "PDF page 50" in render_review(report)
    assert "Verdict: pending" in render_review(report)


def test_related_retrieval_can_fail_empty_check_while_answer_acknowledges_limits(monkeypatch):
    dataset, retrieval_dataset = datasets("unsupported-exact-price-forecast")
    mock_answer_model(monkeypatch, "The supplied excerpts do not provide an exact future price prediction.")
    docs = [Document(page_content="Historical net sales were reported.", metadata={
        "source": "raw/apple/annual_report_2025.pdf", "page_number": 39,
    })]

    result = evaluate_answers(
        dataset, retrieval_dataset, fake_retriever(docs), write_market_answer,
    )["results"][0]

    assert result["retrieval"]["passed"] is False
    assert result["status"] == "checks_passed_review_required"
    assert result["checks"]["limitation_language_detected"] is True
    assert result["human_review"]["verdict"] == "pending"


def test_static_limitations_and_disclaimer_do_not_satisfy_narrative_check(monkeypatch):
    dataset, retrieval_dataset = datasets("missing-apple-report-year")
    mock_answer_model(monkeypatch, "Apple operates across several geographic regions.")
    result = evaluate_answers(
        dataset, retrieval_dataset, fake_retriever([]), write_market_answer,
    )["results"][0]

    assert "No relevant reference documents were found" in result["final_answer"]
    assert result["checks"]["limitation_language_detected"] is False
    assert result["status"] == "flagged_review_required"


def test_disclaimer_plus_invented_price_is_flagged():
    dataset, _ = datasets("unsupported-exact-price-forecast")
    signals = check_answer_signals(
        dataset.cases[0],
        "I cannot verify a forecast from the evidence, but AAPL will close at $1,234.50.",
    )
    assert signals["checks"]["limitation_language_detected"] is True
    assert signals["checks"]["no_currency_amount_detected"] is False
    assert signals["currency_amounts"] == ["$1,234.50"]


def test_checks_do_not_certify_semantic_correctness():
    dataset, _ = datasets("unsupported-exact-price-forecast")
    signals = check_answer_signals(
        dataset.cases[0],
        "I cannot verify a forecast, but it will close at one thousand per share.",
    )
    # Spelled-out predictions can escape these heuristics: human review matters.
    assert all(signals["checks"].values())


def test_generation_failure_is_recorded_without_counting_as_abstention():
    dataset, retrieval_dataset = datasets("missing-apple-report-year")

    def broken_writer(state, stream_writer):
        raise RuntimeError("Provider unavailable")

    report = evaluate_answers(dataset, retrieval_dataset, fake_retriever([]), broken_writer)
    assert report["summary"]["errors"] == 1
    assert report["summary"]["checks_passed"] == 0
    assert report["results"][0]["error"] == "Provider unavailable"
    assert report["results"][0]["human_review"]["verdict"] == "not_generated"
    assert report["summary"]["human_reviews_pending"] == 0


def test_unknown_or_duplicate_cases_are_rejected_before_generation():
    dataset, retrieval_dataset = datasets("missing-apple-report-year")
    with pytest.raises(ValidationError, match="unique"):
        AnswerDataset(version=1, cases=dataset.cases * 2)
    dataset.cases[0].retrieval_case_id = "unknown"
    with pytest.raises(ValueError, match="Unknown retrieval case"):
        evaluate_answers(dataset, retrieval_dataset, fake_retriever([]))


def test_saved_answer_cases_resolve_to_existing_retrieval_cases():
    dataset = AnswerDataset.model_validate_json(DEFAULT_ANSWER_CASES_PATH.read_bytes())
    retrieval_dataset = RetrievalDataset.model_validate_json(DEFAULT_CASES_PATH.read_bytes())
    assert {case.retrieval_case_id for case in dataset.cases} <= {
        case.id for case in retrieval_dataset.cases
    }


def test_invalid_output_is_rejected_before_any_model_or_retrieval_work(monkeypatch):
    from market_research.evaluate_answers import main

    def unexpected_retrieval(**kwargs):
        raise AssertionError("Should validate the output path before doing work")

    monkeypatch.setattr("sys.argv", ["evaluate_answers", "--output", "invalid.txt"])
    monkeypatch.setattr("market_research.evaluate_answers.build_knowledge_retriever", unexpected_retrieval)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
