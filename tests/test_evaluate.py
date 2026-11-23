from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from market_research.evaluate import (
    DEFAULT_CASES_PATH,
    RetrievalCase,
    RetrievalDataset,
    evaluate_retrieval,
    normalize_evidence,
    score_case,
)
from market_research.rag import load_knowledge_documents


def make_case(**changes):
    return RetrievalCase.model_validate({
        "id": "segments", "question": "What are Apple's reportable segments?",
        "tickers": ["AAPL"], "answerable": True,
        "expected_evidence": [{"source": "apple.pdf", "pages": [50],
                               "required_terms": ["Greater China", "Japan"]}],
        "notes": "A small labelled case.", **changes,
    })


def passage(content, source="apple.pdf", page=50):
    return Document(page_content=content, metadata={"source": source, "page_number": page})


def test_correct_page_without_facts_is_not_an_evidence_hit():
    result = score_case(make_case(), [passage("The financial statements are audited.")])
    assert result["source_rank"] == 1
    assert result["evidence_rank"] is None
    assert result["passed"] is False


def test_evidence_requires_correct_source_and_page_and_normalizes_line_wraps():
    documents = [
        passage("Greater China and Japan", source="other.pdf"),
        passage("Greater China and Japan", page=80),
        passage("GREATER\nCHINA and Japan"),
    ]
    result = score_case(make_case(), documents)
    assert result["source_rank"] == 3
    assert result["evidence_rank"] == 3
    assert result["passed"] is True


def test_unknown_answer_has_separate_empty_result_expectation():
    case = make_case(answerable=False, expected_evidence=[])
    assert score_case(case, [])["passed"] is True
    assert score_case(case, [passage("Related financial information.")])["passed"] is False


def test_metrics_include_misses_and_do_not_count_errors_as_abstentions():
    cases = [make_case(), make_case(id="missing"),
             make_case(id="unknown", answerable=False, expected_evidence=[])]
    responses = iter([
        [passage("Unrelated", source="other.pdf"), passage("Greater China and Japan")],
        [],
        RuntimeError("Index unavailable"),
    ])
    ticker_scopes = []

    def scoped_copy(update):
        ticker_scopes.append(update["tickers"])

        def invoke(question):
            response = next(responses)
            if isinstance(response, Exception):
                raise response
            return response

        return SimpleNamespace(invoke=invoke)

    retriever = SimpleNamespace(model_copy=scoped_copy, k=2, candidate_k=20, score_threshold=0.45)
    report = evaluate_retrieval(RetrievalDataset(version=1, cases=cases), retriever)
    metrics = report["metrics"]
    assert metrics["source_hit_rate_at_k"] == 0.5
    assert metrics["evidence_hit_rate_at_k"] == 0.5
    assert metrics["mean_reciprocal_rank"] == 0.25
    assert metrics["empty_result_rate_on_unanswerable"] == 0
    assert metrics["errors"] == 1
    assert metrics["passed"] == 1
    assert ticker_scopes == [["AAPL"], ["AAPL"], ["AAPL"]]


def test_invalid_expectations_and_duplicate_case_ids_are_rejected():
    with pytest.raises(ValidationError, match="expected evidence"):
        make_case(answerable=False)
    with pytest.raises(ValidationError, match="unique"):
        RetrievalDataset(version=1, cases=[make_case(), make_case()])


def test_dataset_answerable_labels_are_supported_by_the_real_sources():
    """Validate labels without embeddings, network, or the answer model."""
    dataset = RetrievalDataset.model_validate_json(DEFAULT_CASES_PATH.read_bytes())
    documents = load_knowledge_documents()
    for case in dataset.cases:
        for expected in case.expected_evidence:
            for page in expected.pages or [None]:
                matches = [doc for doc in documents
                           if doc.metadata["source"] == expected.source
                           and doc.metadata.get("page_number") == page]
                assert any(
                    all(normalize_evidence(term) in normalize_evidence(doc.page_content)
                        for term in expected.required_terms)
                    for doc in matches
                ), (case.id, expected.source, page)


def test_real_report_metadata_keeps_apple_and_microsoft_distinct():
    documents = load_knowledge_documents()
    for company, ticker in [("apple", "AAPL"), ("microsoft", "MSFT")]:
        source = f"raw/{company}/annual_report_2025.pdf"
        pages = [document for document in documents if document.metadata["source"] == source]
        assert pages
        assert all(document.metadata["ticker"] == ticker for document in pages)
        assert all(document.metadata["ticker_scope"] == ticker for document in pages)
        assert all(document.metadata["fiscal_year"] == 2025 for document in pages)
