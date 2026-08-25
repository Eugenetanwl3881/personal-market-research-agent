import pytest

from market_research.guardrails import check_scope


@pytest.mark.parametrize(
    "question",
    [
        "What is the price of AAPL?",
        "How much would 15 shares of MSFT cost?",
        "What is the latest news about Nvidia?",
    ],
)
def test_market_research_questions_are_accepted(question):
    accepted, message = check_scope(question)

    assert accepted is True
    assert message == ""


@pytest.mark.parametrize(
    "question",
    [
        "Should I buy Tesla?",
        "Buy 10 shares of AAPL.",
        "How do I cook pasta?",
        "",
    ],
)
def test_unsafe_or_unrelated_questions_are_rejected(question):
    accepted, message = check_scope(question)

    assert accepted is False
    assert message