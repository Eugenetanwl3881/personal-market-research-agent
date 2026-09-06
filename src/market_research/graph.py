from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from .answer import create_model, write_market_answer
from .calculations import calculate_share_cost
from .data import get_stock_quote, search_market_news
from .guardrails import check_scope
from .parser import parse_market_request
from .state import MarketResearchState


def _complete(state: MarketResearchState, step: str) -> dict:
    """Record a completed step while the graph is still a learning skeleton."""
    return {"steps": [*state.get("steps", []), f"{step} completed"]}


def scope_check(state: MarketResearchState) -> dict:
    accepted, message = check_scope(state["question"])
    step_update = _complete(state, "scope_check")

    if accepted:
        return {
            **step_update,
            "intent": "market_research",
        }

    return {
        **step_update,
        "intent": "refused",
        "errors": [message],
        "final_answer": message,
    }


def route_after_scope_check(state: MarketResearchState) -> str:
    """Choose whether the graph continues or ends after scope validation."""
    if state.get("intent") == "refused":
        return "refused"

    return "continue"


def parse_request(state: MarketResearchState) -> dict:
    try:
        parsed = parse_market_request(state["question"], model=create_model())
    except Exception as exc:
        return {
            **_complete(state, "parse_request"),
            "intent": "error",
            "ticker": [],
            "shares": None,
            "errors": [f"Could not parse the request: {exc}"],
        }

    return {
        **_complete(state, "parse_request"),
        "intent": parsed["intent"],
        "ticker": parsed["ticker"],
        "shares": parsed["shares"],
        "needs_quote": parsed["needs_quote"],
        "needs_news": parsed["needs_news"],
    }


def route_after_parse(state: MarketResearchState) -> str:
    """Choose the first data step required by the parsed request."""
    if not state.get("ticker") or state.get("intent") == "error":
        return "partial_answer"

    if state.get("needs_quote"):
        return "fetch_quote"

    if state.get("needs_news"):
        return "fetch_news"

    return "write_answer"


def fetch_quote(state: MarketResearchState) -> dict:
    quotes = {}
    errors = list(state.get("errors", []))

    for ticker in state.get("ticker", []):
        try:
            quotes[ticker] = get_stock_quote(ticker)
        except (ValueError, RuntimeError) as exc:
            errors.append(f"Could not retrieve quote for {ticker}: {exc}")

    update = {
        **_complete(state, "fetch_quote"),
        "quote": quotes,
    }

    if errors:
        update["errors"] = errors

    return update


def route_after_quote(state: MarketResearchState) -> str:
    """Choose the next step after quote retrieval."""
    if not state.get("quote"):
        return "partial_answer"

    if state.get("needs_news"):
        return "fetch_news"

    if state.get("shares") is not None:
        return "calculate_cost"

    return "write_answer"


def fetch_news(state: MarketResearchState) -> dict:
    tickers = state.get("ticker", [])
    query = " ".join(tickers) + " latest market news"

    try:
        news = search_market_news(query)
        return {
            **_complete(state, "fetch_news"),
            "news": news,
            "news_status": "ok" if news else "no_results",
        }
    except (ValueError, RuntimeError) as exc:
        errors = [
            *state.get("errors", []),
            f"Could not retrieve market news: {exc}",
        ]
        return {
            **_complete(state, "fetch_news"),
            "news": [],
            "news_status": "error",
            "errors": errors,
        }


def route_after_news(state: MarketResearchState) -> str:
    """Choose whether a news request also needs a cost calculation."""
    if state.get("shares") is not None:
        return "calculate_cost"

    return "write_answer"


def calculate_cost(state: MarketResearchState) -> dict:
    shares = state.get("shares")
    quotes = state.get("quote", {})

    if shares is None:
        return {
            **_complete(state, "calculate_cost"),
            "calculation": "No share quantity was provided.",
        }

    calculations = []
    errors = list(state.get("errors", []))

    for ticker, quote in quotes.items():
        try:
            total = calculate_share_cost(quote["price"], shares)
            calculations.append(
                f"{ticker}: {shares} shares × ${quote['price']:.2f} = ${total:.2f}"
            )
        except (KeyError, ValueError) as exc:
            errors.append(f"Could not calculate cost for {ticker}: {exc}")

    update = {
        **_complete(state, "calculate_cost"),
        "calculation": "\n".join(calculations) or "No calculations available.",
    }

    if errors:
        update["errors"] = errors

    return update


def write_answer(state: MarketResearchState) -> dict:
    try:
        final_answer = write_market_answer(
            state,
            stream_writer=get_stream_writer(),
        )
    except Exception as exc:
        errors = [
            *state.get("errors", []),
            f"Could not generate the model answer: {exc}",
        ]
        return {
            **_complete(state, "write_answer"),
            "final_answer": write_partial_answer({**state, "errors": errors}),
            "errors": errors,
        }

    return {
        **_complete(state, "write_answer"),
        "final_answer": final_answer,
    }


def write_partial_answer(state: MarketResearchState) -> str:
    """Return a deterministic answer when parsing or model generation fails."""
    lines = [
        "I could not complete the full market-research request.",
    ]

    quotes = state.get("quote", {})
    for ticker, quote in quotes.items():
        currency = quote.get("currency", "")
        lines.append(
            f"Latest available close for {ticker}: "
            f"{quote.get('price', 'unavailable')} {currency}."
        )

    calculation = state.get("calculation")
    if calculation and calculation != "No calculations available.":
        lines.append(calculation)

    errors = state.get("errors", [])
    if errors:
        lines.append("Details: " + " ".join(errors))

    lines.append("This is informational market research, not investment advice.")
    return "\n".join(lines)


def build_graph():
    workflow = StateGraph(MarketResearchState)

    workflow.add_node("scope_check", scope_check)
    workflow.add_node("parse_request", parse_request)
    workflow.add_node("fetch_quote", fetch_quote)
    workflow.add_node("fetch_news", fetch_news)
    workflow.add_node("calculate_cost", calculate_cost)
    workflow.add_node("write_answer", write_answer)
    workflow.add_node("write_partial_answer", lambda state: {
        **_complete(state, "write_partial_answer"),
        "final_answer": write_partial_answer(state),
    })

    workflow.add_edge(START, "scope_check")
    workflow.add_conditional_edges(
        "scope_check",
        route_after_scope_check,
        {
            "continue": "parse_request",
            "refused": END,
        },
    )
    workflow.add_conditional_edges(
        "parse_request",
        route_after_parse,
        {
            "fetch_quote": "fetch_quote",
            "fetch_news": "fetch_news",
            "write_answer": "write_answer",
            "partial_answer": "write_partial_answer",
        },
    )
    workflow.add_conditional_edges(
        "fetch_quote",
        route_after_quote,
        {
            "fetch_news": "fetch_news",
            "calculate_cost": "calculate_cost",
            "write_answer": "write_answer",
            "partial_answer": "write_partial_answer",
        },
    )
    workflow.add_conditional_edges(
        "fetch_news",
        route_after_news,
        {
            "calculate_cost": "calculate_cost",
            "write_answer": "write_answer",
        },
    )
    workflow.add_edge("calculate_cost", "write_answer")
    workflow.add_edge("write_answer", END)
    workflow.add_edge("write_partial_answer", END)

    return workflow.compile()


if __name__ == "__main__":
    graph = build_graph()
    result = graph.invoke({"question": "What is the price of AAPL?"})
    print(result["steps"])
    print(result["final_answer"])
