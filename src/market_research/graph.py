from langgraph.graph import END, START, StateGraph

from .state import MarketResearchState


def _complete(state: MarketResearchState, step: str) -> dict:
    """Record a completed step while the graph is still a learning skeleton."""
    return {"steps": [*state.get("steps", []), f"{step} completed"]}


def scope_check(state: MarketResearchState) -> dict:
    return _complete(state, "scope_check")


def parse_request(state: MarketResearchState) -> dict:
    return _complete(state, "parse_request")


def fetch_quote(state: MarketResearchState) -> dict:
    return _complete(state, "fetch_quote")


def fetch_news(state: MarketResearchState) -> dict:
    return _complete(state, "fetch_news")


def calculate_cost(state: MarketResearchState) -> dict:
    return _complete(state, "calculate_cost")


def write_answer(state: MarketResearchState) -> dict:
    return {
        **_complete(state, "write_answer"),
        "final_answer": "Graph skeleton completed; data and model steps are not connected yet.",
    }


def build_graph():
    workflow = StateGraph(MarketResearchState)

    workflow.add_node("scope_check", scope_check)
    workflow.add_node("parse_request", parse_request)
    workflow.add_node("fetch_quote", fetch_quote)
    workflow.add_node("fetch_news", fetch_news)
    workflow.add_node("calculate_cost", calculate_cost)
    workflow.add_node("write_answer", write_answer)

    workflow.add_edge(START, "scope_check")
    workflow.add_edge("scope_check", "parse_request")
    workflow.add_edge("parse_request", "fetch_quote")
    workflow.add_edge("fetch_quote", "fetch_news")
    workflow.add_edge("fetch_news", "calculate_cost")
    workflow.add_edge("calculate_cost", "write_answer")
    workflow.add_edge("write_answer", END)

    return workflow.compile()


if __name__ == "__main__":
    graph = build_graph()
    result = graph.invoke({"question": "What is the price of AAPL?"})
    print(result["steps"])
    print(result["final_answer"])
