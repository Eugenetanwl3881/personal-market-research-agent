import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from .state import MarketResearchState


load_dotenv()


def create_model() -> ChatOpenAI:
    """Create a chat model using the OpenCode Go OpenAI-compatible endpoint."""
    api_key = os.getenv("OPENCODE_API_KEY")
    base_url = os.getenv("OPENCODE_BASE_URL", "https://opencode.ai/zen/go/v1")
    model_name = os.getenv("OPENCODE_MODEL", "glm-5.3")

    if not api_key:
        raise ValueError("OPENCODE_API_KEY is not configured.")

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )


def write_market_answer(state: MarketResearchState) -> str:
    """Write a sourced informational answer from collected graph state."""
    prompt = f"""
You are a market-research assistant.

Answer the user's question using only the collected data below.

User question:
{state["question"]}

Quotes:
{state.get("quote", {})}

News:
{state.get("news", [])}

Calculation:
{state.get("calculation", "")}

Errors:
{state.get("errors", [])}

Rules:
- Be factual and concise.
- Clearly mention unavailable data.
- Include relevant news URLs as sources.
- Do not place or recommend trades.
- Include: "This is informational market research, not investment advice."
"""

    response = create_model().invoke(prompt)
    return response.content
