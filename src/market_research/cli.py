import argparse

from .graph import build_graph


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Research stock prices, share costs, and recent market news."
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="The market-research question to answer.",
    )
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    if not question:
        question = input("Market research question: ").strip()

    if not question:
        parser.error("A market-research question is required.")

    graph = build_graph()
    final_state = {}

    print("Steps:")
    for update in graph.stream(
        {"question": question},
        stream_mode="updates",
    ):
        final_state.update(next(iter(update.values())))
        step_updates = final_state.get("steps", [])
        if step_updates:
            print(f"- {step_updates[-1]}")

    print("\nFinal answer:")
    print(final_state.get("final_answer", "No final answer was produced."))

    errors = final_state.get("errors", [])
    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"- {error}")


if __name__ == "__main__":
    main()
