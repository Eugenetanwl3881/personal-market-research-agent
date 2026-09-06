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
    streamed_answer = False

    print("Steps:")
    for mode, data in graph.stream(
        {"question": question},
        stream_mode=["updates", "custom"],
    ):
        if mode == "custom":
            if data.get("type") == "answer_section":
                print(data["text"], end="", flush=True)
                streamed_answer = True
            elif data.get("type") == "answer_token":
                print(data["text"], end="", flush=True)
                streamed_answer = True
            continue

        final_state.update(next(iter(data.values())))
        step_updates = final_state.get("steps", [])
        if step_updates:
            print(f"- {step_updates[-1]}")

    if not streamed_answer:
        print("\nFinal answer:")
        print(final_state.get("final_answer", "No final answer was produced."))
    else:
        print()

    errors = final_state.get("errors", [])
    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"- {error}")


if __name__ == "__main__":
    main()
