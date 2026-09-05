"""
Entry point — run the multi-tool agent interactively from the command line.

Usage:
    python main.py
"""

from agent import ask


def main():
    print("Bangladesh Multi-Tool AI Agent — type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        answer = ask(question)
        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    main()
