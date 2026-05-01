"""Manual smoke test for RSS context retrieval."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pmi_agent.context.news_context import NewsContextService
from pmi_agent.interpretation.query_interpreter import QueryInterpreter


QUESTIONS = [
    "Will the Fed cut rates by September?",
    "Will OpenAI IPO before 2027?",
    "Will gas prices rise this summer?",
]


def main() -> None:
    interpreter = QueryInterpreter()
    service = NewsContextService()
    for question_text in QUESTIONS:
        interpreted = interpreter.interpret(question_text)
        items = service.fetch_context(interpreted, max_items=5)
        summary = service.summarize_context(interpreted, items)

        print(f"\nQuestion: {question_text}")
        print(f"Interpreted event: {interpreted.target_event}")
        print(f"Context summary: {summary}")
        if service.last_warnings:
            print("Warnings:")
            for warning in service.last_warnings[:3]:
                print(f"  - {warning}")
        print("Top context items:")
        if not items:
            print("  none")
        for item in items[:5]:
            print(
                f"  rel={item.relevance_score:.3f} "
                f"source={item.source or '-'} "
                f"title={item.title[:100]}"
            )


if __name__ == "__main__":
    main()
