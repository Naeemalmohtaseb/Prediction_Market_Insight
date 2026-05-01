"""Show recent saved analyses."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pmi_agent.storage.db import StorageManager


def main() -> None:
    manager = StorageManager()
    rows = manager.list_recent_analyses(limit=20)
    if not rows:
        print("No saved analyses found.")
        return

    print(f"{'ID':<5} {'CREATED_AT':<25} {'PROB':>8} {'CONF':<8} QUESTION")
    print("-" * 90)
    for row in rows:
        print(
            f"{row['id']:<5} "
            f"{row['created_at'][:25]:<25} "
            f"{_fmt_probability(row['estimated_probability']):>8} "
            f"{(row['confidence_label'] or '-'):<8} "
            f"{row['user_question'][:80]}"
        )


def _fmt_probability(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


if __name__ == "__main__":
    main()
