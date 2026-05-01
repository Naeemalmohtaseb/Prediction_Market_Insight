"""Initialize the local SQLite database."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pmi_agent.storage.db import StorageManager


def main() -> None:
    manager = StorageManager()
    manager.init_db()
    print(f"Database path: {manager.db_path.resolve()}")
    with manager._connect() as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name").fetchall()
    print("Tables:")
    for row in rows:
        print(f"- {row['name']}")


if __name__ == "__main__":
    main()
