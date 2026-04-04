from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genesis.taste.gp_model import TasteGP


def main() -> None:
    root = Path("taste_db")
    root.mkdir(exist_ok=True)
    TasteGP().save(root / "taste_model.json")
    print("initialized taste model")


if __name__ == "__main__":
    main()
