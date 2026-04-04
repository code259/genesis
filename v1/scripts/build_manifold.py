from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genesis.storage.manifold import ManifoldIndex


def main() -> None:
    root = Path("manifold_index")
    root.mkdir(exist_ok=True)
    manifold = ManifoldIndex(root)
    paper = {
        "paper_id": "seed-paper",
        "title": "Seed paper",
        "abstract": "A seed abstract for manifold initialization.",
        "latent_z": [0.1, 0.2, 0.3],
        "density_score": 0.9,
    }
    manifold.add_paper(paper)
    print(json.dumps({"status": "ok", "papers_indexed": 1}))


if __name__ == "__main__":
    main()
