import json
import subprocess
from pathlib import Path

from genesis.modules.ideation.greedy import GreedyAdjacencySearch
from genesis.modules.ideation.low_density import LowDensityExplorer
from genesis.modules.ideation.orchestrator import IdeationOrchestrator
from genesis.modules.ideation.pollination import PollinationSearch
from genesis.storage.manifold import ManifoldIndex


def test_custom_benchmark_suite_builds_manifold_and_scores_ideas(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    seed_path = workspace / "papers.json"
    seed_path.write_text(
        json.dumps(
            [
                {
                    "paper_id": "paper-1",
                    "title": "Warmup schedules improve convergence",
                    "abstract": "Warmup stabilizes optimization in compact networks.",
                    "year": 2024,
                    "authors": [{"name": "A. Researcher"}],
                    "citation_count": 4,
                    "domain": "ml_efficiency",
                    "citations": [{"paper_id": "paper-2"}],
                },
                {
                    "paper_id": "paper-2",
                    "title": "Sparse training regimes reveal novel optimizer behavior",
                    "abstract": "Sparse optimization regimes reveal surprising trajectories.",
                    "year": 2025,
                    "authors": [{"name": "B. Researcher"}],
                    "citation_count": 6,
                    "domain": "ml_efficiency",
                    "citations": [{"paper_id": "paper-1"}],
                },
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(workspace)
    result = subprocess.run(
        [
            "python3",
            "/Users/nikhilmaturi/Files/Projects/genesis-worktrees/optimizer-ideation-taste/v1/scripts/build_manifold.py",
            "--domain",
            "ml_efficiency",
            "--input",
            str(seed_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    manifest = json.loads(result.stdout)
    assert manifest["papers_indexed"] == 2

    manifold = ManifoldIndex(workspace / "manifold_index")
    orchestrator = IdeationOrchestrator(
        greedy=GreedyAdjacencySearch(manifold),
        pollination=PollinationSearch(manifold),
        low_density=LowDensityExplorer(manifold),
    )
    ideas = orchestrator.run("optimizer warmup surprise", n_failed_iterations=5)
    assert ideas
    assert ideas[0].score.composite_score >= 0.3


def test_build_citation_adjacency_accepts_string_citation_ids():
    from genesis.manifold_utils import build_citation_adjacency

    adjacency, index = build_citation_adjacency(
        [
            {"paper_id": "paper-1", "citations": ["paper-2"]},
            {"paper_id": "paper-2", "citations": []},
        ]
    )
    assert index == {"paper-1": 0, "paper-2": 1}
    assert adjacency[0, 1] == 1.0
    assert adjacency[1, 0] == 1.0
