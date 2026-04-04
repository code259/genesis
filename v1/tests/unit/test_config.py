import json

from genesis.config import load_project_config


def test_load_project_config(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "research_question": "Does warmup help convergence?",
                "domain": "ml_efficiency",
                "success_criteria": ["improve convergence"],
                "oracle_hints": ["metric consistency"],
                "compute_budget": "local_gpu",
                "time_budget_hours": 4,
                "domain_knowledge_model": "none",
                "output_dir": str(tmp_path / "projects"),
            }
        ),
        encoding="utf-8",
    )
    config = load_project_config(spec_path)
    assert config.domain == "ml_efficiency"
    assert config.success_criteria == ["improve convergence"]
