import json
from pathlib import Path

import pytest

from genesis.config import load_api_config, load_project_config


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


def test_load_api_config_from_dotenv(tmp_path, monkeypatch):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "SEMANTIC_SCHOLAR_API_KEY=test-key\nGROQ_API_KEY=test-groq\nOLLAMA_BASE_URL=http://localhost:11434\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    api_config = load_api_config()
    assert api_config["semantic_scholar_api_key"] == "test-key"
    assert api_config["groq_api_keys"] == ["test-groq"]


def test_load_project_config_rejects_invalid_time_budget(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "research_question": "Does warmup help convergence?",
                "domain": "ml_efficiency",
                "success_criteria": ["improve convergence"],
                "oracle_hints": ["metric consistency"],
                "compute_budget": "local_gpu",
                "time_budget_hours": "4",
                "domain_knowledge_model": "none",
                "output_dir": str(tmp_path / "projects"),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_project_config(spec_path)


def test_dependency_manifest_tracks_required_spec_groups():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert '"pydantic>=2.0"' in pyproject
    assert '"rich>=13.0"' in pyproject
    assert '"loguru>=0.7"' in pyproject
    assert '"pytest-asyncio>=0.23"' in pyproject
    assert '"alembic>=1.13"' in pyproject
    assert '"bibtexparser>=1.4,<2"' in pyproject
    assert '"numpy>=1.26,<2"' in pyproject
    assert '"gpytorch>=1.11; platform_system != \'Darwin\' or platform_machine != \'x86_64\'"' in pyproject
