import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts import new_project as new_project_module


def test_new_project_cleans_up_on_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "projects").mkdir()
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "constraints.md").write_text("constraints")

    monkeypatch.setattr(new_project_module, "decompose", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(new_project_module, "adversarial_review", lambda *_args, **_kwargs: "review")
    monkeypatch.setattr(new_project_module, "parse_project_spec", lambda path: {"Title": "t"})
    monkeypatch.setattr(new_project_module, "validate_project_spec", lambda _spec: [])
    monkeypatch.setattr(
        new_project_module,
        "normalize_project_spec",
        lambda _spec: {"title": "Title", "research_goal": "Goal", "domain_context": "Context"},
    )
    monkeypatch.setattr(new_project_module, "project_spec_to_context", lambda _spec: "Context")

    spec_path = tmp_path / "spec.md"
    spec_path.write_text("# spec")

    with pytest.raises(RuntimeError, match="boom"):
        new_project_module.new_project(spec_path=str(spec_path))

    assert list((tmp_path / "projects").iterdir()) == []
