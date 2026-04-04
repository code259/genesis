import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.project_spec import legacy_project_spec_markdown, normalize_project_spec, parse_project_spec, project_spec_to_context, validate_project_spec


SPEC = """# Project Spec

## Title
TOI Validation

## Research Goal
Determine whether a candidate is a plausible exoplanet.

## Domain Context
Transit photometry and catalog vetting.

## Inputs / Resources
- TOI identifier
- MAST API

## Success Criteria
- Verified claims exist

## Constraints
- Use explicit evidence

## Deliverables
- Figures
- Draft paper

## Verification Expectations
- Verifier sign-off

## Known Unknowns
- Exact API path may need discovery
"""


def test_parse_and_validate_project_spec(tmp_path: Path):
    path = tmp_path / "project_spec.md"
    path.write_text(SPEC)
    spec = parse_project_spec(path)
    assert validate_project_spec(spec) == []
    normalized = normalize_project_spec(spec)
    assert normalized["title"] == "TOI Validation"
    assert "MAST API" in normalized["inputs_resources"]


def test_project_spec_to_context_contains_sections(tmp_path: Path):
    path = tmp_path / "project_spec.md"
    path.write_text(SPEC)
    context = project_spec_to_context(parse_project_spec(path))
    assert "Research Goal" in context
    assert "Inputs / Resources" in context


def test_legacy_project_spec_markdown_parses():
    markdown = legacy_project_spec_markdown("Goal", "Context")
    assert "## Research Goal" in markdown
