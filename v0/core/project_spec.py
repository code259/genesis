from __future__ import annotations

from pathlib import Path
import re


REQUIRED_SECTIONS = [
    "Title",
    "Research Goal",
    "Domain Context",
    "Inputs / Resources",
    "Success Criteria",
    "Constraints",
    "Deliverables",
    "Verification Expectations",
    "Known Unknowns",
]


def parse_project_spec(path: str | Path) -> dict:
    spec_path = Path(path)
    content = spec_path.read_text()
    sections = _split_sections(content)
    return {
        "source_path": str(spec_path),
        "raw_markdown": content,
        "title": sections.get("Title", "").strip(),
        "research_goal": sections.get("Research Goal", "").strip(),
        "domain_context": sections.get("Domain Context", "").strip(),
        "inputs_resources": _parse_list_section(sections.get("Inputs / Resources", "")),
        "success_criteria": _parse_list_section(sections.get("Success Criteria", "")),
        "constraints": _parse_list_section(sections.get("Constraints", "")),
        "deliverables": _parse_list_section(sections.get("Deliverables", "")),
        "verification_expectations": _parse_list_section(sections.get("Verification Expectations", "")),
        "known_unknowns": _parse_list_section(sections.get("Known Unknowns", "")),
    }


def validate_project_spec(spec: dict) -> list[str]:
    errors = []
    for field in [
        "title",
        "research_goal",
        "domain_context",
        "inputs_resources",
        "success_criteria",
        "constraints",
        "deliverables",
        "verification_expectations",
        "known_unknowns",
    ]:
        value = spec.get(field)
        if isinstance(value, list):
            if not value:
                errors.append(f"{field}: section is required")
        elif not value:
            errors.append(f"{field}: section is required")
    return errors


def normalize_project_spec(spec: dict) -> dict:
    return {
        "title": spec["title"].strip(),
        "research_goal": spec["research_goal"].strip(),
        "domain_context": spec["domain_context"].strip(),
        "inputs_resources": [item.strip() for item in spec.get("inputs_resources", []) if item.strip()],
        "success_criteria": [item.strip() for item in spec.get("success_criteria", []) if item.strip()],
        "constraints": [item.strip() for item in spec.get("constraints", []) if item.strip()],
        "deliverables": [item.strip() for item in spec.get("deliverables", []) if item.strip()],
        "verification_expectations": [item.strip() for item in spec.get("verification_expectations", []) if item.strip()],
        "known_unknowns": [item.strip() for item in spec.get("known_unknowns", []) if item.strip()],
    }


def legacy_project_spec_markdown(
    research_goal: str,
    domain_context: str,
    domain: str = "general",
) -> str:
    return f"""# Project Spec

## Title
{research_goal.strip()[:120]}

## Research Goal
{research_goal.strip()}

## Domain Context
Domain: {domain}

{domain_context.strip()}

## Inputs / Resources
- No explicit external resources provided yet.

## Success Criteria
- Produce a verified staged research execution plan.
- Generate artifacts and a LaTeX-ready paper package.

## Constraints
- Follow project constraints from `constraints.md`.
- Preserve explicit verification and human checkpoints.

## Deliverables
- Verified task outputs
- Figures and data artifacts when applicable
- LaTeX-ready paper package

## Verification Expectations
- Every accepted task has verifier sign-off.
- Oracle checks run when configured and applicable.

## Known Unknowns
- Domain-specific workflow details may need to be discovered during execution.
"""


def project_spec_to_context(spec: dict) -> str:
    normalized = normalize_project_spec(spec)
    return "\n\n".join(
        [
            f"Title:\n{normalized['title']}",
            f"Research Goal:\n{normalized['research_goal']}",
            f"Domain Context:\n{normalized['domain_context']}",
            _section_text("Inputs / Resources", normalized["inputs_resources"]),
            _section_text("Success Criteria", normalized["success_criteria"]),
            _section_text("Constraints", normalized["constraints"]),
            _section_text("Deliverables", normalized["deliverables"]),
            _section_text("Verification Expectations", normalized["verification_expectations"]),
            _section_text("Known Unknowns", normalized["known_unknowns"]),
        ]
    )


def _split_sections(content: str) -> dict[str, str]:
    matches = list(re.finditer(r"^##\s+(.+?)\s*$", content, re.MULTILINE))
    sections = {}
    for index, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        sections[title] = content[start:end].strip()
    return sections


def _parse_list_section(body: str) -> list[str]:
    lines = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            lines.append(stripped[2:].strip())
        elif stripped:
            lines.append(stripped)
    return lines


def _section_text(name: str, items: list[str]) -> str:
    return f"{name}:\n" + "\n".join(f"- {item}" for item in items)
