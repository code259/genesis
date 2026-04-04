from __future__ import annotations

from genesis.config import ProjectConfig


class DomainOracleGenerator:
    def generate(self, project_config: ProjectConfig) -> str:
        hints = project_config.oracle_hints or ["No explicit oracle hints provided."]
        domain_specific = {
            "astrophysics": [
                "if any(abs(value) > 1e9 for value in numeric_values):",
                "    failures.append('CRITICAL_PHYSICS_VIOLATION')",
            ],
            "ml_efficiency": [
                "if numeric_values and max(numeric_values) < 0.0:",
                "    failures.append('negative_metric_detected')",
            ],
        }.get(project_config.domain.lower(), [])
        lines = [
            "def run_all_checks(outputs_dir: str):",
            "    import json",
            "    import re",
            "    from pathlib import Path",
            "",
            "    failures = []",
            "    warnings = []",
            "    outputs = Path(outputs_dir)",
            '    if not outputs.exists():',
            '        failures.append("outputs_dir_missing")',
            "    numeric_values = []",
            "    for candidate in sorted(outputs.rglob('*')):",
            "        if not candidate.is_file():",
            "            continue",
            "        if candidate.suffix == '.json':",
            "            try:",
            "                payload = json.loads(candidate.read_text(encoding='utf-8'))",
            "            except Exception:",
            "                failures.append(f'invalid_json::{candidate.name}')",
            "                continue",
            "            if isinstance(payload, dict):",
            "                for value in payload.values():",
            "                    if isinstance(value, (int, float)):",
            "                        numeric_values.append(float(value))",
            "        elif candidate.suffix in {'.txt', '.md', '.tex'}:",
            "            text = candidate.read_text(encoding='utf-8')",
            r"            numeric_values.extend(float(match) for match in re.findall(r'[-+]?\d*\.?\d+', text))",
            "    if outputs.exists() and not list(outputs.iterdir()):",
            "        failures.append('outputs_dir_empty')",
        ]
        lines.extend(f"    warnings.append({hint!r})" for hint in hints)
        lines.extend(f"    {line}" for line in domain_specific)
        lines.extend(
            [
                "    return {",
                "        'pass_rate': 1.0 if not failures else 0.0,",
                "        'failures': failures,",
                "        'warnings': warnings,",
                "        'is_critical_fail': bool(failures),",
                "    }",
            ]
        )
        return "\n".join(lines) + "\n"

    def validate_oracle(self, oracle_path: str) -> bool:
        return "run_all_checks" in open(oracle_path, encoding="utf-8").read()
