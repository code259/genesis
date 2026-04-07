from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from genesis.agents.runtime import CodingAgentRuntime, ProviderRuntimeError
from genesis.config import ProjectConfig


class DomainOracleGenerator:
    def __init__(self, runtime: CodingAgentRuntime | None = None):
        self.runtime = runtime

    def generate(self, project_config: ProjectConfig) -> str:
        if self.runtime is not None:
            try:
                payload = self.runtime.generate_task(
                    category="genesis-oracle",
                    instruction=f"Generate oracle rules for: {project_config.research_question}",
                    context=project_config.to_dict(),
                    budget={"max_rules": 6},
                )
                generated = self._from_runtime_payload(payload)
                if generated:
                    return generated
            except ProviderRuntimeError:
                pass
        return self._build_default_oracle(project_config)

    def validate_oracle(self, oracle_path: str | Path) -> bool:
        module_path = Path(oracle_path)
        if not module_path.exists():
            return False
        spec = importlib.util.spec_from_file_location("genesis_generated_oracle", module_path)
        if spec is None or spec.loader is None:
            return False
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return callable(getattr(module, "run_all_checks", None))

    def _build_default_oracle(self, project_config: ProjectConfig) -> str:
        hints = [hint.strip() for hint in project_config.oracle_hints if hint.strip()]
        hint_rules = []
        for hint in hints:
            lowered = hint.lower()
            if "metric consistency" in lowered:
                hint_rules.extend(
                    [
                        "    if primary_metric is None:",
                        "        failures.append('missing_primary_metric')",
                    ]
                )
            if "sample" in lowered or "consistency" in lowered:
                hint_rules.extend(
                    [
                        "    if not json_files:",
                        "        warnings.append('no_json_artifacts_detected')",
                    ]
                )

        domain_rules = []
        domain = project_config.domain.lower().strip()
        if domain == "astrophysics":
            domain_rules.extend(
                [
                    "    if any(abs(value) > 1e9 for value in numeric_values):",
                    "        failures.append('CRITICAL_PHYSICS_VIOLATION')",
                    "    if any('ra' in key.lower() and not (0.0 <= float(value) <= 360.0) for payload in parsed_payloads for key, value in payload.items() if isinstance(payload, dict) and isinstance(value, (int, float))):",
                    "        failures.append('invalid_ra_range')",
                    "    if any('dec' in key.lower() and not (-90.0 <= float(value) <= 90.0) for payload in parsed_payloads for key, value in payload.items() if isinstance(payload, dict) and isinstance(value, (int, float))):",
                    "        failures.append('invalid_dec_range')",
                    "    if any('redshift' in key.lower() and not (0.0 <= float(value) <= 15.0) for payload in parsed_payloads for key, value in payload.items() if isinstance(payload, dict) and isinstance(value, (int, float))):",
                    "        failures.append('invalid_redshift_range')",
                ]
            )
        elif domain == "ml_efficiency":
            domain_rules.extend(
                [
                    "    if primary_metric is not None and primary_metric < 0.0:",
                    "        failures.append('negative_metric_detected')",
                    "    if primary_metric is not None and primary_metric > 1.0:",
                    "        warnings.append('metric_above_unit_range')",
                    "    if not any('loss' in key.lower() or 'accuracy' in key.lower() for payload in parsed_payloads for key in payload.keys() if isinstance(payload, dict)):",
                    "        warnings.append('missing_ml_metric_fields')",
                ]
            )
        else:
            domain_rules.extend(
                [
                    "    if primary_metric is None:",
                    "        warnings.append('primary_metric_missing')",
                    "    if not json_files:",
                    "        failures.append('no_structured_outputs')",
                ]
            )

        rules_text = "\n".join(hint_rules + domain_rules)
        hint_text = "\n".join(f"    warnings.append({hint!r})" for hint in hints) or "    warnings.append('No explicit oracle hints provided.')"

        lines = [
            "# Auto-generated Genesis project oracle",
            "from __future__ import annotations",
            "",
            "import json",
            "import re",
            "from pathlib import Path",
            "",
            "",
            "def _flatten_numeric_values(value):",
            "    values = []",
            "    if isinstance(value, (int, float)) and not isinstance(value, bool):",
            "        values.append(float(value))",
            "    elif isinstance(value, dict):",
            "        for nested in value.values():",
            "            values.extend(_flatten_numeric_values(nested))",
            "    elif isinstance(value, list):",
            "        for nested in value:",
            "            values.extend(_flatten_numeric_values(nested))",
            "    return values",
            "",
            "",
            "def run_all_checks(outputs_dir: str):",
            "    outputs = Path(outputs_dir)",
            "    failures = []",
            "    warnings = []",
            "    numeric_values = []",
            "    json_files = []",
            "    parsed_payloads = []",
            "    primary_metric = None",
            "",
            "    if not outputs.exists():",
            "        return {'pass_rate': 0.0, 'failures': ['outputs_dir_missing'], 'warnings': warnings, 'is_critical_fail': True}",
            "",
            "    for candidate in sorted(outputs.rglob('*')):",
            "        if not candidate.is_file():",
            "            continue",
            "        if candidate.suffix == '.json':",
            "            json_files.append(candidate.name)",
            "            try:",
            "                payload = json.loads(candidate.read_text(encoding='utf-8'))",
            "            except Exception:",
            "                failures.append(f'invalid_json::{candidate.name}')",
            "                continue",
            "            parsed_payloads.append(payload)",
            "            numeric_values.extend(_flatten_numeric_values(payload))",
            "            if isinstance(payload, dict) and primary_metric is None and isinstance(payload.get('primary_metric'), (int, float)):",
            "                primary_metric = float(payload['primary_metric'])",
            "        elif candidate.suffix in {'.txt', '.md', '.tex'}:",
            "            text = candidate.read_text(encoding='utf-8')",
            r"            numeric_values.extend(float(match) for match in re.findall(r'[-+]?\d*\.?\d+', text))",
            "",
            "    if outputs.exists() and not list(outputs.iterdir()):",
            "        failures.append('outputs_dir_empty')",
        ]
        if hint_text:
            lines.extend(hint_text.split("\n"))
        if rules_text:
            lines.extend(rules_text.split("\n"))
        lines.extend(
            [
                "",
                "    pass_rate = 1.0 if not failures else max(0.0, 1.0 - (len(failures) / max(1, len(json_files) + 1)))",
                "    return {'pass_rate': round(pass_rate, 4), 'failures': failures, 'warnings': warnings, 'is_critical_fail': bool(failures)}",
            ]
        )
        return "\n".join(lines) + "\n"

    def _from_runtime_payload(self, payload: dict[str, Any]) -> str:
        rules = payload.get("oracle_rules")
        if not isinstance(rules, list) or not rules:
            return ""

        lines = [
            "from __future__ import annotations",
            "",
            "import json",
            "from pathlib import Path",
            "",
            "",
            "def run_all_checks(outputs_dir: str):",
            "    outputs = Path(outputs_dir)",
            "    failures = []",
            "    warnings = []",
            "    combined = {}",
            "    if not outputs.exists():",
            "        return {'pass_rate': 0.0, 'failures': ['outputs_dir_missing'], 'warnings': warnings, 'is_critical_fail': True}",
            "    for candidate in outputs.rglob('*.json'):",
            "        try:",
            "            payload = json.loads(candidate.read_text(encoding='utf-8'))",
            "            if isinstance(payload, dict):",
            "                combined.update(payload)",
            "        except Exception:",
            "            warnings.append(f'ignored_invalid_json::{candidate.name}')",
        ]
        for rule in rules:
            if isinstance(rule, dict):
                expression = str(rule.get("expression", "")).strip()
                failure = str(rule.get("failure", "oracle_rule_failed")).strip()
                warning = str(rule.get("warning", "")).strip()
                if expression:
                    lines.append(f"    if not ({expression}):")
                    lines.append(f"        failures.append({failure!r})")
                if warning:
                    lines.append(f"    warnings.append({warning!r})")
            elif isinstance(rule, str) and rule.strip():
                lines.append(f"    warnings.append({rule.strip()!r})")
        lines.extend(
            [
                "    pass_rate = 1.0 if not failures else 0.0",
                "    return {'pass_rate': pass_rate, 'failures': failures, 'warnings': warnings, 'is_critical_fail': bool(failures)}",
            ]
        )
        return "\n".join(lines) + "\n"
