from __future__ import annotations

from textwrap import dedent

from genesis.config import ProjectConfig


class DomainOracleGenerator:
    def generate(self, project_config: ProjectConfig) -> str:
        hints = project_config.oracle_hints or ["No explicit oracle hints provided."]
        joined_hints = "\\n".join(f"    warnings.append({hint!r})" for hint in hints)
        return dedent(
            f"""
            def run_all_checks(outputs_dir: str):
                failures = []
                warnings = []
            {joined_hints if joined_hints else "    pass"}
                return {{
                    "pass_rate": 1.0 if not failures else 0.0,
                    "failures": failures,
                    "warnings": warnings,
                    "is_critical_fail": bool(failures),
                }}
            """
        ).strip() + "\n"

    def validate_oracle(self, oracle_path: str) -> bool:
        return "run_all_checks" in open(oracle_path, encoding="utf-8").read()
