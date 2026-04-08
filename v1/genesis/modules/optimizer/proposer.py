from __future__ import annotations

from genesis.models import ExperimentProposal
from genesis.storage.causal_dag import CausalDAG
from genesis.storage.ledger import ExperimentLedger


class ExperimentProposer:
    def propose_next(
        self,
        task_id: str,
        n: int = 3,
        *,
        prior_metric: float = 0.0,
        compute_budget: str = "local_gpu",
        ledger: ExperimentLedger | None = None,
        causal_dag: CausalDAG | None = None,
        domain: str = "",
    ) -> list[ExperimentProposal]:
        history = ledger.get_by_task(task_id) if ledger else []
        prior_metric = max([prior_metric] + [float(item.get("primary_metric", 0.0)) for item in history])
        anomalies = [item for item in history if float(item.get("anomaly_score", 0.0)) >= 0.5]
        best_record = history[0] if history else {}
        base_end = max(prior_metric, float(best_record.get("primary_metric", 0.0)), 0.35)
        scale = 0.7 if "cpu" in compute_budget else 1.0
        target = f"metric:{task_id}"
        high_confidence_edges = causal_dag.top_edges_for_target(target, domain=domain) if causal_dag else []
        positive_features = [
            str(edge.get("source", "")).strip()
            for edge in high_confidence_edges
            if float(edge.get("effect_size", 0.0)) > 0.0 and str(edge.get("source", "")).strip()
        ]
        negative_features = {
            str(edge.get("source", "")).strip()
            for edge in high_confidence_edges
            if float(edge.get("effect_size", 0.0)) < 0.0 and str(edge.get("source", "")).strip()
        }
        warmup_candidates = self._warmup_candidates(history, scale=scale)
        step_candidates = self._step_candidates(history, scale=scale)
        learning_rates = self._learning_rates(history, scale=scale)
        proposals: list[ExperimentProposal] = []
        for index in range(n):
            warmup_ratio = warmup_candidates[index % len(warmup_candidates)]
            steps = step_candidates[index % len(step_candidates)]
            learning_rate = learning_rates[index % len(learning_rates)]
            stability_bonus = 0.03 if anomalies and index == 0 else 0.0
            exploration_bonus = 0.02 * (index + 1)
            expected_metric = min(0.98, base_end + stability_bonus + exploration_bonus)
            start_metric = round(max(0.05, base_end * (0.45 if index == 0 else 0.35)), 4)
            mid_metric = round((start_metric + expected_metric) / 2.0, 4)
            config_parts = [
                f"learning_rate={learning_rate}",
                f"warmup_ratio={warmup_ratio}",
                f"steps={steps}",
            ]
            config_parts = [part for part in config_parts if part not in negative_features]
            if anomalies and index == 0:
                config_parts.append("stabilize_after_anomaly=true")
            if positive_features and index == 0:
                for feature in positive_features[:2]:
                    if feature not in config_parts:
                        config_parts.append(feature)
            proposals.append(
                ExperimentProposal(
                    description=self._describe_variant(
                        task_id,
                        index=index,
                        anomalies=bool(anomalies),
                        has_causal_guidance=bool(positive_features),
                    ),
                    code_diff="; ".join(config_parts),
                    expected_metric=round(expected_metric, 4),
                    expected_trajectory=[
                        start_metric,
                        mid_metric,
                        round(expected_metric, 4),
                    ],
                    compute_budget=compute_budget,
                    model_parameter_count=int(1_000_000 * scale * (index + 1)),
                )
            )
        return proposals

    def _warmup_candidates(self, history: list[dict[str, object]], *, scale: float) -> list[float]:
        if not history:
            return [0.2, 0.35, 0.5]
        return [round(value, 2) for value in (0.15 * scale, 0.25 * scale, 0.4 * scale)]

    def _step_candidates(self, history: list[dict[str, object]], *, scale: float) -> list[int]:
        base = 6 if scale < 1.0 else 8
        if history:
            base += min(4, len(history))
        return [base, base + 2, base + 4]

    def _learning_rates(self, history: list[dict[str, object]], *, scale: float) -> list[float]:
        if history:
            best_metric = max(float(item.get("primary_metric", 0.0)) for item in history)
            base = 0.08 if best_metric > 0.7 else 0.12
        else:
            base = 0.12
        return [round(base * scale, 3), round(base * 0.85 * scale, 3), round(base * 1.1 * scale, 3)]

    def _describe_variant(self, task_id: str, *, index: int, anomalies: bool, has_causal_guidance: bool = False) -> str:
        if anomalies and index == 0:
            return f"Stabilization experiment for anomalous task {task_id}"
        if has_causal_guidance and index == 0:
            return f"Causally guided experiment variant {index + 1} for task {task_id}"
        return f"Optimizer experiment variant {index + 1} for task {task_id}"
