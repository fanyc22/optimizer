from __future__ import annotations

from codesign_optimizer.config.settings import ObjectiveWeights
from codesign_optimizer.models.feedback import SimulationFeedback


class ObjectiveFunction:
    """
    Lower score is better.
    """

    def __init__(self, weights: ObjectiveWeights) -> None:
        self._weights = weights

    def score(self, feedback: SimulationFeedback) -> float:
        gm = feedback.global_metrics
        thermal_penalty = 1.0 if gm.thermal_violation else 0.0
        budget_penalty = max(0.0, gm.budget_utilization_percent - 100.0) / 100.0

        normalized_makespan = gm.makespan_us / 10_000.0
        normalized_energy = gm.total_energy_joules / 10_000.0

        return (
            self._weights.makespan * normalized_makespan
            + self._weights.energy * normalized_energy
            + self._weights.thermal_penalty * thermal_penalty
            + self._weights.budget_penalty * budget_penalty
        )
