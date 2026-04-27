from __future__ import annotations

from dataclasses import dataclass

from codesign_optimizer.config.settings import ConstraintLimits
from codesign_optimizer.models.feedback import SimulationFeedback
from codesign_optimizer.models.hardware import HardwareProposal


@dataclass(frozen=True)
class ConstraintReport:
    feasible: bool
    thermal_ok: bool
    power_ok: bool
    budget_ok: bool
    messages: list[str]


class ConstraintEvaluator:
    def __init__(self, limits: ConstraintLimits) -> None:
        self._limits = limits

    def evaluate(
        self,
        proposal: HardwareProposal,
        feedback: SimulationFeedback,
    ) -> ConstraintReport:
        msgs: list[str] = []

        thermal_ok = not feedback.global_metrics.thermal_violation
        if not thermal_ok:
            msgs.append("Thermal violation reported by simulator.")

        power_ok = feedback.global_metrics.peak_power_watts <= self._limits.max_peak_power_watts
        if not power_ok:
            msgs.append(
                "Peak power exceeds limit: "
                f"{feedback.global_metrics.peak_power_watts:.1f}W > "
                f"{self._limits.max_peak_power_watts:.1f}W."
            )

        budget_ok = (
            feedback.global_metrics.budget_utilization_percent
            <= self._limits.max_budget_utilization_percent
        )
        if not budget_ok:
            msgs.append(
                "Budget exceeds limit: "
                f"{feedback.global_metrics.budget_utilization_percent:.1f}% > "
                f"{self._limits.max_budget_utilization_percent:.1f}%."
            )

        if proposal.total_estimated_cost() <= 0:
            msgs.append("Estimated hardware cost is non-positive; invalid proposal detected.")
            budget_ok = False

        feasible = thermal_ok and power_ok and budget_ok
        return ConstraintReport(
            feasible=feasible,
            thermal_ok=thermal_ok,
            power_ok=power_ok,
            budget_ok=budget_ok,
            messages=msgs,
        )
