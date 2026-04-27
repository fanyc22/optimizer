from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from codesign_optimizer.config.settings import OptimizerSettings
from codesign_optimizer.io.jsonc import dump_json
from codesign_optimizer.models.feedback import SimulationFeedback
from codesign_optimizer.models.hardware import HardwareProposal
from codesign_optimizer.models.workload import WorkloadSpec
from codesign_optimizer.optimizer.constraints import ConstraintEvaluator, ConstraintReport
from codesign_optimizer.optimizer.inner_loop import MappingResult, SoftwareMapper
from codesign_optimizer.optimizer.objective import ObjectiveFunction
from codesign_optimizer.optimizer.outer_loop import HardwareTopologyOptimizer
from codesign_optimizer.simulator.interface import SimulatorClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IterationResult:
    iteration: int
    objective_score: float
    constraints: ConstraintReport
    mapping: MappingResult
    feedback: SimulationFeedback
    hardware: HardwareProposal


class CoDesignOrchestrator:
    def __init__(
        self,
        settings: OptimizerSettings,
        simulator: SimulatorClient,
        inner_loop: SoftwareMapper | None = None,
        outer_loop: HardwareTopologyOptimizer | None = None,
    ) -> None:
        self._settings = settings
        self._simulator = simulator
        self._inner = inner_loop or SoftwareMapper()
        self._outer = outer_loop or HardwareTopologyOptimizer()
        self._objective = ObjectiveFunction(settings.objective_weights)
        self._constraints = ConstraintEvaluator(settings.limits)

    def run(
        self,
        initial_hardware: HardwareProposal,
        workload: WorkloadSpec,
    ) -> list[IterationResult]:
        artifacts_dir = self._settings.artifacts_dir
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        current_hw = initial_hardware
        history: list[IterationResult] = []
        for outer_idx in range(self._settings.max_outer_iterations):
            logger.info("Starting iteration %d", outer_idx)
            feedback = self._simulator.run(current_hw, workload)
            mapping = self._inner.optimize_mapping(workload, current_hw, feedback)
            score = self._objective.score(feedback) + 0.01 * mapping.estimated_mapping_cost
            constraints = self._constraints.evaluate(current_hw, feedback)

            result = IterationResult(
                iteration=outer_idx,
                objective_score=score,
                constraints=constraints,
                mapping=mapping,
                feedback=feedback,
                hardware=current_hw,
            )
            history.append(result)

            self._persist_iteration(artifacts_dir, result)
            logger.info(
                "Iteration %d complete | score=%.4f | feasible=%s",
                outer_idx,
                score,
                constraints.feasible,
            )

            if constraints.feasible:
                logger.info("Feasible solution found. Stopping early at iteration %d.", outer_idx)
                break

            current_hw = self._outer.propose_next(current_hw, feedback, mapping)

        self._persist_summary(artifacts_dir, history)
        return history

    def _persist_iteration(self, base_dir: Path, item: IterationResult) -> None:
        iteration_dir = base_dir / f"iter_{item.iteration:03d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)

        dump_json(iteration_dir / "hardware_proposal.json", item.hardware.to_dict())
        dump_json(iteration_dir / "mapping.json", {"assignment": item.mapping.assignment})
        dump_json(
            iteration_dir / "result.json",
            {
                "objective_score": item.objective_score,
                "constraints": {
                    "feasible": item.constraints.feasible,
                    "thermal_ok": item.constraints.thermal_ok,
                    "power_ok": item.constraints.power_ok,
                    "budget_ok": item.constraints.budget_ok,
                    "messages": item.constraints.messages,
                },
                "feedback": item.feedback.model_dump(mode="json"),
            },
        )

    def _persist_summary(self, base_dir: Path, history: list[IterationResult]) -> None:
        if not history:
            return

        best = min(history, key=lambda h: h.objective_score)
        dump_json(
            base_dir / "summary.json",
            {
                "iterations_ran": len(history),
                "best_iteration": best.iteration,
                "best_score": best.objective_score,
                "feasible_found": any(h.constraints.feasible for h in history),
            },
        )
