from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import math
import random
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from codesign_optimizer.io.jsonc import dump_json, load_jsonc
from codesign_optimizer.models.hardware import ComponentLibrary, NodeTypeSpec
from codesign_optimizer.optimizer.chromosome import (
    Chromosome,
    RackGene,
    chromosome_from_template,
    infer_type_pools,
    rack_gene_from_archetype,
    role_of_type,
)
from codesign_optimizer.optimizer.exporter import ExportedHardware, HardwareTopologyExporter
from codesign_optimizer.optimizer.feedback_parser import ParsedPipelineFeedback
from codesign_optimizer.optimizer.link_scope import ordered_link_types_for_scope
from codesign_optimizer.optimizer.pipeline_client import PipelineClient
from codesign_optimizer.optimizer.repair import CandidateRepairer, RepairReport
from codesign_optimizer.optimizer.search_space import SearchObjectiveWeights, SearchSpace
from codesign_optimizer.optimizer.tcro import softmax
from codesign_optimizer.optimizer.workload_suite import MultiWorkloadFeedback


logger = logging.getLogger(__name__)


ActionType = Literal[
    "add_node_to_slot",
    "remove_node_from_slot",
    "replace_node_type",
    "upgrade_node",
    "downgrade_node",
    "change_intra_rack_topology",
    "upgrade_intra_rack_link",
    "downgrade_intra_rack_link",
    "change_inter_rack_topology",
    "upgrade_inter_rack_link",
    "downgrade_inter_rack_link",
    "activate_optional_rack",
    "deactivate_optional_rack",
    "add_rack_from_template",
    "remove_rack",
]


class TGRLConfig(BaseModel):
    mode: Literal["v0", "v1"] = "v0"
    temperature: float = Field(default=1.0, gt=0)
    heuristic_weight: float = Field(default=1.0, ge=0)
    learning_rate: float = Field(default=0.05, ge=0)
    kl_weight: float = Field(default=0.05, ge=0)
    greedy: bool = False
    duplicate_penalty: float = Field(default=0.05, ge=0)
    freeze_topology: bool = False
    allow_empty_slots: bool = True


TOPOLOGY_CHANGING_ACTION_TYPES: set[ActionType] = {
    "add_node_to_slot",
    "remove_node_from_slot",
    "change_intra_rack_topology",
    "upgrade_intra_rack_link",
    "downgrade_intra_rack_link",
    "change_inter_rack_topology",
    "upgrade_inter_rack_link",
    "downgrade_inter_rack_link",
    "activate_optional_rack",
    "deactivate_optional_rack",
    "add_rack_from_template",
    "remove_rack",
}


@dataclass(frozen=True)
class GraphEditAction:
    action_type: ActionType
    rack_id: str = ""
    resource: str = ""
    target: str = ""
    delta: int = 0

    @property
    def key(self) -> str:
        return "|".join(
            [
                self.action_type,
                self.rack_id,
                self.resource,
                self.target,
                str(self.delta),
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "rack_id": self.rack_id,
            "resource": self.resource,
            "target": self.target,
            "delta": self.delta,
            "key": self.key,
        }


@dataclass
class MaskedAction:
    action: GraphEditAction
    chromosome: Chromosome
    repair: RepairReport
    heuristic_score: float
    features: dict[str, float]
    prior_prob: float = 0.0
    policy_prob: float = 0.0
    logprob: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "heuristic_score": self.heuristic_score,
            "prior_prob": self.prior_prob,
            "policy_prob": self.policy_prob,
            "logprob": self.logprob,
            "features": self.features,
            "chromosome": self.chromosome.to_dict(),
            "repair": {
                "feasible": self.repair.feasible,
                "messages": self.repair.messages,
                "estimated_cost": self.repair.estimated_cost,
                "estimated_power_watts": self.repair.estimated_power_watts,
                "penalty": self.repair.penalty,
            },
        }


@dataclass
class TGRLEvaluation:
    episode: int
    step: int
    candidate_index: int
    masked_action: MaskedAction
    feasible: bool
    messages: list[str]
    objectives: tuple[float, float, float, float, float, float]
    weighted_score: float
    repair: RepairReport
    exported: ExportedHardware | None = None
    feedback: ParsedPipelineFeedback | None = None
    suite_feedback: MultiWorkloadFeedback | None = None
    cache_hit: bool = False

    @property
    def chromosome(self) -> Chromosome:
        return self.repair.chromosome

    @property
    def action(self) -> GraphEditAction:
        return self.masked_action.action

    def to_summary(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "step": self.step,
            "candidate_index": self.candidate_index,
            "feasible": self.feasible,
            "messages": self.messages,
            "objectives": {
                "makespan_us": self.objectives[0],
                "estimated_cost": self.objectives[1],
                "estimated_power_watts": self.objectives[2],
                "max_link_utilization": self.objectives[3],
                "max_queue_delay_ns": self.objectives[4],
                "remote_memory_contention_ns": self.objectives[5],
            },
            "weighted_score": self.weighted_score,
            "cache_hit": self.cache_hit,
            "action": self.action.to_dict(),
            "prior_prob": self.masked_action.prior_prob,
            "policy_prob": self.masked_action.policy_prob,
            "logprob": self.masked_action.logprob,
            "chromosome": self.chromosome.to_dict(),
            "suite": self.suite_feedback.to_dict() if self.suite_feedback is not None else None,
        }


@dataclass
class TrajectoryItem:
    episode: int
    step: int
    action: GraphEditAction
    features: dict[str, float]
    reward: float
    prior_prob: float
    policy_prob: float
    old_logprob: float
    candidate_signature: str
    weighted_score: float
    feasible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "step": self.step,
            "action": self.action.to_dict(),
            "features": self.features,
            "reward": self.reward,
            "prior_prob": self.prior_prob,
            "policy_prob": self.policy_prob,
            "old_logprob": self.old_logprob,
            "candidate_signature": self.candidate_signature,
            "weighted_score": self.weighted_score,
            "feasible": self.feasible,
        }


@dataclass(frozen=True)
class TGRLSearchResult:
    history: list[TGRLEvaluation]
    trajectory: list[TrajectoryItem]
    best: TGRLEvaluation
    policy_state: dict[str, float]


@dataclass
class TelemetryContext:
    compute_utilization: float = 0.0
    network_utilization: float = 0.0
    queue_pressure: float = 0.0
    remote_memory_pressure: float = 0.0
    top_domain: str = ""
    cost_pressure: float = 0.0
    power_pressure: float = 0.0

    @property
    def constraint_pressure(self) -> float:
        return min(2.0, self.cost_pressure + self.power_pressure)


class LinearPolicy:
    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = dict(weights or {})

    def score(self, features: dict[str, float]) -> float:
        return sum(self.weights.get(name, 0.0) * value for name, value in features.items())

    def update(
        self,
        trajectory: list[TrajectoryItem],
        *,
        learning_rate: float,
        kl_weight: float,
    ) -> None:
        if not trajectory or learning_rate <= 0:
            return
        mean_reward = sum(item.reward for item in trajectory) / len(trajectory)
        for item in trajectory:
            advantage = item.reward - mean_reward
            prior_pull = kl_weight * (item.prior_prob - item.policy_prob)
            scale = learning_rate * (advantage + prior_pull)
            for name, value in item.features.items():
                self.weights[name] = self.weights.get(name, 0.0) + scale * value

    def to_dict(self) -> dict[str, Any]:
        return {"weights": dict(sorted(self.weights.items()))}

    @classmethod
    def from_path(cls, path: Path) -> "LinearPolicy":
        if not path.exists():
            return cls()
        payload = load_jsonc(path)
        return cls(weights=payload.get("weights", {}))


class TGRLSearchRunner:
    def __init__(
        self,
        *,
        component_library: ComponentLibrary,
        search_space: SearchSpace,
        pipeline_client: PipelineClient,
        workload_path: Path,
        out_dir: Path,
        episodes: int,
        steps_per_episode: int,
        concurrency: int,
        config: TGRLConfig | None = None,
    ) -> None:
        self._library = component_library
        self._space = search_space
        self._pipeline = pipeline_client
        self._workload_path = workload_path
        self._out_dir = out_dir
        self._episodes = episodes
        self._steps_per_episode = steps_per_episode
        self._concurrency = max(1, concurrency)
        self._config = config or TGRLConfig()
        self._rng = random.Random(search_space.seed)
        self._exporter = HardwareTopologyExporter(component_library)
        self._repairer = CandidateRepairer(component_library, search_space)
        self._policy = LinearPolicy.from_path(out_dir / "policy_state.json")
        self._cache: dict[str, TGRLEvaluation] = {}
        self._cache_lock = threading.Lock()
        self._seen_signatures: set[str] = set()

    def run(self) -> TGRLSearchResult:
        self._out_dir.mkdir(parents=True, exist_ok=True)
        history: list[TGRLEvaluation] = []
        trajectory: list[TrajectoryItem] = []
        best: TGRLEvaluation | None = None
        logger.info(
            "Starting TG-RL search: mode=%s episodes=%d steps_per_episode=%d concurrency=%d out=%s",
            self._config.mode,
            self._episodes,
            self._steps_per_episode,
            self._concurrency,
            self._out_dir,
        )

        for episode in range(self._episodes):
            logger.info("TG-RL episode %d/%d: evaluating initial proposal", episode + 1, self._episodes)
            current = self._initial_chromosome()
            current_eval = self._evaluate_initial(current, episode)
            previous_score = current_eval.weighted_score
            last_feedback = current_eval.feedback
            history.append(current_eval)
            if best is None or current_eval.weighted_score < best.weighted_score:
                best = current_eval
            logger.info(
                "TG-RL episode %d/%d initial: score=%.4f feasible=%s",
                episode + 1,
                self._episodes,
                current_eval.weighted_score,
                current_eval.feasible,
            )

            episode_trajectory: list[TrajectoryItem] = []
            for step in range(self._steps_per_episode):
                current_repair = self._repairer.repair_and_validate(current)
                context = telemetry_context(last_feedback, current_repair, self._space)
                masked_actions = build_masked_actions(
                    current,
                    component_library=self._library,
                    search_space=self._space,
                    repairer=self._repairer,
                    exporter=self._exporter,
                    feedback=last_feedback,
                    current_repair=current_repair,
                    policy=self._policy if self._config.mode == "v1" else None,
                    config=self._config,
                )
                if not masked_actions:
                    logger.info(
                        "TG-RL episode %d/%d step %d/%d: no feasible actions, stopping episode",
                        episode + 1,
                        self._episodes,
                        step + 1,
                        self._steps_per_episode,
                    )
                    break

                sampled = sample_masked_actions(
                    masked_actions,
                    count=min(self._concurrency, len(masked_actions)),
                    rng=self._rng,
                    greedy=self._config.greedy,
                )
                logger.info(
                    "TG-RL episode %d/%d step %d/%d: actions=%d sampled=%d context(compute=%.3f network=%.3f remote=%.3f)",
                    episode + 1,
                    self._episodes,
                    step + 1,
                    self._steps_per_episode,
                    len(masked_actions),
                    len(sampled),
                    context.compute_utilization,
                    context.network_utilization,
                    context.remote_memory_pressure,
                )
                evaluations = self._evaluate_actions(sampled, episode, step)
                history.extend(evaluations)
                step_best = min(evaluations, key=lambda item: item.weighted_score)
                if best is None or step_best.weighted_score < best.weighted_score:
                    best = step_best
                reward_base = best.weighted_score if best is not None else previous_score
                for evaluation in evaluations:
                    reward = previous_score - evaluation.weighted_score
                    if not evaluation.feasible:
                        reward -= 1.0
                    signature = evaluation.chromosome.signature()
                    if signature in self._seen_signatures:
                        reward -= self._config.duplicate_penalty
                    self._seen_signatures.add(signature)
                    item = TrajectoryItem(
                        episode=episode,
                        step=step,
                        action=evaluation.action,
                        features=evaluation.masked_action.features,
                        reward=reward,
                        prior_prob=evaluation.masked_action.prior_prob,
                        policy_prob=evaluation.masked_action.policy_prob,
                        old_logprob=evaluation.masked_action.logprob,
                        candidate_signature=signature,
                        weighted_score=evaluation.weighted_score,
                        feasible=evaluation.feasible,
                    )
                    trajectory.append(item)
                    episode_trajectory.append(item)
                    self._append_trajectory(item)

                current = step_best.chromosome.model_copy(deep=True)
                previous_score = min(previous_score, step_best.weighted_score, reward_base)
                last_feedback = step_best.feedback
                self._persist_step(episode, step, step_best, evaluations)
                logger.info(
                    "TG-RL episode %d/%d step %d/%d complete: best_action=%s best_score=%.4f feasible=%s cache_hits=%d",
                    episode + 1,
                    self._episodes,
                    step + 1,
                    self._steps_per_episode,
                    step_best.action.key,
                    step_best.weighted_score,
                    step_best.feasible,
                    sum(1 for item in evaluations if item.cache_hit),
                )

            if self._config.mode == "v1":
                self._policy.update(
                    episode_trajectory,
                    learning_rate=self._config.learning_rate,
                    kl_weight=self._config.kl_weight,
                )
                dump_json(self._out_dir / "policy_state.json", self._policy.to_dict())
                logger.info(
                    "TG-RL episode %d/%d: updated linear policy with %d trajectory items",
                    episode + 1,
                    self._episodes,
                    len(episode_trajectory),
                )

        if best is None:
            raise RuntimeError("TG-RL did not evaluate any candidate")
        self._persist_final(history, trajectory, best)
        logger.info(
            "TG-RL search finished: evaluations=%d trajectory_items=%d best_score=%.4f feasible=%s",
            len(history),
            len(trajectory),
            best.weighted_score,
            best.feasible,
        )
        return TGRLSearchResult(
            history=history,
            trajectory=trajectory,
            best=best,
            policy_state=self._policy.weights,
        )

    def _initial_chromosome(self) -> Chromosome:
        if not self._space.templates:
            raise ValueError("search space must contain at least one template")
        report = self._repairer.repair_and_validate(chromosome_from_template(self._space.templates[0]))
        return report.chromosome

    def _evaluate_initial(self, chromosome: Chromosome, episode: int) -> TGRLEvaluation:
        action = GraphEditAction(action_type="change_inter_rack_topology", target="initial")
        repair = self._repairer.repair_and_validate(chromosome)
        masked_action = MaskedAction(
            action=action,
            chromosome=repair.chromosome,
            repair=repair,
            heuristic_score=0.0,
            features={"bias": 1.0, "type:initial": 1.0},
            prior_prob=1.0,
            policy_prob=1.0,
            logprob=0.0,
        )
        return self._evaluate_masked_action(
            masked_action,
            episode=episode,
            step=-1,
            candidate_index=0,
            candidate_dir=self._out_dir / f"episode_{episode:03d}" / "initial",
        )

    def _evaluate_actions(
        self,
        actions: list[MaskedAction],
        episode: int,
        step: int,
    ) -> list[TGRLEvaluation]:
        if self._concurrency <= 1 or len(actions) <= 1:
            return [
                self._evaluate_masked_action(
                    action,
                    episode=episode,
                    step=step,
                    candidate_index=index,
                    candidate_dir=self._out_dir
                    / f"episode_{episode:03d}"
                    / f"step_{step:03d}"
                    / f"candidate_{index:03d}",
                )
                for index, action in enumerate(actions)
            ]

        result: list[TGRLEvaluation | None] = [None] * len(actions)
        with ThreadPoolExecutor(max_workers=min(self._concurrency, len(actions))) as executor:
            futures = {
                executor.submit(
                    self._evaluate_masked_action,
                    action,
                    episode,
                    step,
                    index,
                    self._out_dir
                    / f"episode_{episode:03d}"
                    / f"step_{step:03d}"
                    / f"candidate_{index:03d}",
                ): index
                for index, action in enumerate(actions)
            }
            for future in as_completed(futures):
                index = futures[future]
                result[index] = future.result()
        return [item for item in result if item is not None]

    def _evaluate_masked_action(
        self,
        masked_action: MaskedAction,
        episode: int,
        step: int,
        candidate_index: int,
        candidate_dir: Path,
    ) -> TGRLEvaluation:
        candidate_dir.mkdir(parents=True, exist_ok=True)
        dump_json(candidate_dir / "action.json", masked_action.action.to_dict())
        dump_json(candidate_dir / "candidate.json", masked_action.to_dict())

        repair = masked_action.repair
        chromosome = repair.chromosome
        signature = chromosome.signature()
        exported: ExportedHardware | None = None
        try:
            exported = self._exporter.export(chromosome, iteration=max(0, episode))
            dump_json(candidate_dir / "proposal.json", exported.proposal.to_dict())
            dump_json(candidate_dir / "hardware_topology.json", exported.hardware_topology)
        except Exception as exc:
            repair = RepairReport(
                chromosome=chromosome,
                feasible=False,
                messages=repair.messages + [f"export failed: {exc}"],
                estimated_cost=repair.estimated_cost,
                estimated_power_watts=repair.estimated_power_watts,
                penalty=repair.penalty + 1_000_000.0,
            )
        logger.info(
            "TG-RL candidate episode=%03d step=%03d idx=%03d action=%s prepared: ranks=%s racks=%d cost=%.2f power=%.2f feasible=%s",
            episode,
            step,
            candidate_index,
            masked_action.action.key,
            exported.rank_count if exported is not None else "n/a",
            _active_rack_count(chromosome),
            repair.estimated_cost,
            repair.estimated_power_watts,
            repair.feasible and exported is not None,
        )

        cached = self._cached_evaluation(signature)
        if cached is not None:
            copied = TGRLEvaluation(
                episode=episode,
                step=step,
                candidate_index=candidate_index,
                masked_action=masked_action,
                feasible=cached.feasible,
                messages=cached.messages + ["cache hit"],
                objectives=cached.objectives,
                weighted_score=cached.weighted_score,
                repair=repair,
                exported=exported,
                feedback=cached.feedback,
                cache_hit=True,
            )
            dump_json(candidate_dir / "score.json", copied.to_summary())
            if copied.feedback is not None:
                dump_json(candidate_dir / "feedback.json", _feedback_to_dict(copied.feedback))
            logger.info(
                "TG-RL candidate episode=%03d step=%03d idx=%03d cache hit: score=%.4f feasible=%s",
                episode,
                step,
                candidate_index,
                copied.weighted_score,
                copied.feasible,
            )
            return copied

        if not repair.feasible or exported is None:
            evaluation = self._penalty_evaluation(
                masked_action,
                episode,
                step,
                candidate_index,
                repair,
                exported,
                repair.messages,
            )
            dump_json(candidate_dir / "score.json", evaluation.to_summary())
            self._store_cached_evaluation(signature, evaluation)
            logger.info(
                "TG-RL candidate episode=%03d step=%03d idx=%03d skipped with penalty: score=%.4f reason=%s",
                episode,
                step,
                candidate_index,
                evaluation.weighted_score,
                _message_preview(evaluation.messages),
            )
            return evaluation

        feedback: ParsedPipelineFeedback | None = None
        messages = list(repair.messages)
        feasible = True
        logger.info(
            "TG-RL candidate episode=%03d step=%03d idx=%03d running mapper/simulator: topology=%s",
            episode,
            step,
            candidate_index,
            candidate_dir / "hardware_topology.json",
        )
        try:
            feedback = self._pipeline.run(
                topology_path=candidate_dir / "hardware_topology.json",
                workload_path=self._workload_path,
                out_dir=candidate_dir / "wrapper",
            )
            if not feedback.summary.get("success", False):
                feasible = False
                messages.append("mapper/simulator wrapper reported success=false")
        except Exception as exc:
            feasible = False
            messages.append(f"pipeline failed: {exc}")

        objectives = self._objectives(repair, feedback, feasible)
        evaluation = TGRLEvaluation(
            episode=episode,
            step=step,
            candidate_index=candidate_index,
            masked_action=masked_action,
            feasible=feasible,
            messages=messages,
            objectives=objectives,
            weighted_score=self._weighted_score(objectives, feasible, repair.penalty),
            repair=repair,
            exported=exported,
            feedback=feedback,
        )
        dump_json(candidate_dir / "score.json", evaluation.to_summary())
        if feedback is not None:
            dump_json(candidate_dir / "feedback.json", _feedback_to_dict(feedback))
        self._store_cached_evaluation(signature, evaluation)
        logger.info(
            "TG-RL candidate episode=%03d step=%03d idx=%03d done: score=%.4f feasible=%s makespan_us=%.3f max_link_util=%.3f queue_ns=%.3f remote_ns=%.3f",
            episode,
            step,
            candidate_index,
            evaluation.weighted_score,
            evaluation.feasible,
            evaluation.objectives[0],
            evaluation.objectives[3],
            evaluation.objectives[4],
            evaluation.objectives[5],
        )
        return evaluation

    def _penalty_evaluation(
        self,
        masked_action: MaskedAction,
        episode: int,
        step: int,
        candidate_index: int,
        repair: RepairReport,
        exported: ExportedHardware | None,
        messages: list[str],
    ) -> TGRLEvaluation:
        objectives = (
            1_000_000_000.0 + repair.penalty,
            repair.estimated_cost,
            repair.estimated_power_watts,
            1_000_000.0,
            1_000_000_000.0,
            1_000_000_000.0,
        )
        return TGRLEvaluation(
            episode=episode,
            step=step,
            candidate_index=candidate_index,
            masked_action=masked_action,
            feasible=False,
            messages=messages,
            objectives=objectives,
            weighted_score=self._weighted_score(objectives, False, repair.penalty),
            repair=repair,
            exported=exported,
        )

    def _objectives(
        self,
        repair: RepairReport,
        feedback: ParsedPipelineFeedback | None,
        feasible: bool,
    ) -> tuple[float, float, float, float, float, float]:
        if feedback is None:
            return (
                1_000_000_000.0 + repair.penalty,
                repair.estimated_cost,
                repair.estimated_power_watts,
                1_000_000.0,
                1_000_000_000.0,
                1_000_000_000.0,
            )
        penalty = 0.0 if feasible else 1_000_000_000.0
        return (
            feedback.makespan_us + penalty,
            repair.estimated_cost,
            repair.estimated_power_watts,
            feedback.max_link_utilization + (1_000_000.0 if not feasible else 0.0),
            feedback.max_queue_delay_ns + penalty,
            feedback.remote_memory_contention_ns + penalty,
        )

    def _weighted_score(
        self,
        objectives: tuple[float, float, float, float, float, float],
        feasible: bool,
        penalty: float,
    ) -> float:
        weights: SearchObjectiveWeights = self._space.objective_weights
        score = (
            weights.makespan * (objectives[0] / 10_000.0)
            + weights.cost * (objectives[1] / 1_000_000.0)
            + weights.power * (objectives[2] / 100_000.0)
            + weights.max_link_utilization * objectives[3]
            + weights.max_queue_delay * (objectives[4] / 1_000_000.0)
            + weights.remote_memory_contention * (objectives[5] / 1_000_000.0)
        )
        if not feasible:
            score += 1_000_000.0 + penalty
        return score

    def _persist_step(
        self,
        episode: int,
        step: int,
        best: TGRLEvaluation,
        evaluations: list[TGRLEvaluation],
    ) -> None:
        step_dir = self._out_dir / f"episode_{episode:03d}" / f"step_{step:03d}"
        dump_json(
            step_dir / "step_summary.json",
            {
                "best_candidate_index": best.candidate_index,
                "best_score": best.weighted_score,
                "best_feasible": best.feasible,
                "candidates": [item.to_summary() for item in evaluations],
            },
        )
        dump_json(step_dir / "action.json", best.action.to_dict())
        dump_json(step_dir / "candidate.json", best.masked_action.to_dict())
        dump_json(step_dir / "score.json", best.to_summary())
        if best.exported is not None:
            dump_json(step_dir / "proposal.json", best.exported.proposal.to_dict())
            dump_json(step_dir / "hardware_topology.json", best.exported.hardware_topology)
        if best.feedback is not None:
            dump_json(step_dir / "feedback.json", _feedback_to_dict(best.feedback))

    def _persist_final(
        self,
        history: list[TGRLEvaluation],
        trajectory: list[TrajectoryItem],
        best: TGRLEvaluation,
    ) -> None:
        dump_json(
            self._out_dir / "tgrl_summary.json",
            {
                "mode": self._config.mode,
                "episodes": self._episodes,
                "steps_per_episode": self._steps_per_episode,
                "concurrency": self._concurrency,
                "evaluations": len(history),
                "trajectory_items": len(trajectory),
                "best": best.to_summary(),
            },
        )
        dump_json(self._out_dir / "policy_state.json", self._policy.to_dict())
        if best.exported is not None:
            dump_json(self._out_dir / "best_hardware_topology.json", best.exported.hardware_topology)
            dump_json(self._out_dir / "best_proposal.json", best.exported.proposal.to_dict())

    def _append_trajectory(self, item: TrajectoryItem) -> None:
        path = self._out_dir / "trajectory.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")

    def _cached_evaluation(self, signature: str) -> TGRLEvaluation | None:
        with self._cache_lock:
            return self._cache.get(signature)

    def _store_cached_evaluation(self, signature: str, evaluation: TGRLEvaluation) -> None:
        with self._cache_lock:
            self._cache.setdefault(signature, evaluation)


def enumerate_graph_edit_actions(
    chromosome: Chromosome,
    *,
    component_library: ComponentLibrary,
    search_space: SearchSpace,
) -> list[GraphEditAction]:
    pools = infer_type_pools(search_space, component_library.node_types, component_library.link_types)
    intra_link_order = ordered_link_types_for_scope(component_library, "intra")
    inter_link_order = ordered_link_types_for_scope(component_library, "inter")
    actions: list[GraphEditAction] = []
    for archetype in search_space.rack_archetypes:
        actions.append(GraphEditAction("add_rack_from_template", target=archetype.name))
    for rack in chromosome.racks:
        if rack.optional and not rack.active:
            actions.append(GraphEditAction("activate_optional_rack", rack_id=rack.rack_id))
            continue
        if rack.optional and rack.active:
            actions.append(GraphEditAction("deactivate_optional_rack", rack_id=rack.rack_id))
        if rack.dynamic or rack.optional or (
            rack.origin == "seed" and search_space.mutation.allow_remove_initial_racks
        ):
            actions.append(GraphEditAction("remove_rack", rack_id=rack.rack_id))
        for slot in rack.slots:
            if slot.node_type:
                actions.append(GraphEditAction("remove_node_from_slot", rack_id=rack.rack_id, resource=slot.slot_id))
                choices = [item for item in pools.gpu + pools.cpu if item != slot.node_type]
                for target in choices:
                    actions.append(
                        GraphEditAction(
                            "replace_node_type",
                            rack_id=rack.rack_id,
                            resource=slot.slot_id,
                            target=target,
                        )
                    )
                upgrade = _adjacent_node_type(slot.node_type, pools, component_library, direction=1)
                downgrade = _adjacent_node_type(slot.node_type, pools, component_library, direction=-1)
                if upgrade:
                    actions.append(
                        GraphEditAction("upgrade_node", rack_id=rack.rack_id, resource=slot.slot_id, target=upgrade)
                    )
                if downgrade:
                    actions.append(
                        GraphEditAction("downgrade_node", rack_id=rack.rack_id, resource=slot.slot_id, target=downgrade)
                    )
            else:
                for target in pools.gpu + pools.cpu:
                    actions.append(
                        GraphEditAction(
                            "add_node_to_slot",
                            rack_id=rack.rack_id,
                            resource=slot.slot_id,
                            target=target,
                        )
                    )
        for mode in ("ring", "fully_connected", "switch"):
            if mode != rack.intra_rack_topology:
                actions.append(GraphEditAction("change_intra_rack_topology", rack_id=rack.rack_id, target=mode))
        for action_type, direction in [
            ("upgrade_intra_rack_link", 1),
            ("downgrade_intra_rack_link", -1),
        ]:
            target = _adjacent_link_type(rack.intra_rack_link_type, intra_link_order, direction=direction)
            if target:
                actions.append(GraphEditAction(action_type, rack_id=rack.rack_id, target=target))
            else:
                delta = 1 if direction > 0 else -1
                actions.append(GraphEditAction(action_type, rack_id=rack.rack_id, delta=delta))
    actions.append(GraphEditAction("upgrade_inter_rack_link", delta=1))
    actions.append(GraphEditAction("downgrade_inter_rack_link", delta=-1))
    for action_type, direction in [
        ("upgrade_inter_rack_link", 1),
        ("downgrade_inter_rack_link", -1),
    ]:
        target = _adjacent_link_type(chromosome.inter_rack_link_type, inter_link_order, direction=direction)
        if target:
            actions.append(GraphEditAction(action_type, target=target))
    for mode in ("ring", "fully_connected"):
        if mode != chromosome.inter_rack:
            actions.append(GraphEditAction("change_inter_rack_topology", target=mode))
    return _unique_actions(actions)


def build_masked_actions(
    chromosome: Chromosome,
    *,
    component_library: ComponentLibrary,
    search_space: SearchSpace,
    repairer: CandidateRepairer,
    exporter: HardwareTopologyExporter,
    feedback: ParsedPipelineFeedback | MultiWorkloadFeedback | None,
    current_repair: RepairReport,
    policy: LinearPolicy | None,
    config: TGRLConfig,
) -> list[MaskedAction]:
    current_signature = current_repair.chromosome.signature()
    context = telemetry_context(feedback, current_repair, search_space)
    actions: list[MaskedAction] = []
    for action in enumerate_graph_edit_actions(
        chromosome,
        component_library=component_library,
        search_space=search_space,
    ):
        if not _action_allowed_by_config(
            action,
            chromosome=chromosome,
            search_space=search_space,
            config=config,
        ):
            continue
        candidate = apply_graph_edit_action(chromosome, action, search_space=search_space)
        repair = repairer.repair_and_validate(candidate)
        if not repair.feasible:
            continue
        if repair.chromosome.signature() == current_signature:
            continue
        try:
            exporter.export(repair.chromosome)
        except Exception:
            continue
        score = heuristic_action_score(
            action,
            repair.chromosome,
            component_library=component_library,
            context=context,
        )
        features = action_features(action, repair.chromosome, context=context)
        actions.append(
            MaskedAction(
                action=action,
                chromosome=repair.chromosome,
                repair=repair,
                heuristic_score=score,
                features=features,
            )
        )
    if not actions:
        return []
    prior = softmax({item.action.key: item.heuristic_score for item in actions}, temperature=config.temperature)
    logits: dict[str, float] = {}
    for item in actions:
        learned = policy.score(item.features) if policy is not None else 0.0
        logits[item.action.key] = config.heuristic_weight * item.heuristic_score + learned
    probs = softmax(logits, temperature=config.temperature)
    for item in actions:
        item.prior_prob = prior.get(item.action.key, 0.0)
        item.policy_prob = probs.get(item.action.key, 0.0)
        item.logprob = math.log(max(1e-12, item.policy_prob))
    return actions


def _action_allowed_by_config(
    action: GraphEditAction,
    *,
    chromosome: Chromosome,
    search_space: SearchSpace,
    config: TGRLConfig,
) -> bool:
    if action.action_type in TOPOLOGY_CHANGING_ACTION_TYPES:
        if config.freeze_topology:
            return False
        if not _topology_action_allowed_by_exhaustive_space(action, chromosome, search_space):
            return False
    if action.action_type in {"replace_node_type", "upgrade_node", "downgrade_node"}:
        allowed_node_types = _exhaustive_slot_node_types(search_space)
        if allowed_node_types and action.target not in allowed_node_types:
            return False
    if action.action_type == "add_node_to_slot":
        allowed_node_types = _exhaustive_slot_node_types(search_space)
        if allowed_node_types and action.target not in allowed_node_types:
            return False
    if (
        action.action_type == "remove_node_from_slot"
        and not config.allow_empty_slots
    ):
        return False
    return True


def _topology_action_allowed_by_exhaustive_space(
    action: GraphEditAction,
    chromosome: Chromosome,
    search_space: SearchSpace,
) -> bool:
    exhaustive = search_space.exhaustive
    if action.action_type == "change_intra_rack_topology":
        return _target_in_optional_values(action.target, exhaustive.intra_rack_topologies)
    if action.action_type == "change_inter_rack_topology":
        return _target_in_optional_values(action.target, exhaustive.inter_rack_topologies)
    if action.action_type in {"upgrade_intra_rack_link", "downgrade_intra_rack_link"}:
        if action.target and not _target_in_optional_values(action.target, exhaustive.intra_rack_link_types):
            return False
        if action.delta and exhaustive.intra_rack_link_qty is not None:
            rack = _find_rack(chromosome, action.rack_id)
            if rack is None:
                return False
            next_qty = max(1, rack.intra_rack_link_qty + action.delta)
            if next_qty not in exhaustive.intra_rack_link_qty:
                return False
        return True
    if action.action_type in {"upgrade_inter_rack_link", "downgrade_inter_rack_link"}:
        if action.target and not _target_in_optional_values(action.target, exhaustive.inter_rack_link_types):
            return False
        if action.delta and exhaustive.inter_rack_link_qty is not None:
            next_qty = max(1, chromosome.inter_rack_link_qty + action.delta)
            if next_qty not in exhaustive.inter_rack_link_qty:
                return False
        return True
    return True


def _target_in_optional_values(target: str, values: list[str] | None) -> bool:
    return values is None or target in values


def apply_graph_edit_action(
    chromosome: Chromosome,
    action: GraphEditAction,
    *,
    search_space: SearchSpace | None = None,
) -> Chromosome:
    result = chromosome.model_copy(deep=True)
    rack = _find_rack(result, action.rack_id) if action.rack_id else None
    if action.action_type == "add_rack_from_template" and search_space is not None:
        archetype = _find_rack_archetype(search_space, action.target)
        if archetype is None:
            return result
        result.racks.append(rack_gene_from_archetype(archetype, _next_dynamic_rack_id(result, archetype.name)))
        if result.inter_rack == "none" and len([item for item in result.racks if item.active or not item.optional]) > 1:
            result.inter_rack = "ring"
        return result
    if action.action_type == "remove_rack" and rack is not None:
        if rack.dynamic or rack.optional or (
            rack.origin == "seed" and search_space is not None and search_space.mutation.allow_remove_initial_racks
        ):
            result.racks = [item for item in result.racks if item.rack_id != rack.rack_id]
        return result
    if action.action_type == "activate_optional_rack" and rack is not None:
        rack.active = True
        if rack.activation_alpha is not None:
            rack.activation_alpha = max(rack.activation_alpha, 1.0)
        if rack.intra_rack_topology == "switch" and rack.switch_count <= 0:
            rack.switch_count = 1
        if rack.role == "memory" and rack.memory_pool_count <= 0:
            rack.memory_pool_count = 1
        return result
    if action.action_type == "deactivate_optional_rack" and rack is not None and rack.optional:
        rack.active = False
        for slot in rack.slots:
            slot.node_type = None
            slot.link_type = None
            slot.link_qty = None
        rack.memory_pool_count = 0
        rack.switch_count = 0
        return result
    if action.action_type == "add_node_to_slot" and rack is not None:
        slot = _find_slot(rack, action.resource)
        if slot is not None and not slot.node_type:
            slot.node_type = action.target
            if search_space is not None:
                _apply_exhaustive_slot_link_option(slot, action.target, search_space)
            slot.link_type = slot.link_type or rack.intra_rack_link_type
            slot.link_qty = slot.link_qty or rack.intra_rack_link_qty
        return result
    if action.action_type == "remove_node_from_slot" and rack is not None:
        slot = _find_slot(rack, action.resource)
        if slot is not None:
            slot.node_type = None
            slot.link_type = None
            slot.link_qty = None
        return result
    if action.action_type in {"replace_node_type", "upgrade_node", "downgrade_node"} and rack is not None:
        slot = _find_slot(rack, action.resource)
        if slot is not None and slot.node_type:
            slot.node_type = action.target
            if search_space is not None:
                _apply_exhaustive_slot_link_option(slot, action.target, search_space)
        return result
    if action.action_type == "change_intra_rack_topology" and rack is not None:
        rack.intra_rack_topology = "switch" if action.target == "none" else action.target
        if rack.intra_rack_topology == "switch" and rack.switch_count <= 0:
            rack.switch_count = 1
        return result
    if action.action_type in {"upgrade_intra_rack_link", "downgrade_intra_rack_link"} and rack is not None:
        if action.target:
            rack.intra_rack_link_type = action.target
        elif action.delta:
            rack.intra_rack_link_qty = max(1, rack.intra_rack_link_qty + action.delta)
        return result
    if action.action_type in {"upgrade_inter_rack_link", "downgrade_inter_rack_link"}:
        if action.target:
            result.inter_rack_link_type = action.target
        elif action.delta:
            result.inter_rack_link_qty = max(1, result.inter_rack_link_qty + action.delta)
            if result.inter_rack == "none" and action.delta > 0:
                result.inter_rack = "ring"
        return result
    if action.action_type == "change_inter_rack_topology":
        result.inter_rack = "ring" if action.target == "none" else action.target
        return result
    return result


def heuristic_action_score(
    action: GraphEditAction,
    chromosome: Chromosome,
    *,
    component_library: ComponentLibrary,
    context: TelemetryContext,
) -> float:
    score = 0.0
    rack = _find_rack(chromosome, action.rack_id) if action.rack_id else None
    domain_hit = bool(rack and rack.rack_id in context.top_domain)
    network_pressure = max(context.network_utilization - 0.70, context.queue_pressure)
    low_network = context.network_utilization < 0.15 and context.queue_pressure <= 0
    if action.action_type == "add_node_to_slot":
        score += max(0.0, context.compute_utilization - 0.60) * 3.0
        score -= context.constraint_pressure
    elif action.action_type == "remove_node_from_slot":
        score += context.constraint_pressure * 2.5
        if context.compute_utilization < 0.25:
            score += 0.5
    elif action.action_type in {"replace_node_type", "upgrade_node", "downgrade_node"} and rack is not None:
        score += _slot_type_mutation_score(rack, action, component_library, context)
    elif action.action_type == "upgrade_intra_rack_link":
        score += max(0.0, network_pressure) * (3.0 if domain_hit else 1.0)
        score -= context.constraint_pressure * 0.3
    elif action.action_type == "downgrade_intra_rack_link":
        if low_network:
            score += 1.0
        score += context.constraint_pressure * 1.5
    elif action.action_type == "upgrade_inter_rack_link":
        score += max(0.0, network_pressure) * 4.0 if context.top_domain.startswith("cluster:") else 0.5
        score -= context.constraint_pressure * 0.3
    elif action.action_type == "downgrade_inter_rack_link":
        if low_network:
            score += 1.0
        score += context.constraint_pressure * 1.5
    elif action.action_type == "change_intra_rack_topology":
        if action.target == "fully_connected":
            score += max(0.0, network_pressure) * 2.5
        elif action.target == "switch":
            score += max(0.0, network_pressure) * 1.5
        elif action.target == "ring":
            score += 0.4 + context.constraint_pressure * 0.2
        else:
            score += context.constraint_pressure
    elif action.action_type == "change_inter_rack_topology":
        if action.target == "fully_connected":
            score += max(0.0, network_pressure) * 3.0
        elif action.target == "ring":
            score += 0.5 + context.constraint_pressure * 0.3
    elif action.action_type == "activate_optional_rack" and rack is not None:
        if rack.role in {"compute", "hybrid"}:
            score += max(0.0, context.compute_utilization - 0.75) * 4.0
        if rack.role in {"memory", "hybrid"}:
            score += context.remote_memory_pressure * 5.0
        score += max(0.0, network_pressure) * 0.5
        score -= context.constraint_pressure * 1.5
    elif action.action_type == "deactivate_optional_rack":
        score += context.constraint_pressure * 3.0
        if context.compute_utilization < 0.25 and context.remote_memory_pressure <= 0:
            score += 0.5
    elif action.action_type == "add_rack_from_template":
        lowered_target = action.target.lower()
        if any(token in lowered_target for token in ["gpu", "cpu", "compute", "hybrid"]):
            score += max(0.0, context.compute_utilization - 0.75) * 4.0
        if any(token in lowered_target for token in ["mem", "memory", "pool", "hybrid"]):
            score += context.remote_memory_pressure * 5.0
        score += max(0.0, network_pressure) * 0.5
        score -= context.constraint_pressure * 1.5
    elif action.action_type == "remove_rack":
        score += context.constraint_pressure * 3.0
        if context.compute_utilization < 0.25 and context.remote_memory_pressure <= 0:
            score += 0.5
    return score


def action_features(
    action: GraphEditAction,
    chromosome: Chromosome,
    *,
    context: TelemetryContext,
) -> dict[str, float]:
    rack = _find_rack(chromosome, action.rack_id) if action.rack_id else None
    slot = _find_slot(rack, action.resource) if rack is not None else None
    features: dict[str, float] = {
        "bias": 1.0,
        f"type:{action.action_type}": 1.0,
        f"resource:{'slot' if slot is not None else action.resource or 'none'}": 1.0,
        "compute_utilization": context.compute_utilization,
        "network_utilization": context.network_utilization,
        "queue_pressure": context.queue_pressure,
        "remote_memory_pressure": context.remote_memory_pressure,
        "constraint_pressure": context.constraint_pressure,
    }
    if rack is not None:
        features[f"rack_role:{rack.role}"] = 1.0
        features["rack_optional"] = 1.0 if rack.optional else 0.0
        features["rack_active"] = 1.0 if rack.active else 0.0
        features["domain_hit"] = 1.0 if rack.rack_id in context.top_domain else 0.0
        features["rack_occupied_slots"] = len(rack.occupied_slots) / max(1.0, rack.max_slots)
        features["rack_free_slots"] = len(rack.free_slots) / max(1.0, rack.max_slots)
        features["slot_occupied"] = 1.0 if slot and slot.node_type else 0.0
        features["rack_memory_count"] = rack.memory_pool_count / 8.0
    return features


def telemetry_context(
    feedback: ParsedPipelineFeedback | MultiWorkloadFeedback | None,
    repair: RepairReport,
    space: SearchSpace,
) -> TelemetryContext:
    if isinstance(feedback, MultiWorkloadFeedback):
        feedback = feedback.aggregate_feedback
    compute_util = _average_compute_utilization(feedback)
    network_util = feedback.max_link_utilization if feedback is not None else 0.0
    queue_pressure = min(1.0, (feedback.max_queue_delay_ns / 1_000_000.0) if feedback else 0.0)
    remote_pressure = min(1.0, (feedback.remote_memory_contention_ns / 1_000_000.0) if feedback else 0.0)
    top_link = feedback.link_stats[0] if feedback and feedback.link_stats else {}
    top_domain = str(top_link.get("stats_domain") or top_link.get("domain") or "")
    return TelemetryContext(
        compute_utilization=compute_util,
        network_utilization=network_util,
        queue_pressure=queue_pressure,
        remote_memory_pressure=remote_pressure,
        top_domain=top_domain,
        cost_pressure=_positive_ratio(repair.estimated_cost, space.limits.max_total_cost),
        power_pressure=_positive_ratio(repair.estimated_power_watts, space.limits.max_peak_power_watts),
    )


def sample_masked_actions(
    actions: list[MaskedAction],
    *,
    count: int,
    rng: random.Random,
    greedy: bool = False,
) -> list[MaskedAction]:
    if count <= 0 or not actions:
        return []
    if greedy:
        return sorted(actions, key=lambda item: item.policy_prob, reverse=True)[:count]
    remaining = list(actions)
    selected: list[MaskedAction] = []
    while remaining and len(selected) < count:
        total = sum(max(0.0, item.policy_prob) for item in remaining)
        if total <= 0:
            selected.append(remaining.pop(rng.randrange(len(remaining))))
            continue
        threshold = rng.random() * total
        acc = 0.0
        for idx, item in enumerate(remaining):
            acc += max(0.0, item.policy_prob)
            if acc >= threshold:
                selected.append(item)
                remaining.pop(idx)
                break
    return selected


def _slot_type_mutation_score(
    rack: RackGene,
    action: GraphEditAction,
    library: ComponentLibrary,
    context: TelemetryContext,
) -> float:
    slot = _find_slot(rack, action.resource)
    current = slot.node_type if slot else None
    target = action.target
    if not current or target not in library.node_types or current not in library.node_types:
        return 0.0
    current_spec = library.node_types[current]
    target_spec = library.node_types[target]
    score = 0.0
    delta_rank = _node_rank_score(target_spec) - _node_rank_score(current_spec)
    if action.action_type == "upgrade_node" or delta_rank > 0:
        score += max(0.0, context.compute_utilization - 0.65) * 3.0
    elif action.action_type == "downgrade_node" or delta_rank < 0:
        score += context.constraint_pressure * 1.5
    cost_delta = target_spec.cost_unit + target_spec.tdp_watts - current_spec.cost_unit - current_spec.tdp_watts
    if cost_delta < 0:
        score += context.constraint_pressure * 1.5
    elif context.constraint_pressure > 0:
        score -= context.constraint_pressure
    return score


def _average_compute_utilization(feedback: ParsedPipelineFeedback | None) -> float:
    if feedback is None or not feedback.simulation_feedback.compute_profile:
        return 0.0
    values = [item.avg_utilization for item in feedback.simulation_feedback.compute_profile.values()]
    return sum(values) / len(values)


def _positive_ratio(value: float, limit: float) -> float:
    if limit <= 0 or value <= limit:
        return 0.0
    return min(2.0, (value - limit) / limit)


def _node_peak_tflops(spec: NodeTypeSpec) -> float:
    return spec.peak_tflops or spec.compute_teraflops_dense or spec.compute_teraflops_sparse or 0.0


def _node_rank_score(spec: NodeTypeSpec) -> float:
    return (
        _node_peak_tflops(spec) * 1_000_000.0
        + (spec.memory_bw_gbps or 0.0) * 1_000.0
        + (spec.local_memory_gb or 0.0)
    )


def _find_rack(chromosome: Chromosome, rack_id: str) -> RackGene | None:
    for rack in chromosome.racks:
        if rack.rack_id == rack_id:
            return rack
    return None


def _find_slot(rack: RackGene | None, slot_id: str) -> Any | None:
    if rack is None:
        return None
    for slot in rack.slots:
        if slot.slot_id == slot_id:
            return slot
    return None


def _find_rack_archetype(search_space: SearchSpace, name: str) -> Any | None:
    for archetype in search_space.rack_archetypes:
        if archetype.name == name:
            return archetype
    return None


def _exhaustive_slot_node_types(search_space: SearchSpace) -> set[str]:
    return {
        option.node_type
        for option in search_space.exhaustive.slot_options
        if option.node_type is not None
    }


def _apply_exhaustive_slot_link_option(slot: Any, node_type: str, search_space: SearchSpace) -> None:
    for option in search_space.exhaustive.slot_options:
        if option.node_type != node_type:
            continue
        if option.link_type is not None:
            slot.link_type = option.link_type
        if option.link_qty is not None:
            slot.link_qty = option.link_qty
        return


def _next_dynamic_rack_id(chromosome: Chromosome, archetype_name: str) -> str:
    base = _safe_id(archetype_name)
    existing = {rack.rack_id for rack in chromosome.racks}
    index = 0
    while True:
        candidate = f"dyn-{base}-{index}"
        if candidate not in existing:
            return candidate
        index += 1


def _safe_id(text: str) -> str:
    cleaned = []
    for char in text.lower():
        cleaned.append(char if char.isalnum() else "-")
    result = "-".join(part for part in "".join(cleaned).split("-") if part)
    return result or "rack"


def _ordered_node_types(types: list[str], library: ComponentLibrary) -> list[str]:
    return sorted(types, key=lambda name: _node_rank_score(library.node_types[name]))


def _adjacent_node_type(
    current: str,
    pools: Any,
    library: ComponentLibrary,
    *,
    direction: int,
) -> str | None:
    if current not in library.node_types:
        return None
    role = role_of_type(current, library.node_types[current].role)
    candidates = pools.cpu if role == "cpu" else pools.gpu
    ordered = _ordered_node_types([item for item in candidates if item in library.node_types], library)
    if current not in ordered:
        return None
    idx = ordered.index(current) + direction
    if idx < 0 or idx >= len(ordered):
        return None
    return ordered[idx]


def _adjacent_link_type(current: str | None, ordered: list[str], *, direction: int) -> str | None:
    if not current or current not in ordered:
        return None
    idx = ordered.index(current) + direction
    if idx < 0 or idx >= len(ordered):
        return None
    return ordered[idx]


def _unique_actions(actions: list[GraphEditAction]) -> list[GraphEditAction]:
    seen: set[str] = set()
    result: list[GraphEditAction] = []
    for action in actions:
        if action.key in seen:
            continue
        seen.add(action.key)
        result.append(action)
    return result


def _feedback_to_dict(feedback: ParsedPipelineFeedback) -> dict[str, Any]:
    return {
        "simulation_feedback": feedback.simulation_feedback.model_dump(mode="json"),
        "summary": feedback.summary,
        "link_stats": feedback.link_stats,
        "domain_stats": feedback.domain_stats,
        "remote_memory_stats": feedback.remote_memory_stats,
        "scaling_report": feedback.scaling_report,
        "operator_times": feedback.operator_times,
        "compute_comm_overlap_ns": feedback.compute_comm_overlap_ns,
        "simulator_stdout": str(feedback.simulator_stdout_path) if feedback.simulator_stdout_path else "",
    }


def _active_rack_count(chromosome: Chromosome) -> int:
    return sum(1 for rack in chromosome.racks if rack.active or not rack.optional)


def _message_preview(messages: list[str]) -> str:
    if not messages:
        return "none"
    text = "; ".join(messages[:2])
    if len(messages) > 2:
        text += "; ..."
    return text
