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
)
from codesign_optimizer.optimizer.exporter import ExportedHardware, HardwareTopologyExporter
from codesign_optimizer.optimizer.feedback_parser import ParsedPipelineFeedback
from codesign_optimizer.optimizer.pipeline_client import PipelineClient
from codesign_optimizer.optimizer.repair import CandidateRepairer, RepairReport
from codesign_optimizer.optimizer.search_space import SearchObjectiveWeights, SearchSpace
from codesign_optimizer.optimizer.tcro import softmax


logger = logging.getLogger(__name__)


ActionType = Literal[
    "expand_rack_resource",
    "contract_rack_resource",
    "mutate_node_type",
    "upgrade_link_qty",
    "downgrade_link_qty",
    "change_inter_rack_mode",
    "activate_optional_rack",
    "deactivate_optional_rack",
]


class TGRLConfig(BaseModel):
    mode: Literal["v0", "v1"] = "v0"
    temperature: float = Field(default=1.0, gt=0)
    heuristic_weight: float = Field(default=1.0, ge=0)
    learning_rate: float = Field(default=0.05, ge=0)
    kl_weight: float = Field(default=0.05, ge=0)
    greedy: bool = False
    duplicate_penalty: float = Field(default=0.05, ge=0)


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
        action = GraphEditAction(action_type="change_inter_rack_mode", target="initial")
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
    actions: list[GraphEditAction] = []
    for rack in chromosome.racks:
        if rack.optional and not rack.active:
            actions.append(GraphEditAction("activate_optional_rack", rack_id=rack.rack_id))
            continue
        if rack.optional and rack.active:
            actions.append(GraphEditAction("deactivate_optional_rack", rack_id=rack.rack_id))
        for resource in ("gpu", "cpu", "memory"):
            if _resource_allowed(rack, resource):
                actions.append(GraphEditAction("expand_rack_resource", rack_id=rack.rack_id, resource=resource, delta=1))
                actions.append(GraphEditAction("contract_rack_resource", rack_id=rack.rack_id, resource=resource, delta=-1))
        for resource, options in [
            ("gpu", pools.gpu),
            ("cpu", pools.cpu),
            ("memory", pools.memory),
            ("switch", pools.switch),
        ]:
            current = _get_type(rack, resource)
            if current:
                for target in options:
                    if target != current:
                        actions.append(
                            GraphEditAction(
                                "mutate_node_type",
                                rack_id=rack.rack_id,
                                resource=resource,
                                target=target,
                            )
                        )
        for resource in ("endpoint_link", "gpu_link", "cpu_link", "memory_link"):
            if _link_resource_present(rack, resource):
                actions.append(GraphEditAction("upgrade_link_qty", rack_id=rack.rack_id, resource=resource, delta=1))
                actions.append(GraphEditAction("downgrade_link_qty", rack_id=rack.rack_id, resource=resource, delta=-1))
    actions.append(GraphEditAction("upgrade_link_qty", resource="inter_rack_link", delta=1))
    actions.append(GraphEditAction("downgrade_link_qty", resource="inter_rack_link", delta=-1))
    for mode in ("none", "ring", "fully_connected"):
        if mode != chromosome.inter_rack:
            actions.append(GraphEditAction("change_inter_rack_mode", target=mode))
    return _unique_actions(actions)


def build_masked_actions(
    chromosome: Chromosome,
    *,
    component_library: ComponentLibrary,
    search_space: SearchSpace,
    repairer: CandidateRepairer,
    exporter: HardwareTopologyExporter,
    feedback: ParsedPipelineFeedback | None,
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
        candidate = apply_graph_edit_action(chromosome, action)
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


def apply_graph_edit_action(chromosome: Chromosome, action: GraphEditAction) -> Chromosome:
    result = chromosome.model_copy(deep=True)
    rack = _find_rack(result, action.rack_id) if action.rack_id else None
    if action.action_type == "activate_optional_rack" and rack is not None:
        rack.active = True
        if rack.activation_alpha is not None:
            rack.activation_alpha = max(rack.activation_alpha, 1.0)
        if rack.fabric == "switch" and rack.switch_count <= 0:
            rack.switch_count = 1
        return result
    if action.action_type == "deactivate_optional_rack" and rack is not None and rack.optional:
        rack.active = False
        rack.gpu_count = 0
        rack.cpu_count = 0
        rack.memory_pool_count = 0
        rack.switch_count = 0
        return result
    if action.action_type in {"expand_rack_resource", "contract_rack_resource"} and rack is not None:
        _set_count(rack, action.resource, max(0, _get_count(rack, action.resource) + action.delta))
        return result
    if action.action_type == "mutate_node_type" and rack is not None:
        _set_type(rack, action.resource, action.target)
        return result
    if action.action_type in {"upgrade_link_qty", "downgrade_link_qty"}:
        if action.resource == "inter_rack_link":
            result.inter_rack_link_qty = max(1, result.inter_rack_link_qty + action.delta)
            if result.inter_rack == "none" and action.delta > 0:
                result.inter_rack = "ring"
            return result
        if rack is not None:
            _set_link_qty(rack, action.resource, max(1, _get_link_qty(rack, action.resource) + action.delta))
            return result
    if action.action_type == "change_inter_rack_mode":
        result.inter_rack = action.target
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
    if action.action_type == "expand_rack_resource":
        if action.resource in {"gpu", "cpu"}:
            score += max(0.0, context.compute_utilization - 0.75) * 3.0
        if action.resource == "memory":
            score += context.remote_memory_pressure * 4.0
        score -= context.constraint_pressure
    elif action.action_type == "contract_rack_resource":
        score += context.constraint_pressure * 2.5
        if context.compute_utilization < 0.25 and action.resource in {"gpu", "cpu"}:
            score += 0.5
        if context.remote_memory_pressure <= 0 and action.resource == "memory":
            score += 0.25
    elif action.action_type == "mutate_node_type" and rack is not None:
        score += _type_mutation_score(rack, action, component_library, context)
    elif action.action_type == "upgrade_link_qty":
        if action.resource == "inter_rack_link":
            score += max(0.0, network_pressure) * 4.0 if context.top_domain.startswith("cluster:") else 0.5
        elif action.resource == "memory_link":
            score += max(context.remote_memory_pressure, network_pressure) * 3.0
        else:
            score += max(0.0, network_pressure) * (3.0 if domain_hit else 1.0)
        score -= context.constraint_pressure * 0.3
    elif action.action_type == "downgrade_link_qty":
        if low_network:
            score += 1.0
        score += context.constraint_pressure * 1.5
    elif action.action_type == "change_inter_rack_mode":
        if action.target == "fully_connected":
            score += max(0.0, network_pressure) * 3.0
        elif action.target == "ring":
            score += 0.5 + context.constraint_pressure * 0.3
        elif action.target == "none":
            score += context.constraint_pressure * 2.0
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
    return score


def action_features(
    action: GraphEditAction,
    chromosome: Chromosome,
    *,
    context: TelemetryContext,
) -> dict[str, float]:
    rack = _find_rack(chromosome, action.rack_id) if action.rack_id else None
    features: dict[str, float] = {
        "bias": 1.0,
        f"type:{action.action_type}": 1.0,
        f"resource:{action.resource or 'none'}": 1.0,
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
        features["rack_gpu_count"] = rack.gpu_count / 16.0
        features["rack_cpu_count"] = rack.cpu_count / 32.0
        features["rack_memory_count"] = rack.memory_pool_count / 8.0
    return features


def telemetry_context(
    feedback: ParsedPipelineFeedback | None,
    repair: RepairReport,
    space: SearchSpace,
) -> TelemetryContext:
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


def _type_mutation_score(
    rack: RackGene,
    action: GraphEditAction,
    library: ComponentLibrary,
    context: TelemetryContext,
) -> float:
    current = _get_type(rack, action.resource)
    target = action.target
    if not current or target not in library.node_types or current not in library.node_types:
        return 0.0
    current_spec = library.node_types[current]
    target_spec = library.node_types[target]
    score = 0.0
    if action.resource in {"gpu", "cpu"}:
        delta_peak = _node_peak_tflops(target_spec) - _node_peak_tflops(current_spec)
        if delta_peak > 0:
            score += max(0.0, context.compute_utilization - 0.75) * 3.0
        else:
            score += context.constraint_pressure
    if action.resource == "memory":
        delta_bw = (target_spec.memory_bw_gbps or 0.0) - (current_spec.memory_bw_gbps or 0.0)
        delta_cap = (target_spec.capacity_gb or target_spec.local_memory_gb or 0.0) - (
            current_spec.capacity_gb or current_spec.local_memory_gb or 0.0
        )
        if delta_bw + delta_cap > 0:
            score += context.remote_memory_pressure * 3.0
        else:
            score += context.constraint_pressure
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


def _resource_allowed(rack: RackGene, resource: str) -> bool:
    if resource == "gpu":
        return rack.role in {"compute", "hybrid"} and rack.gpu_type is not None
    if resource == "cpu":
        return rack.role in {"compute", "hybrid"} and rack.cpu_type is not None
    if resource == "memory":
        return rack.role in {"memory", "hybrid"} and rack.memory_pool_type is not None
    return False


def _link_resource_present(rack: RackGene, resource: str) -> bool:
    if resource == "endpoint_link":
        return True
    if resource == "gpu_link":
        return rack.gpu_type is not None
    if resource == "cpu_link":
        return rack.cpu_type is not None
    if resource == "memory_link":
        return rack.memory_pool_type is not None
    return False


def _find_rack(chromosome: Chromosome, rack_id: str) -> RackGene | None:
    for rack in chromosome.racks:
        if rack.rack_id == rack_id:
            return rack
    return None


def _get_count(rack: RackGene, resource: str) -> int:
    if resource == "gpu":
        return rack.gpu_count
    if resource == "cpu":
        return rack.cpu_count
    if resource == "memory":
        return rack.memory_pool_count
    return 0


def _set_count(rack: RackGene, resource: str, value: int) -> None:
    if resource == "gpu":
        rack.gpu_count = value
    elif resource == "cpu":
        rack.cpu_count = value
    elif resource == "memory":
        rack.memory_pool_count = value


def _get_type(rack: RackGene, resource: str) -> str | None:
    if resource == "gpu":
        return rack.gpu_type
    if resource == "cpu":
        return rack.cpu_type
    if resource == "memory":
        return rack.memory_pool_type
    if resource == "switch":
        return rack.switch_type
    return None


def _set_type(rack: RackGene, resource: str, value: str) -> None:
    if resource == "gpu":
        rack.gpu_type = value
    elif resource == "cpu":
        rack.cpu_type = value
    elif resource == "memory":
        rack.memory_pool_type = value
    elif resource == "switch":
        rack.switch_type = value


def _get_link_qty(rack: RackGene, resource: str) -> int:
    if resource == "gpu_link":
        return rack.gpu_link_qty or rack.endpoint_link_qty
    if resource == "cpu_link":
        return rack.cpu_link_qty or rack.endpoint_link_qty
    if resource == "memory_link":
        return rack.memory_link_qty
    return rack.endpoint_link_qty


def _set_link_qty(rack: RackGene, resource: str, value: int) -> None:
    if resource == "gpu_link":
        rack.gpu_link_qty = value
    elif resource == "cpu_link":
        rack.cpu_link_qty = value
    elif resource == "memory_link":
        rack.memory_link_qty = value
    else:
        rack.endpoint_link_qty = value


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
