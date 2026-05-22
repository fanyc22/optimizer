from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import dataclass
import html
import json
import logging
import random
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
import torch

from codesign_optimizer.io.jsonc import dump_json
from codesign_optimizer.models.hardware import ComponentLibrary
from codesign_optimizer.optimizer.chromosome import Chromosome, chromosome_from_template
from codesign_optimizer.optimizer.exporter import ExportedHardware, HardwareTopologyExporter
from codesign_optimizer.optimizer.feedback_parser import ParsedPipelineFeedback
from codesign_optimizer.optimizer.pipeline_client import PipelineClient
from codesign_optimizer.optimizer.repair import CandidateRepairer, RepairReport
from codesign_optimizer.optimizer.search_space import SearchObjectiveWeights, SearchSpace
from codesign_optimizer.optimizer.tgrl import (
    MaskedAction,
    TGRLConfig,
    TGRLEvaluation,
    _feedback_to_dict,
    build_masked_actions,
)
from codesign_optimizer.optimizer.tgrl_v2.model import TGRLGNNPolicy, policy_distribution
from codesign_optimizer.optimizer.tgrl_v2.observation import GraphObservationBuilder
from codesign_optimizer.optimizer.tgrl_v2.ppo import PPOConfig, PPOTransition, attach_gae, ppo_update


logger = logging.getLogger(__name__)


class TGRLPPOConfig(BaseModel):
    ppo_epochs: int = Field(default=4, ge=1)
    minibatch_size: int = Field(default=16, ge=1)
    gamma: float = Field(default=0.95, ge=0.0, le=1.0)
    gae_lambda: float = Field(default=0.90, ge=0.0, le=1.0)
    clip_range: float = Field(default=0.2, ge=0.0)
    value_coef: float = Field(default=0.5, ge=0.0)
    entropy_coef: float = Field(default=0.01, ge=0.0)
    kl_weight: float = Field(default=0.1, ge=0.0)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    heuristic_weight: float = Field(default=1.0, ge=0.0)
    temperature: float = Field(default=1.0, gt=0.0)
    duplicate_penalty: float = Field(default=0.05, ge=0.0)
    best_improvement_bonus: float = Field(default=0.1, ge=0.0)
    reward_clip: float = Field(default=5.0, gt=0.0)
    device: str = "auto"
    resume: Path | None = None

    def to_ppo_config(self) -> PPOConfig:
        return PPOConfig(
            ppo_epochs=self.ppo_epochs,
            minibatch_size=self.minibatch_size,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            value_coef=self.value_coef,
            entropy_coef=self.entropy_coef,
            kl_weight=self.kl_weight,
            learning_rate=self.learning_rate,
            heuristic_weight=self.heuristic_weight,
            reward_clip=self.reward_clip,
        )


@dataclass
class EnvState:
    env_id: int
    chromosome: Chromosome
    previous_score: float
    last_feedback: ParsedPipelineFeedback | None
    trajectory: list[PPOTransition]
    initial_evaluation: TGRLEvaluation


@dataclass(frozen=True)
class TGRLPPOResult:
    history: list[TGRLEvaluation]
    transitions: list[PPOTransition]
    best: TGRLEvaluation


class TGRLPPOTrainer:
    def __init__(
        self,
        *,
        component_library: ComponentLibrary,
        search_space: SearchSpace,
        pipeline_client: PipelineClient,
        workload_path: Path,
        out_dir: Path,
        updates: int,
        rollout_steps: int,
        env_count: int,
        config: TGRLPPOConfig | None = None,
    ) -> None:
        self._library = component_library
        self._space = search_space
        self._pipeline = pipeline_client
        self._workload_path = workload_path
        self._out_dir = out_dir
        self._updates = updates
        self._rollout_steps = rollout_steps
        self._env_count = max(1, env_count)
        self._config = config or TGRLPPOConfig()
        self._device = _select_device(self._config.device)
        self._rng = random.Random(search_space.seed)
        self._exporter = HardwareTopologyExporter(component_library)
        self._repairer = CandidateRepairer(component_library, search_space)
        self._observation_builder = GraphObservationBuilder(component_library, search_space)
        self._model = TGRLGNNPolicy().to(self._device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._config.learning_rate)
        self._cache: dict[str, TGRLEvaluation] = {}
        self._cache_lock = threading.Lock()
        self._seen_signatures: set[str] = set()
        self._best_score = float("inf")
        self._start_update = 0
        if self._config.resume is not None:
            self._load_checkpoint(self._config.resume)

    @property
    def global_best_score(self) -> float:
        return self._best_score

    def run(self) -> TGRLPPOResult:
        self._out_dir.mkdir(parents=True, exist_ok=True)
        (self._out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        history: list[TGRLEvaluation] = []
        all_transitions: list[PPOTransition] = []
        previous_candidate_rows = _load_curve_rows(self._out_dir / "curves" / "candidate_scores.json")
        previous_update_rows = _load_curve_rows(self._out_dir / "curves" / "update_scores.json")
        metrics_history: list[dict[str, Any]] = _load_curve_rows(self._out_dir / "curves" / "ppo_metrics.json")
        curve_initial_best = _best_from_existing_rows(
            previous_candidate_rows,
            previous_update_rows,
            self._best_score,
        )
        best: TGRLEvaluation | None = None
        logger.info(
            "Starting TG-RL v2 search: additional_updates=%d start_update=%d rollout_steps=%d envs=%d device=%s out=%s",
            self._updates,
            self._start_update,
            self._rollout_steps,
            self._env_count,
            self._device,
            self._out_dir,
        )
        if self._config.resume is not None:
            logger.info(
                "Resumed TG-RL v2 checkpoint: path=%s best_score=%.4f seen_signatures=%d",
                self._config.resume,
                self._best_score,
                len(self._seen_signatures),
            )

        for local_update, update in enumerate(range(self._start_update, self._start_update + self._updates)):
            logger.info(
                "TG-RL v2 update %d (%d/%d this run): initializing %d rollout envs",
                update,
                local_update + 1,
                self._updates,
                self._env_count,
            )
            envs = self._initialize_envs(update)
            for env in envs:
                history.append(env.initial_evaluation)
                if best is None or env.initial_evaluation.weighted_score < best.weighted_score:
                    best = env.initial_evaluation
                if env.initial_evaluation.weighted_score < self._best_score:
                    self._best_score = env.initial_evaluation.weighted_score
            if envs:
                initial_best = min((env.initial_evaluation for env in envs), key=lambda item: item.weighted_score)
                logger.info(
                    "TG-RL v2 update %d initial complete: best_score=%.4f feasible=%s",
                    update,
                    initial_best.weighted_score,
                    initial_best.feasible,
                )

            update_trajectories: list[list[PPOTransition]] = [[] for _ in envs]
            update_evaluations: list[TGRLEvaluation] = []
            for step in range(self._rollout_steps):
                selections = self._select_actions(
                    envs,
                    update=update,
                    update_position=local_update,
                    step=step,
                    best_score=self._best_score,
                )
                if not selections:
                    logger.info(
                        "TG-RL v2 update %d step %d/%d: no feasible actions, ending rollout",
                        update,
                        step + 1,
                        self._rollout_steps,
                    )
                    break
                logger.info(
                    "TG-RL v2 update %d step %d/%d: selected %d env actions",
                    update,
                    step + 1,
                    self._rollout_steps,
                    len(selections),
                )
                evaluations = self._evaluate_selections(selections, update=update, step=step)
                for evaluation, transition, env in evaluations:
                    history.append(evaluation)
                    update_evaluations.append(evaluation)
                    reward = self._reward(
                        previous_score=env.previous_score,
                        new_score=evaluation.weighted_score,
                        feasible=evaluation.feasible,
                        signature=evaluation.chromosome.signature(),
                    )
                    transition.reward = reward
                    transition.candidate_signature = evaluation.chromosome.signature()
                    transition.done = False
                    update_trajectories[env.env_id].append(transition)
                    all_transitions.append(transition)
                    self._append_trajectory(update, transition, evaluation)
                    env.chromosome = evaluation.chromosome.model_copy(deep=True)
                    env.previous_score = evaluation.weighted_score
                    env.last_feedback = evaluation.feedback
                    if best is None or evaluation.weighted_score < best.weighted_score:
                        best = evaluation
                    if evaluation.weighted_score < self._best_score:
                        self._best_score = evaluation.weighted_score
                self._persist_step(update, step, [item[0] for item in evaluations])
                step_best = min((item[0] for item in evaluations), key=lambda item: item.weighted_score)
                logger.info(
                    "TG-RL v2 update %d step %d/%d complete: best_action=%s best_score=%.4f feasible=%s cache_hits=%d",
                    update,
                    step + 1,
                    self._rollout_steps,
                    step_best.action.key,
                    step_best.weighted_score,
                    step_best.feasible,
                    sum(1 for item, _transition, _env in evaluations if item.cache_hit),
                )

            flattened = attach_gae(
                update_trajectories,
                gamma=self._config.gamma,
                gae_lambda=self._config.gae_lambda,
            )
            metrics = ppo_update(
                model=self._model,
                optimizer=self._optimizer,
                transitions=flattened,
                config=self._config.to_ppo_config(),
                device=self._device,
                rng=self._rng,
            )
            self._persist_update(update, update_evaluations, metrics)
            metrics_history.append(
                {
                    "update": update,
                    "transitions": len(flattened),
                    "evaluations": len(update_evaluations),
                    "best_score": min((item.weighted_score for item in update_evaluations), default=float("nan")),
                    "mean_score": _mean(item.weighted_score for item in update_evaluations),
                    **metrics,
                }
            )
            self._persist_curves(
                history,
                metrics_history,
                previous_candidate_rows=previous_candidate_rows,
                previous_update_rows=previous_update_rows,
                initial_best_score=curve_initial_best,
            )
            logger.info(
                "TG-RL v2 update %d PPO update complete: transitions=%d policy_loss=%.6f value_loss=%.6f entropy=%.6f kl_prior=%.6f",
                update,
                len(flattened),
                metrics.get("policy_loss", 0.0),
                metrics.get("value_loss", 0.0),
                metrics.get("entropy", 0.0),
                metrics.get("kl_prior", 0.0),
            )
            self._save_checkpoint(self._out_dir / "checkpoints" / "policy_latest.pt", update=update)
            if best is not None and best.weighted_score <= self._best_score:
                self._save_checkpoint(self._out_dir / "checkpoints" / "policy_best.pt", update=update)

        if best is None:
            raise RuntimeError("TG-RL v2 did not evaluate any candidate")
        self._persist_final(history, all_transitions, best)
        self._persist_curves(
            history,
            metrics_history,
            previous_candidate_rows=previous_candidate_rows,
            previous_update_rows=previous_update_rows,
            initial_best_score=curve_initial_best,
        )
        logger.info(
            "TG-RL v2 search finished: evaluations=%d transitions=%d best_score=%.4f feasible=%s",
            len(history),
            len(all_transitions),
            best.weighted_score,
            best.feasible,
        )
        return TGRLPPOResult(history=history, transitions=all_transitions, best=best)

    def _initialize_envs(self, update: int) -> list[EnvState]:
        chromosomes = [self._initial_chromosome() for _ in range(self._env_count)]
        evaluations = self._evaluate_initials(chromosomes, update)
        envs: list[EnvState] = []
        for env_id, evaluation in enumerate(evaluations):
            env = EnvState(
                env_id=env_id,
                chromosome=evaluation.chromosome.model_copy(deep=True),
                previous_score=evaluation.weighted_score,
                last_feedback=evaluation.feedback,
                trajectory=[],
                initial_evaluation=evaluation,
            )
            envs.append(env)
        return envs

    def _initial_chromosome(self) -> Chromosome:
        if not self._space.templates:
            raise ValueError("search space must contain at least one template")
        report = self._repairer.repair_and_validate(chromosome_from_template(self._space.templates[0]))
        return report.chromosome

    def _evaluate_initials(self, chromosomes: list[Chromosome], update: int) -> list[TGRLEvaluation]:
        result: list[TGRLEvaluation | None] = [None] * len(chromosomes)
        with ThreadPoolExecutor(max_workers=min(self._env_count, len(chromosomes))) as executor:
            futures = {
                executor.submit(
                    self._evaluate_chromosome,
                    chromosome,
                    update,
                    env_id,
                    -1,
                    self._out_dir / f"update_{update:03d}" / f"env_{env_id:03d}" / "initial",
                    MaskedAction(
                        action=_initial_action(),
                        chromosome=chromosome,
                        repair=self._repairer.repair_and_validate(chromosome),
                        heuristic_score=0.0,
                        features={"bias": 1.0},
                        prior_prob=1.0,
                        policy_prob=1.0,
                        logprob=0.0,
                    ),
                ): env_id
                for env_id, chromosome in enumerate(chromosomes)
            }
            for future in as_completed(futures):
                env_id = futures[future]
                result[env_id] = future.result()
        return [item for item in result if item is not None]

    def _select_actions(
        self,
        envs: list[EnvState],
        *,
        update: int,
        update_position: int,
        step: int,
        best_score: float,
    ) -> list[tuple[EnvState, MaskedAction, PPOTransition]]:
        selections: list[tuple[EnvState, MaskedAction, PPOTransition]] = []
        self._model.eval()
        for env in envs:
            repair = self._repairer.repair_and_validate(env.chromosome)
            masked_actions = build_masked_actions(
                env.chromosome,
                component_library=self._library,
                search_space=self._space,
                repairer=self._repairer,
                exporter=self._exporter,
                feedback=env.last_feedback,
                current_repair=repair,
                policy=None,
                config=TGRLConfig(temperature=self._config.temperature, heuristic_weight=self._config.heuristic_weight),
            )
            if not masked_actions:
                continue
            observation = self._observation_builder.build(
                chromosome=repair.chromosome,
                repair=repair,
                feedback=env.last_feedback,
                masked_actions=masked_actions,
                current_score=env.previous_score,
                best_score=best_score if best_score < float("inf") else env.previous_score,
                update=update_position,
                step=step,
                total_updates=self._updates,
                rollout_steps=self._rollout_steps,
            )
            with torch.no_grad():
                dist, _logits, value, _tensor_observation = policy_distribution(
                    self._model,
                    observation,
                    device=self._device,
                    heuristic_weight=self._config.heuristic_weight,
                )
                action_tensor = dist.sample()
                action_index = int(action_tensor.item())
                old_logprob = float(dist.log_prob(action_tensor).item())
                probs = dist.probs.detach().cpu().tolist()
            for idx, item in enumerate(observation.masked_actions):
                item.policy_prob = float(probs[idx])
                item.logprob = float(torch.log(torch.tensor(max(1e-12, probs[idx]))).item())
            selected = observation.masked_actions[action_index]
            transition = PPOTransition(
                observation=observation,
                action_index=action_index,
                old_logprob=old_logprob,
                value=float(value.item()),
                reward=0.0,
                done=False,
                candidate_signature="",
                episode_env=env.env_id,
                rollout_step=step,
            )
            selections.append((env, selected, transition))
        return selections

    def _evaluate_selections(
        self,
        selections: list[tuple[EnvState, MaskedAction, PPOTransition]],
        *,
        update: int,
        step: int,
    ) -> list[tuple[TGRLEvaluation, PPOTransition, EnvState]]:
        result: list[tuple[TGRLEvaluation, PPOTransition, EnvState] | None] = [None] * len(selections)
        with ThreadPoolExecutor(max_workers=min(self._env_count, len(selections))) as executor:
            futures = {}
            for index, (env, masked_action, transition) in enumerate(selections):
                future = executor.submit(
                    self._evaluate_chromosome,
                    masked_action.chromosome,
                    update,
                    env.env_id,
                    step,
                    self._out_dir / f"update_{update:03d}" / f"env_{env.env_id:03d}" / f"step_{step:03d}",
                    masked_action,
                )
                futures[future] = (index, transition, env)
            for future in as_completed(futures):
                index, transition, env = futures[future]
                evaluation = future.result()
                obs_path = (
                    self._out_dir
                    / f"update_{update:03d}"
                    / f"env_{env.env_id:03d}"
                    / f"step_{step:03d}"
                    / "observation.json"
                )
                dump_json(obs_path, transition.observation.to_dict())
                result[index] = (evaluation, transition, env)
        return [item for item in result if item is not None]

    def _evaluate_chromosome(
        self,
        chromosome: Chromosome,
        update: int,
        env_id: int,
        step: int,
        candidate_dir: Path,
        masked_action: MaskedAction,
    ) -> TGRLEvaluation:
        candidate_dir.mkdir(parents=True, exist_ok=True)
        repair = self._repairer.repair_and_validate(chromosome)
        masked_action.repair = repair
        masked_action.chromosome = repair.chromosome
        dump_json(candidate_dir / "action.json", masked_action.action.to_dict())
        dump_json(candidate_dir / "candidate.json", masked_action.to_dict())

        signature = repair.chromosome.signature()
        exported: ExportedHardware | None = None
        try:
            exported = self._exporter.export(repair.chromosome, iteration=update)
            dump_json(candidate_dir / "proposal.json", exported.proposal.to_dict())
            dump_json(candidate_dir / "hardware_topology.json", exported.hardware_topology)
        except Exception as exc:
            repair = RepairReport(
                chromosome=repair.chromosome,
                feasible=False,
                messages=repair.messages + [f"export failed: {exc}"],
                estimated_cost=repair.estimated_cost,
                estimated_power_watts=repair.estimated_power_watts,
                penalty=repair.penalty + 1_000_000.0,
            )
        logger.info(
            "TG-RL v2 candidate update=%03d step=%03d env=%03d action=%s prepared: ranks=%s racks=%d cost=%.2f power=%.2f feasible=%s",
            update,
            step,
            env_id,
            masked_action.action.key,
            exported.rank_count if exported is not None else "n/a",
            _active_rack_count(repair.chromosome),
            repair.estimated_cost,
            repair.estimated_power_watts,
            repair.feasible and exported is not None,
        )

        cached = self._cached_evaluation(signature)
        if cached is not None:
            copied = TGRLEvaluation(
                episode=update,
                step=step,
                candidate_index=env_id,
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
            self._write_evaluation(candidate_dir, copied)
            logger.info(
                "TG-RL v2 candidate update=%03d step=%03d env=%03d cache hit: score=%.4f feasible=%s",
                update,
                step,
                env_id,
                copied.weighted_score,
                copied.feasible,
            )
            return copied

        if not repair.feasible or exported is None:
            evaluation = self._penalty_evaluation(masked_action, update, step, env_id, repair, exported, repair.messages)
            self._write_evaluation(candidate_dir, evaluation)
            self._store_cached_evaluation(signature, evaluation)
            logger.info(
                "TG-RL v2 candidate update=%03d step=%03d env=%03d skipped with penalty: score=%.4f reason=%s",
                update,
                step,
                env_id,
                evaluation.weighted_score,
                _message_preview(evaluation.messages),
            )
            return evaluation

        feedback: ParsedPipelineFeedback | None = None
        messages = list(repair.messages)
        feasible = True
        logger.info(
            "TG-RL v2 candidate update=%03d step=%03d env=%03d running mapper/simulator: topology=%s",
            update,
            step,
            env_id,
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
            episode=update,
            step=step,
            candidate_index=env_id,
            masked_action=masked_action,
            feasible=feasible,
            messages=messages,
            objectives=objectives,
            weighted_score=self._weighted_score(objectives, feasible, repair.penalty),
            repair=repair,
            exported=exported,
            feedback=feedback,
        )
        self._write_evaluation(candidate_dir, evaluation)
        self._store_cached_evaluation(signature, evaluation)
        logger.info(
            "TG-RL v2 candidate update=%03d step=%03d env=%03d done: score=%.4f feasible=%s makespan_us=%.3f max_link_util=%.3f queue_ns=%.3f remote_ns=%.3f",
            update,
            step,
            env_id,
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
        update: int,
        step: int,
        env_id: int,
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
            episode=update,
            step=step,
            candidate_index=env_id,
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

    def _reward(self, *, previous_score: float, new_score: float, feasible: bool, signature: str) -> float:
        scale = max(1.0, abs(previous_score))
        reward = (previous_score - new_score) / scale
        if not feasible:
            reward -= 1.0
        if signature in self._seen_signatures:
            reward -= self._config.duplicate_penalty
        if new_score < self._best_score:
            reward += self._config.best_improvement_bonus
        self._seen_signatures.add(signature)
        return max(-self._config.reward_clip, min(self._config.reward_clip, reward))

    def _write_evaluation(self, candidate_dir: Path, evaluation: TGRLEvaluation) -> None:
        dump_json(candidate_dir / "score.json", evaluation.to_summary())
        if evaluation.feedback is not None:
            dump_json(candidate_dir / "feedback.json", _feedback_to_dict(evaluation.feedback))

    def _persist_step(self, update: int, step: int, evaluations: list[TGRLEvaluation]) -> None:
        dump_json(
            self._out_dir / f"update_{update:03d}" / f"step_{step:03d}_summary.json",
            {"evaluations": [item.to_summary() for item in evaluations]},
        )

    def _persist_update(
        self,
        update: int,
        evaluations: list[TGRLEvaluation],
        metrics: dict[str, float],
    ) -> None:
        update_dir = self._out_dir / f"update_{update:03d}"
        dump_json(update_dir / "ppo_metrics.json", metrics)
        with (update_dir / "rollout.jsonl").open("w", encoding="utf-8") as handle:
            for evaluation in evaluations:
                handle.write(json.dumps(evaluation.to_summary(), sort_keys=True, separators=(",", ":")) + "\n")

    def _persist_final(
        self,
        history: list[TGRLEvaluation],
        transitions: list[PPOTransition],
        best: TGRLEvaluation,
    ) -> None:
        dump_json(
            self._out_dir / "tgrl_summary.json",
            {
                "mode": "v2",
                "updates": self._updates,
                "start_update": self._start_update,
                "end_update": self._start_update + self._updates - 1,
                "rollout_steps": self._rollout_steps,
                "env_count": self._env_count,
                "evaluations": len(history),
                "transitions": len(transitions),
                "global_best_score": self._best_score,
                "best": best.to_summary(),
            },
        )
        if best.exported is not None:
            dump_json(self._out_dir / "best_hardware_topology.json", best.exported.hardware_topology)
            dump_json(self._out_dir / "best_proposal.json", best.exported.proposal.to_dict())

    def _persist_curves(
        self,
        history: list[TGRLEvaluation],
        metrics_history: list[dict[str, Any]],
        *,
        previous_candidate_rows: list[dict[str, Any]],
        previous_update_rows: list[dict[str, Any]],
        initial_best_score: float,
    ) -> None:
        curve_dir = self._out_dir / "curves"
        curve_dir.mkdir(parents=True, exist_ok=True)
        candidate_rows = previous_candidate_rows + _candidate_score_rows(
            history,
            ordinal_offset=len(previous_candidate_rows),
            initial_best_score=initial_best_score,
        )
        update_rows = previous_update_rows + _update_score_rows(
            history,
            initial_best_score=initial_best_score,
        )
        dump_json(curve_dir / "candidate_scores.json", {"rows": candidate_rows})
        dump_json(curve_dir / "update_scores.json", {"rows": update_rows})
        dump_json(curve_dir / "ppo_metrics.json", {"rows": metrics_history})
        _write_csv(curve_dir / "candidate_scores.csv", candidate_rows)
        _write_csv(curve_dir / "update_scores.csv", update_rows)
        _write_csv(curve_dir / "ppo_metrics.csv", metrics_history)
        _write_svg_lines(
            curve_dir / "score_curve.svg",
            title="TG-RL v2 Score By Update",
            x_label="update",
            y_label="weighted score (lower is better)",
            series=[
                ("best_score", [(float(row["update"]), float(row["best_score"])) for row in update_rows]),
                ("mean_score", [(float(row["update"]), float(row["mean_score"])) for row in update_rows]),
                ("global_best_score", [(float(row["update"]), float(row["global_best_score"])) for row in update_rows]),
            ],
        )
        _write_svg_lines(
            curve_dir / "ppo_loss_curve.svg",
            title="TG-RL v2 PPO Metrics By Update",
            x_label="update",
            y_label="metric value",
            series=[
                ("loss", [(float(row["update"]), float(row.get("loss", 0.0))) for row in metrics_history]),
                ("policy_loss", [(float(row["update"]), float(row.get("policy_loss", 0.0))) for row in metrics_history]),
                ("value_loss", [(float(row["update"]), float(row.get("value_loss", 0.0))) for row in metrics_history]),
                ("kl_prior", [(float(row["update"]), float(row.get("kl_prior", 0.0))) for row in metrics_history]),
                ("entropy", [(float(row["update"]), float(row.get("entropy", 0.0))) for row in metrics_history]),
            ],
        )
        logger.info("TG-RL v2 curves updated: %s", curve_dir)

    def _append_trajectory(self, update: int, transition: PPOTransition, evaluation: TGRLEvaluation) -> None:
        path = self._out_dir / "trajectory.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            payload = transition.to_dict()
            payload["update"] = update
            payload["score"] = evaluation.weighted_score
            payload["feasible"] = evaluation.feasible
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")

    def _save_checkpoint(self, path: Path, *, update: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "update": update,
                "model_state": self._model.state_dict(),
                "optimizer_state": self._optimizer.state_dict(),
                "config": self._config.model_dump(mode="json"),
                "best_score": self._best_score,
                "rng_state": self._rng.getstate(),
                "torch_rng_state": torch.random.get_rng_state(),
                "seen_signatures": sorted(self._seen_signatures),
            },
            path,
        )

    def _load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self._device, weights_only=False)
        model_state = checkpoint["model_state"]
        current_state = self._model.state_dict()
        compatible_state = {
            key: value
            for key, value in model_state.items()
            if key in current_state and current_state[key].shape == value.shape
        }
        skipped = sorted(set(model_state) - set(compatible_state))
        current_state.update(compatible_state)
        self._model.load_state_dict(current_state)
        if skipped:
            logger.warning(
                "Checkpoint %s has %d incompatible model tensors; skipped optimizer state. First skipped keys: %s",
                path,
                len(skipped),
                ", ".join(skipped[:4]),
            )
        elif "optimizer_state" in checkpoint:
            self._optimizer.load_state_dict(checkpoint["optimizer_state"])
        self._best_score = float(checkpoint.get("best_score", float("inf")))
        self._start_update = int(checkpoint.get("update", -1)) + 1
        if "rng_state" in checkpoint:
            self._rng.setstate(checkpoint["rng_state"])
        if "torch_rng_state" in checkpoint:
            torch.random.set_rng_state(checkpoint["torch_rng_state"].detach().cpu())
        if "seen_signatures" in checkpoint:
            self._seen_signatures = set(str(item) for item in checkpoint["seen_signatures"])

    def _cached_evaluation(self, signature: str) -> TGRLEvaluation | None:
        with self._cache_lock:
            return self._cache.get(signature)

    def _store_cached_evaluation(self, signature: str, evaluation: TGRLEvaluation) -> None:
        with self._cache_lock:
            self._cache.setdefault(signature, evaluation)


def _select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is unavailable")
    if name == "mps" and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):
        raise RuntimeError("Requested --device mps but MPS is unavailable")
    return torch.device(name)


def _initial_action() -> Any:
    from codesign_optimizer.optimizer.tgrl import GraphEditAction

    return GraphEditAction(action_type="change_inter_rack_topology", target="initial")


def _candidate_score_rows(
    history: list[TGRLEvaluation],
    *,
    ordinal_offset: int = 0,
    initial_best_score: float = float("inf"),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best_so_far = initial_best_score
    for ordinal, evaluation in enumerate(history):
        best_so_far = min(best_so_far, evaluation.weighted_score)
        rows.append(
            {
                "ordinal": ordinal_offset + ordinal,
                "update": evaluation.episode,
                "step": evaluation.step,
                "env": evaluation.candidate_index,
                "score": evaluation.weighted_score,
                "best_score_so_far": best_so_far,
                "feasible": evaluation.feasible,
                "cache_hit": evaluation.cache_hit,
                "action": evaluation.action.key,
                "makespan_us": evaluation.objectives[0],
                "estimated_cost": evaluation.objectives[1],
                "estimated_power_watts": evaluation.objectives[2],
                "max_link_utilization": evaluation.objectives[3],
                "max_queue_delay_ns": evaluation.objectives[4],
                "remote_memory_contention_ns": evaluation.objectives[5],
            }
        )
    return rows


def _update_score_rows(
    history: list[TGRLEvaluation],
    *,
    initial_best_score: float = float("inf"),
) -> list[dict[str, Any]]:
    by_update: dict[int, list[TGRLEvaluation]] = {}
    for evaluation in history:
        by_update.setdefault(evaluation.episode, []).append(evaluation)
    rows: list[dict[str, Any]] = []
    global_best = initial_best_score
    for update in sorted(by_update):
        values = by_update[update]
        best = min(values, key=lambda item: item.weighted_score)
        global_best = min(global_best, best.weighted_score)
        rows.append(
            {
                "update": update,
                "evaluations": len(values),
                "feasible_evaluations": sum(1 for item in values if item.feasible),
                "cache_hits": sum(1 for item in values if item.cache_hit),
                "best_score": best.weighted_score,
                "mean_score": _mean(item.weighted_score for item in values),
                "global_best_score": global_best,
                "best_step": best.step,
                "best_env": best.candidate_index,
                "best_action": best.action.key,
                "best_makespan_us": best.objectives[0],
                "best_max_link_utilization": best.objectives[3],
                "best_max_queue_delay_ns": best.objectives[4],
                "best_remote_memory_contention_ns": best.objectives[5],
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_curve_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    rows = payload.get("rows", [])
    return rows if isinstance(rows, list) else []


def _best_from_existing_rows(
    candidate_rows: list[dict[str, Any]],
    update_rows: list[dict[str, Any]],
    checkpoint_best_score: float,
) -> float:
    values = [checkpoint_best_score]
    for row in candidate_rows:
        for key in ("best_score_so_far", "score"):
            try:
                values.append(float(row[key]))
            except (KeyError, TypeError, ValueError):
                pass
    for row in update_rows:
        for key in ("global_best_score", "best_score"):
            try:
                values.append(float(row[key]))
            except (KeyError, TypeError, ValueError):
                pass
    finite = [value for value in values if _is_finite(value)]
    return min(finite) if finite else float("inf")


def _write_svg_lines(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: list[tuple[str, list[tuple[float, float]]]],
) -> None:
    width = 920
    height = 460
    margin_left = 78
    margin_right = 24
    margin_top = 48
    margin_bottom = 68
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    clean_series = [
        (name, [(x, y) for x, y in points if _is_finite(x) and _is_finite(y)])
        for name, points in series
    ]
    all_points = [point for _name, points in clean_series for point in points]
    if not all_points:
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<text x="24" y="40" font-family="sans-serif" font-size="18">{html.escape(title)}</text>'
            '<text x="24" y="80" font-family="sans-serif" font-size="13">No data yet</text></svg>',
            encoding="utf-8",
        )
        return
    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    axis_ys = [
        value
        for _name, points in clean_series
        for value in _robust_axis_values([point[1] for point in points])
    ]
    if not axis_ys:
        axis_ys = ys
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = _padded_bounds(min(axis_ys), max(axis_ys))
    if min_x == max_x:
        min_x -= 0.5
        max_x += 0.5
    clipped_points = sum(1 for y in ys if y < min_y or y > max_y)

    def sx(x: float) -> float:
        return margin_left + (x - min_x) / (max_x - min_x) * plot_width

    def sy(y: float) -> float:
        return margin_top + (max_y - y) / (max_y - min_y) * plot_height

    colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2"]
    clip_id = "plot-area"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" data-y-min="{min_y:.12g}" data-y-max="{max_y:.12g}" '
        f'data-clipped-points="{clipped_points}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<defs><clipPath id="{clip_id}"><rect x="{margin_left}" y="{margin_top}" '
        f'width="{plot_width}" height="{plot_height}"/></clipPath></defs>',
        f'<text x="{margin_left}" y="28" font-family="sans-serif" font-size="18" font-weight="700">{html.escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1"/>',
    ]
    for i in range(5):
        ratio = i / 4
        x = margin_left + ratio * plot_width
        y = margin_top + ratio * plot_height
        x_value = min_x + ratio * (max_x - min_x)
        y_value = max_y - ratio * (max_y - min_y)
        parts.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + plot_height}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_width}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{x - 10:.1f}" y="{margin_top + plot_height + 20}" font-family="sans-serif" font-size="11" fill="#374151">{_format_tick(x_value)}</text>')
        parts.append(f'<text x="8" y="{y + 4:.1f}" font-family="sans-serif" font-size="11" fill="#374151">{_format_tick(y_value)}</text>')
    for idx, (name, points) in enumerate(clean_series):
        if not points:
            continue
        color = colors[idx % len(colors)]
        for segment in _line_segments(points):
            if len(segment) < 2:
                continue
            path_data = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in segment)
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2.2" '
                f'clip-path="url(#{clip_id})" points="{path_data}"/>'
            )
        for x, y in points:
            parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3" fill="{color}" clip-path="url(#{clip_id})"/>')
        legend_x = margin_left + 12 + (idx % 3) * 220
        legend_y = height - 34 + (idx // 3) * 16
        parts.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="16" height="4" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 22}" y="{legend_y - 5}" font-family="sans-serif" font-size="12" fill="#111827">{html.escape(name)}</text>')
    if clipped_points:
        parts.append(
            f'<text x="{margin_left}" y="44" font-family="sans-serif" font-size="11" fill="#6b7280">'
            f'Y axis uses robust range; {clipped_points} outlier point(s) clipped.</text>'
        )
    parts.append(f'<text x="{margin_left + plot_width / 2 - 28:.1f}" y="{height - 16}" font-family="sans-serif" font-size="12" fill="#111827">{html.escape(x_label)}</text>')
    parts.append(
        f'<text x="18" y="{margin_top + plot_height / 2:.1f}" transform="rotate(-90 18 {margin_top + plot_height / 2:.1f})" '
        f'font-family="sans-serif" font-size="12" fill="#111827">{html.escape(y_label)}</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _robust_axis_bounds(values: list[float]) -> tuple[float, float]:
    visible = _robust_axis_values(values)
    if not visible:
        return 0.0, 1.0
    return _padded_bounds(min(visible), max(visible))


def _robust_axis_values(values: list[float]) -> list[float]:
    finite = sorted(value for value in values if _is_finite(value))
    if not finite:
        return []

    visible = finite
    if len(finite) >= 3:
        median = _percentile(finite, 0.50)
        deviations = sorted(abs(value - median) for value in finite)
        mad = _percentile(deviations, 0.50)
        if mad > 1e-12:
            scaled_mad = 1.4826 * mad
            lower = median - 8.0 * scaled_mad
            upper = median + 8.0 * scaled_mad
            candidate = [value for value in finite if lower <= value <= upper]
            if len(candidate) >= max(2, len(finite) // 3):
                visible = candidate

    return visible


def _padded_bounds(min_value: float, max_value: float) -> tuple[float, float]:
    if min_value == max_value:
        pad = max(1.0, abs(min_value) * 0.05)
    else:
        pad = (max_value - min_value) * 0.08
    return min_value - pad, max_value + pad


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    index = max(0.0, min(1.0, fraction)) * (len(values) - 1)
    lower = int(index)
    upper = min(len(values) - 1, lower + 1)
    weight = index - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _line_segments(points: list[tuple[float, float]]) -> list[list[tuple[float, float]]]:
    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    previous_x: float | None = None
    for x, y in points:
        if current and previous_x is not None and x < previous_x:
            segments.append(current)
            current = []
        current.append((x, y))
        previous_x = x
    if current:
        segments.append(current)
    return segments


def _mean(values: Any) -> float:
    items = [float(value) for value in values if _is_finite(float(value))]
    if not items:
        return float("nan")
    return sum(items) / len(items)


def _is_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def _format_tick(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _active_rack_count(chromosome: Chromosome) -> int:
    return sum(1 for rack in chromosome.racks if rack.active or not rack.optional)


def _message_preview(messages: list[str]) -> str:
    if not messages:
        return "none"
    text = "; ".join(messages[:2])
    if len(messages) > 2:
        text += "; ..."
    return text
