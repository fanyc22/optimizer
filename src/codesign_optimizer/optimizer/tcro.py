from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import math
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from codesign_optimizer.io.jsonc import dump_json
from codesign_optimizer.models.hardware import ComponentLibrary, NodeTypeSpec
from codesign_optimizer.optimizer.chromosome import (
    Chromosome,
    RackGene,
    chromosome_from_template,
    infer_type_pools,
)
from codesign_optimizer.optimizer.exporter import ExportedHardware, HardwareTopologyExporter
from codesign_optimizer.optimizer.feedback_parser import ParsedPipelineFeedback
from codesign_optimizer.optimizer.link_scope import LinkScope, link_type_allowed_for_scope, link_types_for_scope
from codesign_optimizer.optimizer.pipeline_client import PipelineClient
from codesign_optimizer.optimizer.repair import CandidateRepairer, RepairReport
from codesign_optimizer.optimizer.search_space import SearchObjectiveWeights, SearchSpace


logger = logging.getLogger(__name__)


class TCROConfig(BaseModel):
    learning_rate: float = Field(default=0.35, gt=0)
    initial_temperature: float = Field(default=1.0, ge=0)
    min_temperature: float = Field(default=0.05, ge=0)
    temperature_decay: float = Field(default=0.92, gt=0, le=1)
    noise_scale: float = Field(default=0.02, ge=0)
    link_prune_threshold: float = Field(default=0.25, ge=0)
    rack_activation_threshold: float = Field(default=0.5, ge=0)
    latent_rack_initial_alpha: float = Field(default=0.2, ge=0)
    constraint_penalty_weight: float = Field(default=1.0, ge=0)
    checkpoint_interval: int = Field(default=1, ge=1)
    telemetry_top_k: int = Field(default=1, ge=1)


class RackSupernetState(BaseModel):
    rack_id: str
    role: str
    optional: bool = False
    active_alpha: float = 1.0
    count_alpha: dict[str, float]
    link_alpha: dict[str, float]
    type_logits: dict[str, dict[str, float]]
    link_type_logits: dict[str, dict[str, float]]


class SupernetState(BaseModel):
    template_name: str
    base_chromosome: dict[str, Any]
    racks: list[RackSupernetState]
    inter_rack_alpha: float
    inter_rack_mode_logits: dict[str, float]
    inter_rack_link_type_logits: dict[str, float]
    temperature: float
    step: int = 0
    seed: int = 1

    def rack_state(self, rack_id: str) -> RackSupernetState | None:
        for rack in self.racks:
            if rack.rack_id == rack_id:
                return rack
        return None


@dataclass
class TCROCandidate:
    step: int
    sample: int
    chromosome: Chromosome
    source_temperature: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "sample": self.sample,
            "source_temperature": self.source_temperature,
            "chromosome": self.chromosome.to_dict(),
        }


@dataclass
class TCROCandidateEvaluation:
    step: int
    sample: int
    candidate: TCROCandidate
    feasible: bool
    messages: list[str]
    objectives: tuple[float, float, float, float, float, float]
    weighted_score: float
    repair: RepairReport
    exported: ExportedHardware | None = None
    feedback: ParsedPipelineFeedback | None = None
    cache_hit: bool = False

    def to_summary(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "sample": self.sample,
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
            "candidate": self.candidate.to_dict(),
        }


@dataclass(frozen=True)
class TCROSearchResult:
    history: list[TCROCandidateEvaluation]
    best: TCROCandidateEvaluation
    final_state: SupernetState


def softmax(logits: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    if not logits:
        return {}
    temp = max(1e-9, temperature)
    max_logit = max(logits.values())
    exp_values = {
        key: math.exp((value - max_logit) / temp)
        for key, value in logits.items()
    }
    total = sum(exp_values.values())
    if total <= 0:
        uniform = 1.0 / len(logits)
        return {key: uniform for key in logits}
    return {key: value / total for key, value in exp_values.items()}


def project_simplex(values: list[float]) -> list[float]:
    if not values:
        return []
    clipped = [max(0.0, value) for value in values]
    total = sum(clipped)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in clipped]


def quantize_alpha(alpha: float, *, threshold: float, minimum: int, maximum: int) -> int:
    if alpha < threshold:
        return 0
    return max(minimum, min(maximum, int(math.ceil(alpha))))


class TCROSearchRunner:
    def __init__(
        self,
        *,
        component_library: ComponentLibrary,
        search_space: SearchSpace,
        pipeline_client: PipelineClient,
        workload_path: Path,
        out_dir: Path,
        steps: int,
        samples_per_step: int,
        concurrency: int,
        config: TCROConfig | None = None,
    ) -> None:
        self._library = component_library
        self._space = search_space
        self._pipeline = pipeline_client
        self._workload_path = workload_path
        self._out_dir = out_dir
        self._steps = steps
        self._samples_per_step = samples_per_step
        self._concurrency = max(1, concurrency)
        self._config = config or TCROConfig()
        self._exporter = HardwareTopologyExporter(component_library)
        self._repairer = CandidateRepairer(component_library, search_space)
        self._cache: dict[str, TCROCandidateEvaluation] = {}
        self._cache_lock = threading.Lock()

    def run(self) -> TCROSearchResult:
        self._out_dir.mkdir(parents=True, exist_ok=True)
        state = self._initial_state()
        history: list[TCROCandidateEvaluation] = []
        telemetry_history: list[dict[str, Any]] = []
        best: TCROCandidateEvaluation | None = None
        logger.info(
            "Starting TCRO search: steps=%d samples_per_step=%d concurrency=%d template=%s out=%s",
            self._steps,
            self._samples_per_step,
            self._concurrency,
            state.template_name,
            self._out_dir,
        )

        for step in range(self._steps):
            logger.info(
                "TCRO step %d/%d: sampling %d candidates temperature=%.4f",
                step + 1,
                self._steps,
                self._samples_per_step,
                state.temperature,
            )
            evaluated = self._evaluate_step(state, step)
            history.extend(evaluated)
            step_best = min(evaluated, key=lambda item: item.weighted_score)
            if best is None or step_best.weighted_score < best.weighted_score:
                best = step_best
            pseudo_gradient = self._apply_pseudo_gradient(state, step_best)
            step_summary = self._persist_step(step, state, evaluated, step_best, pseudo_gradient)
            telemetry_history.append(step_summary)
            logger.info(
                "TCRO step %d/%d complete: best_sample=%d best_score=%.4f feasible=%s cache_hits=%d next_temperature=%.4f",
                step + 1,
                self._steps,
                step_best.sample,
                step_best.weighted_score,
                step_best.feasible,
                sum(1 for item in evaluated if item.cache_hit),
                state.temperature,
            )
            if (step + 1) % self._config.checkpoint_interval == 0:
                dump_json(self._out_dir / "supernet_state.json", state.model_dump(mode="json"))

        if best is None:
            raise RuntimeError("TCRO did not evaluate any candidate")
        dump_json(self._out_dir / "supernet_state.json", state.model_dump(mode="json"))
        dump_json(self._out_dir / "telemetry_history.json", {"steps": telemetry_history})
        self._persist_final(history, best, state)
        logger.info(
            "TCRO search finished: evaluations=%d best_score=%.4f feasible=%s final_temperature=%.4f",
            len(history),
            best.weighted_score,
            best.feasible,
            state.temperature,
        )
        return TCROSearchResult(history=history, best=best, final_state=state)

    def _initial_state(self) -> SupernetState:
        if not self._space.templates:
            raise ValueError("search space must contain at least one template")
        base = chromosome_from_template(self._space.templates[0])
        pools = infer_type_pools(self._space, self._library.node_types, self._library.link_types)
        intra_links = link_types_for_scope(self._library, "intra")
        inter_links = link_types_for_scope(self._library, "inter")
        racks: list[RackSupernetState] = []
        for rack in base.racks:
            racks.append(
                RackSupernetState(
                    rack_id=rack.rack_id,
                    role=rack.role,
                    optional=rack.optional,
                    active_alpha=self._initial_active_alpha(rack),
                    count_alpha={
                        "gpu": float(rack.gpu_count),
                        "cpu": float(rack.cpu_count),
                        "memory": float(rack.memory_pool_count),
                        "switch": float(rack.switch_count),
                    },
                    link_alpha={
                        "endpoint": float(rack.endpoint_link_qty),
                        "gpu": float(rack.gpu_link_qty or rack.endpoint_link_qty),
                        "cpu": float(rack.cpu_link_qty or rack.endpoint_link_qty),
                        "memory": float(rack.memory_link_qty),
                    },
                    type_logits={
                        "gpu": _initial_logits(pools.gpu, rack.gpu_type),
                        "cpu": _initial_logits(pools.cpu, rack.cpu_type),
                        "memory": _initial_logits(pools.memory, rack.memory_pool_type),
                        "switch": _initial_logits(pools.switch, rack.switch_type),
                    },
                    link_type_logits={
                        "endpoint": _initial_logits(
                            intra_links,
                            _scoped_link_preferred(self._library, rack.endpoint_link_type, "intra"),
                        ),
                        "gpu": _initial_logits(
                            intra_links,
                            _scoped_link_preferred(
                                self._library,
                                rack.gpu_link_type or rack.endpoint_link_type,
                                "intra",
                            ),
                        ),
                        "cpu": _initial_logits(
                            intra_links,
                            _scoped_link_preferred(
                                self._library,
                                rack.cpu_link_type or rack.endpoint_link_type,
                                "intra",
                            ),
                        ),
                        "memory": _initial_logits(
                            intra_links,
                            _scoped_link_preferred(
                                self._library,
                                rack.memory_link_type or rack.endpoint_link_type,
                                "intra",
                            ),
                        ),
                    },
                )
            )
        return SupernetState(
            template_name=base.template_name,
            base_chromosome=base.to_dict(),
            racks=racks,
            inter_rack_alpha=float(base.inter_rack_link_qty),
            inter_rack_mode_logits=_initial_logits(["none", "ring", "fully_connected"], base.inter_rack),
            inter_rack_link_type_logits=_initial_logits(
                inter_links,
                _scoped_link_preferred(self._library, base.inter_rack_link_type, "inter"),
            ),
            temperature=self._config.initial_temperature,
            step=0,
            seed=self._space.seed,
        )

    def _evaluate_step(
        self,
        state: SupernetState,
        step: int,
    ) -> list[TCROCandidateEvaluation]:
        candidates = [self._sample_candidate(state, step, sample) for sample in range(self._samples_per_step)]
        if self._concurrency <= 1 or len(candidates) <= 1:
            return [
                self._evaluate_candidate(
                    candidate,
                    self._out_dir / f"step_{step:03d}" / f"sample_{candidate.sample:03d}",
                )
                for candidate in candidates
            ]

        result: list[TCROCandidateEvaluation | None] = [None] * len(candidates)
        with ThreadPoolExecutor(max_workers=min(self._concurrency, len(candidates))) as executor:
            futures = {
                executor.submit(
                    self._evaluate_candidate,
                    candidate,
                    self._out_dir / f"step_{step:03d}" / f"sample_{candidate.sample:03d}",
                ): candidate.sample
                for candidate in candidates
            }
            for future in as_completed(futures):
                sample = futures[future]
                result[sample] = future.result()
        return [item for item in result if item is not None]

    def _sample_candidate(self, state: SupernetState, step: int, sample: int) -> TCROCandidate:
        rng = random.Random(state.seed + step * 100_003 + sample * 9_973)
        chromosome = Chromosome.model_validate(state.base_chromosome).model_copy(deep=True)
        rack_state_by_id = {rack.rack_id: rack for rack in state.racks}
        for rack in chromosome.racks:
            rack_state = rack_state_by_id[rack.rack_id]
            self._lower_rack(rack, rack_state, state.temperature, rng)

        inter_qty = quantize_alpha(
            state.inter_rack_alpha,
            threshold=self._config.link_prune_threshold,
            minimum=self._space.mutation.min_inter_rack_link_qty,
            maximum=self._space.mutation.max_inter_rack_link_qty,
        )
        if inter_qty <= 0:
            chromosome.inter_rack = "none"
            chromosome.inter_rack_link_qty = 1
        else:
            chromosome.inter_rack = _select_from_logits(
                state.inter_rack_mode_logits,
                fallback=chromosome.inter_rack,
                temperature=state.temperature,
                rng=rng,
            )
            if chromosome.inter_rack == "none":
                chromosome.inter_rack = "ring"
            chromosome.inter_rack_link_qty = inter_qty
        chromosome.inter_rack_link_type = _select_from_logits(
            state.inter_rack_link_type_logits,
            fallback=chromosome.inter_rack_link_type,
            temperature=state.temperature,
            rng=rng,
        )
        return TCROCandidate(step=step, sample=sample, chromosome=chromosome, source_temperature=state.temperature)

    def _lower_rack(
        self,
        rack: RackGene,
        rack_state: RackSupernetState,
        temperature: float,
        rng: random.Random,
    ) -> None:
        rack.optional = rack_state.optional
        rack.activation_alpha = rack_state.active_alpha
        rack.active = self._sample_rack_active(rack_state, temperature, rng)
        if not rack.active:
            rack.gpu_count = 0
            rack.cpu_count = 0
            rack.memory_pool_count = 0
            rack.switch_count = 0
            return

        rack.gpu_count = _quantize_count(
            rack_state.count_alpha.get("gpu", 0.0),
            self._count_min(rack, "gpu"),
            self._count_max(rack, "gpu"),
        )
        rack.cpu_count = _quantize_count(
            rack_state.count_alpha.get("cpu", 0.0),
            self._count_min(rack, "cpu"),
            self._count_max(rack, "cpu"),
        )
        rack.memory_pool_count = _quantize_count(
            rack_state.count_alpha.get("memory", 0.0),
            self._count_min(rack, "memory"),
            self._count_max(rack, "memory"),
        )
        rack.switch_count = _quantize_count(
            rack_state.count_alpha.get("switch", 1.0),
            1 if rack.fabric == "switch" else 0,
            self._count_max(rack, "switch"),
        )

        if rack.role == "memory":
            rack.gpu_count = 0
            rack.cpu_count = 0
        if rack.role == "compute":
            rack.memory_pool_count = 0

        if rack.gpu_count > 0:
            rack.gpu_type = _select_from_logits(rack_state.type_logits.get("gpu", {}), rack.gpu_type, temperature, rng)
        if rack.cpu_count > 0:
            rack.cpu_type = _select_from_logits(rack_state.type_logits.get("cpu", {}), rack.cpu_type, temperature, rng)
        if rack.memory_pool_count > 0:
            rack.memory_pool_type = _select_from_logits(
                rack_state.type_logits.get("memory", {}),
                rack.memory_pool_type,
                temperature,
                rng,
            )
        if rack.switch_count > 0:
            rack.switch_type = _select_from_logits(
                rack_state.type_logits.get("switch", {}),
                rack.switch_type,
                temperature,
                rng,
            )

        rack.endpoint_link_type = _select_from_logits(
            rack_state.link_type_logits.get("endpoint", {}),
            rack.endpoint_link_type,
            temperature,
            rng,
        )
        rack.gpu_link_type = _select_from_logits(
            rack_state.link_type_logits.get("gpu", {}),
            rack.gpu_link_type or rack.endpoint_link_type,
            temperature,
            rng,
        )
        rack.cpu_link_type = _select_from_logits(
            rack_state.link_type_logits.get("cpu", {}),
            rack.cpu_link_type or rack.endpoint_link_type,
            temperature,
            rng,
        )
        rack.memory_link_type = _select_from_logits(
            rack_state.link_type_logits.get("memory", {}),
            rack.memory_link_type or rack.endpoint_link_type,
            temperature,
            rng,
        )
        rack.endpoint_link_qty = self._quantize_endpoint_alpha(rack_state.link_alpha.get("endpoint", 1.0))
        rack.gpu_link_qty = self._quantize_endpoint_alpha(rack_state.link_alpha.get("gpu", rack.endpoint_link_qty))
        rack.cpu_link_qty = self._quantize_endpoint_alpha(rack_state.link_alpha.get("cpu", rack.endpoint_link_qty))
        rack.memory_link_qty = self._quantize_endpoint_alpha(rack_state.link_alpha.get("memory", 1.0))

    def _evaluate_candidate(self, candidate: TCROCandidate, candidate_dir: Path) -> TCROCandidateEvaluation:
        candidate_dir.mkdir(parents=True, exist_ok=True)
        dump_json(candidate_dir / "candidate.json", candidate.to_dict())
        repair = self._repairer.repair_and_validate(candidate.chromosome)
        chromosome = repair.chromosome
        candidate = TCROCandidate(
            step=candidate.step,
            sample=candidate.sample,
            chromosome=chromosome,
            source_temperature=candidate.source_temperature,
        )
        signature = chromosome.signature()
        cached = self._cached_evaluation(signature)
        exported: ExportedHardware | None = None
        try:
            exported = self._exporter.export(chromosome, iteration=candidate.step)
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
            "TCRO candidate step=%03d sample=%03d prepared: ranks=%s racks=%d cost=%.2f power=%.2f feasible=%s",
            candidate.step,
            candidate.sample,
            exported.rank_count if exported is not None else "n/a",
            _active_rack_count(chromosome),
            repair.estimated_cost,
            repair.estimated_power_watts,
            repair.feasible and exported is not None,
        )

        if cached is not None:
            copied = TCROCandidateEvaluation(
                step=candidate.step,
                sample=candidate.sample,
                candidate=candidate,
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
                "TCRO candidate step=%03d sample=%03d cache hit: score=%.4f feasible=%s",
                candidate.step,
                candidate.sample,
                copied.weighted_score,
                copied.feasible,
            )
            return copied

        if not repair.feasible or exported is None:
            evaluation = self._penalty_evaluation(candidate, repair, exported, repair.messages)
            dump_json(candidate_dir / "score.json", evaluation.to_summary())
            self._store_cached_evaluation(signature, evaluation)
            logger.info(
                "TCRO candidate step=%03d sample=%03d skipped with penalty: score=%.4f reason=%s",
                candidate.step,
                candidate.sample,
                evaluation.weighted_score,
                _message_preview(evaluation.messages),
            )
            return evaluation

        feedback: ParsedPipelineFeedback | None = None
        messages = list(repair.messages)
        feasible = True
        logger.info(
            "TCRO candidate step=%03d sample=%03d running mapper/simulator: topology=%s",
            candidate.step,
            candidate.sample,
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
        evaluation = TCROCandidateEvaluation(
            step=candidate.step,
            sample=candidate.sample,
            candidate=candidate,
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
            "TCRO candidate step=%03d sample=%03d done: score=%.4f feasible=%s makespan_us=%.3f max_link_util=%.3f queue_ns=%.3f remote_ns=%.3f",
            candidate.step,
            candidate.sample,
            evaluation.weighted_score,
            evaluation.feasible,
            evaluation.objectives[0],
            evaluation.objectives[3],
            evaluation.objectives[4],
            evaluation.objectives[5],
        )
        return evaluation

    def _apply_pseudo_gradient(
        self,
        state: SupernetState,
        best: TCROCandidateEvaluation,
    ) -> dict[str, Any]:
        feedback = best.feedback
        repair = best.repair
        compute_util = _average_compute_utilization(feedback)
        bubble = max(0.0, 1.0 - compute_util)
        network_util = feedback.max_link_utilization if feedback is not None else 0.0
        queue_pressure = min(1.0, (feedback.max_queue_delay_ns / 1_000_000.0) if feedback else 0.0)
        remote_pressure = min(1.0, (feedback.remote_memory_contention_ns / 1_000_000.0) if feedback else 0.0)
        cost_pressure = _positive_ratio(repair.estimated_cost, self._space.limits.max_total_cost)
        power_pressure = _positive_ratio(repair.estimated_power_watts, self._space.limits.max_peak_power_watts)
        constraint_pressure = min(2.0, cost_pressure + power_pressure + (0.5 if not best.feasible else 0.0))
        lr = self._config.learning_rate

        gradient: dict[str, Any] = {
            "compute_utilization": compute_util,
            "bubble_proxy": bubble,
            "network_utilization": network_util,
            "queue_pressure": queue_pressure,
            "remote_memory_pressure": remote_pressure,
            "constraint_pressure": constraint_pressure,
            "rack_updates": {},
        }

        top_link = feedback.link_stats[0] if feedback and feedback.link_stats else {}
        top_domain = str(top_link.get("stats_domain") or top_link.get("domain") or "")

        for rack_state in state.racks:
            rack_update: dict[str, float] = {}
            domain_hit = rack_state.rack_id in top_domain
            activation_delta = 0.0
            if compute_util > 0.85:
                rack_update["compute_pressure"] = compute_util - 0.85
                self._nudge_high_perf(rack_state.type_logits.get("gpu", {}), "peak")
                self._nudge_high_perf(rack_state.type_logits.get("cpu", {}), "peak")
                if rack_state.role in {"compute", "hybrid"}:
                    delta = lr * 0.25 * (compute_util - 0.85)
                    rack_state.count_alpha["gpu"] = rack_state.count_alpha.get("gpu", 0.0) + delta
                    activation_delta += delta
            if remote_pressure > 0 and rack_state.role in {"memory", "hybrid"}:
                delta = lr * remote_pressure
                rack_state.count_alpha["memory"] = rack_state.count_alpha.get("memory", 0.0) + delta
                rack_state.link_alpha["memory"] = rack_state.link_alpha.get("memory", 1.0) + delta
                self._nudge_high_perf(rack_state.type_logits.get("memory", {}), "memory")
                rack_update["remote_memory_delta"] = delta
                activation_delta += delta
            if network_util > 0.75 or queue_pressure > 0:
                delta = lr * max(network_util - 0.75, queue_pressure)
                if top_domain.startswith("cluster:") or not top_domain:
                    state.inter_rack_alpha += delta * 2.0
                    state.inter_rack_mode_logits["fully_connected"] = state.inter_rack_mode_logits.get("fully_connected", 0.0) + delta
                    rack_update["cluster_link_delta"] = delta * 2.0
                    activation_delta += delta * 0.5
                if domain_hit:
                    for key in ("endpoint", "gpu", "cpu", "memory"):
                        rack_state.link_alpha[key] = rack_state.link_alpha.get(key, 1.0) + delta
                    rack_update["rack_link_delta"] = delta
                    activation_delta += delta * 0.5
            elif network_util < 0.10 and state.inter_rack_alpha > 0:
                delta = lr * 0.10
                state.inter_rack_alpha = max(0.0, state.inter_rack_alpha - delta)
                rack_update["idle_inter_rack_delta"] = -delta

            if constraint_pressure > 0:
                delta = lr * self._config.constraint_penalty_weight * constraint_pressure
                for key in ("gpu", "cpu", "memory"):
                    rack_state.count_alpha[key] = max(0.0, rack_state.count_alpha.get(key, 0.0) - delta * 0.15)
                for key in ("endpoint", "gpu", "cpu", "memory"):
                    rack_state.link_alpha[key] = max(0.0, rack_state.link_alpha.get(key, 1.0) - delta * 0.10)
                self._nudge_low_cost_power(rack_state.type_logits.get("gpu", {}))
                self._nudge_low_cost_power(rack_state.type_logits.get("cpu", {}))
                self._nudge_low_cost_power(rack_state.type_logits.get("memory", {}))
                rack_update["constraint_delta"] = -delta
                activation_delta -= delta * 0.25
            if rack_state.optional:
                rack_state.active_alpha = max(0.0, rack_state.active_alpha + activation_delta)
                rack_update["active_alpha"] = rack_state.active_alpha
            gradient["rack_updates"][rack_state.rack_id] = rack_update

        if constraint_pressure > 0:
            state.inter_rack_alpha = max(0.0, state.inter_rack_alpha - lr * constraint_pressure * 0.25)
            self._nudge_low_cost_link(state.inter_rack_link_type_logits)

        self._apply_noise_and_clamp(state)
        state.temperature = max(self._config.min_temperature, state.temperature * self._config.temperature_decay)
        state.step += 1
        gradient["new_temperature"] = state.temperature
        return gradient

    def _nudge_high_perf(self, logits: dict[str, float], metric: str) -> None:
        if not logits:
            return
        if metric == "memory":
            score = lambda item: (self._library.node_types[item].memory_bw_gbps or 0.0) + (
                self._library.node_types[item].capacity_gb or 0.0
            )
        else:
            score = lambda item: _node_peak_tflops(self._library.node_types[item])
        winner = max(logits, key=score)
        for key in logits:
            logits[key] += self._config.learning_rate * (0.25 if key == winner else -0.05)

    def _nudge_low_cost_power(self, logits: dict[str, float]) -> None:
        if not logits:
            return
        winner = min(
            logits,
            key=lambda item: self._library.node_types[item].cost_unit + self._library.node_types[item].tdp_watts,
        )
        for key in logits:
            logits[key] += self._config.learning_rate * (0.25 if key == winner else -0.05)

    def _nudge_low_cost_link(self, logits: dict[str, float]) -> None:
        if not logits:
            return
        winner = min(logits, key=lambda item: self._library.link_types[item].cost_unit)
        for key in logits:
            logits[key] += self._config.learning_rate * (0.25 if key == winner else -0.05)

    def _apply_noise_and_clamp(self, state: SupernetState) -> None:
        rng = random.Random(state.seed + state.step * 65_537)
        noise_sigma = state.temperature * self._config.noise_scale
        for rack in state.racks:
            for logits_by_type in rack.type_logits.values():
                _add_logit_noise(logits_by_type, rng, noise_sigma)
            for logits_by_link in rack.link_type_logits.values():
                _add_logit_noise(logits_by_link, rng, noise_sigma)
            for key in list(rack.count_alpha):
                rack.count_alpha[key] = max(0.0, rack.count_alpha[key])
            for key in list(rack.link_alpha):
                upper = self._space.mutation.max_endpoint_link_qty
                rack.link_alpha[key] = max(0.0, min(float(upper), rack.link_alpha[key]))
            rack.active_alpha = max(0.0, min(2.0, rack.active_alpha))
        _add_logit_noise(state.inter_rack_mode_logits, rng, noise_sigma)
        _add_logit_noise(state.inter_rack_link_type_logits, rng, noise_sigma)
        state.inter_rack_alpha = max(0.0, min(float(self._space.mutation.max_inter_rack_link_qty), state.inter_rack_alpha))

    def _persist_step(
        self,
        step: int,
        state: SupernetState,
        evaluated: list[TCROCandidateEvaluation],
        best: TCROCandidateEvaluation,
        pseudo_gradient: dict[str, Any],
    ) -> dict[str, Any]:
        summary = {
            "step": step,
            "temperature": state.temperature,
            "best_sample": best.sample,
            "best_score": best.weighted_score,
            "best_feasible": best.feasible,
            "pseudo_gradient": pseudo_gradient,
            "samples": [item.to_summary() for item in evaluated],
        }
        dump_json(self._out_dir / f"step_{step:03d}" / "step_summary.json", summary)
        return summary

    def _persist_final(
        self,
        history: list[TCROCandidateEvaluation],
        best: TCROCandidateEvaluation,
        state: SupernetState,
    ) -> None:
        dump_json(
            self._out_dir / "tcro_summary.json",
            {
                "steps": self._steps,
                "samples_per_step": self._samples_per_step,
                "concurrency": self._concurrency,
                "evaluations": len(history),
                "best": best.to_summary(),
                "final_temperature": state.temperature,
            },
        )
        if best.exported is not None:
            dump_json(self._out_dir / "best_hardware_topology.json", best.exported.hardware_topology)
            dump_json(self._out_dir / "best_proposal.json", best.exported.proposal.to_dict())

    def _penalty_evaluation(
        self,
        candidate: TCROCandidate,
        repair: RepairReport,
        exported: ExportedHardware | None,
        messages: list[str],
    ) -> TCROCandidateEvaluation:
        objectives = (
            1_000_000_000.0 + repair.penalty,
            repair.estimated_cost,
            repair.estimated_power_watts,
            1_000_000.0,
            1_000_000_000.0,
            1_000_000_000.0,
        )
        return TCROCandidateEvaluation(
            step=candidate.step,
            sample=candidate.sample,
            candidate=candidate,
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

    def _cached_evaluation(self, signature: str) -> TCROCandidateEvaluation | None:
        with self._cache_lock:
            return self._cache.get(signature)

    def _store_cached_evaluation(self, signature: str, evaluation: TCROCandidateEvaluation) -> None:
        with self._cache_lock:
            self._cache.setdefault(signature, evaluation)

    def _quantize_endpoint_alpha(self, value: float) -> int:
        qty = quantize_alpha(
            value,
            threshold=0.0,
            minimum=self._space.mutation.min_endpoint_link_qty,
            maximum=self._space.mutation.max_endpoint_link_qty,
        )
        return max(1, qty)

    def _initial_active_alpha(self, rack: RackGene) -> float:
        if rack.activation_alpha is not None:
            return rack.activation_alpha
        if rack.optional and not rack.active:
            return self._config.latent_rack_initial_alpha
        return 1.0

    def _sample_rack_active(
        self,
        rack_state: RackSupernetState,
        temperature: float,
        rng: random.Random,
    ) -> bool:
        if not rack_state.optional:
            return True
        if temperature <= 0:
            return rack_state.active_alpha >= self._config.rack_activation_threshold
        margin = rack_state.active_alpha - self._config.rack_activation_threshold
        noise = rng.gauss(0.0, max(1e-9, temperature * 0.10))
        return margin + noise >= 0.0

    def _count_min(self, rack: RackGene, kind: str) -> int:
        settings = self._space.mutation
        if kind == "gpu":
            return settings.min_gpu_per_rack if rack.role in {"compute", "hybrid"} and rack.gpu_type else 0
        if kind == "cpu":
            return settings.min_cpu_per_rack if rack.role in {"compute", "hybrid"} and rack.cpu_type else 0
        if kind == "memory":
            return settings.min_memory_pools_per_rack if rack.role in {"memory", "hybrid"} and rack.memory_pool_type else 0
        return 1 if rack.fabric == "switch" else 0

    def _count_max(self, rack: RackGene, kind: str) -> int:
        settings = self._space.mutation
        if kind == "gpu":
            return _limit(rack.limits.max_gpu_count, settings.max_gpu_per_rack)
        if kind == "cpu":
            return _limit(rack.limits.max_cpu_count, settings.max_cpu_per_rack)
        if kind == "memory":
            return _limit(rack.limits.max_memory_pool_count, settings.max_memory_pools_per_rack)
        return _limit(rack.limits.max_switch_count, max(1, rack.switch_count))


def _initial_logits(options: list[str], preferred: str | None) -> dict[str, float]:
    logits = {option: 0.0 for option in options}
    if preferred and preferred not in logits:
        logits[preferred] = 0.0
    if preferred and preferred in logits:
        logits[preferred] = 1.0
    return logits


def _scoped_link_preferred(
    library: ComponentLibrary,
    preferred: str | None,
    scope: LinkScope,
) -> str | None:
    if preferred and link_type_allowed_for_scope(library, preferred, scope):
        return preferred
    return None


def _select_from_logits(
    logits: dict[str, float],
    fallback: str | None,
    temperature: float,
    rng: random.Random,
) -> str | None:
    if not logits:
        return fallback
    if temperature <= 0:
        return max(logits, key=logits.get)
    best_key = None
    best_score = -math.inf
    for key, logit in logits.items():
        u = min(1.0 - 1e-12, max(1e-12, rng.random()))
        gumbel = -math.log(-math.log(u))
        score = logit / max(temperature, 1e-9) + gumbel
        if score > best_score:
            best_key = key
            best_score = score
    return best_key or fallback


def _quantize_count(alpha: float, minimum: int, maximum: int) -> int:
    if maximum <= 0:
        return 0
    if maximum < minimum:
        return maximum
    return max(minimum, min(maximum, int(round(max(0.0, alpha)))))


def _node_peak_tflops(spec: NodeTypeSpec) -> float:
    return (
        spec.peak_tflops
        or spec.compute_teraflops_dense
        or spec.compute_teraflops_sparse
        or 0.0
    )


def _average_compute_utilization(feedback: ParsedPipelineFeedback | None) -> float:
    if feedback is None or not feedback.simulation_feedback.compute_profile:
        return 0.0
    values = [
        item.avg_utilization
        for item in feedback.simulation_feedback.compute_profile.values()
    ]
    return sum(values) / len(values)


def _positive_ratio(value: float, limit: float) -> float:
    if limit <= 0 or value <= limit:
        return 0.0
    return min(2.0, (value - limit) / limit)


def _add_logit_noise(logits: dict[str, float], rng: random.Random, sigma: float) -> None:
    if sigma <= 0:
        return
    for key in list(logits):
        logits[key] += rng.gauss(0.0, sigma)


def _limit(local_limit: int | None, global_limit: int) -> int:
    if local_limit is None:
        return global_limit
    return min(local_limit, global_limit)


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
