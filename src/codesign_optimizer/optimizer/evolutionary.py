from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import math
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codesign_optimizer.io.jsonc import dump_json
from codesign_optimizer.models.hardware import ComponentLibrary
from codesign_optimizer.optimizer.chromosome import (
    Chromosome,
    crossover,
    initial_population,
    mutate_random,
)
from codesign_optimizer.optimizer.exporter import ExportedHardware, HardwareTopologyExporter
from codesign_optimizer.optimizer.feedback_parser import ParsedPipelineFeedback
from codesign_optimizer.optimizer.pipeline_client import PipelineClient
from codesign_optimizer.optimizer.repair import CandidateRepairer, RepairReport
from codesign_optimizer.optimizer.search_space import SearchObjectiveWeights, SearchSpace


logger = logging.getLogger(__name__)


@dataclass
class CandidateEvaluation:
    generation: int
    index: int
    chromosome: Chromosome
    feasible: bool
    messages: list[str]
    objectives: tuple[float, float, float, float, float, float]
    weighted_score: float
    repair: RepairReport
    exported: ExportedHardware | None = None
    feedback: ParsedPipelineFeedback | None = None
    rank: int = 0
    crowding_distance: float = 0.0
    cache_hit: bool = False

    def to_summary(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "index": self.index,
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
            "rank": self.rank,
            "crowding_distance": _json_number(self.crowding_distance),
            "cache_hit": self.cache_hit,
            "chromosome": self.chromosome.to_dict(),
        }


@dataclass(frozen=True)
class SearchResult:
    history: list[CandidateEvaluation]
    pareto_frontier: list[CandidateEvaluation]
    best: CandidateEvaluation


class HeuristicSearchRunner:
    def __init__(
        self,
        *,
        component_library: ComponentLibrary,
        search_space: SearchSpace,
        pipeline_client: PipelineClient,
        workload_path: Path,
        out_dir: Path,
        population_size: int,
        generations: int,
        concurrency: int = 1,
    ) -> None:
        if search_space.mutation.search_granularity == "host":
            raise ValueError("host search_granularity is currently supported only by TG-RL/TG-RL v2")
        self._library = component_library
        self._space = search_space
        self._pipeline = pipeline_client
        self._workload_path = workload_path
        self._out_dir = out_dir
        self._population_size = population_size
        self._generations = generations
        self._concurrency = max(1, concurrency)
        self._rng = random.Random(search_space.seed)
        self._exporter = HardwareTopologyExporter(component_library)
        self._repairer = CandidateRepairer(component_library, search_space)
        self._cache: dict[str, CandidateEvaluation] = {}
        self._cache_lock = threading.Lock()

    def run(self) -> SearchResult:
        self._out_dir.mkdir(parents=True, exist_ok=True)
        population = initial_population(self._space, self._population_size, self._rng)
        history: list[CandidateEvaluation] = []
        logger.info(
            "Starting heuristic search: generations=%d population=%d concurrency=%d templates=%d out=%s",
            self._generations,
            self._population_size,
            self._concurrency,
            len(self._space.templates),
            self._out_dir,
        )

        for generation in range(self._generations):
            logger.info(
                "Generation %d/%d: evaluating %d candidates",
                generation + 1,
                self._generations,
                len(population),
            )
            evaluated = self._evaluate_generation(population, generation)
            rank_fronts(evaluated)
            history.extend(evaluated)
            self._persist_generation(generation, evaluated)
            generation_best = min(evaluated, key=lambda item: item.weighted_score)
            logger.info(
                "Generation %d/%d complete: best_score=%.4f feasible=%s feasible_count=%d cache_hits=%d",
                generation + 1,
                self._generations,
                generation_best.weighted_score,
                generation_best.feasible,
                sum(1 for item in evaluated if item.feasible),
                sum(1 for item in evaluated if item.cache_hit),
            )
            selected = select_next_population(evaluated, self._population_size)
            if generation + 1 < self._generations:
                population = self._make_next_population(selected, self._population_size)
                logger.info(
                    "Generation %d/%d: produced next population with %d candidates",
                    generation + 2,
                    self._generations,
                    len(population),
                )

        rank_fronts(history)
        pareto = [item for item in history if item.rank == 0 and item.feasible]
        if not pareto:
            pareto = sorted(history, key=lambda item: item.weighted_score)[: max(1, self._population_size // 4)]
        best = min(pareto, key=lambda item: item.weighted_score)
        self._persist_final(history, pareto, best)
        logger.info(
            "Heuristic search finished: evaluations=%d pareto=%d best_score=%.4f feasible=%s",
            len(history),
            len(pareto),
            best.weighted_score,
            best.feasible,
        )
        return SearchResult(history=history, pareto_frontier=pareto, best=best)

    def _evaluate_generation(
        self,
        population: list[Chromosome],
        generation: int,
    ) -> list[CandidateEvaluation]:
        if self._concurrency <= 1 or len(population) <= 1:
            result: list[CandidateEvaluation] = []
            for index, chromosome in enumerate(population):
                candidate_dir = self._out_dir / f"iter_{generation:03d}" / f"candidate_{index:03d}"
                result.append(self._evaluate_candidate(chromosome, generation, index, candidate_dir))
            return result

        result: list[CandidateEvaluation | None] = [None] * len(population)
        max_workers = min(self._concurrency, len(population))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for index, chromosome in enumerate(population):
                candidate_dir = self._out_dir / f"iter_{generation:03d}" / f"candidate_{index:03d}"
                future = executor.submit(
                    self._evaluate_candidate,
                    chromosome,
                    generation,
                    index,
                    candidate_dir,
                )
                futures[future] = index
            for future in as_completed(futures):
                index = futures[future]
                result[index] = future.result()

        return [item for item in result if item is not None]

    def _evaluate_candidate(
        self,
        chromosome: Chromosome,
        generation: int,
        index: int,
        candidate_dir: Path,
    ) -> CandidateEvaluation:
        candidate_dir.mkdir(parents=True, exist_ok=True)
        repair = self._repairer.repair_and_validate(chromosome)
        chromosome = repair.chromosome
        signature = chromosome.signature()
        exported: ExportedHardware | None = None
        try:
            exported = self._exporter.export(chromosome, iteration=generation)
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
            "Candidate gen=%03d idx=%03d prepared: template=%s ranks=%s racks=%d cost=%.2f power=%.2f feasible=%s",
            generation,
            index,
            chromosome.template_name,
            exported.rank_count if exported is not None else "n/a",
            _active_rack_count(chromosome),
            repair.estimated_cost,
            repair.estimated_power_watts,
            repair.feasible and exported is not None,
        )

        cached = self._cached_evaluation(signature)
        if cached is not None:
            copied = CandidateEvaluation(
                generation=generation,
                index=index,
                chromosome=chromosome,
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
                "Candidate gen=%03d idx=%03d cache hit: score=%.4f feasible=%s",
                generation,
                index,
                copied.weighted_score,
                copied.feasible,
            )
            return copied

        if not repair.feasible or exported is None:
            evaluation = self._penalty_evaluation(
                chromosome,
                generation,
                index,
                repair,
                exported,
                repair.messages,
            )
            dump_json(candidate_dir / "score.json", evaluation.to_summary())
            self._store_cached_evaluation(signature, evaluation)
            logger.info(
                "Candidate gen=%03d idx=%03d skipped with penalty: score=%.4f reason=%s",
                generation,
                index,
                evaluation.weighted_score,
                _message_preview(evaluation.messages),
            )
            return evaluation

        feedback: ParsedPipelineFeedback | None = None
        messages = list(repair.messages)
        feasible = True
        logger.info(
            "Candidate gen=%03d idx=%03d running mapper/simulator: topology=%s",
            generation,
            index,
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
        evaluation = CandidateEvaluation(
            generation=generation,
            index=index,
            chromosome=chromosome,
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
            "Candidate gen=%03d idx=%03d done: score=%.4f feasible=%s makespan_us=%.3f max_link_util=%.3f queue_ns=%.3f remote_ns=%.3f",
            generation,
            index,
            evaluation.weighted_score,
            evaluation.feasible,
            evaluation.objectives[0],
            evaluation.objectives[3],
            evaluation.objectives[4],
            evaluation.objectives[5],
        )
        return evaluation

    def _cached_evaluation(self, signature: str) -> CandidateEvaluation | None:
        with self._cache_lock:
            return self._cache.get(signature)

    def _store_cached_evaluation(self, signature: str, evaluation: CandidateEvaluation) -> None:
        with self._cache_lock:
            self._cache.setdefault(signature, evaluation)

    def _penalty_evaluation(
        self,
        chromosome: Chromosome,
        generation: int,
        index: int,
        repair: RepairReport,
        exported: ExportedHardware | None,
        messages: list[str],
    ) -> CandidateEvaluation:
        objectives = (
            1_000_000_000.0 + repair.penalty,
            repair.estimated_cost,
            repair.estimated_power_watts,
            1_000_000.0,
            1_000_000_000.0,
            1_000_000_000.0,
        )
        return CandidateEvaluation(
            generation=generation,
            index=index,
            chromosome=chromosome,
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

    def _make_next_population(
        self,
        selected: list[CandidateEvaluation],
        population_size: int,
    ) -> list[Chromosome]:
        if not selected:
            return initial_population(self._space, population_size, self._rng)
        next_population = [item.chromosome.model_copy(deep=True) for item in selected[: self._elite_count()]]
        while len(next_population) < population_size:
            parent_a = self._tournament(selected)
            parent_b = self._tournament(selected)
            child = crossover(parent_a.chromosome, parent_b.chromosome, self._rng)
            if self._rng.random() < self._space.mutation.bottleneck_mutation_rate:
                child = self._mutate_from_bottleneck(child, parent_a)
            if self._rng.random() < self._space.mutation.mutation_rate:
                child = mutate_random(child, self._space, self._rng)
            next_population.append(child)
        return next_population[:population_size]

    def _mutate_from_bottleneck(
        self,
        chromosome: Chromosome,
        parent: CandidateEvaluation,
    ) -> Chromosome:
        child = chromosome.model_copy(deep=True)
        feedback = parent.feedback
        if feedback is None:
            return mutate_random(child, self._space, self._rng)
        top_link = feedback.link_stats[0] if feedback.link_stats else {}
        top_domain = str(top_link.get("stats_domain") or top_link.get("domain") or "")
        if feedback.remote_memory_contention_ns > 0:
            memory_racks = [rack for rack in child.racks if rack.role in {"memory", "hybrid"}]
            for rack in memory_racks:
                if rack.memory_pool_count < self._space.mutation.max_memory_pools_per_rack:
                    rack.memory_pool_count += 1
                rack.memory_link_qty = min(
                    self._space.mutation.max_intra_rack_link_qty,
                    rack.memory_link_qty + 1,
                )
                return child
        if top_domain.startswith("cluster:"):
            child.inter_rack_link_qty = min(
                self._space.mutation.max_inter_rack_link_qty,
                child.inter_rack_link_qty + 1,
            )
            if child.inter_rack == "none":
                child.inter_rack = "ring"
            return child
        for rack in child.racks:
            if rack.rack_id in top_domain:
                rack.endpoint_link_qty = min(
                    self._space.mutation.max_intra_rack_link_qty,
                    rack.endpoint_link_qty + 1,
                )
                return child
        return mutate_random(child, self._space, self._rng)

    def _tournament(self, candidates: list[CandidateEvaluation]) -> CandidateEvaluation:
        left = self._rng.choice(candidates)
        right = self._rng.choice(candidates)
        return better_candidate(left, right)

    def _elite_count(self) -> int:
        return max(1, int(math.ceil(self._population_size * self._space.mutation.elite_fraction)))

    def _persist_generation(self, generation: int, evaluated: list[CandidateEvaluation]) -> None:
        dump_json(
            self._out_dir / f"iter_{generation:03d}" / "generation_summary.json",
            {"candidates": [item.to_summary() for item in evaluated]},
        )

    def _persist_final(
        self,
        history: list[CandidateEvaluation],
        pareto: list[CandidateEvaluation],
        best: CandidateEvaluation,
    ) -> None:
        dump_json(
            self._out_dir / "summary.json",
            {
                "generations": self._generations,
                "population": self._population_size,
                "concurrency": self._concurrency,
                "evaluations": len(history),
                "feasible_evaluations": sum(1 for item in history if item.feasible),
                "best": best.to_summary(),
            },
        )
        dump_json(
            self._out_dir / "pareto_frontier.json",
            {"candidates": [item.to_summary() for item in pareto]},
        )
        if best.exported is not None:
            dump_json(self._out_dir / "best_hardware_topology.json", best.exported.hardware_topology)
            dump_json(self._out_dir / "best_proposal.json", best.exported.proposal.to_dict())


def dominates(left: CandidateEvaluation, right: CandidateEvaluation) -> bool:
    if left.feasible and not right.feasible:
        return True
    if right.feasible and not left.feasible:
        return False
    if not left.feasible and not right.feasible:
        return left.weighted_score < right.weighted_score
    return all(a <= b for a, b in zip(left.objectives, right.objectives, strict=True)) and any(
        a < b for a, b in zip(left.objectives, right.objectives, strict=True)
    )


def rank_fronts(candidates: list[CandidateEvaluation]) -> list[list[CandidateEvaluation]]:
    dominated_by_count = {id(item): 0 for item in candidates}
    dominates_map: dict[int, list[CandidateEvaluation]] = {id(item): [] for item in candidates}
    fronts: list[list[CandidateEvaluation]] = [[]]

    for left in candidates:
        for right in candidates:
            if left is right:
                continue
            if dominates(left, right):
                dominates_map[id(left)].append(right)
            elif dominates(right, left):
                dominated_by_count[id(left)] += 1
        if dominated_by_count[id(left)] == 0:
            left.rank = 0
            fronts[0].append(left)

    idx = 0
    while idx < len(fronts) and fronts[idx]:
        next_front: list[CandidateEvaluation] = []
        for item in fronts[idx]:
            for dominated in dominates_map[id(item)]:
                dominated_by_count[id(dominated)] -= 1
                if dominated_by_count[id(dominated)] == 0:
                    dominated.rank = idx + 1
                    next_front.append(dominated)
        idx += 1
        if next_front:
            fronts.append(next_front)

    for front in fronts:
        assign_crowding_distance(front)
    return fronts


def assign_crowding_distance(front: list[CandidateEvaluation]) -> None:
    if not front:
        return
    for item in front:
        item.crowding_distance = 0.0
    if len(front) <= 2:
        for item in front:
            item.crowding_distance = float("inf")
        return
    objective_count = len(front[0].objectives)
    for objective_idx in range(objective_count):
        ordered = sorted(front, key=lambda item: item.objectives[objective_idx])
        ordered[0].crowding_distance = float("inf")
        ordered[-1].crowding_distance = float("inf")
        min_value = ordered[0].objectives[objective_idx]
        max_value = ordered[-1].objectives[objective_idx]
        if max_value == min_value:
            continue
        for idx in range(1, len(ordered) - 1):
            if math.isinf(ordered[idx].crowding_distance):
                continue
            ordered[idx].crowding_distance += (
                ordered[idx + 1].objectives[objective_idx]
                - ordered[idx - 1].objectives[objective_idx]
            ) / (max_value - min_value)


def select_next_population(
    candidates: list[CandidateEvaluation],
    population_size: int,
) -> list[CandidateEvaluation]:
    fronts = rank_fronts(candidates)
    selected: list[CandidateEvaluation] = []
    for front in fronts:
        if len(selected) + len(front) <= population_size:
            selected.extend(front)
        else:
            ordered = sorted(front, key=lambda item: item.crowding_distance, reverse=True)
            selected.extend(ordered[: population_size - len(selected)])
            break
    return sorted(selected, key=lambda item: (item.rank, -item.crowding_distance, item.weighted_score))


def better_candidate(left: CandidateEvaluation, right: CandidateEvaluation) -> CandidateEvaluation:
    if left.rank != right.rank:
        return left if left.rank < right.rank else right
    if left.crowding_distance != right.crowding_distance:
        return left if left.crowding_distance > right.crowding_distance else right
    return left if left.weighted_score <= right.weighted_score else right


def _feedback_to_dict(feedback: ParsedPipelineFeedback) -> dict[str, Any]:
    return {
        "simulation_feedback": feedback.simulation_feedback.model_dump(mode="json"),
        "summary": feedback.summary,
        "link_stats": feedback.link_stats,
        "domain_stats": feedback.domain_stats,
        "remote_memory_stats": feedback.remote_memory_stats,
        "scaling_report": feedback.scaling_report,
        "simulator_stdout": str(feedback.simulator_stdout_path) if feedback.simulator_stdout_path else "",
    }


def _json_number(value: float) -> float | str:
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return value


def _active_rack_count(chromosome: Chromosome) -> int:
    return sum(1 for rack in chromosome.racks if rack.active or not rack.optional)


def _message_preview(messages: list[str]) -> str:
    if not messages:
        return "none"
    text = "; ".join(messages[:2])
    if len(messages) > 2:
        text += "; ..."
    return text
