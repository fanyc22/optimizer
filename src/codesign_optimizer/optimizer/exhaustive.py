from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import itertools
import json
import logging
import threading
from pathlib import Path
from typing import Any

from codesign_optimizer.io.jsonc import dump_json
from codesign_optimizer.models.hardware import ComponentLibrary
from codesign_optimizer.optimizer.chromosome import Chromosome, RackGene, chromosome_from_template
from codesign_optimizer.optimizer.exporter import ExportedHardware, HardwareTopologyExporter
from codesign_optimizer.optimizer.feedback_parser import ParsedPipelineFeedback
from codesign_optimizer.optimizer.pipeline_client import PipelineClient
from codesign_optimizer.optimizer.repair import CandidateRepairer, RepairReport
from codesign_optimizer.optimizer.scoring import (
    ObjectiveTuple,
    objectives_to_dict,
    penalty_objectives,
    tgrl_v2_objectives,
    weighted_score_from_objectives,
)
from codesign_optimizer.optimizer.search_space import ExhaustiveSlotOption, SearchSpace


logger = logging.getLogger(__name__)


@dataclass
class ExhaustiveEvaluation:
    index: int
    chromosome: Chromosome
    feasible: bool
    messages: list[str]
    objectives: ObjectiveTuple
    weighted_score: float
    repair: RepairReport
    exported: ExportedHardware | None = None
    feedback: ParsedPipelineFeedback | None = None
    cache_hit: bool = False

    def to_summary(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "feasible": self.feasible,
            "messages": self.messages,
            "objectives": objectives_to_dict(self.objectives),
            "weighted_score": self.weighted_score,
            "cache_hit": self.cache_hit,
            "chromosome": self.chromosome.to_dict(),
        }


@dataclass(frozen=True)
class ExhaustiveSearchResult:
    history: list[ExhaustiveEvaluation]
    best: ExhaustiveEvaluation
    total_candidates: int
    unique_candidates: int


class ExhaustiveSearchRunner:
    def __init__(
        self,
        *,
        component_library: ComponentLibrary,
        search_space: SearchSpace,
        pipeline_client: PipelineClient,
        workload_path: Path,
        out_dir: Path,
        concurrency: int = 1,
        max_candidates: int | None = None,
        freeze_topology: bool = False,
    ) -> None:
        self._library = component_library
        self._space = search_space
        self._pipeline = pipeline_client
        self._workload_path = workload_path
        self._out_dir = out_dir
        self._concurrency = max(1, concurrency)
        self._max_candidates = max_candidates
        self._freeze_topology = freeze_topology
        self._exporter = HardwareTopologyExporter(component_library)
        self._repairer = CandidateRepairer(component_library, search_space)
        self._cache: dict[str, ExhaustiveEvaluation] = {}
        self._cache_lock = threading.Lock()
        validate_exhaustive_space(search_space, component_library=component_library)

    def run(self) -> ExhaustiveSearchResult:
        self._out_dir.mkdir(parents=True, exist_ok=True)
        total_candidates = count_exhaustive_candidates(self._space, freeze_topology=self._freeze_topology)
        cap = self._max_candidates or self._space.exhaustive.max_candidates
        if cap is not None and total_candidates > cap:
            raise ValueError(
                f"exhaustive space has {total_candidates} candidates, exceeding max_candidates={cap}"
            )
        chromosomes = list(iter_exhaustive_chromosomes(self._space, freeze_topology=self._freeze_topology))
        unique_candidates = len(chromosomes)
        if not chromosomes:
            raise RuntimeError("exhaustive enumeration produced no candidates")

        dump_json(
            self._out_dir / "enumeration.json",
            {
                "total_candidates": total_candidates,
                "unique_candidates": unique_candidates,
                "templates": len(self._space.templates),
                "slot_options": [
                    option.model_dump(mode="json", exclude_none=True)
                    for option in self._space.exhaustive.slot_options
                ],
                "max_candidates": cap,
                "freeze_topology": self._freeze_topology,
            },
        )
        logger.info(
            "Starting exhaustive search: total_candidates=%d unique_candidates=%d concurrency=%d out=%s",
            total_candidates,
            unique_candidates,
            self._concurrency,
            self._out_dir,
        )

        history = self._evaluate_chromosomes(chromosomes)
        best = min(history, key=lambda item: item.weighted_score)
        self._persist_final(history, best, total_candidates=total_candidates, unique_candidates=unique_candidates)
        logger.info(
            "Exhaustive search finished: evaluated=%d feasible=%d best_score=%.4f feasible_best=%s",
            len(history),
            sum(1 for item in history if item.feasible),
            best.weighted_score,
            best.feasible,
        )
        return ExhaustiveSearchResult(
            history=history,
            best=best,
            total_candidates=total_candidates,
            unique_candidates=unique_candidates,
        )

    def _evaluate_chromosomes(self, chromosomes: list[Chromosome]) -> list[ExhaustiveEvaluation]:
        if self._concurrency <= 1 or len(chromosomes) <= 1:
            return [
                self._evaluate_chromosome(
                    chromosome,
                    index,
                    self._out_dir / f"candidate_{index:06d}",
                )
                for index, chromosome in enumerate(chromosomes)
            ]

        result: list[ExhaustiveEvaluation | None] = [None] * len(chromosomes)
        max_workers = min(self._concurrency, len(chromosomes))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._evaluate_chromosome,
                    chromosome,
                    index,
                    self._out_dir / f"candidate_{index:06d}",
                ): index
                for index, chromosome in enumerate(chromosomes)
            }
            for future in as_completed(futures):
                result[futures[future]] = future.result()
        return [item for item in result if item is not None]

    def _evaluate_chromosome(
        self,
        chromosome: Chromosome,
        index: int,
        candidate_dir: Path,
    ) -> ExhaustiveEvaluation:
        candidate_dir.mkdir(parents=True, exist_ok=True)
        repair = self._repairer.repair_and_validate(chromosome)
        chromosome = repair.chromosome
        signature = chromosome.signature()
        dump_json(candidate_dir / "chromosome.json", chromosome.to_dict())

        exported: ExportedHardware | None = None
        try:
            exported = self._exporter.export(chromosome, iteration=index)
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

        cached = self._cached_evaluation(signature)
        if cached is not None:
            evaluation = ExhaustiveEvaluation(
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
            self._write_evaluation(candidate_dir, evaluation)
            return evaluation

        if not repair.feasible or exported is None:
            objectives = penalty_objectives(repair)
            evaluation = ExhaustiveEvaluation(
                index=index,
                chromosome=chromosome,
                feasible=False,
                messages=repair.messages,
                objectives=objectives,
                weighted_score=weighted_score_from_objectives(
                    objectives,
                    weights=self._space.objective_weights,
                    feasible=False,
                    penalty=repair.penalty,
                ),
                repair=repair,
                exported=exported,
            )
            self._write_evaluation(candidate_dir, evaluation)
            self._store_cached_evaluation(signature, evaluation)
            return evaluation

        feedback: ParsedPipelineFeedback | None = None
        messages = list(repair.messages)
        feasible = True
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

        objectives = tgrl_v2_objectives(repair, feedback, feasible)
        evaluation = ExhaustiveEvaluation(
            index=index,
            chromosome=chromosome,
            feasible=feasible,
            messages=messages,
            objectives=objectives,
            weighted_score=weighted_score_from_objectives(
                objectives,
                weights=self._space.objective_weights,
                feasible=feasible,
                penalty=repair.penalty,
            ),
            repair=repair,
            exported=exported,
            feedback=feedback,
        )
        self._write_evaluation(candidate_dir, evaluation)
        self._store_cached_evaluation(signature, evaluation)
        return evaluation

    def _write_evaluation(self, candidate_dir: Path, evaluation: ExhaustiveEvaluation) -> None:
        dump_json(candidate_dir / "score.json", evaluation.to_summary())
        if evaluation.feedback is not None:
            dump_json(candidate_dir / "feedback.json", _feedback_to_dict(evaluation.feedback))

    def _persist_final(
        self,
        history: list[ExhaustiveEvaluation],
        best: ExhaustiveEvaluation,
        *,
        total_candidates: int,
        unique_candidates: int,
    ) -> None:
        summary = {
            "mode": "exhaustive",
            "total_candidates": total_candidates,
            "unique_candidates": unique_candidates,
            "evaluations": len(history),
            "feasible_evaluations": sum(1 for item in history if item.feasible),
            "best_score": best.weighted_score,
            "best_feasible": best.feasible,
            "freeze_topology": self._freeze_topology,
            "best": best.to_summary(),
        }
        dump_json(self._out_dir / "exhaustive_summary.json", summary)
        with (self._out_dir / "candidate_scores.jsonl").open("w", encoding="utf-8") as handle:
            for evaluation in history:
                handle.write(json.dumps(evaluation.to_summary(), sort_keys=True, separators=(",", ":")) + "\n")
        if best.exported is not None:
            dump_json(self._out_dir / "best_hardware_topology.json", best.exported.hardware_topology)
            dump_json(self._out_dir / "best_proposal.json", best.exported.proposal.to_dict())

    def _cached_evaluation(self, signature: str) -> ExhaustiveEvaluation | None:
        with self._cache_lock:
            return self._cache.get(signature)

    def _store_cached_evaluation(self, signature: str, evaluation: ExhaustiveEvaluation) -> None:
        with self._cache_lock:
            self._cache.setdefault(signature, evaluation)


def validate_exhaustive_space(
    space: SearchSpace,
    *,
    component_library: ComponentLibrary | None = None,
) -> None:
    if not space.exhaustive.slot_options:
        raise ValueError("exhaustive.slot_options must list the finite slot choices")
    if space.rack_archetypes:
        raise ValueError("exhaustive search requires rack_archetypes to be empty")
    if space.mutation.allow_remove_initial_racks:
        raise ValueError("exhaustive search requires mutation.allow_remove_initial_racks=false")
    for template in space.templates:
        for rack in template.racks:
            if rack.optional:
                raise ValueError("exhaustive search requires fixed racks; optional racks are not supported")
    if component_library is None:
        return
    for option in space.exhaustive.slot_options:
        if option.node_type is not None and option.node_type not in component_library.node_types:
            raise ValueError(f"unknown exhaustive slot node_type: {option.node_type}")
        if option.link_type is not None and option.link_type not in component_library.link_types:
            raise ValueError(f"unknown exhaustive slot link_type: {option.link_type}")


def count_exhaustive_candidates(space: SearchSpace, *, freeze_topology: bool = False) -> int:
    total = 0
    for template in space.templates:
        count = 1
        for rack in template.racks:
            slot_count = 1
            for slot in rack.slots:
                slot_count *= len(_slot_options_for_slot(slot, space))
            count *= (
                slot_count
                * len(_choices(space.exhaustive.intra_rack_topologies, rack.intra_rack_topology, freeze_topology))
                * len(_choices(space.exhaustive.intra_rack_link_types, rack.intra_rack_link_type, freeze_topology))
                * len(_choices(space.exhaustive.intra_rack_link_qty, rack.intra_rack_link_qty, freeze_topology))
            )
        count *= (
            len(_choices(space.exhaustive.inter_rack_topologies, template.inter_rack, freeze_topology))
            * len(_choices(space.exhaustive.inter_rack_link_types, template.inter_rack_link_type, freeze_topology))
            * len(_choices(space.exhaustive.inter_rack_link_qty, template.inter_rack_link_qty, freeze_topology))
        )
        total += count
    return total


def iter_exhaustive_chromosomes(space: SearchSpace, *, freeze_topology: bool = False) -> list[Chromosome]:
    seen: set[str] = set()
    result: list[Chromosome] = []
    for template in space.templates:
        base = chromosome_from_template(template)
        rack_variants = [_rack_variants(rack, space, freeze_topology=freeze_topology) for rack in base.racks]
        inter_topologies = _choices(space.exhaustive.inter_rack_topologies, base.inter_rack, freeze_topology)
        inter_link_types = _choices(space.exhaustive.inter_rack_link_types, base.inter_rack_link_type, freeze_topology)
        inter_link_qty = _choices(space.exhaustive.inter_rack_link_qty, base.inter_rack_link_qty, freeze_topology)
        for rack_combo, inter_topology, inter_link_type, inter_qty in itertools.product(
            itertools.product(*rack_variants),
            inter_topologies,
            inter_link_types,
            inter_link_qty,
        ):
            chromosome = base.model_copy(deep=True)
            chromosome.racks = [rack.model_copy(deep=True) for rack in rack_combo]
            chromosome.inter_rack = inter_topology
            chromosome.inter_rack_link_type = inter_link_type
            chromosome.inter_rack_link_qty = inter_qty
            signature = chromosome.signature()
            if signature in seen:
                continue
            seen.add(signature)
            result.append(chromosome)
    return result


def _rack_variants(rack: RackGene, space: SearchSpace, *, freeze_topology: bool = False) -> list[RackGene]:
    slot_option_sets = [_slot_options_for_slot(slot, space) for slot in rack.slots]
    intra_topologies = _choices(space.exhaustive.intra_rack_topologies, rack.intra_rack_topology, freeze_topology)
    intra_link_types = _choices(space.exhaustive.intra_rack_link_types, rack.intra_rack_link_type, freeze_topology)
    intra_link_qty = _choices(space.exhaustive.intra_rack_link_qty, rack.intra_rack_link_qty, freeze_topology)
    variants: list[RackGene] = []
    for slot_options, topology, link_type, link_qty in itertools.product(
        itertools.product(*slot_option_sets),
        intra_topologies,
        intra_link_types,
        intra_link_qty,
    ):
        variant = rack.model_copy(deep=True)
        for slot, option in zip(variant.slots, slot_options, strict=True):
            slot.node_type = option.node_type
            if option.link_type is not None:
                slot.link_type = option.link_type
            if option.link_qty is not None:
                slot.link_qty = option.link_qty
        variant.intra_rack_topology = topology
        variant.intra_rack_link_type = link_type
        variant.intra_rack_link_qty = link_qty
        variants.append(variant)
    return variants


def _slot_options_for_slot(slot: Any, space: SearchSpace) -> list[ExhaustiveSlotOption]:
    if space.exhaustive.slot_options:
        return space.exhaustive.slot_options
    return [
        ExhaustiveSlotOption(
            node_type=slot.node_type,
            link_type=slot.link_type,
            link_qty=slot.link_qty,
        )
    ]


def _choices(values: list[Any] | None, fallback: Any, freeze_topology: bool = False) -> list[Any]:
    if freeze_topology:
        return [fallback]
    if values is None:
        return [fallback]
    return list(values)


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
