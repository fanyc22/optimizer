from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
import math
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from codesign_optimizer.io.jsonc import load_jsonc
from codesign_optimizer.models.feedback import (
    CxlMemoryPoolStats,
    GlobalMetrics,
    LinkBottleneck,
    MemoryProfile,
    NetworkProfile,
    SimulationFeedback,
    UtilizationStats,
)
from codesign_optimizer.optimizer.feedback_parser import ParsedPipelineFeedback
from codesign_optimizer.optimizer.pipeline_client import PipelineClient


_EPS = 1e-9
_FAILED_MAKESPAN_US = 1_000_000_000.0


class WorkloadSuiteItem(BaseModel):
    name: str
    path: Path
    weight: float | None = Field(default=None, gt=0)
    workload_rank_parallel: bool = False


class WorkloadSuite(BaseModel):
    name: str
    workloads: list[WorkloadSuiteItem]
    metric: Literal["makespan"] = "makespan"
    workload_concurrency: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_and_normalize(self) -> "WorkloadSuite":
        if not self.workloads:
            raise ValueError("workload suite must contain at least one workload")
        names = [item.name for item in self.workloads]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate workload names: {', '.join(duplicates)}")
        raw_weights = [1.0 if item.weight is None else float(item.weight) for item in self.workloads]
        total = sum(raw_weights)
        if total <= 0:
            raise ValueError("workload suite weights must sum to a positive value")
        for item, raw in zip(self.workloads, raw_weights, strict=True):
            item.weight = raw / total
        return self

    @property
    def signature(self) -> str:
        pieces = [self.name, self.metric, str(self.workload_concurrency)]
        for item in self.workloads:
            pieces.append(f"{item.name}:{item.path}:{item.weight:.12f}:rank_parallel={item.workload_rank_parallel}")
        return "|".join(pieces)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric": self.metric,
            "workload_concurrency": self.workload_concurrency,
            "workloads": [
                {
                    "name": item.name,
                    "path": str(item.path),
                    "weight": item.weight,
                    "workload_rank_parallel": item.workload_rank_parallel,
                }
                for item in self.workloads
            ],
            "signature": self.signature,
        }


@dataclass(frozen=True)
class WorkloadSuiteBaseline:
    suite_name: str
    makespans_us: dict[str, float]
    suite_signature: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "suite_signature": self.suite_signature,
            "makespans_us": dict(self.makespans_us),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorkloadSuiteBaseline":
        return cls(
            suite_name=str(payload.get("suite_name", "")),
            makespans_us={str(key): float(value) for key, value in payload.get("makespans_us", {}).items()},
            suite_signature=str(payload.get("suite_signature", "")),
        )


@dataclass(frozen=True)
class WorkloadRunFeedback:
    name: str
    path: Path
    weight: float
    workload_rank_parallel: bool
    out_dir: Path
    feedback: ParsedPipelineFeedback | None
    error: str = ""
    speedup: float = 1.0
    baseline_makespan_us: float | None = None

    @property
    def success(self) -> bool:
        return self.feedback is not None and bool(self.feedback.summary.get("success", False)) and not self.error

    @property
    def makespan_us(self) -> float:
        if self.feedback is None:
            return _FAILED_MAKESPAN_US
        value = float(self.feedback.makespan_us)
        return value if value > 0 else _FAILED_MAKESPAN_US

    @property
    def effective_baseline_makespan_us(self) -> float:
        value = self.baseline_makespan_us if self.baseline_makespan_us is not None else self.makespan_us
        return max(_EPS, float(value))

    @property
    def normalized_score(self) -> float:
        """Relative makespan score for this workload; lower is better."""
        return self.makespan_us / self.effective_baseline_makespan_us

    @property
    def weighted_log_score(self) -> float:
        """Contribution to log(suite_makespan_score)."""
        return self.weight * math.log(max(_EPS, self.normalized_score))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "weight": self.weight,
            "workload_rank_parallel": self.workload_rank_parallel,
            "success": self.success,
            "makespan_us": self.makespan_us,
            "score": self.makespan_us,
            "baseline_makespan_us": self.effective_baseline_makespan_us,
            "normalized_score": self.normalized_score,
            "weighted_log_score": self.weighted_log_score,
            "speedup": self.speedup,
            "error": self.error,
            "out_dir": str(self.out_dir),
            "simulator_stdout": (
                str(self.feedback.simulator_stdout_path)
                if self.feedback is not None and self.feedback.simulator_stdout_path is not None
                else ""
            ),
        }


@dataclass(frozen=True)
class MultiWorkloadFeedback:
    suite: WorkloadSuite
    runs: list[WorkloadRunFeedback]
    baseline: WorkloadSuiteBaseline
    aggregate_feedback: ParsedPipelineFeedback
    geomean_speedup: float
    min_speedup: float
    max_speedup: float
    speedup_cv: float
    suite_makespan_score: float

    @property
    def success(self) -> bool:
        return all(item.success for item in self.runs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite.to_dict(),
            "baseline": self.baseline.to_dict(),
            "workloads": [item.to_dict() for item in self.runs],
            "aggregate": {
                "success": self.success,
                "geomean_speedup": self.geomean_speedup,
                "min_speedup": self.min_speedup,
                "max_speedup": self.max_speedup,
                "speedup_cv": self.speedup_cv,
                "suite_makespan_score": self.suite_makespan_score,
                "max_link_utilization": self.aggregate_feedback.max_link_utilization,
                "max_queue_delay_ns": self.aggregate_feedback.max_queue_delay_ns,
                "remote_memory_contention_ns": self.aggregate_feedback.remote_memory_contention_ns,
                "compute_utilization": _average_compute_utilization(self.aggregate_feedback),
            },
        }


class MultiWorkloadPipelineRunner:
    def __init__(self, pipeline_client: PipelineClient, suite: WorkloadSuite) -> None:
        self._pipeline = pipeline_client
        self._suite = suite

    @property
    def suite(self) -> WorkloadSuite:
        return self._suite

    def run(
        self,
        *,
        topology_path: Path,
        out_dir: Path,
        baseline: WorkloadSuiteBaseline | None,
    ) -> MultiWorkloadFeedback:
        out_dir.mkdir(parents=True, exist_ok=True)
        runs: list[WorkloadRunFeedback | None] = [None] * len(self._suite.workloads)
        max_workers = min(max(1, self._suite.workload_concurrency), len(self._suite.workloads))
        if max_workers == 1:
            for index, workload in enumerate(self._suite.workloads):
                workload_dir = out_dir / "workloads" / _sanitize_name(workload.name)
                runs[index] = self._run_one_feedback(
                    topology_path=topology_path,
                    workload=workload,
                    workload_dir=workload_dir,
                )
            return self._aggregate(runs, baseline)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for index, workload in enumerate(self._suite.workloads):
                workload_dir = out_dir / "workloads" / _sanitize_name(workload.name)
                future = executor.submit(
                    self._run_one,
                    topology_path=topology_path,
                    workload=workload,
                    out_dir=workload_dir / "wrapper",
                )
                futures[future] = (index, workload, workload_dir)
            for future in as_completed(futures):
                index, workload, workload_dir = futures[future]
                runs[index] = self._feedback_from_future(future, workload=workload, workload_dir=workload_dir)
        return self._aggregate(runs, baseline)

    def _aggregate(
        self,
        runs: list[WorkloadRunFeedback | None],
        baseline: WorkloadSuiteBaseline | None,
    ) -> MultiWorkloadFeedback:
        complete_runs = [item for item in runs if item is not None]
        current_baseline = baseline or WorkloadSuiteBaseline(
            suite_name=self._suite.name,
            makespans_us={item.name: item.makespan_us for item in complete_runs},
            suite_signature=self._suite.signature,
        )
        speedup_runs: list[WorkloadRunFeedback] = []
        for item in complete_runs:
            baseline_makespan = max(_EPS, current_baseline.makespans_us.get(item.name, item.makespan_us))
            speedup = baseline_makespan / max(_EPS, item.makespan_us)
            speedup_runs.append(
                WorkloadRunFeedback(
                    name=item.name,
                    path=item.path,
                    weight=item.weight,
                    workload_rank_parallel=item.workload_rank_parallel,
                    out_dir=item.out_dir,
                    feedback=item.feedback,
                    error=item.error,
                    speedup=speedup,
                    baseline_makespan_us=baseline_makespan,
                )
            )
        return aggregate_multi_workload_feedback(self._suite, speedup_runs, current_baseline)

    def _run_one_feedback(
        self,
        *,
        topology_path: Path,
        workload: WorkloadSuiteItem,
        workload_dir: Path,
    ) -> WorkloadRunFeedback:
        try:
            feedback = self._run_one(topology_path=topology_path, workload=workload, out_dir=workload_dir / "wrapper")
            return WorkloadRunFeedback(
                name=workload.name,
                path=workload.path,
                weight=float(workload.weight or 0.0),
                workload_rank_parallel=workload.workload_rank_parallel,
                out_dir=workload_dir,
                feedback=feedback,
            )
        except Exception as exc:
            return WorkloadRunFeedback(
                name=workload.name,
                path=workload.path,
                weight=float(workload.weight or 0.0),
                workload_rank_parallel=workload.workload_rank_parallel,
                out_dir=workload_dir,
                feedback=None,
                error=str(exc),
            )

    def _feedback_from_future(
        self,
        future: Any,
        *,
        workload: WorkloadSuiteItem,
        workload_dir: Path,
    ) -> WorkloadRunFeedback:
        try:
            feedback = future.result()
            return WorkloadRunFeedback(
                name=workload.name,
                path=workload.path,
                weight=float(workload.weight or 0.0),
                workload_rank_parallel=workload.workload_rank_parallel,
                out_dir=workload_dir,
                feedback=feedback,
            )
        except Exception as exc:
            return WorkloadRunFeedback(
                name=workload.name,
                path=workload.path,
                weight=float(workload.weight or 0.0),
                workload_rank_parallel=workload.workload_rank_parallel,
                out_dir=workload_dir,
                feedback=None,
                error=str(exc),
            )

    def _run_one(self, *, topology_path: Path, workload: WorkloadSuiteItem, out_dir: Path) -> ParsedPipelineFeedback:
        return self._pipeline.run(
            topology_path=topology_path,
            workload_path=workload.path,
            out_dir=out_dir,
            workload_rank_parallel=workload.workload_rank_parallel,
        )


def load_workload_suite(path: Path, *, repo_root: Path) -> WorkloadSuite:
    suite = WorkloadSuite.model_validate(load_jsonc(path))
    resolved: list[WorkloadSuiteItem] = []
    for item in suite.workloads:
        resolved_path = _resolve_workload_path(item.path, suite_path=path, repo_root=repo_root)
        if not resolved_path.exists():
            raise ValueError(f"workload suite item {item.name} path does not exist: {resolved_path}")
        resolved.append(item.model_copy(update={"path": resolved_path}))
    return suite.model_copy(update={"workloads": resolved})


def aggregate_multi_workload_feedback(
    suite: WorkloadSuite,
    runs: list[WorkloadRunFeedback],
    baseline: WorkloadSuiteBaseline,
) -> MultiWorkloadFeedback:
    runs = [
        replace(
            item,
            baseline_makespan_us=(
                item.baseline_makespan_us
                if item.baseline_makespan_us is not None
                else max(_EPS, baseline.makespans_us.get(item.name, item.makespan_us))
            ),
        )
        for item in runs
    ]
    speeds = [max(_EPS, item.speedup) for item in runs]
    weights = [item.weight for item in runs]
    geomean = math.exp(sum(weight * math.log(speed) for speed, weight in zip(speeds, weights, strict=True)))
    min_speed = min(speeds) if speeds else 0.0
    max_speed = max(speeds) if speeds else 0.0
    mean_speed = sum(speeds) / len(speeds) if speeds else 0.0
    variance = sum((speed - mean_speed) ** 2 for speed in speeds) / len(speeds) if speeds else 0.0
    speedup_cv = math.sqrt(variance) / mean_speed if mean_speed > 0 else 0.0
    aggregate = _aggregate_feedback(suite, runs, geomean)
    return MultiWorkloadFeedback(
        suite=suite,
        runs=runs,
        baseline=baseline,
        aggregate_feedback=aggregate,
        geomean_speedup=geomean,
        min_speedup=min_speed,
        max_speedup=max_speed,
        speedup_cv=speedup_cv,
        suite_makespan_score=1.0 / max(_EPS, geomean),
    )


def _aggregate_feedback(
    suite: WorkloadSuite,
    runs: list[WorkloadRunFeedback],
    geomean_speedup: float,
) -> ParsedPipelineFeedback:
    parsed = [item.feedback for item in runs if item.feedback is not None]
    max_link_util = max((item.max_link_utilization for item in parsed), default=0.0)
    max_queue = max((item.max_queue_delay_ns for item in parsed), default=0.0)
    max_remote = max((item.remote_memory_contention_ns for item in parsed), default=0.0)
    max_compute = max((_average_compute_utilization(item) for item in parsed), default=0.0)
    top_feedback = max(
        parsed,
        key=lambda item: (
            item.max_link_utilization
            + min(1.0, item.max_queue_delay_ns / 1_000_000.0)
            + min(1.0, item.remote_memory_contention_ns / 1_000_000.0)
        ),
        default=None,
    )
    link_stats = list(top_feedback.link_stats if top_feedback is not None else [])
    domain_stats = list(top_feedback.domain_stats if top_feedback is not None else [])
    remote_stats = {"provider_queue_time": int(max_remote), "provider_service_time": 0}
    top_bottlenecks = [
        LinkBottleneck(
            link_id=str(item.get("id", "")),
            utilization=float(item.get("utilization", 0.0)),
            queue_depth_pkts=int(item.get("max_queue_depth", 0)),
            dominant_traffic=str(item.get("technology", "") or item.get("stats_domain", "")),
        )
        for item in link_stats[:8]
    ]
    feedback = SimulationFeedback(
        simulation_id=f"{suite.name}_aggregate",
        workload=suite.name,
        global_metrics=GlobalMetrics(
            makespan_us=max((item.makespan_us for item in runs), default=0.0),
            total_energy_joules=0.0,
            peak_power_watts=0.0,
            thermal_violation=False,
            budget_utilization_percent=0.0,
        ),
        compute_profile={
            "joint": UtilizationStats(
                avg_utilization=max_compute,
                bubble_time_percent=max(0.0, 1.0 - max_compute),
            )
        },
        memory_profile=MemoryProfile(
            local_hbm_bandwidth_util=0.0,
            cxl_memory_pool=CxlMemoryPoolStats(
                avg_access_latency_ns=0.0,
                conflict_rate=1.0 if max_remote > 0 else 0.0,
            ),
        ),
        network_profile=NetworkProfile(
            average_link_utilization=max_link_util,
            top_bottlenecks=top_bottlenecks,
        ),
    )
    return ParsedPipelineFeedback(
        simulation_feedback=feedback,
        summary={
            "case_name": suite.name,
            "success": all(item.success for item in runs),
            "inputs": {"workload_suite": suite.name},
            "aggregate": {"geomean_speedup": geomean_speedup},
        },
        link_stats=link_stats,
        domain_stats=domain_stats,
        remote_memory_stats=remote_stats,
    )


def _resolve_workload_path(path: Path, *, suite_path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    repo_relative = repo_root / path
    if repo_relative.exists():
        return repo_relative
    return suite_path.parent / path


def _average_compute_utilization(feedback: ParsedPipelineFeedback) -> float:
    if not feedback.simulation_feedback.compute_profile:
        return 0.0
    values = [item.avg_utilization for item in feedback.simulation_feedback.compute_profile.values()]
    return sum(values) / len(values)


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_") or "workload"
