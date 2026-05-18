from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


_FINISHED_RE = re.compile(r"sys\[(?P<sys>\d+)\] finished,\s+(?P<cycles>\d+) cycles")
_WALL_RE = re.compile(r"sys\[(?P<sys>\d+)\], Wall time:\s+(?P<cycles>\d+)")
_COMPUTE_UTIL_RE = re.compile(
    r"sys\[(?P<sys>\d+)\], Average compute utilization:\s+(?P<util>[0-9.]+)%"
)
_MEMORY_UTIL_RE = re.compile(
    r"sys\[(?P<sys>\d+)\], Average memory utilization:\s+(?P<util>[0-9.]+)%"
)
_REMOTE_QUEUE_RE = re.compile(
    r"sys\[(?P<sys>\d+)\], Remote mem provider queue time:\s+(?P<time>\d+)"
)
_REMOTE_SERVICE_RE = re.compile(
    r"sys\[(?P<sys>\d+)\], Remote mem provider service time:\s+(?P<time>\d+)"
)
_TYPE_TIME_RE = re.compile(
    r"sys\[(?P<sys>\d+)\], (?P<kind>CPU|GPU|Comm|Remote mem) time:\s+(?P<time>\d+)"
)
_OVERLAP_RE = re.compile(
    r"sys\[(?P<sys>\d+)\], Total compute-communication overlap:\s+(?P<time>\d+)"
)
_LINK_RE = re.compile(
    r"Network top congested link rank=(?P<rank>\d+) id=(?P<id>\S+) "
    r"src_device=(?P<src>\d+) dst_device=(?P<dst>\d+) "
    r"level=(?P<level>\S*) domain=(?P<domain>\S*) stats_domain=(?P<stats_domain>\S*) "
    r"technology=(?P<technology>\S*) route_class=(?P<route_class>\S*) "
    r"bytes=(?P<bytes>\d+) busy_time_ns=(?P<busy>\d+) "
    r"queue_delay_ns=(?P<queue>\d+) transmissions=(?P<tx>\d+) "
    r"max_queue_depth=(?P<depth>\d+) utilization=(?P<util>[0-9.]+)"
)
_DOMAIN_RE = re.compile(
    r"Network top congested domain rank=(?P<rank>\d+) stats_domain=(?P<stats_domain>\S+) "
    r"bytes=(?P<bytes>\d+) busy_time_ns=(?P<busy>\d+) "
    r"queue_delay_ns=(?P<queue>\d+) transmissions=(?P<tx>\d+) "
    r"max_queue_depth=(?P<depth>\d+) utilization=(?P<util>[0-9.]+)"
)
_SCALING_RE = re.compile(r"Scaling report (?P<name>[a-zA-Z0-9_]+) (?P<fields>.*)$")


@dataclass(frozen=True)
class ParsedPipelineFeedback:
    simulation_feedback: SimulationFeedback
    summary: dict[str, Any]
    link_stats: list[dict[str, Any]] = field(default_factory=list)
    domain_stats: list[dict[str, Any]] = field(default_factory=list)
    remote_memory_stats: dict[str, int] = field(default_factory=dict)
    scaling_report: dict[str, dict[str, int]] = field(default_factory=dict)
    operator_times: dict[str, dict[str, int]] = field(default_factory=dict)
    compute_comm_overlap_ns: int = 0
    simulator_stdout_path: Path | None = None

    @property
    def makespan_us(self) -> float:
        return self.simulation_feedback.global_metrics.makespan_us

    @property
    def max_link_utilization(self) -> float:
        if not self.link_stats:
            return 0.0
        return max(float(item.get("utilization", 0.0)) for item in self.link_stats)

    @property
    def max_queue_delay_ns(self) -> float:
        if not self.link_stats:
            return 0.0
        return max(float(item.get("queue_delay_ns", 0.0)) for item in self.link_stats)

    @property
    def remote_memory_contention_ns(self) -> float:
        return float(self.remote_memory_stats.get("provider_queue_time", 0))


def parse_pipeline_outputs(case_dir: Path) -> ParsedPipelineFeedback:
    summary_path = case_dir / "outputs" / "run_summary.json"
    summary = load_jsonc(summary_path)
    stdout_path = Path(summary["simulator"]["stdout"])
    stdout = stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else ""
    return parse_pipeline_feedback(summary=summary, simulator_stdout=stdout, stdout_path=stdout_path)


def parse_pipeline_feedback(
    *,
    summary: dict[str, Any],
    simulator_stdout: str,
    stdout_path: Path | None = None,
) -> ParsedPipelineFeedback:
    wall_times: dict[str, int] = {}
    compute_util: dict[str, float] = {}
    memory_util: dict[str, float] = {}
    remote_queue = 0
    remote_service = 0
    link_stats: list[dict[str, Any]] = []
    domain_stats: list[dict[str, Any]] = []
    scaling: dict[str, dict[str, int]] = {}
    operator_times: dict[str, dict[str, int]] = {}
    compute_comm_overlap = 0

    for line in simulator_stdout.splitlines():
        if match := _FINISHED_RE.search(line):
            wall_times[match.group("sys")] = int(match.group("cycles"))
        if match := _WALL_RE.search(line):
            wall_times[match.group("sys")] = int(match.group("cycles"))
        if match := _COMPUTE_UTIL_RE.search(line):
            compute_util[match.group("sys")] = float(match.group("util")) / 100.0
        if match := _MEMORY_UTIL_RE.search(line):
            memory_util[match.group("sys")] = float(match.group("util")) / 100.0
        if match := _REMOTE_QUEUE_RE.search(line):
            remote_queue += int(match.group("time"))
        if match := _REMOTE_SERVICE_RE.search(line):
            remote_service += int(match.group("time"))
        if match := _TYPE_TIME_RE.search(line):
            sys_id = match.group("sys")
            kind = match.group("kind").lower().replace(" ", "_")
            operator_times.setdefault(sys_id, {})[kind] = int(match.group("time"))
        if match := _OVERLAP_RE.search(line):
            compute_comm_overlap += int(match.group("time"))
        if match := _LINK_RE.search(line):
            link_stats.append(
                {
                    "rank": int(match.group("rank")),
                    "id": match.group("id"),
                    "src_device": int(match.group("src")),
                    "dst_device": int(match.group("dst")),
                    "level": match.group("level"),
                    "domain": match.group("domain"),
                    "stats_domain": match.group("stats_domain"),
                    "technology": match.group("technology"),
                    "route_class": match.group("route_class"),
                    "bytes": int(match.group("bytes")),
                    "busy_time_ns": int(match.group("busy")),
                    "queue_delay_ns": int(match.group("queue")),
                    "transmissions": int(match.group("tx")),
                    "max_queue_depth": int(match.group("depth")),
                    "utilization": float(match.group("util")),
                }
            )
        if match := _DOMAIN_RE.search(line):
            domain_stats.append(
                {
                    "rank": int(match.group("rank")),
                    "stats_domain": match.group("stats_domain"),
                    "bytes": int(match.group("bytes")),
                    "busy_time_ns": int(match.group("busy")),
                    "queue_delay_ns": int(match.group("queue")),
                    "transmissions": int(match.group("tx")),
                    "max_queue_depth": int(match.group("depth")),
                    "utilization": float(match.group("util")),
                }
            )
        if match := _SCALING_RE.search(line):
            scaling[match.group("name")] = _parse_scaling_fields(match.group("fields"))

    makespan = float(max(wall_times.values()) if wall_times else 0)
    if makespan <= 0:
        makespan = 1_000_000_000.0 if not summary.get("success", False) else 0.0

    avg_memory_util = (
        sum(memory_util.values()) / len(memory_util)
        if memory_util
        else 0.0
    )
    top_bottlenecks = [
        LinkBottleneck(
            link_id=item["id"],
            utilization=item["utilization"],
            queue_depth_pkts=item["max_queue_depth"],
            dominant_traffic=item.get("technology", "") or item.get("stats_domain", ""),
        )
        for item in link_stats[:8]
    ]
    feedback = SimulationFeedback(
        simulation_id=str(summary.get("case_name", "mapper_sim_pipeline")),
        workload=str(summary.get("inputs", {}).get("workload", "")),
        global_metrics=GlobalMetrics(
            makespan_us=makespan,
            total_energy_joules=0.0,
            peak_power_watts=0.0,
            thermal_violation=False,
            budget_utilization_percent=0.0,
        ),
        compute_profile={
            f"sys{sys_id}": UtilizationStats(
                avg_utilization=util,
                bubble_time_percent=max(0.0, 1.0 - util),
            )
            for sys_id, util in compute_util.items()
        },
        memory_profile=MemoryProfile(
            local_hbm_bandwidth_util=avg_memory_util,
            cxl_memory_pool=CxlMemoryPoolStats(
                avg_access_latency_ns=float(remote_service),
                conflict_rate=1.0 if remote_queue > 0 else 0.0,
            ),
        ),
        network_profile=NetworkProfile(
            average_link_utilization=(
                sum(item["utilization"] for item in link_stats) / len(link_stats)
                if link_stats
                else 0.0
            ),
            top_bottlenecks=top_bottlenecks,
        ),
    )
    return ParsedPipelineFeedback(
        simulation_feedback=feedback,
        summary=summary,
        link_stats=link_stats,
        domain_stats=domain_stats,
        remote_memory_stats={
            "provider_queue_time": remote_queue,
            "provider_service_time": remote_service,
        },
        scaling_report=scaling,
        operator_times=operator_times,
        compute_comm_overlap_ns=compute_comm_overlap,
        simulator_stdout_path=stdout_path,
    )


def _parse_scaling_fields(fields: str) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for token in fields.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        try:
            parsed[key] = int(value)
        except ValueError:
            continue
    return parsed
