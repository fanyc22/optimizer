from __future__ import annotations

from typing import Any

from codesign_optimizer.optimizer.feedback_parser import ParsedPipelineFeedback
from codesign_optimizer.optimizer.repair import RepairReport
from codesign_optimizer.optimizer.search_space import SearchLimits, SearchObjectiveWeights


ObjectiveTuple = tuple[float, float, float, float, float, float]

FAILED_MAKESPAN_US = 1_000_000_000.0
FAILED_LINK_UTILIZATION = 1_000_000.0
FAILED_QUEUE_NS = 1_000_000_000.0
INFEASIBLE_PENALTY = 1_000_000_000.0


def budget_wall_pressure(
    value: float,
    limit: float,
    *,
    weights: SearchObjectiveWeights,
) -> float:
    if limit <= 0:
        return 0.0 if value <= 0 else float("inf")
    utilization = max(0.0, value / limit)
    pressure = weights.budget_tiebreak * utilization
    if utilization <= weights.budget_wall_knee:
        return pressure
    span = max(1e-12, 1.0 - weights.budget_wall_knee)
    wall = ((utilization - weights.budget_wall_knee) / span) ** weights.budget_wall_exponent
    return pressure + wall


def penalty_objectives(repair: RepairReport) -> ObjectiveTuple:
    return (
        FAILED_MAKESPAN_US + repair.penalty,
        repair.estimated_cost,
        repair.estimated_power_watts,
        FAILED_LINK_UTILIZATION,
        FAILED_QUEUE_NS,
        FAILED_QUEUE_NS,
    )


def single_workload_objectives(
    repair: RepairReport,
    feedback: ParsedPipelineFeedback | None,
    feasible: bool,
) -> ObjectiveTuple:
    if feedback is None:
        return penalty_objectives(repair)
    penalty = 0.0 if feasible else INFEASIBLE_PENALTY
    return (
        feedback.makespan_us + penalty,
        repair.estimated_cost,
        repair.estimated_power_watts,
        feedback.max_link_utilization + (FAILED_LINK_UTILIZATION if not feasible else 0.0),
        feedback.max_queue_delay_ns + penalty,
        feedback.remote_memory_contention_ns + penalty,
    )


def tgrl_v2_objectives(
    repair: RepairReport,
    feedback: ParsedPipelineFeedback | None,
    feasible: bool,
    *,
    suite_feedback: Any | None = None,
) -> ObjectiveTuple:
    if feedback is None:
        return penalty_objectives(repair)
    penalty = 0.0 if feasible else INFEASIBLE_PENALTY
    primary_performance = (
        suite_feedback.suite_makespan_score * 10_000.0
        if suite_feedback is not None
        else feedback.makespan_us
    )
    return (
        primary_performance + penalty,
        repair.estimated_cost,
        repair.estimated_power_watts,
        feedback.max_link_utilization + (FAILED_LINK_UTILIZATION if not feasible else 0.0),
        feedback.max_queue_delay_ns + penalty,
        feedback.remote_memory_contention_ns + penalty,
    )


def weighted_score_from_objectives(
    objectives: ObjectiveTuple,
    *,
    weights: SearchObjectiveWeights,
    limits: SearchLimits | None = None,
    feasible: bool,
    penalty: float,
) -> float:
    cost_pressure = (
        budget_wall_pressure(objectives[1], limits.max_total_cost, weights=weights)
        if limits is not None
        else objectives[1] / 1_000_000.0
    )
    power_pressure = (
        budget_wall_pressure(objectives[2], limits.max_peak_power_watts, weights=weights)
        if limits is not None
        else objectives[2] / 100_000.0
    )
    score = (
        weights.makespan * (objectives[0] / 10_000.0)
        + weights.cost * cost_pressure
        + weights.power * power_pressure
        + weights.max_link_utilization * objectives[3]
        + weights.max_queue_delay * (objectives[4] / 1_000_000.0)
        + weights.remote_memory_contention * (objectives[5] / 1_000_000.0)
    )
    if not feasible:
        score += 1_000_000.0 + penalty
    return score


def objectives_to_dict(objectives: ObjectiveTuple) -> dict[str, float]:
    return {
        "makespan_us": objectives[0],
        "estimated_cost": objectives[1],
        "estimated_power_watts": objectives[2],
        "max_link_utilization": objectives[3],
        "max_queue_delay_ns": objectives[4],
        "remote_memory_contention_ns": objectives[5],
    }
