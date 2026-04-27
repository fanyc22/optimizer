from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class ObjectiveWeights(BaseModel):
    makespan: float = 0.60
    energy: float = 0.20
    thermal_penalty: float = 0.15
    budget_penalty: float = 0.05


class ConstraintLimits(BaseModel):
    max_peak_power_watts: float = 100_000.0
    max_budget_utilization_percent: float = 100.0


class OptimizerSettings(BaseModel):
    max_outer_iterations: int = Field(default=8, ge=1, le=1000)
    inner_trials_per_outer: int = Field(default=6, ge=1, le=500)
    artifacts_dir: Path = Path("artifacts")
    objective_weights: ObjectiveWeights = ObjectiveWeights()
    limits: ConstraintLimits = ConstraintLimits()
