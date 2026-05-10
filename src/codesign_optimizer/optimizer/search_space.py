from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from codesign_optimizer.models.hardware import ComponentLibrary


class SearchLimits(BaseModel):
    max_total_cost: float = Field(default=1_000_000_000.0, ge=0)
    max_peak_power_watts: float = Field(default=1_000_000_000.0, ge=0)
    max_rack_power_watts: float = Field(default=100_000.0, ge=0)
    max_rack_units: float = Field(default=42.0, ge=0)


class SearchObjectiveWeights(BaseModel):
    makespan: float = Field(default=1.0, ge=0)
    cost: float = Field(default=0.15, ge=0)
    power: float = Field(default=0.10, ge=0)
    max_link_utilization: float = Field(default=0.20, ge=0)
    max_queue_delay: float = Field(default=0.20, ge=0)
    remote_memory_contention: float = Field(default=0.10, ge=0)


class MutationSettings(BaseModel):
    min_gpu_per_rack: int = Field(default=0, ge=0)
    max_gpu_per_rack: int = Field(default=8, ge=0)
    min_cpu_per_rack: int = Field(default=0, ge=0)
    max_cpu_per_rack: int = Field(default=8, ge=0)
    min_memory_pools_per_rack: int = Field(default=0, ge=0)
    max_memory_pools_per_rack: int = Field(default=4, ge=0)
    min_endpoint_link_qty: int = Field(default=1, ge=1)
    max_endpoint_link_qty: int = Field(default=8, ge=1)
    min_inter_rack_link_qty: int = Field(default=1, ge=1)
    max_inter_rack_link_qty: int = Field(default=8, ge=1)
    mutation_rate: float = Field(default=0.35, ge=0, le=1)
    bottleneck_mutation_rate: float = Field(default=0.35, ge=0, le=1)
    elite_fraction: float = Field(default=0.20, ge=0, le=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> "MutationSettings":
        if self.min_gpu_per_rack > self.max_gpu_per_rack:
            raise ValueError("min_gpu_per_rack must be <= max_gpu_per_rack")
        if self.min_cpu_per_rack > self.max_cpu_per_rack:
            raise ValueError("min_cpu_per_rack must be <= max_cpu_per_rack")
        if self.min_memory_pools_per_rack > self.max_memory_pools_per_rack:
            raise ValueError("min_memory_pools_per_rack must be <= max_memory_pools_per_rack")
        if self.min_endpoint_link_qty > self.max_endpoint_link_qty:
            raise ValueError("min_endpoint_link_qty must be <= max_endpoint_link_qty")
        if self.min_inter_rack_link_qty > self.max_inter_rack_link_qty:
            raise ValueError("min_inter_rack_link_qty must be <= max_inter_rack_link_qty")
        return self


class EvaluationSettings(BaseModel):
    mapper: str = "heft"
    parallel: str = "auto"
    topology_format: str = "hardware"
    sim_extra: list[str] = Field(default_factory=list)
    mapper_extra: list[str] = Field(default_factory=list)
    scaling_report: bool = False
    timeout_seconds: int | None = Field(default=None, ge=1)
    repo_root: Path | None = None


class RackTemplate(BaseModel):
    name: str
    rack_count: int = Field(default=1, ge=1)
    gpu_count: int = Field(default=0, ge=0)
    cpu_count: int = Field(default=0, ge=0)
    memory_pool_count: int = Field(default=0, ge=0)
    switch_count: int = Field(default=1, ge=0)
    gpu_type: str | None = None
    cpu_type: str | None = None
    memory_pool_type: str | None = None
    switch_type: str | None = None
    endpoint_link_type: str
    gpu_link_type: str | None = None
    cpu_link_type: str | None = None
    memory_link_type: str | None = None
    inter_rack_link_type: str | None = None
    endpoint_link_qty: int = Field(default=1, ge=1)
    gpu_link_qty: int | None = Field(default=None, ge=1)
    cpu_link_qty: int | None = Field(default=None, ge=1)
    memory_link_qty: int = Field(default=1, ge=1)
    inter_rack_link_qty: int = Field(default=1, ge=1)
    fabric: Literal["switch", "ring"] = "switch"
    inter_rack: Literal["none", "ring", "fully_connected"] = "ring"

    @model_validator(mode="after")
    def validate_compute_exists(self) -> "RackTemplate":
        if self.gpu_count + self.cpu_count <= 0:
            raise ValueError(f"template {self.name} must contain at least one compute node")
        if self.fabric == "switch" and self.switch_count <= 0:
            raise ValueError(f"template {self.name} uses switch fabric but switch_count is 0")
        return self


class SearchSpace(BaseModel):
    seed: int = 1
    templates: list[RackTemplate]
    mutation: MutationSettings = MutationSettings()
    limits: SearchLimits = SearchLimits()
    objective_weights: SearchObjectiveWeights = SearchObjectiveWeights()
    evaluation: EvaluationSettings = EvaluationSettings()

    @model_validator(mode="after")
    def validate_templates(self) -> "SearchSpace":
        if not self.templates:
            raise ValueError("search space must contain at least one template")
        return self


def load_component_library(payload: dict) -> ComponentLibrary:
    if "component_library" in payload:
        payload = payload["component_library"]
    return ComponentLibrary.model_validate(payload)
