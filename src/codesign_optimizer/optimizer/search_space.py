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


RackRole = Literal["compute", "memory", "hybrid"]


class RackCapacityLimits(BaseModel):
    max_gpu_count: int | None = Field(default=None, ge=0)
    max_cpu_count: int | None = Field(default=None, ge=0)
    max_memory_pool_count: int | None = Field(default=None, ge=0)
    max_switch_count: int | None = Field(default=None, ge=0)
    max_rack_units: float | None = Field(default=None, ge=0)
    max_power_watts: float | None = Field(default=None, ge=0)


class RackSpec(BaseModel):
    rack_id: str
    role: RackRole | None = None
    optional: bool = False
    active: bool = True
    activation_alpha: float | None = Field(default=None, ge=0)
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
    endpoint_link_qty: int = Field(default=1, ge=1)
    gpu_link_qty: int | None = Field(default=None, ge=1)
    cpu_link_qty: int | None = Field(default=None, ge=1)
    memory_link_qty: int = Field(default=1, ge=1)
    fabric: Literal["switch", "ring"] = "switch"
    limits: RackCapacityLimits = Field(default_factory=RackCapacityLimits)

    @model_validator(mode="after")
    def validate_rack(self) -> "RackSpec":
        compute_count = self.gpu_count + self.cpu_count
        if compute_count + self.memory_pool_count <= 0:
            if not self.optional:
                raise ValueError(f"rack {self.rack_id} must contain compute or memory nodes")
            if self.role is None:
                raise ValueError(f"optional empty rack {self.rack_id} must set role")
            if self.active:
                raise ValueError(f"optional empty rack {self.rack_id} must set active=false")
        if self.fabric == "switch" and self.switch_count <= 0 and self.active:
            raise ValueError(f"rack {self.rack_id} uses switch fabric but switch_count is 0")
        if self.role is None:
            if compute_count > 0 and self.memory_pool_count > 0:
                self.role = "hybrid"
            elif self.memory_pool_count > 0:
                self.role = "memory"
            else:
                self.role = "compute"
        if self.role == "memory" and compute_count > 0:
            raise ValueError(f"rack {self.rack_id} role=memory cannot contain GPU/CPU nodes")
        if self.role == "compute" and self.memory_pool_count > 0:
            raise ValueError(f"rack {self.rack_id} role=compute cannot contain memory pools")
        return self


class RackTemplate(BaseModel):
    name: str
    racks: list[RackSpec] | None = None
    rack_count: int = Field(default=1, ge=1)
    gpu_count: int = Field(default=0, ge=0)
    cpu_count: int = Field(default=0, ge=0)
    memory_pool_count: int = Field(default=0, ge=0)
    switch_count: int = Field(default=1, ge=0)
    gpu_type: str | None = None
    cpu_type: str | None = None
    memory_pool_type: str | None = None
    switch_type: str | None = None
    endpoint_link_type: str | None = None
    gpu_link_type: str | None = None
    cpu_link_type: str | None = None
    memory_link_type: str | None = None
    inter_rack_link_type: str | None = None
    rack_limits: RackCapacityLimits | None = None
    endpoint_link_qty: int = Field(default=1, ge=1)
    gpu_link_qty: int | None = Field(default=None, ge=1)
    cpu_link_qty: int | None = Field(default=None, ge=1)
    memory_link_qty: int = Field(default=1, ge=1)
    inter_rack_link_qty: int = Field(default=1, ge=1)
    fabric: Literal["switch", "ring"] = "switch"
    inter_rack: Literal["none", "ring", "fully_connected"] = "ring"

    @model_validator(mode="after")
    def validate_template(self) -> "RackTemplate":
        if self.racks:
            rack_ids = [rack.rack_id for rack in self.racks]
            if len(set(rack_ids)) != len(rack_ids):
                raise ValueError(f"template {self.name} has duplicate rack_id")
            active_compute_racks = [
                rack
                for rack in self.racks
                if (rack.active or not rack.optional) and rack.gpu_count + rack.cpu_count > 0
            ]
            if not active_compute_racks:
                raise ValueError(f"template {self.name} must contain at least one active compute rack")
            if self.inter_rack != "none" and not (
                self.inter_rack_link_type or self.endpoint_link_type or self.racks[0].endpoint_link_type
            ):
                raise ValueError(f"template {self.name} needs inter_rack_link_type for inter-rack fabric")
            return self

        if self.endpoint_link_type is None:
            raise ValueError(f"template {self.name} must set endpoint_link_type")
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
