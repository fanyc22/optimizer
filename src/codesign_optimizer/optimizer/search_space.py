from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from codesign_optimizer.models.hardware import ComponentLibrary


class SearchLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_total_cost: float = Field(default=1_000_000_000.0, ge=0)
    max_peak_power_watts: float = Field(default=1_000_000_000.0, ge=0)
    max_rack_cost: float = Field(default=1_000_000_000.0, ge=0)
    max_rack_power_watts: float = Field(default=100_000.0, ge=0)
    max_rack_units: float = Field(default=42.0, ge=0)
    max_total_racks: int | None = Field(default=None, ge=1)
    min_compute_racks: int = Field(default=1, ge=0)
    max_compute_racks: int | None = Field(default=None, ge=0)
    max_memory_racks: int | None = Field(default=None, ge=0)
    max_hybrid_racks: int | None = Field(default=None, ge=0)


class SearchObjectiveWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    makespan: float = Field(default=1.0, ge=0)
    cost: float = Field(default=0.15, ge=0)
    power: float = Field(default=0.10, ge=0)
    max_link_utilization: float = Field(default=0.20, ge=0)
    max_queue_delay: float = Field(default=0.20, ge=0)
    remote_memory_contention: float = Field(default=0.10, ge=0)


class MutationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    search_granularity: Literal["slot", "host"] = "slot"
    min_intra_rack_link_qty: int = Field(default=1, ge=1)
    max_intra_rack_link_qty: int = Field(default=8, ge=1)
    min_inter_rack_link_qty: int = Field(default=1, ge=1)
    max_inter_rack_link_qty: int = Field(default=8, ge=1)
    mutation_rate: float = Field(default=0.35, ge=0, le=1)
    bottleneck_mutation_rate: float = Field(default=0.35, ge=0, le=1)
    elite_fraction: float = Field(default=0.20, ge=0, le=1)
    allow_remove_initial_racks: bool = False

    @property
    def min_gpu_per_rack(self) -> int:
        return 0

    @property
    def max_gpu_per_rack(self) -> int:
        return 1_000_000

    @property
    def min_cpu_per_rack(self) -> int:
        return 0

    @property
    def max_cpu_per_rack(self) -> int:
        return 1_000_000

    @property
    def min_memory_pools_per_rack(self) -> int:
        return 0

    @property
    def max_memory_pools_per_rack(self) -> int:
        return 1_000_000

    @property
    def min_endpoint_link_qty(self) -> int:
        return self.min_intra_rack_link_qty

    @property
    def max_endpoint_link_qty(self) -> int:
        return self.max_intra_rack_link_qty

    @model_validator(mode="after")
    def validate_bounds(self) -> "MutationSettings":
        if self.min_intra_rack_link_qty > self.max_intra_rack_link_qty:
            raise ValueError("min_intra_rack_link_qty must be <= max_intra_rack_link_qty")
        if self.min_inter_rack_link_qty > self.max_inter_rack_link_qty:
            raise ValueError("min_inter_rack_link_qty must be <= max_inter_rack_link_qty")
        return self


class EvaluationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workload_kind: Literal["mapper", "llm"] = "mapper"
    mapper: str = "heft"
    parallel: str = "auto"
    topology_format: str = "hardware"
    llm_prefill_batch_size: int = Field(default=1, ge=1)
    llm_prompt_len: int = Field(default=2048, ge=0)
    llm_decode_batch_size: int = Field(default=1, ge=1)
    llm_decode_steps: int = Field(default=0, ge=0)
    llm_avg_context_len: int = Field(default=2048, ge=0)
    llm_tp: int = Field(default=1, ge=1)
    llm_pp: int = Field(default=1, ge=1)
    llm_cp: int = Field(default=1, ge=1)
    llm_dp: int = Field(default=1, ge=1)
    sim_extra: list[str] = Field(default_factory=list)
    mapper_extra: list[str] = Field(default_factory=list)
    calibration_fit_model: Path | None = None
    mapper_calibration_mode: Literal["baked", "full"] = "baked"
    mapper_calibration_group: str = "native"
    scaling_report: bool = False
    cleanup_wrapper_intermediate: bool = True
    timeout_seconds: int | None = Field(default=None, ge=1)
    repo_root: Path | None = None


class ExhaustiveSlotOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_type: str | None = None
    link_type: str | None = None
    link_qty: int | None = Field(default=None, ge=1)


class ExhaustiveSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot_options: list[ExhaustiveSlotOption] = Field(default_factory=list)
    intra_rack_topologies: list[str] | None = None
    intra_rack_link_types: list[str] | None = None
    intra_rack_link_qty: list[int] | None = None
    inter_rack_topologies: list[str] | None = None
    inter_rack_link_types: list[str] | None = None
    inter_rack_link_qty: list[int] | None = None
    max_candidates: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_exhaustive_settings(self) -> "ExhaustiveSettings":
        for field_name in ("intra_rack_link_qty", "inter_rack_link_qty"):
            values = getattr(self, field_name)
            if values is not None and any(value < 1 for value in values):
                raise ValueError(f"{field_name} values must be >= 1")
        return self


RackRole = Literal["compute", "memory", "hybrid"]
RackOrigin = Literal["seed", "dynamic"]
FabricMode = Literal["none", "ring", "fully_connected", "switch"]
InterRackMode = Literal["none", "ring", "fully_connected"]


class SlotSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot_id: str
    node_type: str | None = None
    link_type: str | None = None
    link_qty: int | None = Field(default=None, ge=1)


class HostTemplateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    template_id: str
    rack_units: float = Field(default=1.0, ge=0)
    slots: list[SlotSpec] = Field(default_factory=list)
    host_topology: FabricMode = "switch"
    host_switch_type: str | None = None
    host_link_type: str | None = None
    host_link_qty: int = Field(default=1, ge=1)
    rack_uplink_link_type: str | None = None
    rack_uplink_link_qty: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_host_template(self) -> "HostTemplateSpec":
        slot_ids = [slot.slot_id for slot in self.slots]
        if len(slot_ids) != len(set(slot_ids)):
            raise ValueError(f"host template {self.template_id} has duplicate slot_id")
        occupied = [slot for slot in self.slots if slot.node_type]
        if self.host_topology == "switch" and occupied and self.host_switch_type is None:
            raise ValueError(f"host template {self.template_id} uses switch topology but has no host_switch_type")
        if self.host_topology != "none" and len(occupied) > 1 and self.host_link_type is None:
            raise ValueError(f"host template {self.template_id} needs host_link_type")
        if self.host_topology == "none" and len(occupied) > 1:
            raise ValueError(f"host template {self.template_id} has multiple slots but host_topology=none")
        return self


class HostSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host_id: str
    template_id: str | None = None


class RackCapacityLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_slots: int | None = Field(default=None, ge=0)
    max_memory_pool_count: int | None = Field(default=None, ge=0)
    max_switch_count: int | None = Field(default=None, ge=0)
    max_cost: float | None = Field(default=None, ge=0)
    max_rack_units: float | None = Field(default=None, ge=0)
    max_power_watts: float | None = Field(default=None, ge=0)

    @property
    def max_gpu_count(self) -> int | None:
        return self.max_slots

    @property
    def max_cpu_count(self) -> int | None:
        return self.max_slots


class RackSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rack_id: str
    role: RackRole = "compute"
    optional: bool = False
    active: bool = True
    activation_alpha: float | None = Field(default=None, ge=0)
    origin: RackOrigin = "seed"
    max_slots: int = Field(default=0, ge=0)
    slots: list[SlotSpec] = Field(default_factory=list)
    max_hosts: int | None = Field(default=None, ge=0)
    hosts: list[HostSpec] = Field(default_factory=list)
    memory_pool_count: int = Field(default=0, ge=0)
    switch_count: int = Field(default=1, ge=0)
    memory_pool_type: str | None = None
    switch_type: str | None = None
    intra_rack_topology: FabricMode = "switch"
    intra_rack_link_type: str | None = None
    intra_rack_link_qty: int = Field(default=1, ge=1)
    memory_link_type: str | None = None
    memory_link_qty: int = Field(default=1, ge=1)
    limits: RackCapacityLimits = Field(default_factory=RackCapacityLimits)

    @model_validator(mode="after")
    def validate_rack(self) -> "RackSpec":
        slot_ids = [slot.slot_id for slot in self.slots]
        if len(slot_ids) != len(set(slot_ids)):
            raise ValueError(f"rack {self.rack_id} has duplicate slot_id")
        host_ids = [host.host_id for host in self.hosts]
        if len(host_ids) != len(set(host_ids)):
            raise ValueError(f"rack {self.rack_id} has duplicate host_id")
        if self.hosts and self.slots:
            raise ValueError(f"rack {self.rack_id} cannot define both legacy slots and hosts")
        if self.max_hosts is None:
            self.max_hosts = len(self.hosts)
        if self.max_hosts is not None and len(self.hosts) > self.max_hosts:
            raise ValueError(f"rack {self.rack_id} has more hosts than max_hosts")
        if len(self.slots) > self.max_slots:
            raise ValueError(f"rack {self.rack_id} has more slots than max_slots")
        if self.limits.max_slots is None:
            self.limits.max_slots = self.max_slots
        if self.limits.max_slots is not None and self.max_slots > self.limits.max_slots:
            raise ValueError(f"rack {self.rack_id} max_slots exceeds limits.max_slots")
        if self.optional and not self.active:
            return self
        if self.intra_rack_topology == "switch" and self.switch_count <= 0:
            raise ValueError(f"rack {self.rack_id} uses switch topology but switch_count is 0")
        if self.memory_pool_count > 0 and self.memory_pool_type is None:
            raise ValueError(f"rack {self.rack_id} has memory pools but memory_pool_type is not set")
        occupied = [slot for slot in self.slots if slot.node_type]
        occupied_hosts = [host for host in self.hosts if host.template_id]
        if self.role in {"compute", "hybrid"} and not occupied:
            if not occupied_hosts:
                raise ValueError(f"rack {self.rack_id} must contain at least one occupied compute slot or host")
        if self.role == "memory" and self.memory_pool_count <= 0:
            raise ValueError(f"rack {self.rack_id} role=memory must contain memory pools")
        return self


class RackArchetype(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    rack_id_prefix: str | None = None
    role: RackRole = "compute"
    max_slots: int = Field(default=0, ge=0)
    slots: list[SlotSpec] = Field(default_factory=list)
    max_hosts: int | None = Field(default=None, ge=0)
    hosts: list[HostSpec] = Field(default_factory=list)
    memory_pool_count: int = Field(default=0, ge=0)
    switch_count: int = Field(default=1, ge=0)
    memory_pool_type: str | None = None
    switch_type: str | None = None
    intra_rack_topology: FabricMode = "switch"
    intra_rack_link_type: str | None = None
    intra_rack_link_qty: int = Field(default=1, ge=1)
    memory_link_type: str | None = None
    memory_link_qty: int = Field(default=1, ge=1)
    limits: RackCapacityLimits = Field(default_factory=RackCapacityLimits)

    @model_validator(mode="after")
    def validate_archetype(self) -> "RackArchetype":
        self.to_rack_spec(self.rack_id_prefix or self.name, origin="dynamic")
        return self

    def to_rack_spec(self, rack_id: str, *, origin: RackOrigin = "dynamic") -> RackSpec:
        return RackSpec(
            rack_id=rack_id,
            role=self.role,
            optional=False,
            active=True,
            origin=origin,
            max_slots=self.max_slots,
            slots=[slot.model_copy(deep=True) for slot in self.slots],
            max_hosts=self.max_hosts,
            hosts=[host.model_copy(deep=True) for host in self.hosts],
            memory_pool_count=self.memory_pool_count,
            switch_count=self.switch_count,
            memory_pool_type=self.memory_pool_type,
            switch_type=self.switch_type,
            intra_rack_topology=self.intra_rack_topology,
            intra_rack_link_type=self.intra_rack_link_type,
            intra_rack_link_qty=self.intra_rack_link_qty,
            memory_link_type=self.memory_link_type,
            memory_link_qty=self.memory_link_qty,
            limits=self.limits.model_copy(deep=True),
        )


class RackTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    racks: list[RackSpec]
    inter_rack: InterRackMode = "ring"
    inter_rack_link_type: str | None = None
    inter_rack_link_qty: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_template(self) -> "RackTemplate":
        rack_ids = [rack.rack_id for rack in self.racks]
        if len(rack_ids) != len(set(rack_ids)):
            raise ValueError(f"template {self.name} has duplicate rack_id")
        active_compute_racks = [
            rack
            for rack in self.racks
            if (rack.active or not rack.optional)
            and rack.role in {"compute", "hybrid"}
            and (any(slot.node_type for slot in rack.slots) or any(host.template_id for host in rack.hosts))
        ]
        if not active_compute_racks:
            raise ValueError(f"template {self.name} must contain at least one active compute rack")
        if self.inter_rack != "none" and not self.inter_rack_link_type:
            raise ValueError(f"template {self.name} needs inter_rack_link_type for inter-rack fabric")
        return self


class SearchSpace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 1
    host_templates: list[HostTemplateSpec] = Field(default_factory=list)
    templates: list[RackTemplate]
    rack_archetypes: list[RackArchetype] = Field(default_factory=list)
    mutation: MutationSettings = Field(default_factory=MutationSettings)
    limits: SearchLimits = Field(default_factory=SearchLimits)
    objective_weights: SearchObjectiveWeights = Field(default_factory=SearchObjectiveWeights)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    exhaustive: ExhaustiveSettings = Field(default_factory=ExhaustiveSettings)

    @model_validator(mode="after")
    def validate_templates(self) -> "SearchSpace":
        if not self.templates:
            raise ValueError("search space must contain at least one template")
        names = [item.name for item in self.rack_archetypes]
        if len(names) != len(set(names)):
            raise ValueError("rack_archetypes must have unique names")
        template_ids = [item.template_id for item in self.host_templates]
        if len(template_ids) != len(set(template_ids)):
            raise ValueError("host_templates must have unique template_id")
        host_templates = self.host_template_map()
        host_template_ids = set(template_ids)
        if self.mutation.search_granularity == "host" and not host_template_ids:
            raise ValueError("host search_granularity requires host_templates")
        for rack in _all_rack_specs(self):
            fixed_rank_slots = 0
            fixed_host_units = 0.0
            for host in rack.hosts:
                if host.template_id is not None and host.template_id not in host_template_ids:
                    raise ValueError(f"rack {rack.rack_id} references unknown host template {host.template_id}")
                if host.template_id is not None:
                    template = host_templates[host.template_id]
                    fixed_rank_slots += sum(1 for slot in template.slots if slot.node_type)
                    fixed_host_units += template.rack_units
            if rack.hosts and rack.max_slots == 0 and fixed_rank_slots > 0:
                rack.max_slots = fixed_rank_slots
                if rack.limits.max_slots == 0:
                    rack.limits.max_slots = fixed_rank_slots
            rack_unit_limit = (
                rack.limits.max_rack_units
                if rack.limits.max_rack_units is not None
                else self.limits.max_rack_units
            )
            if rack.hosts and fixed_host_units > rack_unit_limit:
                raise ValueError(
                    f"rack {rack.rack_id} host rack_units exceed limit: "
                    f"{fixed_host_units:.3f} > {rack_unit_limit:.3f}"
                )
        return self

    def host_template_map(self) -> dict[str, HostTemplateSpec]:
        return {template.template_id: template for template in self.host_templates}


def load_component_library(payload: dict) -> ComponentLibrary:
    if "component_library" in payload:
        payload = payload["component_library"]
    return ComponentLibrary.model_validate(payload)


def _all_rack_specs(space: SearchSpace) -> list[RackSpec]:
    racks: list[RackSpec] = []
    for template in space.templates:
        racks.extend(template.racks)
    for archetype in space.rack_archetypes:
        racks.append(archetype.to_rack_spec(archetype.rack_id_prefix or archetype.name))
    return racks
