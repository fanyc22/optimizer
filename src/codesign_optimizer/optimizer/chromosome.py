from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from codesign_optimizer.optimizer.search_space import (
    RackArchetype,
    RackCapacityLimits,
    RackSpec,
    RackTemplate,
    SearchSpace,
    SlotSpec,
)


class SlotGene(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot_id: str
    node_type: str | None = None
    link_type: str | None = None
    link_qty: int | None = Field(default=None, ge=1)

    @classmethod
    def from_spec(cls, slot: SlotSpec) -> "SlotGene":
        return cls(
            slot_id=slot.slot_id,
            node_type=slot.node_type,
            link_type=slot.link_type,
            link_qty=slot.link_qty,
        )


class RackGene(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rack_id: str
    role: str = "compute"
    optional: bool = False
    active: bool = True
    dynamic: bool = False
    origin: str = "seed"
    activation_alpha: float | None = Field(default=None, ge=0)
    max_slots: int = Field(ge=0)
    slots: list[SlotGene] = Field(default_factory=list)
    memory_pool_count: int = Field(default=0, ge=0)
    switch_count: int = Field(default=1, ge=0)
    memory_pool_type: str | None = None
    switch_type: str | None = None
    intra_rack_topology: str = "switch"
    intra_rack_link_type: str | None = None
    intra_rack_link_qty: int = Field(default=1, ge=1)
    memory_link_type: str | None = None
    memory_link_qty: int = Field(default=1, ge=1)
    limits: RackCapacityLimits = Field(default_factory=RackCapacityLimits)

    @property
    def occupied_slots(self) -> list[SlotGene]:
        return [slot for slot in self.slots if slot.node_type]

    @property
    def free_slots(self) -> list[SlotGene]:
        return [slot for slot in self.slots if not slot.node_type]

    @property
    def gpu_count(self) -> int:
        return sum(1 for slot in self.occupied_slots if role_of_type(slot.node_type or "", None) == "gpu")

    @gpu_count.setter
    def gpu_count(self, value: int) -> None:
        self._set_role_count("gpu", value)

    @property
    def cpu_count(self) -> int:
        return sum(1 for slot in self.occupied_slots if role_of_type(slot.node_type or "", None) == "cpu")

    @cpu_count.setter
    def cpu_count(self, value: int) -> None:
        self._set_role_count("cpu", value)

    @property
    def gpu_type(self) -> str | None:
        return self._first_role_type("gpu")

    @gpu_type.setter
    def gpu_type(self, value: str | None) -> None:
        self._set_role_type("gpu", value)

    @property
    def cpu_type(self) -> str | None:
        return self._first_role_type("cpu")

    @cpu_type.setter
    def cpu_type(self, value: str | None) -> None:
        self._set_role_type("cpu", value)

    @property
    def endpoint_link_type(self) -> str | None:
        return self.intra_rack_link_type

    @endpoint_link_type.setter
    def endpoint_link_type(self, value: str | None) -> None:
        self.intra_rack_link_type = value

    @property
    def endpoint_link_qty(self) -> int:
        return self.intra_rack_link_qty

    @endpoint_link_qty.setter
    def endpoint_link_qty(self, value: int) -> None:
        self.intra_rack_link_qty = value

    @property
    def gpu_link_type(self) -> str | None:
        return self._first_role_link_type("gpu")

    @gpu_link_type.setter
    def gpu_link_type(self, value: str | None) -> None:
        self._set_role_link_type("gpu", value)

    @property
    def cpu_link_type(self) -> str | None:
        return self._first_role_link_type("cpu")

    @cpu_link_type.setter
    def cpu_link_type(self, value: str | None) -> None:
        self._set_role_link_type("cpu", value)

    @property
    def gpu_link_qty(self) -> int | None:
        return self._first_role_link_qty("gpu")

    @gpu_link_qty.setter
    def gpu_link_qty(self, value: int | None) -> None:
        if value is not None:
            self._set_role_link_qty("gpu", value)

    @property
    def cpu_link_qty(self) -> int | None:
        return self._first_role_link_qty("cpu")

    @cpu_link_qty.setter
    def cpu_link_qty(self, value: int | None) -> None:
        if value is not None:
            self._set_role_link_qty("cpu", value)

    @property
    def fabric(self) -> str:
        return self.intra_rack_topology

    @fabric.setter
    def fabric(self, value: str) -> None:
        self.intra_rack_topology = value

    def _first_role_type(self, role: str) -> str | None:
        for slot in self.occupied_slots:
            if role_of_type(slot.node_type or "", None) == role:
                return slot.node_type
        return None

    def _set_role_type(self, role: str, value: str | None) -> None:
        for slot in self.occupied_slots:
            if role_of_type(slot.node_type or "", None) == role:
                slot.node_type = value

    def _first_role_link_qty(self, role: str) -> int | None:
        for slot in self.occupied_slots:
            if role_of_type(slot.node_type or "", None) == role:
                return slot.link_qty
        return None

    def _first_role_link_type(self, role: str) -> str | None:
        for slot in self.occupied_slots:
            if role_of_type(slot.node_type or "", None) == role:
                return slot.link_type
        return None

    def _set_role_link_qty(self, role: str, value: int) -> None:
        for slot in self.occupied_slots:
            if role_of_type(slot.node_type or "", None) == role:
                slot.link_qty = value

    def _set_role_link_type(self, role: str, value: str | None) -> None:
        for slot in self.occupied_slots:
            if role_of_type(slot.node_type or "", None) == role:
                slot.link_type = value

    def _set_role_count(self, role: str, value: int) -> None:
        value = max(0, value)
        role_slots = [slot for slot in self.occupied_slots if role_of_type(slot.node_type or "", None) == role]
        current = len(role_slots)
        if value < current:
            for slot in role_slots[value:]:
                slot.node_type = None
                slot.link_type = None
                slot.link_qty = None
            return
        if value == current:
            return
        type_name = self._first_role_type(role)
        if type_name is None:
            return
        for slot in self.free_slots[: value - current]:
            slot.node_type = type_name
            slot.link_type = slot.link_type or self.intra_rack_link_type
            slot.link_qty = self.intra_rack_link_qty


class Chromosome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    template_name: str
    racks: list[RackGene]
    inter_rack: str = "ring"
    inter_rack_link_type: str | None = None
    inter_rack_link_qty: int = Field(default=1, ge=1)

    def signature(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


@dataclass(frozen=True)
class TypePools:
    gpu: list[str]
    cpu: list[str]
    memory: list[str]
    switch: list[str]
    link: list[str]


def role_of_type(type_name: str, role: str | None) -> str:
    text = f"{role or ''} {type_name}".lower()
    if "switch" in text or "router" in text or "spine" in text or "tor" in text:
        return "switch"
    if "mem" in text or "pool" in text or "cxl_type3" in text:
        return "memory"
    if "cpu" in text:
        return "cpu"
    return "gpu"


def infer_type_pools(space: SearchSpace, node_types: dict[str, Any], link_types: dict[str, Any]) -> TypePools:
    gpu: list[str] = []
    cpu: list[str] = []
    memory: list[str] = []
    switch: list[str] = []
    for name, spec in node_types.items():
        role = role_of_type(name, getattr(spec, "role", None))
        if role == "switch":
            switch.append(name)
        elif role == "memory":
            memory.append(name)
        elif role == "cpu":
            cpu.append(name)
        else:
            gpu.append(name)

    for template in space.templates:
        for rack in template.racks:
            _add_rack_types_to_pools(rack, gpu, cpu, memory, switch)
    for archetype in space.rack_archetypes:
        _add_rack_types_to_pools(
            archetype.to_rack_spec(archetype.rack_id_prefix or archetype.name),
            gpu,
            cpu,
            memory,
            switch,
        )

    return TypePools(
        gpu=sorted(set(gpu)),
        cpu=sorted(set(cpu)),
        memory=sorted(set(memory)),
        switch=sorted(set(switch)),
        link=sorted(link_types.keys()),
    )


def chromosome_from_template(template: RackTemplate) -> Chromosome:
    return _sanitize_topologies(
        Chromosome(
            template_name=template.name,
            racks=[_rack_gene_from_spec(rack, dynamic=False) for rack in template.racks],
            inter_rack=template.inter_rack,
            inter_rack_link_type=template.inter_rack_link_type,
            inter_rack_link_qty=template.inter_rack_link_qty,
        )
    )


def rack_gene_from_archetype(archetype: RackArchetype, rack_id: str) -> RackGene:
    rack = _rack_gene_from_spec(archetype.to_rack_spec(rack_id, origin="dynamic"), dynamic=True)
    rack.optional = False
    rack.active = True
    rack.origin = "dynamic"
    _sanitize_rack_topology(rack)
    return rack


def _sanitize_topologies(chromosome: Chromosome) -> Chromosome:
    if chromosome.inter_rack == "none":
        chromosome.inter_rack = "ring"
    for rack in chromosome.racks:
        _sanitize_rack_topology(rack)
    return chromosome


def _sanitize_rack_topology(rack: RackGene) -> None:
    if rack.intra_rack_topology != "none":
        return
    max_switches = rack.limits.max_switch_count
    if rack.switch_type and (max_switches is None or max_switches > 0):
        rack.intra_rack_topology = "switch"
        if rack.switch_count <= 0:
            rack.switch_count = 1
    else:
        rack.intra_rack_topology = "ring"


def initial_population(space: SearchSpace, population_size: int, rng: random.Random) -> list[Chromosome]:
    base = [chromosome_from_template(template) for template in space.templates]
    if not base:
        return []
    population: list[Chromosome] = []
    while len(population) < population_size:
        candidate = base[len(population) % len(base)].model_copy(deep=True)
        if len(population) >= len(base):
            candidate = mutate_random(candidate, space, rng, intensity=1 + len(population) % 3)
        population.append(candidate)
    return population


def mutate_random(
    chromosome: Chromosome,
    space: SearchSpace,
    rng: random.Random,
    *,
    intensity: int = 1,
) -> Chromosome:
    result = chromosome.model_copy(deep=True)
    pools = _type_pools_from_space(space)
    for _ in range(max(1, intensity)):
        active_racks = [rack for rack in result.racks if rack.active or not rack.optional]
        if not active_racks:
            break
        rack = rng.choice(active_racks)
        ops = _mutation_ops_for_rack(rack, pools, space)
        if not ops:
            continue
        op = rng.choice(ops)
        if op == "add_node":
            slot = rng.choice(rack.free_slots)
            choices = pools.gpu + pools.cpu
            if choices:
                slot.node_type = rng.choice(choices)
                slot.link_type = slot.link_type or rack.intra_rack_link_type
                slot.link_qty = rack.intra_rack_link_qty
        elif op == "remove_node":
            slot = rng.choice(rack.occupied_slots)
            slot.node_type = None
            slot.link_type = None
            slot.link_qty = None
        elif op == "replace_node":
            slot = rng.choice(rack.occupied_slots)
            choices = [item for item in pools.gpu + pools.cpu if item != slot.node_type]
            if choices:
                slot.node_type = rng.choice(choices)
        elif op == "intra_qty":
            rack.intra_rack_link_qty = _mutate_int(
                rack.intra_rack_link_qty,
                space.mutation.min_intra_rack_link_qty,
                space.mutation.max_intra_rack_link_qty,
                rng,
            )
        elif op == "inter_qty":
            result.inter_rack_link_qty = _mutate_int(
                result.inter_rack_link_qty,
                space.mutation.min_inter_rack_link_qty,
                space.mutation.max_inter_rack_link_qty,
                rng,
            )
        elif op == "intra_mode":
            rack.intra_rack_topology = rng.choice(["ring", "fully_connected", "switch"])
            if rack.intra_rack_topology == "switch" and rack.switch_count <= 0:
                rack.switch_count = 1
        elif op == "inter_mode":
            result.inter_rack = rng.choice(["ring", "fully_connected"])
        elif op == "remove_rack":
            result.racks = [item for item in result.racks if item.rack_id != rack.rack_id]
    return _sanitize_topologies(result)


def crossover(left: Chromosome, right: Chromosome, rng: random.Random) -> Chromosome:
    if not left.racks or not right.racks:
        return _sanitize_topologies(left.model_copy(deep=True))
    child = left.model_copy(deep=True)
    for idx, rack in enumerate(child.racks):
        donor = _matching_rack(rack, right.racks, idx)
        if donor is not None and rng.random() < 0.5:
            child.racks[idx] = donor.model_copy(deep=True)
    if rng.random() < 0.5:
        child.inter_rack = right.inter_rack
        child.inter_rack_link_type = right.inter_rack_link_type
        child.inter_rack_link_qty = right.inter_rack_link_qty
    return _sanitize_topologies(child)


def _rack_gene_from_spec(rack: RackSpec, *, dynamic: bool) -> RackGene:
    return RackGene(
        rack_id=rack.rack_id,
        role=rack.role,
        optional=rack.optional,
        active=rack.active,
        dynamic=dynamic,
        origin=rack.origin,
        activation_alpha=rack.activation_alpha,
        max_slots=rack.max_slots,
        slots=[SlotGene.from_spec(slot) for slot in rack.slots],
        memory_pool_count=rack.memory_pool_count,
        switch_count=rack.switch_count,
        memory_pool_type=rack.memory_pool_type,
        switch_type=rack.switch_type,
        intra_rack_topology=rack.intra_rack_topology,
        intra_rack_link_type=rack.intra_rack_link_type,
        intra_rack_link_qty=rack.intra_rack_link_qty,
        memory_link_type=rack.memory_link_type,
        memory_link_qty=rack.memory_link_qty,
        limits=rack.limits.model_copy(deep=True),
    )


def _add_rack_types_to_pools(
    rack: RackSpec,
    gpu: list[str],
    cpu: list[str],
    memory: list[str],
    switch: list[str],
) -> None:
    for slot in rack.slots:
        if not slot.node_type:
            continue
        role = role_of_type(slot.node_type, None)
        if role == "cpu" and slot.node_type not in cpu:
            cpu.append(slot.node_type)
        elif role == "gpu" and slot.node_type not in gpu:
            gpu.append(slot.node_type)
    if rack.memory_pool_type and rack.memory_pool_type not in memory:
        memory.append(rack.memory_pool_type)
    if rack.switch_type and rack.switch_type not in switch:
        switch.append(rack.switch_type)


def _type_pools_from_space(space: SearchSpace) -> TypePools:
    gpu: list[str] = []
    cpu: list[str] = []
    memory: list[str] = []
    switch: list[str] = []
    for template in space.templates:
        for rack in template.racks:
            _add_rack_types_to_pools(rack, gpu, cpu, memory, switch)
    for archetype in space.rack_archetypes:
        _add_rack_types_to_pools(archetype.to_rack_spec(archetype.name), gpu, cpu, memory, switch)
    return TypePools(sorted(set(gpu)), sorted(set(cpu)), sorted(set(memory)), sorted(set(switch)), [])


def _mutation_ops_for_rack(rack: RackGene, pools: TypePools, space: SearchSpace) -> list[str]:
    ops = ["inter_qty", "inter_mode", "intra_qty", "intra_mode"]
    if rack.free_slots and (pools.gpu or pools.cpu):
        ops.append("add_node")
    if len(rack.occupied_slots) > 1:
        ops.extend(["replace_node", "remove_node"])
    elif rack.occupied_slots:
        ops.append("replace_node")
    if rack.dynamic or (rack.origin == "seed" and space.mutation.allow_remove_initial_racks):
        ops.append("remove_rack")
    return ops


def _mutate_int(value: int, minimum: int, maximum: int, rng: random.Random) -> int:
    if maximum < minimum:
        return maximum
    if minimum == maximum:
        return minimum
    step = rng.choice([-1, 1])
    return max(minimum, min(maximum, value + step))


def _matching_rack(target: RackGene, candidates: list[RackGene], index: int) -> RackGene | None:
    for candidate in candidates:
        if candidate.rack_id == target.rack_id:
            return candidate
    same_role = [candidate for candidate in candidates if candidate.role == target.role]
    if same_role:
        return same_role[index % len(same_role)]
    if index < len(candidates):
        return candidates[index]
    return None
