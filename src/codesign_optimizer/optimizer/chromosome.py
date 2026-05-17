from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from codesign_optimizer.optimizer.search_space import (
    RackCapacityLimits,
    RackSpec,
    RackTemplate,
    SearchSpace,
)


class RackGene(BaseModel):
    rack_id: str
    role: str = "compute"
    gpu_count: int = Field(ge=0)
    cpu_count: int = Field(ge=0)
    memory_pool_count: int = Field(ge=0)
    switch_count: int = Field(ge=0)
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
    fabric: str = "switch"
    limits: RackCapacityLimits = Field(default_factory=RackCapacityLimits)


class Chromosome(BaseModel):
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
        rack_specs = template.racks or [
            RackSpec(
                rack_id="rack0",
                gpu_count=template.gpu_count,
                cpu_count=template.cpu_count,
                memory_pool_count=template.memory_pool_count,
                switch_count=template.switch_count,
                gpu_type=template.gpu_type,
                cpu_type=template.cpu_type,
                memory_pool_type=template.memory_pool_type,
                switch_type=template.switch_type,
                endpoint_link_type=_require_endpoint_link_type(template),
                gpu_link_type=template.gpu_link_type,
                cpu_link_type=template.cpu_link_type,
                memory_link_type=template.memory_link_type,
                endpoint_link_qty=template.endpoint_link_qty,
                gpu_link_qty=template.gpu_link_qty,
                cpu_link_qty=template.cpu_link_qty,
                memory_link_qty=template.memory_link_qty,
                fabric=template.fabric,
                limits=template.rack_limits or RackCapacityLimits(),
            )
        ]
        for rack in rack_specs:
            if rack.gpu_type and rack.gpu_type not in gpu:
                gpu.append(rack.gpu_type)
            if rack.cpu_type and rack.cpu_type not in cpu:
                cpu.append(rack.cpu_type)
            if rack.memory_pool_type and rack.memory_pool_type not in memory:
                memory.append(rack.memory_pool_type)
            if rack.switch_type and rack.switch_type not in switch:
                switch.append(rack.switch_type)

    return TypePools(
        gpu=sorted(set(gpu)),
        cpu=sorted(set(cpu)),
        memory=sorted(set(memory)),
        switch=sorted(set(switch)),
        link=sorted(link_types.keys()),
    )


def chromosome_from_template(template: RackTemplate) -> Chromosome:
    if template.racks:
        racks = [_rack_gene_from_spec(rack) for rack in template.racks]
        inter_rack_link_type = (
            template.inter_rack_link_type
            or template.endpoint_link_type
            or racks[0].endpoint_link_type
        )
    else:
        endpoint_link_type = _require_endpoint_link_type(template)
        racks = [
            RackGene(
                rack_id=f"rack{idx}",
                role=_infer_rack_role(
                    template.gpu_count,
                    template.cpu_count,
                    template.memory_pool_count,
                ),
                gpu_count=template.gpu_count,
                cpu_count=template.cpu_count,
                memory_pool_count=template.memory_pool_count,
                switch_count=template.switch_count,
                gpu_type=template.gpu_type,
                cpu_type=template.cpu_type,
                memory_pool_type=template.memory_pool_type,
                switch_type=template.switch_type,
                endpoint_link_type=endpoint_link_type,
                gpu_link_type=template.gpu_link_type or endpoint_link_type,
                cpu_link_type=template.cpu_link_type or endpoint_link_type,
                memory_link_type=template.memory_link_type or endpoint_link_type,
                endpoint_link_qty=template.endpoint_link_qty,
                gpu_link_qty=template.gpu_link_qty or template.endpoint_link_qty,
                cpu_link_qty=template.cpu_link_qty or template.endpoint_link_qty,
                memory_link_qty=template.memory_link_qty,
                fabric=template.fabric,
                limits=template.rack_limits or RackCapacityLimits(),
            )
            for idx in range(template.rack_count)
        ]
        inter_rack_link_type = template.inter_rack_link_type or endpoint_link_type
    return Chromosome(
        template_name=template.name,
        racks=racks,
        inter_rack=template.inter_rack,
        inter_rack_link_type=inter_rack_link_type,
        inter_rack_link_qty=template.inter_rack_link_qty,
    )


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
    settings = space.mutation
    for _ in range(max(1, intensity)):
        rack = rng.choice(result.racks)
        op = rng.choice(_mutation_ops_for_rack(rack))
        if op == "gpu_count":
            rack.gpu_count = _mutate_int(
                rack.gpu_count,
                settings.min_gpu_per_rack,
                _count_limit(rack, "max_gpu_count", settings.max_gpu_per_rack),
                rng,
            )
        elif op == "cpu_count":
            rack.cpu_count = _mutate_int(
                rack.cpu_count,
                settings.min_cpu_per_rack,
                _count_limit(rack, "max_cpu_count", settings.max_cpu_per_rack),
                rng,
            )
        elif op == "memory_count":
            rack.memory_pool_count = _mutate_int(
                rack.memory_pool_count,
                settings.min_memory_pools_per_rack,
                _count_limit(
                    rack,
                    "max_memory_pool_count",
                    settings.max_memory_pools_per_rack,
                ),
                rng,
            )
        elif op == "endpoint_qty":
            rack.endpoint_link_qty = _mutate_int(
                rack.endpoint_link_qty,
                settings.min_endpoint_link_qty,
                settings.max_endpoint_link_qty,
                rng,
            )
            if rack.gpu_link_qty is not None:
                rack.gpu_link_qty = _mutate_int(
                    rack.gpu_link_qty,
                    settings.min_endpoint_link_qty,
                    settings.max_endpoint_link_qty,
                    rng,
                )
            if rack.cpu_link_qty is not None:
                rack.cpu_link_qty = _mutate_int(
                    rack.cpu_link_qty,
                    settings.min_endpoint_link_qty,
                    settings.max_endpoint_link_qty,
                    rng,
                )
        elif op == "memory_qty":
            rack.memory_link_qty = _mutate_int(
                rack.memory_link_qty,
                settings.min_endpoint_link_qty,
                settings.max_endpoint_link_qty,
                rng,
            )
        elif op == "inter_qty":
            result.inter_rack_link_qty = _mutate_int(
                result.inter_rack_link_qty,
                settings.min_inter_rack_link_qty,
                settings.max_inter_rack_link_qty,
                rng,
            )
        elif op == "inter_mode":
            result.inter_rack = rng.choice(["none", "ring", "fully_connected"])

        _repair_role_minimums(rack, space)
        if rack.fabric == "switch" and rack.switch_count == 0:
            rack.switch_count = 1
    return result


def crossover(left: Chromosome, right: Chromosome, rng: random.Random) -> Chromosome:
    if not left.racks or not right.racks:
        return left.model_copy(deep=True)
    child = left.model_copy(deep=True)
    for idx, rack in enumerate(child.racks):
        donor = _matching_rack(rack, right.racks, idx)
        if donor is not None and rng.random() < 0.5:
            child.racks[idx] = donor.model_copy(deep=True)
    if rng.random() < 0.5:
        child.inter_rack = right.inter_rack
        child.inter_rack_link_type = right.inter_rack_link_type
        child.inter_rack_link_qty = right.inter_rack_link_qty
    return child


def _mutate_int(value: int, minimum: int, maximum: int, rng: random.Random) -> int:
    if maximum < minimum:
        return maximum
    if minimum == maximum:
        return minimum
    step = rng.choice([-1, 1])
    return max(minimum, min(maximum, value + step))


def _rack_gene_from_spec(rack: RackSpec) -> RackGene:
    return RackGene(
        rack_id=rack.rack_id,
        role=rack.role or _infer_rack_role(rack.gpu_count, rack.cpu_count, rack.memory_pool_count),
        gpu_count=rack.gpu_count,
        cpu_count=rack.cpu_count,
        memory_pool_count=rack.memory_pool_count,
        switch_count=rack.switch_count,
        gpu_type=rack.gpu_type,
        cpu_type=rack.cpu_type,
        memory_pool_type=rack.memory_pool_type,
        switch_type=rack.switch_type,
        endpoint_link_type=rack.endpoint_link_type,
        gpu_link_type=rack.gpu_link_type or rack.endpoint_link_type,
        cpu_link_type=rack.cpu_link_type or rack.endpoint_link_type,
        memory_link_type=rack.memory_link_type or rack.endpoint_link_type,
        endpoint_link_qty=rack.endpoint_link_qty,
        gpu_link_qty=rack.gpu_link_qty or rack.endpoint_link_qty,
        cpu_link_qty=rack.cpu_link_qty or rack.endpoint_link_qty,
        memory_link_qty=rack.memory_link_qty,
        fabric=rack.fabric,
        limits=rack.limits,
    )


def _infer_rack_role(gpu_count: int, cpu_count: int, memory_pool_count: int) -> str:
    compute_count = gpu_count + cpu_count
    if compute_count > 0 and memory_pool_count > 0:
        return "hybrid"
    if memory_pool_count > 0:
        return "memory"
    return "compute"


def _require_endpoint_link_type(template: RackTemplate) -> str:
    if template.endpoint_link_type is None:
        raise ValueError(f"template {template.name} must set endpoint_link_type")
    return template.endpoint_link_type


def _mutation_ops_for_rack(rack: RackGene) -> list[str]:
    ops = ["endpoint_qty", "inter_qty", "inter_mode"]
    if rack.role in {"compute", "hybrid"}:
        if rack.gpu_type:
            ops.append("gpu_count")
        if rack.cpu_type:
            ops.append("cpu_count")
    if rack.role in {"memory", "hybrid"} and rack.memory_pool_type:
        ops.extend(["memory_count", "memory_qty"])
    return ops


def _repair_role_minimums(rack: RackGene, space: SearchSpace) -> None:
    settings = space.mutation
    if rack.role == "memory":
        rack.gpu_count = 0
        rack.cpu_count = 0
        if rack.memory_pool_count == 0:
            rack.memory_pool_count = _minimal_count(
                settings.min_memory_pools_per_rack,
                _count_limit(
                    rack,
                    "max_memory_pool_count",
                    settings.max_memory_pools_per_rack,
                ),
            )
        return

    if rack.role == "compute":
        rack.memory_pool_count = 0
        _ensure_compute_node(rack, space)
        return

    _ensure_compute_node(rack, space)
    if rack.memory_pool_count == 0:
        rack.memory_pool_count = _minimal_count(
            settings.min_memory_pools_per_rack,
            _count_limit(
                rack,
                "max_memory_pool_count",
                settings.max_memory_pools_per_rack,
            ),
        )


def _ensure_compute_node(rack: RackGene, space: SearchSpace) -> None:
    settings = space.mutation
    if rack.gpu_count + rack.cpu_count > 0:
        return
    if rack.gpu_type:
        rack.gpu_count = _minimal_count(
            settings.min_gpu_per_rack,
            _count_limit(rack, "max_gpu_count", settings.max_gpu_per_rack),
        )
    elif rack.cpu_type:
        rack.cpu_count = _minimal_count(
            settings.min_cpu_per_rack,
            _count_limit(rack, "max_cpu_count", settings.max_cpu_per_rack),
        )


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


def _minimal_count(minimum: int, maximum: int) -> int:
    if maximum <= 0:
        return 0
    return min(maximum, max(1, minimum))


def _count_limit(rack: RackGene, field_name: str, global_maximum: int) -> int:
    rack_limit = getattr(rack.limits, field_name)
    if rack_limit is None:
        return global_maximum
    return min(global_maximum, rack_limit)
