from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from codesign_optimizer.optimizer.search_space import RackTemplate, SearchSpace


class RackGene(BaseModel):
    rack_id: str
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
        if template.gpu_type and template.gpu_type not in gpu:
            gpu.append(template.gpu_type)
        if template.cpu_type and template.cpu_type not in cpu:
            cpu.append(template.cpu_type)
        if template.memory_pool_type and template.memory_pool_type not in memory:
            memory.append(template.memory_pool_type)
        if template.switch_type and template.switch_type not in switch:
            switch.append(template.switch_type)

    return TypePools(
        gpu=sorted(set(gpu)),
        cpu=sorted(set(cpu)),
        memory=sorted(set(memory)),
        switch=sorted(set(switch)),
        link=sorted(link_types.keys()),
    )


def chromosome_from_template(template: RackTemplate) -> Chromosome:
    racks = [
        RackGene(
            rack_id=f"rack{idx}",
            gpu_count=template.gpu_count,
            cpu_count=template.cpu_count,
            memory_pool_count=template.memory_pool_count,
            switch_count=template.switch_count,
            gpu_type=template.gpu_type,
            cpu_type=template.cpu_type,
            memory_pool_type=template.memory_pool_type,
            switch_type=template.switch_type,
            endpoint_link_type=template.endpoint_link_type,
            gpu_link_type=template.gpu_link_type or template.endpoint_link_type,
            cpu_link_type=template.cpu_link_type or template.endpoint_link_type,
            memory_link_type=template.memory_link_type or template.endpoint_link_type,
            endpoint_link_qty=template.endpoint_link_qty,
            gpu_link_qty=template.gpu_link_qty or template.endpoint_link_qty,
            cpu_link_qty=template.cpu_link_qty or template.endpoint_link_qty,
            memory_link_qty=template.memory_link_qty,
            fabric=template.fabric,
        )
        for idx in range(template.rack_count)
    ]
    return Chromosome(
        template_name=template.name,
        racks=racks,
        inter_rack=template.inter_rack,
        inter_rack_link_type=template.inter_rack_link_type or template.endpoint_link_type,
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
        op = rng.choice(
            [
                "gpu_count",
                "cpu_count",
                "memory_count",
                "endpoint_qty",
                "memory_qty",
                "inter_qty",
                "inter_mode",
            ]
        )
        if op == "gpu_count":
            rack.gpu_count = _mutate_int(
                rack.gpu_count, settings.min_gpu_per_rack, settings.max_gpu_per_rack, rng
            )
        elif op == "cpu_count":
            rack.cpu_count = _mutate_int(
                rack.cpu_count, settings.min_cpu_per_rack, settings.max_cpu_per_rack, rng
            )
        elif op == "memory_count":
            rack.memory_pool_count = _mutate_int(
                rack.memory_pool_count,
                settings.min_memory_pools_per_rack,
                settings.max_memory_pools_per_rack,
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

        if rack.gpu_count + rack.cpu_count == 0:
            rack.gpu_count = 1
        if rack.fabric == "switch" and rack.switch_count == 0:
            rack.switch_count = 1
    return result


def crossover(left: Chromosome, right: Chromosome, rng: random.Random) -> Chromosome:
    if not left.racks or not right.racks:
        return left.model_copy(deep=True)
    child = left.model_copy(deep=True)
    for idx, rack in enumerate(child.racks):
        if idx < len(right.racks) and rng.random() < 0.5:
            child.racks[idx] = right.racks[idx].model_copy(deep=True)
    if rng.random() < 0.5:
        child.inter_rack = right.inter_rack
        child.inter_rack_link_type = right.inter_rack_link_type
        child.inter_rack_link_qty = right.inter_rack_link_qty
    return child


def _mutate_int(value: int, minimum: int, maximum: int, rng: random.Random) -> int:
    if minimum == maximum:
        return minimum
    step = rng.choice([-1, 1])
    return max(minimum, min(maximum, value + step))
