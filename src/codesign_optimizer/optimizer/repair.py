from __future__ import annotations

from dataclasses import dataclass

from codesign_optimizer.models.hardware import ComponentLibrary
from codesign_optimizer.optimizer.chromosome import Chromosome, RackGene
from codesign_optimizer.optimizer.exporter import HardwareTopologyExporter, node_role
from codesign_optimizer.optimizer.search_space import SearchLimits, SearchSpace


@dataclass(frozen=True)
class RepairReport:
    chromosome: Chromosome
    feasible: bool
    messages: list[str]
    estimated_cost: float
    estimated_power_watts: float
    penalty: float


class CandidateRepairer:
    def __init__(self, component_library: ComponentLibrary, search_space: SearchSpace) -> None:
        self._library = component_library
        self._space = search_space

    def repair_and_validate(self, chromosome: Chromosome) -> RepairReport:
        repaired = chromosome.model_copy(deep=True)
        messages: list[str] = []
        for rack in repaired.racks:
            self._repair_rack(rack, messages)

        feasible = True
        penalty = 0.0
        try:
            exported = HardwareTopologyExporter(self._library).export(repaired)
        except Exception as exc:
            return RepairReport(
                chromosome=repaired,
                feasible=False,
                messages=[f"export failed: {exc}"],
                estimated_cost=0.0,
                estimated_power_watts=0.0,
                penalty=1_000_000.0,
            )

        cost = exported.proposal.total_estimated_cost()
        power = exported.proposal.estimated_peak_power()
        limits = self._space.limits
        if exported.rank_count <= 0:
            feasible = False
            messages.append("candidate has no rank compute nodes")
            penalty += 100_000.0

        feasible, penalty = self._check_limits(
            cost=cost,
            power=power,
            limits=limits,
            feasible=feasible,
            penalty=penalty,
            messages=messages,
        )
        feasible, penalty = self._check_racks(
            repaired,
            limits,
            feasible=feasible,
            penalty=penalty,
            messages=messages,
        )
        feasible, penalty = self._check_switch_radix(
            repaired,
            feasible=feasible,
            penalty=penalty,
            messages=messages,
        )
        return RepairReport(
            chromosome=repaired,
            feasible=feasible,
            messages=messages,
            estimated_cost=cost,
            estimated_power_watts=power,
            penalty=penalty,
        )

    def _repair_rack(self, rack: RackGene, messages: list[str]) -> None:
        settings = self._space.mutation
        original = rack.model_copy(deep=True)
        rack.gpu_count = max(settings.min_gpu_per_rack, min(settings.max_gpu_per_rack, rack.gpu_count))
        rack.cpu_count = max(settings.min_cpu_per_rack, min(settings.max_cpu_per_rack, rack.cpu_count))
        rack.memory_pool_count = max(
            settings.min_memory_pools_per_rack,
            min(settings.max_memory_pools_per_rack, rack.memory_pool_count),
        )
        rack.endpoint_link_qty = max(
            settings.min_endpoint_link_qty,
            min(settings.max_endpoint_link_qty, rack.endpoint_link_qty),
        )
        if rack.gpu_link_qty is not None:
            rack.gpu_link_qty = max(
                settings.min_endpoint_link_qty,
                min(settings.max_endpoint_link_qty, rack.gpu_link_qty),
            )
        if rack.cpu_link_qty is not None:
            rack.cpu_link_qty = max(
                settings.min_endpoint_link_qty,
                min(settings.max_endpoint_link_qty, rack.cpu_link_qty),
            )
        rack.memory_link_qty = max(
            settings.min_endpoint_link_qty,
            min(settings.max_endpoint_link_qty, rack.memory_link_qty),
        )
        if rack.gpu_count + rack.cpu_count <= 0:
            rack.gpu_count = 1
        if rack.fabric == "switch" and rack.switch_count <= 0:
            rack.switch_count = 1
        if rack.model_dump() != original.model_dump():
            messages.append(f"repaired rack bounds for {rack.rack_id}")

    def _check_limits(
        self,
        *,
        cost: float,
        power: float,
        limits: SearchLimits,
        feasible: bool,
        penalty: float,
        messages: list[str],
    ) -> tuple[bool, float]:
        if cost > limits.max_total_cost:
            feasible = False
            excess = cost - limits.max_total_cost
            penalty += excess
            messages.append(f"cost exceeds limit: {cost:.3f} > {limits.max_total_cost:.3f}")
        if power > limits.max_peak_power_watts:
            feasible = False
            excess = power - limits.max_peak_power_watts
            penalty += excess
            messages.append(
                f"peak power exceeds limit: {power:.3f} > {limits.max_peak_power_watts:.3f}"
            )
        return feasible, penalty

    def _check_racks(
        self,
        chromosome: Chromosome,
        limits: SearchLimits,
        *,
        feasible: bool,
        penalty: float,
        messages: list[str],
    ) -> tuple[bool, float]:
        for rack in chromosome.racks:
            rack_power = 0.0
            rack_units = 0.0
            for type_name, count in self._rack_type_counts(rack):
                spec = self._library.node_types[type_name]
                rack_power += spec.tdp_watts * count
                rack_units += spec.rack_units * count
            if rack_power > limits.max_rack_power_watts:
                feasible = False
                penalty += rack_power - limits.max_rack_power_watts
                messages.append(
                    f"{rack.rack_id} power exceeds limit: "
                    f"{rack_power:.3f} > {limits.max_rack_power_watts:.3f}"
                )
            if rack_units > limits.max_rack_units:
                feasible = False
                penalty += (rack_units - limits.max_rack_units) * 100.0
                messages.append(
                    f"{rack.rack_id} rack units exceed limit: "
                    f"{rack_units:.3f} > {limits.max_rack_units:.3f}"
                )
        return feasible, penalty

    def _check_switch_radix(
        self,
        chromosome: Chromosome,
        *,
        feasible: bool,
        penalty: float,
        messages: list[str],
    ) -> tuple[bool, float]:
        for rack in chromosome.racks:
            if rack.fabric != "switch" or not rack.switch_type:
                continue
            switch_spec = self._library.node_types[rack.switch_type]
            radix = switch_spec.radix
            if radix is None:
                continue
            gpu_link_qty = rack.gpu_link_qty or rack.endpoint_link_qty
            cpu_link_qty = rack.cpu_link_qty or rack.endpoint_link_qty
            gpu_link_lanes = self._link_lanes(rack.gpu_link_type or rack.endpoint_link_type)
            cpu_link_lanes = self._link_lanes(rack.cpu_link_type or rack.endpoint_link_type)
            memory_link_lanes = self._link_lanes(rack.memory_link_type or rack.endpoint_link_type)
            endpoint_ports = (
                rack.gpu_count * gpu_link_qty * gpu_link_lanes
                + rack.cpu_count * cpu_link_qty * cpu_link_lanes
                + rack.memory_pool_count * rack.memory_link_qty * memory_link_lanes
            )
            if chromosome.inter_rack != "none" and len(chromosome.racks) > 1:
                inter_lanes = self._link_lanes(chromosome.inter_rack_link_type)
                endpoint_ports += chromosome.inter_rack_link_qty * inter_lanes * 2
            per_switch = endpoint_ports / max(rack.switch_count, 1)
            if per_switch > radix:
                feasible = False
                penalty += (per_switch - radix) * 1000.0
                messages.append(
                    f"{rack.rack_id} switch radix exceeded: {per_switch:.3f} > {radix}"
                )
        return feasible, penalty

    def _link_lanes(self, link_type: str | None) -> int:
        if link_type is None or link_type not in self._library.link_types:
            return 1
        return self._library.link_types[link_type].lanes

    def _rack_type_counts(self, rack: RackGene) -> list[tuple[str, int]]:
        result: list[tuple[str, int]] = []
        for type_name, count in [
            (rack.gpu_type, rack.gpu_count),
            (rack.cpu_type, rack.cpu_count),
            (rack.memory_pool_type, rack.memory_pool_count),
            (rack.switch_type, rack.switch_count),
        ]:
            if type_name and count:
                if type_name not in self._library.node_types:
                    result.append((type_name, count))
                    continue
                role = node_role(type_name, self._library.node_types[type_name])
                if role in {"gpu", "cpu", "memory", "switch"}:
                    result.append((type_name, count))
        return result
