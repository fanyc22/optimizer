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
        if rack.role == "memory":
            rack.gpu_count = 0
            rack.cpu_count = 0
        else:
            rack.gpu_count = self._clamp_count(
                rack.gpu_count,
                settings.min_gpu_per_rack,
                self._count_limit(rack, "max_gpu_count", settings.max_gpu_per_rack),
            )
            rack.cpu_count = self._clamp_count(
                rack.cpu_count,
                settings.min_cpu_per_rack,
                self._count_limit(rack, "max_cpu_count", settings.max_cpu_per_rack),
            )

        if rack.role == "compute":
            rack.memory_pool_count = 0
        else:
            rack.memory_pool_count = self._clamp_count(
                rack.memory_pool_count,
                settings.min_memory_pools_per_rack,
                self._count_limit(
                    rack,
                    "max_memory_pool_count",
                    settings.max_memory_pools_per_rack,
                ),
            )
        if rack.limits.max_switch_count is not None:
            rack.switch_count = min(rack.switch_count, rack.limits.max_switch_count)
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
        if rack.role == "memory" and rack.memory_pool_count <= 0:
            rack.memory_pool_count = self._minimal_count(
                settings.min_memory_pools_per_rack,
                self._count_limit(
                    rack,
                    "max_memory_pool_count",
                    settings.max_memory_pools_per_rack,
                ),
            )
        elif rack.role == "compute" and rack.gpu_count + rack.cpu_count <= 0:
            self._add_minimal_compute_node(rack, settings)
        elif rack.role == "hybrid":
            if rack.gpu_count + rack.cpu_count <= 0:
                self._add_minimal_compute_node(rack, settings)
            if rack.memory_pool_count <= 0:
                rack.memory_pool_count = self._minimal_count(
                    settings.min_memory_pools_per_rack,
                    self._count_limit(
                        rack,
                        "max_memory_pool_count",
                        settings.max_memory_pools_per_rack,
                    ),
                )
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
            if rack.gpu_count + rack.cpu_count + rack.memory_pool_count <= 0:
                feasible = False
                penalty += 100_000.0
                messages.append(f"{rack.rack_id} is empty after repair")
                continue
            feasible, penalty = self._check_rack_device_limits(
                rack,
                feasible=feasible,
                penalty=penalty,
                messages=messages,
            )
            rack_power = 0.0
            rack_units = 0.0
            for type_name, count in self._rack_type_counts(rack):
                spec = self._library.node_types[type_name]
                rack_power += spec.tdp_watts * count
                rack_units += spec.rack_units * count
            rack_power_limit = (
                rack.limits.max_power_watts
                if rack.limits.max_power_watts is not None
                else limits.max_rack_power_watts
            )
            rack_units_limit = (
                rack.limits.max_rack_units
                if rack.limits.max_rack_units is not None
                else limits.max_rack_units
            )
            if rack_power > rack_power_limit:
                feasible = False
                penalty += rack_power - rack_power_limit
                messages.append(
                    f"{rack.rack_id} power exceeds limit: "
                    f"{rack_power:.3f} > {rack_power_limit:.3f}"
                )
            if rack_units > rack_units_limit:
                feasible = False
                penalty += (rack_units - rack_units_limit) * 100.0
                messages.append(
                    f"{rack.rack_id} rack units exceed limit: "
                    f"{rack_units:.3f} > {rack_units_limit:.3f}"
                )
        return feasible, penalty

    def _check_rack_device_limits(
        self,
        rack: RackGene,
        *,
        feasible: bool,
        penalty: float,
        messages: list[str],
    ) -> tuple[bool, float]:
        checks = [
            ("gpu_count", rack.gpu_count, rack.limits.max_gpu_count),
            ("cpu_count", rack.cpu_count, rack.limits.max_cpu_count),
            ("memory_pool_count", rack.memory_pool_count, rack.limits.max_memory_pool_count),
            ("switch_count", rack.switch_count, rack.limits.max_switch_count),
        ]
        for label, value, maximum in checks:
            if maximum is not None and value > maximum:
                feasible = False
                penalty += (value - maximum) * 1000.0
                messages.append(f"{rack.rack_id} {label} exceeds limit: {value} > {maximum}")
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
            inter_degree = self._inter_rack_degree(chromosome, rack.rack_id)
            if inter_degree > 0:
                inter_lanes = self._link_lanes(chromosome.inter_rack_link_type)
                endpoint_ports += chromosome.inter_rack_link_qty * inter_lanes * inter_degree
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

    def _add_minimal_compute_node(self, rack: RackGene, settings: object) -> None:
        if rack.gpu_type:
            rack.gpu_count = self._minimal_count(
                getattr(settings, "min_gpu_per_rack"),
                self._count_limit(rack, "max_gpu_count", getattr(settings, "max_gpu_per_rack")),
            )
        elif rack.cpu_type:
            rack.cpu_count = self._minimal_count(
                getattr(settings, "min_cpu_per_rack"),
                self._count_limit(rack, "max_cpu_count", getattr(settings, "max_cpu_per_rack")),
            )

    def _minimal_count(self, minimum: int, maximum: int) -> int:
        if maximum <= 0:
            return 0
        return min(maximum, max(1, minimum))

    def _clamp_count(self, value: int, minimum: int, maximum: int) -> int:
        if maximum < minimum:
            return maximum
        return max(minimum, min(maximum, value))

    def _count_limit(self, rack: RackGene, field_name: str, global_maximum: int) -> int:
        rack_limit = getattr(rack.limits, field_name)
        if rack_limit is None:
            return global_maximum
        return min(global_maximum, rack_limit)

    def _inter_rack_degree(self, chromosome: Chromosome, rack_id: str) -> int:
        if chromosome.inter_rack == "none" or len(chromosome.racks) <= 1:
            return 0
        if chromosome.inter_rack == "fully_connected":
            return len(chromosome.racks) - 1
        if chromosome.inter_rack == "ring":
            return 1 if len(chromosome.racks) == 2 else 2
        return 0

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
