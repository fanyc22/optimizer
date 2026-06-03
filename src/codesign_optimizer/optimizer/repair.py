from __future__ import annotations

from dataclasses import dataclass

from codesign_optimizer.models.hardware import ComponentLibrary
from codesign_optimizer.optimizer.chromosome import Chromosome, RackGene, role_of_type
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
        rack_ids = [rack.rack_id for rack in repaired.racks]
        if len(set(rack_ids)) != len(rack_ids):
            return RepairReport(
                chromosome=repaired,
                feasible=False,
                messages=["candidate has duplicate rack ids"],
                estimated_cost=0.0,
                estimated_power_watts=0.0,
                penalty=1_000_000.0,
            )
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
        feasible, penalty = self._check_rack_count_limits(
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
        if rack.optional and not rack.active:
            for slot in rack.slots:
                slot.node_type = None
                slot.link_type = None
                slot.link_qty = None
            rack.memory_pool_count = 0
            rack.switch_count = 0
            if rack.model_dump() != original.model_dump():
                messages.append(f"deactivated optional rack {rack.rack_id}")
            return

        rack.intra_rack_link_qty = max(
            settings.min_intra_rack_link_qty,
            min(settings.max_intra_rack_link_qty, rack.intra_rack_link_qty),
        )
        for slot in rack.slots:
            if slot.link_qty is not None:
                slot.link_qty = max(
                    settings.min_intra_rack_link_qty,
                    min(settings.max_intra_rack_link_qty, slot.link_qty),
                )
        if rack.limits.max_switch_count is not None:
            rack.switch_count = min(rack.switch_count, rack.limits.max_switch_count)
        if rack.intra_rack_topology == "switch" and rack.switch_count <= 0:
            rack.switch_count = 1
        if rack.limits.max_memory_pool_count is not None:
            rack.memory_pool_count = min(rack.memory_pool_count, rack.limits.max_memory_pool_count)
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
            if rack.optional and not rack.active:
                continue
            if rack.role in {"compute", "hybrid"} and not rack.occupied_slots:
                feasible = False
                penalty += 100_000.0
                messages.append(f"{rack.rack_id} has no occupied compute slots")
                continue
            if rack.role == "memory" and rack.memory_pool_count <= 0:
                feasible = False
                penalty += 100_000.0
                messages.append(f"{rack.rack_id} has no memory pools")
                continue
            feasible, penalty = self._check_rack_device_limits(
                rack,
                feasible=feasible,
                penalty=penalty,
                messages=messages,
            )
            feasible, penalty = self._check_slot_node_types(
                rack,
                feasible=feasible,
                penalty=penalty,
                messages=messages,
            )
            rack_power = 0.0
            rack_cost = 0.0
            rack_units = 0.0
            for type_name, count in self._rack_type_counts(rack):
                spec = self._library.node_types[type_name]
                rack_cost += spec.cost_unit * count
                rack_power += spec.tdp_watts * count
                rack_units += spec.rack_units * count
            rack_cost += self._rack_local_link_cost(rack)
            rack_cost_limit = (
                rack.limits.max_cost
                if rack.limits.max_cost is not None
                else limits.max_rack_cost
            )
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
            if rack_cost > rack_cost_limit:
                feasible = False
                penalty += rack_cost - rack_cost_limit
                messages.append(
                    f"{rack.rack_id} cost exceeds limit: "
                    f"{rack_cost:.3f} > {rack_cost_limit:.3f}"
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

    def _rack_local_link_cost(self, rack: RackGene) -> float:
        if rack.intra_rack_topology == "none":
            return 0.0
        if rack.intra_rack_topology == "switch":
            endpoint_cost = sum(
                self._link_cost(slot.link_type or rack.intra_rack_link_type, slot.link_qty or rack.intra_rack_link_qty)
                for slot in rack.occupied_slots
            )
            memory_cost = 0.0
            if rack.memory_pool_count:
                memory_cost = rack.memory_pool_count * self._link_cost(
                    rack.memory_link_type or rack.intra_rack_link_type,
                    rack.memory_link_qty,
                )
            return endpoint_cost + memory_cost

        node_count = len(rack.occupied_slots) + rack.memory_pool_count
        if node_count <= 1:
            return 0.0
        if rack.intra_rack_topology == "fully_connected":
            pair_count = node_count * (node_count - 1) // 2
        else:
            pair_count = 1 if node_count == 2 else node_count
        return pair_count * self._link_cost(rack.intra_rack_link_type, rack.intra_rack_link_qty)

    def _link_cost(self, link_type: str | None, qty: int) -> float:
        if link_type is None or link_type not in self._library.link_types:
            return 0.0
        return self._library.link_types[link_type].cost_unit * qty

    def _check_rack_device_limits(
        self,
        rack: RackGene,
        *,
        feasible: bool,
        penalty: float,
        messages: list[str],
    ) -> tuple[bool, float]:
        max_slots = rack.limits.max_slots if rack.limits.max_slots is not None else rack.max_slots
        if rack.max_slots > max_slots:
            feasible = False
            penalty += (rack.max_slots - max_slots) * 1000.0
            messages.append(f"{rack.rack_id} max_slots exceeds limit: {rack.max_slots} > {max_slots}")
        if len(rack.slots) > rack.max_slots:
            feasible = False
            penalty += (len(rack.slots) - rack.max_slots) * 1000.0
            messages.append(f"{rack.rack_id} slot count exceeds max_slots: {len(rack.slots)} > {rack.max_slots}")
        if len(rack.occupied_slots) > rack.max_slots:
            feasible = False
            penalty += (len(rack.occupied_slots) - rack.max_slots) * 1000.0
            messages.append(
                f"{rack.rack_id} occupied slots exceeds max_slots: {len(rack.occupied_slots)} > {rack.max_slots}"
            )
        checks = [
            ("memory_pool_count", rack.memory_pool_count, rack.limits.max_memory_pool_count),
            ("switch_count", rack.switch_count, rack.limits.max_switch_count),
        ]
        for label, value, maximum in checks:
            if maximum is not None and value > maximum:
                feasible = False
                penalty += (value - maximum) * 1000.0
                messages.append(f"{rack.rack_id} {label} exceeds limit: {value} > {maximum}")
        return feasible, penalty

    def _check_slot_node_types(
        self,
        rack: RackGene,
        *,
        feasible: bool,
        penalty: float,
        messages: list[str],
    ) -> tuple[bool, float]:
        seen: set[str] = set()
        for slot in rack.slots:
            if slot.slot_id in seen:
                feasible = False
                penalty += 1000.0
                messages.append(f"{rack.rack_id} has duplicate slot_id {slot.slot_id}")
            seen.add(slot.slot_id)
            if not slot.node_type:
                continue
            if slot.node_type not in self._library.node_types:
                feasible = False
                penalty += 100_000.0
                messages.append(f"{rack.rack_id}.{slot.slot_id} unknown node type {slot.node_type}")
                continue
            role = node_role(slot.node_type, self._library.node_types[slot.node_type])
            if role not in {"gpu", "cpu"}:
                feasible = False
                penalty += 100_000.0
                messages.append(f"{rack.rack_id}.{slot.slot_id} node type must be gpu or cpu: {slot.node_type}")
        return feasible, penalty

    def _check_rack_count_limits(
        self,
        chromosome: Chromosome,
        limits: SearchLimits,
        *,
        feasible: bool,
        penalty: float,
        messages: list[str],
    ) -> tuple[bool, float]:
        active_racks = [rack for rack in chromosome.racks if rack.active or not rack.optional]
        compute_racks = [
            rack
            for rack in active_racks
            if rack.role in {"compute", "hybrid"} and rack.occupied_slots
        ]
        memory_racks = [rack for rack in active_racks if rack.role == "memory"]
        hybrid_racks = [rack for rack in active_racks if rack.role == "hybrid"]
        checks = [
            ("total racks", len(active_racks), limits.max_total_racks),
            ("compute racks", len(compute_racks), limits.max_compute_racks),
            ("memory racks", len(memory_racks), limits.max_memory_racks),
            ("hybrid racks", len(hybrid_racks), limits.max_hybrid_racks),
        ]
        if len(compute_racks) < limits.min_compute_racks:
            feasible = False
            penalty += (limits.min_compute_racks - len(compute_racks)) * 100_000.0
            messages.append(
                f"compute rack count below minimum: {len(compute_racks)} < {limits.min_compute_racks}"
            )
        for label, value, maximum in checks:
            if maximum is not None and value > maximum:
                feasible = False
                penalty += (value - maximum) * 100_000.0
                messages.append(f"{label} exceeds limit: {value} > {maximum}")
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
            if rack.optional and not rack.active:
                continue
            if rack.intra_rack_topology != "switch" or not rack.switch_type:
                continue
            switch_spec = self._library.node_types[rack.switch_type]
            radix = switch_spec.radix
            if radix is None:
                continue
            endpoint_ports = 0
            for slot in rack.occupied_slots:
                link_type = slot.link_type or rack.intra_rack_link_type
                endpoint_ports += (slot.link_qty or rack.intra_rack_link_qty) * self._link_lanes(link_type)
            if rack.memory_pool_count:
                endpoint_ports += rack.memory_pool_count * rack.memory_link_qty * self._link_lanes(
                    rack.memory_link_type or rack.intra_rack_link_type
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

    def _inter_rack_degree(self, chromosome: Chromosome, rack_id: str) -> int:
        active_racks = [rack for rack in chromosome.racks if rack.active or not rack.optional]
        if chromosome.inter_rack == "none" or len(active_racks) <= 1:
            return 0
        if chromosome.inter_rack == "fully_connected":
            return len(active_racks) - 1
        if chromosome.inter_rack == "ring":
            return 1 if len(active_racks) == 2 else 2
        return 0

    def _rack_type_counts(self, rack: RackGene) -> list[tuple[str, int]]:
        counts: dict[str, int] = {}
        for slot in rack.occupied_slots:
            if slot.node_type and slot.node_type in self._library.node_types:
                role = role_of_type(slot.node_type, self._library.node_types[slot.node_type].role)
                if role in {"gpu", "cpu"}:
                    counts[slot.node_type] = counts.get(slot.node_type, 0) + 1
        if rack.memory_pool_type and rack.memory_pool_count:
            counts[rack.memory_pool_type] = counts.get(rack.memory_pool_type, 0) + rack.memory_pool_count
        if rack.switch_type and rack.switch_count:
            counts[rack.switch_type] = counts.get(rack.switch_type, 0) + rack.switch_count
        return list(counts.items())
