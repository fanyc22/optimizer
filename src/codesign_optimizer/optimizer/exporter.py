from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from codesign_optimizer.models.hardware import (
    ComponentLibrary,
    HardwareProposal,
    InstantiatedLink,
    InstantiatedNode,
    LinkTypeSpec,
    NodeTypeSpec,
)
from codesign_optimizer.optimizer.chromosome import Chromosome, RackGene, role_of_type


@dataclass(frozen=True)
class ExportedHardware:
    proposal: HardwareProposal
    hardware_topology: dict[str, Any]
    rank_count: int


class HardwareTopologyExporter:
    def __init__(self, component_library: ComponentLibrary) -> None:
        self._library = component_library

    def export(self, chromosome: Chromosome, *, iteration: int = 0) -> ExportedHardware:
        nodes: list[dict[str, Any]] = []
        links: list[dict[str, Any]] = []
        groups: list[dict[str, Any]] = [
            {
                "id": "cluster0",
                "level": "L4",
                "type": "SuperPOD",
                "children": [rack.rack_id for rack in chromosome.racks],
            }
        ]
        proposal_nodes: list[InstantiatedNode] = []
        proposal_links: list[InstantiatedLink] = []
        rank_map: list[dict[str, Any]] = []
        rank = 0
        memory_provider: dict[str, str] | None = None

        rack_switches: list[str] = []
        for rack in chromosome.racks:
            rack_children: list[str] = []
            switch_ids = self._add_switches(rack, nodes, proposal_nodes, rack_children)
            rack_switches.extend(switch_ids[:1])

            gpu_ids: list[str] = []
            cpu_ids: list[str] = []
            for idx in range(rack.gpu_count):
                node_id = f"{rack.rack_id}_gpu{idx}"
                gpu_ids.append(node_id)
                rank = self._add_compute_node(
                    nodes,
                    proposal_nodes,
                    rank_map,
                    rack_children,
                    node_id=node_id,
                    rack_id=rack.rack_id,
                    type_name=self._require_type(rack.gpu_type, "gpu_type"),
                    kind="gpu",
                    rank=rank,
                )

            for idx in range(rack.cpu_count):
                node_id = f"{rack.rack_id}_cpu{idx}"
                cpu_ids.append(node_id)
                rank = self._add_compute_node(
                    nodes,
                    proposal_nodes,
                    rank_map,
                    rack_children,
                    node_id=node_id,
                    rack_id=rack.rack_id,
                    type_name=self._require_type(rack.cpu_type, "cpu_type"),
                    kind="cpu",
                    rank=rank,
                )

            memory_ids: list[str] = []
            for idx in range(rack.memory_pool_count):
                node_id = f"{rack.rack_id}_mem{idx}"
                memory_ids.append(node_id)
                self._add_memory_pool_node(
                    nodes,
                    proposal_nodes,
                    rack_children,
                    node_id=node_id,
                    rack_id=rack.rack_id,
                    type_name=self._require_type(rack.memory_pool_type, "memory_pool_type"),
                )
                if memory_provider is None:
                    memory_provider = {"node_id": node_id, "capability_id": "pool_mem"}

            self._connect_rack(
                rack,
                gpu_ids,
                cpu_ids,
                memory_ids,
                switch_ids,
                links,
                proposal_links,
            )

            groups.append(
                {
                    "id": rack.rack_id,
                    "level": "L3",
                    "type": "Physical_Rack",
                    "parent": "cluster0",
                    "children": rack_children,
                    "attrs": {
                        "estimated_power_watts": self._rack_power(rack),
                        "estimated_cost": self._rack_cost(rack),
                        "rack_units": self._rack_units(rack),
                    },
                }
            )

        self._connect_racks(chromosome, rack_switches, links, proposal_links)

        if memory_provider is None and rank_map:
            memory_provider = {"node_id": rank_map[0]["node_id"], "capability_id": "local_mem"}

        hardware: dict[str, Any] = {
            "schema": "terrapod.hardware_topology.v2",
            "name": f"optimizer_iter_{iteration:03d}_{chromosome.template_name}",
            "time_unit": "ms",
            "hierarchy": {"groups": groups},
            "nodes": nodes,
            "links": links,
            "rank_map": rank_map,
        }
        if memory_provider is not None:
            hardware["defaults"] = {"memory_provider": memory_provider}

        proposal = HardwareProposal(
            optimizer_version="heuristic_nsga2_v1",
            iteration=iteration,
            component_library=self._library,
            system_instantiation={
                "nodes": proposal_nodes,
                "topology": proposal_links,
            },
        )
        return ExportedHardware(proposal=proposal, hardware_topology=hardware, rank_count=rank)

    def _add_switches(
        self,
        rack: RackGene,
        nodes: list[dict[str, Any]],
        proposal_nodes: list[InstantiatedNode],
        rack_children: list[str],
    ) -> list[str]:
        switch_ids: list[str] = []
        if rack.fabric != "switch":
            return switch_ids
        type_name = self._require_type(rack.switch_type, "switch_type")
        spec = self._library.node_types[type_name]
        for idx in range(rack.switch_count):
            node_id = f"{rack.rack_id}_sw{idx}"
            switch_ids.append(node_id)
            rack_children.append(node_id)
            proposal_nodes.append(InstantiatedNode(node_id=node_id, type=type_name))
            nodes.append(
                {
                    "id": node_id,
                    "parent": rack.rack_id,
                    "level": "L3",
                    "domain": f"rack:{rack.rack_id}",
                    "role": "switch",
                    "capabilities": {
                        "network": {
                            "kind": "switch",
                            "radix": spec.radix or 64,
                        }
                    },
                }
            )
        return switch_ids

    def _add_compute_node(
        self,
        nodes: list[dict[str, Any]],
        proposal_nodes: list[InstantiatedNode],
        rank_map: list[dict[str, Any]],
        rack_children: list[str],
        *,
        node_id: str,
        rack_id: str,
        type_name: str,
        kind: str,
        rank: int,
    ) -> int:
        spec = self._library.node_types[type_name]
        peak = self._peak_tflops(spec)
        memory_bw = spec.memory_bw_gbps or 200.0
        slots = spec.slots or (4 if kind == "cpu" else 1)
        rack_children.append(node_id)
        proposal_nodes.append(InstantiatedNode(node_id=node_id, type=type_name))
        rank_map.append({"rank": rank, "node_id": node_id})
        nodes.append(
            {
                "id": node_id,
                "rank": rank,
                "parent": rack_id,
                "level": "L3",
                "domain": f"rack:{rack_id}",
                "role": "rank_compute",
                "capabilities": {
                    "compute": [
                        {
                            "id": f"{kind}0",
                            "kind": kind,
                            "peak_tflops": peak,
                            "memory_bw_gbps": memory_bw,
                            "slots": slots,
                            "max_parallelism": spec.max_parallelism or slots,
                            "default": True,
                        }
                    ],
                    "memory": [
                        {
                            "id": "local_mem",
                            "kind": "memory",
                            "capacity_gb": spec.local_memory_gb or 0,
                            "bandwidth_gbps": memory_bw,
                            "latency_ns": spec.memory_latency_ns or 0,
                            "queue_domain": f"{node_id}:local_mem",
                            "default": True,
                        }
                    ],
                    "network": {
                        "kind": "endpoint",
                        "comm_slots": spec.comm_slots or 1,
                    },
                },
                "defaults": {
                    "compute": {kind: f"{kind}0"},
                    "memory": "local_mem",
                },
            }
        )
        return rank + 1

    def _add_memory_pool_node(
        self,
        nodes: list[dict[str, Any]],
        proposal_nodes: list[InstantiatedNode],
        rack_children: list[str],
        *,
        node_id: str,
        rack_id: str,
        type_name: str,
    ) -> None:
        spec = self._library.node_types[type_name]
        rack_children.append(node_id)
        proposal_nodes.append(InstantiatedNode(node_id=node_id, type=type_name))
        nodes.append(
            {
                "id": node_id,
                "parent": rack_id,
                "level": "L3",
                "domain": f"rack:{rack_id}",
                "role": "memory_pool",
                "capabilities": {
                    "memory": [
                        {
                            "id": "pool_mem",
                            "kind": "memory_pool",
                            "capacity_gb": spec.capacity_gb or spec.local_memory_gb or 0,
                            "bandwidth_gbps": spec.memory_bw_gbps or 200.0,
                            "latency_ns": spec.memory_latency_ns or 300.0,
                            "queue_domain": f"{node_id}:pool_mem",
                            "default": True,
                        }
                    ],
                    "network": {
                        "kind": "endpoint",
                        "comm_slots": spec.comm_slots or 4,
                    },
                },
                "defaults": {"memory": "pool_mem"},
            }
        )

    def _connect_rack(
        self,
        rack: RackGene,
        gpu_ids: list[str],
        cpu_ids: list[str],
        memory_ids: list[str],
        switch_ids: list[str],
        links: list[dict[str, Any]],
        proposal_links: list[InstantiatedLink],
    ) -> None:
        if rack.fabric == "switch" and switch_ids:
            switch_id = switch_ids[0]
            gpu_link_type = rack.gpu_link_type or rack.endpoint_link_type
            cpu_link_type = rack.cpu_link_type or rack.endpoint_link_type
            gpu_link_qty = rack.gpu_link_qty or rack.endpoint_link_qty
            cpu_link_qty = rack.cpu_link_qty or rack.endpoint_link_qty
            for node_id in gpu_ids:
                self._add_link(
                    links,
                    proposal_links,
                    link_id=f"{node_id}_to_{switch_id}",
                    src=node_id,
                    dst=switch_id,
                    link_type=gpu_link_type,
                    qty=gpu_link_qty,
                    stats_domain=f"rack:{rack.rack_id}",
                    bidirectional=True,
                )
            for node_id in cpu_ids:
                self._add_link(
                    links,
                    proposal_links,
                    link_id=f"{node_id}_to_{switch_id}",
                    src=node_id,
                    dst=switch_id,
                    link_type=cpu_link_type,
                    qty=cpu_link_qty,
                    stats_domain=f"rack:{rack.rack_id}",
                    bidirectional=True,
                )
            memory_link_type = rack.memory_link_type or rack.endpoint_link_type
            for node_id in memory_ids:
                self._add_link(
                    links,
                    proposal_links,
                    link_id=f"{node_id}_to_{switch_id}",
                    src=node_id,
                    dst=switch_id,
                    link_type=memory_link_type,
                    qty=rack.memory_link_qty,
                    stats_domain=f"rack:{rack.rack_id}",
                    bidirectional=True,
                )
            return

        ring_nodes = gpu_ids + cpu_ids + memory_ids
        if len(ring_nodes) == 1:
            return
        rack_pairs = (
            [(ring_nodes[0], ring_nodes[1])]
            if len(ring_nodes) == 2
            else [
                (src, ring_nodes[(idx + 1) % len(ring_nodes)])
                for idx, src in enumerate(ring_nodes)
            ]
        )
        for src, dst in rack_pairs:
            self._add_link(
                links,
                proposal_links,
                link_id=f"{src}_to_{dst}",
                src=src,
                dst=dst,
                link_type=rack.endpoint_link_type,
                qty=rack.endpoint_link_qty,
                stats_domain=f"rack:{rack.rack_id}",
                bidirectional=True,
            )

    def _connect_racks(
        self,
        chromosome: Chromosome,
        rack_switches: list[str],
        links: list[dict[str, Any]],
        proposal_links: list[InstantiatedLink],
    ) -> None:
        if chromosome.inter_rack == "none" or len(rack_switches) <= 1:
            return
        link_type = chromosome.inter_rack_link_type
        if link_type is None:
            return

        pairs: list[tuple[str, str]] = []
        if chromosome.inter_rack == "ring":
            pairs = (
                [(rack_switches[0], rack_switches[1])]
                if len(rack_switches) == 2
                else [
                    (rack_switches[idx], rack_switches[(idx + 1) % len(rack_switches)])
                    for idx in range(len(rack_switches))
                ]
            )
        elif chromosome.inter_rack == "fully_connected":
            pairs = [
                (src, dst)
                for idx, src in enumerate(rack_switches)
                for dst in rack_switches[idx + 1 :]
            ]
        for src, dst in pairs:
            self._add_link(
                links,
                proposal_links,
                link_id=f"{src}_to_{dst}",
                src=src,
                dst=dst,
                link_type=link_type,
                qty=chromosome.inter_rack_link_qty,
                stats_domain="cluster:cluster0",
                bidirectional=True,
            )

    def _add_link(
        self,
        links: list[dict[str, Any]],
        proposal_links: list[InstantiatedLink],
        *,
        link_id: str,
        src: str,
        dst: str,
        link_type: str,
        qty: int,
        stats_domain: str,
        bidirectional: bool,
    ) -> None:
        spec = self._library.link_types[link_type]
        proposal_links.append(InstantiatedLink(src=src, dst=dst, link_type=link_type, qty=qty))
        links.append(
            {
                "id": link_id,
                "src": src,
                "dst": dst,
                "bandwidth_gbps": spec.bandwidth_gbps * qty,
                "latency_ns": spec.latency_ns,
                "bidirectional": bidirectional,
                "level": spec.level or "L3",
                "domain": stats_domain,
                "technology": spec.technology or spec.protocol,
                "stats_domain": stats_domain,
            }
        )

    def _rack_power(self, rack: RackGene) -> float:
        return sum(
            count * self._library.node_types[type_name].tdp_watts
            for type_name, count in self._rack_type_counts(rack)
        )

    def _rack_cost(self, rack: RackGene) -> float:
        return sum(
            count * self._library.node_types[type_name].cost_unit
            for type_name, count in self._rack_type_counts(rack)
        )

    def _rack_units(self, rack: RackGene) -> float:
        return sum(
            count * self._library.node_types[type_name].rack_units
            for type_name, count in self._rack_type_counts(rack)
        )

    def _rack_type_counts(self, rack: RackGene) -> list[tuple[str, int]]:
        items: list[tuple[str, int]] = []
        if rack.gpu_type and rack.gpu_count:
            items.append((rack.gpu_type, rack.gpu_count))
        if rack.cpu_type and rack.cpu_count:
            items.append((rack.cpu_type, rack.cpu_count))
        if rack.memory_pool_type and rack.memory_pool_count:
            items.append((rack.memory_pool_type, rack.memory_pool_count))
        if rack.switch_type and rack.switch_count:
            items.append((rack.switch_type, rack.switch_count))
        return items

    def _peak_tflops(self, spec: NodeTypeSpec) -> float:
        return (
            spec.peak_tflops
            or spec.compute_teraflops_dense
            or spec.compute_teraflops_sparse
            or 1.0
        )

    def _require_type(self, type_name: str | None, field_name: str) -> str:
        if not type_name:
            raise ValueError(f"{field_name} must be set before export")
        if type_name not in self._library.node_types:
            raise ValueError(f"unknown node type {type_name}")
        return type_name


def estimate_link_cost(link: InstantiatedLink, link_types: dict[str, LinkTypeSpec]) -> float:
    return link_types[link.link_type].cost_unit * link.qty


def node_role(type_name: str, spec: NodeTypeSpec) -> str:
    return role_of_type(type_name, spec.role)
