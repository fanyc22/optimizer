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
from codesign_optimizer.optimizer.chromosome import Chromosome, HostGene, RackGene, SlotGene, role_of_type
from codesign_optimizer.optimizer.link_scope import (
    LinkScope,
    default_level_for_scope,
    link_type_allowed_for_scope,
    scope_label,
)


@dataclass(frozen=True)
class ExportedHardware:
    proposal: HardwareProposal
    hardware_topology: dict[str, Any]
    rank_count: int


class HardwareTopologyExporter:
    def __init__(self, component_library: ComponentLibrary) -> None:
        self._library = component_library

    def export(self, chromosome: Chromosome, *, iteration: int = 0) -> ExportedHardware:
        active_racks = [rack for rack in chromosome.racks if _rack_is_active(rack)]
        nodes: list[dict[str, Any]] = []
        links: list[dict[str, Any]] = []
        groups: list[dict[str, Any]] = [
            {
                "id": "cluster0",
                "level": "L4",
                "type": "SuperPOD",
                "children": [rack.rack_id for rack in active_racks],
            }
        ]
        proposal_nodes: list[InstantiatedNode] = []
        proposal_links: list[InstantiatedLink] = []
        rank_map: list[dict[str, Any]] = []
        rank = 0
        memory_provider: dict[str, str] | None = None
        rack_gateways: list[str] = []

        for rack in active_racks:
            rack_children: list[str] = []
            switch_ids = self._add_switches(rack, nodes, proposal_nodes, rack_children)

            compute_ids: list[str] = []
            host_gateways: list[tuple[str, HostGene]] = []
            if rack.hosts:
                for host in rack.occupied_hosts:
                    host_group_id = self._host_group_id(rack.rack_id, host.host_id)
                    rack_children.append(host_group_id)
                    host_children: list[str] = []
                    gateway, host_compute_ids, rank = self._add_host(
                        rack,
                        host,
                        host_group_id,
                        nodes,
                        proposal_nodes,
                        rank_map,
                        host_children,
                        links,
                        proposal_links,
                        rank,
                    )
                    compute_ids.extend(host_compute_ids)
                    if gateway:
                        host_gateways.append((gateway, host))
                    groups.append(
                        {
                            "id": host_group_id,
                            "level": "L2",
                            "type": "Host",
                            "parent": rack.rack_id,
                            "children": host_children,
                            "attrs": {
                                "host_id": host.host_id,
                                "template_id": host.template_id,
                                "rack_units": host.rack_units,
                                "host_topology": host.host_topology,
                            },
                        }
                    )
            else:
                for slot in rack.occupied_slots:
                    node_id = f"{rack.rack_id}_{slot.slot_id}"
                    compute_ids.append(node_id)
                    rank = self._add_compute_node(
                        nodes,
                        proposal_nodes,
                        rank_map,
                        rack_children,
                        node_id=node_id,
                        rack_id=rack.rack_id,
                        type_name=self._require_node_type(slot.node_type, "slot.node_type"),
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
                    type_name=self._require_node_type(rack.memory_pool_type, "memory_pool_type"),
                )
                if memory_provider is None:
                    memory_provider = {"node_id": node_id, "capability_id": "pool_mem"}

            if rack.hosts:
                self._connect_rack_hosts(
                    rack,
                    host_gateways,
                    memory_ids,
                    switch_ids,
                    links,
                    proposal_links,
                )
            else:
                self._connect_rack(
                    rack,
                    compute_ids,
                    memory_ids,
                    switch_ids,
                    links,
                    proposal_links,
                )
            gateway = self._rack_gateway(switch_ids, [item[0] for item in host_gateways] or compute_ids, memory_ids)
            if gateway:
                rack_gateways.append(gateway)

            groups.append(
                {
                    "id": rack.rack_id,
                    "level": "L3",
                    "type": "Physical_Rack",
                    "parent": "cluster0",
                    "children": rack_children,
                    "attrs": {
                        "role": rack.role,
                        "origin": rack.origin,
                        "max_slots": rack.max_slots,
                        "intra_rack_topology": rack.intra_rack_topology,
                        "capacity_limits": rack.limits.model_dump(exclude_none=True),
                        "estimated_power_watts": self._rack_power(rack),
                        "estimated_cost": self._rack_cost(rack),
                        "rack_units": self._rack_units(rack),
                    },
                }
            )

        self._connect_racks(chromosome, rack_gateways, links, proposal_links)

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
            optimizer_version="slot_tgrl_v1",
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
        if rack.intra_rack_topology != "switch" or rack.switch_count <= 0:
            return switch_ids
        type_name = self._require_node_type(rack.switch_type, "switch_type")
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

    def _add_host(
        self,
        rack: RackGene,
        host: HostGene,
        host_group_id: str,
        nodes: list[dict[str, Any]],
        proposal_nodes: list[InstantiatedNode],
        rank_map: list[dict[str, Any]],
        host_children: list[str],
        links: list[dict[str, Any]],
        proposal_links: list[InstantiatedLink],
        rank: int,
    ) -> tuple[str | None, list[str], int]:
        host_switch_ids = self._add_host_switches(host, host_group_id, rack.rack_id, nodes, proposal_nodes, host_children)
        compute_ids: list[str] = []
        for slot in host.occupied_slots:
            node_id = f"{rack.rack_id}_{host.host_id}_{slot.slot_id}"
            compute_ids.append(node_id)
            rank = self._add_compute_node(
                nodes,
                proposal_nodes,
                rank_map,
                host_children,
                node_id=node_id,
                rack_id=rack.rack_id,
                type_name=self._require_node_type(slot.node_type, "host slot.node_type"),
                rank=rank,
                parent_id=host_group_id,
                level="L2",
                domain=f"host:{rack.rack_id}/{host.host_id}",
                attrs={
                    "slot_id": slot.slot_id,
                    "host_id": host.host_id,
                    "host_template_id": host.template_id,
                    "host_rack_units": host.rack_units,
                    "node_type": slot.node_type,
                },
            )
        self._connect_host(host, compute_ids, host_switch_ids, links, proposal_links, rack_id=rack.rack_id)
        return self._rack_gateway(host_switch_ids, compute_ids, []), compute_ids, rank

    def _add_host_switches(
        self,
        host: HostGene,
        host_group_id: str,
        rack_id: str,
        nodes: list[dict[str, Any]],
        proposal_nodes: list[InstantiatedNode],
        host_children: list[str],
    ) -> list[str]:
        if host.host_topology != "switch" or not host.occupied_slots:
            return []
        type_name = self._require_node_type(host.host_switch_type, "host_switch_type")
        spec = self._library.node_types[type_name]
        node_id = f"{host_group_id}_sw0"
        host_children.append(node_id)
        proposal_nodes.append(InstantiatedNode(node_id=node_id, type=type_name))
        nodes.append(
            {
                "id": node_id,
                "parent": host_group_id,
                "level": "L2",
                "domain": f"host:{rack_id}/{host.host_id}",
                "role": "switch",
                "capabilities": {
                    "network": {
                        "kind": "switch",
                        "radix": spec.radix or 64,
                    }
                },
            }
        )
        return [node_id]

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
        rank: int,
        parent_id: str | None = None,
        level: str = "L3",
        domain: str | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> int:
        spec = self._library.node_types[type_name]
        kind = node_role(type_name, spec)
        if kind not in {"gpu", "cpu"}:
            raise ValueError(f"slot node type {type_name} must be gpu or cpu")
        peak = self._peak_tflops(spec)
        memory_bw = spec.memory_bw_gbps or 200.0
        slots = spec.slots or (4 if kind == "cpu" else 1)
        rack_children.append(node_id)
        proposal_nodes.append(InstantiatedNode(node_id=node_id, type=type_name))
        rank_map.append({"rank": rank, "node_id": node_id})
        node_attrs = attrs or {"slot_id": node_id.rsplit("_", 1)[-1], "node_type": type_name}
        nodes.append(
            {
                "id": node_id,
                "rank": rank,
                "parent": parent_id or rack_id,
                "level": level,
                "domain": domain or f"rack:{rack_id}",
                "role": "rank_compute",
                "attrs": node_attrs,
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
        compute_ids: list[str],
        memory_ids: list[str],
        switch_ids: list[str],
        links: list[dict[str, Any]],
        proposal_links: list[InstantiatedLink],
    ) -> None:
        if rack.intra_rack_topology == "none":
            rack_nodes = compute_ids + memory_ids
            if rack_nodes:
                raise ValueError(f"rack {rack.rack_id} has disallowed intra_rack_topology=none")
            return
        if rack.intra_rack_topology == "switch":
            if not switch_ids:
                raise ValueError(f"rack {rack.rack_id} switch topology has no switch")
            switch_id = switch_ids[0]
            slots_by_node = {f"{rack.rack_id}_{slot.slot_id}": slot for slot in rack.occupied_slots}
            for node_id in compute_ids:
                slot = slots_by_node[node_id]
                self._add_link(
                    links,
                    proposal_links,
                    link_id=f"{node_id}_to_{switch_id}",
                    src=node_id,
                    dst=switch_id,
                    link_type=slot.link_type or rack.intra_rack_link_type,
                    qty=slot.link_qty or rack.intra_rack_link_qty,
                    stats_domain=f"rack:{rack.rack_id}",
                    bidirectional=True,
                    scope="intra",
                )
            for node_id in memory_ids:
                self._add_link(
                    links,
                    proposal_links,
                    link_id=f"{node_id}_to_{switch_id}",
                    src=node_id,
                    dst=switch_id,
                    link_type=rack.memory_link_type or rack.intra_rack_link_type,
                    qty=rack.memory_link_qty,
                    stats_domain=f"rack:{rack.rack_id}",
                    bidirectional=True,
                    scope="intra",
                )
            return

        rack_nodes = compute_ids + memory_ids
        if len(rack_nodes) <= 1:
            return
        if rack.intra_rack_topology == "fully_connected":
            pairs = [(src, dst) for idx, src in enumerate(rack_nodes) for dst in rack_nodes[idx + 1 :]]
        else:
            pairs = (
                [(rack_nodes[0], rack_nodes[1])]
                if len(rack_nodes) == 2
                else [(src, rack_nodes[(idx + 1) % len(rack_nodes)]) for idx, src in enumerate(rack_nodes)]
            )
        for src, dst in pairs:
            self._add_link(
                links,
                proposal_links,
                link_id=f"{src}_to_{dst}",
                src=src,
                dst=dst,
                link_type=rack.intra_rack_link_type,
                qty=rack.intra_rack_link_qty,
                stats_domain=f"rack:{rack.rack_id}",
                bidirectional=True,
                scope="intra",
            )

    def _connect_host(
        self,
        host: HostGene,
        compute_ids: list[str],
        switch_ids: list[str],
        links: list[dict[str, Any]],
        proposal_links: list[InstantiatedLink],
        *,
        rack_id: str,
    ) -> None:
        if host.host_topology == "none":
            if len(compute_ids) <= 1:
                return
            raise ValueError(f"host {host.host_id} has multiple ranks but host_topology=none")
        if host.host_topology == "switch":
            if not switch_ids:
                raise ValueError(f"host {host.host_id} switch topology has no switch")
            switch_id = switch_ids[0]
            for node_id in compute_ids:
                self._add_link(
                    links,
                    proposal_links,
                    link_id=f"{node_id}_to_{switch_id}",
                    src=node_id,
                    dst=switch_id,
                    link_type=host.host_link_type,
                    qty=host.host_link_qty,
                    stats_domain=f"host:{rack_id}/{host.host_id}",
                    bidirectional=True,
                    scope="host",
            )
            return

        if len(compute_ids) <= 1:
            return
        if host.host_topology == "fully_connected":
            pairs = [(src, dst) for idx, src in enumerate(compute_ids) for dst in compute_ids[idx + 1 :]]
        else:
            pairs = (
                [(compute_ids[0], compute_ids[1])]
                if len(compute_ids) == 2
                else [(src, compute_ids[(idx + 1) % len(compute_ids)]) for idx, src in enumerate(compute_ids)]
            )
        for src, dst in pairs:
            self._add_link(
                links,
                proposal_links,
                link_id=f"{src}_to_{dst}",
                src=src,
                dst=dst,
                link_type=host.host_link_type,
                qty=host.host_link_qty,
                stats_domain=f"host:{rack_id}/{host.host_id}",
                bidirectional=True,
                scope="host",
            )

    def _connect_rack_hosts(
        self,
        rack: RackGene,
        host_gateways: list[tuple[str, HostGene]],
        memory_ids: list[str],
        switch_ids: list[str],
        links: list[dict[str, Any]],
        proposal_links: list[InstantiatedLink],
    ) -> None:
        if rack.intra_rack_topology == "none":
            if host_gateways or memory_ids:
                raise ValueError(f"rack {rack.rack_id} has disallowed intra_rack_topology=none")
            return
        if rack.intra_rack_topology == "switch":
            if not switch_ids:
                raise ValueError(f"rack {rack.rack_id} switch topology has no switch")
            switch_id = switch_ids[0]
            for gateway_id, host in host_gateways:
                self._add_link(
                    links,
                    proposal_links,
                    link_id=f"{gateway_id}_to_{switch_id}",
                    src=gateway_id,
                    dst=switch_id,
                    link_type=host.rack_uplink_link_type or rack.intra_rack_link_type,
                    qty=host.rack_uplink_link_qty or rack.intra_rack_link_qty,
                    stats_domain=f"rack:{rack.rack_id}",
                    bidirectional=True,
                    scope="intra",
                )
            for node_id in memory_ids:
                self._add_link(
                    links,
                    proposal_links,
                    link_id=f"{node_id}_to_{switch_id}",
                    src=node_id,
                    dst=switch_id,
                    link_type=rack.memory_link_type or rack.intra_rack_link_type,
                    qty=rack.memory_link_qty,
                    stats_domain=f"rack:{rack.rack_id}",
                    bidirectional=True,
                    scope="intra",
                )
            return

        rack_nodes = [gateway for gateway, _host in host_gateways] + memory_ids
        if len(rack_nodes) <= 1:
            return
        if rack.intra_rack_topology == "fully_connected":
            pairs = [(src, dst) for idx, src in enumerate(rack_nodes) for dst in rack_nodes[idx + 1 :]]
        else:
            pairs = (
                [(rack_nodes[0], rack_nodes[1])]
                if len(rack_nodes) == 2
                else [(src, rack_nodes[(idx + 1) % len(rack_nodes)]) for idx, src in enumerate(rack_nodes)]
            )
        for src, dst in pairs:
            self._add_link(
                links,
                proposal_links,
                link_id=f"{src}_to_{dst}",
                src=src,
                dst=dst,
                link_type=rack.intra_rack_link_type,
                qty=rack.intra_rack_link_qty,
                stats_domain=f"rack:{rack.rack_id}",
                bidirectional=True,
                scope="intra",
            )

    def _connect_racks(
        self,
        chromosome: Chromosome,
        rack_gateways: list[str],
        links: list[dict[str, Any]],
        proposal_links: list[InstantiatedLink],
    ) -> None:
        if len(rack_gateways) <= 1:
            return
        if chromosome.inter_rack == "none":
            raise ValueError("inter_rack topology=none disconnects active racks")
        link_type = chromosome.inter_rack_link_type
        if link_type is None:
            raise ValueError(f"inter_rack topology={chromosome.inter_rack} has no link_type")

        if chromosome.inter_rack == "ring":
            pairs = (
                [(rack_gateways[0], rack_gateways[1])]
                if len(rack_gateways) == 2
                else [
                    (rack_gateways[idx], rack_gateways[(idx + 1) % len(rack_gateways)])
                    for idx in range(len(rack_gateways))
                ]
            )
        elif chromosome.inter_rack == "fully_connected":
            pairs = [(src, dst) for idx, src in enumerate(rack_gateways) for dst in rack_gateways[idx + 1 :]]
        else:
            pairs = []
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
                scope="inter",
            )

    def _add_link(
        self,
        links: list[dict[str, Any]],
        proposal_links: list[InstantiatedLink],
        *,
        link_id: str,
        src: str,
        dst: str,
        link_type: str | None,
        qty: int,
        stats_domain: str,
        bidirectional: bool,
        scope: LinkScope,
    ) -> None:
        if link_type is None:
            raise ValueError(f"link {link_id} has no link_type")
        if not link_type_allowed_for_scope(self._library, link_type, scope):
            spec = self._library.link_types.get(link_type)
            level = spec.level if spec is not None else "unknown"
            raise ValueError(
                f"link {link_id} uses {link_type} ({level}) in {scope_label(scope)} scope"
            )
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
                "level": spec.level or default_level_for_scope(scope),
                "domain": stats_domain,
                "technology": spec.technology or spec.protocol,
                "stats_domain": stats_domain,
            }
        )

    def _rack_gateway(self, switch_ids: list[str], compute_ids: list[str], memory_ids: list[str]) -> str | None:
        if switch_ids:
            return switch_ids[0]
        if compute_ids:
            return compute_ids[0]
        if memory_ids:
            return memory_ids[0]
        return None

    def _host_group_id(self, rack_id: str, host_id: str) -> str:
        return f"{rack_id}_{host_id}"

    def _rack_power(self, rack: RackGene) -> float:
        return sum(
            count * self._library.node_types[type_name].tdp_watts
            for type_name, count in self._rack_type_counts(rack)
        )

    def _rack_cost(self, rack: RackGene) -> float:
        node_cost = sum(
            count * self._library.node_types[type_name].cost_unit
            for type_name, count in self._rack_type_counts(rack)
        )
        return node_cost + self._rack_local_link_cost(rack)

    def _rack_local_link_cost(self, rack: RackGene) -> float:
        if rack.hosts:
            host_cost = sum(self._host_local_link_cost(host) for host in rack.occupied_hosts)
            if rack.intra_rack_topology == "none":
                return host_cost
            if rack.intra_rack_topology == "switch":
                uplink_cost = sum(
                    self._link_cost(host.rack_uplink_link_type or rack.intra_rack_link_type, host.rack_uplink_link_qty)
                    for host in rack.occupied_hosts
                )
                memory_cost = 0.0
                if rack.memory_pool_count:
                    memory_cost = rack.memory_pool_count * self._link_cost(
                        rack.memory_link_type or rack.intra_rack_link_type,
                        rack.memory_link_qty,
                    )
                return host_cost + uplink_cost + memory_cost

            node_count = len(rack.occupied_hosts) + rack.memory_pool_count
            if node_count <= 1:
                return host_cost
            if rack.intra_rack_topology == "fully_connected":
                pair_count = node_count * (node_count - 1) // 2
            else:
                pair_count = 1 if node_count == 2 else node_count
            return host_cost + pair_count * self._link_cost(rack.intra_rack_link_type, rack.intra_rack_link_qty)

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

    def _host_local_link_cost(self, host: HostGene) -> float:
        node_count = len(host.occupied_slots)
        if node_count <= 1:
            return 0.0 if host.host_topology != "switch" else node_count * self._link_cost(host.host_link_type, host.host_link_qty)
        if host.host_topology == "switch":
            return node_count * self._link_cost(host.host_link_type, host.host_link_qty)
        if host.host_topology == "fully_connected":
            pair_count = node_count * (node_count - 1) // 2
        else:
            pair_count = 1 if node_count == 2 else node_count
        return pair_count * self._link_cost(host.host_link_type, host.host_link_qty)

    def _rack_units(self, rack: RackGene) -> float:
        if rack.hosts:
            units = sum(host.rack_units for host in rack.occupied_hosts)
            if rack.memory_pool_type and rack.memory_pool_count:
                units += rack.memory_pool_count * self._library.node_types[rack.memory_pool_type].rack_units
            if rack.switch_type and rack.switch_count:
                units += rack.switch_count * self._library.node_types[rack.switch_type].rack_units
            return units
        return sum(
            count * self._library.node_types[type_name].rack_units
            for type_name, count in self._rack_type_counts(rack)
        )

    def _rack_type_counts(self, rack: RackGene) -> list[tuple[str, int]]:
        counts: dict[str, int] = {}
        for slot in rack.occupied_slots:
            if slot.node_type:
                counts[slot.node_type] = counts.get(slot.node_type, 0) + 1
        for host in rack.occupied_hosts:
            if host.host_switch_type:
                counts[host.host_switch_type] = counts.get(host.host_switch_type, 0) + 1
        if rack.memory_pool_type and rack.memory_pool_count:
            counts[rack.memory_pool_type] = counts.get(rack.memory_pool_type, 0) + rack.memory_pool_count
        if rack.switch_type and rack.switch_count:
            counts[rack.switch_type] = counts.get(rack.switch_type, 0) + rack.switch_count
        return list(counts.items())

    def _peak_tflops(self, spec: NodeTypeSpec) -> float:
        return (
            spec.peak_tflops
            or spec.compute_teraflops_dense
            or spec.compute_teraflops_sparse
            or 1.0
        )

    def _require_node_type(self, type_name: str | None, field_name: str) -> str:
        if not type_name:
            raise ValueError(f"{field_name} must be set before export")
        if type_name not in self._library.node_types:
            raise ValueError(f"unknown node type {type_name}")
        return type_name


def estimate_link_cost(link: InstantiatedLink, link_types: dict[str, LinkTypeSpec]) -> float:
    return link_types[link.link_type].cost_unit * link.qty


def node_role(type_name: str, spec: NodeTypeSpec) -> str:
    return role_of_type(type_name, spec.role)


def _rack_is_active(rack: RackGene) -> bool:
    return rack.active or not rack.optional
