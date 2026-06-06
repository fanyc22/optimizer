from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from codesign_optimizer.models.hardware import ComponentLibrary
from codesign_optimizer.optimizer.chromosome import Chromosome, RackGene
from codesign_optimizer.optimizer.feedback_parser import ParsedPipelineFeedback
from codesign_optimizer.optimizer.repair import RepairReport
from codesign_optimizer.optimizer.search_space import SearchSpace
from codesign_optimizer.optimizer.tcro import softmax
from codesign_optimizer.optimizer.tgrl import (
    MaskedAction,
    TelemetryContext,
    telemetry_context,
)
from codesign_optimizer.optimizer.workload_suite import MultiWorkloadFeedback


NODE_FEATURE_DIM = 24
EDGE_FEATURE_DIM = 8
GLOBAL_FEATURE_DIM = 15
ACTION_FEATURE_DIM = 40


@dataclass
class GraphObservation:
    node_features: list[list[float]]
    edge_index: list[list[int]]
    edge_features: list[list[float]]
    global_features: list[float]
    action_features: list[list[float]]
    action_target_indices: list[int]
    heuristic_logits: list[float]
    action_keys: list[str]
    masked_actions: list[MaskedAction]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_features": self.node_features,
            "edge_index": self.edge_index,
            "edge_features": self.edge_features,
            "global_features": self.global_features,
            "action_features": self.action_features,
            "action_target_indices": self.action_target_indices,
            "heuristic_logits": self.heuristic_logits,
            "action_keys": self.action_keys,
            "actions": [item.action.to_dict() for item in self.masked_actions],
        }


class GraphObservationBuilder:
    def __init__(self, component_library: ComponentLibrary, search_space: SearchSpace) -> None:
        self._library = component_library
        self._space = search_space

    def build(
        self,
        *,
        chromosome: Chromosome,
        repair: RepairReport,
        feedback: ParsedPipelineFeedback | None,
        masked_actions: list[MaskedAction],
        suite_feedback: MultiWorkloadFeedback | None = None,
        current_score: float,
        best_score: float,
        update: int,
        step: int,
        total_updates: int,
        rollout_steps: int,
    ) -> GraphObservation:
        context = telemetry_context(suite_feedback or feedback, repair, self._space)
        node_features: list[list[float]] = []
        edge_index: list[list[int]] = [[], []]
        edge_features: list[list[float]] = []
        node_lookup: dict[tuple[str, ...], int] = {}

        global_features = self._global_features(
            repair=repair,
            context=context,
            current_score=current_score,
            best_score=best_score,
            update=update,
            step=step,
            total_updates=total_updates,
            rollout_steps=rollout_steps,
            suite_feedback=suite_feedback,
        )
        node_features.append(_pad([1.0, 0.0, 0.0, 0.0, 0.0, 0.0] + global_features[:17], NODE_FEATURE_DIM))
        node_lookup[("global", "")] = 0

        for rack in chromosome.racks:
            rack_idx = len(node_features)
            node_lookup[("rack", rack.rack_id)] = rack_idx
            node_features.append(self._rack_features(rack, context))
            self._add_edge(edge_index, edge_features, 0, rack_idx, [1.0, 0.0, 0.0, float(rack.active), 0.0, 0.0, 0.0, 0.0])
            self._add_edge(edge_index, edge_features, rack_idx, 0, [1.0, 0.0, 0.0, float(rack.active), 0.0, 0.0, 0.0, 0.0])
            if rack.hosts:
                for host in rack.hosts:
                    host_idx = len(node_features)
                    node_lookup[("host", rack.rack_id, host.host_id)] = host_idx
                    node_features.append(self._host_features(rack, host, context))
                    host_edge = [
                        1.0,
                        0.0,
                        1.0,
                        float(rack.active),
                        min(1.0, host.rack_uplink_link_qty / max(1.0, self._space.mutation.max_intra_rack_link_qty)),
                        1.0 if rack.intra_rack_topology == "switch" else 0.0,
                        0.0,
                        1.0 if rack.rack_id in context.top_domain else 0.0,
                    ]
                    self._add_edge(edge_index, edge_features, rack_idx, host_idx, host_edge)
                    self._add_edge(edge_index, edge_features, host_idx, rack_idx, host_edge)
                    for slot in host.slots:
                        res_idx = len(node_features)
                        node_lookup[("slot", rack.rack_id, host.host_id, slot.slot_id)] = res_idx
                        node_features.append(self._slot_features(rack, slot, context))
                        slot_edge = [
                            1.0,
                            0.0,
                            1.0,
                            float(rack.active),
                            min(1.0, host.host_link_qty / max(1.0, self._space.mutation.max_intra_rack_link_qty)),
                            1.0 if host.host_topology == "switch" else 0.0,
                            0.0,
                            1.0 if f"host:{rack.rack_id}/{host.host_id}" in context.top_domain else 0.0,
                        ]
                        self._add_edge(edge_index, edge_features, host_idx, res_idx, slot_edge)
                        self._add_edge(edge_index, edge_features, res_idx, host_idx, slot_edge)
            else:
                for slot in rack.slots:
                    res_idx = len(node_features)
                    node_lookup[("slot", rack.rack_id, slot.slot_id)] = res_idx
                    node_features.append(self._slot_features(rack, slot, context))
                    link_qty = (slot.link_qty or rack.intra_rack_link_qty) if slot.node_type else rack.intra_rack_link_qty
                    edge_feat = [
                        1.0,
                        0.0,
                        1.0,
                        float(rack.active),
                        min(1.0, link_qty / max(1.0, self._space.mutation.max_intra_rack_link_qty)),
                        1.0 if rack.intra_rack_topology == "switch" else 0.0,
                        0.0,
                        1.0 if rack.rack_id in context.top_domain else 0.0,
                    ]
                    self._add_edge(edge_index, edge_features, rack_idx, res_idx, edge_feat)
                    self._add_edge(edge_index, edge_features, res_idx, rack_idx, edge_feat)

        active_racks = [rack for rack in chromosome.racks if rack.active or not rack.optional]
        if chromosome.inter_rack != "none" and len(active_racks) > 1:
            for left, right in self._inter_rack_pairs(active_racks, chromosome.inter_rack):
                left_idx = node_lookup[("rack", left.rack_id)]
                right_idx = node_lookup[("rack", right.rack_id)]
                edge_feat = [
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    min(1.0, chromosome.inter_rack_link_qty / max(1.0, self._space.mutation.max_inter_rack_link_qty)),
                    0.0,
                    1.0,
                    1.0 if context.top_domain.startswith("cluster:") else 0.0,
                ]
                self._add_edge(edge_index, edge_features, left_idx, right_idx, edge_feat)
                self._add_edge(edge_index, edge_features, right_idx, left_idx, edge_feat)

        prior = softmax({item.action.key: item.heuristic_score for item in masked_actions}, temperature=1.0)
        action_features = [self._action_features(item, context) for item in masked_actions]
        action_target_indices = [self._action_target_index(item, node_lookup) for item in masked_actions]
        heuristic_logits = [item.heuristic_score for item in masked_actions]
        for item in masked_actions:
            item.prior_prob = prior.get(item.action.key, 0.0)

        return GraphObservation(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            global_features=global_features,
            action_features=action_features,
            action_target_indices=action_target_indices,
            heuristic_logits=heuristic_logits,
            action_keys=[item.action.key for item in masked_actions],
            masked_actions=masked_actions,
        )

    def _global_features(
        self,
        *,
        repair: RepairReport,
        context: TelemetryContext,
        current_score: float,
        best_score: float,
        update: int,
        step: int,
        total_updates: int,
        rollout_steps: int,
        suite_feedback: MultiWorkloadFeedback | None,
    ) -> list[float]:
        cost_limit = max(1.0, self._space.limits.max_total_cost)
        power_limit = max(1.0, self._space.limits.max_peak_power_watts)
        return _pad(
            [
                max(0.0, 1.0 - repair.estimated_cost / cost_limit),
                max(0.0, 1.0 - repair.estimated_power_watts / power_limit),
                min(10.0, current_score / 1_000_000.0),
                min(10.0, best_score / 1_000_000.0),
                update / max(1.0, total_updates),
                step / max(1.0, rollout_steps),
                context.compute_utilization,
                context.network_utilization,
                context.queue_pressure,
                context.remote_memory_pressure,
                min(32.0, float(len(suite_feedback.runs))) / 32.0 if suite_feedback is not None else 1.0 / 32.0,
                suite_feedback.geomean_speedup if suite_feedback is not None else 1.0,
                suite_feedback.min_speedup if suite_feedback is not None else 1.0,
                suite_feedback.speedup_cv if suite_feedback is not None else 0.0,
                _worst_workload_weight(suite_feedback),
            ],
            GLOBAL_FEATURE_DIM,
        )

    def _rack_features(self, rack: RackGene, context: TelemetryContext) -> list[float]:
        power = self._rack_power(rack)
        units = self._rack_units(rack)
        power_limit = rack.limits.max_power_watts or self._space.limits.max_rack_power_watts or 1.0
        unit_limit = rack.limits.max_rack_units or self._space.limits.max_rack_units or 1.0
        features = self._kind_one_hot("rack") + self._role_one_hot(rack.role)
        features += [
            float(rack.active),
            float(rack.optional),
            rack.gpu_count / 16.0,
            rack.cpu_count / 32.0,
            rack.memory_pool_count / 8.0,
            rack.switch_count / 4.0,
            min(2.0, power / max(1.0, power_limit)),
            min(2.0, units / max(1.0, unit_limit)),
            self._rack_cost(rack) / 1_000_000.0,
            context.compute_utilization,
            context.network_utilization,
            context.queue_pressure,
            context.remote_memory_pressure,
            1.0 if rack.rack_id in context.top_domain else 0.0,
        ]
        return _pad(features, NODE_FEATURE_DIM)

    def _slot_features(self, rack: RackGene, slot: Any, context: TelemetryContext) -> list[float]:
        type_name = slot.node_type
        spec = self._library.node_types.get(type_name) if type_name else None
        slot_role = _node_kind(spec)
        features = self._kind_one_hot(slot_role or "slot") + self._role_one_hot(rack.role)
        features += [
            float(rack.active),
            float(rack.optional),
            1.0 if slot_role == "gpu" else 0.0,
            1.0 if slot_role == "cpu" else 0.0,
            0.0,
            0.0,
            1.0 if slot.node_type else 0.0,
            (_node_peak(spec) / 200.0) if spec else 0.0,
            ((spec.memory_bw_gbps or 0.0) / 4000.0) if spec else 0.0,
            ((spec.cost_unit + spec.tdp_watts) / 50_000.0) if spec else 0.0,
            context.compute_utilization,
            context.network_utilization,
            context.remote_memory_pressure,
            1.0 if rack.rack_id in context.top_domain else 0.0,
        ]
        return _pad(features, NODE_FEATURE_DIM)

    def _host_features(self, rack: RackGene, host: Any, context: TelemetryContext) -> list[float]:
        peak = 0.0
        memory_bw = 0.0
        cost_power = 0.0
        gpu_count = 0
        cpu_count = 0
        for slot in host.occupied_slots:
            spec = self._library.node_types.get(slot.node_type) if slot.node_type else None
            kind = _node_kind(spec)
            if kind == "gpu":
                gpu_count += 1
            elif kind == "cpu":
                cpu_count += 1
            if spec:
                peak += _node_peak(spec)
                memory_bw += spec.memory_bw_gbps or 0.0
                cost_power += spec.cost_unit + spec.tdp_watts
        features = self._kind_one_hot("host") + self._role_one_hot(rack.role)
        features += [
            float(rack.active),
            float(rack.optional),
            gpu_count / 16.0,
            cpu_count / 32.0,
            0.0,
            1.0 if host.template_id else 0.0,
            1.0 if host.host_topology == "switch" else 0.0,
            peak / 400.0,
            memory_bw / 8000.0,
            cost_power / 200_000.0,
            context.compute_utilization,
            context.network_utilization,
            context.remote_memory_pressure,
            1.0 if f"host:{rack.rack_id}/{host.host_id}" in context.top_domain else 0.0,
        ]
        return _pad(features, NODE_FEATURE_DIM)

    def _action_features(self, item: MaskedAction, context: TelemetryContext) -> list[float]:
        action = item.action
        action_type_order = [
            "add_host_to_bay",
            "remove_host_from_bay",
            "replace_host_template",
            "add_node_to_slot",
            "remove_node_from_slot",
            "replace_node_type",
            "upgrade_node",
            "downgrade_node",
            "change_intra_rack_topology",
            "upgrade_intra_rack_link",
            "downgrade_intra_rack_link",
            "change_inter_rack_topology",
            "upgrade_inter_rack_link",
            "downgrade_inter_rack_link",
            "activate_optional_rack",
            "deactivate_optional_rack",
            "add_rack_from_template",
            "remove_rack",
        ]
        resource_order = ["host", "slot", "intra_rack_link", "inter_rack_link", "none"]
        rack = _find_rack(item.chromosome, action.rack_id) if action.rack_id else None
        features = [1.0 if action.action_type == value else 0.0 for value in action_type_order]
        resource_label = _resource_label(rack, action.resource)
        features += [1.0 if resource_label == value else 0.0 for value in resource_order]
        features += self._role_one_hot(rack.role if rack else "")
        features += [
            1.0 if action.delta > 0 else 0.0,
            1.0 if action.delta < 0 else 0.0,
            1.0 if rack and rack.optional else 0.0,
            1.0 if rack and rack.active else 0.0,
            1.0 if rack and rack.rack_id in context.top_domain else 0.0,
            item.heuristic_score / 10.0,
            context.compute_utilization,
            context.network_utilization,
            context.queue_pressure,
            context.remote_memory_pressure,
            context.constraint_pressure,
        ]
        return _pad(features, ACTION_FEATURE_DIM)

    def _action_target_index(self, item: MaskedAction, node_lookup: dict[tuple[str, ...], int]) -> int:
        action = item.action
        if action.rack_id:
            if ("host", action.rack_id, action.resource) in node_lookup:
                return node_lookup[("host", action.rack_id, action.resource)]
            if ("slot", action.rack_id, action.resource) in node_lookup:
                return node_lookup[("slot", action.rack_id, action.resource)]
            if ("rack", action.rack_id) in node_lookup:
                return node_lookup[("rack", action.rack_id)]
        return 0

    def _add_edge(
        self,
        edge_index: list[list[int]],
        edge_features: list[list[float]],
        src: int,
        dst: int,
        features: list[float],
    ) -> None:
        edge_index[0].append(src)
        edge_index[1].append(dst)
        edge_features.append(_pad(features, EDGE_FEATURE_DIM))

    def _kind_one_hot(self, kind: str) -> list[float]:
        order = ["global", "rack", "host", "slot", "gpu", "cpu", "memory", "switch"]
        return [1.0 if kind == value else 0.0 for value in order]

    def _role_one_hot(self, role: str) -> list[float]:
        return [1.0 if role == value else 0.0 for value in ["compute", "memory", "hybrid"]]

    def _rack_power(self, rack: RackGene) -> float:
        return sum(count * self._library.node_types[type_name].tdp_watts for type_name, count in self._rack_type_counts(rack))

    def _rack_cost(self, rack: RackGene) -> float:
        return sum(count * self._library.node_types[type_name].cost_unit for type_name, count in self._rack_type_counts(rack))

    def _rack_units(self, rack: RackGene) -> float:
        return sum(count * self._library.node_types[type_name].rack_units for type_name, count in self._rack_type_counts(rack))

    def _rack_type_counts(self, rack: RackGene) -> list[tuple[str, int]]:
        result: list[tuple[str, int]] = []
        for type_name, count in [
            (rack.gpu_type, rack.gpu_count),
            (rack.cpu_type, rack.cpu_count),
            (rack.memory_pool_type, rack.memory_pool_count),
            (rack.switch_type, rack.switch_count),
        ]:
            if type_name and count and type_name in self._library.node_types:
                result.append((type_name, count))
        for host in rack.occupied_hosts:
            if host.host_switch_type and host.host_switch_type in self._library.node_types:
                result.append((host.host_switch_type, 1))
        return result

    def _resource_count(self, rack: RackGene, resource: str) -> int:
        if resource == "gpu":
            return rack.gpu_count
        if resource == "cpu":
            return rack.cpu_count
        if resource == "memory":
            return rack.memory_pool_count
        if resource == "switch":
            return rack.switch_count
        return 0

    def _resource_type(self, rack: RackGene, resource: str) -> str | None:
        if resource == "gpu":
            return rack.gpu_type
        if resource == "cpu":
            return rack.cpu_type
        if resource == "memory":
            return rack.memory_pool_type
        if resource == "switch":
            return rack.switch_type
        return None

    def _resource_link_qty(self, rack: RackGene, resource: str) -> int:
        if resource == "gpu":
            return rack.gpu_link_qty or rack.endpoint_link_qty
        if resource == "cpu":
            return rack.cpu_link_qty or rack.endpoint_link_qty
        if resource == "memory":
            return rack.memory_link_qty
        return rack.endpoint_link_qty

    def _inter_rack_pairs(self, racks: list[RackGene], mode: str) -> list[tuple[RackGene, RackGene]]:
        if mode == "fully_connected":
            return [(left, right) for idx, left in enumerate(racks) for right in racks[idx + 1 :]]
        if mode == "ring":
            if len(racks) == 2:
                return [(racks[0], racks[1])]
            return [(rack, racks[(idx + 1) % len(racks)]) for idx, rack in enumerate(racks)]
        return []


def _find_rack(chromosome: Chromosome, rack_id: str) -> RackGene | None:
    for rack in chromosome.racks:
        if rack.rack_id == rack_id:
            return rack
    return None


def _resource_label(rack: RackGene | None, resource: str) -> str:
    if rack is None:
        return resource or "none"
    if any(host.host_id == resource for host in rack.hosts):
        return "host"
    if any(slot.slot_id == resource for slot in rack.slots):
        return "slot"
    for host in rack.hosts:
        if any(slot.slot_id == resource for slot in host.slots):
            return "slot"
    return resource or "none"


def _node_peak(spec: Any) -> float:
    if spec is None:
        return 0.0
    return spec.peak_tflops or spec.compute_teraflops_dense or spec.compute_teraflops_sparse or 0.0


def _node_kind(spec: Any) -> str:
    if spec is None:
        return ""
    role = (getattr(spec, "role", "") or "").lower()
    if "cpu" in role:
        return "cpu"
    if "gpu" in role:
        return "gpu"
    if "memory" in role or "mem" in role:
        return "memory"
    if "switch" in role:
        return "switch"
    return ""


def _worst_workload_weight(suite_feedback: MultiWorkloadFeedback | None) -> float:
    if suite_feedback is None or not suite_feedback.runs:
        return 1.0
    worst = min(suite_feedback.runs, key=lambda item: item.speedup)
    return float(worst.weight)


def _pad(values: list[float], size: int) -> list[float]:
    if len(values) >= size:
        return values[:size]
    return values + [0.0] * (size - len(values))
