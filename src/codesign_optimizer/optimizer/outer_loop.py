from __future__ import annotations

from collections import Counter

from codesign_optimizer.models.feedback import SimulationFeedback
from codesign_optimizer.models.hardware import HardwareProposal, InstantiatedLink, InstantiatedNode
from codesign_optimizer.optimizer.inner_loop import MappingResult


class HardwareTopologyOptimizer:
    """
    Outer loop optimizer:
    Propose incremental hardware improvements from simulator bottlenecks.
    """

    def propose_next(
        self,
        current: HardwareProposal,
        feedback: SimulationFeedback,
        mapping: MappingResult,
    ) -> HardwareProposal:
        next_hw = current.model_copy(deep=True)
        next_hw.iteration = current.iteration + 1

        self._mitigate_network_bottlenecks(next_hw, feedback)
        self._mitigate_thermal_risk(next_hw, feedback)
        self._rebalance_compute_supply(next_hw, mapping)
        return next_hw

    def _mitigate_network_bottlenecks(
        self,
        proposal: HardwareProposal,
        feedback: SimulationFeedback,
    ) -> None:
        if not feedback.network_profile.top_bottlenecks:
            return

        for bottleneck in feedback.network_profile.top_bottlenecks:
            if bottleneck.utilization < 0.9:
                continue
            src, _, dst = bottleneck.link_id.partition("_to_")
            for edge in proposal.system_instantiation.topology:
                if edge.src == src and edge.dst == dst:
                    edge.qty += 1
                    break

    def _mitigate_thermal_risk(
        self,
        proposal: HardwareProposal,
        feedback: SimulationFeedback,
    ) -> None:
        if not feedback.global_metrics.thermal_violation:
            return

        # If thermal alarm is on, convert one Dense_NPU into Sparse_NPU when possible.
        for node in proposal.system_instantiation.nodes:
            if node.type == "Dense_NPU" and "Sparse_NPU" in proposal.component_library.node_types:
                node.type = "Sparse_NPU"
                return

    def _rebalance_compute_supply(
        self,
        proposal: HardwareProposal,
        mapping: MappingResult,
    ) -> None:
        if not mapping.assignment:
            return
        counts = Counter(node_id for node_id in mapping.assignment.values())
        hottest_node_id, hottest_count = counts.most_common(1)[0]
        if hottest_count < 2:
            return

        original = next(
            (n for n in proposal.system_instantiation.nodes if n.node_id == hottest_node_id),
            None,
        )
        if original is None:
            return

        # Clone a similar node to reduce mapping contention in next simulator run.
        new_node_id = f"{original.node_id}_clone"
        if any(n.node_id == new_node_id for n in proposal.system_instantiation.nodes):
            return
        proposal.system_instantiation.nodes.append(
            InstantiatedNode(node_id=new_node_id, type=original.type)
        )

        # Attach clone to same first switch-like node if available.
        switch = next(
            (
                n
                for n in proposal.system_instantiation.nodes
                if "switch" in n.type.lower() or n.node_id.lower().startswith("sw")
            ),
            None,
        )
        if switch and proposal.component_library.link_types:
            first_link_type = next(iter(proposal.component_library.link_types.keys()))
            proposal.system_instantiation.topology.append(
                InstantiatedLink(
                    src=new_node_id,
                    dst=switch.node_id,
                    link_type=first_link_type,
                    qty=1,
                )
            )
