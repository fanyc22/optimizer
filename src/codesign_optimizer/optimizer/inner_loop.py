from __future__ import annotations

from dataclasses import dataclass

from codesign_optimizer.models.feedback import SimulationFeedback
from codesign_optimizer.models.hardware import HardwareProposal
from codesign_optimizer.models.workload import TaskSpec, WorkloadSpec


@dataclass(frozen=True)
class MappingResult:
    assignment: dict[str, str]
    estimated_mapping_cost: float


class SoftwareMapper:
    """
    Inner loop optimizer:
    Assign tasks to concrete nodes considering task type and simulator utilization.
    """

    def optimize_mapping(
        self,
        workload: WorkloadSpec,
        hardware: HardwareProposal,
        feedback: SimulationFeedback,
    ) -> MappingResult:
        node_type_by_id = {
            node.node_id: hardware.component_library.node_types[node.type]
            for node in hardware.system_instantiation.nodes
        }
        available_nodes = list(node_type_by_id.keys())
        if not available_nodes:
            return MappingResult(assignment={}, estimated_mapping_cost=float("inf"))

        sparse_pressure = feedback.compute_profile.get("Sparse_NPU")
        dense_pressure = feedback.compute_profile.get("Dense_NPU")

        assignment: dict[str, str] = {}
        cumulative_cost = 0.0
        for task in workload.tasks:
            best_node = min(
                available_nodes,
                key=lambda node_id: self._task_node_cost(
                    task=task,
                    node_id=node_id,
                    node_type_by_id=node_type_by_id,
                    sparse_pressure=sparse_pressure.avg_utilization if sparse_pressure else 0.0,
                    dense_pressure=dense_pressure.avg_utilization if dense_pressure else 0.0,
                ),
            )
            node_cost = self._task_node_cost(
                task=task,
                node_id=best_node,
                node_type_by_id=node_type_by_id,
                sparse_pressure=sparse_pressure.avg_utilization if sparse_pressure else 0.0,
                dense_pressure=dense_pressure.avg_utilization if dense_pressure else 0.0,
            )
            assignment[task.task_id] = best_node
            cumulative_cost += node_cost

        return MappingResult(assignment=assignment, estimated_mapping_cost=cumulative_cost)

    def _task_node_cost(
        self,
        task: TaskSpec,
        node_id: str,
        node_type_by_id: dict[str, object],
        sparse_pressure: float,
        dense_pressure: float,
    ) -> float:
        node = node_type_by_id[node_id]
        dense_tflops = getattr(node, "compute_teraflops_dense", 0.0) or 0.0
        sparse_tflops = getattr(node, "compute_teraflops_sparse", 0.0) or 0.0
        memory_gb = getattr(node, "local_memory_gb", 0.0) or 0.0

        # Lower is better: compute time + pressure-adjusted contention + memory mismatch penalty.
        dense_time = task.flops_dense / max(dense_tflops, 1e-6)
        sparse_time = task.flops_sparse / max(sparse_tflops, 1e-6)
        memory_penalty = max(0.0, task.memory_gb - memory_gb) * 10.0

        contention_penalty = 0.0
        if task.task_type.lower().startswith("sparse"):
            contention_penalty += sparse_pressure * 0.2
        else:
            contention_penalty += dense_pressure * 0.2

        return dense_time + sparse_time + memory_penalty + contention_penalty
