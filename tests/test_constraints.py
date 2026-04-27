from codesign_optimizer.config.settings import ConstraintLimits
from codesign_optimizer.models.feedback import SimulationFeedback
from codesign_optimizer.models.hardware import HardwareProposal
from codesign_optimizer.optimizer.constraints import ConstraintEvaluator


def test_constraints_detect_thermal_and_power_violations() -> None:
    hardware = HardwareProposal.model_validate(
        {
            "optimizer_version": "v1",
            "iteration": 1,
            "component_library": {
                "node_types": {
                    "Dense_NPU": {"tdp_watts": 700, "cost_unit": 20000},
                },
                "link_types": {"NVLink": {"bandwidth_gbps": 100, "latency_ns": 100, "protocol": "NVLink"}},
            },
            "system_instantiation": {
                "nodes": [{"node_id": "n0", "type": "Dense_NPU"}],
                "topology": [],
            },
        }
    )
    feedback = SimulationFeedback.model_validate(
        {
            "simulation_id": "s1",
            "workload": "w",
            "global_metrics": {
                "makespan_us": 1000,
                "total_energy_joules": 100,
                "peak_power_watts": 5000,
                "thermal_violation": True,
                "budget_utilization_percent": 120,
            },
            "compute_profile": {},
            "memory_profile": {
                "local_hbm_bandwidth_util": "10%",
                "cxl_memory_pool": {"avg_access_latency_ns": 10, "conflict_rate": "1%"},
            },
            "network_profile": {"average_link_utilization": "10%", "top_bottlenecks": []},
        }
    )

    evaluator = ConstraintEvaluator(
        ConstraintLimits(max_peak_power_watts=4000, max_budget_utilization_percent=100)
    )
    report = evaluator.evaluate(hardware, feedback)
    assert not report.feasible
    assert not report.thermal_ok
    assert not report.power_ok
    assert not report.budget_ok
