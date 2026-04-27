from codesign_optimizer.config.settings import ObjectiveWeights
from codesign_optimizer.models.feedback import SimulationFeedback
from codesign_optimizer.optimizer.objective import ObjectiveFunction


def _feedback(thermal: bool) -> SimulationFeedback:
    return SimulationFeedback.model_validate(
        {
            "simulation_id": "sim",
            "workload": "w",
            "global_metrics": {
                "makespan_us": 12000,
                "total_energy_joules": 8000,
                "peak_power_watts": 90000,
                "thermal_violation": thermal,
                "budget_utilization_percent": 90,
            },
            "compute_profile": {},
            "memory_profile": {
                "local_hbm_bandwidth_util": "20%",
                "cxl_memory_pool": {"avg_access_latency_ns": 100, "conflict_rate": "5%"},
            },
            "network_profile": {"average_link_utilization": "50%", "top_bottlenecks": []},
        }
    )


def test_objective_penalizes_thermal_violation() -> None:
    objective = ObjectiveFunction(ObjectiveWeights())
    clean = objective.score(_feedback(thermal=False))
    hot = objective.score(_feedback(thermal=True))
    assert hot > clean
