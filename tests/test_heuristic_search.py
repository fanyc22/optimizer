from pathlib import Path

from codesign_optimizer.models.hardware import ComponentLibrary
from codesign_optimizer.optimizer.evolutionary import HeuristicSearchRunner
from codesign_optimizer.optimizer.feedback_parser import ParsedPipelineFeedback, parse_pipeline_feedback
from codesign_optimizer.optimizer.repair import CandidateRepairer
from codesign_optimizer.optimizer.search_space import SearchSpace


class FakePipelineClient:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, *, topology_path: Path, workload_path: Path, out_dir: Path) -> ParsedPipelineFeedback:
        self.calls += 1
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "case_name": topology_path.stem,
            "success": True,
            "inputs": {"workload": str(workload_path)},
            "simulator": {"finished_count": 1, "expected_finished_count": 1},
        }
        makespan = max(10, 1000 - self.calls * 10)
        stdout = f"""
        [x] [workload] [info] sys[0] finished, {makespan} cycles, exposed communication 1 cycles.
        [x] [statistics] [info] sys[0], Wall time: {makespan}
        [x] [statistics] [info] sys[0], Average compute utilization: 60.000%
        [x] [statistics] [info] sys[0], Average memory utilization: 20.000%
        [x] [network] [info] Network top congested link rank=1 id=rack0_sw0_to_rack1_sw0 src_device=0 dst_device=1 level=L4 domain=cluster:cluster0 stats_domain=cluster:cluster0 technology=optical route_class= bytes=4096 busy_time_ns=80 queue_delay_ns=30 transmissions=2 max_queue_depth=3 utilization=0.750000
        """
        return parse_pipeline_feedback(summary=summary, simulator_stdout=stdout)


def _library(radix: int = 32) -> ComponentLibrary:
    return ComponentLibrary.model_validate(
        {
            "node_types": {
                "GPU": {
                    "role": "gpu",
                    "peak_tflops": 80,
                    "memory_bw_gbps": 1800,
                    "tdp_watts": 700,
                    "cost_unit": 20000,
                },
                "CPU": {
                    "role": "cpu",
                    "peak_tflops": 6,
                    "memory_bw_gbps": 220,
                    "tdp_watts": 350,
                    "cost_unit": 6000,
                },
                "MEM": {
                    "role": "memory_pool",
                    "capacity_gb": 1024,
                    "memory_bw_gbps": 320,
                    "tdp_watts": 250,
                    "cost_unit": 12000,
                },
                "SW": {
                    "role": "switch",
                    "radix": radix,
                    "tdp_watts": 180,
                    "cost_unit": 8000,
                },
            },
            "link_types": {
                "FAST": {
                    "bandwidth_gbps": 100,
                    "latency_ns": 100,
                    "protocol": "NVLink",
                    "cost_unit": 1000,
                },
                "CXL": {
                    "bandwidth_gbps": 64,
                    "latency_ns": 250,
                    "protocol": "CXL",
                    "cost_unit": 300,
                },
                "OPTICAL": {
                    "bandwidth_gbps": 400,
                    "latency_ns": 800,
                    "protocol": "Optical",
                    "level": "L4",
                    "cost_unit": 3000,
                },
            },
        }
    )


def _space() -> SearchSpace:
    return SearchSpace.model_validate(
        {
            "seed": 3,
            "templates": [
                {
                    "name": "two_rack",
                    "rack_count": 2,
                    "gpu_count": 1,
                    "cpu_count": 1,
                    "memory_pool_count": 1,
                    "switch_count": 1,
                    "gpu_type": "GPU",
                    "cpu_type": "CPU",
                    "memory_pool_type": "MEM",
                    "switch_type": "SW",
                    "endpoint_link_type": "FAST",
                    "memory_link_type": "CXL",
                    "inter_rack_link_type": "OPTICAL",
                    "inter_rack": "ring",
                    "fabric": "switch",
                }
            ],
            "mutation": {
                "min_gpu_per_rack": 1,
                "max_gpu_per_rack": 2,
                "min_cpu_per_rack": 0,
                "max_cpu_per_rack": 2,
                "min_memory_pools_per_rack": 0,
                "max_memory_pools_per_rack": 2,
                "max_endpoint_link_qty": 3,
                "max_inter_rack_link_qty": 3,
            },
            "limits": {
                "max_total_cost": 200000,
                "max_peak_power_watts": 20000,
                "max_rack_power_watts": 10000,
                "max_rack_units": 42,
            },
        }
    )


def test_heuristic_search_runs_with_fake_pipeline(tmp_path: Path) -> None:
    workload = tmp_path / "workload.json"
    workload.write_text("{}", encoding="utf-8")
    pipeline = FakePipelineClient()
    runner = HeuristicSearchRunner(
        component_library=_library(),
        search_space=_space(),
        pipeline_client=pipeline,
        workload_path=workload,
        out_dir=tmp_path / "search",
        population_size=4,
        generations=2,
    )

    result = runner.run()

    assert pipeline.calls > 0
    assert result.best.feasible
    assert (tmp_path / "search" / "pareto_frontier.json").exists()
    assert (tmp_path / "search" / "best_hardware_topology.json").exists()


def test_repair_marks_switch_radix_violation() -> None:
    space = _space()
    library = _library(radix=1)
    chromosome = space.templates[0]
    from codesign_optimizer.optimizer.chromosome import chromosome_from_template

    report = CandidateRepairer(library, space).repair_and_validate(chromosome_from_template(chromosome))

    assert not report.feasible
    assert any("switch radix exceeded" in msg for msg in report.messages)
