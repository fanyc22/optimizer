import threading
import time
from pathlib import Path

from codesign_optimizer.models.hardware import ComponentLibrary
from codesign_optimizer.optimizer.feedback_parser import ParsedPipelineFeedback, parse_pipeline_feedback
from codesign_optimizer.optimizer.repair import CandidateRepairer
from codesign_optimizer.optimizer.search_space import SearchSpace
from codesign_optimizer.optimizer.tcro import (
    TCROConfig,
    TCROSearchRunner,
    project_simplex,
    quantize_alpha,
    softmax,
)


class TelemetryFakePipeline:
    def __init__(self, delay_s: float = 0.03) -> None:
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.delay_s = delay_s
        self._lock = threading.Lock()

    def run(self, *, topology_path: Path, workload_path: Path, out_dir: Path) -> ParsedPipelineFeedback:
        out_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            call_id = self.calls
        try:
            time.sleep(self.delay_s)
            makespan = max(100, 1000 - call_id * 10)
            summary = {
                "case_name": topology_path.stem,
                "success": True,
                "inputs": {"workload": str(workload_path)},
                "simulator": {"finished_count": 1, "expected_finished_count": 1},
            }
            stdout = f"""
            [x] [statistics] [info] sys[0], Wall time: {makespan}
            [x] [statistics] [info] sys[0], Average compute utilization: 92.000%
            [x] [statistics] [info] sys[0], Average memory utilization: 40.000%
            [x] [statistics] [info] sys[0], GPU time: 800
            [x] [statistics] [info] sys[0], Comm time: 500
            [x] [statistics] [info] sys[0], Remote mem time: 300
            [x] [statistics] [info] sys[0], Total compute-communication overlap: 250
            [x] [statistics] [info] sys[0], Remote mem provider queue time: 2000000
            [x] [statistics] [info] sys[0], Remote mem provider service time: 100
            [x] [network] [info] Network top congested link rank=1 id=rack0_sw0_to_rack1_sw0 src_device=0 dst_device=1 level=L4 domain=cluster:cluster0 stats_domain=cluster:cluster0 technology=optical route_class= bytes=4096 busy_time_ns=80 queue_delay_ns=900000 transmissions=2 max_queue_depth=3 utilization=0.920000
            """
            return parse_pipeline_feedback(summary=summary, simulator_stdout=stdout)
        finally:
            with self._lock:
                self.active -= 1


def _library() -> ComponentLibrary:
    return ComponentLibrary.model_validate(
        {
            "node_types": {
                "GPU_SMALL": {
                    "role": "gpu",
                    "peak_tflops": 40,
                    "memory_bw_gbps": 900,
                    "tdp_watts": 350,
                    "cost_unit": 9000,
                },
                "GPU_FAST": {
                    "role": "gpu",
                    "peak_tflops": 120,
                    "memory_bw_gbps": 2200,
                    "tdp_watts": 700,
                    "cost_unit": 30000,
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
                    "radix": 64,
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
            "seed": 13,
            "templates": [
                {
                    "name": "tcro_base",
                    "racks": [
                        {
                            "rack_id": "rack0",
                            "role": "hybrid",
                            "gpu_count": 1,
                            "cpu_count": 1,
                            "memory_pool_count": 1,
                            "switch_count": 1,
                            "gpu_type": "GPU_SMALL",
                            "cpu_type": "CPU",
                            "memory_pool_type": "MEM",
                            "switch_type": "SW",
                            "endpoint_link_type": "FAST",
                            "memory_link_type": "FAST",
                            "fabric": "switch",
                            "limits": {
                                "max_gpu_count": 2,
                                "max_cpu_count": 2,
                                "max_memory_pool_count": 3,
                                "max_switch_count": 1,
                            },
                        },
                        {
                            "rack_id": "rack1",
                            "role": "compute",
                            "gpu_count": 1,
                            "cpu_count": 0,
                            "memory_pool_count": 0,
                            "switch_count": 1,
                            "gpu_type": "GPU_SMALL",
                            "switch_type": "SW",
                            "endpoint_link_type": "FAST",
                            "fabric": "switch",
                            "limits": {
                                "max_gpu_count": 2,
                                "max_cpu_count": 0,
                                "max_memory_pool_count": 0,
                                "max_switch_count": 1,
                            },
                        },
                    ],
                    "inter_rack": "ring",
                    "inter_rack_link_type": "OPTICAL",
                }
            ],
            "mutation": {
                "min_gpu_per_rack": 0,
                "max_gpu_per_rack": 2,
                "min_cpu_per_rack": 0,
                "max_cpu_per_rack": 2,
                "min_memory_pools_per_rack": 0,
                "max_memory_pools_per_rack": 3,
                "max_endpoint_link_qty": 4,
                "max_inter_rack_link_qty": 4,
            },
            "limits": {
                "max_total_cost": 300000,
                "max_peak_power_watts": 20000,
                "max_rack_power_watts": 10000,
                "max_rack_units": 42,
            },
        }
    )


def test_softmax_projection_and_alpha_quantization() -> None:
    probs = softmax({"a": 0.0, "b": 2.0}, temperature=1.0)
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert probs["b"] > probs["a"]

    projected = project_simplex([-1.0, 1.0, 3.0])
    assert abs(sum(projected) - 1.0) < 1e-9
    assert projected[0] == 0.0

    assert quantize_alpha(0.1, threshold=0.25, minimum=1, maximum=4) == 0
    assert quantize_alpha(1.2, threshold=0.25, minimum=1, maximum=4) == 2


def test_tcro_lowering_exports_valid_topology(tmp_path: Path) -> None:
    runner = TCROSearchRunner(
        component_library=_library(),
        search_space=_space(),
        pipeline_client=TelemetryFakePipeline(),
        workload_path=tmp_path / "workload.json",
        out_dir=tmp_path / "tcro",
        steps=1,
        samples_per_step=1,
        concurrency=1,
        config=TCROConfig(initial_temperature=0.0),
    )
    state = runner._initial_state()
    candidate = runner._sample_candidate(state, step=0, sample=0)
    report = CandidateRepairer(_library(), _space()).repair_and_validate(candidate.chromosome)

    assert report.feasible
    assert candidate.chromosome.inter_rack in {"ring", "fully_connected"}


def test_tcro_run_updates_state_from_telemetry_and_writes_artifacts(tmp_path: Path) -> None:
    workload = tmp_path / "workload.json"
    workload.write_text("{}", encoding="utf-8")
    pipeline = TelemetryFakePipeline()
    runner = TCROSearchRunner(
        component_library=_library(),
        search_space=_space(),
        pipeline_client=pipeline,
        workload_path=workload,
        out_dir=tmp_path / "tcro_run",
        steps=3,
        samples_per_step=2,
        concurrency=2,
        config=TCROConfig(
            learning_rate=0.4,
            initial_temperature=0.0,
            temperature_decay=1.0,
            link_prune_threshold=0.25,
        ),
    )

    result = runner.run()
    rack0 = result.final_state.rack_state("rack0")

    assert result.best.feasible
    assert pipeline.max_active >= 2
    assert result.final_state.inter_rack_alpha > 1.0
    assert rack0 is not None
    assert rack0.count_alpha["memory"] > 1.0
    assert rack0.link_alpha["memory"] > 1.0
    assert (tmp_path / "tcro_run" / "tcro_summary.json").exists()
    assert (tmp_path / "tcro_run" / "telemetry_history.json").exists()
    assert (tmp_path / "tcro_run" / "best_hardware_topology.json").exists()
