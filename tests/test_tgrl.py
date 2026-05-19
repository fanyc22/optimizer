import threading
import time
import importlib.util
import json
from pathlib import Path

from typer.testing import CliRunner

from codesign_optimizer.cli import app
from codesign_optimizer.models.hardware import ComponentLibrary
from codesign_optimizer.optimizer.chromosome import chromosome_from_template
from codesign_optimizer.optimizer.exporter import HardwareTopologyExporter
from codesign_optimizer.optimizer.feedback_parser import ParsedPipelineFeedback, parse_pipeline_feedback
from codesign_optimizer.optimizer.repair import CandidateRepairer
from codesign_optimizer.optimizer.search_space import SearchSpace
from codesign_optimizer.optimizer.tgrl import (
    GraphEditAction,
    LinearPolicy,
    TGRLConfig,
    TGRLSearchRunner,
    TrajectoryItem,
    action_features,
    apply_graph_edit_action,
    build_masked_actions,
    enumerate_graph_edit_actions,
    heuristic_action_score,
    telemetry_context,
)


class TelemetryPipeline:
    def __init__(
        self,
        *,
        compute_util: float = 0.92,
        network_util: float = 0.80,
        remote_queue_ns: int = 0,
        delay_s: float = 0.01,
    ) -> None:
        self.compute_util = compute_util
        self.network_util = network_util
        self.remote_queue_ns = remote_queue_ns
        self.delay_s = delay_s
        self.calls = 0
        self.active = 0
        self.max_active = 0
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
            makespan = max(100, 1000 - call_id * 20)
            summary = {
                "case_name": topology_path.stem,
                "success": True,
                "inputs": {"workload": str(workload_path)},
                "simulator": {"finished_count": 1, "expected_finished_count": 1},
            }
            stdout = f"""
            [x] [statistics] [info] sys[0], Wall time: {makespan}
            [x] [statistics] [info] sys[0], Average compute utilization: {self.compute_util * 100:.3f}%
            [x] [statistics] [info] sys[0], Average memory utilization: 40.000%
            [x] [statistics] [info] sys[0], Remote mem provider queue time: {self.remote_queue_ns}
            [x] [statistics] [info] sys[0], Remote mem provider service time: 100
            [x] [network] [info] Network top congested link rank=1 id=rack0_sw0_to_rack1_sw0 src_device=0 dst_device=1 level=L4 domain=cluster:cluster0 stats_domain=cluster:cluster0 technology=optical route_class= bytes=4096 busy_time_ns=80 queue_delay_ns=900000 transmissions=2 max_queue_depth=3 utilization={self.network_util:.6f}
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
                "MEM_FAST": {
                    "role": "memory_pool",
                    "capacity_gb": 2048,
                    "memory_bw_gbps": 800,
                    "tdp_watts": 500,
                    "cost_unit": 24000,
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
            "seed": 23,
            "templates": [
                {
                    "name": "tgrl_latent",
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
                            "memory_link_type": "CXL",
                            "fabric": "switch",
                            "limits": {
                                "max_gpu_count": 2,
                                "max_cpu_count": 2,
                                "max_memory_pool_count": 2,
                                "max_switch_count": 1,
                            },
                        },
                        {
                            "rack_id": "latent-mem",
                            "role": "memory",
                            "optional": True,
                            "active": False,
                            "gpu_count": 0,
                            "cpu_count": 0,
                            "memory_pool_count": 0,
                            "switch_count": 0,
                            "memory_pool_type": "MEM",
                            "switch_type": "SW",
                            "endpoint_link_type": "FAST",
                            "memory_link_type": "CXL",
                            "fabric": "switch",
                            "limits": {
                                "max_gpu_count": 0,
                                "max_cpu_count": 0,
                                "max_memory_pool_count": 2,
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
                "max_memory_pools_per_rack": 2,
                "max_endpoint_link_qty": 3,
                "max_inter_rack_link_qty": 3,
            },
            "limits": {
                "max_total_cost": 300000,
                "max_peak_power_watts": 20000,
                "max_rack_power_watts": 10000,
                "max_rack_units": 42,
            },
        }
    )


def _feedback(*, compute: float = 0.9, network: float = 0.8, remote_queue: int = 0) -> ParsedPipelineFeedback:
    summary = {
        "case_name": "fake",
        "success": True,
        "inputs": {"workload": "fake.json"},
        "simulator": {"finished_count": 1, "expected_finished_count": 1},
    }
    stdout = f"""
    [x] [statistics] [info] sys[0], Wall time: 1000
    [x] [statistics] [info] sys[0], Average compute utilization: {compute * 100:.3f}%
    [x] [statistics] [info] sys[0], Remote mem provider queue time: {remote_queue}
    [x] [network] [info] Network top congested link rank=1 id=rack0_sw0_to_rack1_sw0 src_device=0 dst_device=1 level=L4 domain=cluster:cluster0 stats_domain=cluster:cluster0 technology=optical route_class= bytes=4096 busy_time_ns=80 queue_delay_ns=900000 transmissions=2 max_queue_depth=3 utilization={network:.6f}
    """
    return parse_pipeline_feedback(summary=summary, simulator_stdout=stdout)


def test_action_enumerator_and_mask_include_valid_graph_edits() -> None:
    library = _library()
    space = _space()
    chromosome = chromosome_from_template(space.templates[0])
    actions = enumerate_graph_edit_actions(chromosome, component_library=library, search_space=space)

    assert any(action.action_type == "activate_optional_rack" for action in actions)
    assert any(action.action_type == "expand_rack_resource" and action.resource == "gpu" for action in actions)
    assert any(action.action_type == "upgrade_link_qty" for action in actions)

    masked = build_masked_actions(
        chromosome,
        component_library=library,
        search_space=space,
        repairer=CandidateRepairer(library, space),
        exporter=HardwareTopologyExporter(library),
        feedback=_feedback(compute=0.95),
        current_repair=CandidateRepairer(library, space).repair_and_validate(chromosome),
        policy=None,
        config=TGRLConfig(temperature=1.0),
    )

    assert masked
    assert all(item.repair.feasible for item in masked)


def test_action_mask_blocks_capacity_exceeding_expansion() -> None:
    library = _library()
    space = _space()
    chromosome = chromosome_from_template(space.templates[0])
    rack0 = next(rack for rack in chromosome.racks if rack.rack_id == "rack0")
    rack0.gpu_count = rack0.limits.max_gpu_count or 2

    masked = build_masked_actions(
        chromosome,
        component_library=library,
        search_space=space,
        repairer=CandidateRepairer(library, space),
        exporter=HardwareTopologyExporter(library),
        feedback=_feedback(compute=0.95),
        current_repair=CandidateRepairer(library, space).repair_and_validate(chromosome),
        policy=None,
        config=TGRLConfig(temperature=1.0),
    )

    assert not any(
        item.action.action_type == "expand_rack_resource"
        and item.action.rack_id == "rack0"
        and item.action.resource == "gpu"
        for item in masked
    )


def test_heuristic_prior_prefers_bottleneck_relevant_actions() -> None:
    library = _library()
    space = _space()
    chromosome = chromosome_from_template(space.templates[0])
    report = CandidateRepairer(library, space).repair_and_validate(chromosome)
    compute_context = telemetry_context(_feedback(compute=0.96, network=0.1), report, space)
    remote_context = telemetry_context(_feedback(compute=0.3, network=0.2, remote_queue=2_000_000), report, space)

    expand_gpu = GraphEditAction("expand_rack_resource", rack_id="rack0", resource="gpu", delta=1)
    contract_gpu = GraphEditAction("contract_rack_resource", rack_id="rack0", resource="gpu", delta=-1)
    activate_mem = GraphEditAction("activate_optional_rack", rack_id="latent-mem")

    assert heuristic_action_score(expand_gpu, chromosome, component_library=library, context=compute_context) > heuristic_action_score(
        contract_gpu,
        chromosome,
        component_library=library,
        context=compute_context,
    )
    assert heuristic_action_score(activate_mem, chromosome, component_library=library, context=remote_context) > 0


def test_action_apply_repair_and_export_activate_optional_rack() -> None:
    library = _library()
    space = _space()
    chromosome = chromosome_from_template(space.templates[0])
    activated = apply_graph_edit_action(
        chromosome,
        GraphEditAction("activate_optional_rack", rack_id="latent-mem"),
    )
    report = CandidateRepairer(library, space).repair_and_validate(activated)
    exported = HardwareTopologyExporter(library).export(report.chromosome)

    assert report.feasible
    assert any(group["id"] == "latent-mem" for group in exported.hardware_topology["hierarchy"]["groups"])
    assert any(node["id"].startswith("latent-mem_mem") for node in exported.hardware_topology["nodes"])


def test_linear_policy_update_increases_positive_reward_action_score() -> None:
    policy = LinearPolicy()
    action = GraphEditAction("expand_rack_resource", rack_id="rack0", resource="gpu", delta=1)
    features = {"bias": 1.0, "type:expand_rack_resource": 1.0}
    before = policy.score(features)
    policy.update(
        [
            TrajectoryItem(
                episode=0,
                step=0,
                action=action,
                features=features,
                reward=1.0,
                prior_prob=0.5,
                policy_prob=0.5,
                old_logprob=-0.69,
                candidate_signature="a",
                weighted_score=1.0,
                feasible=True,
            ),
            TrajectoryItem(
                episode=0,
                step=0,
                action=GraphEditAction("contract_rack_resource", rack_id="rack0", resource="gpu", delta=-1),
                features={"bias": 1.0, "type:contract_rack_resource": 1.0},
                reward=-1.0,
                prior_prob=0.5,
                policy_prob=0.5,
                old_logprob=-0.69,
                candidate_signature="b",
                weighted_score=2.0,
                feasible=True,
            ),
        ],
        learning_rate=0.1,
        kl_weight=0.0,
    )

    assert policy.score(features) > before


def test_tgrl_v0_and_v1_run_with_fake_pipeline(tmp_path: Path) -> None:
    workload = tmp_path / "workload.json"
    workload.write_text("{}", encoding="utf-8")
    pipeline = TelemetryPipeline(remote_queue_ns=2_000_000)

    runner = TGRLSearchRunner(
        component_library=_library(),
        search_space=_space(),
        pipeline_client=pipeline,
        workload_path=workload,
        out_dir=tmp_path / "tgrl_v0",
        episodes=1,
        steps_per_episode=2,
        concurrency=2,
        config=TGRLConfig(mode="v0", temperature=0.8),
    )
    result = runner.run()

    assert result.best.feasible
    assert pipeline.max_active >= 2
    assert (tmp_path / "tgrl_v0" / "tgrl_summary.json").exists()
    assert (tmp_path / "tgrl_v0" / "trajectory.jsonl").exists()
    assert (tmp_path / "tgrl_v0" / "best_hardware_topology.json").exists()

    pipeline_v1 = TelemetryPipeline(compute_util=0.95, remote_queue_ns=0)
    runner_v1 = TGRLSearchRunner(
        component_library=_library(),
        search_space=_space(),
        pipeline_client=pipeline_v1,
        workload_path=workload,
        out_dir=tmp_path / "tgrl_v1",
        episodes=2,
        steps_per_episode=2,
        concurrency=1,
        config=TGRLConfig(mode="v1", temperature=0.8, learning_rate=0.1),
    )
    result_v1 = runner_v1.run()

    assert result_v1.best.feasible
    assert result_v1.policy_state
    assert (tmp_path / "tgrl_v1" / "policy_state.json").exists()


def test_tgrl_v2_reports_missing_torch(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is not None:
        return
    catalog = tmp_path / "catalog.json"
    space = tmp_path / "space.json"
    workload = tmp_path / "workload.json"
    catalog.write_text(json.dumps(_library().model_dump(mode="json")), encoding="utf-8")
    space.write_text(json.dumps(_space().model_dump(mode="json")), encoding="utf-8")
    workload.write_text("{}", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "tgrl",
            "--catalog",
            str(catalog),
            "--space",
            str(space),
            "--workload",
            str(workload),
            "--mode",
            "v2",
            "--episodes",
            "1",
            "--steps-per-episode",
            "1",
        ],
    )

    assert result.exit_code == 2
    assert "TG-RL v2 requires PyTorch" in result.output
