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
    TOPOLOGY_CHANGING_ACTION_TYPES,
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
                    "level": "L3",
                    "cost_unit": 1000,
                },
                "CXL": {
                    "bandwidth_gbps": 64,
                    "latency_ns": 250,
                    "protocol": "CXL",
                    "level": "L3",
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
                            "max_slots": 4,
                            "slots": [
                                {"slot_id": "slot0", "node_type": "GPU_SMALL"},
                                {"slot_id": "slot1", "node_type": "CPU"},
                                {"slot_id": "slot2"},
                                {"slot_id": "slot3"},
                            ],
                            "memory_pool_count": 1,
                            "switch_count": 1,
                            "memory_pool_type": "MEM",
                            "switch_type": "SW",
                            "intra_rack_topology": "switch",
                            "intra_rack_link_type": "FAST",
                            "memory_link_type": "CXL",
                            "limits": {
                                "max_slots": 4,
                                "max_memory_pool_count": 2,
                                "max_switch_count": 1,
                            },
                        },
                        {
                            "rack_id": "latent-mem",
                            "role": "memory",
                            "optional": True,
                            "active": False,
                            "max_slots": 0,
                            "slots": [],
                            "memory_pool_count": 0,
                            "switch_count": 0,
                            "memory_pool_type": "MEM",
                            "switch_type": "SW",
                            "intra_rack_topology": "switch",
                            "intra_rack_link_type": "FAST",
                            "memory_link_type": "CXL",
                            "limits": {
                                "max_slots": 0,
                                "max_memory_pool_count": 2,
                                "max_switch_count": 1,
                            },
                        },
                    ],
                    "inter_rack": "ring",
                    "inter_rack_link_type": "OPTICAL",
                }
            ],
            "rack_archetypes": [
                {
                    "name": "gpu_leaf",
                    "role": "compute",
                    "max_slots": 2,
                    "slots": [
                        {"slot_id": "slot0", "node_type": "GPU_SMALL"},
                        {"slot_id": "slot1"},
                    ],
                    "switch_count": 1,
                    "switch_type": "SW",
                    "intra_rack_topology": "switch",
                    "intra_rack_link_type": "FAST",
                    "limits": {
                        "max_slots": 2,
                        "max_memory_pool_count": 0,
                        "max_switch_count": 1,
                    },
                },
                {
                    "name": "memory_leaf",
                    "role": "memory",
                    "max_slots": 0,
                    "slots": [],
                    "memory_pool_count": 1,
                    "switch_count": 1,
                    "memory_pool_type": "MEM",
                    "switch_type": "SW",
                    "intra_rack_topology": "switch",
                    "intra_rack_link_type": "FAST",
                    "memory_link_type": "CXL",
                    "limits": {
                        "max_slots": 0,
                        "max_memory_pool_count": 2,
                        "max_switch_count": 1,
                    },
                },
            ],
            "mutation": {
                "max_intra_rack_link_qty": 3,
                "max_inter_rack_link_qty": 3,
            },
            "limits": {
                "max_total_cost": 300000,
                "max_peak_power_watts": 20000,
                "max_rack_power_watts": 10000,
                "max_rack_units": 42,
                "max_total_racks": 4,
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
    assert any(action.action_type == "add_rack_from_template" and action.target == "gpu_leaf" for action in actions)
    assert any(action.action_type == "add_node_to_slot" and action.target == "GPU_FAST" for action in actions)
    assert any(action.action_type == "upgrade_intra_rack_link" for action in actions)

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


def test_freeze_topology_masks_topology_changing_actions() -> None:
    library = _library()
    space_payload = _space().model_dump(mode="json")
    space_payload["exhaustive"] = {
        "slot_options": [
            {"node_type": "GPU_SMALL", "link_type": "FAST", "link_qty": 1},
            {"node_type": "CPU", "link_type": "FAST", "link_qty": 1},
        ]
    }
    space = SearchSpace.model_validate(space_payload)
    chromosome = chromosome_from_template(space.templates[0])
    repairer = CandidateRepairer(library, space)
    masked = build_masked_actions(
        chromosome,
        component_library=library,
        search_space=space,
        repairer=repairer,
        exporter=HardwareTopologyExporter(library),
        feedback=_feedback(compute=0.95),
        current_repair=repairer.repair_and_validate(chromosome),
        policy=None,
        config=TGRLConfig(temperature=1.0, freeze_topology=True),
    )

    assert masked
    assert not any(item.action.action_type in TOPOLOGY_CHANGING_ACTION_TYPES for item in masked)
    assert {item.action.action_type for item in masked} <= {
        "replace_node_type",
        "upgrade_node",
        "downgrade_node",
    }
    assert all(item.action.target in {"GPU_SMALL", "CPU"} for item in masked)


def test_node_replacement_applies_exhaustive_link_option() -> None:
    library = _library()
    space_payload = _space().model_dump(mode="json")
    rack = space_payload["templates"][0]["racks"][0]
    rack["slots"][0]["node_type"] = "CPU"
    rack["slots"][0]["link_type"] = "CXL"
    space_payload["exhaustive"] = {
        "slot_options": [
            {"node_type": "GPU_FAST", "link_type": "FAST", "link_qty": 1},
            {"node_type": "CPU", "link_type": "CXL", "link_qty": 1},
        ]
    }
    space = SearchSpace.model_validate(space_payload)
    chromosome = chromosome_from_template(space.templates[0])

    updated = apply_graph_edit_action(
        chromosome,
        GraphEditAction(
            "replace_node_type",
            rack_id="rack0",
            resource="slot0",
            target="GPU_FAST",
        ),
        search_space=space,
    )

    assert updated.racks[0].slots[0].node_type == "GPU_FAST"
    assert updated.racks[0].slots[0].link_type == "FAST"


def test_link_actions_respect_rack_hierarchy_scopes() -> None:
    library = _library()
    space = _space()
    chromosome = chromosome_from_template(space.templates[0])
    actions = enumerate_graph_edit_actions(chromosome, component_library=library, search_space=space)

    intra_targets = {
        action.target
        for action in actions
        if action.action_type in {"upgrade_intra_rack_link", "downgrade_intra_rack_link"}
    }
    inter_targets = {
        action.target
        for action in actions
        if action.action_type in {"upgrade_inter_rack_link", "downgrade_inter_rack_link"}
    }

    assert "OPTICAL" not in intra_targets
    assert "FAST" not in inter_targets
    assert "CXL" not in inter_targets


def test_slot_replace_and_upgrade_actions_apply_to_nodes() -> None:
    library = _library()
    space = _space()
    chromosome = chromosome_from_template(space.templates[0])
    actions = enumerate_graph_edit_actions(chromosome, component_library=library, search_space=space)

    cpu_to_gpu = next(
        action
        for action in actions
        if action.action_type == "replace_node_type"
        and action.rack_id == "rack0"
        and action.resource == "slot1"
        and action.target == "GPU_FAST"
    )
    gpu_to_cpu = next(
        action
        for action in actions
        if action.action_type == "replace_node_type"
        and action.rack_id == "rack0"
        and action.resource == "slot0"
        and action.target == "CPU"
    )
    upgrade_gpu = next(
        action
        for action in actions
        if action.action_type == "upgrade_node"
        and action.rack_id == "rack0"
        and action.resource == "slot0"
        and action.target == "GPU_FAST"
    )

    replaced_cpu = apply_graph_edit_action(chromosome, cpu_to_gpu, search_space=space)
    replaced_gpu = apply_graph_edit_action(chromosome, gpu_to_cpu, search_space=space)
    upgraded = apply_graph_edit_action(chromosome, upgrade_gpu, search_space=space)

    assert next(slot for slot in replaced_cpu.racks[0].slots if slot.slot_id == "slot1").node_type == "GPU_FAST"
    assert next(slot for slot in replaced_gpu.racks[0].slots if slot.slot_id == "slot0").node_type == "CPU"
    assert next(slot for slot in upgraded.racks[0].slots if slot.slot_id == "slot0").node_type == "GPU_FAST"


def test_seed_rack_remove_action_respects_flag() -> None:
    library = _library()
    space = _space()
    chromosome = chromosome_from_template(space.templates[0])

    default_actions = enumerate_graph_edit_actions(chromosome, component_library=library, search_space=space)
    assert not any(action.action_type == "remove_rack" and action.rack_id == "rack0" for action in default_actions)

    payload = space.model_dump(mode="json")
    payload["mutation"]["allow_remove_initial_racks"] = True
    removable_space = SearchSpace.model_validate(payload)
    removable_actions = enumerate_graph_edit_actions(chromosome, component_library=library, search_space=removable_space)

    assert any(action.action_type == "remove_rack" and action.rack_id == "rack0" for action in removable_actions)


def test_dynamic_add_and_remove_rack_from_archetype() -> None:
    library = _library()
    space = _space()
    chromosome = chromosome_from_template(space.templates[0])

    added = apply_graph_edit_action(
        chromosome,
        GraphEditAction("add_rack_from_template", target="gpu_leaf"),
        search_space=space,
    )
    dynamic_rack = next(rack for rack in added.racks if rack.dynamic)
    assert dynamic_rack.rack_id.startswith("dyn-gpu-leaf-")
    assert dynamic_rack.gpu_count == 1
    assert added.inter_rack == "ring"

    report = CandidateRepairer(library, space).repair_and_validate(added)
    exported = HardwareTopologyExporter(library).export(report.chromosome)
    assert report.feasible
    assert any(group["id"] == dynamic_rack.rack_id for group in exported.hardware_topology["hierarchy"]["groups"])

    actions = enumerate_graph_edit_actions(added, component_library=library, search_space=space)
    assert any(action.action_type == "remove_rack" and action.rack_id == dynamic_rack.rack_id for action in actions)
    removed = apply_graph_edit_action(
        added,
        GraphEditAction("remove_rack", rack_id=dynamic_rack.rack_id),
        search_space=space,
    )
    assert all(not rack.dynamic for rack in removed.racks)


def test_action_mask_blocks_capacity_exceeding_expansion() -> None:
    library = _library()
    space = _space()
    chromosome = chromosome_from_template(space.templates[0])
    rack0 = next(rack for rack in chromosome.racks if rack.rack_id == "rack0")
    for slot in rack0.slots:
        slot.node_type = slot.node_type or "GPU_SMALL"

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
        item.action.action_type == "add_node_to_slot"
        and item.action.rack_id == "rack0"
        for item in masked
    )


def test_heuristic_prior_prefers_bottleneck_relevant_actions() -> None:
    library = _library()
    space = _space()
    chromosome = chromosome_from_template(space.templates[0])
    report = CandidateRepairer(library, space).repair_and_validate(chromosome)
    compute_context = telemetry_context(_feedback(compute=0.96, network=0.1), report, space)
    remote_context = telemetry_context(_feedback(compute=0.3, network=0.2, remote_queue=2_000_000), report, space)

    add_gpu = GraphEditAction("add_node_to_slot", rack_id="rack0", resource="slot2", target="GPU_SMALL")
    remove_gpu = GraphEditAction("remove_node_from_slot", rack_id="rack0", resource="slot0")
    activate_mem = GraphEditAction("activate_optional_rack", rack_id="latent-mem")

    assert heuristic_action_score(add_gpu, chromosome, component_library=library, context=compute_context) > heuristic_action_score(
        remove_gpu,
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
    action = GraphEditAction("add_node_to_slot", rack_id="rack0", resource="slot2", target="GPU_SMALL")
    features = {"bias": 1.0, "type:add_node_to_slot": 1.0}
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
                action=GraphEditAction("remove_node_from_slot", rack_id="rack0", resource="slot0"),
                features={"bias": 1.0, "type:remove_node_from_slot": 1.0},
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
