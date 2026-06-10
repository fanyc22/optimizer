from pathlib import Path

import pytest

from codesign_optimizer.models.hardware import ComponentLibrary
from codesign_optimizer.optimizer.chromosome import chromosome_from_template
from codesign_optimizer.optimizer.feedback_parser import parse_pipeline_feedback
from codesign_optimizer.optimizer.repair import CandidateRepairer
from codesign_optimizer.optimizer.search_space import SearchSpace
from codesign_optimizer.optimizer.tgrl import GraphEditAction, heuristic_action_score, telemetry_context
from codesign_optimizer.optimizer.workload_suite import (
    MultiWorkloadPipelineRunner,
    WorkloadRunFeedback,
    WorkloadSuite,
    WorkloadSuiteBaseline,
    aggregate_multi_workload_feedback,
    apply_workload_rank_parallel_default,
    load_workload_suite,
    workload_suite_rank_parallel_fields,
)


def _library() -> ComponentLibrary:
    return ComponentLibrary.model_validate(
        {
            "node_types": {
                "GPU": {"role": "gpu", "peak_tflops": 80, "memory_bw_gbps": 1800, "tdp_watts": 700, "cost_unit": 20000},
                "CPU": {"role": "cpu", "peak_tflops": 6, "memory_bw_gbps": 220, "tdp_watts": 350, "cost_unit": 6000},
                "MEM": {"role": "memory_pool", "capacity_gb": 1024, "memory_bw_gbps": 320, "tdp_watts": 250, "cost_unit": 12000},
                "SW": {"role": "switch", "radix": 64, "tdp_watts": 180, "cost_unit": 8000},
            },
            "link_types": {
                "FAST": {"bandwidth_gbps": 100, "latency_ns": 100, "protocol": "NVLink", "cost_unit": 1000},
                "CXL": {"bandwidth_gbps": 64, "latency_ns": 250, "protocol": "CXL", "cost_unit": 300},
                "OPTICAL": {"bandwidth_gbps": 400, "latency_ns": 800, "protocol": "Optical", "level": "L4", "cost_unit": 3000},
            },
        }
    )


def _space() -> SearchSpace:
    return SearchSpace.model_validate(
        {
            "seed": 31,
            "templates": [
                {
                    "name": "suite_small",
                    "racks": [
                        {
                            "rack_id": "rack0",
                            "role": "hybrid",
                            "max_slots": 3,
                            "slots": [
                                {"slot_id": "slot0", "node_type": "GPU"},
                                {"slot_id": "slot1", "node_type": "CPU"},
                                {"slot_id": "slot2"},
                            ],
                            "memory_pool_count": 1,
                            "switch_count": 1,
                            "memory_pool_type": "MEM",
                            "switch_type": "SW",
                            "intra_rack_topology": "switch",
                            "intra_rack_link_type": "FAST",
                            "memory_link_type": "CXL",
                            "limits": {"max_slots": 3, "max_memory_pool_count": 2, "max_switch_count": 1},
                        }
                    ],
                    "inter_rack": "ring",
                    "inter_rack_link_type": "OPTICAL",
                }
            ],
            "mutation": {"max_intra_rack_link_qty": 3, "max_inter_rack_link_qty": 3},
            "limits": {
                "max_total_cost": 300000,
                "max_peak_power_watts": 20000,
                "max_rack_power_watts": 10000,
                "max_rack_units": 42,
            },
        }
    )


def _feedback(*, compute: float, network: float, queue_ns: int = 0, remote_ns: int = 0, workload: str = "w"):
    summary = {
        "case_name": workload,
        "success": True,
        "inputs": {"workload": workload},
        "simulator": {"finished_count": 1, "expected_finished_count": 1},
    }
    stdout = f"""
    [x] [statistics] [info] sys[0], Wall time: 1000
    [x] [statistics] [info] sys[0], Average compute utilization: {compute * 100:.3f}%
    [x] [statistics] [info] sys[0], Remote mem provider queue time: {remote_ns}
    [x] [network] [info] Network top congested link rank=1 id=link_{workload} src_device=0 dst_device=1 level=L4 domain=cluster:cluster0 stats_domain=cluster:cluster0 technology=optical route_class= bytes=4096 busy_time_ns=80 queue_delay_ns={queue_ns} transmissions=2 max_queue_depth=3 utilization={network:.6f}
    """
    return parse_pipeline_feedback(summary=summary, simulator_stdout=stdout)


class RecordingPipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def run(
        self,
        *,
        topology_path: Path,
        workload_path: Path,
        out_dir: Path,
        workload_rank_parallel: bool | None = None,
        workload_kind: str | None = None,
        llm_use_all_gpus: bool | None = None,
    ):
        self.calls.append(
            {
                "topology_path": topology_path,
                "workload_path": workload_path,
                "out_dir": out_dir,
                "workload_rank_parallel": workload_rank_parallel,
                "workload_kind": workload_kind,
                "llm_use_all_gpus": llm_use_all_gpus,
            }
        )
        return _feedback(compute=0.9, network=0.1, workload=workload_path.stem)


def test_workload_suite_parser_normalizes_weights_and_resolves_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    workload_a = repo_root / "mapper" / "examples" / "a.json"
    workload_b = repo_root / "mapper" / "examples" / "b.json"
    workload_a.parent.mkdir(parents=True)
    workload_a.write_text("{}", encoding="utf-8")
    workload_b.write_text("{}", encoding="utf-8")
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        """
        {
          "name": "mixed",
          "workloads": [
            {"name": "a", "path": "mapper/examples/a.json", "weight": 2, "workload_rank_parallel": true},
            {"name": "b", "path": "mapper/examples/b.json", "kind": "llm-config"}
          ]
        }
        """,
        encoding="utf-8",
    )

    suite = load_workload_suite(suite_path, repo_root=repo_root)

    assert suite.workloads[0].path == workload_a
    assert suite.workloads[1].path == workload_b
    assert pytest.approx(sum(item.weight or 0 for item in suite.workloads)) == 1.0
    assert pytest.approx(suite.workloads[0].weight or 0) == 2 / 3
    assert pytest.approx(suite.workloads[1].weight or 0) == 1 / 3
    assert suite.workloads[0].workload_rank_parallel is True
    assert suite.workloads[1].workload_rank_parallel is False
    assert suite.workloads[0].workload_kind == "mapper"
    assert suite.workloads[1].workload_kind == "llm-config"
    assert suite.to_dict()["workloads"][0]["workload_rank_parallel"] is True
    assert suite.to_dict()["workloads"][1]["workload_kind"] == "llm-config"


def test_workload_rank_parallel_default_preserves_explicit_suite_values(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        """
        {
          "name": "mixed",
          "workloads": [
            {"name": "a", "path": "a.json", "workload_rank_parallel": true},
            {"name": "b", "path": "b.json", "workload_rank_parallel": false},
            {"name": "c", "path": "c.json"}
          ]
        }
        """,
        encoding="utf-8",
    )
    for name in ["a.json", "b.json", "c.json"]:
        (tmp_path / name).write_text("{}", encoding="utf-8")

    suite = load_workload_suite(suite_path, repo_root=tmp_path)
    suite = apply_workload_rank_parallel_default(
        suite,
        True,
        explicit_fields=workload_suite_rank_parallel_fields(suite_path),
    )

    assert [item.workload_rank_parallel for item in suite.workloads] == [True, False, True]


def test_multi_workload_runner_passes_workload_rank_parallel_per_item(tmp_path: Path) -> None:
    topology = tmp_path / "topology.json"
    workload_a = tmp_path / "a.json"
    workload_b = tmp_path / "b.json"
    topology.write_text("{}", encoding="utf-8")
    workload_a.write_text("{}", encoding="utf-8")
    workload_b.write_text("{}", encoding="utf-8")
    suite = WorkloadSuite.model_validate(
        {
            "name": "mixed",
            "workload_concurrency": 1,
            "workloads": [
                {"name": "a", "path": str(workload_a), "workload_rank_parallel": True},
                {"name": "b", "path": str(workload_b)},
            ],
        }
    )
    pipeline = RecordingPipeline()

    feedback = MultiWorkloadPipelineRunner(pipeline, suite).run(
        topology_path=topology,
        out_dir=tmp_path / "out",
        baseline=None,
    )

    assert feedback.baseline.suite_signature == suite.signature
    assert [call["workload_rank_parallel"] for call in pipeline.calls] == [True, False]
    payload = feedback.to_dict()
    assert payload["suite"]["workloads"][0]["workload_rank_parallel"] is True
    assert payload["workloads"][0]["workload_rank_parallel"] is True


def test_multi_workload_runner_uses_llm_config_item_without_workload_rank_parallel(tmp_path: Path) -> None:
    topology = tmp_path / "topology.json"
    mapper_workload = tmp_path / "taskgraph.json"
    llm_config = tmp_path / "qwenconfig.json"
    topology.write_text("{}", encoding="utf-8")
    mapper_workload.write_text("{}", encoding="utf-8")
    llm_config.write_text("{}", encoding="utf-8")
    suite = WorkloadSuite.model_validate(
        {
            "name": "mixed",
            "workload_concurrency": 1,
            "workloads": [
                {"name": "mapper", "path": str(mapper_workload), "workload_rank_parallel": True},
                {
                    "name": "qwen",
                    "path": str(llm_config),
                    "workload_kind": "llm-config",
                    "workload_rank_parallel": True,
                },
            ],
        }
    )
    pipeline = RecordingPipeline()

    feedback = MultiWorkloadPipelineRunner(pipeline, suite).run(
        topology_path=topology,
        out_dir=tmp_path / "out_llm",
        baseline=None,
    )

    assert pipeline.calls[0]["workload_kind"] == "mapper"
    assert pipeline.calls[0]["workload_rank_parallel"] is True
    assert pipeline.calls[0]["llm_use_all_gpus"] is False
    assert pipeline.calls[1]["workload_kind"] == "llm-config"
    assert pipeline.calls[1]["workload_rank_parallel"] is False
    assert pipeline.calls[1]["llm_use_all_gpus"] is True
    payload = feedback.to_dict()
    assert payload["suite"]["workloads"][1]["workload_kind"] == "llm-config"
    assert payload["suite"]["workloads"][1]["workload_rank_parallel"] is False
    assert payload["workloads"][1]["llm_use_all_gpus"] is True


def test_workload_suite_parser_rejects_duplicate_names(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        """
        {
          "name": "bad",
          "workloads": [
            {"name": "dup", "path": "a.json"},
            {"name": "dup", "path": "b.json"}
          ]
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate workload names"):
        load_workload_suite(suite_path, repo_root=tmp_path)


def test_multi_workload_speedup_and_high_water_telemetry() -> None:
    suite = WorkloadSuite.model_validate(
        {
            "name": "mixed",
            "workloads": [
                {"name": "llm", "path": "llm.json", "weight": 0.5},
                {"name": "gnn", "path": "gnn.json", "weight": 0.5},
            ],
        }
    )
    runs = [
            WorkloadRunFeedback(
                name="llm",
                path=Path("llm.json"),
                weight=0.5,
                workload_rank_parallel=False,
                out_dir=Path("/tmp/llm"),
            feedback=_feedback(compute=0.97, network=0.1, workload="llm"),
            speedup=2.0,
        ),
            WorkloadRunFeedback(
                name="gnn",
                path=Path("gnn.json"),
                weight=0.5,
                workload_rank_parallel=False,
                out_dir=Path("/tmp/gnn"),
            feedback=_feedback(compute=0.3, network=0.95, queue_ns=2_000_000, remote_ns=3_000_000, workload="gnn"),
            speedup=0.5,
        ),
    ]
    feedback = aggregate_multi_workload_feedback(
        suite,
        runs,
        WorkloadSuiteBaseline(suite_name="mixed", makespans_us={"llm": 2000, "gnn": 500}),
    )

    assert pytest.approx(feedback.geomean_speedup) == 1.0
    assert feedback.min_speedup == 0.5
    assert feedback.aggregate_feedback.max_link_utilization == 0.95
    assert feedback.aggregate_feedback.remote_memory_contention_ns == 3_000_000
    payload = feedback.to_dict()
    assert pytest.approx(payload["workloads"][0]["normalized_score"]) == 0.5
    assert pytest.approx(payload["workloads"][1]["normalized_score"]) == 2.0
    assert payload["workloads"][0]["baseline_makespan_us"] == 2000
    assert "weighted_log_score" in payload["workloads"][0]

    library = _library()
    space = _space()
    chromosome = chromosome_from_template(space.templates[0])
    report = CandidateRepairer(library, space).repair_and_validate(chromosome)
    context = telemetry_context(feedback, report, space)
    add_compute = GraphEditAction("add_node_to_slot", rack_id="rack0", resource="slot2", target="GPU")
    upgrade_link = GraphEditAction("upgrade_inter_rack_link", delta=1)
    downgrade_link = GraphEditAction("downgrade_inter_rack_link", delta=-1)

    assert heuristic_action_score(add_compute, chromosome, component_library=library, context=context) > 0
    assert heuristic_action_score(upgrade_link, chromosome, component_library=library, context=context) > 0
    assert heuristic_action_score(downgrade_link, chromosome, component_library=library, context=context) < heuristic_action_score(
        upgrade_link,
        chromosome,
        component_library=library,
        context=context,
    )
