import json
from pathlib import Path

from codesign_optimizer.io.jsonc import load_jsonc
from codesign_optimizer.models.hardware import ComponentLibrary
from codesign_optimizer.optimizer.exhaustive import (
    ExhaustiveSearchRunner,
    count_exhaustive_candidates,
    iter_exhaustive_chromosomes,
    validate_exhaustive_space,
)
from codesign_optimizer.optimizer.feedback_parser import ParsedPipelineFeedback, parse_pipeline_feedback
from codesign_optimizer.optimizer.search_space import SearchSpace, load_component_library


class TopologyAwarePipeline:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, *, topology_path: Path, workload_path: Path, out_dir: Path) -> ParsedPipelineFeedback:
        self.calls += 1
        out_dir.mkdir(parents=True, exist_ok=True)
        topology = json.loads(topology_path.read_text(encoding="utf-8"))
        gpu_count = sum(
            1
            for node in topology["nodes"]
            if node.get("attrs", {}).get("node_type") == "GPU"
        )
        makespan = 1000 - gpu_count * 100
        summary = {
            "case_name": topology_path.stem,
            "success": True,
            "inputs": {"workload": str(workload_path)},
            "simulator": {"finished_count": 1, "expected_finished_count": 1},
        }
        stdout = f"""
        [x] [statistics] [info] sys[0], Wall time: {makespan}
        [x] [statistics] [info] sys[0], Average compute utilization: 75.000%
        [x] [network] [info] Network top congested link rank=1 id=rack0_sw0_to_rack1_sw0 src_device=0 dst_device=1 level=L4 domain=cluster:cluster0 stats_domain=cluster:cluster0 technology=optical route_class= bytes=4096 busy_time_ns=80 queue_delay_ns=30 transmissions=2 max_queue_depth=3 utilization=0.500000
        """
        return parse_pipeline_feedback(summary=summary, simulator_stdout=stdout)


def _library() -> ComponentLibrary:
    return ComponentLibrary.model_validate(
        {
            "node_types": {
                "GPU": {"role": "gpu", "peak_tflops": 80, "tdp_watts": 700, "cost_unit": 20000},
                "CPU": {"role": "cpu", "peak_tflops": 6, "tdp_watts": 350, "cost_unit": 6000},
                "SW": {"role": "switch", "radix": 32, "tdp_watts": 180, "cost_unit": 8000},
            },
            "link_types": {
                "FAST": {
                    "bandwidth_gbps": 100,
                    "latency_ns": 100,
                    "protocol": "NVLink",
                    "level": "L3",
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
            "seed": 7,
            "templates": [
                {
                    "name": "tiny",
                    "racks": [
                        {
                            "rack_id": "rack0",
                            "role": "compute",
                            "max_slots": 1,
                            "slots": [{"slot_id": "slot0", "node_type": "GPU", "link_type": "FAST"}],
                            "switch_count": 1,
                            "switch_type": "SW",
                            "intra_rack_topology": "switch",
                            "intra_rack_link_type": "FAST",
                            "limits": {"max_slots": 1, "max_memory_pool_count": 0, "max_switch_count": 1},
                        },
                        {
                            "rack_id": "rack1",
                            "role": "compute",
                            "max_slots": 1,
                            "slots": [{"slot_id": "slot0", "node_type": "CPU", "link_type": "FAST"}],
                            "switch_count": 1,
                            "switch_type": "SW",
                            "intra_rack_topology": "switch",
                            "intra_rack_link_type": "FAST",
                            "limits": {"max_slots": 1, "max_memory_pool_count": 0, "max_switch_count": 1},
                        },
                    ],
                    "inter_rack": "ring",
                    "inter_rack_link_type": "OPTICAL",
                }
            ],
            "exhaustive": {
                "slot_options": [
                    {"node_type": "GPU", "link_type": "FAST", "link_qty": 1},
                    {"node_type": "CPU", "link_type": "FAST", "link_qty": 1},
                ],
                "intra_rack_topologies": ["switch"],
                "intra_rack_link_types": ["FAST"],
                "intra_rack_link_qty": [1],
                "inter_rack_topologies": ["ring"],
                "inter_rack_link_types": ["OPTICAL"],
                "inter_rack_link_qty": [1],
                "max_candidates": 4,
            },
            "limits": {
                "max_total_cost": 200000,
                "max_peak_power_watts": 20000,
                "max_rack_power_watts": 10000,
                "max_total_racks": 2,
                "min_compute_racks": 2,
                "max_compute_racks": 2,
                "max_memory_racks": 0,
                "max_hybrid_racks": 0,
            },
        }
    )


def test_exhaustive_enumeration_counts_example_space() -> None:
    optimizer_root = Path(__file__).resolve().parents[1]
    catalog = load_component_library(load_jsonc(optimizer_root / "examples" / "component_catalog_tcro_latent_rack.json"))
    examples = [
        ("search_space_3rack_exhaustive_tiny.json", 3, 2, 64),
        ("search_space_4rack_exhaustive_tiny.json", 4, 2, 256),
        ("search_space_5rack_exhaustive_tiny.json", 5, 2, 1024),
        ("search_space_1rack_4device_exhaustive_topology.json", 1, 4, 64),
        ("search_space_2rack_4device_exhaustive_topology.json", 2, 4, 12288),
    ]
    for filename, rack_count, slots_per_rack, candidate_count in examples:
        space = SearchSpace.model_validate(load_jsonc(optimizer_root / "examples" / filename))

        validate_exhaustive_space(space, component_library=catalog)

        assert len(space.templates[0].racks) == rack_count
        assert all(rack.max_slots == slots_per_rack for rack in space.templates[0].racks)
        assert count_exhaustive_candidates(space) == candidate_count
        chromosomes = (
            iter_exhaustive_chromosomes(space)
            if candidate_count <= 1024
            else iter_exhaustive_chromosomes(space, freeze_topology=True)
        )
        assert len(chromosomes) == (candidate_count if candidate_count <= 1024 else 256)
        assert all(
            slot.link_type == ("GPU_FABRIC_200G" if slot.node_type == "GPU_BALANCED_80TF_80GB" else "CPU_PCIE_128G")
            for chromosome in chromosomes
            for rack in chromosome.racks
            for slot in rack.slots
        )


def test_exhaustive_freeze_topology_reduces_topology_examples() -> None:
    optimizer_root = Path(__file__).resolve().parents[1]
    examples = [
        ("search_space_1rack_4device_exhaustive_topology.json", 64, 16),
        ("search_space_2rack_4device_exhaustive_topology.json", 12288, 256),
    ]
    for filename, unfrozen_count, frozen_count in examples:
        space = SearchSpace.model_validate(load_jsonc(optimizer_root / "examples" / filename))

        assert count_exhaustive_candidates(space) == unfrozen_count
        assert count_exhaustive_candidates(space, freeze_topology=True) == frozen_count
        frozen_chromosomes = iter_exhaustive_chromosomes(space, freeze_topology=True)
        assert len(frozen_chromosomes) == frozen_count
        assert all(
            rack.intra_rack_topology == "switch"
            for chromosome in frozen_chromosomes
            for rack in chromosome.racks
        )


def test_exhaustive_runner_finds_best_candidate(tmp_path: Path) -> None:
    workload = tmp_path / "workload.json"
    workload.write_text("{}", encoding="utf-8")
    pipeline = TopologyAwarePipeline()
    runner = ExhaustiveSearchRunner(
        component_library=_library(),
        search_space=_space(),
        pipeline_client=pipeline,
        workload_path=workload,
        out_dir=tmp_path / "exhaustive",
        concurrency=2,
    )

    result = runner.run()

    best_types = [
        slot.node_type
        for rack in result.best.chromosome.racks
        for slot in rack.slots
    ]
    assert result.total_candidates == 4
    assert result.unique_candidates == 4
    assert pipeline.calls == 4
    assert result.best.feasible
    assert len(result.feasible_candidates) == 4
    assert best_types == ["GPU", "GPU"]
    assert (tmp_path / "exhaustive" / "best_hardware_topology.json").exists()
    assert (tmp_path / "exhaustive" / "exhaustive_summary.json").exists()
    assert (tmp_path / "exhaustive" / "feasible_candidates.jsonl").exists()
    assert (tmp_path / "exhaustive" / "feasible_summary.json").exists()


def test_exhaustive_runner_outputs_feasible_candidates_under_rack_cost_limit(tmp_path: Path) -> None:
    workload = tmp_path / "workload.json"
    workload.write_text("{}", encoding="utf-8")
    space = _space().model_copy(deep=True)
    space.limits.max_rack_cost = 15_000
    pipeline = TopologyAwarePipeline()
    runner = ExhaustiveSearchRunner(
        component_library=_library(),
        search_space=space,
        pipeline_client=pipeline,
        workload_path=workload,
        out_dir=tmp_path / "exhaustive",
        concurrency=2,
    )

    result = runner.run()

    assert result.total_candidates == 4
    assert result.unique_candidates == 4
    assert len(result.feasible_candidates) == 1
    assert pipeline.calls == 1
    feasible_types = [
        slot.node_type
        for rack in result.feasible_candidates[0].chromosome.racks
        for slot in rack.slots
    ]
    assert feasible_types == ["CPU", "CPU"]
    feasible_lines = (tmp_path / "exhaustive" / "feasible_candidates.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(feasible_lines) == 1
    feasible_summary = json.loads((tmp_path / "exhaustive" / "feasible_summary.json").read_text(encoding="utf-8"))
    assert feasible_summary["feasible_evaluations"] == 1
