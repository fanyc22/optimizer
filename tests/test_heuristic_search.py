import random
import threading
import time
from pathlib import Path

import pytest

from codesign_optimizer.models.hardware import ComponentLibrary
from codesign_optimizer.optimizer.chromosome import chromosome_from_template, mutate_random
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


class SlowFakePipelineClient:
    def __init__(self, delay_s: float = 0.05) -> None:
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
            makespan = max(10, 1000 - call_id * 10)
            summary = {
                "case_name": topology_path.stem,
                "success": True,
                "inputs": {"workload": str(workload_path)},
                "simulator": {"finished_count": 1, "expected_finished_count": 1},
            }
            stdout = f"""
            [x] [statistics] [info] sys[0], Wall time: {makespan}
            [x] [network] [info] Network top congested link rank=1 id=link{call_id} src_device=0 dst_device=1 level=L4 domain=cluster:cluster0 stats_domain=cluster:cluster0 technology=optical route_class= bytes=4096 busy_time_ns=80 queue_delay_ns=30 transmissions=2 max_queue_depth=3 utilization=0.750000
            """
            return parse_pipeline_feedback(summary=summary, simulator_stdout=stdout)
        finally:
            with self._lock:
                self.active -= 1


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
            "seed": 3,
            "templates": [
                {
                    "name": "two_rack",
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
                        },
                        {
                            "rack_id": "rack1",
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
                        }
                    ],
                    "inter_rack_link_type": "OPTICAL",
                    "inter_rack": "ring",
                }
            ],
            "mutation": {
                "max_intra_rack_link_qty": 3,
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


def _hetero_space() -> SearchSpace:
    return SearchSpace.model_validate(
        {
            "seed": 11,
            "templates": [
                {
                    "name": "hetero_racks",
                    "racks": [
                        {
                            "rack_id": "gpu-rack",
                            "role": "compute",
                            "max_slots": 2,
                            "slots": [
                                {"slot_id": "slot0", "node_type": "GPU"},
                                {"slot_id": "slot1", "node_type": "GPU"},
                            ],
                            "switch_count": 1,
                            "switch_type": "SW",
                            "intra_rack_topology": "switch",
                            "intra_rack_link_type": "FAST",
                            "limits": {
                                "max_slots": 2,
                                "max_memory_pool_count": 0,
                                "max_switch_count": 1,
                                "max_rack_units": 8,
                                "max_power_watts": 4000,
                            },
                        },
                        {
                            "rack_id": "cpu-rack",
                            "role": "compute",
                            "max_slots": 2,
                            "slots": [
                                {"slot_id": "slot0", "node_type": "CPU"},
                                {"slot_id": "slot1", "node_type": "CPU"},
                            ],
                            "switch_count": 1,
                            "switch_type": "SW",
                            "intra_rack_topology": "switch",
                            "intra_rack_link_type": "FAST",
                            "limits": {
                                "max_slots": 2,
                                "max_memory_pool_count": 0,
                                "max_switch_count": 1,
                                "max_rack_units": 8,
                                "max_power_watts": 4000,
                            },
                        },
                        {
                            "rack_id": "mem-rack",
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
                                "max_memory_pool_count": 1,
                                "max_switch_count": 1,
                                "max_rack_units": 8,
                                "max_power_watts": 4000,
                            },
                        },
                    ],
                    "inter_rack": "ring",
                    "inter_rack_link_type": "OPTICAL",
                }
            ],
            "mutation": {
                "max_intra_rack_link_qty": 3,
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


def test_heuristic_search_evaluates_candidates_concurrently(tmp_path: Path) -> None:
    workload = tmp_path / "workload.json"
    workload.write_text("{}", encoding="utf-8")
    pipeline = SlowFakePipelineClient()
    runner = HeuristicSearchRunner(
        component_library=_library(),
        search_space=_space(),
        pipeline_client=pipeline,
        workload_path=workload,
        out_dir=tmp_path / "parallel_search",
        population_size=4,
        generations=1,
        concurrency=3,
    )

    result = runner.run()

    assert result.best.feasible
    assert pipeline.calls > 0
    assert pipeline.max_active >= 2
    assert (tmp_path / "parallel_search" / "summary.json").exists()


def test_heterogeneous_rack_search_runs_with_fake_pipeline(tmp_path: Path) -> None:
    workload = tmp_path / "workload.json"
    workload.write_text("{}", encoding="utf-8")
    pipeline = FakePipelineClient()
    runner = HeuristicSearchRunner(
        component_library=_library(),
        search_space=_hetero_space(),
        pipeline_client=pipeline,
        workload_path=workload,
        out_dir=tmp_path / "hetero_search",
        population_size=3,
        generations=2,
    )

    result = runner.run()

    assert pipeline.calls > 0
    assert result.best.feasible
    best_topology = tmp_path / "hetero_search" / "best_hardware_topology.json"
    assert best_topology.exists()


def test_repair_marks_switch_radix_violation() -> None:
    space = _space()
    library = _library(radix=1)
    chromosome = space.templates[0]
    from codesign_optimizer.optimizer.chromosome import chromosome_from_template

    report = CandidateRepairer(library, space).repair_and_validate(chromosome_from_template(chromosome))

    assert not report.feasible
    assert any("switch radix exceeded" in msg for msg in report.messages)


def test_repair_allows_memory_only_rack_without_adding_compute() -> None:
    space = _hetero_space()
    chromosome = chromosome_from_template(space.templates[0])

    report = CandidateRepairer(_library(), space).repair_and_validate(chromosome)

    assert report.feasible
    memory_rack = next(rack for rack in report.chromosome.racks if rack.role == "memory")
    assert memory_rack.gpu_count == 0
    assert memory_rack.cpu_count == 0
    assert memory_rack.memory_pool_count == 1


def test_slots_schema_rejects_legacy_count_template() -> None:
    with pytest.raises(ValueError):
        SearchSpace.model_validate(
            {
                "templates": [
                    {
                        "name": "legacy",
                        "rack_count": 1,
                        "gpu_count": 1,
                        "cpu_count": 0,
                        "gpu_type": "GPU",
                        "endpoint_link_type": "FAST",
                    }
                ]
            }
        )


def test_slots_schema_rejects_bad_slot_layouts() -> None:
    base_rack = {
        "rack_id": "rack0",
        "role": "compute",
        "max_slots": 2,
        "slots": [{"slot_id": "slot0", "node_type": "GPU"}],
        "switch_count": 1,
        "switch_type": "SW",
        "intra_rack_topology": "switch",
        "intra_rack_link_type": "FAST",
        "limits": {"max_slots": 2, "max_memory_pool_count": 0, "max_switch_count": 1},
    }

    for rack in [
        {**base_rack, "max_slots": 1, "slots": [{"slot_id": "slot0", "node_type": "GPU"}, {"slot_id": "slot1"}]},
        {**base_rack, "slots": [{"slot_id": "slot0", "node_type": "GPU"}, {"slot_id": "slot0"}]},
        {**base_rack, "slots": []},
    ]:
        with pytest.raises(ValueError):
            SearchSpace.model_validate({"templates": [{"name": "bad", "racks": [rack], "inter_rack": "none"}]})


def test_unknown_slot_node_type_is_rejected_by_repair() -> None:
    space = SearchSpace.model_validate(
        {
            "templates": [
                {
                    "name": "unknown_node",
                    "racks": [
                        {
                            "rack_id": "rack0",
                            "role": "compute",
                            "max_slots": 1,
                            "slots": [{"slot_id": "slot0", "node_type": "NOT_IN_CATALOG"}],
                            "switch_count": 1,
                            "switch_type": "SW",
                            "intra_rack_topology": "switch",
                            "intra_rack_link_type": "FAST",
                            "limits": {"max_slots": 1, "max_memory_pool_count": 0, "max_switch_count": 1},
                        }
                    ],
                    "inter_rack": "none",
                }
            ]
        }
    )

    report = CandidateRepairer(_library(), space).repair_and_validate(chromosome_from_template(space.templates[0]))

    assert not report.feasible
    assert any("unknown node type" in message for message in report.messages)


def test_mutation_respects_rack_roles() -> None:
    space = _hetero_space()
    chromosome = chromosome_from_template(space.templates[0])

    mutated = mutate_random(chromosome, space, random.Random(5), intensity=50)

    for rack in mutated.racks:
        if rack.role == "memory":
            assert rack.gpu_count == 0
            assert rack.cpu_count == 0
            assert rack.memory_pool_count > 0
        if rack.role == "compute":
            assert rack.memory_pool_count == 0
            assert rack.gpu_count + rack.cpu_count > 0
            if rack.gpu_type is None:
                assert rack.gpu_count == 0
            if rack.cpu_type is None:
                assert rack.cpu_count == 0


def test_ring_radix_uses_actual_inter_rack_degree() -> None:
    library = _library(radix=2)
    space = SearchSpace.model_validate(
        {
            "templates": [
                {
                    "name": "two_rack_degree_one",
                    "racks": [
                        {
                            "rack_id": "rack0",
                            "role": "compute",
                            "max_slots": 1,
                            "slots": [{"slot_id": "slot0", "node_type": "GPU"}],
                            "switch_count": 1,
                            "switch_type": "SW",
                            "intra_rack_topology": "switch",
                            "intra_rack_link_type": "FAST",
                        },
                        {
                            "rack_id": "rack1",
                            "role": "compute",
                            "max_slots": 1,
                            "slots": [{"slot_id": "slot0", "node_type": "GPU"}],
                            "switch_count": 1,
                            "switch_type": "SW",
                            "intra_rack_topology": "switch",
                            "intra_rack_link_type": "FAST",
                        },
                    ],
                    "inter_rack": "ring",
                    "inter_rack_link_type": "OPTICAL",
                }
            ],
            "limits": {
                "max_total_cost": 200000,
                "max_peak_power_watts": 20000,
                "max_rack_power_watts": 10000,
                "max_rack_units": 42,
            },
        }
    )

    report = CandidateRepairer(library, space).repair_and_validate(
        chromosome_from_template(space.templates[0])
    )

    assert report.feasible


def test_rack_capacity_limits_bound_repair() -> None:
    space = _hetero_space()
    chromosome = chromosome_from_template(space.templates[0])
    gpu_rack = next(rack for rack in chromosome.racks if rack.rack_id == "gpu-rack")
    mem_rack = next(rack for rack in chromosome.racks if rack.rack_id == "mem-rack")
    gpu_rack.slots.append(type(gpu_rack.slots[0])(slot_id="slot2", node_type="GPU"))
    mem_rack.memory_pool_count = 8

    report = CandidateRepairer(_library(), space).repair_and_validate(chromosome)

    repaired_gpu_rack = next(rack for rack in report.chromosome.racks if rack.rack_id == "gpu-rack")
    repaired_mem_rack = next(rack for rack in report.chromosome.racks if rack.rack_id == "mem-rack")
    assert repaired_gpu_rack.gpu_count == 3
    assert not report.feasible
    assert repaired_mem_rack.memory_pool_count == 1


def test_explicit_rack_slot_limits_are_copied() -> None:
    space = SearchSpace.model_validate(
        {
            "templates": [
                {
                    "name": "limited_homogeneous",
                    "racks": [
                        {
                            "rack_id": "rack0",
                            "role": "compute",
                            "max_slots": 2,
                            "slots": [{"slot_id": "slot0", "node_type": "GPU"}, {"slot_id": "slot1"}],
                            "switch_count": 1,
                            "switch_type": "SW",
                            "intra_rack_topology": "switch",
                            "intra_rack_link_type": "FAST",
                            "limits": {"max_slots": 2, "max_memory_pool_count": 0, "max_switch_count": 1},
                        },
                        {
                            "rack_id": "rack1",
                            "role": "compute",
                            "max_slots": 2,
                            "slots": [{"slot_id": "slot0", "node_type": "GPU"}, {"slot_id": "slot1"}],
                            "switch_count": 1,
                            "switch_type": "SW",
                            "intra_rack_topology": "switch",
                            "intra_rack_link_type": "FAST",
                            "limits": {"max_slots": 2, "max_memory_pool_count": 0, "max_switch_count": 1},
                        },
                    ],
                    "inter_rack": "none",
                }
            ],
        }
    )
    chromosome = chromosome_from_template(space.templates[0])

    report = CandidateRepairer(_library(), space).repair_and_validate(chromosome)

    assert report.feasible
    assert [rack.gpu_count for rack in report.chromosome.racks] == [1, 1]
    assert all(rack.limits.max_slots == 2 for rack in report.chromosome.racks)


def test_rack_specific_power_limit_marks_candidate_infeasible() -> None:
    space = _hetero_space()
    chromosome = chromosome_from_template(space.templates[0])
    gpu_rack = next(rack for rack in chromosome.racks if rack.rack_id == "gpu-rack")
    gpu_rack.limits.max_power_watts = 100

    report = CandidateRepairer(_library(), space).repair_and_validate(chromosome)

    assert not report.feasible
    assert any("gpu-rack power exceeds limit" in msg for msg in report.messages)
