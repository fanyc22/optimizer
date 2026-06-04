import pytest

from codesign_optimizer.models.hardware import ComponentLibrary
from codesign_optimizer.optimizer.chromosome import chromosome_from_template
from codesign_optimizer.optimizer.exporter import HardwareTopologyExporter
from codesign_optimizer.optimizer.search_space import RackTemplate


def _library() -> ComponentLibrary:
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
                    "radix": 16,
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
                    "bandwidth_gbps": 100,
                    "latency_ns": 800,
                    "protocol": "Optical",
                    "level": "L4",
                    "cost_unit": 3000,
                },
            },
        }
    )


def test_exporter_generates_v2_topology_with_rank_and_memory_pool() -> None:
    template = RackTemplate(
        name="case",
        racks=[
            {
                "rack_id": "rack0",
                "role": "hybrid",
                "max_slots": 3,
                "slots": [
                    {"slot_id": "slot0", "node_type": "GPU"},
                    {"slot_id": "slot1", "node_type": "GPU"},
                    {"slot_id": "slot2", "node_type": "CPU"},
                ],
                "memory_pool_count": 1,
                "switch_count": 1,
                "memory_pool_type": "MEM",
                "switch_type": "SW",
                "intra_rack_topology": "switch",
                "intra_rack_link_type": "FAST",
                "intra_rack_link_qty": 2,
                "memory_link_type": "CXL",
            }
        ],
        inter_rack="none",
    )
    exported = HardwareTopologyExporter(_library()).export(chromosome_from_template(template))
    topology = exported.hardware_topology

    assert topology["schema"] == "terrapod.hardware_topology.v2"
    assert len(topology["rank_map"]) == 3
    assert topology["defaults"]["memory_provider"]["node_id"] == "rack0_mem0"
    assert all("rack0_sw0" != item["node_id"] for item in topology["rank_map"])

    gpu_link = next(link for link in topology["links"] if link["id"] == "rack0_slot0_to_rack0_sw0")
    assert gpu_link["bandwidth_gbps"] == 200
    assert gpu_link["bidirectional"] is True


def test_exporter_supports_explicit_heterogeneous_racks() -> None:
    template = RackTemplate.model_validate(
        {
            "name": "hetero",
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
                    "rack_id": "mem-rack",
                    "role": "memory",
                    "max_slots": 0,
                    "slots": [],
                    "memory_pool_count": 2,
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
                        "max_rack_units": 8,
                        "max_power_watts": 4000,
                    },
                },
            ],
            "inter_rack": "ring",
            "inter_rack_link_type": "OPTICAL",
        }
    )

    exported = HardwareTopologyExporter(_library()).export(chromosome_from_template(template))
    topology = exported.hardware_topology

    assert exported.rank_count == 2
    assert {item["node_id"] for item in topology["rank_map"]} == {"gpu-rack_slot0", "gpu-rack_slot1"}
    assert "mem-rack_mem0" not in {item["node_id"] for item in topology["rank_map"]}
    assert topology["defaults"]["memory_provider"]["node_id"] == "mem-rack_mem0"
    rack_groups = {group["id"]: group for group in topology["hierarchy"]["groups"]}
    assert rack_groups["gpu-rack"]["attrs"]["role"] == "compute"
    assert rack_groups["mem-rack"]["attrs"]["role"] == "memory"
    assert rack_groups["mem-rack"]["attrs"]["capacity_limits"]["max_memory_pool_count"] == 2


def test_exporter_emits_intra_rack_topology_modes() -> None:
    expected_counts = {
        "ring": 3,
        "fully_connected": 3,
        "switch": 3,
    }

    for topology, expected_count in expected_counts.items():
        template = RackTemplate.model_validate(
            {
                "name": topology,
                "racks": [
                    {
                        "rack_id": "rack0",
                        "role": "compute",
                        "max_slots": 3,
                        "slots": [
                            {"slot_id": "slot0", "node_type": "GPU"},
                            {"slot_id": "slot1", "node_type": "GPU"},
                            {"slot_id": "slot2", "node_type": "CPU"},
                        ],
                        "switch_count": 1 if topology == "switch" else 0,
                        "switch_type": "SW" if topology == "switch" else None,
                        "intra_rack_topology": topology,
                        "intra_rack_link_type": "FAST",
                        "limits": {"max_slots": 3, "max_memory_pool_count": 0, "max_switch_count": 1},
                    }
                ],
                "inter_rack": "none",
            }
        )

        topology_json = HardwareTopologyExporter(_library()).export(chromosome_from_template(template)).hardware_topology
        rack_links = [link for link in topology_json["links"] if link["domain"] == "rack:rack0"]

        assert len(rack_links) == expected_count


def test_exporter_emits_inter_rack_topology_modes() -> None:
    expected_counts = {
        "ring": 3,
        "fully_connected": 3,
    }

    for inter_rack, expected_count in expected_counts.items():
        template = RackTemplate.model_validate(
            {
                "name": inter_rack,
                "racks": [
                    {
                        "rack_id": f"rack{idx}",
                        "role": "compute",
                        "max_slots": 1,
                        "slots": [{"slot_id": "slot0", "node_type": "GPU"}],
                        "switch_count": 1,
                        "switch_type": "SW",
                        "intra_rack_topology": "switch",
                        "intra_rack_link_type": "FAST",
                        "limits": {"max_slots": 1, "max_memory_pool_count": 0, "max_switch_count": 1},
                    }
                    for idx in range(3)
                ],
                "inter_rack": inter_rack,
                "inter_rack_link_type": "OPTICAL",
                "inter_rack_link_qty": 2,
            }
        )

        topology_json = HardwareTopologyExporter(_library()).export(chromosome_from_template(template)).hardware_topology
        rack_links = [link for link in topology_json["links"] if link["domain"] == "cluster:cluster0"]

        assert len(rack_links) == expected_count
        assert all(link["bandwidth_gbps"] == 200 for link in rack_links)


def test_exporter_rejects_disconnected_none_topologies() -> None:
    intra_none_template = RackTemplate.model_validate(
        {
            "name": "bad_intra_none",
            "racks": [
                {
                    "rack_id": "rack0",
                    "role": "compute",
                    "max_slots": 2,
                    "slots": [
                        {"slot_id": "slot0", "node_type": "GPU"},
                        {"slot_id": "slot1", "node_type": "GPU"},
                    ],
                    "switch_count": 0,
                    "intra_rack_topology": "none",
                    "intra_rack_link_type": "FAST",
                    "limits": {"max_slots": 2, "max_memory_pool_count": 0, "max_switch_count": 1},
                }
            ],
            "inter_rack": "none",
        }
    )
    intra_none = chromosome_from_template(intra_none_template)
    intra_none.racks[0].intra_rack_topology = "none"
    with pytest.raises(ValueError, match="intra_rack_topology=none"):
        HardwareTopologyExporter(_library()).export(intra_none)

    inter_none_template = RackTemplate.model_validate(
        {
            "name": "bad_inter_none",
            "racks": [
                {
                    "rack_id": f"rack{idx}",
                    "role": "compute",
                    "max_slots": 1,
                    "slots": [{"slot_id": "slot0", "node_type": "GPU"}],
                    "switch_count": 1,
                    "switch_type": "SW",
                    "intra_rack_topology": "switch",
                    "intra_rack_link_type": "FAST",
                    "limits": {"max_slots": 1, "max_memory_pool_count": 0, "max_switch_count": 1},
                }
                for idx in range(2)
            ],
            "inter_rack": "none",
        }
    )
    missing_link_type = chromosome_from_template(inter_none_template)
    with pytest.raises(ValueError, match="has no link_type"):
        HardwareTopologyExporter(_library()).export(missing_link_type)

    inter_none = chromosome_from_template(inter_none_template)
    inter_none.inter_rack = "none"
    with pytest.raises(ValueError, match="inter_rack topology=none"):
        HardwareTopologyExporter(_library()).export(inter_none)


def test_exporter_rejects_lower_level_link_for_inter_rack_scope() -> None:
    template = RackTemplate.model_validate(
        {
            "name": "bad_inter_scope",
            "racks": [
                {
                    "rack_id": f"rack{idx}",
                    "role": "compute",
                    "max_slots": 1,
                    "slots": [{"slot_id": "slot0", "node_type": "GPU"}],
                    "switch_count": 1,
                    "switch_type": "SW",
                    "intra_rack_topology": "switch",
                    "intra_rack_link_type": "FAST",
                    "limits": {"max_slots": 1, "max_memory_pool_count": 0, "max_switch_count": 1},
                }
                for idx in range(2)
            ],
            "inter_rack": "ring",
            "inter_rack_link_type": "FAST",
        }
    )

    with pytest.raises(ValueError, match="inter-rack scope"):
        HardwareTopologyExporter(_library()).export(chromosome_from_template(template))
