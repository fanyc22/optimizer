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
                    "cost_unit": 1000,
                },
                "CXL": {
                    "bandwidth_gbps": 64,
                    "latency_ns": 250,
                    "protocol": "CXL",
                    "cost_unit": 300,
                },
            },
        }
    )


def test_exporter_generates_v2_topology_with_rank_and_memory_pool() -> None:
    template = RackTemplate(
        name="case",
        rack_count=1,
        gpu_count=2,
        cpu_count=1,
        memory_pool_count=1,
        switch_count=1,
        gpu_type="GPU",
        cpu_type="CPU",
        memory_pool_type="MEM",
        switch_type="SW",
        endpoint_link_type="FAST",
        memory_link_type="CXL",
        endpoint_link_qty=2,
        fabric="switch",
        inter_rack="none",
    )
    exported = HardwareTopologyExporter(_library()).export(chromosome_from_template(template))
    topology = exported.hardware_topology

    assert topology["schema"] == "terrapod.hardware_topology.v2"
    assert len(topology["rank_map"]) == 3
    assert topology["defaults"]["memory_provider"]["node_id"] == "rack0_mem0"
    assert all("rack0_sw0" != item["node_id"] for item in topology["rank_map"])

    gpu_link = next(link for link in topology["links"] if link["id"] == "rack0_gpu0_to_rack0_sw0")
    assert gpu_link["bandwidth_gbps"] == 200
    assert gpu_link["bidirectional"] is True
