from codesign_optimizer.optimizer.feedback_parser import parse_pipeline_feedback


def test_feedback_parser_extracts_network_and_remote_memory_metrics() -> None:
    summary = {
        "case_name": "case",
        "success": True,
        "inputs": {"workload": "workload.json"},
        "simulator": {"finished_count": 2, "expected_finished_count": 2},
    }
    stdout = """
    [x] [workload] [info] sys[0] finished, 120 cycles, exposed communication 10 cycles.
    [x] [statistics] [info] sys[0], Wall time: 120
    [x] [statistics] [info] sys[0], Average compute utilization: 50.000%
    [x] [statistics] [info] sys[0], Average memory utilization: 25.000%
    [x] [statistics] [info] sys[0], Remote mem provider queue time: 7
    [x] [statistics] [info] sys[0], Remote mem provider service time: 11
    [x] [workload] [info] sys[1] finished, 140 cycles, exposed communication 10 cycles.
    [x] [statistics] [info] sys[1], Wall time: 140
    [x] [network] [info] Network top congested link rank=1 id=a_to_b src_device=0 dst_device=1 level=L4 domain=cluster:cluster0 stats_domain=cluster:cluster0 technology=optical route_class= bytes=4096 busy_time_ns=80 queue_delay_ns=30 transmissions=2 max_queue_depth=3 utilization=0.750000
    [x] [network] [info] Network top congested domain rank=1 stats_domain=cluster:cluster0 bytes=4096 busy_time_ns=80 queue_delay_ns=30 transmissions=2 max_queue_depth=3 utilization=0.750000
    [x] [system] [info] Scaling report route_cache requests=4 hits=2 misses=2 dijkstra_runs=2 source_cache_entries=2 cached_paths=2
    """

    parsed = parse_pipeline_feedback(summary=summary, simulator_stdout=stdout)

    assert parsed.makespan_us == 140
    assert parsed.max_link_utilization == 0.75
    assert parsed.max_queue_delay_ns == 30
    assert parsed.remote_memory_contention_ns == 7
    assert parsed.scaling_report["route_cache"]["dijkstra_runs"] == 2
    assert parsed.simulation_feedback.compute_profile["sys0"].avg_utilization == 0.5
