import json
import sys
from pathlib import Path

from codesign_optimizer.optimizer.pipeline_client import MapperSimulatorPipelineClient
from codesign_optimizer.optimizer.search_space import EvaluationSettings


def _fake_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    tools = repo / "tools"
    tools.mkdir(parents=True)
    (tools / "run_mapper_sim_pipeline.py").write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--topology")
parser.add_argument("--workload")
parser.add_argument("--out")
parser.add_argument("--mapper")
parser.add_argument("--parallel")
parser.add_argument("--topology-format")
parser.add_argument("--mapper-extra", action="append", default=[])
parser.add_argument("--sim-extra", action="append", default=[])
args = parser.parse_args()
case_dir = Path(args.out)
(case_dir / "intermediate" / "split_json").mkdir(parents=True, exist_ok=True)
(case_dir / "intermediate" / "split_json" / "rank0.json").write_text("{}", encoding="utf-8")
(case_dir / "workload").mkdir(parents=True, exist_ok=True)
(case_dir / "workload" / "chakra_trace.0.et").write_text("trace", encoding="utf-8")
(case_dir / "outputs").mkdir(parents=True, exist_ok=True)
stdout = case_dir / "outputs" / "simulator_stdout.txt"
stdout.write_text("[x] [statistics] [info] sys[0], Wall time: 123\\n", encoding="utf-8")
summary = {
    "case_name": "fake",
    "success": True,
    "inputs": {"workload": args.workload},
    "simulator": {
        "stdout": str(stdout),
        "finished_count": 1,
        "expected_finished_count": 1,
    },
}
(case_dir / "outputs" / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
""",
        encoding="utf-8",
    )
    return repo


def test_pipeline_client_cleans_large_wrapper_outputs_by_default(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    topology = tmp_path / "topology.json"
    workload = tmp_path / "workload.json"
    topology.write_text("{}", encoding="utf-8")
    workload.write_text("{}", encoding="utf-8")
    out_dir = tmp_path / "case"

    feedback = MapperSimulatorPipelineClient(
        repo_root=repo,
        evaluation=EvaluationSettings(),
        python=sys.executable,
    ).run(topology_path=topology, workload_path=workload, out_dir=out_dir)

    assert feedback.makespan_us == 123
    assert not (out_dir / "intermediate").exists()
    assert not (out_dir / "workload").exists()
    assert (out_dir / "outputs" / "run_summary.json").exists()
    assert (out_dir / "outputs" / "simulator_stdout.txt").exists()


def test_pipeline_client_can_keep_wrapper_outputs(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    topology = tmp_path / "topology.json"
    workload = tmp_path / "workload.json"
    topology.write_text("{}", encoding="utf-8")
    workload.write_text("{}", encoding="utf-8")
    out_dir = tmp_path / "case_keep"

    MapperSimulatorPipelineClient(
        repo_root=repo,
        evaluation=EvaluationSettings(cleanup_wrapper_intermediate=False),
        python=sys.executable,
    ).run(topology_path=topology, workload_path=workload, out_dir=out_dir)

    assert (out_dir / "intermediate" / "split_json").exists()
    assert (out_dir / "workload").exists()
