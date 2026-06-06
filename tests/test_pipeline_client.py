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
parser.add_argument("--llm-config")
parser.add_argument("--llm-prefill-batch")
parser.add_argument("--llm-prompt-len")
parser.add_argument("--llm-decode-batch")
parser.add_argument("--llm-decode-steps")
parser.add_argument("--llm-avg-context-len")
parser.add_argument("--llm-tp")
parser.add_argument("--llm-pp")
parser.add_argument("--llm-cp")
parser.add_argument("--llm-dp")
parser.add_argument("--out")
parser.add_argument("--mapper")
parser.add_argument("--parallel")
parser.add_argument("--topology-format")
parser.add_argument("--mapper-extra", action="append", default=[])
parser.add_argument("--sim-extra", action="append", default=[])
parser.add_argument("--calibration-fit-model")
args = parser.parse_args()
if bool(args.workload) == bool(args.llm_config):
    raise SystemExit(3)
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
    "inputs": {
        "workload": args.workload or args.llm_config,
        "llm_config": args.llm_config,
        "llm_prompt_len": args.llm_prompt_len,
        "llm_decode_steps": args.llm_decode_steps,
        "llm_tp": args.llm_tp,
        "topology": args.topology,
        "out": args.out,
        "calibration_fit_model": args.calibration_fit_model,
        "materialized_calibration": args.calibration_fit_model is not None,
    },
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
    summary = json.loads((out_dir / "outputs" / "run_summary.json").read_text(encoding="utf-8"))
    assert Path(summary["inputs"]["topology"]).is_absolute()
    assert Path(summary["inputs"]["workload"]).is_absolute()
    assert Path(summary["inputs"]["out"]).is_absolute()


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


def test_pipeline_client_passes_calibration_fit_model(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    topology = tmp_path / "topology.json"
    workload = tmp_path / "workload.json"
    model = repo / "calibration" / "calibration_fit_model.json"
    topology.write_text("{}", encoding="utf-8")
    workload.write_text("{}", encoding="utf-8")
    model.parent.mkdir(parents=True)
    model.write_text("{}", encoding="utf-8")
    out_dir = tmp_path / "case_calibration"

    MapperSimulatorPipelineClient(
        repo_root=repo,
        evaluation=EvaluationSettings(
            calibration_fit_model=Path("calibration/calibration_fit_model.json"),
            cleanup_wrapper_intermediate=False,
        ),
        python=sys.executable,
    ).run(topology_path=topology, workload_path=workload, out_dir=out_dir)

    summary = json.loads((out_dir / "outputs" / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["inputs"]["calibration_fit_model"] == str(model.resolve())
    assert summary["inputs"]["materialized_calibration"] is True


def test_pipeline_client_can_pass_llm_evaluation_args(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    topology = tmp_path / "topology.json"
    llm_config = tmp_path / "config.json"
    topology.write_text("{}", encoding="utf-8")
    llm_config.write_text("{}", encoding="utf-8")
    out_dir = tmp_path / "case_llm"

    MapperSimulatorPipelineClient(
        repo_root=repo,
        evaluation=EvaluationSettings(
            workload_kind="llm",
            cleanup_wrapper_intermediate=False,
            llm_prefill_batch_size=2,
            llm_prompt_len=128,
            llm_decode_batch_size=3,
            llm_decode_steps=4,
            llm_avg_context_len=256,
            llm_tp=2,
            llm_pp=1,
            llm_cp=1,
            llm_dp=1,
        ),
        python=sys.executable,
    ).run(topology_path=topology, workload_path=llm_config, out_dir=out_dir)

    summary = json.loads((out_dir / "outputs" / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["inputs"]["workload"] == str(llm_config.resolve())
    assert summary["inputs"]["llm_config"] == str(llm_config.resolve())
    assert summary["inputs"]["llm_prompt_len"] == "128"
    assert summary["inputs"]["llm_decode_steps"] == "4"
    assert summary["inputs"]["llm_tp"] == "2"
