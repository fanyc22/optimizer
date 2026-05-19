from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from codesign_optimizer.optimizer.feedback_parser import (
    ParsedPipelineFeedback,
    parse_pipeline_outputs,
)
from codesign_optimizer.optimizer.search_space import EvaluationSettings


logger = logging.getLogger(__name__)


class PipelineClient(Protocol):
    def run(self, *, topology_path: Path, workload_path: Path, out_dir: Path) -> ParsedPipelineFeedback:
        ...


@dataclass(frozen=True)
class MapperSimulatorPipelineClient:
    repo_root: Path
    evaluation: EvaluationSettings
    python: str = "python3"

    def run(self, *, topology_path: Path, workload_path: Path, out_dir: Path) -> ParsedPipelineFeedback:
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.python,
            str(self.repo_root / "tools" / "run_mapper_sim_pipeline.py"),
            "--topology",
            str(topology_path),
            "--workload",
            str(workload_path),
            "--out",
            str(out_dir),
            "--mapper",
            self.evaluation.mapper,
            "--parallel",
            self.evaluation.parallel,
            "--topology-format",
            self.evaluation.topology_format,
        ]
        for item in self.evaluation.mapper_extra:
            cmd.extend(["--mapper-extra", item])
        sim_extra = list(self.evaluation.sim_extra)
        if self.evaluation.scaling_report and "--scaling-report=true" not in sim_extra:
            sim_extra.append("--scaling-report=true")
        for item in sim_extra:
            cmd.extend(["--sim-extra", item])

        result = subprocess.run(
            cmd,
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.evaluation.timeout_seconds,
            check=False,
        )
        (out_dir / "optimizer_pipeline_stdout.txt").write_text(result.stdout, encoding="utf-8")
        (out_dir / "optimizer_pipeline_stderr.txt").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(
                f"mapper/simulator pipeline failed with exit code {result.returncode}; "
                f"see {out_dir / 'optimizer_pipeline_stderr.txt'}"
            )
        feedback = parse_pipeline_outputs(out_dir)
        if self.evaluation.cleanup_wrapper_intermediate:
            _cleanup_large_wrapper_outputs(out_dir)
        return feedback


def _cleanup_large_wrapper_outputs(out_dir: Path) -> None:
    targets = [
        out_dir / "intermediate",
        out_dir / "workload",
    ]
    removed: list[str] = []
    for target in targets:
        if not target.exists():
            continue
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        removed.append(str(target))
    if removed:
        logger.info("Cleaned large wrapper outputs: %s", ", ".join(removed))
