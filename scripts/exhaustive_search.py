#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import typer
from rich.console import Console


OPTIMIZER_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = OPTIMIZER_ROOT / "src"
REPO_ROOT = OPTIMIZER_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from codesign_optimizer.io.jsonc import load_jsonc  # noqa: E402
from codesign_optimizer.optimizer.exhaustive import ExhaustiveSearchRunner  # noqa: E402
from codesign_optimizer.optimizer.pipeline_client import MapperSimulatorPipelineClient  # noqa: E402
from codesign_optimizer.optimizer.search_space import SearchSpace, load_component_library  # noqa: E402
from codesign_optimizer.utils.logging import configure_logging  # noqa: E402


console = Console()


def main(
    catalog: Path = typer.Option(..., exists=True, readable=True, help="Component catalog JSON/JSONC."),
    space: Path = typer.Option(..., exists=True, readable=True, help="Finite exhaustive search-space JSON/JSONC."),
    workload: Path = typer.Option(..., exists=True, readable=True, help="Mapper workload JSON."),
    concurrency: int = typer.Option(1, min=1, max=1024, help="Concurrent mapper/simulator evaluations."),
    max_candidates: int | None = typer.Option(None, min=1, help="Override exhaustive.max_candidates guard."),
    out: Path = typer.Option(Path("artifacts/exhaustive_run"), help="Search output directory."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logs."),
) -> None:
    configure_logging(verbose=verbose)
    component_library = load_component_library(load_jsonc(catalog))
    search_space = SearchSpace.model_validate(load_jsonc(space))
    repo_root = search_space.evaluation.repo_root or REPO_ROOT
    pipeline = MapperSimulatorPipelineClient(repo_root=repo_root, evaluation=search_space.evaluation)
    runner = ExhaustiveSearchRunner(
        component_library=component_library,
        search_space=search_space,
        pipeline_client=pipeline,
        workload_path=workload,
        out_dir=out,
        concurrency=concurrency,
        max_candidates=max_candidates,
    )
    result = runner.run()
    console.print(
        "[green]Exhaustive search completed[/green]\n"
        f"Total candidates: {result.total_candidates}\n"
        f"Unique candidates: {result.unique_candidates}\n"
        f"Evaluations: {len(result.history)}\n"
        f"Feasible evaluations: {sum(1 for item in result.history if item.feasible)}\n"
        f"Best score: {result.best.weighted_score:.4f}\n"
        f"Best feasible: {result.best.feasible}\n"
        f"Artifacts: {out}"
    )


if __name__ == "__main__":
    typer.run(main)
