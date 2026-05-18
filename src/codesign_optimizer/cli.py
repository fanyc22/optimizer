from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from codesign_optimizer.config.settings import OptimizerSettings
from codesign_optimizer.io.jsonc import load_jsonc
from codesign_optimizer.models.hardware import HardwareProposal
from codesign_optimizer.models.workload import WorkloadSpec
from codesign_optimizer.optimizer.evolutionary import HeuristicSearchRunner
from codesign_optimizer.optimizer.orchestrator import CoDesignOrchestrator
from codesign_optimizer.optimizer.pipeline_client import MapperSimulatorPipelineClient
from codesign_optimizer.optimizer.search_space import SearchSpace, load_component_library
from codesign_optimizer.optimizer.tcro import TCROConfig, TCROSearchRunner
from codesign_optimizer.simulator.file_adapter import FileBackedSimulatorClient
from codesign_optimizer.utils.logging import configure_logging

app = typer.Typer(help="Two-stage hardware-software co-design optimizer CLI.", no_args_is_help=True)
console = Console()


@app.callback()
def main() -> None:
    """
    CLI root callback so `codesign-opt run ...` works as expected.
    """


@app.command("run")
def run_optimizer(
    hardware: Path = typer.Option(..., exists=True, readable=True, help="Input hardware JSON/JSONC."),
    workload: Path = typer.Option(..., exists=True, readable=True, help="Input workload JSON/JSONC."),
    feedback: Path = typer.Option(..., exists=True, readable=True, help="Simulator feedback JSON/JSONC."),
    iterations: int = typer.Option(8, min=1, max=1000, help="Max outer-loop iterations."),
    artifacts_dir: Path = typer.Option(Path("artifacts"), help="Output directory."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logs."),
) -> None:
    configure_logging(verbose=verbose)

    hardware_payload = load_jsonc(hardware)
    workload_payload = load_jsonc(workload)

    hardware_model = HardwareProposal.model_validate(hardware_payload)
    workload_model = WorkloadSpec.model_validate(workload_payload)
    settings = OptimizerSettings(max_outer_iterations=iterations, artifacts_dir=artifacts_dir)

    simulator = FileBackedSimulatorClient(
        proposal_out_path=artifacts_dir / "last_submitted_proposal.json",
        feedback_in_path=feedback,
    )
    orchestrator = CoDesignOrchestrator(settings=settings, simulator=simulator)
    history = orchestrator.run(initial_hardware=hardware_model, workload=workload_model)

    if not history:
        console.print("[red]No iterations were executed.[/red]")
        raise typer.Exit(code=1)

    best = min(history, key=lambda h: h.objective_score)
    console.print(
        "[green]Optimization completed[/green]\n"
        f"Iterations: {len(history)}\n"
        f"Best iteration: {best.iteration}\n"
        f"Best score: {best.objective_score:.4f}\n"
        f"Feasible: {best.constraints.feasible}\n"
        f"Artifacts: {artifacts_dir}"
    )


@app.command("search")
def search_optimizer(
    catalog: Path = typer.Option(..., exists=True, readable=True, help="Component catalog JSON/JSONC."),
    space: Path = typer.Option(..., exists=True, readable=True, help="Heuristic search-space JSON/JSONC."),
    workload: Path = typer.Option(..., exists=True, readable=True, help="Mapper workload JSON."),
    generations: int = typer.Option(4, min=1, max=1000, help="Number of generations."),
    population: int = typer.Option(8, min=1, max=10000, help="Population size."),
    concurrency: int = typer.Option(
        1,
        min=1,
        max=1024,
        help="Maximum number of candidate mapper/simulator pipeline runs to execute concurrently per generation.",
    ),
    out: Path = typer.Option(Path("artifacts/search_run"), help="Search output directory."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logs."),
) -> None:
    configure_logging(verbose=verbose)

    component_library = load_component_library(load_jsonc(catalog))
    search_space = SearchSpace.model_validate(load_jsonc(space))
    repo_root = search_space.evaluation.repo_root or _default_repo_root()
    pipeline = MapperSimulatorPipelineClient(repo_root=repo_root, evaluation=search_space.evaluation)
    runner = HeuristicSearchRunner(
        component_library=component_library,
        search_space=search_space,
        pipeline_client=pipeline,
        workload_path=workload,
        out_dir=out,
        population_size=population,
        generations=generations,
        concurrency=concurrency,
    )
    result = runner.run()
    console.print(
        "[green]Heuristic search completed[/green]\n"
        f"Generations: {generations}\n"
        f"Population: {population}\n"
        f"Concurrency: {concurrency}\n"
        f"Evaluations: {len(result.history)}\n"
        f"Pareto candidates: {len(result.pareto_frontier)}\n"
        f"Best score: {result.best.weighted_score:.4f}\n"
        f"Best feasible: {result.best.feasible}\n"
        f"Artifacts: {out}"
    )


@app.command("tcro")
def tcro_optimizer(
    catalog: Path = typer.Option(..., exists=True, readable=True, help="Component catalog JSON/JSONC."),
    space: Path = typer.Option(..., exists=True, readable=True, help="TCRO search-space JSON/JSONC."),
    workload: Path = typer.Option(..., exists=True, readable=True, help="Mapper workload JSON."),
    steps: int = typer.Option(8, min=1, max=10000, help="Number of TCRO continuous-update steps."),
    samples_per_step: int = typer.Option(4, min=1, max=10000, help="Discrete samples evaluated per TCRO step."),
    concurrency: int = typer.Option(1, min=1, max=1024, help="Maximum concurrent mapper/simulator runs per step."),
    learning_rate: float = typer.Option(0.35, min=0.000001, help="Pseudo-gradient learning rate."),
    initial_temperature: float = typer.Option(1.0, min=0.0, help="Initial Gumbel sampling temperature."),
    temperature_decay: float = typer.Option(0.92, min=0.000001, max=1.0, help="Temperature decay per step."),
    min_temperature: float = typer.Option(0.05, min=0.0, help="Lower bound for sampling temperature."),
    link_prune_threshold: float = typer.Option(0.25, min=0.0, help="Inter-rack alpha below this value is pruned."),
    rack_activation_threshold: float = typer.Option(0.5, min=0.0, help="Optional rack active_alpha threshold."),
    latent_rack_initial_alpha: float = typer.Option(0.2, min=0.0, help="Default initial alpha for inactive optional rack slots."),
    checkpoint_interval: int = typer.Option(1, min=1, help="Write supernet_state.json every N steps."),
    out: Path = typer.Option(Path("artifacts/tcro_run"), help="TCRO output directory."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logs."),
) -> None:
    configure_logging(verbose=verbose)

    component_library = load_component_library(load_jsonc(catalog))
    search_space = SearchSpace.model_validate(load_jsonc(space))
    repo_root = search_space.evaluation.repo_root or _default_repo_root()
    pipeline = MapperSimulatorPipelineClient(repo_root=repo_root, evaluation=search_space.evaluation)
    runner = TCROSearchRunner(
        component_library=component_library,
        search_space=search_space,
        pipeline_client=pipeline,
        workload_path=workload,
        out_dir=out,
        steps=steps,
        samples_per_step=samples_per_step,
        concurrency=concurrency,
        config=TCROConfig(
            learning_rate=learning_rate,
            initial_temperature=initial_temperature,
            temperature_decay=temperature_decay,
            min_temperature=min_temperature,
            link_prune_threshold=link_prune_threshold,
            rack_activation_threshold=rack_activation_threshold,
            latent_rack_initial_alpha=latent_rack_initial_alpha,
            checkpoint_interval=checkpoint_interval,
        ),
    )
    result = runner.run()
    console.print(
        "[green]TCRO search completed[/green]\n"
        f"Steps: {steps}\n"
        f"Samples per step: {samples_per_step}\n"
        f"Concurrency: {concurrency}\n"
        f"Evaluations: {len(result.history)}\n"
        f"Best score: {result.best.weighted_score:.4f}\n"
        f"Best feasible: {result.best.feasible}\n"
        f"Artifacts: {out}"
    )


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


if __name__ == "__main__":
    app()
