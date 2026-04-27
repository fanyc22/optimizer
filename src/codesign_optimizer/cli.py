from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from codesign_optimizer.config.settings import OptimizerSettings
from codesign_optimizer.io.jsonc import load_jsonc
from codesign_optimizer.models.hardware import HardwareProposal
from codesign_optimizer.models.workload import WorkloadSpec
from codesign_optimizer.optimizer.orchestrator import CoDesignOrchestrator
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


if __name__ == "__main__":
    app()
