# Two-Stage Hardware-Software Co-Design Optimizer

Production-oriented Python framework for iterative optimization of future SuperPOD architectures with:

- **Outer loop**: hardware topology and bill-of-material updates.
- **Inner loop**: software task-to-node mapping and placement strategy.
- **Simulator coupling**: JSONC-safe file interface for proposal/feedback exchange.
- **Constraint handling**: thermal, budget, and power-aware feasibility checks.

## Architecture

```text
.
├── src/codesign_optimizer
│   ├── cli.py
│   ├── config/settings.py
│   ├── io/jsonc.py
│   ├── models
│   │   ├── feedback.py
│   │   ├── hardware.py
│   │   └── workload.py
│   ├── optimizer
│   │   ├── constraints.py
│   │   ├── inner_loop.py
│   │   ├── objective.py
│   │   ├── orchestrator.py
│   │   └── outer_loop.py
│   ├── simulator
│   │   ├── file_adapter.py
│   │   └── interface.py
│   └── utils/logging.py
├── tests
│   ├── test_constraints.py
│   ├── test_jsonc.py
│   └── test_objective.py
└── examples
    └── workload_example.json
```

## Installation

```bash
PYTHON=/path/to/python3.11-or-newer
$PYTHON -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

If the local pip is old and reports that editable mode requires `setup.py` or
`setup.cfg`, this repository includes a small `setup.py` compatibility shim.
Make sure the command is run with Python 3.11 or newer; macOS `/usr/bin/python3`
is often Python 3.9 and is not supported by this package.

On this workstation, the bundled Codex runtime has a compatible Python:

```bash
/Users/tianyi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

Use the provided reference files in this workspace:

```bash
codesign-opt run \
  --hardware ./hw_config_proposal.json \
  --workload ./examples/workload_example.json \
  --feedback ./sim_feedback.json \
  --iterations 5
```

The optimizer writes updated hardware proposals and software mapping outputs under `./artifacts`.

## Heuristic Mapper/Simulator Search

The first production search path is a constraint-aware heuristic search. It
generates discrete rack/fabric candidates, exports each candidate to simulator
`hardware_topology.v2`, and invokes the existing mapper -> simulator wrapper:

```bash
codesign-opt search \
  --catalog ./examples/component_catalog.json \
  --space ./examples/search_space.json \
  --workload ../mapper/examples/cg_iteration_workload.json \
  --generations 4 \
  --population 8 \
  --out ./artifacts/search_run
```

The search writes per-candidate artifacts under `iter_*/candidate_*`:

- `proposal.json`
- `hardware_topology.json`
- `wrapper/` with mapper/simulator outputs
- `feedback.json`
- `score.json`

It also writes `summary.json`, `pareto_frontier.json`,
`best_hardware_topology.json`, and `best_proposal.json`.

The search does not modify mapper, ET, or simulator semantics. The wrapper
still projects only rank compute nodes to mapper; switch/router/memory-only
nodes remain simulator-only explicit graph nodes.

## Key Design Notes

- The sample simulator files are **JSONC** (comments included), so parser supports inline `// ...` comments.
- Hardware and simulator feedback are validated with `pydantic` models.
- The inner loop and outer loop are decoupled behind stable interfaces for easy replacement with learned policies or advanced solvers.
- The heuristic search path keeps topology search at rack/template level instead
  of directly enumerating a large adjacency matrix.
