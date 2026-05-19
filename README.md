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
  --concurrency 4 \
  --out ./artifacts/search_run
```

`--concurrency` caps how many candidates in the same generation are evaluated
at once. Each concurrent evaluation gets its own `iter_*/candidate_*/wrapper`
directory and launches an independent mapper/simulator wrapper process.

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

## TCRO Continuous-Relaxed Search

TCRO keeps a continuous supernet inside the optimizer, samples discrete
`hardware_topology.v2` candidates for evaluation, then uses simulator telemetry
as pseudo-gradients to update node-type logits and link alpha values:

```bash
codesign-opt tcro \
  --catalog ./examples/component_catalog_tcro_latent_rack.json \
  --space ./examples/search_space_tcro_latent_rack.json \
  --workload ../mapper/examples/cg_iteration_workload.json \
  --steps 8 \
  --samples-per-step 4 \
  --concurrency 2 \
  --rack-activation-threshold 0.5 \
  --out ./artifacts/tcro_run
```

TCRO writes `step_*/sample_*` artifacts plus `supernet_state.json`,
`telemetry_history.json`, `tcro_summary.json`, `best_proposal.json`, and
`best_hardware_topology.json`. The simulator still only sees legal discrete
hardware graphs; the continuous relaxation is optimizer-internal. TCRO v1
initializes from the first template in the search space, so use a single
starting template when running focused continuous relaxation experiments.

TCRO search spaces may include latent rack slots by setting a rack to
`"optional": true`, `"active": false`, and a small `activation_alpha`. Inactive
optional racks are kept in the continuous supernet but are omitted from exported
`hardware_topology.v2` until their `active_alpha` crosses the activation
threshold. This lets TCRO grow extra compute, memory, or hybrid racks without
changing the mapper/simulator interface.

## TG-RL Masked Graph-Edit Search

TG-RL keeps the candidate hardware discrete. Each step enumerates legal
rack-level graph edits, masks out edits that fail existing repair/export
constraints, scores the remaining edits with simulator telemetry priors, and
evaluates one or more sampled candidates:

```bash
codesign-opt tgrl \
  --catalog ./examples/component_catalog_tcro_latent_rack.json \
  --space ./examples/search_space_tcro_latent_rack.json \
  --workload ../mapper/examples/cg_iteration_workload.json \
  --episodes 20 \
  --steps-per-episode 8 \
  --mode v0 \
  --concurrency 2 \
  --out ./artifacts/tgrl_run
```

`--mode v0` uses the telemetry heuristic prior directly. `--mode v1` adds a
small pure-Python linear policy over graph-edit features and updates it from
trajectory rewards with a KL-style pull toward the heuristic prior. TG-RL writes
`episode_*/step_*` artifacts plus `trajectory.jsonl`, `policy_state.json`,
`tgrl_summary.json`, `best_proposal.json`, and `best_hardware_topology.json`.

`--mode v2` enables the optional GNN-PPO trainer. Install the RL extra first:

```bash
pip install -e ".[dev,rl]"
codesign-opt tgrl \
  --catalog ./examples/component_catalog_tcro_latent_rack.json \
  --space ./examples/search_space_tcro_latent_rack.json \
  --workload ../mapper/examples/cg_iteration_workload.json \
  --episodes 4 \
  --steps-per-episode 4 \
  --mode v2 \
  --concurrency 2 \
  --ppo-epochs 2 \
  --device auto \
  --out ./artifacts/tgrl_v2_run
```

In v2, `episodes` are PPO updates, `steps-per-episode` is rollout length, and
`concurrency` is the number of parallel rollout environments. The trainer writes
`update_*/env_*/step_*` artifacts, `ppo_metrics.json`, and
`checkpoints/policy_latest.pt`.

## Key Design Notes

- The sample simulator files are **JSONC** (comments included), so parser supports inline `// ...` comments.
- Hardware and simulator feedback are validated with `pydantic` models.
- The inner loop and outer loop are decoupled behind stable interfaces for easy replacement with learned policies or advanced solvers.
- The heuristic search path keeps topology search at rack/template level instead
  of directly enumerating a large adjacency matrix.
