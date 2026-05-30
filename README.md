# Two-Stage Hardware-Software Co-Design Optimizer

Production-oriented Python framework for iterative optimization of future SuperPOD architectures with:

- **Outer loop**: hardware topology and bill-of-material updates.
- **Inner loop**: software task-to-node mapping and placement strategy.
- **Simulator coupling**: JSONC-safe file interface for proposal/feedback exchange.
- **Constraint handling**: thermal, budget, and power-aware feasibility checks.
- **Search policies**: constraint-aware evolutionary search, TCRO continuous relaxation, and TG-RL masked graph-edit RL.

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
│   │   ├── chromosome.py
│   │   ├── constraints.py
│   │   ├── evolutionary.py
│   │   ├── exporter.py
│   │   ├── feedback_parser.py
│   │   ├── inner_loop.py
│   │   ├── objective.py
│   │   ├── orchestrator.py
│   │   ├── outer_loop.py
│   │   ├── pipeline_client.py
│   │   ├── repair.py
│   │   ├── search_space.py
│   │   ├── tcro.py
│   │   ├── tgrl.py
│   │   ├── tgrl_v2
│   │   │   ├── model.py
│   │   │   ├── observation.py
│   │   │   ├── ppo.py
│   │   │   └── trainer.py
│   │   └── workload_suite.py
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

TCRO search spaces may include latent rack templates by setting a rack to
`"optional": true`, `"active": false`, and a small `activation_alpha`. Inactive
optional racks are kept in the continuous supernet but are omitted from exported
`hardware_topology.v2` until their `active_alpha` crosses the activation
threshold. This lets TCRO grow extra compute, memory, or hybrid racks without
changing the mapper/simulator interface.

## TG-RL Masked Graph-Edit Search

TG-RL keeps the candidate hardware discrete. Each step enumerates legal
slot/node, rack-topology, inter-rack, and rack add/remove graph edits, masks out
edits that fail existing repair/export constraints, scores the remaining edits
with simulator telemetry priors, and evaluates one or more sampled candidates:

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

The v2 policy is deliberately constrained. It does not generate arbitrary
graphs. For each rollout state, TG-RL first enumerates graph edits and masks out
anything that fails repair/export checks. The GNN policy then chooses among the
remaining legal actions:

- The observation graph contains a global node, rack nodes, slot nodes,
  rack-slot edges, inter-rack edges, budget/power margins, current and best
  score, rollout progress, simulator telemetry pressure, and optional workload
  suite speedup features.
- The actor scores each masked action from the graph embedding, the action
  target embedding, the action feature vector, and the telemetry heuristic
  logit.
- The action distribution is `actor_logits + heuristic_weight *
  heuristic_logits`, so PPO learns a correction over the telemetry prior rather
  than exploring from scratch.
- The reward is the normalized weighted-score improvement. Infeasible
  candidates and duplicate candidates are penalized; new global-best candidates
  receive a small bonus; rewards are clipped before PPO.
- PPO uses GAE, clipped policy loss, value loss, entropy regularization, and a
  KL penalty toward the telemetry prior.

v2 also supports multi-workload optimization:

```bash
codesign-opt tgrl \
  --catalog ./examples/component_catalog_tcro_latent_rack.json \
  --space ./examples/search_space_tcro_latent_rack.json \
  --workload-suite ./examples/workload_suites/small_mixed_suite.json \
  --episodes 4 \
  --steps-per-episode 4 \
  --mode v2 \
  --concurrency 2 \
  --ppo-epochs 2 \
  --device auto \
  --out ./artifacts/tgrl_v2_suite_run
```

For workload suites, the trainer evaluates or loads `baseline_suite.json`, then
optimizes the weighted geometric mean speedup by using
`suite_makespan_score = 1 / geomean_speedup` as the primary performance term.
It writes per-workload feedback and curve files under `curves/`, including JSON,
CSV, and SVG summaries. The final CLI output and `tgrl_summary.json` report a
per-workload `single_task_score` for the aggregate-best topology. That score is
recomputed with the same weighted-score formula used by single-workload TG-RL,
so it is comparable to a single-workload run's `Best score` for the same
workload. It is report-only and does not change the multi-workload optimization
objective.

Resume training with:

```bash
codesign-opt tgrl \
  --catalog ./examples/component_catalog_tcro_latent_rack.json \
  --space ./examples/search_space_tcro_latent_rack.json \
  --workload ../mapper/examples/cg_iteration_workload.json \
  --mode v2 \
  --resume ./artifacts/tgrl_v2_run/checkpoints/policy_latest.pt \
  --episodes 2 \
  --steps-per-episode 4 \
  --out ./artifacts/tgrl_v2_run
```

The checkpoint stores compatible model weights, optimizer state, RNG state,
seen candidate signatures, workload-suite baseline, global best score, and the
best chromosome from the previous update as the next rollout seed.

For apples-to-apples comparison with a fixed-topology exhaustive run, add
`--freeze-topology` to a TG-RL command. This masks graph edits that change rack
count, slot occupancy, or fabric topology, leaving only node-type substitutions
on already occupied slots. When the search space defines
`exhaustive.slot_options`, frozen-topology TG-RL also restricts replacement
targets to those node types and applies any link type/quantity specified by the
selected slot option.

## Exhaustive Finite Search

For very small spaces, use the exhaustive runner to evaluate every candidate
once and report the global optimum under the same weighted-score formula used by
TG-RL v2 single-workload runs:

```bash
codesign-opt exhaustive \
  --catalog ./examples/component_catalog_tcro_latent_rack.json \
  --space ./examples/search_space_4rack_exhaustive_tiny.json \
  --workload ../mapper/examples/cg_iteration_workload.json \
  --concurrency 4 \
  --out ./artifacts/exhaustive_tiny_run
```

The tiny example fixes four racks with at most two compute devices per rack.
Its `exhaustive.slot_options` contains two node choices, so the full space is
`2^(4 racks * 2 slots) = 256` candidates. The runner validates that the space is
finite, checks `exhaustive.max_candidates`, writes `candidate_*/score.json`, and
exports `best_proposal.json`, `best_hardware_topology.json`, and
`exhaustive_summary.json`.

The same entrypoint is also available as a standalone script:

```bash
python ./scripts/exhaustive_search.py \
  --catalog ./examples/component_catalog_tcro_latent_rack.json \
  --space ./examples/search_space_4rack_exhaustive_tiny.json \
  --workload ../mapper/examples/cg_iteration_workload.json
```

## Key Design Notes

- The sample simulator files are **JSONC** (comments included), so parser supports inline `// ...` comments.
- Hardware and simulator feedback are validated with `pydantic` models.
- The inner loop and outer loop are decoupled behind stable interfaces for easy replacement with learned policies or advanced solvers.
- The heuristic search path keeps topology search at rack/template level instead
  of directly enumerating a large adjacency matrix.
- TG-RL v2 remains a black-box optimizer over real mapper/simulator evaluations;
  simulator telemetry guides sampling, but the simulator itself is not treated as
  differentiable.
