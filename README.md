# CHASE Optimizer

The CHASE optimizer is the outer hardware-search loop for **Cross-Layer
Heterogeneous Agile System-Exploration**. It proposes physically feasible
hardware candidates, exports each candidate in the v2 hardware-topology format,
evaluates the candidate through the CHASE mapper and simulator, then uses parsed
telemetry to choose the next architecture to explore.

## Overview

CHASE separates architecture exploration into two coupled loops:

```text
component catalog + search space
        |
        v
optimizer candidate proposal
        |
        v
v2 hardware topology
        |
        v
mapper -> event traces -> simulator
        |
        v
runtime, utilization, congestion, remote-memory telemetry
        |
        v
optimizer update
```

The optimizer is responsible for:

- Modeling candidate systems as racks, hosts, slots, switches, memory pools, and
  typed links.
- Enforcing cost, power, rack-unit, topology, and export-validity constraints
  before expensive simulation.
- Calling `tools/run_mapper_sim_pipeline.py` for fair candidate evaluation.
- Parsing simulator telemetry into comparable objectives.
- Exporting the best architecture and per-candidate artifacts for inspection.

The optimizer does not replace the mapper or simulator. Every evaluated
candidate still goes through the same mapper -> simulator path used by the rest
of CHASE.

## Quick Start

These commands assume the repository root has already been prepared following
the top-level `README.md`: submodules are initialized, `mapper/mapper_demo` is
built, and the simulator binary exists at
`simulator/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware`.

Install the optimizer from `optimizer/`:

```bash
cd optimizer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

Confirm the CLI:

```bash
codesign-opt --help
```

Run a small evolutionary search:

```bash
codesign-opt search \
  --catalog ./examples/component_catalog.json \
  --space ./examples/search_space.json \
  --workload ../mapper/examples/cg_iteration_workload.json \
  --generations 1 \
  --population 2 \
  --concurrency 1 \
  --out /tmp/chase_optimizer_search_smoke
```

Expected outputs include:

```text
/tmp/chase_optimizer_search_smoke/
  iter_*/candidate_*/
  summary.json
  pareto_frontier.json
  best_proposal.json
  best_hardware_topology.json
```

Run a tiny exhaustive search:

```bash
codesign-opt exhaustive \
  --catalog ./examples/component_catalog_tgrl.json \
  --space ./examples/search_space_tgrl_exhaustive_tiny.json \
  --workload ../mapper/examples/cg_iteration_workload.json \
  --concurrency 1 \
  --no-allow-empty-slots \
  --out /tmp/chase_optimizer_exhaustive_smoke
```

This enumerates four finite candidates and writes `exhaustive_summary.json`,
`candidate_scores.jsonl`, and the best exported topology.

## Public Examples

| File | Purpose |
| --- | --- |
| `examples/component_catalog.json` | Small smoke catalog for fast search tests. |
| `examples/search_space.json` | Small evolutionary-search space used by quick start. |
| `examples/component_catalog_tgrl.json` | Larger typed catalog for TCRO and TG-RL examples. |
| `examples/search_space_tgrl.json` | Rack/slot search space with latent racks for TCRO and TG-RL. |
| `examples/search_space_tgrl_exhaustive_tiny.json` | Finite four-candidate exhaustive smoke fixture. |
| `examples/component_catalog_host_templates.json` | Host-template catalog for host-granularity TG-RL tests. |
| `examples/search_space_host_template.json` | Host-template search-space example. |
| `examples/component_catalog_enterprise.json` | Enterprise H100/H200/L40S/L4-style component catalog. |
| `examples/search_space_enterprise.json` | Larger host-template enterprise topology example. |
| `examples/workload_suites/sparse_suite_example.json` | Small runnable workload-suite smoke fixture. |

Some larger workload-suite JSON files intentionally refer to experiment inputs
that are not part of the lightweight smoke path. Use `sparse_suite_example.json`
when you want a suite that is self-contained in this repository.

## Optimizer Modes

### `search`

`codesign-opt search` is the simplest practical mode. It performs a constrained
evolutionary search over candidate topology templates, evaluates each candidate
through the mapper/simulator wrapper, and ranks candidates by weighted runtime,
cost, power, network, and remote-memory objectives.

Use this first when validating a new installation or a new component catalog.

```bash
codesign-opt search \
  --catalog ./examples/component_catalog.json \
  --space ./examples/search_space.json \
  --workload ../mapper/examples/cg_iteration_workload.json \
  --generations 1 \
  --population 2 \
  --concurrency 1 \
  --out /tmp/chase_search_smoke
```

### `exhaustive`

`codesign-opt exhaustive` enumerates a finite search space and is useful for
near-optimality checks on deliberately tiny spaces. The search space must define
finite `exhaustive.slot_options`; host-granularity spaces are not exhaustive
search inputs.

```bash
codesign-opt exhaustive \
  --catalog ./examples/component_catalog_tgrl.json \
  --space ./examples/search_space_tgrl_exhaustive_tiny.json \
  --workload ../mapper/examples/cg_iteration_workload.json \
  --no-allow-empty-slots \
  --out /tmp/chase_exhaustive_smoke
```

The same mode is available through `scripts/exhaustive_search.py` for users who
prefer a standalone script.

### `tcro`

`codesign-opt tcro` keeps a continuous relaxation inside the optimizer and
samples legal discrete topologies for mapper/simulator evaluation. The simulator
only receives exported discrete hardware graphs; the continuous supernet is an
optimizer-internal state.

```bash
codesign-opt tcro \
  --catalog ./examples/component_catalog_tgrl.json \
  --space ./examples/search_space_tgrl.json \
  --workload ../mapper/examples/cg_iteration_workload.json \
  --steps 2 \
  --samples-per-step 1 \
  --concurrency 1 \
  --out /tmp/chase_tcro_smoke
```

Outputs include `supernet_state.json`, `telemetry_history.json`,
`tcro_summary.json`, and best-topology artifacts.

### `tgrl`

`codesign-opt tgrl` implements telemetry-guided graph-edit search. It enumerates
legal graph edits, masks infeasible actions, combines simulator telemetry with a
policy score, evaluates sampled candidates, and records the resulting
trajectory.

Available modes:

- `--mode v0`: telemetry heuristic prior only.
- `--mode v1`: lightweight learned linear policy.
- `--mode v2`: GNN-PPO policy over masked graph edits.

Smoke run without PyTorch:

```bash
codesign-opt tgrl \
  --catalog ./examples/component_catalog_tgrl.json \
  --space ./examples/search_space_tgrl.json \
  --workload ../mapper/examples/cg_iteration_workload.json \
  --episodes 1 \
  --steps-per-episode 1 \
  --mode v0 \
  --concurrency 1 \
  --out /tmp/chase_tgrl_v0_smoke
```

TG-RL v2 requires the optional RL dependency:

```bash
pip install -e ".[dev,rl]"
```

On CPU-only machines, install the CPU PyTorch wheel first to avoid pulling a
large CUDA-enabled wheel:

```bash
pip install "torch>=2.3" --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,rl]"
```

```bash
codesign-opt tgrl \
  --catalog ./examples/component_catalog_tgrl.json \
  --space ./examples/search_space_tgrl.json \
  --workload ../mapper/examples/cg_iteration_workload.json \
  --episodes 1 \
  --steps-per-episode 1 \
  --mode v2 \
  --concurrency 1 \
  --ppo-epochs 1 \
  --device auto \
  --out /tmp/chase_tgrl_v2_smoke
```

TG-RL v2 also supports workload suites:

```bash
codesign-opt tgrl \
  --catalog ./examples/component_catalog_tgrl.json \
  --space ./examples/search_space_tgrl.json \
  --workload-suite ./examples/workload_suites/sparse_suite_example.json \
  --episodes 1 \
  --steps-per-episode 1 \
  --mode v2 \
  --concurrency 1 \
  --ppo-epochs 1 \
  --device auto \
  --out /tmp/chase_tgrl_v2_suite_smoke
```

## Inputs

### Component Catalog

A component catalog defines available node and link types:

- `node_types`: GPUs, CPUs, switches, memory pools, host switches, and their
  cost, power, rack-unit, compute, memory, and radix properties.
- `link_types`: bandwidth, latency, protocol, hierarchy level, and per-link
  cost.

### Search Space

A search space defines how components may be assembled:

- `templates`: seed topology candidates.
- `host_templates`: optional host-level building blocks for host-granularity
  search.
- `rack_archetypes`: templates for dynamically added racks.
- `mutation`: graph-edit and link-quantity bounds.
- `limits`: global and per-rack cost, power, rack-unit, and rack-count limits.
- `evaluation`: mapper/simulator wrapper options.
- `exhaustive`: finite choices for exhaustive search, when applicable.

### Workload

Use `--workload` for a mapper workload DAG JSON. To optimize an LLM config, set
`evaluation.workload_kind` to `"llm"` in the search-space JSON and pass a config
such as `../mapper/examples/qwenconfig.json` through `--workload`.

For multi-workload TG-RL v2, use `--workload-suite` with a suite JSON.

## Outputs

Most modes create an output directory containing:

- Candidate directories with `proposal.json`, `hardware_topology.json`,
  `score.json`, and wrapper outputs.
- Mode summaries such as `summary.json`, `exhaustive_summary.json`,
  `tcro_summary.json`, or `tgrl_summary.json`.
- `best_proposal.json` and `best_hardware_topology.json`.
- Optional SVG topology visualizations.

The wrapper removes large intermediate mapper/simulator traces by default for
optimizer runs. Set `evaluation.save_wrapper_inputs` or
`evaluation.save_operator_stats` only when debugging a specific case.

## Choosing Run Budgets

The quick-start examples deliberately use one or two evaluations so that users
can validate an installation quickly. For an actual search, increase
generations, population, TCRO steps, or TG-RL episodes incrementally while
watching evaluation time, failure counts, convergence curves, and disk use.

Start with the self-contained sparse workload suite before introducing external
workloads. Preserve the catalog, search space, workload or suite, calibration
model, random seed, and output summary together when comparing runs.

## Tests

From `optimizer/`:

```bash
source .venv/bin/activate
python -m pytest tests -q
```

The optimizer tests exercise schema loading, repair and export logic,
evolutionary search, exhaustive search, TCRO, TG-RL, host templates, workload
suites, and telemetry parsing.

## More Documentation

- Root optimizer manual: `../docs/optimizer.md`
- End-to-end wrapper details: `../docs/pipeline.md`
- End-user workflows: `../docs/workflows.md`

## License

Copyright (c) 2026 Yuchen Fan, Minghong Sun, Jikui Ma, and Shunyu Mao.
Released under the [MIT License](LICENSE).
