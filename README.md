# LLM in Queuing

A queueing-model and token-level simulator for **LLM serving** with **continuous
batching** (prefill + decode) under a **KV-cache capacity constraint**. The
repository studies the controllable trade-off between **throughput, latency,
tail-latency protection, and prediction-accuracy sensitivity** across several
scheduling policies, and ships reproducible experiments together with frozen
result artifacts and a regression check.

## Overview

- **Simulation engine**: event-driven, token-level continuous batching with
  explicit prefill/decode semantics and KV-cache accounting.
- **Scheduling policies**: `FCFS`, `PSF`, `AA` (PSF + aging), `QBA`
  (quota-based aging), and `RR` (round-robin over short/long classes).
- **Analytical layer**: an M/G/n surrogate model with trace-calibrated
  `φ(rho)` closure for PSFA/PSFAQ, alongside simulation curves.
- **Workloads**: a synthetic mixed normal–Pareto + lognormal workload, plus an
  Azure LLM inference trace pipeline (`data/process.py`) and a real-trace
  driver (`scripts/real_trace_analysis.py`).
- **Frozen results**: `freeze/` contains the exact `.npz` + `.json` artifacts
  behind figures 2–6 and the real-trace study; `reproduction.py` re-runs the
  engine and asserts numerical agreement against those artifacts.

## Repository layout

```
LLM_in_Queuing/
├── src/llm_in_queuing/        # core package (engine, schedulers, formula, ...)
│   ├── simulator.py           # event loop / run() entry
│   ├── engine.py              # prefill+decode semantics, KV accounting, batch time
│   ├── metrics.py             # per-run metric extraction
│   ├── workload.py            # synthetic workload generation + label noise
│   ├── state.py               # per-request / per-batch state transitions
│   ├── config.py              # YAML config loading
│   ├── types.py               # Request / EngineState / BatchEvent dataclasses
│   ├── admission.py           # admission policy
│   ├── predictor.py           # short/long prediction interface
│   ├── io.py, cli.py          # I/O helpers and CLI entry points
│   ├── schedulers/            # fcfs.py psf.py aging.py qba.py rr.py
│   ├── formula/               # phi_model.py: M/G/n surrogate + φ calibration
│   ├── plotting/              # fig2/fig3/fig4 drawing + style
│   └── analytical/            # calibration.py mg_n.py phi.py (lower-level helpers)
├── scripts/
│   ├── experiment/            # fig2–fig6 sweep drivers + shared _common.py
│   │   └── configs/           # per-figure base YAML configs
│   ├── draw/                  # draw_fig2.py … draw_fig6.py
│   ├── AE.ps1                 # one-click runner for fig2–fig6
│   ├── real_trace_analysis.py # Azure real-trace per-window driver
│   └── run_experiment.py      # single-simulation demo + diagnostic plots
├── data/
│   └── process.py             # download / sample / normalize / token-scale CLI
├── configs/                   # example top-level configs (conference/journal/…)
├── freeze/                    # frozen artifacts (fig2–fig6 + real-trace results)
├── tests/                     # pytest suite
├── reproduction.py            # reproducibility regression check (fig2/3/4)
└── pyproject.toml
```

## Installation

```bash
cd LLM_in_Queuing
pip install -e .
```

Core runtime dependencies are `numpy` and `pyyaml` (declared in
`pyproject.toml`). The real-trace pipeline additionally needs `pandas`
(`data/process.py`, `scripts/real_trace_analysis.py`), and
`scripts/run_experiment.py` needs `matplotlib`. Install them on demand:

```bash
pip install pandas matplotlib
```

## Quick start (single simulation)

```bash
python scripts/run_experiment.py
```

This runs one simulation with the default config
(`configs/conference_default.yaml`), prints the waiting-time mean / P95 / P99
for the configured policy, and writes a 3×3 diagnostic figure to
`scripts/output_exp/`.

## Frozen results & reproducibility

`freeze/` holds the reference artifacts used in the paper:

- `fig2_data.npz` / `fig2_meta.json` — lambda sweep (SIM + formula layers)
- `fig3_data.npz` / `fig3_meta.json` — `quota_k` sweep
- `fig4_data.npz` / `fig4_meta.json` — `gamma` sweep
- `fig5_data.npz` / `fig5_meta.json` — `epsilonSL` sweep
- `fig6_data.npz` / `fig6_meta.json` — `epsilonLS` sweep
- `{FCFS,PSF,AA,QBA}_real_trace_results.npz` / `.json` — Azure real-trace study

`reproduction.py` re-runs the refactored engine over the first `k` seeds of the
SIM-layer sweeps and asserts bit-level closeness (default tolerance `1e-8`)
against the corresponding `freeze/` segments:

```bash
python reproduction.py            # k=1
python reproduction.py --k 3      # first 3 seeds
python reproduction.py --tol 1e-6 # custom tolerance
```

It covers the seed-indexed `*_runs` arrays of `fig2`, `fig3`, and `fig4`
(formula-layer curves are deterministic and therefore not included). Warm-up /
steady-state trimming is explicitly disabled to match how `freeze/` was
generated.

## Figure experiments (fig2–fig6)

All sweeps run as modules and write `<fig>_data.npz` + `<fig>_meta.json` into
`scripts/plot/`. Common arguments are `--seed-start 42`, `--seed-count 100`
(seeds `42..141`), `--sim-admission-number`, `--plot-dir`, and `--out-prefix`.

Each sweep is followed by its drawing script:

```bash
python -m scripts.draw.draw_figN --data scripts/plot/figN_data.npz
```

### fig2 — arrival-rate (lambda) sweep

Simulation layer (FCFS/PSF/AA/QBA/RR) plus the analytical formula layer.

```bash
python -m scripts.experiment.sweep_lambda --seed-count 100
python -m scripts.draw.draw_fig2 --data scripts/plot/fig2_data.npz
```

Key arguments: `--sim-lambda-min 26 --sim-lambda-max 40 --sim-lambda-step 1`,
`--formula-lambda-min 27 --formula-lambda-max 32 --formula-lambda-step 0.25`,
`--phi-fit-lambdas 25,27,29,31,33`, `--reference-load`.

Output arrays: `sim_{mean,p95,p99}_{FCFS,PSF,PSFA,PSFAQ,RR}_runs` and the
deterministic `formula_*` curves.

### fig3 — `quota_k` sweep (QBA)

```bash
python -m scripts.experiment.sweep_quotak --seed-count 100 --reference-load 32.0
python -m scripts.draw.draw_fig3 --data scripts/plot/fig3_data.npz
```

Key arguments: `--reference-load 32.0`,
`--quota-ks "1,2,3,4,5,6,7,8,9,10,11,12,13,14"`.

Sweeps `QBA` over `quota_k`; runs `FCFS`, `PSF`, `AA`, `RR` as fixed
references. Note `quota_k` is clamped to `max(1, k)` in `state.py`.

### fig4 — aging threshold (`gamma`) sweep (AA / QBA)

```bash
python -m scripts.experiment.sweep_gamma --seed-count 100 --reference-load 32.0
python -m scripts.draw.draw_fig4 --data scripts/plot/fig4_data.npz
```

Key arguments: `--gamma-min 0.50 --gamma-max 1.00 --gamma-step 0.05`,
`--quota-k 5`.

Sweeps `AA` and `QBA` over `gamma`; runs `FCFS`, `PSF`, `RR` as references.

### fig5 — `epsilonSL` sweep (short predicted long)

```bash
python -m scripts.experiment.sweep_epsilonSL --seed-count 100
python -m scripts.draw.draw_fig5 --data scripts/plot/fig5_data.npz
```

Key arguments: `--reference-load 32.0`,
`--epsilon-sl-min 0.0 --epsilon-sl-max 0.40 --epsilon-sl-step 0.05`,
`--epsilon-ls-fixed 0.0`.

Sweeps `PSF`, `AA`, `QBA`, `RR` over `epsilonSL = P(pred=long | true=short)`;
`FCFS` is the noise-free reference. Per-class waiting times and realized error
rates are also recorded.

### fig6 — `epsilonLS` sweep (long predicted short)

```bash
python -m scripts.experiment.sweep_epsilonLS --seed-count 100
python -m scripts.draw.draw_fig6 --data scripts/plot/fig6_data.npz
```

Key arguments: `--reference-load 32.0`,
`--epsilon-ls-min 0.0 --epsilon-ls-max 0.40 --epsilon-ls-step 0.05`,
`--epsilon-sl-fixed 0.0`.

Sweeps `PSF`, `AA`, `QBA`, `RR` over `epsilonLS = P(pred=short | true=long)`;
`FCFS` is the reference.

## One-click runner (`scripts/AE.ps1`)

Runs fig2–fig6 (100 seeds each) and draws all figures in sequence. From the
repository root (PowerShell):

```powershell
.\scripts\AE.ps1
```

It sets `PYTHONPATH` to `src`, runs each `sweep_*` module with
`--seed-count 100`, then invokes the matching `draw_fig*` script. Output lands
in `scripts/plot/`.

## Real-trace analysis (`scripts/real_trace_analysis.py`)

Runs `FCFS`, `PSF`, `AA`, `QBA` over every `data/Azure_work_load_subset*.csv`
window (5000 requests per window) and saves per-window diagnostics:

```bash
python -m scripts.real_trace_analysis
```

Output (in `scripts/real_trace_output/`):

- `_pre_stats.npz` — window-level pre-simulation statistics
- `{Policy}_real_trace_results.npz` — per-window post-simulation metrics
- `{Policy}_real_trace_results.json` — config metadata

Requires the subset CSVs to exist (see the data pipeline below).

## Data pipeline (`data/process.py`)

Prepares the Azure LLM inference trace. Sub-commands:

```bash
python data/process.py download      # download the raw CSV (~1 GB) if not cached
python data/process.py sample        # extract 100 non-overlapping subsets × 5000 rows
python data/process.py normalize     # add per-subset TimeSeconds column (each t=0)
python data/process.py token-scale   # compute global token scale s_tok
```

- `download` fetches `AzureLLMInferenceTrace_conv_1week.csv` from the Azure
  public dataset release (use `--force` to re-download).
- `sample` streams the CSV in chunks and writes `Azure_work_load_subset001.csv`
  … `subset100.csv` with a fixed seed for reproducibility
  (`--n-subsets`, `--subset-size`, `--seed`).
- `normalize` inserts `TimeSeconds = (TIMESTAMP - subset_origin)`, zeroing each
  subset independently.
- `token-scale` computes the global token scale
  `s_tok = E[P+D]_synthetic / E[P+D]_Azure_pooled`, matching the synthetic
  workload to the Azure token magnitude (default `0.44` in
  `real_trace_analysis.py`).

## Package components (`src/llm_in_queuing/`)

- **`simulator.py`** — the event-driven `run(cfg)` loop: generates arrivals,
  dispatches batches, advances time, and returns the metrics dict.
- **`engine.py`** — continuous-batching semantics: prefill/decode token
  budgeting, KV-cache accounting, batch-time model.
- **`schedulers/`** — pluggable scheduling strategies:
  - `fcfs.py` — first-come-first-served.
  - `psf.py` — predict-shortest-first.
  - `aging.py` — `AA` (PSF + aging threshold, a.k.a. `PSFA`).
  - `qba.py` — `QBA` (aging with a promotion quota, a.k.a. `PSFAQ`).
  - `rr.py` — `RR` (round-robin over short/long classes).
- **`formula/phi_model.py`** — the M/G/n surrogate: parameter calibration from
  an FCFS trace, `φ(rho)` curve fitting for PSFA/PSFAQ, and analytical wait-time
  curves.
- **`metrics.py`** — extracts waiting / E2E / TTFT / throughput / OOM /
  batch-utilization metrics (internal keys are currently Chinese; the sweep and
  real-trace drivers normalize them into stable English keys).
- **`workload.py`** — synthetic workload generation (normal–Pareto prompts,
  lognormal answers) and prediction-noise injection (`TN/TP/FN/FP` →
  `epsilonSL`/`epsilonLS`).
- **`state.py`** — request/batch state transitions (queue insertion, aging
  threshold, start marking, `quota_k` clamping).
- **`config.py`** — YAML config loading with defaults.
- **`types.py`** — `Request`, `EngineState`, `BatchEvent` dataclasses.
- **`admission.py`**, **`predictor.py`** — admission and short/long prediction
  interfaces.
- **`plotting/`** — figure drawing helpers (legacy fig2/fig3/fig4 + style).
- **`analytical/`** — lower-level analytical helpers (`calibration.py`,
  `mg_n.py`, `phi.py`).

## Configuration

Per-figure base configs live in `scripts/experiment/configs/`
(`sweep_lambda_base.yaml`, `sweep_quotak_base.yaml`, `sweep_gamma_base.yaml`,
`sweep_epsilonSL_base.yaml`, `sweep_epsilonLS_base.yaml`). Example top-level
configs are in `configs/` (`conference_default.yaml`, `journal_base.yaml`,
`ipccc_base.yaml`, and per-policy `*_base.yaml`).

Representative keys:

```yaml
admission_number: 5000        # requests per run
lambda_arrival: 25            # arrival rate
budget_tokens: 128            # per-step token budget
request_block_size: 8
KV_total_capacity: 8192
KV_safety_threshold: 0.7
schedule_method: FCFS         # FCFS | PSF | AA | QBA | RR
gamma: 0.80                   # aging threshold
quota_k: 5                    # QBA promotion quota
ageing_measure: history
label_threshold_percentail: 0.85
TN/TP/FN/FP: 0.0              # raw prediction-noise counts
steady_state:
  enabled: false              # warm-up/steady-state trimming (disabled by default)
  trim_ratio: 0.1
```

When `steady_state.enabled` is `true`, the first and last `trim_ratio` of
requests are excluded from statistics; the freeze artifacts were generated with
it disabled.

## Scheduling policies

| Policy | Engine name | NPZ key | Description |
|---|---|---|---|
| First-come-first-served | `FCFS` | `FCFS` | Baseline, no prediction/aging. |
| Predict-shortest-first | `PSF` | `PSF` | Prioritizes requests predicted short. |
| PSF + aging | `AA` | `PSFA` | Aging threshold promotes long requests past `gamma`. |
| Quota-based aging | `QBA` | `PSFAQ` | `AA` with a `quota_k` promotion budget. |
| Round-robin | `RR` | `RR` | Alternates short/long classes. |

## Tests

```bash
pytest tests/ -v
```

The suite covers engine KV accounting, schedulers (including QBA/RR), the
workload generator, and conference baseline regression points.
