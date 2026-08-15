"""Real trace analysis: run scheduling policies on Azure LLM trace subsets.

Loops over all ``Azure_work_load_subset*.csv`` files, runs each scheduling
policy per window, and saves per-window diagnostic metrics as ``.npz`` files.
Config metadata is preserved as per-policy JSON files.

Usage:
    python -m scripts.real_trace_analysis

Output:
    scripts/real_trace_output/<Policy>_real_trace_results.npz   ← per-window arrays
    scripts/real_trace_output/<Policy>_real_trace_results.json  ← config metadata
    scripts/real_trace_output/_pre_stats.npz                    ← window-level pre-sim stats
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from llm_in_queuing.simulator import run

# ── constants ──────────────────────────────────────────────────────────────

POLICIES = ["FCFS", "PSF", "AA", "QBA"]
_SUBSET_GLOB = "data/Azure_work_load_subset[0-9][0-9][0-9].csv"

_CFG: dict = {
    "admission_number": 5000,
    "budget_tokens": 128,
    "request_block_size": 8,
    "KV_total_capacity": 8192,
    "KV_safety_threshold": 0.7,
    "gamma": 0.80,
    "quota_k": 5,
    "ageing_measure": "history",
    "label_threshold_percentail": 0.85,
    "TN": 0.0,
    "TP": 0.0,
    "FN": 0.0,
    "FP": 0.0,
    "seed": 42,
    "prompt_len_cap": 4096,
    "answer_len_cap": 4096,
    "lambda_arrival": 32.0,
    "token_scale": 0.44,
    "steady_state": {"enabled": False, "trim_ratio": 0.1},
}

_SNAPSHOT_KEYS = [
    "admission_number", "seed", "budget_tokens", "request_block_size",
    "KV_total_capacity", "KV_safety_threshold", "gamma", "quota_k",
    "ageing_measure", "label_threshold_percentail",
    "lambda_arrival", "token_scale", "steady_state",
]

# ── pre-simulation stats (per window, shared across policies) ──────────────

_PRE_STAT_NAMES = [
    "n_requests",
    "raw_lambda",
    "time_scale",
    "trace_span_s",
    "scaled_E_P",
    "scaled_E_D",
    "scaled_E_PD",
    "p85_answer_threshold",
    "true_long_fraction",
    "inter_arrival_Ca2",
]

# ── post-simulation per-policy metrics ─────────────────────────────────────

_POST_METRIC_NAMES = [
    "waiting_mean",
    "waiting_p95",
    "waiting_p99",
    "max_waiting",
    "ts_mean",
    "ts_p99",
    "tl_mean",
    "tl_p99",
    "n_true_short",
    "n_true_long",
    "actual_epsilon_SL",
    "actual_epsilon_LS",
    "e2e_mean",
    "e2e_p99",
    "ttft_mean",
    "ttft_p99",
    "throughput",
    "oom_prob",
    "avg_batch_utilization",
    "qba_promoted_total",
    "elapsed_seconds",
]


# ── pre-sim helpers ────────────────────────────────────────────────────────

def _compute_pre_stats(csv_path: Path, cfg: dict) -> dict[str, float]:
    """Compute per-window diagnostic stats from the raw CSV, before simulation."""
    df = pd.read_csv(csv_path)
    n = len(df)
    col_p = "ContextTokens"
    col_a = "GeneratedTokens"

    # ── time / lambda / scale ───────────────────────────────────────────
    ts = df["TimeSeconds"].values.astype(np.float64)
    trace_span = float(ts[-1] - ts[0])  # seconds (already zeroed by normalize_time)
    raw_lambda = float(n) / trace_span if trace_span > 0.0 else float("inf")
    target_lambda = float(cfg.get("lambda_arrival", 0.0))
    time_scale = (raw_lambda / target_lambda) if (target_lambda > 0 and raw_lambda > 0) else 1.0

    # ── scaled token stats ──────────────────────────────────────────────
    tok_s = float(cfg.get("token_scale", 1.0))
    p_raw = df[col_p].values.astype(np.float64)
    a_raw = df[col_a].values.astype(np.float64)
    p_scaled = np.clip(p_raw * tok_s, 5, int(cfg.get("prompt_len_cap", 4096)))
    a_scaled = np.clip(a_raw * tok_s, 1, int(cfg.get("answer_len_cap", 4096)))
    pd_scaled = p_scaled + a_scaled

    # ── label threshold (same rule as _assign_labels) ───────────────────
    pct = float(cfg.get("label_threshold_percentail", 0.85))
    ans_threshold = float(np.quantile(a_scaled, pct))
    long_frac = float(np.mean(a_scaled > ans_threshold))

    # ── inter-arrival C_a^2 ─────────────────────────────────────────────
    inter_arr = np.diff(ts)
    inter_arr = inter_arr[inter_arr > 0]  # guard against zero / negative deltas
    if len(inter_arr) >= 2:
        mu_ia = float(np.mean(inter_arr))
        var_ia = float(np.var(inter_arr, ddof=1))
        ca2 = var_ia / (mu_ia ** 2) if mu_ia > 0 else float("nan")
    else:
        ca2 = float("nan")

    return {
        "n_requests": float(n),
        "raw_lambda": raw_lambda,
        "time_scale": time_scale,
        "trace_span_s": trace_span,
        "scaled_E_P": float(np.mean(p_scaled)),
        "scaled_E_D": float(np.mean(a_scaled)),
        "scaled_E_PD": float(np.mean(pd_scaled)),
        "p85_answer_threshold": ans_threshold,
        "true_long_fraction": long_frac,
        "inter_arrival_Ca2": ca2,
    }


# ── post-sim helpers ───────────────────────────────────────────────────────

def _extract_post_stats(metrics: dict, elapsed: float) -> dict[str, float]:
    """Extract per-window post-simulation metrics from the raw metrics dict."""
    return {
        "waiting_mean": float(metrics.get("waiting_time平均", float("nan"))),
        "waiting_p95": float(metrics.get("waiting_time p95", float("nan"))),
        "waiting_p99": float(metrics.get("waiting_time p99", float("nan"))),
        "max_waiting": float(metrics.get("max waiting_time", float("nan"))),
        "ts_mean": float(metrics.get("waiting_time_true_short_mean", float("nan"))),
        "ts_p99": float(metrics.get("waiting_time_true_short_p99", float("nan"))),
        "tl_mean": float(metrics.get("waiting_time_true_long_mean", float("nan"))),
        "tl_p99": float(metrics.get("waiting_time_true_long_p99", float("nan"))),
        "n_true_short": float(metrics.get("n_true_short", 0)),
        "n_true_long": float(metrics.get("n_true_long", 0)),
        "actual_epsilon_SL": float(metrics.get("actual_epsilon_SL", 0.0)),
        "actual_epsilon_LS": float(metrics.get("actual_epsilon_LS", 0.0)),
        "e2e_mean": float(metrics.get("E2E Latency平均", float("nan"))),
        "e2e_p99": float(metrics.get("E2E Latency p99", float("nan"))),
        "ttft_mean": float(metrics.get("TTFT平均", float("nan"))),
        "ttft_p99": float(metrics.get("TTFT p99", float("nan"))),
        "throughput": float(metrics.get("平均吞吐量", float("nan"))),
        "oom_prob": float(metrics.get("OOM概率", float("nan"))),
        "avg_batch_utilization": float(metrics.get("平均batch使用率", float("nan"))),
        "qba_promoted_total": float(metrics.get("PSFAQ_promoted_total", 0.0)),
        "elapsed_seconds": float(elapsed),
    }


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"

    # ── discover subsets ─────────────────────────────────────────────────
    subsets = sorted(data_dir.glob("Azure_work_load_subset[0-9][0-9][0-9].csv"))
    if not subsets:
        raise FileNotFoundError(
            f"No subset CSVs found in {data_dir}.  "
            f"Run 'python data/process.py sample' first."
        )
    n_windows = len(subsets)

    out_dir = Path(__file__).resolve().parent / "real_trace_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Subsets:    {n_windows}")
    print(f"Policies:   {', '.join(POLICIES)}")
    print(f"Output dir: {out_dir}\n")

    # ── allocate result arrays ───────────────────────────────────────────
    pre_arrays: dict[str, np.ndarray] = {
        name: np.full(n_windows, np.nan, dtype=np.float64)
        for name in _PRE_STAT_NAMES
    }
    post_arrays: dict[str, dict[str, np.ndarray]] = {
        p: {name: np.full(n_windows, np.nan, dtype=np.float64)
            for name in _POST_METRIC_NAMES}
        for p in POLICIES
    }

    # ── loop over windows ────────────────────────────────────────────────
    t_start_total = time.perf_counter()

    for wi, csv_path in enumerate(subsets):
        window_label = csv_path.stem.replace("Azure_work_load_", "")
        print(f"\n{'=' * 55}")
        print(f"[{wi + 1:3d}/{n_windows}]  {window_label}")
        print(f"{'=' * 55}")

        # ── pre-sim stats (once per window) ──────────────────────────
        pre_stats = _compute_pre_stats(csv_path, _CFG)
        for name in _PRE_STAT_NAMES:
            pre_arrays[name][wi] = pre_stats[name]

        print(f"  λ_raw={pre_stats['raw_lambda']:.2f}  "
              f"time_scale={pre_stats['time_scale']:.4f}  "
              f"span={pre_stats['trace_span_s']:.1f}s")
        print(f"  scaled E[P]={pre_stats['scaled_E_P']:.0f}  "
              f"E[D]={pre_stats['scaled_E_D']:.0f}  "
              f"E[P+D]={pre_stats['scaled_E_PD']:.0f}  "
              f"long_frac={pre_stats['true_long_fraction']:.3f}  "
              f"C_a²={pre_stats['inter_arrival_Ca2']:.2f}")

        # ── run each policy ──────────────────────────────────────────
        for policy in POLICIES:
            t0 = time.perf_counter()
            local_cfg = dict(_CFG)
            local_cfg["trace_csv"] = str(csv_path)
            local_cfg["schedule_method"] = policy

            metrics = run(local_cfg)
            elapsed = time.perf_counter() - t0

            pstats = _extract_post_stats(metrics, elapsed)
            for name in _POST_METRIC_NAMES:
                post_arrays[policy][name][wi] = pstats[name]

            promo_str = (
                f"promo={pstats['qba_promoted_total']:.0f}"
                if policy == "QBA" else ""
            )
            print(
                f"  [{policy:>4s}]  "
                f"w_mean={pstats['waiting_mean']:.4f}  "
                f"w_p99={pstats['waiting_p99']:.4f}  "
                f"ts_mean={pstats['ts_mean']:.4f}  "
                f"tl_p99={pstats['tl_p99']:.4f}  "
                f"thr={pstats['throughput']:.0f}  "
                f"OOM={pstats['oom_prob']:.4f}  "
                f"{promo_str}  "
                f"({elapsed:.1f}s)"
            )

    # ── save npz files ───────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print("Saving npz files …")

    # Shared pre-sim stats
    pre_path = out_dir / "_pre_stats.npz"
    np.savez_compressed(pre_path, **pre_arrays, subset_names=np.array(
        [p.stem.replace("Azure_work_load_", "") for p in subsets]
    ))
    print(f"  {pre_path}")

    # Per-policy post-sim stats
    for policy in POLICIES:
        npz_path = out_dir / f"{policy}_real_trace_results.npz"
        np.savez_compressed(npz_path, **post_arrays[policy])
        print(f"  {npz_path}")

    # ── save JSON config metadata ────────────────────────────────────────
    for policy in POLICIES:
        payload = {
            "config_snapshot": {k: _CFG.get(k) for k in _SNAPSHOT_KEYS},
            "n_windows": n_windows,
            "n_total_requests": n_windows * int(_CFG["admission_number"]),
            "pre_stat_names": _PRE_STAT_NAMES,
            "post_metric_names": _POST_METRIC_NAMES,
        }
        json_path = out_dir / f"{policy}_real_trace_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"  {json_path}")

    total_elapsed = time.perf_counter() - t_start_total
    print(f"\nTotal: {n_windows} windows × {len(POLICIES)} policies "
          f"= {n_windows * len(POLICIES)} runs in {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
