"""fig7: label_threshold_percentail（p）扫参实验 — 观察长短划分阈值对调度的影响。

p 是 answer_len 的分位数阈值，决定真值短/长标签的划分（workload.py 中读取）。
- 参与扫参的策略：FCFS / PSF / AA / QBA。其中 FCFS 的调度不使用标签，但其
  per-class 指标（true_short / true_long）依赖由 p 决定的真值划分，因此也随 p
  变化，必须一并扫参（其整体指标会近似水平，可作为 sanity check）。
- RR 为内部策略、不进入会议、也不绘图，仅作为参考跑一次保留数据。

注意: 与 sweep_gamma 不同，此处 FCFS 不作单次水平参考线，而是随 p 变化的曲线。

用法:
    python -m scripts.experiment.sweep_threshold
    python -m scripts.experiment.sweep_threshold --reference-load 32.0 --p-min 0.55 --p-max 0.95 --p-step 0.05
    python -m scripts.experiment.sweep_threshold --seed-count 10 --plot-dir ../plot

输出:
    plot/fig7_data.npz + plot/fig7_meta.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from llm_in_queuing.config import load_config
from ._common import (
    progress_update,
    progress_done,
    run_sim_stats,
    run_references,
    save_results,
)


def _run_threshold_sweep(
    cfg: dict,
    lam: float,
    percentiles: np.ndarray,
    policies: list[str],
    seeds: list[int],
) -> tuple[dict, np.ndarray]:
    """按 label_threshold_percentail 扫参多个策略。

    Returns:
        (out, pi_s_runs)
        out: {policy: {"mean_runs": np.ndarray[n_seeds, n_points], ...}}
        pi_s_runs: np.ndarray[n_seeds, n_points]，真值短类的实际占比 π_S（仅诊断用）
    """
    n_seeds = len(seeds)
    n_points = len(percentiles)

    metrics_keys = [
        "mean_runs", "p95_runs", "p99_runs",
        "ts_mean_runs", "ts_p99_runs",
        "tl_mean_runs", "tl_p99_runs",
        "actual_epsilon_SL_runs", "actual_epsilon_LS_runs",
    ]
    out = {
        p: {k: np.full((n_seeds, n_points), np.nan, dtype=np.float64)
            for k in metrics_keys}
        for p in policies
    }
    pi_s_runs = np.full((n_seeds, n_points), np.nan, dtype=np.float64)
    total = n_points * len(policies) * n_seeds
    done = 0
    print(f"stage: SIM threshold sweep runs={total} seeds={n_seeds} "
          f"points={n_points} policies={len(policies)}")
    for si, seed in enumerate(seeds):
        for li, pct in enumerate(percentiles.tolist()):
            for p in policies:
                s = run_sim_stats(cfg, {
                    "lambda_arrival": float(lam),
                    "schedule_method": p,
                    "seed": int(seed),
                    "label_threshold_percentail": float(pct),
                })
                out[p]["mean_runs"][si, li] = float(s["mean"])
                out[p]["p95_runs"][si, li] = float(s["p95"])
                out[p]["p99_runs"][si, li] = float(s["p99"])
                out[p]["ts_mean_runs"][si, li] = float(s["ts_mean"])
                out[p]["ts_p99_runs"][si, li] = float(s["ts_p99"])
                out[p]["tl_mean_runs"][si, li] = float(s["tl_mean"])
                out[p]["tl_p99_runs"][si, li] = float(s["tl_p99"])
                out[p]["actual_epsilon_SL_runs"][si, li] = float(s["actual_epsilon_SL"])
                out[p]["actual_epsilon_LS_runs"][si, li] = float(s["actual_epsilon_LS"])
                n_ts = int(s["n_true_short"])
                n_tl = int(s["n_true_long"])
                if (n_ts + n_tl) > 0:
                    pi_s_runs[si, li] = n_ts / float(n_ts + n_tl)
                done += 1
                progress_update(done, total, "SIM threshold sweep")
    progress_done("SIM threshold sweep", total)
    return out, pi_s_runs


def main():
    parser = argparse.ArgumentParser(description="fig7: label_threshold_percentail 扫参")
    parser.add_argument("--reference-load", type=float, default=32.0)
    parser.add_argument("--p-min", type=float, default=0.55)
    parser.add_argument("--p-max", type=float, default=0.95)
    parser.add_argument("--p-step", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.80)
    parser.add_argument("--quota-k", type=int, default=5)
    parser.add_argument("--plot-dir", type=str, default=os.path.join(
        os.path.dirname(__file__), "..", "plot"))
    parser.add_argument("--out-prefix", type=str, default="fig7")
    parser.add_argument("--sim-admission-number", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--seed-count", type=int, default=100)
    args = parser.parse_args()

    # 加载基础配置
    cfg_path = Path(__file__).resolve().parent / "configs" / "sweep_threshold_base.yaml"
    cfg = load_config(cfg_path)
    if args.sim_admission_number is not None:
        cfg["admission_number"] = int(args.sim_admission_number)
    # 固定老化相关参数，仅让 p 变化（控制变量）
    cfg["gamma"] = float(args.gamma)
    cfg["quota_k"] = int(args.quota_k)

    plot_dir = os.path.abspath(args.plot_dir)

    reference_lambda = float(args.reference_load)
    percentiles = np.arange(
        float(args.p_min), float(args.p_max) + 1e-9,
        float(args.p_step), dtype=np.float64,
    )
    seeds = [int(args.seed_start) + i for i in range(int(args.seed_count))]

    # 参与扫参的策略：FCFS 的 per-class 指标随 p 变化，一并扫参
    sweep_policies = ["FCFS", "PSF", "AA", "QBA"]
    # RR 为内部策略，仅作参考跑一次（不绘图）
    ref_policies = ["RR"]

    print(f"stage: SIM references lambda={reference_lambda}")
    refs = run_references(cfg, reference_lambda, ref_policies, seeds, extended=True)

    print(f"stage: SIM threshold sweep points={len(percentiles)} lambda={reference_lambda} "
          f"gamma={cfg['gamma']} quota_k={cfg['quota_k']} repeats={len(seeds)}")
    rows, pi_s_runs = _run_threshold_sweep(
        cfg, reference_lambda, percentiles, sweep_policies, seeds)

    # 保存输出
    arrays = {
        "reference_lambda": np.asarray([float(reference_lambda)], dtype=np.float64),
        "percentiles": np.asarray(percentiles, dtype=np.float64),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "pi_s_runs": pi_s_runs,
    }

    # 扫参曲线数据（FCFS 也在其中：其 per-class 指标随 p 变化）
    for p, prefix in [("FCFS", "fcfs"), ("PSF", "psf"), ("AA", "psfa"), ("QBA", "psfaq")]:
        for mk in ["mean", "p95", "p99", "ts_mean", "ts_p99", "tl_mean", "tl_p99"]:
            arrays[f"{prefix}_{mk}_runs"] = rows[p][f"{mk}_runs"]
        arrays[f"{prefix}_actual_epsilon_SL_runs"] = rows[p]["actual_epsilon_SL_runs"]
        arrays[f"{prefix}_actual_epsilon_LS_runs"] = rows[p]["actual_epsilon_LS_runs"]

    # RR 参考数据（内部策略，不绘图）
    for mk in ["mean", "p95", "p99", "ts_mean", "ts_p99", "tl_mean", "tl_p99"]:
        arrays[f"ref_rr_{mk}_runs"] = refs["RR"][f"{mk}_runs"]
    arrays["ref_rr_actual_epsilon_SL_runs"] = refs["RR"]["actual_epsilon_SL_runs"]
    arrays["ref_rr_actual_epsilon_LS_runs"] = refs["RR"]["actual_epsilon_LS_runs"]

    extra_meta = {
        "sweep_policies": sweep_policies,
        "ref_policies": ref_policies,
        "refs": {
            p: {k: np.asarray(v, dtype=np.float64).tolist()
                for k, v in refs[p].items()}
            for p in refs
        },
    }

    npz_path, meta_path = save_results(
        plot_dir, str(args.out_prefix), arrays, cfg,
        vars(args), config_path=str(cfg_path), extra_meta=extra_meta,
    )
    print(f"saved data: {npz_path}")
    print(f"saved meta: {meta_path}")
    print(f"next: python draw_fig7.py --data \"{npz_path}\"")


if __name__ == "__main__":
    main()
