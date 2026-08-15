"""fig5: epsilonSL 扫参实验 — 观察 PSF/AA/QBA/RR 在不同 epsilonSL（短被预测为长）下的表现。

epsilonSL = P(pred=long | true=short)，epsilonLS 固定为 0。

用法:
    python -m scripts.experiment.sweep_epsilonSL
    python -m scripts.experiment.sweep_epsilonSL --seed-count 10

输出:
    plot/fig5_data.npz + plot/fig5_meta.json
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


def _compute_noise_overrides(epsilon_sl: float, epsilon_ls: float) -> dict:
    """计算 raw TN/TP/FN/FP 使 workload.py 归一化后得到目标 epsilon。

    workload.py 归一化:
      epsilonSL = FP/(TN+FP), epsilonLS = FN/(TP+FN)
    给定 TN=1.0, TP=1.0:
      目标 epsilonSL = eps_sl → raw FP = eps_sl/(1-eps_sl)
      目标 epsilonLS = eps_ls → raw FN = eps_ls/(1-eps_ls)
    """
    fp = epsilon_sl / (1.0 - epsilon_sl) if epsilon_sl > 0 else 0.0
    fn = epsilon_ls / (1.0 - epsilon_ls) if epsilon_ls > 0 else 0.0
    return {"TN": 1.0, "TP": 1.0, "FN": fn, "FP": fp}


def _run_epsilon_sweep(
    cfg: dict,
    lam: float,
    epsilons_sl: np.ndarray,
    eps_ls_fixed: float,
    policies: list[str],
    seeds: list[int],
) -> dict:
    """按 epsilonSL 扫参多个策略。

    Returns:
        {policy: {"mean_runs": np.ndarray[n_seeds, n_points], ...,
                  "ts_mean_runs": ..., "tl_mean_runs": ...,
                  "actual_epsilon_SL_runs": ..., "actual_epsilon_LS_runs": ...}}
    """
    n_seeds = len(seeds)
    n_points = len(epsilons_sl)

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
    total = n_points * len(policies) * n_seeds
    done = 0
    print(f"stage: SIM epsilonSL sweep runs={total} seeds={n_seeds} "
          f"points={n_points} policies={len(policies)}")
    for si, seed in enumerate(seeds):
        for li, eps_sl in enumerate(epsilons_sl.tolist()):
            noise = _compute_noise_overrides(float(eps_sl), eps_ls_fixed)
            for p in policies:
                overrides = {
                    "lambda_arrival": float(lam),
                    "schedule_method": p,
                    "seed": int(seed),
                }
                overrides.update(noise)
                s = run_sim_stats(cfg, overrides)
                out[p]["mean_runs"][si, li] = float(s["mean"])
                out[p]["p95_runs"][si, li] = float(s["p95"])
                out[p]["p99_runs"][si, li] = float(s["p99"])
                out[p]["ts_mean_runs"][si, li] = float(s["ts_mean"])
                out[p]["ts_p99_runs"][si, li] = float(s["ts_p99"])
                out[p]["tl_mean_runs"][si, li] = float(s["tl_mean"])
                out[p]["tl_p99_runs"][si, li] = float(s["tl_p99"])
                out[p]["actual_epsilon_SL_runs"][si, li] = float(s["actual_epsilon_SL"])
                out[p]["actual_epsilon_LS_runs"][si, li] = float(s["actual_epsilon_LS"])
                done += 1
                progress_update(done, total, "SIM epsilonSL")
    progress_done("SIM epsilonSL", total)
    return out


def main():
    parser = argparse.ArgumentParser(description="fig5: epsilonSL 扫参")
    parser.add_argument("--reference-load", type=float, default=32.0)
    parser.add_argument("--epsilon-sl-min", type=float, default=0.0)
    parser.add_argument("--epsilon-sl-max", type=float, default=0.40)
    parser.add_argument("--epsilon-sl-step", type=float, default=0.05)
    parser.add_argument("--epsilon-ls-fixed", type=float, default=0.0)
    parser.add_argument("--plot-dir", type=str, default=os.path.join(
        os.path.dirname(__file__), "..", "plot"))
    parser.add_argument("--out-prefix", type=str, default="fig5")
    parser.add_argument("--sim-admission-number", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--seed-count", type=int, default=100)
    args = parser.parse_args()

    # 加载基础配置
    cfg_path = Path(__file__).resolve().parent / "configs" / "sweep_epsilonSL_base.yaml"
    cfg = load_config(cfg_path)
    if args.sim_admission_number is not None:
        cfg["admission_number"] = int(args.sim_admission_number)

    plot_dir = os.path.abspath(args.plot_dir)

    reference_lambda = float(args.reference_load)
    epsilons_sl = np.arange(
        float(args.epsilon_sl_min), float(args.epsilon_sl_max) + 1e-9,
        float(args.epsilon_sl_step), dtype=np.float64,
    )
    eps_ls_fixed = float(args.epsilon_ls_fixed)
    seeds = [int(args.seed_start) + i for i in range(int(args.seed_count))]

    # 受预测噪声影响的策略
    sweep_policies = ["PSF", "AA", "QBA", "RR"]
    # 唯一不受预测噪声影响的参考策略
    ref_policies = ["FCFS"]

    print(f"stage: SIM references lambda={reference_lambda}")
    refs = run_references(cfg, reference_lambda, ref_policies, seeds, extended=True)

    print(f"stage: SIM epsilonSL sweep points={len(epsilons_sl)} lambda={reference_lambda} "
          f"epsilonLS={eps_ls_fixed} repeats={len(seeds)}")
    rows = _run_epsilon_sweep(cfg, reference_lambda, epsilons_sl, eps_ls_fixed,
                              sweep_policies, seeds)

    # 保存输出
    arrays = {
        "reference_lambda": np.asarray([float(reference_lambda)], dtype=np.float64),
        "epsilons_sl": np.asarray(epsilons_sl, dtype=np.float64),
        "epsilon_ls_fixed": np.asarray([eps_ls_fixed], dtype=np.float64),
        "seeds": np.asarray(seeds, dtype=np.int64),
    }

    # 扫参曲线数据
    for p, prefix in [("PSF", "psf"), ("AA", "psfa"), ("QBA", "psfaq"), ("RR", "rr")]:
        for mk in ["mean", "p95", "p99", "ts_mean", "ts_p99", "tl_mean", "tl_p99"]:
            arrays[f"{prefix}_{mk}_runs"] = rows[p][f"{mk}_runs"]
        arrays[f"{prefix}_actual_epsilon_SL_runs"] = rows[p]["actual_epsilon_SL_runs"]
        arrays[f"{prefix}_actual_epsilon_LS_runs"] = rows[p]["actual_epsilon_LS_runs"]

    # FCFS 参考数据
    for mk in ["mean", "p95", "p99", "ts_mean", "ts_p99", "tl_mean", "tl_p99"]:
        arrays[f"ref_fcfs_{mk}_runs"] = refs["FCFS"][f"{mk}_runs"]
    arrays["ref_fcfs_actual_epsilon_SL_runs"] = refs["FCFS"]["actual_epsilon_SL_runs"]
    arrays["ref_fcfs_actual_epsilon_LS_runs"] = refs["FCFS"]["actual_epsilon_LS_runs"]

    extra_meta = {
        "epsilon_ls_fixed": eps_ls_fixed,
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
    print(f"next: python draw_fig5.py --data \"{npz_path}\"")


if __name__ == "__main__":
    main()
