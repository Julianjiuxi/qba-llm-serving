"""fig4: gamma 扫参实验 — 观察 AA/QBA 在不同老化阈值下的表现。

用法:
    python scripts/experiment/sweep_gamma.py
    python scripts/experiment/sweep_gamma.py --reference-load 32.0 --gamma-min 0.50 --gamma-max 1.00 --gamma-step 0.05
    python scripts/experiment/sweep_gamma.py --seed-count 10 --plot-dir ../plot

输出:
    plot/fig4_data.npz + plot/fig4_meta.json
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


def _run_gamma_sweep(
    cfg: dict,
    lam: float,
    gammas: np.ndarray,
    quota_k: int,
    seeds: list[int],
) -> dict:
    """按 gamma 扫参 AA 和 QBA。

    AA 不需要 quota_k，QBA 需要。
    """
    n_seeds = len(seeds)
    n_points = len(gammas)

    out = {
        "PSFA": {
            "mean_runs": np.full((n_seeds, n_points), np.nan, dtype=np.float64),
            "p95_runs": np.full((n_seeds, n_points), np.nan, dtype=np.float64),
            "p99_runs": np.full((n_seeds, n_points), np.nan, dtype=np.float64),
        },
        "PSFAQ": {
            "mean_runs": np.full((n_seeds, n_points), np.nan, dtype=np.float64),
            "p95_runs": np.full((n_seeds, n_points), np.nan, dtype=np.float64),
            "p99_runs": np.full((n_seeds, n_points), np.nan, dtype=np.float64),
        },
    }

    total = n_points * 2 * n_seeds  # AA + QBA
    done = 0
    for si, seed in enumerate(seeds):
        for li, g in enumerate(gammas.tolist()):
            gf = float(g)
            # AA
            s_aa = run_sim_stats(cfg, {
                "lambda_arrival": float(lam),
                "schedule_method": "AA",
                "seed": int(seed),
                "gamma": gf,
            })
            out["PSFA"]["mean_runs"][si, li] = float(s_aa["mean"])
            out["PSFA"]["p95_runs"][si, li] = float(s_aa["p95"])
            out["PSFA"]["p99_runs"][si, li] = float(s_aa["p99"])
            done += 1
            progress_update(done, total, "SIM gamma sweep")

            # QBA
            s_qba = run_sim_stats(cfg, {
                "lambda_arrival": float(lam),
                "schedule_method": "QBA",
                "seed": int(seed),
                "gamma": gf,
                "quota_k": int(quota_k),
            })
            out["PSFAQ"]["mean_runs"][si, li] = float(s_qba["mean"])
            out["PSFAQ"]["p95_runs"][si, li] = float(s_qba["p95"])
            out["PSFAQ"]["p99_runs"][si, li] = float(s_qba["p99"])
            done += 1
            progress_update(done, total, "SIM gamma sweep")

    progress_done("SIM gamma sweep", total)
    return out


def main():
    parser = argparse.ArgumentParser(description="fig4: gamma 扫参")
    parser.add_argument("--reference-load", type=float, default=32.0)
    parser.add_argument("--gamma-min", type=float, default=0.50)
    parser.add_argument("--gamma-max", type=float, default=1.00)
    parser.add_argument("--gamma-step", type=float, default=0.05)
    parser.add_argument("--quota-k", type=int, default=5)
    parser.add_argument("--plot-dir", type=str, default=os.path.join(
        os.path.dirname(__file__), "..", "plot"))
    parser.add_argument("--out-prefix", type=str, default="fig4")
    parser.add_argument("--sim-admission-number", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--seed-count", type=int, default=100)
    args = parser.parse_args()

    # 加载基础配置
    cfg_path = Path(__file__).resolve().parent / "configs" / "sweep_gamma_base.yaml"
    cfg = load_config(cfg_path)
    if args.sim_admission_number is not None:
        cfg["admission_number"] = int(args.sim_admission_number)
    cfg["quota_k"] = int(args.quota_k)

    plot_dir = os.path.abspath(args.plot_dir)

    reference_lambda = float(args.reference_load)
    gammas = np.arange(
        float(args.gamma_min), float(args.gamma_max) + 1e-9,
        float(args.gamma_step), dtype=np.float64,
    )
    seeds = [int(args.seed_start) + i for i in range(int(args.seed_count))]

    # 参考策略: FCFS, PSF（不参与 gamma 扫参）
    print(f"stage: SIM references lambda={reference_lambda}")
    refs = run_references(cfg, reference_lambda, ["FCFS", "PSF", "RR"], seeds)

    # gamma 扫参: AA, QBA
    print(f"stage: SIM gamma sweep points={len(gammas)} lambda={reference_lambda} "
          f"quota_k={int(args.quota_k)} repeats={len(seeds)}")
    rows = _run_gamma_sweep(cfg, reference_lambda, gammas, int(args.quota_k), seeds)

    # 保存输出（npz key 名保持旧脚本兼容格式）
    arrays = {
        "reference_lambda": np.asarray([float(reference_lambda)], dtype=np.float64),
        "quota_k": np.asarray([int(args.quota_k)], dtype=np.int64),
        "gammas": np.asarray(gammas, dtype=np.float64),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "psfa_mean_runs": rows["PSFA"]["mean_runs"],
        "psfa_p95_runs": rows["PSFA"]["p95_runs"],
        "psfa_p99_runs": rows["PSFA"]["p99_runs"],
        "psfaq_mean_runs": rows["PSFAQ"]["mean_runs"],
        "psfaq_p95_runs": rows["PSFAQ"]["p95_runs"],
        "psfaq_p99_runs": rows["PSFAQ"]["p99_runs"],
        "ref_fcfs_mean_runs": refs["FCFS"]["mean_runs"],
        "ref_fcfs_p95_runs": refs["FCFS"]["p95_runs"],
        "ref_fcfs_p99_runs": refs["FCFS"]["p99_runs"],
        "ref_psf_mean_runs": refs["PSF"]["mean_runs"],
        "ref_psf_p95_runs": refs["PSF"]["p95_runs"],
        "ref_psf_p99_runs": refs["PSF"]["p99_runs"],
        "ref_rr_mean_runs": refs["RR"]["mean_runs"],
        "ref_rr_p95_runs": refs["RR"]["p95_runs"],
        "ref_rr_p99_runs": refs["RR"]["p99_runs"],
    }

    extra_meta = {
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
    print(f"next: python draw_fig4.py --data \"{npz_path}\"")


if __name__ == "__main__":
    main()
