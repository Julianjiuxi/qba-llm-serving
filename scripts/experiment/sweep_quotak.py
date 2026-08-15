"""fig3: quota_k 扫参实验 — 观察 QBA 在不同配额参数下的表现。

注意: state.py 中 quota_k = max(1, quota_k)，因此 k=0 实际执行为 k=1。
使用 --quota-ks 显式指定扫描列表，避免 ambiguous 数据点。

用法:
    python -m scripts.experiment.sweep_quotak.py
    python -m scripts.experiment.sweep_quotak.py --reference-load 32.0 --quota-ks "1,2,3,4,5,6,8,10,12,14"
    python -m scripts.experiment.sweep_quotak.py --seed-count 10 --plot-dir ../plot

输出:
    plot/fig3_data.npz + plot/fig3_meta.json
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


def _run_quotak_sweep(
    cfg: dict,
    lam: float,
    ks: np.ndarray,
    seeds: list[int],
) -> dict:
    """按 quota_k 扫参 QBA。

    Returns:
        {"mean_runs": np.ndarray[n_seeds, n_ks], ...}
    """
    n_seeds = len(seeds)
    n_points = len(ks)
    out = {
        "mean_runs": np.full((n_seeds, n_points), np.nan, dtype=np.float64),
        "p95_runs": np.full((n_seeds, n_points), np.nan, dtype=np.float64),
        "p99_runs": np.full((n_seeds, n_points), np.nan, dtype=np.float64),
    }
    total = n_points * n_seeds
    done = 0
    for si, seed in enumerate(seeds):
        for li, k in enumerate(ks.tolist()):
            s = run_sim_stats(cfg, {
                "lambda_arrival": float(lam),
                "schedule_method": "QBA",
                "seed": int(seed),
                "quota_k": int(k),
            })
            out["mean_runs"][si, li] = float(s["mean"])
            out["p95_runs"][si, li] = float(s["p95"])
            out["p99_runs"][si, li] = float(s["p99"])
            done += 1
            progress_update(done, total, "SIM PSFAQ quota_k")
    progress_done("SIM PSFAQ quota_k", total)
    return out


def main():
    parser = argparse.ArgumentParser(description="fig3: quota_k 扫参")
    parser.add_argument("--reference-load", type=float, default=32.0)
    parser.add_argument("--quota-ks", type=str, default="1,2,3,4,5,6,7,8,9,10,11,12,13,14",
                        help="逗号分隔的 quota_k 列表，例如 \"1,2,3,4,5,6,7,8,9,10,11,12,13,14\"")
    parser.add_argument("--plot-dir", type=str, default=os.path.join(
        os.path.dirname(__file__), "..", "plot"))
    parser.add_argument("--out-prefix", type=str, default="fig3")
    parser.add_argument("--sim-admission-number", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--seed-count", type=int, default=100)
    args = parser.parse_args()

    # 加载基础配置
    cfg_path = Path(__file__).resolve().parent / "configs" / "sweep_quotak_base.yaml"
    cfg = load_config(cfg_path)
    if args.sim_admission_number is not None:
        cfg["admission_number"] = int(args.sim_admission_number)

    plot_dir = os.path.abspath(args.plot_dir)

    reference_lambda = float(args.reference_load)
    quota_ks_nominal = [int(x.strip()) for x in str(args.quota_ks).split(",") if x.strip()]
    quota_ks = np.asarray(quota_ks_nominal, dtype=np.int64)
    # state.py 中 quota_k = max(1, quota_k)，记录实际生效值
    quota_ks_effective = np.asarray([max(1, k) for k in quota_ks_nominal], dtype=np.int64)
    seeds = [int(args.seed_start) + i for i in range(int(args.seed_count))]

    # 参考策略: FCFS, PSF, AA（不参与 quota_k 扫参）
    print(f"stage: SIM references lambda={reference_lambda}")
    refs = run_references(cfg, reference_lambda, ["FCFS", "PSF", "AA", "RR"], seeds)

    # quota_k 扫参: QBA
    print(f"stage: SIM PSFAQ sweep points={len(quota_ks)} lambda={reference_lambda} "
          f"repeats={len(seeds)}")
    rows = _run_quotak_sweep(cfg, reference_lambda, quota_ks, seeds)

    # 保存输出（npz key 名保持旧脚本兼容格式）
    arrays = {
        "reference_lambda": np.asarray([float(reference_lambda)], dtype=np.float64),
        "quota_ks": np.asarray(quota_ks, dtype=np.int64),
        "quota_ks_effective": np.asarray(quota_ks_effective, dtype=np.int64),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "psfaq_mean_runs": rows["mean_runs"],
        "psfaq_p95_runs": rows["p95_runs"],
        "psfaq_p99_runs": rows["p99_runs"],
        "ref_fcfs_mean_runs": refs["FCFS"]["mean_runs"],
        "ref_fcfs_p95_runs": refs["FCFS"]["p95_runs"],
        "ref_fcfs_p99_runs": refs["FCFS"]["p99_runs"],
        "ref_psf_mean_runs": refs["PSF"]["mean_runs"],
        "ref_psf_p95_runs": refs["PSF"]["p95_runs"],
        "ref_psf_p99_runs": refs["PSF"]["p99_runs"],
        "ref_psfa_mean_runs": refs["AA"]["mean_runs"],
        "ref_psfa_p95_runs": refs["AA"]["p95_runs"],
        "ref_psfa_p99_runs": refs["AA"]["p99_runs"],
        "ref_rr_mean_runs": refs["RR"]["mean_runs"],
        "ref_rr_p95_runs": refs["RR"]["p95_runs"],
        "ref_rr_p99_runs": refs["RR"]["p99_runs"],
    }

    extra_meta = {
        "quota_ks_nominal": [int(k) for k in quota_ks_nominal],
        "quota_ks_effective": [int(k) for k in quota_ks_effective],
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
    print(f"next: python draw_fig3.py --data \"{npz_path}\"")


if __name__ == "__main__":
    main()
