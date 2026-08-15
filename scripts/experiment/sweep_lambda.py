"""fig2: lambda 扫参实验 — SIM 层 + 公式层。

对给定 lambda 范围：
- SIM 层：FCFS / PSF / AA / QBA 四种策略的多 seed 仿真
- 公式层：从 FCFS trace 校准参数，拟合 phi 曲线，计算分析曲线

用法:
    python -m scripts.experiment.sweep_lambda
    python -m scripts.experiment.sweep_lambda --seed-count 10

输出:
    scripts/plot/fig2_data.npz + scripts/plot/fig2_meta.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from llm_in_queuing.config import load_config
from llm_in_queuing.formula import (
    estimate_model_params_from_fcfs_trace,
    mixture_moments,
    psfa_phi_from_params,
    psfaq_phi_from_params,
    promoted_class_moments,
    psf_wait_approx,
    allen_cunneen_mean_wq,
    compute_curves,
    empirical_phi_from_policy_state,
    maybe_attach_empirical_phi,
    maybe_attach_empirical_phi_psfaq,
    fit_phi_curve_from_policy_traces,
)

from ._common import (
    progress_update,
    progress_done,
    run_sim_stats,
    save_results,
)


# 策略映射：新引擎名 → npz 输出 key 前缀（保持旧兼容格式）
_POLICY_MAP = {
    "FCFS": "FCFS",
    "PSF": "PSF",
    "AA": "PSFA",
    "QBA": "PSFAQ",
    "RR": "RR",
}


def _run_sim_sweep(
    cfg: dict,
    lambdas: np.ndarray,
    policies: list[str],
    seeds: list[int],
) -> dict:
    """SIM 层：多 seed × 多 lambda × 多策略扫参。"""
    n_seeds = len(seeds)
    n_points = len(lambdas)
    out = {
        p: {
            "mean_runs": np.full((n_seeds, n_points), np.nan, dtype=np.float64),
            "p95_runs": np.full((n_seeds, n_points), np.nan, dtype=np.float64),
            "p99_runs": np.full((n_seeds, n_points), np.nan, dtype=np.float64),
        }
        for p in policies
    }
    total = n_points * len(policies) * n_seeds
    done = 0
    print(f"stage: SIM runs={total} seeds={n_seeds} points={n_points} policies={len(policies)}")
    for si, seed in enumerate(seeds):
        cfg_seed = dict(cfg)
        cfg_seed["seed"] = int(seed)
        for li, lam in enumerate(lambdas.tolist()):
            for p in policies:
                s = run_sim_stats(cfg_seed, {
                    "lambda_arrival": float(lam),
                    "schedule_method": p,
                })
                out[p]["mean_runs"][si, li] = float(s["mean"])
                out[p]["p95_runs"][si, li] = float(s["p95"])
                out[p]["p99_runs"][si, li] = float(s["p99"])
                done += 1
                progress_update(done, total, "SIM")
    progress_done("SIM", total)
    return out


def _run_formula_sweep(
    cfg: dict,
    lambdas: np.ndarray,
    *,
    reference_load: float | None,
    phi_fit_lambdas: list[float],
) -> dict:
    """公式层：校准参数 → 拟合 phi → 计算曲线。"""
    cfg_formula = dict(cfg)
    print("stage: FORMULA calibrate_fcfs_trace")
    params, aux = estimate_model_params_from_fcfs_trace(
        cfg_formula, reference_load=reference_load,
    )
    print("stage: FORMULA phi_fit (AA/QBA)")

    p = float(params.p_short)
    es, _ = mixture_moments(p, params.m1_short, params.m1_long,
                            params.m2_short, params.m2_long)

    # 用原函数格式——手动执行 phi 拟合循环以输出进度
    rho_psfa, phi_psfa, debug_psfa = fit_phi_curve_from_policy_traces(
        cfg_formula, params, phi_fit_lambdas, "AA",
    )
    rho_psfaq, phi_psfaq, debug_psfaq = fit_phi_curve_from_policy_traces(
        cfg_formula, params, phi_fit_lambdas, "QBA",
    )

    params = type(params)(
        n=params.n, p_short=params.p_short,
        m1_short=params.m1_short, m1_long=params.m1_long,
        m2_short=params.m2_short, m2_long=params.m2_long,
        quota_k=int(cfg_formula.get("quota_k", 1)),
        reference_load=params.reference_load,
        reference_mean_wait=params.reference_mean_wait,
        reference_mean_service=params.reference_mean_service,
        phi_knots_rho=rho_psfa, phi_knots_phi=phi_psfa,
        phi_knots_rho_psfaq=rho_psfaq, phi_knots_phi_psfaq=phi_psfaq,
    )
    phi_aux = {
        "psfa": {"phi_knots_n": int(len(rho_psfa)), "phi_debug": debug_psfa},
        "psfaq": {"phi_knots_n": int(len(rho_psfaq)), "phi_debug": debug_psfaq},
    }

    print("stage: FORMULA curves")
    curves = compute_curves(lambdas, params)

    return {"params": params, "aux": aux, "phi_aux": phi_aux, "curves": curves}


def main():
    parser = argparse.ArgumentParser(description="fig2: lambda 扫参 (SIM + FORMULA)")
    parser.add_argument("--sim-lambda-min", type=float, default=26.0)
    parser.add_argument("--sim-lambda-max", type=float, default=40.0)
    parser.add_argument("--sim-lambda-step", type=float, default=1.0)
    parser.add_argument("--formula-lambda-min", type=float, default=27.0)
    parser.add_argument("--formula-lambda-max", type=float, default=32.0)
    parser.add_argument("--formula-lambda-step", type=float, default=0.25)
    parser.add_argument("--reference-load", type=float, default=None)
    parser.add_argument("--phi-fit-lambdas", type=str, default="25,27,29,31,33")
    parser.add_argument("--sim-admission-number", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument("--out-prefix", type=str, default="fig2")
    args = parser.parse_args()

    # 加载基础配置
    cfg_path = Path(__file__).resolve().parent / "configs" / "sweep_lambda_base.yaml"
    cfg = load_config(cfg_path)
    if args.sim_admission_number is not None:
        cfg["admission_number"] = int(args.sim_admission_number)

    # 输出目录
    plot_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "plot"))
    os.makedirs(plot_dir, exist_ok=True)

    # 策略（新引擎名）
    policies = ["FCFS", "PSF", "AA", "QBA", "RR"]
    phi_fit_lambdas = [float(x.strip()) for x in str(args.phi_fit_lambdas).split(",") if str(x).strip()]
    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.seed_count)))

    sim_lambdas = np.arange(
        float(args.sim_lambda_min), float(args.sim_lambda_max) + 1e-9,
        float(args.sim_lambda_step), dtype=np.float64,
    )
    formula_lambdas = np.arange(
        float(args.formula_lambda_min), float(args.formula_lambda_max) + 1e-9,
        float(args.formula_lambda_step), dtype=np.float64,
    )

    # ── SIM 层 ──
    sim_rows = _run_sim_sweep(cfg, sim_lambdas, policies, seeds)

    # ── 公式层 ──
    formula_rows = _run_formula_sweep(
        cfg, formula_lambdas,
        reference_load=args.reference_load,
        phi_fit_lambdas=phi_fit_lambdas,
    )

    # ── 保存 ──
    arrays = {
        "sim_lambdas": np.asarray(sim_lambdas, dtype=np.float64),
        "formula_lambdas": np.asarray(formula_lambdas, dtype=np.float64),
    }
    for policy in policies:
        key = _POLICY_MAP[policy]
        mean_runs = np.asarray(sim_rows[policy]["mean_runs"], dtype=np.float64)
        p95_runs = np.asarray(sim_rows[policy]["p95_runs"], dtype=np.float64)
        p99_runs = np.asarray(sim_rows[policy]["p99_runs"], dtype=np.float64)
        arrays[f"sim_mean_{key}_runs"] = mean_runs
        arrays[f"sim_p95_{key}_runs"] = p95_runs
        arrays[f"sim_p99_{key}_runs"] = p99_runs
        arrays[f"sim_mean_{key}"] = np.nanmean(mean_runs, axis=0)
        arrays[f"sim_p95_{key}"] = np.nanmean(p95_runs, axis=0)
        arrays[f"sim_p99_{key}"] = np.nanmean(p99_runs, axis=0)

    curves = formula_rows["curves"]
    arrays["formula_EWq_FCFS"] = np.asarray(curves["E[Wq]_FCFS"], dtype=np.float64)
    arrays["formula_EWq_PSF"] = np.asarray(curves["E[Wq]_PSF"], dtype=np.float64)
    arrays["formula_EWq_PSFA"] = np.asarray(curves["E[Wq]_PSFA"], dtype=np.float64)
    arrays["formula_phi_PSFA"] = np.asarray(curves["phi_PSFA"], dtype=np.float64)
    if "E[Wq]_PSFAQ" in curves and "phi_PSFAQ" in curves:
        arrays["formula_EWq_PSFAQ"] = np.asarray(curves["E[Wq]_PSFAQ"], dtype=np.float64)
        arrays["formula_phi_PSFAQ"] = np.asarray(curves["phi_PSFAQ"], dtype=np.float64)
    params_f = formula_rows["params"]
    arrays["formula_reference_load"] = np.asarray([float(params_f.reference_load)], dtype=np.float64)

    # phi_aux 和 aux 是 formula 层特有字段，其他字段由 save_results 统一生成
    extra_meta = {
        "phi_aux": formula_rows.get("phi_aux", {}),
        "aux": formula_rows.get("aux", {}),
    }

    npz_path, meta_path = save_results(
        plot_dir, str(args.out_prefix), arrays, cfg,
        vars(args), config_path=str(cfg_path), extra_meta=extra_meta,
    )
    print(f"saved data: {npz_path}")
    print(f"saved meta: {meta_path}")
    print(f"next: python draw_fig2.py --data \"{npz_path}\"")


if __name__ == "__main__":
    main()
