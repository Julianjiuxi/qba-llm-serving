"""面向论文可复现性的回归测试。

用重构后的 LLM_in_Queuing 引擎重跑 fig2/fig3/fig4/fig5/fig6 的 SIM 层实验，
对前 k 个 seed（默认 k=1）逐 seed 与 freeze/ 下对应 npz 的同 seed 段
做数值断言，确认可独立复现论文结果。

用法:
    python reproduction.py            # k=1
    python reproduction.py --k 3      # 前 3 个 seed
    python reproduction.py --tol 1e-6  # 自定义容差

说明:
    - 显式关闭 warm-up 窗口（steady_state.enabled=false，与 freeze 一致）。
    - 不写任何输出文件，仅在内存中比较。
    - fig2 的公式层（formula_*）是确定性计算、不依赖 seed，故不在本
      回归范围内；本测试只覆盖含 seed 维度的 SIM 层 *_runs 数据。
    - freeze 中 fig2/3/4 不含 RR 策略，因此本测试同样不跑 RR。
    - fig5/6 覆盖 PSF/AA/QBA 三策略及 FCFS 参考，校验 per-class 指标。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
for _p in (ROOT, SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llm_in_queuing.config import load_config
from scripts.experiment import (
    sweep_lambda,
    sweep_quotak,
    sweep_gamma,
    sweep_epsilonSL,
    sweep_epsilonLS,
)
from scripts.experiment._common import run_references

FREEZE_DIR = os.path.join(ROOT, "freeze")
CONFIG_DIR = os.path.join(ROOT, "scripts", "experiment", "configs")

# freeze 里的策略名 -> 新引擎策略名（freeze 无 RR）
FIG2_POLICIES = ["FCFS", "PSF", "AA", "QBA"]
FIG2_NPZ_KEY = {"FCFS": "FCFS", "PSF": "PSF", "AA": "PSFA", "QBA": "PSFAQ"}
FIG3_REF_POLICIES = ["FCFS", "PSF", "AA"]
FIG3_REF_NPZ = {"FCFS": "fcfs", "PSF": "psf", "AA": "psfa"}
FIG4_REF_POLICIES = ["FCFS", "PSF"]
FIG4_REF_NPZ = {"FCFS": "fcfs", "PSF": "psf"}

STATS = ["mean", "p95", "p99"]

# fig5/fig6 使用扩展的 per-class 指标；RR 不在此三策略论文 panel 中。
FIG56_POLICIES = ["PSF", "AA", "QBA"]
FIG56_NPZ = {"PSF": "psf", "AA": "psfa", "QBA": "psfaq"}
FIG56_METRICS = ["mean", "p95", "p99", "ts_mean", "ts_p99", "tl_mean", "tl_p99"]


def _assert_close(got, ref, label, tol):
    """断言两个数组在容差内一致，失败时报出 label 与最大偏差。"""
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    try:
        np.testing.assert_allclose(got, ref, rtol=tol, atol=tol)
    except AssertionError as exc:
        diff = np.abs(got - ref)
        raise AssertionError(
            f"{label} mismatch: max_abs_diff={float(diff.max()):.6e} "
            f"tol={tol}"
        ) from exc


def _close_warmup(cfg: dict) -> dict:
    """显式关闭 warm-up / 稳态窗口，与 freeze 生成时保持一致。"""
    cfg = dict(cfg)
    cfg["steady_state"] = {"enabled": False, "trim_ratio": 0.1}
    return cfg


def _check_fig2(tol: float, k: int) -> None:
    data = np.load(os.path.join(FREEZE_DIR, "fig2_data.npz"))
    cfg = _close_warmup(load_config(os.path.join(CONFIG_DIR, "sweep_lambda_base.yaml")))
    sim_lambdas = data["sim_lambdas"]
    # fig2_data.npz 未存 seeds 数组，其 runs 第 0 行对应 seed_start=42
    seeds = list(range(42, 42 + k))
    print(f"[fig2] seeds={seeds} lambdas={len(sim_lambdas)} policies={FIG2_POLICIES}")
    rows = sweep_lambda._run_sim_sweep(cfg, sim_lambdas, FIG2_POLICIES, seeds)
    for p in FIG2_POLICIES:
        key = FIG2_NPZ_KEY[p]
        for stat in STATS:
            _assert_close(
                rows[p][f"{stat}_runs"],
                data[f"sim_{stat}_{key}_runs"][:k],
                f"fig2 {p} {stat}",
                tol,
            )
    print("[fig2] OK")


def _check_fig3(tol: float, k: int) -> None:
    data = np.load(os.path.join(FREEZE_DIR, "fig3_data.npz"))
    cfg = _close_warmup(load_config(os.path.join(CONFIG_DIR, "sweep_quotak_base.yaml")))
    lam = float(data["reference_lambda"][0])
    quota_ks = data["quota_ks"]
    seeds = [int(x) for x in data["seeds"][:k]]
    print(f"[fig3] seeds={seeds} quota_ks={quota_ks.tolist()} lambda={lam}")
    # QBA 扫参
    rows = sweep_quotak._run_quotak_sweep(cfg, lam, quota_ks, seeds)
    for stat in STATS:
        _assert_close(
            rows[f"{stat}_runs"],
            data[f"psfaq_{stat}_runs"][:k],
            f"fig3 QBA {stat}",
            tol,
        )
    # 参考策略 FCFS / PSF / AA
    refs = run_references(cfg, lam, FIG3_REF_POLICIES, seeds)
    for p in FIG3_REF_POLICIES:
        nkey = FIG3_REF_NPZ[p]
        for stat in STATS:
            _assert_close(
                refs[p][f"{stat}_runs"],
                data[f"ref_{nkey}_{stat}_runs"][:k],
                f"fig3 ref {p} {stat}",
                tol,
            )
    print("[fig3] OK")


def _check_fig4(tol: float, k: int) -> None:
    data = np.load(os.path.join(FREEZE_DIR, "fig4_data.npz"))
    cfg = _close_warmup(load_config(os.path.join(CONFIG_DIR, "sweep_gamma_base.yaml")))
    lam = float(data["reference_lambda"][0])
    gammas = data["gammas"]
    quota_k = int(data["quota_k"][0])
    seeds = [int(x) for x in data["seeds"][:k]]
    print(f"[fig4] seeds={seeds} gammas={gammas.tolist()} quota_k={quota_k} lambda={lam}")
    # AA 与 QBA 扫参
    rows = sweep_gamma._run_gamma_sweep(cfg, lam, gammas, quota_k, seeds)
    for pkey, nkey in (("PSFA", "psfa"), ("PSFAQ", "psfaq")):
        for stat in STATS:
            _assert_close(
                rows[pkey][f"{stat}_runs"],
                data[f"{nkey}_{stat}_runs"][:k],
                f"fig4 {pkey} {stat}",
                tol,
            )
    # 参考策略 FCFS / PSF
    refs = run_references(cfg, lam, FIG4_REF_POLICIES, seeds)
    for p in FIG4_REF_POLICIES:
        nkey = FIG4_REF_NPZ[p]
        for stat in STATS:
            _assert_close(
                refs[p][f"{stat}_runs"],
                data[f"ref_{nkey}_{stat}_runs"][:k],
                f"fig4 ref {p} {stat}",
                tol,
            )
    print("[fig4] OK")


def _check_fig5(tol: float, k: int) -> None:
    data = np.load(os.path.join(FREEZE_DIR, "fig5_data.npz"))
    cfg = _close_warmup(load_config(os.path.join(CONFIG_DIR, "sweep_epsilonSL_base.yaml")))
    lam = float(data["reference_lambda"][0])
    epsilons_sl = data["epsilons_sl"]
    eps_ls_fixed = float(data["epsilon_ls_fixed"][0])
    seeds = [int(x) for x in data["seeds"][:k]]
    print(f"[fig5] seeds={seeds} epsilons_sl={epsilons_sl.tolist()} lambda={lam}")
    rows = sweep_epsilonSL._run_epsilon_sweep(
        cfg, lam, epsilons_sl, eps_ls_fixed, FIG56_POLICIES, seeds)
    for p in FIG56_POLICIES:
        nkey = FIG56_NPZ[p]
        for mk in FIG56_METRICS:
            _assert_close(
                rows[p][f"{mk}_runs"], data[f"{nkey}_{mk}_runs"][:k],
                f"fig5 {p} {mk}", tol)
    refs = run_references(cfg, lam, ["FCFS"], seeds, extended=True)
    for mk in FIG56_METRICS:
        _assert_close(
            refs["FCFS"][f"{mk}_runs"], data[f"ref_fcfs_{mk}_runs"][:k],
            f"fig5 ref FCFS {mk}", tol)
    print("[fig5] OK")


def _check_fig6(tol: float, k: int) -> None:
    data = np.load(os.path.join(FREEZE_DIR, "fig6_data.npz"))
    cfg = _close_warmup(load_config(os.path.join(CONFIG_DIR, "sweep_epsilonLS_base.yaml")))
    lam = float(data["reference_lambda"][0])
    epsilons_ls = data["epsilons_ls"]
    eps_sl_fixed = float(data["epsilon_sl_fixed"][0])
    seeds = [int(x) for x in data["seeds"][:k]]
    print(f"[fig6] seeds={seeds} epsilons_ls={epsilons_ls.tolist()} lambda={lam}")
    rows = sweep_epsilonLS._run_epsilon_sweep(
        cfg, lam, epsilons_ls, eps_sl_fixed, FIG56_POLICIES, seeds)
    for p in FIG56_POLICIES:
        nkey = FIG56_NPZ[p]
        for mk in FIG56_METRICS:
            _assert_close(
                rows[p][f"{mk}_runs"], data[f"{nkey}_{mk}_runs"][:k],
                f"fig6 {p} {mk}", tol)
    refs = run_references(cfg, lam, ["FCFS"], seeds, extended=True)
    for mk in FIG56_METRICS:
        _assert_close(
            refs["FCFS"][f"{mk}_runs"], data[f"ref_fcfs_{mk}_runs"][:k],
            f"fig6 ref FCFS {mk}", tol)
    print("[fig6] OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="论文 fig2/3/4/5/6 可复现性回归测试")
    parser.add_argument("--k", type=int, default=1, help="抽取并校验的 seed 数量")
    parser.add_argument("--tol", type=float, default=1e-8, help="数值比较容差（rtol=atol）")
    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k 必须 >= 1")

    print(f"reproduction check: k={args.k} tol={args.tol}")
    _check_fig2(args.tol, args.k)
    _check_fig3(args.tol, args.k)
    _check_fig4(args.tol, args.k)
    _check_fig5(args.tol, args.k)
    _check_fig6(args.tol, args.k)
    print("ALL REPRODUCTION CHECKS PASSED")


if __name__ == "__main__":
    main()
