"""fig12: p × 预测误差 × 构成 的联合分析（fig11 的逐请求总 token 守恒生成 + 误差注入）。

背景（由 fig11 引出）:
    fig11 在"逐请求总 token 严格守恒"下扫描 prefill/decode 构成，发现仅改变构成就能显著
    改变调度性能。本实验把 fig11 的生成方式固定下来，再叠加两个预测侧变量：

        (1) 标签阈值 p（决定 true-short/long 的划分，从而决定 π_S）；
        (2) 预测误差（ε_SL = P(pred=long|true=short)，ε_LS = P(pred=short|true=long)）。

研究问题:
    在不同阶段构成与类别比例下，同样的预测错误，会怎样改变预测类别数量、实际提升比例
    以及真实长短请求的等待？

误差如何同时改变两件事（对应两条理论连接）:
    π̂_S = π_S (1 - ε_SL) + (1 - π_S) ε_LS,
    χ_pred = π̂_S / (k (1 - π̂_S)).
    - χ_pred 衡量"提升机会相对于 predicted-long 数量"的充足程度；
    - 误差还决定 predicted-long 里混入了哪些真实短请求。
    因此本实验不把 n_true_long 当作 promotion fraction 的分母，而保存预测类别计数并计算
        φ_Q = N_prom / N_predicted_long。

关键实现点（务必遵守）:
    误差标签必须在 fig11 的长度变换 + 真标签重算**之后**注入；否则会被 Oracle 重算覆盖。
    为此本脚本在适配层内：先生成基础请求 → 逐请求守恒变换（真值/预测标签置为 Oracle）
    → 再按 (ε_SL, ε_LS) 扰动 predict_label。

设计:
- 网格：构成倍率 a × 标签阈值 p × 误差设置 × 策略 × 种子。
  默认 a ∈ {0.5, 1, 2}，p ∈ {0.75, 0.85, 0.95}，
  误差 ∈ {oracle, sl0.1, sl0.3, ls0.1, ls0.3}，策略 ∈ {FCFS, PSF, AA, QBA}。
  完整首轮 = 3 × 3 × 5 × 4 × 20 = 3600 次。
- 主数组形状统一为 [n_seeds, n_a, n_p, n_err]（与 fig9 的 [n_seeds, n_k, n_p] 风格一致）。
- a=1、p=基础值、oracle 的单元应与 fig11 的 a=1（进而 fig10 的 a=1）逐位一致（锚点）。
- 误差设置假定方向互斥：任一设置只在一个方向注入（SL 或 LS），另一方向为 0。
- 只改请求的 P/D 与标签，不改到达时间与 id；守恒变换不消耗随机数，误差注入才消耗。
- 模块级临时替换 ``simulator.generate_heapq``，每个 (seed, a, p, err) 单元替换一次、
  四个策略共用同一份变换与同一注入，跑完在 ``finally`` 恢复。同一进程内不要并行跑不同
  单元（云端通过独立子进程按 seed 并行）。

不变量声明:
    逐请求总 token 相同并不保证实际负载相同。本实验不记录"被提升者的真实类别"，因此
    **不声称**测得 true-long promotion fraction；φ_Q 只是分母改为 N_predicted_long 的
    promotion 比率。

用法:
    python -m scripts.experiment.sweep_p_error_composition
    python -m scripts.experiment.sweep_p_error_composition --seed-count 20
    python -m scripts.experiment.sweep_p_error_composition \
        --composition-scales "0.5,1,2" --p-values "0.75,0.85,0.95" \
        --seed-count 2 --sim-admission-number 300 --plot-dir scripts/plot/smoke_fig12

输出:
    plot/fig12_data.npz + plot/fig12_meta.json
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from llm_in_queuing import simulator
from llm_in_queuing.config import load_config
from ._common import (
    progress_update,
    progress_done,
    run_one_sim,
    save_results,
)
from .sweep_request_composition import (
    _apply_request_composition,
    _extract_metrics,
    _parse_composition_scales,
    _METRICS_KEYS,
    _POLICY_DIAG_KEYS as _RC_POLICY_DIAG_KEYS,
)

# 误差设置规范表：名称 -> (ε_SL, ε_LS)。
# ε_SL = P(pred=long | true=short)，ε_LS = P(pred=short | true=long)。
_ERROR_SPEC_TABLE: dict[str, tuple[float, float]] = {
    "oracle": (0.0, 0.0),
    "sl0.1": (0.1, 0.0),
    "sl0.3": (0.3, 0.0),
    "ls0.1": (0.0, 0.1),
    "ls0.3": (0.0, 0.3),
}
_DEFAULT_ERROR_ORDER = ["oracle", "sl0.1", "sl0.3", "ls0.1", "ls0.3"]

# 引擎资源诊断（复用 fig11 的键）+ φ_Q
_POLICY_DIAG_KEYS = list(_RC_POLICY_DIAG_KEYS) + ["phi_Q"]

# workload 级诊断（同一 (seed, a, p, err) 只记录一次，与策略无关）
_WORK_DIAG_KEYS = [
    "prompt_mean", "answer_mean", "token_ratio",
    "base_total_mean", "total_mean",
    "base_total_var", "total_var",
    "max_request_total_error",
    "lower_bound_fraction", "upper_bound_fraction",
    "pi_s", "pi_s_pred", "chi_pred",
    "n_true_short", "n_true_long", "n_pred_short", "n_pred_long",
    "label_flip_fraction",
    "actual_epsilon_SL", "actual_epsilon_LS",
]


def _parse_p_values(raw: str) -> np.ndarray:
    """解析 p 列表：有限、位于 (0, 1)、无重复，按升序返回。"""
    parts = [s.strip() for s in str(raw).split(",") if s.strip()]
    if not parts:
        raise ValueError("--p-values 不能为空")
    vals = [float(s) for s in parts]
    for v in vals:
        if not np.isfinite(v) or not (0.0 < v < 1.0):
            raise ValueError(f"非法 p: {v}（必须为有限且位于 (0, 1)）")
    if len(set(vals)) != len(vals):
        raise ValueError(f"--p-values 存在重复值: {parts}")
    return np.asarray(sorted(vals), dtype=np.float64)


def _parse_error_specs(raw: str) -> list[tuple[str, float, float]]:
    """解析误差设置列表：名称必须在规范表内、无重复，保持给定顺序。"""
    parts = [s.strip() for s in str(raw).split(",") if s.strip()]
    if not parts:
        raise ValueError("--error-specs 不能为空")
    for name in parts:
        if name not in _ERROR_SPEC_TABLE:
            raise ValueError(
                f"未知误差设置: {name!r}（可用: {sorted(_ERROR_SPEC_TABLE)}）")
    if len(set(parts)) != len(parts):
        raise ValueError(f"--error-specs 存在重复值: {parts}")
    return [(name, _ERROR_SPEC_TABLE[name][0], _ERROR_SPEC_TABLE[name][1])
            for name in parts]


def _inject_prediction_errors(reqs: list, eps_sl: float, eps_ls: float) -> None:
    """在预测标签上注入方向互斥的误差（就地修改 predict_label）。

    与 ``workload._assign_labels`` 的口径一致：每个请求消耗一次 ``np.random.rand()``，
    true-short 以 ε_SL 概率被判为 long，true-long 以 ε_LS 概率被判为 short。
    Oracle（两个 ε 都为 0）时不消耗任何随机数，保持与 fig11/fig10 的随机流一致。

    注意：本函数必须在真标签重算**之后**调用，否则会被 Oracle 重算覆盖。
    """
    if eps_sl == 0.0 and eps_ls == 0.0:
        return
    for req in reqs:
        r = float(np.random.rand())
        if req.true_label == "short":
            req.predict_label = "short" if r < (1.0 - eps_sl) else "long"
        else:
            req.predict_label = "long" if r < (1.0 - eps_ls) else "short"


def _make_generator(original_generate, scale: float, eps_sl: float, eps_ls: float,
                    diag: dict, base_labels: dict):
    """返回包装生成器：基础请求 → 逐请求守恒变换（真标签重算）→ 误差注入 → 记录计数。"""
    def _gen(cfg: dict):
        heap = original_generate(cfg)
        # 1) fig11 的逐请求守恒变换：调整 P/D 并按新 answer 分布重算真值标签（Oracle）
        _apply_request_composition(heap, scale, cfg, diag, base_labels)
        reqs = [req for _, _, req in heap]
        # 2) 在真标签确定之后注入预测误差（方向互斥）
        _inject_prediction_errors(reqs, eps_sl, eps_ls)

        # 3) 记录预测类别计数与实际误判率（与策略无关，写回 diag）
        n = len(reqs)
        n_true_long = sum(1 for r in reqs if r.true_label == "long")
        n_true_short = n - n_true_long
        n_pred_long = sum(1 for r in reqs if r.predict_label == "long")
        n_pred_short = n - n_pred_long
        n_sl = sum(1 for r in reqs
                   if r.true_label == "short" and r.predict_label == "long")
        n_ls = sum(1 for r in reqs
                   if r.true_label == "long" and r.predict_label == "short")
        diag["n_true_short"] = n_true_short
        diag["n_true_long"] = n_true_long
        diag["n_pred_short"] = n_pred_short
        diag["n_pred_long"] = n_pred_long
        diag["pi_s_pred"] = (n_pred_short / n) if n > 0 else float("nan")
        diag["actual_epsilon_SL"] = (n_sl / n_true_short) if n_true_short > 0 else 0.0
        diag["actual_epsilon_LS"] = (n_ls / n_true_long) if n_true_long > 0 else 0.0
        return heap

    return _gen


def _run_p_error_composition_sweep(
    cfg: dict,
    lam: float,
    scales: np.ndarray,
    percentiles: np.ndarray,
    error_specs: list[tuple[str, float, float]],
    policies: list[str],
    seeds: list[int],
) -> tuple[dict, dict, dict]:
    """在 (a, p, 误差) 三维网格上扫参多个策略（逐请求总 token 守恒 + 误差后注入）。

    Returns:
        (out, policy_diag, work_diag)
        out: {policy: {metric: np.ndarray[n_seeds, n_a, n_p, n_err]}}
        policy_diag: {policy: {diag_key: np.ndarray[...]}}（含 φ_Q）
        work_diag: {diag_key: np.ndarray[...]}（与策略无关）
    """
    n_seeds = len(seeds)
    n_a = len(scales)
    n_p = len(percentiles)
    n_err = len(error_specs)
    shape = (n_seeds, n_a, n_p, n_err)

    out = {
        p: {k: np.full(shape, np.nan, dtype=np.float64) for k in _METRICS_KEYS}
        for p in policies
    }
    policy_diag = {
        p: {k: np.full(shape, np.nan, dtype=np.float64) for k in _POLICY_DIAG_KEYS}
        for p in policies
    }
    work_diag = {k: np.full(shape, np.nan, dtype=np.float64)
                 for k in _WORK_DIAG_KEYS}

    original_generate = simulator.generate_heapq
    quota_k = int(cfg["quota_k"])
    total = n_seeds * n_a * n_p * n_err * len(policies)
    done = 0
    print(f"stage: SIM p-error-composition sweep runs={total} seeds={n_seeds} "
          f"a={n_a} p={n_p} err={n_err} policies={len(policies)}")
    for si, seed in enumerate(seeds):
        for pj, pct in enumerate(percentiles.tolist()):
            # 该 (seed, p) 下的基础真值标签参照（a=1 原始生成器），用于记录 flip 比例
            base_cfg = dict(cfg)
            base_cfg.update({
                "lambda_arrival": float(lam),
                "seed": int(seed),
                "label_threshold_percentail": float(pct),
            })
            base_heap = original_generate(base_cfg)
            base_labels = {r.id: r.true_label for _, _, r in base_heap}

            for ai, a in enumerate(scales.tolist()):
                for ej, (ename, eps_sl, eps_ls) in enumerate(error_specs):
                    cell_diag: dict = {}
                    simulator.generate_heapq = _make_generator(
                        original_generate, float(a), float(eps_sl), float(eps_ls),
                        cell_diag, base_labels,
                    )
                    try:
                        for pol in policies:
                            local = dict(cfg)
                            local.update({
                                "lambda_arrival": float(lam),
                                "schedule_method": pol,
                                "seed": int(seed),
                                "label_threshold_percentail": float(pct),
                                # 误差由适配层注入：workload 侧保持 Oracle 配置
                                "TN": 0.0, "TP": 0.0, "FN": 0.0, "FP": 0.0,
                            })
                            try:
                                s = _extract_metrics(run_one_sim(local))
                            except Exception as exc:  # 保留原异常，补充实验上下文
                                raise RuntimeError(
                                    f"Experiment failed: seed={seed} a={a} p={pct} "
                                    f"error={ename} policy={pol}"
                                ) from exc
                            for mk in _METRICS_KEYS:
                                out[pol][mk][si, ai, pj, ej] = float(s[mk])
                            for dk in _RC_POLICY_DIAG_KEYS:
                                policy_diag[pol][dk][si, ai, pj, ej] = float(s[dk])
                            # φ_Q = N_prom / N_predicted_long（分母为预测 long 计数）
                            n_pred_long = float(cell_diag.get("n_pred_long", float("nan")))
                            policy_diag[pol]["phi_Q"][si, ai, pj, ej] = (
                                float(s["promoted_total"]) / n_pred_long
                                if n_pred_long > 0 else float("nan")
                            )
                            done += 1
                            progress_update(done, total, "SIM p-error-composition")
                    finally:
                        simulator.generate_heapq = original_generate

                    # workload 诊断（与策略无关，单元跑完后记录一次）
                    for dk in _WORK_DIAG_KEYS:
                        if dk == "chi_pred":
                            continue
                        work_diag[dk][si, ai, pj, ej] = float(
                            cell_diag.get(dk, float("nan")))
                    pis = float(cell_diag.get("pi_s_pred", float("nan")))
                    work_diag["chi_pred"][si, ai, pj, ej] = (
                        pis / (quota_k * (1.0 - pis))
                        if np.isfinite(pis) and pis < 1.0 else float("nan")
                    )
    progress_done("SIM p-error-composition", total)
    return out, policy_diag, work_diag


def main():
    parser = argparse.ArgumentParser(
        description="fig12: p × 预测误差 × 构成 联合分析")
    parser.add_argument("--composition-scales", type=str, default="0.5,1,2",
                        help="逗号分隔的构成倍率列表")
    parser.add_argument("--p-values", type=str, default="0.75,0.85,0.95",
                        help="逗号分隔的标签阈值 p 列表（显式指定，避免浮点步长误差）")
    parser.add_argument("--error-specs", type=str,
                        default=",".join(_DEFAULT_ERROR_ORDER),
                        help="逗号分隔的误差设置列表，可选: "
                             f"{','.join(_DEFAULT_ERROR_ORDER)}")
    parser.add_argument("--reference-load", type=float, default=32.0)
    parser.add_argument("--gamma", type=float, default=0.80)
    parser.add_argument("--quota-k", type=int, default=5)
    parser.add_argument("--plot-dir", type=str, default=os.path.join(
        os.path.dirname(__file__), "..", "plot"))
    parser.add_argument("--out-prefix", type=str, default="fig12")
    parser.add_argument("--sim-admission-number", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--seed-count", type=int, default=100,
                        help="种子数量，默认 100；首轮探索建议 20（42-61）")
    args = parser.parse_args()

    # 加载基础配置
    cfg_path = Path(__file__).resolve().parent / "configs" / "sweep_p_error_composition_base.yaml"
    cfg = load_config(cfg_path)
    if args.sim_admission_number is not None:
        cfg["admission_number"] = int(args.sim_admission_number)
    # 固定老化相关参数，仅让 (a, p, 误差) 变化（控制变量）
    cfg["gamma"] = float(args.gamma)
    cfg["quota_k"] = int(args.quota_k)

    plot_dir = os.path.abspath(args.plot_dir)

    reference_lambda = float(args.reference_load)
    scales = _parse_composition_scales(args.composition_scales)
    percentiles = _parse_p_values(args.p_values)
    error_specs = _parse_error_specs(args.error_specs)
    seeds = [int(args.seed_start) + i for i in range(int(args.seed_count))]

    # 构成与误差都会改变整体 workload，FCFS 一并扫参作为相对基线
    sweep_policies = ["FCFS", "PSF", "AA", "QBA"]

    error_names = [name for name, _, _ in error_specs]
    error_eps_sl = np.asarray([eps for _, eps, _ in error_specs], dtype=np.float64)
    error_eps_ls = np.asarray([eps for _, _, eps in error_specs], dtype=np.float64)

    print(f"stage: SIM p-error-composition sweep scales={scales.tolist()} "
          f"p={percentiles.tolist()} errors={error_names} "
          f"lambda={reference_lambda} gamma={cfg['gamma']} quota_k={cfg['quota_k']} "
          f"repeats={len(seeds)}")
    rows, policy_diag, work_diag = _run_p_error_composition_sweep(
        cfg, reference_lambda, scales, percentiles, error_specs,
        sweep_policies, seeds,
    )

    # 保存输出（主数组统一 [n_seeds, n_a, n_p, n_err]）
    arrays = {
        "reference_lambda": np.asarray([float(reference_lambda)], dtype=np.float64),
        "composition_scales": np.asarray(scales, dtype=np.float64),
        "percentiles": np.asarray(percentiles, dtype=np.float64),
        "error_names": np.asarray(error_names),
        "error_eps_sl": error_eps_sl,
        "error_eps_ls": error_eps_ls,
        "seeds": np.asarray(seeds, dtype=np.int64),
    }
    for dk in _WORK_DIAG_KEYS:
        arrays[f"{dk}_runs"] = work_diag[dk]

    for p, prefix in [("FCFS", "fcfs"), ("PSF", "psf"), ("AA", "psfa"), ("QBA", "psfaq")]:
        for mk in _METRICS_KEYS:
            arrays[f"{prefix}_{mk}_runs"] = rows[p][mk]
        for dk in _POLICY_DIAG_KEYS:
            arrays[f"{prefix}_{dk}_runs"] = policy_diag[p][dk]

    extra_meta = {
        "experiment": "p_error_composition_joint",
        "sweep_policies": sweep_policies,
        "composition_scales": [float(v) for v in scales.tolist()],
        "percentiles": [float(v) for v in percentiles.tolist()],
        "quota_k": int(cfg["quota_k"]),
        "error_specs": {
            name: {"epsilon_SL": float(sl), "epsilon_LS": float(ls)}
            for name, sl, ls in error_specs
        },
        "normalization": "exact per-request total-token preservation (fig11 transform)",
        "answer_preserved": False,
        "labels_recomputed": (
            "oracle quantile on transformed answers; prediction errors injected "
            "afterward (never overwritten by the oracle recompute)"
        ),
        "error_injection": "post-transform perturbation of predict_label",
        "identity_anchor": "a=1, p=base, error=oracle (matches fig11/fig10 a=1)",
        "phi_Q_definition": "PSFAQ_promoted_total / N_predicted_long",
        "chi_pred_formula": "pi_s_pred / (k * (1 - pi_s_pred))",
        "pi_s_pred_formula": "pi_S (1 - eps_SL) + (1 - pi_S) eps_LS",
        "note_no_true_long_promotion": (
            "promoted requests' true class is not recorded; true-long promotion "
            "fraction is NOT measured"
        ),
    }

    npz_path, meta_path = save_results(
        plot_dir, str(args.out_prefix), arrays, cfg,
        vars(args), config_path=str(cfg_path), extra_meta=extra_meta,
    )
    print(f"saved data: {npz_path}")
    print(f"saved meta: {meta_path}")
    print(f"next: python -m scripts.draw.draw_fig12 --data \"{npz_path}\"")


if __name__ == "__main__":
    main()
