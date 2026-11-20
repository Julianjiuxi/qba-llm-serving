"""fig10: prefill/decode 构成扫描（总 token 需求归一化）— 固定到达率下服务构成对调度的影响。

背景（由 fig8 引出）:
    fig8 固定 answer 与 λ、只放大 prompt，于是"构成变化"和"总需求增加"混在一起，
    高负载区被总需求膨胀主导，看不到纯粹的 prefill/decode 异构效应。本实验改为在
    **匹配平均总 token 需求**的前提下，只改变 prefill/decode 的构成。

归一化规则:
    以每个 seed 的基础请求 (P_i, D_i)（a=1，即原始 workload）为基准，设
        target = E[P] + E[D]
        s(a)   = target / (a * E[P] + E[D])
    再同时变换
        P_i' = clip(round(s(a) * a * P_i), 5, prompt_len_cap)
        D_i' = clip(round(s(a) * D_i),    1, answer_len_cap)
    忽略取整/上限时，E[P'+D'] = E[P+D]（总 token 需求不变），且
    E[P']/E[D'] = a * E[P]/E[D]（构成比按倍率 a 改变）。由于 round/clip 会引入偏差，
    脚本对 s 做有限次迭代校准，使实测平均总 token 的相对误差 <= 0.1%；若迭代耗尽仍未
    达标则直接报错（带上下文），不会在未收敛的情况下继续仿真。

设计要点:
- a=1 时 s=1 且不做多余舍入/截断，逐字段还原原始 workload，可与 fig7 的
  (p=0.85, k=5, 同 seed) 结果对位（验收锚点）。
- **answer 长度会随 a 改变**（相对 fig8 的明确设计变化）——这是"固定总需求"的代价：
  要同时满足 λ 固定、总 token 固定、构成变化，增加 prompt 就必须减少 decode，反之亦然。
- 标签在变换后按**新的 answer 分布**重新计算（p = label_threshold_percentail）。本实验
  要求 oracle 配置（TN/TP/FN/FP 全为 0），标签采用确定性重算（不调用会消耗全局随机数的
  ``_assign_labels``）；公共缩放保持 answer 排序，故绝大多数标签不变，但取整/截断造成的
  并列可能翻转少量标签，脚本记录 label flip 比例与实际 short 占比 pi_s。
- 只改请求的 P/D 与标签，不改到达时间；**不消耗任何额外随机数**（由单测保证）。
- 实现方式与 fig8 一致：模块级临时替换 ``simulator.generate_heapq``，每个
  (seed, scale) 单元替换一次、四个策略共用同一份变换，跑完在 ``finally`` 恢复原函数。
  注意这是模块级替换，同一进程内不要并行跑不同倍率。
- **不变量声明**：总 token 相同并不保证实际负载相同（prefill 可用批次剩余预算、decode
  占多轮服务、KV 驻留与阶段构成相关）。因此本实验应称为"固定到达率、匹配平均总 token
  需求的 prefill/decode 构成扫描"，不等价于"严格等负载"或"独立 R 扫描"。

用法:
    python -m scripts.experiment.sweep_composition
    python -m scripts.experiment.sweep_composition --composition-scales "0.25,0.5,1,2,4"
    python -m scripts.experiment.sweep_composition --seed-count 2 --sim-admission-number 300

输出:
    plot/fig10_data.npz + plot/fig10_meta.json
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

# 与 workload.py 生成阶段一致的长度硬下限
_PROMPT_LOWER_BOUND = 5
_ANSWER_LOWER_BOUND = 1

# 归一化迭代校准（容差 0.1%；迭代耗尽仍未达标则报错，不再继续仿真）
_NORM_MAX_ITER = 12
_NORM_TOL = 1e-3

# 每条 (policy) 的等待指标（与 fig8 对齐，便于横向比较）
_METRICS_KEYS = [
    "mean", "p95", "p99",
    "ts_mean", "ts_p99",
    "tl_mean", "tl_p99",
    "actual_epsilon_SL", "actual_epsilon_LS",
]

# 每条 (policy) 的引擎资源诊断
_POLICY_DIAG_KEYS = [
    "oom_prob", "throughput", "batch_util", "promoted_total",
]

# 与策略无关的 workload 诊断（同一 (seed, scale) 只记录一次）
_WORK_DIAG_KEYS = [
    "prompt_mean", "answer_mean", "total_mean", "token_ratio",
    "base_total_mean", "norm_scale", "total_rel_error",
    "prompt_upper_clip_fraction", "answer_upper_clip_fraction",
    "label_flip_fraction", "pi_s",
]


def _parse_composition_scales(raw: str) -> np.ndarray:
    """解析构成倍率列表：有限、严格大于 0、无重复，按升序返回。"""
    parts = [s.strip() for s in str(raw).split(",") if s.strip()]
    if not parts:
        raise ValueError("--composition-scales 不能为空")
    scales = [float(s) for s in parts]
    for a in scales:
        if not np.isfinite(a) or a <= 0.0:
            raise ValueError(f"非法 composition_scale: {a}（必须为有限且 > 0）")
    if len(set(scales)) != len(scales):
        raise ValueError(f"--composition-scales 存在重复值: {parts}")
    return np.asarray(sorted(scales), dtype=np.float64)


def _apply_composition(
    heap: list,
    scale: float,
    cfg: dict,
    diag: dict | None = None,
    base_labels: dict | None = None,
) -> None:
    """就地按构成倍率 ``scale`` 同时缩放 P/D，使平均总 token 需求保持不变。

    归一化系数 s 由基础请求的均值闭式给出，并对 round/clip 的偏差做有限次迭代校准。
    变换后按新的 answer 分布重算真值/预测标签。所有统计写入 ``diag``。
    """
    reqs = [req for _, _, req in heap]
    n = len(reqs)
    if n == 0:
        return

    p_cap = int(cfg.get("prompt_len_cap", 10 ** 9))
    a_cap = int(cfg.get("answer_len_cap", 10 ** 9))
    p0 = np.array([int(r.prompt_len) for r in reqs], dtype=np.float64)
    d0 = np.array([int(r.answer_len) for r in reqs], dtype=np.float64)

    mean_p0 = float(p0.mean())
    mean_d0 = float(d0.mean())
    target = mean_p0 + mean_d0  # E[P] + E[D]（基准总需求）

    # 闭式初值 s(a) = target / (a*E[P] + E[D])；a=1 时严格等于 1。
    # 每次迭代都用「当前 s」重新算出 p_new/d_new，只有验收通过（或 realized<=0 无法继续）
    # 才跳出；这样跳出时保存的 s 一定对应实际写回的那一批长度，不会出现 s 与变换脱节。
    s = target / (float(scale) * mean_p0 + mean_d0)
    p_new = d_new = None
    realized = float("nan")
    rel_err = float("nan")
    converged = False
    for _ in range(_NORM_MAX_ITER):
        p_pre = s * float(scale) * p0
        d_pre = s * d0
        p_new = np.clip(np.round(p_pre), _PROMPT_LOWER_BOUND, p_cap)
        d_new = np.clip(np.round(d_pre), _ANSWER_LOWER_BOUND, a_cap)
        realized = float(p_new.mean() + d_new.mean())
        if realized <= 0.0:
            break
        rel_err = abs(realized - target) / target if target > 0 else float("nan")
        if rel_err <= _NORM_TOL:
            converged = True
            break
        s = s * (target / realized)

    if not converged:
        raise RuntimeError(
            f"构成归一化未在 {_NORM_MAX_ITER} 次迭代内收敛: scale={scale} "
            f"target={target:.6f} realized={realized:.6f} "
            f"rel_err={rel_err:.6e} tol={_NORM_TOL:.1e} s={s:.6f}"
        )

    # 写回请求（heap 持有同一批 Request 对象，就地更新即可）
    for i, req in enumerate(reqs):
        req.prompt_len = int(p_new[i])
        req.prefill_res_token = int(p_new[i])
        req.answer_len = int(d_new[i])

    # 变换后按新的 answer 分布「确定性」重算标签（不消耗全局随机数）。
    # 本实验要求 oracle 配置：先校验误判率全为 0，再直接赋值，避免 _assign_labels 内部的
    # np.random.rand() 调用改变随机流（与"不引入额外随机数"的承诺一致）。
    tf = [float(cfg.get(k, 0.0)) for k in ("TN", "TP", "FN", "FP")]
    if any(v != 0.0 for v in tf):
        raise ValueError(
            f"fig10 标签重算假定 oracle（TN/TP/FN/FP 全为 0），实际为 TN/TP/FN/FP={tf}"
        )
    ans = np.array([int(r.answer_len) for r in reqs], dtype=np.float64)
    threshold = float(np.quantile(ans, float(cfg["label_threshold_percentail"])))
    n_short = 0
    for req in reqs:
        lab = "short" if req.answer_len <= threshold else "long"
        req.true_label = lab
        req.predict_label = lab
        if lab == "short":
            n_short += 1

    if diag is not None:
        mean_p = float(p_new.mean())
        mean_d = float(d_new.mean())
        diag["prompt_mean"] = mean_p
        diag["answer_mean"] = mean_d
        diag["total_mean"] = mean_p + mean_d
        diag["base_total_mean"] = float(target)
        diag["token_ratio"] = (mean_p / mean_d) if mean_d > 0 else float("nan")
        diag["norm_scale"] = float(s)
        diag["total_rel_error"] = float(rel_err)
        diag["prompt_upper_clip_fraction"] = float(np.mean(p_pre > p_cap))
        diag["answer_upper_clip_fraction"] = float(np.mean(d_pre > a_cap))
        diag["pi_s"] = n_short / n
        if base_labels is not None:
            flips = sum(1 for r in reqs if base_labels.get(r.id) != r.true_label)
            diag["label_flip_fraction"] = flips / n


def _make_composition_generator(original_generate, scale: float, diag: dict,
                                base_labels: dict):
    """返回一个包装后的生成器：先产出基础请求堆，再做确定性的构成归一化。"""
    def _composition_generate(cfg: dict):
        heap = original_generate(cfg)
        _apply_composition(heap, scale, cfg, diag, base_labels)
        return heap

    return _composition_generate


def _extract_metrics(metrics: dict) -> dict:
    """从引擎 metrics 提取等待指标与资源诊断（适配层，不改旧接口）。"""
    return {
        "mean": float(metrics.get("waiting_time平均", float("nan"))),
        "p95": float(metrics.get("waiting_time p95", float("nan"))),
        "p99": float(metrics.get("waiting_time p99", float("nan"))),
        "ts_mean": float(metrics.get("waiting_time_true_short_mean", float("nan"))),
        "ts_p99": float(metrics.get("waiting_time_true_short_p99", float("nan"))),
        "tl_mean": float(metrics.get("waiting_time_true_long_mean", float("nan"))),
        "tl_p99": float(metrics.get("waiting_time_true_long_p99", float("nan"))),
        "actual_epsilon_SL": float(metrics.get("actual_epsilon_SL", 0.0)),
        "actual_epsilon_LS": float(metrics.get("actual_epsilon_LS", 0.0)),
        "oom_prob": float(metrics.get("OOM概率", float("nan"))),
        "throughput": float(metrics.get("平均吞吐量", float("nan"))),
        "batch_util": float(metrics.get("平均batch使用率", float("nan"))),
        "promoted_total": float(metrics.get("PSFAQ_promoted_total", float("nan"))),
    }


def _run_composition_sweep(
    cfg: dict,
    lam: float,
    scales: np.ndarray,
    policies: list[str],
    seeds: list[int],
) -> tuple[dict, dict, dict]:
    """按构成倍率扫参多个策略。

    Returns:
        (out, policy_diag, work_diag)
        out: {policy: {metric: np.ndarray[n_seeds, n_points]}}
        policy_diag: {policy: {diag_key: np.ndarray[n_seeds, n_points]}}
        work_diag: {diag_key: np.ndarray[n_seeds, n_points]}（与策略无关）
    """
    n_seeds = len(seeds)
    n_points = len(scales)

    out = {
        p: {k: np.full((n_seeds, n_points), np.nan, dtype=np.float64)
            for k in _METRICS_KEYS}
        for p in policies
    }
    policy_diag = {
        p: {k: np.full((n_seeds, n_points), np.nan, dtype=np.float64)
            for k in _POLICY_DIAG_KEYS}
        for p in policies
    }
    work_diag = {k: np.full((n_seeds, n_points), np.nan, dtype=np.float64)
                 for k in _WORK_DIAG_KEYS}

    original_generate = simulator.generate_heapq
    total = n_points * len(policies) * n_seeds
    done = 0
    print(f"stage: SIM composition sweep runs={total} seeds={n_seeds} "
          f"points={n_points} policies={len(policies)}")
    for si, seed in enumerate(seeds):
        # 基准 workload：用于取真值标签参照（真值标签随 a 变化，需逐倍率对比）
        base_cfg = dict(cfg)
        base_cfg.update({"lambda_arrival": float(lam), "seed": int(seed)})
        base_heap = original_generate(base_cfg)
        base_labels = {r.id: r.true_label for _, _, r in base_heap}

        for li, a in enumerate(scales.tolist()):
            scale = float(a)
            cell_diag: dict = {}
            simulator.generate_heapq = _make_composition_generator(
                original_generate, scale, cell_diag, base_labels,
            )
            try:
                for p in policies:
                    local = dict(cfg)
                    local.update({
                        "lambda_arrival": float(lam),
                        "schedule_method": p,
                        "seed": int(seed),
                    })
                    try:
                        s = _extract_metrics(run_one_sim(local))
                    except Exception as exc:  # 保留原异常，补充实验上下文
                        raise RuntimeError(
                            f"Experiment failed: seed={seed} "
                            f"composition_scale={scale} policy={p}"
                        ) from exc
                    for mk in _METRICS_KEYS:
                        out[p][mk][si, li] = float(s[mk])
                    for dk in _POLICY_DIAG_KEYS:
                        policy_diag[p][dk][si, li] = float(s[dk])
                    done += 1
                    progress_update(done, total, "SIM composition sweep")
            finally:
                simulator.generate_heapq = original_generate
            # workload 诊断与策略无关，单元跑完后记录一次
            for dk in _WORK_DIAG_KEYS:
                work_diag[dk][si, li] = float(cell_diag.get(dk, float("nan")))
    progress_done("SIM composition sweep", total)
    return out, policy_diag, work_diag


def main():
    parser = argparse.ArgumentParser(description="fig10: prefill/decode 构成扫描（总 token 归一化）")
    parser.add_argument("--composition-scales", type=str, default="0.25,0.5,1,2,4",
                        help="逗号分隔的构成倍率列表，例如 \"0.25,0.5,1,2,4\"")
    parser.add_argument("--reference-load", type=float, default=32.0)
    parser.add_argument("--gamma", type=float, default=0.80)
    parser.add_argument("--quota-k", type=int, default=5)
    parser.add_argument("--plot-dir", type=str, default=os.path.join(
        os.path.dirname(__file__), "..", "plot"))
    parser.add_argument("--out-prefix", type=str, default="fig10")
    parser.add_argument("--sim-admission-number", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--seed-count", type=int, default=100)
    args = parser.parse_args()

    # 加载基础配置
    cfg_path = Path(__file__).resolve().parent / "configs" / "sweep_composition_base.yaml"
    cfg = load_config(cfg_path)
    if args.sim_admission_number is not None:
        cfg["admission_number"] = int(args.sim_admission_number)
    # 固定老化相关参数，仅让构成倍率变化（控制变量）
    cfg["gamma"] = float(args.gamma)
    cfg["quota_k"] = int(args.quota_k)

    plot_dir = os.path.abspath(args.plot_dir)

    reference_lambda = float(args.reference_load)
    scales = _parse_composition_scales(args.composition_scales)
    seeds = [int(args.seed_start) + i for i in range(int(args.seed_count))]

    # 构成变化会改变整体 workload，FCFS 整体指标亦变化，一并扫参
    sweep_policies = ["FCFS", "PSF", "AA", "QBA"]

    print(f"stage: SIM composition sweep scales={scales.tolist()} lambda={reference_lambda} "
          f"gamma={cfg['gamma']} quota_k={cfg['quota_k']} repeats={len(seeds)}")
    rows, policy_diag, work_diag = _run_composition_sweep(
        cfg, reference_lambda, scales, sweep_policies, seeds,
    )

    # 保存输出
    arrays = {
        "reference_lambda": np.asarray([float(reference_lambda)], dtype=np.float64),
        "composition_scales": np.asarray(scales, dtype=np.float64),
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
        "experiment": "prefill_decode_composition",
        "sweep_policies": sweep_policies,
        "ref_policies": [],
        "composition_scales": np.asarray(scales, dtype=np.float64).tolist(),
        "normalization": "match mean total token demand E[P+D] to the a=1 baseline",
        "normalization_formula": "s(a) = (E[P]+E[D]) / (a*E[P]+E[D]); P'=clip(round(s*a*P)); D'=clip(round(s*D))",
        "norm_tol": _NORM_TOL,
        "answer_preserved": False,
        "labels_recomputed": "true labels recomputed on transformed answer distribution; oracle prediction",
        "identity_anchor": "a=1 restores the raw workload (aligns with fig7 p=0.85, k=5)",
        "invariant_note": "equal mean total tokens does NOT imply equal effective load (stage heterogeneity)",
    }

    npz_path, meta_path = save_results(
        plot_dir, str(args.out_prefix), arrays, cfg,
        vars(args), config_path=str(cfg_path), extra_meta=extra_meta,
    )
    print(f"saved data: {npz_path}")
    print(f"saved meta: {meta_path}")
    print(f"next: python -m scripts.draw.draw_fig10 --data \"{npz_path}\"")


if __name__ == "__main__":
    main()
