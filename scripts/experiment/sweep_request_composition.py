"""fig11: 逐请求总 token 守恒的构成扫描 — fig10 的控制实验。

背景（由 fig10 引出）:
    fig10 只固定了**总体平均**：E[P'+D'] ≈ E[P+D]，但各条请求的总长度及其方差仍会
    改变，因此无法排除"总 token 分布改变"是性能差异的来源。本实验改为对**每条请求**
    固定总长度：

        P_i' + D_i' = P_i + D_i = T_i,

    于是同一 seed 下各倍率共享完全相同的到达时间、总 token 序列 T_i（含均值/方差/
    分位数），只改变每条请求内部 prefill/decode 的分配。

研究问题:
    在逐请求总 token 需求保持不变的情况下，改变 prefill/decode 分配，是否仍然显著
    改变调度性能？若仍有明显变化，可排除"总 token 分布改变"作为唯一解释；但不能直接
    归因于某个特定机制。

变换规则（对构成倍率 a > 0，记 T_i = P_i + D_i）:
    未取整的目标 prompt:  P~_i(a) = T_i * (a*P_i) / (a*P_i + D_i)
    联合可行区间（同时满足 prompt/answer 的上下限）:
        L_i = max(P_min, T_i - D_max)
        U_i = min(P_max, T_i - D_min)
    P_min = 5, D_min = 1, P_max = prompt_len_cap, D_max = answer_len_cap。
    取整并夹到可行区间:
        P_i'(a) = min(U_i, max(L_i, round(P~_i(a))))
        D_i'(a) = T_i - P_i'(a)
    这样可同时保证：prompt/answer 均在合法范围内；每条请求总 token **严格相等**；
    不需要 fig10 的迭代归一化；不引入随机数。

设计要点:
- a=1 时直接保留原始整数 P_i、D_i，逐字段还原原始 workload，可与 fig7/fig10 的
  (p=0.85, k=5, 同 seed) 结果对位（验收锚点）。
- **answer 长度随 a 改变**（相对 fig8 的明确设计变化）：要同时满足"逐请求总 token 固定"
  与"构成变化"，增加 prompt 就必须减少该请求的 decode，反之亦然。
- 标签在变换后按**新的 answer 分布**重新计算（p = label_threshold_percentail）。本实验
  要求 oracle 配置（TN/TP/FN/FP 全为 0），标签采用确定性重算（不调用会消耗全局随机数的
  ``_assign_labels``）。与 fig10 不同：fig11 的缩放因请求而异，answer 排序可能变化，
  因此标签变化比例可能更高——本轮**不设定**"标签变化必须很小"的通过标准，只记录。
- 只改请求的 P/D 与标签，不改到达时间与 id；**不消耗任何额外随机数**（由单测保证）。
- 实现方式与 fig10 一致：模块级临时替换 ``simulator.generate_heapq``，每个
  (seed, scale) 单元替换一次、四个策略共用同一份变换，跑完在 ``finally`` 恢复原函数。
  注意这是模块级替换，同一进程内不要并行跑不同倍率（云端通过独立子进程按 seed 并行）。

不变量声明:
    逐请求总 token 相同**并不保证**实际负载相同（prefill 可用批次剩余预算、decode 占多轮
    服务、KV 驻留与阶段构成相关）。本实验应称为"逐请求总 token 守恒的 prefill/decode
    构成扫描"，不等价于"严格等负载"或"独立 R 扫描"。

用法:
    python -m scripts.experiment.sweep_request_composition
    python -m scripts.experiment.sweep_request_composition --composition-scales "0.25,0.5,1,2,4"
    python -m scripts.experiment.sweep_request_composition --seed-count 2 --sim-admission-number 300

输出:
    plot/fig11_data.npz + plot/fig11_meta.json
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

# 每条 (policy) 的等待指标（与 fig8/fig10 对齐，便于横向比较）
_METRICS_KEYS = [
    "mean", "p95", "p99",
    "ts_mean", "ts_p99",
    "tl_mean", "tl_p99",
    "actual_epsilon_SL", "actual_epsilon_LS",
]

# 每条 (policy) 的引擎资源诊断（复用 fig10 的键，不扩展引擎）
_POLICY_DIAG_KEYS = [
    "oom_prob", "throughput", "batch_util", "promoted_total",
]

# 与策略无关的 workload 诊断（同一 (seed, scale) 只记录一次）
_WORK_DIAG_KEYS = [
    "prompt_mean", "answer_mean", "token_ratio",
    "base_total_mean", "total_mean",
    "base_total_var", "total_var",
    "max_request_total_error",
    "lower_bound_fraction", "upper_bound_fraction",
    "pi_s", "label_flip_fraction",
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


def _apply_request_composition(
    heap: list,
    scale: float,
    cfg: dict,
    diag: dict | None = None,
    base_labels: dict | None = None,
) -> None:
    """就地按构成倍率 ``scale`` 调整每条请求的 P/D，严格保持 P_i + D_i 不变。

    先按 ``P~_i(a) = T_i * (a*P_i)/(a*P_i + D_i)`` 求未取整目标，再夹到"逐请求联合可行
    区间" [L_i, U_i]（L_i = max(P_min, T_i - D_max)，U_i = min(P_max, T_i - D_min)），
    answer 由 ``D_i' = T_i - P_i'`` 反推。变换后按新的 answer 分布重算真值/预测标签。
    所有统计写入 ``diag``。
    """
    reqs = [req for _, _, req in heap]
    n = len(reqs)
    if n == 0:
        return

    p_cap = int(cfg.get("prompt_len_cap", 10 ** 9))
    a_cap = int(cfg.get("answer_len_cap", 10 ** 9))
    p0 = np.array([int(r.prompt_len) for r in reqs], dtype=np.float64)
    d0 = np.array([int(r.answer_len) for r in reqs], dtype=np.float64)
    total = p0 + d0  # T_i：逐请求守恒目标

    # 逐请求联合可行区间（同时满足 prompt/answer 上下限）
    p_lo = np.maximum(float(_PROMPT_LOWER_BOUND), total - float(a_cap))
    p_hi = np.minimum(float(p_cap), total - float(_ANSWER_LOWER_BOUND))

    if float(scale) == 1.0:
        # a=1 直接保留原始整数，不引入任何取整/夹取，保证基线完全一致
        p_new = p0.copy()
        lower_adj = int(np.sum(p0 < p_lo))
        upper_adj = int(np.sum(p0 > p_hi))
    else:
        a = float(scale)
        target = total * (a * p0) / (a * p0 + d0)  # P~_i(a)
        rounded = np.round(target)
        p_new = np.minimum(p_hi, np.maximum(p_lo, rounded))
        lower_adj = int(np.sum(rounded < p_lo))
        upper_adj = int(np.sum(rounded > p_hi))

    p_new = p_new.astype(np.int64)
    d_new = (total - p_new.astype(np.float64)).astype(np.int64)

    # 写回请求（heap 持有同一批 Request 对象，就地更新；到达时间与 id 不动）
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
            f"fig11 标签重算假定 oracle（TN/TP/FN/FP 全为 0），实际为 TN/TP/FN/FP={tf}"
        )
    ans = d_new.astype(np.float64)
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
        realized_total = p_new.astype(np.float64) + d_new.astype(np.float64)
        diag["prompt_mean"] = mean_p
        diag["answer_mean"] = mean_d
        diag["token_ratio"] = (mean_p / mean_d) if mean_d > 0 else float("nan")
        diag["base_total_mean"] = float(total.mean())
        diag["total_mean"] = float(realized_total.mean())
        diag["base_total_var"] = float(total.var())
        diag["total_var"] = float(realized_total.var())
        diag["max_request_total_error"] = float(np.max(np.abs(realized_total - total)))
        diag["lower_bound_fraction"] = lower_adj / n
        diag["upper_bound_fraction"] = upper_adj / n
        diag["pi_s"] = n_short / n
        if base_labels is not None:
            flips = sum(1 for r in reqs if base_labels.get(r.id) != r.true_label)
            diag["label_flip_fraction"] = flips / n


def _make_request_composition_generator(original_generate, scale: float, diag: dict,
                                        base_labels: dict):
    """返回一个包装后的生成器：先产出基础请求堆，再做逐请求守恒的构成变换。"""
    def _request_composition_generate(cfg: dict):
        heap = original_generate(cfg)
        _apply_request_composition(heap, scale, cfg, diag, base_labels)
        return heap

    return _request_composition_generate


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


def _run_request_composition_sweep(
    cfg: dict,
    lam: float,
    scales: np.ndarray,
    policies: list[str],
    seeds: list[int],
) -> tuple[dict, dict, dict]:
    """按构成倍率扫参多个策略（逐请求总 token 守恒）。

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
    print(f"stage: SIM request-composition sweep runs={total} seeds={n_seeds} "
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
            simulator.generate_heapq = _make_request_composition_generator(
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
                    progress_update(done, total, "SIM request-composition sweep")
            finally:
                simulator.generate_heapq = original_generate
            # workload 诊断与策略无关，单元跑完后记录一次
            for dk in _WORK_DIAG_KEYS:
                work_diag[dk][si, li] = float(cell_diag.get(dk, float("nan")))
    progress_done("SIM request-composition sweep", total)
    return out, policy_diag, work_diag


def main():
    parser = argparse.ArgumentParser(
        description="fig11: 逐请求总 token 守恒的构成扫描")
    parser.add_argument("--composition-scales", type=str, default="0.25,0.5,1,2,4",
                        help="逗号分隔的构成倍率列表，例如 \"0.25,0.5,1,2,4\"")
    parser.add_argument("--reference-load", type=float, default=32.0)
    parser.add_argument("--gamma", type=float, default=0.80)
    parser.add_argument("--quota-k", type=int, default=5)
    parser.add_argument("--plot-dir", type=str, default=os.path.join(
        os.path.dirname(__file__), "..", "plot"))
    parser.add_argument("--out-prefix", type=str, default="fig11")
    parser.add_argument("--sim-admission-number", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--seed-count", type=int, default=100)
    args = parser.parse_args()

    # 加载基础配置
    cfg_path = Path(__file__).resolve().parent / "configs" / "sweep_request_composition_base.yaml"
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

    print(f"stage: SIM request-composition sweep scales={scales.tolist()} "
          f"lambda={reference_lambda} gamma={cfg['gamma']} quota_k={cfg['quota_k']} "
          f"repeats={len(seeds)}")
    rows, policy_diag, work_diag = _run_request_composition_sweep(
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
        "experiment": "per_request_total_preserving_composition",
        "sweep_policies": sweep_policies,
        "ref_policies": [],
        "composition_scales": np.asarray(scales, dtype=np.float64).tolist(),
        "normalization": "exact per-request total-token preservation",
        "normalization_formula": (
            "P~_i(a)=T_i*(a*P_i)/(a*P_i+D_i); "
            "L_i=max(Pmin,T_i-Dmax); U_i=min(Pmax,T_i-Dmin); "
            "P'_i=clip(round(P~_i),L_i,U_i); D'_i=T_i-P'_i"
        ),
        "answer_preserved": False,
        "labels_recomputed": "oracle quantile on transformed answers",
        "identity_anchor": "a=1",
        "invariant_note": (
            "per-request total tokens are preserved; the total-token distribution "
            "(mean/var/quantiles) is identical across scales, unlike fig10"
        ),
    }

    npz_path, meta_path = save_results(
        plot_dir, str(args.out_prefix), arrays, cfg,
        vars(args), config_path=str(cfg_path), extra_meta=extra_meta,
    )
    print(f"saved data: {npz_path}")
    print(f"saved meta: {meta_path}")
    print(f"next: python -m scripts.draw.draw_fig11 --data \"{npz_path}\"")


if __name__ == "__main__":
    main()
