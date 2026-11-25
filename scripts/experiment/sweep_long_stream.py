"""fig14: 长请求流下的负载—构成实验（Tier1）。

背景（由 fig11 / fig13 引出）:
    fig11 在固定 λ=32、N=5000 下发现 prefill/decode 构成（逐请求总 token 守恒）会显著
    改变等待与提升比例；fig13 把结论迁移到不同 λ，发现"提升比例随负载升高"。
    但两者都只跑了 N=5000。本实验直接检查这些结论是否**依赖运行长度**：

        a ∈ {0.5, 1, 2, 4} × λ ∈ {26,30,32,34,36,38,40} × N ∈ {5000, 20000, 100000}。

研究问题:
1. λ=38 等重载点的等待时间很高——增加 N 后是进入相对稳定区间，还是随请求流长度持续
   增长（即当前结果主要是「积压随时间增长」而非稳态）？
2. 构成效应与 promotion 比例是否随 N 改变？
3. 机制（旁路统计）上，提升来自「有 aged-long 候选」还是「资源阻塞」？

关键设计（前缀一致性）:
    同一 seed 下，先生成**参考基础流**（长度 ``N_ref = max(max(lengths), --label-reference-n)``），
    再截取前缀作为较短 N 的请求流。由于 ``generate_heapq`` 顺序生成（固定 seed），前 k 条
    请求与 N=k 逐位一致；逐请求守恒变换 P_i'(a) 只依赖单条请求自身，故前缀一致。标签使用
    **共同的分类阈值**（取自 ``N_ref`` 变换后 answer 分布的 p 分位数），并记录各前缀实际
    short 比例（pi_s）。固定 ``--label-reference-n`` 后，分阶段补跑更长 N 时已跑短流的
    真标签保持不变。

窗口 / 积压双口径:
    不只看全程 mean/P99。按「到达序号」与「时间」两种口径各切 W 个窗口，记录窗口内
    mean/P99、到达数、完成数、积压数、等待队列长度、promotion 比例、phi_Q；并记录
    「到达停止时未完成数」与「最后清空队列所需时间（drain time）」。时间窗口覆盖
    ``[0, t_last_arrival]``（**不含排空阶段**，排空由 drain_time 单独记录），以区分
    「稳定运行」与「过载积压」。窗口边界随结果一并保存（``*_edges_runs``）。

不变量声明:
    与 fig11 相同——逐请求总 token 守恒并不保证实际负载相同。本实验用于检验结论对运行
    长度的敏感性，不等价于"严格等负载"。

积压/提升口径:
    waiting_at_arrival_stop   : 尚未开始服务数（start > t_last_arrival）
    unfinished_at_arrival_stop: 尚未完成数（end > t_last_arrival），与窗口 backlog 一致
    promotion_ratio = N_prom / N_all；phi_Q = N_prom / N_pred_long（同 fig12）

用法:
    python -m scripts.experiment.sweep_long_stream
    python -m scripts.experiment.sweep_long_stream \\
        --composition-scales "0.5,1,2,4" --lambdas "26,32,38" --lengths "5000,20000" \\
        --label-reference-n 100000 --seed-count 20 --plot-dir scripts/plot/fig14
    # Tier2 机制 k 网格（固定 N，QBA 扫 quota_k）
    python -m scripts.experiment.sweep_long_stream \\
        --composition-scales "0.5,1,2,4" --lambdas "26,32,38" --lengths "20000" \\
        --mech-ks "3,5,9" --mech-n 20000 --label-reference-n 100000 --seed-count 100
    python -m scripts.experiment.sweep_long_stream --smoke   # 极小规模快速自检

输出:
    plot/fig14_data.npz + plot/fig14_meta.json
    - Tier1 主路径：各策略的等待/资源 vs (a, λ, N) + 双口径窗口（含边界） + 积压标量
    - Tier2 机制：mech_*（N 依赖）与 mechk_*（固定 N，扫 quota_k）
"""

from __future__ import annotations

import argparse
import heapq
import os
from pathlib import Path

import numpy as np

from llm_in_queuing import simulator
from llm_in_queuing.config import load_config
from llm_in_queuing.metrics import mechanism_metrics
from llm_in_queuing.types import Request
from ._common import (
    progress_update,
    progress_done,
    save_results,
)
from .sweep_request_composition import (
    _extract_metrics,
    _parse_composition_scales,
    _METRICS_KEYS,
    _POLICY_DIAG_KEYS,
    _PROMPT_LOWER_BOUND,
    _ANSWER_LOWER_BOUND,
)
from .sweep_composition_load import _parse_lambdas

# ── 输出键定义 ────────────────────────────────────────────────

# 每个 (seed, a, λ) 单元、基于 N_ref 的变换诊断（与策略、N 无关）
_LONG_TRANSFORM_KEYS = [
    "prompt_mean", "answer_mean", "token_ratio",
    "total_mean", "total_var", "max_request_total_error",
    "lower_bound_fraction", "upper_bound_fraction",
    "label_threshold",
]

# 每个 (seed, a, λ, N) 单元的前缀诊断（与策略无关）
_PREFIX_DIAG_KEYS = ["pi_s", "n_short"]

# 每个 (seed, a, λ, N, policy) 单元的串长 / 积压标量
# waiting/unfinished 两种口径明确区分；promotion_ratio 与 phi_Q 分母不同（见 _scalar_stream_diag）
_STREAM_KEYS = [
    "t_last_arrival", "t_last_completion", "drain_time",
    "waiting_at_arrival_stop", "unfinished_at_arrival_stop",
    "unfinished_fraction", "promotion_ratio", "phi_Q",
]

# 窗口指标（两种口径共用；不适用的键填 NaN）
_WINDOW_KEYS = [
    "wait_mean", "wait_p99", "count",
    "arrivals", "completions", "backlog",
    "queue_len_mean", "promotion_ratio", "phi_Q",
]

# 旁路机制统计（仅 QBA 有意义；其余策略近似 0）
_MECH_KEYS = [
    "mech_debt_episode_count", "mech_debt_episode_mean_duration",
    "mech_debt_episode_p99_duration", "mech_debt_episode_max_duration",
    "mech_debt_unresolved_at_end", "mech_debt_unresolved_duration",
    "mech_promotion_success_count",
    "mech_debt_batch_no_eligible", "mech_debt_batch_candidate_blocked",
    "mech_debt_batch_success", "mech_debt_batch_kv_gate_closed",
    "mech_debt_batch_time_no_eligible", "mech_debt_batch_time_candidate_blocked",
    "mech_debt_batch_time_success",
    "mech_eta_eligibility", "mech_eta_feasibility",
    "mech_long_admit_normal", "mech_long_admit_promoted",
    "mech_long_admit_eligible_at_entry",
    "mech_budget_break_count", "mech_kv_break_count",
    "mech_concurrency_break_count", "mech_continuation_break_count",
    "mech_prefill_token_sum", "mech_decode_token_sum",
    "mech_continuation_count", "mech_batch_count",
]

# Tier2 机制 k 网格：固定 N 下扫 quota_k（QBA 为主）。在 _MECH_KEYS 基础上补等待/提升指标。
_MECHK_KEYS = _MECH_KEYS + ["mean", "p99", "promotion_ratio", "phi_Q", "promoted_total"]
_MECHK_POLICY = "QBA"

_POLICY_PREFIX = [("FCFS", "fcfs"), ("PSF", "psf"), ("AA", "psfa"), ("QBA", "psfaq")]
_WINDOW_MODES = ("index", "time")


def _parse_lengths(raw: str) -> np.ndarray:
    """解析请求流长度列表：正整数、无重复，按升序返回（严格整数解析）。"""
    parts = [s.strip() for s in str(raw).split(",") if s.strip()]
    if not parts:
        raise ValueError("--lengths 不能为空")
    vals: list[int] = []
    for s in parts:
        try:
            v = int(s)
        except ValueError as exc:
            raise ValueError(f"非法请求流长度: {s!r}（必须为整数）") from exc
        vals.append(v)
    for v in vals:
        if v <= 0:
            raise ValueError(f"非法请求流长度: {v}（必须 > 0）")
    if len(set(vals)) != len(vals):
        raise ValueError(f"--lengths 存在重复值: {parts}")
    return np.asarray(sorted(vals), dtype=np.int64)


def _parse_ks(raw: str) -> np.ndarray:
    """解析 quota_k 网格：正整数、无重复，按升序返回（严格整数解析）。"""
    parts = [s.strip() for s in str(raw).split(",") if s.strip()]
    if not parts:
        raise ValueError("--mech-ks 不能为空")
    vals: list[int] = []
    for s in parts:
        try:
            v = int(s)
        except ValueError as exc:
            raise ValueError(f"非法 quota_k: {s!r}（必须为整数）") from exc
        if v <= 0:
            raise ValueError(f"非法 quota_k: {v}（必须 > 0）")
        vals.append(v)
    if len(set(vals)) != len(vals):
        raise ValueError(f"--mech-ks 存在重复值: {parts}")
    return np.asarray(sorted(vals), dtype=np.int64)


def _index_windows(n: int, w: int) -> np.ndarray:
    """按到达序号把 [0, n) 均分成 w 段的边界（长度 w+1）。"""
    return np.linspace(0, n, w + 1).round().astype(np.int64)


def _time_windows(t_end: float, w: int) -> np.ndarray:
    """把 [0, t_end] 均分成 w 段的时间边界（长度 w+1）。"""
    if not np.isfinite(t_end) or t_end <= 0.0:
        return np.linspace(0.0, 1.0, w + 1)
    return np.linspace(0.0, float(t_end), w + 1)


def _transform_lengths(base_reqs: list[Request], scale: float, cfg: dict):
    """对完整基础流做逐请求总 token 守恒变换，返回 (p_new, d_new, threshold)。

    与 fig11 的 ``_apply_request_composition`` 同一公式，但只算数组、不就地写回，
    以便对同一变换结果截取不同长度的前缀。``threshold`` 取自 **完整流**（N_max）变换后
    answer 分布的分位数，作为所有前缀**共同的分类阈值**。
    """
    if not base_reqs:
        raise ValueError("基础请求流为空")
    p0 = np.array([int(r.prompt_len) for r in base_reqs], dtype=np.float64)
    d0 = np.array([int(r.answer_len) for r in base_reqs], dtype=np.float64)
    total = p0 + d0
    p_cap = int(cfg.get("prompt_len_cap", 10 ** 9))
    a_cap = int(cfg.get("answer_len_cap", 10 ** 9))
    p_lo = np.maximum(float(_PROMPT_LOWER_BOUND), total - float(a_cap))
    p_hi = np.minimum(float(p_cap), total - float(_ANSWER_LOWER_BOUND))

    if float(scale) == 1.0:
        p_new = p0.copy()
    else:
        a = float(scale)
        target = total * (a * p0) / (a * p0 + d0)
        p_new = np.minimum(p_hi, np.maximum(p_lo, np.round(target)))
    p_new = p_new.astype(np.int64)
    d_new = (total - p_new.astype(np.float64)).astype(np.int64)
    threshold = float(np.quantile(d_new.astype(np.float64),
                                  float(cfg["label_threshold_percentail"])))
    return p_new, d_new, threshold


def _make_prefix_generator(base_arrival, p_new, d_new, ids, n_take, threshold):
    """返回一个生成器：每次调用重建「前 n_take 条」请求堆（每策略独立、互不污染）。"""
    n_take = int(n_take)

    def _generate(cfg: dict):
        heap: list = []
        for i in range(n_take):
            d = int(d_new[i])
            lab = "short" if d <= float(threshold) else "long"
            req = Request(
                id=int(ids[i]),
                arrival_time=float(base_arrival[i]),
                prompt_len=int(p_new[i]),
                answer_len=d,
                prefill_res_token=int(p_new[i]),
                true_label=lab,
                predict_label=lab,
            )
            heap.append((req.arrival_time, req.id, req))
        heapq.heapify(heap)
        return heap

    return _generate


def _prefix_short_ratio(d_new, n_take, threshold) -> tuple[int, float]:
    """按共同阈值统计前缀 [0, n_take) 中的 short 数量与比例。"""
    d = d_new[: int(n_take)].astype(np.float64)
    n_short = int(np.sum(d <= float(threshold)))
    return n_short, (float(n_short) / float(n_take) if n_take > 0 else float("nan"))


def _scalar_stream_diag(finished: list[Request]) -> dict:
    """串长与积压标量。

    积压口径（明确区分）:
        waiting_at_arrival_stop   : 到达结束时**尚未开始服务**（start > t_last_arrival）
        unfinished_at_arrival_stop: 到达结束时**尚未完成**（end > t_last_arrival），
            与时间窗口 backlog（累计到达−累计完成）一致，用于「积压增长」主图
    提升口径（分母不同，不可直接比较）:
        promotion_ratio = N_prom / N_all
        phi_Q           = N_prom / N_pred_long（与 fig12 的 φ_Q 同口径）
    """
    if not finished:
        return {k: float("nan") for k in _STREAM_KEYS}
    arr = np.array([float(r.arrival_time) for r in finished], dtype=np.float64)
    end = np.array([float(r.end_service_time) for r in finished], dtype=np.float64)
    start = np.array([float(r.start_service_time) for r in finished], dtype=np.float64)
    promoted = np.array([bool(getattr(r, "promoted", False)) for r in finished])
    pred_long = np.array([getattr(r, "predict_label", "") == "long" for r in finished])
    t_last_arrival = float(arr.max())
    t_last_completion = float(end.max())
    waiting = int(np.sum(start > t_last_arrival))
    unfinished = int(np.sum(end > t_last_arrival))
    n_all = float(len(finished))
    n_prom = int(promoted.sum())
    n_pred_long = int(pred_long.sum())
    return {
        "t_last_arrival": t_last_arrival,
        "t_last_completion": t_last_completion,
        "drain_time": t_last_completion - t_last_arrival,
        "waiting_at_arrival_stop": float(waiting),
        "unfinished_at_arrival_stop": float(unfinished),
        "unfinished_fraction": float(unfinished) / n_all,
        "promotion_ratio": float(n_prom) / n_all,
        "phi_Q": (float(n_prom) / float(n_pred_long)
                  if n_pred_long > 0 else float("nan")),
    }


def _window_arrays(finished: list[Request], snaps: list[dict],
                   w: int, mode: str) -> tuple[dict, np.ndarray]:
    """把完成请求切成 w 个窗口，返回 ``(每窗口 _WINDOW_KEYS 数组, 窗口边界)``。

    mode="index": 按到达序号均分，边界为请求序号（与策略/负载无关）。
    mode="time" : 按 ``[0, t_last_arrival]`` 均分（**不含排空阶段**），边界为时间；
        排空阶段由标量 ``drain_time`` 单独记录。``queue_len_mean`` 为按 batch 快照
        采样的算术平均，非严格时间加权平均。
    """
    out = {k: np.full(w, np.nan, dtype=np.float64) for k in _WINDOW_KEYS}
    edges = np.full(w + 1, np.nan, dtype=np.float64)
    if not finished:
        return out, edges

    arr = np.array([float(r.arrival_time) for r in finished], dtype=np.float64)
    end = np.array([float(r.end_service_time) for r in finished], dtype=np.float64)
    wait = np.array([float(r.waiting_time) for r in finished], dtype=np.float64)
    promoted = np.array([bool(getattr(r, "promoted", False)) for r in finished])
    pred_long = np.array([getattr(r, "predict_label", "") == "long" for r in finished])

    if mode == "index":
        order = np.argsort(arr, kind="stable")
        e = _index_windows(len(order), w)
        edges[:] = e.astype(np.float64)
        for wi in range(w):
            lo, hi = int(e[wi]), int(e[wi + 1])
            idx = order[lo:hi]
            if idx.size == 0:
                continue
            out["count"][wi] = float(idx.size)
            out["arrivals"][wi] = float(idx.size)
            out["completions"][wi] = float(idx.size)
            out["backlog"][wi] = 0.0
            out["wait_mean"][wi] = float(wait[idx].mean())
            out["wait_p99"][wi] = float(np.percentile(wait[idx], 99))
            out["promotion_ratio"][wi] = float(promoted[idx].mean())
            pl = pred_long[idx]
            if pl.any():
                out["phi_Q"][wi] = float(promoted[idx][pl].mean())
        return out, edges

    # mode == "time"：主窗口覆盖 [0, t_last_arrival]，排空阶段不混入
    t_end = float(arr.max())
    edges[:] = _time_windows(t_end, w)
    snap_t = np.array([float(b.get("sim_time", 0.0)) for b in snaps], dtype=np.float64)
    snap_q = np.array([float(b.get("wait_queue_len", 0)) for b in snaps], dtype=np.float64)
    cum_arr = 0
    cum_cmp = 0
    for wi in range(w):
        lo, hi = float(edges[wi]), float(edges[wi + 1])
        last = (wi == w - 1)
        if last:
            a_mask = (arr >= lo) & (arr <= hi)
            c_mask = (end >= lo) & (end <= hi)
            q_mask = (snap_t >= lo) & (snap_t <= hi)
        else:
            a_mask = (arr >= lo) & (arr < hi)
            c_mask = (end >= lo) & (end < hi)
            q_mask = (snap_t >= lo) & (snap_t < hi)
        n_arr = int(a_mask.sum())
        n_cmp = int(c_mask.sum())
        cum_arr += n_arr
        cum_cmp += n_cmp
        out["arrivals"][wi] = float(n_arr)
        out["completions"][wi] = float(n_cmp)
        out["backlog"][wi] = float(cum_arr - cum_cmp)
        out["count"][wi] = float(n_arr)
        if n_arr > 0:
            out["wait_mean"][wi] = float(wait[a_mask].mean())
            out["wait_p99"][wi] = float(np.percentile(wait[a_mask], 99))
            out["promotion_ratio"][wi] = float(promoted[a_mask].mean())
            pl = pred_long[a_mask]
            if pl.any():
                out["phi_Q"][wi] = float(promoted[a_mask][pl].mean())
        if q_mask.any():
            out["queue_len_mean"][wi] = float(snap_q[q_mask].mean())
    return out, edges


def _run_long_stream_sweep(
    cfg: dict,
    scales: np.ndarray,
    loads: np.ndarray,
    lengths: np.ndarray,
    policies: list[str],
    seeds: list[int],
    n_windows: int,
    quota_ks: np.ndarray | None = None,
    mech_n: int | None = None,
    label_reference_n: int | None = None,
) -> tuple[dict, dict, dict, dict, dict, dict, dict, dict]:
    """在 (a, λ, N) 网格上扫参多个策略（逐请求总 token 守恒，前缀共享阈值）。

    Args:
        quota_ks: Tier2 机制 k 网格的 quota_k 列表。若给定，则额外在固定 N=``mech_n`` 下
            对 QBA 扫 k，输出 ``mechk``（形状 [n_seeds, n_a, n_l, n_k]）。
        mech_n: 机制 k 网格固定使用的请求流长度（须 ≤ max(lengths)）。
        label_reference_n: 分类阈值参考长度。基础流按 ``N_ref = max(max(lengths),
            label_reference_n)`` 生成，阈值取自该 **完整参考流** 变换后 answer 分布；实际
            仿真仍只跑各指定前缀。这样分阶段补跑（如先 5k/20k、后补 100k）时，已跑短流的
            真标签保持稳定。为 None 时退化为 ``N_ref = max(lengths)``（旧行为）。

    Returns:
        (rows, policy_diag, stream, mech, transform_diag, prefix_diag, window, mechk)
        rows[p][metric]   : [n_seeds, n_a, n_l, n_N]
        policy_diag[p][k] : [n_seeds, n_a, n_l, n_N]
        stream[p][k]      : [n_seeds, n_a, n_l, n_N]
        mech[p][k]        : [n_seeds, n_a, n_l, n_N]
        transform_diag[k] : [n_seeds, n_a, n_l]        （N_ref 基准，含 label_threshold）
        prefix_diag[k]    : [n_seeds, n_a, n_l, n_N]
        window[p][mode][k]: [n_seeds, n_a, n_l, n_N, n_windows]
        window[p][mode]["edges"]: [n_seeds, n_a, n_l, n_N, n_windows+1]（实际窗口边界）
        mechk[k]          : [n_seeds, n_a, n_l, n_k]    （QBA，固定 N=mech_n）
    """
    n_seeds, n_a, n_l, n_N = len(seeds), len(scales), len(loads), len(lengths)
    w = int(n_windows)
    shape = (n_seeds, n_a, n_l, n_N)
    shape_tr = (n_seeds, n_a, n_l)
    shape_w = (n_seeds, n_a, n_l, n_N, w)
    shape_we = (n_seeds, n_a, n_l, n_N, w + 1)
    n_max = int(np.max(lengths))
    # 分类阈值的参考长度：分阶段补跑时固定不变，避免已跑短流的真标签被改写
    n_ref = max(n_max, int(label_reference_n) if label_reference_n is not None else 0)

    run_mechk = quota_ks is not None and mech_n is not None
    ks_arr = np.asarray(quota_ks, dtype=np.int64) if run_mechk else np.asarray([], dtype=np.int64)
    n_k = int(ks_arr.size)
    shape_mk = (n_seeds, n_a, n_l, n_k)

    rows = {p: {k: np.full(shape, np.nan) for k in _METRICS_KEYS} for p in policies}
    policy_diag = {p: {k: np.full(shape, np.nan) for k in _POLICY_DIAG_KEYS}
                   for p in policies}
    stream = {p: {k: np.full(shape, np.nan) for k in _STREAM_KEYS} for p in policies}
    mech = {p: {k: np.full(shape, np.nan) for k in _MECH_KEYS} for p in policies}
    transform_diag = {k: np.full(shape_tr, np.nan) for k in _LONG_TRANSFORM_KEYS}
    prefix_diag = {k: np.full(shape, np.nan) for k in _PREFIX_DIAG_KEYS}
    window = {
        p: {mode: {k: np.full(shape_w, np.nan) for k in _WINDOW_KEYS}
            for mode in _WINDOW_MODES}
        for p in policies
    }
    window_edge = {p: {mode: np.full(shape_we, np.nan) for mode in _WINDOW_MODES}
                   for p in policies}
    mechk = {k: np.full(shape_mk, np.nan) for k in _MECHK_KEYS}

    original_generate = simulator.generate_heapq
    total = n_seeds * n_l * n_a * n_N * len(policies)
    if run_mechk:
        total += n_seeds * n_l * n_a * n_k
    done = 0
    print(f"stage: SIM long-stream sweep runs={total} seeds={n_seeds} "
          f"a={n_a} lambda={n_l} N={n_N} policies={len(policies)} N_ref={n_ref}"
          + (f" mechk_k={ks_arr.tolist()} mech_N={int(mech_n)}" if run_mechk else ""))
    try:
        for si, seed in enumerate(seeds):
            for li, lam in enumerate(loads.tolist()):
                # 生成参考基础流（长度 N_ref ≥ N_max），后续所有 N 取自其前缀
                base_cfg = dict(cfg)
                base_cfg.update({"lambda_arrival": float(lam), "seed": int(seed),
                                 "admission_number": int(n_ref)})
                base_heap = original_generate(base_cfg)
                base_reqs = sorted((req for _, _, req in base_heap),
                                   key=lambda r: int(r.id))
                base_arrival = np.array([float(r.arrival_time) for r in base_reqs])
                base_ids = np.array([int(r.id) for r in base_reqs])
                base_total = (np.array([int(r.prompt_len) for r in base_reqs], dtype=np.float64)
                              + np.array([int(r.answer_len) for r in base_reqs], dtype=np.float64))

                for ai, a in enumerate(scales.tolist()):
                    scale = float(a)
                    p_new, d_new, threshold = _transform_lengths(base_reqs, scale, cfg)

                    # 变换诊断（基于完整流，与策略、N 无关）
                    realized = p_new.astype(np.float64) + d_new.astype(np.float64)
                    mean_p = float(p_new.mean())
                    mean_d = float(d_new.mean())
                    transform_diag["prompt_mean"][si, ai, li] = mean_p
                    transform_diag["answer_mean"][si, ai, li] = mean_d
                    transform_diag["token_ratio"][si, ai, li] = (
                        mean_p / mean_d if mean_d > 0 else float("nan"))
                    transform_diag["total_mean"][si, ai, li] = float(realized.mean())
                    transform_diag["total_var"][si, ai, li] = float(realized.var())
                    transform_diag["max_request_total_error"][si, ai, li] = float(
                        np.max(np.abs(realized - base_total)))
                    transform_diag["lower_bound_fraction"][si, ai, li] = float(
                        np.mean(p_new.astype(np.float64)
                                <= np.maximum(float(_PROMPT_LOWER_BOUND),
                                              base_total - float(cfg["answer_len_cap"])) + 1e-9))
                    transform_diag["upper_bound_fraction"][si, ai, li] = float(
                        np.mean(p_new.astype(np.float64)
                                >= np.minimum(float(cfg["prompt_len_cap"]),
                                              base_total - float(_ANSWER_LOWER_BOUND)) - 1e-9))
                    # 实际分类阈值（供分阶段补跑合并校验）
                    transform_diag["label_threshold"][si, ai, li] = float(threshold)

                    for ni, n_take in enumerate(lengths.tolist()):
                        n_take = int(n_take)
                        n_short, pi_s = _prefix_short_ratio(d_new, n_take, threshold)
                        prefix_diag["pi_s"][si, ai, li, ni] = pi_s
                        prefix_diag["n_short"][si, ai, li, ni] = float(n_short)

                        simulator.generate_heapq = _make_prefix_generator(
                            base_arrival, p_new, d_new, base_ids, n_take, threshold)
                        for pol in policies:
                            local = dict(cfg)
                            local.update({
                                "lambda_arrival": float(lam),
                                "schedule_method": pol,
                                "seed": int(seed),
                                "admission_number": int(n_take),
                            })
                            try:
                                metrics, state = simulator.run(local, return_state=True)
                            except Exception as exc:  # 附实验上下文
                                raise RuntimeError(
                                    f"Experiment failed: seed={seed} a={a} "
                                    f"lambda={lam} N={n_take} policy={pol}"
                                ) from exc
                            s = _extract_metrics(metrics)
                            for mk in _METRICS_KEYS:
                                rows[pol][mk][si, ai, li, ni] = float(s[mk])
                            for dk in _POLICY_DIAG_KEYS:
                                policy_diag[pol][dk][si, ai, li, ni] = float(s[dk])
                            mm = mechanism_metrics(state)
                            for k in _MECH_KEYS:
                                mech[pol][k][si, ai, li, ni] = float(
                                    mm.get(k, float("nan")))
                            for k, v in _scalar_stream_diag(
                                    state.finished_queue).items():
                                stream[pol][k][si, ai, li, ni] = float(v)
                            for mode in _WINDOW_MODES:
                                wa, edges = _window_arrays(state.finished_queue,
                                                           state.batch_list, w, mode)
                                for wk in _WINDOW_KEYS:
                                    window[pol][mode][wk][si, ai, li, ni] = wa[wk]
                                window_edge[pol][mode][si, ai, li, ni, :] = edges
                            done += 1
                            progress_update(done, total, "SIM long-stream")
                        simulator.generate_heapq = original_generate

                    # ── Tier2 机制 k 网格：固定 N=mech_n，QBA 扫 quota_k ──
                    if run_mechk:
                        simulator.generate_heapq = _make_prefix_generator(
                            base_arrival, p_new, d_new, base_ids, int(mech_n), threshold)
                        for ki, kv in enumerate(ks_arr.tolist()):
                            local = dict(cfg)
                            local.update({
                                "lambda_arrival": float(lam),
                                "schedule_method": _MECHK_POLICY,
                                "seed": int(seed),
                                "admission_number": int(mech_n),
                                "quota_k": int(kv),
                            })
                            try:
                                metrics, state = simulator.run(local, return_state=True)
                            except Exception as exc:
                                raise RuntimeError(
                                    f"Experiment failed: seed={seed} a={a} "
                                    f"lambda={lam} N={int(mech_n)} k={kv} "
                                    f"policy={_MECHK_POLICY}"
                                ) from exc
                            merged = dict(_extract_metrics(metrics))
                            merged.update(mechanism_metrics(state))
                            merged.update(_scalar_stream_diag(state.finished_queue))
                            for mk in _MECHK_KEYS:
                                mechk[mk][si, ai, li, ki] = float(
                                    merged.get(mk, float("nan")))
                            done += 1
                            progress_update(done, total, "SIM long-stream")
                        simulator.generate_heapq = original_generate
    finally:
        simulator.generate_heapq = original_generate
    progress_done("SIM long-stream", total)
    # 把实际窗口边界并入 window（键 "edges"，长度 w+1）
    for p in policies:
        for mode in _WINDOW_MODES:
            window[p][mode]["edges"] = window_edge[p][mode]
    return (rows, policy_diag, stream, mech, transform_diag, prefix_diag, window,
            mechk)


def main():
    parser = argparse.ArgumentParser(
        description="fig14: 长请求流下的负载—构成实验（Tier1）")
    parser.add_argument("--composition-scales", type=str, default="0.5,1,2,4")
    parser.add_argument("--lambdas", type=str, default="26,30,32,34,36,38,40")
    parser.add_argument("--lengths", type=str, default="5000,20000,100000")
    parser.add_argument("--windows", type=int, default=8,
                        help="窗口数（序号 / 时间两种口径共用）")
    parser.add_argument("--gamma", type=float, default=0.80)
    parser.add_argument("--quota-k", type=int, default=5,
                        help="Tier1 主路径固定使用的 quota_k")
    parser.add_argument("--mech-ks", type=str, default="5",
                        help="Tier2 机制 k 网格的 quota_k 列表（QBA，固定 N=mech-n）")
    parser.add_argument("--mech-n", type=int, default=None,
                        help="机制 k 网格固定的请求流长度（默认取 max(lengths)）")
    parser.add_argument("--label-reference-n", type=int, default=None,
                        help="分类阈值参考长度；基础流按 max(max(lengths), 本值) 生成，"
                             "阈值取自完整参考流，使分阶段补跑时已跑短流标签稳定")
    parser.add_argument("--plot-dir", type=str, default=os.path.join(
        os.path.dirname(__file__), "..", "plot"))
    parser.add_argument("--out-prefix", type=str, default="fig14")
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--smoke", action="store_true",
                        help="极小规模自检（覆盖默认网格，仅用于流程验证）")
    args = parser.parse_args()

    cfg_path = Path(__file__).resolve().parent / "configs" / "sweep_long_stream_base.yaml"
    cfg = load_config(cfg_path)
    cfg["gamma"] = float(args.gamma)
    cfg["quota_k"] = int(args.quota_k)
    cfg["mech_diag_enabled"] = True

    if args.smoke:
        args.composition_scales = "0.5,1"
        args.lambdas = "30,32"
        args.lengths = "120,240,480"
        args.seed_count = 2
        args.windows = 4
        args.mech_ks = "3,5"
        args.label_reference_n = 960

    plot_dir = os.path.abspath(args.plot_dir)
    scales = _parse_composition_scales(args.composition_scales)
    loads = _parse_lambdas(args.lambdas)
    lengths = _parse_lengths(args.lengths)
    quota_ks = _parse_ks(args.mech_ks)
    mech_n = int(args.mech_n) if args.mech_n is not None else int(np.max(lengths))
    if mech_n > int(np.max(lengths)):
        raise ValueError(
            f"--mech-n={mech_n} 超过 max(lengths)={int(np.max(lengths))}；"
            f"机制网格的请求流必须是基础流的前缀")
    seeds = [int(args.seed_start) + i for i in range(int(args.seed_count))]
    sweep_policies = ["FCFS", "PSF", "AA", "QBA"]

    print(f"stage: SIM long-stream scales={scales.tolist()} lambdas={loads.tolist()} "
          f"lengths={lengths.tolist()} gamma={cfg['gamma']} quota_k={cfg['quota_k']} "
          f"mech_ks={quota_ks.tolist()} mech_n={mech_n} "
          f"label_reference_n={args.label_reference_n} "
          f"windows={args.windows} repeats={len(seeds)}")

    (rows, policy_diag, stream, mech, transform_diag, prefix_diag, window,
     mechk) = _run_long_stream_sweep(
        cfg, scales, loads, lengths, sweep_policies, seeds, int(args.windows),
        quota_ks=quota_ks, mech_n=mech_n, label_reference_n=args.label_reference_n)

    arrays = {
        "composition_scales": np.asarray(scales, dtype=np.float64),
        "lambdas": np.asarray(loads, dtype=np.float64),
        "lengths": np.asarray(lengths, dtype=np.int64),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "n_windows": np.asarray([int(args.windows)], dtype=np.int64),
        "mech_ks": np.asarray(quota_ks, dtype=np.int64),
        "mech_n": np.asarray([int(mech_n)], dtype=np.int64),
    }
    for k in _LONG_TRANSFORM_KEYS:
        arrays[f"transform_{k}"] = transform_diag[k]
    for k in _PREFIX_DIAG_KEYS:
        arrays[f"prefix_{k}"] = prefix_diag[k]

    for p, prefix in _POLICY_PREFIX:
        for mk in _METRICS_KEYS:
            arrays[f"{prefix}_{mk}_runs"] = rows[p][mk]
        for dk in _POLICY_DIAG_KEYS:
            arrays[f"{prefix}_{dk}_runs"] = policy_diag[p][dk]
        for sk in _STREAM_KEYS:
            arrays[f"{prefix}_{sk}_runs"] = stream[p][sk]
        for mk in _MECH_KEYS:
            arrays[f"{prefix}_{mk}_runs"] = mech[p][mk]
        for mode in _WINDOW_MODES:
            for wk in _WINDOW_KEYS:
                arrays[f"{prefix}_{mode}_{wk}_runs"] = window[p][mode][wk]
            arrays[f"{prefix}_{mode}_edges_runs"] = window[p][mode]["edges"]

    # Tier2 机制 k 网格（QBA，固定 N=mech_n）
    for mk in _MECHK_KEYS:
        arrays[f"mechk_{mk}"] = mechk[mk]

    extra_meta = {
        "experiment": "long_stream_load_composition",
        "sweep_policies": sweep_policies,
        "composition_scales": [float(v) for v in scales.tolist()],
        "lambdas": [float(v) for v in loads.tolist()],
        "lengths": [int(v) for v in lengths.tolist()],
        "n_windows": int(args.windows),
        "label_threshold_percentail": float(cfg["label_threshold_percentail"]),
        "normalization": "exact per-request total-token preservation (fig11 transform)",
        "label_reference": {
            "label_reference_n": (int(args.label_reference_n)
                                  if args.label_reference_n is not None else None),
            "n_ref": int(max(int(np.max(lengths)),
                             int(args.label_reference_n)
                             if args.label_reference_n is not None else 0)),
            "note": ("label threshold taken from the full N_ref stream; staged "
                     "top-ups of longer N keep shorter-stream labels unchanged"),
        },
        "prefix_consistency": (
            "base stream generated once per (seed,lambda) at length N_ref; every N is "
            "its prefix; label threshold taken from the full N_ref transformed answers"
        ),
        "stream_metric_scope": {
            "waiting_at_arrival_stop": "start_service_time > t_last_arrival",
            "unfinished_at_arrival_stop": "end_service_time > t_last_arrival (matches window backlog)",
            "promotion_ratio": "N_prom / N_all",
            "phi_Q": "N_prom / N_pred_long (matches fig12 phi_Q)",
        },
        "mech_diag_enabled": True,
        "mech_k_grid": {
            "policy": _MECHK_POLICY,
            "quota_ks": [int(v) for v in quota_ks.tolist()],
            "mech_n": int(mech_n),
            "note": "bypass mechanism stats swept over quota_k at fixed N (prefix)",
        },
        "invariant_note": (
            "per-request total tokens are preserved; N differences come only from "
            "run length; time windows cover [0, t_last_arrival] (drain separate); "
            "debt batch decomposition includes the kv_gate_closed branch"
        ),
    }

    npz_path, meta_path = save_results(
        plot_dir, str(args.out_prefix), arrays, cfg,
        vars(args), config_path=str(cfg_path), extra_meta=extra_meta,
    )
    print(f"saved data: {npz_path}")
    print(f"saved meta: {meta_path}")
    print(f"next: python -m scripts.draw.draw_fig14 --data \"{npz_path}\"")


if __name__ == "__main__":
    main()
