"""fig15: 固定安全比例下的 KV 容量敏感性与调度折中实验。

背景（由 fig14 引出）:
    fig14 在固定 K=8192（安全比例 r=0.7）下研究了长请求流（N up to 100k）的负载—构成
    效应，发现重载点（λ=34/38）会出现持续积压。这些结论是否依赖 KV 容量这一资源约束？
    本实验把 ``KV_total_capacity`` 提升为**扫参维度**，同时保持：
      - quota 数量条件不变（同一请求流、同一分类标签、同一固定 k）：
            χ_pred = N_Ŝ / (k·N_L̂)  在不同容量下完全相同；
      - 固定安全比例 r=0.7（绝对安全门 = r·K 随 K 同增）。
    因此这是一组直接的控制实验：**保持 quota 的数量条件不变，干预它所处的资源环境。**

研究问题:
1. fig14 中的积压与等待，对 KV 容量有多敏感？
2. KV 约束减弱后，QBA 相对 PSF 的长尾保护及平均等待代价是否仍然存在？
3. quota 数量上界不变时，资源变化如何影响实际 promotion？

设计要点（继承 fig14）:
- **参考流锚定**：同一 (seed, λ) 先生成长度 ``N_ref = max(N, --label-reference-n)`` 的参考
  基础流，再截取前缀作为各 N 的请求流；分类阈值取自完整 N_ref 变换后 answer 分布。
  短流（N=5000）与长流（N=100000）共用同一条 N_ref=100000 参考流的阈值，避免分类差异。
- **逐请求总 token 守恒**：复用 fig11/fig14 的变换 P_i' + D_i' = P_i + D_i。
- **每次仿真独立请求状态**：前缀生成器每次重建堆，四策略 × 各容量共享同一 (arrival,
  length, label)，互不污染。
- **不另跑 mech_k 网格**：本轮正式 QBA 运行已开启机制统计，无需重复 fig14 的额外网格。

KV / 执行诊断口径:
    kv_hard_touch_rate     : kv_peak >= KV_total_capacity 的批次占比（硬容量越界，仅记录，
                             不触发真实 OOM 恢复）
    kv_gate_closed_rate    : 批次开始时 KV >= 安全门（r·K）的批次占比（所有策略同一 batch 分母）
    kv_peak_*              : 批次内 KV 峰值（绝对值 / 相对容量）
    exec_concurrent_*      : 批次开始时并发 decode 数（state.current_concurrent；所有策略均采集，
                             budget_tokens//request_block_size=16 为名义上界）
    exec_active_decode_*   : 调度时刻 decode 列表长度（**仅 QBA 调度器填充**；非 QBA 策略未采集，
                             记为 NaN，勿当作 0）；且它是调度时刻的列表长度，不等于本批执行后、
                             包含新准入请求的并发数
    n_pred_long_normal     : 从完成请求统计的唯一普通准入 predicted-long 数（不直接用
                             mech_long_admit_normal，后者可能重复计入 continuation chunk）

分片与恢复（云端）:
    --ckpt-dir DIR   每个 seed 完成后原子落盘一个分片 ``{stage}_seed{seed:07d}.npz``（先写
                     ``*.tmp.npz`` 再 ``os.replace``）；不同分片使用各自独立目录即可互不覆盖。
    --resume         若某 seed 分片已存在且坐标与本轮网格一致，则跳过该 seed 的全部仿真。
    --merge-dirs D1,D2
                     merge 模式：读取各目录分片，校验**容量与种子坐标**（跨分片坐标必须一致、
                     seed 必须互不重复），合并写出规范化的 ``fig15_{stage}_data.npz``。

用法:
    python -m scripts.experiment.sweep_kv_capacity
    python -m scripts.experiment.sweep_kv_capacity --stage short \\
        --capacities-short "8192,12288,16384,32768" --seed-count-short 100 --resume
    python -m scripts.experiment.sweep_kv_capacity --stage long --merge-dirs "ckpt/a,ckpt/b"
    python -m scripts.experiment.sweep_kv_capacity --smoke   # 极小规模快速自检

输出:
    plot/fig15_short_data.npz + plot/fig15_short_meta.json
    plot/fig15_long_data.npz  + plot/fig15_long_meta.json
    数组形状（单个阶段）: [n_seeds, n_a, n_lambda, n_K, n_N]；窗口多一维 n_windows；
    前缀诊断只依赖 (seed,a,λ,N)，形状 [n_seeds, n_a, n_lambda, n_N]（不含 K）。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from llm_in_queuing import simulator
from llm_in_queuing.config import load_config
from llm_in_queuing.metrics import mechanism_metrics
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
from .sweep_long_stream import (
    _transform_lengths,
    _make_prefix_generator,
    _prefix_short_ratio,
    _scalar_stream_diag,
    _window_arrays,
    _LONG_TRANSFORM_KEYS,
    _PREFIX_DIAG_KEYS,
    _STREAM_KEYS,
    _WINDOW_KEYS,
    _WINDOW_MODES,
    _MECH_KEYS,
    _POLICY_PREFIX,
)

# ── 输出键定义 ────────────────────────────────────────────────

# KV 资源状态（基于 batch 快照，分母为所有策略一致的批次）
_KV_DIAG_KEYS = [
    "kv_hard_touch_rate",
    "kv_gate_closed_rate",
    "kv_peak_mean", "kv_peak_max",
    "kv_peak_over_capacity_mean", "kv_peak_over_capacity_max",
]

# 执行状态（观察瓶颈是否转移）
_EXEC_DIAG_KEYS = [
    "exec_concurrent_mean", "exec_concurrent_max",
    "exec_active_decode_mean", "exec_active_decode_max",
]

# 唯一 predicted-long 准入去向（从完成请求统计，避免重复计 continuation chunk）
_LONG_ADMIT_KEYS = ["n_pred_long", "n_pred_long_promoted", "n_pred_long_normal"]

_POLICIES = ["FCFS", "PSF", "AA", "QBA"]

# 只有 QBA 调度器填充 mech_active_decode；其余策略该字段保持默认 0，须视为「未采集」
_ACTIVE_DECODE_POLICY = "QBA"

# 每策略子字典（用于统一保存 / 分片 / 合并遍历）
_POLICY_SUBDICTS = [
    ("rows", _METRICS_KEYS),
    ("policy_diag", _POLICY_DIAG_KEYS),
    ("stream", _STREAM_KEYS),
    ("kv", _KV_DIAG_KEYS),
    ("exec", _EXEC_DIAG_KEYS),
    ("long_admit", _LONG_ADMIT_KEYS),
    ("mech", _MECH_KEYS),
]

# 分片文件中的坐标字段（跨分片须一致，seed 除外）
_CKPT_COORD_KEYS = (
    "composition_scales", "lambdas", "capacities", "lengths",
    "n_windows", "n_ref", "safety_ratio", "absolute_safety_gate",
)
_CKPT_SEED_KEY = "seed"

# 策略名 → 数组前缀（分片/合并按实际运行的策略遍历，而非全局四策略）
_POLICY_PREFIX_MAP = dict(_POLICY_PREFIX)


# ── 解析 ────────────────────────────────────────────────────

def _parse_capacities(raw: str) -> np.ndarray:
    """解析 KV 容量列表：正整数、无重复，按升序返回（严格整数解析）。"""
    parts = [s.strip() for s in str(raw).split(",") if s.strip()]
    if not parts:
        raise ValueError("KV 容量网格不能为空")
    vals: list[int] = []
    for s in parts:
        try:
            v = int(s)
        except ValueError as exc:
            raise ValueError(f"非法 KV 容量: {s!r}（必须为整数）") from exc
        if v <= 0:
            raise ValueError(f"非法 KV 容量: {v}（必须 > 0）")
        vals.append(v)
    if len(set(vals)) != len(vals):
        raise ValueError(f"KV 容量存在重复值: {parts}")
    return np.asarray(sorted(vals), dtype=np.int64)


# ── 诊断 ────────────────────────────────────────────────────

def _kv_diag(state) -> dict:
    """KV 资源状态（batch 快照，分母为所有策略一致的 batch 数）。

    硬容量越界（``current_OOM`` / ``kv_peak >= KV_total_capacity``）在当前模拟器中只被
    记录，不触发真实 OOM 恢复，故不能解释为「消除了 OOM 失败」。
    """
    out = {k: float("nan") for k in _KV_DIAG_KEYS}
    snaps = state.batch_list
    if not snaps:
        return out
    K = float(state.KV_total_capacity)
    safety = float(state.KV_safety_threshold)
    oom = np.array([bool(b.get("current_OOM", False)) for b in snaps], dtype=np.float64)
    start_kv = np.array([float(b.get("current_KV_capacity", 0.0))
                         for b in snaps], dtype=np.float64)
    peak = np.array([float(b.get("kv_peak", 0.0)) for b in snaps], dtype=np.float64)
    out["kv_hard_touch_rate"] = float(oom.mean())
    out["kv_gate_closed_rate"] = float(np.mean(start_kv >= safety))
    out["kv_peak_mean"] = float(peak.mean())
    out["kv_peak_max"] = float(peak.max())
    if K > 0.0:
        out["kv_peak_over_capacity_mean"] = float((peak / K).mean())
        out["kv_peak_over_capacity_max"] = float((peak / K).max())
    return out


def _exec_diag(state, policy: str) -> dict:
    """执行状态：批次起始并发 decode（所有策略）+ 调度时刻 decode 列表长度（仅 QBA）。

    ``current_concurrent`` 由所有调度器写入（批次起始并发 decode 数），可直接比较；
    ``mech_active_decode`` 只有 QBA 调度器填充，其余策略保持默认 0——若直接读取会
    把「未采集」误当作 0，故非 QBA 返回 NaN。
    """
    out = {k: float("nan") for k in _EXEC_DIAG_KEYS}
    snaps = state.batch_list
    if not snaps:
        return out
    conc = np.array([float(b.get("current_concurrent", 0)) for b in snaps],
                    dtype=np.float64)
    out["exec_concurrent_mean"] = float(conc.mean())
    out["exec_concurrent_max"] = float(conc.max())
    if str(policy) == _ACTIVE_DECODE_POLICY and state.mech.enabled:
        ad = np.array([float(b.get("mech_active_decode", 0)) for b in snaps],
                      dtype=np.float64)
        out["exec_active_decode_mean"] = float(ad.mean())
        out["exec_active_decode_max"] = float(ad.max())
    return out


def _long_admit_diag(finished) -> dict:
    """从完成请求统计唯一 predicted-long 准入去向。

    ``mech_long_admit_normal`` 可能重复计入 continuation chunk，故这里是唯一普通准入
    long 请求数 = predicted-long 且未 promoted。
    """
    pred_long = [r for r in finished if getattr(r, "predict_label", "") == "long"]
    n_pl = len(pred_long)
    n_prom = sum(1 for r in pred_long if bool(getattr(r, "promoted", False)))
    return {
        "n_pred_long": float(n_pl),
        "n_pred_long_promoted": float(n_prom),
        "n_pred_long_normal": float(n_pl - n_prom),
    }


def _fill_transform_diag(transform_diag: dict, si: int, ai: int, li: int,
                         p_new, d_new, threshold: float, base_total, cfg: dict) -> None:
    """写入基于完整 N_ref 流的变换诊断（与策略、N、K 无关）。"""
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
    transform_diag["label_threshold"][si, ai, li] = float(threshold)


# ── 分片落盘 / 恢复 ─────────────────────────────────────────

def _seed_ckpt_path(ckpt_dir: str, stage: str, seed: int) -> str:
    return os.path.join(ckpt_dir, f"{stage}_seed{int(seed):07d}.npz")


def _build_coord_payload(scales, loads, capacities, lengths, n_windows, n_ref,
                         safety_ratio) -> dict:
    return {
        "composition_scales": np.asarray(scales, dtype=np.float64),
        "lambdas": np.asarray(loads, dtype=np.float64),
        "capacities": np.asarray(capacities, dtype=np.int64),
        "lengths": np.asarray(lengths, dtype=np.int64),
        "n_windows": np.asarray([int(n_windows)], dtype=np.int64),
        "n_ref": np.asarray([int(n_ref)], dtype=np.int64),
        "safety_ratio": np.asarray([float(safety_ratio)], dtype=np.float64),
        "absolute_safety_gate": (np.asarray(capacities, dtype=np.float64)
                                 * float(safety_ratio)),
    }


def _slice_seed_payload(si: int, seed: int, *, subs: dict, transform_diag: dict,
                        prefix_diag: dict, window: dict, coords: dict) -> dict:
    """把某个 seed 的整片结果切出来（首维=seed 被去掉），用于原子落盘。"""
    payload = {_CKPT_SEED_KEY: np.asarray([int(seed)], dtype=np.int64)}
    payload.update(coords)
    for k in _LONG_TRANSFORM_KEYS:
        payload[f"transform_{k}"] = transform_diag[k][si]
    for k in _PREFIX_DIAG_KEYS:
        payload[f"prefix_{k}"] = prefix_diag[k][si]
    for _p in subs["rows"]:
        prefix = _POLICY_PREFIX_MAP[_p]
        for name, keys in _POLICY_SUBDICTS:
            for k in keys:
                payload[f"{prefix}_{k}_runs"] = subs[name][_p][k][si]
        for mode in _WINDOW_MODES:
            for k in _WINDOW_KEYS:
                payload[f"{prefix}_{mode}_{k}_runs"] = window[_p][mode][k][si]
            payload[f"{prefix}_{mode}_edges_runs"] = window[_p][mode]["edges"][si]
    return payload


def _fill_seed_from_payload(payload: dict, si: int, *, subs: dict,
                            transform_diag: dict, prefix_diag: dict,
                            window: dict) -> None:
    """把分片内容填回整片数组的 seed 切片（resume 跳过仿真）。"""
    for k in _LONG_TRANSFORM_KEYS:
        transform_diag[k][si] = payload[f"transform_{k}"]
    for k in _PREFIX_DIAG_KEYS:
        prefix_diag[k][si] = payload[f"prefix_{k}"]
    for _p in subs["rows"]:
        prefix = _POLICY_PREFIX_MAP[_p]
        for name, keys in _POLICY_SUBDICTS:
            for k in keys:
                subs[name][_p][k][si] = payload[f"{prefix}_{k}_runs"]
        for mode in _WINDOW_MODES:
            for k in _WINDOW_KEYS:
                window[_p][mode][k][si] = payload[f"{prefix}_{mode}_{k}_runs"]
            window[_p][mode]["edges"][si] = payload[f"{prefix}_{mode}_edges_runs"]


def _ckpt_coords_match(payload: dict, coords: dict) -> bool:
    """校验分片坐标与本轮网格是否一致（用于 resume 判定是否可跳过）。"""
    for key, cur in coords.items():
        if key not in payload:
            return False
        if not np.array_equal(np.asarray(payload[key]), np.asarray(cur)):
            return False
    return True


def _write_seed_ckpt(path: str, payload: dict) -> None:
    """原子落盘：先写 ``*.tmp.npz`` 再 ``os.replace``，避免半截文件被误判为已完成。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


def _load_npz(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ── 主扫参：[n_seeds, n_a, n_lambda, n_K, n_N] ───────────────

def _run_kv_capacity_sweep(
    cfg: dict,
    scales: np.ndarray,
    loads: np.ndarray,
    capacities: np.ndarray,
    lengths: np.ndarray,
    policies: list[str],
    seeds: list[int],
    n_windows: int,
    label_reference_n: int | None = None,
    *,
    stage: str | None = None,
    ckpt_dir: str | None = None,
    resume: bool = False,
) -> dict:
    """在 (a, λ, K, N) 网格上扫参多个策略（逐请求总 token 守恒，前缀共享阈值）。

    容量 K 只进入 ``simulator.run`` 的本地配置（``KV_total_capacity``），不改变参考流 /
    变换 / 前缀 / 标签——因此同一 (seed, a, λ, N) 下各 K、各策略共享完全相同的请求流与标签。

    Args:
        label_reference_n: 分类阈值参考长度。基础流按 ``N_ref = max(max(lengths),
            本值)`` 生成；短流与长流共用同一 N_ref 时标签阈值一致。
        stage: 分片文件命名用的阶段名（short / long）。
        ckpt_dir: 若给定，每个 seed 完成后原子落盘一个分片；``resume`` 时若分片已存在
            且坐标一致则跳过该 seed 仿真。

    Returns:
        dict，包含以下键（维度见各值说明）:
            rows[p][k]        : [n_seeds, n_a, n_l, n_K, n_N]
            policy_diag[p][k] : [n_seeds, n_a, n_l, n_K, n_N]
            stream[p][k]      : [n_seeds, n_a, n_l, n_K, n_N]
            kv[p][k]          : [n_seeds, n_a, n_l, n_K, n_N]
            exec[p][k]        : [n_seeds, n_a, n_l, n_K, n_N]
            long_admit[p][k]  : [n_seeds, n_a, n_l, n_K, n_N]
            mech[p][k]        : [n_seeds, n_a, n_l, n_K, n_N]
            transform[k]      : [n_seeds, n_a, n_l]（N_ref 基准，含 label_threshold）
            prefix[k]         : [n_seeds, n_a, n_l, n_N]（与策略、K 无关）
            window[p][mode][k]: [n_seeds, n_a, n_l, n_K, n_N, n_windows]
            window[p][mode]["edges"]: [n_seeds, n_a, n_l, n_K, n_N, n_windows+1]
            n_ref             : int
    """
    n_seeds, n_a, n_l, n_K, n_N = (len(seeds), len(scales), len(loads),
                                   len(capacities), len(lengths))
    w = int(n_windows)
    shape = (n_seeds, n_a, n_l, n_K, n_N)
    shape_tr = (n_seeds, n_a, n_l)
    shape_pf = (n_seeds, n_a, n_l, n_N)
    shape_w = (n_seeds, n_a, n_l, n_K, n_N, w)
    shape_we = (n_seeds, n_a, n_l, n_K, n_N, w + 1)
    n_max = int(np.max(lengths))
    n_ref = max(n_max, int(label_reference_n) if label_reference_n is not None else 0)

    rows = {p: {k: np.full(shape, np.nan) for k in _METRICS_KEYS} for p in policies}
    policy_diag = {p: {k: np.full(shape, np.nan) for k in _POLICY_DIAG_KEYS}
                   for p in policies}
    stream = {p: {k: np.full(shape, np.nan) for k in _STREAM_KEYS} for p in policies}
    kv = {p: {k: np.full(shape, np.nan) for k in _KV_DIAG_KEYS} for p in policies}
    exe = {p: {k: np.full(shape, np.nan) for k in _EXEC_DIAG_KEYS} for p in policies}
    long_admit = {p: {k: np.full(shape, np.nan) for k in _LONG_ADMIT_KEYS}
                  for p in policies}
    mech = {p: {k: np.full(shape, np.nan) for k in _MECH_KEYS} for p in policies}
    transform_diag = {k: np.full(shape_tr, np.nan) for k in _LONG_TRANSFORM_KEYS}
    # 前缀诊断只依赖 (seed,a,λ,N)，不含 K 维
    prefix_diag = {k: np.full(shape_pf, np.nan) for k in _PREFIX_DIAG_KEYS}
    window = {
        p: {mode: {k: np.full(shape_w, np.nan) for k in _WINDOW_KEYS}
            for mode in _WINDOW_MODES}
        for p in policies
    }
    window_edge = {p: {mode: np.full(shape_we, np.nan) for mode in _WINDOW_MODES}
                   for p in policies}
    # window[...]["edges"] 直接引用 window_edge，使 resume 跳过路径也能写入窗口边界
    for p in policies:
        for mode in _WINDOW_MODES:
            window[p][mode]["edges"] = window_edge[p][mode]

    subs = {
        "rows": rows, "policy_diag": policy_diag, "stream": stream, "kv": kv,
        "exec": exe, "long_admit": long_admit, "mech": mech,
    }
    coords = _build_coord_payload(scales, loads, capacities, lengths, w, n_ref,
                                  cfg["KV_safety_threshold"])

    original_generate = simulator.generate_heapq
    runs_per_seed = n_l * n_a * n_N * n_K * len(policies)
    total = n_seeds * runs_per_seed
    done = 0
    skipped = 0
    print(f"stage: SIM kv-capacity runs={total} seeds={n_seeds} "
          f"a={n_a} lambda={n_l} K={n_K} N={n_N} policies={len(policies)} "
          f"capacities={capacities.tolist()} N_ref={n_ref} "
          f"ckpt={'on' if ckpt_dir else 'off'} resume={bool(resume)}")
    try:
        for si, seed in enumerate(seeds):
            ckpt_path = (_seed_ckpt_path(ckpt_dir, stage, seed)
                         if (ckpt_dir and stage) else None)

            # ── 恢复：分片存在且坐标一致 → 跳过该 seed 的全部仿真 ──
            if resume and ckpt_path is not None and os.path.exists(ckpt_path):
                payload = _load_npz(ckpt_path)
                if _ckpt_coords_match(payload, coords):
                    _fill_seed_from_payload(payload, si, subs=subs,
                                            transform_diag=transform_diag,
                                            prefix_diag=prefix_diag, window=window)
                    done += runs_per_seed
                    skipped += 1
                    progress_update(done, total, "SIM kv-capacity")
                    continue
                print(f"warn: 分片坐标不一致，忽略并重算: {ckpt_path}")

            for li, lam in enumerate(loads.tolist()):
                # 参考基础流（长度 N_ref ≥ N_max），所有 N 取自其前缀
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
                    p_new, d_new, threshold = _transform_lengths(base_reqs, float(a), cfg)
                    _fill_transform_diag(transform_diag, si, ai, li,
                                         p_new, d_new, threshold, base_total, cfg)

                    for ni, n_take in enumerate(lengths.tolist()):
                        n_take = int(n_take)
                        n_short, pi_s = _prefix_short_ratio(d_new, n_take, threshold)
                        prefix_diag["pi_s"][si, ai, li, ni] = pi_s
                        prefix_diag["n_short"][si, ai, li, ni] = float(n_short)

                        # 前缀生成器与 K 无关：同一 (seed,a,λ,N) 下所有 K、策略复用
                        simulator.generate_heapq = _make_prefix_generator(
                            base_arrival, p_new, d_new, base_ids, n_take, threshold)
                        for ki, K in enumerate(capacities.tolist()):
                            K = int(K)
                            for pol in policies:
                                local = dict(cfg)
                                local.update({
                                    "lambda_arrival": float(lam),
                                    "schedule_method": pol,
                                    "seed": int(seed),
                                    "admission_number": int(n_take),
                                    "KV_total_capacity": int(K),
                                })
                                try:
                                    metrics, state = simulator.run(local, return_state=True)
                                except Exception as exc:  # 附实验上下文
                                    raise RuntimeError(
                                        f"Experiment failed: seed={seed} a={a} "
                                        f"lambda={lam} K={K} N={n_take} policy={pol}"
                                    ) from exc
                                s = _extract_metrics(metrics)
                                for mk in _METRICS_KEYS:
                                    rows[pol][mk][si, ai, li, ki, ni] = float(s[mk])
                                for dk in _POLICY_DIAG_KEYS:
                                    policy_diag[pol][dk][si, ai, li, ki, ni] = float(s[dk])
                                mm = mechanism_metrics(state)
                                for k in _MECH_KEYS:
                                    mech[pol][k][si, ai, li, ki, ni] = float(
                                        mm.get(k, float("nan")))
                                for k, v in _scalar_stream_diag(
                                        state.finished_queue).items():
                                    stream[pol][k][si, ai, li, ki, ni] = float(v)
                                for k, v in _kv_diag(state).items():
                                    kv[pol][k][si, ai, li, ki, ni] = float(v)
                                for k, v in _exec_diag(state, pol).items():
                                    exe[pol][k][si, ai, li, ki, ni] = float(v)
                                for k, v in _long_admit_diag(
                                        state.finished_queue).items():
                                    long_admit[pol][k][si, ai, li, ki, ni] = float(v)
                                for mode in _WINDOW_MODES:
                                    wa, edges = _window_arrays(state.finished_queue,
                                                               state.batch_list, w, mode)
                                    for wk in _WINDOW_KEYS:
                                        window[pol][mode][wk][si, ai, li, ki, ni] = wa[wk]
                                    window_edge[pol][mode][si, ai, li, ki, ni, :] = edges
                                done += 1
                                progress_update(done, total, "SIM kv-capacity")
                        simulator.generate_heapq = original_generate

            # ── 该 seed 全部单元完成 → 原子落盘分片 ──
            for p in policies:
                for mode in _WINDOW_MODES:
                    window[p][mode]["edges"] = window_edge[p][mode]
            if ckpt_path is not None:
                payload = _slice_seed_payload(
                    si, seed, subs=subs, transform_diag=transform_diag,
                    prefix_diag=prefix_diag, window=window, coords=coords)
                _write_seed_ckpt(ckpt_path, payload)
    finally:
        simulator.generate_heapq = original_generate
    progress_done("SIM kv-capacity", total)
    if skipped:
        print(f"resume: 跳过 {skipped}/{n_seeds} 个 seed（命中分片）")
    return {
        "rows": rows,
        "policy_diag": policy_diag,
        "stream": stream,
        "kv": kv,
        "exec": exe,
        "long_admit": long_admit,
        "mech": mech,
        "transform": transform_diag,
        "prefix": prefix_diag,
        "window": window,
        "n_ref": int(n_ref),
    }


# ── 结果数组组装 / 合并 / 保存 ───────────────────────────────

def _build_arrays(res: dict, scales, loads, capacities, lengths, seeds,
                  n_windows: int, safety_ratio: float) -> dict:
    """把扫参结果整理成待落盘的扁平数组字典。"""
    arrays = {
        "composition_scales": np.asarray(scales, dtype=np.float64),
        "lambdas": np.asarray(loads, dtype=np.float64),
        "capacities": np.asarray(capacities, dtype=np.int64),
        "lengths": np.asarray(lengths, dtype=np.int64),
        "safety_ratio": np.asarray([float(safety_ratio)], dtype=np.float64),
        "absolute_safety_gate": (np.asarray(capacities, dtype=np.float64)
                                 * float(safety_ratio)),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "n_windows": np.asarray([int(n_windows)], dtype=np.int64),
        "n_ref": np.asarray([int(res["n_ref"])], dtype=np.int64),
    }
    for k in _LONG_TRANSFORM_KEYS:
        arrays[f"transform_{k}"] = res["transform"][k]
    for k in _PREFIX_DIAG_KEYS:
        arrays[f"prefix_{k}"] = res["prefix"][k]
    for p in res["rows"]:
        prefix = _POLICY_PREFIX_MAP[p]
        for name, keys in _POLICY_SUBDICTS:
            for k in keys:
                arrays[f"{prefix}_{k}_runs"] = res[name][p][k]
        for mode in _WINDOW_MODES:
            for k in _WINDOW_KEYS:
                arrays[f"{prefix}_{mode}_{k}_runs"] = res["window"][p][mode][k]
            arrays[f"{prefix}_{mode}_edges_runs"] = res["window"][p][mode]["edges"]
    return arrays


def _merge_ckpt_payloads(payloads: list[dict]) -> dict:
    """合并各 seed 分片：校验坐标与 seed 坐标，按 seed 堆叠。

    跨分片的坐标（容量、构成、λ、长度、窗口数、N_ref、安全比例）必须完全一致，
    seed 必须互不重复；否则报错，避免静默产生错误合并。
    """
    if not payloads:
        raise ValueError("没有可合并的分片")
    ref = payloads[0]
    for key in _CKPT_COORD_KEYS:
        if key not in ref:
            raise ValueError(f"分片缺少坐标字段: {key}")
        base = np.asarray(ref[key])
        for pi, pl in enumerate(payloads[1:], start=1):
            if key not in pl or not np.array_equal(np.asarray(pl[key]), base):
                raise ValueError(
                    f"分片坐标不一致: {key}（分片 0 vs 分片 {pi}）")
    seeds = [int(np.asarray(pl[_CKPT_SEED_KEY]).reshape(-1)[0]) for pl in payloads]
    if len(set(seeds)) != len(seeds):
        dup = sorted({s for s in seeds if seeds.count(s) > 1})
        raise ValueError(f"分片存在重复 seed 坐标: {dup}")
    order = np.argsort(np.asarray(seeds))
    seeds_sorted = [seeds[i] for i in order]
    ordered = [payloads[i] for i in order]

    data_keys = [k for k in ref.keys()
                 if k not in _CKPT_COORD_KEYS and k != _CKPT_SEED_KEY]
    n_seeds = len(seeds_sorted)
    arrays: dict = {}
    for key in data_keys:
        sample = np.asarray(ref[key], dtype=np.float64)
        arrays[key] = np.full((n_seeds,) + sample.shape, np.nan, dtype=np.float64)
    for si, pl in enumerate(ordered):
        for key in data_keys:
            arrays[key][si] = np.asarray(pl[key], dtype=np.float64)
    # 坐标字段取参考分片
    for key in _CKPT_COORD_KEYS:
        arr = np.asarray(ref[key])
        arrays[key] = arr.astype(np.float64) if arr.dtype.kind == "f" else arr
    arrays["seeds"] = np.asarray(seeds_sorted, dtype=np.int64)
    return arrays


def _save_stage(plot_dir: str, stage: str, cfg: dict, cfg_path: Path,
                args: argparse.Namespace, arrays: dict) -> tuple[str, str]:
    """根据扁平数组字典写出阶段 npz + meta。"""
    scales = np.asarray(arrays["composition_scales"], dtype=np.float64)
    loads = np.asarray(arrays["lambdas"], dtype=np.float64)
    capacities = np.asarray(arrays["capacities"], dtype=np.int64)
    lengths = np.asarray(arrays["lengths"], dtype=np.int64)
    seeds = np.asarray(arrays["seeds"], dtype=np.int64)
    n_windows = int(np.asarray(arrays["n_windows"]).reshape(-1)[0])
    n_ref = int(np.asarray(arrays["n_ref"]).reshape(-1)[0])
    safety_ratio = float(np.asarray(arrays["safety_ratio"]).reshape(-1)[0])
    absolute_gate = {int(K): float(K) * safety_ratio for K in capacities.tolist()}

    extra_meta = {
        "experiment": "kv_capacity_sensitivity_fixed_safety_ratio",
        "stage": str(stage),
        "sweep_policies": _POLICIES,
        "composition_scales": [float(v) for v in scales.tolist()],
        "lambdas": [float(v) for v in loads.tolist()],
        "capacities": [int(v) for v in capacities.tolist()],
        "lengths": [int(v) for v in lengths.tolist()],
        "seeds": [int(v) for v in seeds.tolist()],
        "safety_ratio": safety_ratio,
        "absolute_safety_gate": absolute_gate,
        "n_windows": n_windows,
        "label_threshold_percentail": float(cfg["label_threshold_percentail"]),
        "normalization": "exact per-request total-token preservation (fig11 transform)",
        "label_reference": {
            "label_reference_n": (int(args.label_reference_n)
                                  if getattr(args, "label_reference_n", None) is not None
                                  else None),
            "n_ref": n_ref,
            "note": ("label threshold taken from the full N_ref stream; short and long "
                     "stages share the same N_ref threshold"),
        },
        "prefix_consistency": (
            "base stream generated once per (seed,lambda) at length N_ref; every N is its "
            "prefix; label threshold from the full N_ref transformed answers; capacities "
            "share the same requests/labels"
        ),
        "prefix_diag_scope": (
            "prefix_pi_s / prefix_n_short depend only on (seed,a,lambda,N); shape "
            "[n_seeds,n_a,n_lambda,n_N] (no K axis)"
        ),
        "chi_pred_invariant": (
            "chi_pred = N_short_hat / (k * N_long_hat) is unchanged across capacities: "
            "same stream, same labels, fixed k"
        ),
        "stream_metric_scope": {
            "waiting_at_arrival_stop": "start_service_time > t_last_arrival",
            "unfinished_at_arrival_stop": "end_service_time > t_last_arrival (matches window backlog)",
            "promotion_ratio": "N_prom / N_all",
            "phi_Q": "N_prom / N_pred_long (matches fig12 phi_Q)",
        },
        "kv_metric_scope": {
            "kv_hard_touch_rate": "fraction of batches with kv_peak >= KV_total_capacity (recorded only)",
            "kv_gate_closed_rate": "fraction of batches with batch-start KV >= safety gate (r*K)",
            "kv_peak_over_capacity": "batch kv_peak / KV_total_capacity",
            "note": ("out-of-capacity is recorded, not a real OOM recovery; do not read the "
                     "change as eliminating OOM failures"),
        },
        "exec_metric_scope": {
            "exec_concurrent": "batch-start concurrent decode count (state.current_concurrent; all policies)",
            "max_active_decode_nominal": "budget_tokens // request_block_size (=16)",
            "exec_active_decode": ("scheduler-time decode list length; QBA only (NaN for "
                                   "FCFS/PSF/AA); not the post-batch concurrency"),
        },
        "mech_diag_enabled": True,
        "checkpoint_schema": [
            "seed", "a", "lambda", "K", "safety_ratio", "N", "policy", "N_ref",
            "source_version",
        ],
        "checkpoint": {
            "mode": "per_seed_shard",
            "ckpt_dir": getattr(args, "ckpt_dir", None),
            "resume": bool(getattr(args, "resume", False)),
            "file_pattern": "{stage}_seed{seed:07d}.npz",
            "atomic": "write .tmp.npz then os.replace",
            "merge": ("--merge-dirs validates identical coordinates and disjoint seeds "
                      "before merging"),
        },
        "source_version": {"git_commit": "recorded in this meta under 'git'"},
        "invariant_note": (
            "per-request total tokens preserved; K enters only KV_total_capacity; "
            "time windows cover [0, t_last_arrival] (drain separate); "
            "unique normal long admits computed from finished requests (pred long & not promoted)"
        ),
    }

    npz_path, meta_path = save_results(
        plot_dir, f"fig15_{stage}", arrays, cfg, vars(args),
        config_path=str(cfg_path), extra_meta=extra_meta,
    )
    print(f"saved data: {npz_path}")
    print(f"saved meta: {meta_path}")
    return npz_path, meta_path


def _run_stage(
    stage: str,
    cfg: dict,
    cfg_path: Path,
    plot_dir: str,
    args: argparse.Namespace,
    scales: np.ndarray,
    loads: np.ndarray,
    capacities: np.ndarray,
    lengths: np.ndarray,
    seeds: list[int],
    n_windows: int,
) -> tuple[str, str]:
    """运行一个阶段并保存 npz + meta，返回 (npz_path, meta_path)。"""
    res = _run_kv_capacity_sweep(
        cfg, scales, loads, capacities, lengths, _POLICIES, seeds,
        int(n_windows), label_reference_n=args.label_reference_n,
        stage=stage, ckpt_dir=args.ckpt_dir, resume=bool(args.resume))
    arrays = _build_arrays(res, scales, loads, capacities, lengths, seeds,
                           int(n_windows), float(cfg["KV_safety_threshold"]))
    return _save_stage(plot_dir, stage, cfg, cfg_path, args, arrays)


def _merge_stage(stage: str, cfg: dict, cfg_path: Path, plot_dir: str,
                 args: argparse.Namespace, ckpt_dirs: list[str]) -> tuple[str, str]:
    """读取多个分片目录（或分片文件），校验坐标后合并写出规范结果。"""
    paths: list[str] = []
    for d in ckpt_dirs:
        if os.path.isdir(d):
            paths.extend(sorted(
                os.path.join(d, f) for f in os.listdir(d)
                if f.startswith(f"{stage}_seed") and f.endswith(".npz")
                and ".tmp." not in f))
        elif os.path.isfile(d):
            paths.append(d)
        else:
            raise FileNotFoundError(f"分片目录/文件不存在: {d}")
    if not paths:
        raise FileNotFoundError(f"未找到 {stage} 分片（pattern: {stage}_seed*.npz）")
    print(f"stage: MERGE kv-capacity[{stage}] shards={len(paths)}")
    payloads = [_load_npz(p) for p in paths]
    arrays = _merge_ckpt_payloads(payloads)
    print(f"merge: seeds={arrays['seeds'].tolist()} "
          f"capacities={arrays['capacities'].tolist()}")
    return _save_stage(plot_dir, stage, cfg, cfg_path, args, arrays)


def main():
    parser = argparse.ArgumentParser(
        description="fig15: 固定安全比例下的 KV 容量敏感性与调度折中实验")
    parser.add_argument("--stage", choices=["short", "long", "both"], default="both",
                        help="运行短流容量扫描、长流端点验证，或两者（默认 both）")
    # 短流容量扫描
    parser.add_argument("--capacities-short", type=str,
                        default="8192,12288,16384,32768")
    parser.add_argument("--scales-short", type=str, default="0.5,1,2,4")
    parser.add_argument("--length-short", type=int, default=5000)
    parser.add_argument("--seed-count-short", type=int, default=100)
    # 长流端点验证
    parser.add_argument("--capacities-long", type=str, default="8192,32768")
    parser.add_argument("--scales-long", type=str, default="1,2")
    parser.add_argument("--length-long", type=int, default=100000)
    parser.add_argument("--seed-count-long", type=int, default=32)
    # 公共
    parser.add_argument("--lambdas", type=str, default="30,32,34,38")
    parser.add_argument("--windows", type=int, default=8,
                        help="窗口数（序号 / 时间两种口径共用）")
    parser.add_argument("--gamma", type=float, default=0.80)
    parser.add_argument("--quota-k", type=int, default=5)
    parser.add_argument("--label-reference-n", type=int, default=100000,
                        help="分类阈值参考长度；基础流按 max(max(lengths), 本值) 生成，"
                             "短流与长流共用同一 N_ref 阈值")
    parser.add_argument("--plot-dir", type=str, default=os.path.join(
        os.path.dirname(__file__), "..", "plot"))
    parser.add_argument("--seed-start", type=int, default=42)
    # 分片 / 恢复 / 合并
    parser.add_argument("--ckpt-dir", type=str, default=None,
                        help="每个 seed 的原子分片落盘目录（不同分片用各自目录互不覆盖）；"
                             "缺省时不落盘")
    parser.add_argument("--resume", action="store_true",
                        help="若某 seed 分片已存在且坐标一致则跳过该 seed 仿真")
    parser.add_argument("--merge-dirs", type=str, default=None,
                        help="merge 模式：逗号分隔的分片目录/文件列表，校验坐标后合并")
    parser.add_argument("--smoke", action="store_true",
                        help="极小规模自检（覆盖默认网格，仅用于流程验证）")
    args = parser.parse_args()

    cfg_path = Path(__file__).resolve().parent / "configs" / "sweep_kv_capacity_base.yaml"
    cfg = load_config(cfg_path)
    cfg["gamma"] = float(args.gamma)
    cfg["quota_k"] = int(args.quota_k)
    cfg["mech_diag_enabled"] = True

    if args.smoke:
        args.stage = "both"
        args.capacities_short = "8192,16384"
        args.scales_short = "1,2"
        args.length_short = 120
        args.seed_count_short = 2
        args.capacities_long = "8192,16384"
        args.scales_long = "1"
        args.length_long = 240
        args.seed_count_long = 2
        args.lambdas = "30,32"
        args.windows = 4
        args.label_reference_n = 480

    plot_dir = os.path.abspath(args.plot_dir)

    # ── merge 模式 ──
    if args.merge_dirs:
        dirs = [d.strip() for d in args.merge_dirs.split(",") if d.strip()]
        stages = ["short", "long"] if args.stage == "both" else [args.stage]
        for st in stages:
            try:
                _merge_stage(st, cfg, cfg_path, plot_dir, args, dirs)
            except FileNotFoundError as exc:
                print(f"skip merge[{st}]: {exc}")
        return

    loads = _parse_lambdas(args.lambdas)
    print(f"stage: SIM kv-capacity lambdas={loads.tolist()} gamma={cfg['gamma']} "
          f"quota_k={cfg['quota_k']} label_reference_n={args.label_reference_n} "
          f"windows={args.windows} run_stage={args.stage} "
          f"ckpt_dir={args.ckpt_dir} resume={bool(args.resume)}")

    if args.stage in ("short", "both"):
        caps_s = _parse_capacities(args.capacities_short)
        scales_s = _parse_composition_scales(args.scales_short)
        lengths_s = np.asarray([int(args.length_short)], dtype=np.int64)
        seeds_s = [int(args.seed_start) + i for i in range(int(args.seed_count_short))]
        print(f"stage: SIM kv-capacity[short] capacities={caps_s.tolist()} "
              f"scales={scales_s.tolist()} N={lengths_s.tolist()} seeds={len(seeds_s)}")
        npz_s, _ = _run_stage("short", cfg, cfg_path, plot_dir, args,
                              scales_s, loads, caps_s, lengths_s, seeds_s,
                              int(args.windows))
        print(f"next: python -m scripts.draw.draw_fig15 --data-short \"{npz_s}\"")

    if args.stage in ("long", "both"):
        caps_l = _parse_capacities(args.capacities_long)
        scales_l = _parse_composition_scales(args.scales_long)
        lengths_l = np.asarray([int(args.length_long)], dtype=np.int64)
        seeds_l = [int(args.seed_start) + i for i in range(int(args.seed_count_long))]
        print(f"stage: SIM kv-capacity[long] capacities={caps_l.tolist()} "
              f"scales={scales_l.tolist()} N={lengths_l.tolist()} seeds={len(seeds_l)}")
        npz_l, _ = _run_stage("long", cfg, cfg_path, plot_dir, args,
                              scales_l, loads, caps_l, lengths_l, seeds_l,
                              int(args.windows))
        print(f"next: python -m scripts.draw.draw_fig15 --data-long \"{npz_l}\"")


if __name__ == "__main__":
    main()
