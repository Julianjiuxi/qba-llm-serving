"""scaled real-trace replay：Fig11–15 机制在真实 Azure trace 上的理论迁移验证（模块 A–E）。

定位
====
本脚本**不进入主系列命名（figN）**，与 ``real_trace_analysis.py`` 并列。论文中的准确
名称是 **scaled real-trace replay**：真实到达形状与长度关系经**统一缩放**后在当前
模拟器中回放。它验证输入分布的迁移性，**不能**替代真实 GPU 上的系统验证。

要检验的三件事（允许数值、转折位置甚至曲线形状改变）:
    1. quota 的数量约束是否仍有解释力（有限样本上界与**实际兑现**之间的距离）；
    2. 阶段构成与 KV 资源是否仍共同影响调度（固定安全比例下的资源响应）；
    3. 这些折中是否经得住真实时间序列的长流检验（积压 / 兑现 / 服务质量）。

关键事实（analysis/realtrace_design_20261128/*.json）
=====================================================
真实轨迹总 token 规模与合成基线接近（≈774 vs ≈771），但 prefill/decode 构成相差极大：
    Azure:  E[P]≈727.96  E[D]≈46.51  R_tok=E[P]/E[D]≈15.65  prompt 占总 token≈94%
    合成:   E[P]≈106.9   E[D]≈664.0  R_tok≈0.161            prompt 占比≈13.9%
故真实轨迹特别适合检验「总 token 数不足以描述引擎中的实际服务过程」。注意 R_tok 是
**token 比例**，与旧规划中服务成本加权比 R=c_pE[P]/(c_dE[D]) 不可直接比较。

模块（--module）
================
A  真实窗口上的构成 × KV × 到达率主实验
   100 短窗口 × a{1/16,1/4,1} × K{8192,16384,32768} × λ{26,32,38} × 4 策略 = 10,800 次
B  小规模 p×k：quota 耦合能否迁移（a=1, λ=38, K{8192,32768}, p=0.70–0.95, k∈{3,5,9}）
C  方向性误判：真实长度结构下的错误后果（a=1,p=0.85,k=5,λ=38；ε_SL/ε_LS∈{0.1,0.3}）
D  32 段真正连续长流（N=5,000 前缀 / 100,000 完整；a{1/16,1}；K{8192,32768}；λ{32,38}）
E  统一时间缩放对照（32 长窗口，a=1，共享 c_t，保留跨窗口真实强度差异）

数据与统计口径（实现前固定）
============================
- **统一 token_scale=0.44**，逐请求整数化 ``max(min_cap, min(cap, int(raw*0.44)))``；
- **保留同时到达事件**：不删除零间隔，也不重新生成泊松到达；
- **不对每个窗口重新匹配平均总 token**，保留真实窗口间长度差异；
- **从最终请求数组计算预统计**（修正旧脚本未复现整数化、删零间隔的问题）；
- **评价群体固定**：基线 answer 最长约 15% 与基线总 token 最长约 5%，所有 a/K/策略共享请求 ID；
- **loader 陷阱**：旧 ``_generate_heapq_from_csv`` 会按本次截取 ``n_take`` 重算时间缩放与
  标签阈值。故模块 D/E 必须先在**完整区间**上完成长度处理与构成变换、确定时间缩放系数
  与标签阈值、冻结完整请求数组，**再截前缀**；不能分别设 admission_number=5000/100000。

用法
====
    python -m scripts.real_trace_transfer --module A
    python -m scripts.real_trace_transfer --module A --ckpt-dir ckpt/rtA --resume
    python -m scripts.real_trace_transfer --module A --merge-dirs "ckpt/rtA_a,ckpt/rtA_b"
    python -m scripts.real_trace_transfer --module D --long-source week --n-long-windows 32
    python -m scripts.real_trace_transfer --smoke          # 极小规模快速自检

输出
====
    scripts/real_trace_output/rt_{module}_data.npz  +  rt_{module}_meta.json
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ── 环境引导：确保导入本仓库 src（避免被同名旧 editable 安装遮蔽）──
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
for _p in (str(_SRC), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llm_in_queuing import simulator  # noqa: E402
from llm_in_queuing.config import load_config  # noqa: E402
from llm_in_queuing.metrics import mechanism_metrics  # noqa: E402
from llm_in_queuing.types import Request  # noqa: E402

from scripts.experiment._common import (  # noqa: E402
    progress_update,
    progress_done,
    save_results,
)
from scripts.experiment.sweep_request_composition import (  # noqa: E402
    _extract_metrics,
    _METRICS_KEYS,
    _POLICY_DIAG_KEYS,
    _PROMPT_LOWER_BOUND,
    _ANSWER_LOWER_BOUND,
)
from scripts.experiment.sweep_long_stream import (  # noqa: E402
    _prefix_short_ratio,
    _scalar_stream_diag,
    _window_arrays,
    _LONG_TRANSFORM_KEYS,
    _PREFIX_DIAG_KEYS,
    _STREAM_KEYS,
    _WINDOW_KEYS,
    _WINDOW_MODES,
    _MECH_KEYS,
    _parse_ks,
)
from scripts.experiment.sweep_kv_capacity import (  # noqa: E402
    _KV_DIAG_KEYS,
    _EXEC_DIAG_KEYS,
    _LONG_ADMIT_KEYS,
    _POLICY_SUBDICTS,
    _parse_capacities,
)
from scripts.experiment.sweep_composition_load import _parse_lambdas  # noqa: E402
from scripts.experiment.sweep_request_composition import (  # noqa: E402
    _parse_composition_scales,
)

_POLICY_PREFIX = [("FCFS", "fcfs"), ("PSF", "psf"), ("AA", "psfa"), ("QBA", "psfaq")]
_POLICY_PREFIX_MAP = dict(_POLICY_PREFIX)

_ALL_POLICIES = ["FCFS", "PSF", "AA", "QBA"]

# 与策略无关的窗口预统计（从最终请求数组计算；保留零间隔）
_PRE_STAT_KEYS = [
    "n_requests", "raw_lambda", "trace_span_s",
    "scaled_E_P", "scaled_E_D", "scaled_E_PD",
    "p_answer_threshold", "true_long_fraction", "inter_arrival_Ca2",
]

# 固定评价群体（基线 a=1 参考流上预定；所有 a/K/策略共享请求 ID）
_FIXED_GROUP_KEYS = [
    "fx_ans_long_n", "fx_ans_long_wait_mean", "fx_ans_long_wait_p99",
    "fx_tot_long_n", "fx_tot_long_wait_mean", "fx_tot_long_wait_p99",
]

# 模块 C 四群体统计（真类别 × 预测类别）
#   0: short→short  1: short→long  2: long→short  3: long→long
_QUAD_NAMES = ["ss", "sl", "ls", "ll"]
_QUAD_STATS = ["n", "wait_mean", "wait_p99", "promo_n", "P_mean", "D_mean", "T_mean"]


# ── 真实窗口容器与 loader ──────────────────────────────────────

@dataclass
class ReplayWindow:
    """一个真实 trace 窗口的冻结请求数组（已完成长度处理与整数化）。

    时间与长度在**完整窗口**上确定；模块 D/E 的前缀直接取自这些完整数组。
    """
    wid: int
    arrivals: np.ndarray      # [N] 已 re-zero 到 0 的原始秒（未做 λ 缩放）
    p_base: np.ndarray        # [N] int64 经 token_scale + 上下限后的 prompt
    d_base: np.ndarray        # [N] int64 经 token_scale + 上下限后的 answer
    ids: np.ndarray           # [N] int64
    raw_lambda: float
    span: float
    label_threshold_a1: float  # a=1 下 answer 的 p=0.85 分位（固定评价群体基准）
    fixed_ans_mask: np.ndarray  # 基线 answer 最长约 15%
    fixed_tot_mask: np.ndarray  # 基线总 token 最长约 5%
    source_key: str = ""        # 输入文件 + 行区间的稳定身份（checkpoint 校验用）

    @property
    def total_base(self) -> np.ndarray:
        return (self.p_base.astype(np.float64) + self.d_base.astype(np.float64))


def _scale_tokens(raw: np.ndarray, cfg: dict, lo: int, cap_key: str) -> np.ndarray:
    """复现旧 loader 的整数化：``max(lo, min(cap, int(raw * token_scale)))``。"""
    tok = float(cfg.get("token_scale", 1.0))
    cap = int(cfg.get(cap_key, 10 ** 9))
    v = (np.asarray(raw, dtype=np.float64) * tok).astype(np.int64)  # astype = 向零截断
    v = np.minimum(v, cap)
    v = np.maximum(v, lo)
    return v


_UTC_EPOCH = None


def _parse_timestamp_seconds(col) -> np.ndarray:
    """把 ISO8601 时间戳字符串列解析成**绝对秒**（float64，相对 1970-01-01Z）。

    原始整周 CSV 的时间列形如 ``2024-05-12 00:00:00.001163+00:00``，可能带/不带小数秒。
    ``format='mixed'`` 在千万行规模下开销巨大（~60s/2M 行），故走快速路径：整秒部分用
    固定格式解析，小数秒单独用正则提取后拼接（实测 200k 行 ~1s，与 ISO8601 数值等价）。
    相对首行的秒数由调用方相减得到。
    """
    import pandas as pd

    global _UTC_EPOCH
    if _UTC_EPOCH is None:
        _UTC_EPOCH = pd.Timestamp("1970-01-01T00:00:00Z")
    s = pd.Series(col).astype(str)
    base = pd.to_datetime(s.str.slice(0, 19), format="%Y-%m-%d %H:%M:%S", utc=True)
    sec = (base - _UTC_EPOCH).dt.total_seconds().to_numpy(np.float64)
    frac_str = s.str.extract(r"\.(\d+)", expand=False)   # 无小数秒 → NaN
    nd = frac_str.str.len().fillna(0).to_numpy(np.int64)
    fv = pd.to_numeric(frac_str, errors="coerce").fillna(0.0).to_numpy(np.float64)
    frac = np.where(nd > 0, fv / np.power(10.0, nd.astype(np.float64)), 0.0)
    return sec + frac


def _time_seconds(df) -> np.ndarray:
    """从窗口 DataFrame 取**相对秒**时间戳。

    处理过的 subset CSV 已带 ``TimeSeconds`` 列，直接使用；原始整周 CSV 只有
    ``TIMESTAMP``（UTC 字符串），需解析后取相对首行的秒数。两条路径都不删除零间隔，
    保留同时到达。
    """
    if "TimeSeconds" in df.columns:
        return np.asarray(df["TimeSeconds"].to_numpy(np.float64))
    if "TIMESTAMP" not in df.columns:
        raise KeyError("窗口缺少 TimeSeconds 与 TIMESTAMP 两列，无法解析时间")
    t = _parse_timestamp_seconds(df["TIMESTAMP"])
    if t.size == 0:
        return t
    return t - t[0]


def _make_window(wid: int, ts: np.ndarray, p_raw: np.ndarray, d_raw: np.ndarray,
                 cfg: dict, source_key: str = "") -> ReplayWindow:
    ts = np.asarray(ts, dtype=np.float64)
    if ts.size >= 2 and np.any(np.diff(ts) < 0.0):
        # 保留真实到达顺序：不允许时间倒流（否则 re-zero 后的到达序列无意义）
        raise ValueError(f"窗口 {wid} 内时间戳非非递减（真实到达顺序被破坏）")
    arrivals = ts - ts[0]                     # re-zero，未做 λ 缩放
    span = float(arrivals[-1] - arrivals[0]) if arrivals.size else 0.0
    raw_lambda = float(arrivals.size) / span if span > 0 else float("inf")
    p_base = _scale_tokens(p_raw, cfg, _PROMPT_LOWER_BOUND, "prompt_len_cap")
    d_base = _scale_tokens(d_raw, cfg, _ANSWER_LOWER_BOUND, "answer_len_cap")
    pct = float(cfg["label_threshold_percentail"])
    thr_a1 = float(np.quantile(d_base.astype(np.float64), pct))
    total = p_base.astype(np.float64) + d_base.astype(np.float64)
    ans_top = float(np.quantile(d_base.astype(np.float64), 0.85))
    tot_top = float(np.quantile(total, 0.95))
    return ReplayWindow(
        wid=int(wid), arrivals=arrivals, p_base=p_base, d_base=d_base,
        ids=np.arange(arrivals.shape[0], dtype=np.int64),
        raw_lambda=raw_lambda, span=span, label_threshold_a1=thr_a1,
        fixed_ans_mask=(d_base.astype(np.float64) > ans_top),
        fixed_tot_mask=(total > tot_top),
        source_key=str(source_key),
    )


def _compute_pre_stats(win: ReplayWindow, cfg: dict) -> dict:
    """从最终请求数组计算窗口预统计（保留零间隔，不用旧脚本的删零做法）。"""
    n = win.arrivals.shape[0]
    inter = np.diff(win.arrivals)             # 保留同时到达（零间隔）
    if inter.size >= 2:
        mu = float(inter.mean())
        var = float(inter.var(ddof=1))
        ca2 = var / (mu ** 2) if mu > 0 else float("nan")
    else:
        ca2 = float("nan")
    thr = win.label_threshold_a1
    d = win.d_base.astype(np.float64)
    return {
        "n_requests": float(n),
        "raw_lambda": float(win.raw_lambda),
        "trace_span_s": float(win.span),
        "scaled_E_P": float(win.p_base.mean()),
        "scaled_E_D": float(win.d_base.mean()),
        "scaled_E_PD": float(win.total_base.mean()),
        "p_answer_threshold": float(thr),
        "true_long_fraction": float(np.mean(d > thr)),
        "inter_arrival_Ca2": ca2,
    }


def _load_subset_windows(cfg: dict, n_windows: int,
                         max_n: int | None = None, start: int = 0
                         ) -> tuple[list[ReplayWindow], dict]:
    """读取 data/Azure_work_load_subset*.csv（各 5000 请求，已 TimeSeconds 归零）。

    ``max_n`` 仅用于 smoke 自检：截取每个窗口前 ``max_n`` 条请求，缩小仿真规模。
    ``start`` 为**全局窗口起始编号**（并行分片用）：取第 ``[start, start+n_windows)`` 个
    CSV，窗口 ``wid`` 保留全局编号，保证不同分片 ID / 掩码对齐。

    返回 ``(windows, partition)``；``partition`` 为**全体**窗口的来源列表（只含文件名，
    不含内容依赖的请求数），故各 worker 计算一致，可作为公共签名的一部分。
    """
    import pandas as pd

    data_dir = _REPO_ROOT / str(cfg.get("trace_dir", "data"))
    glob_pat = str(cfg.get("subset_glob", "Azure_work_load_subset[0-9][0-9][0-9].csv"))
    all_paths = sorted(data_dir.glob(glob_pat))
    if not all_paths:
        raise FileNotFoundError(f"未找到 subset CSV：{data_dir}/{glob_pat}")
    partition = {"source": "subset", "glob": glob_pat,
                 "sources": [f"subset:{p.name}" for p in all_paths]}
    start = int(start)
    paths = all_paths[start:]
    if n_windows is not None and n_windows > 0:
        paths = paths[: int(n_windows)]
    wins: list[ReplayWindow] = []
    for wi, p in enumerate(paths):
        df = pd.read_csv(p)
        ts = _time_seconds(df)
        ct = df["ContextTokens"].to_numpy(np.float64)
        gt = df["GeneratedTokens"].to_numpy(np.float64)
        if max_n is not None and max_n > 0:
            ts, ct, gt = ts[: int(max_n)], ct[: int(max_n)], gt[: int(max_n)]
        key = f"subset:{p.name}"
        wins.append(_make_window(start + wi, ts, ct, gt, cfg, source_key=key))
    return wins, partition


def _subset_grid_rate(cfg: dict, args) -> float:
    """subset 源下模块 E 的**全局**统一时间缩放：对全体长窗口（grid）求 pooled rate。

    只读时间列，跨 worker 计算一致（避免各进程按自己的窗口子集重复计算导致强度差异被抹平）。
    """
    import pandas as pd

    data_dir = _REPO_ROOT / str(cfg.get("trace_dir", "data"))
    glob_pat = str(cfg.get("subset_glob", "Azure_work_load_subset[0-9][0-9][0-9].csv"))
    paths = sorted(data_dir.glob(glob_pat))
    grid = int(args.window_total) if int(args.window_total) > 0 else int(args.n_long_windows)
    paths = paths[: max(1, grid)]
    total_n, total_span = 0, 0.0
    for p in paths:
        ts = _time_seconds(pd.read_csv(p))
        if ts.size == 0:
            continue
        total_n += int(ts.size)
        total_span += float(ts[-1] - ts[0])
    return float(total_n) / total_span if total_span > 0 else float("inf")


def _week_long_starts(ts: np.ndarray, n_windows: int, n_full: int) -> list[int]:
    """按**时间分层**选取 n_windows 个互不重叠、各含连续 n_full 请求的窗口起点。

    把整周相对时间跨度均分为 n_windows 层；每层取该层起始时刻对应的行索引作为起点，
    并强制与前一窗口不重叠、为后续窗口预留空间（避免行号均匀选点的边界重叠）。
    """
    ts = np.asarray(ts, dtype=np.float64)
    n_total = int(ts.size)
    n_windows = int(n_windows)
    n_full = int(n_full)
    if n_windows <= 0 or n_full <= 0:
        raise ValueError("n_windows / n_full 必须为正整数")
    if n_full * n_windows > n_total:
        raise ValueError(f"1 周仅 {n_total} 行，无法取 {n_windows}×{n_full}")
    t0, t1 = float(ts[0]), float(ts[-1])
    span = t1 - t0
    starts: list[int] = []
    prev_end = 0
    for k in range(n_windows):
        target_t = t0 + span * (float(k) / float(n_windows))
        idx = int(np.searchsorted(ts, target_t, side="left"))
        max_idx = n_total - n_full * (n_windows - k)   # 为剩余窗口预留空间
        idx = max(idx, prev_end)                        # 与前一窗口不重叠
        if idx > max_idx:
            idx = max_idx
        starts.append(int(idx))
        prev_end = int(idx) + n_full
    return starts


def _load_week_windows(cfg: dict, n_windows: int, n_full: int,
                       start: int = 0, window_total: int = 0
                       ) -> tuple[list[ReplayWindow], list[int], int, dict, float]:
    """从完整一周 CSV 抽取 n_windows 个**按时间分层、互不重叠**的连续 n_full 请求区间。

    原始整周文件只有 ``TIMESTAMP,ContextTokens,GeneratedTokens`` 三列（无 ``TimeSeconds``），
    故用 ``_time_seconds`` / ``_parse_timestamp_seconds`` 解析 ``TIMESTAMP``。``start`` 为
    **全局窗口起始编号**（并行分片）：输出窗口 ``wid`` 保留全局编号；``window_total`` 为整周
    划分的总窗口数（默认由 ``n_windows + start`` 推断），保证不同分片计算出的起点网格一致。

    返回 ``(windows, starts, n_total, partition, grid_rate)``。``partition`` 与 ``grid_rate``
    都基于**全体 grid 窗口**计算，因此各 worker 结果一致：前者作为公共签名的一部分，
    后者是模块 E 的统一时间缩放（避免各进程按自己的窗口子集重复计算而抹平窗口间强度差异）。
    """
    import pandas as pd

    data_dir = _REPO_ROOT / str(cfg.get("trace_dir", "data"))
    csv_path = data_dir / str(cfg.get("week_csv", "AzureLLMInferenceTrace_conv_1week.csv"))
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到 1 周 CSV：{csv_path}")

    # 1. 完整时间轴（相对秒）+ 总行数（单次仅读 TIMESTAMP 列）
    ts_parts: list[np.ndarray] = []
    for chunk in pd.read_csv(csv_path, chunksize=500_000, usecols=["TIMESTAMP"],
                             encoding="utf-8-sig"):
        ts_parts.append(_parse_timestamp_seconds(chunk["TIMESTAMP"]))
    ts_all = np.concatenate(ts_parts) if ts_parts else np.zeros(0, dtype=np.float64)
    n_total = int(ts_all.size)
    if n_total == 0:
        raise ValueError(f"1 周 CSV 为空：{csv_path}")
    ts_all = ts_all - ts_all[0]                       # 全局相对秒

    grid = int(window_total) if int(window_total) > 0 else int(n_windows) + int(start)
    all_starts = _week_long_starts(ts_all, grid, int(n_full))
    sel = range(int(start), int(start) + int(n_windows))
    starts = [all_starts[i] for i in sel]
    ranges = [(s, s + int(n_full)) for s in starts]
    print(f"  [week] rows={n_total:,} grid={grid} starts={starts[:3]}…"
          f"(+{max(0, len(starts) - 3)}) n_full={n_full}")

    # 全体 grid 窗口的公共身份（含来源与 pooled rate）——各 worker 一致
    partition = {
        "source": "week", "file": csv_path.name, "n_full": int(n_full), "grid": int(grid),
        "sources": [f"week:{csv_path.name}:rows={s}:{s + int(n_full)}" for s in all_starts],
    }
    grid_rate = _grid_pooled_rate(ts_all, all_starts, int(n_full))

    # 2. 单次流式抽取所有区间（坐标已定，时间轴直接切片，无需重解析）
    buckets: list[list] = [[] for _ in ranges]
    idx = 0
    for chunk in pd.read_csv(csv_path, chunksize=500_000,
                             usecols=["ContextTokens", "GeneratedTokens"],
                             encoding="utf-8-sig"):
        n = len(chunk)
        cend = idx + n
        for ri, (s, e) in enumerate(ranges):
            if e <= idx:
                continue
            if s >= cend:
                break
            rs = max(s, idx) - idx
            re_ = min(e, cend) - idx
            buckets[ri].append(chunk.iloc[rs:re_])
        idx = cend
        sys.stdout.write(f"\r  [week] read {idx:,}/{n_total:,}")
        sys.stdout.flush()
    sys.stdout.write("\n")

    wins: list[ReplayWindow] = []
    for gi, (b, (s, e)) in enumerate(zip(buckets, ranges)):
        df = pd.concat(b, ignore_index=True)
        wid = int(start) + gi
        key = f"week:{csv_path.name}:rows={s}:{e}"
        wins.append(_make_window(wid, ts_all[s:e],
                                 df["ContextTokens"].to_numpy(np.float64),
                                 df["GeneratedTokens"].to_numpy(np.float64), cfg,
                                 source_key=key))
    return wins, starts, n_total, partition, grid_rate


def _grid_pooled_rate(ts: np.ndarray, starts: list[int], n_full: int) -> float:
    """全体 grid 窗口的 pooled rate = Σn_i / Σspan_i。"""
    if not starts or n_full <= 0:
        return float("nan")
    span = 0.0
    for s in starts:
        span += float(ts[int(s) + int(n_full) - 1] - ts[int(s)])
    return float(len(starts) * int(n_full)) / span if span > 0 else float("inf")


# ── 逐请求守恒构成变换 / 标签 ──────────────────────────────────

def _compose_lengths(p_base: np.ndarray, d_base: np.ndarray, scale: float,
                     cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """逐请求总 token 守恒的构成变换（与 fig11/fig14 同一公式，数组实现）。"""
    p0 = np.asarray(p_base, dtype=np.float64)
    d0 = np.asarray(d_base, dtype=np.float64)
    total = p0 + d0
    p_cap = int(cfg.get("prompt_len_cap", 10 ** 9))
    a_cap = int(cfg.get("answer_len_cap", 10 ** 9))
    p_lo = np.maximum(float(_PROMPT_LOWER_BOUND), total - float(a_cap))
    p_hi = np.minimum(float(p_cap), total - float(_ANSWER_LOWER_BOUND))
    if float(scale) == 1.0:
        p_new = p0.astype(np.int64)
    else:
        a = float(scale)
        target = total * (a * p0) / (a * p0 + d0)
        p_new = np.minimum(p_hi, np.maximum(p_lo, np.round(target))).astype(np.int64)
    d_new = (total - p_new.astype(np.float64)).astype(np.int64)
    return p_new, d_new


def _label_stream(d_new: np.ndarray, threshold: float, *, err, window_id: int,
                  err_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """确定性 oracle 标签 + 可选的**方向性误判**（在最终真标签之后注入）。

    err = None            → oracle（predict == true）
    err = ("SL", eps)     → 真 short 中 eps 比例被预测为 long（单方向）
    err = ("LS", eps)     → 真 long 中 eps 比例被预测为 short（单方向）
    误判掩码仅由 (err_seed, window_id, 方向) 决定，**跨策略 / 容量共享**。
    """
    d = np.asarray(d_new, dtype=np.float64)
    lab = np.where(d <= float(threshold), "short", "long")
    if err is None:
        return lab, lab.copy()
    direction, eps = err
    rng = np.random.default_rng([int(err_seed), int(window_id), 0 if direction == "SL" else 1])
    u = rng.random(d.shape[0])
    pred = lab.copy()
    if direction == "SL":
        flip = (lab == "short") & (u < float(eps))
        pred[flip] = "long"
    else:
        flip = (lab == "long") & (u < float(eps))
        pred[flip] = "short"
    return lab, pred


def _make_request_generator(arrivals: np.ndarray, p: np.ndarray, d: np.ndarray,
                            ids: np.ndarray, true_lab: np.ndarray,
                            pred_lab: np.ndarray):
    """返回生成器：每次调用重建一份请求堆（每策略/容量独立，互不污染）。"""
    arr = np.asarray(arrivals, dtype=np.float64)
    n = int(arr.shape[0])

    def _gen(cfg: dict):
        heap = []
        for i in range(n):
            req = Request(
                id=int(ids[i]), arrival_time=float(arr[i]),
                prompt_len=int(p[i]), answer_len=int(d[i]),
                prefill_res_token=int(p[i]),
                true_label=str(true_lab[i]), predict_label=str(pred_lab[i]),
            )
            heap.append((req.arrival_time, req.id, req))
        heapq.heapify(heap)
        return heap

    return _gen


# ── 单次仿真 + 记录 ────────────────────────────────────────────

def _run_cell(cfg: dict, gen, policies: list[str], w: int, kv_capacity: int) -> dict:
    """在给定请求流上依次跑各策略，返回 ``{policy: {rec, window, finished}}``。"""
    original = simulator.generate_heapq
    simulator.generate_heapq = gen
    out: dict = {}
    try:
        for pol in policies:
            local = dict(cfg)
            local["schedule_method"] = pol
            local["KV_total_capacity"] = int(kv_capacity)
            metrics, state = simulator.run(local, return_state=True)
            rec: dict = {}
            rec.update(_extract_metrics(metrics))
            for k, v in _kv_diag(state).items():
                rec[k] = v
            for k, v in _exec_diag(state, pol).items():
                rec[k] = v
            for k, v in _long_admit_diag(state.finished_queue).items():
                rec[k] = v
            for k, v in _scalar_stream_diag(state.finished_queue).items():
                rec[k] = v
            mm = mechanism_metrics(state)
            for k in _MECH_KEYS:
                rec[k] = float(mm.get(k, float("nan")))
            win = {}
            for mode in _WINDOW_MODES:
                wa, edges = _window_arrays(state.finished_queue, state.batch_list, w, mode)
                win[mode] = (wa, edges)
            out[pol] = {"rec": rec, "window": win, "finished": state.finished_queue}
    finally:
        simulator.generate_heapq = original
    return out


def _kv_diag(state) -> dict:
    from scripts.experiment.sweep_kv_capacity import _kv_diag as _f
    return _f(state)


def _exec_diag(state, policy: str) -> dict:
    from scripts.experiment.sweep_kv_capacity import _exec_diag as _f
    return _f(state, policy)


def _long_admit_diag(finished) -> dict:
    from scripts.experiment.sweep_kv_capacity import _long_admit_diag as _f
    return _f(finished)


def _mask_wait_stats(finished, mask: np.ndarray) -> tuple[int, float, float]:
    """固定评价群体的 (人数, 平均等待, P99 等待)。"""
    if not finished:
        return 0, float("nan"), float("nan")
    ids = np.fromiter((int(r.id) for r in finished), dtype=np.int64, count=len(finished))
    waits = np.fromiter((float(r.waiting_time) for r in finished),
                        dtype=np.float64, count=len(finished))
    m = mask[ids]
    if not np.any(m):
        return 0, float("nan"), float("nan")
    wsub = waits[m]
    return int(wsub.size), float(wsub.mean()), float(np.percentile(wsub, 99))


def _true_long_stats(finished, d_arr: np.ndarray, thr: float) -> tuple[int, float, float]:
    """**按给定 p 阈值重新分组**统计 true-long 群体的 (人数, 平均等待, P99 等待)。

    调度不依赖标签，故 FCFS 一次运行的完成请求长度/等待可跨 p 复用；这里只是把同一批
    完成请求按 ``d_arr[id] > thr`` 重新归类为 true-long，不重跑仿真。
    """
    if not finished:
        return 0, float("nan"), float("nan")
    ids = np.fromiter((int(r.id) for r in finished), dtype=np.int64, count=len(finished))
    waits = np.fromiter((float(r.waiting_time) for r in finished),
                        dtype=np.float64, count=len(finished))
    m = d_arr[ids] > float(thr)
    if not np.any(m):
        return 0, float("nan"), float("nan")
    wsub = waits[m]
    return int(wsub.size), float(wsub.mean()), float(np.percentile(wsub, 99))


def _quad_stats(finished, p_arr: np.ndarray, d_arr: np.ndarray) -> np.ndarray:
    """模块 C 四群体统计，形状 (4, len(_QUAD_STATS))。

    空群体的人数与 promotion 数记 0（该类别确实没有请求），
    其余派生量（等待 / 组内平均 P,D,T）保持 NaN 以区分「无样本」。
    """
    out = np.full((4, len(_QUAD_STATS)), np.nan, dtype=np.float64)
    out[:, 0] = 0.0     # n
    out[:, 3] = 0.0     # promo_n
    if not finished:
        return out
    n = len(finished)
    ids = np.fromiter((int(r.id) for r in finished), dtype=np.int64, count=n)
    waits = np.fromiter((float(r.waiting_time) for r in finished), dtype=np.float64, count=n)
    promo = np.fromiter((bool(getattr(r, "promoted", False)) for r in finished),
                        dtype=bool, count=n)
    tru = np.fromiter((getattr(r, "true_label", "") for r in finished), dtype=object, count=n)
    prd = np.fromiter((getattr(r, "predict_label", "") for r in finished),
                      dtype=object, count=n)
    for qi, (tl, pl) in enumerate([("short", "short"), ("short", "long"),
                                   ("long", "short"), ("long", "long")]):
        m = np.fromiter(((tru[i] == tl and prd[i] == pl) for i in range(n)),
                        dtype=bool, count=n)
        if not np.any(m):
            continue
        sel = ids[m]
        ws = waits[m]
        out[qi, 0] = float(sel.size)
        out[qi, 1] = float(ws.mean())
        out[qi, 2] = float(np.percentile(ws, 99))
        out[qi, 3] = float(promo[m].sum())
        out[qi, 4] = float(p_arr[sel].mean())
        out[qi, 5] = float(d_arr[sel].mean())
        out[qi, 6] = float((p_arr[sel].astype(np.float64) + d_arr[sel]).mean())
    return out


# ── 结果容器 / 分片 / 恢复 / 合并 ──────────────────────────────

class _WindowStore:
    """按窗口维（axis 0）组织的扁平结果容器，统一支持切片 / 回填 / 合并。"""

    def __init__(self, n_windows: int, spec: dict, coords: dict):
        self.n = int(n_windows)
        self.spec = dict(spec)
        self.coords = dict(coords)
        self.data = {k: np.full((self.n,) + tuple(s), np.nan, dtype=np.float64)
                     for k, s in self.spec.items()}

    def slice_window(self, wi: int) -> dict:
        return {k: np.array(self.data[k][wi], copy=True) for k in self.data}

    def fill_window(self, wi: int, payload: dict) -> None:
        for k in self.data:
            self.data[k][wi] = payload[k]


def _win_ckpt_path(ckpt_dir: str, module: str, wi: int) -> str:
    return os.path.join(ckpt_dir, f"rt_{module}_win{int(wi):04d}.npz")


def _write_ckpt(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


def _load_npz(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _coords_match(payload: dict, coords: dict) -> bool:
    for k, v in coords.items():
        key = f"coord_{k}"
        if key not in payload:
            return False
        if not np.array_equal(np.asarray(payload[key]), np.asarray(v)):
            return False
    return True


def _win_digest(win) -> str:
    """单窗口内容指纹（最终请求数组的 md5 摘要）。"""
    h = hashlib.md5()
    for arr in (win.arrivals, win.p_base, win.d_base):
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _win_id_payload(win) -> dict:
    """单窗口**输入身份**（随分片保存，恢复 / 合并逐个全局窗口校验）。"""
    return {
        "win_source": np.asarray(str(win.source_key)),
        "win_digest": np.asarray(_win_digest(win)),
        "win_n": np.asarray([int(win.arrivals.shape[0])], dtype=np.int64),
        "win_span": np.asarray([round(float(win.span), 6)], dtype=np.float64),
    }


def _win_id_match(payload: dict, win) -> bool:
    if "win_source" not in payload or "win_digest" not in payload:
        return False
    return (str(payload["win_source"]) == str(win.source_key)
            and str(payload["win_digest"]) == _win_digest(win))


def _norm_value(v):
    """把配置值归一化为 JSON 安全类型（dict 递归、未知类型转 str）。"""
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return float(v)
    if isinstance(v, str) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _norm_value(x) for k, x in sorted(v.items())}
    if isinstance(v, (list, tuple)):
        return [_norm_value(x) for x in v]
    return str(v)


def _cfg_fingerprint(cfg: dict) -> dict:
    """**完整有效配置**指纹（规范化后逐字段）。

    手工列举参数总会漏项（budget_tokens / request_block_size / ageing_measure /
    batch-time / steady_state 等），故直接对整个 cfg 生成指纹，避免漏列导致静默复用。
    """
    return {str(k): _norm_value(cfg[k]) for k in sorted(cfg)}


def _run_signature(cfg: dict, args, module: str, partition: dict | None = None,
                   w: int | None = None) -> dict:
    """**公共运行签名**（跨分片一致；恢复 / 合并共用同一套校验）。

    分两层校验：
      - 公共层（本签名）：模块、err_seed、**统计分窗数** ``w``（来自 CLI，不在 cfg 中，
        故须单独纳入）、**完整有效配置**指纹、全局窗口网格 ``partition``（与内容无关的
        来源列表）与源码版本 —— 同一实验的所有 worker 必然一致，处理不同窗口的合法
        worker 因此仍可合并；
      - 单窗口层（随分片保存）：见 ``_win_id_payload``（每个全局窗口的来源 + 内容指纹）。
    换参数 / 换输入 / 换代码 / 换分窗数都会被拒绝，而"不同进程各处理一部分窗口"不会被误拒。
    """
    from scripts.experiment._common import _get_git_info
    git = _get_git_info()
    return {
        "module": str(module),
        "err_seed": int(args.err_seed),
        "stat_windows": (int(w) if w is not None else None),
        "config": _cfg_fingerprint(cfg),
        "partition": partition if partition is not None else {},
        "git_commit": str(git.get("git_commit") or ""),
        "git_dirty": bool(git.get("git_dirty") or False),
    }


def _sig_payload(sig: dict, run_meta: dict) -> dict:
    """把签名与原始运行 meta 序列化为 npz 可存（allow_pickle=False）的字符串数组。"""
    return {
        "sig_json": np.asarray(json.dumps(sig, sort_keys=True, ensure_ascii=False)),
        "meta_json": np.asarray(json.dumps(run_meta, sort_keys=True, ensure_ascii=False)),
    }


def _read_json_field(payload: dict, key: str) -> dict | None:
    if key not in payload:
        return None
    try:
        return json.loads(str(payload[key]))
    except (ValueError, TypeError):
        return None


def _sig_match(payload: dict, sig: dict) -> bool:
    stored = _read_json_field(payload, "sig_json")
    return stored is not None and stored == sig


# 分片内非数据键（不参与数组堆叠；单窗口身份 + 元信息）
_CKPT_META_KEYS = ("sig_json", "meta_json",
                   "win_source", "win_digest", "win_n", "win_span")


def _drive(module: str, store: _WindowStore, windows: list, run_window, *,
           ckpt_dir, resume, sig: dict, run_meta: dict) -> None:
    """逐窗口运行并落盘分片。

    分片文件名与 ``coord_window_index`` 一律使用**全局窗口编号 ``win.wid``**（而非进程内
    的 ``wi``），这样两个独立进程分别处理全局窗口 0 / 1 时不会互相覆盖，且能正确合并。
    恢复校验同时要求：公共坐标一致 + 公共签名一致 + 本窗口输入身份一致。
    """
    n = int(store.n)
    for wi in range(n):
        wid = int(windows[wi].wid)
        path = _win_ckpt_path(ckpt_dir, module, wid) if ckpt_dir else None
        if resume and path and os.path.exists(path):
            payload = _load_npz(path)
            if (_coords_match(payload, store.coords) and _sig_match(payload, sig)
                    and _win_id_match(payload, windows[wi])):
                store.fill_window(wi, payload)
                print(f"  resume: 命中分片 {os.path.basename(path)}，跳过 window {wid}")
                continue
            if not _coords_match(payload, store.coords):
                print(f"  warn: 分片坐标不一致，重算 window {wid}: {path}")
            elif not _sig_match(payload, sig):
                print(f"  warn: 分片运行签名不一致（参数/配置/代码已变），"
                      f"拒绝复用并重算 window {wid}: {path}")
            else:
                print(f"  warn: 分片输入身份不一致（输入文件/内容已变），"
                      f"拒绝复用并重算 window {wid}: {path}")
        print(f"  [{module}] window {wi + 1}/{n}")
        run_window(wi)
        if path:
            payload = store.slice_window(wi)
            for k, v in store.coords.items():
                payload[f"coord_{k}"] = np.asarray(v)
            payload["coord_window_index"] = np.asarray([wid], dtype=np.int64)
            payload.update(_win_id_payload(windows[wi]))
            payload.update(_sig_payload(sig, run_meta))
            _write_ckpt(path, payload)


def _module_arrays(store: _WindowStore, axis_arrays: dict) -> dict:
    arrays = dict(store.data)
    arrays.update(axis_arrays)
    arrays["n_windows"] = np.asarray([store.n], dtype=np.int64)
    return arrays


_COORD_AXIS_RENAME = {"scales": "composition_scales"}


def _axis_equal(a, b) -> bool:
    """轴相等比较：字符串轴按字符串比，数值轴按容差比（忽略 int/float 差异）。"""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "US" or b.dtype.kind in "US":
        return bool(np.array_equal(a.astype(str), b.astype(str)))
    return bool(np.allclose(a.astype(np.float64), b.astype(np.float64),
                            rtol=0.0, atol=1e-12))


def _axis_from_coords(coords: dict) -> dict:
    """从**已验证坐标**生成输出轴（避免把合并时 CLI 参数当作真实运行坐标）。"""
    out: dict = {}
    for k, v in coords.items():
        if k in ("module", "n_windows"):
            continue
        out[_COORD_AXIS_RENAME.get(k, k)] = np.asarray(v)
    return out


def _merge_module(module: str, ckpt_dirs: list[str], cli_axis: dict | None = None
                  ) -> tuple[dict, dict, dict]:
    """合并分片：校验运行签名 + 坐标，从坐标生成输出轴，返回 (arrays, sig, run_meta)。

    若同时给出 ``cli_axis``，其与 checkpoint 坐标重合的键必须完全一致，否则拒绝合并。
    """
    paths: list[str] = []
    for d in ckpt_dirs:
        if os.path.isdir(d):
            paths.extend(sorted(os.path.join(d, f) for f in os.listdir(d)
                                if f.startswith(f"rt_{module}_win") and f.endswith(".npz")
                                and ".tmp." not in f))
        elif os.path.isfile(d):
            paths.append(d)
        else:
            raise FileNotFoundError(f"分片目录/文件不存在：{d}")
    if not paths:
        raise FileNotFoundError(f"未找到 {module} 分片（pattern: rt_{module}_win*.npz）")
    payloads = [_load_npz(p) for p in paths]

    # 1. 公共签名一致性（err_seed / 完整有效配置 / 全局网格 partition / 代码版本）
    ref_sig = _read_json_field(payloads[0], "sig_json")
    if ref_sig is None:
        raise ValueError(f"分片缺少运行签名 sig_json（旧格式，需重跑）：{paths[0]}")
    ref_meta = _read_json_field(payloads[0], "meta_json") or {}
    for pi, pl in enumerate(payloads[1:], 1):
        sj = _read_json_field(pl, "sig_json")
        if sj != ref_sig:
            raise ValueError(
                f"分片运行签名不一致（分片 0 vs 分片 {pi}）：{sj} != {ref_sig}")

    # 1b. 单窗口输入身份：每个全局窗口的来源必须与公共网格一致
    grid_sources = list((ref_sig.get("partition") or {}).get("sources") or [])
    for pi, pl in enumerate(payloads):
        wid = int(np.asarray(pl["coord_window_index"]).reshape(-1)[0])
        if not grid_sources:
            continue
        if wid < 0 or wid >= len(grid_sources):
            raise ValueError(f"分片窗口编号 {wid} 超出全局网格 {len(grid_sources)}（分片 {pi}）")
        src = str(pl["win_source"]) if "win_source" in pl else ""
        if src != grid_sources[wid]:
            raise ValueError(
                f"分片输入身份与全局网格不符：window {wid} 期望 {grid_sources[wid]}，实际 {src}")

    # 2. 坐标一致性（window_index 逐分片不同，不参与比较）
    ref_coords = {k[len("coord_"):]: v for k, v in payloads[0].items()
                  if k.startswith("coord_") and k != "coord_window_index"}
    for pi, pl in enumerate(payloads[1:], 1):
        c = {k[len("coord_"):]: v for k, v in pl.items()
             if k.startswith("coord_") and k != "coord_window_index"}
        for k, v in ref_coords.items():
            if k not in c or not np.array_equal(np.asarray(c[k]), np.asarray(v)):
                raise ValueError(f"分片坐标不一致：{k}（分片 0 vs 分片 {pi}）")

    # 3. 输出轴从已验证坐标生成；CLI 若给出重合坐标必须一致
    axis_arrays = _axis_from_coords(ref_coords)
    if cli_axis:
        for k, v in cli_axis.items():
            if k in axis_arrays and not _axis_equal(axis_arrays[k], v):
                raise ValueError(
                    f"合并轴与分片坐标不符：{k}（CLI={np.asarray(v)} vs "
                    f"ckpt={np.asarray(axis_arrays[k])}）")
            axis_arrays.setdefault(k, np.asarray(v))

    # 窗口编号
    wins = sorted(int(np.asarray(pl["coord_window_index"]).reshape(-1)[0]) for pl in payloads)
    if len(set(wins)) != len(wins):
        raise ValueError(f"分片存在重复窗口编号：{wins}")
    order = np.argsort([int(np.asarray(pl["coord_window_index"]).reshape(-1)[0])
                        for pl in payloads])
    ordered = [payloads[i] for i in order]
    data_keys = [k for k in ordered[0].keys()
                 if not k.startswith("coord_") and k not in _CKPT_META_KEYS]
    n_w = len(ordered)
    arrays: dict = {}
    for k in data_keys:
        sample = np.asarray(ordered[0][k], dtype=np.float64)
        arrays[k] = np.full((n_w,) + sample.shape, np.nan, dtype=np.float64)
    for wi, pl in enumerate(ordered):
        for k in data_keys:
            arrays[k][wi] = np.asarray(pl[k], dtype=np.float64)
    arrays.update(axis_arrays)
    arrays["windows"] = np.asarray(wins, dtype=np.int64)
    arrays["n_windows"] = np.asarray([n_w], dtype=np.int64)
    print(f"merge: module={module} windows={wins}")
    return arrays, ref_sig, ref_meta


# ── 通用扫参驱动 ──────────────────────────────────────────────

def _scaled_arrivals(win: ReplayWindow, scale: float) -> np.ndarray:
    return win.arrivals * float(scale)


def _local_cfg(cfg: dict, lam: float, n_take: int, seed: int) -> dict:
    local = dict(cfg)
    local.update({
        "lambda_arrival": float(lam),
        "admission_number": int(n_take),
        "seed": int(seed),
    })
    return local


def _ckpt_run_meta(module: str, cfg: dict, args) -> dict:
    """checkpoint 内保存的**原始运行 meta**（合并时继承，避免被 merge 参数替代）。"""
    from scripts.experiment._common import _get_git_info
    meta = _common_meta(module, cfg, args)
    meta["args"] = {k: v for k, v in vars(args).items()}
    meta["git"] = _get_git_info()
    return meta


# ── 模块 A：构成 × KV × 到达率 ─────────────────────────────────

def _module_a_spec(policies, w, n_a, n_K, n_l) -> dict:
    from scripts.experiment.sweep_kv_capacity import _POLICY_SUBDICTS as PS
    spec: dict = {}
    for p in policies:
        pfx = _POLICY_PREFIX_MAP[p]
        for _name, keys in PS:
            for k in keys:
                spec[f"{pfx}_{k}"] = (n_a, n_K, n_l)
        for mode in _WINDOW_MODES:
            for k in _WINDOW_KEYS:
                spec[f"{pfx}_{mode}_{k}"] = (n_a, n_K, n_l, w)
            spec[f"{pfx}_{mode}_edges"] = (n_a, n_K, n_l, w + 1)
        for k in _FIXED_GROUP_KEYS:
            spec[f"{pfx}_fixed_{k}"] = (n_a, n_K, n_l)
    for k in _LONG_TRANSFORM_KEYS:
        spec[f"transform_{k}"] = (n_a,)
    for k in _PREFIX_DIAG_KEYS:
        spec[f"prefix_{k}"] = (n_a,)
    for k in _PRE_STAT_KEYS:
        spec[f"pre_{k}"] = ()
    return spec


def _run_module_a(cfg, windows, scales, caps, loads, policies, w, args, partition=None):
    from scripts.experiment.sweep_kv_capacity import _POLICY_SUBDICTS as PS
    n_a, n_K, n_l = len(scales), len(caps), len(loads)
    coords = {
        "module": np.asarray([ord("A")]),
        "scales": np.asarray(scales, dtype=np.float64),
        "capacities": np.asarray(caps, dtype=np.int64),
        "lambdas": np.asarray(loads, dtype=np.float64),
    }
    store = _WindowStore(len(windows), _module_a_spec(policies, w, n_a, n_K, n_l), coords)

    def run_window(wi: int) -> None:
        win = windows[wi]
        for k, v in _compute_pre_stats(win, cfg).items():
            store.data[f"pre_{k}"][wi] = float(v)
        for ai, a in enumerate(scales.tolist()):
            p_new, d_new = _compose_lengths(win.p_base, win.d_base, float(a), cfg)
            thr = float(np.quantile(d_new.astype(np.float64),
                                    float(cfg["label_threshold_percentail"])))
            _, pi_s = _prefix_short_ratio(d_new, d_new.shape[0], thr)
            store.data["prefix_pi_s"][wi, ai] = pi_s
            store.data["prefix_n_short"][wi, ai] = float(np.sum(d_new <= thr))
            _fill_transform(store, wi, ai, p_new, d_new, thr, win, cfg)
            true_lab, _ = _label_stream(d_new, thr, err=None, window_id=win.wid,
                                        err_seed=args.err_seed)
            for li, lam in enumerate(loads.tolist()):
                scale = win.raw_lambda / float(lam)
                arr = _scaled_arrivals(win, scale)
                gen = _make_request_generator(arr, p_new, d_new, win.ids, true_lab, true_lab)
                for ki, K in enumerate(caps.tolist()):
                    local = _local_cfg(cfg, float(lam), d_new.shape[0], win.wid)
                    res = _run_cell(local, gen, policies, w, int(K))
                    for pol in policies:
                        pfx = _POLICY_PREFIX_MAP[pol]
                        _store_policy(store, wi, (ai, ki, li), pfx, res[pol], PS)
                        _store_fixed(store, wi, (ai, ki, li), pfx, res[pol]["finished"], win)

    _drive("A", store, windows, run_window, ckpt_dir=args.ckpt_dir, resume=bool(args.resume),
           sig=_run_signature(cfg, args, "A", partition, w),
           run_meta=_ckpt_run_meta("A", cfg, args))
    axis = {
        "composition_scales": np.asarray(scales, dtype=np.float64),
        "capacities": np.asarray(caps, dtype=np.int64),
        "lambdas": np.asarray(loads, dtype=np.float64),
        "windows": np.asarray([win.wid for win in windows], dtype=np.int64),
    }
    return _module_arrays(store, axis)


def _fill_transform(store, wi, ai, p_new, d_new, thr, win, cfg):
    from scripts.experiment.sweep_request_composition import (
        _PROMPT_LOWER_BOUND as PLO, _ANSWER_LOWER_BOUND as ALO,
    )
    total = win.total_base
    realized = p_new.astype(np.float64) + d_new.astype(np.float64)
    mp = float(p_new.mean())
    md = float(d_new.mean())
    store.data["transform_prompt_mean"][wi, ai] = mp
    store.data["transform_answer_mean"][wi, ai] = md
    store.data["transform_token_ratio"][wi, ai] = mp / md if md > 0 else float("nan")
    store.data["transform_total_mean"][wi, ai] = float(realized.mean())
    store.data["transform_total_var"][wi, ai] = float(realized.var())
    store.data["transform_max_request_total_error"][wi, ai] = float(
        np.max(np.abs(realized - total)))
    store.data["transform_lower_bound_fraction"][wi, ai] = float(
        np.mean(p_new.astype(np.float64)
                <= np.maximum(float(PLO), total - float(cfg["answer_len_cap"])) + 1e-9))
    store.data["transform_upper_bound_fraction"][wi, ai] = float(
        np.mean(p_new.astype(np.float64)
                >= np.minimum(float(cfg["prompt_len_cap"]), total - float(ALO)) - 1e-9))
    store.data["transform_label_threshold"][wi, ai] = float(thr)


def _store_policy(store, wi, idx, pfx, entry, PS) -> None:
    rec = entry["rec"]
    for _name, keys in PS:
        for k in keys:
            store.data[f"{pfx}_{k}"][wi][idx] = float(rec[k])
    for mode in _WINDOW_MODES:
        wa, edges = entry["window"][mode]
        for k in _WINDOW_KEYS:
            store.data[f"{pfx}_{mode}_{k}"][wi][idx] = wa[k]
        store.data[f"{pfx}_{mode}_edges"][wi][idx] = edges


def _store_fixed(store, wi, idx, prefix: str, finished, win) -> None:
    """固定评价群体统计（按策略分别记录；请求 ID 跨策略共享，等待随策略不同）。"""
    n_a, m_a, p_a = _mask_wait_stats(finished, win.fixed_ans_mask)
    n_t, m_t, p_t = _mask_wait_stats(finished, win.fixed_tot_mask)
    store.data[f"{prefix}_fixed_fx_ans_long_n"][wi][idx] = float(n_a)
    store.data[f"{prefix}_fixed_fx_ans_long_wait_mean"][wi][idx] = m_a
    store.data[f"{prefix}_fixed_fx_ans_long_wait_p99"][wi][idx] = p_a
    store.data[f"{prefix}_fixed_fx_tot_long_n"][wi][idx] = float(n_t)
    store.data[f"{prefix}_fixed_fx_tot_long_wait_mean"][wi][idx] = m_t
    store.data[f"{prefix}_fixed_fx_tot_long_wait_p99"][wi][idx] = p_t


# ── 模块 B：p × k（quota 耦合迁移）─────────────────────────────

def _module_b_spec(w, n_K, n_p, n_k) -> dict:
    from scripts.experiment.sweep_kv_capacity import _POLICY_SUBDICTS as PS
    spec: dict = {}
    for p, shape in [("QBA", (n_K, n_p, n_k)), ("PSF", (n_K, n_p)),
                     ("AA", (n_K, n_p)), ("FCFS", (n_K,))]:
        pfx = _POLICY_PREFIX_MAP[p]
        for _name, keys in PS:
            for k in keys:
                spec[f"{pfx}_{k}"] = shape
        for mode in _WINDOW_MODES:
            for k in _WINDOW_KEYS:
                spec[f"{pfx}_{mode}_{k}"] = shape + (w,)
            spec[f"{pfx}_{mode}_edges"] = shape + (w + 1,)
        for k in _FIXED_GROUP_KEYS:
            spec[f"{pfx}_fixed_{k}"] = shape
    spec["b_pi_s"] = (n_p,)
    spec["b_thr"] = (n_p,)
    spec["b_n_short_hat"] = (n_p,)
    spec["b_n_long_hat"] = (n_p,)
    spec["b_n_pred_short"] = (n_p,)
    spec["b_chi_pred"] = (n_p, n_k)
    # FCFS 整体指标与固定评价群体跨 p 复用（调度不依赖标签）；但 **true-long 群体随 p 改变**，
    # 需按各 p 阈值对同一批完成请求重新分组统计（不重跑 FCFS）。
    spec["fcfs_tl_group_n"] = (n_K, n_p)
    spec["fcfs_tl_group_wait_mean"] = (n_K, n_p)
    spec["fcfs_tl_group_wait_p99"] = (n_K, n_p)
    spec["transform_token_ratio"] = ()
    spec["transform_total_mean"] = ()
    spec["transform_max_request_total_error"] = ()
    for k in _PRE_STAT_KEYS:
        spec[f"pre_{k}"] = ()
    return spec


def _run_module_b(cfg, windows, caps, ps, ks, lam, policies, w, args, partition=None):
    from scripts.experiment.sweep_kv_capacity import _POLICY_SUBDICTS as PS
    n_K, n_p, n_k = len(caps), len(ps), len(ks)
    coords = {
        "module": np.asarray([ord("B")]),
        "capacities": np.asarray(caps, dtype=np.int64),
        "ps": np.asarray(ps, dtype=np.float64),
        "ks": np.asarray(ks, dtype=np.int64),
        "lambda": np.asarray([float(lam)]),
    }
    store = _WindowStore(len(windows), _module_b_spec(w, n_K, n_p, n_k), coords)

    def run_window(wi: int) -> None:
        win = windows[wi]
        for k, v in _compute_pre_stats(win, cfg).items():
            store.data[f"pre_{k}"][wi] = float(v)
        p_new, d_new = win.p_base.copy(), win.d_base.copy()   # a = 1，保留整数
        realized = p_new.astype(np.float64) + d_new.astype(np.float64)
        store.data["transform_total_mean"][wi] = float(realized.mean())
        store.data["transform_max_request_total_error"][wi] = float(
            np.max(np.abs(realized - win.total_base)))
        store.data["transform_token_ratio"][wi] = float(p_new.mean()) / float(d_new.mean())
        scale = win.raw_lambda / float(lam)
        arr = _scaled_arrivals(win, scale)
        # FCFS 与标签无关 → 每个 K 只跑一次（整体指标 + 固定评价群体可跨 p 复用）
        tl_a1, _ = _label_stream(d_new, win.label_a1_thr(), err=None,
                                 window_id=win.wid, err_seed=args.err_seed)
        gen_base = _make_request_generator(arr, p_new, d_new, win.ids, tl_a1, tl_a1)
        d_f64 = d_new.astype(np.float64)
        thr_by_p = [float(np.quantile(d_f64, float(p))) for p in ps.tolist()]
        for ki, K in enumerate(caps.tolist()):
            local = _local_cfg(cfg, float(lam), d_new.shape[0], win.wid)
            res = _run_cell(local, gen_base, ["FCFS"], w, int(K))
            _store_policy(store, wi, (ki,), "fcfs", res["FCFS"], PS)
            finished = res["FCFS"]["finished"]
            _store_fixed(store, wi, (ki,), "fcfs", finished, win)
            # true-long 群体随 p 改变：对同一批完成请求按各 p 阈值重新分组（不重跑 FCFS）
            for pi, thr_p in enumerate(thr_by_p):
                n_tl, m_tl, p_tl = _true_long_stats(finished, d_f64, thr_p)
                store.data["fcfs_tl_group_n"][wi, ki, pi] = float(n_tl)
                store.data["fcfs_tl_group_wait_mean"][wi, ki, pi] = m_tl
                store.data["fcfs_tl_group_wait_p99"][wi, ki, pi] = p_tl
        # PSF/AA 依赖 p（标签）但不依赖 k
        for pi, p in enumerate(ps.tolist()):
            thr = float(np.quantile(d_new.astype(np.float64), float(p)))
            _, pi_s = _prefix_short_ratio(d_new, d_new.shape[0], thr)
            store.data["b_pi_s"][wi, pi] = pi_s
            store.data["b_thr"][wi, pi] = thr
            tl, _ = _label_stream(d_new, thr, err=None, window_id=win.wid,
                                  err_seed=args.err_seed)
            n_short = int(np.sum(tl == "short"))
            n_long = int(np.sum(tl == "long"))
            store.data["b_n_short_hat"][wi, pi] = float(n_short)
            store.data["b_n_long_hat"][wi, pi] = float(n_long)
            store.data["b_n_pred_short"][wi, pi] = float(n_short)
            for kk, k in enumerate(ks.tolist()):
                store.data["b_chi_pred"][wi, pi, kk] = (
                    float(n_short) / (float(k) * float(n_long)) if n_long > 0 else float("nan"))
            gen = _make_request_generator(arr, p_new, d_new, win.ids, tl, tl)
            for ki, K in enumerate(caps.tolist()):
                local = _local_cfg(cfg, float(lam), d_new.shape[0], win.wid)
                res_kn = _run_cell(local, gen, ["PSF", "AA"], w, int(K))
                _store_policy(store, wi, (ki, pi), "psf", res_kn["PSF"], PS)
                _store_fixed(store, wi, (ki, pi), "psf", res_kn["PSF"]["finished"], win)
                _store_policy(store, wi, (ki, pi), "psfa", res_kn["AA"], PS)
                _store_fixed(store, wi, (ki, pi), "psfa", res_kn["AA"]["finished"], win)
                for kk, k in enumerate(ks.tolist()):
                    local_k = dict(local)
                    local_k["quota_k"] = int(k)
                    res_q = _run_cell(local_k, gen, ["QBA"], w, int(K))
                    _store_policy(store, wi, (ki, pi, kk), "psfaq", res_q["QBA"], PS)
                    _store_fixed(store, wi, (ki, pi, kk), "psfaq",
                                 res_q["QBA"]["finished"], win)

    _drive("B", store, windows, run_window, ckpt_dir=args.ckpt_dir, resume=bool(args.resume),
           sig=_run_signature(cfg, args, "B", partition, w),
           run_meta=_ckpt_run_meta("B", cfg, args))
    axis = {
        "capacities": np.asarray(caps, dtype=np.int64),
        "ps": np.asarray(ps, dtype=np.float64),
        "ks": np.asarray(ks, dtype=np.int64),
        "lambda": np.asarray([float(lam)], dtype=np.float64),
        "windows": np.asarray([win.wid for win in windows], dtype=np.int64),
    }
    return _module_arrays(store, axis)


def _label_a1_thr(self) -> float:  # 便捷访问（避免每处重复取分位）
    return self.label_threshold_a1


ReplayWindow.label_a1_thr = _label_a1_thr  # type: ignore[attr-defined]


# ── 模块 C：方向性误判 + 四群体 ────────────────────────────────

_ERR_CONFIGS = [("oracle", None), ("SL0.1", ("SL", 0.1)), ("SL0.3", ("SL", 0.3)),
                ("LS0.1", ("LS", 0.1)), ("LS0.3", ("LS", 0.3))]


def _module_c_spec(w, n_K, n_err) -> dict:
    from scripts.experiment.sweep_kv_capacity import _POLICY_SUBDICTS as PS
    spec: dict = {}
    for p in _ALL_POLICIES:
        pfx = _POLICY_PREFIX_MAP[p]
        for _name, keys in PS:
            for k in keys:
                spec[f"{pfx}_{k}"] = (n_K, n_err)
        for mode in _WINDOW_MODES:
            for k in _WINDOW_KEYS:
                spec[f"{pfx}_{mode}_{k}"] = (n_K, n_err, w)
            spec[f"{pfx}_{mode}_edges"] = (n_K, n_err, w + 1)
        spec[f"quad_{pfx}"] = (n_K, n_err, 4, len(_QUAD_STATS))
        for k in _FIXED_GROUP_KEYS:
            spec[f"{pfx}_fixed_{k}"] = (n_K, n_err)
    for k in _PRE_STAT_KEYS:
        spec[f"pre_{k}"] = ()
    return spec


def _run_module_c(cfg, windows, caps, errs, lam, p, k, policies, w, args, partition=None):
    from scripts.experiment.sweep_kv_capacity import _POLICY_SUBDICTS as PS
    n_K, n_err = len(caps), len(errs)
    coords = {
        "module": np.asarray([ord("C")]),
        "capacities": np.asarray(caps, dtype=np.int64),
        "err_names": np.asarray([e[0] for e in errs]),
        "p": np.asarray([float(p)]),
        "k": np.asarray([int(k)]),
        "lambda": np.asarray([float(lam)]),
    }
    store = _WindowStore(len(windows), _module_c_spec(w, n_K, n_err), coords)

    def run_window(wi: int) -> None:
        win = windows[wi]
        for kk, v in _compute_pre_stats(win, cfg).items():
            store.data[f"pre_{kk}"][wi] = float(v)
        p_new, d_new = win.p_base.copy(), win.d_base.copy()   # a = 1
        thr = float(np.quantile(d_new.astype(np.float64), float(p)))
        scale = win.raw_lambda / float(lam)
        arr = _scaled_arrivals(win, scale)
        for ei, (_name, err) in enumerate(errs):
            tl, pl = _label_stream(d_new, thr, err=err, window_id=win.wid,
                                   err_seed=args.err_seed)
            gen = _make_request_generator(arr, p_new, d_new, win.ids, tl, pl)
            for ki, K in enumerate(caps.tolist()):
                local = _local_cfg(cfg, float(lam), d_new.shape[0], win.wid)
                local["quota_k"] = int(k)
                res = _run_cell(local, gen, policies, w, int(K))
                for pol in policies:
                    pfx = _POLICY_PREFIX_MAP[pol]
                    _store_policy(store, wi, (ki, ei), pfx, res[pol], PS)
                    _store_fixed(store, wi, (ki, ei), pfx, res[pol]["finished"], win)
                    store.data[f"quad_{pfx}"][wi, ki, ei] = _quad_stats(
                        res[pol]["finished"], p_new, d_new)

    _drive("C", store, windows, run_window, ckpt_dir=args.ckpt_dir, resume=bool(args.resume),
           sig=_run_signature(cfg, args, "C", partition, w),
           run_meta=_ckpt_run_meta("C", cfg, args))
    axis = {
        "capacities": np.asarray(caps, dtype=np.int64),
        "err_names": np.asarray([e[0] for e in errs]),
        "p": np.asarray([float(p)], dtype=np.float64),
        "k": np.asarray([int(k)], dtype=np.int64),
        "lambda": np.asarray([float(lam)], dtype=np.float64),
        "windows": np.asarray([win.wid for win in windows], dtype=np.int64),
    }
    return _module_arrays(store, axis)


# ── 模块 D/E：连续长流 ─────────────────────────────────────────

def _module_d_spec(w, n_N, n_a, n_K, n_l) -> dict:
    from scripts.experiment.sweep_kv_capacity import _POLICY_SUBDICTS as PS
    spec: dict = {}
    for p in _ALL_POLICIES:
        pfx = _POLICY_PREFIX_MAP[p]
        for _name, keys in PS:
            for k in keys:
                spec[f"{pfx}_{k}"] = (n_N, n_a, n_K, n_l)
        for mode in _WINDOW_MODES:
            for k in _WINDOW_KEYS:
                spec[f"{pfx}_{mode}_{k}"] = (n_N, n_a, n_K, n_l, w)
            spec[f"{pfx}_{mode}_edges"] = (n_N, n_a, n_K, n_l, w + 1)
        for k in _FIXED_GROUP_KEYS:
            spec[f"{pfx}_fixed_{k}"] = (n_N, n_a, n_K, n_l)
    for k in _LONG_TRANSFORM_KEYS:
        spec[f"transform_{k}"] = (n_a,)
    for k in _PREFIX_DIAG_KEYS:
        spec[f"prefix_{k}"] = (n_a, n_N)
    for k in _PRE_STAT_KEYS:
        spec[f"pre_{k}"] = ()
    return spec


def _run_module_d(cfg, windows, lengths, scales, caps, loads, policies, w, args,
                  partition=None):
    from scripts.experiment.sweep_kv_capacity import _POLICY_SUBDICTS as PS
    n_N, n_a, n_K, n_l = len(lengths), len(scales), len(caps), len(loads)
    coords = {
        "module": np.asarray([ord("D")]),
        "lengths": np.asarray(lengths, dtype=np.int64),
        "scales": np.asarray(scales, dtype=np.float64),
        "capacities": np.asarray(caps, dtype=np.int64),
        "lambdas": np.asarray(loads, dtype=np.float64),
    }
    store = _WindowStore(len(windows), _module_d_spec(w, n_N, n_a, n_K, n_l), coords)

    def run_window(wi: int) -> None:
        win = windows[wi]
        for k, v in _compute_pre_stats(win, cfg).items():
            store.data[f"pre_{k}"][wi] = float(v)
        for ai, a in enumerate(scales.tolist()):
            p_new, d_new = _compose_lengths(win.p_base, win.d_base, float(a), cfg)
            thr = float(np.quantile(d_new.astype(np.float64),
                                    float(cfg["label_threshold_percentail"])))
            # 变换诊断基于完整区间（与策略/K/λ/N 无关）；复用 _fill_transform 以补齐
            # lower/upper_bound_fraction 诊断（旧版此处恒为 NaN）。
            _fill_transform(store, wi, ai, p_new, d_new, thr, win, cfg)
            tl, _ = _label_stream(d_new, thr, err=None, window_id=win.wid,
                                  err_seed=args.err_seed)
            for ni, n_take in enumerate(lengths.tolist()):
                n_take = int(n_take)
                _, pi_s = _prefix_short_ratio(d_new, n_take, thr)
                store.data["prefix_pi_s"][wi, ai, ni] = pi_s
                store.data["prefix_n_short"][wi, ai, ni] = float(np.sum(d_new[:n_take] <= thr))
                for li, lam in enumerate(loads.tolist()):
                    scale = win.raw_lambda / float(lam)
                    arr = _scaled_arrivals(win, scale)
                    gen = _make_request_generator(arr[:n_take], p_new[:n_take], d_new[:n_take],
                                                  win.ids[:n_take], tl[:n_take], tl[:n_take])
                    for ki, K in enumerate(caps.tolist()):
                        local = _local_cfg(cfg, float(lam), n_take, win.wid)
                        res = _run_cell(local, gen, policies, w, int(K))
                        for pol in policies:
                            pfx = _POLICY_PREFIX_MAP[pol]
                            _store_policy(store, wi, (ni, ai, ki, li), pfx, res[pol], PS)
                            # 固定评价群体掩码在**完整参考窗口**（a=1 基线）确定，
                            # 这里按前缀 ID 统计（win.ids[:n_take] == arange(n_take)）。
                            _store_fixed(store, wi, (ni, ai, ki, li), pfx,
                                         res[pol]["finished"], win)

    _drive("D", store, windows, run_window, ckpt_dir=args.ckpt_dir, resume=bool(args.resume),
           sig=_run_signature(cfg, args, "D", partition, w),
           run_meta=_ckpt_run_meta("D", cfg, args))
    axis = {
        "lengths": np.asarray(lengths, dtype=np.int64),
        "composition_scales": np.asarray(scales, dtype=np.float64),
        "capacities": np.asarray(caps, dtype=np.int64),
        "lambdas": np.asarray(loads, dtype=np.float64),
        "windows": np.asarray([win.wid for win in windows], dtype=np.int64),
    }
    return _module_arrays(store, axis)


def _module_e_spec(w, n_K) -> dict:
    from scripts.experiment.sweep_kv_capacity import _POLICY_SUBDICTS as PS
    spec: dict = {}
    for p in _ALL_POLICIES:
        pfx = _POLICY_PREFIX_MAP[p]
        for _name, keys in PS:
            for k in keys:
                spec[f"{pfx}_{k}"] = (n_K,)
        for mode in _WINDOW_MODES:
            for k in _WINDOW_KEYS:
                spec[f"{pfx}_{mode}_{k}"] = (n_K, w)
            spec[f"{pfx}_{mode}_edges"] = (n_K, w + 1)
        for k in _FIXED_GROUP_KEYS:
            spec[f"{pfx}_fixed_{k}"] = (n_K,)
    spec["e_lambda_eff"] = ()
    for k in _PRE_STAT_KEYS:
        spec[f"pre_{k}"] = ()
    return spec


def _run_module_e(cfg, windows, caps, shared_time_scale, policies, w, args,
                  partition=None):
    from scripts.experiment.sweep_kv_capacity import _POLICY_SUBDICTS as PS
    n_K = len(caps)
    coords = {
        "module": np.asarray([ord("E")]),
        "capacities": np.asarray(caps, dtype=np.int64),
        "shared_time_scale": np.asarray([float(shared_time_scale)]),
    }
    store = _WindowStore(len(windows), _module_e_spec(w, n_K), coords)

    def run_window(wi: int) -> None:
        win = windows[wi]
        for k, v in _compute_pre_stats(win, cfg).items():
            store.data[f"pre_{k}"][wi] = float(v)
        store.data["e_lambda_eff"][wi] = win.raw_lambda / float(shared_time_scale)
        p_new, d_new = win.p_base.copy(), win.d_base.copy()   # a = 1
        thr = float(np.quantile(d_new.astype(np.float64),
                                float(cfg["label_threshold_percentail"])))
        tl, _ = _label_stream(d_new, thr, err=None, window_id=win.wid,
                              err_seed=args.err_seed)
        arr = win.arrivals * float(shared_time_scale)
        gen = _make_request_generator(arr, p_new, d_new, win.ids, tl, tl)
        lam_eff = win.raw_lambda / float(shared_time_scale)
        for ki, K in enumerate(caps.tolist()):
            local = _local_cfg(cfg, float(lam_eff), d_new.shape[0], win.wid)
            res = _run_cell(local, gen, policies, w, int(K))
            for pol in policies:
                pfx = _POLICY_PREFIX_MAP[pol]
                _store_policy(store, wi, (ki,), pfx, res[pol], PS)
                _store_fixed(store, wi, (ki,), pfx, res[pol]["finished"], win)

    _drive("E", store, windows, run_window, ckpt_dir=args.ckpt_dir, resume=bool(args.resume),
           sig=_run_signature(cfg, args, "E", partition, w),
           run_meta=_ckpt_run_meta("E", cfg, args))
    axis = {
        "capacities": np.asarray(caps, dtype=np.int64),
        "shared_time_scale": np.asarray([float(shared_time_scale)], dtype=np.float64),
        "windows": np.asarray([win.wid for win in windows], dtype=np.int64),
    }
    return _module_arrays(store, axis)


# ── 保存 ──────────────────────────────────────────────────────

def _save_module(module: str, out_dir: str, arrays: dict, cfg: dict, args,
                 cfg_path: Path, extra_meta: dict) -> tuple[str, str]:
    npz_path, meta_path = save_results(
        out_dir, f"rt_{module}", arrays, cfg, vars(args),
        config_path=str(cfg_path), extra_meta=extra_meta,
    )
    print(f"saved data: {npz_path}")
    print(f"saved meta: {meta_path}")
    return npz_path, meta_path


def _common_meta(module: str, cfg: dict, args) -> dict:
    return {
        "experiment": "scaled_real_trace_replay",
        "module": module,
        "positioning": ("scaled real-trace replay: real arrival shape + length relation after a "
                        "single uniform scaling, replayed in the current simulator; validates "
                        "transfer of input distribution, not a real-GPU system validation"),
        "policies": _ALL_POLICIES,
        "token_scale": float(cfg.get("token_scale", 1.0)),
        "label_threshold_percentail": float(cfg["label_threshold_percentail"]),
        "safety_ratio": float(cfg["KV_safety_threshold"]),
        "normalization": "per-request total-token preserving composition transform (a=1 identity)",
        "arrival_note": ("simultaneous arrivals preserved; no re-generated Poisson; zero "
                         "inter-arrival gaps kept"),
        "pre_stat_note": "pre-stats computed from the final (integerized) request arrays",
        "fixed_group_note": ("baseline a=1 answer-top ~15% and total-top ~5%; request IDs shared "
                             "across a/K/policy"),
        "loader_trap_note": ("time scale and label threshold determined on the FULL interval, "
                             "then frozen and prefix-sliced (module D/E)"),
    }


# ── main ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="scaled real-trace replay：Fig11–15 机制的真实轨迹迁移验证（模块 A–E）")
    parser.add_argument("--module", choices=["A", "B", "C", "D", "E", "all"], default="all")
    # 短窗口
    parser.add_argument("--n-windows", type=int, default=100, help="短窗口数量（subset CSV 数）")
    parser.add_argument("--window-start", type=int, default=0,
                        help="全局窗口起始编号（并行分片：取第 [start, start+n) 个窗口，"
                             "窗口 ID 保留全局编号以保证掩码/评估群体对齐）")
    parser.add_argument("--window-total", type=int, default=0,
                        help="全局窗口总网格数（长窗口分片时用于对齐起点网格；0=自动推断）")
    parser.add_argument("--short-max-n", type=int, default=None,
                        help="仅 smoke：截取每个短窗口前 N 条请求以缩小规模")
    parser.add_argument("--caps", type=str, default="8192,16384,32768")
    parser.add_argument("--scales", type=str, default="0.0625,0.25,1")
    parser.add_argument("--lambdas", type=str, default="26,32,38")
    parser.add_argument("--windows", type=int, default=8)
    # 模块 B / C
    parser.add_argument("--ps", type=str, default="0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--ks", type=str, default="3,5,9")
    parser.add_argument("--module-c-p", type=float, default=0.85)
    parser.add_argument("--module-c-k", type=int, default=5)
    parser.add_argument("--lambda-focus", type=float, default=38.0, help="模块 B/C 的固定 λ")
    # 模块 D / E
    parser.add_argument("--long-source", choices=["week", "subset"], default="week")
    parser.add_argument("--n-long-windows", type=int, default=32)
    parser.add_argument("--long-full-n", type=int, default=100000)
    parser.add_argument("--long-prefix-n", type=int, default=5000)
    parser.add_argument("--long-scales", type=str, default="0.0625,1")
    parser.add_argument("--long-caps", type=str, default="8192,32768")
    parser.add_argument("--long-lambdas", type=str, default="32,38")
    parser.add_argument("--nominal-lambda", type=float, default=32.0,
                        help="模块 E 统一时间缩放的标称 λ（c_t = pooled_rate / 本值）")
    # 公共
    parser.add_argument("--gamma", type=float, default=0.80)
    parser.add_argument("--quota-k", type=int, default=5)
    parser.add_argument("--err-seed", type=int, default=20261128,
                        help="误判掩码固定种子（与窗口编号共同决定掩码，跨策略/容量共享）")
    parser.add_argument("--out-dir", type=str, default=os.path.join(
        os.path.dirname(__file__), "real_trace_output"))
    parser.add_argument("--ckpt-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--merge-dirs", type=str, default=None,
                        help="merge 模式：逗号分隔的分片目录/文件列表")
    parser.add_argument("--smoke", action="store_true", help="极小规模自检（覆盖默认网格）")
    args = parser.parse_args()

    cfg_path = Path(__file__).resolve().parent / "experiment" / "configs" / "real_trace_transfer_base.yaml"
    cfg = load_config(cfg_path)
    cfg["gamma"] = float(args.gamma)
    cfg["quota_k"] = int(args.quota_k)
    cfg["mech_diag_enabled"] = True

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.smoke:
        _apply_smoke(args)

    modules = ["A", "B", "C", "D", "E"] if args.module == "all" else [args.module]

    # merge 模式：不需要加载窗口
    if args.merge_dirs:
        dirs = [d.strip() for d in args.merge_dirs.split(",") if d.strip()]
        for m in modules:
            try:
                arrays, _sig, ref_meta = _merge_module(m, dirs, _axis_stub(cfg, args, m))
                # 输出 meta 继承**原始运行信息**（完整运行签名/参数/代码版本），
                # 不用合并时的 CLI 参数替代。
                meta = dict(ref_meta) if ref_meta else _common_meta(m, cfg, args)
                # 配置快照必须来自**已验证的原始运行配置**（_sig['config']），而非合并时
                # 的 CLI 参数；否则同一 JSON 内 args（原始 gamma=0.6）与
                # cfg_sim_snapshot（合并默认 gamma=0.8）会相互矛盾，误导后续分析。
                orig_cfg = dict(_sig.get("config") or cfg)
                meta["merge"] = {
                    "sources": dirs,
                    "note": ("output meta inherited from the verified shards' original run "
                             "meta/signature; merge-time CLI params not substituted"),
                }
                meta["run_signature"] = _sig
                _save_module(m, out_dir, arrays, orig_cfg, args, cfg_path, meta)
            except FileNotFoundError as exc:
                print(f"skip merge[{m}]: {exc}")
        return

    # 短窗口（A/B/C）
    if any(m in ("A", "B", "C") for m in modules):
        short_windows, short_partition = _load_subset_windows(
            cfg, int(args.n_windows),
            max_n=(int(args.short_max_n) if args.short_max_n else None),
            start=int(args.window_start))
        print(f"short windows: {len(short_windows)} "
              f"(each {short_windows[0].arrivals.shape[0] if short_windows else 0} reqs)")
    else:
        short_windows, short_partition = [], {}

    # 长窗口（D/E）
    if any(m in ("D", "E") for m in modules):
        if args.long_source == "week":
            long_windows, starts, n_total, long_partition, grid_rate = _load_week_windows(
                cfg, int(args.n_long_windows), int(args.long_full_n),
                start=int(args.window_start), window_total=int(args.window_total))
        else:
            long_windows, long_partition = _load_subset_windows(
                cfg, int(args.n_long_windows),
                max_n=(int(args.short_max_n) if args.short_max_n else None),
                start=int(args.window_start))
            if long_windows and int(args.long_full_n) > long_windows[0].arrivals.shape[0]:
                raise ValueError(
                    f"--long-full-n={args.long_full_n} 超过窗口请求数 "
                    f"{long_windows[0].arrivals.shape[0]}（subset 源需满足 long_full_n ≤ 窗口大小）")
            starts, n_total = [], 0
            # 模块 E 的**全局**时间缩放：对全体 grid 窗口（非本进程子集）求 pooled rate，
            # 保证各 worker 用同一系数，保留窗口间强度差异。
            grid_rate = _subset_grid_rate(cfg, args)
        print(f"long windows: {len(long_windows)} source={args.long_source} "
              f"grid_rate={grid_rate:.4f}")
    else:
        long_windows, starts, n_total, long_partition, grid_rate = [], [], 0, {}, float("nan")

    if "A" in modules:
        caps = _parse_capacities(args.caps)
        scales = _parse_composition_scales(args.scales)
        loads = _parse_lambdas(args.lambdas)
        print(f"[A] windows={len(short_windows)} a={scales.tolist()} K={caps.tolist()} "
              f"lambda={loads.tolist()} policies={len(_ALL_POLICIES)}")
        arrays = _run_module_a(cfg, short_windows, scales, caps, loads,
                               _ALL_POLICIES, int(args.windows), args, short_partition)
        _save_module("A", out_dir, arrays, cfg, args, cfg_path, _common_meta("A", cfg, args))

    if "B" in modules:
        caps = _parse_capacities(args.long_caps)
        ps = _parse_ps(args.ps)
        ks = _parse_ks(args.ks)
        lam = float(args.lambda_focus)
        print(f"[B] windows={len(short_windows)} K={caps.tolist()} p={ps.tolist()} "
              f"k={ks.tolist()} lambda={lam}")
        arrays = _run_module_b(cfg, short_windows, caps, ps, ks, lam,
                               _ALL_POLICIES, int(args.windows), args, short_partition)
        _save_module("B", out_dir, arrays, cfg, args, cfg_path, _common_meta("B", cfg, args))

    if "C" in modules:
        caps = _parse_capacities(args.long_caps)
        print(f"[C] windows={len(short_windows)} K={caps.tolist()} "
              f"p={args.module_c_p} k={args.module_c_k} lambda={args.lambda_focus} "
              f"errs={[e[0] for e in _ERR_CONFIGS]}")
        arrays = _run_module_c(cfg, short_windows, caps, _ERR_CONFIGS,
                               float(args.lambda_focus), float(args.module_c_p),
                               int(args.module_c_k), _ALL_POLICIES, int(args.windows), args,
                               short_partition)
        _save_module("C", out_dir, arrays, cfg, args, cfg_path, _common_meta("C", cfg, args))

    if "D" in modules or "E" in modules:
        # 全局统一时间缩放：基于**全体 grid 窗口**（跨 worker 一致），而非本进程子集
        shared_ct = float(grid_rate) / float(args.nominal_lambda)
        print(f"[D/E] long windows={len(long_windows)} grid_rate={grid_rate:.3f} "
              f"shared_c_t={shared_ct:.4f} starts[:3]={starts[:3]}")

    if "D" in modules:
        caps = _parse_capacities(args.long_caps)
        scales = _parse_composition_scales(args.long_scales)
        loads = _parse_lambdas(args.long_lambdas)
        lengths = np.asarray([int(args.long_prefix_n), int(args.long_full_n)], dtype=np.int64)
        print(f"[D] N={lengths.tolist()} a={scales.tolist()} K={caps.tolist()} "
              f"lambda={loads.tolist()}")
        arrays = _run_module_d(cfg, long_windows, lengths, scales, caps, loads,
                               _ALL_POLICIES, int(args.windows), args, long_partition)
        meta_extra = _common_meta("D", cfg, args)
        meta_extra["week_starts"] = starts
        meta_extra["week_total_rows"] = int(n_total)
        meta_extra["loader_trap_note"] = (
            "full interval processed first (length transform + threshold + time scale), "
            "frozen, then prefix sliced; short/long share the identical prefix")
        _save_module("D", out_dir, arrays, cfg, args, cfg_path, meta_extra)

    if "E" in modules:
        caps = _parse_capacities(args.long_caps)
        print(f"[E] windows={len(long_windows)} K={caps.tolist()} c_t={shared_ct:.4f} "
              f"nominal_lambda={args.nominal_lambda}")
        arrays = _run_module_e(cfg, long_windows, caps, shared_ct,
                               _ALL_POLICIES, int(args.windows), args, long_partition)
        meta_extra = _common_meta("E", cfg, args)
        meta_extra["shared_time_scale"] = float(shared_ct)
        meta_extra["nominal_lambda"] = float(args.nominal_lambda)
        meta_extra["grid_rate"] = float(grid_rate)
        meta_extra["note"] = ("all windows share one time scale; per-window effective lambda "
                              "differs (see e_lambda_eff) and must be reported individually")
        _save_module("E", out_dir, arrays, cfg, args, cfg_path, meta_extra)


def _apply_smoke(args) -> None:
    args.n_windows = 2
    args.short_max_n = 400
    args.caps = "8192,16384"
    args.scales = "0.25,1"
    args.lambdas = "32"
    args.windows = 2
    args.ps = "0.80,0.85"
    args.ks = "3"
    args.long_caps = "8192"
    args.long_scales = "0.25,1"
    args.long_lambdas = "32"
    args.long_source = "subset"
    args.n_long_windows = 2
    args.long_full_n = 400
    args.long_prefix_n = 200


def _parse_ps(raw: str) -> np.ndarray:
    parts = [s.strip() for s in str(raw).split(",") if s.strip()]
    if not parts:
        raise ValueError("--ps 不能为空")
    vals = [float(s) for s in parts]
    for v in vals:
        if not (0.0 < v < 1.0):
            raise ValueError(f"非法 p: {v}（必须在 (0,1)）")
    if len(set(vals)) != len(vals):
        raise ValueError(f"--ps 存在重复值: {parts}")
    return np.asarray(sorted(vals), dtype=np.float64)


def _axis_stub(cfg: dict, args, module: str) -> dict:
    """merge 模式下补充轴数组（分片中已含坐标，这里给绘图用的轴名）。"""
    if module == "A":
        return {
            "composition_scales": _parse_composition_scales(args.scales),
            "capacities": _parse_capacities(args.caps),
            "lambdas": _parse_lambdas(args.lambdas),
        }
    if module == "B":
        return {
            "capacities": _parse_capacities(args.long_caps),
            "ps": _parse_ps(args.ps),
            "ks": _parse_ks(args.ks),
            "lambda": np.asarray([float(args.lambda_focus)]),
        }
    if module == "C":
        return {
            "capacities": _parse_capacities(args.long_caps),
            "err_names": np.asarray([e[0] for e in _ERR_CONFIGS]),
        }
    if module == "D":
        return {
            "lengths": np.asarray([int(args.long_prefix_n), int(args.long_full_n)]),
            "composition_scales": _parse_composition_scales(args.long_scales),
            "capacities": _parse_capacities(args.long_caps),
            "lambdas": _parse_lambdas(args.long_lambdas),
        }
    return {"capacities": _parse_capacities(args.long_caps)}


if __name__ == "__main__":
    main()
