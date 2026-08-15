"""公式层核心：reduced M/G/n mean-level surrogate model。

从旧 formula_calculate_trace_phi.py 移植，trace 相关函数已适配新引擎的 EngineState dataclass。
纯数学函数（排队论公式）无改动。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from llm_in_queuing.simulator import run
from llm_in_queuing.types import EngineState, Request


# ============================================================
# ModelParams
# ============================================================

@dataclass(frozen=True)
class ModelParams:
    n: int
    p_short: float
    m1_short: float
    m1_long: float
    m2_short: float
    m2_long: float
    quota_k: int = 1
    reference_load: float = 30.0
    reference_mean_wait: float = 0.0
    reference_mean_service: float = 0.0
    phi_knots_rho: Tuple[float, ...] = ()
    phi_knots_phi: Tuple[float, ...] = ()
    phi_knots_rho_psfaq: Tuple[float, ...] = ()
    phi_knots_phi_psfaq: Tuple[float, ...] = ()


# ============================================================
# 排队论公式（纯数学，无 trace 依赖）
# ============================================================

def erlang_c_delay_prob(n: int, rho: float) -> float:
    """Erlang-C 公式：M/M/n 中顾客需要等待的概率。"""
    if n <= 0:
        return 1.0
    if rho <= 0.0:
        return 0.0
    if rho >= 1.0:
        return 1.0
    a = float(n) * float(rho)
    denom = 0.0
    term = 1.0
    for k in range(0, n):
        if k > 0:
            term *= a / float(k)
        denom += term
    num = term * (a / float(n)) / (1.0 - float(rho))
    return float(num / (denom + num))


def allen_cunneen_mean_wq(lam: float, n: int, es: float, es2: float) -> float:
    """Allen-Cunneen 近似：M/G/n 平均等待时间。"""
    if n <= 0:
        return float("inf")
    if lam <= 0.0:
        return 0.0
    if es <= 0.0:
        return 0.0
    mu = 1.0 / float(es)
    rho = float(lam) * float(es) / float(n)
    if rho >= 1.0:
        return float("inf")
    var_s = max(0.0, float(es2) - float(es) * float(es))
    cs2 = 0.0 if es <= 0.0 else var_s / (float(es) * float(es))
    ca2 = 1.0
    p_wait = erlang_c_delay_prob(n, rho)
    denom = float(n) * mu - float(lam)
    if denom <= 0.0:
        return float("inf")
    return float(((ca2 + cs2) / 2.0) * (p_wait / denom))


def mixture_moments(p_short: float, m1_s: float, m1_l: float, m2_s: float, m2_l: float):
    """混合分布的 1、2 阶矩。"""
    p_short = float(p_short)
    p_long = 1.0 - p_short
    es = p_short * float(m1_s) + p_long * float(m1_l)
    es2 = p_short * float(m2_s) + p_long * float(m2_l)
    return es, es2


def psf_wait_approx(lam: float, n: int, p_short: float, m1_s: float, m1_l: float,
                    es: float, es2: float):
    """两阶段近似：用 omega 修正分离短/长任务的等待时间。"""
    mu = 1.0 / float(es) if es > 0.0 else 0.0
    rho = float(lam) * float(es) / float(n) if es > 0.0 else float("inf")
    if rho >= 1.0 or mu <= 0.0:
        return float("inf"), float("inf"), float("inf")
    wq_fcfs = allen_cunneen_mean_wq(lam, n, es, es2)
    omega = wq_fcfs * (1.0 - rho)
    lam_s = float(lam) * float(p_short)
    rho_s = lam_s * float(m1_s) / float(n) if m1_s > 0.0 else 0.0
    if rho_s >= 1.0:
        return float("inf"), float("inf"), float("inf")
    w_s = omega / (1.0 - rho_s)
    w_l = omega / ((1.0 - rho_s) * (1.0 - rho))
    w_avg = float(p_short) * w_s + (1.0 - float(p_short)) * w_l
    return float(w_s), float(w_l), float(w_avg)


def promoted_class_moments(lam: float, p_short: float, phi: float,
                           m1_s: float, m1_l: float, m2_s: float, m2_l: float):
    """promotion 后提升类（short + phi * long）的矩。"""
    lam = float(lam)
    lam_s = lam * float(p_short)
    lam_l = lam * (1.0 - float(p_short))
    lam_h = lam_s + float(phi) * lam_l
    if lam_h <= 0.0:
        return float(p_short), float(m1_s), float(m2_s)
    p_h = lam_h / lam
    m1_h = (lam_s * float(m1_s) + float(phi) * lam_l * float(m1_l)) / lam_h
    m2_h = (lam_s * float(m2_s) + float(phi) * lam_l * float(m2_l)) / lam_h
    return float(p_h), float(m1_h), float(m2_h)


def _interp01(x: float, xs: Tuple[float, ...], ys: Tuple[float, ...]) -> float:
    """线性插值，输出 clamp 到 [0, 1]。"""
    if len(xs) == 0 or len(ys) == 0 or len(xs) != len(ys):
        return 0.0
    if len(xs) == 1:
        return float(max(0.0, min(1.0, ys[0])))
    x = float(x)
    if x <= float(xs[0]):
        return float(max(0.0, min(1.0, ys[0])))
    if x >= float(xs[-1]):
        return float(max(0.0, min(1.0, ys[-1])))
    for i in range(len(xs) - 1):
        x0 = float(xs[i])
        x1 = float(xs[i + 1])
        if x0 <= x <= x1:
            y0 = float(ys[i])
            y1 = float(ys[i + 1])
            t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
            return float(max(0.0, min(1.0, (1.0 - t) * y0 + t * y1)))
    return 0.0


def psfa_phi_from_params(params: ModelParams, rho: float) -> float:
    """从 AA 的 phi 经验曲线插值。"""
    if len(params.phi_knots_rho) == 0 or len(params.phi_knots_phi) == 0:
        raise RuntimeError("phi knots are empty; fit/attach empirical phi before computing curves")
    return _interp01(float(rho), params.phi_knots_rho, params.phi_knots_phi)


def psfaq_phi_from_params(params: ModelParams, rho: float) -> float:
    """从 QBA 的 phi 经验曲线插值，受 quota_k 上限约束。"""
    if len(params.phi_knots_rho_psfaq) == 0 or len(params.phi_knots_phi_psfaq) == 0:
        raise RuntimeError("psfaq phi knots are empty; fit/attach empirical phi before computing curves")
    phi_interp = _interp01(float(rho), params.phi_knots_rho_psfaq, params.phi_knots_phi_psfaq)
    p_s = float(params.p_short)
    p_l = max(1e-12, 1.0 - p_s)
    k = max(1, int(params.quota_k))
    phi_cap = min(1.0, p_s / (float(k) * p_l))
    return float(min(phi_interp, phi_cap))


def compute_curves(lambdas: np.ndarray, params: ModelParams) -> dict:
    """计算 FCFS / PSF / AA / QBA 的近似平均等待时间曲线。"""
    p = params.p_short
    es, es2 = mixture_moments(p, params.m1_short, params.m1_long, params.m2_short, params.m2_long)

    fcfs_w = []
    psf_w = []
    psfa_w = []
    psfa_phi = []
    psfaq_w = []
    psfaq_phi = []

    for lam in lambdas:
        rho = float(lam) * float(es) / float(params.n)
        wq_fcfs = allen_cunneen_mean_wq(float(lam), params.n, es, es2)
        fcfs_w.append(wq_fcfs)
        _, _, wq_psf = psf_wait_approx(float(lam), params.n, p, params.m1_short, params.m1_long, es, es2)
        psf_w.append(wq_psf)
        phi = psfa_phi_from_params(params, rho)
        psfa_phi.append(phi)
        p_h, m1_h, m2_h = promoted_class_moments(float(lam), p, phi, params.m1_short, params.m1_long, params.m2_short, params.m2_long)
        _, _, wq_psfa = psf_wait_approx(float(lam), params.n, p_h, m1_h, params.m1_long, es, es2)
        psfa_w.append(wq_psfa)
        if len(params.phi_knots_rho_psfaq) > 0 and len(params.phi_knots_phi_psfaq) > 0:
            phi_q = psfaq_phi_from_params(params, rho)
            psfaq_phi.append(phi_q)
            p_hq, m1_hq, m2_hq = promoted_class_moments(float(lam), p, phi_q, params.m1_short, params.m1_long, params.m2_short, params.m2_long)
            _, _, wq_psfaq = psf_wait_approx(float(lam), params.n, p_hq, m1_hq, params.m1_long, es, es2)
            psfaq_w.append(wq_psfaq)

    fcfs_w = np.asarray(fcfs_w, dtype=np.float64)
    psf_w = np.asarray(psf_w, dtype=np.float64)
    psfa_w = np.asarray(psfa_w, dtype=np.float64)
    psfa_phi = np.asarray(psfa_phi, dtype=np.float64)
    psfaq_w_arr = np.asarray(psfaq_w, dtype=np.float64) if len(psfaq_w) > 0 else None
    psfaq_phi_arr = np.asarray(psfaq_phi, dtype=np.float64) if len(psfaq_phi) > 0 else None

    out = {
        "E[S]": float(es),
        "E[Wq]_FCFS": fcfs_w,
        "E[Wq]_PSF": psf_w,
        "E[Wq]_PSFA": psfa_w,
        "phi_PSFA": psfa_phi,
        "E[T]_FCFS": fcfs_w + float(es),
        "E[T]_PSF": psf_w + float(es),
        "E[T]_PSFA": psfa_w + float(es),
    }
    if psfaq_w_arr is not None and psfaq_phi_arr is not None:
        out["E[Wq]_PSFAQ"] = psfaq_w_arr
        out["phi_PSFAQ"] = psfaq_phi_arr
        out["E[T]_PSFAQ"] = psfaq_w_arr + float(es)
    return out


# ============================================================
# Trace 函数（已适配新引擎 EngineState）
# ============================================================

def run_trace(cfg: dict, policy: str) -> EngineState:
    """运行仿真 trace，返回 EngineState。"""
    cfg_local = dict(cfg)
    cfg_local["schedule_method"] = policy
    _, state = run(cfg_local, return_state=True)
    return state


def _threshold_at_start_times(state: EngineState, start_times: np.ndarray) -> np.ndarray:
    """从 trace 中提取 aging threshold，按 start_times 插值。"""
    ts = np.asarray(state.aging_threshold_times, dtype=np.float64)
    th = np.asarray(state.aging_threshold_values, dtype=np.float64)
    if ts.size == 0 or th.size == 0 or ts.size != th.size:
        return np.zeros_like(start_times, dtype=np.float64)
    idx = np.searchsorted(ts, start_times, side="right") - 1
    idx = np.clip(idx, 0, ts.size - 1)
    return th[idx]


def extract_reference_stats_from_state(state: EngineState) -> dict:
    """从 EngineState 提取模型校准所需的统计量。"""
    finished = state.finished_queue
    if not finished:
        raise RuntimeError("empty finished_queue in calibration trace")

    service = np.asarray(
        [float(r.end_service_time) - float(r.start_service_time) for r in finished],
        dtype=np.float64,
    )
    waits = np.asarray(
        [float(r.waiting_time) for r in finished],
        dtype=np.float64,
    )
    pred_short_mask = np.asarray(
        [r.predict_label == "short" for r in finished],
        dtype=bool,
    )

    s_short = service[pred_short_mask] if pred_short_mask.any() else service
    s_long = service[~pred_short_mask] if (~pred_short_mask).any() else service

    batch_list = state.batch_list
    conc = np.asarray(
        [float(b.get("current_concurrent", 0.0)) for b in batch_list
         if float(b.get("batch_usage", 0.0)) > 0.0],
        dtype=np.float64,
    )
    if conc.size == 0:
        conc = np.asarray([1.0], dtype=np.float64)

    return {
        "mean_wait": float(waits.mean()),
        "mean_service": float(service.mean()),
        "p_short": float(pred_short_mask.mean()),
        "m1_short": float(s_short.mean()),
        "m1_long": float(s_long.mean()),
        "m2_short": float((s_short * s_short).mean()),
        "m2_long": float((s_long * s_long).mean()),
        "mean_concurrency": float(conc.mean()),
        "p90_concurrency": float(np.quantile(conc, 0.90)),
        "max_concurrency": float(conc.max()),
        "n_finished": int(len(finished)),
    }


def calibrate_effective_n(reference_load: float, stats: dict, n_budget: int):
    """根据参考负载校准有效服务台数。"""
    p_short = float(stats["p_short"])
    m1_short = float(stats["m1_short"])
    m1_long = float(stats["m1_long"])
    m2_short = float(stats["m2_short"])
    m2_long = float(stats["m2_long"])
    target_w = float(stats["mean_wait"])
    es, es2 = mixture_moments(p_short, m1_short, m1_long, m2_short, m2_long)

    best = None
    for n in range(1, int(n_budget) + 1):
        pred = allen_cunneen_mean_wq(float(reference_load), int(n), float(es), float(es2))
        err = abs(float(pred) - float(target_w))
        score = (err, abs(n - stats.get("p90_concurrency", n)), -n)
        if best is None or score < best[0]:
            best = (score, int(n), float(pred))
    _, n_eff, pred_w = best
    return int(n_eff), float(pred_w), float(es), float(es2)


def choose_reference_load(cfg: dict, override: float | None) -> float:
    """选择参考负载。"""
    if override is not None:
        return float(override)
    return 30.0


def estimate_model_params_from_fcfs_trace(
    cfg: dict, reference_load: float | None = None,
) -> tuple[ModelParams, dict]:
    """从 FCFS trace 估计模型参数。"""
    cfg_local = dict(cfg)
    ref_load = choose_reference_load(cfg_local, reference_load)
    cfg_local["lambda_arrival"] = float(ref_load)

    state = run_trace(cfg_local, "FCFS")
    stats = extract_reference_stats_from_state(state)
    n_budget = max(1, int(float(cfg_local["budget_tokens"]) // float(cfg_local["request_block_size"])))
    n_eff, pred_w_ref, es, es2 = calibrate_effective_n(ref_load, stats, n_budget)

    params = ModelParams(
        n=int(n_eff),
        p_short=float(stats["p_short"]),
        m1_short=float(stats["m1_short"]),
        m1_long=float(stats["m1_long"]),
        m2_short=float(stats["m2_short"]),
        m2_long=float(stats["m2_long"]),
        reference_load=float(ref_load),
        reference_mean_wait=float(stats["mean_wait"]),
        reference_mean_service=float(stats["mean_service"]),
    )
    aux = {
        "reference_load": float(ref_load),
        "reference_mean_wait_sim": float(stats["mean_wait"]),
        "reference_mean_wait_formula": float(pred_w_ref),
        "reference_mean_service": float(stats["mean_service"]),
        "mean_concurrency": float(stats["mean_concurrency"]),
        "p90_concurrency": float(stats["p90_concurrency"]),
        "max_concurrency": float(stats["max_concurrency"]),
        "n_budget": int(n_budget),
        "n_finished": int(stats["n_finished"]),
        "es": float(es),
        "es2": float(es2),
    }
    return params, aux


# ============================================================
# 经验 phi 统计（trace-based）
# ============================================================

def empirical_phi_from_psfa_state(state: EngineState) -> dict:
    """从 AA trace 统计经验 promotion 比例（job-level）。"""
    finished = state.finished_queue
    if not finished:
        return {"phi_job": 0.0, "n_pred_long": 0, "n_promoted": 0, "phi_long_starts": 0.0}

    pred_long_jobs = [e for e in finished if e.predict_label == "long"]
    if len(pred_long_jobs) == 0:
        return {"phi_job": 0.0, "n_pred_long": 0, "n_promoted": 0, "phi_long_starts": 0.0}

    starts = np.asarray([float(e.start_service_time) for e in pred_long_jobs], dtype=np.float64)
    waits = np.asarray([float(e.waiting_time) for e in pred_long_jobs], dtype=np.float64)
    thresholds = _threshold_at_start_times(state, starts)
    promoted = waits > thresholds

    return {
        "phi_job": float(promoted.mean()) if promoted.size else 0.0,
        "n_pred_long": int(promoted.size),
        "n_promoted": int(promoted.sum()) if promoted.size else 0,
        "phi_long_starts": float(promoted.mean()) if promoted.size else 0.0,
    }


def empirical_phi_from_psfaq_state(state: EngineState) -> dict:
    """从 QBA trace 统计经验 promotion 比例（受 quota_k 约束）。"""
    finished = state.finished_queue
    if not finished:
        return {"phi_job": 0.0, "n_pred_long": 0, "n_promoted": 0, "phi_long_starts": 0.0}

    pred_long_jobs = [e for e in finished if e.predict_label == "long"]
    n_pred_long = int(len(pred_long_jobs))
    if n_pred_long == 0:
        return {"phi_job": 0.0, "n_pred_long": 0, "n_promoted": 0, "phi_long_starts": 0.0}

    n_promoted = int(state.psfaq_promoted_total)
    phi = float(n_promoted) / float(n_pred_long)
    phi = float(np.clip(phi, 0.0, 1.0))
    return {
        "phi_job": phi,
        "n_pred_long": n_pred_long,
        "n_promoted": n_promoted,
        "phi_long_starts": phi,
    }


def _resolve_policy(policy: str) -> str:
    """将策略名解析为新引擎名称（AA/QBA）。"""
    p = str(policy).upper()
    if p in ("PSFA", "AA"):
        return "AA"
    if p in ("PSFAQ", "QBA"):
        return "QBA"
    return "FCFS" if p == "FCFS" else "PSF"


def empirical_phi_from_policy_state(state: EngineState, policy: str) -> dict:
    """按策略分发经验 phi 统计。"""
    p = str(policy).upper()
    if p in ("PSFAQ", "QBA"):
        return empirical_phi_from_psfaq_state(state)
    return empirical_phi_from_psfa_state(state)


# ============================================================
# phi 曲线拟合
# ============================================================

def fit_phi_curve_from_policy_traces(
    cfg: dict, params_base: ModelParams, lambdas_fit: List[float], policy: str,
) -> tuple:
    """在给定 lambda 点集上运行 trace，拟合 phi(rho) 曲线。"""
    p = float(params_base.p_short)
    es, _ = mixture_moments(p, params_base.m1_short, params_base.m1_long,
                            params_base.m2_short, params_base.m2_long)
    if not np.isfinite(es) or es <= 0.0:
        raise RuntimeError("invalid E[S] for phi calibration")

    rho_knots: List[float] = []
    phi_knots: List[float] = []
    debug_rows = []
    for lam in lambdas_fit:
        cfg_local = dict(cfg)
        cfg_local["lambda_arrival"] = float(lam)
        engine_policy = _resolve_policy(str(policy))
        st = run_trace(cfg_local, engine_policy)
        phi_info = empirical_phi_from_policy_state(st, str(policy))
        rho = float(lam) * float(es) / float(params_base.n)
        rho_knots.append(rho)
        phi_knots.append(float(phi_info["phi_job"]))
        debug_rows.append((float(lam), float(rho), float(phi_info["phi_job"]),
                           int(phi_info["n_promoted"]), int(phi_info["n_pred_long"])))

    if len(rho_knots) == 0:
        return (), (), []
    order = np.argsort(np.asarray(rho_knots, dtype=np.float64))
    rho_sorted = tuple(float(rho_knots[i]) for i in order.tolist())
    phi_sorted = tuple(float(phi_knots[i]) for i in order.tolist())
    debug_sorted = [debug_rows[i] for i in order.tolist()]
    return rho_sorted, phi_sorted, debug_sorted


def maybe_attach_empirical_phi(
    cfg: dict, params: ModelParams, *, lambdas_fit: List[float],
) -> tuple[ModelParams, dict]:
    """从 AA trace 拟合 phi 曲线并挂到 ModelParams 上。"""
    if len(lambdas_fit) == 0:
        raise ValueError("phi-fit-lambdas is empty")
    rho_knots, phi_knots, debug_rows = fit_phi_curve_from_policy_traces(cfg, params, lambdas_fit, "AA")
    if len(rho_knots) == 0:
        raise RuntimeError("phi knots empty after fitting")
    updated = ModelParams(
        n=params.n, p_short=params.p_short,
        m1_short=params.m1_short, m1_long=params.m1_long,
        m2_short=params.m2_short, m2_long=params.m2_long,
        reference_load=params.reference_load,
        reference_mean_wait=params.reference_mean_wait,
        reference_mean_service=params.reference_mean_service,
        quota_k=params.quota_k,
        phi_knots_rho=tuple(rho_knots), phi_knots_phi=tuple(phi_knots),
        phi_knots_rho_psfaq=tuple(params.phi_knots_rho_psfaq),
        phi_knots_phi_psfaq=tuple(params.phi_knots_phi_psfaq),
    )
    return updated, {"phi_knots_n": int(len(rho_knots)), "phi_debug": debug_rows}


def maybe_attach_empirical_phi_psfaq(
    cfg: dict, params: ModelParams, *, lambdas_fit: List[float],
) -> tuple[ModelParams, dict]:
    """从 QBA trace 拟合 phi 曲线并挂到 ModelParams 上。"""
    if len(lambdas_fit) == 0:
        raise ValueError("phi-fit-lambdas is empty")
    rho_knots, phi_knots, debug_rows = fit_phi_curve_from_policy_traces(cfg, params, lambdas_fit, "QBA")
    if len(rho_knots) == 0:
        raise RuntimeError("phi knots empty after fitting")
    updated = ModelParams(
        n=params.n, p_short=params.p_short,
        m1_short=params.m1_short, m1_long=params.m1_long,
        m2_short=params.m2_short, m2_long=params.m2_long,
        reference_load=params.reference_load,
        reference_mean_wait=params.reference_mean_wait,
        reference_mean_service=params.reference_mean_service,
        quota_k=params.quota_k,
        phi_knots_rho=tuple(params.phi_knots_rho),
        phi_knots_phi=tuple(params.phi_knots_phi),
        phi_knots_rho_psfaq=tuple(rho_knots),
        phi_knots_phi_psfaq=tuple(phi_knots),
    )
    return updated, {"phi_knots_n": int(len(rho_knots)), "phi_debug": debug_rows}
