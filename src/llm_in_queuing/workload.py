from __future__ import annotations

import heapq

import numpy as np

from llm_in_queuing.types import Request


def _assign_labels(
    raw_requests: list[Request],
    cfg: dict,
) -> None:
    """Assign true/predicted short/long labels to a list of Requests.

    Uses ``cfg["label_threshold_percentail"]`` as the quantile of answer_len
    for the true-label threshold, and ``cfg["TN"]/["TP"]/["FN"]/["FP"]`` for
    prediction noise (all-zero = perfect prediction).
    """
    answer_lens = np.array([r.answer_len for r in raw_requests], dtype=np.float64)
    ans_threshold = float(np.quantile(answer_lens, float(cfg["label_threshold_percentail"])))

    tn = float(cfg["TN"])
    tp = float(cfg["TP"])
    fn = float(cfg["FN"])
    fp = float(cfg["FP"])

    if tn == 0.0 and tp == 0.0 and fn == 0.0 and fp == 0.0:
        tn, tp, fn, fp = 1.0, 1.0, 0.0, 0.0
    else:
        s_short = tn + fp
        s_long = tp + fn
        if s_short > 0.0:
            tn, fp = tn / s_short, fp / s_short
        if s_long > 0.0:
            tp, fn = tp / s_long, fn / s_long

    for req in raw_requests:
        true_label = "short" if req.answer_len <= ans_threshold else "long"
        r = float(np.random.rand())
        if true_label == "long":
            pred_label = "long" if r < tp else "short"
        else:
            pred_label = "short" if r < tn else "long"
        req.true_label = true_label
        req.predict_label = pred_label


def _generate_heapq_from_csv(csv_path: str, cfg: dict) -> list[tuple[float, int, Request]]:
    """Build arrival heap from a real-trace CSV file.

    The CSV must contain columns ``TimeSeconds``, ``ContextTokens``,
    ``GeneratedTokens``.  ``admission_number`` can be smaller than the CSV
    row count to truncate the trace.

    If ``cfg["lambda_arrival"]`` is set, the original arrival rate
    (n / trace_span) is scaled to match the target  λ, preserving the
    relative inter-arrival intervals::

        scale = λ_original / λ_target
        t' = t × scale

    If ``cfg["token_scale"]`` is set, both ``prompt_len`` and ``answer_len``
    are multiplied by this factor (before capping), allowing the workload
    to be shrunk/grown to match engine capacity.
    """
    import pandas as pd

    np.random.seed(int(cfg["seed"]))
    df = pd.read_csv(csv_path)
    n_take = min(len(df), int(cfg.get("admission_number", len(df))))

    prompt_cap = int(cfg.get("prompt_len_cap", 999999))
    answer_cap = int(cfg.get("answer_len_cap", 999999))
    token_scale = float(cfg.get("token_scale", 1.0))

    # ── compute original lambda from trace ───────────────────────────────
    ts_raw = df["TimeSeconds"].iloc[:n_take].values.astype(np.float64)
    trace_span = float(ts_raw[-1] - ts_raw[0])
    lambda_orig = float(n_take) / trace_span if trace_span > 0 else float("inf")

    target_lambda = float(cfg.get("lambda_arrival", 0.0))
    scale = 1.0
    if target_lambda > 0 and lambda_orig > 0:
        scale = lambda_orig / target_lambda

    raw_requests: list[Request] = []
    for idx in range(n_take):
        row = df.iloc[idx]
        raw_p = int(row["ContextTokens"])
        raw_a = int(row["GeneratedTokens"])
        p_len = max(5, min(prompt_cap, int(raw_p * token_scale)))
        a_len = max(1, min(answer_cap, int(raw_a * token_scale)))
        t_raw = float(row["TimeSeconds"])
        t_scaled = (t_raw - float(ts_raw[0])) * scale  # re-zero + scale
        req = Request(
            id=idx,
            arrival_time=t_scaled,
            prompt_len=p_len,
            answer_len=a_len,
            prefill_res_token=p_len,
        )
        raw_requests.append(req)

    _assign_labels(raw_requests, cfg)

    # ── log scaling info ──────────────────────────────────────────────
    parts: list[str] = [f"  [trace_csv]  N={n_take}"]
    if target_lambda > 0:
        scaled_span = trace_span * scale
        parts.append(
            f"λ_orig={lambda_orig:.2f}→λ_target={target_lambda:.2f}  "
            f"time_scale={scale:.4f}  span: {trace_span:.1f}s→{scaled_span:.1f}s"
        )
    if token_scale != 1.0:
        prompts = np.array([r.prompt_len for r in raw_requests], dtype=np.float64)
        answers = np.array([r.answer_len for r in raw_requests], dtype=np.float64)
        parts.append(
            f"token_scale={token_scale:.2f}  "
            f"P_mean={prompts.mean():.0f}  A_mean={answers.mean():.0f}"
        )
    if len(parts) > 1:
        print("  ".join(parts))

    event_heap: list[tuple[float, int, Request]] = []
    for req in raw_requests:
        heapq.heappush(event_heap, (req.arrival_time, req.id, req))
    return event_heap


def generate_heapq(cfg: dict) -> list[tuple[float, int, Request]]:
    """生成请求到达事件堆，返回 heap 列表（可直接传给 heapq 操作）

    If ``cfg["trace_csv"]`` is set, requests are read from a real-trace CSV
    instead of being generated from synthetic distributions.
    """
    if "trace_csv" in cfg:
        return _generate_heapq_from_csv(str(cfg["trace_csv"]), cfg)

    np.random.seed(int(cfg["seed"]))
    event_heap: list[tuple[float, int, Request]] = []

    # 生成泊松到达序列并编号
    t = 0.0
    raw_requests: list[Request] = []
    for idx in range(int(cfg["admission_number"])):
        inter_time = np.random.exponential(1.0 / float(cfg["lambda_arrival"]))
        t += inter_time  # 负指数累计生成泊松

        # 生成 prompt_len（混合分布：正态 + 帕累托）
        if np.random.rand() < float(cfg["norm_ratio"]):
            prompt_len = int(np.random.normal(float(cfg["norm_loc"]), float(cfg["norm_scale"])))
        else:
            prompt_len = int((np.random.pareto(float(cfg["pareto_shape"])) + 1.0) * float(cfg["pareto_scale"]))
        prompt_len = max(5, prompt_len)
        prompt_len = min(int(cfg["prompt_len_cap"]), prompt_len)

        # 生成 answer_len（对数正态，与 prompt_len 对数相关）
        mu_ln = float(cfg["beta0"]) + float(cfg["beta1"]) * np.log1p(prompt_len)
        answer_len_f = float(np.random.lognormal(mu_ln, float(cfg["sigma"])))
        if not np.isfinite(answer_len_f):
            answer_len_f = float(cfg["answer_len_cap"])
        answer_len = int(min(float(cfg["answer_len_cap"]), answer_len_f))
        answer_len = max(1, answer_len)

        req = Request(
            id=idx,
            arrival_time=t,
            prompt_len=prompt_len,
            answer_len=answer_len,
            prefill_res_token=prompt_len,  # 初始剩余 prefill token 数 = 全部 prompt
        )
        raw_requests.append(req)

    _assign_labels(raw_requests, cfg)

    for req in raw_requests:
        heapq.heappush(event_heap, (req.arrival_time, req.id, req))

    return event_heap
