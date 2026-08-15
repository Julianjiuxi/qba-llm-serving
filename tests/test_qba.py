"""QBA 调度器回归测试。

重点测试 quota counter 的基础语义：ordinary predicted-short 完整 prefill
准入会 +1，predicted-long 不计入。

注：partial-prefill 与 continuation-finish 的计数语义（was_continuation 修复）
暂缓，当前沿用 refine.py 旧语义以保证 freeze/ 冻结结果可复现。
"""

from __future__ import annotations

from llm_in_queuing.types import EngineState, Request
from llm_in_queuing.schedulers.qba import schedule_qba


def _make_req(
    req_id: int,
    pred_label: str,
    *,
    arrival: float = 1.0,
    prefill_tokens: int = 200,
    answer_len: int = 100,
    true_label: str = "short",
) -> Request:
    """构造一个测试用 Request。"""
    return Request(
        id=req_id,
        arrival_time=arrival,
        prompt_len=prefill_tokens,
        answer_len=answer_len,
        prefill_res_token=prefill_tokens,
        predict_label=pred_label,
        true_label=true_label,
    )


def _fresh_state() -> EngineState:
    """构造一个干净的 EngineState，用于 QBA 测试。"""
    s = EngineState()
    s.budget_tokens = 128
    s.request_block_size = 8
    s.KV_total_capacity = 8192
    s.KV_safety_threshold = 6000
    s.current_KV_capacity = 100
    s.engine_isbusy = False
    s.quota_k = 5
    s.gamma = 0.80
    return s


def test_qba_ordinary_short_full_prefill_counts_admission():
    """ordinary short 一次完成 prefill：counter 应从 0 变为 1。"""
    state = _fresh_state()
    req = _make_req(1, "short", prefill_tokens=50)  # ≤ budget=128
    state.wait_queue = [req]

    schedule_qba([], state)

    assert state.psfaq_short_admit_since_last_promoted == 1, (
        f"ordinary short full prefill 应使 counter +1，"
        f"实际: {state.psfaq_short_admit_since_last_promoted}"
    )


def test_qba_predicted_long_does_not_count():
    """predicted-long 请求不应增加 short counter。"""
    state = _fresh_state()
    req = _make_req(1, "long", prefill_tokens=50)
    state.wait_queue = [req]

    schedule_qba([], state)

    assert state.psfaq_short_admit_since_last_promoted == 0, (
        f"predicted-long 不应使 short counter 增加，"
        f"实际: {state.psfaq_short_admit_since_last_promoted}"
    )
