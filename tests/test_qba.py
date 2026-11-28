"""QBA 调度器回归测试。

重点测试 quota counter 的基础语义：ordinary predicted-short 完整 prefill
准入会 +1，predicted-long 不计入。

Note: partial-prefill and continuation-finish cases are not exercised here;
they follow the same "count on full admission" rule documented in schedule_qba,
which is what the frozen freeze/ artifacts were produced with.
"""

from __future__ import annotations

from llm_in_queuing.types import EngineState, Request
from llm_in_queuing.state import add_to_wait_queue
from llm_in_queuing.engine import handle_arrival_event
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


def test_qba_t0_chunked_prefill_marks_started_once():
    """t=0 开始 + 多批 prefill：不得因 start_service_time==0 被误判为「尚未开始」而重复计数。

    回归：脚本把首请求到达时间归零，引擎旧逻辑用 ``start_service_time == 0.0`` 表示未开始，
    导致首个批次 t=0 开始后，下一批 chunked prefill 又被当成首次开始，重写等待并重复累加
    （arrived_started_n 变 2，arrived_not_started_n 变 −1）。改用独立 started 布尔后只计一次。
    """
    state = _fresh_state()
    state.current_time = 0.0
    req = _make_req(1, "short", arrival=0.0, prefill_tokens=500)   # > budget=128 → 分批
    handle_arrival_event(req, state)
    assert state.arrived_not_started_n == 1

    heap: list = []
    schedule_qba(heap, state)
    assert req.started
    assert req.waiting_time == 0.0
    assert req.prefill_res_token > 0                    # 确实需要多批 prefill
    assert state.arrived_started_n == 1
    assert state.arrived_not_started_n == 0

    # 第二轮：同一继续 prefill 请求再次被调度（t > 0），started 必须保持
    state.engine_isbusy = False
    state.current_time = 0.05
    add_to_wait_queue(req, state)
    schedule_qba(heap, state)
    assert state.arrived_started_n == 1                 # 不得重复 +1
    assert state.arrived_not_started_n == 0             # 不得出现负计数
