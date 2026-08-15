from __future__ import annotations

from pathlib import Path

import pytest

from llm_in_queuing.types import Request, EngineState
from llm_in_queuing.schedulers.rr import _rr_alternating_order, schedule_rr


# ── 辅助函数 ──────────────────────────────────────────────────

def _make_req(
    req_id: int,
    label: str,
    arrival: float = 1.0,
    prefill_tokens: int = 100,
    answer_len: int = 200,
) -> Request:
    """创建一个用于测试的最小 Request。"""
    return Request(
        id=req_id,
        arrival_time=arrival,
        prompt_len=100,
        answer_len=answer_len,
        predict_label=label,
        prefill_res_token=prefill_tokens,
    )


# ── 测试 _rr_alternating_order（候选顺序，不修改全局状态）───────

def test_rr_alternates_when_both_nonempty():
    """S/L 均非空时，交替顺序正确。_rr_alternating_order 不修改 state.rr_next_class。"""
    state = EngineState()
    state.rr_next_class = "short"
    before = state.rr_next_class

    short = [_make_req(1, "short"), _make_req(2, "short")]
    long = [_make_req(10, "long"), _make_req(11, "long")]

    result = _rr_alternating_order(state, short, long)

    labels = [r.predict_label for r in result]
    assert labels == ["short", "long", "short", "long"]
    # 全局指针不应被 _rr_alternating_order 修改
    assert state.rr_next_class == before, (
        f"_rr_alternating_order 不应修改 state.rr_next_class, "
        f"expected '{before}', got '{state.rr_next_class}'"
    )


def test_rr_alternates_starting_from_long():
    """rr_next_class 指向 long 时，从 long 开始交替。"""
    state = EngineState()
    state.rr_next_class = "long"
    before = state.rr_next_class

    short = [_make_req(1, "short"), _make_req(2, "short")]
    long = [_make_req(10, "long"), _make_req(11, "long")]

    result = _rr_alternating_order(state, short, long)

    labels = [r.predict_label for r in result]
    assert labels == ["long", "short", "long", "short"]
    assert state.rr_next_class == before


def test_rr_short_only_continuous():
    """S 非空、L 为空时，全部返回 S（work-conserving）。"""
    state = EngineState()
    state.rr_next_class = "short"
    before = state.rr_next_class

    short = [_make_req(1, "short"), _make_req(2, "short")]
    long: list[Request] = []

    result = _rr_alternating_order(state, short, long)

    assert len(result) == 2
    assert all(r.predict_label == "short" for r in result)
    assert state.rr_next_class == before


def test_rr_long_only_continuous():
    """L 非空、S 为空时，全部返回 L（work-conserving）。"""
    state = EngineState()
    state.rr_next_class = "short"
    before = state.rr_next_class

    short: list[Request] = []
    long = [_make_req(10, "long"), _make_req(11, "long")]

    result = _rr_alternating_order(state, short, long)

    assert len(result) == 2
    assert all(r.predict_label == "long" for r in result)
    assert state.rr_next_class == before


# ── 测试 schedule_rr 的 pointer 更新语义 ───────────────────────

def test_rr_pointer_after_partial_batch():
    """两短两长，batch 只能准入一个 short：下一轮 pointer 必须是 long。

    构造：2 short + 2 long，prefill_budget 仅够一个请求。
    预期：第一个 short 成功准入 → rr_next_class 变为 "long"。
    """
    state = EngineState()
    state.budget_tokens = 128
    state.request_block_size = 8
    state.KV_total_capacity = 8192
    state.KV_safety_threshold = 6000
    state.current_KV_capacity = 100
    state.engine_isbusy = False
    state.rr_next_class = "short"

    # 4 个请求，每个 prefill 128 tokens，budget = 128 → 只够一个
    s1 = _make_req(1, "short", 1.0, prefill_tokens=128)
    s2 = _make_req(2, "short", 2.0, prefill_tokens=128)
    l1 = _make_req(10, "long", 3.0, prefill_tokens=128)
    l2 = _make_req(11, "long", 4.0, prefill_tokens=128)
    state.wait_queue = [s1, s2, l1, l2]

    schedule_rr([], state)

    admitted_ids = [r.id for r in state.current_batch_prefill]
    admitted_labels = [(r.id, r.predict_label) for r in state.current_batch_prefill]
    assert len(admitted_ids) == 1, f"应只准入 1 个请求，实际: {admitted_labels}"
    assert admitted_ids[0] == 1, f"第一个准入的应为 short(id=1), got {admitted_ids}"
    assert state.rr_next_class == "long", (
        f"成功准入 short 后，pointer 应为 'long'，实际: '{state.rr_next_class}'"
    )


def test_rr_pointer_after_two_admitted():
    """short + long 同一 batch 均准入：pointer 切换两次回到起始类别。"""
    state = EngineState()
    state.budget_tokens = 256
    state.request_block_size = 8
    state.KV_total_capacity = 8192
    state.KV_safety_threshold = 6000
    state.current_KV_capacity = 100
    state.engine_isbusy = False
    state.rr_next_class = "short"

    # 两个小请求，都放进 budget
    s = _make_req(1, "short", 1.0, prefill_tokens=50)
    l = _make_req(10, "long", 2.0, prefill_tokens=50)
    state.wait_queue = [s, l]

    schedule_rr([], state)

    admitted = [(r.id, r.predict_label) for r in state.current_batch_prefill]
    assert len(admitted) == 2, f"应准入 2 个请求，实际: {admitted}"
    # short 准入 → pointer "long"；long 准入 → pointer "short"
    assert state.rr_next_class == "short", (
        f"short→long→short，最终应为 'short'，实际: '{state.rr_next_class}'"
    )


def test_rr_kv_projected_rejection_no_toggle():
    """第一个候选因 projected KV peak 被拒绝时，pointer 保持不变。

    构造：KV 接近安全阈值，第一个候选的 projected_peak 超限。
    预期：无实际准入，rr_next_class 不变。
    """
    state = EngineState()
    state.budget_tokens = 256
    state.request_block_size = 8
    state.KV_total_capacity = 2000
    state.KV_safety_threshold = 500  # 低安全阈值
    state.current_KV_capacity = 400  # 接近阈值
    state.engine_isbusy = False
    state.rr_next_class = "short"

    # 第一个 short 的 prefill = 200，extra_decode0 = 8
    # projected = 400 + 0 + 0 + 200 + 8 = 608 > 500 → 拒绝
    s = _make_req(1, "short", 1.0, prefill_tokens=200, answer_len=200)
    l = _make_req(10, "long", 2.0, prefill_tokens=200, answer_len=200)
    state.wait_queue = [s, l]

    schedule_rr([], state)

    admitted = state.current_batch_prefill
    assert len(admitted) == 0, (
        f"所有请求应被 KV 拒绝，实际准入: "
        f"{[(r.id, r.predict_label) for r in admitted]}"
    )
    assert state.rr_next_class == "short", (
        f"KV 拒绝不应切换 pointer，应为 'short'，实际: '{state.rr_next_class}'"
    )


def test_rr_long_continuation_does_not_consume_turn():
    """long continuation 成功执行不应消耗 RR turn，pointer 保持不变。"""
    state = EngineState()
    state.budget_tokens = 128
    state.request_block_size = 8
    state.KV_total_capacity = 8192
    state.KV_safety_threshold = 6000
    state.current_KV_capacity = 100
    state.engine_isbusy = False
    state.rr_next_class = "short"

    # continuation 用光全部 budget，后续普通请求无法准入
    cont = _make_req(1, "long", 1.0, prefill_tokens=128, answer_len=200)
    cont.prefill_chunk = True
    cont.prefill_res_token = 128

    short = _make_req(2, "short", 2.0, prefill_tokens=50)
    long = _make_req(3, "long", 3.0, prefill_tokens=50)
    state.wait_queue = [cont, short, long]

    schedule_rr([], state)

    admitted = [(r.id, r.predict_label, r.prefill_chunk) for r in state.current_batch_prefill]
    # continuation 被准入
    assert any(r[0] == 1 for r in admitted), f"continuation 应被准入, got {admitted}"
    # pointer 未变：continuation 不消耗 RR turn
    assert state.rr_next_class == "short", (
        f"continuation 不应消耗 RR turn，pointer 应为 'short'，实际: '{state.rr_next_class}'"
    )


# ── 原有测试（已验证正确，保留）───────────────────────────────

def test_rr_partial_prefill_consumes_turn():
    """新 ordinary short 发生 partial prefill：pointer 应从 short 切换到 long。

    构造：新 short 请求 prefill=200，budget=128 → 只能拿 128（partial prefill）。
    旧代码会在 prefill_chunk=True 后再判 is_prefill_continuation，导致误判为 continuation 跳过 toggle。
    """
    state = EngineState()
    state.budget_tokens = 128
    state.request_block_size = 8
    state.KV_total_capacity = 8192
    state.KV_safety_threshold = 6000
    state.current_KV_capacity = 100
    state.engine_isbusy = False
    state.rr_next_class = "short"

    # 新请求，不是 continuation，但 prefill 超出预算
    s = _make_req(1, "short", 1.0, prefill_tokens=200, answer_len=200)
    state.wait_queue = [s]

    schedule_rr([], state)

    admitted = [(r.id, r.predict_label, r.prefill_chunk, r.prefill_res_token)
                 for r in state.current_batch_prefill]
    assert len(admitted) == 1, f"应准入 1 个请求（partial prefill），实际: {admitted}"
    # partial prefill 后 prefill_chunk=True, prefill_res_token=72
    assert admitted[0][2] is True, "partial prefill 应设置 prefill_chunk=True"
    assert admitted[0][3] == 72, f"剩余 prefill 应为 72，实际: {admitted[0][3]}"
    # 新请求首次准入 → 应切换 pointer
    assert state.rr_next_class == "long", (
        f"新 ordinary short 首次准入（partial prefill）应切换 pointer 为 'long'，"
        f"实际: '{state.rr_next_class}'"
    )


def test_rr_continuation_finishing_does_not_toggle():
    """short continuation 恰好完成全部剩余 prefill：pointer 保持不变。

    旧代码在 prefill_res_token=0 后再判 is_prefill_continuation，会误判为非 continuation 并切换。
    """
    state = EngineState()
    state.budget_tokens = 256
    state.request_block_size = 8
    state.KV_total_capacity = 8192
    state.KV_safety_threshold = 6000
    state.current_KV_capacity = 100
    state.engine_isbusy = False
    state.rr_next_class = "short"

    # short continuation，本 batch 恰好完成
    cont = _make_req(1, "short", 1.0, prefill_tokens=50, answer_len=200)
    cont.prefill_chunk = True
    cont.prefill_res_token = 50

    state.wait_queue = [cont]

    schedule_rr([], state)

    admitted = [(r.id, r.predict_label, r.prefill_chunk, r.prefill_res_token)
                 for r in state.current_batch_prefill]
    assert len(admitted) == 1, f"continuation 应被准入, got {admitted}"
    # continuation 完成 → pointer 不切换
    assert state.rr_next_class == "short", (
        f"short continuation 完成不应切换 pointer，应为 'short'，实际: '{state.rr_next_class}'"
    )


def test_rr_short_continuation_does_not_consume_turn():
    """short continuation 成功执行不应消耗 RR turn，pointer 保持不变。

    注：cont label 设为 "short" 以确保无论旧/新代码都检测 toggle 错误。
    旧代码中 cont 标签若为 "long" 则 toggle 结果恰好与起始相同（"short"→"short"），会掩盖 bug。
    """
    state = EngineState()
    state.budget_tokens = 128
    state.request_block_size = 8
    state.KV_total_capacity = 8192
    state.KV_safety_threshold = 6000
    state.current_KV_capacity = 100
    state.engine_isbusy = False
    state.rr_next_class = "short"

    # continuation 用光全部 budget，后续普通请求无法准入
    # label="short" 使旧代码如果错误 toggle，结果会是 "long"，能被检测
    cont = _make_req(1, "short", 1.0, prefill_tokens=128, answer_len=200)
    cont.prefill_chunk = True
    cont.prefill_res_token = 128

    short = _make_req(2, "short", 2.0, prefill_tokens=50)
    long = _make_req(3, "long", 3.0, prefill_tokens=50)
    state.wait_queue = [cont, short, long]

    schedule_rr([], state)

    admitted = [(r.id, r.predict_label, r.prefill_chunk) for r in state.current_batch_prefill]
    assert any(r[0] == 1 for r in admitted), f"continuation 应被准入, got {admitted}"
    assert state.rr_next_class == "short", (
        f"continuation 不应消耗 RR turn，pointer 应为 'short'，实际: '{state.rr_next_class}'"
    )


def test_rr_continuation_above_rr():
    """continuation 请求排在 RR 交替的最前面（在 prefill batch 中最先被服务）。"""
    state = EngineState()
    state.budget_tokens = 128
    state.request_block_size = 8
    state.KV_total_capacity = 8192
    state.KV_safety_threshold = 6000
    state.current_KV_capacity = 100
    state.engine_isbusy = False

    cont = _make_req(1, "short", 1.0, prefill_tokens=50)
    cont.prefill_chunk = True
    cont.prefill_res_token = 50

    short = _make_req(2, "short", 2.0, prefill_tokens=100)
    long = _make_req(3, "long", 3.0, prefill_tokens=100)
    state.wait_queue = [cont, short, long]

    schedule_rr([], state)

    prefill_ids = [r.id for r in state.current_batch_prefill]
    assert prefill_ids[0] == 1, f"continuation 应为 prefill batch 首位, got {prefill_ids}"

    wait_ids = {r.id for r in state.wait_queue}
    assert 1 not in wait_ids, "continuation should have been consumed"


def test_rr_decode_priority_unchanged():
    """decode 不在 prefill 队列中出现，不受 RR 影响。"""
    state = EngineState()
    state.budget_tokens = 128
    state.request_block_size = 8
    state.KV_total_capacity = 8192
    state.KV_safety_threshold = 6000
    state.current_KV_capacity = 100
    state.engine_isbusy = False

    decode_req = _make_req(99, "long", 1.0, prefill_tokens=100)
    decode_req.processed_len = 50
    state.current_batch_decode = [decode_req]

    short = _make_req(1, "short", 2.0, prefill_tokens=100)
    long = _make_req(2, "long", 3.0, prefill_tokens=100)
    state.wait_queue = [short, long]

    schedule_rr([], state)

    prefill_ids = {r.id for r in state.current_batch_prefill}
    assert 99 not in prefill_ids
    assert len(state.current_batch_decode) == 1
    assert state.current_batch_decode[0].id == 99


def test_rr_end_to_end_smoke():
    """端到端冒烟测试：通过 simulator 运行一次完整仿真。"""
    from llm_in_queuing.config import load_config

    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(repo_root / "configs" / "rr_base.yaml")

    from llm_in_queuing.simulator import run

    metrics = run(dict(cfg))

    assert "OOM概率" in metrics
    assert "waiting_time平均" in metrics
    assert metrics["waiting_time平均"] >= 0
