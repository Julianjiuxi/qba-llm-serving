from __future__ import annotations

import heapq

from llm_in_queuing.types import Request, BatchEvent, EngineState
from llm_in_queuing.state import (
    compute_and_record_aging_threshold,
    is_prefill_continuation,
    mark_request_started,
    remove_from_wait_queue,
)
from llm_in_queuing.engine import batch_time_calculate


def _rr_alternating_order(
    state: EngineState,
    short_queue: list[Request],
    long_queue: list[Request],
) -> list[Request]:
    """生成 RR 交替候选顺序：short→long→short→long→...

    从 state.rr_next_class 指向的类别开始交替选取。
    当某一队列为空时，追加另一队列全部剩余请求（work-conserving）。

    请注意：此函数使用局部变量构造顺序，不修改 state.rr_next_class。
    全局 RR pointer 仅在普通 prefill 请求成功准入后才更新。
    """
    result: list[Request] = []
    next_class: str = state.rr_next_class  # 局部副本，不写回 state
    s_idx, l_idx = 0, 0
    s_len = len(short_queue)
    l_len = len(long_queue)

    while s_idx < s_len or l_idx < l_len:
        if next_class == "short" and s_idx < s_len:
            result.append(short_queue[s_idx])
            s_idx += 1
            next_class = "long"
        elif next_class == "long" and l_idx < l_len:
            result.append(long_queue[l_idx])
            l_idx += 1
            next_class = "short"
        elif s_idx < s_len:
            result.extend(short_queue[s_idx:])
            break
        else:
            result.extend(long_queue[l_idx:])
            break

    return result


def schedule_rr(heap: list, state: EngineState) -> bool:
    """Round Robin 调度：类别交替选取 prefill candidates。

    active decode → unfinished continuation → RR-controlled ordinary prefill
    """
    if not state.engine_isbusy:
        state.itr += 1
        processing_in_current_itr: list[Request] = []
        state.current_batch_prefill = []
        state.current_batch = 0

        # --- decode budget 统计（与 AA/QBA/PSF 完全一致）---
        max_active_decode = int(state.budget_tokens) // int(state.request_block_size)
        state.current_concurrent = len(state.current_batch_decode)
        decode_tokens_this_batch = 0
        for r in state.current_batch_decode:
            remaining = int(r.answer_len) - int(r.processed_len)
            if remaining > 0:
                decode_tokens_this_batch += min(int(state.request_block_size), remaining)
        decode_tokens_this_batch = min(int(state.budget_tokens), int(decode_tokens_this_batch))
        prefill_budget = int(state.budget_tokens) - int(decode_tokens_this_batch)

        # --- aging 阈值（保持统计一致性，但不用于 RR 排序）---
        compute_and_record_aging_threshold(state)

        prefill_token_map: dict[int, int] = {}

        if state.current_KV_capacity < state.KV_safety_threshold:
            # 1. 提取 continuation（最高优先级）
            continuation = [e for e in state.wait_queue if is_prefill_continuation(e)]
            rest = [e for e in state.wait_queue if not is_prefill_continuation(e)]

            # 2. 分区 short / long
            short_queue = [e for e in rest if e.predict_label == "short"]
            long_queue = [e for e in rest if e.predict_label == "long"]
            short_queue.sort(key=lambda x: x.arrival_time)
            long_queue.sort(key=lambda x: x.arrival_time)

            # 3. 生成 RR 交替顺序
            rr_ordered = _rr_alternating_order(state, short_queue, long_queue)

            # 4. 最终队列：continuation 最高优先 + RR 交替
            ordered = continuation + rr_ordered
            state.wait_queue = ordered

            # 5. prefill admission 循环（参照 AA：break 语义，无 quantum 限制）
            planned_kv_increase = 0
            for e in ordered:
                if state.current_concurrent >= max_active_decode:
                    break

                remaining_prefill_budget = int(prefill_budget) - int(state.current_batch)
                if remaining_prefill_budget <= 0:
                    break

                rem = int(e.prefill_res_token)
                take = min(rem, int(remaining_prefill_budget))
                if take <= 0:
                    break

                extra_decode0 = 0
                if take == rem:
                    extra_decode0 = min(
                        int(state.request_block_size),
                        int(e.answer_len) - int(e.processed_len),
                    )

                projected_kv_peak = (
                    int(state.current_KV_capacity)
                    + int(decode_tokens_this_batch)
                    + int(planned_kv_increase)
                    + int(take)
                    + int(extra_decode0)
                )
                if (
                    projected_kv_peak > int(state.KV_total_capacity)
                    or projected_kv_peak > float(state.KV_safety_threshold)
                ):
                    break  # KV 不足 → 本批次停止 prefill admission

                # 准入前保存 continuation 状态，避免后续修改 e 导致误判
                was_continuation = is_prefill_continuation(e)

                if rem <= remaining_prefill_budget:
                    state.current_batch_prefill.append(e)
                    state.current_batch += int(rem)
                    e.prefill_res_token = 0
                    prefill_token_map[int(e.id)] = int(rem)
                    planned_kv_increase += int(rem) + int(extra_decode0)
                    processing_in_current_itr.append(e)
                    if e.start_service_time == 0.0:
                        e.start_service_time = state.current_time
                        e.waiting_time = e.start_service_time - e.arrival_time
                        mark_request_started(e, state)
                    # 成功准入普通请求 → 切换 RR pointer
                    if not was_continuation:
                        state.rr_next_class = "long" if e.predict_label == "short" else "short"
                else:
                    e.prefill_chunk = True
                    state.current_batch_prefill.append(e)
                    e.prefill_res_token = int(e.prefill_res_token) - int(take)
                    state.current_batch += int(take)
                    prefill_token_map[int(e.id)] = int(take)
                    planned_kv_increase += int(take)
                    processing_in_current_itr.append(e)
                    if e.start_service_time == 0.0:
                        e.start_service_time = state.current_time
                        e.waiting_time = e.start_service_time - e.arrival_time
                        mark_request_started(e, state)
                    # 部分 prefill chunk 也消耗一个 RR turn
                    if not was_continuation:
                        state.rr_next_class = "long" if e.predict_label == "short" else "short"
                    break  # chunked prefill: 一轮只拿一部分

            # 6. 从 wait_queue 移除已处理的请求
            for e in processing_in_current_itr:
                remove_from_wait_queue(e, state)

        # --- 构建 BatchEvent ---
        batch_usage = int(state.current_batch) + int(decode_tokens_this_batch)
        state.current_batch_time = state.current_time + batch_time_calculate(
            batch_usage, state.current_concurrent
        )
        current_OOM = False
        if state.current_KV_capacity > state.KV_total_capacity:
            current_OOM = True
        batch_event = BatchEvent(
            id=state.itr,
            batch_time=state.current_batch_time,
            batch_usage=batch_usage,
            current_concurrent=state.current_concurrent,
            current_KV_capacity=state.current_KV_capacity,
            current_OOM=current_OOM,
            prefill_requests=list(state.current_batch_prefill),
            prefill_token_map=dict(prefill_token_map),
            decode_requests=state.current_batch_decode,
        )
        if batch_event.batch_usage > 0:
            heapq.heappush(heap, (batch_event.batch_time, batch_event.id, batch_event))
            state.engine_isbusy = True
            return False
        else:
            return True

    return False
