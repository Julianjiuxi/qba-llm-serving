from __future__ import annotations

from dataclasses import dataclass, field

@dataclass
class Request:
    id: int
    arrival_time: float
    prompt_len: int
    answer_len: int
    start_service_time: float = 0.0
    end_service_time: float = 0.0
    TTFT: float = 0.0
    waiting_time: float = 0.0
    processed_len: int = 0
    prefill_chunk: bool = False
    prefill_res_token: int = 0
    true_label: str = ""
    predict_label: str = ""
    in_processing: bool = False


@dataclass
class BatchEvent:
    id: int = 0
    batch_time: float = 0.0
    batch_usage: int = 0
    current_concurrent: int = 0
    current_KV_capacity: int = 0
    current_OOM: bool = False
    prefill_requests: list[Request] = field(default_factory=list)
    prefill_token_map: dict[int, int] = field(default_factory=dict)
    decode_requests: list[Request] = field(default_factory=list)


@dataclass
class EngineState:
    current_time: float = 0.0
    current_batch: int = 0
    current_batch_time: float = 0.0
    current_KV_capacity: int = 0
    current_concurrent: int = 0
    KV_total_capacity: int = 8192
    KV_safety_threshold: float = 0.0
    budget_tokens: int = 128
    request_block_size: int = 8
    wait_queue: list[Request] = field(default_factory=list)
    current_batch_prefill: list[Request] = field(default_factory=list)
    current_batch_decode: list[Request] = field(default_factory=list)
    finished_queue: list[Request] = field(default_factory=list)
    batch_list: list = field(default_factory=list)
    engine_isbusy: bool = False
    schedule_method: str = ""
    # QBA 相关
    quota_k: int = 5
    psfaq_short_admit_since_last_promoted: int = 0
    psfaq_promotion_debt: bool = False
    psfaq_promoted_total: int = 0
    psfaq_promoted_this_batch: int = 0
    # RR 相关
    rr_next_class: str = "short"
    # 迭代/老化相关
    itr: int = 0
    gamma: float = 0.95
    ageing_measure: str = "waiting_queue"
    # 等待队列统计
    wait_queue_n: int = 0
    wait_queue_sum_arrival: float = 0.0
    wait_queue_sum_arrival_sq: float = 0.0
    wait_queue_wait_mean: float = 0.0
    wait_queue_wait_var: float = 0.0
    wait_queue_wait_std: float = 0.0
    # 到达但未开始请求统计
    arrived_not_started_n: int = 0
    arrived_not_started_sum_arrival: float = 0.0
    arrived_not_started_sum_arrival_sq: float = 0.0
    # 已开始请求统计
    arrived_started_n: int = 0
    arrived_started_sum_wait: float = 0.0
    arrived_started_sum_wait_sq: float = 0.0
    arrived_started_waits: list[float] = field(default_factory=list)
    # 老化阈值记录
    aging_threshold_times: list[float] = field(default_factory=list)
    aging_threshold_values: list[float] = field(default_factory=list)
    # 稳态窗口
    t_warm_end: float = 0.0
    t_measure_end: float = 0.0
