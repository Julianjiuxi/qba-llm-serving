"""会议场景配置与核心类型的最小回归测试。

说明:
    迁移期曾用旧版 `refine` 引擎作为基准，逐一对比新引擎与旧引擎在
    FCFS/PSF/AA/QBA 下的数值是否一致（`test_reproduce_conference_baseline`、
    `test_legacy_*`、`test_workload_matches_refine`）。该对比回归已完成使命，
    现已移除，避免 clean-clone 因缺少私有 `refine` 包而失败。
    当前仅保留不依赖外部包的配置加载与类型冒烟测试。
"""

from __future__ import annotations

from pathlib import Path

from llm_in_queuing.config import load_config


def test_conference_config_loads() -> None:
    """会议场景默认配置能正确加载并解析。"""
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(repo_root / "configs" / "conference_default.yaml")

    assert cfg["seed"] == 42
    assert cfg["schedule_method"] == "QBA"
    assert cfg["KV_total_capacity"] == 8192
    assert cfg["KV_safety_threshold"] == 0.85


def test_types_smoke() -> None:
    """核心 dataclass 能正常实例化，默认值与可变字段隔离语义正确。"""
    from llm_in_queuing.types import Request, BatchEvent, EngineState

    # Request 能实例化，默认值生效
    r = Request(id=1, arrival_time=3.5, prompt_len=100, answer_len=200)
    assert r.id == 1
    assert r.waiting_time == 0.0          # 默认值
    assert r.prefill_chunk is False       # 默认值
    r.processed_len = 50                  # 字段可写
    assert r.processed_len == 50

    # BatchEvent 默认工厂不共享
    b1 = BatchEvent(id=1)
    b2 = BatchEvent(id=2)
    b1.prefill_requests.append(r)
    assert len(b1.prefill_requests) == 1
    assert len(b2.prefill_requests) == 0  # 不会污染另一个实例

    # EngineState 同理
    s = EngineState()
    s.wait_queue.append(r)
    assert len(s.wait_queue) == 1
    assert s.engine_isbusy is False
