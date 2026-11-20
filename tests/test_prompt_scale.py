"""prompt_scale 实验适配层的一致性测试（fig8）。

只验证"相对基础 workload 仅改变了 prompt"这一核心不变量，并确认适配层可恢复，
不复现会议版 frozen sweep、不重跑百种子回归。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm_in_queuing import simulator  # noqa: E402
from llm_in_queuing.config import load_config  # noqa: E402
from llm_in_queuing.workload import generate_heapq  # noqa: E402
from scripts.experiment import sweep_prompt_scale as sps  # noqa: E402
from scripts.experiment._common import run_sim_stats  # noqa: E402

_BASE_YAML = _REPO_ROOT / "scripts" / "experiment" / "configs" / "sweep_prompt_scale_base.yaml"


def _small_cfg() -> dict:
    """小规模配置：只用于验证逻辑，不用于科学结论。"""
    cfg = load_config(_BASE_YAML)
    cfg["admission_number"] = 120
    return cfg


def _overrides(policy: str, seed: int) -> dict:
    return {"lambda_arrival": 32.0, "schedule_method": policy, "seed": int(seed)}


def _snapshot(cfg: dict, scale: float) -> dict:
    """按给定倍率生成并缩放请求，返回 {id: Request}。"""
    heap = generate_heapq(dict(cfg))
    sps._apply_prompt_scale(heap, scale, int(cfg["prompt_len_cap"]), {})
    return {r.id: r for _, _, r in heap}


# ── 倍率解析 ────────────────────────────────────────────────

def test_parse_prompt_scales_sorts_ascending():
    scales = sps._parse_prompt_scales("1, 4,0.25,2,0.5")
    assert scales.tolist() == [0.25, 0.5, 1.0, 2.0, 4.0]


@pytest.mark.parametrize("bad", ["", "0", "-1", "1,1", "nan", "inf", "1,abc"])
def test_parse_prompt_scales_rejects_invalid(bad):
    with pytest.raises(ValueError):
        sps._parse_prompt_scales(bad)


# ── 验证 A：请求级不变量 ────────────────────────────────────

def test_scaling_only_changes_prompt():
    """缩放只改 prompt_len / prefill_res_token，其余字段逐条不变。"""
    cfg = _small_cfg()
    cap = int(cfg["prompt_len_cap"])
    base = _snapshot(cfg, 1.0)

    for a in (0.25, 4.0):
        got = _snapshot(cfg, a)
        assert set(got) == set(base)
        for rid, r in got.items():
            b = base[rid]
            # 到达时间 / answer / 标签完全保持
            assert r.arrival_time == b.arrival_time
            assert r.answer_len == b.answer_len
            assert r.true_label == b.true_label
            assert r.predict_label == b.predict_label
            # prompt 满足缩放公式，且剩余 prefill token 同步
            expected = min(cap, max(5, int(round(a * b.prompt_len))))
            assert r.prompt_len == expected
            assert r.prefill_res_token == r.prompt_len


def test_scale_one_is_identity():
    """a=1 必须与原始生成器逐字段一致（不做多余舍入）。"""
    cfg = _small_cfg()
    raw = {r.id: r for _, _, r in generate_heapq(dict(cfg))}
    got = _snapshot(cfg, 1.0)
    assert set(raw) == set(got)
    for rid, r in got.items():
        assert r.prompt_len == raw[rid].prompt_len
        assert r.prefill_res_token == raw[rid].prefill_res_token


# ── 验证 B：a=1 数值锚点 + 恢复路径 ─────────────────────────

def test_scale_one_matches_raw_simulation():
    """经 a=1 适配层的仿真，与直接运行的指标逐位一致，且生成器已被恢复。"""
    cfg = _small_cfg()
    original = simulator.generate_heapq

    policies = ["FCFS", "QBA"]
    seed = int(cfg["seed"])
    rows, _ = sps._run_prompt_scale_sweep(cfg, 32.0, np.asarray([1.0]), policies, [seed])

    assert simulator.generate_heapq is original  # 已恢复
    for pol in policies:
        ref = run_sim_stats(cfg, _overrides(pol, seed))
        assert rows[pol]["mean_runs"][0, 0] == ref["mean"]
        assert rows[pol]["p99_runs"][0, 0] == ref["p99"]


def test_generator_restored_on_failure(monkeypatch):
    """仿真抛异常时，finally 仍恢复原生成器，并带上实验上下文。"""
    cfg = _small_cfg()
    original = simulator.generate_heapq

    def _boom(cfg, overrides=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(sps, "run_sim_stats", _boom)
    with pytest.raises(RuntimeError, match=r"Experiment failed: seed=.*prompt_scale=.*policy="):
        sps._run_prompt_scale_sweep(cfg, 32.0, np.asarray([2.0]), ["FCFS"], [42])
    assert simulator.generate_heapq is original
