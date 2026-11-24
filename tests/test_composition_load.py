"""fig13: 跨负载的构成扫描（fig11 迁移验证）适配层的一致性测试。

只验证核心不变量（λ 解析、3D 数组布局、a=1/λ=32 锚点与直接仿真一致、生成器可恢复、
逐请求总 token 守恒），不复现百种子回归、不重跑全量 sweep。
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
from scripts.experiment import sweep_composition_load as src  # noqa: E402
from scripts.experiment._common import run_sim_stats  # noqa: E402

_BASE_YAML = (_REPO_ROOT / "scripts" / "experiment" / "configs"
              / "sweep_composition_load_base.yaml")


def _small_cfg() -> dict:
    """小规模配置：只用于验证逻辑，不用于科学结论。"""
    cfg = load_config(_BASE_YAML)
    cfg["admission_number"] = 300
    return cfg


def _overrides(policy: str, seed: int) -> dict:
    return {"lambda_arrival": 32.0, "schedule_method": policy, "seed": int(seed)}


# ── λ 解析 ──────────────────────────────────────────────────

def test_parse_lambdas_sorts_ascending():
    loads = src._parse_lambdas("38, 26,32")
    assert loads.tolist() == [26.0, 32.0, 38.0]


@pytest.mark.parametrize("bad", ["", "0", "-1", "32,32", "nan", "inf", "abc"])
def test_parse_lambdas_rejects_invalid(bad):
    with pytest.raises(ValueError):
        src._parse_lambdas(bad)


# ── 主扫参：3D 布局 [n_seeds, n_a, n_lambda] ────────────────

def test_sweep_shape_is_3d():
    cfg = _small_cfg()
    seed = int(cfg["seed"])
    out, policy_diag, work_diag = src._run_composition_load_sweep(
        cfg,
        scales=np.asarray([0.5, 1.0, 2.0]),
        loads=np.asarray([26.0, 32.0]),
        policies=["FCFS"],
        seeds=[seed],
    )
    shape = (1, 3, 2)
    assert out["FCFS"]["mean"].shape == shape
    assert policy_diag["FCFS"]["oom_prob"].shape == shape
    assert work_diag["token_ratio"].shape == shape
    # 逐请求总 token 严格守恒（与 λ、a 无关）
    assert np.all(work_diag["max_request_total_error"] == 0.0)


# ── a=1 / λ=32 锚点：与直接仿真逐位一致 ─────────────────────

def test_identity_anchor_matches_raw_and_restores_generator():
    cfg = _small_cfg()
    original = simulator.generate_heapq
    seed = int(cfg["seed"])

    out, _, work_diag = src._run_composition_load_sweep(
        cfg,
        scales=np.asarray([1.0]),
        loads=np.asarray([32.0]),
        policies=["QBA"],
        seeds=[seed],
    )
    assert simulator.generate_heapq is original  # 已恢复

    ref = run_sim_stats(cfg, _overrides("QBA", seed))
    assert out["QBA"]["mean"][0, 0, 0] == ref["mean"]
    assert out["QBA"]["p99"][0, 0, 0] == ref["p99"]
    assert work_diag["max_request_total_error"][0, 0, 0] == 0.0


def test_generator_restored_on_failure(monkeypatch):
    """仿真抛异常时 finally 仍恢复原生成器，并带上实验上下文。"""
    cfg = _small_cfg()
    original = simulator.generate_heapq

    def _boom(cfg):
        raise RuntimeError("boom")

    monkeypatch.setattr(src, "run_one_sim", _boom)
    with pytest.raises(RuntimeError,
                       match=r"Experiment failed: seed=.* lambda=.* policy="):
        src._run_composition_load_sweep(
            cfg,
            scales=np.asarray([2.0]),
            loads=np.asarray([38.0]),
            policies=["FCFS"],
            seeds=[42],
        )
    assert simulator.generate_heapq is original
