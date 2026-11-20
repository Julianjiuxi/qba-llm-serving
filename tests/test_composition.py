"""prefill/decode 构成扫描（fig10）适配层的一致性测试。

只验证核心不变量（总 token 守恒、构成比按倍率变化、a=1 还原、标签重算、适配层可恢复），
不复现百种子回归、不重跑全量 sweep。
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
from scripts.experiment import sweep_composition as sc  # noqa: E402
from scripts.experiment._common import run_sim_stats  # noqa: E402

_BASE_YAML = _REPO_ROOT / "scripts" / "experiment" / "configs" / "sweep_composition_base.yaml"


def _small_cfg() -> dict:
    """小规模配置：只用于验证逻辑，不用于科学结论。"""
    cfg = load_config(_BASE_YAML)
    cfg["admission_number"] = 300
    return cfg


def _overrides(policy: str, seed: int) -> dict:
    return {"lambda_arrival": 32.0, "schedule_method": policy, "seed": int(seed)}


def _base_labels(cfg: dict) -> dict:
    heap = generate_heapq(dict(cfg))
    return {r.id: r.true_label for _, _, r in heap}


def _snapshot(cfg: dict, scale: float, base_labels: dict | None = None) -> tuple[dict, dict]:
    """按给定倍率生成并变换请求，返回 ({id: Request}, diag)。"""
    diag: dict = {}
    heap = generate_heapq(dict(cfg))
    sc._apply_composition(heap, scale, cfg, diag, base_labels)
    return {r.id: r for _, _, r in heap}, diag


# ── 倍率解析 ────────────────────────────────────────────────

def test_parse_composition_scales_sorts_ascending():
    scales = sc._parse_composition_scales("1, 4,0.25,2,0.5")
    assert scales.tolist() == [0.25, 0.5, 1.0, 2.0, 4.0]


@pytest.mark.parametrize("bad", ["", "0", "-1", "1,1", "nan", "inf", "1,abc"])
def test_parse_composition_scales_rejects_invalid(bad):
    with pytest.raises(ValueError):
        sc._parse_composition_scales(bad)


# ── 验证 A：a=1 完全还原原始 workload（含标签） ──────────────

def test_scale_one_is_identity():
    cfg = _small_cfg()
    raw = {r.id: r for _, _, r in generate_heapq(dict(cfg))}
    got, diag = _snapshot(cfg, 1.0)

    assert set(raw) == set(got)
    for rid, r in got.items():
        b = raw[rid]
        assert r.prompt_len == b.prompt_len
        assert r.prefill_res_token == b.prompt_len
        assert r.answer_len == b.answer_len
        assert r.true_label == b.true_label
        assert r.predict_label == b.predict_label
    assert diag["norm_scale"] == pytest.approx(1.0, abs=1e-12)
    assert abs(diag["total_rel_error"]) <= 1e-9


# ── 验证 B：总 token 需求守恒 ───────────────────────────────

@pytest.mark.parametrize("a", [0.25, 0.5, 2.0, 4.0])
def test_total_token_demand_conserved(a):
    cfg = _small_cfg()
    base, base_diag = _snapshot(cfg, 1.0)
    got, diag = _snapshot(cfg, a)

    base_total = np.mean([r.prompt_len + r.answer_len for r in base.values()])
    new_total = np.mean([r.prompt_len + r.answer_len for r in got.values()])
    # 总 token 需求相对误差应在校准容差内
    assert abs(new_total - base_total) / base_total <= sc._NORM_TOL
    assert abs(diag["total_rel_error"]) <= sc._NORM_TOL


# ── 验证 C：构成比按倍率 a 变化 ─────────────────────────────

@pytest.mark.parametrize("a", [0.25, 0.5, 2.0, 4.0])
def test_composition_ratio_scaled(a):
    cfg = _small_cfg()
    base, _ = _snapshot(cfg, 1.0)
    got, _ = _snapshot(cfg, a)

    base_ratio = np.mean([r.prompt_len for r in base.values()]) / np.mean(
        [r.answer_len for r in base.values()])
    new_ratio = np.mean([r.prompt_len for r in got.values()]) / np.mean(
        [r.answer_len for r in got.values()])
    # 允许 round/clip 造成的小偏差
    assert new_ratio == pytest.approx(a * base_ratio, rel=0.05)


# ── 验证 D：标签按新 answer 分布重算且为 oracle ─────────────

def test_labels_recomputed_on_new_distribution():
    cfg = _small_cfg()
    base = {r.id: r for _, _, r in generate_heapq(dict(cfg))}
    labels = {r.id: r.true_label for r in base.values()}

    got, diag = _snapshot(cfg, 4.0, base_labels=labels)

    ans = np.array([r.answer_len for r in got.values()], dtype=np.float64)
    thr = float(np.quantile(ans, float(cfg["label_threshold_percentail"])))
    for r in got.values():
        expected = "short" if r.answer_len <= thr else "long"
        assert r.true_label == expected
        assert r.predict_label == r.true_label  # oracle（无误判）
    # 公共缩放保持排序，label flip 比例应为极小值
    assert diag["label_flip_fraction"] <= 0.05


# ── 验证 E：a=1 数值锚点 + 恢复路径 ─────────────────────────

def test_scale_one_matches_raw_simulation():
    """经 a=1 适配层的仿真，与直接运行的指标逐位一致，且生成器已被恢复。"""
    cfg = _small_cfg()
    original = simulator.generate_heapq
    policies = ["FCFS", "QBA"]
    seed = int(cfg["seed"])

    rows, _, work_diag = sc._run_composition_sweep(
        cfg, 32.0, np.asarray([1.0]), policies, [seed],
    )
    assert simulator.generate_heapq is original  # 已恢复
    for pol in policies:
        ref = run_sim_stats(cfg, _overrides(pol, seed))
        assert rows[pol]["mean"][0, 0] == ref["mean"]
        assert rows[pol]["p99"][0, 0] == ref["p99"]
    assert abs(work_diag["total_rel_error"][0, 0]) <= 1e-9


def test_generator_restored_on_failure(monkeypatch):
    """仿真抛异常时，finally 仍恢复原生成器，并带上实验上下文。"""
    cfg = _small_cfg()
    original = simulator.generate_heapq

    def _boom(cfg):
        raise RuntimeError("boom")

    monkeypatch.setattr(sc, "run_one_sim", _boom)
    with pytest.raises(RuntimeError, match=r"Experiment failed: seed=.*composition_scale=.*policy="):
        sc._run_composition_sweep(cfg, 32.0, np.asarray([2.0]), ["FCFS"], [42])
    assert simulator.generate_heapq is original


# ── 验证 F：归一化强制验收 / 确定性标签 / 新增诊断 ──────────

def test_apply_composition_does_not_consume_randomness():
    """确定性标签重算不得改变全局随机状态（不消耗额外随机数）。"""
    cfg = _small_cfg()
    heap = generate_heapq(dict(cfg))
    labels = {r.id: r.true_label for _, _, r in heap}

    np.random.seed(20260101)
    before = np.random.get_state()
    sc._apply_composition(heap, 4.0, cfg, {}, labels)
    after = np.random.get_state()

    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_non_convergence_raises(monkeypatch):
    """迭代耗尽仍未达标时必须报错，不得在未收敛时继续仿真。"""
    cfg = _small_cfg()
    heap = generate_heapq(dict(cfg))
    monkeypatch.setattr(sc, "_NORM_MAX_ITER", 1)
    monkeypatch.setattr(sc, "_NORM_TOL", 1e-15)
    with pytest.raises(RuntimeError, match=r"归一化未在 .* 次迭代内收敛"):
        sc._apply_composition(heap, 4.0, cfg, {}, None)


def test_non_oracle_config_rejected():
    """非 oracle 误判率配置应被显式拒绝。"""
    cfg = _small_cfg()
    cfg["FN"] = 0.1
    heap = generate_heapq(dict(cfg))
    with pytest.raises(ValueError, match=r"oracle"):
        sc._apply_composition(heap, 2.0, cfg, {}, None)


def test_work_diag_records_pi_s_and_base_total():
    cfg = _small_cfg()
    got, diag = _snapshot(cfg, 4.0)

    # pi_s 应与真实 short 占比一致
    n_short = sum(1 for r in got.values() if r.true_label == "short")
    assert diag["pi_s"] == pytest.approx(n_short / len(got))
    assert 0.0 <= diag["pi_s"] <= 1.0

    # base_total_mean 为 a=1 基准总需求，且变换后总 token 在容差内守恒
    _, diag_one = _snapshot(cfg, 1.0)
    assert diag["base_total_mean"] == pytest.approx(diag_one["base_total_mean"])
    assert diag["total_mean"] == pytest.approx(diag["base_total_mean"], rel=sc._NORM_TOL)
