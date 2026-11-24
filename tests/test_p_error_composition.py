"""fig12: p × 预测误差 × 构成 联合分析适配层的一致性测试。

只验证核心不变量（误差在真标签重算之后注入、方向互斥、Oracle 不消耗随机数、
预测计数与 π̂_S/χ_pred 公式、φ_Q 分母为 N_predicted_long、a=1/oracle 锚点与直接仿真一致、
生成器可恢复、4D 数组布局），不复现百种子回归、不重跑全量 sweep。
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
from llm_in_queuing.types import Request  # noqa: E402
from llm_in_queuing.workload import generate_heapq  # noqa: E402
from scripts.experiment import sweep_p_error_composition as src  # noqa: E402
from scripts.experiment._common import run_sim_stats  # noqa: E402

_BASE_YAML = (_REPO_ROOT / "scripts" / "experiment" / "configs"
              / "sweep_p_error_composition_base.yaml")


def _small_cfg() -> dict:
    """小规模配置：只用于验证逻辑，不用于科学结论。"""
    cfg = load_config(_BASE_YAML)
    cfg["admission_number"] = 300
    return cfg


def _overrides(policy: str, seed: int) -> dict:
    return {"lambda_arrival": 32.0, "schedule_method": policy, "seed": int(seed)}


def _synth_reqs(n_short: int, n_long: int) -> list:
    """构造带 true_label 的合成请求（predict_label 由注入函数写入）。"""
    reqs = []
    for i in range(n_short):
        r = Request(id=i, arrival_time=0.0, prompt_len=100,
                    answer_len=10, prefill_res_token=100)
        r.true_label = "short"
        r.predict_label = "short"
        reqs.append(r)
    for i in range(n_long):
        r = Request(id=n_short + i, arrival_time=0.0, prompt_len=100,
                    answer_len=10, prefill_res_token=100)
        r.true_label = "long"
        r.predict_label = "long"
        reqs.append(r)
    return reqs


def _gen_once(cfg: dict, a: float, eps_sl: float, eps_ls: float):
    """按 (a, ε_SL, ε_LS) 生成一份变换+注入后的请求堆与诊断。"""
    original = simulator.generate_heapq
    diag: dict = {}
    heap_base = original(dict(cfg))
    base_labels = {r.id: r.true_label for _, _, r in heap_base}
    gen = src._make_generator(original, float(a), float(eps_sl), float(eps_ls),
                              diag, base_labels)
    heap = gen(dict(cfg))
    return [req for _, _, req in heap], diag


# ── 解析 ────────────────────────────────────────────────────

def test_parse_p_values_sorts_ascending():
    p = src._parse_p_values("0.95, 0.75,0.85")
    assert p.tolist() == [0.75, 0.85, 0.95]


@pytest.mark.parametrize("bad", ["", "0", "1", "-0.1", "1.5", "0.5,0.5", "nan", "abc"])
def test_parse_p_values_rejects_invalid(bad):
    with pytest.raises(ValueError):
        src._parse_p_values(bad)


def test_parse_error_specs_preserves_order():
    specs = src._parse_error_specs("ls0.3, oracle, sl0.1")
    assert [name for name, _, _ in specs] == ["ls0.3", "oracle", "sl0.1"]
    assert specs[0][1:] == (0.0, 0.3)
    assert specs[1][1:] == (0.0, 0.0)
    assert specs[2][1:] == (0.1, 0.0)


@pytest.mark.parametrize("bad", ["", "xx0.1", "oracle,oracle", "SL0.1"])
def test_parse_error_specs_rejects_invalid(bad):
    with pytest.raises(ValueError):
        src._parse_error_specs(bad)


def test_error_table_directions_mutually_exclusive():
    """每个误差设置只在单一方向注入（SL 或 LS），不会同时非零。"""
    for name, (sl, ls) in src._ERROR_SPEC_TABLE.items():
        assert not (sl > 0.0 and ls > 0.0), name
        assert 0.0 <= sl <= 1.0 and 0.0 <= ls <= 1.0


# ── 误差注入：Oracle 不消耗随机数、方向互斥 ─────────────────

def test_inject_oracle_consumes_no_randomness():
    reqs = _synth_reqs(50, 50)
    np.random.seed(20260101)
    before = np.random.get_state()
    src._inject_prediction_errors(reqs, 0.0, 0.0)
    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    # Oracle：预测标签保持原样（未被改写）
    assert all(r.predict_label == r.true_label for r in reqs)


def _count(reqs):
    n_sl = sum(1 for r in reqs if r.true_label == "short" and r.predict_label == "long")
    n_ls = sum(1 for r in reqs if r.true_label == "long" and r.predict_label == "short")
    n_short = sum(1 for r in reqs if r.true_label == "short")
    n_long = sum(1 for r in reqs if r.true_label == "long")
    return n_sl, n_ls, n_short, n_long


def test_inject_sl_only_has_no_ls_errors():
    reqs = _synth_reqs(2000, 2000)
    np.random.seed(7)
    src._inject_prediction_errors(reqs, 0.3, 0.0)
    n_sl, n_ls, n_short, n_long = _count(reqs)
    assert n_ls == 0  # ε_LS = 0 → 不产生 long→short
    assert n_sl > 0
    assert 0.27 <= n_sl / n_short <= 0.33  # ≈ ε_SL，固定种子下确定性


def test_inject_ls_only_has_no_sl_errors():
    reqs = _synth_reqs(2000, 2000)
    np.random.seed(7)
    src._inject_prediction_errors(reqs, 0.0, 0.3)
    n_sl, n_ls, n_short, n_long = _count(reqs)
    assert n_sl == 0  # ε_SL = 0 → 不产生 short→long
    assert n_ls > 0
    assert 0.27 <= n_ls / n_long <= 0.33


# ── 误差在真标签重算之后注入（不被 Oracle 覆盖） ────────────

def test_errors_injected_after_oracle_recompute():
    cfg = _small_cfg()
    reqs, diag = _gen_once(cfg, a=2.0, eps_sl=0.3, eps_ls=0.0)

    # 真标签仍是变换后 answer 分布上的 Oracle 重算（未受注入影响）
    ans = np.array([r.answer_len for r in reqs], dtype=np.float64)
    thr = float(np.quantile(ans, float(cfg["label_threshold_percentail"])))
    for r in reqs:
        assert r.true_label == ("short" if r.answer_len <= thr else "long")

    # 预测标签携带 SL 误差：确实存在 true-short 被判为 long
    n_sl, n_ls, _, _ = _count(reqs)
    assert n_sl > 0
    assert n_ls == 0

    # 诊断中的实际误判率与计数一致
    assert diag["actual_epsilon_SL"] == pytest.approx(
        n_sl / sum(1 for r in reqs if r.true_label == "short"))
    assert diag["actual_epsilon_LS"] == 0.0


def test_work_diag_counts_and_pi_s_pred_identity():
    cfg = _small_cfg()
    reqs, diag = _gen_once(cfg, a=0.5, eps_sl=0.1, eps_ls=0.3)

    n = len(reqs)
    assert diag["n_true_short"] + diag["n_true_long"] == n
    assert diag["n_pred_short"] + diag["n_pred_long"] == n
    assert diag["pi_s_pred"] == pytest.approx(diag["n_pred_short"] / n)

    # π̂_S = π_S(1-ε_SL) + (1-π_S)ε_LS（用实际误判率，等式严格成立）
    pi_s = diag["pi_s"]
    pi_s_pred = pi_s * (1.0 - diag["actual_epsilon_SL"]) \
        + (1.0 - pi_s) * diag["actual_epsilon_LS"]
    assert diag["pi_s_pred"] == pytest.approx(pi_s_pred)


# ── 主扫参：4D 布局 + φ_Q 分母为 N_predicted_long ───────────

def test_sweep_shape_and_phi_q_denominator():
    cfg = _small_cfg()
    seed = int(cfg["seed"])
    out, policy_diag, work_diag = src._run_p_error_composition_sweep(
        cfg, 32.0,
        scales=np.asarray([1.0]),
        percentiles=np.asarray([0.85]),
        error_specs=[("oracle", 0.0, 0.0), ("sl0.3", 0.3, 0.0)],
        policies=["QBA"],
        seeds=[seed],
    )
    shape = (1, 1, 1, 2)
    assert out["QBA"]["mean"].shape == shape
    assert work_diag["n_pred_long"].shape == shape
    assert policy_diag["QBA"]["phi_Q"].shape == shape

    # φ_Q = promoted_total / N_predicted_long（分母为预测 long 计数）
    n_pred_long = work_diag["n_pred_long"][0, 0, 0, 0]
    promoted = policy_diag["QBA"]["promoted_total"][0, 0, 0, 0]
    phi = policy_diag["QBA"]["phi_Q"][0, 0, 0, 0]
    assert n_pred_long > 0
    assert phi == pytest.approx(promoted / n_pred_long)

    # χ_pred = π̂_S / (k(1-π̂_S))
    pis = work_diag["pi_s_pred"][0, 0, 0, 0]
    k = int(cfg["quota_k"])
    assert work_diag["chi_pred"][0, 0, 0, 0] == pytest.approx(pis / (k * (1.0 - pis)))


# ── a=1 / p=base / oracle 锚点：与直接仿真逐位一致 ───────────

def test_identity_anchor_matches_raw_and_restores_generator():
    cfg = _small_cfg()
    original = simulator.generate_heapq
    seed = int(cfg["seed"])

    out, _, work_diag = src._run_p_error_composition_sweep(
        cfg, 32.0,
        scales=np.asarray([1.0]),
        percentiles=np.asarray([float(cfg["label_threshold_percentail"])]),
        error_specs=[("oracle", 0.0, 0.0)],
        policies=["QBA"],
        seeds=[seed],
    )
    assert simulator.generate_heapq is original  # 已恢复

    ref = run_sim_stats(cfg, _overrides("QBA", seed))
    assert out["QBA"]["mean"][0, 0, 0, 0] == ref["mean"]
    assert out["QBA"]["p99"][0, 0, 0, 0] == ref["p99"]
    # a=1 时逐请求总 token 严格守恒，且 oracle 下无误判
    assert work_diag["max_request_total_error"][0, 0, 0, 0] == 0.0
    assert work_diag["actual_epsilon_SL"][0, 0, 0, 0] == 0.0
    assert work_diag["actual_epsilon_LS"][0, 0, 0, 0] == 0.0


def test_generator_restored_on_failure(monkeypatch):
    """仿真抛异常时 finally 仍恢复原生成器，并带上实验上下文。"""
    cfg = _small_cfg()
    original = simulator.generate_heapq

    def _boom(cfg):
        raise RuntimeError("boom")

    monkeypatch.setattr(src, "run_one_sim", _boom)
    with pytest.raises(RuntimeError,
                       match=r"Experiment failed: seed=.* error=.*policy="):
        src._run_p_error_composition_sweep(
            cfg, 32.0,
            scales=np.asarray([2.0]),
            percentiles=np.asarray([0.85]),
            error_specs=[("sl0.1", 0.1, 0.0)],
            policies=["FCFS"],
            seeds=[42],
        )
    assert simulator.generate_heapq is original
