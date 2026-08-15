"""公式层模块：reduced M/G/n mean-level surrogate model。

核心功能：
- 从 FCFS trace 校准模型参数（有效服务台数 n_eff、长/短类别矩）
- 从 AA/QBA trace 拟合经验 promotion phi 曲线
- 计算 FCFS / PSF / AA / QBA 的近似平均等待时间曲线

所有 trace 相关函数已适配新引擎的 EngineState dataclass。
"""

from llm_in_queuing.formula.phi_model import (
    ModelParams,
    erlang_c_delay_prob,
    allen_cunneen_mean_wq,
    mixture_moments,
    psf_wait_approx,
    promoted_class_moments,
    psfa_phi_from_params,
    psfaq_phi_from_params,
    compute_curves,
    run_trace,
    extract_reference_stats_from_state,
    empirical_phi_from_policy_state,
    calibrate_effective_n,
    choose_reference_load,
    estimate_model_params_from_fcfs_trace,
    fit_phi_curve_from_policy_traces,
    maybe_attach_empirical_phi,
    maybe_attach_empirical_phi_psfaq,
)
