# ============================================================
# Fig.2–6 全套实验一键运行脚本（seed_count=1）
# 运行方式：在 LLM_in_Queuing 目录下执行 .\scripts\AE.ps1
# ============================================================

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = "$ROOT\src"
$PLOT_DIR = "$ROOT\scripts\plot"

Set-Location $ROOT

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Fig.2 — lambda sweep (SIM + FORMULA)" -ForegroundColor Cyan
Write-Host "========================================"
python -m scripts.experiment.sweep_lambda --seed-count 100
python -m scripts.draw.draw_fig2 --data "$PLOT_DIR\fig2_data.npz"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Fig.3 — quota_k sensitivity" -ForegroundColor Cyan
Write-Host "========================================"
python -m scripts.experiment.sweep_quotak --seed-count 100 --reference-load 32.0
python -m scripts.draw.draw_fig3 --data "$PLOT_DIR\fig3_data.npz"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Fig.4 — gamma sensitivity" -ForegroundColor Cyan
Write-Host "========================================"
python -m scripts.experiment.sweep_gamma --seed-count 100 --reference-load 32.0
python -m scripts.draw.draw_fig4 --data "$PLOT_DIR\fig4_data.npz"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Fig.5 — epsilonSL sweep (epsilonLS=0)" -ForegroundColor Cyan
Write-Host "========================================"
python -m scripts.experiment.sweep_epsilonSL --seed-count 100
python -m scripts.draw.draw_fig5 --data "$PLOT_DIR\fig5_data.npz"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Fig.6 — epsilonLS sweep (epsilonSL=0)" -ForegroundColor Cyan
Write-Host "========================================"
python -m scripts.experiment.sweep_epsilonLS --seed-count 100
python -m scripts.draw.draw_fig6 --data "$PLOT_DIR\fig6_data.npz"

Write-Host "========================================" -ForegroundColor Green
Write-Host "  全部完成！输出目录: $PLOT_DIR"        -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
