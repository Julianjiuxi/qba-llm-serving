"""Real trace data processing — download, parse, extract workload parameters."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── constants ──────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent
CSV_NAME = "AzureLLMInferenceTrace_conv_1week.csv"
CSV_URL = (
    "https://github.com/Azure/AzurePublicDataset/releases/download/"
    f"dataset-llm-2024/{CSV_NAME}"
)

# ── download ───────────────────────────────────────────────────────────────

def download_csv(force: bool = False) -> Path:
    """Download the Azure LLM inference trace CSV if not already cached.

    The trace contains ~27M conversation requests from production Azure
    endpoints (May 2024, 1 week).  Each row is a single request with
    ``TIMESTAMP``, ``ContextTokens``, ``GeneratedTokens``.

    Args:
        force: Re-download even if the file already exists.

    Returns:
        Absolute path to the local CSV file.
    """
    csv_path = DATA_DIR / CSV_NAME

    if csv_path.exists() and not force:
        size_gb = csv_path.stat().st_size / (1024**3)
        print(f"CSV already cached: {csv_path} ({size_gb:.2f} GB)")
        return csv_path

    print(f"Downloading {CSV_NAME} …")
    print(f"  from: {CSV_URL}")

    import urllib.request

    def _report(count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        pct = min(count * block_size / total_size * 100.0, 100.0)
        downloaded_gb = count * block_size / (1024**3)
        total_gb = total_size / (1024**3)
        sys.stdout.write(
            f"\r  {pct:5.1f} %   {downloaded_gb:.2f} / {total_gb:.2f} GB"
        )
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(str(CSV_URL), str(csv_path), reporthook=_report)
    except Exception as exc:
        # Clean up partial download
        if csv_path.exists():
            csv_path.unlink()
        raise RuntimeError(f"Download failed: {exc}") from exc

    print()  # newline after progress bar
    final_gb = csv_path.stat().st_size / (1024**3)
    print(f"Downloaded  →  {csv_path} ({final_gb:.2f} GB)")
    return csv_path


# ── sample subsets ─────────────────────────────────────────────────────────

_SAMPLE_SEED: int = 42
_SAMPLE_N: int = 100
_SAMPLE_SIZE: int = 5000
_CHUNK_SIZE: int = 500_000


def sample_subsets(
    n_subsets: int = _SAMPLE_N,
    subset_size: int = _SAMPLE_SIZE,
    seed: int = _SAMPLE_SEED,
) -> list[Path]:
    """Extract non-overlapping consecutive subsets from the trace CSV.

    Uses a fixed seed so that repeated calls produce identical subsets.
    Subsets are streamed through the 1 GB CSV in chunks — the full file
    is never loaded into memory.

    Args:
        n_subsets:   Number of subset files to generate (default 100).
        subset_size: Requests per subset (default 5000).
        seed:        Random seed for reproducibility.

    Returns:
        Absolute paths to the generated ``Azure_work_load_subset*.csv`` files.
    """
    import random

    import pandas as pd

    csv_path = download_csv()

    # ── 1. count total rows (chunked, no full load) ──────────────────────
    print(f"Counting rows in {csv_path.name} …")
    n_total = 0
    for chunk in pd.read_csv(csv_path, chunksize=_CHUNK_SIZE, usecols=[0]):
        n_total += len(chunk)
    print(f"  Total rows: {n_total:,}")

    # ── 2. generate guaranteed non-overlapping starts ────────────────────
    rng = random.Random(seed)
    total_gap = n_total - n_subsets * subset_size
    if total_gap < 0:
        raise ValueError(
            f"CSV too small: need {n_subsets * subset_size:,} rows, "
            f"have {n_total:,}"
        )
    # Sample K cut points from [0, total_gap]; sort → shift by i*S
    cuts = sorted(rng.sample(range(total_gap + 1), n_subsets))
    starts = [cuts[i] + i * subset_size for i in range(n_subsets)]

    intervals = [(s, s + subset_size, i) for i, s in enumerate(starts)]
    # intervals are sorted by construction (starts is ascending)

    total_extracted = n_subsets * subset_size
    print(f"Extracting {n_subsets} subsets × {subset_size:,} = "
          f"{total_extracted:,} rows …")

    # ── 3. open output files (overwrite) ────────────────────────────────
    writers: list = []
    paths: list[Path] = []
    header_written: list[bool] = [False] * n_subsets

    for i in range(n_subsets):
        fname = f"Azure_work_load_subset{i + 1:03d}.csv"
        fpath = DATA_DIR / fname
        writers.append(open(str(fpath), "w", newline="", encoding="utf-8"))
        paths.append(fpath)

    # ── 4. single-pass streaming extraction ─────────────────────────────
    global_idx = 0
    n_chunks = 0

    try:
        for chunk in pd.read_csv(csv_path, chunksize=_CHUNK_SIZE):
            n = len(chunk)
            chunk_end = global_idx + n

            for s, e, sub_idx in intervals:
                if e <= global_idx:
                    continue          # interval fully before this chunk
                if s >= chunk_end:
                    break             # interval fully after → remaining too

                # Overlap region within this chunk
                row_start = max(s, global_idx) - global_idx
                row_end = min(e, chunk_end) - global_idx
                sub_chunk = chunk.iloc[row_start:row_end]

                hdr = not header_written[sub_idx]
                sub_chunk.to_csv(writers[sub_idx], index=False, header=hdr)
                header_written[sub_idx] = True

            global_idx += n
            n_chunks += 1
            if n_chunks % 10 == 0:
                pct = min(global_idx / n_total * 100.0, 100.0)
                sys.stdout.write(f"\r  {pct:.1f} %  ({global_idx:,} / {n_total:,})")
                sys.stdout.flush()

        print(f"\r  100.0 %  ({n_total:,} / {n_total:,})")
    finally:
        for f in writers:
            f.close()

    # ── 5. verify ───────────────────────────────────────────────────────
    print("Verifying …")
    all_ok = True
    for i, p in enumerate(paths):
        n = 0
        with open(p, "r") as f:
            # count lines minus header
            n = sum(1 for _ in f) - 1
        status = "OK" if n == subset_size else f"MISMATCH (expected {subset_size})"
        if n != subset_size:
            all_ok = False
        print(f"  {p.name}: {n} rows  {status}")
    if all_ok:
        print(f"All {n_subsets} subsets verified ({subset_size:,} rows each).")
    else:
        print("WARNING: some subsets have unexpected row counts!")

    return paths


# ── normalize time ────────────────────────────────────────────────────────

_SUBSET_GLOB = "Azure_work_load_subset*.csv"


def normalize_time() -> list[Path]:
    """Add a ``TimeSeconds`` column to every subset CSV.

    Each subset is normalized **independently**: its own earliest
    ``TIMESTAMP`` becomes ``t=0.0``.  The column is computed as::

        TimeSeconds = (TIMESTAMP - subset_origin).total_seconds()

    This makes each subset a self-contained workload trace ready for
    multi-seed simulation — every subset starts at the same logical
    time zero.

    The column order becomes::

        TimeSeconds, TIMESTAMP, ContextTokens, GeneratedTokens

    Files are overwritten in-place.

    Returns:
        Paths to the updated CSV files.
    """
    import pandas as pd

    paths = sorted(DATA_DIR.glob(_SUBSET_GLOB))
    if not paths:
        raise FileNotFoundError(
            f"No '{_SUBSET_GLOB}' files found in {DATA_DIR}.  "
            f"Run 'python data/process.py sample' first."
        )

    print(f"Normalizing {len(paths)} subsets (independent per-subset origin) …")
    for i, p in enumerate(paths):
        df = pd.read_csv(p)
        ts = pd.to_datetime(df["TIMESTAMP"], format="mixed", utc=True)
        origin = ts.iloc[0]
        df.drop(columns=["TimeSeconds"], inplace=True, errors="ignore")
        df.insert(0, "TimeSeconds", (ts - origin).dt.total_seconds())
        df.to_csv(p, index=False)

        last_sec = df["TimeSeconds"].iloc[-1]
        if (i + 1) % 10 == 0 or i == 0 or i == len(paths) - 1:
            sys.stdout.write(
                f"\r  [{i + 1:3d}/{len(paths)}]  {p.name}  "
                f"origin={origin}  span=0–{last_sec:.1f}s"
            )
            sys.stdout.flush()

    print(f"\nDone.  All {len(paths)} subsets updated (each starts at t=0).")
    print(f"Column order: TimeSeconds, TIMESTAMP, ContextTokens, GeneratedTokens")
    return paths


# ── compute token scale ────────────────────────────────────────────────────

# Synthetic workload parameters matching sweep_lambda_base.yaml
# (the config actually used to generate Fig.2–6)
_SYN_DEFAULTS: dict = {
    "norm_ratio": 0.7,
    "norm_loc": 100,
    "norm_scale": 30,
    "pareto_shape": 1.4,
    "pareto_scale": 40,
    "beta0": 3,
    "beta1": 0.75,
    "sigma": 0.3,
    "prompt_len_cap": 4096,
    "answer_len_cap": 4096,
    "seed": 42,
}


def _synthetic_e_pd(syn_cfg: dict, n_samples: int = 500_000) -> float:
    """Estimate E[P+D] of the synthetic workload by Monte Carlo.

    Uses the same distribution logic as ``workload.py::generate_heapq``
    without the scheduling simulation overhead.
    """
    import numpy as np

    rng = np.random.default_rng(int(syn_cfg["seed"]))

    nc = float(syn_cfg["norm_ratio"])
    p_cap = int(syn_cfg["prompt_len_cap"])
    a_cap = int(syn_cfg["answer_len_cap"])

    # P: normal-pareto mixture (same as workload.py)
    use_norm = rng.random(n_samples) < nc
    p_norm = rng.normal(float(syn_cfg["norm_loc"]), float(syn_cfg["norm_scale"]), n_samples)
    p_pareto = (rng.pareto(float(syn_cfg["pareto_shape"]), n_samples) + 1.0) * float(syn_cfg["pareto_scale"])
    p = np.where(use_norm, p_norm, p_pareto)
    p = np.clip(np.maximum(p, 5), 0, p_cap)

    # A: lognormal conditional on P
    mu = float(syn_cfg["beta0"]) + float(syn_cfg["beta1"]) * np.log1p(p)
    a = rng.lognormal(mu, float(syn_cfg["sigma"]))
    a = np.clip(np.maximum(a, 1), 0, a_cap)

    total = p + a
    return float(total.mean())


def _azure_pooled_e_pd(
    subset_glob: str = _SUBSET_GLOB,
) -> tuple[float, int]:
    """Read all Azure subset CSVs and return pooled E[P+D] and total row count.

    Returns:
        (mean_pd, total_rows) — mean of ``ContextTokens + GeneratedTokens``
        across all matching subset files, and the total number of requests.
    """
    import pandas as pd

    paths = sorted(DATA_DIR.glob(subset_glob))
    if not paths:
        raise FileNotFoundError(
            f"No files matching '{subset_glob}' found in {DATA_DIR}."
        )

    total_sum = 0.0
    total_rows = 0
    for i, p in enumerate(paths):
        df = pd.read_csv(p)
        col_p = "ContextTokens"
        col_a = "GeneratedTokens"
        if col_p not in df.columns or col_a not in df.columns:
            raise KeyError(
                f"Expected columns {col_p}/{col_a} not in {p.name}. "
                f"Available: {list(df.columns)}"
            )
        pd_sum = (df[col_p].astype(float) + df[col_a].astype(float)).sum()
        total_sum += float(pd_sum)
        total_rows += len(df)
        if (i + 1) % 10 == 0 or i == 0 or i == len(paths) - 1:
            sys.stdout.write(f"\r  [{i + 1:3d}/{len(paths)}]  {p.name}")
            sys.stdout.flush()

    mean_pd = total_sum / total_rows
    print(f"\n  Pooled: {total_rows:,} requests across {len(paths)} subsets")
    print(f"  Pooled E[P+D] = {mean_pd:.2f}")
    return mean_pd, total_rows


def compute_token_scale(
    subset_glob: str = _SUBSET_GLOB,
    syn_cfg: dict | None = None,
    n_mc: int = 500_000,
) -> float:
    """Compute the global token scale ``s_tok``.

    .. math::

        s_{\rm tok} = \\frac{E[P+D]_{\\rm synthetic}}{E[P+D]_{\\rm Azure, pooled}}

    The denominator pools all request rows from every matching subset CSV.
    The numerator is a Monte Carlo estimate using the synthetic workload
    distribution (default: ``sweep_lambda_base.yaml`` parameters).

    All Azure windows share the same single ``s_tok`` — window-level
    request-size variation is deliberately preserved.

    Args:
        subset_glob: Glob pattern for subset CSVs (default ``Azure_work_load_subset*.csv``).
        syn_cfg:     Override synthetic distribution parameters.
        n_mc:        Monte Carlo sample count for the synthetic E[P+D] estimate.

    Returns:
        ``s_tok`` as a float.
    """
    print("=== Computing global token scale ======================")
    print(f"  Subset glob: {subset_glob}")
    print(f"  MC samples:  {n_mc:,}")

    # ── Azure pooled E[P+D] ─────────────────────────────────────────────
    print("\n[1/2] Azure pooled E[P+D] …")
    az_mean, az_rows = _azure_pooled_e_pd(subset_glob)

    # ── Synthetic E[P+D] ────────────────────────────────────────────────
    print("\n[2/2] Synthetic E[P+D] …")
    if syn_cfg is None:
        syn_cfg = dict(_SYN_DEFAULTS)
    syn_mean = _synthetic_e_pd(syn_cfg, n_samples=n_mc)
    print(f"  E[P+D]_synthetic = {syn_mean:.2f}")
    for k in ["norm_ratio", "norm_loc", "norm_scale", "pareto_shape",
              "pareto_scale", "beta0", "beta1", "sigma"]:
        print(f"    {k} = {syn_cfg[k]}")

    # ── Scale ───────────────────────────────────────────────────────────
    s_tok = syn_mean / az_mean
    print(f"\n  s_tok = {syn_mean:.2f} / {az_mean:.2f} = {s_tok:.6f}")
    print("========================================================")
    return s_tok


# ── cli ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Azure LLM trace data utilities")
    sub = parser.add_subparsers(dest="command")

    dl = sub.add_parser("download", help="Download the raw CSV (if not cached)")
    dl.add_argument("--force", action="store_true",
                    help="Re-download even if cached")

    sm = sub.add_parser("sample", help="Extract non-overlapping random subsets")
    sm.add_argument("--n-subsets", type=int, default=_SAMPLE_N,
                    help=f"Number of subsets (default: {_SAMPLE_N})")
    sm.add_argument("--subset-size", type=int, default=_SAMPLE_SIZE,
                    help=f"Rows per subset (default: {_SAMPLE_SIZE})")
    sm.add_argument("--seed", type=int, default=_SAMPLE_SEED,
                    help=f"Random seed (default: {_SAMPLE_SEED})")

    nm = sub.add_parser("normalize", help="Add TimeSeconds column to all subsets")

    ts = sub.add_parser("token-scale",
                         help="Compute global token scale s_tok = E[P+D]_syn / E[P+D]_Azure_pooled")
    ts.add_argument("--subset-glob", default=_SUBSET_GLOB,
                    help=f"Glob for subset CSVs (default: {_SUBSET_GLOB})")
    ts.add_argument("--n-mc", type=int, default=500_000,
                    help="Monte Carlo samples for synthetic E[P+D] (default: 500_000)")
    ts.add_argument("--norm-ratio", type=float, default=None,
                    help="Override synthetic norm_ratio")
    ts.add_argument("--norm-loc", type=float, default=None,
                    help="Override synthetic norm_loc")
    ts.add_argument("--norm-scale", type=float, default=None,
                    help="Override synthetic norm_scale")
    ts.add_argument("--beta0", type=float, default=None,
                    help="Override synthetic beta0")
    ts.add_argument("--beta1", type=float, default=None,
                    help="Override synthetic beta1")
    ts.add_argument("--sigma", type=float, default=None,
                    help="Override synthetic sigma")

    args = parser.parse_args()

    if args.command == "download":
        download_csv(force=args.force)
    elif args.command == "sample":
        sample_subsets(
            n_subsets=args.n_subsets,
            subset_size=args.subset_size,
            seed=args.seed,
        )
    elif args.command == "normalize":
        normalize_time()
    elif args.command == "token-scale":
        syn_cfg = dict(_SYN_DEFAULTS)
        for key in ["norm_ratio", "norm_loc", "norm_scale",
                    "pareto_shape", "pareto_scale",
                    "beta0", "beta1", "sigma"]:
            val = getattr(args, key.replace("_", "_"), None)
            if val is not None:
                syn_cfg[key] = val
        compute_token_scale(
            subset_glob=args.subset_glob,
            syn_cfg=syn_cfg,
            n_mc=args.n_mc,
        )
    else:
        parser.print_help()
