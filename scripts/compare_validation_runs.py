"""Quick comparison for ISMN validation output folders.

Usage:
  python scripts/compare_validation_runs.py \
    --run f2pyw=results/ismn_validation_f2pyw \
    --run gdf6e=results/ismn_validation_gdf6e

Prints station-level pass rate, metric summaries, and depth/horizon breakdown.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _latest_csv(run_dir: Path, pattern: str) -> Path:
    files = sorted(run_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match {pattern} in {run_dir}")
    return files[-1]


def _pick_csv(run_dir: Path, fixed_name: str, pattern: str) -> Path:
    fixed = run_dir / fixed_name
    if fixed.exists():
        return fixed
    return _latest_csv(run_dir, pattern)


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _pick_col(cols: list[str], *candidates: str) -> str:
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"None of columns found: {candidates}. Available: {cols}")


def summarize_run(name: str, run_dir: Path) -> None:
    station_path = _pick_csv(
        run_dir, "station_summary.csv", "station_summary_*.csv")
    depth_path = _pick_csv(run_dir, "depth_summary.csv", "depth_summary_*.csv")
    horizon_path = _pick_csv(
        run_dir, "horizon_summary.csv", "horizon_summary_*.csv")

    st = _norm_cols(pd.read_csv(station_path))
    station_col = next(c for c in st.columns if c.startswith("station"))
    pass_col = next((c for c in st.columns if c in {"pass", "passed", "is_pass"}), None) or next(
        c for c in st.columns if "pass" in c
    )
    rmse_col = next((c for c in st.columns if c == "rmse" or c.endswith("rmse")), None) or next(
        c for c in st.columns if "rmse" in c
    )
    kge_col = next((c for c in st.columns if c == "kge" or c.endswith("kge")), None) or next(
        c for c in st.columns if c.startswith("kge")
    )
    nse_col = next((c for c in st.columns if c ==
                   "nse" or c.endswith("nse")), None)

    passed = st[pass_col].astype(str).str.contains(
        "✓|true|1", case=False, regex=True)
    st["_rmse"] = pd.to_numeric(st[rmse_col], errors="coerce")
    st["_kge"] = pd.to_numeric(st[kge_col], errors="coerce")
    if nse_col:
        st["_nse"] = pd.to_numeric(st[nse_col], errors="coerce")

    print("\n" + "=" * 28 + f" {name} " + "=" * 28)
    print(f"dir: {run_dir}")
    print(f"station_summary: {station_path.name}")
    print(
        f"stations: {len(st)} | pass: {passed.sum()}/{len(st)} ({passed.mean() * 100:.1f}%)")
    print(
        f"RMSE mean±std: {st['_rmse'].mean():.4f} ± {st['_rmse'].std():.4f} | min={st['_rmse'].min():.4f} max={st['_rmse'].max():.4f}"
    )
    print(
        f"KGE  mean±std: {st['_kge'].mean():.4f} ± {st['_kge'].std():.4f} | min={st['_kge'].min():.4f} max={st['_kge'].max():.4f}"
    )
    if nse_col:
        print(
            f"NSE  mean±std: {st['_nse'].mean():.4f} ± {st['_nse'].std():.4f} | min={st['_nse'].min():.4f} max={st['_nse'].max():.4f}"
        )

    worst = st.sort_values("_kge").head(
        8)[[station_col, "_rmse", "_kge", pass_col]]
    best = st.sort_values("_kge", ascending=False).head(
        8)[[station_col, "_rmse", "_kge", pass_col]]
    print("\nworst KGE:")
    print(worst.to_string(index=False))
    print("\nbest KGE:")
    print(best.to_string(index=False))

    d = _norm_cols(pd.read_csv(depth_path))
    depth_col = next(c for c in d.columns if "depth" in c)
    rmse_d = next(c for c in d.columns if "rmse" in c)
    kge_d = next(c for c in d.columns if c.startswith("kge"))
    d["_rmse"] = pd.to_numeric(d[rmse_d], errors="coerce")
    d["_kge"] = pd.to_numeric(d[kge_d], errors="coerce")
    print("\nby depth (rmse,kge):")
    print(d[[depth_col, "_rmse", "_kge"]].sort_values(
        depth_col).to_string(index=False))

    h = _norm_cols(pd.read_csv(horizon_path))
    horizon_col = next(c for c in h.columns if "horizon" in c)
    rmse_h = next(c for c in h.columns if "rmse" in c)
    kge_h = next(c for c in h.columns if c.startswith("kge"))
    h["_rmse"] = pd.to_numeric(h[rmse_h], errors="coerce")
    h["_kge"] = pd.to_numeric(h[kge_h], errors="coerce")
    print("\nby horizon (rmse,kge):")
    print(h[[horizon_col, "_rmse", "_kge"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec like name=path/to/results_dir (repeatable)",
    )
    args = parser.parse_args()

    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"Invalid --run {spec!r}; expected name=path")
        name, run_dir = spec.split("=", 1)
        summarize_run(name, Path(run_dir))


if __name__ == "__main__":
    main()
