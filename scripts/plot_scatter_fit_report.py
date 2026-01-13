"""Create scatter+fit evaluation plots from paired obs/sim exports.

Reads `paired_obs_sim.csv` (or latest `paired_obs_sim_*.csv`) from each run folder and
creates:
- Per-station scatter plots (3 horizons) for a selected set of stations.
- Multi-site aggregate scatter plots (3 horizons), colored by vegetation proxy bins.

Usage:
  .venv/bin/python scripts/plot_scatter_fit_report.py \
    --run f2pyw=results/ismn_validation/f2pyw/latest \
    --run gdf6e=results/ismn_validation/gdf6e/latest \
    --out results/validation_figures/latest_figures \
    --overwrite

Notes:
- Scatter is x=simulated (pred) vs y=observed (obs).
- Linear fit is least squares: y = a x + b.
- Metrics shown: slope/intercept vs ideal (1/0), RMSE, R².
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required for plotting. Install with: .venv/bin/python -m pip install matplotlib\n"
            f"Original error: {e}"
        )


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


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _rmse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x - y) ** 2)))


def _nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom <= 0:
        return float("nan")
    return float(1.0 - float(np.sum((y_true - y_pred) ** 2)) / denom)


def _r2_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Squared Pearson correlation, bounded to [0, 1].
    if len(y_true) < 2:
        return float("nan")
    if float(np.std(y_true)) <= 1e-12 or float(np.std(y_pred)) <= 1e-12:
        return float("nan")
    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    if not np.isfinite(r):
        return float("nan")
    return float(np.clip(r * r, 0.0, 1.0))


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    # y = a x + b
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def _finite_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def _quantile_bins(values: pd.Series) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    # Use quantiles; fall back to a single bin if too few unique values.
    try:
        q1 = float(v.quantile(1 / 3))
        q2 = float(v.quantile(2 / 3))
    except Exception:
        q1, q2 = float("nan"), float("nan")

    def _bin(x: float) -> str:
        if not np.isfinite(x) or (not np.isfinite(q1)) or (not np.isfinite(q2)) or q1 == q2:
            return "unknown"
        if x <= q1:
            return "low"
        if x <= q2:
            return "mid"
        return "high"

    return v.apply(lambda x: _bin(float(x)) if np.isfinite(x) else "unknown")


def pick_stations(pairs: pd.DataFrame, station_summary: pd.DataFrame, n_sites: int, seed: int) -> list[str]:
    # If station_summary has KGE_mean, sort by it descending
    if "KGE_mean" in station_summary.columns:
        top_stations = station_summary.sort_values("KGE_mean", ascending=False)[
            "station_id"].head(n_sites).tolist()
        return top_stations
    else:
        # Fallback to original logic
        station = pairs.groupby("station_id").agg(
            {"lai_mean": "mean", "ndvi_mean": "mean"}).reset_index()
        station["veg_proxy"] = station["lai_mean"].where(
            np.isfinite(station["lai_mean"]), station["ndvi_mean"])
        station["veg_bin"] = _quantile_bins(station["veg_proxy"])

        rng = np.random.default_rng(seed)

        selected: list[str] = []
        # Aim for diversity across veg bins.
        for b in ["low", "mid", "high"]:
            cand = station[station["veg_bin"] == b]["station_id"].tolist()
            if not cand:
                continue
            k = min(max(1, n_sites // 3), len(cand))
            chosen = rng.choice(cand, size=k, replace=False).tolist()
            selected.extend(chosen)

        if len(selected) < n_sites:
            remaining = [s for s in station["station_id"].tolist()
                         if s not in selected]
            if remaining:
                k = min(n_sites - len(selected), len(remaining))
                selected.extend(rng.choice(
                    remaining, size=k, replace=False).tolist())

        return selected[:n_sites]


@dataclass(frozen=True)
class RunInputs:
    name: str
    run_dir: Path
    pairs_csv: Path
    station_summary_csv: Path


def load_run(name: str, run_dir: Path) -> RunInputs:
    return RunInputs(
        name=name,
        run_dir=run_dir,
        pairs_csv=_pick_csv(run_dir, "paired_obs_sim.csv",
                            "paired_obs_sim_*.csv"),
        station_summary_csv=_pick_csv(run_dir, "station_summary.csv",
                                      "station_summary_*.csv"),
    )


def plot_station_scatter(
    pairs: pd.DataFrame,
    run_name: str,
    station_id: str,
    out_path: Path,
    horizons: list[str],
    seed: int,
    max_points: int,
) -> None:
    import matplotlib.pyplot as plt

    d = pairs[pairs["station_id"] == station_id].copy()
    if d.empty:
        return

    # Down-select depths if there are too many
    depths = sorted(pd.unique(pd.to_numeric(
        d["depth_cm"], errors="coerce").dropna()))
    if len(depths) > 8:
        idx = np.linspace(0, len(depths) - 1, 8).round().astype(int)
        depths = [depths[i] for i in idx]
        d = d[d["depth_cm"].isin(depths)]

    fig, axes = plt.subplots(1, len(horizons), figsize=(
        14, 4.5), constrained_layout=True, sharex=True, sharey=True)
    if len(horizons) == 1:
        axes = [axes]

    all_pred = pd.to_numeric(d["pred"], errors="coerce")
    all_obs = pd.to_numeric(d["obs"], errors="coerce")
    vmin = float(np.nanmin(np.r_[all_pred.values, all_obs.values]))
    vmax = float(np.nanmax(np.r_[all_pred.values, all_obs.values]))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 0.6

    for ax, hz in zip(axes, horizons):
        hz_df = d[d["horizon_name"] == hz].copy()
        hz_df = hz_df.dropna(subset=["pred", "obs", "depth_cm"])
        hz_df = _subsample(hz_df, n=max_points, seed=seed)

        # Scatter by depth
        for depth in depths:
            sub = hz_df[hz_df["depth_cm"] == depth]
            if sub.empty:
                continue
            ax.scatter(sub["pred"], sub["obs"], s=10,
                       alpha=0.35, label=f"{int(depth)} cm")

        # Fit line + metrics (on up to 20k points)
        fit_df = d[d["horizon_name"] == hz].dropna(subset=["pred", "obs"])
        fit_df = _subsample(fit_df, n=20000, seed=seed)
        x = fit_df["pred"].to_numpy(dtype=float)
        y = fit_df["obs"].to_numpy(dtype=float)
        x, y = _finite_xy(x, y)

        ax.plot([vmin, vmax], [vmin, vmax], linestyle="--",
                color="0.4", linewidth=1.25, label="ideal")

        if len(x) >= 3:
            a, b = _fit_line(x, y)
            yhat = a * x + b
            ax.plot([vmin, vmax], [a * vmin + b, a * vmax + b],
                    linestyle="--", color="#d62728", linewidth=2.0, label="fit")

            txt = (
                f"slope={a:.3f} (ideal 1)\n"
                f"intercept={b:.3f} (ideal 0)\n"
                f"RMSE={_rmse(x, y):.3f}\n"
                f"NSE={_nse(y, x):.3f}\n"
                f"n={len(x)}"
            )
        else:
            txt = "n<3"

        ax.set_title(f"{run_name} {station_id} — {hz}")
        ax.set_xlabel("Simulated (m³/m³)")
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.grid(True, alpha=0.25)
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.8"))

    axes[0].set_ylabel("Observed (m³/m³)")
    # Keep legend light
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(6, len(labels)), frameon=False)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_multisite_scatter(
    pairs: pd.DataFrame,
    run_name: str,
    out_path: Path,
    horizons: list[str],
    seed: int,
    max_points: int,
) -> None:
    import matplotlib.pyplot as plt

    d = pairs.copy()
    d["veg_proxy"] = d["lai_mean"].where(
        np.isfinite(d["lai_mean"]), d["ndvi_mean"])
    d["veg_bin"] = _quantile_bins(d["veg_proxy"])

    fig, axes = plt.subplots(1, len(horizons), figsize=(
        14, 4.5), constrained_layout=True, sharex=True, sharey=True)
    if len(horizons) == 1:
        axes = [axes]

    all_pred = pd.to_numeric(d["pred"], errors="coerce")
    all_obs = pd.to_numeric(d["obs"], errors="coerce")
    vmin = float(np.nanmin(np.r_[all_pred.values, all_obs.values]))
    vmax = float(np.nanmax(np.r_[all_pred.values, all_obs.values]))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 0.6

    colors = {"low": "#1f77b4", "mid": "#2ca02c",
              "high": "#ff7f0e", "unknown": "0.5"}

    for ax, hz in zip(axes, horizons):
        hz_df = d[d["horizon_name"] == hz].dropna(
            subset=["pred", "obs"]).copy()
        hz_df = _subsample(hz_df, n=max_points, seed=seed)

        for veg_bin, sub in hz_df.groupby("veg_bin"):
            ax.scatter(sub["pred"], sub["obs"], s=10, alpha=0.25,
                       color=colors.get(veg_bin, "0.5"), label=veg_bin)

        fit_df = d[d["horizon_name"] == hz].dropna(
            subset=["pred", "obs"]).copy()
        fit_df = _subsample(fit_df, n=20000, seed=seed)
        x = fit_df["pred"].to_numpy(dtype=float)
        y = fit_df["obs"].to_numpy(dtype=float)
        x, y = _finite_xy(x, y)

        ax.plot([vmin, vmax], [vmin, vmax], linestyle="--",
                color="0.4", linewidth=1.25, label="ideal")

        if len(x) >= 3:
            a, b = _fit_line(x, y)
            ax.plot([vmin, vmax], [a * vmin + b, a * vmax + b],
                    linestyle="--", color="#d62728", linewidth=2.0, label="fit")

            txt = (
                f"slope={a:.3f} (ideal 1)\n"
                f"intercept={b:.3f} (ideal 0)\n"
                f"RMSE={_rmse(x, y):.3f}\n"
                f"NSE={_nse(y, x):.3f}\n"

                f"n={len(x)}"
            )
        else:
            txt = "n<3"

        ax.set_title(f"{run_name} — multi-site ({hz})")
        ax.set_xlabel("Simulated (m³/m³)")
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.grid(True, alpha=0.25)
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.8"))

    axes[0].set_ylabel("Observed (m³/m³)")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        # De-dup labels
        seen = set()
        uniq = []
        for h, l in zip(handles, labels):
            if l in seen:
                continue
            seen.add(l)
            uniq.append((h, l))
        fig.legend([h for h, _ in uniq], [l for _, l in uniq],
                   loc="lower center", ncol=min(6, len(uniq)), frameon=False)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    _ensure_matplotlib()

    parser = argparse.ArgumentParser(
        description="Scatter+fit evaluation plots for ISMN validation outputs")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run input mapping name=folder (repeatable).",
    )
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Write directly into --out and remove existing scatter-fit PNGs",
    )
    parser.add_argument("--n-sites", type=int, default=6,
                        help="Number of stations to plot per run")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for sampling")
    parser.add_argument("--max-points", type=int, default=3000,
                        help="Max scatter points per subplot")
    parser.add_argument(
        "--station-list",
        nargs="*",
        help="List of specific station IDs to plot (overrides automatic selection)",
    )

    args = parser.parse_args()

    runs: list[RunInputs] = []
    for item in args.run:
        if "=" not in item:
            raise SystemExit(
                f"Invalid --run entry: {item}. Expected name=path")
        name, path = item.split("=", 1)
        runs.append(load_run(name.strip(), Path(path)))

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for p in out_dir.glob("*_scatter_fit_*.png"):
            p.unlink(missing_ok=True)

    horizons = ["24h", "72h", "168h"]

    for run in runs:
        pairs = pd.read_csv(run.pairs_csv)
        # Normalize types
        for c in ["pred", "obs", "depth_cm", "lai_mean", "ndvi_mean"]:
            if c in pairs.columns:
                pairs[c] = _to_num(pairs[c])

        pairs = pairs[pairs["horizon_name"].isin(horizons)].copy()

        # Load station summary for KGE-based selection
        station_summary = pd.read_csv(run.station_summary_csv)

        # Select stations
        stations = pick_stations(
            pairs, station_summary, n_sites=args.n_sites, seed=args.seed)

        # Per-station plots
        for station_id in stations:
            out_path = out_dir / f"{run.name}_scatter_fit_{station_id}.png"
            plot_station_scatter(
                pairs=pairs,
                run_name=run.name,
                station_id=station_id,
                out_path=out_path,
                horizons=horizons,
                seed=args.seed,
                max_points=args.max_points,
            )

        # Multi-site aggregate
        out_path = out_dir / f"{run.name}_scatter_fit_multisite.png"
        plot_multisite_scatter(
            pairs=pairs,
            run_name=run.name,
            out_path=out_path,
            horizons=horizons,
            seed=args.seed,
            max_points=max(8000, args.max_points),
        )

    print(f"Wrote scatter-fit figures to: {out_dir}")


if __name__ == "__main__":
    main()
