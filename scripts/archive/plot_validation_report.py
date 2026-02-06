"""Create report-ready plots from ISMN validation output folders.

Usage:
  .venv/bin/python scripts/plot_validation_report.py \
    --run f2pyw=results/ismn_validation_f2pyw \
    --run gdf6e=results/ismn_validation_gdf6e \
    --out results/validation_figures

The script reads the *latest* CSVs in each run directory and writes PNGs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

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


def _pick_first(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    raise KeyError(
        f"None of columns found: {list(candidates)}. Available: {list(df.columns)}")


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


@dataclass(frozen=True)
class RunInputs:
    name: str
    run_dir: Path
    station_summary: Path
    depth_summary: Path
    horizon_summary: Path
    seasonal_summary: Path
    results: Path


def load_run(name: str, run_dir: Path) -> RunInputs:
    return RunInputs(
        name=name,
        run_dir=run_dir,
        station_summary=_pick_csv(
            run_dir, "station_summary.csv", "station_summary_*.csv"),
        depth_summary=_pick_csv(
            run_dir, "depth_summary.csv", "depth_summary_*.csv"),
        horizon_summary=_pick_csv(
            run_dir, "horizon_summary.csv", "horizon_summary_*.csv"),
        seasonal_summary=_pick_csv(
            run_dir, "seasonal_summary.csv", "seasonal_summary_*.csv"),
        results=_pick_csv(run_dir, "ismn_validation_results.csv",
                          "ismn_validation_results_*.csv"),
    )


def _ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required for plotting. Install with: .venv/bin/python -m pip install matplotlib\n"
            f"Original error: {e}"
        )


def plot_depth_skill(depth_df: pd.DataFrame, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    depth_col = _pick_first(depth_df, ["depth_cm", "depth", "depth (cm)"])
    kge_col = next((c for c in depth_df.columns if c.lower().startswith(
        "kge") and c.lower().endswith("mean")), None)
    if kge_col is None:
        kge_col = next(
            (c for c in depth_df.columns if c.lower().startswith("kge")), None)
    rmse_col = next((c for c in depth_df.columns if "rmse" in c.lower()
                    and c.lower().endswith("mean")), None)
    if rmse_col is None:
        rmse_col = next(
            (c for c in depth_df.columns if "rmse" in c.lower()), None)

    d = depth_df.copy()
    d["_depth"] = _to_num(d[depth_col])
    if kge_col:
        d["_kge"] = _to_num(d[kge_col])
    if rmse_col:
        d["_rmse"] = _to_num(d[rmse_col])
    d = d.sort_values("_depth")

    fig, ax1 = plt.subplots(figsize=(8, 4.5), constrained_layout=True)

    if kge_col:
        ax1.plot(d["_depth"], d["_kge"], marker="o", linewidth=2)
        ax1.axhline(0.0, color="black", linewidth=1, alpha=0.4)
        ax1.set_ylabel("KGE")
        ax1.set_xlabel("Depth (cm)")
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)

        # Optional RMSE on secondary axis if available
        if rmse_col:
            ax2 = ax1.twinx()
            ax2.plot(d["_depth"], d["_rmse"], marker="s",
                     linestyle="--", linewidth=1.5, color="#d62728")
            ax2.set_ylabel("RMSE")

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_horizon_skill(h_df: pd.DataFrame, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    horizon_col = _pick_first(
        h_df, ["horizon_hours", "horizon", "forecast_horizon_hours"])
    kge_col = next((c for c in h_df.columns if c.lower().startswith(
        "kge") and c.lower().endswith("mean")), None)
    if kge_col is None:
        kge_col = next(
            (c for c in h_df.columns if c.lower().startswith("kge")), None)
    rmse_col = next((c for c in h_df.columns if "rmse" in c.lower()
                    and c.lower().endswith("mean")), None)
    if rmse_col is None:
        rmse_col = next((c for c in h_df.columns if "rmse" in c.lower()), None)

    d = h_df.copy()
    d["_h"] = _to_num(d[horizon_col])
    if kge_col:
        d["_kge"] = _to_num(d[kge_col])
    if rmse_col:
        d["_rmse"] = _to_num(d[rmse_col])
    d = d.sort_values("_h")

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    if kge_col:
        ax.plot(d["_h"], d["_kge"], marker="o", linewidth=2)
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.4)
        ax.set_xlabel("Forecast horizon (hours)")
        ax.set_ylabel("KGE")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    if rmse_col:
        ax2 = ax.twinx()
        ax2.plot(d["_h"], d["_rmse"], marker="s",
                 linestyle="--", linewidth=1.5, color="#d62728")
        ax2.set_ylabel("RMSE")

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_station_map(st_df: pd.DataFrame, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    lat_col = _pick_first(st_df, ["latitude", "lat"])
    lon_col = _pick_first(st_df, ["longitude", "lon", "lng"])

    # Prefer station-level mean KGE if present
    kge_col = next((c for c in st_df.columns if c.lower(
    ).startswith("kge") and "mean" in c.lower()), None)
    if kge_col is None:
        kge_col = next((c for c in st_df.columns if c.lower() == "kge"), None)

    pass_col = next(
        (c for c in st_df.columns if "passes_validation" in c.lower()), None)
    if pass_col is None:
        pass_col = next(
            (c for c in st_df.columns if "passes_any_depth" in c.lower()), None)

    d = st_df.copy()
    d["_lat"] = _to_num(d[lat_col])
    d["_lon"] = _to_num(d[lon_col])
    if kge_col:
        d["_kge"] = _to_num(d[kge_col])
    else:
        d["_kge"] = pd.NA

    passed = None
    if pass_col:
        # robust boolean parsing
        passed = d[pass_col].astype(str).str.contains(
            "true|1|✓", case=False, regex=True)

    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)

    if passed is None:
        sc = ax.scatter(d["_lon"], d["_lat"], c=d["_kge"],
                        s=40, cmap="viridis", alpha=0.85)
    else:
        # plot failed first then passed on top
        d0 = d[~passed]
        d1 = d[passed]
        sc = ax.scatter(d0["_lon"], d0["_lat"], c=d0["_kge"],
                        s=36, cmap="viridis", alpha=0.65, marker="x")
        ax.scatter(d1["_lon"], d1["_lat"], c=d1["_kge"], s=46,
                   cmap="viridis", alpha=0.95, marker="o")
        ax.legend(["fail", "pass"], loc="best")

    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("KGE")

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.2)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_obs_variance_diagnostics(results_df: pd.DataFrame, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    depth_col = _pick_first(results_df, ["depth_cm", "depth", "depth (cm)"])
    kge_col = next((c for c in results_df.columns if c.lower() == "kge"), None)
    obs_std_col = next(
        (c for c in results_df.columns if c.lower() == "obs_std"), None)
    flag_col = next((c for c in results_df.columns if c.lower()
                    == "obs_low_variance_flag"), None)

    if not (kge_col and obs_std_col):
        # Not all runs will have the QC columns; skip gracefully.
        return

    d = results_df.copy()
    d["_obs_std"] = _to_num(d[obs_std_col])
    d["_kge"] = _to_num(d[kge_col])
    d["_depth"] = _to_num(d[depth_col])

    flagged = pd.Series(False, index=d.index)
    if flag_col:
        flagged = d[flag_col].astype(str).str.contains(
            "true|1", case=False, regex=True)

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)

    ax.scatter(d.loc[~flagged, "_obs_std"],
               d.loc[~flagged, "_kge"], s=18, alpha=0.55, label="ok")
    ax.scatter(d.loc[flagged, "_obs_std"], d.loc[flagged,
               "_kge"], s=28, alpha=0.9, label="low-variance")

    ax.set_xscale("log")
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.4)
    ax.set_xlabel("Observation std (m³/m³) [log]")
    ax.set_ylabel("KGE")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_depth_horizon_heatmap(results_df: pd.DataFrame, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    depth_col = _pick_first(results_df, ["depth_cm", "depth", "depth (cm)"])

    # Expect per-depth horizon metrics columns from run_ismn_validation.py
    horizon_cols = [
        ("0h", "KGE"),
        ("24h", "KGE_24h"),
        ("72h", "KGE_72h"),
        ("168h", "KGE_168h"),
    ]
    available = [(h, c) for (h, c) in horizon_cols if c in results_df.columns]
    if len(available) < 2:
        return

    d = results_df.copy()
    d["_depth"] = _to_num(d[depth_col])
    d = d.dropna(subset=["_depth"])

    # Build a depth x horizon matrix of mean KGE
    rows = []
    for depth in sorted(d["_depth"].unique()):
        dd = d[d["_depth"] == depth]
        row = {"depth": depth}
        for horizon_label, col in available:
            row[horizon_label] = float(_to_num(dd[col]).mean())
        rows.append(row)

    mat_df = pd.DataFrame(rows).set_index("depth")
    mat = mat_df.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=-0.5, vmax=0.8)

    ax.set_title(title)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Depth (cm)")

    ax.set_xticks(range(mat_df.shape[1]))
    ax.set_xticklabels(list(mat_df.columns))
    ax.set_yticks(range(mat_df.shape[0]))
    ax.set_yticklabels([str(int(v)) for v in mat_df.index])

    # Annotate values
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center",
                        va="center", fontsize=8, color="white")

    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Mean KGE")

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def generate_figures(run: RunInputs, out_dir: Path) -> None:
    _ensure_matplotlib()

    out_dir.mkdir(parents=True, exist_ok=True)

    station_df = pd.read_csv(run.station_summary)
    depth_df = pd.read_csv(run.depth_summary)
    horizon_df = pd.read_csv(run.horizon_summary)
    results_df = pd.read_csv(run.results)

    plot_depth_skill(
        depth_df,
        title=f"{run.name}: Skill vs depth",
        out_path=out_dir / f"{run.name}_depth_skill.png",
    )
    plot_horizon_skill(
        horizon_df,
        title=f"{run.name}: Skill vs forecast horizon",
        out_path=out_dir / f"{run.name}_horizon_skill.png",
    )
    plot_station_map(
        station_df,
        title=f"{run.name}: Station performance map",
        out_path=out_dir / f"{run.name}_station_map.png",
    )
    plot_obs_variance_diagnostics(
        results_df,
        title=f"{run.name}: KGE vs observation variance",
        out_path=out_dir / f"{run.name}_kge_vs_obs_std.png",
    )

    plot_depth_horizon_heatmap(
        results_df,
        title=f"{run.name}: Depth × horizon (mean KGE)",
        out_path=out_dir / f"{run.name}_depth_horizon_kge_heatmap.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ISMN validation outputs")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec like name=results/ismn_validation_f2pyw (repeatable)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/validation_figures"),
        help="Output folder for figures",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Write figures directly into --out (no timestamp subfolder). "
            "Deletes existing *.png files in that folder first."
        ),
    )

    args = parser.parse_args()

    if args.overwrite:
        out_root = args.out
        out_root.mkdir(parents=True, exist_ok=True)
        for old_png in out_root.glob("*.png"):
            old_png.unlink()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = args.out / timestamp

    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"Invalid --run {spec!r}; expected name=path")
        name, run_dir = spec.split("=", 1)
        run = load_run(name, Path(run_dir))
        generate_figures(run, out_root)

    print(f"Wrote figures to: {out_root}")


if __name__ == "__main__":
    main()
