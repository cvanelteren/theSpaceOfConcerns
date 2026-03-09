# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
import ultraplot as uplt

from utils import (
    generate_interaction_matrix,
    get_rca,
    load_data,
    standardize_index_labels,
)

RPA_THRESHOLD = 1.0
RETENTION_WINDOW_YEARS = 1
MAX_ALLOWED_CONSEC_GAPS = 5
LOG_EPS = 1e-12
N_DISTANCE_BINS = 30

HAZARD_PANEL_PATH = Path("output/hazard_panel_at_risk.parquet")
HAZARD_META_PATH = Path("output/hazard_panel_at_risk_meta.json")

DATA_PATHS = [
    "antarctic-database-go/data/processed/document-summary.parquet",
    "antarctic-treaty-system-ATCM-papers/dataset-DATESTAMP-HASH/summary.parquet",
    "Parsayarya-Scraping-ATCM-d1329da/ATCMDataset.csv",
    "document-summary.csv",
]

OUT_PNG = Path("figures/fig03_local_portfolio_movement.png")
OUT_PDF = Path("figures/fig03_local_portfolio_movement.pdf")
SLIDES_DIR = Path("slides")
OUT_ADOPTION_PNG = SLIDES_DIR / "fig03_local_portfolio_movement_adoption_panel.png"
OUT_ADOPTION_PDF = SLIDES_DIR / "fig03_local_portfolio_movement_adoption_panel.pdf"
OUT_RETENTION_PNG = SLIDES_DIR / "fig03_local_portfolio_movement_retention_panel.png"
OUT_RETENTION_PDF = SLIDES_DIR / "fig03_local_portfolio_movement_retention_panel.pdf"
OUT_TRANSITION_PNG = SLIDES_DIR / "fig03_local_portfolio_movement_transition_panel.png"
OUT_TRANSITION_PDF = SLIDES_DIR / "fig03_local_portfolio_movement_transition_panel.pdf"

REGIME_SUMMARY_PATHS = [
    Path("output/fig45_regime_transition_summary_window5.csv"),
    Path("output/fig45_regime_transition_summary_window1.csv"),
]
REGIME_MATRIX_PATHS = [
    Path("output/fig45_regime_transition_matrix_row_normalized_window5.csv"),
    Path("output/fig45_regime_transition_matrix_row_normalized_window1.csv"),
]


def _load_json_or_empty(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}


def _load_data_with_fallback(paths: list[str]):
    last_err = None
    for p in paths:
        try:
            return load_data(p)
        except Exception as exc:
            last_err = exc
    raise RuntimeError("Failed to load ATS data from fallback paths") from last_err


def _sanitize_years(df_in: pd.DataFrame, year_col: str) -> pd.DataFrame:
    out = df_in.copy()
    out[year_col] = pd.to_numeric(out[year_col], errors="coerce")
    out = out.dropna(subset=[year_col]).copy()
    out[year_col] = out[year_col].astype(int)
    return out


def _build_periods(year_min: int, year_max: int, window: int):
    return [(y - window + 1, y) for y in range(year_min + window - 1, year_max + 1)]


def _build_window_interaction(
    submitted_df: pd.DataFrame,
    year_col: str,
    year_start: int,
    year_end: int,
    all_members_raw: set[str],
    all_topics_raw: set[str],
    topics_order: list[str],
    members_order: list[str],
) -> pd.DataFrame:
    window_df = submitted_df[
        (submitted_df[year_col] >= int(year_start))
        & (submitted_df[year_col] <= int(year_end))
    ]
    inter = generate_interaction_matrix(window_df, all_members_raw, all_topics_raw)
    inter = standardize_index_labels(inter)
    if inter.index.has_duplicates:
        inter = inter.groupby(level=0).sum()
    return inter.reindex(index=topics_order, columns=members_order, fill_value=0)


def _km_survival(duration_windows: np.ndarray, censored: np.ndarray) -> pd.DataFrame:
    d = duration_windows.astype(int)
    c = censored.astype(bool)
    if d.size == 0:
        return pd.DataFrame({"t": [], "S": []})

    t_max = int(np.max(d))
    S = 1.0
    rows = [{"t": 0, "S": 1.0}]
    for t in range(1, t_max + 1):
        at_risk = d >= t
        n_at_risk = int(at_risk.sum())
        if n_at_risk == 0:
            break
        events = int(((d == t) & (~c)).sum())
        if events > 0:
            S *= 1.0 - events / n_at_risk
        rows.append({"t": t, "S": S})
    return pd.DataFrame(rows)


def _compute_retention_curves() -> dict[int, dict]:
    counts_df, submitted_df, members_raw, topics_raw = _load_data_with_fallback(
        DATA_PATHS
    )
    year_col = "meeting_year" if "meeting_year" in submitted_df.columns else "year"
    if year_col not in submitted_df.columns:
        raise KeyError("No meeting_year or year column found in source data.")
    submitted_df = _sanitize_years(submitted_df, year_col)

    topics = counts_df.index.tolist()
    members = counts_df.columns.tolist()
    all_members_raw = set(members_raw)
    all_topics_raw = set(topics_raw)

    year_min = int(submitted_df[year_col].min())
    year_max = int(submitted_df[year_col].max())
    periods = _build_periods(year_min, year_max, RETENTION_WINDOW_YEARS)

    rcas = []
    entry_active = []
    for start, end in periods:
        inter = _build_window_interaction(
            submitted_df=submitted_df,
            year_col=year_col,
            year_start=int(start),
            year_end=int(end),
            all_members_raw=all_members_raw,
            all_topics_raw=all_topics_raw,
            topics_order=topics,
            members_order=members,
        )
        rca = get_rca(inter).reindex(index=topics, columns=members, fill_value=0.0)
        active = (rca > RPA_THRESHOLD).reindex(
            index=topics, columns=members, fill_value=False
        )
        rcas.append(rca)
        entry_active.append(active)

    T = len(periods)
    colors = uplt.Colormap("538")(np.linspace(0.15, 0.9, MAX_ALLOWED_CONSEC_GAPS))
    curves: dict[int, dict] = {}

    for allowed_gaps, color in zip(range(1, MAX_ALLOWED_CONSEC_GAPS + 1), colors):
        durations: list[int] = []
        censored: list[bool] = []
        for member in members:
            A_entry = np.stack(
                [entry_active[t][member].to_numpy(bool) for t in range(T)], axis=0
            )
            prev = np.vstack(
                [np.zeros((1, A_entry.shape[1]), dtype=bool), A_entry[:-1]]
            )
            entries = A_entry & (~prev)
            entry_ts, topic_js = np.where(entries)
            for t0, j in zip(entry_ts, topic_js):
                consecutive_gaps = 0
                t = int(t0)
                while t < T:
                    on_now = bool(rcas[t].iloc[j][member] > RPA_THRESHOLD)
                    if on_now:
                        consecutive_gaps = 0
                        t += 1
                        continue
                    consecutive_gaps += 1
                    if consecutive_gaps <= allowed_gaps:
                        t += 1
                        continue
                    break
                durations.append(int(t - t0))
                censored.append(bool(t == T))

        km = _km_survival(
            duration_windows=np.asarray(durations, dtype=int),
            censored=np.asarray(censored, dtype=bool),
        )
        km["t_years"] = km["t"]
        curves[allowed_gaps] = {
            "curve": km,
            "label": f"{allowed_gaps}-gap tolerance",
            "color": color,
        }

    return curves


def _load_regime_transition_summary(paths: list[Path]) -> dict:
    for path in paths:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        row["_source_path"] = str(path)
        return row
    return {}


def _load_regime_transition_matrix(paths: list[Path]) -> tuple[pd.DataFrame, str]:
    for path in paths:
        if not path.exists():
            continue
        try:
            mat = pd.read_csv(path, index_col=0)
        except Exception:
            continue
        if mat.empty:
            continue
        mat.index = pd.to_numeric(mat.index, errors="coerce")
        mat.columns = pd.to_numeric(mat.columns, errors="coerce")
        mat = mat.reindex(index=[1, 2, 3], columns=[1, 2, 3])
        return mat, str(path)
    return pd.DataFrame(), ""


def _compute_log_distance(
    raw_distance: np.ndarray, distance_definition: str
) -> np.ndarray:
    distance_definition = str(distance_definition).strip().lower()
    if distance_definition == "shortest_path_neg_log_proximity":
        return raw_distance
    if np.nanmin(raw_distance) >= -1e-9 and np.nanmax(raw_distance) <= 1.0 + 1e-9:
        phi = np.clip(1.0 - raw_distance, LOG_EPS, 1.0)
        return -np.log(phi)
    return raw_distance


def _binned_adoption_curve(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    max_log_distance = float(np.nanmax(df["log_distance"].to_numpy(dtype=float)))
    if not np.isfinite(max_log_distance) or max_log_distance <= 0:
        max_log_distance = 1.0
    bins = np.linspace(0.0, max_log_distance, N_DISTANCE_BINS)
    binned = df.copy()
    binned["dist_bin"] = pd.cut(binned["log_distance"], bins=bins, include_lowest=True)

    g = binned.groupby("dist_bin", observed=True)["adopted"]
    agg = g.agg(n="size", k="sum").reset_index()
    agg["x"] = agg["dist_bin"].apply(lambda iv: float(iv.mid)).astype(float)

    z = 1.959963984540054
    n = agg["n"].to_numpy(dtype=float)
    k = agg["k"].to_numpy(dtype=float)
    p = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    den = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / den
    half = (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / den

    agg["p"] = p
    agg["ci_low"] = np.clip(center - half, 0, 1)
    agg["ci_high"] = np.clip(center + half, 0, 1)
    event_rate = float(k.sum() / np.clip(n.sum(), 1.0, None))
    return agg, event_rate


def _build_regime_summary_lines(
    hazard_meta: dict, regime_transition_meta: dict
) -> list[str]:
    topic_persistence_rate = float(hazard_meta.get("persistence_rate", np.nan))
    window_years = regime_transition_meta.get("window_years", np.nan)
    same_region_rate = float(regime_transition_meta.get("same_region_rate", np.nan))
    adjacent_or_same_rate = float(
        regime_transition_meta.get("adjacent_or_same_rate", np.nan)
    )
    far_jump_rate = float(regime_transition_meta.get("far_jump_rate", np.nan))

    lines: list[str] = []
    if np.isfinite(topic_persistence_rate):
        lines.append(f"Topic persistence (1y): {topic_persistence_rate:.1%}")
    if np.isfinite(same_region_rate):
        regime_label = (
            f"{int(window_years)}y" if np.isfinite(window_years) else "windowed"
        )
        lines.append(f"Same regime ({regime_label}): {same_region_rate:.1%}")
    if np.isfinite(adjacent_or_same_rate):
        lines.append(f"Same/adjacent regime: {adjacent_or_same_rate:.1%}")
    if np.isfinite(far_jump_rate):
        lines.append(f"Far 1<->3 jumps: {far_jump_rate:.1%}")
    return lines


def _plot_adoption_panel(
    ax,
    agg: pd.DataFrame,
    log_distance: np.ndarray,
    event_rate: float,
    *,
    xlabel: str = "Min -log(proximity) to prior portfolio (t-1)",
    show_legend: bool = True,
    show_title: bool = True,
):
    ax.plot(agg["x"], agg["p"], marker="o", lw=1.8, ms=4, label="Binned mean")
    ax.fill_between(
        agg["x"], agg["ci_low"], agg["ci_high"], alpha=0.2, label="Wilson 95% CI"
    )
    med = float(np.nanmedian(log_distance))
    ax.axvline(med, color="black", ls="--", lw=1.2, alpha=0.8)
    ax.text(
        med,
        0.98,
        f" median={med:.2f}",
        transform=ax.get_xaxis_transform(),
        va="top",
        ha="left",
        fontsize=9,
    )
    if show_legend:
        ax.legend(frameon=False, loc="cr", ncols=1)
    format_kwargs = {
        "xlabel": xlabel,
        "ylabel": "P(adopt at t | at risk at t-1)",
    }
    if show_title:
        format_kwargs["title"] = f"Adoption vs log-distance (event rate={event_rate:.2%})"
    ax.format(**format_kwargs)


def _plot_retention_panel(
    ax, retention_curves: dict[int, dict], regime_lines: list[str]
):
    for allowed_gaps in sorted(retention_curves):
        item = retention_curves[allowed_gaps]
        km_df = item["curve"]
        ax.step(
            km_df["t_years"].to_numpy(dtype=float),
            km_df["S"].to_numpy(dtype=float),
            where="post",
            color=item["color"],
            lw=1.6,
            alpha=0.95,
            label=item["label"],
        )
    ax.format(
        xlabel="Time since adoption (years)",
        ylabel="P(topic remains active ≥ t years after adoption)",
        ylim=(0.0, 1.02),
        title="Retention sensitivity with regime-stability summary",
    )
    ax.legend(
        title="Allowed consecutive gaps",
        loc="cr",
        frameon=False,
        fontsize=7,
        titlefontsize=8,
        ncols=1,
    )
    if regime_lines:
        ax.text(
            0.97,
            0.97,
            "\n".join(regime_lines),
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=8,
            bbox={
                "facecolor": "white",
                "edgecolor": "0.7",
                "alpha": 0.9,
                "boxstyle": "round,pad=0.25",
            },
        )


def _plot_transition_matrix_panel(
    ax,
    regime_transition_matrix: pd.DataFrame,
    window_years,
    regime_matrix_source: str,
    *,
    show_title: bool = True,
    xlabel: str = "To regime",
    ylabel: str = "From regime",
    xlabel_on_top: bool = True,
):
    if regime_transition_matrix.empty:
        ax.text(
            0.5,
            0.5,
            "No regime transition matrix available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )
        format_kwargs = {"xticks": [], "yticks": []}
        if show_title:
            format_kwargs["title"] = "Regime transition matrix"
        ax.format(**format_kwargs)
        return

    mat = regime_transition_matrix.to_numpy(dtype=float)
    img = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="fire")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = float(mat[i, j]) if np.isfinite(mat[i, j]) else np.nan
            txt = "NA" if not np.isfinite(v) else f"{v:.1%}"
            color = "white" if np.isfinite(v) and v >= 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    matrix_label = (
        f"{int(window_years)}y"
        if np.isfinite(window_years)
        else Path(regime_matrix_source).stem.replace("_", " ")
    )
    format_kwargs = {
        "xlabel": xlabel,
        "ylabel": ylabel,
        "xticks": [0, 1, 2],
        "yticks": [0, 1, 2],
        "xticklabels": ["R1", "R2", "R3"],
        "yticklabels": ["R1", "R2", "R3"],
    }
    if show_title:
        format_kwargs["title"] = f"Regime transition matrix ({matrix_label})"
    ax.format(**format_kwargs)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    if xlabel_on_top:
        ax.xaxis.set_label_position("top")
    else:
        ax.xaxis.set_label_position("bottom")
    cbar = ax.colorbar(img)
    cbar.set_label("Transition probability")


def _save_panel(fig, out_png: Path, out_pdf: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_png,
        dpi=240,
        bbox_inches="tight",
        pad_inches=0.14,
        transparent=True,
    )
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.14)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


def main():
    df = pd.read_parquet(HAZARD_PANEL_PATH)
    hazard_meta = _load_json_or_empty(HAZARD_META_PATH)
    retention_curves = _compute_retention_curves()
    regime_transition_meta = _load_regime_transition_summary(REGIME_SUMMARY_PATHS)
    regime_transition_matrix, regime_matrix_source = _load_regime_transition_matrix(
        REGIME_MATRIX_PATHS
    )

    raw_distance = df["distance"].to_numpy(dtype=float)
    distance_definition = hazard_meta.get("distance_definition", "")
    df = df.copy()
    df["log_distance"] = _compute_log_distance(raw_distance, distance_definition)
    agg, event_rate = _binned_adoption_curve(df)
    regime_lines = _build_regime_summary_lines(hazard_meta, regime_transition_meta)
    window_years = regime_transition_meta.get("window_years", np.nan)

    fig, axs = uplt.subplots(ncols=3, share=0)
    axs.format(abc="[A]", abcloc="ul")
    ax, ax2, ax3 = axs

    _plot_adoption_panel(
        ax=ax,
        agg=agg,
        log_distance=df["log_distance"].to_numpy(dtype=float),
        event_rate=event_rate,
    )
    _plot_retention_panel(
        ax=ax2, retention_curves=retention_curves, regime_lines=regime_lines
    )
    _plot_transition_matrix_panel(
        ax=ax3,
        regime_transition_matrix=regime_transition_matrix,
        window_years=window_years,
        regime_matrix_source=regime_matrix_source,
    )

    _save_panel(fig, OUT_PNG, OUT_PDF)

    # Standalone panels for slide use.
    fig_a, ax_a = uplt.subplots()
    _plot_adoption_panel(
        ax=ax_a,
        agg=agg,
        log_distance=df["log_distance"].to_numpy(dtype=float),
        event_rate=event_rate,
        xlabel="Distance in topic space",
        show_legend=False,
        show_title=False,
    )
    _save_panel(fig_a, OUT_ADOPTION_PNG, OUT_ADOPTION_PDF)

    fig_r, ax_r = uplt.subplots()
    _plot_retention_panel(
        ax=ax_r, retention_curves=retention_curves, regime_lines=regime_lines
    )
    _save_panel(fig_r, OUT_RETENTION_PNG, OUT_RETENTION_PDF)

    fig_t, ax_t = uplt.subplots()
    _plot_transition_matrix_panel(
        ax=ax_t,
        regime_transition_matrix=regime_transition_matrix,
        window_years=window_years,
        regime_matrix_source=regime_matrix_source,
        show_title=False,
        xlabel="to regime",
        ylabel="from regime",
        xlabel_on_top=False,
    )
    _save_panel(fig_t, OUT_TRANSITION_PNG, OUT_TRANSITION_PDF)

    uplt.show(block=1)


if __name__ == "__main__":
    main()

# %%
