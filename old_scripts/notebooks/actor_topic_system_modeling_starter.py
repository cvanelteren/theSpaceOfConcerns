# %%
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Actor-Topic System Modeling Starter
#
# This notebook is a modeling note in Jupytext form. It is meant to bridge the
# current empirical analysis to a more explicit theory of how interests are
# organized in the ATS.
#
# The core question is not:
#
# - "What is the probability of the exact observed matrix?"
#
# but rather:
#
# - "Given `N` topics, `M` actors, and actor-specific budgets `K_it`, how unusual
#   is the observed macrostructure relative to the set of feasible actor-topic
#   systems?"
#
# That shift matters because the exact observed microstate is usually the wrong
# object. The more useful question is whether the ATS exhibits a rare
# macrostructure:
#
# - a shared issue arena
# - differentiated actor positions
# - local portfolio growth in topic space
# - persistent regime structure
#
# This notebook captures that reasoning and provides a practical scaffold for
# building null models and generative simulations.

# %% [markdown]
# ## Conceptual Setup
#
# Let:
#
# - `i = 1, ..., M` index actors
# - `j = 1, ..., N` index topics
# - `t` index years or rolling windows
#
# Define the raw actor-topic allocation matrix:
#
# - `X_t in R_+^(M x N)`
# - `x_ijt >= 0` is the attention actor `i` allocates to topic `j` at time `t`
#
# With actor budget constraint:
#
# - `sum_j x_ijt = K_it`
#
# A normalized portfolio is:
#
# - `s_ijt = x_ijt / K_it`
#
# The empirical object in the paper is often a transformed version of this raw
# allocation:
#
# - RCA / RPA
# - active set (`RCA > 1`)
# - concern-space proximity
# - hazard of entry into new topics
#
# For theory, the clean base object is still `X_t`.

# %%
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import (
    compute_product_space,
    generate_interaction_matrix,
    get_rca,
    load_data,
    standardize_index_labels,
)

# %% [markdown]
# ## Static Versus Dynamic Questions
#
# There are two related but distinct theoretical questions.
#
# Static:
#
# - Why does the system exhibit this degree of overlap, concentration, and
#   differentiation at one point in time?
#
# Dynamic:
#
# - Why do actors grow their portfolios locally from existing positions in the
#   concern space rather than spreading uniformly?
#
# The current paper already addresses the dynamic side empirically via the hazard
# analysis. A practical theory program should still begin with the static
# constrained ensemble, then add dynamic growth.

# %% [markdown]
# ## Feasible Systems
#
# There are two natural formalizations of the feasible set.
#
# Binary support system:
#
# - `S_ijt in {0, 1}`
# - `sum_j S_ijt = k_it`
#
# This is useful if the main object is whether an actor is active in a topic.
#
# Weighted count system:
#
# - `x_ijt in Z_+`
# - `sum_j x_ijt = K_it`
#
# This is useful if the main object is raw allocation across topics.
#
# In either case the set of feasible systems is huge. The theoretical task is
# therefore not to enumerate exact systems, but to define a probability measure
# over the feasible set and ask whether the ATS looks typical or atypical under
# that ensemble.

# %% [markdown]
# ## Null Models To Compare
#
# A clean sequence of nulls is:
#
# 1. Budget-only null
#    - preserve actor budgets `K_it`
#    - topics are otherwise exchangeable
#
# 2. Budget + topic salience null
#    - preserve actor budgets
#    - allow some topics to be intrinsically more attractive
#
# 3. Budget + topic salience + local growth null
#    - preserve actor budgets
#    - allow topic popularity / salience
#    - actors expand into nearby topics in the concern space
#
# 4. Add actor heterogeneity
#    - some actors are drawn to some regions more than others
#
# The theoretical question becomes:
#
# - Which is the first model that reproduces the observed macrostructure?

# %% [markdown]
# ## Eligibility, Participation, And Allocation
#
# One correction is essential: not all actors are active at all times.
#
# A cleaner full model would distinguish:
#
# - `E_it`: actor is eligible / exists in the system at time `t`
# - `Z_it`: actor is active at time `t`
# - `K_it`: actor budget conditional on being active
# - `X_it`: actor-topic allocation conditional on being active
#
# For the simplest implementation, we do **not** model participation yet.
# Instead we condition on the observed active set:
#
# - `M_t = {i : K_it > 0}`
#
# and only simulate allocations for actors who are actually active in each
# observed period. This means the first forward simulation is only about the
# **intensive margin** of attention allocation, not the extensive margin of who
# participates.
#
# We also use the simplest reset rule after inactivity:
#
# - if an actor is not active in the previous period, their simulated portfolio
#   memory resets to zero before re-entry.
#
# This is deliberately minimal and meant only to build intuition.

# %% [markdown]
# ## Practical Empirical Program
#
# In practice, we should not "just fit a model."
#
# The more defensible workflow is:
#
# 1. estimate a local decision rule for topic entry / allocation
# 2. simulate synthetic ATS histories from that rule
# 3. compare observed and simulated systems on macro statistics
# 4. add complexity only when the simpler model fails
#
# That means we need both:
#
# - micro fit: does the model explain local choices?
# - macro fit: does it reproduce regime structure, overlap, and local growth?

# %% [markdown]
# ## Notebook Goals
#
# This notebook sets up:
#
# - a first actor-topic-time panel on raw counts
# - a fixed topic-space matrix `Phi`
# - simple null samplers
# - a minimal dynamic allocation rule
# - a checklist for simulation-based validation

# %% [markdown]
# ## Configuration


# %%
@dataclass(frozen=True)
class ModelingConfig:
    data_paths: tuple[Path, ...] = (
        Path("antarctic-database-go/data/processed/document-summary.parquet"),
        Path(
            "antarctic-treaty-system-ATCM-papers/dataset-DATESTAMP-HASH/summary.parquet"
        ),
        Path("Parsayarya-Scraping-ATCM-d1329da/ATCMDataset.csv"),
        Path("document-summary.csv"),
    )
    time_unit: str = os.getenv("ACTOR_TOPIC_MODEL_TIME_UNIT", "year").strip().lower()
    # Rolling window size, interpreted in `time_unit`.
    window_size: int = int(
        os.getenv(
            "ACTOR_TOPIC_MODEL_WINDOW_SIZE",
            os.getenv("ACTOR_TOPIC_MODEL_WINDOW_YEARS", "1"),
        )
    )
    rca_threshold: float = 1.0
    random_seed: int = 7
    beta_demo: float = 2.0
    rho_demo: float = 0.0
    gamma_demo: float = 0.0
    out_dir: Path = Path("output")
    out_summary_json: Path = Path("output/actor_topic_modeling_starter_summary.json")
    out_null_draw_csv: Path = Path("output/actor_topic_modeling_starter_null_draw_latest.csv")
    out_history_summary_csv: Path = Path(
        "output/actor_topic_modeling_starter_history_summary.csv"
    )
    out_sim_history_csv: Path = Path(
        "output/actor_topic_modeling_starter_sim_history.csv"
    )
    out_param_json: Path = Path(
        "output/actor_topic_modeling_starter_param_estimates.json"
    )
    out_support_param_json: Path = Path(
        "output/actor_topic_modeling_starter_support_param_estimates.json"
    )
    out_entry_param_json: Path = Path(
        "output/actor_topic_modeling_starter_entry_param_estimates.json"
    )
    out_retention_param_json: Path = Path(
        "output/actor_topic_modeling_starter_retention_param_estimates.json"
    )
    out_two_stage_history_summary_csv: Path = Path(
        "output/actor_topic_modeling_starter_two_stage_history_summary.csv"
    )
    out_two_stage_sim_history_csv: Path = Path(
        "output/actor_topic_modeling_starter_two_stage_sim_history.csv"
    )
    out_split_history_summary_csv: Path = Path(
        "output/actor_topic_modeling_starter_split_history_summary.csv"
    )
    out_split_sim_history_csv: Path = Path(
        "output/actor_topic_modeling_starter_split_sim_history.csv"
    )
    out_centroid_shift_csv: Path = Path(
        "output/actor_topic_modeling_starter_centroid_shifts.csv"
    )
    out_entry_phi_csv: Path = Path(
        "output/actor_topic_modeling_starter_entry_phi_proximity.csv"
    )
    out_concentration_latest_csv: Path = Path(
        "output/actor_topic_modeling_starter_concentration_latest.csv"
    )
    out_subsample_validation_csv: Path = Path(
        "output/actor_topic_modeling_starter_subsample_validation.csv"
    )
    out_subsample_validation_summary_csv: Path = Path(
        "output/actor_topic_modeling_starter_subsample_validation_summary.csv"
    )
    process_uncertainty_reps: int = 64
    out_process_uncertainty_history_csv: Path = Path(
        "output/actor_topic_modeling_starter_process_uncertainty_history.csv"
    )
    out_process_uncertainty_entry_csv: Path = Path(
        "output/actor_topic_modeling_starter_process_uncertainty_entry.csv"
    )
    out_split_param_uncertainty_csv: Path = Path(
        "output/actor_topic_modeling_starter_split_param_uncertainty.csv"
    )


CFG = ModelingConfig()

# %% [markdown]
# ## Helpers: Data Loading And Panel Construction


# %%
def load_data_with_fallback(paths: Iterable[Path]):
    """Load ATS data from the first working path."""
    last_err: Optional[Exception] = None
    for path in paths:
        if not path.exists():
            continue
        try:
            return load_data(str(path))
        except Exception as exc:  # pragma: no cover
            last_err = exc
    if last_err is not None:
        raise RuntimeError("Failed to load ATS data from fallback paths.") from last_err
    raise FileNotFoundError("No known ATS data path exists. Update CFG.data_paths.")


def detect_year_col(df: pd.DataFrame) -> str:
    for candidate in ("meeting year", "year"):
        if candidate in df.columns:
            return candidate
    raise KeyError("No meeting year or year column found in source data.")


def detect_meeting_col(df: pd.DataFrame) -> str | None:
    for candidate in ("meeting number", "meeting_number"):
        if candidate in df.columns:
            return candidate
    return None


def sanitize_years(df: pd.DataFrame, year_col: str) -> pd.DataFrame:
    out = df.copy()
    out[year_col] = pd.to_numeric(out[year_col], errors="coerce")
    out = out.dropna(subset=[year_col]).copy()
    out[year_col] = out[year_col].astype(int)
    return out


def sanitize_int_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=[col]).copy()
    out[col] = out[col].astype(int)
    return out


def build_periods(year_min: int, year_max: int, window: int) -> list[tuple[int, int]]:
    return [(y - window + 1, y) for y in range(year_min + window - 1, year_max + 1)]


def build_periods_for_unit(
    submitted_df: pd.DataFrame,
    *,
    year_col: str,
    meeting_col: str | None,
    time_unit: str,
    window_size: int,
) -> list[tuple[int, int]]:
    time_unit = str(time_unit).strip().lower()
    window_size = max(1, int(window_size))

    if time_unit == "year":
        year_min = int(submitted_df[year_col].min())
        year_max = int(submitted_df[year_col].max())
        return build_periods(year_min, year_max, window_size)

    if time_unit == "meeting":
        if meeting_col is None or meeting_col not in submitted_df.columns:
            raise KeyError(
                "No meeting-number column found in source data for time_unit='meeting'."
            )
        values = (
            pd.to_numeric(submitted_df[meeting_col], errors="coerce")
            .dropna()
            .astype(int)
            .sort_values()
            .unique()
            .tolist()
        )
        return [
            (int(values[idx - window_size + 1]), int(values[idx]))
            for idx in range(window_size - 1, len(values))
        ]

    raise ValueError(f"Unknown time_unit={time_unit!r}. Use meeting|year.")


def build_window_interaction(
    submitted_df: pd.DataFrame,
    time_col: str,
    window_start: int,
    window_end: int,
    all_members_raw: set[str],
    all_topics_raw: set[str],
    topics_order: list[str],
    actors_order: list[str],
) -> pd.DataFrame:
    window_df = submitted_df[
        (submitted_df[time_col] >= int(window_start))
        & (submitted_df[time_col] <= int(window_end))
    ]
    interaction = generate_interaction_matrix(
        window_df, all_members_raw, all_topics_raw
    )
    interaction = standardize_index_labels(interaction)
    if interaction.index.has_duplicates:
        interaction = interaction.groupby(level=0).sum()
    return interaction.reindex(index=topics_order, columns=actors_order, fill_value=0)


def build_actor_topic_window_panel(
    submitted_df: pd.DataFrame,
    time_col: str,
    periods: list[tuple[int, int]],
    all_members_raw: set[str],
    all_topics_raw: set[str],
    topics_order: list[str],
    actors_order: list[str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for start, end in periods:
        interaction = build_window_interaction(
            submitted_df=submitted_df,
            time_col=time_col,
            window_start=int(start),
            window_end=int(end),
            all_members_raw=all_members_raw,
            all_topics_raw=all_topics_raw,
            topics_order=topics_order,
            actors_order=actors_order,
        )
        long_df = interaction.stack().rename("x_ijt").reset_index()
        long_df.columns = ["topic", "actor", "x_ijt"]
        long_df["window_start"] = int(start)
        long_df["window_end"] = int(end)
        rows.append(long_df)

    panel = pd.concat(rows, ignore_index=True)
    panel["K_it"] = panel.groupby(["window_end", "actor"])["x_ijt"].transform("sum")
    panel["s_ijt"] = np.divide(
        panel["x_ijt"].to_numpy(dtype=float),
        np.clip(panel["K_it"].to_numpy(dtype=float), 1.0, None),
    )
    panel["active_raw"] = panel["x_ijt"] > 0
    return panel


def build_rca_active_panel(
    submitted_df: pd.DataFrame,
    time_col: str,
    periods: list[tuple[int, int]],
    all_members_raw: set[str],
    all_topics_raw: set[str],
    topics_order: list[str],
    actors_order: list[str],
    rca_threshold: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for start, end in periods:
        interaction = build_window_interaction(
            submitted_df=submitted_df,
            time_col=time_col,
            window_start=int(start),
            window_end=int(end),
            all_members_raw=all_members_raw,
            all_topics_raw=all_topics_raw,
            topics_order=topics_order,
            actors_order=actors_order,
        )
        rca = get_rca(interaction).reindex(
            index=topics_order, columns=actors_order, fill_value=0.0
        )
        long_df = rca.stack().rename("rca").reset_index()
        long_df.columns = ["topic", "actor", "rca"]
        long_df["window_start"] = int(start)
        long_df["window_end"] = int(end)
        long_df["active_rca"] = long_df["rca"] > float(rca_threshold)
        rows.append(long_df)
    return pd.concat(rows, ignore_index=True)


def build_count_panel_with_active_rca(
    submitted_df: pd.DataFrame,
    time_col: str,
    periods: list[tuple[int, int]],
    all_members_raw: set[str],
    all_topics_raw: set[str],
    topics_order: list[str],
    actors_order: list[str],
    rca_threshold: float,
) -> pd.DataFrame:
    """Rebuild the raw-count panel and RCA-active panel from a submission subset."""
    count_panel = build_actor_topic_window_panel(
        submitted_df=submitted_df,
        time_col=time_col,
        periods=periods,
        all_members_raw=all_members_raw,
        all_topics_raw=all_topics_raw,
        topics_order=topics_order,
        actors_order=actors_order,
    )
    active_panel = build_rca_active_panel(
        submitted_df=submitted_df,
        time_col=time_col,
        periods=periods,
        all_members_raw=all_members_raw,
        all_topics_raw=all_topics_raw,
        topics_order=topics_order,
        actors_order=actors_order,
        rca_threshold=rca_threshold,
    )
    return count_panel.merge(
        active_panel[["topic", "actor", "window_end", "rca", "active_rca"]],
        on=["topic", "actor", "window_end"],
        how="left",
    )


# %% [markdown]
# ## Load ATS Data And Build A First Panel
#
# This gives us the basic actor-topic-window panel on raw counts. That panel is a
# clean starting point for both static nulls and dynamic simulations.


# %%
counts_df, submitted_df, members_raw, topics_raw = load_data_with_fallback(CFG.data_paths)
year_col = detect_year_col(submitted_df)
submitted_df = sanitize_years(submitted_df, year_col)
meeting_col = detect_meeting_col(submitted_df)
if meeting_col is not None:
    submitted_df = sanitize_int_column(submitted_df, meeting_col)

time_unit = str(CFG.time_unit).strip().lower()
if time_unit not in {"meeting", "year"}:
    raise ValueError(f"Unknown CFG.time_unit={CFG.time_unit!r}. Use meeting|year.")
time_col = meeting_col if time_unit == "meeting" else year_col
if time_col is None:
    raise KeyError("Meeting-based mode requested but no meeting-number column exists.")

topics_order = counts_df.index.tolist()
actors_order = counts_df.columns.tolist()
all_members_raw = set(members_raw)
all_topics_raw = set(topics_raw)

year_min = int(submitted_df[year_col].min())
year_max = int(submitted_df[year_col].max())
meeting_min = int(submitted_df[meeting_col].min()) if meeting_col is not None else np.nan
meeting_max = int(submitted_df[meeting_col].max()) if meeting_col is not None else np.nan
periods = build_periods_for_unit(
    submitted_df=submitted_df,
    year_col=year_col,
    meeting_col=meeting_col,
    time_unit=time_unit,
    window_size=CFG.window_size,
)
period_min = int(periods[0][0]) if periods else 0
period_max = int(periods[-1][1]) if periods else 0

count_panel = build_count_panel_with_active_rca(
    submitted_df=submitted_df,
    time_col=time_col,
    periods=periods,
    all_members_raw=all_members_raw,
    all_topics_raw=all_topics_raw,
    topics_order=topics_order,
    actors_order=actors_order,
    rca_threshold=CFG.rca_threshold,
)

system_summary = pd.Series(
    {
        "time_unit": time_unit,
        "time_axis_label": "Meeting" if time_unit == "meeting" else "Year",
        "window_size": int(CFG.window_size),
        "window_years": int(CFG.window_size),
        "n_topics": len(topics_order),
        "n_actors": len(actors_order),
        "year_min": year_min,
        "year_max": year_max,
        "meeting_min": meeting_min,
        "meeting_max": meeting_max,
        "period_min": period_min,
        "period_max": period_max,
        "n_periods": len(periods),
        "mean_budget_per_actor_period": count_panel.groupby(
            ["window_end", "actor"]
        )["K_it"].first().mean(),
        "mean_active_topics_per_actor_period_raw": count_panel.groupby(
            ["window_end", "actor"]
        )["active_raw"].sum().mean(),
        "mean_active_topics_per_actor_period_rca": count_panel.groupby(
            ["window_end", "actor"]
        )["active_rca"].sum().mean(),
    }
)

system_summary

# %% [markdown]
# ## Build A Fixed Topic Space
#
# A minimal dynamic model needs a common topic-space matrix `Phi`. The simplest
# starting point is the pooled concern space from the full-history RCA matrix.


# %%
overall_rca = get_rca(counts_df).reindex(
    index=topics_order, columns=actors_order, fill_value=0.0
)
phi = compute_product_space(overall_rca, threshold=CFG.rca_threshold).reindex(
    index=topics_order, columns=topics_order, fill_value=0.0
)

phi.shape

# %% [markdown]
# ## First Nulls
#
# These are not the final models. They are baseline generators for comparison.
#
# Budget-only null:
#
# - each actor has budget `K_it`
# - topics are exchangeable
#
# Budget + topic-salience null:
#
# - same budgets
# - topic weights allow some topics to be more attractive overall


# %%
def sample_support_budget_only(
    rng: np.random.Generator, n_topics: int, k_i: int
) -> np.ndarray:
    """Binary support draw with exactly `k_i` active topics."""
    k_i = int(np.clip(k_i, 0, n_topics))
    out = np.zeros(n_topics, dtype=int)
    if k_i == 0:
        return out
    choice = rng.choice(n_topics, size=k_i, replace=False)
    out[choice] = 1
    return out


def sample_count_budget_only(
    rng: np.random.Generator,
    K_i: int,
    n_topics: int,
    topic_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Count allocation with total budget `K_i`."""
    if K_i <= 0:
        return np.zeros(n_topics, dtype=int)
    if topic_weights is None:
        probs = np.full(n_topics, 1.0 / n_topics, dtype=float)
    else:
        probs = np.asarray(topic_weights, dtype=float)
        probs = probs / probs.sum()
    return rng.multinomial(int(K_i), probs)


def sample_system_count_null(
    rng: np.random.Generator,
    budgets: pd.Series,
    topics_order: list[str],
    topic_weights: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Draw a row-constrained system for one period.

    Parameters
    ----------
    budgets:
        Series indexed by actor, with budget `K_i`.
    """
    n_topics = len(topics_order)
    rows: list[pd.DataFrame] = []
    for actor, K_i in budgets.items():
        alloc = sample_count_budget_only(
            rng=rng,
            K_i=int(K_i),
            n_topics=n_topics,
            topic_weights=topic_weights,
        )
        actor_df = pd.DataFrame(
            {"topic": topics_order, "actor": actor, "x_ijt_sim": alloc}
        )
        rows.append(actor_df)
    return pd.concat(rows, ignore_index=True)


# %% [markdown]
# ## Minimal Dynamic Allocation Rule
#
# A stripped-down dynamic model close to the hazard logic is:
#
# `Pr(i allocates to j at t) proportional exp(alpha_j + rho * s_ij,t-1 + beta * Fit_ijt + gamma * popularity_j,t-1 + a_ij)`
#
# where:
#
# - `alpha_j` is baseline topic attractiveness
# - `rho` captures persistence on already-used topics
# - `Fit_ijt` is proximity of topic `j` to actor `i`'s previous portfolio
# - `gamma` captures popularity or crowding effects
# - `a_ij` is an optional actor-topic affinity term
#
# A natural local-fit term is:
#
# `Fit_ijt = sum_l s_il,t-1 * Phi_lj`
#
# This is the formal version of local portfolio growth in the concern space.


# %%
def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    logits = logits - np.nanmax(logits)
    exp_logits = np.exp(logits)
    return exp_logits / np.clip(exp_logits.sum(), 1e-12, None)


def local_fit_from_phi(shares_prev: np.ndarray, phi_matrix: np.ndarray) -> np.ndarray:
    shares_prev = np.asarray(shares_prev, dtype=float)
    phi_matrix = np.asarray(phi_matrix, dtype=float)
    return shares_prev @ phi_matrix


def dynamic_topic_probs(
    alpha: np.ndarray,
    shares_prev: np.ndarray,
    phi_matrix: np.ndarray,
    *,
    rho: float = 0.0,
    beta: float = 0.0,
    topic_popularity: Optional[np.ndarray] = None,
    gamma: float = 0.0,
    actor_affinity: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Starter dynamic rule for one actor-period.

    This is not yet estimated. It is the scaffold we can later plug estimates
    into for simulation.
    """
    logits = np.asarray(alpha, dtype=float).copy()
    logits += float(rho) * np.asarray(shares_prev, dtype=float)
    logits += float(beta) * local_fit_from_phi(shares_prev, phi_matrix)
    if topic_popularity is not None:
        logits += float(gamma) * np.asarray(topic_popularity, dtype=float)
    if actor_affinity is not None:
        logits += np.asarray(actor_affinity, dtype=float)
    return softmax(logits)


def simulate_actor_period_counts(
    rng: np.random.Generator,
    K_i: int,
    alpha: np.ndarray,
    shares_prev: np.ndarray,
    phi_matrix: np.ndarray,
    *,
    rho: float = 0.0,
    beta: float = 0.0,
    topic_popularity: Optional[np.ndarray] = None,
    gamma: float = 0.0,
    actor_affinity: Optional[np.ndarray] = None,
) -> np.ndarray:
    probs = dynamic_topic_probs(
        alpha=alpha,
        shares_prev=shares_prev,
        phi_matrix=phi_matrix,
        rho=rho,
        beta=beta,
        topic_popularity=topic_popularity,
        gamma=gamma,
        actor_affinity=actor_affinity,
    )
    return rng.multinomial(int(K_i), probs)


def estimate_topic_salience_alpha(
    count_panel: pd.DataFrame, topics_order: list[str]
) -> np.ndarray:
    """
    Estimate baseline topic attractiveness from pooled topic mass.

    This is a reduced-form empirical prior, not yet a structural estimate.
    """
    topic_mass = (
        count_panel.groupby("topic")["x_ijt"]
        .sum()
        .reindex(topics_order, fill_value=0.0)
        .to_numpy(dtype=float)
    )
    topic_probs = topic_mass / np.clip(topic_mass.sum(), 1.0, None)
    alpha = np.log(np.clip(topic_probs, 1e-12, None))
    alpha -= alpha.mean()
    return alpha


def build_budget_table(count_panel: pd.DataFrame) -> pd.DataFrame:
    """Budget table indexed by period with actor columns."""
    budget_table = count_panel.groupby(["window_end", "actor"])["K_it"].first().unstack()
    budget_table = budget_table.fillna(0).sort_index()
    budget_table = budget_table.reindex(
        columns=sorted(budget_table.columns.tolist()), fill_value=0
    )
    return budget_table


def simulate_history_conditioned_on_active_set(
    rng: np.random.Generator,
    budget_table: pd.DataFrame,
    topics_order: list[str],
    phi_matrix: np.ndarray,
    alpha: np.ndarray,
    *,
    rho: float = 0.0,
    beta: float = 0.0,
    gamma: float = 0.0,
) -> pd.DataFrame:
    """
    Simplest forward simulation:

    - preserve the observed active set and budgets period by period
    - simulate only allocations for active actors
    - reset portfolio memory after inactivity gaps
    """
    n_topics = len(topics_order)
    zero_shares = np.zeros(n_topics, dtype=float)
    prev_shares_by_actor: dict[str, np.ndarray] = {}
    prev_topic_popularity = np.zeros(n_topics, dtype=float)
    rows: list[pd.DataFrame] = []

    for window_end, budget_row in budget_table.iterrows():
        current_shares_by_actor: dict[str, np.ndarray] = {}
        active_actors = [actor for actor, K_i in budget_row.items() if float(K_i) > 0.0]

        for actor in active_actors:
            K_i = int(budget_row[actor])
            shares_prev = prev_shares_by_actor.get(actor, zero_shares)
            alloc = simulate_actor_period_counts(
                rng=rng,
                K_i=K_i,
                alpha=alpha,
                shares_prev=shares_prev,
                phi_matrix=phi_matrix,
                rho=rho,
                beta=beta,
                topic_popularity=prev_topic_popularity,
                gamma=gamma,
            )
            shares_now = alloc.astype(float) / np.clip(float(K_i), 1.0, None)
            current_shares_by_actor[actor] = shares_now
            actor_df = pd.DataFrame(
                {
                    "window_end": int(window_end),
                    "actor": actor,
                    "topic": topics_order,
                    "K_it": int(K_i),
                    "x_ijt_sim": alloc,
                    "s_ijt_sim": shares_now,
                    "active_sim": alloc > 0,
                }
            )
            rows.append(actor_df)

        if current_shares_by_actor:
            prev_shares_by_actor = current_shares_by_actor
            prev_topic_popularity = (
                np.vstack(list(current_shares_by_actor.values())) > 0
            ).mean(axis=0)
        else:
            prev_shares_by_actor = {}
            prev_topic_popularity = np.zeros(n_topics, dtype=float)

    return pd.concat(rows, ignore_index=True)


def summarize_observed_and_simulated_history(
    count_panel: pd.DataFrame, sim_history: pd.DataFrame
) -> pd.DataFrame:
    observed_actor_period = (
        count_panel.groupby(["window_end", "actor"])
        .agg(
            K_it=("K_it", "first"),
            active_topics_obs=("active_rca", "sum"),
            active_topics_raw_obs=("active_raw", "sum"),
        )
        .reset_index()
    )
    observed_actor_period = observed_actor_period[observed_actor_period["K_it"] > 0].copy()

    observed_topic_period = count_panel.groupby(["window_end", "topic"])[
        "active_rca"
    ].sum()

    observed_summary = observed_actor_period.groupby("window_end").agg(
        n_active_actors_obs=("actor", "nunique"),
        mean_budget_obs=("K_it", "mean"),
        mean_active_topics_obs=("active_topics_obs", "mean"),
        mean_active_topics_raw_obs=("active_topics_raw_obs", "mean"),
    )
    observed_summary["mean_topic_popularity_obs"] = observed_topic_period.groupby(
        "window_end"
    ).mean()

    sim_actor_period = (
        sim_history.groupby(["window_end", "actor"])
        .agg(
            K_it=("K_it", "first"),
            active_topics_sim=("active_sim", "sum"),
            herfindahl_sim=("s_ijt_sim", lambda x: float(np.square(x).sum())),
        )
        .reset_index()
    )

    sim_topic_period = sim_history.groupby(["window_end", "topic"])[
        "active_sim"
    ].sum()

    sim_summary = sim_actor_period.groupby("window_end").agg(
        n_active_actors_sim=("actor", "nunique"),
        mean_budget_sim=("K_it", "mean"),
        mean_active_topics_sim=("active_topics_sim", "mean"),
        mean_herfindahl_sim=("herfindahl_sim", "mean"),
    )
    sim_summary["mean_topic_popularity_sim"] = sim_topic_period.groupby(
        "window_end"
    ).mean()

    return observed_summary.join(sim_summary, how="outer").reset_index()


def fit_reduced_form_allocation_params(
    count_panel: pd.DataFrame,
    phi_matrix: np.ndarray,
    topics_order: list[str],
    actors_order: list[str],
) -> dict[str, float]:
    """
    Estimate reduced-form allocation parameters conditional on:

    - observed active actor set by period
    - observed actor-period budgets
    - fixed pooled topic salience alpha_j

    Parameters:
    - rho: persistence on previously held topics
    - beta: local fit / proximity expansion
    - gamma: effect of previous topic popularity
    """
    alpha = estimate_topic_salience_alpha(count_panel, topics_order)
    budget_table = build_budget_table(count_panel)
    periods = budget_table.index.to_numpy()
    period_index = {p: i for i, p in enumerate(periods)}
    actor_names = budget_table.columns.tolist()
    actor_index = {a: i for i, a in enumerate(actor_names)}
    topic_index = {t: i for i, t in enumerate(topics_order)}

    T = len(periods)
    A = len(actor_names)
    N = len(topics_order)

    X = np.zeros((T, A, N), dtype=float)
    for row in count_panel[["window_end", "actor", "topic", "x_ijt"]].itertuples(
        index=False
    ):
        ti = period_index[row.window_end]
        ai = actor_index[row.actor]
        ji = topic_index[row.topic]
        X[ti, ai, ji] = float(row.x_ijt)

    K = X.sum(axis=2)
    shares = np.divide(X, np.clip(K[..., None], 1.0, None))
    active_raw = (X > 0).astype(float)

    pop_prev_all = np.zeros_like(X)
    for t in range(1, T):
        active_mask = K[t - 1] > 0
        if active_mask.any():
            pop_prev = active_raw[t - 1, active_mask].mean(axis=0)
        else:
            pop_prev = np.zeros(N, dtype=float)
        pop_prev_all[t] = pop_prev

    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for t in range(1, T):
        for a in range(A):
            if K[t, a] <= 0:
                continue
            x = X[t, a]
            s_prev = shares[t - 1, a] if K[t - 1, a] > 0 else np.zeros(N, dtype=float)
            fit = s_prev @ phi_matrix
            pop_prev = pop_prev_all[t, a]
            rows.append((x, s_prev, fit, pop_prev))

    def neg_ll(theta: np.ndarray) -> float:
        rho, beta, gamma = theta
        total = 0.0
        for x, s_prev, fit, pop_prev in rows:
            u = alpha + rho * s_prev + beta * fit + gamma * pop_prev
            total -= float(np.dot(x, u - logsumexp(u)))
        return total

    res = minimize(neg_ll, x0=np.array([0.0, 0.0, 0.0]), method="L-BFGS-B")
    out = {
        "rho_mle": float(res.x[0]),
        "beta_mle": float(res.x[1]),
        "gamma_mle": float(res.x[2]),
        "neg_loglik_full": float(res.fun),
        "n_actor_periods_fit": int(len(rows)),
        "opt_success": bool(res.success),
    }
    try:
        hess_inv = np.asarray(res.hess_inv.todense(), dtype=float)
        se = np.sqrt(np.diag(hess_inv))
        out["rho_se"] = float(se[0])
        out["beta_se"] = float(se[1])
        out["gamma_se"] = float(se[2])
    except Exception:  # pragma: no cover
        pass
    out["neg_loglik_alpha_only"] = float(neg_ll(np.array([0.0, 0.0, 0.0])))
    out["neg_loglik_rho_only"] = float(neg_ll(np.array([out["rho_mle"], 0.0, 0.0])))
    out["neg_loglik_beta_only"] = float(neg_ll(np.array([0.0, out["beta_mle"], 0.0])))
    out["neg_loglik_rho_beta"] = float(
        neg_ll(np.array([out["rho_mle"], out["beta_mle"], 0.0]))
    )
    return out


def estimate_support_baseline_logit(
    count_panel: pd.DataFrame, topics_order: list[str]
) -> np.ndarray:
    """
    Estimate baseline support logits from pooled active-topic frequency among
    active actor-periods.
    """
    actor_period_active = (
        count_panel.groupby(["window_end", "actor"])["K_it"].first().rename("K_it")
    )
    n_active_actor_periods = int((actor_period_active > 0).sum())
    active_counts = (
        count_panel.loc[count_panel["K_it"] > 0]
        .groupby("topic")["active_rca"]
        .sum()
        .reindex(topics_order, fill_value=0.0)
        .to_numpy(dtype=float)
    )
    probs = active_counts / np.clip(float(n_active_actor_periods), 1.0, None)
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    return np.log(probs / (1.0 - probs))


def build_support_transition_arrays(
    count_panel: pd.DataFrame,
    topics_order: list[str],
    actors_order: list[str],
    phi_matrix: np.ndarray,
) -> dict[str, np.ndarray | list[str]]:
    """
    Shared state arrays for support-transition estimation.

    The arrays are indexed as `[time, actor, topic]`.
    """
    budget_table = build_budget_table(count_panel)
    periods = budget_table.index.to_numpy()
    period_index = {p: i for i, p in enumerate(periods)}
    actor_names = budget_table.columns.tolist()
    actor_index = {a: i for i, a in enumerate(actor_names)}
    topic_index = {t: i for i, t in enumerate(topics_order)}

    T = len(periods)
    A = len(actor_names)
    N = len(topics_order)

    Y = np.zeros((T, A, N), dtype=float)
    X_counts = np.zeros((T, A, N), dtype=float)
    for row in count_panel[
        ["window_end", "actor", "topic", "active_rca", "x_ijt"]
    ].itertuples(index=False):
        ti = period_index[row.window_end]
        ai = actor_index[row.actor]
        ji = topic_index[row.topic]
        Y[ti, ai, ji] = float(row.active_rca)
        X_counts[ti, ai, ji] = float(row.x_ijt)

    K = X_counts.sum(axis=2)
    shares = np.divide(X_counts, np.clip(K[..., None], 1.0, None))
    fit_prev_all = np.zeros_like(Y)
    pop_prev_all = np.zeros_like(Y)
    for t in range(1, T):
        active_mask = K[t - 1] > 0
        if active_mask.any():
            pop_prev = Y[t - 1, active_mask].mean(axis=0)
        else:
            pop_prev = np.zeros(N, dtype=float)
        pop_prev_all[t] = pop_prev
        for a in range(A):
            if K[t - 1, a] <= 0:
                continue
            prev_support = Y[t - 1, a]
            prev_support_sum = prev_support.sum()
            if prev_support_sum > 0:
                fit_prev_all[t, a] = (prev_support / prev_support_sum) @ phi_matrix

    return {
        "periods": periods,
        "actor_names": actor_names,
        "Y": Y,
        "X_counts": X_counts,
        "K": K,
        "shares": shares,
        "fit_prev_all": fit_prev_all,
        "pop_prev_all": pop_prev_all,
    }


def fit_support_selection_params(
    count_panel: pd.DataFrame,
    phi_matrix: np.ndarray,
    topics_order: list[str],
    actors_order: list[str],
) -> dict[str, float]:
    """
    Fit a pooled binary support-selection model for whether an active actor
    touches a topic at all in a given period.
    """
    alpha_support = estimate_support_baseline_logit(count_panel, topics_order)
    arrays = build_support_transition_arrays(
        count_panel=count_panel,
        topics_order=topics_order,
        actors_order=actors_order,
        phi_matrix=phi_matrix,
    )
    Y = arrays["Y"]
    K = arrays["K"]
    fit_prev_all = arrays["fit_prev_all"]
    pop_prev_all = arrays["pop_prev_all"]
    T, A, N = Y.shape

    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for t in range(1, T):
        for a in range(A):
            if K[t, a] <= 0:
                continue
            y = Y[t, a]
            prev_support = Y[t - 1, a] if K[t - 1, a] > 0 else np.zeros(N, dtype=float)
            fit = fit_prev_all[t, a]
            pop_prev = pop_prev_all[t, a]
            rows.append((y, prev_support, fit, pop_prev))

    def neg_ll(theta: np.ndarray) -> float:
        rho_s, beta_s, gamma_s = theta
        total = 0.0
        for y, prev_support, fit, pop_prev in rows:
            u = alpha_support + rho_s * prev_support + beta_s * fit + gamma_s * pop_prev
            p = np.clip(expit(u), 1e-9, 1.0 - 1e-9)
            total -= float(np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
        return total

    res = minimize(neg_ll, x0=np.array([0.0, 0.0, 0.0]), method="L-BFGS-B")
    out = {
        "rho_support_mle": float(res.x[0]),
        "beta_support_mle": float(res.x[1]),
        "gamma_support_mle": float(res.x[2]),
        "neg_loglik_support_full": float(res.fun),
        "n_actor_periods_support_fit": int(len(rows)),
        "opt_support_success": bool(res.success),
    }
    try:
        hess_inv = np.asarray(res.hess_inv.todense(), dtype=float)
        se = np.sqrt(np.diag(hess_inv))
        out["rho_support_se"] = float(se[0])
        out["beta_support_se"] = float(se[1])
        out["gamma_support_se"] = float(se[2])
    except Exception:  # pragma: no cover
        pass
    out["neg_loglik_support_alpha_only"] = float(neg_ll(np.array([0.0, 0.0, 0.0])))
    return out


def estimate_entry_baseline_logit(
    count_panel: pd.DataFrame,
    topics_order: list[str],
    actors_order: list[str],
    phi_matrix: np.ndarray,
) -> np.ndarray:
    """Topic-specific baseline logits for new support entry."""
    arrays = build_support_transition_arrays(
        count_panel=count_panel,
        topics_order=topics_order,
        actors_order=actors_order,
        phi_matrix=phi_matrix,
    )
    Y = arrays["Y"]
    K = arrays["K"]
    T, A, N = Y.shape
    entry_success = np.zeros(N, dtype=float)
    entry_trials = np.zeros(N, dtype=float)
    for t in range(1, T):
        for a in range(A):
            if K[t, a] <= 0:
                continue
            prev_support = Y[t - 1, a] if K[t - 1, a] > 0 else np.zeros(N, dtype=float)
            mask = prev_support <= 0
            entry_success += Y[t, a] * mask
            entry_trials += mask.astype(float)
    overall_prob = float(entry_success.sum() / np.clip(entry_trials.sum(), 1.0, None))
    probs = np.divide(
        entry_success,
        np.clip(entry_trials, 1.0, None),
        out=np.full(N, overall_prob, dtype=float),
        where=entry_trials > 0,
    )
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    return np.log(probs / (1.0 - probs))


def estimate_retention_baseline_logit(
    count_panel: pd.DataFrame,
    topics_order: list[str],
    actors_order: list[str],
    phi_matrix: np.ndarray,
) -> np.ndarray:
    """Topic-specific baseline logits for retaining previously held support."""
    arrays = build_support_transition_arrays(
        count_panel=count_panel,
        topics_order=topics_order,
        actors_order=actors_order,
        phi_matrix=phi_matrix,
    )
    Y = arrays["Y"]
    K = arrays["K"]
    T, A, N = Y.shape
    retention_success = np.zeros(N, dtype=float)
    retention_trials = np.zeros(N, dtype=float)
    for t in range(1, T):
        for a in range(A):
            if K[t, a] <= 0 or K[t - 1, a] <= 0:
                continue
            prev_support = Y[t - 1, a]
            mask = prev_support > 0
            retention_success += Y[t, a] * mask
            retention_trials += mask.astype(float)
    overall_prob = float(
        retention_success.sum() / np.clip(retention_trials.sum(), 1.0, None)
    )
    probs = np.divide(
        retention_success,
        np.clip(retention_trials, 1.0, None),
        out=np.full(N, overall_prob, dtype=float),
        where=retention_trials > 0,
    )
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    return np.log(probs / (1.0 - probs))


def fit_support_entry_params(
    count_panel: pd.DataFrame,
    phi_matrix: np.ndarray,
    topics_order: list[str],
    actors_order: list[str],
) -> dict[str, float]:
    """
    Fit support entry on not-yet-held topics only.

    This isolates the local expansion mechanism from portfolio retention.
    """
    alpha_entry = estimate_entry_baseline_logit(
        count_panel=count_panel,
        topics_order=topics_order,
        actors_order=actors_order,
        phi_matrix=phi_matrix,
    )
    arrays = build_support_transition_arrays(
        count_panel=count_panel,
        topics_order=topics_order,
        actors_order=actors_order,
        phi_matrix=phi_matrix,
    )
    Y = arrays["Y"]
    K = arrays["K"]
    fit_prev_all = arrays["fit_prev_all"]
    pop_prev_all = arrays["pop_prev_all"]
    T, A, N = Y.shape

    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for t in range(1, T):
        for a in range(A):
            if K[t, a] <= 0:
                continue
            prev_support = Y[t - 1, a] if K[t - 1, a] > 0 else np.zeros(N, dtype=float)
            entry_mask = prev_support <= 0
            if not entry_mask.any():
                continue
            y = Y[t, a, entry_mask]
            alpha = alpha_entry[entry_mask]
            fit = fit_prev_all[t, a, entry_mask]
            pop_prev = pop_prev_all[t, a, entry_mask]
            rows.append((y, alpha, fit, pop_prev))

    def neg_ll(theta: np.ndarray) -> float:
        delta_entry, beta_entry, gamma_entry = theta
        total = 0.0
        for y, alpha, fit, pop_prev in rows:
            u = alpha + delta_entry + beta_entry * fit + gamma_entry * pop_prev
            p = np.clip(expit(u), 1e-9, 1.0 - 1e-9)
            total -= float(np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
        return total

    res = minimize(neg_ll, x0=np.array([0.0, 0.0, 0.0]), method="L-BFGS-B")
    out = {
        "delta_entry_mle": float(res.x[0]),
        "beta_entry_mle": float(res.x[1]),
        "gamma_entry_mle": float(res.x[2]),
        "neg_loglik_entry_full": float(res.fun),
        "n_actor_periods_entry_fit": int(len(rows)),
        "opt_entry_success": bool(res.success),
    }
    try:
        hess_inv = np.asarray(res.hess_inv.todense(), dtype=float)
        se = np.sqrt(np.diag(hess_inv))
        out["delta_entry_se"] = float(se[0])
        out["beta_entry_se"] = float(se[1])
        out["gamma_entry_se"] = float(se[2])
    except Exception:  # pragma: no cover
        pass
    out["neg_loglik_entry_alpha_only"] = float(neg_ll(np.array([0.0, 0.0, 0.0])))
    return out


def fit_support_retention_params(
    count_panel: pd.DataFrame,
    phi_matrix: np.ndarray,
    topics_order: list[str],
    actors_order: list[str],
) -> dict[str, float]:
    """
    Fit support retention on previously held topics only.

    A small share-based term allows strongly held issues to be retained more
    easily than marginal ones.
    """
    alpha_ret = estimate_retention_baseline_logit(
        count_panel=count_panel,
        topics_order=topics_order,
        actors_order=actors_order,
        phi_matrix=phi_matrix,
    )
    arrays = build_support_transition_arrays(
        count_panel=count_panel,
        topics_order=topics_order,
        actors_order=actors_order,
        phi_matrix=phi_matrix,
    )
    Y = arrays["Y"]
    K = arrays["K"]
    shares = arrays["shares"]
    pop_prev_all = arrays["pop_prev_all"]
    T, A, N = Y.shape

    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for t in range(1, T):
        for a in range(A):
            if K[t, a] <= 0 or K[t - 1, a] <= 0:
                continue
            prev_support = Y[t - 1, a]
            retain_mask = prev_support > 0
            if not retain_mask.any():
                continue
            y = Y[t, a, retain_mask]
            alpha = alpha_ret[retain_mask]
            share_prev = shares[t - 1, a, retain_mask]
            pop_prev = pop_prev_all[t, a, retain_mask]
            rows.append((y, alpha, share_prev, pop_prev))

    def neg_ll(theta: np.ndarray) -> float:
        delta_ret, lambda_ret, gamma_ret = theta
        total = 0.0
        for y, alpha, share_prev, pop_prev in rows:
            u = alpha + delta_ret + lambda_ret * share_prev + gamma_ret * pop_prev
            p = np.clip(expit(u), 1e-9, 1.0 - 1e-9)
            total -= float(np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
        return total

    res = minimize(neg_ll, x0=np.array([0.0, 0.0, 0.0]), method="L-BFGS-B")
    out = {
        "delta_retention_mle": float(res.x[0]),
        "lambda_retention_mle": float(res.x[1]),
        "gamma_retention_mle": float(res.x[2]),
        "neg_loglik_retention_full": float(res.fun),
        "n_actor_periods_retention_fit": int(len(rows)),
        "opt_retention_success": bool(res.success),
    }
    try:
        hess_inv = np.asarray(res.hess_inv.todense(), dtype=float)
        se = np.sqrt(np.diag(hess_inv))
        out["delta_retention_se"] = float(se[0])
        out["lambda_retention_se"] = float(se[1])
        out["gamma_retention_se"] = float(se[2])
    except Exception:  # pragma: no cover
        pass
    out["neg_loglik_retention_alpha_only"] = float(neg_ll(np.array([0.0, 0.0, 0.0])))
    return out


def simulate_history_two_stage_conditioned_on_active_set(
    rng: np.random.Generator,
    budget_table: pd.DataFrame,
    topics_order: list[str],
    phi_matrix: np.ndarray,
    alpha_alloc: np.ndarray,
    alpha_support: np.ndarray,
    *,
    rho_alloc: float,
    beta_alloc: float,
    gamma_alloc: float,
    rho_support: float,
    beta_support: float,
    gamma_support: float,
) -> pd.DataFrame:
    """
    Two-stage history simulator:

    1. Select support (which topics are touched at all).
    2. Allocate the observed budget across the selected support.
    """
    n_topics = len(topics_order)
    zero = np.zeros(n_topics, dtype=float)
    prev_shares_by_actor: dict[str, np.ndarray] = {}
    prev_support_by_actor: dict[str, np.ndarray] = {}
    prev_topic_popularity = np.zeros(n_topics, dtype=float)
    rows: list[pd.DataFrame] = []

    for window_end, budget_row in budget_table.iterrows():
        current_shares_by_actor: dict[str, np.ndarray] = {}
        current_support_by_actor: dict[str, np.ndarray] = {}
        active_actors = [actor for actor, K_i in budget_row.items() if float(K_i) > 0.0]

        for actor in active_actors:
            K_i = int(budget_row[actor])
            shares_prev = prev_shares_by_actor.get(actor, zero)
            support_prev = prev_support_by_actor.get(actor, zero)
            support_prev_sum = support_prev.sum()
            if support_prev_sum > 0:
                fit_support = (support_prev / support_prev_sum) @ phi_matrix
            else:
                fit_support = zero

            u_support = (
                alpha_support
                + rho_support * support_prev
                + beta_support * fit_support
                + gamma_support * prev_topic_popularity
            )
            p_support = np.clip(expit(u_support), 1e-9, 1.0 - 1e-9)
            support = rng.binomial(1, p_support).astype(int)

            # A support set cannot exceed budget if each selected topic must receive
            # at least one count. Truncate to the highest-probability topics.
            n_selected = int(support.sum())
            if n_selected == 0:
                support[np.argmax(p_support)] = 1
                n_selected = 1
            if n_selected > K_i:
                top_idx = np.argsort(p_support)[::-1][:K_i]
                support = np.zeros(n_topics, dtype=int)
                support[top_idx] = 1
                n_selected = int(support.sum())

            u_alloc = (
                alpha_alloc
                + rho_alloc * shares_prev
                + beta_alloc * local_fit_from_phi(shares_prev, phi_matrix)
                + gamma_alloc * prev_topic_popularity
            )
            selected_idx = np.flatnonzero(support > 0)
            alloc = np.zeros(n_topics, dtype=int)
            alloc[selected_idx] = 1
            remaining = int(K_i - n_selected)
            if remaining > 0:
                probs_selected = softmax(u_alloc[selected_idx])
                alloc_extra = rng.multinomial(remaining, probs_selected)
                alloc[selected_idx] += alloc_extra

            shares_now = alloc.astype(float) / np.clip(float(K_i), 1.0, None)
            support_now = (alloc > 0).astype(float)
            current_shares_by_actor[actor] = shares_now
            current_support_by_actor[actor] = support_now
            actor_df = pd.DataFrame(
                {
                    "window_end": int(window_end),
                    "actor": actor,
                    "topic": topics_order,
                    "K_it": int(K_i),
                    "x_ijt_sim": alloc,
                    "s_ijt_sim": shares_now,
                    "active_sim": alloc > 0,
                    "support_selected_sim": support.astype(bool),
                }
            )
            rows.append(actor_df)

        if current_shares_by_actor:
            prev_shares_by_actor = current_shares_by_actor
            prev_support_by_actor = current_support_by_actor
            prev_topic_popularity = (
                np.vstack(list(current_support_by_actor.values())) > 0
            ).mean(axis=0)
        else:
            prev_shares_by_actor = {}
            prev_support_by_actor = {}
            prev_topic_popularity = np.zeros(n_topics, dtype=float)

    return pd.concat(rows, ignore_index=True)


def simulate_history_split_support_conditioned_on_active_set(
    rng: np.random.Generator,
    budget_table: pd.DataFrame,
    topics_order: list[str],
    phi_matrix: np.ndarray,
    alpha_alloc: np.ndarray,
    alpha_entry: np.ndarray,
    alpha_retention: np.ndarray,
    *,
    rho_alloc: float,
    beta_alloc: float,
    gamma_alloc: float,
    delta_entry: float,
    beta_entry: float,
    gamma_entry: float,
    delta_retention: float,
    lambda_retention: float,
    gamma_retention: float,
) -> pd.DataFrame:
    """
    Split support simulator:

    1. retain or drop previously held topics
    2. add new topics from the unheld set
    3. allocate the observed budget within the resulting support

    This keeps the support process simple while separating persistence from
    expansion.
    """
    n_topics = len(topics_order)
    zero = np.zeros(n_topics, dtype=float)
    prev_shares_by_actor: dict[str, np.ndarray] = {}
    prev_support_by_actor: dict[str, np.ndarray] = {}
    prev_topic_popularity = np.zeros(n_topics, dtype=float)
    rows: list[pd.DataFrame] = []

    for window_end, budget_row in budget_table.iterrows():
        current_shares_by_actor: dict[str, np.ndarray] = {}
        current_support_by_actor: dict[str, np.ndarray] = {}
        active_actors = [actor for actor, K_i in budget_row.items() if float(K_i) > 0.0]

        for actor in active_actors:
            K_i = int(budget_row[actor])
            shares_prev = prev_shares_by_actor.get(actor, zero)
            support_prev = prev_support_by_actor.get(actor, zero)
            support_prev_sum = support_prev.sum()
            fit_support = (
                (support_prev / support_prev_sum) @ phi_matrix
                if support_prev_sum > 0
                else zero
            )

            u_entry = (
                alpha_entry
                + delta_entry
                + beta_entry * fit_support
                + gamma_entry * prev_topic_popularity
            )
            u_ret = (
                alpha_retention
                + delta_retention
                + lambda_retention * shares_prev
                + gamma_retention * prev_topic_popularity
            )

            support = np.zeros(n_topics, dtype=int)
            prev_mask = support_prev > 0
            if prev_mask.any():
                p_ret = np.clip(expit(u_ret[prev_mask]), 1e-9, 1.0 - 1e-9)
                support[prev_mask] = rng.binomial(1, p_ret).astype(int)
            entry_mask = ~prev_mask
            if entry_mask.any():
                p_entry = np.clip(expit(u_entry[entry_mask]), 1e-9, 1.0 - 1e-9)
                support[entry_mask] = rng.binomial(1, p_entry).astype(int)

            support_score = np.where(prev_mask, u_ret, u_entry)
            n_selected = int(support.sum())
            if n_selected == 0:
                support[np.argmax(support_score)] = 1
                n_selected = 1
            if n_selected > K_i:
                top_idx = np.argsort(support_score)[::-1][:K_i]
                support = np.zeros(n_topics, dtype=int)
                support[top_idx] = 1
                n_selected = int(support.sum())

            u_alloc = (
                alpha_alloc
                + rho_alloc * shares_prev
                + beta_alloc * local_fit_from_phi(shares_prev, phi_matrix)
                + gamma_alloc * prev_topic_popularity
            )
            selected_idx = np.flatnonzero(support > 0)
            alloc = np.zeros(n_topics, dtype=int)
            alloc[selected_idx] = 1
            remaining = int(K_i - n_selected)
            if remaining > 0:
                probs_selected = softmax(u_alloc[selected_idx])
                alloc_extra = rng.multinomial(remaining, probs_selected)
                alloc[selected_idx] += alloc_extra

            shares_now = alloc.astype(float) / np.clip(float(K_i), 1.0, None)
            support_now = (alloc > 0).astype(float)
            current_shares_by_actor[actor] = shares_now
            current_support_by_actor[actor] = support_now
            actor_df = pd.DataFrame(
                {
                    "window_end": int(window_end),
                    "actor": actor,
                    "topic": topics_order,
                    "K_it": int(K_i),
                    "x_ijt_sim": alloc,
                    "s_ijt_sim": shares_now,
                    "active_sim": alloc > 0,
                    "support_selected_sim": support.astype(bool),
                }
            )
            rows.append(actor_df)

        if current_shares_by_actor:
            prev_shares_by_actor = current_shares_by_actor
            prev_support_by_actor = current_support_by_actor
            prev_topic_popularity = (
                np.vstack(list(current_support_by_actor.values())) > 0
            ).mean(axis=0)
        else:
            prev_shares_by_actor = {}
            prev_support_by_actor = {}
            prev_topic_popularity = np.zeros(n_topics, dtype=float)

    return pd.concat(rows, ignore_index=True)


# %% [markdown]
# ## What We Would Estimate
#
# The practical goal is not to estimate every parameter at once.
#
# A sensible sequence is:
#
# 1. reduced-form entry model
#    - topic intercepts
#    - actor intercepts
#    - locality / fit term
#    - persistence term
#    - topic popularity term
#
# 2. simulation from the estimated reduced-form model
#
# 3. macro validation:
#    - does the simulated system recover the observed structure?
#
# Only after that should we consider a richer latent-affinity model.
#
# In the first empirical pass below:
#
# - `alpha_j` is estimated from pooled topic mass
# - `rho`, `beta`, and `gamma` are estimated by maximum likelihood conditional on
#   the observed active set and observed budgets
#
# This is still a reduced-form model, but it is already enough to ask whether the
# allocation process is locally path-dependent in the data.
#
# The next simplest extension is a two-stage model:
#
# 1. select support (which topics are touched at all)
# 2. allocate budget within the selected support
#
# This directly tests whether the missing ingredient in the one-stage model is
# sharper portfolio selectivity.
#
# If that still fails, the next clean refinement is to split support dynamics
# into:
#
# 1. retention of already-held topics
# 2. entry into not-yet-held topics
#
# That matters because sparse support and local entry are different empirical
# behaviors and should not automatically be forced into the same logit.
#
# If the split model gets the relative structure right but misses average
# support size, a final simple calibration is to adjust only the entry and
# retention intercepts in simulation. This keeps the slope estimates fixed while
# aligning the level of support with the observed ATS.

# %% [markdown]
# ## Macro Statistics To Compare
#
# The comparison target should be macrostructure, not exact microstate equality.
#
# Candidate statistics:
#
# - actor overlap distribution
# - topic concentration distribution
# - share of active topics per actor
# - regime separation / regime persistence
# - locality of new topic entry
# - persistence / retention of portfolio positions


# %%
def actor_overlap_matrix_from_support(X_binary: np.ndarray) -> np.ndarray:
    """Simple actor-actor overlap count from a binary topic support matrix."""
    X_binary = np.asarray(X_binary, dtype=int)
    return X_binary @ X_binary.T


def herfindahl_by_actor(X_counts: np.ndarray) -> np.ndarray:
    """Portfolio concentration for each actor."""
    X_counts = np.asarray(X_counts, dtype=float)
    row_sum = X_counts.sum(axis=1, keepdims=True)
    shares = np.divide(X_counts, np.clip(row_sum, 1.0, None))
    return np.square(shares).sum(axis=1)


def normalized_entropy_by_actor(X_counts: np.ndarray) -> np.ndarray:
    """Normalized portfolio entropy on `[0, 1]`."""
    X_counts = np.asarray(X_counts, dtype=float)
    row_sum = X_counts.sum(axis=1, keepdims=True)
    shares = np.divide(X_counts, np.clip(row_sum, 1.0, None))
    safe_shares = np.clip(shares, 1e-12, None)
    entropy = -np.sum(np.where(shares > 0.0, shares * np.log(safe_shares), 0.0), axis=1)
    return entropy / np.clip(np.log(X_counts.shape[1]), 1e-12, None)


def topic_popularity(X_binary: np.ndarray) -> np.ndarray:
    """How many actors are active in each topic."""
    X_binary = np.asarray(X_binary, dtype=int)
    return X_binary.sum(axis=0)


def build_observed_history_like(count_panel: pd.DataFrame) -> pd.DataFrame:
    """Observed history in the same column layout as simulated histories."""
    return pd.DataFrame(
        {
            "window_end": count_panel["window_end"].astype(int),
            "actor": count_panel["actor"],
            "topic": count_panel["topic"],
            "K_it": count_panel["K_it"].astype(int),
            "x_ijt_sim": count_panel["x_ijt"].astype(int),
            "s_ijt_sim": count_panel["s_ijt"].astype(float),
            "active_sim": count_panel["active_raw"].astype(bool),
        }
    )


def embed_topics_from_phi(
    phi_matrix: np.ndarray, topics_order: list[str], n_components: int = 2
) -> pd.DataFrame:
    """
    Simple spectral embedding of the pooled topic-space matrix.

    This is only a diagnostic coordinate system. It is not used by the
    simulator itself.
    """
    phi_sym = 0.5 * (np.asarray(phi_matrix, dtype=float) + np.asarray(phi_matrix, dtype=float).T)
    evals, evecs = np.linalg.eigh(phi_sym)
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order][:n_components], 0.0, None)
    evecs = evecs[:, order][:, :n_components]
    coords = evecs * np.sqrt(evals)
    cols = [f"coord_{i + 1}" for i in range(coords.shape[1])]
    return pd.DataFrame(coords, index=topics_order, columns=cols).reset_index(names="topic")


def compute_actor_centroid_shifts(
    history_df: pd.DataFrame, topic_coords: pd.DataFrame
) -> pd.DataFrame:
    """
    Consecutive-period portfolio centroid movement for active actors.

    Interpretation:
    lower centroid shift implies actors remain in a more stable region of the
    concern space through time.
    """
    coord_cols = [c for c in topic_coords.columns if c.startswith("coord_")]
    coords = topic_coords.set_index("topic")[coord_cols]
    centroid_rows: list[dict[str, float | int | str]] = []
    for (actor, window_end), group in history_df.groupby(["actor", "window_end"], sort=True):
        active = group[group["active_sim"]].copy()
        if active.empty:
            continue
        weights = active["s_ijt_sim"].to_numpy(dtype=float)
        if weights.sum() <= 0.0:
            weights = np.full(len(active), 1.0 / len(active), dtype=float)
        else:
            weights = weights / weights.sum()
        centroid = weights @ coords.loc[active["topic"]].to_numpy(dtype=float)
        row: dict[str, float | int | str] = {
            "actor": actor,
            "window_end": int(window_end),
            "support_size": int(active["active_sim"].sum()),
        }
        row.update({c: float(v) for c, v in zip(coord_cols, centroid)})
        centroid_rows.append(row)

    centroids = pd.DataFrame(centroid_rows).sort_values(["actor", "window_end"])
    for col in coord_cols:
        centroids[f"prev_{col}"] = centroids.groupby("actor")[col].shift(1)
    centroids["prev_window_end"] = centroids.groupby("actor")["window_end"].shift(1)
    centroids["period_gap"] = centroids["window_end"] - centroids["prev_window_end"]
    consecutive = centroids[centroids["period_gap"] == 1].copy()
    delta_sq = np.zeros(len(consecutive), dtype=float)
    for col in coord_cols:
        delta_sq += np.square(
            consecutive[col].to_numpy(dtype=float)
            - consecutive[f"prev_{col}"].to_numpy(dtype=float)
        )
    consecutive["centroid_shift"] = np.sqrt(delta_sq)
    return consecutive[
        ["actor", "window_end", "prev_window_end", "support_size", "centroid_shift"]
    ].reset_index(drop=True)


def compute_new_entry_phi_proximity(
    history_df: pd.DataFrame, phi_matrix: np.ndarray, topics_order: list[str]
) -> pd.DataFrame:
    """
    Phi-based proximity of newly entered topics to the actor's prior support.

    We record three simple diagnostics for each newly entered topic:
    - mean Phi proximity to the previous support
    - max Phi proximity to the previous support
    - percentile rank of mean Phi proximity among all available topics not yet in
      the previous support

    If actors expand locally in the native concern space, these Phi measures
    should be relatively high.
    """
    phi_df = pd.DataFrame(phi_matrix, index=topics_order, columns=topics_order)
    rows: list[dict[str, float | int | str]] = []

    for actor, actor_df in history_df.groupby("actor", sort=True):
        actor_df = actor_df.sort_values(["window_end", "topic"])
        period_support: dict[int, set[str]] = {}

        for window_end, period_df in actor_df.groupby("window_end", sort=True):
            active_df = period_df[period_df["active_sim"]].copy()
            if active_df.empty:
                continue
            period_support[int(window_end)] = set(active_df["topic"].tolist())

        actor_periods = sorted(period_support)
        for prev_end, curr_end in zip(actor_periods[:-1], actor_periods[1:]):
            if curr_end - prev_end != 1:
                continue
            prev_topics = sorted(period_support[prev_end])
            curr_topics = sorted(period_support[curr_end])
            new_topics = sorted(set(curr_topics) - set(prev_topics))
            if not new_topics:
                continue
            candidate_topics = sorted(set(topics_order) - set(prev_topics))
            if not candidate_topics:
                continue
            phi_to_prev = phi_df.loc[prev_topics, candidate_topics]
            mean_phi = phi_to_prev.mean(axis=0)
            max_phi = phi_to_prev.max(axis=0)
            mean_phi_rank = mean_phi.rank(method="average", pct=True)
            max_phi_rank = max_phi.rank(method="average", pct=True)
            for topic in new_topics:
                if topic not in mean_phi.index:
                    continue
                rows.append(
                    {
                        "actor": actor,
                        "window_end": int(curr_end),
                        "topic": topic,
                        "mean_phi_to_prev_support": float(mean_phi.loc[topic]),
                        "max_phi_to_prev_support": float(max_phi.loc[topic]),
                        "mean_phi_rank_pct": float(mean_phi_rank.loc[topic]),
                        "max_phi_rank_pct": float(max_phi_rank.loc[topic]),
                        "n_candidate_topics": int(len(candidate_topics)),
                    }
                )

    return pd.DataFrame(rows)


def compare_latest_period_concentration_to_null(
    latest_slice: pd.DataFrame, null_draw: pd.DataFrame, topics_order: list[str]
) -> pd.DataFrame:
    """Observed latest-period concentration versus the budget-only null."""
    observed = (
        latest_slice.pivot(index="actor", columns="topic", values="x_ijt")
        .fillna(0)
        .reindex(columns=topics_order, fill_value=0)
        .sort_index()
    )
    null_counts = (
        null_draw.pivot(index="actor", columns="topic", values="x_ijt_sim")
        .fillna(0)
        .reindex(columns=topics_order, fill_value=0)
        .sort_index()
    )

    def _summarize(label: str, counts_df: pd.DataFrame) -> pd.DataFrame:
        X = counts_df.to_numpy(dtype=float)
        return pd.DataFrame(
            {
                "actor": counts_df.index,
                "label": label,
                "budget": X.sum(axis=1),
                "support_size": (X > 0).sum(axis=1),
                "herfindahl": herfindahl_by_actor(X),
                "entropy_norm": normalized_entropy_by_actor(X),
            }
        )

    return pd.concat(
        [_summarize("observed_latest", observed), _summarize("budget_only_null", null_counts)],
        ignore_index=True,
    )


def calibrate_split_support_offsets(
    count_panel: pd.DataFrame,
    budget_table: pd.DataFrame,
    topics_order: list[str],
    phi_matrix: np.ndarray,
    alpha_alloc: np.ndarray,
    alpha_entry: np.ndarray,
    alpha_retention: np.ndarray,
    *,
    rho_alloc: float,
    beta_alloc: float,
    gamma_alloc: float,
    delta_entry: float,
    beta_entry: float,
    gamma_entry: float,
    delta_retention: float,
    lambda_retention: float,
    gamma_retention: float,
    random_seed: int,
) -> dict[str, float]:
    """
    Calibrate only the split-model intercepts after fitting.

    This keeps the fitted slope structure intact and asks a narrower practical
    question: what small intercept shifts best align support levels and local
    entry with the observed ATS?
    """
    observed_summary = summarize_observed_and_simulated_history(
        count_panel, build_observed_history_like(count_panel)
    )
    observed_entry_phi = compute_new_entry_phi_proximity(
        build_observed_history_like(count_panel), phi_matrix, topics_order
    )
    obs_mean_active = float(observed_summary["mean_active_topics_obs"].mean())
    obs_mean_pop = float(observed_summary["mean_topic_popularity_obs"].mean())
    obs_entry_rank = float(observed_entry_phi["mean_phi_rank_pct"].mean())

    best: dict[str, float] | None = None
    entry_shift_grid = np.linspace(-0.25, 1.25, 7)
    retention_shift_grid = np.linspace(-0.25, 0.75, 5)

    for entry_shift in entry_shift_grid:
        for retention_shift in retention_shift_grid:
            sim_history = simulate_history_split_support_conditioned_on_active_set(
                rng=np.random.default_rng(random_seed),
                budget_table=budget_table,
                topics_order=topics_order,
                phi_matrix=phi_matrix,
                alpha_alloc=alpha_alloc,
                alpha_entry=alpha_entry,
                alpha_retention=alpha_retention,
                rho_alloc=rho_alloc,
                beta_alloc=beta_alloc,
                gamma_alloc=gamma_alloc,
                delta_entry=delta_entry + float(entry_shift),
                beta_entry=beta_entry,
                gamma_entry=gamma_entry,
                delta_retention=delta_retention + float(retention_shift),
                lambda_retention=lambda_retention,
                gamma_retention=gamma_retention,
            )
            hist = summarize_observed_and_simulated_history(count_panel, sim_history)
            entry_phi = compute_new_entry_phi_proximity(sim_history, phi_matrix, topics_order)
            mean_active = float(hist["mean_active_topics_sim"].mean())
            mean_pop = float(hist["mean_topic_popularity_sim"].mean())
            corr_active = float(
                hist["mean_active_topics_obs"].corr(hist["mean_active_topics_sim"])
            )
            corr_pop = float(
                hist["mean_topic_popularity_obs"].corr(hist["mean_topic_popularity_sim"])
            )
            entry_rank = float(entry_phi["mean_phi_rank_pct"].mean())

            loss = (
                abs(mean_active - obs_mean_active) / np.clip(obs_mean_active, 1e-9, None)
                + 0.75
                * abs(mean_pop - obs_mean_pop)
                / np.clip(obs_mean_pop, 1e-9, None)
                + 0.75
                * abs(entry_rank - obs_entry_rank)
                / np.clip(obs_entry_rank, 1e-9, None)
                + 0.5 * (1.0 - corr_active)
                + 0.25 * (1.0 - corr_pop)
            )

            candidate = {
                "entry_shift_best": float(entry_shift),
                "retention_shift_best": float(retention_shift),
                "calibration_loss": float(loss),
                "mean_active_topics_split_calibrated_avg": mean_active,
                "mean_topic_popularity_split_calibrated_avg": mean_pop,
                "corr_mean_active_topics_split_calibrated": corr_active,
                "corr_mean_topic_popularity_split_calibrated": corr_pop,
                "mean_entry_phi_rank_split_calibrated": entry_rank,
            }
            if best is None or candidate["calibration_loss"] < best["calibration_loss"]:
                best = candidate

    assert best is not None
    return best


def run_model_suite(
    count_panel: pd.DataFrame,
    phi_matrix: np.ndarray,
    topics_order: list[str],
    actors_order: list[str],
    *,
    random_seed: int,
) -> dict[str, object]:
    """
    Fit the one-stage, pooled-support, and split-support models, then simulate
    conditioned histories from each.

    Keeping this in one helper makes it easy to re-run the same validation logic
    on thinned versions of the ATS without duplicating the full pipeline.
    """
    budget_table = build_budget_table(count_panel)
    alpha_hat = estimate_topic_salience_alpha(count_panel, topics_order)
    alpha_support_hat = estimate_support_baseline_logit(count_panel, topics_order)
    alpha_entry_hat = estimate_entry_baseline_logit(
        count_panel=count_panel,
        topics_order=topics_order,
        actors_order=actors_order,
        phi_matrix=phi_matrix,
    )
    alpha_retention_hat = estimate_retention_baseline_logit(
        count_panel=count_panel,
        topics_order=topics_order,
        actors_order=actors_order,
        phi_matrix=phi_matrix,
    )

    param_estimates = fit_reduced_form_allocation_params(
        count_panel=count_panel,
        phi_matrix=phi_matrix,
        topics_order=topics_order,
        actors_order=actors_order,
    )
    support_param_estimates = fit_support_selection_params(
        count_panel=count_panel,
        phi_matrix=phi_matrix,
        topics_order=topics_order,
        actors_order=actors_order,
    )
    entry_param_estimates = fit_support_entry_params(
        count_panel=count_panel,
        phi_matrix=phi_matrix,
        topics_order=topics_order,
        actors_order=actors_order,
    )
    retention_param_estimates = fit_support_retention_params(
        count_panel=count_panel,
        phi_matrix=phi_matrix,
        topics_order=topics_order,
        actors_order=actors_order,
    )

    sim_history = simulate_history_conditioned_on_active_set(
        rng=np.random.default_rng(random_seed),
        budget_table=budget_table,
        topics_order=topics_order,
        phi_matrix=phi_matrix,
        alpha=alpha_hat,
        rho=param_estimates["rho_mle"],
        beta=param_estimates["beta_mle"],
        gamma=param_estimates["gamma_mle"],
    )
    history_summary = summarize_observed_and_simulated_history(count_panel, sim_history)

    sim_history_two_stage = simulate_history_two_stage_conditioned_on_active_set(
        rng=np.random.default_rng(random_seed),
        budget_table=budget_table,
        topics_order=topics_order,
        phi_matrix=phi_matrix,
        alpha_alloc=alpha_hat,
        alpha_support=alpha_support_hat,
        rho_alloc=param_estimates["rho_mle"],
        beta_alloc=param_estimates["beta_mle"],
        gamma_alloc=param_estimates["gamma_mle"],
        rho_support=support_param_estimates["rho_support_mle"],
        beta_support=support_param_estimates["beta_support_mle"],
        gamma_support=support_param_estimates["gamma_support_mle"],
    )
    history_summary_two_stage = summarize_observed_and_simulated_history(
        count_panel, sim_history_two_stage
    )

    split_calibration = calibrate_split_support_offsets(
        count_panel=count_panel,
        budget_table=budget_table,
        topics_order=topics_order,
        phi_matrix=phi_matrix,
        alpha_alloc=alpha_hat,
        alpha_entry=alpha_entry_hat,
        alpha_retention=alpha_retention_hat,
        rho_alloc=param_estimates["rho_mle"],
        beta_alloc=param_estimates["beta_mle"],
        gamma_alloc=param_estimates["gamma_mle"],
        delta_entry=entry_param_estimates["delta_entry_mle"],
        beta_entry=entry_param_estimates["beta_entry_mle"],
        gamma_entry=entry_param_estimates["gamma_entry_mle"],
        delta_retention=retention_param_estimates["delta_retention_mle"],
        lambda_retention=retention_param_estimates["lambda_retention_mle"],
        gamma_retention=retention_param_estimates["gamma_retention_mle"],
        random_seed=random_seed,
    )
    sim_history_split = simulate_history_split_support_conditioned_on_active_set(
        rng=np.random.default_rng(random_seed),
        budget_table=budget_table,
        topics_order=topics_order,
        phi_matrix=phi_matrix,
        alpha_alloc=alpha_hat,
        alpha_entry=alpha_entry_hat,
        alpha_retention=alpha_retention_hat,
        rho_alloc=param_estimates["rho_mle"],
        beta_alloc=param_estimates["beta_mle"],
        gamma_alloc=param_estimates["gamma_mle"],
        delta_entry=entry_param_estimates["delta_entry_mle"]
        + split_calibration["entry_shift_best"],
        beta_entry=entry_param_estimates["beta_entry_mle"],
        gamma_entry=entry_param_estimates["gamma_entry_mle"],
        delta_retention=retention_param_estimates["delta_retention_mle"]
        + split_calibration["retention_shift_best"],
        lambda_retention=retention_param_estimates["lambda_retention_mle"],
        gamma_retention=retention_param_estimates["gamma_retention_mle"],
    )
    history_summary_split = summarize_observed_and_simulated_history(
        count_panel, sim_history_split
    )

    history_fit_summary = {
        "mean_active_topics_obs_avg": float(history_summary["mean_active_topics_obs"].mean()),
        "mean_active_topics_sim_avg": float(history_summary["mean_active_topics_sim"].mean()),
        "mean_topic_popularity_obs_avg": float(
            history_summary["mean_topic_popularity_obs"].mean()
        ),
        "mean_topic_popularity_sim_avg": float(
            history_summary["mean_topic_popularity_sim"].mean()
        ),
        "corr_mean_active_topics": float(
            history_summary["mean_active_topics_obs"].corr(
                history_summary["mean_active_topics_sim"]
            )
        ),
        "corr_mean_topic_popularity": float(
            history_summary["mean_topic_popularity_obs"].corr(
                history_summary["mean_topic_popularity_sim"]
            )
        ),
        "mean_active_topics_two_stage_avg": float(
            history_summary_two_stage["mean_active_topics_sim"].mean()
        ),
        "mean_topic_popularity_two_stage_avg": float(
            history_summary_two_stage["mean_topic_popularity_sim"].mean()
        ),
        "corr_mean_active_topics_two_stage": float(
            history_summary_two_stage["mean_active_topics_obs"].corr(
                history_summary_two_stage["mean_active_topics_sim"]
            )
        ),
        "corr_mean_topic_popularity_two_stage": float(
            history_summary_two_stage["mean_topic_popularity_obs"].corr(
                history_summary_two_stage["mean_topic_popularity_sim"]
            )
        ),
        "mean_active_topics_split_avg": float(
            history_summary_split["mean_active_topics_sim"].mean()
        ),
        "mean_topic_popularity_split_avg": float(
            history_summary_split["mean_topic_popularity_sim"].mean()
        ),
        "corr_mean_active_topics_split": float(
            history_summary_split["mean_active_topics_obs"].corr(
                history_summary_split["mean_active_topics_sim"]
            )
        ),
        "corr_mean_topic_popularity_split": float(
            history_summary_split["mean_topic_popularity_obs"].corr(
                history_summary_split["mean_topic_popularity_sim"]
            )
        ),
        **split_calibration,
    }

    return {
        "budget_table": budget_table,
        "alpha_hat": alpha_hat,
        "alpha_support_hat": alpha_support_hat,
        "alpha_entry_hat": alpha_entry_hat,
        "alpha_retention_hat": alpha_retention_hat,
        "param_estimates": param_estimates,
        "support_param_estimates": support_param_estimates,
        "entry_param_estimates": entry_param_estimates,
        "retention_param_estimates": retention_param_estimates,
        "split_calibration": split_calibration,
        "sim_history": sim_history,
        "history_summary": history_summary,
        "sim_history_two_stage": sim_history_two_stage,
        "history_summary_two_stage": history_summary_two_stage,
        "sim_history_split": sim_history_split,
        "history_summary_split": history_summary_split,
        "history_fit_summary": history_fit_summary,
    }


def summarize_process_uncertainty_from_repeated_simulation(
    count_panel: pd.DataFrame,
    model_suite: dict[str, object],
    phi_matrix: np.ndarray,
    topics_order: list[str],
    *,
    n_reps: int,
    base_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Re-simulate each fitted model repeatedly with fixed parameters.

    This captures process uncertainty only: the active set, budgets, pooled
    concern space, and fitted coefficients are held fixed while the stochastic
    portfolio draws vary across repetitions.
    """
    budget_table = model_suite["budget_table"]
    alpha_hat = model_suite["alpha_hat"]
    alpha_support_hat = model_suite["alpha_support_hat"]
    alpha_entry_hat = model_suite["alpha_entry_hat"]
    alpha_retention_hat = model_suite["alpha_retention_hat"]
    param_estimates = model_suite["param_estimates"]
    support_param_estimates = model_suite["support_param_estimates"]
    entry_param_estimates = model_suite["entry_param_estimates"]
    retention_param_estimates = model_suite["retention_param_estimates"]
    split_calibration = model_suite["split_calibration"]

    observed_history = build_observed_history_like(count_panel)
    observed_summary = summarize_observed_and_simulated_history(
        count_panel, observed_history
    )[
        ["window_end", "mean_active_topics_obs", "mean_topic_popularity_obs"]
    ].copy()
    observed_entry = compute_new_entry_phi_proximity(
        observed_history, phi_matrix, topics_order
    )
    observed_entry_rank = float(observed_entry["mean_phi_rank_pct"].mean())

    history_rows: list[pd.DataFrame] = []
    entry_rows: list[dict[str, float | int | str]] = []

    for rep in range(n_reps):
        rep_seed = int(base_seed + 10_000 + rep)

        sim_histories = {
            "one_stage": simulate_history_conditioned_on_active_set(
                rng=np.random.default_rng(rep_seed),
                budget_table=budget_table,
                topics_order=topics_order,
                phi_matrix=phi_matrix,
                alpha=alpha_hat,
                rho=param_estimates["rho_mle"],
                beta=param_estimates["beta_mle"],
                gamma=param_estimates["gamma_mle"],
            ),
            "two_stage": simulate_history_two_stage_conditioned_on_active_set(
                rng=np.random.default_rng(rep_seed + 1),
                budget_table=budget_table,
                topics_order=topics_order,
                phi_matrix=phi_matrix,
                alpha_alloc=alpha_hat,
                alpha_support=alpha_support_hat,
                rho_alloc=param_estimates["rho_mle"],
                beta_alloc=param_estimates["beta_mle"],
                gamma_alloc=param_estimates["gamma_mle"],
                rho_support=support_param_estimates["rho_support_mle"],
                beta_support=support_param_estimates["beta_support_mle"],
                gamma_support=support_param_estimates["gamma_support_mle"],
            ),
            "split_support": simulate_history_split_support_conditioned_on_active_set(
                rng=np.random.default_rng(rep_seed + 2),
                budget_table=budget_table,
                topics_order=topics_order,
                phi_matrix=phi_matrix,
                alpha_alloc=alpha_hat,
                alpha_entry=alpha_entry_hat,
                alpha_retention=alpha_retention_hat,
                rho_alloc=param_estimates["rho_mle"],
                beta_alloc=param_estimates["beta_mle"],
                gamma_alloc=param_estimates["gamma_mle"],
                delta_entry=entry_param_estimates["delta_entry_mle"]
                + split_calibration["entry_shift_best"],
                beta_entry=entry_param_estimates["beta_entry_mle"],
                gamma_entry=entry_param_estimates["gamma_entry_mle"],
                delta_retention=retention_param_estimates["delta_retention_mle"]
                + split_calibration["retention_shift_best"],
                lambda_retention=retention_param_estimates["lambda_retention_mle"],
                gamma_retention=retention_param_estimates["gamma_retention_mle"],
            ),
        }

        for model_name, sim_history_model in sim_histories.items():
            hist = summarize_observed_and_simulated_history(count_panel, sim_history_model)
            history_rows.append(
                hist[
                    [
                        "window_end",
                        "mean_active_topics_sim",
                        "mean_topic_popularity_sim",
                    ]
                ].assign(model=model_name, replicate=int(rep))
            )
            entry_phi = compute_new_entry_phi_proximity(
                sim_history_model, phi_matrix, topics_order
            )
            entry_rows.append(
                {
                    "model": model_name,
                    "replicate": int(rep),
                    "mean_entry_phi_rank": float(entry_phi["mean_phi_rank_pct"].mean()),
                }
            )

    process_history = pd.concat(history_rows, ignore_index=True)
    process_entry = pd.DataFrame(entry_rows)

    def _q(col: str, q: float):
        return (col, lambda x, q=q: float(x.quantile(q)))

    process_history_summary = (
        process_history.groupby(["model", "window_end"], as_index=False)
        .agg(
            mean_active_topics_mean=("mean_active_topics_sim", "mean"),
            mean_active_topics_q05=_q("mean_active_topics_sim", 0.05),
            mean_active_topics_q25=_q("mean_active_topics_sim", 0.25),
            mean_active_topics_q75=_q("mean_active_topics_sim", 0.75),
            mean_active_topics_q95=_q("mean_active_topics_sim", 0.95),
            mean_topic_popularity_mean=("mean_topic_popularity_sim", "mean"),
            mean_topic_popularity_q05=_q("mean_topic_popularity_sim", 0.05),
            mean_topic_popularity_q25=_q("mean_topic_popularity_sim", 0.25),
            mean_topic_popularity_q75=_q("mean_topic_popularity_sim", 0.75),
            mean_topic_popularity_q95=_q("mean_topic_popularity_sim", 0.95),
        )
        .merge(observed_summary, on="window_end", how="left")
    )

    process_entry_summary = process_entry.groupby("model", as_index=False).agg(
        n_reps=("replicate", "nunique"),
        mean_entry_phi_rank_mean=("mean_entry_phi_rank", "mean"),
        mean_entry_phi_rank_q05=_q("mean_entry_phi_rank", 0.05),
        mean_entry_phi_rank_q25=_q("mean_entry_phi_rank", 0.25),
        mean_entry_phi_rank_q75=_q("mean_entry_phi_rank", 0.75),
        mean_entry_phi_rank_q95=_q("mean_entry_phi_rank", 0.95),
        mean_entry_phi_rank_sd=("mean_entry_phi_rank", "std"),
    )
    observed_entry_row = pd.DataFrame(
        [
            {
                "model": "observed",
                "n_reps": 1,
                "mean_entry_phi_rank_mean": observed_entry_rank,
                "mean_entry_phi_rank_q05": observed_entry_rank,
                "mean_entry_phi_rank_q25": observed_entry_rank,
                "mean_entry_phi_rank_q75": observed_entry_rank,
                "mean_entry_phi_rank_q95": observed_entry_rank,
                "mean_entry_phi_rank_sd": 0.0,
            }
        ]
    )
    process_entry_summary = pd.concat(
        [observed_entry_row, process_entry_summary], ignore_index=True
    )
    return process_history_summary, process_entry_summary


def build_split_parameter_uncertainty_table(
    param_estimates: dict[str, float],
    entry_param_estimates: dict[str, float],
    retention_param_estimates: dict[str, float],
) -> pd.DataFrame:
    """Approximate split-model parameter intervals from inverse-Hessian SEs."""

    rows: list[dict[str, float | str | int]] = []
    specs = [
        ("allocation", 1, "Persistence (rho)", "rho_mle", "rho_se"),
        ("allocation", 2, "Local fit (beta)", "beta_mle", "beta_se"),
        ("allocation", 3, "Popularity (gamma)", "gamma_mle", "gamma_se"),
        ("entry", 1, "Entry intercept (delta)", "delta_entry_mle", "delta_entry_se"),
        ("entry", 2, "Local fit (beta_ent)", "beta_entry_mle", "beta_entry_se"),
        ("entry", 3, "Popularity (gamma_ent)", "gamma_entry_mle", "gamma_entry_se"),
        (
            "retention",
            1,
            "Retention intercept (delta)",
            "delta_retention_mle",
            "delta_retention_se",
        ),
        (
            "retention",
            2,
            "Prior share (lambda_ret)",
            "lambda_retention_mle",
            "lambda_retention_se",
        ),
        (
            "retention",
            3,
            "Popularity (gamma_ret)",
            "gamma_retention_mle",
            "gamma_retention_se",
        ),
    ]
    lookup = {
        "allocation": param_estimates,
        "entry": entry_param_estimates,
        "retention": retention_param_estimates,
    }
    for stage, stage_order, label, estimate_key, se_key in specs:
        source = lookup[stage]
        estimate = float(source[estimate_key])
        se = float(source.get(se_key, np.nan))
        lower95 = float(estimate - 1.96 * se) if np.isfinite(se) else np.nan
        upper95 = float(estimate + 1.96 * se) if np.isfinite(se) else np.nan
        rows.append(
            {
                "stage": stage,
                "stage_order": stage_order,
                "term": label,
                "estimate": estimate,
                "se": se,
                "lower95": lower95,
                "upper95": upper95,
            }
        )
    return pd.DataFrame(rows)


def subsample_submitted_by_time_unit(
    submitted_df: pd.DataFrame,
    time_col: str,
    keep_fraction: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Thin the raw submission data within each period.

    This preserves broad temporal coverage while making dense periods sparser by
    the same proportion as early periods.
    """
    keep_fraction = float(np.clip(keep_fraction, 0.0, 1.0))
    sampled_parts: list[pd.DataFrame] = []
    for period_value, period_df in submitted_df.groupby(time_col, sort=True):
        n_period = int(len(period_df))
        if n_period == 0:
            continue
        n_keep = int(np.ceil(keep_fraction * n_period))
        n_keep = max(1, min(n_period, n_keep))
        chosen_idx = rng.choice(period_df.index.to_numpy(), size=n_keep, replace=False)
        sampled_parts.append(period_df.loc[np.sort(chosen_idx)])
    return (
        pd.concat(sampled_parts, ignore_index=False)
        .sort_values([time_col])
        .reset_index(drop=True)
    )


def run_time_balanced_subsample_validation(
    submitted_df: pd.DataFrame,
    time_col: str,
    periods: list[tuple[int, int]],
    all_members_raw: set[str],
    all_topics_raw: set[str],
    topics_order: list[str],
    actors_order: list[str],
    phi_matrix: np.ndarray,
    *,
    rca_threshold: float,
    keep_fractions: tuple[float, ...] = (0.4, 0.6, 0.8),
    n_reps: int = 5,
    base_seed: int = 7,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Refit and re-evaluate the model on time-balanced thinned corpora.

    This is a robustness check against the concern that later dense periods are
    doing all the work. The topic space `Phi` is held fixed to isolate the model
    from space re-estimation noise.
    """
    rows: list[dict[str, float | int]] = []
    total_jobs = len(keep_fractions) * n_reps
    job_idx = 0
    for keep_fraction in keep_fractions:
        for rep in range(n_reps):
            job_idx += 1
            if verbose:
                print(
                    f"[subsample {job_idx}/{total_jobs}] "
                    f"keep_fraction={keep_fraction:.2f}, rep={rep + 1}/{n_reps}: "
                    "rebuilding panel and refitting models...",
                    flush=True,
                )
            rng = np.random.default_rng(base_seed + int(round(1000 * keep_fraction)) + rep)
            submitted_sub = subsample_submitted_by_time_unit(
                submitted_df=submitted_df,
                time_col=time_col,
                keep_fraction=keep_fraction,
                rng=rng,
            )
            count_panel_sub = build_count_panel_with_active_rca(
                submitted_df=submitted_sub,
                time_col=time_col,
                periods=periods,
                all_members_raw=all_members_raw,
                all_topics_raw=all_topics_raw,
                topics_order=topics_order,
                actors_order=actors_order,
                rca_threshold=rca_threshold,
            )
            suite = run_model_suite(
                count_panel=count_panel_sub,
                phi_matrix=phi_matrix,
                topics_order=topics_order,
                actors_order=actors_order,
                random_seed=base_seed + rep,
            )
            observed_sub = build_observed_history_like(count_panel_sub)
            entry_obs = compute_new_entry_phi_proximity(
                observed_sub, phi_matrix, topics_order
            )
            entry_split = compute_new_entry_phi_proximity(
                suite["sim_history_split"], phi_matrix, topics_order
            )
            active_actor_counts = (
                count_panel_sub.groupby(["window_end", "actor"])["K_it"].first().gt(0).reset_index()
            )
            active_actor_counts = (
                active_actor_counts[active_actor_counts["K_it"]]
                .groupby("window_end")["actor"]
                .nunique()
            )
            row = {
                "keep_fraction": float(keep_fraction),
                "replicate": int(rep),
                "n_submitted_rows": int(len(submitted_sub)),
                "mean_active_actors_per_period": float(active_actor_counts.mean()),
                "rho_mle": float(suite["param_estimates"]["rho_mle"]),
                "beta_mle": float(suite["param_estimates"]["beta_mle"]),
                "gamma_mle": float(suite["param_estimates"]["gamma_mle"]),
                "beta_entry_mle": float(suite["entry_param_estimates"]["beta_entry_mle"]),
                "lambda_retention_mle": float(
                    suite["retention_param_estimates"]["lambda_retention_mle"]
                ),
                "corr_mean_active_topics": float(
                    suite["history_fit_summary"]["corr_mean_active_topics"]
                ),
                "corr_mean_active_topics_two_stage": float(
                    suite["history_fit_summary"]["corr_mean_active_topics_two_stage"]
                ),
                "corr_mean_active_topics_split": float(
                    suite["history_fit_summary"]["corr_mean_active_topics_split"]
                ),
                "corr_mean_topic_popularity": float(
                    suite["history_fit_summary"]["corr_mean_topic_popularity"]
                ),
                "corr_mean_topic_popularity_two_stage": float(
                    suite["history_fit_summary"]["corr_mean_topic_popularity_two_stage"]
                ),
                "corr_mean_topic_popularity_split": float(
                    suite["history_fit_summary"]["corr_mean_topic_popularity_split"]
                ),
                "mean_active_topics_obs_avg": float(
                    suite["history_fit_summary"]["mean_active_topics_obs_avg"]
                ),
                "mean_active_topics_split_avg": float(
                    suite["history_fit_summary"]["mean_active_topics_split_avg"]
                ),
                "mean_topic_popularity_obs_avg": float(
                    suite["history_fit_summary"]["mean_topic_popularity_obs_avg"]
                ),
                "mean_topic_popularity_split_avg": float(
                    suite["history_fit_summary"]["mean_topic_popularity_split_avg"]
                ),
                "mean_entry_phi_rank_observed": float(
                    entry_obs["mean_phi_rank_pct"].mean()
                ),
                "mean_entry_phi_rank_split": float(
                    entry_split["mean_phi_rank_pct"].mean()
                ),
            }
            rows.append(row)
            if verbose:
                print(
                    f"[subsample {job_idx}/{total_jobs}] done: "
                    f"split active r={row['corr_mean_active_topics_split']:.3f}, "
                    f"two-stage active r={row['corr_mean_active_topics_two_stage']:.3f}, "
                    f"one-stage active r={row['corr_mean_active_topics']:.3f}, "
                    f"split entry rank={row['mean_entry_phi_rank_split']:.3f}",
                    flush=True,
                )

    validation_df = pd.DataFrame(rows).sort_values(["keep_fraction", "replicate"])
    summary_df = (
        validation_df.groupby("keep_fraction")
        .agg(
            n_reps=("replicate", "nunique"),
            mean_active_actor_periods=("mean_active_actors_per_period", "mean"),
            mean_corr_active_one_stage=("corr_mean_active_topics", "mean"),
            mean_corr_active_two_stage=("corr_mean_active_topics_two_stage", "mean"),
            mean_corr_active_split=("corr_mean_active_topics_split", "mean"),
            sd_corr_active_split=("corr_mean_active_topics_split", "std"),
            mean_corr_pop_split=("corr_mean_topic_popularity_split", "mean"),
            sd_corr_pop_split=("corr_mean_topic_popularity_split", "std"),
            mean_entry_phi_rank_observed=("mean_entry_phi_rank_observed", "mean"),
            mean_entry_phi_rank_split=("mean_entry_phi_rank_split", "mean"),
            mean_beta_entry=("beta_entry_mle", "mean"),
            sd_beta_entry=("beta_entry_mle", "std"),
            mean_lambda_retention=("lambda_retention_mle", "mean"),
            sd_lambda_retention=("lambda_retention_mle", "std"),
        )
        .reset_index()
    )
    return validation_df, summary_df


# %% [markdown]
# ## Immediate Practical Roadmap For This Project
#
# A good first implementation sequence in this repo would be:
#
# 1. Use the hazard panel machinery to estimate a reduced-form topic-entry model.
# 2. Preserve observed `K_it` budgets by actor-period.
# 3. Simulate synthetic histories under nested models:
#    - budget only
#    - budget + topic salience
#    - budget + salience + local fit
# 4. Compare observed ATS to simulated histories on:
#    - entry locality
#    - overlap
#    - concentration
#    - regime persistence
# 5. Only if needed, add actor-topic latent affinity terms.
#
# That would let us say something much sharper than:
#
# - "actors are strategically differentiated"
#
# We could instead say:
#
# - "the observed ATS occupies a highly non-random region of the feasible
#   actor-topic space, and reproducing it requires local path-dependent growth
#   within a shared concern space."

# %% [markdown]
# ## Starter Checks
#
# These are just placeholders to keep the notebook executable while making it
# easy to inspect the basic ingredients before building the first simulation.


# %%
rng = np.random.default_rng(CFG.random_seed)

latest_period = int(count_panel["window_end"].max())
latest_slice = count_panel[count_panel["window_end"] == latest_period].copy()
latest_budgets = latest_slice.groupby("actor")["K_it"].first().sort_index()

latest_support = (
    latest_slice.pivot(index="actor", columns="topic", values="active_rca")
    .fillna(False)
    .astype(int)
    .reindex(index=sorted(latest_budgets.index), columns=topics_order, fill_value=0)
)

null_draw = sample_system_count_null(
    rng=rng,
    budgets=latest_budgets,
    topics_order=topics_order,
)

model_suite = run_model_suite(
    count_panel=count_panel,
    phi_matrix=phi.to_numpy(dtype=float),
    topics_order=topics_order,
    actors_order=actors_order,
    random_seed=CFG.random_seed,
)
budget_table = model_suite["budget_table"]
alpha_hat = model_suite["alpha_hat"]
alpha_support_hat = model_suite["alpha_support_hat"]
alpha_entry_hat = model_suite["alpha_entry_hat"]
alpha_retention_hat = model_suite["alpha_retention_hat"]
param_estimates = model_suite["param_estimates"]
support_param_estimates = model_suite["support_param_estimates"]
entry_param_estimates = model_suite["entry_param_estimates"]
retention_param_estimates = model_suite["retention_param_estimates"]
split_calibration = model_suite["split_calibration"]
sim_history = model_suite["sim_history"]
history_summary = model_suite["history_summary"]
sim_history_two_stage = model_suite["sim_history_two_stage"]
history_summary_two_stage = model_suite["history_summary_two_stage"]
sim_history_split = model_suite["sim_history_split"]
history_summary_split = model_suite["history_summary_split"]
history_fit_summary = model_suite["history_fit_summary"]
process_uncertainty_history, process_uncertainty_entry = (
    summarize_process_uncertainty_from_repeated_simulation(
        count_panel=count_panel,
        model_suite=model_suite,
        phi_matrix=phi.to_numpy(dtype=float),
        topics_order=topics_order,
        n_reps=CFG.process_uncertainty_reps,
        base_seed=CFG.random_seed,
    )
)
split_param_uncertainty = build_split_parameter_uncertainty_table(
    param_estimates=param_estimates,
    entry_param_estimates=entry_param_estimates,
    retention_param_estimates=retention_param_estimates,
)

# %% [markdown]
# ## Direct Specialization Diagnostics
#
# The two-stage result suggests that actors do not simply diffuse across all
# nearby topics. To probe that more directly, we compute three diagnostics.
#
# 1. Centroid stability through time
#    - if actors occupy persistent niches, their portfolio centroid should not
#      drift too far between consecutive periods
#
# 2. Concentration versus a budget-only null
#    - if specialization matters, observed portfolios should be more
#      concentrated and lower-entropy than a null that only preserves budgets
#
# 3. Phi-based proximity of new topic entries to the prior support
#    - if actors expand locally in the native concern space, new topics should
#      have high proximity to previously held topics under `Phi`
#
# These diagnostics do not prove strategic intent on their own. They do tell us
# whether the observed system looks more like persistent niche occupation than a
# smooth diffusion process.

# %%
observed_history = build_observed_history_like(count_panel)
topic_coords = embed_topics_from_phi(phi.to_numpy(dtype=float), topics_order)

centroid_shifts_observed = compute_actor_centroid_shifts(observed_history, topic_coords)
centroid_shifts_one_stage = compute_actor_centroid_shifts(sim_history, topic_coords)
centroid_shifts_two_stage = compute_actor_centroid_shifts(
    sim_history_two_stage, topic_coords
)
centroid_shifts_split = compute_actor_centroid_shifts(sim_history_split, topic_coords)
centroid_shifts = pd.concat(
    [
        centroid_shifts_observed.assign(model="observed"),
        centroid_shifts_one_stage.assign(model="one_stage"),
        centroid_shifts_two_stage.assign(model="two_stage"),
        centroid_shifts_split.assign(model="split_support"),
    ],
    ignore_index=True,
)

entry_phi_observed = compute_new_entry_phi_proximity(
    observed_history, phi.to_numpy(dtype=float), topics_order
)
entry_phi_one_stage = compute_new_entry_phi_proximity(
    sim_history, phi.to_numpy(dtype=float), topics_order
)
entry_phi_two_stage = compute_new_entry_phi_proximity(
    sim_history_two_stage, phi.to_numpy(dtype=float), topics_order
)
entry_phi_split = compute_new_entry_phi_proximity(
    sim_history_split, phi.to_numpy(dtype=float), topics_order
)
entry_phi = pd.concat(
    [
        entry_phi_observed.assign(model="observed"),
        entry_phi_one_stage.assign(model="one_stage"),
        entry_phi_two_stage.assign(model="two_stage"),
        entry_phi_split.assign(model="split_support"),
    ],
    ignore_index=True,
)

concentration_latest = compare_latest_period_concentration_to_null(
    latest_slice=latest_slice, null_draw=null_draw, topics_order=topics_order
)

specialization_summary = {
    "mean_centroid_shift_observed": float(
        centroid_shifts.loc[centroid_shifts["model"] == "observed", "centroid_shift"].mean()
    ),
    "mean_centroid_shift_one_stage": float(
        centroid_shifts.loc[centroid_shifts["model"] == "one_stage", "centroid_shift"].mean()
    ),
    "mean_centroid_shift_two_stage": float(
        centroid_shifts.loc[centroid_shifts["model"] == "two_stage", "centroid_shift"].mean()
    ),
    "mean_centroid_shift_split": float(
        centroid_shifts.loc[
            centroid_shifts["model"] == "split_support", "centroid_shift"
        ].mean()
    ),
    "mean_entry_phi_observed": float(
        entry_phi.loc[
            entry_phi["model"] == "observed", "mean_phi_to_prev_support"
        ].mean()
    ),
    "mean_entry_phi_one_stage": float(
        entry_phi.loc[
            entry_phi["model"] == "one_stage", "mean_phi_to_prev_support"
        ].mean()
    ),
    "mean_entry_phi_two_stage": float(
        entry_phi.loc[
            entry_phi["model"] == "two_stage", "mean_phi_to_prev_support"
        ].mean()
    ),
    "mean_entry_phi_split": float(
        entry_phi.loc[
            entry_phi["model"] == "split_support", "mean_phi_to_prev_support"
        ].mean()
    ),
    "max_entry_phi_observed": float(
        entry_phi.loc[
            entry_phi["model"] == "observed", "max_phi_to_prev_support"
        ].mean()
    ),
    "max_entry_phi_one_stage": float(
        entry_phi.loc[
            entry_phi["model"] == "one_stage", "max_phi_to_prev_support"
        ].mean()
    ),
    "max_entry_phi_two_stage": float(
        entry_phi.loc[
            entry_phi["model"] == "two_stage", "max_phi_to_prev_support"
        ].mean()
    ),
    "max_entry_phi_split": float(
        entry_phi.loc[
            entry_phi["model"] == "split_support", "max_phi_to_prev_support"
        ].mean()
    ),
    "mean_entry_phi_rank_observed": float(
        entry_phi.loc[
            entry_phi["model"] == "observed", "mean_phi_rank_pct"
        ].mean()
    ),
    "mean_entry_phi_rank_one_stage": float(
        entry_phi.loc[
            entry_phi["model"] == "one_stage", "mean_phi_rank_pct"
        ].mean()
    ),
    "mean_entry_phi_rank_two_stage": float(
        entry_phi.loc[
            entry_phi["model"] == "two_stage", "mean_phi_rank_pct"
        ].mean()
    ),
    "mean_entry_phi_rank_split": float(
        entry_phi.loc[
            entry_phi["model"] == "split_support", "mean_phi_rank_pct"
        ].mean()
    ),
    "mean_herfindahl_latest_observed": float(
        concentration_latest.loc[
            concentration_latest["label"] == "observed_latest", "herfindahl"
        ].mean()
    ),
    "mean_herfindahl_latest_budget_only_null": float(
        concentration_latest.loc[
            concentration_latest["label"] == "budget_only_null", "herfindahl"
        ].mean()
    ),
    "mean_entropy_latest_observed": float(
        concentration_latest.loc[
            concentration_latest["label"] == "observed_latest", "entropy_norm"
        ].mean()
    ),
    "mean_entropy_latest_budget_only_null": float(
        concentration_latest.loc[
            concentration_latest["label"] == "budget_only_null", "entropy_norm"
        ].mean()
    ),
}

specialization_summary

# %% [markdown]
# ## Reading The Specialization Checks
#
# A useful way to read the diagnostics is:
#
# - lower centroid drift: actors remain in more stable regions over time
# - higher Herfindahl and lower entropy than the null: portfolios are more
#   selective than budget constraints alone would imply
# - higher entry `Phi` and higher entry `Phi` rank: new topics are closer to the
#   prior support in the native concern space
#
# If the observed ATS shows this pattern more strongly than the one-stage model,
# then "local growth in a shared space" is not the whole story. The data would
# be more consistent with persistent selective niches inside that shared space.

# %% [markdown]
# ## Density-Robust Validation
#
# A reasonable overfitting concern is that later, denser ATS years may make the
# split model look stronger than it really is. To probe that, we thin the raw
# submission records *within each year*, rebuild the actor-topic panels, refit
# the models, and ask whether the split model still outperforms the simpler
# alternatives.
#
# Important design choice:
#
# - we thin the raw submissions rather than dropping actor-topic rows from the
#   panel, because RCA support and budgets must be recomputed coherently
# - we keep the pooled `Phi` fixed to the full-data estimate so this check tests
#   model robustness rather than instability in space estimation

# %%
subsample_validation, subsample_validation_summary = run_time_balanced_subsample_validation(
    submitted_df=submitted_df,
    time_col=time_col,
    periods=periods,
    all_members_raw=all_members_raw,
    all_topics_raw=all_topics_raw,
    topics_order=topics_order,
    actors_order=actors_order,
    phi_matrix=phi.to_numpy(dtype=float),
    rca_threshold=CFG.rca_threshold,
    keep_fractions=(0.4, 0.6, 0.8),
    n_reps=4,
    base_seed=CFG.random_seed,
    verbose=True,
)

subsample_validation_summary

starter_summary = {
    "latest_period": latest_period,
    "n_actor_budgets": int(len(latest_budgets)),
    "mean_budget": float(latest_budgets.mean()),
    "mean_observed_support_size": float(latest_support.sum(axis=1).mean()),
    "mean_null_allocation_per_topic": float(
        null_draw.groupby("topic")["x_ijt_sim"].sum().mean()
    ),
    "n_topics": int(len(topics_order)),
    "n_actors": int(len(actors_order)),
    "n_periods": int(len(periods)),
    **param_estimates,
    **support_param_estimates,
    **entry_param_estimates,
    **retention_param_estimates,
    **history_fit_summary,
    **specialization_summary,
    "process_uncertainty_reps": int(CFG.process_uncertainty_reps),
    "split_mean_active_topics_q05_avg": float(
        process_uncertainty_history.loc[
            process_uncertainty_history["model"] == "split_support",
            "mean_active_topics_q05",
        ].mean()
    ),
    "split_mean_active_topics_q95_avg": float(
        process_uncertainty_history.loc[
            process_uncertainty_history["model"] == "split_support",
            "mean_active_topics_q95",
        ].mean()
    ),
    "split_mean_topic_popularity_q05_avg": float(
        process_uncertainty_history.loc[
            process_uncertainty_history["model"] == "split_support",
            "mean_topic_popularity_q05",
        ].mean()
    ),
    "split_mean_topic_popularity_q95_avg": float(
        process_uncertainty_history.loc[
            process_uncertainty_history["model"] == "split_support",
            "mean_topic_popularity_q95",
        ].mean()
    ),
    "split_mean_entry_phi_rank_q05": float(
        process_uncertainty_entry.loc[
            process_uncertainty_entry["model"] == "split_support",
            "mean_entry_phi_rank_q05",
        ].iloc[0]
    ),
    "split_mean_entry_phi_rank_q95": float(
        process_uncertainty_entry.loc[
            process_uncertainty_entry["model"] == "split_support",
            "mean_entry_phi_rank_q95",
        ].iloc[0]
    ),
    "subsample_min_mean_corr_active_split": float(
        subsample_validation_summary["mean_corr_active_split"].min()
    ),
    "subsample_min_mean_corr_pop_split": float(
        subsample_validation_summary["mean_corr_pop_split"].min()
    ),
    "subsample_max_abs_entry_phi_rank_gap_split": float(
        (
            subsample_validation_summary["mean_entry_phi_rank_split"]
            - subsample_validation_summary["mean_entry_phi_rank_observed"]
        )
        .abs()
        .max()
    ),
    "subsample_split_better_than_two_stage_active_all": bool(
        (
            subsample_validation_summary["mean_corr_active_split"]
            > subsample_validation_summary["mean_corr_active_two_stage"]
        ).all()
    ),
}

starter_summary

# %% [markdown]
# ## Closing Note
#
# This notebook is intentionally only a starter. The next real step is to choose
# one concrete reduced-form model and one concrete simulation target, then test
# whether the observed macrostructure is reproduced under that model.

# %% [markdown]
# ## Script Outputs
#
# When run as a plain Python script, this notebook writes:
#
# - `output/actor_topic_modeling_starter_summary.json`
# - `output/actor_topic_modeling_starter_null_draw_latest.csv`
# - `output/actor_topic_modeling_starter_param_estimates.json`
# - `output/actor_topic_modeling_starter_support_param_estimates.json`
# - `output/actor_topic_modeling_starter_entry_param_estimates.json`
# - `output/actor_topic_modeling_starter_retention_param_estimates.json`
# - `output/actor_topic_modeling_starter_history_summary.csv`
# - `output/actor_topic_modeling_starter_sim_history.csv`
# - `output/actor_topic_modeling_starter_two_stage_history_summary.csv`
# - `output/actor_topic_modeling_starter_two_stage_sim_history.csv`
# - `output/actor_topic_modeling_starter_split_history_summary.csv`
# - `output/actor_topic_modeling_starter_split_sim_history.csv`
# - `output/actor_topic_modeling_starter_centroid_shifts.csv`
# - `output/actor_topic_modeling_starter_entry_phi_proximity.csv`
# - `output/actor_topic_modeling_starter_concentration_latest.csv`
# - `output/actor_topic_modeling_starter_process_uncertainty_history.csv`
# - `output/actor_topic_modeling_starter_process_uncertainty_entry.csv`
# - `output/actor_topic_modeling_starter_split_param_uncertainty.csv`
# - `output/actor_topic_modeling_starter_subsample_validation.csv`
# - `output/actor_topic_modeling_starter_subsample_validation_summary.csv`
#
# and prints a short console summary.


# %%
CFG.out_dir.mkdir(parents=True, exist_ok=True)
CFG.out_summary_json.write_text(
    pd.Series({**system_summary.to_dict(), **starter_summary}).to_json(indent=2),
    encoding="utf-8",
)
null_draw.to_csv(CFG.out_null_draw_csv, index=False)
history_summary.to_csv(CFG.out_history_summary_csv, index=False)
sim_history.to_csv(CFG.out_sim_history_csv, index=False)
history_summary_two_stage.to_csv(CFG.out_two_stage_history_summary_csv, index=False)
sim_history_two_stage.to_csv(CFG.out_two_stage_sim_history_csv, index=False)
history_summary_split.to_csv(CFG.out_split_history_summary_csv, index=False)
sim_history_split.to_csv(CFG.out_split_sim_history_csv, index=False)
centroid_shifts.to_csv(CFG.out_centroid_shift_csv, index=False)
entry_phi.to_csv(CFG.out_entry_phi_csv, index=False)
concentration_latest.to_csv(CFG.out_concentration_latest_csv, index=False)
process_uncertainty_history.to_csv(CFG.out_process_uncertainty_history_csv, index=False)
process_uncertainty_entry.to_csv(CFG.out_process_uncertainty_entry_csv, index=False)
split_param_uncertainty.to_csv(CFG.out_split_param_uncertainty_csv, index=False)
subsample_validation.to_csv(CFG.out_subsample_validation_csv, index=False)
subsample_validation_summary.to_csv(CFG.out_subsample_validation_summary_csv, index=False)
CFG.out_param_json.write_text(
    pd.Series(param_estimates).to_json(indent=2),
    encoding="utf-8",
)
CFG.out_support_param_json.write_text(
    pd.Series(support_param_estimates).to_json(indent=2),
    encoding="utf-8",
)
CFG.out_entry_param_json.write_text(
    pd.Series(entry_param_estimates).to_json(indent=2),
    encoding="utf-8",
)
CFG.out_retention_param_json.write_text(
    pd.Series(retention_param_estimates).to_json(indent=2),
    encoding="utf-8",
)

print("Wrote", CFG.out_summary_json)
print("Wrote", CFG.out_null_draw_csv)
print("Wrote", CFG.out_history_summary_csv)
print("Wrote", CFG.out_sim_history_csv)
print("Wrote", CFG.out_param_json)
print("Wrote", CFG.out_support_param_json)
print("Wrote", CFG.out_entry_param_json)
print("Wrote", CFG.out_retention_param_json)
print("Wrote", CFG.out_two_stage_history_summary_csv)
print("Wrote", CFG.out_two_stage_sim_history_csv)
print("Wrote", CFG.out_split_history_summary_csv)
print("Wrote", CFG.out_split_sim_history_csv)
print("Wrote", CFG.out_centroid_shift_csv)
print("Wrote", CFG.out_entry_phi_csv)
print("Wrote", CFG.out_concentration_latest_csv)
print("Wrote", CFG.out_process_uncertainty_history_csv)
print("Wrote", CFG.out_process_uncertainty_entry_csv)
print("Wrote", CFG.out_split_param_uncertainty_csv)
print("Wrote", CFG.out_subsample_validation_csv)
print("Wrote", CFG.out_subsample_validation_summary_csv)
print(
    "Starter summary:",
    {
        "n_topics": starter_summary["n_topics"],
        "n_actors": starter_summary["n_actors"],
        "n_periods": starter_summary["n_periods"],
        "latest_period": starter_summary["latest_period"],
        "mean_budget": round(starter_summary["mean_budget"], 3),
        "mean_observed_support_size": round(
            starter_summary["mean_observed_support_size"], 3
        ),
        "rho_mle": round(starter_summary["rho_mle"], 3),
        "beta_mle": round(starter_summary["beta_mle"], 3),
        "gamma_mle": round(starter_summary["gamma_mle"], 3),
        "rho_support_mle": round(starter_summary["rho_support_mle"], 3),
        "beta_support_mle": round(starter_summary["beta_support_mle"], 3),
        "beta_entry_mle": round(starter_summary["beta_entry_mle"], 3),
        "lambda_retention_mle": round(
            starter_summary["lambda_retention_mle"], 3
        ),
        "entry_shift_best": round(starter_summary["entry_shift_best"], 3),
        "retention_shift_best": round(
            starter_summary["retention_shift_best"], 3
        ),
        "corr_mean_active_topics": round(
            starter_summary["corr_mean_active_topics"], 3
        ),
        "corr_mean_active_topics_two_stage": round(
            starter_summary["corr_mean_active_topics_two_stage"], 3
        ),
        "corr_mean_active_topics_split": round(
            starter_summary["corr_mean_active_topics_split"], 3
        ),
        "corr_mean_topic_popularity": round(
            starter_summary["corr_mean_topic_popularity"], 3
        ),
        "corr_mean_topic_popularity_two_stage": round(
            starter_summary["corr_mean_topic_popularity_two_stage"], 3
        ),
        "corr_mean_topic_popularity_split": round(
            starter_summary["corr_mean_topic_popularity_split"], 3
        ),
        "mean_centroid_shift_observed": round(
            starter_summary["mean_centroid_shift_observed"], 3
        ),
        "mean_centroid_shift_two_stage": round(
            starter_summary["mean_centroid_shift_two_stage"], 3
        ),
        "mean_centroid_shift_split": round(
            starter_summary["mean_centroid_shift_split"], 3
        ),
        "mean_entry_phi_observed": round(
            starter_summary["mean_entry_phi_observed"], 3
        ),
        "mean_entry_phi_two_stage": round(
            starter_summary["mean_entry_phi_two_stage"], 3
        ),
        "mean_entry_phi_split": round(
            starter_summary["mean_entry_phi_split"], 3
        ),
        "mean_entry_phi_rank_observed": round(
            starter_summary["mean_entry_phi_rank_observed"], 3
        ),
        "mean_entry_phi_rank_two_stage": round(
            starter_summary["mean_entry_phi_rank_two_stage"], 3
        ),
        "mean_entry_phi_rank_split": round(
            starter_summary["mean_entry_phi_rank_split"], 3
        ),
        "subsample_min_mean_corr_active_split": round(
            starter_summary["subsample_min_mean_corr_active_split"], 3
        ),
        "subsample_min_mean_corr_pop_split": round(
            starter_summary["subsample_min_mean_corr_pop_split"], 3
        ),
    },
)
