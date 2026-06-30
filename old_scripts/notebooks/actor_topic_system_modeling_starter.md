```python
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
```

# Actor-Topic System Modeling Starter

This notebook is a modeling note in Jupytext form. It is meant to bridge the
current empirical analysis to a more explicit theory of how interests are
organized in the ATS.

The core question is not:

- "What is the probability of the exact observed matrix?"

but rather:

- "Given `N` topics, `M` actors, and actor-specific budgets `K_it`, how unusual
  is the observed macrostructure relative to the set of feasible actor-topic
  systems?"

That shift matters because the exact observed microstate is usually the wrong
object. The more useful question is whether the ATS exhibits a rare
macrostructure:

- a shared issue arena
- differentiated actor positions
- local portfolio growth in topic space
- persistent regime structure

This notebook captures that reasoning and provides a practical scaffold for
building null models and generative simulations.


## Conceptual Setup

Let:

- `i = 1, ..., M` index actors
- `j = 1, ..., N` index topics
- `t` index years or rolling windows

Define the raw actor-topic allocation matrix:

- `X_t in R_+^(M x N)`
- `x_ijt >= 0` is the attention actor `i` allocates to topic `j` at time `t`

With actor budget constraint:

- `sum_j x_ijt = K_it`

A normalized portfolio is:

- `s_ijt = x_ijt / K_it`

The empirical object in the paper is often a transformed version of this raw
allocation:

- RCA / RPA
- active set (`RCA > 1`)
- concern-space proximity
- hazard of entry into new topics

For theory, the clean base object is still `X_t`.

```python
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import (
    compute_product_space,
    generate_interaction_matrix,
    get_rca,
    load_data,
    standardize_index_labels,
)
```

## Static Versus Dynamic Questions

There are two related but distinct theoretical questions.

Static:

- Why does the system exhibit this degree of overlap, concentration, and
  differentiation at one point in time?

Dynamic:

- Why do actors grow their portfolios locally from existing positions in the
  concern space rather than spreading uniformly?

The current paper already addresses the dynamic side empirically via the hazard
analysis. A practical theory program should still begin with the static
constrained ensemble, then add dynamic growth.


## Feasible Systems

There are two natural formalizations of the feasible set.

Binary support system:

- `S_ijt in {0, 1}`
- `sum_j S_ijt = k_it`

This is useful if the main object is whether an actor is active in a topic.

Weighted count system:

- `x_ijt in Z_+`
- `sum_j x_ijt = K_it`

This is useful if the main object is raw allocation across topics.

In either case the set of feasible systems is huge. The theoretical task is
therefore not to enumerate exact systems, but to define a probability measure
over the feasible set and ask whether the ATS looks typical or atypical under
that ensemble.


## Null Models To Compare

A clean sequence of nulls is:

1. Budget-only null
   - preserve actor budgets `K_it`
   - topics are otherwise exchangeable

2. Budget + topic salience null
   - preserve actor budgets
   - allow some topics to be intrinsically more attractive

3. Budget + topic salience + local growth null
   - preserve actor budgets
   - allow topic popularity / salience
   - actors expand into nearby topics in the concern space

4. Add actor heterogeneity
   - some actors are drawn to some regions more than others

The theoretical question becomes:

- Which is the first model that reproduces the observed macrostructure?


## Practical Empirical Program

In practice, we should not "just fit a model."

The more defensible workflow is:

1. estimate a local decision rule for topic entry / allocation
2. simulate synthetic ATS histories from that rule
3. compare observed and simulated systems on macro statistics
4. add complexity only when the simpler model fails

That means we need both:

- micro fit: does the model explain local choices?
- macro fit: does it reproduce regime structure, overlap, and local growth?


## Notebook Goals

This notebook sets up:

- a first actor-topic-time panel on raw counts
- a fixed topic-space matrix `Phi`
- simple null samplers
- a minimal dynamic allocation rule
- a checklist for simulation-based validation


## Configuration


```python
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
    window_years: int = 1
    rca_threshold: float = 1.0
    random_seed: int = 7
    out_dir: Path = Path("output")
    out_summary_json: Path = Path("output/actor_topic_modeling_starter_summary.json")
    out_null_draw_csv: Path = Path("output/actor_topic_modeling_starter_null_draw_latest.csv")


CFG = ModelingConfig()
```

## Helpers: Data Loading And Panel Construction


```python
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
    for candidate in ("meeting_year", "year"):
        if candidate in df.columns:
            return candidate
    raise KeyError("No meeting_year or year column found in source data.")


def sanitize_years(df: pd.DataFrame, year_col: str) -> pd.DataFrame:
    out = df.copy()
    out[year_col] = pd.to_numeric(out[year_col], errors="coerce")
    out = out.dropna(subset=[year_col]).copy()
    out[year_col] = out[year_col].astype(int)
    return out


def build_periods(year_min: int, year_max: int, window: int) -> list[tuple[int, int]]:
    return [(y - window + 1, y) for y in range(year_min + window - 1, year_max + 1)]


def build_window_interaction(
    submitted_df: pd.DataFrame,
    year_col: str,
    year_start: int,
    year_end: int,
    all_members_raw: set[str],
    all_topics_raw: set[str],
    topics_order: list[str],
    actors_order: list[str],
) -> pd.DataFrame:
    window_df = submitted_df[
        (submitted_df[year_col] >= int(year_start))
        & (submitted_df[year_col] <= int(year_end))
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
    year_col: str,
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
            year_col=year_col,
            year_start=int(start),
            year_end=int(end),
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
    year_col: str,
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
            year_col=year_col,
            year_start=int(start),
            year_end=int(end),
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
```


## Load ATS Data And Build A First Panel

This gives us the basic actor-topic-window panel on raw counts. That panel is a
clean starting point for both static nulls and dynamic simulations.


```python
counts_df, submitted_df, members_raw, topics_raw = load_data_with_fallback(CFG.data_paths)
year_col = detect_year_col(submitted_df)
submitted_df = sanitize_years(submitted_df, year_col)

topics_order = counts_df.index.tolist()
actors_order = counts_df.columns.tolist()
all_members_raw = set(members_raw)
all_topics_raw = set(topics_raw)

year_min = int(submitted_df[year_col].min())
year_max = int(submitted_df[year_col].max())
periods = build_periods(year_min, year_max, CFG.window_years)

count_panel = build_actor_topic_window_panel(
    submitted_df=submitted_df,
    year_col=year_col,
    periods=periods,
    all_members_raw=all_members_raw,
    all_topics_raw=all_topics_raw,
    topics_order=topics_order,
    actors_order=actors_order,
)

active_panel = build_rca_active_panel(
    submitted_df=submitted_df,
    year_col=year_col,
    periods=periods,
    all_members_raw=all_members_raw,
    all_topics_raw=all_topics_raw,
    topics_order=topics_order,
    actors_order=actors_order,
    rca_threshold=CFG.rca_threshold,
)

count_panel = count_panel.merge(
    active_panel[["topic", "actor", "window_end", "rca", "active_rca"]],
    on=["topic", "actor", "window_end"],
    how="left",
)

system_summary = pd.Series(
    {
        "n_topics": len(topics_order),
        "n_actors": len(actors_order),
        "year_min": year_min,
        "year_max": year_max,
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
```

## Build A Fixed Topic Space

A minimal dynamic model needs a common topic-space matrix `Phi`. The simplest
starting point is the pooled concern space from the full-history RCA matrix.


```python
overall_rca = get_rca(counts_df).reindex(
    index=topics_order, columns=actors_order, fill_value=0.0
)
phi = compute_product_space(overall_rca, threshold=CFG.rca_threshold).reindex(
    index=topics_order, columns=topics_order, fill_value=0.0
)

phi.shape
```

## First Nulls

These are not the final models. They are baseline generators for comparison.

Budget-only null:

- each actor has budget `K_it`
- topics are exchangeable

Budget + topic-salience null:

- same budgets
- topic weights allow some topics to be more attractive overall


```python
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
```


## Minimal Dynamic Allocation Rule

A stripped-down dynamic model close to the hazard logic is:

`Pr(i allocates to j at t) proportional exp(alpha_j + rho * s_ij,t-1 + beta * Fit_ijt + gamma * popularity_j,t-1 + a_ij)`

where:

- `alpha_j` is baseline topic attractiveness
- `rho` captures persistence on already-used topics
- `Fit_ijt` is proximity of topic `j` to actor `i`'s previous portfolio
- `gamma` captures popularity or crowding effects
- `a_ij` is an optional actor-topic affinity term

A natural local-fit term is:

`Fit_ijt = sum_l s_il,t-1 * Phi_lj`

This is the formal version of local portfolio growth in the concern space.


```python
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
```


## What We Would Estimate

The practical goal is not to estimate every parameter at once.

A sensible sequence is:

1. reduced-form entry model
   - topic intercepts
   - actor intercepts
   - locality / fit term
   - persistence term
   - topic popularity term

2. simulation from the estimated reduced-form model

3. macro validation:
   - does the simulated system recover the observed structure?

Only after that should we consider a richer latent-affinity model.


## Macro Statistics To Compare

The comparison target should be macrostructure, not exact microstate equality.

Candidate statistics:

- actor overlap distribution
- topic concentration distribution
- share of active topics per actor
- regime separation / regime persistence
- locality of new topic entry
- persistence / retention of portfolio positions


```python
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


def topic_popularity(X_binary: np.ndarray) -> np.ndarray:
    """How many actors are active in each topic."""
    X_binary = np.asarray(X_binary, dtype=int)
    return X_binary.sum(axis=0)
```


## Immediate Practical Roadmap For This Project

A good first implementation sequence in this repo would be:

1. Use the hazard panel machinery to estimate a reduced-form topic-entry model.
2. Preserve observed `K_it` budgets by actor-period.
3. Simulate synthetic histories under nested models:
   - budget only
   - budget + topic salience
   - budget + salience + local fit
4. Compare observed ATS to simulated histories on:
   - entry locality
   - overlap
   - concentration
   - regime persistence
5. Only if needed, add actor-topic latent affinity terms.

That would let us say something much sharper than:

- "actors are strategically differentiated"

We could instead say:

- "the observed ATS occupies a highly non-random region of the feasible
  actor-topic space, and reproducing it requires local path-dependent growth
  within a shared concern space."


## Starter Checks

These are just placeholders to keep the notebook executable while making it
easy to inspect the basic ingredients before building the first simulation.


```python
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
}

starter_summary
```

## Closing Note

This notebook is intentionally only a starter. The next real step is to choose
one concrete reduced-form model and one concrete simulation target, then test
whether the observed macrostructure is reproduced under that model.


## Script Outputs

When run as a plain Python script, this notebook writes:

- `output/actor_topic_modeling_starter_summary.json`
- `output/actor_topic_modeling_starter_null_draw_latest.csv`

and prints a short console summary.


```python
CFG.out_dir.mkdir(parents=True, exist_ok=True)
CFG.out_summary_json.write_text(
    pd.Series({**system_summary.to_dict(), **starter_summary}).to_json(indent=2),
    encoding="utf-8",
)
null_draw.to_csv(CFG.out_null_draw_csv, index=False)

print("Wrote", CFG.out_summary_json)
print("Wrote", CFG.out_null_draw_csv)
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
    },
)
```
