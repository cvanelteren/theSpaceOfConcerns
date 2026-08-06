#!/usr/bin/env python
"""Test the manuscript's central negative claim: the space is not semantic similarity.

Four places in the paper assert that proximity in the space of concerns
"reflects recurrent joint engagement rather than semantic similarity". That is
the load-bearing justification for estimating the space from actor behaviour at
all: if topic proximity were recoverable from the topic labels, the whole
construction would be an expensive way to compute string similarity. Until now
the claim was asserted and never tested.

This builds both matrices over the same 45 Secretariat topic labels:

  phi(i, j)  co-specialization proximity -- the paper's space
  s(i, j)    TF-IDF cosine over the label text

and compares them. A lexical baseline rather than a neural embedding is
deliberate. It needs no model download, so anyone cloning the repository gets
identical numbers, and it tests the objection a reviewer actually raises: that
similar-sounding topics are adjacent because they sound similar. Word and
character n-grams are both reported, since short institutional labels
("Marine Acoustics", "Marine Protected Areas") share substrings that a
word-level vectorizer alone would miss.

Significance uses a Mantel permutation. Rows and columns of one matrix are
shuffled jointly, which preserves the dependence among pairwise entries that
makes an ordinary correlation p-value invalid here.

Usage::

    micromamba run -n ultraplot-dev python scripts/semantic_baseline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import compute_product_space, get_rca, load_data  # noqa: E402

OUT_JSON = ROOT / "output" / "semantic_baseline.json"
OUT_CSV = ROOT / "output" / "semantic_baseline_pairs.csv"

# Same candidates and the same "all"/"other" exclusion as
# fig01_space_of_concerns_topology.py, so this compares against the geometry the
# figure actually draws rather than a near-miss reconstruction of it.
DATA_CANDIDATES = [
    ROOT / "antarctic-database-go/data/processed/document-summary.parquet",
    ROOT.parent / "antarctic-database-go/data/processed/document-summary.parquet",
]
EXCLUDED_TOPICS = {"all", "other"}

N_PERMUTATIONS = 10_000
SEED = 20260806


def load_phi() -> pd.DataFrame:
    path = next((p for p in DATA_CANDIDATES if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "Raw submission table not found. See README: it is an external "
            "dependency archived in the Zenodo record. Looked for "
            + ", ".join(str(p) for p in DATA_CANDIDATES)
        )
    counts, _, _, _ = load_data(str(path))
    keep = [t for t in counts.index if str(t).strip().lower() not in EXCLUDED_TOPICS]
    return compute_product_space(get_rca(counts.loc[keep]))


def label_similarity(labels: list[str], analyzer: str, ngram: tuple[int, int]):
    vec = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram,
        lowercase=True,
        sublinear_tf=True,
        stop_words="english" if analyzer == "word" else None,
    )
    X = vec.fit_transform(labels).toarray()
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-12)
    return X @ X.T


def mantel(a: np.ndarray, b: np.ndarray, n_perm: int, seed: int) -> dict:
    """Correlate two symmetric matrices with a joint row/column shuffle null."""
    iu = np.triu_indices_from(a, k=1)
    x, y = a[iu], b[iu]
    r = float(stats.pearsonr(x, y).statistic)
    rho = float(stats.spearmanr(x, y).statistic)

    rng = np.random.default_rng(seed)
    n = a.shape[0]
    null = np.empty(n_perm)
    for k in range(n_perm):
        p = rng.permutation(n)
        null[k] = stats.pearsonr(a[np.ix_(p, p)][iu], y).statistic
    return {
        "pearson_r": r,
        "spearman_rho": rho,
        "r_squared": r * r,
        "mantel_p": float((np.abs(null) >= abs(r)).mean()),
        "null_mean_r": float(null.mean()),
        "null_sd_r": float(null.std()),
        "n_pairs": int(x.size),
    }


def main() -> int:
    phi = load_phi()
    labels = [str(t) for t in phi.index]
    P = phi.to_numpy(dtype=float)
    np.fill_diagonal(P, 0.0)
    print(f"topics: {len(labels)}   pairs: {len(labels) * (len(labels) - 1) // 2}")

    results = {}
    for name, (analyzer, ngram) in {
        "word_1_2": ("word", (1, 2)),
        "char_3_5": ("char_wb", (3, 5)),
    }.items():
        S = label_similarity(labels, analyzer, ngram)
        np.fill_diagonal(S, 0.0)
        results[name] = mantel(P, S, N_PERMUTATIONS, SEED)
        r = results[name]
        print(
            f"  {name:9s} r={r['pearson_r']:+.3f}  R2={r['r_squared']:.3f}  "
            f"rho={r['spearman_rho']:+.3f}  Mantel p={r['mantel_p']:.4f}"
        )

    # Concrete cases, which is what a caption can use: pairs the space calls
    # close that the labels call unrelated, and the reverse.
    S = label_similarity(labels, "char_wb", (3, 5))
    np.fill_diagonal(S, 0.0)
    iu = np.triu_indices_from(P, k=1)
    pairs = pd.DataFrame({
        "topic_i": [labels[i] for i in iu[0]],
        "topic_j": [labels[j] for j in iu[1]],
        "phi": P[iu],
        "label_cosine": S[iu],
    })
    pairs["divergence"] = pairs["phi"].rank(pct=True) - pairs["label_cosine"].rank(pct=True)
    pairs = pairs.sort_values("divergence", ascending=False)

    print("\nadjacent in the space, unrelated by label:")
    for _, r in pairs.head(6).iterrows():
        print(f"  phi={r['phi']:.3f} cos={r['label_cosine']:.3f}  {r['topic_i']} <-> {r['topic_j']}")
    print("\nsimilar labels, distant in the space:")
    sim = pairs[pairs.label_cosine > 0].tail(6)
    for _, r in sim.iterrows():
        print(f"  phi={r['phi']:.3f} cos={r['label_cosine']:.3f}  {r['topic_i']} <-> {r['topic_j']}")

    OUT_JSON.write_text(json.dumps({
        "n_topics": len(labels), "n_permutations": N_PERMUTATIONS,
        "seed": SEED, "variants": results}, indent=2))
    pairs.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_JSON.relative_to(ROOT)}, {OUT_CSV.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
