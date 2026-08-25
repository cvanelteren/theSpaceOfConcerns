#!/usr/bin/env python3
"""Merge independent human codings and score the output-title classifier.

At least two complete, independently produced human exports are required.
Unanimous primary labels are accepted as consensus.  Any disagreement is sent
to a blind adjudication queue; classifier accuracy is withheld until those
rows have been adjudicated.  Automated or model-consensus files are rejected.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_DIR = (
    ROOT / "output" / "category_treatment_comparison" / "human_output_audit"
)
DEFAULT_SAMPLE = DEFAULT_AUDIT_DIR / "output_topic_audit_sample_blind.csv"
DEFAULT_KEY = DEFAULT_AUDIT_DIR / "output_topic_audit_model_key.csv"
DEFAULT_MANIFEST = DEFAULT_AUDIT_DIR / "output_topic_audit_manifest.json"
DEFAULT_PROBABILITIES = (
    ROOT
    / "output"
    / "category_treatment_comparison"
    / "fractional_multilabel"
    / "outcome_topic_probabilities.csv"
)
SCHEMA = "ats-output-topic-audit-v1"


def wilson_interval(successes: int, trials: int, z: float = 1.959964) -> tuple[float, float]:
    if trials == 0:
        return math.nan, math.nan
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return center - half, center + half


def load_export(path: Path, sample_id: str, expected_items: set[str]) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"{path}: unsupported schema")
    if payload.get("sample_id") != sample_id:
        raise ValueError(f"{path}: belongs to a different audit sample")
    source = str(payload.get("coding_source", "")).strip().lower()
    if source != "independent_human":
        raise ValueError(
            f"{path}: rejected coding_source={source!r}. Automated/model consensus "
            "is not human validation."
        )
    if payload.get("blind_to_model_predictions") is not True:
        raise ValueError(f"{path}: missing blind-coding attestation")
    coder = str(payload.get("coder", "")).strip()
    if not coder:
        raise ValueError(f"{path}: missing coder identifier")
    codings = payload.get("codings")
    if not isinstance(codings, dict):
        raise ValueError(f"{path}: codings must be an object")
    observed = set(codings)
    missing = expected_items - observed
    extra = observed - expected_items
    if missing or extra:
        raise ValueError(
            f"{path}: incomplete or mismatched export "
            f"({len(missing)} missing, {len(extra)} extra)"
        )
    return payload


def validate_coding(coding: dict, official_topics: set[str], label: str) -> None:
    primary = str(coding.get("primary", "")).strip()
    secondary = str(coding.get("secondary", "")).strip()
    confidence = str(coding.get("confidence", "")).strip()
    if primary not in official_topics:
        raise ValueError(f"{label}: invalid primary concern {primary!r}")
    if secondary and secondary not in official_topics:
        raise ValueError(f"{label}: invalid secondary concern {secondary!r}")
    if secondary == primary:
        raise ValueError(f"{label}: primary and secondary concerns must differ")
    if confidence not in {"high", "medium", "low"}:
        raise ValueError(f"{label}: invalid confidence {confidence!r}")


def agreement_tables(
    exports: list[dict], item_ids: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    long_rows = []
    for payload in exports:
        for item_id in item_ids:
            coding = payload["codings"][item_id]
            long_rows.append(
                {
                    "coder": payload["coder"],
                    "item_id": item_id,
                    "primary": coding["primary"],
                    "secondary": coding.get("secondary", ""),
                    "confidence": coding["confidence"],
                    "notes": coding.get("notes", ""),
                }
            )
    long = pd.DataFrame(long_rows)
    wide = long.pivot(index="item_id", columns="coder", values="primary")
    pair_rows = []
    for left, right in combinations(wide.columns, 2):
        exact = wide[left].eq(wide[right])
        pair_rows.append(
            {
                "coder_left": left,
                "coder_right": right,
                "n": int(len(wide)),
                "primary_exact_agreement": float(exact.mean()),
                "cohen_kappa": float(cohen_kappa_score(wide[left], wide[right])),
            }
        )
    pairwise = pd.DataFrame(pair_rows)
    unanimous = wide.nunique(axis=1).eq(1)
    summary = {
        "n_coders": int(len(exports)),
        "n_items": int(len(item_ids)),
        "unanimous_primary_n": int(unanimous.sum()),
        "unanimous_primary_share": float(unanimous.mean()),
        "mean_pairwise_primary_agreement": float(
            pairwise["primary_exact_agreement"].mean()
        ),
        "mean_pairwise_cohen_kappa": float(pairwise["cohen_kappa"].mean()),
    }
    return long, pairwise, summary


def build_consensus_and_queue(
    sample: pd.DataFrame, long: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    consensus_rows = []
    queue_rows = []
    sample_by_id = sample.set_index("item_id")
    for item_id, group in long.groupby("item_id", sort=False):
        primary_values = group["primary"].tolist()
        secondary_values = group["secondary"].fillna("").tolist()
        base = sample_by_id.loc[item_id].to_dict()
        base["item_id"] = item_id
        if len(set(primary_values)) == 1:
            nonempty_secondary = [value for value in secondary_values if value]
            secondary = (
                nonempty_secondary[0]
                if nonempty_secondary and len(set(nonempty_secondary)) == 1
                else ""
            )
            consensus_rows.append(
                {
                    **base,
                    "adjudicated_primary_concern": primary_values[0],
                    "adjudicated_secondary_concern": secondary,
                    "label_source": "unanimous_independent_coders",
                    "adjudication_notes": "",
                }
            )
            continue
        queue_rows.append(
            {
                **base,
                "coder_primary_labels": " | ".join(primary_values),
                "coder_secondary_labels": " | ".join(value or "[none]" for value in secondary_values),
                "coder_confidences": " | ".join(group["confidence"].tolist()),
                "adjudicated_primary_concern": "",
                "adjudicated_secondary_concern": "",
                "adjudication_notes": "",
            }
        )
    return pd.DataFrame(consensus_rows), pd.DataFrame(queue_rows)


def load_adjudication(
    path: Path, queue: pd.DataFrame, official_topics: set[str]
) -> pd.DataFrame:
    adjudication = pd.read_csv(path, dtype=str).fillna("")
    required = {
        "item_id",
        "adjudicated_primary_concern",
        "adjudicated_secondary_concern",
    }
    missing_columns = required - set(adjudication.columns)
    if missing_columns:
        raise ValueError(f"Adjudication file lacks columns: {sorted(missing_columns)}")
    expected = set(queue["item_id"])
    observed = set(adjudication["item_id"])
    if expected != observed:
        raise ValueError(
            "Adjudication rows must exactly match the unresolved queue: "
            f"{len(expected - observed)} missing, {len(observed - expected)} extra"
        )
    rows = []
    queue_by_id = queue.set_index("item_id")
    for _, row in adjudication.iterrows():
        primary = row["adjudicated_primary_concern"].strip()
        secondary = row["adjudicated_secondary_concern"].strip()
        if primary not in official_topics:
            raise ValueError(
                f"Adjudication {row['item_id']}: invalid primary concern {primary!r}"
            )
        if secondary and secondary not in official_topics:
            raise ValueError(
                f"Adjudication {row['item_id']}: invalid secondary concern {secondary!r}"
            )
        if primary == secondary:
            raise ValueError(
                f"Adjudication {row['item_id']}: primary and secondary must differ"
            )
        base = queue_by_id.loc[row["item_id"]].to_dict()
        rows.append(
            {
                **base,
                "item_id": row["item_id"],
                "adjudicated_primary_concern": primary,
                "adjudicated_secondary_concern": secondary,
                "label_source": "independent_human_adjudication",
                "adjudication_notes": row.get("adjudication_notes", ""),
            }
        )
    return pd.DataFrame(rows)


def accuracy_row(group: pd.DataFrame, label: str, value: str) -> dict:
    n = len(group)
    metrics = {
        "top1_matches_primary": group["top1_matches_primary"],
        "top3_contains_primary": group["top3_contains_primary"],
        "top1_matches_primary_or_secondary": group["top1_matches_any_human_concern"],
    }
    row: dict[str, object] = {"stratum": label, "value": value, "n": n}
    for name, values in metrics.items():
        successes = int(values.sum())
        lower, upper = wilson_interval(successes, n)
        row[name] = successes / n
        row[f"{name}_ci_low"] = lower
        row[f"{name}_ci_high"] = upper
    return row


def score_classifier(labels: pd.DataFrame, key: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    duplicate_metadata = [
        column
        for column in ["instrument", "year", "meeting"]
        if column in labels.columns and column in key.columns
    ]
    model_key = key.drop(columns=duplicate_metadata)
    scored = labels.merge(
        model_key, on=["sample_id", "outcome_id"], validate="one_to_one"
    )
    scored["top1_matches_primary"] = scored["topic_top1"].eq(
        scored["adjudicated_primary_concern"]
    )
    scored["top3_contains_primary"] = scored.apply(
        lambda row: row["adjudicated_primary_concern"]
        in {row["topic_top1"], row["topic_top2"], row["topic_top3"]},
        axis=1,
    )
    scored["top1_matches_any_human_concern"] = scored.apply(
        lambda row: row["topic_top1"]
        in {
            row["adjudicated_primary_concern"],
            row["adjudicated_secondary_concern"],
        },
        axis=1,
    )
    rows = [accuracy_row(scored, "all", "all")]
    for column in ["instrument", "time_band", "confidence_band"]:
        for value, group in scored.groupby(column, sort=True, observed=True):
            rows.append(accuracy_row(group, column, str(value)))
    return scored, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exports", type=Path, nargs="+")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--probabilities", type=Path, default=DEFAULT_PROBABILITIES)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_AUDIT_DIR / "merged")
    args = parser.parse_args()

    if len(args.exports) < 2:
        parser.error("At least two independent human coder exports are required")
    manifest = json.loads(args.manifest.read_text())
    sample_id = manifest["sample_id"]
    sample = pd.read_csv(args.sample, dtype={"item_id": str})
    key = pd.read_csv(args.key)
    if set(sample["sample_id"].astype(str)) != {sample_id}:
        raise ValueError("Blind sample does not match its manifest")
    if set(key["sample_id"].astype(str)) != {sample_id}:
        raise ValueError("Model key does not match its manifest")
    official_topics = set(pd.read_csv(args.probabilities, usecols=["topic"])["topic"])
    if len(official_topics) != 45:
        raise ValueError(f"Expected 45 official concerns, found {len(official_topics)}")

    expected_items = set(sample["item_id"])
    exports = [load_export(path, sample_id, expected_items) for path in args.exports]
    coders = [payload["coder"] for payload in exports]
    if len(set(coders)) != len(coders):
        raise ValueError("Coder identifiers must be unique; duplicate exports are not independent")
    for payload in exports:
        for item_id, coding in payload["codings"].items():
            validate_coding(coding, official_topics, f"{payload['coder']} / {item_id}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    long, pairwise, agreement = agreement_tables(exports, sample["item_id"].tolist())
    long.to_csv(args.outdir / "human_codings_long.csv", index=False)
    pairwise.to_csv(args.outdir / "human_coder_pairwise_agreement.csv", index=False)
    (args.outdir / "human_coder_agreement.json").write_text(
        json.dumps(agreement, indent=2) + "\n"
    )

    consensus, queue = build_consensus_and_queue(sample, long)
    if not queue.empty:
        queue.to_csv(args.outdir / "output_topic_adjudication_queue.csv", index=False)
        if args.adjudication is None:
            print(json.dumps(agreement, indent=2))
            raise SystemExit(
                "Classifier accuracy withheld: independent coders disagreed on "
                f"{len(queue)} items. Complete output_topic_adjudication_queue.csv "
                "blind to the model, then rerun with --adjudication."
            )
        resolved = load_adjudication(args.adjudication, queue, official_topics)
        labels = pd.concat([consensus, resolved], ignore_index=True)
    else:
        labels = consensus

    if len(labels) != len(sample):
        raise AssertionError("Every sampled output must have one adjudicated label")
    labels.to_csv(args.outdir / "output_topic_adjudicated_labels.csv", index=False)
    scored, accuracy = score_classifier(labels, key)
    scored.to_csv(args.outdir / "output_topic_classifier_human_scored.csv", index=False)
    accuracy.to_csv(args.outdir / "output_topic_classifier_human_accuracy.csv", index=False)
    summary = {
        **agreement,
        "sample_id": sample_id,
        "n_adjudicated": int(len(labels)),
        "n_disagreements_requiring_adjudication": int(len(queue)),
        "human_validation": True,
        "automated_consensus_used_as_human_truth": False,
        "overall_accuracy": accuracy.iloc[0].to_dict(),
    }
    (args.outdir / "output_topic_classifier_human_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
