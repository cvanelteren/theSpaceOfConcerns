# Scientific adversarial audit

## Overall verdict

The manuscript's scientific spine survives, but two claims require narrower
language. The network is connected, weakly modular, and locally organized. Its
connectivity is mostly a consequence of actor portfolio breadth and concern
prevalence, while its aggregate local organization exceeds exact-margin nulls.
Documentary attention predicts Resolution categories in retrospective
rolling-origin tests, including a specification that uses only the preceding
meeting. The present data do not establish a contemporaneous pre-adoption
forecast, and they provide weaker evidence for the network increment by itself.
An independent closure review found no numerical or timing contradiction, but
it identified one major submission blocker: the 45-to-15 crosswalk has not yet
been validated by independent blinded coders.

## Claim-by-claim assessment

| Claim | Adversarial test | Verdict |
|---|---|---|
| The ATS forms one connected concern space | 1,000 Curveball nulls that preserve every actor's specialization breadth and every concern's number of holders | Descriptively correct but not distinctive. Every null network is connected. Observed positive pairs are 956, versus a null median of 945 and 95% interval of [930, 957]. |
| The space contains local organization | Re-optimize weighted modularity within every fixed-margin null graph; repeat after sparse-actor filters | Supported. Observed optimized Q is 0.0898, versus null median 0.0518 and 95% interval [0.0439, 0.0627], p=0.001. The excess survives every filter. |
| The seven displayed regions are meaningful | Hold observed labels fixed in null draws and compare within-region enrichment | Supporting evidence only. The displayed labels were selected from the observed graph. Use the re-optimized modularity result as the selection-aware test. |
| Individual edges have precise substantive meaning | Pair-specific fixed-margin intervals | Too strong. Of 990 observed proximities, 928 fall inside their pair-specific null intervals. Interpret aggregate local structure and avoid strong claims about most individual edges. |
| Actors add new specializations locally | Within-actor option model, popularity-weighted matched redraws, temporal maps, first-paper outcomes, leave-one-actor-out maps, actor bootstrap, and disjoint windows | Supported. The primary odds ratio is 0.817 per 0.1 farther, with interval [0.783, 0.853]. Realized shifts are nearer in 62.5% of matched comparisons, with actor-bootstrap interval [58.9%, 66.5%]. |
| Attention improves Resolution category forecasts | Same-meeting rolling origins, nested reselection, strictly lagged predictors, moving-block intervals, exact sign flips, and a future-attention falsification | Supported as a retrospective association. The strictly lagged combined model improves 15 of 19 meetings and changes the mean score by -0.0816, with interval [-0.1278, -0.0326]. No corresponding improvement appears for Measures or Decisions. |
| The forecast was operationally available before adoption | Audit query timestamps, payload fields, document headers, and a January 2024 category snapshot | Not established. Category-query responses were retrieved in February 2026 and contain no assignment timestamp. The paper files predate adoption, but the analysis is a retrospective nowcast. |
| Network weighting adds information beyond direct attention | Label permutations, generic-paper baseline, January 2024 single-label sensitivity, and crosswalk perturbations | Mixed. The current multi-label analysis favors network weighting, but the older single-label snapshot does not preserve the increment. Treat the network component as conditional evidence. |
| The 45-to-15 crosswalk does not determine the forecast | 50 isolated alternative assignments, 96 sampled joint mappings, and a post hoc failure stack | Stable to local ambiguity but not invariant. Every isolated and sampled mapping retains a negative point estimate. A coordinated 19-change stack erases the gain. |
| Measures follow a narrow formal record | Year-qualified citation audit keyed by type, number, and year | Supported as descriptive formal continuity. The audit does not establish causation, legal force, implementation, or entry into force. |

## Remaining submission checks

Two independent coders must complete the blinded 45-row crosswalk packet,
followed by blind adjudication and a rerun with the consensus mapping. A future
operational forecast would require frozen category metadata and model settings
before the meeting outputs are known. These checks affect the forecast claim;
they do not affect the topology or actor-locality results.

## Reproducibility map

- Topology null: `scripts/topology_fixed_margin_null.py`,
  `scripts/topology_curveball_mixing_check.py`, and
  `output/scientific_checks/topology_fixed_margin_summary.json`.
- Metadata timing: `scripts/audit_paper_metadata_timing.py`,
  `scripts/audit_snapshot_nowcast_sensitivity.py`, and
  `output/scientific_checks/paper_metadata_timing_report.md`.
- Forecast timing: `scripts/adversarial_forecast_checks.py` and
  `output/scientific_checks/forecast_adversarial_summary.json`.
- Crosswalk stress test: `scripts/check_crosswalk_uncertainty.py` and
  `output/scientific_checks/crosswalk_summary.json`.
- Blind crosswalk workflow: `tools/build_crosswalk_blind_validation.py`,
  `tools/merge_crosswalk_blind_validation.py`, and
  `output/scientific_checks/crosswalk_blind_validation_protocol.md`.

## Writing and build gate

The final mechanical grep returned zero em dashes, passive-voice patterns,
promotional constructions, throat-clearing openers, fancy verbs, content-free
openers, wordiness patterns, and precision-pair errors. The broader scans
returned 93 antithesis matches, 4 banned-word matches, 1 hype match, 20 weak-
qualifier matches, 7 `-wise` matches, and 22 exclamation-mark matches. Manual
inspection classified the antithesis and qualifier matches as factual scope
limits or comparisons, the four word matches as a required venue label and
official category names, the hype match as the substring in "this set to", the
seven `-wise` matches as the technical term "pairwise", and all exclamation
marks as LaTeX float or spacing syntax. No prose violation remains. The PDF
build reports 21 pages, zero undefined references, zero undefined citations,
and zero overfull boxes.
