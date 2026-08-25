# Do central paper concerns predict formal contribution?

All concern-network quantities are calculated from meetings preceding the focal ATCM. The unit is a candidate paper--output pair. Centrality describes the paper's labelled concern, not the output concern, because the latter does not vary among papers competing around the same output.

## Conditional estimates

Each row adds one structural quantity to exact alignment, neighbouring-concern proximity, title overlap, Working Paper status, and prior concern-level document volume. Odds ratios are per one standard deviation in the concern's within-meeting rank. Intervals below are model-based; the meeting-held-out results are the predictive check.

| Quantity | Odds ratio | 95% CI | BH-adjusted p |
|---|---:|---:|---:|
| topic complexity | 0.881 | 0.702--1.106 | 0.988 |
| betweenness | 0.897 | 0.731--1.099 | 0.988 |
| clustering | 0.911 | 0.731--1.136 | 0.988 |
| degree | 1.089 | 0.861--1.377 | 0.988 |
| holder breadth | 0.921 | 0.716--1.184 | 0.988 |
| ubiquity | 1.056 | 0.797--1.398 | 0.988 |
| eigenvector | 0.979 | 0.782--1.227 | 0.988 |
| pagerank | 0.996 | 0.794--1.249 | 0.988 |
| closeness | 1.004 | 0.804--1.255 | 0.988 |
| strength | 0.998 | 0.796--1.251 | 0.988 |

## Held-out meetings

The baseline contains exact alignment, neighbouring-concern proximity, title overlap, Working Paper status, and prior concern volume. Positive changes mean that adding the structural quantity ranked linked papers better in meetings excluded from model fitting.

| Added quantity | Baseline AUC | Augmented AUC | Change | 95% CI |
|---|---:|---:|---:|---:|
| clustering | 0.792 | 0.795 | +0.003 | -0.009--+0.017 |
| topic complexity | 0.782 | 0.785 | +0.003 | -0.010--+0.017 |
| eigenvector | 0.792 | 0.791 | -0.001 | -0.010--+0.008 |
| pagerank | 0.792 | 0.790 | -0.001 | -0.008--+0.006 |
| strength | 0.792 | 0.790 | -0.002 | -0.008--+0.005 |
| closeness | 0.792 | 0.789 | -0.003 | -0.012--+0.006 |
| degree | 0.792 | 0.786 | -0.006 | -0.014--+0.001 |
| holder breadth | 0.782 | 0.763 | -0.019 | -0.038---0.003 |
| betweenness | 0.792 | 0.767 | -0.025 | -0.047---0.005 |
| ubiquity | 0.792 | 0.767 | -0.025 | -0.041---0.010 |

## Which concerns receive formal output?

A second analysis uses concern--meeting rather than paper--output pairs. The full-sample model relates formal output at a meeting to centrality calculated before that meeting, alongside paper volume on the concern and its neighbours during the preceding three meetings, prior output, cumulative concern volume, and concern and meeting fixed effects.

| Quantity | Full-sample IRR | 95% CI | BH-adjusted p | Later-meeting rank change | 95% CI |
|---|---:|---:|---:|---:|---:|
| holder breadth | 0.786 | 0.641--0.963 | 0.200 | +0.003 | -0.007--+0.014 |
| topic complexity | 1.161 | 1.001--1.347 | 0.243 | +0.012 | +0.005--+0.019 |
| ubiquity | 1.222 | 0.927--1.610 | 0.515 | +0.016 | +0.008--+0.024 |
| closeness | 0.963 | 0.840--1.104 | 0.919 | -0.006 | -0.011---0.001 |
| pagerank | 0.967 | 0.846--1.106 | 0.919 | -0.004 | -0.007--+0.000 |
| strength | 0.972 | 0.844--1.119 | 0.919 | -0.005 | -0.008---0.001 |
| clustering | 1.023 | 0.910--1.149 | 0.919 | -0.006 | -0.013--+0.002 |
| eigenvector | 0.977 | 0.852--1.119 | 0.919 | -0.003 | -0.005---0.000 |
| betweenness | 0.993 | 0.904--1.091 | 0.919 | -0.004 | -0.008--+0.002 |
| degree | 0.992 | 0.856--1.151 | 0.919 | -0.002 | -0.005--+0.001 |

The later-meeting column is the change in the mean within-meeting Spearman correlation when the metric is added. Models are trained only on earlier meetings. This prevents an in-sample association from being described as predictive when it does not travel forward in time.

Topic complexity is not independent of the simpler count of actors specialized in a concern. Once that ubiquity benchmark is included, the complexity IRR is 1.155 (0.983--1.356) and its later-meeting rank gain is +0.000 (-0.006--+0.007).

## Interpretation rule

A useful structural predictor should have a stable conditional association and improve ranking in held-out meetings. A small p-value without a held-out gain is treated as description, not predictive evidence. Strength, closeness, eigenvector centrality, and PageRank are expected to overlap heavily; the correlation table should therefore be consulted before giving any one of them a distinct substantive interpretation.
