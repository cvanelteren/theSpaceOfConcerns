# Directed transitions and formal outputs

The directed maps are learned from rolling five-meeting actor portfolios. For an output at ATCM t, only portfolio transitions completed before that meeting are used. Exact concern matches are kept separate because entry into a concern is observed only while that concern is not already held.

## Does direction predict later portfolio movement?

Each transition is scored using maps estimated from earlier transitions. Values below are actor-period-balanced probabilities that a newly specialized concern ranks above another available concern. Intervals resample complete actor histories.

| Period | Score | Probability | 95% CI | Actor-periods |
|---|---|---:|---:|---:|
| all validated meetings | directed rate | 0.537 | [0.518, 0.556] | 862 |
| all validated meetings | directed ridge | 0.569 | [0.553, 0.585] | 862 |
| all validated meetings | prior popularity | 0.729 | [0.704, 0.753] | 862 |
| all validated meetings | symmetric proximity | 0.648 | [0.628, 0.669] | 862 |
| verified lineage era atcm16 onward | directed rate | 0.538 | [0.519, 0.557] | 846 |
| verified lineage era atcm16 onward | directed ridge | 0.568 | [0.552, 0.586] | 846 |
| verified lineage era atcm16 onward | prior popularity | 0.729 | [0.705, 0.753] | 846 |
| verified lineage era atcm16 onward | symmetric proximity | 0.649 | [0.629, 0.669] | 846 |

## Paper-ranking results

| Comparison | Probability linked paper ranks higher | 95% CI | Outputs |
|---|---:|---:|---:|
| symmetric full | 0.660 | [0.601, 0.721] | 89 |
| exact only | 0.639 | [0.595, 0.688] | 89 |
| symmetric nearby only | 0.535 | [0.467, 0.602] | 64 |
| directed rate full | 0.656 | [0.591, 0.720] | 89 |
| directed rate nearby only | 0.536 | [0.467, 0.604] | 64 |
| directed ridge full | 0.616 | [0.547, 0.684] | 89 |
| directed ridge nearby only | 0.475 | [0.406, 0.545] | 64 |
| discussion directed rate full | 0.536 | [0.479, 0.592] | 79 |

## Conditional comparison

The models below include exact concern alignment, title overlap, and Working Paper status. The displayed rows show the symmetric or directed off-label term when both are allowed to compete.

| Model | Term | Odds ratio | 95% CI |
|---|---|---:|---:|
| exact symmetric directed rate | related concern proximity | 1.171 | [0.886, 1.548] |
| exact symmetric directed rate | directed rate rank 10 | 0.953 | [0.744, 1.220] |
| exact symmetric directed ridge | related concern proximity | 1.186 | [0.912, 1.543] |
| exact symmetric directed ridge | directed ridge rank | 0.900 | [0.718, 1.127] |

## Map diagnostics

The primary empirical-Bayes map is built from 64,559 actor--target transition opportunities containing 1,724 entries. In the latest map, the median directed pair has 150 observed opportunities; 0.8% have fewer than ten. Forward and reverse scores correlate at 0.054 for the shrunk rate and 0.144 for the joint ridge map. Lower correlations indicate that direction adds information beyond the symmetric space.

The transparent rate can still credit several concerns held together for the same transition. The joint ridge map is the direct test of whether one of those concerns carries more conditional predictive information, but its coefficients are regularized and should not be interpreted causally.
