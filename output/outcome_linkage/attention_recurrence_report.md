# Does attention have to recur before formal output?

All windows use ordered ATCM meetings. The models compare each concern with itself over time, account for meeting-wide changes, and hold nearby-concern papers and earlier focal-concern outputs constant.

## The main decomposition

Across the preceding five meetings, moving from no focal paper to one is associated with a hard-output rate ratio of 2.79 [1.30, 6.00]. Among concerns already receiving direct attention, each doubling of focal papers is associated with 1.18 [1.03, 1.36] times as many outputs. Nearby-concern attention adds little (0.93 [0.79, 1.10]), while earlier focal-concern outputs remain associated with later output (1.38 [1.17, 1.61]).

The soft-assignment estimates point in the same direction: 1.64 [1.11, 2.43] for the first focal paper and 1.13 [1.01, 1.25] per additional doubling. Restricting hard assignments to high-confidence titles gives 6.43 [3.04, 13.59] and 1.19 [1.00, 1.42], respectively. In the binary occurrence model, the first focal paper is strongly associated with whether any output appears (4.78 [2.70, 8.45]), while extra paper volume is not (1.08 [0.94, 1.24]). After routine site-administration concerns are removed, the first-paper association remains (4.78 [2.60, 8.77]) but the additional-volume estimate becomes imprecise (1.12 [0.97, 1.30]).

This separates two signals that the original cumulative-paper coefficient combined. Direct activation mainly distinguishes where any formal output appears; additional paper volume is more closely related to how much output appears on concerns that are already active.

## Does temporal recurrence add information?

Across 3 preceding meetings, spreading the same focal-paper volume from one to two effective meetings yields a hard-output ratio of 0.75 [0.58, 0.96], a soft-output ratio of 0.79 [0.67, 0.93], and an any-output ratio of 0.92 [0.71, 1.19] after also accounting for how recently attention occurred. The hard-output ratio for a 25-percentage-point increase in returning-actor activity is 1.24 [0.96, 1.59].

Across 5 preceding meetings, spreading the same focal-paper volume from one to two effective meetings yields a hard-output ratio of 1.02 [0.88, 1.19], a soft-output ratio of 0.98 [0.88, 1.08], and an any-output ratio of 1.08 [0.93, 1.26] after also accounting for how recently attention occurred. The hard-output ratio for a 25-percentage-point increase in returning-actor activity is 1.07 [0.91, 1.26].

Across 8 preceding meetings, spreading the same focal-paper volume from one to two effective meetings yields a hard-output ratio of 1.05 [0.95, 1.17], a soft-output ratio of 1.02 [0.93, 1.11], and an any-output ratio of 1.02 [0.87, 1.19] after also accounting for how recently attention occurred. The hard-output ratio for a 25-percentage-point increase in returning-actor activity is 0.95 [0.83, 1.10].

Across 10 preceding meetings, spreading the same focal-paper volume from one to two effective meetings yields a hard-output ratio of 1.09 [1.00, 1.19], a soft-output ratio of 1.04 [0.97, 1.11], and an any-output ratio of 1.00 [0.89, 1.13] after also accounting for how recently attention occurred. The hard-output ratio for a 25-percentage-point increase in returning-actor activity is 0.93 [0.74, 1.16].

Temporal spread does not add a stable signal beyond volume. Its hard-count estimate approaches a positive association at ten meetings, but this does not reproduce for soft output, output occurrence, or the site-administration exclusion. Returning activity by the same actors is also unstable once volume, timing, nearby attention, and earlier output are held fixed. The record therefore does not support a strong claim that repetition across meetings or by the same actors independently drives output.

## Quiet-to-active transitions

At 5 meetings, the temporal-spread ratio is 1.01 [0.92, 1.11] for soft-assigned output and 1.95 [0.97, 3.92] for the onset of output after a window with no earlier focal-concern output. The onset model contains 58 events.

At 10 meetings, the temporal-spread ratio is 1.05 [0.99, 1.12] for soft-assigned output and 1.25 [0.29, 5.34] for the onset of output after a window with no earlier focal-concern output. The onset model contains 31 events.

The onset estimates are exploratory because the number of quiet-to-active transitions is small and the predictors are correlated. They do not establish that repeated attention causes formal action.

## Narrative implication

The concern space describes where documentary portfolios expand. Formal action is associated most clearly with direct attention to the focal concern and with earlier formal instruments, not with proximity alone. A precise narrative is therefore: actors explore locally through the concern space; formal output concentrates where explored concerns receive direct documentary attention. More papers accompany more output, but the evidence does not show that simply spreading those papers across meetings, or having the same actors return, independently carries a concern into formal action.

These are descriptive associations. Output may sustain later papers, and stable issue streams may generate both recurring attention and repeated output.

## Reproducibility

Run `micromamba run -n ultraplot-dev python scripts/analyze_attention_recurrence.py` from the manuscript repository.
