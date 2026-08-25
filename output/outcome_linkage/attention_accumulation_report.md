# How documentary attention relates to formal output

All time windows are ordered ATCM meetings, not calendar years. The models compare a concern with itself over time and account for changes shared by each meeting. They are descriptive associations, not causal effects.

## Main result

Across the preceding five meetings, doubling same-concern paper attention plus one is associated with later hard-assigned output at a rate ratio of 1.12 [0.95, 1.32] after accounting for nearby-concern papers and earlier output. The soft-assignment estimate is 1.08 [0.99, 1.18]. The corresponding ratios are 1.16 [0.99, 1.36] for nearby-concern attention and 1.10 [0.87, 1.38] for earlier formal output. The same-concern result remains when the outcome is only whether any output occurs (1.14 [0.99, 1.30]) and after removing routine site-administration concerns (1.21 [1.02, 1.45]).

The same-concern association grows through roughly five preceding meetings and remains positive through ten. Nearby-concern attention fades as the window widens. A longer institutional horizon therefore reveals sustained attention--output alignment at the concern level rather than through the wider concern network.

## New episodes versus continuing streams

There are 41 concern--meeting output episodes after five meetings with no output on that concern. Prior same-concern attention does not clearly distinguish those onsets (OR 1.47 [1.05, 2.07]), nor does nearby attention (OR 0.65 [0.40, 1.06]). Event-centered trajectories show the clearest lead-up within continuing output streams, not before genuinely quiet-to-active transitions.

## Temporal caution

When five preceding and five subsequent meetings enter together, doubling past same-concern attention plus one is associated with output at 1.09 [0.90, 1.33]. The future estimate is smaller and uncertain at 1.12 [0.94, 1.33]. Anticipated output, later attention, and stable issue streams can still contribute to both sides of the association. The analysis does not identify a one-way causal pipeline.

## Interpretation

The concern space predicts where actors shift relative documentary attention. Adopted ATCM outputs are tied more closely to accumulated attention on the focal concern than to activity in nearby concerns; earlier output provides no stable additional association. New output on a previously quiet concern remains difficult to anticipate.

## Reproducibility

Run `micromamba run -n ultraplot-dev python scripts/analyze_attention_accumulation.py` from the manuscript repository.
