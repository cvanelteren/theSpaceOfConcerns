# Final formal-output validation

## Status

The adversarial coding process reached consensus on every title. No disputed
case remains unresolved.

The validation covers 246 unique formal-output titles. This set combines a
120-title audit balanced across instrument type and classifier confidence with
every output in the verified paper-lineage comparison. The latter contains 157
outputs: 95 with a documented adoption or contribution link and 85 with a
discussion-only link, with overlap between those groups.

## Blind coding and consensus

Three independently prompted GPT-5-family Codex agents coded title text using a
fixed 45-concern codebook; the packet retained year and instrument type as
identifiers. They saw neither the classifier output nor the submitted papers,
lineages, figures, or one another's decisions. The protocol
allowed `INSUFFICIENT_TITLE` and `OUTSIDE_TAXONOMY`, so unclear titles did not
have to be forced into a concern.

- The three first-pass coders agreed unanimously on 232 of 246 titles (94.3%).
- Pairwise exact agreement ranged from 95.1% to 97.6%.
- Pairwise Cohen's kappa ranged from 0.947 to 0.973.
- An adversarial adjudicator considered the strongest case for every proposed
  label on the remaining 14 titles.
- A fresh reviewer challenged every provisional label without seeing the
  classifier or lineage. It did see the proposed label and whether it came
  from unanimity or adjudication.
- A final arbitrator received the two competing label pairs in random order
  and selected one of the independent proposals in all four cases.
- Sixteen titles were left unassigned from title text alone. The complete
  lineage set contains 149 codable and eight unassignable outputs.

This is an adversarial model audit, not human expert coding. The coders share a
model family, so their agreement does not remove correlated language-model
error. The manuscript now states that limit directly.

## Classifier validity

The 120-title audit deliberately contains equal numbers from the four
instrument types and from high and lower classifier-confidence strata. The
population estimates therefore weight those eight strata back to their
frequency among all 740 titled outputs.

| Check | Denominator | Population-weighted estimate | 95% stratified-bootstrap interval |
|---|---|---:|---:|
| Title can be assigned from the codebook | All titles | 91.6% | 85.3–97.0% |
| Classifier's first concern equals consensus | Assignable titles | 77.9% | 70.3–84.8% |
| Consensus concern appears in classifier's top three | Assignable titles | 94.7% | 91.3–97.6% |
| Classifier and consensus lie in the same descriptive region | Assignable titles | 92.7% | 88.7–96.3% |
| Mean concern proximity between classifier and consensus | Assignable titles | 0.866 | 0.820–0.907 |

Counting unassignable titles as non-matches gives a population-weighted exact
agreement of 71.3%. The conditional estimate is the relevant classifier check,
while coverage reports how often that check can be made from title text.

For the 95 adoption-lineage outputs, 89 titles were codable. Exact agreement
was 70.8%, top-three agreement was 93.3%, and 87.6% fell in the same descriptive
region.

## Effect on the substantive results

The paper-link and actor-conversion analyses were rerun with a one-hot concern
at the final consensus label. Unassignable titles were omitted. The original
probability-based results remain available as sensitivities.

Both columns below use the same consensus-codable outputs. This separates the
effect of relabelling from the omission of eight lineage titles that could not
be assigned.

| Result | Classifier probabilities, matched outputs | Consensus labels | Decision |
|---|---:|---:|---|
| Adoption-linked paper ranks closer than another same-meeting paper | 67.2% | 66.0% | Stable |
| Discussion-only paper ranks closer | 49.8% | 50.2% | Stable at chance |
| Exact concern alone ranks adoption-linked paper closer | 61.5% | 63.9% | Stable |
| Nearby-only comparison after exact matches are removed | 55.9% | 53.5% | Both intervals include chance |
| Nearby-concern term in the within-output model | OR 1.55 [1.22, 1.96] | OR 1.33 [1.04, 1.71] | Smaller but detectable |
| Nearby-concern term after accounting for title overlap | OR 1.48 [1.17, 1.88] | OR 1.26 [0.98, 1.62] | Only classifier estimate remains distinct |
| Prior portfolio proximity in actor conversion model | OR 1.05 [0.77, 1.43] | OR 1.04 [0.79, 1.36] | Unchanged null result |
| Prior portfolio size in actor conversion model | OR 1.01 [0.65, 1.56] | OR 1.03 [0.68, 1.56] | Unchanged null result |

The off-label row starts from the same codable output set, but removing exact
matches leaves 66 usable classifier-positioned outputs and 64 usable
consensus-positioned outputs because soft and hard labels define exact matching
differently.

The validation does not reverse the paper's conclusion. Papers documented as
contributing to adoption are close to the output, while discussion-only papers
are not. Exact concern matching carries most of the result. The additional
signal from nearby concerns is modest and is not distinct after accounting for
paper–output title overlap. Current papers are associated with documented
contribution; the actor's portfolio at the preceding meeting is not.

## Proximity direction

The validation and rerun use proximity in the intended direction. Larger
`phi` means that concerns are closer. Distance is `1 - phi`, so a larger
distance means farther apart. The outcome analyses rank larger proximity as
closer; they do not invert this relation.

## Manuscript and figure changes

- Figure 3a now uses consensus-coded outputs for the adoption and discussion
  comparisons.
- Figure 3c now uses consensus-coded outputs in the exposure-corrected actor
  model.
- Figure 3b still uses classifier probability distributions because it covers
  all 740 titled outputs. Its stratified validation is reported in the text.
- The supplementary discrimination figure now uses the consensus-coded
  lineage set.
- The abstract, Results, Discussion, figure captions, and Methods report the
  consensus estimates and the language-model audit limitation.

## Reproducibility

The main scripts are:

- `scripts/build_outcome_consensus_packet.py`
- `scripts/merge_outcome_consensus_coders.py`
- `scripts/finalize_outcome_consensus.py`
- `scripts/evaluate_outcome_consensus.py`
- `scripts/analyze_outcome_consensus_sensitivity.py`
- `scripts/plot_attention_to_outcomes.py`
- `scripts/plot_space_discrimination.py`

The fixed codebook, final consensus files, agreement statistics, validation
metrics, and consensus sensitivity results are in `output/outcome_linkage/`.
