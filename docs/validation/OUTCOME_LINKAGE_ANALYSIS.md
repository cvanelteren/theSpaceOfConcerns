# From attention to formal outcomes

> **Superseded analysis note (14 August 2026).** The paper--output lineage
> results in this working document must not be used. An adversarial audit found
> that the parser discarded cited instrument years and sometimes matched an
> earlier Measure, Decision, Resolution, or Recommendation to a same-number
> output at the report's current meeting. The manuscript now uses only the
> complete 1995--2025 regular-ATCM output inventory and concern--meeting
> associations that do not depend on reconstructed paper--output links. The
> active publication gate is `scripts/verify_attention_to_outcomes.py`.

## Headline claim

> **The space of concerns identifies which documents reach formal action.**
> Among the papers submitted to a single meeting, concern proximity to an
> instrument's independently classified subject ranks the paper documented as
> contributing to its adoption above a randomly chosen co-submitted paper 65%
> of the time. For papers recorded only in the discussion, proximity is
> uninformative. The discrimination is not a restatement of the Secretariat's
> concern label: holding constant the probability that the instrument concerns
> the paper's own topic, position relative to the instrument's *other* likely
> concerns still identifies documented contributors, and neither term does so
> for discussion-only papers.

Three properties make this a claim about the space rather than about the
taxonomy or about hindsight.

1. **No parameters are fitted to the lineage.** The ranking statistic contains
   no trained model. The space is estimated only from documentary history at
   ATCMs preceding each outcome's meeting, the outcome's concern comes from a classifier
   that never saw a single paper--outcome edge, and the ranking is a direct
   computation on those two inputs.
2. **The comparison set is the institution's own.** Every paper is ranked only
   against papers submitted to the same meeting, so the meeting's agenda,
   period, and documentary volume are held fixed by construction.
3. **The placebo is clean.** The identical statistic applied to papers that
   merely appear in the discussion record returns chance. Whatever the space is
   detecting, it is specific to documented contribution.

The obvious circularity is addressed. Because an instrument's concern is read
off its title, an instrument that reuses its source paper's title would be
classified into that paper's concern partly by construction. Removing the
families where such reuse is institutionalised *strengthens* the result
(0.696), and controlling for paper--instrument title overlap directly leaves
the geometric term essentially unchanged (OR 1.46 to 1.40) while the label term
absorbs the reuse (1.93 to 1.64). Title reuse therefore explains part of what
the concern label contributes and almost none of what the geometry contributes.
See *The circularity risk* below.

The claim is about **discrimination within an opportunity set**, not about
forecasting. It says which of the documents in front of a meeting is the one on
a route to adoption. It does not say whether a concern will produce an outcome
at a later meeting, and the analysis should not be described as predicting
formal output. See *What the space does not do* below.

## The supporting results

The concern space is consequential at a specific stage of the documentary process: papers that are documented as contributing to adoption lie unusually close to the independently classified concern of the resulting output, and prior attention robustly predicts later output within the same concern. Evidence for adjacent-concern spillover is inconclusive, while an actor's exact prior proximity does not independently distinguish documented contributors.

In the main analysis, attention is ordered by ATCM rather than calendar year. The association is strongest at the adopting meeting. Across the preceding three ATCMs, same-concern attention remains associated with later output, while the nearby-concern estimate is uncertain. The longer legal histories of Measures are a separate lineage question discussed below.

This is evidence that formal action follows concern-specific documentary pathways. It is not evidence that attention mechanically diffuses through the network or that documentary activity causes legal adoption.

## What is being linked

The analysis separates three observed layers.

1. **Documentary attention.** Working papers and information papers retain the Secretariat's 45 concern labels. Their actor-level co-specialization generates the space of concerns.
2. **Annual ATCM outputs.** The outcome inventory contains 768 Recommendations, Measures, Decisions, and Resolutions from 1961--2025. Titles are available for 740 (96.4%). Their concern positions are assigned independently from their titles; paper--outcome edges are not used in this assignment.
3. **Verified documentary lineage.** The lineage database identifies papers that were discussed, proposed, or documented as contributing to particular outcomes. Verified paper--outcome coverage begins in 1991, so the direct test covers 1991--2025 even though the outcome-rate models cover 1961--2025.

Constitutional instruments adopted in other fora remain a fourth, separate layer. The Madrid Protocol was adopted at SATCM XI-4 and is not counted as an annual ATCM output. The Ross Sea region MPA was adopted by CCAMLR and is likewise outside the ATCM outcome corpus. Their subsequent ATCM follow-up remains observable.

This distinction resolves the apparent Drilling anomaly. Recommendation XIV-3 (1987), *Human impact on the Antarctic environment: Safeguards for scientific drilling*, is a high-confidence annual output assigned to Drilling. The 1991 Madrid Protocol is instead external constitutional context: Article 7 prohibits Antarctic mineral-resource activity other than scientific research, linking primarily to Mineral resources and secondarily to Drilling's research context.

## Independent outcome positioning

A word- and character-level title classifier is trained on 6,572 submitted-paper titles that carry exactly one Secretariat concern label. Five-fold cross-validation holds out entire ATCM meetings, preventing titles from the same meeting from appearing in both training and test folds. The classifier's decision scores are temperature-calibrated and retained as a 45-element probability distribution for each outcome.

Cross-validated performance is:

| Validation target | Result |
|---|---:|
| Exact concern | 63.2% |
| Top-three concerns | 80.5% |
| Same broad descriptive region | 81.3% |
| Mean proximity between true and predicted concern | 0.756 |
| High-confidence coverage | 57.9% |
| Accuracy within high-confidence subset | 81.0% |

The exact label is therefore noisy, but most errors remain close in the concern space. Main models integrate the full probability distribution rather than treating the top label as certain. Hard top-one and high-confidence-only specifications provide sensitivity checks. A 120-outcome blind author-coding packet is available in `output/outcome_linkage/outcome_topic_validation_blind.csv`; it should be completed before submission because cross-validation against paper labels is not a substitute for substantive legal coding of outcome titles.

## Does the geometry add anything beyond the concern label?

The proximity results have an obvious deflationary reading: a paper filed under
the same Secretariat concern as the instrument is trivially "close", so the
apparent role of the space could be an exact label match wearing a geometric
costume. `scripts/analyze_space_discrimination.py` separates the two.

Every paper in this archive carries exactly one Secretariat concern, so the
expected proximity of a paper to an instrument decomposes cleanly:

- **same-concern mass** -- the probability the instrument is about the paper's
  own concern, which is the exact-label-match explanation in continuous form;
- **related-concern proximity** -- the same expectation computed with the
  diagonal of the space removed, so only the paper's position relative to the
  instrument's *other* likely concerns can contribute.

### Discrimination

The ranking statistic is the probability that a linked paper outranks a
randomly chosen unlinked paper submitted to the same meeting, averaged equally
over instruments. It contains no fitted parameters.

| Comparison | Statistic | 95% CI | Instruments |
|---|---:|---:|---:|
| Papers that reached adoption, full proximity | 0.648 | 0.586--0.710 | 95 |
| Papers in the discussion only, full proximity | 0.491 | 0.429--0.551 | 83 |
| Papers that reached adoption, same-concern mass alone | 0.602 | 0.532--0.673 | 95 |
| Papers that reached adoption, off-label geometry, exact matches deleted | 0.537 | 0.467--0.606 | 72 |
| Papers in the discussion only, off-label geometry, exact matches deleted | 0.473 | 0.415--0.533 | 77 |

Concern proximity ranks the paper that reached adoption above a randomly chosen
co-submitted paper 65% of the time, and is at chance for papers that only
joined the discussion. Same-concern mass alone reaches 0.602, so the concern
label does much of the work -- as it should, since adoption-linked papers carry
the instrument's leading concern 30% of the time against 5% for unlinked
papers.

This statistic is deliberately more conservative than the within-meeting
percentile reported below. The percentile ranks the observed paper against the
full candidate list including itself; the discrimination statistic excludes each
linked paper from its own comparison set, which is why it returns 0.648 where
the percentile returns 67.5%. The two agree, and the headline uses the stricter
of them.

The strictest non-parametric test deletes every paper sharing the instrument's
leading concern, leaving only off-label geometry to rank the remainder. It is
directionally right but not individually decisive (0.537, 95% CI
0.467--0.606). This test discards 70% of the available information and retains
only 72 instruments, so it is underpowered by construction; its value is that
it points the same way as the better-powered test below, while the
discussion-only placebo run through the identical filter does not (0.473).

One diagnostic in the output looks contradictory and is not: ranking by
off-label geometry *without* deleting exact matches gives 0.425, below chance.
Zeroing the diagonal actively penalises exact-match papers, and adoption-linked
papers are disproportionately exact matches, so that statistic is confounded by
construction. It is reported for transparency and should not be read as
evidence against the space.

### The circularity risk, and how far it can be ruled out

An instrument's concern is assigned from its title. If an instrument reuses the
title of the paper that proposed it, the classifier will place it in that
paper's concern almost by construction, and the discrimination would be
partly circular. Two guards address this, and they do not agree.

| Guard | Statistic | 95% CI | Instruments |
|---|---:|---:|---:|
| Unrestricted | 0.648 | 0.586--0.710 | 95 |
| Excluding site administration | **0.696** | 0.604--0.779 | 49 |
| Paper--instrument title overlap below 0.30 | 0.597 | 0.528--0.664 | 81 |
| Paper--instrument title overlap below 0.15 | 0.527 | 0.459--0.599 | 67 |

Excluding routine site administration -- management plans, historic sites,
protected areas, visitor guidelines -- *strengthens* the result to 0.696. That
is the targeted guard, because verbatim title reuse is standard practice
precisely in those families: a working paper titled "Revised Management Plan
for ASPA 116" becomes a Measure of nearly the same name. If title copying drove
the discrimination, removing the families where copying is institutionalised
should have weakened it. It did the opposite.

Filtering on lexical similarity instead attenuates the result monotonically,
and at the strictest threshold the interval covers chance. That filter should
not be read as the decisive test, because title overlap is not a clean measure
of circularity: a paper genuinely about the subject of an instrument shares
words with it whether or not anything was copied. Conditioning on low overlap
therefore removes on-topic papers along with circular ones -- it conditions on a
proxy for the signal being measured. The distribution supports this reading:
the median adoption-linked paper shares **no** content words with the
instrument it fed, and only 12% exceed an overlap of 0.30, so the filter
discards instruments rather than isolating a confound.

The decisive version of the check controls for title overlap inside the race
rather than deleting instruments, so that all the data is retained and lexical
similarity is held fixed rather than used as a sample filter.

| Term | Without control | With title-overlap control | 95% CI | p |
|---|---:|---:|---:|---:|
| Instrument is about the paper's own concern | 1.93 | 1.64 | 1.40--1.91 | <1e-9 |
| Proximity through related concerns | 1.46 | **1.40** | 1.12--1.77 | 0.0039 |
| Paper--instrument title overlap | -- | 1.22 | 1.15--1.30 | <1e-9 |

Title overlap is itself a genuine predictor of documented contribution, which
is why the subsetting filter destroyed so much signal. But it does not explain
the geometry. Holding it constant moves the related-concern term from 1.46 to
1.40 -- a 4% change in the odds ratio, with the interval still clearly
excluding one -- while the label term absorbs considerably more of it
(1.93 to 1.64). Lexical reuse of titles therefore accounts for part of what the
*concern label* contributes, and essentially none of what the *geometry*
contributes.

Taken together, the three guards converge. Removing the families where title
reuse is institutionalised strengthens the result; deleting instruments by
lexical similarity weakens it, but only because that filter also deletes
on-topic papers; and controlling for lexical similarity directly, which does
neither, leaves the geometric term intact. The circularity objection is
answered for the geometric component of the claim. It remains a live
qualification for the label component, which is the larger of the two.

### Racing label against geometry

The better-powered test keeps every paper and races the two components inside a
conditional logit stratified by instrument, so each comparison is between
papers submitted to the same meeting and evaluated against the same instrument.
The two terms are only mildly correlated (-0.35), so both are separately
identified.

| Papers | Term | Odds ratio per 1 SD | 95% CI | p |
|---|---|---:|---:|---:|
| Reached adoption | Instrument is about the paper's own concern | 1.93 | 1.68--2.21 | 6e-21 |
| Reached adoption | Proximity through related concerns | **1.46** | 1.16--1.84 | 0.0014 |
| Discussion only | Instrument is about the paper's own concern | 1.12 | 0.95--1.32 | 0.180 |
| Discussion only | Proximity through related concerns | 1.03 | 0.84--1.27 | 0.756 |

Holding constant the probability that the instrument is about the paper's own
concern, a one-standard-deviation increase in proximity to the instrument's
other likely concerns is associated with 46% higher odds of being a documented
contributor. The label carries more weight than the geometry, and the honest
summary is that the space adds a real but secondary increment rather than
replacing the taxonomy.

The placebo is what makes this credible. Run on papers that appear only in the
discussion record, both terms collapse to null. A confound that inflated
proximity for documents near an instrument's subject would inflate it for
discussion papers too; the contrast is specific to documented contribution.

## Direct evidence: where a paper lands

For every verified paper--outcome edge, the source paper's concern is compared with the independently assigned probability distribution of the outcome. Proximity is computed in a cumulative concern space estimated only from ATCMs preceding the outcome's meeting. The observed paper is then ranked against every categorized paper available at that meeting. This produces a within-meeting percentile that controls for the meeting's documentary opportunity set.

The null draws the same number of source papers from each outcome's meeting and averages outcomes equally, so outcomes with many recovered links cannot dominate the test.

| Verified relation | Edges | Outcomes | Observed percentile | Null percentile | One-sided permutation p |
|---|---:|---:|---:|---:|---:|
| Adoption or documented contribution | 133 | 95 | 67.5% | 50.3% | 0.0001 |
| Adoption/contribution, excluding routine site administration | 70 | 49 | 72.0% | 50.3% | 0.0001 |
| Adoption/contribution, high-confidence outcome coding | 85 | 64 | 68.0% | 50.3% | 0.0001 |
| Proposal or discussion only | 138 | 85 | 51.3% | 50.3% | 0.357 |

The contrast is more informative than a pooled lineage result. Merely appearing in the discussion record does not make a paper spatially distinctive. Papers connected to adoption do. The space therefore locates the documentary routes that reach formal action rather than simply distinguishing documents that receive attention.

## Indirect evidence: whether attention precedes output

The second test uses all 45 concerns at each of the 47 ATCMs (2,115 concern--meeting observations). Formal output is the sum of each outcome's independently estimated probability mass at a concern--meeting pair. Fixed-effect Poisson models compare each concern with itself over the ordered meeting sequence and include meeting effects shared by all concerns. Predictors are standardized after `log(1+x)` transformation. Standard errors are clustered by both concern and meeting.

The prospective specification relates output at meeting *t* to:

- papers in the same concern during the preceding three ATCMs;
- papers in the five nearest concerns during those meetings, where neighbours come from cumulative spaces estimated before meeting *t*; and
- output in the same concern during the preceding three ATCMs.

| Outcome definition | Prior same-concern attention, IRR per 1 SD | 95% CI | p |
|---|---:|---:|---:|
| Probability-weighted outputs | 1.295 | 1.093--1.534 | 0.0027 |
| Hard top-one assignments | 1.406 | 1.130--1.750 | 0.0023 |
| High-confidence assignments only | 1.517 | 1.152--1.997 | 0.0030 |
| Excluding site-administration concerns | 1.288 | 1.038--1.598 | 0.0218 |
| Resolutions | 1.420 | 1.142--1.765 | 0.0016 |
| Decisions | 1.183 | 0.939--1.490 | 0.155 |
| Measures | 1.067 | 0.803--1.416 | 0.656 |

In the main probability-weighted model, a one-standard-deviation increase in same-concern attention across the preceding three ATCMs is associated with 29.5% more output at the next meeting. The nearby-concern estimate is positive but imprecise (IRR 1.148, 95% CI 0.948--1.390). Same-concern evidence is therefore robust, whereas adjacent-concern spillover is inconclusive. The two coefficients are not themselves distinguishable (own-to-nearby IRR ratio 1.128, 95% CI 0.839--1.517; `p=0.43`). The analysis cannot claim that own-concern accumulation is stronger than spillover merely because one coefficient crosses a significance threshold and the other does not.

The instrument split also bounds the claim. Across the preceding three ATCMs the association is visible for non-binding Resolutions but not for Decisions or legally binding Measures. The Measure null is not a general absence of historical association; the section *What Measures actually are* below examines their longer formal lineages separately.

Separate meeting lags show the strongest association at the adopting ATCM. Both same-concern and nearby attention weaken at earlier meetings, although their intervals overlap. The three-meeting aggregate is the prospective summary; the lag profile is descriptive rather than evidence of a precisely timed causal process.

## What Measures actually are

The Measure result needed its own analysis, because the three-year test assumed
a short, direct transition from recent papers to a Measure. The record does not
work that way. `scripts/analyze_measure_pathways.py` classifies all 279 titled
Measures, separates strong from contextual formal predecessors, refits the
timing question over pre-specified windows, and tests spatial continuity along
validated lineage edges. `MEASURE_CODING_PROTOCOL.md` states every coding rule.

### Composition

Measures, Decisions, and Resolutions exist only from 1995; Recommendations run
1961--1994. The published instrument split was fitted over 1961--2025, so 34 of
65 years contributed structural zeros. Refitting the same three-year
specification on 1995--2025 gives IRR 1.123 (95% CI 0.827--1.525), essentially
the published estimate, so those zeros were not the cause of the null.

The composition is the more consequential fact. A rule-based functional
typology assigns 273 of 279 Measures (97.8%) to recurring site administration:
218 management plans, 29 historic sites, and 26 protected-area designations.
Site identifiers are recoverable for 230 Measures, covering 118 distinct sites.
Six Measures fall outside that category across the whole period: the
Secretariat instrument (2003), insurance and contingency planning for tourism
(2004), Annex VI on liability (2005), specially protected species (2006),
landing of persons from passenger vessels (2009), and the amendment of Annex II
(2009). A nominal sample of 279 instruments therefore does not represent 279
independent political processes, and no substantive-hardening subsample can be
estimated: models fitted to those six Measures return incidence-rate ratios in
the thousands with intervals spanning orders of magnitude, and are marked
`estimable = False` in the output.

### How Measures are assembled

| Observed pathway | Measures | Share |
|---|---:|---:|
| Earlier formal outcome only | 148 | 53.0% |
| Neither recovered | 83 | 29.7% |
| Both paper and formal outcome | 34 | 12.2% |
| Paper only | 14 | 5.0% |

Two-thirds of Measures have a recovered formal predecessor; 17.2% have a
recovered paper. Strong transformation relations (`supersedes`, `amends`,
`pursuant_to`, `designates_under`) reach a Measure after a median of six years;
Measure-to-Measure supersession, the dominant relation with 131 edges, also has
a median of six years. Contextual references (`recalls`, `cites`) have a median
of fourteen years and reach back much further, with recalled Recommendations at
a median of 31 years. The published three-year exposure window sits below
essentially all of this formal inheritance.

`neither` is a property of the parsed record, not evidence of institutional
independence. `measure_unconnected_audit_sample.csv` holds a stratified audit
packet for those 83 Measures, with blank coder columns, and should be completed
before the absence is interpreted.

### Timing

Pre-specified windows, on 1995--2025, with concern and year fixed effects and
two-way clustered errors:

| Exposure window | Measures, IRR per 1 SD | 95% CI | p |
|---|---:|---:|---:|
| Same meeting | 1.181 | 0.987--1.414 | 0.069 |
| 1--3 years prior | 1.047 | 0.792--1.385 | 0.744 |
| 4--7 years prior | 0.999 | 0.821--1.217 | 0.996 |
| 8--15 years prior | 1.744 | 1.119--2.720 | 0.014 |

The association is real but slow. It is absent at one to seven years and
appears at eight to fifteen. It strengthens under hard top-one assignment
(IRR 3.669, 95% CI 2.380--5.656), so it is not an artefact of probability
weighting. The published null was principally a timing mismatch.

The obvious worry is that a fifteen-year window is reading a concern's own
growth in both attention and output rather than a sequence. The direct test,
concern-specific linear trends, is not estimable here: 45 trends on top of
concern and year effects make the panel singular, which is unsurprising when
174 of 279 Measures sit in a single concern. The mirror-image placebo is
estimable and is informative in the same direction. Papers submitted eight to
fifteen years *after* a year, which cannot have contributed to that year's
Measures, carry the opposite sign (IRR 0.608, 95% CI 0.428--0.863); the
one-to-three and four-to-seven leads are null. A symmetric trend artefact would
produce a positive lead of similar size. The asymmetry favours a directional
reading, and the negative lead is itself consistent with attention subsiding in
a concern once its register has been revised, though nothing here identifies
that mechanism.

Instruments run on different clocks. Resolutions respond to attention at the
adopting meeting itself (IRR 1.518, 95% CI 1.208--1.909) and show nothing at
eight to fifteen years (IRR 0.846). Measures show the reverse. Decisions show
neither.

### Formal inheritance

Adding formal predecessors changes the picture in the direction mediation
predicts, without establishing it. Earlier non-Measure instruments in the same
concern at eight to fifteen years predict Measures (IRR 1.346, 95% CI
1.025--1.766), as do earlier Measures at four to seven years (IRR 1.206, 95% CI
1.028--1.415). In the combined model the eight-to-fifteen-year paper
coefficient loses 28% of its log-IRR and its interval widens across one (IRR
1.489, 95% CI 0.918--2.417), while same-meeting papers become clearly
associated (IRR 1.256, 95% CI 1.054--1.496). This is descriptive evidence
consistent with a multi-stage pathway, not a causal mediation estimate.

Within the recurring-site families, which are 97.8% of the population, the
eight-to-fifteen-year formal-precedent term is the one that survives adjustment
(IRR 1.418, 95% CI 1.118--1.799). Measure production in this corpus is largely
the administrative renewal of a protected-area register on a decadal cycle,
with papers arriving at the adopting meeting to carry the individual plan.

### Whether the space matters along the lineage

One estimand for every edge type: expected concern proximity in a
cumulative-lagged space built only from records preceding the target year. Each
observed predecessor is ranked against alternatives matched on source type,
calendar period, lag bin, and availability before the target year. Targets are
averaged equally.

| Edge set | Edges | Targets | Observed percentile | Matched null | p |
|---|---:|---:|---:|---:|---:|
| Paper to intermediate outcome to Measure, adoption-opened | 317 | 120 | 75.0% | 50.4% | 0.0001 |
| Contextual predecessor to Measure | 852 | 178 | 71.0% | 52.5% | 0.0001 |
| Paper to outcome, adoption-linked | 133 | 95 | 67.5% | 50.4% | 0.0001 |
| Strong predecessor to Measure | 167 | 143 | 58.9% | 51.7% | 0.0007 |
| Paper to intermediate outcome to Measure, discussion-opened | 31 | 20 | 52.2% | 50.3% | 0.369 |
| Paper to outcome, discussion-only | 138 | 85 | 51.3% | 50.3% | 0.355 |

The adoption-versus-discussion contrast holds at one step and at two. A paper
documented as contributing to adoption sits at the 67.5th percentile of
proximity to the instrument it fed; a paper recorded only in the discussion
record sits at chance. Extending the path by one institutional step preserves
the contrast: paths opened by an adoption-linked paper reach the 75.0th
percentile of proximity to a Measure adopted several steps and often more than
a decade later, while paths opened by a discussion-only paper stay at chance
(52.2%, `p=0.369`). Proximity therefore tracks documented contribution rather
than mere documentary presence, and it survives transmission through an
intermediate instrument.

Strong transformation edges clear the null by the smallest margin. This is a
property of their matched pool rather than a weaker relation: supersession is
overwhelmingly Measure-to-Measure, so the alternatives are other management
plans already concentrated in the same corner of the space, leaving little room
to be distinctive. Points are only interpretable against their own null, never
against each other.

### Decision-table row

The evidence selects the timing row, qualified by composition and by partial
formal mediation:

> The original Measure null was principally a timing mismatch. Attention eight
> to fifteen years before adoption predicts Measures; attention one to three
> years before does not. Part of that long-run association runs through formal
> precedent, and the population it describes is 97.8% recurring site
> administration, so it is a statement about the renewal of a protected-area
> register rather than about general agenda-to-law conversion. Along that
> lineage the concern space is informative: observed links are unusually
> concern-proximate relative to a time- and source-matched null, and the
> contrast between documented contribution and mere documentary presence
> survives transmission through an intermediate instrument.

## Actor movement and whose papers link

The final test separates opportunity from conversion. The earlier binary model
asked whether an actor appeared at least once in an output lineage, so actors
submitting more papers mechanically had more chances to appear. The corrected
model instead counts linked papers out of all papers that an actor submitted to
the output's meeting. It covers 3,710 actor--output pairs, 22,561 eligible
actor--paper opportunities, and 191 linked actor--papers around 95 outputs.
Actor and output fixed effects hold stable submitter differences and each
output's opportunity set constant.

| Characteristic, per 1 SD | Odds ratio for the documented-link rate | 95% CI | p |
|---|---:|---:|---:|
| Papers share the output concern | 1.63 | 1.22--2.20 | 0.0011 |
| Papers address nearby concerns | 1.60 | 1.15--2.24 | 0.0057 |
| Earlier portfolio proximity | 0.96 | 0.75--1.22 | 0.725 |
| Earlier concerns covered | 1.88 | 0.85--4.12 | 0.117 |
| Working-paper share | 2.03 | 1.47--2.80 | <0.0001 |

Once submission volume is the denominator, the route visible in the public
record is tied to what the actor submitted for that output. Exact concern match,
nearby concerns, and document type matter. The actor's earlier position in the
map does not add a detectable association, and the estimate for earlier concern
coverage is too uncertain to support an inequality claim about conversion.

The trajectory figure makes this distinction visible for the three actors already shown in Figure 1. Lines show each actor's rolling five-year position on the continuous concern axis. Diamonds locate the independently classified concern of outcomes to which that actor has a verified contribution. Australia has 16 such links, the Netherlands eight, and Ukraine none in the verified lineage corpus. These are observed documentary links, not a ranking of political influence; zero means no recovered verified contribution, not proof of no influence.

## Narrative spine for the paper

The headline claim gives the space a job that formal-output counts cannot do:
it says which of the documents in front of a meeting is on a route to adoption.
The rest of the outcome analysis says what happens to that route afterwards --
that it is local, that it survives transmission through intermediate
instruments, and that it runs on instrument-specific clocks.

The outcome analysis gives Figure 1 a concrete purpose.

1. The space maps where documentary attention is organized.
2. Actor portfolios move locally through that space.
3. Formal outputs are distributed unevenly across it.
4. Verified adoption-linked papers follow concern-space routes, while discussion-only papers do not.
5. Sustained attention is associated with subsequent output within the same concern, particularly in Resolutions; adjacent-concern spillover remains inconclusive.
6. Measures run on a much longer clock. They inherit from formal precedent, respond to attention eight to fifteen years earlier rather than one to three, and are overwhelmingly the renewal of a protected-area register.
7. The actor's position at the preceding meeting does not secure a higher documented-link rate; the relevance and type of the papers submitted at the focal meeting do.

This supports the following central claim:

> The space of concerns connects agenda development to formal action by identifying the documentary pathways through which attention reaches outcomes. Among the documents in front of a meeting, position in the space picks out those on a route to adoption, and it does so beyond what the concern label alone conveys. The route is tied to what an actor submits at the focal meeting rather than secured by its position at the preceding ATCM. These pathways do not imply automatic diffusion across the network, and they do not forecast whether or in which instrument a concern will produce formal output.

## Suggested Results text

### The space identifies documents on a route to formal action

Formal output records what reached agreement; it does not show which of the documents in front of a meeting was on a route there. The concern space does. For each of 95 instruments with verified adoption lineage, we ranked every categorized paper submitted to the adopting meeting by its expected concern proximity to the instrument's independently classified subject, in a space estimated only from documentary history at preceding ATCMs. The ranking uses no fitted parameters: the outcome's concern comes from a title classifier that never saw a paper--outcome edge, and the space is constructed prospectively. A paper documented as contributing to adoption outranks a randomly chosen co-submitted paper 64.8% of the time (95% CI 0.586--0.710). A paper recorded only in the discussion is ranked at chance (0.491, 95% CI 0.429--0.551).

This is not a restatement of the Secretariat's taxonomy. Because each paper carries exactly one concern label, expected proximity decomposes into the probability that the instrument concerns the paper's own topic and the paper's proximity, through the space, to the instrument's other likely concerns. In a conditional logit stratified by instrument, both components identify documented contributors: the label term at OR 1.93 per standard deviation (95% CI 1.68--2.21) and the geometric term at OR 1.46 (95% CI 1.16--1.84, `p=0.0014`). Run on discussion-only papers, both collapse to null (OR 1.12 and 1.03). Position in the space therefore carries information about routes into formal action that the concern label alone does not, and that information is specific to documented contribution rather than to documentary presence.

Because instrument concerns are assigned from titles, an instrument that reuses the title of the paper proposing it would be classified into that paper's concern partly by construction. Excluding the instrument families where such reuse is institutionalised -- management plans, historic sites, protected areas, and visitor guidelines -- raises the statistic to 0.696 (95% CI 0.604--0.779), the opposite of what title reuse would produce. Controlling for paper--instrument title overlap directly leaves the geometric term essentially unchanged (OR 1.40, 95% CI 1.12--1.77) while attenuating the concern-label term from 1.93 to 1.64. Lexical reuse therefore accounts for part of what the concern label contributes and almost none of what position in the space contributes.

### Attention reaches outcomes through concern-specific pathways

Annual ATCM outputs are distributed unevenly across the concern space. We independently positioned 740 titled Recommendations, Measures, Decisions, and Resolutions from 1961--2025 by classifying their titles against the Secretariat's 45 concerns. The classifier was trained on submitted-paper titles but did not use paper--outcome links; cross-validation holding out complete meetings recovered the exact concern in 63.2% of cases, placed 80.5% within the top three, and placed 81.3% in the same broad region of the space. We therefore retain the full probability distribution over concerns rather than treating a single assigned label as known.

Verified documentary lineage shows where attention reaches formal action. In a cumulative-lagged space constructed only from prior records, papers documented as contributing to adoption lie at the 67.5th percentile of proximity to their outcome concern when compared with all categorized papers submitted at the same meeting. An outcome-balanced randomization gives a 50.3rd-percentile expectation (`p=0.0001`; 133 links to 95 outcomes). The result strengthens when routine management-plan, historic-site, area-protection, and visitor-guideline outputs are excluded (72.0th percentile; `p=0.0001`) and remains under high-confidence outcome classification (68.0th percentile; `p=0.0001`). Papers recorded only as proposals or discussion inputs show no comparable alignment (51.3rd percentile; `p=0.357`). Spatial alignment therefore characterizes documents that reach adoption, not documentary attention in general.

The full concern--meeting record shows the corresponding temporal pattern. With concern and meeting fixed effects, a one-standard-deviation increase in papers submitted to a concern during the preceding three ATCMs is associated with 29.5% more probability-weighted output in that concern at the next meeting (IRR 1.295, 95% CI 1.093--1.534). The estimate survives hard assignment, high-confidence-only coding, and removal of routine site administration. Attention in the five nearest concerns is more weakly and imprecisely associated with later output (IRR 1.148, 95% CI 0.948--1.390), leaving adjacent-concern spillover inconclusive; the own- and nearby-concern coefficients are not statistically distinguishable (`p=0.43`).

Instruments differ in how quickly they respond. Over the years in which all three exist (1995--2025), Resolutions track attention at the adopting meeting itself (IRR 1.518, 95% CI 1.208--1.909), Decisions track it at no window, and Measures track attention eight to fifteen years earlier (IRR 1.744, 95% CI 1.119--2.720; IRR 3.669 under hard assignment) while showing nothing at one to three years. The mirror-image lead window carries the opposite sign (IRR 0.608, 95% CI 0.428--0.863), so the Measure result is asymmetric in time rather than a shared trend. Documentary attention therefore enters the annual formal record on instrument-specific timescales, and the legally binding instrument is the slowest of the three.

Measures deserve separate treatment because of what they are. A functional classification of all 279 assigns 273 (97.8%) to recurring site administration -- management plans, historic sites, and protected-area designations -- so this is a result about the renewal of a protected-area register rather than about general agenda-to-law conversion. Two-thirds of Measures inherit from a recovered earlier instrument, with strong transformation relations arriving after a median of six years and contextual references after fourteen. Adjusting for those predecessors removes 28% of the log-IRR of the eight-to-fifteen-year paper term and widens its interval across one (IRR 1.489, 95% CI 0.918--2.417), which is consistent with part of the pathway running through formal precedent. Along that lineage the space remains informative: paths that open with an adoption-linked paper, run through an intermediate instrument, and end at a Measure sit at the 75.0th percentile of concern proximity against a time- and source-matched null of 50.4 (`p=0.0001`; 317 paths to 120 Measures), whereas paths opening with a discussion-only paper sit at chance (52.2%, `p=0.369`).

Actor position adds a different constraint. Across 95 independently positioned outcomes with verified contributing papers, the corrected model treats submitted papers as the opportunity denominator. Papers matching the output concern (OR 1.60, 95% CI 1.15--2.21) and papers on nearby concerns (OR 1.59, 95% CI 1.13--2.25) have higher documented-link rates. Neither portfolio proximity at the immediately preceding ATCM (OR 1.04, 95% CI 0.76--1.41) nor the number of concerns covered there (OR 1.07, 95% CI 0.69--1.64) adds a detectable association. The observable route into action is therefore paper-specific rather than secured by prior position alone.

## Figure allocation and captions

### Main figure: evidence synthesis

Use `figures/exploratory_attention_outcome_evidence.pdf` in the main text.

**Caption. Attention reaches formal action through concern-specific pathways.** (A) Within-meeting discrimination for verified paper--output links. Adoption/contribution papers are unusually close to their independently classified output concerns, whereas discussion-only papers are not. (B) Concern- and meeting-fixed-effect estimates across the ordered ATCM sequence. The association is strongest within the adopting meeting. (C) Binomial estimates of linked papers out of all eligible papers for each actor--output pair, with actor and output fixed effects. Papers on the output concern and nearby concerns have higher documented-link rates; earlier actor position does not.

### Main figure: the headline discrimination

Use `figures/exploratory_space_discrimination.pdf` as the lead outcome figure.

**Caption. The space of concerns identifies which documents reach formal action.** (A) Discrimination within a single meeting's opportunity set. For each instrument, every categorized paper submitted to the adopting meeting is ranked by expected concern proximity to the instrument's independently classified concern distribution, in a space estimated only from earlier ATCMs. The statistic is the probability that a linked paper outranks a randomly chosen unlinked paper from the same meeting, averaged equally over instruments; bars are instrument-bootstrap 95% intervals and $n$ is the number of instruments. No parameters are fitted to the lineage. Papers documented as contributing to adoption are ranked above chance; papers recorded only in the discussion are not. Rows three and four are circularity guards: excluding the instrument families where a Measure routinely reuses its source paper's title strengthens the estimate, whereas filtering on lexical similarity attenuates it, because that filter also removes papers genuinely about the instrument's subject. The final two rows delete every paper sharing the instrument's leading concern, so that only off-label geometry can rank the remainder. (B) Conditional logit stratified by instrument, racing the two components of proximity against each other: the probability that the instrument concerns the paper's own topic, and proximity to the instrument's other likely concerns through the space. Adding paper--instrument title overlap as a control attenuates the concern-label term from 1.93 to 1.64 and leaves the geometric term essentially unchanged at 1.40, so lexical reuse explains part of what the label contributes and almost none of what the space contributes. Neither term identifies papers that only joined the discussion.

### Main figure: outcome pathways

Use `figures/exploratory_measure_pathways.pdf` in the main text.

**Caption. Measures follow long formal lineages, not recent attention.** (A) Observed pathway composition for all 279 titled Measures, 1995--2025, from the verified lineage graph. Categories are mutually exclusive and record what the parser recovered: "neither recovered" is a property of the parsed record, not evidence that a Measure arose independently, and "formal predecessor" combines transforming and merely citing relations (143 Measures have at least one transforming predecessor; 39 have only citations). (B) Time from a recovered predecessor to the Measure, separating strong transformation relations (supersedes, amends, pursuant to, designates under) from contextual references (recalls, cites). Densities are normalised within each relation class, so the curves compare shape rather than volume. Dashed lines mark medians; the shaded band is the three-year exposure window used in the concern-year models, which sits below essentially all of the formal inheritance. (C) Expected concern proximity of each observed link in a cumulative-lagged space built only from prior records, expressed as a percentile against alternatives matched on source type, calendar period, lag bin, and availability before the target year. The upper block tests papers reaching any instrument; the lower block tests links reaching a Measure. Points average targets equally, bars are target-bootstrap 95% intervals, grey ticks mark each row's own matched-null expectation, and $n$ is the number of target instruments. Matched pools differ in size and composition between rows, so each point is interpretable only against its own tick, not against the other rows. Documented contribution is concern-proximate at one step and at two; documentary presence alone is not.

### Descriptive map

Use `figures/exploratory_independent_outcomes_map.pdf` as the opening outcome figure or Supplementary overview.

**Caption. Formal outputs are unevenly distributed across the attention space.** The network uses the same concern positions and attention-space edges as Figure 1. Node area is the number of annual Recommendations, Measures, Decisions, or Resolutions whose title is independently assigned primarily to that concern from 1961--2025. Hollow nodes have exactly zero primary annual-output assignments; they do not indicate an absence of governance through other concerns, constitutional instruments, implementation, or informal practice. Dark edges are bootstrap-supported co-specialization ties and dotted edges retain the Figure 1 scaffold; neither edge type is an outcome relation. Of 768 annual outcomes, 740 have titles and enter the map. Four of 45 concerns receive no primary assignment.

### Scope diagnostic

Use `figures/exploratory_outcome_scope_boundary.pdf` in the Supplement or methods presentation.

**Caption. Annual outputs and constitutional instruments form separate outcome layers.** Recommendation XIV-3 (1987) is an annual ATCM output assigned to Drilling. The Madrid Protocol was adopted at SATCM XI-4 in 1991 and is not counted in annual ATCM output totals; Article 7 supplies primary legal context for Mineral resources and secondary research context for Drilling.

### Supplementary diagnostics

- `figures/exploratory_attention_outcome_lag_profile.pdf`: individual lags 0--5; use to show that the timing is irregular and motivates the three-year aggregate.
- `figures/exploratory_independent_outcome_trajectories.pdf`: actor trajectories and verified outcome contributions for Australia, the Netherlands, and Ukraine.

## What the space does not do

The headline claim is a discrimination result, and it is worth being explicit
about the neighbouring claims it does **not** license, because each of them is
tempting and each fails for a specific reason.

**It does not predict whether an outcome will occur.** Every proximity test
conditions on an instrument that exists and asks which document reached it.
Nothing here scores concern--meeting pairs in which no outcome occurred.

**It does not predict which instrument type will result.** Instrument type is
very nearly a function of concern identity in this corpus. Over 1995--2025,
97.5% of Measures fall in three concerns, against 64.7% of Decisions and 33.9%
of Resolutions; predicting the instrument from the concern's modal instrument
alone is 87.3% accurate against a 47.4% majority baseline. Any claim that the
space predicts the instrument would have to beat that lookup, and the concern
fixed effects used throughout the panel models deliberately absorb exactly the
variation such a test would need. The space and concern identity are close to
collinear in a fixed-effects design.

**It does not establish that an actor's position secures access.** Once eligible
papers are the denominator, portfolio proximity at the immediately preceding
ATCM is not associated with the documented-link rate (OR 1.04, 95% CI
0.76--1.41), nor is the number of concerns covered there (OR 1.07, 95% CI
0.69--1.64). The measurable route is in the papers submitted around a specific
output, not in prior position alone.

**It does not identify a causal pathway.** A recovered link records
documentation. Conference Room Papers, drafting groups, and informal bargaining
remain outside the archive.

The open question that would extend the claim is a concern--meeting forecasting
exercise: predict the instrument-type composition of output from information
available before ATCM *t*, split by meeting order, and require the space to add
over a baseline of concern identity plus meeting. That test has not been run.

## Boundaries that should remain explicit

- The title classifier measures alignment with the Secretariat taxonomy; it does not establish the full legal scope of an instrument.
- Twenty-eight outcomes lack usable titles and are excluded from concern assignment.
- Verified direct paper--outcome evidence begins in 1991, although the annual outcome panel begins in 1961.
- Paper activity can reflect agenda entrepreneurship, reporting requirements, mandates, scientific programmes, or strategic choice. The models do not isolate intent.
- Concern and meeting fixed effects remove stable concern differences and changes shared across ATCMs, but do not identify a causal effect of submissions.
- Annual output is not implementation, compliance, environmental effectiveness, or legal strength. The instrument split shows why those distinctions matter.
- The conversion model identifies documented paper links, not negotiating influence. Conference Room Papers and informal bargaining remain outside the archive.
- The Measure functional typology is rule-based on titles and site identifiers. It is a reproducible classification of the whole population, not legal coding, and `family_rule` in the inventory records the pattern that fired for every row.
- Attenuation of a paper coefficient when formal predecessors are added is descriptive. It is consistent with mediation and does not identify a mediated effect.
- The Measure result describes adoption at the ATCM. Approval, entry into effect, and implementation are separate outcomes and are not modelled.
- Six Measures fall outside recurring site administration. Nothing about substantive legal hardening can be estimated from that subsample, and models fitted to it are marked `estimable = False`.
- Instrument concerns are assigned from titles, and some instruments reuse the title of the paper that proposed them. Controlling for paper--instrument title overlap leaves the geometric term intact but attenuates the concern-label term, so title reuse remains a live qualification for the label component of the headline, which is the larger of the two.
- Discrimination operates within a meeting's opportunity set. It does not forecast whether a concern will produce an outcome, or which instrument type will result.

## Reproducibility

Run:

```bash
micromamba run -n ultraplot-dev python scripts/analyze_attention_to_outcomes.py
micromamba run -n ultraplot-dev python scripts/plot_attention_to_outcomes.py
micromamba run -n ultraplot-dev python scripts/verify_attention_to_outcomes.py
micromamba run -n ultraplot-dev python scripts/analyze_measure_pathways.py
micromamba run -n ultraplot-dev python scripts/plot_measure_pathways.py
micromamba run -n ultraplot-dev python scripts/analyze_space_discrimination.py
micromamba run -n ultraplot-dev python scripts/plot_space_discrimination.py
```

The Measure scripts read the classifier artifacts produced by the first script, so run them in this order.

Machine-readable results are under `output/outcome_linkage/`. `analysis_summary.json` collects the complete model output and `verification_report.json` records the invariant checks used for the completion audit. The Measure pathway analysis adds:

| File | Contents |
|---|---|
| `measure_pathway_inventory.csv` | All 279 Measures with family, coding rule, site identifiers, pathway, predecessors, and concern assignment |
| `measure_edge_audit.csv` | Every incoming edge with its relation class and lag |
| `measure_unconnected_audit_sample.csv` | Stratified audit packet for the 83 Measures with no recovered predecessor |
| `measure_pathway_composition.csv` | Pathway shares overall and by family |
| `measure_predecessor_lags.csv` | Lag distributions by relation and source instrument |
| `measure_windows_panel.csv` | Concern-year panel with the pre-specified windows |
| `measure_pathway_models.csv` | All fitted models, with an `estimable` flag |
| `measure_spatial_continuity_edges.csv`, `measure_spatial_continuity_tests.csv` | Matched-null proximity per edge and per edge set |
| `measure_pathway_summary.json` | Machine-readable summary and the selected decision-table row |

The headline discrimination analysis adds:

| File | Contents |
|---|---|
| `space_discrimination_panel.csv` | One row per instrument × paper available at its meeting, with proximity decomposed into same-concern mass and related-concern proximity |
| `space_discrimination_auc.csv` | Outcome-balanced discrimination statistics with bootstrap intervals |
| `space_discrimination_race.csv` | Conditional logit racing label against geometry, with outcome-cluster bootstrap intervals |
| `space_discrimination_summary.json` | Machine-readable summary |

Coding rules are in `MEASURE_CODING_PROTOCOL.md`.
