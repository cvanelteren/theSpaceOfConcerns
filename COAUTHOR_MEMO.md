# Space of Concerns — what changed since the version on Overleaf

**Baseline:** `81987fe`, 28 July 2026 ("Resolve rebase and refine One Earth framing") — the version currently in Overleaf.
**Current:** `9f63e93`. Fifteen commits.
**Attached:** `coauthor_diff.pdf` (latexdiff, 37 pp.) and the current build.

Short version: I did a correctness pass, found five errors that were in the
version we were about to send, and in fixing them concluded the framing was
also costing us. **No analysis was rerun and no estimate changed.** Three
decisions below are yours, not mine.

---

## 1. Corrections — these were errors

Not framing preferences. Each was in the Overleaf version.

| | Was | Is | How found |
|---|---|---|---|
| Retain-and-adopt entry rank | prose said simulated **0.679**, caption said 0.675 | **0.675** everywhere | `split_support` mean in the process-uncertainty table is 0.675314; the prose was stale |
| Figure 5 caption | "the fitted line in each panel is pooled across modes"; "actors with broader portfolios enter emerging concerns earlier (A)" | describes what the panels show | Panel B has **no line**; panel A's is a LOWESS smoother. The breadth margin is **ρ = −0.007, p = 0.95** — the caption asserted an effect the data contradicts |
| Figure 3 caption | placed the "concern axis" in Figure 1 | Figure 2A | Figure 1 is a 2D network layout; the 1D axis is what Figure 2A plots |
| Figure 3A | "spacing is even, panel B carries the real spacing" caveat | restored | Lost when the panel moved from Fig 2 to Fig 3; the SI's analogous figure still carried it, so the paper contradicted itself |
| Figure 5 PDF | stale — predated its own script | rebuilt | Regenerating from byte-identical input produced a different image |

Two more worth knowing about:

- **Missing literature.** Dudeney & Walton (2012, *Polar Research*) and Sánchez
  (2016, *Polar Record*) are quantitative predecessors on this exact archive
  and were uncited. Dudeney & Walton use **fractional attribution as their
  primary counting scheme** — which we present as our own robustness check.
  Both are now cited, and their leadership group is used as external
  validation of Mode 3 rather than left for a reviewer to find.
- **The repository could not regenerate its own figures.** There was no
  `.gitignore`, so 407 of 415 files in `output/` were untracked. All five main
  figures now build from a clean clone; a guard script fails if that regresses.

## 2. Restructuring — these need your sign-off

Each is reversible. Say the word and it goes back.

**(a) Figure 5 demoted to SI; four main figures instead of five.**
Rationale: the earlier-entry result is our weakest evidence — cross-sectional,
R² = 0.24, n = 64, with breadth, anchoring and tenure entangled among the same
small set of long-established actors. It was the closing beat, so the paper got
weaker as it went. It now closes on the retain-and-adopt model. **If your
contribution centred on the pioneer analysis, this is the change to push back
on.**

**(b) The framing moved from method to object.**
Was: "we adapt economic-complexity methods to Antarctic governance."
Now: "there is a relational structure to agenda formation; here it is."
Same evidence, larger claim. This is the change most likely to draw fire, and
it is why the missing ATS predecessors above mattered — under the old framing a
predecessor was harmless, under this one it is not.

**(c) The mode caveats were consolidated.**
The partition limit was stated five times in ~110 lines while three of four
figures were built on it — full commitment plus full apology. The complete
statement now sits once in Results; the rest are one-clause pointers.
Substantively unchanged, but the register is different and you should look at
it.

**Still open, and I'd like a view:** the modes are described as an interpretive
division of a continuum, yet carry a lot of the visual argument. Three coherent
exits — validate the partition externally, demote it to a presentational
device, or own it as a finding. We are currently between them.

## 3. Verification

- Every main-text statistic re-checked against `output/`: tenure and breadth
  correlations, complementary pair count, all five hazard specifications, all
  four entry ranks. All match.
- All five main figures rebuild from a clean clone of `main`; four are
  pixel-identical to the committed PDFs, and the fifth was the stale one above.
- Document builds at 32 pp. with zero undefined references or citations.

## 4. One thing that is new

We asserted that proximity reflects co-engagement "rather than semantic
similarity". That was a statement about provenance — the categories are the
Secretariat's, and no edge is computed from text — which is true by
construction. It left open whether the geometry nonetheless *coincides* with
the wording, which a reviewer will ask.

Tested: lexical similarity explains **1.2%** of the variance in φ (word
n-grams) and **1.8%** (character n-grams), Mantel p = 0.002 and 0.001. The
strongest ties join labels with no shared vocabulary at all — Exchange of
information with Opening statements (0.52), Environmental protection with
Liability (0.48), Inspections with Waste management (0.47) — while some of the
most lexically similar pairs sit far apart. This is now in Results and Methods.

## 5. Proposed next steps

1. Sign-off on 2(a)–(c), and a view on the modes question.
2. Target: Nature Human Behaviour. Abstract is rewritten and now 197 words,
   inside their 200 cap — worth a read, since it is the part an editor uses to
   decide whether to send us out.
3. Reconcile the arXiv preprint, which reports 6,591 documents and a
   binding-law finding that is not in this manuscript. We now say 6,573. A
   reviewer who finds the preprint will see both.
