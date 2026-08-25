# Coding protocol: Measure families, site identities, and predecessor relations

This protocol governs `scripts/analyze_measure_pathways.py`. It is written so
that an author can reproduce, audit, or override every assignment. All rules
are applied to the 279 titled Measures in `decision_map_verified.json`
(1995--2025); no Measure is dropped.

Assignments are rule-based rather than hand-coded. That is a deliberate
trade-off: it makes the whole population reproducibly classified rather than a
sample, and it records the exact pattern that fired for every row. It is not a
substitute for legal coding. `measure_pathway_inventory.csv` carries a
`family_rule` column precisely so that manual override is a single edit.

## 1. Functional typology

Each Measure receives exactly one family. Rules are applied in order and the
first match wins, so **rule order is part of the definition**.

| Order | Family | Inclusion rule | Rationale |
|---|---|---|---|
| 1 | `environment_or_liability` | title mentions liability, environmental emergencies, an Annex to the Environment Protocol, or specially protected species | Substantive environmental rule-making |
| 2 | `tourism_safety_operations` | title mentions tourism, tourists, non-governmental activity, landing of persons, passenger vessels, insurance, contingency planning, shipborne activity, air safety, or vessel operations | Regulation of activity rather than of place |
| 3 | `historic_site` | title mentions a historic site, historic monument, monument, or HSM | Historic Sites and Monuments register |
| 4 | `management_plan` | title mentions a management plan | Adoption, revision, or revocation of a plan |
| 5 | `protected_area_designation` | title mentions a protected area, specially managed area, ASPA, ASMA, SSSI, site of special scientific interest, expiry date, or a numbered SPA | Designation or boundary change without a new plan |
| 6 | `other_substantive` | nothing above matched | Residual |

Substantive families are tested **before** site vocabulary. Two title patterns
make this necessary:

- *Specially Protected Species: Fur Seals* (2006) contains "specially
  protected" but designates a species, not an area. Rule 1 catches it first.
- *Revised List of Antarctic Historic Sites and Monuments: Wreck of Sir Ernest
  Shackleton's vessel* Endurance (2019) contains "vessel". Rule 2 therefore
  requires an operational context (`passenger vessel`, `vessel operation`)
  rather than the bare word, so the Measure falls through to rule 3.

`recurring_site_administration` is true for families 3, 4, and 5. This flag,
not the Secretariat concern label, defines the family split in the models.

### Result

| Family | Measures |
|---|---:|
| `management_plan` | 218 |
| `historic_site` | 29 |
| `protected_area_designation` | 26 |
| `environment_or_liability` | 3 |
| `tourism_safety_operations` | 2 |
| `other_substantive` | 1 |
| **Recurring site administration (3+4+5)** | **273 of 279 (97.8%)** |

The six non-site Measures are the Secretariat instrument (2003), insurance and
contingency planning for tourism (2004), Annex VI on liability (2005),
specially protected species (2006), landing of persons from passenger vessels
(2009), and the amendment of Annex II (2009).

## 2. Site identity parsing

Site identifiers are extracted as `<TYPE> <NUMBER>` for ASPA, ASMA, SSSI, SPA,
and HSM. Two forms are recognised:

1. an explicit type followed by an optional `No.`/`Number` and a number
   (*Antarctic Specially Protected Area No 138*, *ASPA 116*, *SSSI 11*);
2. list continuations, where a title names the type once and then lists further
   numbers each followed by a parenthesised site name (*ASPA 106 (Cape
   Hallett), 107 (Emperor Island), 108 (Green Island)*). A bare number followed
   by an opening parenthesis inherits the most recent preceding type.

230 of 279 Measures yield at least one identifier, covering 118 distinct sites.

The 42 recurring-administration Measures without an identifier are almost all
register-wide instruments from 1995--2007 whose titles name no numbered site
(*Antarctic Protected Areas System: Management Plans for Specially Protected
Areas*), plus a handful of titles truncated in the source record. Absence of an
identifier is a property of the title, not evidence that no site is involved.

**Numbering caution.** Site numbers are stable across time; instrument numbers
are not. Measure 1 (1995) and Measure 1 (1997) are different instruments.
Measures are therefore always keyed by the full `Measure N (YYYY)` label.

## 3. Predecessor relation audit

Incoming edges to a Measure in `decision_map_verified.json` are split into four
classes. The split between strong and contextual outcome relations is the one
that matters for interpretation: a cited outcome may supply background law
rather than function as a step in a pathway.

| Class | Relations | Meaning |
|---|---|---|
| `strong_transformation` | `supersedes`, `amends`, `pursuant_to`, `designates_under` | The Measure transforms or executes a specific earlier instrument |
| `contextual_reference` | `recalls`, `cites` | The Measure situates itself in a legal context |
| `paper_adoption_or_contribution` | `direct_adoption_or_approval`, `documented_contribution` | A paper is documented as contributing to adoption |
| `paper_proposal_or_discussion` | `direct_proposal_or_discussion` | A paper appears in the discussion record only |

Edge direction in the source graph runs predecessor → successor, so incoming
edges are the Measure's recovered predecessors. Lag is the target year minus
the source year; paper edges are same-meeting by construction and carry lag 0.

## 4. Pathway assignment

Each Measure receives exactly one observed pathway from its recovered
predecessors: `paper_only`, `outcome_only`, `both`, or `neither`. These are
categories of the parsed record. **`neither` does not mean the Measure arose
independently**; it means no predecessor survived parsing. Conference Room
Papers, drafting groups, annexes, and unrecovered report language all produce
`neither`.

`measure_unconnected_audit_sample.csv` therefore draws a stratified sample of
the unconnected Measures, four per functional family × period cell
(1995--2009, 2010--2025), with blank `coder_verdict` and `coder_note` columns
for author completion before the absence is interpreted.

## 5. What is fixed before estimation

The following were fixed before any coefficient was inspected, and are
constants at the top of the script:

- exposure windows: same meeting, 1--3, 4--7, and 8--15 years prior;
- the strong/contextual relation split above;
- the family split at `recurring_site_administration`;
- the 1995 start year for all Measure models;
- `MINIMUM_EVENTS_FOR_INFERENCE = 30`, below which a fixed-effect Poisson on
  this panel is reported but marked `estimable = False`.

Two sensitivity checks were added after the main estimates were inspected and
are labelled as such: hard top-one Measure counts in place of probability mass,
and the mirror-image lead-window placebo. The placebo replaced a
concern-specific linear trend model, which is not estimable on this panel --
45 trends on top of concern and year effects yield a singular design when 174
of 279 Measures fall in one concern.

That last gate matters here. Only six Measures fall outside recurring site
administration, so the "other Measures" models are fitted for transparency and
produce incidence-rate ratios in the thousands with intervals spanning several
orders of magnitude. They are not interpretable and no conclusion reads them.
