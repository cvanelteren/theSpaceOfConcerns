# Blinded coding protocol for formal Antarctic Treaty outputs

## Purpose and blindness

Assign each formal-output title to the closest concern in the fixed 45-label
Secretariat taxonomy. Code from `outcome_consensus_validation_blind.csv` only.
The packet retains year and instrument type as identifiers, but concern
decisions must follow the visible title text.
Coders must not inspect classifier predictions, probabilities, confidence
scores, paper--output lineages, model keys, or another coder's file. Do not use
external searches or recover missing text from the instrument identifier. The
validation asks what the title itself supports.

## Required fields

For every `validation_id`, record:

- `primary_concern`: exactly one codebook label, `INSUFFICIENT_TITLE`, or
  `OUTSIDE_TAXONOMY`;
- `secondary_concern`: one codebook label or blank;
- `confidence`: `high`, `medium`, or `low`;
- `rationale`: a short title-based reason.

Use `INSUFFICIENT_TITLE` when the visible title is truncated, is only a page
reference, or does not say what the output concerns. Use `OUTSIDE_TAXONOMY`
when the subject is identifiable but none of the 45 labels is defensible.
Abstention is evidence about coverage and must not be replaced by a forced
guess.

## Decision rules

1. Code the concern governed by the output, not the instrument type.
2. Prefer the most specific explicit label. Use a general label only when the
   title does not support a narrower one.
3. A named individual ASPA/ASMA designation, revision, or management plan is
   `Management Plans`; system-wide protected-area rules are
   `Area Protection and Management Plans General`.
4. A named historic site or monument is `Historic Sites and Monuments`, even
   when a list or site record is being revised.
5. Rules, consultative status, Treaty symbols, compilation of Treaty documents,
   and general meeting machinery are
   `Operation of the Antarctic Treaty system General`. Secretariat budgets,
   programmes, audits, and Secretariat reports are
   `Operation of the Antarctic Treaty system The Secretariat`. CEP membership,
   observers, procedure, and administration are `Operation of the CEP`.
6. Use `Institutional and legal matters` for a legal or institutional question
   not assigned to a named operational category. Specific legal subjects such
   as liability retain their specific label.
7. Use `Cooperation with Other Organisations` when the action primarily
   supports, communicates, or coordinates with another organization. Use the
   substantive concern as secondary when explicit.
8. Use `Environmental Protection General` for broad environmental protection
   or Protocol adherence only when no specific environmental concern controls.
9. `Opening statements` applies only to opening statements, not generic
   procedural outputs. `Operational issues` covers logistics and general
   operations; use `Safety and Operations in Antarctica` for explicit safety
   or aviation-safety matters and `Search and Rescue` for explicit SAR.
10. Add a secondary concern only when the title explicitly and materially spans
    two labels. Do not use it merely to record uncertainty between synonyms.

## Fixed concern codebook

| Concern | Scope for title coding |
|---|---|
| Area Protection and Management Plans General | Protected-area system, framework, numbering, or general plan guidance |
| Biological Prospecting | Bioprospecting, genetic or biochemical resource prospecting |
| CEP Strategy Discussions | CEP priorities or strategic agenda |
| Climate Change | Climate processes, impacts, mitigation, or climate research |
| Comprehensive Environmental Evaluations | Explicit comprehensive environmental evaluation or CEE |
| Cooperation with Other Organisations | Coordination, letters, or formal support involving another organization |
| Drilling | Scientific or other drilling |
| Educational issues | Education, outreach, or cultural activity |
| Emergency report and contingency planning | Emergency response plans or contingency planning, except liability |
| Environmental Domains Analysis | Environmental domains, biogeographic regionalization, or comparable spatial framework |
| Environmental Impact Assessment EIA Other EIA Matters | Environmental impact assessment other than explicit CEE |
| Environmental Monitoring and Reporting | Environmental monitoring or general environmental reporting |
| Environmental Protection General | Broad environmental protection or general Protocol adherence |
| Exchange of Information | Information exchange duties, formats, or EIES |
| Fauna and Flora_General | General fauna/flora conservation not covered by a narrower species label |
| Historic Sites and Monuments | Historic sites, monuments, remains, or their list |
| Human Footprint and wilderness values | Wilderness, footprint, cumulative human presence, or landscape values |
| Inspections | Inspection systems, reports, or checklists |
| Institutional and legal matters | Legal/institutional questions lacking a more specific label |
| International Polar Year | Explicit IPY activity or legacy |
| Liability | Liability, especially environmental-emergency liability |
| Management Plans | Individual ASPA/ASMA designation, revision, expiry, or management plan |
| Marine Acoustics | Marine acoustic activity or impacts |
| Marine Protected Areas | Marine protected areas or MPA planning |
| Marine living resources | Fishing, sealing, harvest, or conservation of marine living resources |
| Mineral resources | Mineral-resource activity, regulation, or prohibition |
| Multiyear strategic workplan | ATCM multi-year strategic work plan |
| Nonnative Species and Quarantine | Non-native species, biosecurity, or quarantine |
| Opening statements | Meeting opening statements only |
| Operation of the Antarctic Treaty system General | General ATCM/Treaty procedure, status, rules, symbols, or document system |
| Operation of the Antarctic Treaty system Reports | Reports explicitly about operation of the Treaty System |
| Operation of the Antarctic Treaty system The Secretariat | Secretariat budget, programme, audit, administration, or report |
| Operation of the CEP | CEP procedure, membership, observers, rules, or administration |
| Operational issues | Logistics, transport, communications, or general operations without a safety focus |
| Prevention of marine pollution | Oil or other marine pollution prevention |
| Repair and remediation of environmental damage | Environmental repair, cleanup, or remediation |
| Safety and Operations in Antarctica | Operational safety, aviation safety, or safe field operations |
| Science issues | General scientific research or facilitation without a narrower concern |
| Search and Rescue | Search-and-rescue arrangements or capability |
| Site Guidelines for Visitors | Visitor-site guidelines, landing guidance, or post-visit site reporting |
| Specially Protected Species | Explicit specially protected species or designated species protection |
| State of the Antarctic Environment Report SAER | State of the Antarctic Environment Report |
| Sub glacial Lakes | Sub-glacial lakes |
| Tourism and NG_Activities | Tourism, visitor activity, vessels, or other non-governmental activity |
| Waste management and disposal | Waste handling, disposal, or cleanup where waste is the governing subject |

## Consensus rule

Three coders first work independently. Unanimous primary labels are accepted.
All other records are sent, without model information, to adversarial
adjudication. The adjudicator must state the strongest title-based case for
each distinct proposal before choosing a label or abstention. A fresh consensus
review then checks the adjudicated label against the codebook. If those two
decisions differ, another blinded arbitration round is required; no majority
label is accepted silently.

## Supplemental lineage coverage

The same frozen rules were applied to
`outcome_consensus_supplement_blind.csv`, which contains the additional titles
needed to cover every output in the adoption and discussion lineage
comparison. Only the input and output filenames changed.
