#!/usr/bin/env python3
"""Retired gate for the discarded paper--output lineage analysis.

The former gate validated coding consistency but not whether each parsed
paper--output relation was supported by its source passage. Instrument numbers
repeat across years, and the old parser sometimes resolved an earlier cited
instrument to a same-number output at the current meeting. The manuscript no
longer uses those relations or their model-consensus labels.

Use ``scripts/verify_attention_to_outcomes.py`` for the current publication
gate. The lineage project requires pair-level source validation before this
file can become an active gate again.
"""

raise SystemExit(
    "RETIRED: lineage relation validity was not established; "
    "run scripts/verify_attention_to_outcomes.py instead"
)
