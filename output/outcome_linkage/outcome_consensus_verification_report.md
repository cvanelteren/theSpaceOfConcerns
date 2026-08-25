# Formal-output consensus verification

Overall: **PASS**

- PASS: final consensus has 246 unique titles (rows=246, validation_ids=246, outcomes=246)
- PASS: all final labels resolved or explicitly abstained (abstentions=16)
- PASS: first-pass consensus count (unanimous=232/246)
- PASS: all non-unanimous titles adjudicated (expected 14 argument-first adjudications)
- PASS: all independent challenges arbitrated (expected 4 anonymous final arbitrations)
- PASS: complete lineage set represented (lineage titles=157; codable=149)
- PASS: sensitivity summary uses lineage scope (coded=149; abstained=8)
- PASS: population-weighted classifier agreement recomputed from title rows (coverage=0.915856; exact|assignable=0.778772; top3|assignable=0.947078)
- PASS: consensus paper-link estimates recomputed from candidate rows (adoption=0.660178 (n=89); discussion=0.501821 (n=79); off-label=0.535316 (n=64))
- PASS: matched classifier comparison uses the consensus-codable output set (same output IDs; adoption=0.671903; discussion=0.498074)
- PASS: title-overlap-adjusted nearby effect (OR=1.260296, CI=[0.981113,1.618921])
- PASS: prior-portfolio result remains null (OR=1.037881, CI=[0.794814,1.355283], outputs=89)
- PASS: proximity direction is native phi and higher scores rank as closer (row-level calculation awards a win when linked-paper phi exceeds unlinked-paper phi; distance is 1-phi)
- PASS: manuscript reports model-consensus results (all required result strings present)
