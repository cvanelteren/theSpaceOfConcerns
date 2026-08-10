"""Which founding instrument governs each concern.

AUTHOR REVIEW REQUIRED. These are legal judgements, not data. Three of them were
already wrong on a first pass (wilderness, educational issues, tourism), so treat
every row as provisional until checked against the instruments themselves.

Each entry is (instrument, tier, citation):
  direct   -- the concern IS the instrument's subject matter
  indirect -- regulated THROUGH an instrument without being enumerated by it
  none     -- no founding-document basis; arrived via Recommendations, Measures,
              Resolutions, ATCM practice, or the science programme
"""

AT, PROT = "Antarctic Treaty", "Protocol (body)"
A1, A2, A3 = "Annex I (EIA)", "Annex II (fauna & flora)", "Annex III (waste)"
A4, A5, A6 = "Annex IV (marine pollution)", "Annex V (area protection)", "Annex VI (liability)"
CCAMLR, NONE = "CCAMLR", "no founding-document home"

MAP = {
    # ---- Antarctic Treaty 1959 -------------------------------------------
    "Science issues":                        (AT, "direct",   "Art II"),
    "Exchange of Information":               (AT, "direct",   "Art III(1)"),
    "Cooperation with Other Organisations":  (AT, "direct",   "Art III(2)"),
    "Inspections":                           (AT, "direct",   "Art VII"),
    "International Polar Year":              (AT, "indirect", "Art II/III, a science programme"),
    "Operational issues":                    (AT, "indirect", "Art III(1)(a); no operations annex exists"),
    "Safety and Operations in Antarctica":   (AT, "indirect", "Art VII(5) advance notice"),
    "Institutional and legal matters":       (AT, "indirect", "Art IX competence generally"),
    "Operation of the Antarctic Treaty system General":       (AT, "indirect", "Art IX, ATCM procedure"),
    "Operation of the Antarctic Treaty system Reports":       (AT, "indirect", "Art IX, ATCM procedure"),
    "Operation of the Antarctic Treaty system The Secretariat":(AT, "indirect", "Measure 1 (2003), not the Treaty"),
    "Multiyear strategic workplan":          (AT, "indirect", "ATCM Decision, practice"),

    # ---- Madrid Protocol body --------------------------------------------
    "Environmental Protection General":      (PROT, "direct", "Art 2, 3"),
    "Human Footprint and wilderness values": (PROT, "direct", "Art 3(1) names wilderness and aesthetic values"),
    "Educational issues":                    (PROT, "direct", "Art 6(1)(a) cooperative programmes of educational value"),
    "Tourism and NG_Activities":             (PROT, "direct", "Art 8(2) names tourism explicitly"),
    "Environmental Monitoring and Reporting":(PROT, "direct", "Art 3(2)(d) monitoring"),
    "Operation of the CEP":                  (PROT, "direct", "Art 11 establishes the Committee"),
    "Emergency report and contingency planning": (PROT, "direct", "Art 15 response action"),
    "Mineral resources":                     (PROT, "direct", "Art 7 prohibition"),
    "Drilling":                              (PROT, "indirect", "Art 7 defines its status; science drilling under AT Art II"),
    "CEP Strategy Discussions":              (PROT, "indirect", "Art 11, Committee practice"),
    "State of the Antarctic Environment Report SAER": (PROT, "indirect", "Art 3 / CEP, never a standing obligation"),

    # ---- Protocol annexes -------------------------------------------------
    "Environmental Impact Assessment EIA Other EIA Matters": (A1, "direct", "Annex I"),
    "Comprehensive Environmental Evaluations": (A1, "direct", "Annex I, the CEE procedure"),
    "Sub glacial Lakes":                     (A1, "indirect", "Annex I EIA applies; SCAR codes of conduct"),
    "Fauna and Flora_General":               (A2, "direct",   "Annex II"),
    "Specially Protected Species":           (A2, "direct",   "Annex II Appendix A"),
    "Nonnative Species and Quarantine":      (A2, "direct",   "Annex II Art 4"),
    "Marine Acoustics":                      (A2, "indirect", "Annex II Art 3 harmful interference; not enumerated"),
    "Waste management and disposal":         (A3, "direct",   "Annex III"),
    "Prevention of marine pollution":        (A4, "direct",   "Annex IV"),
    "Area Protection and Management Plans General": (A5, "direct", "Annex V, ASPAs and ASMAs"),
    "Management Plans":                      (A5, "direct",   "Annex V"),
    "Historic Sites and Monuments":          (A5, "direct",   "Annex V Art 8"),
    "Environmental Domains Analysis":        (A5, "indirect", "classification tool supporting Annex V"),
    "Site Guidelines for Visitors":          (A5, "indirect", "adopted as Resolutions, not an Annex V instrument"),
    "Liability":                             (A6, "direct",   "Annex VI (not in force)"),
    "Repair and remediation of environmental damage": (A6, "direct", "Annex VI / Protocol Art 15"),

    # ---- other conventions -------------------------------------------------
    "Marine living resources":               (CCAMLR, "direct", "CCAMLR 1980; also AT Art IX(1)(f)"),
    "Marine Protected Areas":                (CCAMLR, "contested", "DUAL: Annex V marine ASPAs vs CCAMLR MPAs"),

    # ---- no founding-document home ------------------------------------------
    "Opening statements":                    (NONE, "none", "pure ATCM practice"),
    "Climate Change":                        (NONE, "none", "no instrument; reaches the agenda via CEP and science"),
    "Biological Prospecting":                (NONE, "none", "known legal gap: AT Art II freedom of science, or unregulated"),
    "Search and Rescue":                     (NONE, "none", "largely IMO/ICAO; no Antarctic instrument"),
}

ORDER = [AT, PROT, A1, A2, A3, A4, A5, A6, CCAMLR, NONE]

# colourblind-safe categorical; grey reserved for the concerns with no legal home
COLOR = {
    AT:     "#1F4E79",   # Treaty            deep blue
    PROT:   "#2E7D5B",   # Protocol body     green
    A1:     "#D98324",   # Annex I           orange
    A2:     "#7BAF3F",   # Annex II          leaf
    A3:     "#8C5A3C",   # Annex III         brown
    A4:     "#3FA9B5",   # Annex IV          teal
    A5:     "#7B3294",   # Annex V           purple
    A6:     "#B5544B",   # Annex VI          red
    CCAMLR: "#E0B33A",   # CCAMLR            gold
    NONE:   "#B9BEC4",   # no legal home     grey
}


def instrument_of(topic):
    return MAP[topic][0]


def tier_of(topic):
    return MAP[topic][1]


if __name__ == "__main__":
    import collections
    from concern_classes import load_classes
    topics = load_classes()["topics"]
    missing = [t for t in topics if t not in MAP]
    extra = [t for t in MAP if t not in topics]
    print(f"unmapped: {missing or 'none'}   not in archive: {extra or 'none'}")
    c = collections.Counter(tier_of(t) for t in topics)
    print("tiers:", dict(c))
    by = collections.defaultdict(list)
    for t in topics:
        by[instrument_of(t)].append(t)
    for k in ORDER:
        print(f"\n{k} ({len(by[k])})")
        for t in sorted(by[k]):
            print(f"    [{tier_of(t):9s}] {t}  -- {MAP[t][2]}")
