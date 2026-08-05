"""Shared colour vocabulary for the main-text figures.

The paper encodes two different nominal variables in colour, one figure apart:

  * eight **topic themes**, which colour the nodes of the space in Figure 1
  * three **actor modes**, which colour zones and markers in Figures 2 and 5

Before this module the two palettes were declared independently and drifted
into near-collision -- Figure 1's Environmental Protection green sat next to
Figure 2's Strategy green, Operations & Safety orange next to Coordination
orange, Marine & Wildlife blue next to Compliance blue. Nothing in the data
links those pairs, so a reader carrying colour memory across a page break was
being invited to infer a correspondence that does not exist.

The fix is to treat the mode triple as reserved. It keeps the bright, saturated
Okabe-Ito hues; the theme palette is pushed into a darker, lower-chroma register
and its hue angles are chosen to stay clear of the three reserved ones. The two
sets then read as different families even before a reader looks at a legend.

``max_theme_mode_similarity`` exists so this stays a checkable property rather
than a claim in a docstring -- see ``python figstyle.py``.
"""

from __future__ import annotations

# --- Reserved: actor modes -------------------------------------------------
# Okabe-Ito vermilion / blue / bluish-green. These three hues mean "mode"
# everywhere in the paper and must not be reused for any other variable.
MODE_COLORS: dict[int, str] = {
    1: "#D55E00",  # Coordination
    2: "#0072B2",  # Compliance
    3: "#009E73",  # Strategy
}

MODE_GLOSS: dict[int, str] = {
    1: "Coordination",
    2: "Compliance",
    3: "Strategy",
}

# --- Topic themes (Figure 1) -----------------------------------------------
# Darker and less saturated than the mode triple, and spread around the hue
# circle avoiding the reserved angles (~26 deg, ~202 deg, ~164 deg).
THEME_COLORS: dict[str, str] = {
    "Environmental Protection": "#5B8C3A",   # olive, not the mode's teal-green
    "Marine & Wildlife": "#4C4B8A",          # indigo, not the mode's cyan-blue
    "Operations & Safety": "#C89600",        # gold, not the mode's vermilion
    "Governance & Legal": "#6A3D9A",
    "Science & Research": "#A6274F",
    "Tourism & Human Activity": "#C4699E",
    "Infrastructure & Planning": "#6B4A32",
    "Resource Extraction": "#2E7D7D",
}

# --- Structural regions of the space (Figure 1 callouts) -------------------
# The three branches and the procedural core carried their own ad-hoc blue /
# orange / green, which read as a fourth partition competing with the modes.
# They are aliased onto the theme palette instead: each region is dominated by
# a theme, so the callout and the legend entry now agree, and the reserved mode
# hues stay out of the figure entirely.
BRANCH_COLORS: dict[str, str] = {
    "drilling": THEME_COLORS["Marine & Wildlife"],
    "human_impact": THEME_COLORS["Tourism & Human Activity"],
    "environmental": THEME_COLORS["Environmental Protection"],
    "core": THEME_COLORS["Governance & Legal"],
}

# --- Neutrals --------------------------------------------------------------
# Used where a figure deliberately carries no categorical colour, so that the
# absence of hue is itself consistent across figures.
PRIMARY = "#111827"
MUTED = "#8A93A0"
ENVELOPE = "#D8DCE2"
TEXT = "#374151"

# Off-palette accent for one-off binary highlights (e.g. complementary pairs),
# chosen to belong to neither family above.
ACCENT = "#7B3294"
ACCENT_MUTED = "#C3C7CE"


# --- Typography ------------------------------------------------------------
# Sizes were previously set per-figure and had drifted to 8-9pt for axis labels
# and 7pt for legends. At the widths these figures are placed (0.95\textwidth
# for a three-panel row, so roughly 2.3in per panel) that reduces to about 5pt
# on the page, below the ~7pt floor most publishers set. These are the single
# source of truth; call ``apply_typography`` rather than restyling in a script.
FS_LABEL = 11.0
FS_TICK = 10.0
FS_LEGEND = 9.5
FS_ANNOT = 8.5
FS_PANEL = 12.0


def apply_typography(
    axes,
    *,
    label: float = FS_LABEL,
    tick: float = FS_TICK,
    legend: float = FS_LEGEND,
) -> None:
    """Set axis-label, tick-label and legend sizes on every axes given.

    Accepts a single axes or any iterable of them. Legends are restyled in
    place when present, so this can be called after they are created.
    """
    try:
        iter(axes)
        items = list(axes)
    except TypeError:
        items = [axes]

    for ax in items:
        ax.xaxis.label.set_fontsize(label)
        ax.yaxis.label.set_fontsize(label)
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_fontsize(tick)
        existing = ax.get_legend()
        if existing is not None:
            for text in existing.get_texts():
                text.set_fontsize(legend)
            title = existing.get_title()
            if title is not None and title.get_text():
                title.set_fontsize(legend)


def _srgb_to_lab(hex_color: str) -> tuple[float, float, float]:
    """CIE L*a*b* under D65, so distances approximate perceived difference."""
    h = hex_color.lstrip("#")
    rgb = [int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    r, g, b = lin
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def delta_e(a: str, b: str) -> float:
    """CIE76 colour difference. Below ~15 two categorical fills read as alike."""
    la, aa, ba = _srgb_to_lab(a)
    lb, ab, bb = _srgb_to_lab(b)
    return ((la - lb) ** 2 + (aa - ab) ** 2 + (ba - bb) ** 2) ** 0.5


def max_theme_mode_similarity() -> tuple[str, int, float]:
    """The closest theme/mode pair, as (theme, mode_id, delta_e).

    Returned rather than asserted so callers -- and the module's own main --
    can check that the two palettes stayed separated after any edit.
    """
    worst = min(
        (
            (theme, mode_id, delta_e(theme_hex, mode_hex))
            for theme, theme_hex in THEME_COLORS.items()
            for mode_id, mode_hex in MODE_COLORS.items()
        ),
        key=lambda item: item[2],
    )
    return worst


if __name__ == "__main__":
    theme, mode_id, distance = max_theme_mode_similarity()
    print(f"closest theme/mode pair: {theme} vs {MODE_GLOSS[mode_id]}  dE={distance:.1f}")
    for name, hexes in (("themes", THEME_COLORS), ("modes", MODE_COLORS)):
        print(f"{name}: {list(hexes.values())}")
