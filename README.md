# The Space of Concerns

Code and derived data for *The Space of Concerns*, an analysis of 6,573 working
papers and information documents submitted to the Antarctic Treaty Consultative
Meetings between 1961 and 2025.

The paper reconstructs a **space of concerns**: a network in which two policy
topics are linked when the same actors specialize in both more than their usual
rate, estimated with relatedness measures adapted from economic complexity. It
uses that network to show that actor positions are distinct and tenure-linked,
that portfolios expand locally, and that a simple retain-and-adopt rule
reproduces the record.

## Layout

| Path | Contents |
|---|---|
| `theSpaceOfConcerns.tex` | The manuscript. Builds with `latexmk -pdf`. |
| `fig0*.py`, `figS*.py` | One script per figure; each writes to `figures/`. |
| `figstyle.py` | Shared colour and typography vocabulary (see below). |
| `utils.py` | RPA, proximity and space construction. |
| `scripts/` | Hazard specifications, circularity checks, data guard. |
| `output/` | Derived data read by the figure scripts. |
| `figures/` | Generated PDFs, included by the manuscript. |

## Requirements

Python 3.13 with numpy, pandas, scipy, statsmodels, networkx, pyarrow and
[UltraPlot](https://github.com/Ultraplot/UltraPlot). The environment used for
all results in the paper is a micromamba environment named `ultraplot-dev`:

```bash
micromamba run -n ultraplot-dev python fig02_where_actors_sit.py
```

## The one external dependency

Most figures read only from `output/`, which is committed, and run against a
fresh clone with no further setup:

```bash
micromamba run -n ultraplot-dev python fig02_where_actors_sit.py   # works
micromamba run -n ultraplot-dev python fig05_who_enters_first.py   # works
```

Four scripts — `fig01_space_of_concerns_topology.py`,
`fig03_dynamics_are_local.py`, `fig04_retain_and_adopt.py` and
`fig45_portfolio_space_ridgelines.py` — rebuild the space from the raw
submission table and additionally need:

```
antarctic-database-go/data/processed/document-summary.parquet
```

That table is produced by the document scraper at
<https://github.com/carlohamalainen/antarctic-database-go>; it is a generated
artifact and is **not** tracked in that repository, so cloning the scraper is
not by itself sufficient — the pipeline has to be run to produce it. The
archived copy accompanying the paper is the authoritative one:

> Zenodo record: <https://doi.org/10.5281/zenodo.20821775>

Place it at the path above (a symlink to a checkout elsewhere is fine, which is
how the development tree is arranged) and the remaining scripts will run.

## Checking the derived data is complete

`output/` holds roughly 150 MB of intermediates locally, of which about 6 MB is
read by the figure scripts. `.gitignore` therefore ignores `output/*` wholesale
and the needed files are force-added. That is only safe if something verifies
it, so:

```bash
micromamba run -n ultraplot-dev python scripts/check_required_data.py
```

fails if any figure script references a data file that is not committed. Run it
after adding or changing a figure script. Without it, a new script can
reference a file that exists locally, pass every check on the author's machine,
and fail for everyone else.

## Colour

`figstyle.py` is the single source of truth, and it enforces one rule worth
knowing before editing any figure: the Okabe-Ito **orange / blue / green triple
is reserved** and means *engagement mode* everywhere in the paper. Topic themes
in Figure 1 use a darker, lower-chroma palette chosen to avoid those hue
angles, and any other nominal variable needs its own family. Check separation
with:

```bash
micromamba run -n ultraplot-dev python figstyle.py
```

which prints the closest theme/mode pair as a CIE76 ΔE. Below about 15, two
categorical fills read as the same colour.

## Citation

See `theSpaceOfConcerns.bib`. Data and the versioned code release are archived
in the Zenodo record linked above.
