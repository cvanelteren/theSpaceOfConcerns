// Cover letter for One Earth
// Compile from this directory with:
//   typst compile --root .. cover_letter.typ

#set page(
  paper: "a4",
  margin: (x: 2.3cm, top: 1.4cm, bottom: 1.2cm),
)
#set text(font: "TeX Gyre Pagella", size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.82em, spacing: 0.82em)
#show link: set text(fill: rgb("#0a4a8a"))

#grid(
  columns: (auto, 1fr),
  align: (left + top, right + top),
  image("../apparent_consensus/qut.png", height: 2cm),
  {
    set text(size: 7pt, fill: rgb("#004675"))
    show link: set text(fill: rgb("#004675"))
    set par(justify: false, leading: 0.55em, spacing: 0.45em)
    align(right)[
      *Queensland University of Technology* \
      Faculty of Science \
      School of Mathematics \
      Brisbane, QLD 4001, Australia \
      #v(0.3em)
      Casper van Elteren \
      #link("mailto:caspervanelteren@gmail.com")[caspervanelteren\@gmail.com] \
      #datetime.today().display("[day] [month repr:long] [year]")
    ]
  },
)

#v(0.4em)

Dear Editors,

On behalf of my co-authors, I am pleased to submit _"Attention before agreement forecasts the focus of soft-law action"_ for consideration in _One Earth_.

International institutions invest attention before they act, yet this early stage is rarely visible. We use 6,573 papers submitted to Antarctic Treaty Consultative Meetings between 1961 and 2025 to reconstruct how documentary attention is organized. Two concerns are close when the same actors devote a disproportionate share of their papers to both. Earlier portfolios predict where relative attention shifts next: moving 0.1 farther from an actor's earlier portfolio is associated with 18% lower odds of later specialization, and realized shifts are nearer than popularity-weighted alternatives in 62.5% of actor-period comparisons.

We then ask whether the distribution of attention contains information about adopted output. The analysis covers all 584 Measures, Decisions, and Resolutions adopted at regular meetings from 1995 to 2025. After selecting settings on earlier meetings, direct and network-weighted paper attention improve rolling-origin forecasts of the category distribution of non-binding Resolutions across ATCMs 29--47. The model improves 15 of 19 meetings, and the observed concern map outperforms all 200 coherently shuffled maps. The same model does not improve Measures or Decisions.

Measures follow a narrower formal record. Of 277 Measures, 245 cite an earlier Measure, Decision, or Resolution in their adopted body text, and 231 cite an earlier Measure. This distinction is central to the paper. A concern network can reveal where documentary attention is likely to move, but that signal should not be mistaken for binding adaptation. Formal inheritance, entry into force, implementation, and environmental outcomes require separate evidence.

The Antarctic Treaty System makes this distinction unusually observable. It has governed a globally important environmental region by consensus for more than six decades, and its public record contains both submitted papers and adopted instruments with different legal force. The result offers a diagnostic for other document-rich consensus institutions: monitor concern networks to see where attention is moving, then track legal status and implementation separately to determine whether the institution is adapting.

We believe this combination of a globally consequential case, a transferable method, and a direct comparison between attention and instruments of different legal force fits _One Earth_'s interest in how institutions respond to sustainability challenges. The paper does not infer bargaining or implementation from documents. Instead, it shows which parts of institutional adaptation can be observed before agreement and which require later evidence.

The manuscript is original and is not under consideration elsewhere. All authors have approved the submission and declare no competing interests. A self-contained review archive with the reconstructed paper panel, pinned official output records, analysis code, and derived tables accompanies the submission. The same materials will be added to the existing Zenodo record before publication.

Thank you for considering our manuscript.

#v(0.15em)
#grid(
  columns: (0.34fr,),
  column-gutter: 1em,
  align: top,
  [
    Sincerely, \
    Casper van Elteren \
    On behalf of all authors
  ],
)
