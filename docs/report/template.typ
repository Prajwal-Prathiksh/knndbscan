// A small template for technical reports: page style, title block, numbered theorem-like
// environments, proofs, algorithm figures and callout boxes. It uses only Typst built-ins,
// so a report can pick its own packages (pseudocode, diagrams) for the content.
//
//   #import "template.typ": *
//   #show: report.with(title: [...], authors: ((name: "...", affiliation: "..."),), abstract: [...])
//
// Environments are figures, so `@label` references work: #theorem(title: [...])[...] <thm:x>

#let accent = rgb("#1f4e79")
#let muted = luma(100)

// kind: (supplement, background, rule colour)
#let environments = (
  definition: ([Definition], rgb("#f3f6fa"), rgb("#1f4e79")),
  lemma: ([Lemma], rgb("#f4f8f1"), rgb("#3a6b1f")),
  theorem: ([Theorem], rgb("#f4f8f1"), rgb("#3a6b1f")),
  proposition: ([Proposition], rgb("#f4f8f1"), rgb("#3a6b1f")),
  invariant: ([Invariant], rgb("#fbf5ec"), rgb("#8a5a00")),
)

#let environment(kind) = (title: none, body) => figure(
  kind: kind,
  supplement: environments.at(kind).at(0),
  caption: title,
  body,
)
#let definition = environment("definition")
#let lemma = environment("lemma")
#let theorem = environment("theorem")
#let proposition = environment("proposition")
#let invariant = environment("invariant")

#let proof(body) = block(breakable: true, above: 0.6em, below: 1em)[
  _Proof._ #body #h(1fr) $square$
]

// Pseudocode conventions (with lovelace: `pseudocode-list(..pseudocode-style)[...]`):
// keywords in bold, comments as small grey italics after `//`, no end keywords and no
// indentation guides, Require/Ensure lines unnumbered.
#let comment-colour = luma(135)
#let pseudocode-style = (stroke: none, indentation: 1.3em, line-gap: 0.55em)
// A comment at the right end of a line.
#let cmt(body) = {
  h(1fr)
  text(fill: comment-colour, size: 0.88em, style: "italic")[#"//" #body]
}
// A comment on a line of its own.
#let lcmt(body) = text(fill: comment-colour, size: 0.88em, style: "italic")[#"//" #body]
#let require(body) = [*Require:* #body]
#let ensure(body) = [*Ensure:* #body]

// A numbered algorithm; the body is usually a pseudocode list.
#let algorithm(caption: none, body) = figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: caption,
  body,
)

#let callout(title, colour, body) = block(
  width: 100%,
  breakable: true,
  inset: (x: 10pt, y: 8pt),
  radius: 2pt,
  fill: colour.lighten(92%),
  stroke: (left: 2.5pt + colour),
)[
  #text(weight: "bold", fill: colour)[#title.] #body
]
#let note(body, title: [Note]) = callout(title, rgb("#1f4e79"), body)
#let pitfall(body, title: [Pitfall]) = callout(title, rgb("#a1260d"), body)
#let implementation(body, title: [Implementation]) = callout(title, rgb("#5b3f8c"), body)

#let report(
  title: none,
  subtitle: none,
  authors: (),
  date: none,
  abstract: none,
  body,
) = {
  // The PDF's author metadata takes plain strings; names may be given as content.
  let plain(it) = if type(it) == str { it } else if it.has("text") { it.text } else { "" }
  set document(title: title, author: authors.map(a => plain(a.name)))
  set page(
    paper: "a4",
    margin: (x: 2.4cm, top: 2.6cm, bottom: 2.4cm),
    numbering: "1",
    header: context {
      if counter(page).get().first() > 1 {
        set text(8pt, fill: muted)
        title
        h(1fr)
        counter(page).display()
      }
    },
    footer: [],
  )
  set text(font: "Libertinus Serif", size: 10.5pt, lang: "en")
  set par(justify: true, leading: 0.6em, spacing: 1em)
  show math.equation: set text(font: "New Computer Modern Math")
  set math.equation(numbering: "(1)")
  show raw: set text(font: "DejaVu Sans Mono", size: 0.88em)
  show link: set text(fill: accent)
  show ref: set text(fill: accent)

  set heading(numbering: "1.1")
  show heading: set block(above: 1.4em, below: 0.7em)
  show heading.where(level: 1): set text(13.5pt, fill: accent)
  show heading.where(level: 2): set text(11pt, fill: accent)

  set table(stroke: (x, y) => if y == 0 { (bottom: 0.7pt) } else { (bottom: 0.3pt + luma(200)) })
  show table.cell.where(y: 0): set text(weight: "bold")
  show figure.caption: set text(9pt)
  show table.cell: set par(justify: false)
  show table.cell: set text(hyphenate: false)
  show figure: set block(breakable: true)

  // Theorem-like environments and algorithms are drawn as boxes, not floating figures.
  show figure: it => {
    // Ordinary figures (tables, images) have an element function as kind, not a string.
    if type(it.kind) == str and it.kind in environments {
      let (_, fill, rule) = environments.at(it.kind)
      block(
        width: 100%,
        breakable: true,
        inset: (x: 10pt, y: 8pt),
        radius: 2pt,
        fill: fill,
        stroke: (left: 2pt + rule),
        align(left)[
          #text(weight: "bold", fill: rule)[#it.supplement #it.counter.display(it.numbering)]#if it.caption != none [ (#it.caption.body)].
          #it.body
        ],
      )
    } else if it.kind == "algorithm" {
      block(
        width: 100%,
        breakable: true,
        stroke: (top: 0.8pt, bottom: 0.8pt),
        inset: (y: 6pt),
        align(left)[
          #text(weight: "bold")[#it.supplement #it.counter.display(it.numbering)]#if it.caption != none [: #it.caption.body]
          #line(length: 100%, stroke: 0.4pt)
          #it.body
        ],
      )
    } else {
      it
    }
  }

  align(center)[
    #set par(justify: false)
    #set text(hyphenate: false)
    #text(19pt, weight: "bold", title)
    #if subtitle != none {
      v(0.2em)
      text(11.5pt, fill: muted, subtitle)
    }
    #v(0.8em)
    #authors.map(a => [#a.name #if "affiliation" in a [\ #text(9pt, fill: muted, a.affiliation)]]).join(h(3em))
    #if date != none {
      v(0.4em)
      text(9pt, fill: muted, date)
    }
  ]
  if abstract != none {
    v(0.6em)
    block(inset: (x: 2em))[
      #set text(9.5pt)
      #text(weight: "bold")[Abstract.] #abstract
    ]
  }
  v(0.4em)
  block(inset: (x: 2em))[
    #set text(9.5pt)
    #outline(depth: 2, indent: 1.2em)
  ]
  body
}
