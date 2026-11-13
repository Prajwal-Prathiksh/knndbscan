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

// Pseudocode is data, so one algorithm can be drawn in full and again with some lines in
// focus (a storyboard step) without being written twice:
//
//   #let alg = (require[...], ensure[...], ln(0, [*repeat*]), ln(1, [$x <- y$], note: [why]))
//   #pseudocode(alg)                  // every line
//   #pseudocode(alg, focus: (2, 3))   // lines 2-3 highlighted, the rest dimmed
//
// Keywords are bold in the body; notes are drawn as small grey italics after `//` at the
// line's end; there are no end keywords and no indentation guides.
#let comment-colour = luma(135)
#let dimmed-colour = luma(175)
#let focus-colour = rgb("#fdf0c8")
#let focus-rule = rgb("#d39b00")

// One numbered line at indentation `depth`, with an optional end-of-line note.
#let ln(depth, body, note: none) = (depth: depth, body: body, note: note, numbered: true)
#let require(body) = (depth: 0, body: [*Require:* #body], note: none, numbered: false)
#let ensure(body) = (depth: 0, body: [*Ensure:* #body], note: none, numbered: false)

// `focus`: line numbers to highlight, and "in" for the Require/Ensure lines; the other
// lines are dimmed and lose their notes, so that the eye lands on the focus. With
// `focus: none` every line is drawn plainly.
#let pseudocode(lines, focus: none, notes: true) = {
  let number = 0
  let cells = ()
  for line in lines {
    let n = none
    if line.numbered {
      number += 1
      n = number
    }
    let active = focus == none or (if n == none { "in" in focus } else { n in focus })
    let lit = focus != none and active
    let body = pad(left: line.depth * 1.25em, {
      line.body
      if notes and line.note != none and active {
        h(1fr)
        h(0.8em)
        text(fill: comment-colour, size: 0.88em, style: "italic")[#"//" #line.note]
      }
    })
    let colour = if active { black } else { dimmed-colour }
    cells.push(grid.cell(
      fill: if lit { focus-colour },
      stroke: if lit { (left: 2pt + focus-rule) },
      align(right, text(0.8em, fill: if active { muted } else { dimmed-colour }, if n != none [#n])),
    ))
    cells.push(grid.cell(fill: if lit { focus-colour }, {
      set text(fill: colour)
      show ref: set text(fill: colour)
      body
    }))
  }
  set par(justify: false, leading: 0.45em)
  grid(columns: (1.4em, 1fr), column-gutter: 0.5em, inset: (x: 2pt, y: 0.3em), ..cells)
}

// A numbered algorithm; the body is usually a pseudocode list.
#let algorithm(caption: none, body) = figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: caption,
  body,
)

// A storyboard step: on the left the algorithm, with the lines this step is about in focus,
// and the narration; on the right what those lines do, so the reader never has to look
// elsewhere. `code` is (label: <alg:...>, lines: ...), or none for a step that no
// pseudocode line covers.
#let step-counter = counter("storyboard-step")
#let step(title, code: none, focus: none, visual, body) = block(
  breakable: false,
  width: 100%,
  above: 1.3em,
  below: 1.3em,
  {
    step-counter.step()
    block(below: 0.5em, stroke: (bottom: 0.5pt + luma(200)), inset: (bottom: 3pt), width: 100%)[
      #text(fill: accent, weight: "bold")[#context step-counter.display() #h(0.4em) #title]
      #h(1fr)
      #if code != none { text(8pt, fill: muted, ref(code.label)) }
    ]
    let narration = text(9.5pt, body)
    grid(
      columns: (1.08fr, 1fr),
      column-gutter: 1.4em,
      // End-of-line notes would wrap in a half-width column; the narration carries them.
      if code != none {
        set text(8.5pt)
        block(fill: luma(250), inset: (x: 4pt, y: 5pt), radius: 2pt, width: 100%,
          pseudocode(code.lines, focus: focus, notes: false))
        v(0.2em)
        narration
      } else { narration },
      align(center, visual),
    )
  },
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
