// Primitives that explain before they formalize: a question that opens a section, a
// picture beside its words, frames of a process, pseudocode storyboards, words beside
// symbols, and the one-line takeaway. Each idea then gets its formal statement from
// environments.typ, so a reader can check the picture against the mathematics.

#import "theme.typ": theme
#import "pseudocode.typ": pseudocode

// The problem a section solves, asked before any answer is given.
#let question(body) = context block(
  sticky: true,
  above: 1em,
  below: 1em,
  inset: (left: 12pt, y: 2pt),
  stroke: (left: 2.5pt + theme().accent.lighten(45%)),
  text(12pt, style: "italic", fill: theme().accent, body),
)

// A picture beside the words that explain it, vertically centred on each other. `side`
// places the picture, so consecutive scenes can alternate; `width` is its share of the line.
#let scene(visual, body, side: right, width: 44%) = {
  let picture = align(center + horizon, visual)
  let words = align(horizon, body)
  block(breakable: false, above: 1.1em, below: 1.1em, grid(
    columns: if side == right { (1fr, width) } else { (width, 1fr) },
    column-gutter: 1.5em,
    ..if side == right { (words, picture) } else { (picture, words) },
  ))
}

// Frames of one process in time order, read left to right like an animation. Each frame
// is (visual, label); with a caption the strip is a numbered figure.
#let filmstrip(..frames, caption: none, arrows: true) = {
  let fs = frames.pos()
  let n = fs.len()
  let columns = range(2 * n - 1).map(k => if calc.even(k) { 1fr } else { auto })
  let pictures = range(2 * n - 1).map(k => if calc.even(k) {
    align(center + horizon, fs.at(int(k / 2)).at(0))
  } else {
    align(center + horizon, context text(13pt, fill: theme().muted, if arrows [→]))
  })
  let labels = range(2 * n - 1).map(k => if calc.even(k) {
    align(center, text(8pt, fs.at(int(k / 2)).at(1)))
  } else { [] })
  let strip = grid(columns: columns, column-gutter: 0.45em, row-gutter: 0.45em, ..pictures, ..labels)
  if caption == none {
    block(breakable: false, above: 1.1em, below: 1.1em, strip)
  } else {
    figure(strip, kind: image, caption: caption)
  }
}

// A storyboard step: on the left the algorithm, with the lines this step is about in
// focus, and the narration; on the right what those lines do, so the reader never has to
// look elsewhere. `code` is (label: <alg:...>, lines: ...), or none for a step that no
// pseudocode line covers. `window` elides lines far from the focus (see pseudocode);
// `width` is the picture's share of the line.
#let step-counter = counter("companion-step")
#let step(title, code: none, focus: none, window: none, width: 48%, visual, body) = block(
  breakable: false,
  width: 100%,
  above: 1.3em,
  below: 1.3em,
  context {
    let t = theme()
    step-counter.step()
    block(below: 0.5em, stroke: (bottom: 0.5pt + luma(200)), inset: (bottom: 3pt), width: 100%)[
      #text(fill: t.accent, weight: "bold")[#context step-counter.display() #h(0.4em) #title]
      #h(1fr)
      #if code != none { text(8pt, fill: t.muted, ref(code.label)) }
    ]
    let narration = text(9.5pt, body)
    grid(
      columns: (1fr, width),
      column-gutter: 1.4em,
      // End-of-line notes would wrap in a half-width column; the narration carries them.
      if code != none {
        set text(8.5pt)
        block(fill: luma(250), inset: (x: 4pt, y: 5pt), radius: 2pt, width: 100%,
          pseudocode(code.lines, focus: focus, notes: false, window: window))
        v(0.2em)
        narration
      } else { narration },
      align(center, visual),
    )
  },
)
// Steps are numbered within a story; call at the start of each.
#let story() = step-counter.update(0)

// Plain words beside the symbols that say the same thing, one idea per row.
#let translate(..rows) = block(breakable: false, above: 1em, below: 1em, table(
  columns: (1fr, 1fr),
  inset: (x: 6pt, y: 5pt),
  align: left + horizon,
  table.header[In words][In symbols],
  ..rows.pos().flatten(),
))

// The section's idea in one sentence, to close it.
#let takeaway(body) = context block(
  width: 100%,
  above: 1.3em,
  below: 1.5em,
  inset: (x: 12pt, y: 8pt),
  radius: 3pt,
  fill: theme().accent.lighten(93%),
)[#text(weight: "bold", fill: theme().accent)[In one line.] #body]

// Short items of equal weight in a grid, each (title, body).
#let cards(columns: 3, ..items) = context grid(
  columns: (1fr,) * columns,
  gutter: 0.7em,
  ..items.pos().map(((title, body)) => block(
    width: 100%,
    inset: 8pt,
    radius: 3pt,
    stroke: 0.5pt + luma(205),
    fill: luma(252),
  )[#set par(justify: false); #text(weight: "bold", fill: theme().accent, title) \ #text(9pt, body)]),
)

// An inline colour key that matches a figure's marks: a dot, a cross, a line or a dashed line.
#let swatch(colour, shape: "dot") = box(baseline: 0.05em, if shape == "dot" {
  circle(radius: 0.32em, fill: colour)
} else if shape == "cross" {
  text(fill: colour, weight: "bold")[×]
} else if shape == "line" {
  line(length: 1.1em, stroke: 1.4pt + colour, start: (0pt, -0.3em))
} else {
  line(length: 1.1em, stroke: (paint: colour, thickness: 1pt, dash: "dashed"), start: (0pt, -0.3em))
})
