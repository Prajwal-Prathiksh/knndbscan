// Pseudocode as data, so one algorithm can be drawn in full and again with some lines in
// focus (a storyboard step) without being written twice:
//
//   #let alg = (require[...], ensure[...], ln(0, [*repeat*]), ln(1, [$x <- y$], note: [why]))
//   #pseudocode(alg)                             // every line
//   #pseudocode(alg, focus: (2, 3))              // lines 2-3 highlighted, the rest dimmed
//   #pseudocode(alg, focus: (7,), window: 1)     // only lines 6-8, the rest elided
//
// Keywords are bold in the body; notes are small grey italics after `//` at the line's
// end; there are no end keywords and no indentation guides.

#import "theme.typ": theme

// One numbered line at indentation `depth`, with an optional end-of-line note.
#let ln(depth, body, note: none) = (depth: depth, body: body, note: note, numbered: true)
#let require(body) = (depth: 0, body: [*Require:* #body], note: none, numbered: false)
#let ensure(body) = (depth: 0, body: [*Ensure:* #body], note: none, numbered: false)

// `focus`: line numbers to highlight, and "in" for the Require/Ensure lines; other lines
// are dimmed and lose their notes, so the eye lands on the focus. `notes: false` drops
// all notes (for narrow columns). `window: w` shows only lines within w lines of the
// focus and elides each run of the others as one "⋮" line.
#let pseudocode(lines, focus: none, notes: true, window: none) = context {
  let t = theme()
  let rows = ()
  let number = 0
  for line in lines {
    let n = none
    if line.numbered {
      number += 1
      n = number
    }
    rows.push((line: line, n: n))
  }
  let in-focus(row) = focus != none and (if row.n == none { "in" in focus } else { row.n in focus })
  let visible(k) = (
    window == none
      or focus == none
      or range(calc.max(0, k - window), calc.min(rows.len(), k + window + 1)).any(j => in-focus(rows.at(j)))
  )

  let cells = ()
  let elided = false
  for (k, row) in rows.enumerate() {
    if not visible(k) {
      if not elided { cells += ([], text(fill: t.dimmed)[#h(row.line.depth * 1.25em)⋮]) }
      elided = true
      continue
    }
    elided = false
    let active = focus == none or in-focus(row)
    let lit = focus != none and active
    let colour = if active { black } else { t.dimmed }
    let body = pad(left: row.line.depth * 1.25em, {
      row.line.body
      if notes and row.line.note != none and active {
        h(1fr)
        h(0.8em)
        text(fill: t.comment, size: 0.88em, style: "italic")[#"//" #row.line.note]
      }
    })
    cells.push(grid.cell(
      fill: if lit { t.focus },
      stroke: if lit { (left: 2pt + t.focus-rule) },
      align(right, text(0.8em, fill: if active { t.muted } else { t.dimmed }, if row.n != none [#row.n])),
    ))
    cells.push(grid.cell(fill: if lit { t.focus }, {
      set text(fill: colour)
      show ref: set text(fill: colour)
      body
    }))
  }
  set par(justify: false, leading: 0.45em)
  grid(columns: (1.4em, 1fr), column-gutter: 0.5em, inset: (x: 2pt, y: 0.3em), ..cells)
}
