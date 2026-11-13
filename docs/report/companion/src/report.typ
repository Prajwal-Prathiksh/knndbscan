// The page, type and title block of a companion, installed with `#show: report.with(...)`.

#import "theme.typ": default-theme, theme, theme-state
#import "environments.typ": environment-rules

// `theme` overrides entries of default-theme (theme.typ), e.g. (accent: rgb("#5b3f8c")).
#let report(
  title: none,
  subtitle: none,
  authors: (),
  date: none,
  abstract: none,
  theme: (:),
  body,
) = {
  let t = default-theme + theme
  theme-state.update(t)
  // The PDF's author metadata takes plain strings; names may be given as content.
  let plain(it) = if type(it) == str { it } else if it.has("text") { it.text } else { "" }
  // No build timestamp in the PDF, so an unchanged source compiles to identical bytes and a
  // committed PDF changes only when its content does.
  set document(title: title, author: authors.map(a => plain(a.name)), date: none)
  set page(
    paper: "a4",
    margin: (x: 2.4cm, top: 2.6cm, bottom: 2.4cm),
    numbering: "1",
    header: context {
      if counter(page).get().first() > 1 {
        set text(8pt, fill: t.muted)
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
  show raw: set text(font: "DejaVu Sans Mono", size: 0.88em)
  show link: set text(fill: t.accent)
  show ref: set text(fill: t.accent)

  set heading(numbering: "1.1")
  show heading: set block(above: 1.4em, below: 0.7em)
  show heading.where(level: 1): set text(13.5pt, fill: t.accent)
  show heading.where(level: 2): set text(11pt, fill: t.accent)

  set table(stroke: (x, y) => if y == 0 { (bottom: 0.7pt) } else { (bottom: 0.3pt + luma(200)) })
  show table.cell.where(y: 0): set text(weight: "bold")
  show table.cell: set par(justify: false)
  show table.cell: set text(hyphenate: false)
  show figure.caption: set text(9pt)
  show figure: set block(breakable: true)
  show figure: environment-rules

  align(center)[
    #set par(justify: false)
    #set text(hyphenate: false)
    #text(19pt, weight: "bold", title)
    #if subtitle != none {
      v(0.2em)
      text(11.5pt, fill: t.muted, subtitle)
    }
    #v(0.8em)
    #authors.map(a => [#a.name #if "affiliation" in a [\ #text(9pt, fill: t.muted, a.affiliation)]]).join(h(3em))
    #if date != none {
      v(0.4em)
      text(9pt, fill: t.muted, date)
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
    #outline(depth: 1, indent: 1.2em)
  ]
  body
}
