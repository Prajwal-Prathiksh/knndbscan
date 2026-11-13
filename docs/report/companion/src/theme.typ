// The colours every primitive draws with, set once per document by `report(theme: ...)`.

#let default-theme = (
  accent: rgb("#1f4e79"), // headings, step titles, references
  muted: luma(100), // secondary text: line numbers, page header
  comment: luma(135), // pseudocode notes
  dimmed: luma(175), // pseudocode lines out of focus
  focus: rgb("#fdf0c8"), // background of pseudocode lines in focus
  focus-rule: rgb("#d39b00"), // their left rule
  danger: rgb("#a1260d"), // pitfalls
  definition: rgb("#1f4e79"),
  result: rgb("#3a6b1f"), // lemmas, theorems, propositions
  invariant: rgb("#8a5a00"),
  implementation: rgb("#5b3f8c"),
)

#let theme-state = state("companion-theme", default-theme)

// The current theme; call inside `context`.
#let theme() = theme-state.get()
