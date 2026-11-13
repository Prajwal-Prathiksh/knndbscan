// companion: technical companions that show an idea, say it, then state it formally.
//
//   #import "@local/companion:0.1.0": *     // installed as a package
//   #import "companion/lib.typ": *          // or vendored next to the document
//   #show: report.with(title: [...], authors: ((name: [...]),), abstract: [...])
//
// See README.md for every primitive and how they combine.

#import "src/theme.typ": default-theme, theme
#import "src/report.typ": report
#import "src/environments.typ": (
  algorithm, callout, definition, implementation, invariant, lemma, note, pitfall, proof,
  proposition, theorem,
)
#import "src/pseudocode.typ": ensure, ln, pseudocode, require
#import "src/exposition.typ": cards, filmstrip, question, scene, step, story, swatch, takeaway, translate
