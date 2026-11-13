# companion

A Typst package for writing technical companions to research papers: documents that let a
reader understand a method, and the code that implements it, without first reading the
paper.

A companion explains every idea three times, in this order:

1. **a picture** of one running example, used throughout the document;
2. **words**: a plain explanation of what the picture shows;
3. **symbols**: the formal definition, lemma or theorem, in a tinted box.

The picture and the symbols say the same thing, so a reader can check either against the
other. Each section opens with the question it answers, and closes with its idea in one
sentence. Algorithms are written once, and then shown again beside each picture with the
lines that picture is about highlighted, so the reader never has to turn back to find
"line 7".

## Using it

The package uses only Typst's built-in functions and the fonts embedded in Typst (Libertinus
Serif, New Computer Modern Math, DejaVu Sans Mono), and has been tested with Typst 0.15.

```typst
#import "@local/companion:0.1.0": *   // installed (see below)
#import "companion/lib.typ": *         // or vendored next to the document

#show: report.with(
  title: [...],
  subtitle: [...],
  authors: ((name: [...], affiliation: [...]),),
  abstract: [...],
  theme: (accent: rgb("#5b3f8c")),     // optional: any key of default-theme
)
```

`typst init @local/companion:0.1.0 my-companion` starts a document from `template/main.typ`.

To install it for every project on a machine, link or copy this directory to
`{data-dir}/typst/packages/local/companion/0.1.0/`, where `{data-dir}` is
`~/.local/share` on Linux, `~/Library/Application Support` on macOS and `%APPDATA%` on
Windows. To publish it for everyone, submit it to Typst Universe as `@preview/companion`.
Universe requires a template thumbnail, and possibly a more distinctive name.

## Primitives

### Opening and closing a section

| Primitive | Use |
|---|---|
| `question[...]` | The problem the section solves, asked before any answer. It stays on the same page as what follows it. |
| `takeaway[...]` | The section's idea in one sentence. |

### Pictures and words

| Primitive | Use |
|---|---|
| `scene(visual, body, side: right, width: 44%)` | A picture beside the words that explain it, centred on each other. Alternate `side` between consecutive scenes. |
| `filmstrip((visual, label), ..., caption: none, arrows: true)` | Frames of one process in time order, with arrows between them. With a caption it becomes a numbered figure that `@label` can reference. Use `arrows: false` for a side-by-side comparison. |
| `cards((title, body), ..., columns: 3)` | Short items of equal weight, such as test strategies or design options. |
| `swatch(colour, shape: "dot")` | An inline colour key (`"dot"`, `"cross"`, `"line"`, `"dash"`), so the text can name a figure's marks: `#swatch(blue) core point`. |

### Words and symbols

| Primitive | Use |
|---|---|
| `translate([words], [symbols], ...)` | A two-column table with each idea in plain words beside its notation. Use it for the notation, and before a definition. |
| `definition`, `lemma`, `theorem`, `proposition`, `invariant` | Numbered formal statements, called as `#theorem(title: [...])[...] <thm:x>`. |
| `proof[...]` | A proof or proof sketch, closed with a square. |
| `note`, `pitfall`, `implementation` | Callouts for context, a trap an implementation can fall into, and engineering detail. |

### Algorithms

Pseudocode is data, so one algorithm can be drawn in full and again in focus:

```typst
#let method = (
  require[the input], ensure[the output],
  ln(0, [*repeat*]),
  ln(1, [$x <- f(x)$], note: [why]),
  ln(0, [*until* $x$ is unchanged]),
)
#algorithm(caption: [The method.], pseudocode(method)) <alg:method>
```

- `pseudocode(lines, focus: none, notes: true, window: none)`: `focus` takes line numbers,
  plus `"in"` for the Require/Ensure lines. Lines in focus are highlighted; the others are
  dimmed and lose their notes. `window: w` shows only the lines within `w` of the focus,
  and collapses each run of hidden lines into one "⋮" line.
- `step(title, code: (label: <alg:method>, lines: method), focus: (...), window: none, width: 48%, visual)[narration]`:
  a storyboard step. The algorithm is on the left, with the focus lines highlighted and
  the narration below it; the picture of what those lines do is on the right. Steps are
  numbered within a story: call `story()` to start a new one. Pass `code: none` for a step
  that no pseudocode line covers.

## A section, in order

```typst
= Title <sec:x>
#question[What problem does this section solve?]
#scene(picture)[What the picture shows, in words.]
#translate([the idea in words], $the idea in symbols$)
#definition(title: [...])[...]
#story()
#step([First step], code: alg, focus: (1, 2), picture)[What lines 1-2 do.]
#theorem(title: [...])[...]
#proof[...]
#takeaway[The idea in one line.]
```

## Theme

`default-theme` (in `src/theme.typ`) names every colour the primitives use: `accent`,
`muted`, `comment`, `dimmed`, `focus`, `focus-rule`, `danger`, `definition`, `result`,
`invariant` and `implementation`. Pass any of them to `report(theme: (...))` to override
them. The colours of a document's own subject, for example which colour means a "core
point", belong to the document: write them, with the numbers the text quotes, from the
script that draws the figures, and read them with `json(...)`, so the text, the figures
and the diagrams cannot disagree.
