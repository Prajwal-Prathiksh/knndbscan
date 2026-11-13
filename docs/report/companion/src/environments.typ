// Formal statements (definitions, results, invariants), proofs, algorithms and callouts:
// the "in symbols" register of a companion. Environments are figures, so `@label` works.

#import "theme.typ": theme

// kind: (supplement, theme key of its colour)
#let environments = (
  definition: ([Definition], "definition"),
  lemma: ([Lemma], "result"),
  theorem: ([Theorem], "result"),
  proposition: ([Proposition], "result"),
  invariant: ([Invariant], "invariant"),
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

// A numbered algorithm; the body is usually `pseudocode(...)`.
#let algorithm(caption: none, body) = figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: caption,
  body,
)

// Draws environments as tinted boxes and algorithms between rules; `report` installs it.
#let environment-rules(it) = context {
  if type(it.kind) == str and it.kind in environments {
    let colour = theme().at(environments.at(it.kind).at(1))
    block(
      width: 100%,
      breakable: true,
      inset: (x: 10pt, y: 8pt),
      radius: 2pt,
      fill: colour.lighten(94%),
      stroke: (left: 2pt + colour),
      align(left)[
        #text(weight: "bold", fill: colour)[#it.supplement #it.counter.display(it.numbering)]#if it.caption != none [ (#it.caption.body)].
        #it.body
      ],
    )
  } else if it.kind == "algorithm" {
    // An algorithm is read as a whole, so it is never split across pages.
    block(
      width: 100%,
      breakable: false,
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

#let callout(title, key, body) = context {
  let colour = theme().at(key)
  block(
    width: 100%,
    breakable: true,
    inset: (x: 10pt, y: 8pt),
    radius: 2pt,
    fill: colour.lighten(92%),
    stroke: (left: 2.5pt + colour),
  )[#text(weight: "bold", fill: colour)[#title.] #body]
}
#let note(body, title: [Note]) = callout(title, "accent", body)
#let pitfall(body, title: [Pitfall]) = callout(title, "danger", body)
#let implementation(body, title: [Implementation]) = callout(title, "implementation", body)
