#import "@local/companion:0.1.0": *

#show: report.with(
  title: [Method name: what it computes and how],
  subtitle: [A technical companion to the paper that introduced it],
  authors: ((name: [Project name], affiliation: [repository or group]),),
  date: datetime.today().display("[month repr:long] [year]"),
  abstract: [What the method does, who this companion is for, and what it explains.],
)

= The problem <sec:problem>

#question[The question a reader should be able to answer by the end of this section.]

#scene(rect(width: 100%, height: 4cm, fill: luma(240))[a picture of the running example])[
  Show the situation before naming anything: what the reader sees in the picture, and
  why the obvious approach is not enough.
]

#translate(
  [the objects the method works on], $cal(X)$,
  [the parameter that controls it], $epsilon$,
)

#definition(title: [the central object])[The formal statement of what the picture showed.] <def:object>

#let method = (
  require[the input],
  ensure[the output],
  ln(0, [*repeat*]),
  ln(1, [one step of the method], note: [why it is needed]),
  ln(0, [*until* nothing changes]),
)

#algorithm(caption: [The method.], pseudocode(method)) <alg:method>

#story()
#step([The first step], code: (label: <alg:method>, lines: method), focus: (2,),
  rect(width: 100%, height: 4cm, fill: luma(240))[what line 2 does to the example])[
  What the highlighted line does, in words, on the running example.
]

#theorem(title: [why it works])[The property that makes the method correct.] <thm:main>
#proof[Why it holds, in a few sentences.]

#takeaway[The section's idea in one sentence.]
