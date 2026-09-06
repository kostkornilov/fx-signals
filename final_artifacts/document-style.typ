// Общий стиль итоговых материалов; цвета и шрифт взяты из slides.typ.
#let ink = rgb("#101010")
#let muted = rgb("#63636b")
#let paper = rgb("#eeeeee")
#let red = rgb("#f03226")
#let violet = rgb("#6c24ff")

#let report-style(kind, body) = {
  set document(title: "Как поймать выгодный момент для перевода за рубеж — " + kind,
    author: ("Константин Корнилов", "Георгий Грушевский", "Берке Демир"))
  set page(paper: "a4", margin: (x: 17mm, top: 20mm, bottom: 18mm), fill: white,
    header: [
      #set text(size: 8pt, fill: muted)
      #grid(columns: (1fr, auto), [КОМАНДА 101 / AI TALENT HACKATHON], [#kind])
      #v(5pt)
      #line(length: 100%, stroke: 0.6pt + paper.darken(12%))
    ],
    footer: [
      #set text(size: 8pt, fill: muted)
      #line(length: 100%, stroke: 0.6pt + paper.darken(12%))
      #v(4pt)
      #grid(columns: (1fr, auto), [Финальная версия · 6 сентября 2026],
        text(fill: red, weight: "bold", context counter(page).display("01")))
    ],
  )
  set text(font: "YS Text", size: 10.5pt, fill: ink, lang: "ru")
  set par(leading: 0.57em, spacing: 7pt, justify: false)
  set heading(numbering: none)
  show heading.where(level: 1): it => block(above: 0pt, below: 14pt)[
    #text(size: 25pt, weight: "bold")[#it.body]
  ]
  show heading.where(level: 2): it => block(above: 12pt, below: 5pt)[
    #text(size: 13pt, weight: "bold")[#it.body]
  ]
  show heading.where(level: 3): it => block(above: 8pt, below: 4pt)[
    #text(size: 10.5pt, weight: "bold")[#it.body]
  ]
  set list(indent: 0pt, body-indent: 10pt, spacing: 4pt)
  set enum(indent: 0pt, body-indent: 10pt, spacing: 4pt)
  show link: set text(fill: violet)
  show raw: set text(font: "Menlo", size: 7.8pt)
  body
}

#let kicker(body) = block(above: 0pt, below: 8pt)[
  #text(size: 8.5pt, weight: "bold", fill: red, tracking: 0.7pt)[#body]
]

#let banner(label, title, subtitle) = block(
  width: 100%, fill: violet, inset: 18pt, radius: 5pt,
)[
  #text(size: 8pt, fill: white, weight: "bold", tracking: 0.7pt)[#label]
  #v(11pt)
  #text(size: 26pt, weight: "bold", fill: white)[#title]
  #v(9pt)
  #text(size: 10.5pt, fill: white)[#subtitle]
]

#let note(body) = block(width: 100%, fill: paper, inset: 11pt, radius: 4pt)[#body]
#let accent(body) = block(width: 100%, stroke: (left: 2pt + red), inset: (left: 12pt, y: 5pt))[#body]
#let small(body) = text(size: 8.3pt, fill: muted, body)

#let data-table(columns, ..cells) = {
  set text(size: 9.2pt)
  set par(leading: 0.4em)
  table(columns: columns, inset: (x: 8pt, y: 7pt),
    stroke: (x: none, y: 0.5pt + paper.darken(8%)),
    fill: (_, y) => if y == 0 { paper } else { white },
    ..cells.pos())
}

#let source(body) = block(above: 8pt)[#small[#body]]
