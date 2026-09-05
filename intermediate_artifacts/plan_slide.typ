// Standalone companion source for the project plan slide.

#let navy = rgb("#102a43")
#let beige = rgb("#f2e8d8")

#set page(
  width: 13.333in,
  height: 7.5in,
  margin: (left: 62pt, right: 62pt, top: 42pt, bottom: 30pt),
  fill: beige,
)
#set text(font: "Arial", size: 22pt, fill: navy, lang: "ru")
#set par(leading: 1.05em)
#set enum(indent: 28pt, body-indent: 11pt, spacing: 16pt)

#block(width: 100%, breakable: false)[
  #grid(
    columns: (1fr, auto),
    align: (left, top),
    text(size: 36pt, weight: "bold")[Что в планах],
    text(size: 26pt, weight: "bold")[8],
  )
  #v(28pt)
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 48pt,
    align: (left, top),
    enum(
      [Расширить пул индикаторов.],
      [Реализовать универсальные бэктесты.],
      [Использовать ML для предсказания выгодности курса в моменте.],
    ),
    enum(
      start: 4,
      [Использовать данные других валютных пар с большей гранулярностью.],
      [Детально проработать пользовательский опыт: тексты пушей, дизайн страницы переводов и настройку частоты пушей о курсе.],
      [При устаревании пуша показывать статистику по другому индикатору.],
    ),
  )
]
