// Build from the repository root:
// typst compile --root . final_artifacts/slides.typ final_artifacts/slides.pdf

#let navy = rgb("#102a43")
#let beige = rgb("#f2e8d8")
#let cream = rgb("#fbf7ef")
#let sand = rgb("#cdbb9f")

#set page(
  width: 13.333in,
  height: 7.5in,
  margin: (left: 62pt, right: 62pt, top: 42pt, bottom: 30pt),
  fill: beige,
)
#set text(font: "Arial", size: 26pt, fill: navy, lang: "ru")
#set par(leading: 0.65em)
#set list(indent: 20pt, body-indent: 11pt, spacing: 30pt)
#set enum(indent: 20pt, body-indent: 11pt, spacing: 30pt)

#let frame(body) = block(
  width: 100%,
  height: 460pt,
  breakable: false,
)[#body]

#let slide(title, number, body) = [
  #frame[
    #grid(
      columns: (1fr, auto),
      align: (left, top),
      text(size: 36pt, weight: "bold", title),
      text(size: 26pt, weight: "bold", str(number)),
    )
    #v(28pt)
    #body
  ]
  #pagebreak(weak: true)
]

#let member(name, role, responsibility) = block(
  width: 100%,
  fill: cream,
  stroke: 1pt + sand,
  radius: 8pt,
  inset: (x: 18pt, y: 14pt),
)[
  #grid(
    columns: (0.95fr, 1.05fr),
    column-gutter: 26pt,
    align: (left, top),
    [#strong[#name]#linebreak()#role],
    [#responsibility],
  )
]

// 1. Title
#frame[
  #place(
    top + left,
    dx: -62pt,
    dy: -42pt,
    rect(width: 13.333in, height: 7.5in, fill: navy),
  )
  #v(70pt)
  #text(size: 50pt, weight: "bold", fill: beige)[Как поймать выгодный момент\
  для перевода за рубеж]
  #v(15pt)
  #text(size: 24pt, fill: beige)[_Строим триггерную модель для трансграничных переводов_]
]
#pagebreak(weak: true)
// 2. Team
#slide("Команда 10", 1)[
  #member(
    [#text(size: 21pt)[Корнилов Константин Георгиевич]],
    [#text(size: 21pt)[AI Engineer · 1 курс · \@cnstnk]],
    [#text(size: 21pt)[Исследование и разработка логики сигналов, ML для предсказаний]],
  )
  #v(12pt)
  #member(
    [#text(size: 21pt)[Грушевский Георгий Романович]],
    [#text(size: 21pt)[AI Engineer · 1 курс · \@dayzgoby]],
    [#text(size: 21pt)[Исследование офлайн-метрик, оценка потенциала и рисков]],
  )
  #v(12pt)
  #member(
    [#text(size: 21pt)[Демир Берке]],
    [#text(size: 21pt)[AI Product · 1 курс · \@bqrke]],
    [#text(size: 21pt)[Пользовательский опыт взаимодействия с пушами]],
  )
]
// 3. Problem
#slide("Проблематика", 2)[
  Пользователь хочет переводить деньги в другую страну по выгодному курсу, но у него нет времени регулярно следить за курсом валюты

  Можно помочь пользователю переводить выгоднее, отправляя ему push-уведомления
]
// 4. Task
#slide("Постановка задачи", 3)[
  #text(size: 28pt)[
    _Выбрать момент для отправки пуша, когда пользователь может перевести валюту по наиболее выгодному курсу.
    Выгодность курса в момент T определяем в сравнении с его динамикой в будущем._
  ]
]
// 5. Approach
#slide("Выбранный подход", 4)[
  Решение об отправке пуша в момент времени T принимается на основе:
  #v(18pt)
  #enum(
    [данных о динамике курса в прошлом],
    [прогноза вероятности наступления некоторого события в будущем],
  )
  #v(18pt)
  Пользователю сообщается только аналитический "сигнал" к переводу, видимый на прошлых данных.
]
// 6. Benefit
#slide("Откуда берется выгода", 5)[
  Выгода для пользователя достигается тем, что:
  #v(18pt)
  #enum(
    [с помощью коммуникации мы приводим его на страницу переводов в нужный момент],
    [помогаем следить за курсом, сообщая полезную информацию],
  )
  \
  В общем, принимаем решение о коммуникации *на основе прогноза* выгоды в моменте, но сообщаем пользователю лишь часть известной нам информации -- аналитический сигнал
]
// 7. Solution
#slide("Схема решения", 6)[
  #enum(
    [Пул из быстрых и медленных аналитических сигналов, которые можно сообщить пользователю],
    [Для каждого индикатора ML-модель предсказывает, будет ли текущий курс выгодным для некоторого окна в будущем],
    [Мета-алгоритм собирает предсказания моделей для каждого индикатора и принимает решение об отправке пуша],
  )
]
// TODO: слайд со сравнением всех метрик
#slide("Метрики", 7)[

]

// // 8. Current state
// #slide("Что есть сейчас", 7)[
//   #grid(
//     columns: (1fr, 1fr),
//     column-gutter: 38pt,
//     align: (left, top),
//     [
//       #strong[MVP работы с индикаторами]
//       #v(18pt)
//       #list(
//         [4 простейших индикатора],
//         [аналитические правила вместо ML-моделей],
//         [пуш отправляется, если хотя бы одно правило сработало.],
//       )
//     ],
//     image("../reports/figures/baseline_lift_h5.png", width: 100%, height: 285pt, fit: "contain"),
//   )
// ]
// #slide("Что в планах [1/2]", 8)[
//   #set enum(spacing: 12pt)
//   #enum(
//     [Расширить пул индикаторов],
//     [Реализовать универсальные бэктесты],
//     [Использовать ML для предсказания выгодности курса в моменте],
//     [Использовать данные других валютных пар с большей гранулярностью],
//     [Детально проработать пользовательский опыт: тексты пушей, дизайн страницы переводов и настройку частоты пушей о курсе],
//     [При устаревании пуша показывать статистику по другому индикатору],
//     [Мемные пуши, чтобы обучить пользователей на них кликать]
//   )
// ]

// #slide("Пример дизайна в приложении", 9)[
//   #grid(
//     columns: (1fr, 1fr),
//     column-gutter: 60pt,
//     align: (center, top),
//     // [
//     //   #align(center)[#strong[Сигнал актуален]]
//     //   #v(10pt)
//     //   #align(center)[#image("./ux/signal-current.png", height: 320pt)]
//     ],
//     [
//       // #align(center)[#strong[Сигнал устарел]]
//       // #v(10pt)
//       // #align(center)[#image("./ux/signal-stale.png", height: 320pt)]
//     ],
//   )
// ]
