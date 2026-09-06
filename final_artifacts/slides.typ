// Build from the repository root:
// typst compile --root . final_artifacts/slides.typ final_artifacts/slides.pdf

#let navy = rgb("#101010")
#let beige = rgb("#eeeeee")
#let cream = rgb("#ffffff")
#let sand = rgb("#f03226")
#let alfa-violet = rgb("#6c24ff")

#set page(
  width: 13.333in,
  height: 7.5in,
  margin: (left: 62pt, right: 62pt, top: 42pt, bottom: 30pt),
  fill: beige,
)
#set text(font: "YS Text", size: 26pt, fill: navy, lang: "ru")
#set par(leading: 0.65em)
#set list(indent: 20pt, body-indent: 11pt, spacing: 30pt)
#set enum(indent: 20pt, body-indent: 11pt, spacing: 30pt)

#let frame(body) = block(
  width: 100%,
  height: 460pt,
  breakable: false,
)[#body]

#let alfa-logo(width: 120pt) = image("logo-transparent.png", width: width, fit: "contain")

#let slide(title, body) = [
  #frame[
    #place(
      top + right,
      grid(
        columns: (100pt, 50pt),
        column-gutter: 12pt,
        align: (center + horizon, center + horizon),
        alfa-logo(),
        text(size: 44pt, weight: "bold", fill: sand, context counter(page).display("1")),
      ),
    )
    #block(width: 650pt)[#text(size: 44pt, weight: "bold", title)]
    #v(28pt)
    #body
  ]
  #pagebreak(weak: true)
]

#let member(name, role, responsibility) = block(
  width: 100%,
  height: 320pt,
  fill: cream,
  stroke: 1pt + sand,
  radius: 8pt,
  inset: (x: 18pt, y: 18pt),
)[
  #strong[#name]
  #v(12pt)
  #role
  #v(28pt)
  #responsibility
]

// 1. Title
#frame[
  #place(
    top + left,
    dx: -62pt,
    dy: -42pt,
    rect(width: 13.333in, height: 7.5in, fill: alfa-violet),
  )
  #place(top + right, alfa-logo(width: 150pt))
  #v(120pt)
  #text(size: 50pt, weight: "bold", fill: cream)[Как поймать выгодный момент\
  для перевода за рубеж]
  #v(15pt)
  #text(size: 32pt, fill: cream)[_Строим триггерную модель для трансграничных переводов_]
]
#pagebreak(weak: true)
// 2. Team
#slide("Команда 101")[
  #grid(
    columns: (1fr, 1fr, 1fr),
    column-gutter: 18pt,
    align: (top, top, top),
    member(
      [#text(size: 21pt)[Корнилов Константин Георгиевич]],
      [#text(size: 21pt)[AI Engineer · 1 курс · \@cnstnk]],
      [#text(size: 21pt)[Исследование и разработка логики сигналов, ML для предсказаний]],
    ),
    member(
      [#text(size: 21pt)[Грушевский Георгий Романович]],
      [#text(size: 21pt)[AI Engineer · 1 курс · \@dayzgoby]],
      [#text(size: 21pt)[Дизайн метрик, оценка потенциала и рисков]],
    ),
    member(
      [#text(size: 21pt)[Демир Берке]],
      [#text(size: 21pt)[AI Product · 1 курс · \@bqrke]],
      [#text(size: 21pt)[Пользовательский опыт, взаимодействия с пушами]],
    ),
  )
]

// Слайд 0. По желанию: какую сформулировали ценность, юнит экономика пользователя и другое.
// Слайд 1. Клиентский путь целиком (экраны или шевроны с текстом) - суть продукта. Особенно если делаете свой оригинальный клиентский путь, отличающийся от постановки.
// Слайд 2. Какой сигнал выбрали и почему. Как фильтруете сигналы и почему.
// Слайд 3. Тексты пущей. Как решили проблему с часовыми поясами и устареванием сигналов.
// Слайд 4. По желанию.
// Ваши какие-то кастомные доработки, если выше не рассказали о них. Например, способ предзаполнения данных для переводов.

#slide("Схема решения")[
  1. ML-модель определяет, выгоден ли перевод сейчас
  2. Считается пул аналитических сигналов, которые можно сообщить пользователю
  3. Из пула сработавших сигналов выбирается наиболее понятный
  4. Пуш отправляется в 2 часа по местному времени
]

#slide("Сигналы")[
  - *Краткосрочная динамика*

    _Пример:_ в 4 из 5 последних обновлений курс растет

  - *Исторические ориентиры*

    _Пример:_ сегодня курс выгоднее среднего за последние 30 дней

  - *Сезонность*

    _Пример:_ приближение государственного праздника в стране получателя
]

#slide("Выбор текста")[
  Если сработало несколько сигналов — выберем самый понятный
  и заметный факт:

  ✅ "До Навруза осталось 5 дней. Посмотрите текущий курс перевода в Таджикистан"

  🚫 "За 10 000 ₽ сейчас получится на 30 сомони больше,
  чем три обновления назад"

  🚫 "Курс сомони улучшался в 4 из 5 последних обновлений.
  Сейчас 10 000 ₽ — это 1 250 сомони"

]

#slide("Больше, чем факты")[
  Коммуникация — возможность *научить* пользователя понимать состояние рынка

  _"В выходные часть валютных площадок закрыта. Курс перевода
  может двигаться иначе, чем в будни"_

  _"Последние 20 обновлений курс менялся в 2.4 раза
  сильнее обычного. Высокая волатильность —
  это большие движения в обе стороны"_
]

#slide("Пользовательский путь")[
  // Три самостоятельных PNG-макета, созданных по референсу тиммейта.
  // Генерация: client_path/prompts.json. Данные демонстрационные.
  #set text(size: 13pt)
  #set par(leading: 0.35em, spacing: 0pt)
  #set block(above: 0pt, below: 0pt)
  #grid(
    columns: (1fr, 1fr, 1fr),
    column-gutter: 26pt,
    align: center,
    [
      #text(size: 19pt, weight: "bold")[Получили пуш]
      #v(20pt)
      #image("client_path/01-push.png", height: 330pt, fit: "contain")
    ],
    [
      #text(size: 19pt, weight: "bold")[Сигнал актуален]
      #v(20pt)
      #image("client_path/02-current.png", height: 330pt, fit: "contain")
    ],
    [
      #text(size: 19pt, weight: "bold")[Сигнал устарел]
      #v(20pt)
      #image("client_path/03-expired.png", height: 330pt, fit: "contain")
    ],
  )
  #v(30pt)
  #text(size: 10pt, fill: rgb("#73747b"))[
    Концепт интерфейса, данные условные. Курс ЦБ для сигнала, курс приложения для перевода.
  ]

]

#slide("Оценка эффективности")[
  // Источник: reports/tables/ml_summary.csv, числовая сортировка lift_h5 DESC.
  // Колонки: method, lift_h5, lift_h10, lar_h5, lar_h10. Округление до 0.01.
  #set text(size: 20pt)
  // #set par(leading: 0.4em, spacing: 0pt)
  // #set block(above: 0pt, below: 0pt)
  Final OOT, среднее по пяти коридорам по всем валютам

  #table(
    columns: (2.1fr, 0.85fr, 0.85fr, 0.95fr, 1.0fr),
    align: (left, center, center, center, center),
    inset: (x: 12pt, y: 12pt),
    stroke: 0.6pt + navy,
    fill: (_, y) => if y == 0 { sand.lighten(82%) } else { cream },
    [*Подход*], [*Lift\@5*], [*Lift\@10*], [*LAR\@5*], [*LAR\@10*],
    [CatBoost (сигналы + контекст)], [*1.53*], [*1.59*], [*1.84*], [*1.63*],
    [LogReg (сигналы + динамика)], [1.52], [1.39], [1.02], [0.69],
    [Сигнал: недавние изменения плохие], [1.29], [1.38], [0.94], [0.98],
  )

  Порог классификации подбирается на валидации - чтобы получить нужную частоту: $ "LiftAtRisk" times min("SignalsPerWeek" / "TargetSignalsPerWeek", 1) $

//   в общем:
// 1. Сейчас в бектесте положительное предсказание = модель выдала p > порог & сработал один из 11 индикаторов

// 1.  результат на картинке

// 1. сверху этого вешается мета-аглоритм который ты вчера запушил
// 1. после того, как я  добавил твои новые индикаторы, катбуст уступил место в рейтинге. По lift_ar_risk. По обычному lift он все еще выше всех
]

#slide("Результат")[
  1. Репозиторий с кодом, который делает полный цикл работы - от предсказания до текста пуша
  2. Подход к влиянию на поведение пользователя
  3. Демо интерфейса + план на случай устаревания пуша
  4. Доказанная эффективность прогноза
]

#frame[
  #place(
    top + left,
    dx: -62pt,
    dy: -42pt,
    rect(width: 13.333in, height: 7.5in, fill: alfa-violet),
  )
  #place(top + right, alfa-logo(width: 150pt))
  #v(220pt)
  #text(size: 50pt, weight: "bold", fill: cream)[Спасибо за внимание!]
]
#pagebreak(weak: true)

#frame[
  #place(
    top + left,
    dx: -62pt,
    dy: -42pt,
    rect(width: 13.333in, height: 7.5in, fill: alfa-violet),
  )
  #place(top + right, alfa-logo(width: 150pt))
  #v(220pt)
  #text(size: 50pt, weight: "bold", fill: cream)[Приложение к презентации]
]
#pagebreak(weak: true)

#slide("Планы на будущее")[

  Будущее = данные о реакции пользователей

  1. Применить ML/LLM scoring в персонализированной генерации пуша
  2. LLM-переформулировки пушей
  3. Измерить и оптимизировать пользовательское поведение
  4. Онлайн-метрика эффекта от коммуникаций как LATE для суммы переводов
]

#slide("Метрика LiftAtRisk")[
  Lift не учитывает выгоду от "попаданий" и потери от "промахов"

  Модифицируем метрику:

  $ "LiftAtRisk" = "Lift" dot exp(ln q dot (Delta"Value" - rho dot Delta"Risk") / D ) $


  $Delta"Value"$ --- на сколько % курс в момент сигнала выгоднее *в среднем*

  $-Delta"Risk"$ --- на сколько % *меньше потери* в худших 5% случаев

  \

  $rho = 2, D = 1, q=1.3$
]


#slide("Данные")[
  #set text(size: 22pt)
  *Источник*: официальные курсы ЦБ РФ через API за 1 января 2019 — 2 сентября 2026.

  *Наборы признаков*:
  - Индикаторы для курса валюты --- momentum, level, reversal, праздники
  - "Интенсивность" индикаторов --- длина снижения, перцентиль, отскок от минимума, дни до праздника
  - Динамика курса --- доходности 1–20, волатильность и тренд за 20 публикаций
  - Контекст ЦБ по другим валютам --- USD, EUR, CNY и динамика по ним
]

#slide("Тестирование")[
  // #set text(size: 21pt)
  // #set list(spacing: 14pt)
  - Walk-forward: в каждом фолде train только на прошлом, подбор порога на валидации, честный test
  - Модели — LogReg и CatBoost, обученные на неухудшение курса в ближайшие 5 дней
  - "Сигналы" не обучаются, но считаются на тех же test-датах
  - 2022 — отдельный стресс-период. 2023 — август 2025 — walk-forward. Сентябрь 2025 — сентябрь 2026 — финальный OOT

]

#slide("Детализация метрик")[
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 24pt,
    image("../reports/figures/catboost_lift_h5_by_currency.png", width: 100%),
    image("../reports/figures/catboost_lar_h5_by_currency.png", width: 100%),
  )
]
