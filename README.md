# FX Signals

Воспроизводимый сигнальный слой для трансграничных переводов
`RUB -> AMD/KZT/KGS/TJS/UZS`.

Проект определяет подходящий момент для коммуникации о переводе и позволяет проверить,
насколько объяснимые индикаторы — momentum, level, reversal и seasonality — информативнее
случайного дня. Пять воспроизводимых ноутбуков оценивают поиск локального минимума курса на
горизонтах `h=1/3/5/10/20` свежих публикаций ЦБ.

## Быстрый старт

Требуются Python 3.12+ и [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run fx-signals data fetch --config configs/data.yaml
uv run fx-signals baseline --config configs/baseline.yaml
uv run fx-signals research --config configs/research.yaml
uv run fx-signals backtest-report --config configs/backtest_report.yaml
uv run jupyter nbconvert --execute --inplace notebooks/*.ipynb
```

Данные не хранятся в Git. Команда `data fetch` загружает фиксированный период из
[официального XML API Банка России](https://www.cbr.ru/development/SXML/) и создаёт локальный
manifest с URL, числом строк и SHA256. Период эксперимента закреплён в `configs/data.yaml`, поэтому
результат не сдвигается при появлении нового курса.

`backtest-report` запускает purged walk-forward бэктест на дневных официальных курсах ЦБ:
основные коридоры — `RUB→AMD/KZT/KGS/TJS/UZS`, контекстные ряды — `USD/EUR/CNY`.
Фиксированные правила не обучаются; LogReg и CatBoost обучаются на таргете `h=5`, а единый
поток их сигналов проверяется на `h=1/3/5/10/20`. Пороги выбираются только на validation,
final OOT не участвует в обучении. Полные метрики пишутся в
`reports/backtest/backtest_metrics.csv`, краткий вывод — в `reports/backtest/REPORT.md`,
OOT-витрина для финального отчёта — в `reports/tables/ml_summary_lift.csv`.
Период можно переопределить флагами `--first-test-year`, `--oot-start` и `--end`.

## Семантика

`rub_per_unit = value_rub / nominal` — рублей за единицу валюты получателя. Чем значение ниже,
тем выгоднее момент для отправителя.

```text
target_local_min_h(t) = 1,
если q(t) — минимум среди q(t-h), ..., q(t+h),
где h ∈ {1, 3, 5, 10, 20}
```

Горизонт считается в свежих публикациях. XML-история ЦБ уже содержит только даты установки
курса, поэтому выходные не превращаются в фиктивные нулевые изменения.

## Результаты

После запуска появляются:

- `reports/tables/baseline_all_horizons.csv` — общая таблица всех горизонтов;
- `reports/tables/baseline_h*.csv` — отдельная таблица для каждого горизонта;
- `reports/figures/baseline_lift_h*.png` — отдельные heatmap;
- пять выполненных ноутбуков в `notebooks/`, по одному на горизонт.

Сырые данные, промежуточные таблицы и модели игнорируются Git. Агрегированные результаты
экспериментов коммитить можно: они позволяют посмотреть вывод без доступа к API.

## Проверки

```bash
uv run pytest
uv run ruff check .
```

## Исследование дополнительных рядов

Команда `research` запускает одинаковый walk-forward/OOT эксперимент для текущего набора
признаков, усиленного CBR-only baseline и моделей с USD/EUR/CNY. Она формирует:

- `reports/research/data_audit.csv` — покрытие и лаг публикации;
- `reports/research/market_component_eda.csv` — корреляция, beta и остаточная волатильность;
- `reports/research/ablation_metrics.csv` — метрики всех моделей и горизонтов;
- `reports/research/paired_comparisons.csv` — paired moving-block bootstrap относительно baseline;
- `reports/research/README.md` — краткий автоматически собранный вывод.

Национальные банки и MOEX добавляются одним CSV через `external_path`; контракт и правила
point-in-time объединения описаны в [`docs/external-data.md`](docs/external-data.md). Внешние
признаки образуют группу `F`, а эксперимент с ней пропускается, пока snapshot отсутствует.

Тесты используют синтетические данные и не требуют сети.

## Ограничения текущего эксперимента

- Представленный в ноутбуке baseline пока не является итоговым walk-forward бэктестом.
- Параметры правил пока не подбираются по коридорам.
- Seasonality v0 означает близость к государственному празднику страны получателя.
- Reversal срабатывает после минимума и потому структурно не должен оцениваться только по
  target «сегодня локальный минимум»; его продуктовая метрика будет добавлена следующим этапом.
- Официальный курс ЦБ — proxy, а не курс исполнения в приложении банка.
- `available_at` консервативно приравнен к дате действия курса; отдельная модель времени
  публикации будет добавлена перед итоговым бэктестом.

Полный порядок следующих работ описан в `PLAN.md`.
