# Indicator search report

## Итог

Проверено 124 одиночных правил и автоматически построенные интерпретируемые
ансамбли. Quality gate: **FAIL**. Отрицательный результат считается
валидным завершением перебора и не является основанием для продуктового запуска.

## Discovery outer folds

- Средний macro lift: 0.828
- 95% moving-block bootstrap CI lift: [0.732, 0.937]
- Средний эффект: -15.8 б.п.
- 95% CI эффекта: [-36.1, 5.5] б.п.
- Средняя частота по коридорам: 0.863 сигнала в неделю
- Коридоров со средним lift > 1: 0/5
- Публичный внешний контекст: not loaded: configured public sources were unavailable during this run

## Выбранная внутри фолдов политика

- wf_2022: `or__level__pct_w250_p10__oversold__stoch_w60_p20`, cooldown=1
- wf_2023: `or__momentum__ret_h20_w60_p20__relative__eur_ret_1_w120_p10`, cooldown=1
- wf_2024: `vote2__top3`, cooldown=3
- wf_2025: `or__level__z_sma_w120_m0p5__momentum__ret_h10_w120_p20`, cooldown=1
- oot: `or__oversold__rsi_w10_p40__level__near_min_w120_bp10`, cooldown=1

Период после 2025-09-01 показан только как повторно
использованный confirmation и не входит в bootstrap CI.
