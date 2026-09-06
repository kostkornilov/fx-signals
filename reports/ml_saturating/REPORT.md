# ML-матрица с saturating-порогом

Сигналы пересчитаны текущим кодом: порог на validation — `score = lift × min(signals_per_week, 1)`, верхний гейт `target_signals_per_week[1]`. Правила и random не зависят от порога модели.

Обучающий горизонт ML: 5 наблюдений. Те же сигналы оценены на 1/3/5/10/20. 13 конфигураций × 5 коридоров × 5 горизонтов × 5 фолдов.

Это не восстановление архива `4c12350`. Исторический пересчёт с замороженными прогнозами остаётся в [reports/ml_recomputed](../ml_recomputed/REPORT.md).

Канонические таблицы: [ml_summary.csv](../tables/ml_summary.csv), [ml_diagnostics.csv](../tables/ml_diagnostics.csv). Прогнозы этого прогона: [predictions.csv](predictions.csv).
