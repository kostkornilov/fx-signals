import numpy as np
import pandas as pd

from fx_signal.features import add_features
from fx_signal.targets import add_targets
from fx_signal.train import run_method


def _synthetic_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2019-01-01", "2026-09-02")
    rows = []
    for currency, offset in (("TJS", 0.0), ("KZT", 0.4)):
        price = 8 + offset + np.sin(np.arange(len(dates)) / 18.0) + np.linspace(0, 0.8, len(dates))
        rows.append(
            pd.DataFrame(
                {
                    "currency": currency,
                    "effective_date": dates,
                    "rub_per_unit": price,
                }
            )
        )
    rates = pd.concat(rows, ignore_index=True)
    frame = add_targets(rates, horizon=5)
    return add_features(frame, level_window=30, reversal_window=10)


def test_logreg_walk_forward_writes_oot_and_2022_rows() -> None:
    frame = _synthetic_frame()
    config = {
        "horizon": 5,
        "target": "stay_not_worse",
        "feature_groups": ["A", "B"],
        "first_test_year": 2022,
        "oot_start": "2025-09-01",
        "thresholds": [0.6, 0.8],
        "quantile_rates": [0.2],
        "target_signals_per_week": [0.3, 8.0],
        "seed": 0,
    }
    scored, metrics = run_method(frame, config, "logreg")
    assert not metrics.empty
    assert set(metrics["split"]) >= {"y2022", "oot"}
    test_days = scored["eval_signal"].notna()
    assert test_days.any()
    # Predictions exist only on walk-forward test windows, not on the 2019 head.
    early = scored["effective_date"] < pd.Timestamp("2021-06-01")
    assert not scored.loc[early, "eval_signal"].any()
