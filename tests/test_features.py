import pandas as pd

from fx_signal.explain import pick_fact
from fx_signal.features import add_features


def test_features_do_not_change_when_future_is_appended() -> None:
    dates = pd.date_range("2025-01-01", periods=40, freq="D")
    rates = pd.DataFrame(
        {
            "currency": "TJS",
            "effective_date": dates,
            "rub_per_unit": [20 - i * 0.1 for i in range(40)],
        }
    )
    columns = [
        "signal_momentum",
        "down_streak",
        "price_percentile",
        "ret_1",
        "vol_20",
        "has_fact",
    ]
    short = add_features(rates.iloc[:30], level_window=10, reversal_window=5)
    full = add_features(rates, level_window=10, reversal_window=5)
    comparable = [column for column in columns if column in short.columns]
    pd.testing.assert_frame_equal(
        short[comparable].reset_index(drop=True),
        full.iloc[:30][comparable].reset_index(drop=True),
        check_dtype=False,
    )


def test_pick_fact_prefers_level_over_momentum() -> None:
    row = pd.Series(
        {
            "signal_level": True,
            "signal_momentum": True,
            "signal_seasonality": False,
            "signal_reversal": False,
            "price_percentile": 0.08,
            "down_streak": 4,
            "days_to_holiday": 3,
        }
    )
    picked = pick_fact(row, target_kind="stay_not_worse")
    assert picked is not None
    assert picked["indicator"] == "level"
    assert picked["speed"] == "slow"
    assert picked["scenario"] == "now_cheap"


def test_pick_fact_stays_silent_without_allowed_fact() -> None:
    row = pd.Series(
        {
            "signal_level": False,
            "signal_momentum": False,
            "signal_seasonality": False,
            "signal_reversal": True,
        }
    )
    assert pick_fact(row, target_kind="stay_not_worse") is None
    picked = pick_fact(row, target_kind="window_closing")
    assert picked is not None
    assert picked["indicator"] == "reversal"
