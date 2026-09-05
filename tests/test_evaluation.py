import pandas as pd
import pytest

from fx_signal.evaluation import evaluate_lift, evaluate_method, forward_bps


def test_lift_matches_manual_calculation() -> None:
    frame = pd.DataFrame(
        {
            "currency": ["TJS"] * 4,
            "effective_date": pd.date_range("2025-01-01", periods=4),
            "target_local_min_h3": [True, False, True, False],
            "signal_momentum": [True, False, True, False],
            "signal_reversal": [False, True, False, False],
            "signal_seasonality": [True, True, False, False],
            "signal_level": [True, False, False, False],
        }
    )
    result = evaluate_lift(frame).set_index("indicator")
    assert (result["horizon"] == 3).all()
    assert result.loc["momentum", "signal_hit_rate"] == 1.0
    assert result.loc["momentum", "random_hit_rate"] == 0.5
    assert result.loc["momentum", "lift"] == 2.0
    assert result.loc["reversal", "lift"] == 0.0
    assert result.loc["level", "lift"] == pytest.approx(2.0)


def test_forward_bps_is_positive_when_future_is_more_expensive() -> None:
    price = pd.Series([10.0, 10.0])
    future = pd.Series([10.1, 9.9])
    bps = forward_bps(price, future)
    assert bps.iloc[0] == pytest.approx(100.0)
    assert bps.iloc[1] == pytest.approx(-100.0)


def test_evaluate_method_reports_forward_bps_and_frequency() -> None:
    frame = pd.DataFrame(
        {
            "currency": ["TJS"] * 6,
            "effective_date": pd.date_range("2025-01-01", periods=6),
            "rub_per_unit": [10.0, 10.0, 9.0, 11.0, 11.0, 11.0],
            "forward_mean_h3": [10.0, 11.0, 11.0, 11.0, 11.0, 11.0],
            "target_stay_not_worse_h3": [True, True, False, True, True, True],
            "ml_signal": [True, True, False, False, False, False],
        }
    )
    result = evaluate_method(
        frame,
        horizon=3,
        method="logreg",
        split="oot",
        target_col="target_stay_not_worse_h3",
        signal_col="ml_signal",
    ).iloc[0]
    assert result["signal_count"] == 2
    assert result["signal_hit_rate"] == 1.0
    assert result["bps_forward"] == pytest.approx(500.0)
    assert result["signals_per_week"] > 0
