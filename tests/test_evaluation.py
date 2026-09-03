import pandas as pd
import pytest

from fx_signal.evaluation import evaluate_lift


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
    assert result.loc["momentum", "signal_hit_rate"] == 1.0
    assert result.loc["momentum", "random_hit_rate"] == 0.5
    assert result.loc["momentum", "lift"] == 2.0
    assert result.loc["reversal", "lift"] == 0.0
    assert result.loc["level", "lift"] == pytest.approx(2.0)

