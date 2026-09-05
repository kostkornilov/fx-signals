import math

import numpy as np
import pandas as pd
import pytest

from fx_signal.metrics import (
    DEFAULT_SCALE_BPS,
    discounted_lift_at_risk,
    fixed_window_lift_at_risk,
    lar_path_outcomes,
    lift_at_risk,
)


def test_default_scale_makes_100_bps_equal_to_1_3_multiplier() -> None:
    score = lift_at_risk(
        lift=1.0,
        signal_value_bps=100.0,
        random_value_bps=0.0,
        signal_tail_regret_bps=0.0,
        random_tail_regret_bps=0.0,
    )

    assert DEFAULT_SCALE_BPS == pytest.approx(100.0 / math.log(1.3))
    assert score == pytest.approx(1.3)


def test_matched_random_components_leave_lift_unchanged() -> None:
    score = lift_at_risk(
        lift=1.4,
        signal_value_bps=20.0,
        random_value_bps=20.0,
        signal_tail_regret_bps=50.0,
        random_tail_regret_bps=50.0,
    )

    assert score == pytest.approx(1.4)


def test_fixed_path_uses_future_mean_and_worst_regret() -> None:
    frame = pd.DataFrame(
        {
            "currency": "KZT",
            "effective_date": pd.date_range("2025-01-01", periods=4),
            "rub_per_unit": [10.0, 11.0, 9.0, 10.0],
        }
    )

    outcomes = lar_path_outcomes(frame, horizon=2)

    assert outcomes.loc[0, "path_value_bps"] == pytest.approx(0.0)
    assert outcomes.loc[0, "path_regret_bps"] == pytest.approx(10_000 / 9)
    assert outcomes.iloc[-2:].isna().all().all()


def test_path_outcome_is_missing_when_current_price_is_missing() -> None:
    frame = pd.DataFrame(
        {
            "currency": "KZT",
            "effective_date": pd.date_range("2025-01-01", periods=3),
            "rub_per_unit": [np.nan, 11.0, 12.0],
        }
    )

    outcomes = lar_path_outcomes(frame, horizon=1)

    assert outcomes.loc[0].isna().all()


def test_discounted_path_downweights_a_later_opportunity() -> None:
    frame = pd.DataFrame(
        {
            "currency": "KZT",
            "effective_date": pd.date_range("2025-01-01", periods=4),
            "rub_per_unit": [10.0, 11.0, 9.0, 10.0],
        }
    )

    outcomes = lar_path_outcomes(frame, horizon=2, mean_gap_days=1.0)
    discount = math.exp(-1.0)
    expected_value = (1_000.0 - discount * 1_000.0) / (1.0 + discount)
    expected_regret = discount * 10_000 / 9

    assert outcomes.loc[0, "path_value_bps"] == pytest.approx(expected_value)
    assert outcomes.loc[0, "path_regret_bps"] == pytest.approx(expected_regret)


def test_fixed_and_discounted_lar_match_at_horizon_one() -> None:
    frame = pd.DataFrame(
        {
            "currency": "KZT",
            "effective_date": pd.date_range("2025-01-01", periods=6),
            "rub_per_unit": [10.0, 11.0, 10.0, 9.0, 10.0, 11.0],
            "target": [True, False, False, True, True, pd.NA],
            "signal": [True, False, False, True, False, False],
        }
    )
    baselines = np.array(
        [
            [False, True, True, False, False, False],
            [True, False, False, False, True, False],
        ]
    )

    fixed = fixed_window_lift_at_risk(
        frame,
        horizon=1,
        target_col="target",
        signal_col="signal",
        baseline_signals=baselines,
    )
    discounted = discounted_lift_at_risk(
        frame,
        horizon=1,
        target_col="target",
        signal_col="signal",
        mean_gap_days=30.0,
        baseline_signals=baselines,
    )

    assert fixed["variant"] == "fixed"
    assert discounted["variant"] == "discounted"
    for key in (
        "lift",
        "signal_value_bps",
        "random_value_bps",
        "signal_tail_regret_bps",
        "random_tail_regret_bps",
        "delta_utility_bps",
        "lift_at_risk",
    ):
        assert discounted[key] == pytest.approx(fixed[key])


def test_lar_evaluator_reports_and_excludes_incomplete_signals() -> None:
    frame = pd.DataFrame(
        {
            "currency": "KZT",
            "effective_date": pd.date_range("2025-01-01", periods=5),
            "rub_per_unit": [10.0, 11.0, 10.0, 9.0, 10.0],
            "target": [True, False, True, False, pd.NA],
            "signal": [True, False, False, False, True],
        }
    )
    baseline = np.array([False, True, False, False, False])

    result = fixed_window_lift_at_risk(
        frame,
        horizon=1,
        target_col="target",
        signal_col="signal",
        baseline_signals=baseline,
    )

    assert result["eligible_count"] == 4
    assert result["signal_count"] == 1
    assert result["excluded_signal_count"] == 1
    assert result["baseline_draws"] == 1
    assert result["signal_hit_rate"] == 1.0
    assert result["random_hit_rate"] == 0.0
    assert math.isnan(result["lift"])
    assert math.isnan(result["lift_at_risk"])


def test_baseline_stream_must_match_signal_count() -> None:
    frame = pd.DataFrame(
        {
            "currency": "KZT",
            "effective_date": pd.date_range("2025-01-01", periods=4),
            "rub_per_unit": [10.0, 11.0, 10.0, 9.0],
            "target": [True, False, True, pd.NA],
            "signal": [True, False, False, False],
        }
    )

    with pytest.raises(ValueError, match="match the evaluated signal count"):
        fixed_window_lift_at_risk(
            frame,
            horizon=1,
            target_col="target",
            signal_col="signal",
            baseline_signals=np.array([True, True, False, False]),
        )
