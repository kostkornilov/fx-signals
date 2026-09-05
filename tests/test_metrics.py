import math

import numpy as np
import pandas as pd
import pytest

from fx_signal.metrics import (
    add_customer_outcomes,
    cluster_rate,
    customer_regret_bps,
    empirical_cvar,
    evaluate_customer_outcomes,
    hit_rate,
    lift_score,
    moment_advantage_bps,
    push_frequency_summary,
    summarize_stability,
    useful_signals_per_100_weeks,
    useful_signals_per_week,
    weekly_push_counts,
)


def test_lift_uses_only_eligible_targets() -> None:
    target = pd.Series([True, False, True, pd.NA], index=[4, 5, 6, 7], dtype="boolean")
    signal = pd.Series([True, True, True, True], index=[10, 11, 12, 13])

    assert hit_rate(target, signal) == pytest.approx(2 / 3)
    assert lift_score(target, signal) == pytest.approx(1.0)


def test_lift_is_nan_without_signals_or_positive_baseline() -> None:
    assert math.isnan(lift_score(pd.Series([True, False]), pd.Series([False, False])))
    assert math.isnan(lift_score(pd.Series([False, False]), pd.Series([True, False])))


def test_moment_advantage_and_regret_have_expected_direction() -> None:
    price = pd.Series([10.0, 10.0, 9.0])

    advantage = moment_advantage_bps(price, pd.Series([11.0, 9.0, 9.0]))
    regret = customer_regret_bps(price, pd.Series([9.0, 10.0, 10.0]))

    assert advantage.tolist() == pytest.approx([1_000.0, -1_000.0, 0.0])
    assert regret.tolist() == pytest.approx([10_000 / 9, 0.0, 0.0])


def test_customer_price_metrics_reject_non_positive_prices() -> None:
    with pytest.raises(ValueError, match="positive"):
        moment_advantage_bps(pd.Series([0.0]), pd.Series([1.0]))
    with pytest.raises(ValueError, match="positive"):
        customer_regret_bps(pd.Series([1.0]), pd.Series([-1.0]))


def test_empirical_cvar_averages_the_worst_five_percent() -> None:
    losses = pd.Series(np.arange(1.0, 101.0))

    assert empirical_cvar(losses, confidence=0.95) == pytest.approx(98.0)
    assert empirical_cvar(pd.Series([0.0, 100.0]), confidence=0.95) == 100.0
    with pytest.raises(ValueError, match="non-negative"):
        empirical_cvar(pd.Series([-1.0, 2.0]))


def test_customer_outcomes_use_separate_currency_windows() -> None:
    frame = pd.DataFrame(
        {
            "currency": ["TJS"] * 5 + ["KZT"] * 3,
            "effective_date": pd.date_range("2025-01-01", periods=5).tolist()
            + pd.date_range("2025-01-01", periods=3).tolist(),
            "rub_per_unit": [3.0, 2.0, 1.0, 2.0, 3.0, 100.0, 50.0, 100.0],
            "signal": [False, True, True, False, False, False, True, False],
        }
    )

    result = add_customer_outcomes(frame, horizon=1)

    assert result.loc[2, "surrounding_mean_h1"] == pytest.approx(5 / 3)
    assert result.loc[2, "moment_advantage_bps_h1"] == pytest.approx((5 / 3 - 1) * 10_000)
    assert result.loc[1, "customer_regret_bps_h1"] == pytest.approx(10_000.0)
    assert pd.isna(result.loc[4, "future_best_h1"])
    assert result.loc[6, "surrounding_mean_h1"] == pytest.approx(250 / 3)

    summary = evaluate_customer_outcomes(frame, signal_col="signal", horizon=1)
    assert summary["value_observations"] == 3
    assert summary["regret_observations"] == 3
    assert summary["customer_regret_bps_max"] == pytest.approx(10_000.0)


def test_weekly_push_counts_include_silent_weeks() -> None:
    dates = pd.Series(pd.to_datetime(["2025-01-06", "2025-01-07", "2025-01-20"]))
    signal = pd.Series([True, True, True])

    counts = weekly_push_counts(dates, signal)

    assert counts.tolist() == [2, 0, 1]


def test_useful_signal_yield_is_different_from_push_frequency() -> None:
    dates = pd.Series(pd.to_datetime(["2025-01-06", "2025-01-07", "2025-01-20"]))
    target = pd.Series([True, False, True])
    signal = pd.Series([True, True, True])

    useful = useful_signals_per_100_weeks(dates, target, signal)
    useful_weekly = useful_signals_per_week(dates, target, signal)
    frequency = push_frequency_summary(
        dates,
        signal,
        weekly_limit=1,
        cooldown_days=2,
    )

    assert useful == pytest.approx(200 / 3)
    assert useful_weekly == pytest.approx(2 / 3)
    assert frequency["signal_count"] == 3
    assert frequency["eligible_weeks"] == 3
    assert frequency["pushes_per_week"] == 1.0
    assert frequency["pushes_p50_per_week"] == 1.0
    assert frequency["pushes_p90_per_week"] == pytest.approx(1.8)
    assert frequency["zero_push_week_rate"] == pytest.approx(1 / 3)
    assert frequency["over_budget_week_rate"] == pytest.approx(1 / 3)
    assert frequency["cluster_rate"] == pytest.approx(1 / 3)


def test_cluster_rate_sorts_dates_and_handles_no_pushes() -> None:
    dates = pd.Series(pd.to_datetime(["2025-01-10", "2025-01-01", "2025-01-03"]))

    assert cluster_rate(dates, pd.Series([True, True, True]), cooldown_days=2) == pytest.approx(1 / 3)
    assert math.isnan(cluster_rate(dates, pd.Series([False, False, False])))


def test_stability_summary_keeps_weak_slices_visible() -> None:
    slices = pd.DataFrame(
        {
            "lift": [1.5, 1.2, 1.3],
            "customer_regret_cvar_95_bps": [10.0, 30.0, 20.0],
        }
    )

    result = summarize_stability(
        slices,
        ["lift", "customer_regret_cvar_95_bps"],
        lower_is_better=["customer_regret_cvar_95_bps"],
    ).set_index("metric")

    assert result.loc["lift", "slice_count"] == 3
    assert result.loc["lift", "worst"] == 1.2
    assert result.loc["lift", "median"] == 1.3
    assert result.loc["customer_regret_cvar_95_bps", "worst"] == 30.0
    assert result.loc["customer_regret_cvar_95_bps", "best"] == 10.0
