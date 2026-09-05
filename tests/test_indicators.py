import pandas as pd

from fx_signal.indicators import (
    RESEARCH_SIGNAL_COLUMNS,
    add_baseline_indicators,
    add_research_indicators,
)


def _rates_from_recipient(
    currency: str,
    dates: pd.DatetimeIndex | list[str],
    recipient_per_rub: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "currency": currency,
            "effective_date": pd.to_datetime(dates),
            "rub_per_unit": [1.0 / value for value in recipient_per_rub],
        }
    )


def test_momentum_level_and_reversal() -> None:
    dates = pd.date_range("2025-01-01", periods=12, freq="D")
    prices = [5.0] * 7 + [4.0, 3.0, 2.0, 1.0, 1.5]
    rates = pd.DataFrame({"currency": "TJS", "effective_date": dates, "rub_per_unit": prices})
    result = add_baseline_indicators(rates, momentum_days=3, level_window=4, reversal_window=4)
    assert result.loc[10, "signal_momentum"]
    assert result.loc[10, "signal_level"]
    assert result.loc[11, "signal_reversal"]


def test_indicators_do_not_change_when_future_is_appended() -> None:
    dates = pd.date_range("2025-01-01", periods=25, freq="D")
    rates = pd.DataFrame(
        {"currency": "TJS", "effective_date": dates, "rub_per_unit": range(25, 0, -1)}
    )
    short = add_baseline_indicators(rates.iloc[:20], level_window=5, reversal_window=5)
    full = add_baseline_indicators(rates, level_window=5, reversal_window=5)
    columns = ["signal_momentum", "signal_level", "signal_reversal", "signal_seasonality"]
    pd.testing.assert_frame_equal(short[columns], full.iloc[:20][columns])


def test_research_annual_gain_and_loss_signals() -> None:
    dates = ["2024-01-01", "2024-01-02", "2025-01-01", "2025-01-02"]
    better = _rates_from_recipient("TJS", dates, [1.0, 1.0, 1.0, 1.1])
    worse = _rates_from_recipient("KZT", dates, [1.0, 1.0, 1.0, 0.9])
    rates = pd.concat([better, worse], ignore_index=True)

    result = add_research_indicators(
        rates,
        year_gain_threshold=0.05,
        year_loss_threshold=0.05,
    )
    latest = result[result["effective_date"].eq(pd.Timestamp("2025-01-02"))]
    better_latest = latest[latest["currency"].eq("TJS")].iloc[0]
    worse_latest = latest[latest["currency"].eq("KZT")].iloc[0]

    assert bool(better_latest["signal_better_than_one_year_ago"])
    assert not bool(better_latest["signal_less_than_one_year_ago"])
    assert bool(worse_latest["signal_less_than_one_year_ago"])
    assert not bool(worse_latest["signal_better_than_one_year_ago"])


def test_research_short_history_signals() -> None:
    range_rates = _rates_from_recipient(
        "TJS",
        pd.date_range("2025-01-01", periods=10, freq="D"),
        [1.0] * 7 + [1.1] * 3,
    )
    usual_moves = [1.0 * 1.001**step for step in range(11)]
    unusual_rates = _rates_from_recipient(
        "KZT",
        pd.date_range("2025-01-01", periods=12, freq="D"),
        usual_moves + [usual_moves[-1] * 1.05],
    )
    average_rates = _rates_from_recipient(
        "AMD",
        pd.date_range("2025-01-01", periods=31, freq="D"),
        [1.0] * 30 + [1.2],
    )
    favourable_rates = _rates_from_recipient(
        "KGS",
        pd.date_range("2025-01-01", periods=6, freq="D"),
        [1.0, 1.02, 1.04, 1.03, 1.05, 1.07],
    )
    unfavourable_rates = _rates_from_recipient(
        "UZS",
        pd.date_range("2025-01-01", periods=6, freq="D"),
        [1.0, 0.98, 0.96, 0.97, 0.95, 0.93],
    )
    rates = pd.concat(
        [
            range_rates,
            unusual_rates,
            average_rates,
            favourable_rates,
            unfavourable_rates,
        ],
        ignore_index=True,
    )

    result = add_research_indicators(
        rates,
        range_gain_threshold=0.05,
        unusual_improvement_floor=0.02,
        average_30_day_gain_threshold=0.05,
        recent_gain_threshold=0.03,
        recent_loss_threshold=0.03,
    )
    latest = result.groupby("currency").tail(1).set_index("currency")

    assert bool(latest.loc["TJS", "signal_better_range_held"])
    assert bool(latest.loc["KZT", "signal_larger_than_usual_latest_improvement"])
    assert bool(latest.loc["AMD", "signal_better_than_30_day_average"])
    assert bool(latest.loc["KGS", "signal_most_recent_changes_favourable"])
    assert bool(latest.loc["UZS", "signal_most_recent_changes_unfavourable"])


def test_research_signals_do_not_use_future_and_are_not_wired_into_baseline() -> None:
    dates = pd.date_range("2023-01-01", periods=430, freq="D")
    recipient = [1.0 + (step % 17) * 0.002 for step in range(len(dates))]
    rates = _rates_from_recipient("TJS", dates, recipient)

    short = add_research_indicators(rates.iloc[:400])
    full = add_research_indicators(rates)
    pd.testing.assert_frame_equal(
        short[list(RESEARCH_SIGNAL_COLUMNS)],
        full.iloc[:400][list(RESEARCH_SIGNAL_COLUMNS)],
    )

    baseline = add_baseline_indicators(rates.iloc[:20], level_window=5, reversal_window=5)
    assert set(RESEARCH_SIGNAL_COLUMNS).isdisjoint(baseline.columns)
