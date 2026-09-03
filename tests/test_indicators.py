import pandas as pd

from fx_signal.indicators import add_baseline_indicators


def test_momentum_level_and_reversal() -> None:
    dates = pd.date_range("2025-01-01", periods=12, freq="D")
    prices = [5.0] * 7 + [4.0, 3.0, 2.0, 1.0, 1.5]
    rates = pd.DataFrame(
        {"currency": "TJS", "effective_date": dates, "rub_per_unit": prices}
    )
    result = add_baseline_indicators(
        rates, momentum_days=3, level_window=4, reversal_window=4
    )
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

