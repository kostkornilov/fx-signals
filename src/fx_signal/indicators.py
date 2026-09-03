from __future__ import annotations

from bisect import bisect_left

import holidays
import pandas as pd

COUNTRY_BY_CURRENCY = {"AMD": "AM", "KZT": "KZ", "KGS": "KG", "TJS": "TJ", "UZS": "UZ"}


def _momentum(price: pd.Series, days: int) -> pd.Series:
    falling = price.diff().lt(0)
    return falling.rolling(days, min_periods=days).sum().eq(days)


def _level(price: pd.Series, window: int, quantile: float) -> pd.Series:
    threshold = price.rolling(window, min_periods=window).quantile(quantile)
    return price.le(threshold).where(threshold.notna())


def _reversal(price: pd.Series, window: int) -> pd.Series:
    previous = price.shift(1)
    previous_min = previous.rolling(window, min_periods=window).min()
    return (previous.eq(previous_min) & price.gt(previous)).where(previous_min.notna())


def _holiday_signal(dates: pd.Series, currency: str, lookahead: int) -> pd.Series:
    country = COUNTRY_BY_CURRENCY[currency]
    start_year, end_year = int(dates.dt.year.min()), int(dates.dt.year.max()) + 1
    calendar = holidays.country_holidays(country, years=range(start_year, end_year + 1))
    holiday_dates = sorted(calendar.keys())

    def near_holiday(timestamp: pd.Timestamp) -> bool:
        current = timestamp.date()
        index = bisect_left(holiday_dates, current)
        if index == len(holiday_dates):
            return False
        return 0 <= (holiday_dates[index] - current).days <= lookahead

    return dates.map(near_holiday).astype(bool)


def add_baseline_indicators(
    rates: pd.DataFrame,
    momentum_days: int = 3,
    level_window: int = 90,
    level_quantile: float = 0.10,
    reversal_window: int = 20,
    holiday_lookahead_days: int = 7,
) -> pd.DataFrame:
    result_frames: list[pd.DataFrame] = []
    for currency, group in rates.groupby("currency", sort=False):
        group = group.sort_values("effective_date").copy()
        price = group["rub_per_unit"]
        group["signal_momentum"] = _momentum(price, momentum_days)
        group["signal_level"] = _level(price, level_window, level_quantile)
        group["signal_reversal"] = _reversal(price, reversal_window)
        group["signal_seasonality"] = _holiday_signal(
            group["effective_date"], currency, holiday_lookahead_days
        )
        result_frames.append(group)
    return pd.concat(result_frames).sort_index()
