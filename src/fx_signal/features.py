from __future__ import annotations

from bisect import bisect_left

import holidays
import numpy as np
import pandas as pd

from fx_signal.external import safe_series_name
from fx_signal.indicators import (
    ALL_SIGNAL_COLUMNS,
    COUNTRY_BY_CURRENCY,
    RESEARCH_SIGNAL_COLUMNS,
    add_baseline_indicators,
    add_research_indicators,
)

GROUP_A = (
    "signal_momentum",
    "signal_level",
    "signal_reversal",
    "signal_seasonality",
)
GROUP_B = (
    "down_streak",
    "price_percentile",
    "days_since_min",
    "rebound_pct",
    "days_to_holiday",
)
GROUP_C = ("ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "vol_20", "slope_20")
GROUP_E = ("month", "weekday")
CONTEXT_CURRENCIES = ("USD", "EUR", "CNY")
EXTERNAL_SUFFIXES = ("ret_1", "ret_5", "percentile", "vol_20", "staleness_days")


def _down_streak(price: pd.Series) -> pd.Series:
    falling = price.diff().lt(0).fillna(False)
    groups = (~falling).cumsum()
    return falling.groupby(groups).cumsum().astype(float)


def _rolling_percentile(price: pd.Series, window: int) -> pd.Series:
    return price.rolling(window, min_periods=window).apply(
        lambda values: float((values <= values[-1]).mean()), raw=True
    )


def _days_since_min(price: pd.Series, window: int) -> pd.Series:
    return price.rolling(window, min_periods=window).apply(
        lambda values: float(len(values) - 1 - int(np.argmin(values))), raw=True
    )


def _slope(price: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)

    def fit(values: np.ndarray) -> float:
        return float(np.polyfit(x, values, 1)[0])

    return price.rolling(window, min_periods=window).apply(fit, raw=True)


def _days_to_holiday(dates: pd.Series, currency: str) -> pd.Series:
    if currency not in COUNTRY_BY_CURRENCY:
        return pd.Series(np.nan, index=dates.index)
    country = COUNTRY_BY_CURRENCY[currency]
    start_year, end_year = int(dates.dt.year.min()), int(dates.dt.year.max()) + 1
    calendar = holidays.country_holidays(country, years=range(start_year, end_year + 1))
    holiday_dates = sorted(calendar.keys())

    def gap(timestamp: pd.Timestamp) -> float:
        current = timestamp.date()
        index = bisect_left(holiday_dates, current)
        if index == len(holiday_dates):
            return np.nan
        return float((holiday_dates[index] - current).days)

    return dates.map(gap).astype(float)


def _corridor_features(
    group: pd.DataFrame,
    *,
    level_window: int,
    reversal_window: int,
) -> pd.DataFrame:
    price = group["rub_per_unit"]
    group = group.copy()
    group["down_streak"] = _down_streak(price)
    group["price_percentile"] = _rolling_percentile(price, level_window)
    group["days_since_min"] = _days_since_min(price, reversal_window)
    rolling_min = price.rolling(reversal_window, min_periods=reversal_window).min()
    group["rebound_pct"] = (price / rolling_min - 1.0).where(rolling_min.gt(0))
    group["days_to_holiday"] = _days_to_holiday(
        group["effective_date"], group["currency"].iloc[0]
    )
    for horizon in (1, 3, 5, 10, 20):
        group[f"ret_{horizon}"] = price.pct_change(horizon)
    group["vol_20"] = price.pct_change().rolling(20, min_periods=20).std()
    group["slope_20"] = _slope(price, 20)
    group["month"] = group["effective_date"].dt.month.astype(float)
    group["weekday"] = group["effective_date"].dt.weekday.astype(float)
    return group


def _align_context(rates: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    dates = rates[["effective_date"]].drop_duplicates().sort_values("effective_date")
    wide = dates.copy()
    for currency, group in context.groupby("currency", sort=False):
        series = (
            group.sort_values("effective_date")[["effective_date", "rub_per_unit"]]
            .drop_duplicates("effective_date")
            .rename(columns={"rub_per_unit": currency.lower()})
        )
        wide = wide.merge(series, on="effective_date", how="left")
    for currency in CONTEXT_CURRENCIES:
        col = currency.lower()
        if col not in wide.columns:
            continue
        wide[col] = wide[col].ffill()
        wide[f"{col}_ret_1"] = wide[col].pct_change(1)
        wide[f"{col}_ret_5"] = wide[col].pct_change(5)
        wide[f"{col}_percentile"] = _rolling_percentile(wide[col], 90)
        wide[f"{col}_vol_20"] = wide[col].pct_change().rolling(20, min_periods=20).std()
        wide = wide.drop(columns=[col])
    return wide


def _align_external(rates: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    """Align the last actually available external observation to each decision date."""
    decisions = (
        rates[["effective_date"]]
        .drop_duplicates()
        .sort_values("effective_date")
        .rename(columns={"effective_date": "decision_at"})
    )
    result = decisions.copy()
    for series_id, group in external.groupby("series_id", sort=True):
        prefix = f"ext_{safe_series_name(str(series_id))}"
        observations = group.sort_values("available_at").copy()
        observations = observations.drop_duplicates("available_at", keep="last")
        aligned = pd.merge_asof(
            decisions,
            observations[["available_at", "value"]],
            left_on="decision_at",
            right_on="available_at",
            direction="backward",
            allow_exact_matches=True,
        )
        values = aligned["value"]
        result[f"{prefix}_ret_1"] = values.pct_change(1)
        result[f"{prefix}_ret_5"] = values.pct_change(5)
        result[f"{prefix}_percentile"] = _rolling_percentile(values, 90)
        result[f"{prefix}_vol_20"] = values.pct_change().rolling(20, min_periods=20).std()
        result[f"{prefix}_staleness_days"] = (
            aligned["decision_at"] - aligned["available_at"]
        ).dt.total_seconds() / 86400
    return result.rename(columns={"decision_at": "effective_date"})


def group_d_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    columns: list[str] = []
    for currency in CONTEXT_CURRENCIES:
        col = currency.lower()
        for suffix in ("ret_1", "ret_5", "percentile", "vol_20"):
            name = f"{col}_{suffix}"
            if name in frame.columns:
                columns.append(name)
    if "usd_ret_1" in frame.columns and "ret_1" in frame.columns:
        columns.append("residual_usd_1")
    for prefix in ("local_per_usd", "local_per_cny"):
        for suffix in ("ret_1", "ret_5", "percentile", "vol_20"):
            name = f"{prefix}_{suffix}"
            if name in frame.columns:
                columns.append(name)
    return tuple(columns)


def group_f_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        column
        for column in frame.columns
        if column.startswith("ext_") and column.endswith(EXTERNAL_SUFFIXES)
    )


def add_features(
    rates: pd.DataFrame,
    *,
    context: pd.DataFrame | None = None,
    external: pd.DataFrame | None = None,
    momentum_days: int = 3,
    level_window: int = 90,
    level_quantile: float = 0.10,
    reversal_window: int = 20,
    holiday_lookahead_days: int = 7,
) -> pd.DataFrame:
    if "signal_momentum" not in rates.columns:
        rates = add_baseline_indicators(
            rates,
            momentum_days=momentum_days,
            level_window=level_window,
            level_quantile=level_quantile,
            reversal_window=reversal_window,
            holiday_lookahead_days=holiday_lookahead_days,
        )
    if RESEARCH_SIGNAL_COLUMNS[0] not in rates.columns:
        rates = add_research_indicators(rates)
    frames = [
        _corridor_features(
            group, level_window=level_window, reversal_window=reversal_window
        )
        for _, group in rates.groupby("currency", sort=False)
    ]
    result = pd.concat(frames).sort_index()
    if context is not None and not context.empty:
        aligned = _align_context(result, context)
        result = result.merge(aligned, on="effective_date", how="left")
        if "usd_ret_1" in result.columns:
            result["residual_usd_1"] = result["ret_1"] - result["usd_ret_1"]
        for context_currency in ("usd", "cny"):
            context_return = f"{context_currency}_ret_1"
            if context_return not in result.columns:
                continue
            local_prefix = f"local_per_{context_currency}"
            # dlog(LOCAL/CTX) = dlog(RUB/CTX) - dlog(RUB/LOCAL).
            local_ret_1 = result[context_return] - result["ret_1"]
            result[f"{local_prefix}_ret_1"] = local_ret_1
            result[f"{local_prefix}_ret_5"] = (
                result[f"{context_currency}_ret_5"] - result["ret_5"]
            )
            synthetic_level = (
                1.0 + local_ret_1.fillna(0.0)
            ).groupby(result["currency"], sort=False).cumprod()
            result[f"{local_prefix}_percentile"] = synthetic_level.groupby(
                result["currency"], sort=False
            ).transform(
                lambda values: _rolling_percentile(values, 90)
            )
            result[f"{local_prefix}_vol_20"] = local_ret_1.groupby(
                result["currency"], sort=False
            ).transform(lambda values: values.rolling(20, min_periods=20).std())
    if external is not None and not external.empty:
        aligned_external = _align_external(result, external)
        result = result.merge(aligned_external, on="effective_date", how="left")
    facts = result[list(ALL_SIGNAL_COLUMNS)].astype("boolean")
    result["has_fact"] = facts.fillna(False).any(axis=1)
    return result


def columns_for_groups(groups: list[str], frame: pd.DataFrame) -> list[str]:
    mapping = {
        "A": GROUP_A,
        "B": GROUP_B,
        "C": GROUP_C,
        "E": GROUP_E,
        "D": group_d_columns(frame),
        "F": group_f_columns(frame),
    }
    selected: list[str] = []
    for group in groups:
        if group not in mapping:
            raise ValueError(f"Unknown feature group: {group}")
        for column in mapping[group]:
            if column in frame.columns and column not in selected:
                selected.append(column)
    return selected
