from __future__ import annotations

from bisect import bisect_left

import holidays
import pandas as pd

COUNTRY_BY_CURRENCY = {"AMD": "AM", "KZT": "KZ", "KGS": "KG", "TJS": "TJ", "UZS": "UZ"}

BASELINE_SIGNAL_COLUMNS = (
    "signal_momentum",
    "signal_level",
    "signal_reversal",
    "signal_seasonality",
)

RESEARCH_SIGNAL_COLUMNS = (
    "signal_better_than_one_year_ago",
    "signal_better_range_held",
    "signal_larger_than_usual_latest_improvement",
    "signal_better_than_30_day_average",
    "signal_most_recent_changes_favourable",
    "signal_less_than_one_year_ago",
    "signal_most_recent_changes_unfavourable",
)

ALL_SIGNAL_COLUMNS = BASELINE_SIGNAL_COLUMNS + RESEARCH_SIGNAL_COLUMNS
SIGNAL_EFFECT_COLUMN = {signal: f"{signal}_effect" for signal in ALL_SIGNAL_COLUMNS}
BASELINE_SIGNAL_EFFECT_COLUMNS = tuple(
    SIGNAL_EFFECT_COLUMN[signal] for signal in BASELINE_SIGNAL_COLUMNS
)
RESEARCH_SIGNAL_EFFECT_COLUMN = {
    signal: SIGNAL_EFFECT_COLUMN[signal] for signal in RESEARCH_SIGNAL_COLUMNS
}
RESEARCH_SIGNAL_EFFECT_COLUMNS = tuple(RESEARCH_SIGNAL_EFFECT_COLUMN.values())


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


def _recipient_per_rub(price: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(price, errors="coerce")
    return numeric.rdiv(1.0).where(numeric.gt(0))


def _rising_edge(condition: pd.Series) -> pd.Series:
    current = condition.astype("boolean").fillna(False).astype(bool)
    previous = current.shift(1, fill_value=False)
    return (current & ~previous).where(condition.notna()).astype("boolean")


def _threshold_crossing(value: pd.Series, threshold: float) -> pd.Series:
    previous = value.shift(1)
    valid = value.notna() & previous.notna()
    crossing = value.gt(threshold) & previous.le(threshold)
    return crossing.where(valid).astype("boolean")


def _annual_reference(
    dates: pd.Series,
    recipient_per_rub: pd.Series,
    max_gap_days: int,
) -> pd.Series:
    left = pd.DataFrame(
        {
            "target_date": (dates - pd.DateOffset(years=1)).to_numpy(),
            "position": range(len(dates)),
        }
    ).sort_values("target_date")
    right = pd.DataFrame(
        {
            "reference_date": dates.to_numpy(),
            "reference_value": recipient_per_rub.to_numpy(),
        }
    ).sort_values("reference_date")
    matched = pd.merge_asof(
        left,
        right,
        left_on="target_date",
        right_on="reference_date",
        direction="backward",
        tolerance=pd.Timedelta(max_gap_days, unit="D"),
    ).sort_values("position")
    return pd.Series(
        matched["reference_value"].to_numpy(),
        index=recipient_per_rub.index,
        dtype=float,
    )


def _calendar_window_stats(
    dates: pd.Series,
    values: pd.Series,
    window_days: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    date_index = pd.DatetimeIndex(dates.to_numpy())
    indexed_values = pd.Series(values.to_numpy(), index=date_index, dtype=float)
    window = indexed_values.rolling(f"{window_days}D", min_periods=1)

    date_numbers = pd.Series(date_index.asi8.astype(float), index=date_index)
    oldest_date = date_numbers.rolling(f"{window_days}D", min_periods=1).min()
    nanoseconds_per_day = float(pd.Timedelta(1, unit="D").value)
    coverage_days = (date_numbers - oldest_date) / nanoseconds_per_day

    return (
        pd.Series(window.mean().to_numpy(), index=values.index, dtype=float),
        pd.Series(window.count().to_numpy(), index=values.index, dtype=float),
        pd.Series(coverage_days.to_numpy(), index=values.index, dtype=float),
    )


def _add_research_indicators_to_group(
    group: pd.DataFrame,
    *,
    year_gain_threshold: float,
    range_gain_threshold: float,
    unusual_improvement_floor: float,
    unusual_move_multiplier: float,
    unusual_move_epsilon: float,
    average_30_day_gain_threshold: float,
    recent_gain_threshold: float,
    year_loss_threshold: float,
    recent_loss_threshold: float,
    annual_reference_max_gap_days: int,
) -> pd.DataFrame:
    result = group.sort_values("effective_date").copy()
    dates = pd.to_datetime(result["effective_date"])
    recipient = _recipient_per_rub(result["rub_per_unit"])

    annual_reference = _annual_reference(
        dates,
        recipient,
        max_gap_days=annual_reference_max_gap_days,
    )
    annual_gain = recipient / annual_reference - 1.0
    annual_loss = 1.0 - recipient / annual_reference
    result[RESEARCH_SIGNAL_EFFECT_COLUMN["signal_better_than_one_year_ago"]] = annual_gain
    result[RESEARCH_SIGNAL_EFFECT_COLUMN["signal_less_than_one_year_ago"]] = annual_gain
    result["signal_better_than_one_year_ago"] = _threshold_crossing(
        annual_gain,
        year_gain_threshold,
    )
    result["signal_less_than_one_year_ago"] = _threshold_crossing(
        annual_loss,
        year_loss_threshold,
    )

    latest_min = recipient.rolling(3, min_periods=3).min()
    earlier_max = recipient.shift(3).rolling(7, min_periods=7).max()
    range_gain = latest_min / earlier_max - 1.0
    result[RESEARCH_SIGNAL_EFFECT_COLUMN["signal_better_range_held"]] = range_gain
    range_condition = range_gain.gt(range_gain_threshold).where(range_gain.notna())
    result["signal_better_range_held"] = _rising_edge(range_condition)

    latest_move = recipient / recipient.shift(1) - 1.0
    usual_move = latest_move.abs().shift(1).rolling(10, min_periods=10).median()
    unusual_baseline = usual_move.clip(lower=unusual_move_epsilon)
    result[RESEARCH_SIGNAL_EFFECT_COLUMN["signal_larger_than_usual_latest_improvement"]] = (
        latest_move
    )
    unusual_condition = (
        latest_move.gt(unusual_improvement_floor)
        & latest_move.ge(unusual_move_multiplier * unusual_baseline)
    ).where(latest_move.notna() & usual_move.notna())
    result["signal_larger_than_usual_latest_improvement"] = _rising_edge(unusual_condition)

    average_30_day, count_30_day, coverage_30_day = _calendar_window_stats(
        dates,
        recipient,
        window_days=30,
    )
    gain_vs_average = recipient / average_30_day - 1.0
    valid_average = count_30_day.ge(15) & coverage_30_day.ge(25)
    result[RESEARCH_SIGNAL_EFFECT_COLUMN["signal_better_than_30_day_average"]] = (
        gain_vs_average.where(valid_average)
    )
    result["signal_better_than_30_day_average"] = _threshold_crossing(
        gain_vs_average.where(valid_average),
        average_30_day_gain_threshold,
    )

    change = recipient.diff()
    favourable = change.gt(0).where(change.notna())
    unfavourable = change.lt(0).where(change.notna())
    favourable_count = favourable.rolling(5, min_periods=5).sum()
    unfavourable_count = unfavourable.rolling(5, min_periods=5).sum()
    recent_gain = recipient / recipient.shift(5) - 1.0
    recent_loss = 1.0 - recipient / recipient.shift(5)
    result[RESEARCH_SIGNAL_EFFECT_COLUMN["signal_most_recent_changes_favourable"]] = recent_gain
    result[RESEARCH_SIGNAL_EFFECT_COLUMN["signal_most_recent_changes_unfavourable"]] = recent_gain

    favourable_condition = (favourable_count.ge(4) & recent_gain.gt(recent_gain_threshold)).where(
        favourable_count.notna() & recent_gain.notna()
    )
    result["signal_most_recent_changes_favourable"] = _rising_edge(favourable_condition)

    unfavourable_condition = (
        unfavourable_count.ge(4) & recent_loss.gt(recent_loss_threshold)
    ).where(unfavourable_count.notna() & recent_loss.notna())
    result["signal_most_recent_changes_unfavourable"] = _rising_edge(unfavourable_condition)
    return result


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
        recipient = _recipient_per_rub(price)
        group["signal_momentum"] = _momentum(price, momentum_days)
        group[SIGNAL_EFFECT_COLUMN["signal_momentum"]] = (
            recipient / recipient.shift(momentum_days) - 1.0
        )
        group["signal_level"] = _level(price, level_window, level_quantile)
        level_threshold = price.rolling(level_window, min_periods=level_window).quantile(
            level_quantile
        )
        group[SIGNAL_EFFECT_COLUMN["signal_level"]] = (level_threshold / price - 1.0).where(
            price.gt(0)
        )
        group["signal_reversal"] = _reversal(price, reversal_window)
        group[SIGNAL_EFFECT_COLUMN["signal_reversal"]] = recipient / recipient.shift(1) - 1.0
        group["signal_seasonality"] = _holiday_signal(
            group["effective_date"], currency, holiday_lookahead_days
        )
        group[SIGNAL_EFFECT_COLUMN["signal_seasonality"]] = float("nan")
        result_frames.append(group)
    return pd.concat(result_frames).sort_index()


def add_research_indicators(
    rates: pd.DataFrame,
    *,
    year_gain_threshold: float = 0.0,
    range_gain_threshold: float = 0.0,
    unusual_improvement_floor: float = 0.0,
    unusual_move_multiplier: float = 2.0,
    unusual_move_epsilon: float = 1e-12,
    average_30_day_gain_threshold: float = 0.0,
    recent_gain_threshold: float = 0.0,
    year_loss_threshold: float = 0.0,
    recent_loss_threshold: float = 0.0,
    annual_reference_max_gap_days: int = 7,
) -> pd.DataFrame:
    """Add the seven research signals and their signed effects.

    Gain and loss thresholds are decimal fractions. Their zero defaults expose the raw signal
    direction for research; production thresholds must be calibrated per corridor. Effect columns
    are also decimal fractions: positive means more recipient currency and negative means less.
    The research signals are allowed as send-time explanations, but they are not added to
    the model feature groups.
    """
    required = {"currency", "effective_date", "rub_per_unit"}
    missing = required.difference(rates.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    thresholds = {
        "year_gain_threshold": year_gain_threshold,
        "range_gain_threshold": range_gain_threshold,
        "unusual_improvement_floor": unusual_improvement_floor,
        "average_30_day_gain_threshold": average_30_day_gain_threshold,
        "recent_gain_threshold": recent_gain_threshold,
        "year_loss_threshold": year_loss_threshold,
        "recent_loss_threshold": recent_loss_threshold,
    }
    invalid = [name for name, value in thresholds.items() if value < 0]
    if invalid:
        raise ValueError(f"Thresholds must be non-negative: {invalid}")
    if unusual_move_multiplier <= 0:
        raise ValueError("unusual_move_multiplier must be positive")
    if unusual_move_epsilon <= 0:
        raise ValueError("unusual_move_epsilon must be positive")
    if annual_reference_max_gap_days < 0:
        raise ValueError("annual_reference_max_gap_days must be non-negative")

    if rates.empty:
        result = rates.copy()
        for column in RESEARCH_SIGNAL_COLUMNS:
            result[column] = pd.Series(index=result.index, dtype="boolean")
        for column in RESEARCH_SIGNAL_EFFECT_COLUMNS:
            result[column] = pd.Series(index=result.index, dtype=float)
        return result

    result_frames = [
        _add_research_indicators_to_group(
            group,
            year_gain_threshold=year_gain_threshold,
            range_gain_threshold=range_gain_threshold,
            unusual_improvement_floor=unusual_improvement_floor,
            unusual_move_multiplier=unusual_move_multiplier,
            unusual_move_epsilon=unusual_move_epsilon,
            average_30_day_gain_threshold=average_30_day_gain_threshold,
            recent_gain_threshold=recent_gain_threshold,
            year_loss_threshold=year_loss_threshold,
            recent_loss_threshold=recent_loss_threshold,
            annual_reference_max_gap_days=annual_reference_max_gap_days,
        )
        for _, group in rates.groupby("currency", sort=False)
    ]
    return pd.concat(result_frames).sort_index()
