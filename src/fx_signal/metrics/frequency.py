from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def _aligned_events(
    dates: pd.Series | Iterable[object],
    signal: pd.Series | Iterable[bool],
) -> pd.DataFrame:
    date_series = dates.copy() if isinstance(dates, pd.Series) else pd.Series(dates)
    signal_series = signal.copy() if isinstance(signal, pd.Series) else pd.Series(signal)
    if len(date_series) != len(signal_series):
        raise ValueError("dates and signal must have the same length")
    parsed_dates = pd.to_datetime(date_series, errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("dates must not contain missing or invalid values")
    return pd.DataFrame(
        {
            "date": parsed_dates.to_numpy(),
            "signal": signal_series.fillna(False).astype(bool).to_numpy(),
        }
    ).sort_values("date", kind="stable")


def weekly_push_counts(
    dates: pd.Series | Iterable[object],
    signal: pd.Series | Iterable[bool],
) -> pd.Series:
    """Count pushes in every calendar week, including zero-push weeks."""
    events = _aligned_events(dates, signal)
    if events.empty:
        return pd.Series(dtype="int64", name="push_count")
    events["week"] = events["date"].dt.to_period("W-SUN")
    counts = events.groupby("week", sort=True)["signal"].sum().astype(int)
    weeks = pd.period_range(events["week"].min(), events["week"].max(), freq="W-SUN")
    return counts.reindex(weeks, fill_value=0).rename("push_count")


def signals_per_week(
    dates: pd.Series | Iterable[object],
    signal: pd.Series | Iterable[bool],
) -> float:
    """Return average pushes per represented calendar week."""
    counts = weekly_push_counts(dates, signal)
    return float(counts.mean()) if len(counts) else 0.0


def cluster_rate(
    dates: pd.Series | Iterable[object],
    signal: pd.Series | Iterable[bool],
    *,
    cooldown_days: int = 3,
) -> float:
    """Share of pushes no more than `cooldown_days` after the prior push."""
    if cooldown_days < 0:
        raise ValueError("cooldown_days must be non-negative")
    events = _aligned_events(dates, signal)
    sent_dates = events.loc[events["signal"], "date"].reset_index(drop=True)
    if sent_dates.empty:
        return float("nan")
    if len(sent_dates) == 1:
        return 0.0
    clustered = sent_dates.diff().dt.total_seconds().div(86_400).le(cooldown_days)
    return float(clustered.sum() / len(sent_dates))


def cluster_share(signal: pd.Series | Iterable[bool]) -> float:
    """Legacy row-adjacency cluster metric kept for old reports."""
    values = signal.copy() if isinstance(signal, pd.Series) else pd.Series(signal)
    values = values.fillna(False).astype(bool).reset_index(drop=True)
    count = int(values.sum())
    if count == 0:
        return float("nan")
    continued = values & values.shift(1, fill_value=False)
    return float(continued.sum() / count)


def useful_signals_per_100_weeks(
    dates: pd.Series | Iterable[object],
    target: pd.Series | Iterable[bool],
    signal: pd.Series | Iterable[bool],
) -> float:
    """Return correct sent messages per 100 eligible calendar weeks."""
    date_series = dates.copy() if isinstance(dates, pd.Series) else pd.Series(dates)
    target_series = target.copy() if isinstance(target, pd.Series) else pd.Series(target)
    signal_series = signal.copy() if isinstance(signal, pd.Series) else pd.Series(signal)
    if not (len(date_series) == len(target_series) == len(signal_series)):
        raise ValueError("dates, target, and signal must have the same length")
    date_series = date_series.reset_index(drop=True)
    target_series = target_series.reset_index(drop=True)
    signal_series = signal_series.reset_index(drop=True)
    eligible = target_series.notna()
    if not eligible.any():
        return float("nan")
    useful = (
        target_series.loc[eligible].astype(bool).to_numpy()
        & signal_series.loc[eligible].fillna(False).astype(bool).to_numpy()
    )
    counts = weekly_push_counts(date_series.loc[eligible], useful)
    return float(100.0 * counts.sum() / len(counts)) if len(counts) else float("nan")


def useful_signals_per_week(
    dates: pd.Series | Iterable[object],
    target: pd.Series | Iterable[bool],
    signal: pd.Series | Iterable[bool],
) -> float:
    """Return correct sent messages per eligible calendar week."""
    per_100 = useful_signals_per_100_weeks(dates, target, signal)
    return per_100 / 100.0


def push_frequency_summary(
    dates: pd.Series | Iterable[object],
    signal: pd.Series | Iterable[bool],
    *,
    weekly_limit: int = 2,
    cooldown_days: int = 3,
) -> dict[str, float | int]:
    """Summarize raw push load, including silent and over-budget weeks."""
    if weekly_limit < 0:
        raise ValueError("weekly_limit must be non-negative")
    counts = weekly_push_counts(dates, signal)
    signal_series = signal.copy() if isinstance(signal, pd.Series) else pd.Series(signal)
    signal_count = int(signal_series.fillna(False).astype(bool).sum())
    if counts.empty:
        return {
            "signal_count": signal_count,
            "eligible_weeks": 0,
            "pushes_per_week": 0.0,
            "pushes_p50_per_week": 0.0,
            "pushes_p90_per_week": 0.0,
            "zero_push_week_rate": float("nan"),
            "over_budget_week_rate": float("nan"),
            "cluster_rate": float("nan"),
        }
    return {
        "signal_count": signal_count,
        "eligible_weeks": len(counts),
        "pushes_per_week": float(counts.mean()),
        "pushes_p50_per_week": float(counts.quantile(0.50)),
        "pushes_p90_per_week": float(counts.quantile(0.90)),
        "zero_push_week_rate": float(counts.eq(0).mean()),
        "over_budget_week_rate": float(counts.gt(weekly_limit).mean()),
        "cluster_rate": cluster_rate(dates, signal, cooldown_days=cooldown_days),
    }
