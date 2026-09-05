from __future__ import annotations

import numpy as np
import pandas as pd

from fx_signal.metrics.customer_value_and_regret import (
    evaluate_customer_outcomes,
    forward_bps,
)
from fx_signal.metrics.frequency import (
    cluster_share,
    push_frequency_summary,
    signals_per_week,
    useful_signals_per_100_weeks,
    useful_signals_per_week,
)

INDICATORS = ("momentum", "reversal", "seasonality", "level")


def hit_rate(target: pd.Series, signal: pd.Series) -> float:
    """Return the share of sent signals whose message-specific target is true."""
    if len(target) != len(signal):
        raise ValueError("target and signal must have the same length")
    target = target.reset_index(drop=True)
    signal = signal.reset_index(drop=True)
    eligible = target.notna()
    truth = target.loc[eligible].astype(bool).reset_index(drop=True)
    sent = signal.loc[eligible].fillna(False).astype(bool).reset_index(drop=True)
    if not sent.any():
        return float("nan")
    return float(truth.loc[sent].mean())


def lift_score(target: pd.Series, signal: pd.Series) -> float:
    """Return signal hit rate divided by target prevalence on eligible dates."""
    if len(target) != len(signal):
        raise ValueError("target and signal must have the same length")
    eligible = target.notna()
    truth = target.loc[eligible].astype(bool)
    if truth.empty:
        return float("nan")
    baseline = float(truth.mean())
    score = hit_rate(target, signal)
    if baseline == 0.0 or np.isnan(score):
        return float("nan")
    return score / baseline


def evaluate_lift(
    frame: pd.DataFrame, horizon: int = 3, indicators: tuple[str, ...] = INDICATORS
) -> pd.DataFrame:
    """Calculate per-currency lift for each FX timing signal."""
    target_col = f"target_local_min_h{horizon}"
    required = {"currency", "effective_date", target_col}
    required.update(f"signal_{name}" for name in indicators)
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []

    for currency, group in frame.groupby("currency"):
        for indicator in indicators:
            eligible = group.dropna(subset=[target_col])
            target = eligible[target_col].astype(bool)
            signal = eligible[f"signal_{indicator}"].fillna(False).astype(bool)
            random_hit_rate = float(target.mean()) if len(target) else float("nan")
            signal_count = int(signal.sum())
            signal_hit_rate = hit_rate(target, signal)
            rows.append(
                {
                    "horizon": horizon,
                    "corridor": f"RUB->{currency}",
                    "indicator": indicator,
                    "eligible_days": len(eligible),
                    "signal_count": signal_count,
                    "target_positives": int(target.sum()),
                    "signal_hit_rate": signal_hit_rate,
                    "random_hit_rate": random_hit_rate,
                    "lift": lift_score(target, signal),
                    "signals_per_week": signals_per_week(eligible["effective_date"], signal),
                }
            )
    return pd.DataFrame(rows)


def frequency_matched_hit_rate(
    target: pd.Series,
    signal: pd.Series,
    *,
    draws: int = 50,
    rng: np.random.Generator | None = None,
) -> float:
    if len(target) != len(signal):
        raise ValueError("target and signal must have the same length")
    if draws < 1:
        raise ValueError("draws must be at least 1")
    target = target.reset_index(drop=True)
    signal = signal.reset_index(drop=True)
    eligible = target.notna()
    values = target.loc[eligible].astype(bool).to_numpy()
    sent = signal.loc[eligible].fillna(False).astype(bool)
    n_signal = int(sent.sum())
    if n_signal == 0 or n_signal > len(values):
        return float("nan")
    rng = rng or np.random.default_rng(0)
    hits = [
        float(values[rng.choice(len(values), size=n_signal, replace=False)].mean())
        for _ in range(draws)
    ]
    return float(np.mean(hits))


def evaluate_method(
    frame: pd.DataFrame,
    *,
    horizon: int,
    method: str,
    split: str,
    target_col: str,
    signal_col: str,
    rng: np.random.Generator | None = None,
    weekly_limit: int = 2,
    cooldown_days: int = 3,
) -> pd.DataFrame:
    required = {"currency", "effective_date", "rub_per_unit", target_col, signal_col}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    rows: list[dict] = []
    rng = rng or np.random.default_rng(0)
    for currency, full_group in frame.groupby("currency", sort=False):
        full_group = full_group.sort_values("effective_date", kind="stable")
        group = full_group.dropna(subset=[target_col, "rub_per_unit"]).copy()
        target = group[target_col].astype(bool)
        signal = group[signal_col].fillna(False).astype(bool)
        signal_count = int(signal.sum())
        random_hit_rate = float(target.mean()) if len(target) else float("nan")
        signal_hit_rate = hit_rate(target, signal)
        matched = frequency_matched_hit_rate(target, signal, rng=rng)
        forward_col = f"forward_mean_h{horizon}"
        if forward_col in group:
            bps = forward_bps(group["rub_per_unit"], group[forward_col])
            sent_bps = bps.loc[signal].dropna()
            bps_forward = float(sent_bps.mean()) if len(sent_bps) else float("nan")
        else:
            bps_forward = float("nan")
        customer = evaluate_customer_outcomes(
            group,
            signal_col=signal_col,
            horizon=horizon,
        )
        frequency = push_frequency_summary(
            group["effective_date"],
            signal,
            weekly_limit=weekly_limit,
            cooldown_days=cooldown_days,
        )
        useful_per_100 = useful_signals_per_100_weeks(
            group["effective_date"], target, signal
        )
        useful_per_week = useful_signals_per_week(group["effective_date"], target, signal)
        rows.append(
            {
                "horizon": horizon,
                "corridor": f"RUB->{currency}",
                "method": method,
                "split": split,
                "eligible_days": len(group),
                "signal_count": signal_count,
                "target_positives": int(target.sum()),
                "signal_hit_rate": signal_hit_rate,
                "random_hit_rate": random_hit_rate,
                "freq_matched_hit_rate": matched,
                "lift": lift_score(target, signal),
                "lift_freq_matched": signal_hit_rate / matched
                if signal_count and matched
                else float("nan"),
                "bps_forward": bps_forward,
                **customer,
                "useful_signals_per_week": useful_per_week,
                "useful_signals_per_100_weeks": useful_per_100,
                **frequency,
                "signals_per_week": frequency["pushes_per_week"],
                "cluster_share": cluster_share(signal),
            }
        )
    return pd.DataFrame(rows)
