from __future__ import annotations

import numpy as np
import pandas as pd

INDICATORS = ("momentum", "reversal", "seasonality", "level")


def evaluate_lift(
    frame: pd.DataFrame, horizon: int = 3, indicators: tuple[str, ...] = INDICATORS
) -> pd.DataFrame:
    """Calculate per-currency lift for each FX timing signal."""
    target_col = f"target_local_min_h{horizon}"
    required = [target_col, *(f"signal_{name}" for name in indicators)]
    eligible = frame.dropna(subset=required).copy()
    rows: list[dict[str, object]] = []

    for currency, group in eligible.groupby("currency"):
        target = group[target_col].astype(bool)
        random_hit_rate = float(target.mean())
        for indicator in indicators:
            signal = group[f"signal_{indicator}"].astype(bool)
            signal_count = int(signal.sum())
            signal_hit_rate = float(target[signal].mean()) if signal_count else np.nan
            rows.append(
                {
                    "horizon": horizon,
                    "corridor": f"RUB->{currency}",
                    "indicator": indicator,
                    "eligible_days": len(group),
                    "signal_count": signal_count,
                    "target_positives": int(target.sum()),
                    "signal_hit_rate": signal_hit_rate,
                    "random_hit_rate": random_hit_rate,
                    "lift": signal_hit_rate / random_hit_rate if signal_count else np.nan,
                    "signals_per_week": signal_count
                    / max(
                        (group["effective_date"].max() - group["effective_date"].min()).days
                        / 7,
                        1,
                    ),
                }
            )
    return pd.DataFrame(rows)


def signals_per_week(dates: pd.Series, signal: pd.Series) -> float:
    count = int(signal.sum())
    if count == 0 or dates.empty:
        return 0.0
    span_weeks = max((dates.max() - dates.min()).days / 7, 1)
    return count / span_weeks


def cluster_share(signal: pd.Series) -> float:
    count = int(signal.sum())
    if count == 0:
        return float("nan")
    continued = signal.astype(bool) & signal.shift(1).eq(True)
    return float(continued.sum() / count)


def frequency_matched_hit_rate(
    target: pd.Series,
    signal: pd.Series,
    *,
    draws: int = 50,
    rng: np.random.Generator | None = None,
) -> float:
    n_signal = int(signal.sum())
    if n_signal == 0 or n_signal > len(target):
        return float("nan")
    rng = rng or np.random.default_rng(0)
    values = target.to_numpy()
    draws = min(draws, 50)
    hits = [
        float(values[rng.choice(len(values), size=n_signal, replace=False)].mean())
        for _ in range(draws)
    ]
    return float(np.mean(hits))


def forward_bps(price: pd.Series, forward_mean: pd.Series) -> pd.Series:
    return (forward_mean - price) / price * 1e4


def evaluate_method(
    frame: pd.DataFrame,
    *,
    horizon: int,
    method: str,
    split: str,
    target_col: str,
    signal_col: str,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    required = [target_col, signal_col, "rub_per_unit", f"forward_mean_h{horizon}"]
    eligible = frame.dropna(subset=required).copy()
    rows: list[dict] = []
    rng = rng or np.random.default_rng(0)
    for currency, group in eligible.groupby("currency", sort=False):
        target = group[target_col].astype(bool)
        signal = group[signal_col].fillna(False).astype(bool)
        signal_count = int(signal.sum())
        random_hit_rate = float(target.mean())
        signal_hit_rate = float(target[signal].mean()) if signal_count else float("nan")
        matched = frequency_matched_hit_rate(target, signal, rng=rng)
        bps = forward_bps(group["rub_per_unit"], group[f"forward_mean_h{horizon}"])
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
                "lift": signal_hit_rate / random_hit_rate
                if signal_count and random_hit_rate
                else float("nan"),
                "lift_freq_matched": signal_hit_rate / matched
                if signal_count and matched
                else float("nan"),
                "bps_forward": float(bps[signal].mean()) if signal_count else float("nan"),
                "signals_per_week": signals_per_week(group["effective_date"], signal),
                "cluster_share": cluster_share(signal),
            }
        )
    return pd.DataFrame(rows)
