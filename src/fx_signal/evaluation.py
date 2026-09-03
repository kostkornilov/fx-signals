from __future__ import annotations

import numpy as np
import pandas as pd

INDICATORS = ("momentum", "reversal", "seasonality", "level")


def evaluate_lift(
    frame: pd.DataFrame, horizon: int = 3, indicators: tuple[str, ...] = INDICATORS
) -> pd.DataFrame:
    target_col = f"target_local_min_h{horizon}"
    required = [target_col, *(f"signal_{name}" for name in indicators)]
    eligible = frame.dropna(subset=required).copy()
    rows: list[dict] = []

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
                    / max((group["effective_date"].max() - group["effective_date"].min()).days / 7, 1),
                }
            )
    return pd.DataFrame(rows)
