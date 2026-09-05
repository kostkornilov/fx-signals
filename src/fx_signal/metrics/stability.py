from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def summarize_stability(
    metrics: pd.DataFrame,
    metric_columns: Iterable[str],
    *,
    lower_is_better: Iterable[str] = (),
) -> pd.DataFrame:
    """Summarize metric variation across already-computed OOT slices.

    `metrics` should have one row per corridor/fold/regime slice. The function
    deliberately does not pool the underlying observations, because pooling can
    hide a weak slice.
    """
    columns = list(metric_columns)
    missing = set(columns).difference(metrics.columns)
    if missing:
        raise KeyError(f"Missing metric columns: {sorted(missing)}")
    lower = set(lower_is_better)
    unknown = lower.difference(columns)
    if unknown:
        raise ValueError(f"lower_is_better contains unknown metrics: {sorted(unknown)}")

    rows: list[dict[str, float | int | str]] = []
    for column in columns:
        values = pd.to_numeric(metrics[column], errors="coerce").dropna()
        if values.empty:
            rows.append(
                {
                    "metric": column,
                    "slice_count": 0,
                    "worst": float("nan"),
                    "median": float("nan"),
                    "best": float("nan"),
                }
            )
            continue
        low, high = float(values.min()), float(values.max())
        rows.append(
            {
                "metric": column,
                "slice_count": len(values),
                "worst": high if column in lower else low,
                "median": float(values.median()),
                "best": low if column in lower else high,
            }
        )
    return pd.DataFrame(rows)
