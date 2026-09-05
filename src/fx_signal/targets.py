from __future__ import annotations

import pandas as pd


def add_local_min_target(
    rates: pd.DataFrame, horizon: int = 3, price_col: str = "rub_per_unit"
) -> pd.DataFrame:
    """Label minima in a symmetric +/- horizon window, separately by currency."""
    result = rates.copy()

    window_min = result.groupby("currency")[price_col].transform(
        lambda price: price.rolling(
            2 * horizon + 1, center=True, min_periods=2 * horizon + 1
        ).min()
    )
    result[f"target_local_min_h{horizon}"] = (
        result[price_col].eq(window_min).astype("boolean").where(window_min.notna())
    )
    return result


def _future_frame(price: pd.Series, horizon: int) -> pd.DataFrame:
    return pd.concat(
        {f"lead_{step}": price.shift(-step) for step in range(1, horizon + 1)},
        axis=1,
    )


def add_stay_not_worse_target(
    rates: pd.DataFrame, horizon: int = 3, price_col: str = "rub_per_unit"
) -> pd.DataFrame:
    """Label days where the rate does not get cheaper over the next `horizon` prints.

    Lower `rub_per_unit` is better for the sender, so the day stays not-worse when
    the minimum of the next `horizon` prices is at least today's price.
    """
    result = rates.copy()
    labels: list[pd.Series] = []
    means: list[pd.Series] = []
    for _, group in result.groupby("currency", sort=False):
        price = group[price_col]
        future = _future_frame(price, horizon)
        complete = future.notna().all(axis=1)
        stay = future.min(axis=1).ge(price).astype("boolean").where(complete)
        labels.append(stay)
        means.append(future.mean(axis=1).where(complete))
    result[f"target_stay_not_worse_h{horizon}"] = pd.concat(labels).sort_index()
    result[f"forward_mean_h{horizon}"] = pd.concat(means).sort_index()
    return result


def add_window_closing_target(
    rates: pd.DataFrame, horizon: int = 3, price_col: str = "rub_per_unit"
) -> pd.DataFrame:
    """Label days where the rate is strictly higher `horizon` prints later."""
    result = rates.copy()
    future = result.groupby("currency")[price_col].shift(-horizon)
    result[f"target_window_closing_h{horizon}"] = (
        future.gt(result[price_col]).astype("boolean").where(future.notna())
    )
    return result


def add_targets(
    rates: pd.DataFrame, horizon: int = 3, price_col: str = "rub_per_unit"
) -> pd.DataFrame:
    result = add_local_min_target(rates, horizon=horizon, price_col=price_col)
    result = add_stay_not_worse_target(result, horizon=horizon, price_col=price_col)
    return add_window_closing_target(result, horizon=horizon, price_col=price_col)


def target_column(kind: str, horizon: int) -> str:
    names = {
        "local_min": f"target_local_min_h{horizon}",
        "stay_not_worse": f"target_stay_not_worse_h{horizon}",
        "window_closing": f"target_window_closing_h{horizon}",
    }
    if kind not in names:
        raise ValueError(f"Unknown target kind: {kind}")
    return names[kind]
