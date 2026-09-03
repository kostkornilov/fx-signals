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
