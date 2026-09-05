from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd


def _numeric_series(values: pd.Series | Iterable[float], *, name: str) -> pd.Series:
    """Return numeric values without silently accepting non-numeric input."""
    series = values.copy() if isinstance(values, pd.Series) else pd.Series(values)
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = series.notna() & numeric.isna()
    if invalid.any():
        raise ValueError(f"{name} contains non-numeric values")
    return numeric.astype(float)


def _validate_prices(price: pd.Series, *, name: str) -> None:
    if price.dropna().le(0).any():
        raise ValueError(f"{name} must contain only positive prices")


def moment_advantage_bps(
    price: pd.Series | Iterable[float],
    reference_price: pd.Series | Iterable[float],
) -> pd.Series:
    """Return how much better `price` is than a reference price, in basis points.

    Prices are RUB per unit of recipient currency, so lower is better. Positive
    values therefore mean the selected price is better than the reference.
    """
    current = _numeric_series(price, name="price")
    reference = _numeric_series(reference_price, name="reference_price")
    if len(current) != len(reference):
        raise ValueError("price and reference_price must have the same length")
    reference.index = current.index
    _validate_prices(current, name="price")
    _validate_prices(reference, name="reference_price")
    return (reference / current - 1.0) * 10_000.0


def forward_bps(
    price: pd.Series | Iterable[float],
    forward_mean: pd.Series | Iterable[float],
) -> pd.Series:
    """Backward-compatible name for advantage versus a future mean price."""
    return moment_advantage_bps(price, forward_mean)


def customer_regret_bps(
    price: pd.Series | Iterable[float],
    best_later_price: pd.Series | Iterable[float],
) -> pd.Series:
    """Return non-negative loss versus the best later price, in basis points."""
    current = _numeric_series(price, name="price")
    later = _numeric_series(best_later_price, name="best_later_price")
    if len(current) != len(later):
        raise ValueError("price and best_later_price must have the same length")
    later.index = current.index
    _validate_prices(current, name="price")
    _validate_prices(later, name="best_later_price")
    return ((current / later - 1.0) * 10_000.0).clip(lower=0.0)


def empirical_cvar(
    losses: pd.Series | Iterable[float], *, confidence: float = 0.95
) -> float:
    """Average the worst `(1 - confidence)` share of observed losses.

    This discrete definition always includes at least one observation. It is a
    descriptive tail statistic; callers should avoid strong claims when the
    sample contains fewer than 100 alerts at 95% confidence.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    numeric = _numeric_series(losses, name="losses").dropna().to_numpy()
    if len(numeric) == 0:
        return float("nan")
    if np.any(numeric < 0):
        raise ValueError("losses must be non-negative")
    tail_size = max(1, math.ceil((1.0 - confidence) * len(numeric) - 1e-12))
    return float(np.sort(numeric)[-tail_size:].mean())


def add_customer_outcomes(
    frame: pd.DataFrame,
    *,
    horizon: int,
    price_col: str = "rub_per_unit",
    currency_col: str = "currency",
    date_col: str = "effective_date",
) -> pd.DataFrame:
    """Add surrounding-moment advantage and forward regret per currency.

    Windows are based on observations within each currency. The surrounding
    mean uses the complete `[t-h, t+h]` window. The best-later price uses the
    complete next `h` observations. Rows without a complete window get NaN.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    required = {price_col, currency_col, date_col}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    result = frame.copy()
    order_col = "__metrics_original_order__"
    if order_col in result.columns:
        raise ValueError(f"Reserved column already exists: {order_col}")
    result[order_col] = np.arange(len(result))
    result = result.sort_values([currency_col, date_col, order_col], kind="stable")
    price = _numeric_series(result[price_col], name=price_col)
    _validate_prices(price, name=price_col)
    result[price_col] = price

    surrounding = result.groupby(currency_col, sort=False)[price_col].transform(
        lambda values: values.rolling(
            2 * horizon + 1,
            center=True,
            min_periods=2 * horizon + 1,
        ).mean()
    )

    def future_min(values: pd.Series) -> pd.Series:
        future = pd.concat(
            [values.shift(-step) for step in range(1, horizon + 1)],
            axis=1,
        )
        complete = future.notna().all(axis=1)
        return future.min(axis=1).where(complete)

    best_later = result.groupby(currency_col, sort=False)[price_col].transform(future_min)
    result[f"surrounding_mean_h{horizon}"] = surrounding
    result[f"future_best_h{horizon}"] = best_later
    result[f"moment_advantage_bps_h{horizon}"] = moment_advantage_bps(price, surrounding)
    result[f"customer_regret_bps_h{horizon}"] = customer_regret_bps(price, best_later)

    return result.sort_values(order_col, kind="stable").drop(columns=order_col)


def evaluate_customer_outcomes(
    frame: pd.DataFrame,
    *,
    signal_col: str,
    horizon: int,
    price_col: str = "rub_per_unit",
    currency_col: str = "currency",
    date_col: str = "effective_date",
) -> dict[str, float | int]:
    """Summarize customer value and regret for sent signals."""
    if signal_col not in frame:
        raise KeyError(f"Missing required column: {signal_col}")
    value_col = f"moment_advantage_bps_h{horizon}"
    regret_col = f"customer_regret_bps_h{horizon}"
    if value_col in frame and regret_col in frame:
        outcomes = frame
    else:
        outcomes = add_customer_outcomes(
            frame,
            horizon=horizon,
            price_col=price_col,
            currency_col=currency_col,
            date_col=date_col,
        )
    signal = outcomes[signal_col].fillna(False).astype(bool)
    value = outcomes.loc[signal, value_col].dropna()
    regret = outcomes.loc[signal, regret_col].dropna()
    worst_five = regret.nlargest(min(5, len(regret)))
    return {
        "value_observations": len(value),
        "moment_advantage_bps": float(value.mean()) if len(value) else float("nan"),
        "regret_observations": len(regret),
        "customer_regret_bps_mean": float(regret.mean()) if len(regret) else float("nan"),
        "customer_regret_bps_max": float(regret.max()) if len(regret) else float("nan"),
        "customer_regret_cvar_95_bps": empirical_cvar(regret, confidence=0.95),
        "customer_regret_worst_five_mean_bps": (
            float(worst_five.mean()) if len(worst_five) else float("nan")
        ),
    }
