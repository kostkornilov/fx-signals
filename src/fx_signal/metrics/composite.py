from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from fx_signal.metrics.customer_value_and_regret import empirical_cvar

DEFAULT_RHO = 2.0
DEFAULT_MATERIALITY_BPS = 100.0
DEFAULT_SCALE_BPS = DEFAULT_MATERIALITY_BPS / math.log(1.3)


def lift_at_risk(
    *,
    lift: float,
    signal_value_bps: float,
    random_value_bps: float,
    signal_tail_regret_bps: float,
    random_tail_regret_bps: float,
    rho: float = DEFAULT_RHO,
    scale_bps: float = DEFAULT_SCALE_BPS,
) -> float:
    """Combine lift with incremental mean value and tail regret.

    The matched-random adjustment makes 1 the neutral score. With the default
    scale, a 100 bp utility improvement multiplies the score by 1.3.
    """
    values = np.asarray(
        [
            lift,
            signal_value_bps,
            random_value_bps,
            signal_tail_regret_bps,
            random_tail_regret_bps,
            rho,
            scale_bps,
        ],
        dtype=float,
    )
    if np.isnan(values).any():
        return float("nan")
    if not np.isfinite(values).all():
        raise ValueError("Lift-at-Risk inputs must be finite")
    if lift < 0:
        raise ValueError("lift must be non-negative")
    if signal_tail_regret_bps < 0 or random_tail_regret_bps < 0:
        raise ValueError("tail regret must be non-negative")
    if rho < 0:
        raise ValueError("rho must be non-negative")
    if scale_bps <= 0:
        raise ValueError("scale_bps must be positive")
    if lift == 0:
        return 0.0

    delta_utility = (signal_value_bps - random_value_bps) - rho * (
        signal_tail_regret_bps - random_tail_regret_bps
    )
    exponent = delta_utility / scale_bps
    if exponent > math.log(np.finfo(float).max):
        return float("inf")
    if exponent < math.log(np.nextafter(0.0, 1.0)):
        return 0.0
    return float(lift * math.exp(exponent))


def lar_path_outcomes(
    frame: pd.DataFrame,
    *,
    horizon: int,
    mean_gap_days: float | None = None,
    price_col: str = "rub_per_unit",
    currency_col: str = "currency",
    date_col: str = "effective_date",
) -> pd.DataFrame:
    """Return per-moment future-path value and regret for LAR.

    ``mean_gap_days=None`` gives the fixed-window variant. A positive value
    enables exponential calendar-time discounting. Only rows with all next
    ``horizon`` observations available within their currency receive outcomes.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if mean_gap_days is not None and (
        not math.isfinite(mean_gap_days) or mean_gap_days <= 0
    ):
        raise ValueError("mean_gap_days must be a positive finite number")
    required = {price_col, currency_col, date_col}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if not frame.index.is_unique:
        raise ValueError("frame index must be unique")

    work = frame[[currency_col, date_col, price_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    if work[date_col].isna().any():
        raise ValueError(f"{date_col} must not contain missing or invalid dates")
    if work[currency_col].isna().any():
        raise ValueError(f"{currency_col} must not contain missing values")
    price = pd.to_numeric(work[price_col], errors="coerce")
    invalid_price = work[price_col].notna() & price.isna()
    if invalid_price.any():
        raise ValueError(f"{price_col} contains non-numeric values")
    if price.dropna().le(0).any():
        raise ValueError(f"{price_col} must contain only positive prices")
    work[price_col] = price
    if work.duplicated([currency_col, date_col]).any():
        raise ValueError(f"frame must have at most one row per {currency_col} and {date_col}")

    order_col = "__lar_original_order__"
    work[order_col] = np.arange(len(work))
    work = work.sort_values([currency_col, date_col, order_col], kind="stable")
    value = pd.Series(np.nan, index=work.index, dtype=float)
    regret = pd.Series(np.nan, index=work.index, dtype=float)

    for _, group in work.groupby(currency_col, sort=False):
        current_price = group[price_col]
        current_date = group[date_col]
        future_prices = pd.concat(
            [current_price.shift(-step) for step in range(1, horizon + 1)],
            axis=1,
        )
        future_dates = pd.concat(
            [current_date.shift(-step) for step in range(1, horizon + 1)],
            axis=1,
        )
        future_prices.columns = range(1, horizon + 1)
        future_dates.columns = range(1, horizon + 1)
        complete = (
            current_price.notna()
            & future_prices.notna().all(axis=1)
            & future_dates.notna().all(axis=1)
        )

        advantage = future_prices.div(current_price, axis=0).sub(1.0).mul(10_000.0)
        loss = current_price.to_frame().to_numpy() / future_prices - 1.0
        loss = loss.clip(lower=0.0).mul(10_000.0)

        if mean_gap_days is None:
            relevance = pd.DataFrame(1.0, index=group.index, columns=future_prices.columns)
        else:
            elapsed_days = future_dates.sub(current_date, axis=0).apply(
                lambda column: column.dt.total_seconds() / 86_400.0
            )
            relative_days = elapsed_days.sub(elapsed_days.iloc[:, 0], axis=0)
            relevance = np.exp(-relative_days / mean_gap_days)

        weights = relevance.div(relevance.sum(axis=1), axis=0)
        group_value = advantage.mul(weights).sum(axis=1).where(complete)
        group_regret = loss.mul(relevance).max(axis=1).where(complete)
        value.loc[group.index] = group_value
        regret.loc[group.index] = group_regret

    result = pd.DataFrame(
        {
            "path_value_bps": value,
            "path_regret_bps": regret,
        }
    )
    return result.reindex(frame.index)


def fixed_window_lift_at_risk(
    frame: pd.DataFrame,
    *,
    horizon: int,
    target_col: str,
    signal_col: str,
    baseline_signals: Iterable[Iterable[bool] | pd.Series] | np.ndarray | None = None,
    baseline_draws: int = 200,
    rho: float = DEFAULT_RHO,
    scale_bps: float = DEFAULT_SCALE_BPS,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
    price_col: str = "rub_per_unit",
    currency_col: str = "currency",
    date_col: str = "effective_date",
) -> dict[str, float | int | str]:
    """Calculate fixed-window Lift-at-Risk for one corridor/OOT slice.

    By default, random streams reproduce the evaluated signal count in every
    calendar week. Pass ``baseline_signals`` produced by a policy simulator when
    cooldown and other eligibility rules must also be reproduced exactly.
    """
    return _evaluate_lift_at_risk(
        frame,
        horizon=horizon,
        target_col=target_col,
        signal_col=signal_col,
        mean_gap_days=None,
        baseline_signals=baseline_signals,
        baseline_draws=baseline_draws,
        rho=rho,
        scale_bps=scale_bps,
        confidence=confidence,
        rng=rng,
        price_col=price_col,
        currency_col=currency_col,
        date_col=date_col,
    )


def discounted_lift_at_risk(
    frame: pd.DataFrame,
    *,
    horizon: int,
    target_col: str,
    signal_col: str,
    mean_gap_days: float,
    baseline_signals: Iterable[Iterable[bool] | pd.Series] | np.ndarray | None = None,
    baseline_draws: int = 200,
    rho: float = DEFAULT_RHO,
    scale_bps: float = DEFAULT_SCALE_BPS,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
    price_col: str = "rub_per_unit",
    currency_col: str = "currency",
    date_col: str = "effective_date",
) -> dict[str, float | int | str]:
    """Calculate time-discounted Lift-at-Risk for one corridor/OOT slice.

    By default, random streams reproduce the evaluated signal count in every
    calendar week. Pass ``baseline_signals`` produced by a policy simulator when
    cooldown and other eligibility rules must also be reproduced exactly.
    """
    return _evaluate_lift_at_risk(
        frame,
        horizon=horizon,
        target_col=target_col,
        signal_col=signal_col,
        mean_gap_days=mean_gap_days,
        baseline_signals=baseline_signals,
        baseline_draws=baseline_draws,
        rho=rho,
        scale_bps=scale_bps,
        confidence=confidence,
        rng=rng,
        price_col=price_col,
        currency_col=currency_col,
        date_col=date_col,
    )


def _evaluate_lift_at_risk(
    frame: pd.DataFrame,
    *,
    horizon: int,
    target_col: str,
    signal_col: str,
    mean_gap_days: float | None,
    baseline_signals: Iterable[Iterable[bool] | pd.Series] | np.ndarray | None,
    baseline_draws: int,
    rho: float,
    scale_bps: float,
    confidence: float,
    rng: np.random.Generator | None,
    price_col: str,
    currency_col: str,
    date_col: str,
) -> dict[str, float | int | str]:
    required = {target_col, signal_col, currency_col}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if baseline_draws < 1:
        raise ValueError("baseline_draws must be at least 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    if not math.isfinite(rho) or rho < 0:
        raise ValueError("rho must be a non-negative finite number")
    if not math.isfinite(scale_bps) or scale_bps <= 0:
        raise ValueError("scale_bps must be a positive finite number")
    if frame[currency_col].nunique(dropna=False) != 1:
        raise ValueError("Lift-at-Risk must be evaluated separately for each currency")

    outcomes = lar_path_outcomes(
        frame,
        horizon=horizon,
        mean_gap_days=mean_gap_days,
        price_col=price_col,
        currency_col=currency_col,
        date_col=date_col,
    )
    target = frame[target_col]
    signal = frame[signal_col].fillna(False).astype(bool)
    eligible = target.notna() & outcomes.notna().all(axis=1)
    eligible_target = target.loc[eligible].astype(bool).to_numpy()
    eligible_signal = signal.loc[eligible].to_numpy()
    value = outcomes.loc[eligible, "path_value_bps"].to_numpy()
    regret = outcomes.loc[eligible, "path_regret_bps"].to_numpy()
    signal_count = int(eligible_signal.sum())
    excluded_signal_count = int((signal & ~eligible).sum())

    if signal_count == 0:
        return _empty_result(
            variant="discounted" if mean_gap_days is not None else "fixed",
            eligible_count=int(eligible.sum()),
            excluded_signal_count=excluded_signal_count,
            rho=rho,
            scale_bps=scale_bps,
            confidence=confidence,
            mean_gap_days=mean_gap_days,
        )

    signal_hit_rate = float(eligible_target[eligible_signal].mean())
    signal_value = float(value[eligible_signal].mean())
    signal_regret = empirical_cvar(regret[eligible_signal], confidence=confidence)

    masks = _baseline_masks(
        baseline_signals,
        eligible=eligible.to_numpy(),
        eligible_signal=eligible_signal,
        eligible_dates=frame.loc[eligible, date_col],
        signal_count=signal_count,
        baseline_draws=baseline_draws,
        rng=rng,
    )
    random_hit_rates: list[float] = []
    random_values: list[float] = []
    random_regrets: list[float] = []
    for baseline in masks:
        random_hit_rates.append(float(eligible_target[baseline].mean()))
        random_values.append(float(value[baseline].mean()))
        random_regrets.append(empirical_cvar(regret[baseline], confidence=confidence))

    random_hit_rate = float(np.mean(random_hit_rates))
    random_value = float(np.mean(random_values))
    random_regret = float(np.mean(random_regrets))
    lift = signal_hit_rate / random_hit_rate if random_hit_rate > 0 else float("nan")
    delta_utility = (signal_value - random_value) - rho * (signal_regret - random_regret)
    score = lift_at_risk(
        lift=lift,
        signal_value_bps=signal_value,
        random_value_bps=random_value,
        signal_tail_regret_bps=signal_regret,
        random_tail_regret_bps=random_regret,
        rho=rho,
        scale_bps=scale_bps,
    )
    return {
        "variant": "discounted" if mean_gap_days is not None else "fixed",
        "eligible_count": int(eligible.sum()),
        "signal_count": signal_count,
        "excluded_signal_count": excluded_signal_count,
        "baseline_draws": len(masks),
        "baseline_kind": (
            "weekly_count_matched" if baseline_signals is None else "provided"
        ),
        "signal_hit_rate": signal_hit_rate,
        "random_hit_rate": random_hit_rate,
        "lift": lift,
        "signal_value_bps": signal_value,
        "random_value_bps": random_value,
        "signal_tail_regret_bps": signal_regret,
        "random_tail_regret_bps": random_regret,
        "delta_utility_bps": delta_utility,
        "rho": float(rho),
        "scale_bps": float(scale_bps),
        "regret_confidence": float(confidence),
        "mean_gap_days": float(mean_gap_days) if mean_gap_days is not None else float("nan"),
        "lift_at_risk": score,
    }


def _baseline_masks(
    baseline_signals: Iterable[Iterable[bool] | pd.Series] | np.ndarray | None,
    *,
    eligible: np.ndarray,
    eligible_signal: np.ndarray,
    eligible_dates: pd.Series,
    signal_count: int,
    baseline_draws: int,
    rng: np.random.Generator | None,
) -> list[np.ndarray]:
    if baseline_signals is None:
        generator = rng or np.random.default_rng(0)
        weeks = pd.to_datetime(eligible_dates).dt.to_period("W-SUN").to_numpy()
        masks: list[np.ndarray] = []
        for _ in range(baseline_draws):
            mask = np.zeros(len(eligible_signal), dtype=bool)
            for week in pd.unique(weeks):
                positions = np.flatnonzero(weeks == week)
                count = int(eligible_signal[positions].sum())
                if count:
                    selected = generator.choice(positions, size=count, replace=False)
                    mask[selected] = True
            masks.append(mask)
        return masks

    if isinstance(baseline_signals, np.ndarray):
        raw = baseline_signals
        if raw.ndim == 1:
            candidates = [raw]
        elif raw.ndim == 2:
            candidates = list(raw)
        else:
            raise ValueError("baseline_signals array must be one- or two-dimensional")
    else:
        candidates = list(baseline_signals)
        if candidates and all(isinstance(item, (bool, np.bool_)) for item in candidates):
            candidates = [candidates]
    if not candidates:
        raise ValueError("baseline_signals must contain at least one random stream")

    masks = []
    for candidate in candidates:
        series = candidate if isinstance(candidate, pd.Series) else pd.Series(candidate)
        if len(series) != len(eligible):
            raise ValueError("every baseline signal must have the same length as frame")
        full_mask = series.fillna(False).astype(bool).to_numpy()
        if (full_mask & ~eligible).any():
            raise ValueError("baseline signals must select only eligible rows")
        eligible_mask = full_mask[eligible]
        if int(eligible_mask.sum()) != signal_count:
            raise ValueError(
                "every baseline signal must match the evaluated signal count on eligible rows"
            )
        masks.append(eligible_mask)
    return masks


def _empty_result(
    *,
    variant: str,
    eligible_count: int,
    excluded_signal_count: int,
    rho: float,
    scale_bps: float,
    confidence: float,
    mean_gap_days: float | None,
) -> dict[str, float | int | str]:
    return {
        "variant": variant,
        "eligible_count": eligible_count,
        "signal_count": 0,
        "excluded_signal_count": excluded_signal_count,
        "baseline_draws": 0,
        "baseline_kind": "none",
        "signal_hit_rate": float("nan"),
        "random_hit_rate": float("nan"),
        "lift": float("nan"),
        "signal_value_bps": float("nan"),
        "random_value_bps": float("nan"),
        "signal_tail_regret_bps": float("nan"),
        "random_tail_regret_bps": float("nan"),
        "delta_utility_bps": float("nan"),
        "rho": float(rho),
        "scale_bps": float(scale_bps),
        "regret_confidence": float(confidence),
        "mean_gap_days": float(mean_gap_days) if mean_gap_days is not None else float("nan"),
        "lift_at_risk": float("nan"),
    }
