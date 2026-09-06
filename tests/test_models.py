import numpy as np
import pandas as pd

from fx_signal.models import RankingMetric, select_threshold


def _weekly_frame(
    *,
    n_weeks: int,
    target: np.ndarray,
    proba: np.ndarray,
    days_per_week: int = 1,
    prices: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    mondays = pd.date_range("2024-01-01", periods=n_weeks, freq="W-MON")
    dates: list[pd.Timestamp] = []
    for monday in mondays:
        dates.extend(monday + pd.to_timedelta(range(days_per_week), unit="D"))
    n = len(dates)
    if len(target) != n or len(proba) != n:
        raise ValueError("target and proba must match the constructed calendar")
    price = np.ones(n) if prices is None else prices
    if len(price) != n:
        raise ValueError("prices must match the constructed calendar")
    forward_mean = np.roll(price, -5) if prices is not None else price * 1.01
    frame = pd.DataFrame(
        {
            "currency": "KZT",
            "effective_date": dates,
            "has_fact": True,
            "stay_not_worse": target,
            "rub_per_unit": price,
            "forward_mean_h5": forward_mean,
        }
    )
    return frame, proba


def _pick(
    frame: pd.DataFrame,
    proba: np.ndarray,
    grid: list[float],
    *,
    max_spw: float,
    ranking_metric: RankingMetric = "lift",
) -> float:
    return select_threshold(
        frame,
        proba,
        target_col="stay_not_worse",
        horizon=5,
        grid=grid,
        quantiles=[],
        min_signals_per_week=0.8,
        max_signals_per_week=max_spw,
        ranking_metric=ranking_metric,
    )


def test_rare_high_lift_loses_to_saturated_frequency() -> None:
    n_weeks = 10
    target = np.zeros(n_weeks, dtype=bool)
    target[:3] = True
    proba = np.full(n_weeks, 0.55)
    proba[0] = 0.95
    frame, proba = _weekly_frame(n_weeks=n_weeks, target=target, proba=proba)
    # 0.9: 1 hit / 10 weeks, lift ≈ 3.3, score ≈ 0.33
    # 0.5: 10 signals, lift = 1, score = 1
    assert _pick(frame, proba, [0.9, 0.5], max_spw=2.5) == 0.5


def test_frequency_below_one_is_not_saturated() -> None:
    n_weeks = 10
    target = np.zeros(n_weeks, dtype=bool)
    target[:9] = True
    proba = np.full(n_weeks, 0.35)
    proba[:8] = 0.75
    frame, proba = _weekly_frame(n_weeks=n_weeks, target=target, proba=proba)
    # 0.75: spw=0.8, lift > 1, score = lift * 0.8
    # 0.35: spw=1.0, lower lift, score = lift
    # If saturation were 0.8, the rarer threshold would win.
    assert _pick(frame, proba, [0.75, 0.35], max_spw=2.5) == 0.35


def test_threshold_above_max_frequency_is_dropped() -> None:
    n_weeks = 10
    days = 5
    n = n_weeks * days
    target = np.ones(n, dtype=bool)
    proba = np.full(n, 0.2)
    for week in range(n_weeks):
        proba[week * days] = 0.9
    frame, proba = _weekly_frame(
        n_weeks=n_weeks,
        days_per_week=days,
        target=target,
        proba=proba,
    )
    assert _pick(frame, proba, [0.9, 0.2], max_spw=2.5) == 0.9


def test_lar_threshold_prefers_cheaper_forward_path() -> None:
    n_weeks = 20
    target = np.ones(n_weeks, dtype=bool)
    proba = np.full(n_weeks, 0.4)
    proba[:10] = 0.9
    prices = np.concatenate([np.full(10, 1.2), np.full(10, 0.8)])
    frame, proba = _weekly_frame(n_weeks=n_weeks, target=target, proba=proba, prices=prices)
    # Lift is 1 at both thresholds; LAR should prefer the cheaper later path (0.4).
    assert _pick(frame, proba, [0.9, 0.4], max_spw=2.5, ranking_metric="lar") == 0.4
