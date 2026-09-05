from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from pandas.api.types import is_scalar

from fx_signal.indicators import RESEARCH_SIGNAL_COLUMNS

FAVOURABLE_SIGNAL_PRIORITY = (
    "signal_better_than_30_day_average",
    "signal_better_than_one_year_ago",
    "signal_better_range_held",
    "signal_larger_than_usual_latest_improvement",
    "signal_most_recent_changes_favourable",
)

DETERIORATION_SIGNAL_PRIORITY = (
    "signal_most_recent_changes_unfavourable",
    "signal_less_than_one_year_ago",
)

FORECAST_INTENT_BY_KIND = {
    "favourable": "favourable",
    "stay_not_worse": "favourable",
    "local_min": "favourable",
    "deterioration": "deterioration",
    "window_closing": "deterioration",
}

PRIORITY_BY_INTENT = {
    "favourable": FAVOURABLE_SIGNAL_PRIORITY,
    "deterioration": DETERIORATION_SIGNAL_PRIORITY,
}

DecisionReason = Literal[
    "selected_single_signal",
    "selected_by_clarity",
    "stale_market_data",
    "invalid_market_score",
    "below_market_threshold",
    "no_active_signal",
    "no_eligible_signal",
]


@dataclass(frozen=True, slots=True)
class MetaDecision:
    """Auditable result of one market-only push decision."""

    should_send: bool
    selected_signal: str | None
    reason: DecisionReason
    market_score: float | None
    threshold: float
    forecast_intent: str
    active_signals: tuple[str, ...]
    eligible_signals: tuple[str, ...]


def _forecast_intent(forecast_kind: str) -> str:
    try:
        return FORECAST_INTENT_BY_KIND[forecast_kind]
    except KeyError as error:
        supported = ", ".join(sorted(FORECAST_INTENT_BY_KIND))
        raise ValueError(
            f"Unknown forecast_kind {forecast_kind!r}; expected one of: {supported}"
        ) from error


def _finite_threshold(value: object) -> float:
    if isinstance(value, bool) or not is_scalar(value):
        raise ValueError("threshold must be a finite number")
    try:
        threshold = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("threshold must be a finite number") from error
    if not math.isfinite(threshold):
        raise ValueError("threshold must be a finite number")
    return threshold


def _market_score(value: object) -> float | None:
    """Convert a model score, failing closed instead of sending on invalid data."""
    if isinstance(value, bool) or not is_scalar(value) or pd.isna(value):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _active_flag(value: object) -> bool:
    if not is_scalar(value):
        raise TypeError("signal flags must be scalar values")
    if pd.isna(value):
        return False
    return bool(value)


def _allowed_signals(values: Collection[str] | None) -> frozenset[str]:
    if values is None:
        return frozenset(RESEARCH_SIGNAL_COLUMNS)
    allowed = frozenset(values)
    unknown = sorted(allowed.difference(RESEARCH_SIGNAL_COLUMNS))
    if unknown:
        raise ValueError(f"Unknown signal columns: {unknown}")
    return allowed


def decide_push(
    *,
    market_score: object,
    threshold: object,
    signal_values: Mapping[str, object],
    forecast_kind: str = "stay_not_worse",
    data_is_fresh: object = True,
    allowed_signals: Collection[str] | None = None,
) -> MetaDecision:
    """Apply the market gate and choose one clear, forecast-compatible signal.

    The function deliberately does not estimate transfer probability, conversion or profit. A
    ``True`` signal flag is treated as the upstream indicator's assertion that its factual claim is
    active at the decision time.
    """
    intent = _forecast_intent(forecast_kind)
    parsed_threshold = _finite_threshold(threshold)
    parsed_score = _market_score(market_score)
    allowed = _allowed_signals(allowed_signals)

    active = tuple(
        signal
        for signal in RESEARCH_SIGNAL_COLUMNS
        if _active_flag(signal_values.get(signal, False))
    )
    eligible = tuple(
        signal for signal in PRIORITY_BY_INTENT[intent] if signal in allowed and signal in active
    )

    common = {
        "market_score": parsed_score,
        "threshold": parsed_threshold,
        "forecast_intent": intent,
        "active_signals": active,
        "eligible_signals": eligible,
    }

    if not _active_flag(data_is_fresh):
        return MetaDecision(False, None, "stale_market_data", **common)
    if parsed_score is None:
        return MetaDecision(False, None, "invalid_market_score", **common)
    if parsed_score < parsed_threshold:
        return MetaDecision(False, None, "below_market_threshold", **common)
    if not active:
        return MetaDecision(False, None, "no_active_signal", **common)
    if not eligible:
        return MetaDecision(False, None, "no_eligible_signal", **common)

    selected = eligible[0]
    reason: DecisionReason = (
        "selected_single_signal" if len(eligible) == 1 else "selected_by_clarity"
    )
    return MetaDecision(True, selected, reason, **common)


def add_meta_decisions(
    frame: pd.DataFrame,
    *,
    score_col: str = "market_score",
    threshold: float | str = 0.8,
    forecast_kind: str = "stay_not_worse",
    forecast_kind_col: str | None = None,
    freshness_col: str | None = None,
    allowed_signals: Collection[str] | None = None,
) -> pd.DataFrame:
    """Apply :func:`decide_push` to every row without changing other pipeline stages.

    ``threshold`` may be one fixed number or the name of a per-row threshold column. A forecast-kind
    column is useful when favourable and closing-window model targets share one frame.
    """
    allowed = _allowed_signals(allowed_signals)
    required = {score_col, *allowed}
    if isinstance(threshold, str):
        required.add(threshold)
    if forecast_kind_col is not None:
        required.add(forecast_kind_col)
    if freshness_col is not None:
        required.add(freshness_col)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    fixed_threshold = None if isinstance(threshold, str) else _finite_threshold(threshold)
    decisions: list[MetaDecision] = []
    for row in frame.to_dict(orient="records"):
        row_threshold = row[threshold] if isinstance(threshold, str) else fixed_threshold
        row_kind = row[forecast_kind_col] if forecast_kind_col is not None else forecast_kind
        row_freshness = row[freshness_col] if freshness_col is not None else True
        decisions.append(
            decide_push(
                market_score=row[score_col],
                threshold=row_threshold,
                signal_values=row,
                forecast_kind=str(row_kind),
                data_is_fresh=row_freshness,
                allowed_signals=allowed,
            )
        )

    result = frame.copy()
    result["meta_should_send"] = [decision.should_send for decision in decisions]
    result["meta_selected_signal"] = pd.array(
        [decision.selected_signal for decision in decisions], dtype="string"
    )
    result["meta_reason"] = pd.array([decision.reason for decision in decisions], dtype="string")
    result["meta_active_signals"] = [decision.active_signals for decision in decisions]
    result["meta_eligible_signals"] = [decision.eligible_signals for decision in decisions]
    return result
