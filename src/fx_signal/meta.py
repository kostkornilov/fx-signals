from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from pandas.api.types import is_scalar

from fx_signal.indicators import (
    ALL_SIGNAL_COLUMNS,
    SIGNAL_EFFECT_COLUMN,
)

SEASONALITY_SIGNAL = "signal_seasonality"

DEFAULT_CLARITY_COEFFICIENTS = {
    "signal_seasonality": 1.00,
    "signal_better_than_30_day_average": 1.00,
    "signal_better_than_one_year_ago": 0.95,
    "signal_less_than_one_year_ago": 0.95,
    "signal_better_range_held": 0.80,
    "signal_level": 0.75,
    "signal_larger_than_usual_latest_improvement": 0.70,
    "signal_reversal": 0.70,
    "signal_momentum": 0.65,
    "signal_most_recent_changes_favourable": 0.65,
    "signal_most_recent_changes_unfavourable": 0.65,
}

DecisionReason = Literal[
    "selected_single_signal",
    "selected_seasonality_priority",
    "selected_by_effect",
    "stale_market_data",
    "invalid_market_score",
    "below_market_threshold",
    "no_active_signal",
    "no_allowed_signal",
    "invalid_signal_effect",
]


@dataclass(frozen=True, slots=True)
class SignalScore:
    """Comparable score for one active indicator."""

    signal: str
    effect: float | None
    clarity: float
    score: float | None


@dataclass(frozen=True, slots=True)
class MetaDecision:
    """Auditable result of one market-only push decision."""

    should_send: bool
    selected_signal: str | None
    reason: DecisionReason
    market_score: float | None
    threshold: float
    selected_effect: float | None
    selection_score: float | None
    active_signals: tuple[str, ...]
    signal_scores: tuple[SignalScore, ...]


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


def _optional_finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not is_scalar(value) or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _active_flag(value: object) -> bool:
    if not is_scalar(value):
        raise TypeError("signal flags must be scalar values")
    if pd.isna(value):
        return False
    return bool(value)


def _allowed_signals(values: Collection[str] | None) -> frozenset[str]:
    if values is None:
        return frozenset(ALL_SIGNAL_COLUMNS)
    allowed = frozenset(values)
    unknown = sorted(allowed.difference(ALL_SIGNAL_COLUMNS))
    if unknown:
        raise ValueError(f"Unknown signal columns: {unknown}")
    return allowed


def _clarity_coefficients(values: Mapping[str, float] | None) -> dict[str, float]:
    coefficients = dict(DEFAULT_CLARITY_COEFFICIENTS)
    if values is None:
        return coefficients

    unknown = sorted(set(values).difference(ALL_SIGNAL_COLUMNS))
    if unknown:
        raise ValueError(f"Unknown clarity coefficient signals: {unknown}")
    for signal, raw_coefficient in values.items():
        coefficient = _optional_finite_number(raw_coefficient)
        if coefficient is None or not 0.0 < coefficient <= 1.0:
            raise ValueError(f"Clarity coefficient for {signal!r} must be in (0, 1]")
        coefficients[signal] = coefficient
    return coefficients


def _effect_for(
    signal: str,
    signal_values: Mapping[str, object],
    signal_effects: Mapping[str, object] | None,
) -> float | None:
    if signal_effects is not None:
        return _optional_finite_number(signal_effects.get(signal))
    return _optional_finite_number(signal_values.get(SIGNAL_EFFECT_COLUMN[signal]))


def decide_push(
    *,
    market_score: object,
    threshold: object,
    signal_values: Mapping[str, object],
    signal_effects: Mapping[str, object] | None = None,
    data_is_fresh: object = True,
    allowed_signals: Collection[str] | None = None,
    clarity_coefficients: Mapping[str, float] | None = None,
) -> MetaDecision:
    """Gate on one model score, then choose the strongest clear factual indicator.

    Effects are signed decimal changes in recipient currency: ``0.02`` means 2% more and ``-0.02``
    means 2% less. The sign is kept for message rendering; selection uses the absolute magnitude.
    This function does not estimate transfer probability, conversion or bank profit.
    """
    parsed_threshold = _finite_threshold(threshold)
    parsed_market_score = _optional_finite_number(market_score)
    allowed = _allowed_signals(allowed_signals)
    clarity = _clarity_coefficients(clarity_coefficients)
    active = tuple(
        signal for signal in ALL_SIGNAL_COLUMNS if _active_flag(signal_values.get(signal, False))
    )
    candidates = tuple(signal for signal in active if signal in allowed)

    def result(
        should_send: bool,
        selected_signal: str | None,
        reason: DecisionReason,
        *,
        selected_effect: float | None = None,
        selection_score: float | None = None,
        signal_scores: tuple[SignalScore, ...] = (),
    ) -> MetaDecision:
        return MetaDecision(
            should_send=should_send,
            selected_signal=selected_signal,
            reason=reason,
            market_score=parsed_market_score,
            threshold=parsed_threshold,
            selected_effect=selected_effect,
            selection_score=selection_score,
            active_signals=active,
            signal_scores=signal_scores,
        )

    if not _active_flag(data_is_fresh):
        return result(False, None, "stale_market_data")
    if parsed_market_score is None:
        return result(False, None, "invalid_market_score")
    if parsed_market_score < parsed_threshold:
        return result(False, None, "below_market_threshold")
    if not active:
        return result(False, None, "no_active_signal")
    if not candidates:
        return result(False, None, "no_allowed_signal")

    effects = {signal: _effect_for(signal, signal_values, signal_effects) for signal in candidates}
    scores = tuple(
        SignalScore(
            signal=signal,
            effect=effect,
            clarity=clarity[signal],
            score=abs(effect) * clarity[signal] if effect is not None else None,
        )
        for signal in candidates
        if (effect := effects[signal]) is not None or signal == SEASONALITY_SIGNAL
    )

    if len(candidates) == 1:
        selected = candidates[0]
        effect = effects[selected]
        selection_score = abs(effect) * clarity[selected] if effect is not None else None
        return result(
            True,
            selected,
            "selected_single_signal",
            selected_effect=effect,
            selection_score=selection_score,
            signal_scores=scores,
        )

    if SEASONALITY_SIGNAL in candidates:
        return result(
            True,
            SEASONALITY_SIGNAL,
            "selected_seasonality_priority",
            signal_scores=scores,
        )

    if len(scores) != len(candidates) or any(item.score is None for item in scores):
        return result(False, None, "invalid_signal_effect", signal_scores=scores)

    order = {signal: index for index, signal in enumerate(ALL_SIGNAL_COLUMNS)}
    winner = max(
        scores,
        key=lambda item: (
            item.score if item.score is not None else float("-inf"),
            item.clarity,
            abs(item.effect) if item.effect is not None else float("-inf"),
            -order[item.signal],
        ),
    )
    return result(
        True,
        winner.signal,
        "selected_by_effect",
        selected_effect=winner.effect,
        selection_score=winner.score,
        signal_scores=scores,
    )


def add_meta_decisions(
    frame: pd.DataFrame,
    *,
    score_col: str = "market_score",
    threshold: float | str = 0.8,
    freshness_col: str | None = None,
    allowed_signals: Collection[str] | None = None,
    clarity_coefficients: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Apply :func:`decide_push` to every row without wiring it into another pipeline stage.

    ``threshold`` may be one fixed number or the name of a per-row threshold column.
    """
    allowed = _allowed_signals(allowed_signals)
    required = {score_col, *ALL_SIGNAL_COLUMNS}
    required.update(SIGNAL_EFFECT_COLUMN[signal] for signal in allowed)
    if isinstance(threshold, str):
        required.add(threshold)
    if freshness_col is not None:
        required.add(freshness_col)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    fixed_threshold = None if isinstance(threshold, str) else _finite_threshold(threshold)
    decisions: list[MetaDecision] = []
    for row in frame.to_dict(orient="records"):
        row_threshold = row[threshold] if isinstance(threshold, str) else fixed_threshold
        row_freshness = row[freshness_col] if freshness_col is not None else True
        decisions.append(
            decide_push(
                market_score=row[score_col],
                threshold=row_threshold,
                signal_values=row,
                data_is_fresh=row_freshness,
                allowed_signals=allowed,
                clarity_coefficients=clarity_coefficients,
            )
        )

    result = frame.copy()
    result["meta_should_send"] = [decision.should_send for decision in decisions]
    result["meta_selected_signal"] = pd.array(
        [decision.selected_signal for decision in decisions], dtype="string"
    )
    result["meta_reason"] = pd.array([decision.reason for decision in decisions], dtype="string")
    result["meta_selected_effect"] = pd.array(
        [decision.selected_effect for decision in decisions], dtype="Float64"
    )
    result["meta_selection_score"] = pd.array(
        [decision.selection_score for decision in decisions], dtype="Float64"
    )
    result["meta_active_signals"] = [decision.active_signals for decision in decisions]
    result["meta_signal_scores"] = [decision.signal_scores for decision in decisions]
    return result
