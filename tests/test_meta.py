import math

import pandas as pd
import pytest

from fx_signal.indicators import (
    RESEARCH_SIGNAL_COLUMNS,
    RESEARCH_SIGNAL_EFFECT_COLUMN,
)
from fx_signal.meta import add_meta_decisions, decide_push


def _signal_row(**effects: float | None) -> dict[str, object]:
    row: dict[str, object] = {}
    for signal in RESEARCH_SIGNAL_COLUMNS:
        row[signal] = signal in effects
        row[RESEARCH_SIGNAL_EFFECT_COLUMN[signal]] = effects.get(signal, math.nan)
    return row


def test_market_score_is_the_send_gate() -> None:
    row = _signal_row(signal_better_than_30_day_average=0.02)

    below = decide_push(market_score=0.79, threshold=0.8, signal_values=row)
    at_threshold = decide_push(market_score=0.8, threshold=0.8, signal_values=row)

    assert not below.should_send
    assert below.reason == "below_market_threshold"
    assert at_threshold.should_send
    assert at_threshold.selected_signal == "signal_better_than_30_day_average"


def test_one_active_signal_is_selected_without_comparing_effects() -> None:
    row = _signal_row(signal_better_than_one_year_ago=None)

    decision = decide_push(market_score=0.9, threshold=0.8, signal_values=row)

    assert decision.should_send
    assert decision.reason == "selected_single_signal"
    assert decision.selected_signal == "signal_better_than_one_year_ago"
    assert decision.selected_effect is None
    assert decision.selection_score is None


def test_multiple_signals_choose_largest_clarity_adjusted_effect() -> None:
    row = _signal_row(
        signal_better_than_30_day_average=0.02,
        signal_better_than_one_year_ago=0.05,
        signal_most_recent_changes_favourable=0.03,
    )

    decision = decide_push(market_score=0.9, threshold=0.8, signal_values=row)

    assert decision.should_send
    assert decision.reason == "selected_by_effect"
    assert decision.selected_signal == "signal_better_than_one_year_ago"
    assert decision.selected_effect == pytest.approx(0.05)
    assert decision.selection_score == pytest.approx(0.05 * 0.95)


def test_clarity_coefficient_can_make_a_close_effect_win() -> None:
    row = _signal_row(
        signal_better_than_30_day_average=0.01,
        signal_better_than_one_year_ago=0.0104,
    )

    decision = decide_push(market_score=0.9, threshold=0.8, signal_values=row)

    assert decision.selected_signal == "signal_better_than_30_day_average"
    assert decision.selection_score == pytest.approx(0.01)


def test_worsening_effects_are_compared_by_absolute_size() -> None:
    row = _signal_row(
        signal_less_than_one_year_ago=-0.04,
        signal_most_recent_changes_unfavourable=-0.05,
    )

    decision = decide_push(market_score=0.9, threshold=0.8, signal_values=row)

    assert decision.should_send
    assert decision.selected_signal == "signal_less_than_one_year_ago"
    assert decision.selected_effect == pytest.approx(-0.04)
    assert decision.selection_score == pytest.approx(0.04 * 0.95)


def test_multiple_signals_require_all_effects() -> None:
    row = _signal_row(
        signal_better_than_30_day_average=0.02,
        signal_better_than_one_year_ago=None,
    )

    decision = decide_push(market_score=0.9, threshold=0.8, signal_values=row)

    assert not decision.should_send
    assert decision.reason == "invalid_signal_effect"
    assert decision.selected_signal is None


@pytest.mark.parametrize(
    ("score", "is_fresh", "reason"),
    [
        (math.nan, True, "invalid_market_score"),
        (math.inf, True, "invalid_market_score"),
        (0.9, False, "stale_market_data"),
        (0.9, pd.NA, "stale_market_data"),
    ],
)
def test_invalid_or_stale_market_data_fails_closed(
    score: float, is_fresh: object, reason: str
) -> None:
    decision = decide_push(
        market_score=score,
        threshold=0.8,
        signal_values=_signal_row(signal_better_than_one_year_ago=0.02),
        data_is_fresh=is_fresh,
    )

    assert not decision.should_send
    assert decision.selected_signal is None
    assert decision.reason == reason


def test_allowed_signals_can_restrict_selection() -> None:
    row = _signal_row(
        signal_better_than_30_day_average=0.02,
        signal_better_than_one_year_ago=0.05,
    )

    decision = decide_push(
        market_score=0.9,
        threshold=0.8,
        signal_values=row,
        allowed_signals={"signal_better_than_30_day_average"},
    )

    assert decision.should_send
    assert decision.selected_signal == "signal_better_than_30_day_average"
    assert decision.reason == "selected_single_signal"


def test_add_meta_decisions_supports_a_per_row_threshold() -> None:
    frame = pd.DataFrame(
        [
            {
                "market_score": 0.9,
                "market_threshold": 0.8,
                **_signal_row(
                    signal_better_than_30_day_average=0.02,
                    signal_better_than_one_year_ago=0.05,
                ),
            },
            {
                "market_score": 0.7,
                "market_threshold": 0.8,
                **_signal_row(signal_better_than_one_year_ago=0.05),
            },
            {
                "market_score": 0.9,
                "market_threshold": 0.8,
                **_signal_row(signal_most_recent_changes_unfavourable=-0.03),
            },
        ]
    )

    result = add_meta_decisions(frame, threshold="market_threshold")

    assert result["meta_should_send"].tolist() == [True, False, True]
    assert result["meta_selected_signal"].tolist() == [
        "signal_better_than_one_year_ago",
        pd.NA,
        "signal_most_recent_changes_unfavourable",
    ]
    assert result["meta_selected_effect"].tolist() == [0.05, pd.NA, -0.03]
    assert result["meta_reason"].tolist() == [
        "selected_by_effect",
        "below_market_threshold",
        "selected_single_signal",
    ]


def test_custom_clarity_coefficients_are_supported() -> None:
    row = _signal_row(
        signal_better_than_30_day_average=0.02,
        signal_better_than_one_year_ago=0.03,
    )

    decision = decide_push(
        market_score=0.9,
        threshold=0.8,
        signal_values=row,
        clarity_coefficients={"signal_better_than_one_year_ago": 0.5},
    )

    assert decision.selected_signal == "signal_better_than_30_day_average"


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="threshold must be a finite number"):
        decide_push(
            market_score=0.9,
            threshold=math.nan,
            signal_values=_signal_row(),
        )

    with pytest.raises(ValueError, match="Unknown signal columns"):
        decide_push(
            market_score=0.9,
            threshold=0.8,
            signal_values=_signal_row(),
            allowed_signals={"signal_unknown"},
        )

    with pytest.raises(ValueError, match="must be in"):
        decide_push(
            market_score=0.9,
            threshold=0.8,
            signal_values=_signal_row(),
            clarity_coefficients={"signal_better_than_one_year_ago": 1.5},
        )
