import math

import pandas as pd
import pytest

from fx_signal.indicators import RESEARCH_SIGNAL_COLUMNS
from fx_signal.meta import add_meta_decisions, decide_push


def _signals(*active: str) -> dict[str, bool]:
    return {signal: signal in active for signal in RESEARCH_SIGNAL_COLUMNS}


def test_market_score_is_a_gate() -> None:
    signal = "signal_better_than_30_day_average"

    below = decide_push(
        market_score=0.79,
        threshold=0.8,
        signal_values=_signals(signal),
    )
    at_threshold = decide_push(
        market_score=0.8,
        threshold=0.8,
        signal_values=_signals(signal),
    )

    assert not below.should_send
    assert below.reason == "below_market_threshold"
    assert at_threshold.should_send
    assert at_threshold.selected_signal == signal
    assert at_threshold.reason == "selected_single_signal"


def test_multiple_favourable_signals_use_clarity_priority() -> None:
    decision = decide_push(
        market_score=0.9,
        threshold=0.8,
        signal_values=_signals(
            "signal_better_than_one_year_ago",
            "signal_better_than_30_day_average",
            "signal_most_recent_changes_favourable",
        ),
    )

    assert decision.should_send
    assert decision.selected_signal == "signal_better_than_30_day_average"
    assert decision.reason == "selected_by_clarity"
    assert decision.eligible_signals[0] == decision.selected_signal


def test_signals_inconsistent_with_forecast_are_removed() -> None:
    decision = decide_push(
        market_score=0.9,
        threshold=0.8,
        signal_values=_signals("signal_most_recent_changes_unfavourable"),
        forecast_kind="stay_not_worse",
    )

    assert not decision.should_send
    assert decision.reason == "no_eligible_signal"
    assert decision.active_signals == ("signal_most_recent_changes_unfavourable",)
    assert decision.eligible_signals == ()


def test_window_closing_prefers_recent_deterioration() -> None:
    decision = decide_push(
        market_score=0.85,
        threshold=0.8,
        signal_values=_signals(
            "signal_less_than_one_year_ago",
            "signal_most_recent_changes_unfavourable",
        ),
        forecast_kind="window_closing",
    )

    assert decision.should_send
    assert decision.forecast_intent == "deterioration"
    assert decision.selected_signal == "signal_most_recent_changes_unfavourable"


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
        signal_values=_signals("signal_better_than_one_year_ago"),
        data_is_fresh=is_fresh,
    )

    assert not decision.should_send
    assert decision.selected_signal is None
    assert decision.reason == reason


def test_allowed_signals_can_restrict_the_model_explanation() -> None:
    allowed = {"signal_better_than_one_year_ago"}
    decision = decide_push(
        market_score=0.9,
        threshold=0.8,
        signal_values=_signals(
            "signal_better_than_one_year_ago",
            "signal_better_than_30_day_average",
        ),
        allowed_signals=allowed,
    )

    assert decision.should_send
    assert decision.selected_signal == "signal_better_than_one_year_ago"
    assert decision.reason == "selected_single_signal"


def test_add_meta_decisions_supports_per_row_threshold_and_forecast_kind() -> None:
    rows = []
    for score, threshold, kind, active in (
        (0.9, 0.8, "stay_not_worse", "signal_better_than_30_day_average"),
        (0.7, 0.8, "stay_not_worse", "signal_better_than_one_year_ago"),
        (0.9, 0.8, "window_closing", "signal_most_recent_changes_unfavourable"),
    ):
        rows.append(
            {
                "market_score": score,
                "market_threshold": threshold,
                "forecast_kind": kind,
                **_signals(active),
            }
        )
    frame = pd.DataFrame(rows)

    result = add_meta_decisions(
        frame,
        threshold="market_threshold",
        forecast_kind_col="forecast_kind",
    )

    assert result["meta_should_send"].tolist() == [True, False, True]
    assert result["meta_selected_signal"].tolist() == [
        "signal_better_than_30_day_average",
        pd.NA,
        "signal_most_recent_changes_unfavourable",
    ]
    assert result["meta_reason"].tolist() == [
        "selected_single_signal",
        "below_market_threshold",
        "selected_single_signal",
    ]


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown forecast_kind"):
        decide_push(
            market_score=0.9,
            threshold=0.8,
            signal_values=_signals(),
            forecast_kind="unknown",
        )

    with pytest.raises(ValueError, match="threshold must be a finite number"):
        decide_push(
            market_score=0.9,
            threshold=math.nan,
            signal_values=_signals(),
        )

    with pytest.raises(ValueError, match="Unknown signal columns"):
        decide_push(
            market_score=0.9,
            threshold=0.8,
            signal_values=_signals(),
            allowed_signals={"signal_unknown"},
        )
