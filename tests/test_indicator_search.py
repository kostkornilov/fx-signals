import pandas as pd

from fx_signal.indicator_search import apply_communication_policy, purge_validation_tail
from fx_signal.public_context import causal_context_join
from fx_signal.rule_catalog import build_rule_catalog


def _rates(periods: int = 300) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "currency": "TJS",
            "effective_date": pd.bdate_range("2024-01-01", periods=periods),
            "rub_per_unit": [10 - index * 0.001 for index in range(periods)],
            "ret_1": [-0.001] * periods,
        }
    )


def test_rule_catalog_is_causal_when_future_is_appended() -> None:
    rates = _rates()
    short, specs = build_rule_catalog(rates.iloc[:260])
    full, full_specs = build_rule_catalog(rates)
    columns = [spec.rule_id for spec in specs]
    assert columns == [spec.rule_id for spec in full_specs]
    pd.testing.assert_frame_equal(
        short[columns].reset_index(drop=True),
        full.iloc[:260][columns].reset_index(drop=True),
    )


def test_communication_policy_applies_cooldown_and_weekly_cap() -> None:
    frame = _rates(10)
    raw = pd.Series(True, index=frame.index)
    kept = apply_communication_policy(frame, raw, cooldown_prints=1, max_in_7d=2)
    selected_dates = frame.loc[kept, "effective_date"].tolist()
    assert selected_dates[:2] == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03")]
    assert selected_dates[2] >= pd.Timestamp("2024-01-08")


def test_context_join_waits_until_available_at() -> None:
    frame = _rates(4)
    context = pd.DataFrame(
        {
            "effective_date": [pd.Timestamp("2024-01-01")],
            "available_at": [pd.Timestamp("2024-01-03")],
            "name": ["ctx_oil"],
            "value": [80.0],
        }
    )
    result = causal_context_join(frame, context)
    assert pd.isna(result.loc[0, "ctx_oil"])
    assert pd.isna(result.loc[1, "ctx_oil"])
    assert result.loc[2, "ctx_oil"] == 80.0


def test_validation_tail_is_purged_per_currency() -> None:
    first = _rates(10)
    second = first.assign(currency="KZT")
    result = purge_validation_tail(pd.concat([first, second], ignore_index=True), 3)
    assert result.groupby("currency").size().to_dict() == {"KZT": 7, "TJS": 7}
    assert result.groupby("currency")["effective_date"].max().nunique() == 1
