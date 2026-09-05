import pandas as pd

from fx_signal.targets import (
    add_local_min_target,
    add_stay_not_worse_target,
    add_window_closing_target,
)


def test_local_min_h1_is_computed_per_currency() -> None:
    rates = pd.DataFrame(
        {
            "currency": ["TJS"] * 5 + ["KZT"] * 3,
            "rub_per_unit": [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 1.0, 4.0],
        }
    )
    result = add_local_min_target(rates, horizon=1)
    target = result["target_local_min_h1"]
    assert target.iloc[2]
    assert target.iloc[6]
    assert pd.isna(target.iloc[0])
    assert pd.isna(target.iloc[5])


def test_stay_not_worse_requires_all_future_prints() -> None:
    rates = pd.DataFrame(
        {
            "currency": ["TJS"] * 5,
            "rub_per_unit": [2.0, 2.0, 1.5, 1.6, 1.7],
        }
    )
    result = add_stay_not_worse_target(rates, horizon=2)
    target = result["target_stay_not_worse_h2"]
    assert not bool(target.iloc[0])
    assert not bool(target.iloc[1])
    assert bool(target.iloc[2])
    assert pd.isna(target.iloc[3])
    assert pd.isna(target.iloc[4])


def test_window_closing_is_forward_only() -> None:
    rates = pd.DataFrame(
        {"currency": ["TJS"] * 4, "rub_per_unit": [1.0, 1.0, 1.2, 1.1]}
    )
    result = add_window_closing_target(rates, horizon=2)
    target = result["target_window_closing_h2"]
    assert bool(target.iloc[0])
    assert bool(target.iloc[1])
    assert pd.isna(target.iloc[2])

