import pandas as pd

from fx_signal.targets import add_local_min_target


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

