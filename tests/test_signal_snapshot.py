from pathlib import Path

import numpy as np
import pandas as pd

from fx_signal.signal_snapshot import (
    OUTPUT_COLUMNS,
    apply_signal_policy,
    prepare_snapshot_frame,
    run_signal_snapshot,
)


def _rates(dates: pd.DatetimeIndex, currencies: tuple[str, ...] = ("TJS",)) -> pd.DataFrame:
    rows = []
    for number, currency in enumerate(currencies):
        values = 8.0 + number + np.sin(np.arange(len(dates)) / 9.0)
        rows.append(
            pd.DataFrame(
                {
                    "effective_date": dates,
                    "available_at": dates,
                    "currency": currency,
                    "cbr_id": f"id-{currency}",
                    "nominal": 1,
                    "value_rub": values,
                    "rub_per_unit": values,
                    "recipient_per_rub": 1.0 / values,
                    "is_fresh": True,
                    "source": "test",
                    "fetched_at": pd.Timestamp("2026-01-01", tz="UTC"),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _write_data(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_snapshot_features_do_not_change_when_future_rows_change(tmp_path: Path) -> None:
    dates = pd.bdate_range("2021-01-01", "2024-06-30")
    rates = _rates(dates)
    data_path = tmp_path / "rates.csv"
    _write_data(data_path, rates)
    config = {
        "data_path": str(data_path),
        "horizon": 5,
        "momentum_days": 3,
        "level_window": 30,
        "level_quantile": 0.1,
        "reversal_window": 20,
        "holiday_lookahead_days": 7,
    }
    cutoff = "2024-01-31"
    original = prepare_snapshot_frame(config, tmp_path, cutoff)

    future = pd.to_datetime(rates["available_at"]).gt(cutoff)
    rates.loc[future, "rub_per_unit"] *= 100
    rates.loc[future, "value_rub"] *= 100
    rates.loc[future, "recipient_per_rub"] /= 100
    _write_data(data_path, rates)
    changed = prepare_snapshot_frame(config, tmp_path, cutoff)

    columns = ["effective_date", "currency", "ret_1", "signal_momentum"]
    pd.testing.assert_frame_equal(
        original[columns].reset_index(drop=True),
        changed[columns].reset_index(drop=True),
    )


def test_signal_policy_applies_cooldown_and_weekly_limit() -> None:
    frame = pd.DataFrame(
        {
            "currency": ["TJS"] * 5,
            "effective_date": pd.to_datetime(
                ["2026-08-03", "2026-08-04", "2026-08-07", "2026-08-08", "2026-08-10"]
            ),
            "candidate": True,
        }
    )

    result = apply_signal_policy(
        frame,
        candidate_col="candidate",
        cooldown_days=2,
        weekly_limit=2,
    )

    assert result.tolist() == [True, False, True, False, True]


def test_run_snapshot_writes_stable_schema_for_empty_result(tmp_path: Path) -> None:
    dates = pd.bdate_range("2019-01-01", "2024-06-28")
    data_path = tmp_path / "data" / "rates.csv"
    _write_data(data_path, _rates(dates, ("TJS", "KZT")))
    texts_path = Path("configs/texts.yaml").resolve()
    config_path = tmp_path / "configs" / "signals.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "\n".join(
            [
                f"data_path: {data_path}",
                f"texts_path: {texts_path}",
                "corridors: [TJS, KZT]",
                "target: stay_not_worse",
                "horizon: 5",
                "model: logreg",
                "feature_groups: [A, B]",
                "validation_days: 365",
                "policy_lookback_days: 10",
                "thresholds: [1.0]",
                "quantile_rates: []",
                "default_threshold: 1.0",
                "target_signals_per_week: [0.0, 2.0]",
                "momentum_days: 3",
                "level_window: 30",
                "level_quantile: 0.1",
                "reversal_window: 20",
                "holiday_lookahead_days: 7",
                "allowed_signals: [signal_momentum]",
                "cooldown_days: 3",
                "weekly_limit: 0",
                "max_staleness_days: 3",
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "signals.csv"

    path = run_signal_snapshot(config_path, as_of="2024-06-28", output_path=output)
    result = pd.read_csv(path)

    assert list(result.columns) == list(OUTPUT_COLUMNS)
    assert result.empty
