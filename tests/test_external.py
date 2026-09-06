from pathlib import Path

import pandas as pd
import pytest

from fx_signal.external import load_external_series, triangulate_cross, validate_external_series
from fx_signal.features import add_features, columns_for_groups


def test_external_contract_rejects_future_availability() -> None:
    frame = pd.DataFrame(
        {
            "event_date": ["2025-01-02"],
            "available_at": ["2025-01-01"],
            "source": ["test"],
            "series_id": ["usd_local"],
            "value": [10.0],
        }
    )
    with pytest.raises(ValueError, match="earlier"):
        validate_external_series(frame)


def test_external_loader_requires_point_in_time_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2025-01-01"], "value": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_external_series(path)


def test_triangulated_cross() -> None:
    result = triangulate_cross(pd.Series([80.0]), pd.Series([10.0]))
    assert result.iloc[0] == pytest.approx(8.0)


def test_external_features_obey_available_at() -> None:
    dates = pd.date_range("2025-01-01", periods=110, freq="D")
    rates = pd.DataFrame(
        {"currency": "TJS", "effective_date": dates, "rub_per_unit": range(110, 220)}
    )
    external = pd.DataFrame(
        {
            "event_date": dates[:100],
            "available_at": dates[:100] + pd.to_timedelta(2, unit="D"),
            "source": "test",
            "series_id": "usd_local",
            "value": range(100, 200),
        }
    )
    result = add_features(rates, external=validate_external_series(external))
    columns = columns_for_groups(["F"], result)
    assert "ext_usd_local_staleness_days" in columns
    assert result.loc[2, "ext_usd_local_staleness_days"] == pytest.approx(0.0)
    assert pd.isna(result.loc[1, "ext_usd_local_staleness_days"])


def test_public_context_maps_to_external_contract() -> None:
    context = pd.DataFrame(
        {
            "effective_date": ["2025-01-02"],
            "available_at": ["2025-01-03"],
            "name": ["moex_imoex"],
            "value": [3000.0],
        }
    )
    from fx_signal.external import from_public_context

    series = from_public_context(context)
    assert list(series.columns[:5]) == [
        "event_date",
        "available_at",
        "source",
        "series_id",
        "value",
    ]
    assert series.loc[0, "series_id"] == "moex_imoex"
    assert series.loc[0, "source"] == "moex"
