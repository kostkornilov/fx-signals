from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("event_date", "available_at", "source", "series_id", "value")


def safe_series_name(value: str) -> str:
    """Return a stable feature prefix for an external series identifier."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def validate_external_series(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the point-in-time contract for external market data."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"External series missing columns: {', '.join(missing)}")
    result = frame.copy()
    result["event_date"] = pd.to_datetime(result["event_date"], utc=True).dt.tz_localize(None)
    result["available_at"] = pd.to_datetime(result["available_at"], utc=True).dt.tz_localize(None)
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    if result["value"].isna().any():
        raise ValueError("External series contains non-numeric values")
    if (result["available_at"] < result["event_date"]).any():
        raise ValueError("available_at cannot be earlier than event_date")
    if result["series_id"].astype(str).map(safe_series_name).duplicated().any():
        # Duplicated rows are valid, but distinct identifiers must not collapse to one feature name.
        names = result[["series_id"]].drop_duplicates().assign(
            safe=lambda x: x["series_id"].astype(str).map(safe_series_name)
        )
        if names["safe"].duplicated().any():
            raise ValueError("External series identifiers collide after normalization")
    keys = ["source", "series_id", "event_date", "available_at"]
    if result.duplicated(keys).any():
        raise ValueError("External series contains duplicate point-in-time observations")
    return result.sort_values(["series_id", "available_at", "event_date"]).reset_index(drop=True)


def load_external_series(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"External data snapshot not found at {path}")
    return validate_external_series(pd.read_csv(path))


def triangulate_cross(
    base_per_usd: pd.Series,
    recipient_per_usd: pd.Series,
) -> pd.Series:
    """Compute units of base currency per one unit of recipient currency."""
    denominator = pd.to_numeric(recipient_per_usd, errors="coerce")
    numerator = pd.to_numeric(base_per_usd, errors="coerce")
    return numerator.div(denominator.where(denominator.gt(0)))
