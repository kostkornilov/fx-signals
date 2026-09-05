from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"effective_date", "available_at", "name", "value"}


def load_public_context(path: Path) -> pd.DataFrame:
    context = pd.read_csv(path, parse_dates=["effective_date", "available_at"])
    missing = REQUIRED_COLUMNS.difference(context.columns)
    if missing:
        raise ValueError(f"External context misses columns: {sorted(missing)}")
    if (context["available_at"] < context["effective_date"]).any():
        raise ValueError("available_at cannot precede effective_date")
    return context.sort_values(["name", "available_at"])


def causal_context_join(frame: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Backward as-of join using publication availability, never observation date alone."""
    base = frame.copy()
    base["_row_id"] = np.arange(len(base))
    dates = base[["effective_date"]].drop_duplicates().sort_values("effective_date")
    wide = dates.copy()
    for name, group in context.groupby("name", sort=False):
        values = group[["available_at", "value"]].dropna().sort_values("available_at")
        aligned = pd.merge_asof(
            dates,
            values,
            left_on="effective_date",
            right_on="available_at",
            direction="backward",
            allow_exact_matches=True,
        )[["effective_date", "value"]].rename(columns={"value": str(name)})
        wide = wide.merge(aligned, on="effective_date", how="left")
    return base.merge(wide, on="effective_date", how="left").sort_values("_row_id").drop(columns="_row_id")


def fetch_public_context(config: dict, repo_root: Path, force: bool = False) -> Path:
    """Fetch configured public CSV series into one auditable long-form snapshot.

    Every source must explicitly provide a conservative publication lag. Sources whose
    historical availability cannot be represented should not be configured.
    """
    output = repo_root / Path(config.get("public_context_path", "data/raw/public/context.csv"))
    if output.exists() and not force:
        return output
    frames, manifest = [], []
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for source in config.get("public_sources", []):
            raw_parts: list[bytes] = []
            if source.get("format", "csv") == "moex_iss":
                records: list[list] = []
                columns: list[str] = []
                start = 0
                while True:
                    response = client.get(
                        source["url"],
                        params={"iss.meta": "off", "start": start},
                    )
                    response.raise_for_status()
                    raw_parts.append(response.content)
                    block = response.json()[source.get("block", "history")]
                    columns = block["columns"]
                    page = block["data"]
                    if not page:
                        break
                    records.extend(page)
                    start += len(page)
                table = pd.DataFrame(records, columns=columns)
            else:
                response = client.get(source["url"])
                response.raise_for_status()
                raw_parts.append(response.content)
                table = pd.read_csv(io.BytesIO(response.content))
            dates = pd.to_datetime(table[source["date_column"]], errors="coerce")
            values = pd.to_numeric(table[source["value_column"]], errors="coerce")
            lag = int(source["availability_lag_days"])
            frame = pd.DataFrame({
                "effective_date": dates,
                "available_at": dates + pd.to_timedelta(lag, unit="D"),
                "name": source["name"],
                "value": values,
                "source": source["url"],
            }).dropna(subset=["effective_date", "value"])
            frames.append(frame)
            manifest.append({
                "name": source["name"], "url": source["url"], "license": source.get("license"),
                "availability_lag_days": lag, "rows": len(frame),
                "sha256": hashlib.sha256(b"".join(raw_parts)).hexdigest(),
            })
    if not frames:
        raise ValueError("No public_sources configured")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).sort_values(["name", "effective_date"]).to_csv(output, index=False)
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps({
        "schema_version": 1, "fetched_at": datetime.now(UTC).isoformat(), "sources": manifest,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
