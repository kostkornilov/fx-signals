from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from xml.etree import ElementTree

import httpx
import pandas as pd
import yaml


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _as_date(value: str | date) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


def parse_cbr_xml(content: bytes, currency: str, cbr_id: str) -> pd.DataFrame:
    """Parse one XML_dynamic response into the canonical rate schema."""
    root = ElementTree.fromstring(content)
    rows: list[dict] = []
    for record in root.findall("Record"):
        nominal_text = record.findtext("Nominal")
        value_text = record.findtext("Value")
        if nominal_text is None or value_text is None:
            raise ValueError(f"Malformed CBR record for {currency}")
        nominal = int(nominal_text)
        value_rub = float(value_text.replace(",", "."))
        day, month, year = map(int, record.attrib["Date"].split("."))
        effective_date = date(year, month, day)
        rows.append(
            {
                "effective_date": effective_date,
                # Conservative daily convention: use a rate no earlier than its effective date.
                "available_at": effective_date,
                "currency": currency,
                "cbr_id": cbr_id,
                "nominal": nominal,
                "value_rub": value_rub,
                "rub_per_unit": value_rub / nominal,
                "recipient_per_rub": nominal / value_rub,
                "is_fresh": True,
            }
        )
    if not rows:
        raise ValueError(f"CBR returned no records for {currency}")
    return pd.DataFrame(rows)


def _download_group(
    client: httpx.Client,
    config: dict,
    currencies: dict[str, str],
    output_dir: Path,
    fetched_at: str,
) -> tuple[pd.DataFrame, list[dict]]:
    frames: list[pd.DataFrame] = []
    requests_manifest: list[dict] = []
    for currency, cbr_id in currencies.items():
        params = {
            "date_req1": _as_date(config["date_from"]).strftime("%d/%m/%Y"),
            "date_req2": _as_date(config["date_to"]).strftime("%d/%m/%Y"),
            "VAL_NM_RQ": cbr_id,
        }
        response = client.get(config["source"], params=params)
        response.raise_for_status()
        raw_path = output_dir / f"{currency}.xml"
        raw_path.write_bytes(response.content)
        frame = parse_cbr_xml(response.content, currency, cbr_id)
        frame["source"] = str(response.url)
        frame["fetched_at"] = fetched_at
        frames.append(frame)
        requests_manifest.append(
            {
                "currency": currency,
                "cbr_id": cbr_id,
                "url": str(response.url),
                "rows": len(frame),
                "sha256": hashlib.sha256(response.content).hexdigest(),
            }
        )
    rates = pd.concat(frames, ignore_index=True).sort_values(["currency", "effective_date"])
    return rates, requests_manifest


def fetch_snapshot(config_path: Path, force: bool = False) -> Path:
    """Download a fixed, reproducible CBR snapshot configured in YAML."""
    config = load_yaml(config_path)
    output_dir = Path(config["output_dir"])
    rates_path = output_dir / "rates.csv"
    context_path = output_dir / "context_rates.csv"
    context_currencies = config.get("context_currencies") or {}
    need_rates = force or not rates_path.exists()
    need_context = bool(context_currencies) and (force or not context_path.exists())
    if not need_rates and not need_context:
        return rates_path

    output_dir.mkdir(parents=True, exist_ok=True)
    fetched_at = datetime.now(UTC).isoformat()
    manifest: dict = {
        "schema_version": 1,
        "fetched_at": fetched_at,
        "date_from": _as_date(config["date_from"]).isoformat(),
        "date_to": _as_date(config["date_to"]).isoformat(),
    }
    if rates_path.exists() and not force:
        manifest["rates_sha256"] = hashlib.sha256(rates_path.read_bytes()).hexdigest()

    with httpx.Client(timeout=30, follow_redirects=True) as client:
        if need_rates:
            rates, requests_manifest = _download_group(
                client, config, config["currencies"], output_dir, fetched_at
            )
            rates.to_csv(rates_path, index=False)
            manifest["rates_sha256"] = hashlib.sha256(rates_path.read_bytes()).hexdigest()
            manifest["requests"] = requests_manifest
        if need_context:
            context, context_manifest = _download_group(
                client, config, context_currencies, output_dir, fetched_at
            )
            context.to_csv(context_path, index=False)
            manifest["context_sha256"] = hashlib.sha256(context_path.read_bytes()).hexdigest()
            manifest["context_requests"] = context_manifest

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return rates_path


def load_rates(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Data snapshot not found at {path}. Run: "
            "uv run fx-signals data fetch --config configs/data.yaml"
        )
    rates = pd.read_csv(path, parse_dates=["effective_date", "available_at", "fetched_at"])
    return rates.sort_values(["currency", "effective_date"]).reset_index(drop=True)
