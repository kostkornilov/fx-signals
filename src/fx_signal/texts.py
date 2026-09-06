from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Formatter

import yaml


@dataclass(frozen=True, slots=True)
class MessageTemplate:
    signal: str
    scenario: str
    direction: str
    speed: str
    push_title: str
    push_body: str


@dataclass(frozen=True, slots=True)
class ForbiddenPhrase:
    phrase: str
    reason: str


@dataclass(frozen=True, slots=True)
class TextLibrary:
    scenarios: dict[str, MessageTemplate]
    forbidden: tuple[ForbiddenPhrase, ...]


ALLOWED_PLACEHOLDERS = {"currency", "effect_pct"}


def _placeholders(value: str) -> set[str]:
    return {
        field_name
        for _, field_name, _, _ in Formatter().parse(value)
        if field_name is not None
    }


def load_text_library(path: Path, *, required_signals: set[str] | None = None) -> TextLibrary:
    """Load and validate the compliance-safe push-copy library."""
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}

    scenario_rows = raw.get("scenarios")
    if not isinstance(scenario_rows, list) or not scenario_rows:
        raise ValueError("Text library must contain a non-empty 'scenarios' list")

    scenarios: dict[str, MessageTemplate] = {}
    for row in scenario_rows:
        if not isinstance(row, dict):
            raise TypeError("Every text scenario must be a mapping")
        required = {"signal", "scenario", "direction", "speed", "push_title", "push_body"}
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(f"Text scenario is missing fields: {missing}")
        template = MessageTemplate(**{name: str(row[name]) for name in required})
        if template.signal in scenarios:
            raise ValueError(f"Duplicate text scenario for {template.signal!r}")
        if template.speed not in {"fast", "slow"}:
            raise ValueError(f"Invalid speed for {template.signal!r}: {template.speed!r}")
        fields = _placeholders(template.push_title) | _placeholders(template.push_body)
        unknown = sorted(fields.difference(ALLOWED_PLACEHOLDERS))
        if unknown:
            raise ValueError(f"Unknown placeholders for {template.signal!r}: {unknown}")
        scenarios[template.signal] = template

    if required_signals is not None:
        missing = sorted(required_signals.difference(scenarios))
        extra = sorted(set(scenarios).difference(required_signals))
        if missing or extra:
            raise ValueError(
                f"Text-library coverage mismatch; missing={missing}, extra={extra}"
            )

    forbidden_rows = raw.get("forbidden")
    if not isinstance(forbidden_rows, list) or not forbidden_rows:
        raise ValueError("Text library must contain a non-empty 'forbidden' list")
    forbidden: list[ForbiddenPhrase] = []
    for row in forbidden_rows:
        if not isinstance(row, dict) or not row.get("phrase") or not row.get("reason"):
            raise ValueError("Every forbidden phrase must have non-empty phrase and reason")
        forbidden.append(ForbiddenPhrase(phrase=str(row["phrase"]), reason=str(row["reason"])))

    return TextLibrary(scenarios=scenarios, forbidden=tuple(forbidden))


def render_message(template: MessageTemplate, *, currency: str, effect: float | None) -> tuple[str, str]:
    values = {
        "currency": currency,
        "effect_pct": "—" if effect is None else f"{abs(effect) * 100:.2f}",
    }
    return template.push_title.format(**values), template.push_body.format(**values)
