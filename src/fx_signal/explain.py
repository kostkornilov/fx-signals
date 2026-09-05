from __future__ import annotations

import pandas as pd

STAY_NOT_WORSE_FACTS = ("level", "momentum", "seasonality")
WINDOW_CLOSING_FACTS = ("reversal",)
DEFAULT_RANK = {"level": 3, "momentum": 2, "seasonality": 1, "reversal": 3}
FAST_FACTS = {"momentum"}
SCENARIO_BY_TARGET = {
    "stay_not_worse": "now_cheap",
    "local_min": "now_cheap",
    "window_closing": "window_closing",
}


def _candidates_for(target_kind: str) -> tuple[str, ...]:
    if SCENARIO_BY_TARGET[target_kind] == "window_closing":
        return WINDOW_CLOSING_FACTS
    return STAY_NOT_WORSE_FACTS


def _strength(row: pd.Series, fact: str) -> float:
    if fact == "level":
        percentile = row.get("price_percentile")
        return 1.0 - float(percentile) if pd.notna(percentile) else 0.0
    if fact == "momentum":
        streak = row.get("down_streak")
        return float(streak) if pd.notna(streak) else 0.0
    if fact == "seasonality":
        days = row.get("days_to_holiday")
        return -float(days) if pd.notna(days) else float("-inf")
    if fact == "reversal":
        rebound = row.get("rebound_pct")
        return float(rebound) if pd.notna(rebound) else 0.0
    return 0.0


def _true_facts(row: pd.Series, allowed: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    for fact in allowed:
        value = row.get(f"signal_{fact}")
        if bool(value) and pd.notna(value):
            found.append(fact)
    return found


def pick_fact(
    row: pd.Series,
    target_kind: str = "stay_not_worse",
    rank: dict[str, int] | None = None,
) -> dict[str, object] | None:
    """Choose one observable fact for the push, or None to stay silent."""
    allowed = _candidates_for(target_kind)
    facts = _true_facts(row, allowed)
    if not facts:
        return None
    ranking = rank or DEFAULT_RANK
    chosen = max(facts, key=lambda fact: (ranking.get(fact, 0), _strength(row, fact)))
    scenario = SCENARIO_BY_TARGET[target_kind]
    direction = "up" if chosen == "reversal" else "down"
    return {
        "indicator": chosen,
        "direction": direction,
        "strength": _strength(row, chosen),
        "speed": "fast" if chosen in FAST_FACTS else "slow",
        "scenario": scenario,
    }


def add_explanations(
    frame: pd.DataFrame,
    *,
    signal_col: str = "ml_signal",
    target_kind: str = "stay_not_worse",
) -> pd.DataFrame:
    result = frame.copy()
    picked = [
        pick_fact(row, target_kind=target_kind) if bool(row.get(signal_col)) else None
        for row in result.to_dict("records")
    ]
    result["chosen_indicator"] = [item["indicator"] if item else pd.NA for item in picked]
    result["direction"] = [item["direction"] if item else pd.NA for item in picked]
    result["strength"] = [item["strength"] if item else pd.NA for item in picked]
    result["speed"] = [item["speed"] if item else pd.NA for item in picked]
    result["scenario"] = [item["scenario"] if item else pd.NA for item in picked]
    return result
