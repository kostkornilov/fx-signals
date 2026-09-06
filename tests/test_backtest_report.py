import pandas as pd
import pytest

from fx_signal.backtest_report import (
    OOT_SUMMARY_COLUMNS,
    _candidates,
    build_oot_summary,
    render_report,
)


def _metrics() -> pd.DataFrame:
    rows = []
    for method_id, label in (("good", "Good"), ("empty", "Empty")):
        for fold in ("wf_2022", "oot"):
            for horizon in (1, 3, 5, 10, 20):
                for corridor in ("RUB->AMD", "RUB->KZT"):
                    populated = method_id == "good"
                    rows.append(
                        {
                            "method_id": method_id,
                            "method_label": label,
                            "split": fold,
                            "horizon": horizon,
                            "corridor": corridor,
                            "lift": 1.5 if populated else float("nan"),
                            "lift_at_risk": 1.2 if populated else float("nan"),
                            "moment_advantage_bps": 10.0 if populated else float("nan"),
                            "signals_per_week": 1.5 if populated else 0.0,
                            "cluster_rate": 0.1 if populated else float("nan"),
                        }
                    )
    return pd.DataFrame(rows)


def test_oot_summary_keeps_legacy_schema_and_counts_corridors() -> None:
    summary = build_oot_summary(_metrics())

    assert list(summary.columns) == OOT_SUMMARY_COLUMNS
    good = summary.set_index("method").loc["Good"]
    assert good["lift_h5"] == 1.5
    assert good["corridors_with_lar_h10"] == 2
    assert summary.set_index("method").loc["Empty", "corridors_with_lar_h5"] == 0


def test_markdown_separates_oos_from_oot_and_keeps_empty_candidate() -> None:
    report = render_report(
        _metrics(),
        {
            "first_test_year": 2022,
            "end": "2026-09-02",
            "oot_start": "2025-09-01",
            "training_horizon": 5,
            "evaluation_horizons": [1, 3, 5, 10, 20],
        },
        "abc123",
    )

    assert "Good" in report
    assert "Empty" in report
    assert "final OOT в него не входит" in report
    assert "abc123" in report


def test_candidate_registry_rejects_unknown_rule() -> None:
    with pytest.raises(ValueError, match="Unknown rule signal"):
        _candidates({"rule_signals": ["signal_unknown"], "model_candidates": []})
