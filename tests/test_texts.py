from pathlib import Path

import pytest
import yaml

from fx_signal.signal_snapshot import DEFAULT_ALLOWED_SIGNALS
from fx_signal.texts import load_text_library, render_message


def test_repository_text_library_covers_allowed_signals() -> None:
    library = load_text_library(
        Path("configs/texts.yaml"),
        required_signals=set(DEFAULT_ALLOWED_SIGNALS),
    )

    assert set(library.scenarios) == set(DEFAULT_ALLOWED_SIGNALS)
    assert library.forbidden
    title, body = render_message(
        library.scenarios["signal_better_than_one_year_ago"],
        currency="TJS",
        effect=0.01234,
    )
    assert title
    assert "TJS" in body
    assert "1.23%" in body


def test_text_library_rejects_missing_scenario(tmp_path: Path) -> None:
    path = tmp_path / "texts.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "scenarios": [
                    {
                        "signal": "signal_momentum",
                        "scenario": "momentum",
                        "direction": "recipient_value_up",
                        "speed": "fast",
                        "push_title": "Факт",
                        "push_body": "Текст",
                    }
                ],
                "forbidden": [{"phrase": "Обещание", "reason": "Это прогноз"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="coverage mismatch"):
        load_text_library(path, required_signals={"signal_momentum", "signal_level"})
