from pathlib import Path

import pytest

from fx_signal.data import load_rates, parse_cbr_xml

XML = b"""<?xml version="1.0" encoding="windows-1251"?>
<ValCurs ID="R01670"><Record Date="01.08.2026" Id="R01670">
<Nominal>10</Nominal><Value>82,5000</Value></Record></ValCurs>"""


def test_parse_and_normalize_cbr_xml() -> None:
    result = parse_cbr_xml(XML, "TJS", "R01670").iloc[0]
    assert result["nominal"] == 10
    assert result["rub_per_unit"] == pytest.approx(8.25)
    assert result["recipient_per_rub"] == pytest.approx(1 / 8.25)


def test_empty_xml_is_rejected() -> None:
    with pytest.raises(ValueError, match="no records"):
        parse_cbr_xml(b"<ValCurs />", "TJS", "R01670")


def test_missing_snapshot_has_actionable_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="fx-signals data fetch"):
        load_rates(tmp_path / "missing.csv")

