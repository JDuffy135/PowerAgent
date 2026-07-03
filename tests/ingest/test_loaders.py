from pathlib import Path

import pytest

from src.ingest.loaders import UnsupportedFileType, parse_upload

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_parse_upload_reads_txt():
    text = parse_upload(FIXTURES_DIR / "kg_lb_mixed.txt")
    assert "Bench Press" in text


def test_parse_upload_xlsx_emits_one_block_per_sheet():
    text = parse_upload(FIXTURES_DIR / "training_log.xlsx")
    # One header per sheet, in workbook order.
    assert "=== Sheet: Block 1 - Strength ===" in text
    assert "=== Sheet: Cardio ===" in text
    assert text.index("Block 1 - Strength") < text.index("Cardio")


def test_parse_upload_xlsx_preserves_messy_cells_verbatim():
    text = parse_upload(FIXTURES_DIR / "training_log.xlsx")
    # The extractor is built for these strings -- the loader must not clean them.
    assert "1x3 @ 170KG" in text
    assert "385x1, 315x4" in text
    assert "Reps: N/A" in text
    assert "🚴" in text  # emoji preserved


def test_parse_upload_xlsx_drops_fully_blank_rows():
    text = parse_upload(FIXTURES_DIR / "training_log.xlsx")
    # No line should be empty within a sheet block (blank rows are dropped);
    # the only blank lines are the double-newline separators between sheets.
    block = text.split("=== Sheet: Cardio ===")[0]
    body_lines = [ln for ln in block.splitlines() if ln and not ln.startswith("=== Sheet")]
    assert all(ln.strip() for ln in body_lines)


def test_parse_upload_pdf_one_block_per_page():
    text = parse_upload(FIXTURES_DIR / "study.pdf")
    assert "=== Page 1 ===" in text
    assert "=== Page 2 ===" in text
    assert "RPE and Autoregulation" in text
    assert "deadlift volume landmarks" in text


def test_parse_upload_unsupported_type(tmp_path):
    path = tmp_path / "log.docx"
    path.write_text("hi")
    with pytest.raises(UnsupportedFileType):
        parse_upload(path)
