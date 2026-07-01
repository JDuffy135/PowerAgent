from pathlib import Path

import pytest

from src.ingest.loaders import UnsupportedFileType, parse_upload

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_parse_upload_reads_txt():
    text = parse_upload(FIXTURES_DIR / "kg_lb_mixed.txt")
    assert "Bench Press" in text


def test_parse_upload_xlsx_not_implemented(tmp_path):
    path = tmp_path / "log.xlsx"
    path.write_bytes(b"")
    with pytest.raises(NotImplementedError):
        parse_upload(path)


def test_parse_upload_pdf_not_implemented(tmp_path):
    path = tmp_path / "log.pdf"
    path.write_bytes(b"")
    with pytest.raises(NotImplementedError):
        parse_upload(path)


def test_parse_upload_unsupported_type(tmp_path):
    path = tmp_path / "log.docx"
    path.write_text("hi")
    with pytest.raises(UnsupportedFileType):
        parse_upload(path)
