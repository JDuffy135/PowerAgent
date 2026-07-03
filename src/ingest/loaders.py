"""File loaders: raw upload -> raw text, ready for `extract_training_data`.

`.txt` is read directly. `.xlsx` and `.pdf` are implemented in Stage 8; the raw
text they produce is deliberately *un-cleaned* -- the messy strings (mixed
lb/kg, pin settings, slang) are exactly what `extract_training_data` is built to
parse, so pre-normalizing here would throw away signal.

**[DECISION] xlsx layout is not assumed.** Every coach's workbook looks
different (blocks-per-sheet, weeks-as-columns, free-form logs), so the xlsx
loader makes no structural guesses: it emits one text block per sheet with a
header line, rows as tab-joined cell text, and lets the LLM extractor figure out
the shape. Fully blank rows/columns are dropped so the extractor isn't fed a
grid of `None`s.
"""
from __future__ import annotations

from pathlib import Path


class UnsupportedFileType(Exception):
    def __init__(self, suffix: str):
        self.suffix = suffix
        super().__init__(f"Unsupported upload file type: {suffix!r}")


def parse_upload(path: str | Path) -> str:
    """Load a raw upload and return its text content.

    `.txt` files are read directly; `.xlsx` via openpyxl (one block per sheet);
    `.pdf` via pypdf (one block per page). Anything else raises
    `UnsupportedFileType`.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".txt":
        return p.read_text(encoding="utf-8")
    if suffix == ".xlsx":
        return _load_xlsx(p)
    if suffix == ".pdf":
        return _load_pdf(p)

    raise UnsupportedFileType(suffix)


def _cell_text(value) -> str:
    """Render one cell verbatim. Numbers/dates come back as their str()."""
    if value is None:
        return ""
    return str(value).strip()


def _load_xlsx(path: Path) -> str:
    """Read every sheet of a workbook into text, one block per sheet.

    No structural assumptions (see module docstring): each non-empty row becomes
    a tab-joined line, trailing empty cells trimmed. Sheets are separated by a
    `=== Sheet: <name> ===` header so the extractor can tell tabs apart. Reads
    with `data_only=True` so formula cells yield their cached value, not the
    formula string.
    """
    import openpyxl

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        blocks: list[str] = []
        for sheet in workbook.worksheets:
            lines: list[str] = []
            for row in sheet.iter_rows(values_only=True):
                cells = [_cell_text(v) for v in row]
                while cells and cells[-1] == "":
                    cells.pop()
                if not cells:
                    continue
                lines.append("\t".join(cells))
            block = f"=== Sheet: {sheet.title} ===\n" + "\n".join(lines)
            blocks.append(block.rstrip())
        return "\n\n".join(blocks)
    finally:
        workbook.close()


def _load_pdf(path: Path) -> str:
    """Extract text from a PDF, one block per page.

    Uses pypdf's text extraction; pages are separated by a `=== Page N ===`
    header. Pages with no extractable text (e.g. scanned images -- OCR is out of
    scope) are skipped rather than emitting empty blocks.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    blocks: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        blocks.append(f"=== Page {i} ===\n{text}")
    return "\n\n".join(blocks)
