"""File loaders: raw upload -> raw text, ready for `extract_training_data`.

Step 2 scope is `.txt` only (ARCHITECTURE.md §5.3 / HANDOFF_STEP_2.md); `.xlsx`
and `.pdf` are stubbed so the call sites and error surface exist now without
pulling in openpyxl/pandas/pypdf before they're needed.
"""
from __future__ import annotations

from pathlib import Path


class UnsupportedFileType(Exception):
    def __init__(self, suffix: str):
        self.suffix = suffix
        super().__init__(f"Unsupported upload file type: {suffix!r}")


def parse_upload(path: str | Path) -> str:
    """Load a raw upload and return its text content.

    `.txt` files are read directly. `.xlsx`/`.pdf` raise `NotImplementedError`
    for now; anything else raises `UnsupportedFileType`.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".txt":
        return p.read_text(encoding="utf-8")
    if suffix == ".xlsx":
        raise NotImplementedError("xlsx loading is not implemented yet (planned: openpyxl/pandas)")
    if suffix == ".pdf":
        raise NotImplementedError("pdf loading is not implemented yet")

    raise UnsupportedFileType(suffix)
