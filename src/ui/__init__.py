"""Streamlit UI (Stage 9).

Run with:  streamlit run src/ui/app.py

Layout: `app.py` is the entry point (resource wiring + tab layout); each tab
lives in its own module. Everything with actual logic (`driver.py`,
`editing.py`) is streamlit-free so it stays unit-testable; the `*_tab.py`
modules are thin rendering veneers over `src/tools/organize.py`,
`src/tools/admin.py`, and `src/ingest/backfill.py`.
"""
