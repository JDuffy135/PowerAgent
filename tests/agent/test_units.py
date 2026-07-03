"""Unit conversion / presentation tests (pure functions, no models)."""
from __future__ import annotations

from src.agent.units import convert_weights, format_weight, to_display_weight


def test_lb_passthrough_rounds_but_keeps_value():
    assert to_display_weight(315.0, "lb") == 315.0
    assert to_display_weight(None, "lb") is None


def test_kg_conversion():
    assert to_display_weight(315.0, "kg") == 142.9  # 315 * 0.45359237 = 142.88...
    assert to_display_weight(100.0, "kg") == 45.4


def test_format_weight_drops_trailing_zero():
    assert format_weight(225.0, "lb") == "225 lb"
    assert format_weight(142.88, "kg") == "64.8 kg"
    assert format_weight(None, "kg") == "?"


def test_convert_weights_renames_lb_keys_and_converts():
    obj = {"weight_lb": 100.0, "reps": 5, "e1rm": 100.0, "date": "2026-01-01"}
    out = convert_weights(obj, "kg")
    assert "weight_lb" not in out
    assert out["weight"] == 45.4       # renamed + converted
    assert out["e1rm"] == 45.4          # weight key, converted, keeps its name
    assert out["reps"] == 5             # non-weight untouched
    assert out["date"] == "2026-01-01"


def test_convert_weights_lb_unit_is_identity_shape():
    obj = {"target_weight_lb": 305.0, "nested": [{"tonnage_lb": 1000.0}]}
    out = convert_weights(obj, "lb")
    assert out == {"target_weight": 305.0, "nested": [{"tonnage": 1000.0}]}


def test_convert_weights_leaves_bools_alone():
    # is_top_set etc. are ints-as-bools in Python; must never be treated as weight.
    out = convert_weights({"is_top_set_lb": True}, "kg")
    assert out["is_top_set_lb"] is True
