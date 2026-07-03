"""Presentation-layer unit handling (ARCHITECTURE.md §2).

Canonical storage is pounds; a `display_unit` preference (`lb` | `kg`) controls
output formatting only. Nothing upstream of SYNTHESIZE ever converts -- these
helpers are the single conversion point, so evidence stays in lb until it is
rendered for the user.

`convert_weights` walks a JSON-ish structure (the `.model_dump()` of a tool
result) and converts every weight-bearing value. A key is treated as a weight if
it ends in `_lb` (e.g. `weight_lb`, `target_weight_lb`, `tonnage_lb`) or is
`e1rm` (Epley estimates are in weight units). Converted `_lb` keys are renamed to
drop the suffix, so `weight_lb: 315` in lb becomes `weight: 142.9` in kg and the
key never lies about its unit.

Known limitation: `BodyweightTrend`'s summary fields (`first`/`last`/`delta`/
`min`/`max`) don't carry the `_lb` suffix, so they are *not* auto-converted; the
per-row `weight_lb` values are. Bodyweight is conventionally shown in lb here and
the default `display_unit` is lb, so this is acceptable for now.
"""
from __future__ import annotations

KG_PER_LB = 0.45359237

# Keys whose numeric values are weights even though they don't end in "_lb".
_WEIGHT_KEYS = {"e1rm"}


def to_display_weight(weight_lb: float | None, unit: str) -> float | None:
    """Convert a stored (lb) weight to the display unit, rounded to 1 decimal."""
    if weight_lb is None:
        return None
    if unit == "kg":
        return round(weight_lb * KG_PER_LB, 1)
    return round(weight_lb, 1)


def format_weight(weight_lb: float | None, unit: str) -> str:
    """Render a stored (lb) weight for display, e.g. `315 lb` / `142.9 kg`.

    Whole numbers drop the trailing `.0` so `225 lb` reads naturally.
    """
    value = to_display_weight(weight_lb, unit)
    if value is None:
        return "?"
    text = f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{text} {unit}"


def convert_weights(obj, unit: str):
    """Deep-convert every weight value in `obj` to `unit` (see module docstring).

    Returns a new structure; the input is never mutated. Non-weight values pass
    through unchanged.
    """
    if isinstance(obj, dict):
        out: dict = {}
        for key, value in obj.items():
            if isinstance(value, bool):
                out[key] = value  # bools are ints in Python -- never a weight
            elif isinstance(value, (int, float)) and (key.endswith("_lb") or key in _WEIGHT_KEYS):
                new_key = key[:-3] if key.endswith("_lb") else key
                out[new_key] = to_display_weight(float(value), unit)
            else:
                out[key] = convert_weights(value, unit)
        return out
    if isinstance(obj, list):
        return [convert_weights(item, unit) for item in obj]
    return obj
