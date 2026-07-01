from src.tools.resolve import add_exercise, resolve_exercise


def test_exact_alias_hit(conn):
    resolved = resolve_exercise(conn, "bench press")
    assert resolved is not None
    assert resolved.name == "Bench Press"
    assert resolved.matched_via == "exact"


def test_fuzzy_hit_typo(conn):
    resolved = resolve_exercise(conn, "competiton bench")
    assert resolved is not None
    assert resolved.name == "Bench Press"
    assert resolved.matched_via == "fuzzy"


def test_miss_returns_none(conn):
    resolved = resolve_exercise(conn, "underwater basket weaving")
    assert resolved is None


def test_ambiguity_does_not_false_positive(conn):
    # Two near-identical aliases pointing at different exercises: a fuzzy
    # query close to both should NOT silently pick one.
    id_a = add_exercise(conn, "Exercise Alpha", "accessory", None, ["squat variant one"])
    id_b = add_exercise(conn, "Exercise Beta", "accessory", None, ["squat variant two"])
    assert id_a != id_b

    resolved = resolve_exercise(conn, "squat variant three")
    assert resolved is None
