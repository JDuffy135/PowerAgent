"""Seed script: builds data/training.db with realistic sample training data.

Idempotent via wipe-and-reload: all tables are cleared before inserting, so
running this script multiple times always produces the same result.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.connection import get_conn, init_db
from src.tools.resolve import add_exercise

DB_PATH = Path(__file__).parent.parent / "data" / "training.db"

TABLES_IN_DELETE_ORDER = [
    "pr",
    "cardio",
    "lift_set",
    "programmed_slot",
    "session",
    "measurement",
    "injury",
    "bodyweight",
    "exercise_alias",
    "exercise",
    "block",
    "program",
    "ingest_batch",
]


def wipe(conn) -> None:
    for table in TABLES_IN_DELETE_ORDER:
        conn.execute(f"DELETE FROM {table}")
    conn.commit()


def seed(conn) -> None:
    wipe(conn)

    # --- Program & blocks -------------------------------------------------
    program_id = conn.execute(
        """
        INSERT INTO program (name, status, start_date, end_date, goals_text)
        VALUES (?, 'incomplete', '2025-12-27', NULL, 'Build toward Meet 1: raise total, keep joints healthy.')
        """,
        ("2026 Meet 1 Prep",),
    ).lastrowid

    hypertrophy_block_id = conn.execute(
        """
        INSERT INTO block (program_id, name, focus, week_count, start_date, end_date)
        VALUES (?, 'Hypertrophy Phase 1', 'hypertrophy', 13, '2025-12-27', '2026-03-29')
        """,
        (program_id,),
    ).lastrowid

    strength_block_id = conn.execute(
        """
        INSERT INTO block (program_id, name, focus, week_count, start_date, end_date)
        VALUES (?, 'Strength Block 1', 'strength', 4, '2026-03-30', '2026-04-26')
        """,
        (program_id,),
    ).lastrowid

    peaking_block_id = conn.execute(
        """
        INSERT INTO block (program_id, name, focus, week_count, start_date, end_date)
        VALUES (?, 'Peaking Block', 'peaking', 4, '2026-05-25', '2026-06-21')
        """,
        (program_id,),
    ).lastrowid

    # --- Exercises ----------------------------------------------------------
    ex = {}
    ex["bench"] = add_exercise(
        conn, "Bench Press", "competition", "chest",
        ["bench press", "competition bench", "comp bench"],
    )
    ex["squat"] = add_exercise(
        conn, "Low Bar Squat", "competition", "posterior chain",
        ["low bar squat", "squat", "competition squat"],
    )
    ex["deadlift"] = add_exercise(
        conn, "Deadlift", "competition", "posterior chain",
        ["deadlift", "deadlifts", "competition deadlift"],
    )
    ex["paused_squat"] = add_exercise(
        conn, "Paused Low Bar Squat", "variation", "posterior chain",
        ["paused low bar squats", "pause squat"],
    )
    ex["cgb"] = add_exercise(
        conn, "Close Grip Bench", "variation", "triceps",
        ["close grip bench"],
    )
    ex["pullups"] = add_exercise(
        conn, "Weighted Pullups", "accessory", "upper back",
        ["weighted pullups"],
    )
    ex["pulldowns"] = add_exercise(
        conn, "MAG Grip Pulldowns", "accessory", "upper back",
        ["mag grip pulldowns"],
    )
    ex["oh_tricep"] = add_exercise(
        conn, "Standing Overhead Tricep Extensions", "accessory", "triceps",
        [],
    )
    ex["leg_curls"] = add_exercise(
        conn, "Plate Loaded Leg Curls", "accessory", "hamstrings",
        [],
    )
    ex["leg_ext"] = add_exercise(
        conn, "Leg Extensions", "accessory", "quads",
        [],
    )

    def add_session(date, block_id, week_number, day_number, day_label,
                     session_type="lifting", duration_min=None, raw_note=None):
        return conn.execute(
            """
            INSERT INTO session (date, block_id, week_number, day_number, day_label,
                                  duration_min, session_type, raw_note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (date, block_id, week_number, day_number, day_label, duration_min,
             session_type, raw_note),
        ).lastrowid

    def add_set(session_id, exercise_id, set_index, weight_lb, reps, rpe=None,
                is_top_set=0, raw_text=None):
        conn.execute(
            """
            INSERT INTO lift_set (session_id, exercise_id, set_index, weight_lb, reps,
                                   rpe, is_top_set, raw_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, exercise_id, set_index, weight_lb, reps, rpe, is_top_set, raw_text),
        )

    # --- Sessions -------------------------------------------------------

    # S1: 2026-01-10 - paused squat + leg extensions (Hypertrophy)
    s1 = add_session("2026-01-10", hypertrophy_block_id, 3, 6, "w3d6")
    for i in range(5):
        add_set(s1, ex["paused_squat"], i + 1, 185, 5, raw_text="5x5 @ 185")
    for i in range(3):
        add_set(s1, ex["leg_ext"], i + 1, 90, 12, raw_text="3x12 @ 90")

    # S2: 2026-02-07 - close grip bench + standing OH tricep ext (Hypertrophy)
    s2 = add_session("2026-02-07", hypertrophy_block_id, 7, 6, "w7d6")
    for i in range(4):
        add_set(s2, ex["cgb"], i + 1, 155, 6, raw_text="4x6 @ 155")
    for i in range(3):
        add_set(s2, ex["oh_tricep"], i + 1, 40, 12, raw_text="3x12 @ 40")

    # S3: 2026-03-05 - bench, top set 225x1 (Hypertrophy)
    s3 = add_session("2026-03-05", hypertrophy_block_id, 10, 3, "w10d3")
    add_set(s3, ex["bench"], 1, 225, 1, is_top_set=1, raw_text="225x1")
    for i in range(3):
        add_set(s3, ex["bench"], i + 2, 205, 3, raw_text="205x3,3,3")

    # S4: 2026-03-19 - bench, top set 230x1 (Hypertrophy) -- March-best anchor
    s4 = add_session("2026-03-19", hypertrophy_block_id, 12, 3, "w12d3")
    add_set(s4, ex["bench"], 1, 230, 1, is_top_set=1, raw_text="230x1")
    for i in range(3):
        add_set(s4, ex["bench"], i + 2, 205, 3, raw_text="205x3,3,3")

    # S5: 2026-04-02 - deadlift + weighted pullups (Strength Block 1)
    s5 = add_session("2026-04-02", strength_block_id, 1, 4, "w1d4")
    add_set(s5, ex["deadlift"], 1, 335, 2, is_top_set=1, raw_text="335x2")
    for i in range(3):
        add_set(s5, ex["deadlift"], i + 2, 285, 4, raw_text="285x4,4,4")
    for i in range(3):
        add_set(s5, ex["pullups"], i + 1, 45, 8, raw_text="3x8 @ +45")

    # S6: 2026-04-16 - squat (Strength Block 1)
    s6 = add_session("2026-04-16", strength_block_id, 3, 4, "w3d4")
    add_set(s6, ex["squat"], 1, 295, 3, is_top_set=1, raw_text="295x3")
    for i in range(4):
        add_set(s6, ex["squat"], i + 2, 255, 4, raw_text="255x4,4,4,4")

    # S7: 2026-05-27 - squat, top single 315x1 + 6 backoff sets of 4 @ 255 (Peaking)
    s7 = add_session("2026-05-27", peaking_block_id, 1, 1, "w1d1")
    add_set(s7, ex["squat"], 1, 315, 1, is_top_set=1, raw_text="315x1")
    for i in range(6):
        add_set(s7, ex["squat"], i + 2, 255, 4, raw_text="6x4 @ 255")

    # S8: 2026-05-30 - cardio (bike, ~26 min) (Peaking)
    s8 = add_session(
        "2026-05-30", peaking_block_id, 1, 4, "CARDIO",
        session_type="cardio", duration_min=26,
    )
    conn.execute(
        """
        INSERT INTO cardio (session_id, modality, distance_mi, duration_min, intensity, raw_text)
        VALUES (?, 'bike', NULL, 26, 'light', 'bike ~26 min')
        """,
        (s8,),
    )

    # S9: 2026-06-01 - deadlift + MAG Grip Pulldowns (Peaking), week 2 day 'w2d1'
    s9 = add_session("2026-06-01", peaking_block_id, 2, 1, "w2d1")
    add_set(s9, ex["deadlift"], 1, 385, 1, is_top_set=1, raw_text="385x1")
    for i in range(4):
        add_set(s9, ex["deadlift"], i + 2, 315, 3, raw_text="315x3,3,3,3")
    add_set(s9, ex["pulldowns"], 1, 143, 14, raw_text="14 @ 143")
    add_set(s9, ex["pulldowns"], 2, 121, 17, raw_text="17 @ 121 (pin dropped from 143 after set 1)")
    add_set(s9, ex["pulldowns"], 3, 121, 15, raw_text="15 @ 121")

    # S10: 2026-06-08 - bench + plate loaded leg curls (Peaking)
    s10 = add_session("2026-06-08", peaking_block_id, 3, 1, "w3d1")
    add_set(s10, ex["bench"], 1, 240, 1, is_top_set=1, raw_text="240x1")
    for i in range(3):
        add_set(s10, ex["bench"], i + 2, 205, 3, raw_text="205x3,3,3")
    for i in range(3):
        add_set(s10, ex["leg_curls"], i + 1, 70, 12, raw_text="3x12 @ 70")

    # --- Bodyweight (weekly-ish, 138.0 -> 146.0) --------------------------
    bw_rows = [
        ("2026-01-03", 138.0),
        ("2026-01-24", 139.0),
        ("2026-02-14", 140.0),
        ("2026-03-07", 141.0),
        ("2026-03-28", 142.0),
        ("2026-04-18", 143.0),
        ("2026-05-09", 144.0),
        ("2026-05-23", 145.0),
        ("2026-06-01", 146.0),
    ]
    for date, weight in bw_rows:
        conn.execute(
            "INSERT INTO bodyweight (date, weight_lb) VALUES (?, ?)", (date, weight)
        )

    # --- PR ----------------------------------------------------------------
    conn.execute(
        """
        INSERT INTO pr (date, session_id, exercise_id, weight_lb, reps, context)
        VALUES ('2026-06-01', ?, ?, 385, 1, 'gym')
        """,
        (s9, ex["deadlift"]),
    )

    # --- Injury -------------------------------------------------------------
    conn.execute(
        """
        INSERT INTO injury (start_date, end_date, area, severity, note)
        VALUES ('2026-02-15', '2026-03-10', 'right knee', 'niggle', 'Slight ache on deep squats, resolved with rest.')
        """
    )

    conn.commit()


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = get_conn(DB_PATH)
    init_db(conn)
    seed(conn)
    conn.close()
    print(f"Seeded {DB_PATH}")


if __name__ == "__main__":
    main()
