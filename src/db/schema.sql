-- Programs = macrocycle-level containers (e.g. "2026 Meet 1 Prep")
CREATE TABLE IF NOT EXISTS program (
    program_id   INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('complete','incomplete','draft')),
    start_date   TEXT,            -- NULL for drafts
    end_date     TEXT,            -- NULL until complete
    goals_text   TEXT,            -- "MAIN GOALS OF THE PROGRAM" prose
    review_text  TEXT,            -- macrocycle review prose (also embedded in Chroma)
    notes        TEXT
);

-- Blocks = mesocycles within a program (hypertrophy block, strength block 1, peaking block...)
CREATE TABLE IF NOT EXISTS block (
    block_id     INTEGER PRIMARY KEY,
    program_id   INTEGER NOT NULL REFERENCES program(program_id),
    name         TEXT NOT NULL,
    focus        TEXT,            -- 'hypertrophy' | 'strength' | 'peaking' | ...
    week_count   INTEGER,
    start_date   TEXT,
    end_date     TEXT,
    review_text  TEXT             -- block review prose (also embedded in Chroma)
);

-- Exercise dictionary. Canonical names solve the "MAG GRIP PULLDOWNS" vs
-- "mag grip pulldown" problem; aliases map raw log strings to canonical IDs.
CREATE TABLE IF NOT EXISTS exercise (
    exercise_id  INTEGER PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,      -- canonical, e.g. 'Low Bar Squat'
    tier         TEXT NOT NULL CHECK (tier IN ('competition','variation','accessory')),
    muscle_group TEXT CHECK (muscle_group IN (
                    'chest','triceps','upper back','lower back','biceps','core',
                    'front deltoids','side deltoids','rear deltoids','glutes',
                    'adductors','abductors','quads','hamstrings','calves',
                    'posterior chain')),
    equipment_note TEXT                      -- pin settings, seat heights, etc. (default setup)
);

CREATE TABLE IF NOT EXISTS exercise_alias (
    alias        TEXT PRIMARY KEY,           -- raw string as it appears in logs, lowercased
    exercise_id  INTEGER NOT NULL REFERENCES exercise(exercise_id)
);

-- A session = one workout (or cardio day)
CREATE TABLE IF NOT EXISTS session (
    session_id   INTEGER PRIMARY KEY,
    date         TEXT NOT NULL,
    block_id     INTEGER REFERENCES block(block_id),   -- NULL allowed (unattached logs)
    week_number  INTEGER,                    -- within block, if known
    day_number   INTEGER,                    -- within week, if known (w1d3 -> 3)
    day_label    TEXT,                       -- raw text as logged: 'w2d1', 'CARDIO', etc.
    duration_min INTEGER,
    session_type TEXT NOT NULL DEFAULT 'lifting'
                 CHECK (session_type IN ('lifting','cardio','other')),
    raw_note     TEXT                        -- full original log text for this session
);

-- Uniform set-level grain for ALL exercises (competition + accessories).
CREATE TABLE IF NOT EXISTS lift_set (
    set_id       INTEGER PRIMARY KEY,
    session_id   INTEGER NOT NULL REFERENCES session(session_id),
    exercise_id  INTEGER NOT NULL REFERENCES exercise(exercise_id),
    set_index    INTEGER NOT NULL,           -- 1-based order within the exercise
    weight_lb    REAL,                       -- canonical lb; NULL for bodyweight-only.
    reps         INTEGER,
    rpe          REAL,                       -- NULL if not recorded
    is_paused    INTEGER NOT NULL DEFAULT 0,
    is_amrap     INTEGER NOT NULL DEFAULT 0,
    is_top_set   INTEGER NOT NULL DEFAULT 0, -- heavy single/double before backoffs
    is_failed    INTEGER NOT NULL DEFAULT 0,
    raw_text     TEXT                        -- original substring, incl. any kg notation
);

-- Programmed (planned) work, kept separate from performed work so
-- "projected vs actual" comparisons are a JOIN, not a parsing problem.
CREATE TABLE IF NOT EXISTS programmed_slot (
    slot_id      INTEGER PRIMARY KEY,
    block_id     INTEGER NOT NULL REFERENCES block(block_id),
    week_number  INTEGER,
    day_number   INTEGER,                    -- within week, if known
    day_label    TEXT,                       -- raw text: 'MONDAY', 'w1d3', etc.
    exercise_id  INTEGER REFERENCES exercise(exercise_id),
    prescription TEXT NOT NULL,              -- '1x3 @ RPE 7, 4x4 @ RPE 7-8'
    target_weight_lb REAL,                   -- projected weight if specified
    notes        TEXT
);

-- Cardio kept separate from lift_set (different shape entirely)
CREATE TABLE IF NOT EXISTS cardio (
    cardio_id    INTEGER PRIMARY KEY,
    session_id   INTEGER NOT NULL REFERENCES session(session_id),
    modality     TEXT,                       -- 'bike', 'run', ...
    distance_mi  REAL,
    duration_min REAL,
    intensity    TEXT,                       -- 'light', 'moderate', ...
    raw_text     TEXT
);

CREATE TABLE IF NOT EXISTS bodyweight (
    bw_id        INTEGER PRIMARY KEY,
    date         TEXT NOT NULL,
    weight_lb    REAL NOT NULL,
    note         TEXT
);

CREATE TABLE IF NOT EXISTS pr (
    pr_id        INTEGER PRIMARY KEY,
    date         TEXT NOT NULL,
    session_id   INTEGER REFERENCES session(session_id),  -- session where the PR was hit
    exercise_id  INTEGER NOT NULL REFERENCES exercise(exercise_id),
    weight_lb    REAL NOT NULL,
    reps         INTEGER NOT NULL,           -- 1 for a true 1RM PR
    context      TEXT                        -- 'gym', 'mock meet', 'meet', notes
);

CREATE TABLE IF NOT EXISTS injury (
    injury_id    INTEGER PRIMARY KEY,
    start_date   TEXT NOT NULL,
    end_date     TEXT,                       -- NULL = ongoing
    area         TEXT NOT NULL,              -- 'right knee', 'hip', ...
    severity     TEXT,                       -- 'niggle', 'moderate', 'serious'
    note         TEXT
);

CREATE TABLE IF NOT EXISTS measurement (
    m_id         INTEGER PRIMARY KEY,
    date         TEXT NOT NULL,
    site         TEXT NOT NULL,              -- 'arm', 'femur length', ...
    value_in     REAL NOT NULL,
    note         TEXT
);

-- Ingestion audit trail: every upload gets a record; parsed payloads reference it.
CREATE TABLE IF NOT EXISTS ingest_batch (
    batch_id     INTEGER PRIMARY KEY,
    created_at   TEXT NOT NULL,
    source_file  TEXT,
    status       TEXT NOT NULL CHECK (status IN ('pending_review','committed','rejected')),
    parsed_json  TEXT                        -- the JSON shown to the user at review time
);

CREATE INDEX IF NOT EXISTS idx_lift_set_exercise_session ON lift_set(exercise_id, session_id);
CREATE INDEX IF NOT EXISTS idx_session_date ON session(date);
CREATE INDEX IF NOT EXISTS idx_session_block ON session(block_id);
CREATE INDEX IF NOT EXISTS idx_bodyweight_date ON bodyweight(date);
CREATE INDEX IF NOT EXISTS idx_pr_exercise_date ON pr(exercise_id, date);
