"""SQLite access layer.

Schema (app.db)::

    users       (id, username, password_hash, contact, created_at)
    references  (id, user_id, filename, original_name, label, kind, vector_path, created_at)
    sessions    (id, user_id, reference_id, raw_path, skeleton_path, status, created_at)
    scores      (id, session_id, overall_score, angle_score, landmark_score, dtw_distance)
    feedback    (id, session_id, joint_name, error_deg, signed_deg, tip)
    gallery_items  (id, reference_id, user_id, description, upvotes, downvotes, created_at)
    gallery_votes  (id, user_id, gallery_item_id, vote, UNIQUE(user_id, gallery_item_id))
    leaderboard_entries (id, gallery_item_id, session_id, user_id, overall_score, created_at)

``references`` are uploaded reference files; ``sessions`` are recorded
exercise attempts. Both are surfaced on the dashboard. A ``leaderboard_entry``
is a recorded session a user chose to publish onto a gallery item's leaderboard.
"""

import os
import sqlite3
from datetime import datetime

from flask import current_app, g

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    contact       TEXT,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reference_files (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL,
    filename      TEXT NOT NULL,
    original_name TEXT,
    label         TEXT,                    -- user-given exercise name
    kind          TEXT NOT NULL,           -- 'video' | 'image'
    vector_path   TEXT,                    -- .npy of extracted feature vectors
    pose_json     TEXT,                    -- first-frame tracked landmarks [[x,y],...]
    created_at    TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL,
    reference_id  INTEGER,
    raw_path      TEXT,
    skeleton_path TEXT,
    ai_feedback   TEXT,                    -- JSON coaching report from the LLM
    status        TEXT NOT NULL DEFAULT 'processing',
    created_at    TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (reference_id) REFERENCES reference_files(id)
);

CREATE TABLE IF NOT EXISTS scores (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     INTEGER NOT NULL,
    overall_score  REAL,
    angle_score    REAL,
    landmark_score REAL,
    dtw_distance   REAL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL,
    joint_name  TEXT,
    error_deg   REAL,
    signed_deg  REAL,
    tip         TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS gallery_items (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    reference_id  INTEGER NOT NULL,
    user_id       INTEGER NOT NULL,
    description   TEXT,
    upvotes       INTEGER NOT NULL DEFAULT 0,
    downvotes     INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    FOREIGN KEY (reference_id) REFERENCES reference_files(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS gallery_votes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL,
    gallery_item_id INTEGER NOT NULL,
    vote            INTEGER NOT NULL,      -- 1 = upvote, -1 = downvote
    UNIQUE(user_id, gallery_item_id),
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (gallery_item_id) REFERENCES gallery_items(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS gallery_comments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    gallery_item_id INTEGER NOT NULL,
    user_id         INTEGER NOT NULL,
    body            TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (gallery_item_id) REFERENCES gallery_items(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS leaderboard_entries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    gallery_item_id INTEGER NOT NULL,      -- which public reference's leaderboard
    session_id      INTEGER NOT NULL,      -- the published recording (score+feedback+video)
    user_id         INTEGER NOT NULL,      -- who published it
    overall_score   REAL,                  -- denormalised for fast ranking
    created_at      TEXT NOT NULL,
    UNIQUE(session_id),                    -- a session can be on a board at most once
    FOREIGN KEY (gallery_item_id) REFERENCES gallery_items(id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
"""


def get_db():
    """Connection bound to the request/app context (one per request)."""
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DB_PATH"])
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


def close_db(exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def _migrate(db):
    """Idempotent column additions for pre-existing databases."""
    ref_cols = [row[1] for row in db.execute("PRAGMA table_info(reference_files)")]
    if "label" not in ref_cols:
        db.execute("ALTER TABLE reference_files ADD COLUMN label TEXT")
    if "pose_json" not in ref_cols:
        db.execute("ALTER TABLE reference_files ADD COLUMN pose_json TEXT")

    sess_cols = [row[1] for row in db.execute("PRAGMA table_info(sessions)")]
    if "ai_feedback" not in sess_cols:
        db.execute("ALTER TABLE sessions ADD COLUMN ai_feedback TEXT")


def init_db(app):
    """Create tables if missing and register teardown. Idempotent."""
    with app.app_context():
        db = sqlite3.connect(app.config["DB_PATH"])
        db.executescript(SCHEMA)
        _migrate(db)
        db.commit()
        db.close()
    app.teardown_appcontext(close_db)


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# --- A connection usable outside a request (background processing threads) ---
def standalone_connection(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn