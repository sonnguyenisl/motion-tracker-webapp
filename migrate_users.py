"""One-time migration of the legacy users.json into app.db.

The old prototype stored accounts (and their password hashes) in users.json.
This copies any users that don't already exist in the SQLite users table.
Run once after setting up:  python migrate_users.py
"""

import json
import os
import sqlite3
from datetime import datetime

from config import Config

USERS_JSON = os.path.join(Config.BASE_DIR, "users.json")


def main():
    if not os.path.exists(USERS_JSON):
        print("No users.json found — nothing to migrate.")
        return

    with open(USERS_JSON, "r", encoding="utf-8") as f:
        users = json.load(f)

    # Ensure the schema exists.
    from app.db import SCHEMA
    conn = sqlite3.connect(Config.DB_PATH)
    conn.executescript(SCHEMA)

    migrated = 0
    for username, data in users.items():
        exists = conn.execute(
            "SELECT 1 FROM users WHERE username = ?", (username,)
        ).fetchone()
        if exists:
            continue
        conn.execute(
            "INSERT INTO users (username, password_hash, contact, created_at) "
            "VALUES (?, ?, ?, ?)",
            (
                username,
                data.get("password", ""),
                data.get("contact", ""),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        migrated += 1

    conn.commit()
    conn.close()
    print(f"Migrated {migrated} user(s) into {Config.DB_PATH}.")
    print("Note: old uploaded videos in users.json are not reference movements "
          "and were not imported.")


if __name__ == "__main__":
    main()
