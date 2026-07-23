"""Active-reference state for the current user.

Stores which reference the user has selected for their next recording in the
Flask session, and loads/caches that reference's extracted feature vectors so
the scoring stage doesn't re-process the reference video every time.
"""

import os
import threading

from flask import session as flask_session

from config import Config
from . import db
from .utils import load_vectors

# Process-local cache: reference_id -> numpy feature array.
# Capped at 2 entries to prevent unbounded memory growth on long-lived servers.
# Older entries are evicted when the cache exceeds this limit.
_VECTOR_CACHE = {}
_VECTOR_CACHE_LOCK = threading.Lock()
_VECTOR_CACHE_MAX = 2

ACTIVE_KEY = "active_reference_id"


def set_active_reference(reference_id):
    flask_session[ACTIVE_KEY] = reference_id


def get_active_reference_id():
    return flask_session.get(ACTIVE_KEY)


def get_reference_row(reference_id, user_id):
    """Fetch a reference row owned by the user, or None."""
    cur = db.get_db().execute(
        "SELECT * FROM reference_files WHERE id = ? AND user_id = ?",
        (reference_id, user_id),
    )
    return cur.fetchone()


def get_accessible_reference_row(reference_id, user_id):
    """Fetch a reference the user is allowed to record against, or None.

    That means one they own *or* one published to the public gallery (so a
    "Use This" from the gallery can score against another user's reference).
    """
    cur = db.get_db().execute(
        """
        SELECT r.* FROM reference_files r
        WHERE r.id = ?
          AND (r.user_id = ?
               OR EXISTS (SELECT 1 FROM gallery_items g
                          WHERE g.reference_id = r.id))
        """,
        (reference_id, user_id),
    )
    return cur.fetchone()


def load_reference_vectors(vector_path):
    """Return cached feature vectors for a reference, loading from disk once.

    ``vector_path`` is the name stored in the DB (legacy absolute paths are
    tolerated via basename); it's resolved against the uploads dir on this
    machine so the data stays portable across checkouts.

    The cache is capped at ``_VECTOR_CACHE_MAX`` entries. When it exceeds the
    limit the *oldest* entry is evicted first (FIFO), keeping memory bounded.
    Thread-safe via ``_VECTOR_CACHE_LOCK``.
    """
    key = os.path.basename(vector_path)
    with _VECTOR_CACHE_LOCK:
        if key in _VECTOR_CACHE:
            return _VECTOR_CACHE[key]

        # Evict the oldest entry when the cache is full.
        while len(_VECTOR_CACHE) >= _VECTOR_CACHE_MAX:
            oldest = next(iter(_VECTOR_CACHE))
            del _VECTOR_CACHE[oldest]

        vectors = load_vectors(os.path.join(Config.UPLOADS_DIR, key))
        _VECTOR_CACHE[key] = vectors
        return vectors


def invalidate(vector_path):
    with _VECTOR_CACHE_LOCK:
        _VECTOR_CACHE.pop(os.path.basename(vector_path), None)


def clear_cache():
    """Drop all cached reference vectors immediately.

    Call this after a background scoring task completes so the vectors don't
    occupy memory between requests.
    """
    with _VECTOR_CACHE_LOCK:
        _VECTOR_CACHE.clear()
