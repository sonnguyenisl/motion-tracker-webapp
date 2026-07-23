"""Central configuration for the Aximove web app.

All filesystem paths are anchored to this file's directory so the app behaves
the same no matter what the current working directory is when it launches.
"""

import os

try:
    from dotenv import load_dotenv
    # Load environment variables from a local .env file (e.g. OPENROUTER_API_KEY)
    # before anything reads os.environ. Safe no-op if the file is absent.
    load_dotenv()
except ImportError:
    # python-dotenv not installed — rely on real environment variables.
    pass

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    BASE_DIR = BASE_DIR

    # --- Flask ---
    SECRET_KEY = os.environ.get("AXIMOVE_SECRET", "aximove-dev-secret")
    MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB uploads/recordings

    # --- Storage locations ---
    UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
    RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
    RECORDINGS_RAW_DIR = os.path.join(RECORDINGS_DIR, "raw")
    RECORDINGS_SKELETON_DIR = os.path.join(RECORDINGS_DIR, "skeleton")
    DB_PATH = os.path.join(BASE_DIR, "app.db")

    # --- Uploads ---
    ALLOWED_VIDEO_EXT = {"mp4", "mov", "avi", "mkv", "webm"}
    ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "webp"}

    # Max allowed length for any scored clip (reference, uploaded user video, or
    # live recording). Enforced server-side; the client mirrors this to cap the
    # trimmer window and auto-stop recording. Exposed to templates as
    # ``config.MAX_VIDEO_SECONDS``.
    MAX_VIDEO_SECONDS = 20

    # --- SocketIO ---
    # "threading" keeps deployment simple (no eventlet/gevent monkey-patching
    # required) which plays well with the blocking OpenCV/MediaPipe pipeline.
    SOCKETIO_ASYNC_MODE = "threading"

    @classmethod
    def ensure_dirs(cls):
        for path in (
            cls.UPLOADS_DIR,
            cls.RECORDINGS_DIR,
            cls.RECORDINGS_RAW_DIR,
            cls.RECORDINGS_SKELETON_DIR,
        ):
            os.makedirs(path, exist_ok=True)
