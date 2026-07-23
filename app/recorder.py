"""Recording persistence + the post-recording processing pipeline.

The browser records locally and uploads the finished clip. This module saves
it to ``recordings/raw`` and then (in a background task) runs the motion_ml
pipeline: extract user pose -> DTW score vs the reference -> build feedback ->
burn the skeleton overlay into ``recordings/skeleton`` -> persist everything.
Progress is streamed to the client over SocketIO.
"""

import os
import json
import gc
import ctypes
import platform

from . import db
from .utils import unique_name, resolve_under, trim_video, get_video_duration
from .session import load_reference_vectors, clear_cache as clear_vector_cache

from motion_ml.pose_estimator import get_form_data
from motion_ml.similarity import evaluate_motion, print_feedback
from motion_ml.feedback import build_feedback
from motion_ml.skeleton_renderer import burn_skeleton
from motion_ml.coach import generate_coaching


def _trim_heap():
    """Try to return freed heap pages to the OS via platform allocator hooks.

    On Linux (glibc) calls ``malloc_trim(0)``, on Windows calls ``_heapmin()``.
    Safe no-op on other platforms or if the C library can't be loaded.
    """
    try:
        if platform.system() == "Windows":
            ctypes.windll.msvcrt._heapmin()
        else:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def save_recording(file_storage, user_id, reference_id, raw_dir):
    """Save the uploaded clip to recordings/raw and open a session row.

    Returns the new session id. Runs inside a request context.
    """
    name = unique_name(file_storage.filename or "recording.webm", prefix="rec_")
    raw_path = os.path.join(raw_dir, name)
    file_storage.save(raw_path)

    # Store the name only (not the absolute path) so the DB stays portable.
    conn = db.get_db()
    cur = conn.execute(
        "INSERT INTO sessions (user_id, reference_id, raw_path, status, created_at) "
        "VALUES (?, ?, ?, 'processing', ?)",
        (user_id, reference_id, name, db.now()),
    )
    conn.commit()
    return cur.lastrowid


def _emit(socketio, room, stage, fraction, message=""):
    socketio.emit(
        "progress",
        {"stage": stage, "fraction": round(fraction, 3), "message": message},
        to=room,
    )


def process_session(app, socketio, session_id, room, trim_start=None, trim_end=None):
    """Background task: score the recording and persist results.

    Uses a standalone DB connection because it runs outside any request.
    Emits 'progress' updates and a final 'done' (or 'error') to ``room``.
    """
    with app.app_context():
        conn = db.standalone_connection(app.config["DB_PATH"])
        try:
            session_row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if session_row is None:
                socketio.emit("error", {"message": "Session not found."}, to=room)
                return

            # Resolve the stored name to an absolute path on this machine.
            raw_path = resolve_under(
                app.config["RECORDINGS_RAW_DIR"], session_row["raw_path"]
            )
            reference_id = session_row["reference_id"]

            # Trim the recording if requested
            if trim_start is not None and trim_end is not None and trim_end > trim_start:
                _emit(socketio, room, "trim", 0.03, "Trimming recording…")
                base = os.path.basename(raw_path)
                trimmed_name = os.path.splitext(base)[0] + "_trimmed" + os.path.splitext(base)[1]
                trimmed_path = os.path.join(app.config["RECORDINGS_RAW_DIR"], trimmed_name)
                try:
                    # trim_video may change the extension (e.g. .webm -> .mp4),
                    # so use the path it actually wrote — the only file we keep.
                    out_path = trim_video(raw_path, trimmed_path, trim_start, trim_end)
                    if os.path.abspath(out_path) != os.path.abspath(raw_path):
                        os.remove(raw_path)  # drop the full original
                    raw_path = out_path
                    # Update the session's raw_path to the trimmed file (by name)
                    conn.execute(
                        "UPDATE sessions SET raw_path = ? WHERE id = ?",
                        (os.path.basename(out_path), session_id),
                    )
                    conn.commit()
                except Exception as exc:  # noqa: BLE001
                    # If trimming fails, use the original
                    pass

            # Enforce the max clip length on the final (possibly trimmed) clip.
            # Prefer the trim window (always reliable); fall back to measuring
            # the file (live .webm recordings are auto-capped client-side, so a
            # 0/unknown measurement there is fine to let through).
            max_secs = app.config["MAX_VIDEO_SECONDS"]
            if trim_start is not None and trim_end is not None and trim_end > trim_start:
                eff_secs = trim_end - trim_start
            else:
                eff_secs = get_video_duration(raw_path)
            if eff_secs > max_secs + 0.5:
                raise ValueError(
                    f"Videos must be {max_secs} seconds or less — yours is about "
                    f"{eff_secs:.0f}s. Please trim it down and try again."
                )

            # 1. Reference vectors -------------------------------------------
            _emit(socketio, room, "reference", 0.05, "Loading reference…")
            ref_row = conn.execute(
                "SELECT * FROM reference_files WHERE id = ?", (reference_id,)
            ).fetchone()
            if ref_row is None or not ref_row["vector_path"]:
                raise ValueError("Selected reference has no analysed pose data.")
            ref_vectors = load_reference_vectors(ref_row["vector_path"])
            exercise_label = ref_row["label"] if "label" in ref_row.keys() else None

            # 2. Extract user pose -------------------------------------------
            _emit(socketio, room, "analyze", 0.1, "Analysing your movement…")
            user = get_form_data(
                raw_path,
                with_landmarks=True,
                return_landmarks=True,
                progress=lambda f: _emit(
                    socketio, room, "analyze", 0.1 + f * 0.4,
                    "Analysing your movement…",
                ),
            )

            # Extract data from user dict and free it early
            user_vectors = user["vectors"]
            user_landmarks = user.pop("landmarks", None)
            del user

            # 3. Score --------------------------------------------------------
            _emit(socketio, room, "score", 0.55, "Comparing to reference…")
            evaluation = evaluate_motion(ref_vectors, user_vectors)
            # Console dump + capture of the signed angle/landmark/velocity diffs.
            # Reuse the DTW path from evaluate_motion to avoid recomputing DTW.
            dtw_path = evaluation.get("_path")
            fb = print_feedback(ref_vectors, user_vectors, label=exercise_label, path=dtw_path)
            # Free vectors after all DTW work is done
            del ref_vectors, user_vectors, dtw_path
            gc.collect(2)
            _trim_heap()
            report = build_feedback(evaluation)

            # 3b. AI coaching (optional; falls back to per-joint feedback) ----
            _emit(socketio, room, "coach", 0.58, "Generating AI coaching…")
            # Strip the internal _path key before passing evaluation further
            eval_for_coach = {k: v for k, v in evaluation.items() if k != "_path"}
            ai_report = generate_coaching(exercise_label, fb, eval_for_coach)
            ai_json = json.dumps(ai_report) if ai_report else None

            # 4. Burn skeleton overlay ---------------------------------------
            _emit(socketio, room, "skeleton", 0.6, "Rendering skeleton overlay…")
            skel_name = os.path.splitext(os.path.basename(raw_path))[0] + ".mp4"
            skel_path = os.path.join(
                app.config["RECORDINGS_SKELETON_DIR"], skel_name
            )
            burn_skeleton(
                raw_path, skel_path,
                precomputed_landmarks=user_landmarks,
                progress=lambda f: _emit(
                    socketio, room, "skeleton", 0.6 + f * 0.35,
                    "Rendering skeleton overlay…",
                ),
            )
            del user_landmarks
            gc.collect(2)
            _trim_heap()

            # 5. Persist ------------------------------------------------------
            # Store the skeleton by name only (portable across machines).
            _emit(socketio, room, "save", 0.97, "Saving results…")
            conn.execute(
                "UPDATE sessions SET skeleton_path = ?, ai_feedback = ?, "
                "status = 'done' WHERE id = ?",
                (skel_name, ai_json, session_id),
            )
            conn.execute(
                "INSERT INTO scores (session_id, overall_score, angle_score, "
                "landmark_score, dtw_distance) VALUES (?, ?, ?, ?, ?)",
                (
                    session_id,
                    evaluation["overall_score"],
                    evaluation["angle_score"],
                    evaluation["landmark_score"],
                    evaluation["dtw_distance"],
                ),
            )
            for joint in report["joints"]:
                conn.execute(
                    "INSERT INTO feedback (session_id, joint_name, error_deg, "
                    "signed_deg, tip) VALUES (?, ?, ?, ?, ?)",
                    (session_id, joint["name"], joint["error"],
                     joint["signed"], joint["tip"]),
                )
            conn.commit()

            _emit(socketio, room, "done", 1.0, "Done!")
            socketio.emit(
                "done",
                {"session_id": session_id, "redirect": f"/result/{session_id}"},
                to=room,
            )
        except Exception as exc:  # noqa: BLE001 - surface any failure to the user
            conn.execute(
                "UPDATE sessions SET status = 'error' WHERE id = ?", (session_id,)
            )
            conn.commit()
            socketio.emit("error", {"message": str(exc)}, to=room)
        finally:
            clear_vector_cache()
            conn.close()
            gc.collect(2)
            _trim_heap()
