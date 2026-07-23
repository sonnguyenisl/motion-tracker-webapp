"""All HTTP routes: auth, pages, reference upload, recording intake, results."""

import os
import json
import threading
from functools import wraps

from flask import (
    Blueprint, render_template, request, session, redirect, url_for,
    send_from_directory, send_file, jsonify, current_app, abort,
)
from werkzeug.security import generate_password_hash, check_password_hash

from . import db
from . import session as ref_session
from . import recorder
from .utils import (
    classify_upload, unique_name, save_vectors, resolve_under,
    generate_captcha_text, render_captcha_image, trim_video, get_video_duration,
)
from motion_ml.pose_estimator import get_form_data, get_form_data_from_image

bp = Blueprint("main", __name__)

# Semaphore to cap concurrent background scoring tasks. With each session
# consuming ~100-200 MB for MediaPipe + OpenCV + DTW + ffmpeg, this prevents
# OOM on small servers. Adjust based on available RAM.
_BG_SEMAPHORE = threading.Semaphore(1)


# --------------------------------------------------------------------------
# Error handlers
# --------------------------------------------------------------------------
@bp.app_errorhandler(404)
def page_not_found(error):
    """App-wide 404 page (registered on the app via the blueprint)."""
    return render_template("404.html"), 404


# --------------------------------------------------------------------------
# Auth helpers
# --------------------------------------------------------------------------
def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("main.login"))
        return view(*args, **kwargs)
    return wrapper


def current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    return db.get_db().execute(
        "SELECT * FROM users WHERE id = ?", (uid,)
    ).fetchone()


@bp.app_context_processor
def inject_user():
    return {"username": session.get("username")}


# --------------------------------------------------------------------------
# Captcha
# --------------------------------------------------------------------------
@bp.route("/captcha.png")
def captcha_image():
    text = generate_captcha_text(5)
    session["captcha"] = text
    response = send_file(render_captcha_image(text), mimetype="image/png")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# --------------------------------------------------------------------------
# Pages: landing + auth
# --------------------------------------------------------------------------
@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = db.get_db().execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()

        if user is None:
            return render_template("login.html",
                                   message="Account not found. Please sign up first.",
                                   message_type="error")
        if not check_password_hash(user["password_hash"], password):
            return render_template("login.html",
                                   message="Incorrect password. Try again.",
                                   message_type="error")

        session["user_id"] = user["id"]
        session["username"] = user["username"]
        return redirect(url_for("main.dashboard"))

    return render_template("login.html")


@bp.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        contact = request.form.get("contact", "").strip()
        captcha_input = request.form.get("captcha", "").strip().upper()

        if not username or not password:
            return render_template("signup.html",
                                   message="Username and password are required.",
                                   message_type="error")

        expected = session.pop("captcha", None)
        if not expected or captcha_input != expected:
            return render_template("signup.html",
                                   message="Wrong captcha — try again.",
                                   message_type="error")

        conn = db.get_db()
        if conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
            return render_template("signup.html",
                                   message="That username is already taken.",
                                   message_type="error")

        if contact:
            dup = conn.execute(
                "SELECT 1 FROM users WHERE lower(contact) = ?", (contact.lower(),)
            ).fetchone()
            if dup:
                return render_template("signup.html",
                                       message="That email or phone is already registered.",
                                       message_type="error")

        conn.execute(
            "INSERT INTO users (username, password_hash, contact, created_at) "
            "VALUES (?, ?, ?, ?)",
            (username, generate_password_hash(password), contact, db.now()),
        )
        conn.commit()
        return render_template("signup.html",
                               message="Account created! You can now sign in.",
                               message_type="success")

    return render_template("signup.html")


@bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("main.index"))


# --------------------------------------------------------------------------
# Dashboard: references + recorded sessions
# --------------------------------------------------------------------------
@bp.route("/dashboard")
@login_required
def dashboard():
    conn = db.get_db()
    uid = session["user_id"]

    references = conn.execute(
        "SELECT * FROM reference_files WHERE user_id = ? ORDER BY id DESC", (uid,)
    ).fetchall()

    sessions = conn.execute(
        """
        SELECT s.*, sc.overall_score, r.original_name AS reference_name
        FROM sessions s
        LEFT JOIN scores sc ON sc.session_id = s.id
        LEFT JOIN reference_files r ON r.id = s.reference_id
        WHERE s.user_id = ?
        ORDER BY s.id DESC
        """,
        (uid,),
    ).fetchall()

    # Get which reference ids are published to gallery
    published = conn.execute(
        "SELECT reference_id FROM gallery_items WHERE user_id = ?", (uid,)
    ).fetchall()
    published_ref_ids = {row["reference_id"] for row in published}

    return render_template(
        "dashboard.html",
        references=references,
        sessions=sessions,
        active_reference_id=ref_session.get_active_reference_id(),
        published_ref_ids=published_ref_ids,
    )


# --------------------------------------------------------------------------
# Reference upload (video or image) + analysis
# --------------------------------------------------------------------------
@bp.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "GET":
        return render_template("upload.html", result=None)

    file = request.files.get("reference")
    if not file or file.filename == "":
        return render_template("upload.html",
                               result="No file selected.", result_type="error")

    kind = classify_upload(file.filename)
    if kind is None:
        return render_template("upload.html",
                               result="Unsupported file type.", result_type="error")

    uid = session["user_id"]
    label = request.form.get("label", "").strip()
    name = unique_name(file.filename, prefix="ref_")
    save_path = os.path.join(current_app.config["UPLOADS_DIR"], name)
    file.save(save_path)

    # Trim the video if trim parameters were provided
    trim_start = request.form.get("trim_start", type=float)
    trim_end = request.form.get("trim_end", type=float)
    if kind == "video" and trim_start is not None and trim_end is not None and trim_end > trim_start:
        trimmed_name = unique_name(file.filename, prefix="ref_trim_")
        trimmed_path = os.path.join(current_app.config["UPLOADS_DIR"], trimmed_name)
        try:
            # trim_video may change the extension (e.g. .webm -> .mp4), so take
            # the path it actually wrote — that's the only file we keep.
            out_path = trim_video(save_path, trimmed_path, trim_start, trim_end)
            if os.path.abspath(out_path) != os.path.abspath(save_path):
                os.remove(save_path)  # drop the full original
            save_path = out_path
            name = os.path.basename(out_path)
        except Exception as exc:  # noqa: BLE001
            # If trimming fails, use the original
            pass

    # Enforce the max clip length on the final (possibly trimmed) video. Prefer
    # the trim window (always reliable) and fall back to measuring the file.
    if kind == "video":
        max_secs = current_app.config["MAX_VIDEO_SECONDS"]
        if trim_start is not None and trim_end is not None and trim_end > trim_start:
            eff_secs = trim_end - trim_start
        else:
            eff_secs = get_video_duration(save_path)
        if eff_secs > max_secs + 0.5:
            os.remove(save_path)
            return render_template(
                "upload.html",
                result=f"Reference videos must be {max_secs} seconds or less. "
                       f"Trim it down (yours is about {eff_secs:.0f}s) and try again.",
                result_type="error")

    # Analyse the reference now so it's ready for scoring later.
    try:
        if kind == "video":
            data = get_form_data(save_path, with_landmarks=True)
        else:
            data = get_form_data_from_image(save_path, with_landmarks=True)
    except Exception as exc:  # noqa: BLE001
        os.remove(save_path)
        return render_template("upload.html",
                               result=f"Could not analyse file: {exc}",
                               result_type="error")

    if data["vectors"].size == 0:
        os.remove(save_path)
        return render_template("upload.html",
                               result="No person detected in that file.",
                               result_type="error")

    # Save the feature vectors next to the upload, but store only the name in
    # the DB (kept portable across machines — see utils.resolve_under).
    vector_name = name + ".npy"
    save_vectors(data["vectors"], os.path.join(current_app.config["UPLOADS_DIR"], vector_name))

    # First-frame tracked landmarks in reference-pixel coords (so the body's
    # aspect is preserved); the client rescales these to fit the camera frame.
    positions = data.get("positions") or []
    ref_w, ref_h = data.get("width") or 1, data.get("height") or 1
    pose_json = (
        json.dumps([[float(p.x) * ref_w, float(p.y) * ref_h] for p in positions[0]])
        if positions else None
    )

    conn = db.get_db()
    cur = conn.execute(
        "INSERT INTO reference_files (user_id, filename, original_name, label, "
        "kind, vector_path, pose_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (uid, name, file.filename, label or file.filename, kind, vector_name,
         pose_json, db.now()),
    )
    conn.commit()

    # Auto-select the just-uploaded reference for the next recording.
    ref_session.set_active_reference(cur.lastrowid)

    # Publish to gallery if requested
    if request.form.get("publish_to_gallery") and kind == "video":
        description = request.form.get("gallery_description", "").strip()
        conn.execute(
            "INSERT INTO gallery_items (reference_id, user_id, description, upvotes, downvotes, created_at) "
            "VALUES (?, ?, ?, 0, 0, ?)",
            (cur.lastrowid, uid, description, db.now()),
        )
        conn.commit()

    return render_template("upload.html",
                           result="Reference uploaded and analysed — ready to record!",
                           result_type="success",
                           filename=name, kind=kind, label=label or file.filename)


@bp.route("/reference/<int:ref_id>/select", methods=["POST"])
@login_required
def select_reference(ref_id):
    row = ref_session.get_reference_row(ref_id, session["user_id"])
    if row is None:
        abort(404)
    ref_session.set_active_reference(ref_id)
    return redirect(url_for("main.camera"))


@bp.route("/reference/<int:ref_id>/delete", methods=["POST"])
@login_required
def delete_reference(ref_id):
    conn = db.get_db()
    row = ref_session.get_reference_row(ref_id, session["user_id"])
    if row is None:
        abort(404)

    uploads_dir = current_app.config["UPLOADS_DIR"]
    for path in (
        resolve_under(uploads_dir, row["filename"]),
        resolve_under(uploads_dir, row["vector_path"]),
    ):
        if path and os.path.exists(path):
            os.remove(path)
    if row["vector_path"]:
        ref_session.invalidate(row["vector_path"])

    # Delete gallery entries, votes and the leaderboard if this reference was
    # published. The leaderboard goes away with the reference.
    gallery = conn.execute(
        "SELECT id FROM gallery_items WHERE reference_id = ?", (ref_id,)
    ).fetchone()
    if gallery:
        conn.execute("DELETE FROM leaderboard_entries WHERE gallery_item_id = ?", (gallery["id"],))
        conn.execute("DELETE FROM gallery_votes WHERE gallery_item_id = ?", (gallery["id"],))
        conn.execute("DELETE FROM gallery_items WHERE id = ?", (gallery["id"],))

    # Detach any recorded sessions (the user's own and other people's) from this
    # reference instead of deleting them, so each scorer keeps their recording
    # and score on their dashboard. Without this, the sessions.reference_id
    # foreign key blocks the delete with an IntegrityError. The dashboard shows
    # these as "vs deleted reference".
    conn.execute("UPDATE sessions SET reference_id = NULL WHERE reference_id = ?", (ref_id,))

    conn.execute("DELETE FROM reference_files WHERE id = ?", (ref_id,))
    conn.commit()
    return redirect(url_for("main.dashboard"))


# --------------------------------------------------------------------------
# Camera / recording
# --------------------------------------------------------------------------
@bp.route("/camera")
@login_required
def camera():
    conn = db.get_db()
    uid = session["user_id"]

    # If a ref_id was passed in query string (e.g. from gallery "Use This"),
    # auto-select it and redirect to clear the query param. Gallery references
    # may belong to another user, so accept any reference the user can access.
    preselected = request.args.get("ref_id", type=int)
    if preselected:
        row = ref_session.get_accessible_reference_row(preselected, uid)
        if row:
            ref_session.set_active_reference(preselected)
            # Redirect to /camera without ?ref_id= to keep URLs clean.
            return redirect(url_for("main.camera"))

    references = conn.execute(
        "SELECT * FROM reference_files WHERE user_id = ? ORDER BY id DESC",
        (uid,),
    ).fetchall()

    # The active reference may be a gallery reference owned by someone else
    # (picked via "Use This"). Surface it in the picker so it can be recorded
    # against; drop the selection if it's no longer accessible.
    active_id = ref_session.get_active_reference_id()
    if active_id and not any(r["id"] == active_id for r in references):
        extra = ref_session.get_accessible_reference_row(active_id, uid)
        if extra:
            references = [extra, *references]
        else:
            ref_session.set_active_reference(None)

    # First-frame poses keyed by reference id, for the live alignment overlay.
    ref_poses = {}
    for r in references:
        pj = r["pose_json"] if "pose_json" in r.keys() else None
        if pj:
            try:
                ref_poses[str(r["id"])] = json.loads(pj)
            except (ValueError, TypeError):
                pass

    return render_template(
        "camera.html",
        references=references,
        active_reference_id=ref_session.get_active_reference_id(),
        ref_poses=ref_poses,
    )


@bp.route("/api/sessions", methods=["POST"])
@login_required
def create_session():
    """Receive a finished recording, persist it, start background scoring."""
    from . import socketio  # local import avoids a circular import at load time

    file = request.files.get("recording")
    if not file:
        return jsonify({"error": "No recording uploaded."}), 400

    reference_id = request.form.get("reference_id", type=int)
    room = request.form.get("room", "").strip()
    if not reference_id:
        return jsonify({"error": "No reference selected."}), 400

    if ref_session.get_accessible_reference_row(reference_id, session["user_id"]) is None:
        return jsonify({"error": "Reference not found."}), 404

    # Trim the recording if trim parameters were provided
    trim_start = request.form.get("trim_start", type=float)
    trim_end = request.form.get("trim_end", type=float)

    session_id = recorder.save_recording(
        file, session["user_id"], reference_id,
        current_app.config["RECORDINGS_RAW_DIR"],
    )

    app_obj = current_app._get_current_object()
    acquired = _BG_SEMAPHORE.acquire(blocking=False)
    if not acquired:
        return jsonify({"error": "Server is busy processing other recordings. Please try again shortly."}), 503

    def _run_with_semaphore(*args, **kwargs):
        try:
            recorder.process_session(*args, **kwargs)
        finally:
            _BG_SEMAPHORE.release()

    socketio.start_background_task(
        _run_with_semaphore, app_obj, socketio, session_id, room or None,
        trim_start=trim_start, trim_end=trim_end,
    )
    return jsonify({"session_id": session_id})


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------
def _result_context(conn, s):
    """Shared template context for a session's result view (score, feedback,
    video names, AI report). Used by both the owner view and the public
    leaderboard view."""
    session_id = s["id"]
    score = conn.execute(
        "SELECT * FROM scores WHERE session_id = ?", (session_id,)
    ).fetchone()
    feedback_rows = conn.execute(
        "SELECT * FROM feedback WHERE session_id = ? ORDER BY error_deg DESC",
        (session_id,),
    ).fetchall()

    raw_name = os.path.basename(s["raw_path"]) if s["raw_path"] else None
    skel_name = os.path.basename(s["skeleton_path"]) if s["skeleton_path"] else None

    ai = None
    if "ai_feedback" in s.keys() and s["ai_feedback"]:
        try:
            ai = json.loads(s["ai_feedback"])
        except (ValueError, TypeError):
            ai = None

    return {
        "session": s, "score": score, "feedback": feedback_rows,
        "raw_name": raw_name, "skeleton_name": skel_name, "ai": ai,
    }


@bp.route("/result/<int:session_id>")
@login_required
def result(session_id):
    conn = db.get_db()
    s = conn.execute(
        "SELECT * FROM sessions WHERE id = ? AND user_id = ?",
        (session_id, session["user_id"]),
    ).fetchone()
    if s is None:
        abort(404)

    ctx = _result_context(conn, s)

    # If this session's reference is published to the gallery it has a
    # leaderboard, so the owner can publish (or remove) their score.
    gallery_item = None
    if s["reference_id"]:
        gallery_item = conn.execute(
            "SELECT id FROM gallery_items WHERE reference_id = ?", (s["reference_id"],)
        ).fetchone()
    on_leaderboard = False
    if gallery_item:
        on_leaderboard = conn.execute(
            "SELECT 1 FROM leaderboard_entries WHERE session_id = ?", (session_id,)
        ).fetchone() is not None

    return render_template(
        "result.html",
        gallery_item_id=(gallery_item["id"] if gallery_item else None),
        can_publish=(gallery_item is not None and s["status"] == "done"
                     and ctx["score"] is not None),
        on_leaderboard=on_leaderboard,
        **ctx,
    )


@bp.route("/leaderboard/session/<int:session_id>")
@login_required
def leaderboard_result(session_id):
    """Public, read-only result view for a session that has been published to a
    leaderboard — same layout the owner sees after scoring."""
    conn = db.get_db()
    # Only sessions actually on a leaderboard are viewable by other users.
    if conn.execute(
        "SELECT 1 FROM leaderboard_entries WHERE session_id = ?", (session_id,)
    ).fetchone() is None:
        abort(404)

    s = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if s is None:
        abort(404)

    owner = conn.execute(
        "SELECT username FROM users WHERE id = ?", (s["user_id"],)
    ).fetchone()

    return render_template(
        "result.html",
        leaderboard_view=True,
        entry_username=(owner["username"] if owner else "Unknown"),
        back_url=url_for("main.gallery"),
        **_result_context(conn, s),
    )


@bp.route("/session/<int:session_id>/publish_score", methods=["POST"])
@login_required
def publish_score(session_id):
    """Publish a finished recording's score + feedback onto the leaderboard of
    the gallery item it was recorded against."""
    conn = db.get_db()
    uid = session["user_id"]
    s = conn.execute(
        "SELECT * FROM sessions WHERE id = ? AND user_id = ?", (session_id, uid)
    ).fetchone()
    if s is None:
        abort(404)
    if s["status"] != "done":
        return jsonify({"error": "This recording hasn't finished scoring yet."}), 400

    score = conn.execute(
        "SELECT overall_score FROM scores WHERE session_id = ?", (session_id,)
    ).fetchone()
    if score is None or score["overall_score"] is None:
        return jsonify({"error": "This recording has no score to publish."}), 400

    gallery_item = None
    if s["reference_id"]:
        gallery_item = conn.execute(
            "SELECT id FROM gallery_items WHERE reference_id = ?", (s["reference_id"],)
        ).fetchone()
    if gallery_item is None:
        return jsonify({"error": "This reference isn't in the gallery, so it has no leaderboard."}), 400

    # UNIQUE(session_id) keeps a session on the board at most once.
    if conn.execute(
        "SELECT 1 FROM leaderboard_entries WHERE session_id = ?", (session_id,)
    ).fetchone() is None:
        conn.execute(
            "INSERT INTO leaderboard_entries (gallery_item_id, session_id, user_id, "
            "overall_score, created_at) VALUES (?, ?, ?, ?, ?)",
            (gallery_item["id"], session_id, uid, score["overall_score"], db.now()),
        )
        conn.commit()
    return jsonify({"published": True, "gallery_item_id": gallery_item["id"]})


@bp.route("/session/<int:session_id>/unpublish_score", methods=["POST"])
@login_required
def unpublish_score(session_id):
    """Remove the current user's session from its leaderboard."""
    conn = db.get_db()
    uid = session["user_id"]
    if conn.execute(
        "SELECT 1 FROM sessions WHERE id = ? AND user_id = ?", (session_id, uid)
    ).fetchone() is None:
        abort(404)
    conn.execute(
        "DELETE FROM leaderboard_entries WHERE session_id = ? AND user_id = ?",
        (session_id, uid),
    )
    conn.commit()
    return jsonify({"published": False})


@bp.route("/session/<int:session_id>/delete", methods=["POST"])
@login_required
def delete_session(session_id):
    conn = db.get_db()
    s = conn.execute(
        "SELECT * FROM sessions WHERE id = ? AND user_id = ?",
        (session_id, session["user_id"]),
    ).fetchone()
    if s is None:
        abort(404)

    for path in (
        resolve_under(current_app.config["RECORDINGS_RAW_DIR"], s["raw_path"]),
        resolve_under(current_app.config["RECORDINGS_SKELETON_DIR"], s["skeleton_path"]),
    ):
        if path and os.path.exists(path):
            os.remove(path)

    conn.execute("DELETE FROM leaderboard_entries WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM scores WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM feedback WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    return redirect(url_for("main.dashboard"))


# --------------------------------------------------------------------------
# File serving
# --------------------------------------------------------------------------
@bp.route("/uploads/<path:filename>")
@login_required
def serve_upload(filename):
    return send_from_directory(current_app.config["UPLOADS_DIR"], filename)


@bp.route("/uploads/public/<path:filename>")
def serve_upload_public(filename):
    """Serve upload files publicly (no login required) for the gallery."""
    return send_from_directory(current_app.config["UPLOADS_DIR"], filename)


@bp.route("/recordings/raw/<path:filename>")
@login_required
def serve_raw(filename):
    return send_from_directory(current_app.config["RECORDINGS_RAW_DIR"], filename)


@bp.route("/recordings/skeleton/<path:filename>")
@login_required
def serve_skeleton(filename):
    return send_from_directory(current_app.config["RECORDINGS_SKELETON_DIR"], filename)


# --------------------------------------------------------------------------
# Gallery: browse published references
# --------------------------------------------------------------------------
@bp.route("/gallery")
def gallery():
    """Browse published reference videos, sorted by rating, searchable."""
    conn = db.get_db()
    q = request.args.get("q", "").strip()

    query = """
        SELECT g.id, g.description, g.upvotes, g.downvotes, g.created_at,
               r.id AS ref_id, r.filename, r.label, r.original_name, r.kind,
               u.username,
               (g.upvotes - g.downvotes) AS net_rating
        FROM gallery_items g
        JOIN reference_files r ON r.id = g.reference_id
        JOIN users u ON u.id = g.user_id
    """
    params = []

    if q:
        query += " WHERE r.label LIKE ? COLLATE NOCASE"
        params.append(f"%{q}%")

    query += " ORDER BY net_rating DESC, g.created_at DESC"

    items = conn.execute(query, params).fetchall()

    # Check which ones the current user has voted on
    user_votes = {}
    if session.get("user_id"):
        rows = conn.execute(
            "SELECT gallery_item_id, vote FROM gallery_votes WHERE user_id = ?",
            (session["user_id"],),
        ).fetchall()
        user_votes = {row["gallery_item_id"]: row["vote"] for row in rows}

    return render_template(
        "gallery.html",
        items=items,
        q=q,
        user_votes=user_votes,
    )


@bp.route("/reference/<int:ref_id>/publish", methods=["POST"])
@login_required
def publish_reference(ref_id):
    """Publish a reference to the public gallery with a description."""
    conn = db.get_db()
    row = conn.execute(
        "SELECT * FROM reference_files WHERE id = ? AND user_id = ?",
        (ref_id, session["user_id"]),
    ).fetchone()
    if row is None:
        abort(404)

    if row["kind"] != "video":
        return jsonify({"error": "Only video references can be published to the gallery."}), 400

    # Check if already published
    existing = conn.execute(
        "SELECT id FROM gallery_items WHERE reference_id = ?", (ref_id,)
    ).fetchone()
    if existing:
        return jsonify({"error": "This reference is already published."}), 400

    description = request.form.get("description", "").strip()
    conn.execute(
        "INSERT INTO gallery_items (reference_id, user_id, description, upvotes, downvotes, created_at) "
        "VALUES (?, ?, ?, 0, 0, ?)",
        (ref_id, session["user_id"], description, db.now()),
    )
    conn.commit()
    return redirect(url_for("main.dashboard"))


@bp.route("/gallery/<int:item_id>/vote", methods=["POST"])
@login_required
def gallery_vote(item_id):
    """Upvote or downvote a gallery item. Expects JSON: {"vote": 1 or -1}."""
    conn = db.get_db()
    item = conn.execute(
        "SELECT * FROM gallery_items WHERE id = ?", (item_id,)
    ).fetchone()
    if item is None:
        return jsonify({"error": "Gallery item not found."}), 404

    data = request.get_json(silent=True) or {}
    vote = data.get("vote")
    if vote not in (1, -1):
        return jsonify({"error": "Vote must be 1 (upvote) or -1 (downvote)."}), 400

    uid = session["user_id"]

    # Upsert the vote
    existing = conn.execute(
        "SELECT id, vote FROM gallery_votes WHERE user_id = ? AND gallery_item_id = ?",
        (uid, item_id),
    ).fetchone()

    if existing:
        if existing["vote"] == vote:
            # Same vote — remove it (toggle off)
            conn.execute("DELETE FROM gallery_votes WHERE id = ?", (existing["id"],))
            if vote == 1:
                conn.execute("UPDATE gallery_items SET upvotes = upvotes - 1 WHERE id = ?", (item_id,))
            else:
                conn.execute("UPDATE gallery_items SET downvotes = downvotes - 1 WHERE id = ?", (item_id,))
        else:
            # Change vote direction
            conn.execute("UPDATE gallery_votes SET vote = ? WHERE id = ?", (vote, existing["id"]))
            if vote == 1:
                conn.execute("UPDATE gallery_items SET upvotes = upvotes + 1, downvotes = downvotes - 1 WHERE id = ?", (item_id,))
            else:
                conn.execute("UPDATE gallery_items SET upvotes = upvotes - 1, downvotes = downvotes + 1 WHERE id = ?", (item_id,))
    else:
        conn.execute(
            "INSERT INTO gallery_votes (user_id, gallery_item_id, vote) VALUES (?, ?, ?)",
            (uid, item_id, vote),
        )
        if vote == 1:
            conn.execute("UPDATE gallery_items SET upvotes = upvotes + 1 WHERE id = ?", (item_id,))
        else:
            conn.execute("UPDATE gallery_items SET downvotes = downvotes + 1 WHERE id = ?", (item_id,))

    conn.commit()

    updated = conn.execute(
        "SELECT upvotes, downvotes FROM gallery_items WHERE id = ?", (item_id,)
    ).fetchone()
    return jsonify({
        "upvotes": updated["upvotes"],
        "downvotes": updated["downvotes"],
        "net": updated["upvotes"] - updated["downvotes"],
        "user_vote": vote if not existing or existing["vote"] != vote else 0,
    })


@bp.route("/gallery/<int:item_id>")
def gallery_item_detail(item_id):
    """Full detail for one gallery item: description, votes, how many distinct
    users have recorded against it, and its comments. Used by the modal."""
    conn = db.get_db()
    item = conn.execute(
        """
        SELECT g.id, g.description, g.upvotes, g.downvotes, g.created_at,
               r.id AS ref_id, r.filename, r.label, r.original_name,
               u.username
        FROM gallery_items g
        JOIN reference_files r ON r.id = g.reference_id
        JOIN users u ON u.id = g.user_id
        WHERE g.id = ?
        """,
        (item_id,),
    ).fetchone()
    if item is None:
        return jsonify({"error": "Gallery item not found."}), 404

    # Distinct people who recorded against this reference (not # of recordings).
    used_by = conn.execute(
        "SELECT COUNT(DISTINCT user_id) AS n FROM sessions WHERE reference_id = ?",
        (item["ref_id"],),
    ).fetchone()["n"]

    comments = conn.execute(
        """
        SELECT c.body, c.created_at, u.username
        FROM gallery_comments c
        JOIN users u ON u.id = c.user_id
        WHERE c.gallery_item_id = ?
        ORDER BY c.id DESC
        """,
        (item_id,),
    ).fetchall()

    user_vote = 0
    if session.get("user_id"):
        row = conn.execute(
            "SELECT vote FROM gallery_votes WHERE user_id = ? AND gallery_item_id = ?",
            (session["user_id"], item_id),
        ).fetchone()
        if row:
            user_vote = row["vote"]

    return jsonify({
        "id": item["id"],
        "ref_id": item["ref_id"],
        "label": item["label"] or item["original_name"],
        "username": item["username"],
        "created_at": item["created_at"],
        "description": item["description"] or "",
        "upvotes": item["upvotes"],
        "downvotes": item["downvotes"],
        "net": item["upvotes"] - item["downvotes"],
        "user_vote": user_vote,
        "used_by": used_by,
        "video_url": url_for("main.serve_upload_public", filename=item["filename"]),
        "use_url": url_for("main.camera", ref_id=item["ref_id"]),
        "comments": [
            {"username": c["username"], "body": c["body"],
             "created_at": c["created_at"]}
            for c in comments
        ],
    })


@bp.route("/gallery/<int:item_id>/leaderboard")
def gallery_leaderboard(item_id):
    """Ranked list of published scores for one gallery item's reference.

    Returns entries sorted best-first; each links to a public result view.
    """
    conn = db.get_db()
    if conn.execute(
        "SELECT 1 FROM gallery_items WHERE id = ?", (item_id,)
    ).fetchone() is None:
        return jsonify({"error": "Gallery item not found."}), 404

    rows = conn.execute(
        """
        SELECT le.session_id, le.user_id, le.overall_score, le.created_at,
               u.username
        FROM leaderboard_entries le
        JOIN users u ON u.id = le.user_id
        WHERE le.gallery_item_id = ?
        ORDER BY le.overall_score DESC, le.created_at ASC
        """,
        (item_id,),
    ).fetchall()

    uid = session.get("user_id")
    entries = [
        {
            "rank": rank,
            "session_id": r["session_id"],
            "username": r["username"],
            "overall_score": r["overall_score"],
            "created_at": r["created_at"],
            "is_you": uid is not None and r["user_id"] == uid,
            "url": url_for("main.leaderboard_result", session_id=r["session_id"]),
        }
        for rank, r in enumerate(rows, start=1)
    ]
    return jsonify({"item_id": item_id, "entries": entries})


@bp.route("/gallery/<int:item_id>/comment", methods=["POST"])
@login_required
def gallery_comment(item_id):
    """Add a comment to a gallery item. Expects JSON: {"body": "..."}."""
    conn = db.get_db()
    if conn.execute(
        "SELECT 1 FROM gallery_items WHERE id = ?", (item_id,)
    ).fetchone() is None:
        return jsonify({"error": "Gallery item not found."}), 404

    data = request.get_json(silent=True) or {}
    body = (data.get("body") or "").strip()
    if not body:
        return jsonify({"error": "Comment cannot be empty."}), 400
    body = body[:1000]  # keep comments reasonable

    created_at = db.now()
    conn.execute(
        "INSERT INTO gallery_comments (gallery_item_id, user_id, body, created_at) "
        "VALUES (?, ?, ?, ?)",
        (item_id, session["user_id"], body, created_at),
    )
    conn.commit()
    return jsonify({
        "username": session["username"],
        "body": body,
        "created_at": created_at,
    })
