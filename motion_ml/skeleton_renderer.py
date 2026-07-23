"""Burn a pose skeleton onto video frames with OpenCV (MediaPipe Tasks API).

Produces the "skeleton" variant of a recording shown on the result page when
the user toggles the skeleton overlay. Also provides ``draw_pose`` to render a
normalised reference pose onto a frame (used for a live reference overlay).
"""

import ctypes
import gc
import os
import shutil
import subprocess

import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import RunningMode

from .pose_estimator import make_landmarker

# System ffmpeg locations (used to transcode to browser-playable H.264).
_FFMPEG_CANDIDATES = ("ffmpeg", r"C:\ffmpeg\bin\ffmpeg.exe", "/usr/bin/ffmpeg")


def _transcode_to_h264(src, dst):
    """Re-encode ``src`` to H.264 / yuv420p MP4 at ``dst`` with the system
    ffmpeg's software libx264 encoder. Returns True on success.

    OpenCV's bundled ffmpeg only exposes the hardware ``h264_v4l2m2m`` encoder,
    which can't initialise in a headless container (no /dev/video device), and
    the ``mp4v`` fallback it can write doesn't play in browser <video> tags.
    So we write mp4v first (reliable everywhere) and re-encode here to a format
    browsers can actually play.
    """
    for ffmpeg in _FFMPEG_CANDIDATES:
        try:
            result = subprocess.run(
                [ffmpeg, "-y", "-i", src,
                 "-threads", "1",
                 "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", dst],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300,
            )
            if result.returncode == 0 and os.path.exists(dst):
                return True
        except Exception:  # noqa: BLE001
            continue
    return False

# Standard BlazePose body connections (index pairs into the 33-point layout).
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
    (27, 29), (29, 31), (28, 30), (30, 32),
]

# Connections between the 12 tracked landmarks (indices into TRACKED_LANDMARKS).
_REF_CONNECTIONS = [
    (2, 3),
    (2, 0), (0, 8),
    (3, 1), (1, 9),
    (2, 4), (3, 5), (4, 5),
    (4, 10), (10, 6),
    (5, 11), (11, 7),
]


def draw_pose(frame, points, ref_w, ref_h, padding=0.8):
    """Draw a normalised reference pose centred/scaled onto ``frame``.

    ``points`` are the 12 tracked landmarks (objects with .x/.y normalised to
    the reference size ref_w x ref_h). Returns the pixel points drawn.
    """
    screen_h, screen_w = frame.shape[:2]
    xs = [p.x * ref_w for p in points]
    ys = [p.y * ref_h for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    pose_w = max(max_x - min_x, 1e-6)
    pose_h = max(max_y - min_y, 1e-6)
    scale = min(screen_w * padding / pose_w, screen_h * padding / pose_h)
    offset_x = (screen_w - pose_w * scale) / 2
    offset_y = (screen_h - pose_h * scale) / 2

    pixel_points = []
    for p in points:
        x = int((p.x * ref_w - min_x) * scale + offset_x)
        y = int((p.y * ref_h - min_y) * scale + offset_y)
        pixel_points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    for i, j in _REF_CONNECTIONS:
        cv2.line(frame, pixel_points[i], pixel_points[j], (255, 0, 0), 2)
    return pixel_points


def _draw_points(frame, pixel_pts):
    """Draw skeleton from pre-computed pixel points (list of (x, y) tuples).

    Uses the standard 33-point BlazePose POSE_CONNECTIONS.
    """
    for a, b in POSE_CONNECTIONS:
        if a < len(pixel_pts) and b < len(pixel_pts):
            cv2.line(frame, pixel_pts[a], pixel_pts[b], (255, 0, 0), 2)
    for (x, y) in pixel_pts:
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)


def _draw_detected(frame, landmarks):
    """Draw the full detected 33-point skeleton onto a BGR frame."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (255, 0, 0), 2)
    for (x, y) in pts:
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)


def burn_skeleton(input_video, output_video, progress=None, precomputed_landmarks=None):
    """Read ``input_video``, draw the detected skeleton on every frame, write
    the annotated result to ``output_video``. Returns the output path.

    When ``precomputed_landmarks`` (numpy array of shape (n_frames, 33, 2)) is
    provided, MediaPipe pose detection is skipped and the given landmarks are
    used instead. This avoids running the pose model twice per session.
    """
    landmarker = None if precomputed_landmarks is not None else make_landmarker(RunningMode.IMAGE)
    cap = cv2.VideoCapture(input_video)
    out = None
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0

        # Write frames with mp4v (always available, even in a headless container),
        # then transcode to browser-playable H.264 below. OpenCV can't produce
        # H.264 here — its bundled encoder needs hardware that isn't present.
        tmp_path = os.path.splitext(output_video)[0] + "_mp4v.mp4"
        for codec in ("mp4v",):
            candidate = cv2.VideoWriter(
                tmp_path, cv2.VideoWriter_fourcc(*codec), fps, (width, height)
            )
            if candidate.isOpened():
                out = candidate
                break
            candidate.release()
        if out is None:
            raise RuntimeError("Could not open a VideoWriter for skeleton output.")

        seen = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if precomputed_landmarks is not None and seen < len(precomputed_landmarks):
                pts = precomputed_landmarks[seen]
                h, w = frame.shape[:2]
                pixel_pts = [(int(p[0] * w), int(p[1] * h)) for p in pts]
                _draw_points(frame, pixel_pts)
            elif landmarker:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_image)
                if result.pose_landmarks:
                    _draw_detected(frame, result.pose_landmarks[0])

            out.write(frame)
            seen += 1

            if seen % 30 == 0:
                print(f"Processing frame {seen}/{int(total)}...", flush=True)

            if progress and total:
                progress(min(seen / total, 1.0))

    finally:
        cap.release()
        if out is not None:
            out.release()
        # landmarker NOT closed — globally cached in pose_estimator

    # Release the OpenCV encoder *before* ffmpeg transcode so both don't hold
    # frame buffers simultaneously.
    out = None

    # Transcode the mp4v draft to H.264 so it plays in the browser <video>.
    # If ffmpeg isn't available, keep the mp4v file as a best-effort fallback.
    if _transcode_to_h264(tmp_path, output_video):
        os.remove(tmp_path)
    else:
        shutil.move(tmp_path, output_video)
    return output_video
