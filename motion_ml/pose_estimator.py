"""MediaPipe Pose (Tasks API) wrapper + feature extraction.

Turns a video file (or image) into a sequence of per-frame feature vectors
used by the similarity stage. A feature vector is::

    [ 8 joint angles , 24 generalized landmarks , 24 landmark velocities ]  -> length 56

The velocity block is the frame-to-frame change of the generalized landmarks
(zeros on the first frame), giving the similarity stage a sense of motion
dynamics, not just static poses.

Uses the modern MediaPipe Tasks ``PoseLandmarker``. The landmark indices are
the standard 33-point BlazePose layout, so the scoring math is identical to
the original ``mp.solutions.pose`` prototype.
"""

import os
import math
import atexit
import threading
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions, RunningMode,
)

# Global landmarker cache: one per running mode.
# Avoids re-loading the ~15 MB model and re-allocating the TFLite interpreter
# on every session, which prevents cumulative C++ heap growth.
_LANDMARKERS = {}
_LANDMARKERS_LOCK = threading.Lock()
# Monotonically increasing timestamp counter so a cached VIDEO-mode
# PoseLandmarker can be reused across distinct video files.
_NEXT_TS = 0
_TS_LOCK = threading.Lock()

# --- Model asset (auto-downloaded on first use) ---------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "pose_landmarker_lite.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


def ensure_model():
    """Download the pose landmarker model once; return its local path."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODELS_DIR, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


# --- BlazePose landmark indices -------------------------------------------
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

ANGLE_DIM = 8
LANDMARK_DIM = 24       # 12 tracked landmarks x (x, y)
VELOCITY_DIM = 24       # frame-to-frame delta of the generalized landmarks
FEATURE_DIM = ANGLE_DIM + LANDMARK_DIM + VELOCITY_DIM  # 56

ANGLE_NAMES = [
    "Left Elbow", "Right Elbow", "Left Shoulder", "Right Shoulder",
    "Left Hip", "Right Hip", "Left Knee", "Right Knee",
]

# The 12 tracked landmarks, in the order generalize_landmarks/draw_pose expect.
TRACKED_LANDMARKS = [
    L_ELBOW, R_ELBOW, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP,
    L_ANKLE, R_ANKLE, L_WRIST, R_WRIST, L_KNEE, R_KNEE,
]

_LM_KEYS = [
    "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
    "left_hip", "right_hip", "left_ankle", "right_ankle",
    "left_wrist", "right_wrist", "left_knee", "right_knee",
]


def cal_angle(joint_a, joint_b, joint_c):
    """Angle ABC in degrees. Each joint has .x and .y attributes."""
    ax, ay = joint_a.x - joint_b.x, joint_a.y - joint_b.y
    cx, cy = joint_c.x - joint_b.x, joint_c.y - joint_b.y

    dot = ax * cx + ay * cy
    mag_a = math.hypot(ax, ay)
    mag_c = math.hypot(cx, cy)
    if mag_a == 0 or mag_c == 0:
        return 0.0

    cos_angle = max(-1.0, min(1.0, dot / (mag_a * mag_c)))
    return math.degrees(math.acos(cos_angle))


def generalize_landmarks(points):
    """Translation/scale-invariant landmark coords.

    ``points`` is the list of 12 tracked landmarks (objects with .x/.y) in
    _LM_KEYS order. Recentred on the hip midpoint and scaled by the body
    height (shoulder-centre to ankle-centre distance), so the features are
    invariant to where the person stands and how large they appear.
    Returns a flat float32 array of length LANDMARK_DIM (24).
    """
    coords = {name: np.array([lm.x, lm.y]) for name, lm in zip(_LM_KEYS, points)}

    hip_center = (coords["left_hip"] + coords["right_hip"]) / 2.0
    shoulder_center = (coords["left_shoulder"] + coords["right_shoulder"]) / 2.0
    ankle_center = (coords["left_ankle"] + coords["right_ankle"]) / 2.0

    body_height = np.linalg.norm(shoulder_center - ankle_center)
    if body_height < 1e-6:
        body_height = 1.0

    generalized = []
    for point in coords.values():
        generalized.extend((point - hip_center) / body_height)
    return np.array(generalized, dtype=np.float32)


def _frame_angles(lm):
    """The 8 joint angles for one frame; ``lm`` indexable by landmark index."""
    return [
        cal_angle(lm[L_WRIST], lm[L_ELBOW], lm[L_SHOULDER]),
        cal_angle(lm[R_WRIST], lm[R_ELBOW], lm[R_SHOULDER]),
        cal_angle(lm[L_HIP], lm[L_SHOULDER], lm[L_ELBOW]),
        cal_angle(lm[R_HIP], lm[R_SHOULDER], lm[R_ELBOW]),
        cal_angle(lm[L_SHOULDER], lm[L_HIP], lm[L_KNEE]),
        cal_angle(lm[R_SHOULDER], lm[R_HIP], lm[R_KNEE]),
        cal_angle(lm[L_HIP], lm[L_KNEE], lm[L_ANKLE]),
        cal_angle(lm[R_HIP], lm[R_KNEE], lm[R_ANKLE]),
    ]


def _frame_vector(lm, with_landmarks, prev_landmarks=None):
    """Build one frame's feature vector.

    Without landmarks: just the 8 joint angles.
    With landmarks: ``[8 angles, 24 generalized landmarks, 24 velocity]`` (56),
    where velocity is the change in generalized landmarks since the previous
    frame (zeros on the first frame).

    Returns ``(vector, current_landmarks)``; feed ``current_landmarks`` back in
    as ``prev_landmarks`` on the next frame to get velocity. When
    ``with_landmarks`` is False the second item is None.
    """
    angles = np.array(_frame_angles(lm), dtype=np.float32)
    if not with_landmarks:
        return angles, None

    tracked = [lm[idx] for idx in TRACKED_LANDMARKS]
    current = generalize_landmarks(tracked)
    if prev_landmarks is None:
        velocity = np.zeros_like(current)
    else:
        velocity = current - prev_landmarks
    return np.concatenate((angles, current, velocity)), current


def make_landmarker(running_mode=RunningMode.VIDEO):
    with _LANDMARKERS_LOCK:
        if running_mode not in _LANDMARKERS:
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=ensure_model()),
                running_mode=running_mode,
                num_poses=1,
            )
            _LANDMARKERS[running_mode] = PoseLandmarker.create_from_options(options)
        return _LANDMARKERS[running_mode]


def close_landmarkers():
    """Explicitly close all cached landmarkers and clear the cache.
    Intended for app shutdown or idle timeout.
    """
    with _LANDMARKERS_LOCK:
        for lm in _LANDMARKERS.values():
            lm.close()
        _LANDMARKERS.clear()


atexit.register(close_landmarkers)


def _next_timestamp():
    """Return a globally unique, monotonically increasing timestamp in ms."""
    global _NEXT_TS
    with _TS_LOCK:
        _NEXT_TS += 1
        return _NEXT_TS


def get_vid_duration(video_path):
    """Duration of a video in seconds (0 if it can't be read)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return (frame_count / fps) if fps else 0.0


def get_form_data(video_path, with_landmarks=True, progress=None, return_landmarks=False):
    """Extract per-frame feature vectors from a video.

    When ``return_landmarks`` is True, also returns ``landmarks``:
    a numpy array of shape (n_frames, 33, 2) with normalized x,y for all 33
    BlazePose landmarks, so ``burn_skeleton`` can reuse them instead of
    re-running MediaPipe.

    Returns ``{"vectors", "positions", "width", "height"}``.
    """
    landmarker = make_landmarker(RunningMode.VIDEO)
    cap = cv2.VideoCapture(video_path)
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0

        vectors, positions = [], []
        frame_landmarks = [] if return_landmarks else None
        seen = 0
        prev_landmarks = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = _next_timestamp()
            seen += 1

            result = landmarker.detect_for_video(mp_image, ts)
            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                vector, prev_landmarks = _frame_vector(lm, with_landmarks, prev_landmarks)
                vectors.append(vector)
                if seen == 1:
                    positions.append([lm[idx] for idx in TRACKED_LANDMARKS])
                if return_landmarks:
                    frame_landmarks.append([[p.x, p.y] for p in lm])

            if progress and total:
                progress(min(seen / total, 1.0))

        result = {
            "vectors": np.array(vectors, dtype=np.float32),
            "positions": positions,
            "width": width,
            "height": height,
        }
        if return_landmarks and frame_landmarks:
            result["landmarks"] = np.array(frame_landmarks, dtype=np.float32)
        return result
    finally:
        cap.release()
        # landmarker NOT closed — cached globally to avoid C++ alloc churn


def get_form_data_from_image(image_path, with_landmarks=True):
    """Single-frame feature extraction for an image reference."""
    landmarker = make_landmarker(RunningMode.IMAGE)
    try:
        image = cv2.imread(image_path)
        if image is None:
            return {"vectors": np.empty((0, 0), dtype=np.float32),
                    "positions": [], "width": 0, "height": 0}

        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        vectors, positions = [], []
        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            vector, _ = _frame_vector(lm, with_landmarks, None)
            vectors.append(vector)
            positions.append([lm[idx] for idx in TRACKED_LANDMARKS])

        return {
            "vectors": np.array(vectors, dtype=np.float32),
            "positions": positions,
            "width": width,
            "height": height,
        }
    finally:
        # landmarker NOT closed — cached globally
        pass
