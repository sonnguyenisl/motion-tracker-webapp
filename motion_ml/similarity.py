"""Sequence similarity scoring via DTW.

Aligns the reference and user feature sequences with Dynamic Time Warping
(so tempo differences don't penalise the user) and produces an overall
similarity score plus per-joint error stats. Pure computation, no printing.

DTW is used to time-align the two sequences; the per-frame angle and landmark
errors along that alignment drive the scores. ``angle_score`` (technique) and
``landmark_score`` (posture) fall off linearly with error, and ``overall_score``
is a weighted blend of the two so visible mistakes clearly lower the result.
``dtw_distance`` is still reported as a raw similarity metric.
"""

import numpy as np
from tslearn.metrics import dtw_path

from .pose_estimator import ANGLE_DIM, LANDMARK_DIM, ANGLE_NAMES

# --- Scoring sensitivity (the main tuning knobs) ----------------------------
# Posture falls linearly from 100 (perfect) to 0 at LANDMARK_ERROR_ZERO, so a
# SMALLER zero-point = harsher posture scoring. Technique uses the original
# gentle linear penalty. Overall is a weighted blend of the two.
LANDMARK_ERROR_ZERO = 0.25    # avg generalized-landmark error that scores 0% posture
TECHNIQUE_WEIGHT = 0.6        # overall = 0.6*technique(angles) + 0.4*posture(landmarks)


def _compute_path(ref_vector, user_vector):
    """Compute the DTW path once and return (path, distance, ref, user)."""
    ref_vector = np.asarray(ref_vector, dtype=np.float32)
    user_vector = np.asarray(user_vector, dtype=np.float32)
    if ref_vector.size == 0 or user_vector.size == 0:
        raise ValueError("Empty pose sequence — no person detected in a video.")
    path, dtw_distance = dtw_path(ref_vector, user_vector)
    return path, dtw_distance, ref_vector, user_vector


def evaluate_motion(ref_vector, user_vector):
    """Compare two (n_frames, 32) feature sequences.

    Returns a dict of scores and per-joint stats consumed by feedback.build.
    """
    path, dtw_distance, ref_vector, user_vector = _compute_path(ref_vector, user_vector)

    angle_signed = np.zeros(ANGLE_DIM)
    angle_abs = np.zeros(ANGLE_DIM)
    landmark_sum = 0.0
    landmark_count = 0

    for i, j in path:
        diff = ref_vector[i] - user_vector[j]

        angle_diff = diff[:ANGLE_DIM]
        angle_signed += angle_diff
        angle_abs += np.abs(angle_diff)

        # Static-landmark error only (exclude the velocity block that follows).
        lm_diff = diff[ANGLE_DIM:ANGLE_DIM + LANDMARK_DIM]
        if lm_diff.size:
            landmark_sum += float(np.mean(np.abs(lm_diff)))
            landmark_count += 1

    n = len(path)
    angle_signed /= n
    angle_abs /= n
    landmark_error = landmark_sum / landmark_count if landmark_count else 0.0
    avg_angle_error = float(np.mean(angle_abs))

    # Technique: original gentle linear penalty (restored). Posture: harsher
    # linear fall-off to 0 at LANDMARK_ERROR_ZERO so posture errors are visible.
    angle_score = max(0.0, 100.0 - avg_angle_error * 2.0)
    landmark_score = float(np.clip(100.0 * (1.0 - landmark_error / LANDMARK_ERROR_ZERO), 0.0, 100.0))

    # Overall is a blend of the two visible sub-scores (technique-led).
    overall_score = TECHNIQUE_WEIGHT * angle_score + (1.0 - TECHNIQUE_WEIGHT) * landmark_score

    return {
        "overall_score": round(overall_score, 1),
        "angle_score": round(angle_score, 1),
        "landmark_score": round(landmark_score, 1),
        "avg_angle_error": avg_angle_error,
        "avg_landmark_error": landmark_error,
        "angle_abs": angle_abs,
        "angle_signed": angle_signed,
        "angle_names": ANGLE_NAMES,
        "dtw_distance": float(dtw_distance),
        "_path": path,  # reused by feedback to avoid recomputing DTW
    }


def feedback(ref_vec, user_vec, path=None):
    """Signed per-component differences (user - reference) along the DTW path.

    If ``path`` is provided (from ``evaluate_motion``'s ``_path`` key), it is
    reused to avoid recomputing the expensive DTW alignment. Otherwise the path
    is computed from scratch.

    Returns a dict with "angles" (8), "landmarks" (12 joints) and "velocity"
    (12 joints), each a list of {joint, ..., explanation} entries.
    """
    if path is None:
        path, _, ref_vec, user_vec = _compute_path(ref_vec, user_vec)
    else:
        ref_vec = np.asarray(ref_vec, dtype=np.float32)
        user_vec = np.asarray(user_vec, dtype=np.float32)

    ANGLE_DIM = 8
    LANDMARK_DIM = 24
    VELOCITY_DIM = 24

    angle_signed = np.zeros(ANGLE_DIM)
    landmark_signed = np.zeros(LANDMARK_DIM)
    velocity_signed = np.zeros(VELOCITY_DIM)

    for i, j in path:
        diff = user_vec[j] - ref_vec[i]
        angle_signed += diff[:ANGLE_DIM]
        landmark_signed += diff[ANGLE_DIM:ANGLE_DIM + LANDMARK_DIM]
        velocity_signed += diff[ANGLE_DIM + LANDMARK_DIM:]

    angle_signed /= len(path)
    landmark_signed /= len(path)
    velocity_signed /= len(path)

    # --------------------------------------------------
    # Names
    # --------------------------------------------------
    angle_names = [
        "Left Elbow", "Right Elbow", "Left Shoulder", "Right Shoulder",
        "Left Hip", "Right Hip", "Left Knee", "Right Knee",
    ]
    joint_names = [
        "Left Elbow", "Right Elbow", "Left Shoulder", "Right Shoulder",
        "Left Hip", "Right Hip", "Left Ankle", "Right Ankle",
        "Left Wrist", "Right Wrist", "Left Knee", "Right Knee",
    ]

    results = {"angles": [], "landmarks": [], "velocity": []}

    # --------------------------------------------------
    # ANGLES
    # --------------------------------------------------
    for idx, name in enumerate(angle_names):
        diff = angle_signed[idx]
        if diff > 0:
            explanation = f"{name} angle is {abs(diff):.2f} degree larger than reference."
        else:
            explanation = f"{name} angle is {abs(diff):.2f} degree smaller than reference."
        results["angles"].append({
            "joint": name,
            "difference_deg": float(diff),
            "explanation": explanation,
        })

    # --------------------------------------------------
    # LANDMARKS
    # --------------------------------------------------
    for joint_idx, name in enumerate(joint_names):
        dx = landmark_signed[joint_idx * 2]
        dy = landmark_signed[joint_idx * 2 + 1]
        offset = np.sqrt(dx ** 2 + dy ** 2)
        horizontal = "right" if dx > 0 else "left"
        vertical = "lower" if dy > 0 else "higher"
        explanation = (
            f"{name} is offset by {offset:.3f} body units. "
            f"It is positioned more {horizontal} and {vertical} than the reference."
        )
        results["landmarks"].append({
            "joint": name,
            "dx": float(dx),
            "dy": float(dy),
            "offset": float(offset),
            "explanation": explanation,
        })

    # --------------------------------------------------
    # VELOCITY
    # --------------------------------------------------
    for joint_idx, name in enumerate(joint_names):
        vx = velocity_signed[joint_idx * 2]
        vy = velocity_signed[joint_idx * 2 + 1]
        speed_diff = np.sqrt(vx ** 2 + vy ** 2)
        if vx > 0 or vy > 0:
            explanation = (
                f"{name} moves faster than the reference "
                f"by approximately {speed_diff:.3f} units/frame."
            )
        else:
            explanation = (
                f"{name} moves slower than the reference "
                f"by approximately {speed_diff:.3f} units/frame."
            )
        results["velocity"].append({
            "joint": name,
            "vx_diff": float(vx),
            "vy_diff": float(vy),
            "speed_diff": float(speed_diff),
            "explanation": explanation,
        })

    return results


def format_feedback(fb, label=None):
    """Build the human-readable feedback text — exactly what print_feedback
    prints to the console (and what gets fed to the LLM coach)."""
    bar = "=" * 50
    lines = [bar]
    if label:
        lines.append(f"EXERCISE: {label}")
    lines.append("MOTION FEEDBACK (user - reference)")
    lines.append(bar)

    lines.append("")
    lines.append("--- ANGLES ---")
    lines.extend(item["explanation"] for item in fb["angles"])

    lines.append("")
    lines.append("--- LANDMARKS ---")
    lines.extend(item["explanation"] for item in fb["landmarks"])

    lines.append("")
    lines.append("--- VELOCITY ---")
    lines.extend(item["explanation"] for item in fb["velocity"])

    lines.append(bar)
    return "\n".join(lines)


def print_feedback(ref_vec, user_vec, label=None, path=None):
    """Print the signed angle/landmark/velocity differences to the console.

    ``path`` is the DTW alignment path from ``evaluate_motion``, reused to
    avoid recomputing the expensive DTW.

    Returns the raw feedback() dict.
    """
    fb = feedback(ref_vec, user_vec, path=path)
    print("\n" + format_feedback(fb, label) + "\n")
    return fb
