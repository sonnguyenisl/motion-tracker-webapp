"""Turn raw similarity stats into human-readable, per-joint coaching feedback."""

import numpy as np


def _grade(score):
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 60:
        return "Fair"
    return "Needs work"


def build_feedback(evaluation, top_n=3):
    """Build a structured feedback report from an evaluate_motion() result.

    Returns::

        {
            "overall_score", "angle_score", "landmark_score", "grade",
            "summary": str,
            "joints": [ {name, error, signed, direction, tip}, ... ],  # all 8
            "top_issues": [ same dicts, worst first ],
        }
    """
    angle_abs = np.asarray(evaluation["angle_abs"])
    angle_signed = np.asarray(evaluation["angle_signed"])
    names = evaluation["angle_names"]

    joints = []
    for i, name in enumerate(names):
        # Reference minus user: positive signed error means the user's angle
        # was *smaller* (more bent) than the reference.
        direction = "more bent than the reference" if angle_signed[i] > 0 \
            else "more extended than the reference"
        joints.append({
            "name": name,
            "error": round(float(angle_abs[i]), 1),
            "signed": round(float(angle_signed[i]), 1),
            "direction": direction,
            "tip": _tip_for(name, angle_abs[i], angle_signed[i]),
        })

    order = np.argsort(angle_abs)[::-1]
    top_issues = [joints[idx] for idx in order[:top_n] if angle_abs[idx] > 1.0]

    overall = evaluation["overall_score"]
    if top_issues:
        focus = ", ".join(j["name"].lower() for j in top_issues)
        summary = (
            f"{_grade(overall)} form ({overall:.0f}%). "
            f"Focus on your {focus} to tighten up the movement."
        )
    else:
        summary = f"{_grade(overall)} form ({overall:.0f}%). Great alignment overall!"

    return {
        "overall_score": overall,
        "angle_score": evaluation["angle_score"],
        "landmark_score": evaluation["landmark_score"],
        "grade": _grade(overall),
        "summary": summary,
        "joints": joints,
        "top_issues": top_issues,
    }


def _tip_for(name, error, signed):
    if error <= 1.0:
        return f"{name} tracked the reference closely — nice."
    amount = abs(signed)
    if signed > 0:
        return (
            f"Your {name.lower()} was about {amount:.0f}° more bent than the "
            f"reference. Try opening that joint up a little."
        )
    return (
        f"Your {name.lower()} was about {amount:.0f}° more extended than the "
        f"reference. Try bending it slightly more."
    )
