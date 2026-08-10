"""LLM coaching layer.

Turns the raw ``feedback()`` differences (angles / landmarks / velocity) plus
the exercise name into a short, structured coaching report via an OpenRouter
chat model (tencent/hy3:free by default). Kept free of any Flask dependency.

The API key is read from the ``OPENROUTER_API_KEY`` environment variable. If it
is missing — or the call fails for any reason — ``generate_coaching`` returns
None so the caller can fall back to the plain per-joint feedback.
"""

import os
import json

from openai import OpenAI

from .similarity import format_feedback

BASE_URL = "https://openrouter.ai/api/v1"
MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")

# Reusable OpenAI client (created once, used across sessions)
_CLIENT = None

SYSTEM_PROMPT = (
    "You are an expert movement and strength coach. You are given the name of "
    "an exercise and a set of measured differences between a trainee's attempt "
    "and a reference performance. Sign convention: a positive value means the "
    "trainee's value is larger / faster, or positioned more to the right / "
    "lower than the reference. Tiny differences are sensor noise — ignore them "
    "and focus on what actually matters for this exercise. Be specific, "
    "practical and encouraging.\n\n"
    "Respond with ONLY a JSON object (no markdown, no commentary) using exactly "
    "these keys:\n"
    "{\n"
    '  "exercise": string,                 // the exercise name, tidied up\n'
    '  "major_errors": [string],           // 2-4 most important form problems, plain language\n'
    '  "tips": [string],                   // 2-4 concrete cues/fixes for those problems\n'
    '  "supporting_exercises": [string],   // 2-4 drills/accessory exercises that help\n'
    '  "quote": string                     // one short motivational line\n'
    "}"
)


def _build_user_prompt(exercise, feedback, scores=None):
    # Feed the LLM exactly what print_feedback logs to the console.
    parts = [format_feedback(feedback, exercise)]

    if scores:
        parts.append(
            "\nScores — overall {o}%, technique {t}%, posture {p}%".format(
                o=scores.get("overall_score", "?"),
                t=scores.get("angle_score", "?"),
                p=scores.get("landmark_score", "?"),
            )
        )

    parts.append(
        "\nGive coaching for this exercise as the JSON object described above."
    )
    return "\n".join(parts)


def _as_list(value):
    """Coerce an LLM field into a clean list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [str(value).strip()]


def _parse(content, exercise):
    """Extract the JSON object from the model's reply and normalise it."""
    text = content.strip()
    # Strip ```json ... ``` fences if present.
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    # Fall back to the outermost { ... } span.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]

    data = json.loads(text)
    return {
        "exercise": str(data.get("exercise") or exercise or "Your Movement"),
        "major_errors": _as_list(data.get("major_errors")),
        "tips": _as_list(data.get("tips")),
        "supporting_exercises": _as_list(data.get("supporting_exercises")),
        "quote": str(data.get("quote") or "You got this — keep going!"),
    }


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            return None
        _CLIENT = OpenAI(base_url=BASE_URL, api_key=api_key)
    return _CLIENT


def generate_coaching(exercise, feedback, scores=None):
    """Return a structured coaching dict, or None if the LLM is unavailable.

    ``feedback`` is the dict returned by ``similarity.feedback``.
    """
    client = _get_client()
    if client is None:
        print("[coach] OPENROUTER_API_KEY not set — skipping AI coaching.")
        return None

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(exercise, feedback, scores)},
            ],
            extra_body={"reasoning": {"enabled": True}},
        )
        content = response.choices[0].message.content or ""
        return _parse(content, exercise)
    except Exception as exc:  # noqa: BLE001 - never let coaching break scoring
        print(f"[coach] AI coaching failed: {exc}")
        return None
