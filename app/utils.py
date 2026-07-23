"""Small shared helpers: file paths, validation, captcha image, JSON/npy IO,
and video trimming via OpenCV."""

import os
import random
import subprocess
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from werkzeug.utils import secure_filename

from config import Config


# --------------------------------------------------------------------------
# File helpers
# --------------------------------------------------------------------------
def ext_of(filename):
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def classify_upload(filename):
    """Return 'video', 'image', or None based on the extension."""
    ext = ext_of(filename)
    if ext in Config.ALLOWED_VIDEO_EXT:
        return "video"
    if ext in Config.ALLOWED_IMAGE_EXT:
        return "image"
    return None


def unique_name(original, prefix=""):
    """Collision-resistant, secure filename: <prefix><ts>_<safe-original>."""
    from datetime import datetime
    ts = int(datetime.now().timestamp() * 1000)
    return f"{prefix}{ts}_{secure_filename(original)}"


def resolve_under(directory, stored):
    """Absolute path for a DB-stored file, anchored to ``directory``.

    Files are stored in the database by *name only* so the database stays
    portable across machines/checkouts (absolute paths from one PC don't exist
    on another). This rebuilds the absolute path against the given runtime
    directory. Legacy rows that still hold a full path are tolerated by taking
    their basename. Returns None for empty input.
    """
    if not stored:
        return None
    return os.path.join(directory, os.path.basename(stored))


def save_vectors(vectors, dest_path):
    """Persist a numpy feature array to disk; return the path."""
    np.save(dest_path, vectors)
    return dest_path


def load_vectors(path):
    return np.load(path, allow_pickle=False)


def get_video_duration(video_path):
    """Return duration in seconds using ffprobe (fast) or OpenCV fallback."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:  # noqa: BLE001
        pass

    # Fallback: OpenCV
    import cv2
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        return count / fps if fps else 0.0
    finally:
        cap.release()


def trim_video(input_path, output_path, start_sec, end_sec):
    """Trim a video to [start_sec, end_sec] and normalise it for playback.

    ``start_sec`` and ``end_sec`` are float seconds. When the whole clip is
    selected the file is copied as-is (lossless, no re-encode). When an actual
    sub-range is requested the clip is **re-encoded** to H.264 + yuv420p MP4 so
    the cut is frame-accurate: a plain ffmpeg stream-copy (``-c copy``) can only
    cut on keyframes, which for most clips yields a segment with no usable
    keyframe — a black/undecodable video that trips "No person detected" and
    won't play in the browser.

    Returns the path actually written, which may differ in extension from
    ``output_path`` (the ffmpeg re-encode and the OpenCV fallback both emit
    ``.mp4``, so e.g. a ``.webm`` recording comes back as ``.mp4``). Callers
    must use the returned path. Raises if no trimmed file could be produced.
    """
    duration = get_video_duration(input_path)
    start_sec = max(0.0, start_sec)

    if duration > 0:
        end_sec = min(duration, end_sec) if end_sec > 0 else duration
        if start_sec <= 0 and end_sec >= duration:
            # Whole clip selected — nothing to trim, copy as-is (no re-encode).
            import shutil
            shutil.copy2(input_path, output_path)
            return output_path
    # If the duration couldn't be measured (common for MediaRecorder .webm
    # blobs, which carry no duration metadata) we don't bail to a full copy —
    # that would store the untrimmed clip. We trust the requested ``end_sec``
    # and trim by frames in the OpenCV fallback below.

    # Prefer ffmpeg: re-encode (NOT -c copy) so the cut is frame-accurate and
    # the output is a clean H.264/yuv420p MP4 that OpenCV can decode and the
    # browser can play. -ss before -i is a fast, accurate input seek; -t gives
    # the window length. Audio is dropped (-an) — it's irrelevant to pose
    # analysis and avoids audio-encoder edge cases.
    out_mp4 = os.path.splitext(output_path)[0] + ".mp4"
    ffmpeg_paths = ["ffmpeg", r"C:\ffmpeg\bin\ffmpeg.exe", "/usr/bin/ffmpeg"]
    for ffmpeg in ffmpeg_paths:
        try:
            cmd = [ffmpeg, "-y"]
            if start_sec > 0:
                cmd += ["-ss", str(start_sec)]
            cmd += ["-i", input_path]
            if end_sec and end_sec > start_sec:
                cmd += ["-t", str(end_sec - start_sec)]
            cmd += [
                "-threads", "1", "-preset", "ultrafast",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-an", out_mp4,
            ]
            result = subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300,
            )
            if result.returncode == 0 and os.path.exists(out_mp4):
                return out_mp4
        except Exception:  # noqa: BLE001
            continue

    # Fallback: OpenCV frame-by-frame trim (slower but always available).
    # Always write an .mp4 — the mp4v codec can't be muxed into other
    # containers (e.g. a .webm recording), which would produce a broken file.
    import cv2
    output_path = os.path.splitext(output_path)[0] + ".mp4"
    cap = cv2.VideoCapture(input_path)
    out = None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps) if end_sec > 0 else total

        # Encode with mp4v. This matches the skeleton renderer's codec.
        for codec in ("mp4v",):
            candidate = cv2.VideoWriter(
                output_path, cv2.VideoWriter_fourcc(*codec), fps, (width, height)
            )
            if candidate.isOpened():
                out = candidate
                break
            candidate.release()

        if out is None:
            raise RuntimeError("Could not open a VideoWriter for trimming.")

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if start_frame <= frame_idx < end_frame:
                out.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        if out is not None:
            out.release()
    return output_path


# --------------------------------------------------------------------------
# Captcha
# --------------------------------------------------------------------------
_CAPTCHA_CHARS = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"


def generate_captcha_text(length=5):
    return "".join(random.choices(_CAPTCHA_CHARS, k=length))


def _load_captcha_font(size=36):
    for font_name in (
        "arial.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(font_name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def render_captcha_image(text):
    """Return a BytesIO PNG of a noisy, rotated captcha."""
    width, height = 180, 60
    img = Image.new("RGB", (width, height), (17, 24, 39))
    draw = ImageDraw.Draw(img)
    font = _load_captcha_font(36)

    for i, char in enumerate(text):
        char_img = Image.new("RGBA", (50, 60), (0, 0, 0, 0))
        char_draw = ImageDraw.Draw(char_img)
        color = (
            random.randint(220, 255),
            random.randint(120, 180),
            random.randint(40, 90),
        )
        char_draw.text((5, 8), char, font=font, fill=color)
        rotated = char_img.rotate(random.randint(-25, 25), resample=Image.BICUBIC)
        img.paste(rotated, (15 + i * 30, 0), rotated)

    for _ in range(6):
        draw.line(
            [
                (random.randint(0, width), random.randint(0, height)),
                (random.randint(0, width), random.randint(0, height)),
            ],
            fill=(75, 85, 99),
            width=1,
        )

    for _ in range(180):
        draw.point(
            (random.randint(0, width - 1), random.randint(0, height - 1)),
            fill=(107, 114, 128),
        )

    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf