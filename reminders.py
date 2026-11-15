"""Reminder helpers for SCENE-DETECT.

Modular support for the "Reminders" dashboard section:
- JSON persistence (backed by a local file)
- offline speech-to-text (Vosk, 16 kHz mono PCM)
- natural-language trigger-time parsing (dateparser)
- text-to-speech for announcements (pyttsx3 -> audio bytes)

Heavy dependencies (vosk, dateparser, pyttsx3) are imported lazily inside the
functions that need them, so the rest of the app still loads even if a package
or the Vosk model directory is missing.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import wave
from datetime import datetime, timedelta
from typing import Optional

# NOTE: numpy is imported lazily inside the audio-decode helpers so this module
# (and its parser) can be imported and unit-tested without numpy installed.
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Local directory containing an unpacked Vosk model (e.g.
# "vosk-model-small-en-us-0.15"). Download a model from
# https://alphacephei.com/vosk/models and unpack it under ./models/.
# Override at runtime with the VOSK_MODEL_PATH environment variable.
VOSK_MODEL_PATH = os.environ.get(
    "VOSK_MODEL_PATH", "models/vosk-model-small-en-us-0.15"
)

# JSON file used to persist reminders between restarts.
REMINDERS_FILE = os.environ.get("REMINDERS_FILE", "reminders.json")

# Vosk expects 16 kHz mono 16-bit PCM.
VOSK_SAMPLE_RATE = 16000

# Lazy singletons (module-level, cached across Streamlit reruns in a process).
_VOSK_MODEL = None
_VOSK_MODEL_PATH_LOADED = None

# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------


def load_reminders(path: str = REMINDERS_FILE) -> list[dict]:
    """Read reminders from the JSON file; return [] on any failure."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def save_reminders(reminders: list[dict], path: str = REMINDERS_FILE) -> None:
    """Write reminders to the JSON file atomically (write temp, then replace)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(reminders, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Audio decoding -> 16 kHz mono 16-bit PCM (required by Vosk)
# ---------------------------------------------------------------------------


def _to_bytes(value) -> bytes:
    """Normalise the st.audio_input value (an UploadedFile / BytesIO) to bytes."""
    if value is None:
        return b""
    getter = getattr(value, "getvalue", None)
    if callable(getter):
        return getter()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    return bytes(value)


def _wav_to_float_mono(raw: bytes):
    """Decode a WAV file to a float32 mono array in [-1, 1] plus its sample rate."""
    import numpy as np

    with wave.open(io.BytesIO(raw), "rb") as w:
        channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        frames = w.readframes(w.getnframes())

    if sampwidth == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 3:
        b = np.frombuffer(frames, dtype=np.uint8)
        pcm = (
            b[0::3].astype(np.int32)
            | (b[1::3].astype(np.int32) << 8)
            | (b[2::3].astype(np.int32) << 16)
        )
        pcm = (pcm << 8) >> 8  # sign-extend 24-bit
        data = pcm.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sampwidth}")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1).astype(np.float32)
    return data, framerate


def _resample(data, orig_rate: int, target_rate: int):
    """Linear-interpolation resample of a 1-D signal (good enough for speech)."""
    import numpy as np

    if orig_rate == target_rate or data.size == 0:
        return data
    n = data.shape[0]
    num = max(1, int(round(n * target_rate / float(orig_rate))))
    x_old = np.linspace(0.0, 1.0, n, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num, endpoint=False)
    return np.interp(x_new, x_old, data).astype(np.float32)


def _ffmpeg_to_pcm(raw: bytes, sample_rate: int = VOSK_SAMPLE_RATE) -> np.ndarray:
    """Decode a non-WAV container (webm/ogg/mp4) to 16 kHz mono s16 via ffmpeg."""
    import numpy as np

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required to decode this audio format. Install it "
            "(e.g. `brew install ffmpeg`) or record in a WAV-compatible browser."
        )
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "input.audio")
        dst = os.path.join(tmp, "out.pcm")
        with open(src, "wb") as fh:
            fh.write(raw)
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", src, "-ar", str(sample_rate), "-ac", "1", "-f", "s16le", dst,
            ],
            check=True,
            capture_output=True,
        )
        with open(dst, "rb") as fh:
            pcm = fh.read()
    return np.frombuffer(pcm, dtype=np.int16)


def decode_audio_to_pcm16_mono(raw, target_rate: int = VOSK_SAMPLE_RATE) -> np.ndarray:
    """Convert recorded audio to a 16 kHz mono int16 NumPy array for Vosk.

    `st.audio_input` returns audio/wav; this decodes WAV directly (stdlib
    `wave`) and falls back to ffmpeg for other containers. Sample rate and
    channel count are normalised regardless of the browser's microphone
    settings.
    """
    raw = _to_bytes(raw)
    if not raw:
        raise ValueError("No audio was recorded — please try again.")

    import numpy as np

    try:
        data, orig_rate = _wav_to_float_mono(raw)
    except (wave.Error, EOFError, ValueError, RuntimeError):
        return _ffmpeg_to_pcm(raw, target_rate)

    if data.size == 0:
        raise ValueError("The recording was empty — please try again.")

    data = _resample(data, orig_rate, target_rate)
    pcm = np.clip(np.round(data * 32767.0), -32768, 32767).astype(np.int16)
    return np.ascontiguousarray(pcm)


# ---------------------------------------------------------------------------
# Offline speech-to-text (Vosk)
# ---------------------------------------------------------------------------


def get_vosk_model():
    """Return a cached vosk.Model, raising a friendly error if it is missing."""
    global _VOSK_MODEL, _VOSK_MODEL_PATH_LOADED

    if _VOSK_MODEL is not None and _VOSK_MODEL_PATH_LOADED == VOSK_MODEL_PATH:
        return _VOSK_MODEL

    import vosk  # lazy import

    if not os.path.isdir(VOSK_MODEL_PATH):
        raise FileNotFoundError(
            f"Vosk model directory not found: {VOSK_MODEL_PATH!r}. Download a model "
            "from https://alphacephei.com/vosk/models, unpack it to ./models/, and "
            "make sure the folder name matches (or set the VOSK_MODEL_PATH "
            "environment variable to the model directory)."
        )

    _VOSK_MODEL = vosk.Model(VOSK_MODEL_PATH)
    _VOSK_MODEL_PATH_LOADED = VOSK_MODEL_PATH
    return _VOSK_MODEL


def transcribe_pcm(pcm: np.ndarray) -> str:
    """Run Vosk over 16 kHz mono int16 PCM and return the recognised text."""
    if pcm is None or getattr(pcm, "size", 0) == 0:
        return ""

    import numpy as np
    import vosk  # lazy import

    model = get_vosk_model()
    rec = vosk.KaldiRecognizer(model, VOSK_SAMPLE_RATE)
    rec.AcceptWaveform(np.ascontiguousarray(pcm, dtype=np.int16).tobytes())
    try:
        text = json.loads(rec.FinalResult()).get("text", "")
    except (ValueError, TypeError):
        text = ""
    return (text or "").strip()


# ---------------------------------------------------------------------------
# Natural-language reminder parsing (dateparser)
# ---------------------------------------------------------------------------

_COMMAND_PREFIXES = (
    "please remind me to",
    "please remind me",
    "remind me to",
    "remind me about",
    "remind me",
    "set a reminder to",
    "set a reminder for",
    "set a reminder",
    "reminder to",
    "reminder for",
    "reminder",
)


def _strip_command(text: str) -> str:
    lower = text.lower()
    for prefix in sorted(_COMMAND_PREFIXES, key=len, reverse=True):
        if lower.startswith(prefix):
            rest = text[len(prefix):].lstrip(" ,;:")
            while rest.lower().startswith("to "):
                rest = rest[3:].lstrip()
            return rest
    return text


def _clean_task(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" ,;:.!?")
    if text.lower().startswith("to "):
        text = text[3:].strip()
    return text


def parse_reminder(text: str, now: Optional[datetime] = None):
    """Split a reminder into (task, scheduled_at).

    Returns ``scheduled_at`` as an aware-free ``datetime``, or ``None`` when no
    time could be parsed (the caller treats it as unscheduled).
    """
    import dateparser  # lazy import
    from dateparser.search import search_dates  # submodule needs an explicit import

    text = (text or "").strip()
    if now is None:
        now = datetime.now()
    if not text:
        return "", None

    task = _strip_command(text)
    settings = {
        "RELATIVE_BASE": now,
        "PREFER_DATES_FROM": "future",
        "RETURN_AS_TIMEZONE_AWARE": False,
    }

    scheduled = None
    phrase = None

    try:
        matches = search_dates(task, languages=["en"], settings=settings) or []
    except Exception:
        matches = []

    if matches:
        phrase, scheduled = matches[-1]
    else:
        try:
            scheduled = dateparser.parse(task, languages=["en"], settings=settings)
            if scheduled is not None:
                phrase = task
        except Exception:
            scheduled = None

    if scheduled is not None:
        if phrase and phrase in task:
            task = task.replace(phrase, "", 1)
        task = _clean_task(task)
        # A bare time that has already passed today -> push to tomorrow.
        if scheduled <= now:
            scheduled = scheduled + timedelta(days=1)
    else:
        task = _clean_task(task)

    return task.strip(), scheduled


# ---------------------------------------------------------------------------
# Date helpers for storage / display
# ---------------------------------------------------------------------------


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def to_iso(dt: datetime) -> str:
    return dt.isoformat(timespec="minutes")


def parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def format_scheduled(value: Optional[str]) -> str:
    dt = parse_iso(value)
    if dt is None:
        return "—"
    delta = dt - datetime.now()
    minutes = int(round(delta.total_seconds() / 60.0))
    if 0 <= minutes <= 60:
        return "now" if minutes == 0 else f"in {minutes} min"
    return dt.strftime("%a %d %b %Y, %H:%M")


# ---------------------------------------------------------------------------
# Text-to-speech (pyttsx3 / OS `say` -> audible announcement + WAV bytes)
# ---------------------------------------------------------------------------

# pyttsx3's macOS driver (NSSpeechSynthesizer) only runs on the main thread,
# but Streamlit runs the script in a worker thread — in-process synthesis
# there yields empty audio. We run pyttsx3 in a SUBPROCESS instead (which gets
# its own main thread), and on macOS use the built-in `say` command for the
# audible announcement.

_SYNTH_SAY_SCRIPT = (
    "import sys, pyttsx3\n"
    "e = pyttsx3.init()\n"
    "e.say(sys.argv[1])\n"
    "e.runAndWait()\n"
)

_SYNTH_SAVE_SCRIPT = (
    "import sys, pyttsx3\n"
    "e = pyttsx3.init()\n"
    "e.save_to_file(sys.argv[1], sys.argv[2])\n"
    "e.runAndWait()\n"
)


def _convert_to_wav(data: bytes) -> Optional[bytes]:
    """Convert arbitrary audio bytes to WAV using ffmpeg; None if unavailable."""
    if shutil.which("ffmpeg") is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.audio")
        dst = os.path.join(tmp, "out.wav")
        with open(src, "wb") as fh:
            fh.write(data)
        try:
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src, dst],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            return None
        with open(dst, "rb") as fh:
            return fh.read()


def speak(text: str) -> None:
    """Announce `text` aloud through the machine's speakers (non-blocking).

    Uses the built-in `say` command on macOS and pyttsx3 (in a subprocess)
    elsewhere. This runs outside the Streamlit worker thread so macOS TTS works.
    """
    text = (text or "").strip()
    if not text:
        return
    if sys.platform == "darwin" and shutil.which("say"):
        subprocess.Popen(
            ["say", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    else:
        subprocess.Popen(
            [sys.executable, "-c", _SYNTH_SAY_SCRIPT, text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def text_to_audio(text: str):
    """Synthesise `text` to WAV bytes for a browser player; returns (bytes, mime).

    Runs pyttsx3 in a subprocess (see note above). On macOS pyttsx3 writes
    AIFF, which is converted to WAV when ffmpeg is available.
    """
    text = (text or "").strip()
    if not text:
        return b"", "audio/wav"

    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
        path = tmp.name
    try:
        subprocess.run(
            [sys.executable, "-c", _SYNTH_SAVE_SCRIPT, text, path],
            check=True,
            capture_output=True,
        )
        with open(path, "rb") as fh:
            data = fh.read()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return data, "audio/wav"
    if data[:4] == b"FORM" and data[8:12] in (b"AIFF", b"AIFC"):
        wav = _convert_to_wav(data)
        if wav is not None:
            return wav, "audio/wav"
        return data, "audio/aiff"
    return data, "audio/wav"
