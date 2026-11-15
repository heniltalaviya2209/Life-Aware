# AGENTS.md — SCENE-DETECT prototype

Compact, repo-specific guidance for OpenCode sessions. Each line answers "would an agent miss this without help?" — everything else is left out.

## What this is

A single-file **Streamlit UI prototype** for an AI situation-recognition system (OTH Amberg-Weiden university project). It is **simulation-only**: no real AI model, no person/pose/emotion/fall detection, all results pre-written. Not a medical device.

## Quickstart

```bash
source .venv/bin/activate          # macOS/Linux  (Windows: .\.venv\Scripts\Activate.ps1)
pip install -r requirements.txt
streamlit run app.py               # http://localhost:8501
```

- `streamlit: command not found` → venv not active, or run `python -m streamlit run app.py`.
- No `.venv` in repo — create it first (`python -m venv .venv`).

## Project structure

- `app.py` — **the entire app**: config, styles, session state, camera helpers, page (sections A–F). No other source files.
- `reminders.py` — helper module for section F (Vosk transcription, dateparser parsing, JSON storage, pyttsx3 TTS). Heavy deps imported lazily.
- `requirements.txt` — streamlit, pandas, numpy, opencv-python + vosk, dateparser, pyttsx3.
- `docs/` — project docs (PDFs, Word, meeting transcript). Not needed to run the app.
- `README.md` — canonical prose docs (state/scenario tables, setup, troubleshooting). Prefer it over re-deriving.

## Reminders section (F) — setup

- **Vosk model is not bundled**: download one from https://alphacephei.com/vosk/models and unpack to `./models/` (default path `models/vosk-model-small-en-us-0.15`). Override via the `VOSK_MODEL_PATH` env var. Missing model shows a friendly error, not a crash.
- `reminders.json` (persistence) and `models/` are gitignored.
- **Python 3.10–3.12 recommended** — `vosk`/`pyttsx3` wheels lag on 3.13+; `pandas<3.0`/`numpy<2.3` are pinned for Windows.
- **`ffmpeg` recommended** (not a pip dep): used to decode non-WAV mic audio and to convert pyttsx3's AIFF output to WAV on macOS. Without it, macOS announcements fall back to `audio/aiff`.

## Customising content — edit the dicts at the top of `app.py`

All UI text and simulated data is data, not code. No UI changes needed to customise:

| Want to | Where |
|---|---|
| Add/change a demo scenario | `SCENARIOS` — keys: `state`, `score`, `features`, `evidence`, `explanation`. `state` must be a key of `SITUATIONS`. |
| Change a state description | `SITUATIONS` — keys: `tag`, `tone`, `description`. |
| Change roadmap cards | `FUTURE_FEATURES` — `(emoji, title, text)` tuples. |

The four states are `NO PERSON`, `NORMAL`, `CHECK PERSON`, `ALERT`; there are seven scenarios. Streamlit auto-reloads on save.

## Gotchas

- **OpenCV is imported lazily** inside `_show_camera_frame()` — the app runs without `opencv-python`; only the camera preview fails.
- **Camera renders one frame per script run** (`st.rerun()` loop) — deliberate, keeps the Stop button responsive. Don't "fix" it to a tight loop.
- **History is `st.session_state` only** — lost on refresh/restart.
- **Pinned deps matter**: `pandas<3.0`, `numpy<2.3` (newer builds blocked by Windows Code Integrity). Don't bump casually.
- **Windows camera** uses a `CAP_DSHOW` fallback in `_open_capture()`.

## Troubleshooting

| Problem | Fix |
|---|---|
| "OpenCV is not installed" | `pip install opencv-python`, or ignore — simulated demos still work. |
| "No camera could be opened" | Check webcam + privacy settings (Windows: Settings → Privacy → Camera); another app (Teams/Zoom) may hold it. |
| Port 8501 in use | `streamlit run app.py --server.port 8502`. |
