# SCENE-DETECT — Intelligent Scene Recognition

A user-interface prototype for an AI-based situation-recognition system for the home, built as a university project at **OTH Amberg-Weiden**.

The long-term idea is a small home-assistant robot that watches over people at home, especially elderly people (60+). It recognises the situation in a room and gets a human helper when something looks wrong, such as a fall. This repository holds the **first prototype**. It is a Streamlit dashboard that shows the workflow and how the system would *explain* its decisions.

> [!IMPORTANT]
> **Most of this prototype does not run a real AI model.**
> It does not detect people, poses, emotions or falls. The scene demonstrations (sections A–B) are pre-written simulations for fixed scenarios, and the camera preview (section C) only shows the video.
> **The one exception is section G (find my phone):** it runs a real, pretrained object detector on the webcam image — locally, phone only.
> It is not a medical device and does not give a medical diagnosis.

---

## Table of contents

- [What the app shows](#what-the-app-shows)
- [The four system states](#the-four-system-states)
- [Demonstration scenarios](#demonstration-scenarios)
- [Reminders (section F)](#reminders-section-f)
- [Find my phone or keys (section G)](#find-my-phone-or-keys-section-g)
- [Getting started](#getting-started)
- [Using the app](#using-the-app)
- [Project structure](#project-structure)
- [Customising the content](#customising-the-content)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Design principles](#design-principles)

---

## What the app shows

The dashboard is a single page with seven sections:

| Section | What it does |
|---|---|
| **A · Main Dashboard** | Four cards that explain the states the system can report: `NO PERSON`, `NORMAL`, `CHECK PERSON`, `ALERT`. |
| **B · Scene Demonstration** | Pick one of seven predefined scenarios and press **Run demonstration**. You get a simulated result with a state, an illustrative score, body features, the evidence used, and a plain-language reason for the label. |
| **C · Camera Preview** *(optional)* | Shows your live webcam feed through OpenCV, marked "PREVIEW ONLY – NO AI DETECTION". It is there to show where a real camera would plug in. |
| **D · Event History** | A table of every demonstration you ran in this session (time, scenario, state, explanation), with a **Clear history** button. |
| **E · Future Features** | Planned features for later versions. None of them is implemented yet. |
| **F · Reminders** | Record a spoken reminder, which is transcribed offline, scheduled from the words, and announced aloud when due. |
| **G · Find My Phone or Keys** | **Real detection.** The webcam feeds a pretrained YOLO model that recognises a phone, notes its zone (sofa / table / floor) and remembers the last sighting. Ask "Where is my phone?" and the answer is shown and spoken. Keys cannot be detected yet, so their answer is an honest "not sure". |

## The four system states

These describe **software states, never a medical condition**.

| State | Colour | Meaning |
|---|---|---|
| `NO PERSON` | Grey | Nobody is visible. An empty room is not an emergency. |
| `NORMAL` | Green | A person is visible and doing something ordinary: standing, sitting, walking or resting. |
| `CHECK PERSON` | Amber | Something is unclear: the person is partly hidden, the view is bad, or the evidence is too weak. A human should take a look. |
| `ALERT` | Red | Strong evidence of an emergency, such as a sudden fall. In the real system this would notify a helper. |

## Demonstration scenarios

| Scenario | Resulting state | Simulated score |
|---|---|---|
| Empty room | `NO PERSON` | 0.97 |
| Person standing | `NORMAL` | 0.94 |
| Person sitting | `NORMAL` | 0.93 |
| Person walking | `NORMAL` | 0.92 |
| Person lying on the floor | `CHECK PERSON` | 0.71 |
| Possible fall | `ALERT` | 0.88 |
| Uncertain or partially hidden person | `CHECK PERSON` | 0.48 |

Note that *lying on the floor* gives `CHECK PERSON`, not `ALERT`. People lie down to sleep, rest or stretch, and a single moment in time cannot tell resting apart from a fall. Only a sudden change from upright to horizontal, followed by no movement, is treated as an alert.

## Reminders (section F)

Record a spoken reminder — for example *"remind me to take my medicine at 4pm"* or *"remind me to stretch in 30 minutes"*. The app:

1. transcribes your voice **offline** with Vosk (no cloud),
2. uses `dateparser` to separate the task from the trigger time,
3. saves it to `reminders.json` (so it survives a restart) and lists it in a table,
4. checks every 5 seconds and, when the time arrives, **speaks the reminder aloud** through your computer's speakers and marks it done.

If no time is found in the phrase, the reminder is saved as *unscheduled*. Together with section G, this is one of the two sections that perform real work — the rest of the dashboard is simulated.

## Find my phone or keys (section G)

Press **▶ Start finder**, hold your phone up to the webcam, put it away, and ask **📱 Where is my phone?**. The app:

1. reads the webcam at up to **10 frames per second**, each frame with a timestamp,
2. runs a **pretrained YOLO model** (`yolo11m`, COCO class *"cell phone"* — nothing was trained by us) on every 5th frame,
3. looks up which **zone** of the room the phone is in (*sofa*, *table*, *floor* — polygons in `config.yaml`),
4. **remembers the last sighting** (item, zone, time, confidence) and shows it in a *Last seen* card,
5. answers your question in a sentence that is **shown and spoken aloud** — e.g. *"Your phone was last seen on the table at 15:04."*

The answers are honest by design:

| Situation | Answer |
|---|---|
| Seen with good confidence (≥ 0.6) | *"Your phone was last seen on the table at 15:04."* |
| Seen, but the model was unsure (0.4–0.6) | *"I think your phone was last seen on the table at 15:04, but I'm not sure."* |
| Never seen / can't be detected (keys) | *"I have not seen your keys — I'm not sure where they are."* |
| Camera fails | Shows *"Camera lost"* — a broken camera never counts as "the item is gone". |

It is always the **last sighting**, not a promise that the item is still there. Keys are not a COCO class, so they cannot be detected yet (planned: an open-vocabulary detector).

**How it works inside.** The camera and the model run in a background thread, so the dashboard stays responsive; the page only shows the latest picture and the memory.

```
Webcam ─▶ FrameSource ─▶ ObjectDetector ─▶ ZoneMap ─▶ ItemMemory ─▶ UC2 answer ─▶ screen + voice
          camera.py       objects.py        zones.py   memory.py     usecases/lost_items.py
```

| Module (`lifeaware/`) | Job |
|---|---|
| `schema.py` | The shared data contract: every sighting is an `Observation` (time, source, label, confidence, zone); every answer is an `Event`. |
| `config.py` + `config.yaml` | All thresholds, the model name and the zones in one place, with safe defaults. |
| `camera.py` | `FrameSource`: webcam or video file, fps cap, timestamps, status `ok` / `lost` / `ended`. The camera preview in section C reuses its `open_webcam()`. |
| `objects.py` | `ObjectDetector`: loads YOLO **once** (with a warm-up), checks every n-th frame, turns boxes into observations. |
| `zones.py` | `ZoneMap`: pixel position → zone name. |
| `memory.py` | `ItemMemory`: last sighting per item, ignores weak detections, builds the answer sentence. |
| `usecases/lost_items.py` | Use case UC2: question → answer → `Event` → voice (reuses `reminders.speak`). |
| `item_finder.py` | The Streamlit section G itself (background worker + layout). Also shown standalone by `streamlit run demo_items.py`. |

**Speed on our laptop (CPU only):** about 5–8 frames per second on screen, ~0.3–0.7 s per model check. The first start takes 5–10 seconds while the model loads. We use the medium model because the nano model mistook a phone held up to the camera for a *laptop*.

---

## Getting started

### Requirements

- **Python 3.10–3.12** (the speech libraries don't yet ship wheels for 3.13+)
- A webcam *(optional; only needed for sections C and G)*
- No GPU, no cloud services and no paid APIs. Everything runs on a normal CPU.

Dependencies (from [`requirements.txt`](requirements.txt)):

| Package | Why |
|---|---|
| `streamlit` | The web UI |
| `pandas` | The event history table |
| `numpy` | Used by pandas and OpenCV |
| `opencv-python` | Webcam preview only. The app still works without it. |
| `vosk` | Offline speech-to-text for the reminders section. |
| `dateparser` | Extracts the trigger time ("at 4pm", "in 30 minutes") from a reminder. |
| `pyttsx3` | Speaks reminders aloud when they are due. |
| `pyyaml` | Reads `config.yaml` (thresholds, model, zones). |
| `ultralytics` | Pretrained YOLO object detection for section G (pulls in PyTorch, CPU is enough). The model file is downloaded into `models/` on first use. |
| `pytest` | Runs the automatic tests (development only). |

The reminders section also needs two things that are **not** pip packages:

- **A Vosk model** (~40 MB). Download one from <https://alphacephei.com/vosk/models> and unpack it into `models/` (the app looks for `models/vosk-model-small-en-us-0.15`). Override the path with the `VOSK_MODEL_PATH` environment variable.
- **`ffmpeg`** *(recommended)* for converting the text-to-speech output to WAV and decoding some browser audio formats. macOS: `brew install ffmpeg`. Linux: `sudo apt install ffmpeg espeak` (the `espeak` package provides the TTS voice). Windows: optional.

Section G (find my phone) needs **nothing extra**: the first time you press **Start finder**, the YOLO model `yolo11m.pt` (~40 MB) is downloaded automatically into `models/` (internet needed once).

**What you get after cloning** — models are never stored in git:

| | After `git clone` + `pip install` | You must do |
|---|---|---|
| Dashboard, sections A–E | ✅ works | nothing |
| Section F (reminders) | ⚠️ shows "Vosk model directory not found" | download the Vosk model (above) |
| Section G (find my phone) | ✅ works | nothing — model downloads on first start |
| Tests (`python -m pytest`) | ✅ works | nothing — no camera or model needed |

### 1. Clone the repository

> [!TIP]
> **Windows:** clone into a **short** folder such as `C:\Users\<you>\Life-Aware`, not deep inside OneDrive or Documents. PyTorch contains very long file paths, and Windows stops at 260 characters (see Troubleshooting).

```bash
git clone https://github.com/heniltalaviya2209/Life-Aware.git
cd Life-Aware
```

### 2. Create and activate a virtual environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> If PowerShell blocks the activation script, run
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

Streamlit opens the dashboard in your browser at **http://localhost:8501**. To stop the app, press `Ctrl + C` in the terminal.

### 5. Run the tests

```bash
python -m pytest
```

(`python -m pytest` also works where Windows Application Control blocks the
`pytest.exe` launcher.) The tests use fake data only — no camera, microphone or Vosk model is needed.
The same tests run automatically on GitHub (see `.github/workflows/tests.yml`).

### 6. Try the real camera → object → "where is my phone?" chain (optional)

```bash
python tools/zone_setup.py            # click to find zone corners; copy them into config.yaml
python tools/item_demo.py --show      # webcam for 20 s: boxes, zones, then the UC2 answer + fps
python tools/item_demo.py --source clip.mp4
```

For a one-page browser demo of the same chain: `streamlit run demo_items.py`.

The first run downloads the pretrained `yolo11m.pt` (~40 MB) into `models/`
(the model is set in `config.yaml`; the smaller `yolo11n.pt` missed a phone held up
to our webcam).
Only the COCO class "cell phone" is detected for now; keys are not a COCO class,
so the answer for keys is always an honest "not sure".

---

## Using the app

1. **Read the four cards** in section A to learn what each state means.
2. In section B, **choose a scenario** from the dropdown and click **▶ Run demonstration**. The result panel shows:
   - the resulting **state** (colour-coded),
   - an **illustrative score** (simulated, not a real confidence value),
   - the **simulated body features** (orientation and motion),
   - the **evidence** used and **why this label was chosen**.
3. *(Optional)* In section C, click **Start camera preview** to see your webcam, and **Stop camera preview** to turn it off.
4. Check section D to see the **history** of all demonstrations you ran. Click ** Clear history** to reset it.
5. In section F, click **Record a reminder** and speak a phrase with a time. It appears in the reminders table, and when the time arrives the app announces it aloud and marks it completed.
6. In section G, click **Start finder** (first start takes a few seconds), hold your phone up to the camera until a red box appears, hide it, then click **📱 Where is my phone?**. Only one section can use the webcam: starting G stops the preview in C, and vice versa.

The history lives in memory only. It is lost when you refresh the page or restart the app. Reminders, however, are written to `reminders.json` and persist across restarts.

---

## Project structure

```
SCENE-DETECT/
├── app.py               # The Streamlit dashboard (config, styles, UI, camera preview)
├── reminders.py         # Reminder logic: transcription, parsing, storage, speech
├── demo_items.py        # Section G as a standalone page (streamlit run demo_items.py)
├── config.yaml          # Thresholds, model name and room zones for the real chain
├── lifeaware/           # The real (non-simulated) chain — see section G above
│   ├── schema.py  config.py  camera.py  objects.py  zones.py  memory.py  item_finder.py
│   └── usecases/lost_items.py      # UC2: find lost items
├── tools/
│   ├── zone_setup.py    # Click on a camera image to find zone coordinates
│   └── item_demo.py     # Run the chain in a terminal window and print fps
├── tests/               # pytest tests (fake data only: no camera, mic or model download)
├── .github/workflows/tests.yml   # Runs the tests on GitHub for every push
├── pytest.ini
├── requirements.txt     # Python dependencies
├── models/              # Vosk speech model + YOLO weights (downloaded, gitignored)
├── reminders.json       # Saved reminders (gitignored)
├── docs/                # Project documentation
│   ├── SCENE-DETECT_Project_Definition.pdf     # Project definition
│   ├── SCENE-DETECT_Project_Definition.docx    # Same, editable Word version
│   ├── SCENE-DETECT_Analysis_and_Assumptions.pdf
│   └── Standard_mp3_auto_transcript.txt        # Auto-transcript of the project kick-off meeting
├── .gitignore
└── README.md
```

### How `app.py` is organised

| Part | Contents |
|---|---|
| **Configuration** | `SITUATIONS` (the four states), `SCENARIOS` (the seven demos), `FUTURE_FEATURES` and all UI texts. |
| **Styles** | Custom CSS for the cards, chips and result panel, with a consistent grey / green / amber / red colour scheme. |
| **Session state** | Keeps the history, the last result, the camera status and the reminders between Streamlit reruns. |
| **Camera helpers** | Opens the webcam (with a Windows `CAP_DSHOW` fallback) and renders **one frame per rerun**, so the Stop button always stays responsive. If it can't open the camera, it shows a friendly error message instead of crashing. |
| **Page** | Renders sections A to F, section G (one call: `render_item_finder()` from `lifeaware/item_finder.py`) and the footer. |

OpenCV is imported **lazily**, only when the camera is started. If it isn't installed, everything except the camera preview still works.

---

## Customising the content

All texts and simulated data are at the top of `app.py`, so you can change the demo without touching the UI code:

- **Add a scenario:** add a new entry to the `SCENARIOS` dictionary with `state`, `score`, `features`, `evidence` and `explanation`. The `state` must be one of the keys in `SITUATIONS`.
- **Change a state description:** edit `SITUATIONS`.
- **Change the roadmap cards:** edit `FUTURE_FEATURES`.

The real chain (section G) is configured in **`config.yaml`** instead — no code changes needed:

- **Room zones:** run `python tools/zone_setup.py`, click the corners of an area on the camera image, and copy the printed points into `zones`.
- **Model and speed:** `objects.model` (e.g. `models/yolo11n.pt` is faster but less accurate) and `objects.every_n_frames`.
- **How sure is "sure":** `objects.min_conf` (ignore weaker detections) and `items.sure_conf` (below this the answer says *"I'm not sure"*).

Streamlit reloads the app automatically when you save the file.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `streamlit: command not found` | The virtual environment is not active. Activate it (step 2), or run `python -m streamlit run app.py`. |
| "No camera could be opened" | Check that a webcam is connected and not being used by another app (Teams, Zoom, …). On Windows, check **Settings → Privacy & security → Camera** and allow desktop apps to use the camera. |
| "OpenCV is not installed" | Run `pip install opencv-python`, or ignore it. The simulated demos work without it. |
| DLL / import errors for pandas or numpy on Windows | Keep the versions pinned in `requirements.txt` (`pandas<3.0`, `numpy<2.3`). Newer builds were blocked by a Windows Code Integrity policy during development. |
| "Vosk model directory not found" | Download a model from https://alphacephei.com/vosk/models and unpack it to `models/` (see Requirements). |
| Reminder doesn't play aloud | macOS uses the built-in `say` command; Linux needs `espeak`. If the browser replay control is silent, install `ffmpeg`. |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502`. |
| Section G: *"Camera lost"* | Another program holds the webcam — the camera preview in section C, a second Streamlit page, Teams or Zoom. Close it and press **▶ Start finder** again. |
| Section G: no red box around the phone | Use good light and hold the phone 30–50 cm from the camera, screen or back fully visible. Wait ~1 s — only every 5th frame is checked. |
| Section G: first start is slow | Normal: loading PyTorch and the model takes 5–10 s the first time. Later starts are quicker. |
| `pytest.exe` is blocked ("Application Control policy") | Run `python -m pytest` instead. |
| `pip install` fails with *"Could not install packages due to an OSError … Long Path support"* (Windows) | The project folder path is too long for PyTorch's files. Either move/clone the project to a short path (e.g. `C:\Users\<you>\Life-Aware`), delete `.venv` and install again — or enable long paths once (PowerShell as admin: `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force`, then reboot). |
| Section G says the model could not be downloaded | The first start needs internet to fetch `yolo11m.pt`. Alternatively copy `yolo11m.pt` from a teammate into `models/`. |

---

## Roadmap

Planned for later versions. **None of this is implemented yet.**

- 🧍 **Real-time human pose detection.** Estimate body key points (shoulders, hips, knees, ankles) and classify standing, sitting, walking, bending and lying.
- 🧎 **Fall and inactivity detection.** Watch movement across several frames and only raise an alert when strong evidence persists.
- 🔊 **Voice interaction.** Ask the person *"Are you okay?"* and listen for an answer before escalating.
- 📞 **Human assistance alerts.** Notify a caregiver or emergency contact with the scene details.
- 🔑 **Finding keys and glasses.** The phone finder already works (section G); keys and glasses are not in the standard model's classes and need an open-vocabulary detector.
- 🎵 **Favourite music playback.** Play the person's preferred music on request.
- 💬 **Emotion-aware conversation** *(optional)*. Adapt the assistant's tone to the person's mood, but only if this proves reliable.

The target platform is a small home robot that streams camera data to a laptop, where the heavy processing runs.

---

## Design principles

- **Honest uncertainty.** When the evidence is weak, the system says `CHECK PERSON` instead of guessing.
- **No false emergencies from normal behaviour.** Sitting or lying down is never an alert by itself.
- **Explainable decisions.** Every result shows the evidence and the reasoning behind it.
- **Human in the loop.** The system calls for a human. It does not replace one.
- **Privacy-friendly and low-cost.** It runs locally on a CPU, with no cloud and no paid APIs.

---

*SCENE-DETECT prototype. A university project at OTH Amberg-Weiden. Simulated scene demonstrations; real phone detection in section G.*
