# SCENE-DETECT — Intelligent Scene Recognition

A user-interface prototype for an AI-based situation-recognition system for the home, built as a university project at **OTH Amberg-Weiden**.

The long-term idea is a small home-assistant robot that watches over people at home, especially elderly people (60+). It recognises the situation in a room and gets a human helper when something looks wrong, such as a fall. This repository holds the **first prototype**. It is a Streamlit dashboard that shows the workflow and how the system would *explain* its decisions.

> [!IMPORTANT]
> **Parts of this prototype are simulated, parts are real.**
> The scene demonstrations (sections A–B) are pre-written simulations for fixed scenarios, and the camera preview (section C) only shows the video.
> **Two sections run real, pretrained models locally:** section G (find my phone, object detection) and section H (fall check, pose detection). In section H the person's answers still come from buttons (simulated speech).
> It does not recognise emotions. It is not a medical device, does not give a medical diagnosis and does not call emergency services.

---

## Table of contents

- [What the app shows](#what-the-app-shows)
- [The four system states](#the-four-system-states)
- [Demonstration scenarios](#demonstration-scenarios)
- [Reminders (section F)](#reminders-section-f)
- [Find my phone or keys (section G)](#find-my-phone-or-keys-section-g)
- [Fall check (section H)](#fall-check-section-h)
- [Getting started](#getting-started)
- [Using the app](#using-the-app)
- [Project structure](#project-structure)
- [Customising the content](#customising-the-content)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Design principles](#design-principles)

---

## What the app shows

The dashboard is a single page with eight sections:

| Section | What it does |
|---|---|
| **A · Main Dashboard** | Four cards that explain the states the system can report: `NO PERSON`, `NORMAL`, `CHECK PERSON`, `ALERT`. |
| **B · Scene Demonstration** | Pick one of seven predefined scenarios and press **Run demonstration**. You get a simulated result with a state, an illustrative score, body features, the evidence used, and a plain-language reason for the label. |
| **C · Camera Preview** *(optional)* | Shows your live webcam feed through OpenCV, marked "PREVIEW ONLY – NO AI DETECTION". It is there to show where a real camera would plug in. |
| **D · Event History** | A table of every demonstration you ran in this session (time, scenario, state, explanation), with a **Clear history** button. |
| **E · Future Features** | Planned features for later versions. None of them is implemented yet. |
| **F · Reminders** | Record a spoken reminder, which is transcribed offline, scheduled from the words, and announced aloud when due. |
| **G · Find My Phone or Keys** | **Real detection.** The webcam feeds a pretrained YOLO model that recognises a phone, notes its zone (sofa / table / floor) and remembers the last sighting. Ask "Where is my phone?" and the answer is shown and spoken. Keys cannot be detected yet, so their answer is an honest "not sure". |
| **H · Fall Check** | **Real pose detection.** A pretrained YOLO pose model decides standing / sitting / lying. If someone lies on the floor without moving, the system asks *"Are you okay?"*; no answer → local alarm. A broken camera or microphone gives `UNKNOWN`, never an alarm. Answers come from buttons for now. |

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

## Fall check (section H)

Use case **UC3**. Press **▶ Start fall check** (the webcam takes up to ~30 s to open on some Windows laptops), make sure the camera sees the **whole body**, then lie down on the floor and keep still.

**What happens, step by step** (timers from `config.yaml` → `fall:`):

| Situation | State | What the system does |
|---|---|---|
| Nothing unusual | `NORMAL` | Watches. |
| Lying on the **floor** for ≥ 3 s | `POSSIBLE_EVENT` | Keeps watching for movement. |
| … and hardly moving for ≥ 10 s | `CHECKING` | Asks *"Are you okay? Please say yes or no."* (spoken aloud). |
| Answer **yes**, or the person gets up | `SOLVED` | Does not ask again for 60 s. |
| Answer **no** or **help** | `ALARM` | Local alarm (spoken + red card) until a helper presses **Acknowledge**. |
| No answer for 15 s, camera **and** microphone working | `ALARM` | Same as above. |
| Camera lost, microphone off, reply not understood, or person out of view | `UNKNOWN` | Grey card: *"a helper should take a look"*. **Never an alarm.** |

Normal look-alike activities do **not** lead to a question: lying on the **sofa** (or any zone that is not `floor`), lying down only briefly, sitting, or moving around on the floor (e.g. exercises). "Nothing seen" is never treated as "not moving", and a person cut off by the image edge counts as `unclear`, never as lying.

**The page shows:** the live camera with zones and skeleton, what is seen right now (*"lying (floor)"*), frames per second and time per check; a colour-coded **state card** with the plain-language reason for the last change; the answer buttons ✅ **Yes** · ❌ **No** · 🆘 **Help** · ❔ **Not understood** · 🙋 **Acknowledge**; toggles 🔊 *Speak aloud* and 🎤 *Microphone works* (off → a missing answer gives `UNKNOWN`); and the **event history**, which is also appended to `events/fall_events.csv` (gitignored).

> The answer buttons stand in for the speech recogniser until the audio part is connected. Results from them are **simulated answers** and must be reported as such.

**How it works inside.** Camera, pose model and fall check run in a **background thread**; the page only shows the latest picture, the state and the event history.

```
Webcam ─▶ FrameSource ─▶ PoseDetector ─▶ ZoneMap ─▶ FallCheck ─▶ state card, voice, event log
          camera.py       pose.py         zones.py   usecases/fall_check.py   fall_panel.py
```

| Module (`lifeaware/`) | Job |
|---|---|
| `pose.py` | `PoseDetector`: pretrained `yolo11s-pose` (17 body points), checks every 3rd frame, decides `person_standing` / `person_sitting` / `person_lying` / `person_unclear` from the torso angle and knee height, and measures movement (body-lengths per second). The rules are plain geometry in `classify_posture()` and can be tested without a camera. |
| `usecases/fall_check.py` | `FallCheck`: the UC3 state machine (`normal → possible_event → checking → alarm / solved / unknown`). Pure logic, no camera or model; every change is an `Event` with a reason. |
| `fall_panel.py` | The Streamlit section H itself (background worker + layout). Also shown standalone by `streamlit run demo_fall.py`. |

**Speed on our laptop (CPU only, measured):** in the terminal tools about **7 frames per second** and **~90–100 ms per pose check**. **Inside the dashboard it is much slower: about 1.5–2 frames per second and ~1.2–2 s per pose check.**

> **Known issue (open).** `tools/speed_check.py` shows why: the model thread shares Python's global lock (GIL) with the busy Streamlit server. Alone, a check takes ~90 ms; next to a busy Python thread, ~8 s. The fall check still works at this speed (its timers are several seconds long), but the picture is choppy. Planned fix: run camera + model in their own process.

**Try it without the dashboard:**

```bash
python tools/pose_demo.py --show                        # postures, zone and movement per frame + fps
python tools/fall_demo.py --show --still 4 --timeout 8  # whole fall check; keys y/n/h/u answer, a = acknowledge
python tools/speed_check.py                             # compare model speed: main thread / thread / busy thread
```

**Known limits.** One fixed camera: someone lying with their head towards the camera can look upright, and the whole body must be in view (a laptop webcam on a desk only sees the upper body → `unclear`). The posture thresholds are first guesses and must be tuned on our own labelled clips. The `zones` in `config.yaml` are still placeholders until they are set for the final camera position with `tools/zone_setup.py`.

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
python -m streamlit run app.py
```

(`python -m streamlit` also works where Windows Application Control blocks the `streamlit.exe` launcher.)

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

### 7. Try the real camera → pose → fall check chain (optional)

```bash
python tools/pose_demo.py --show                        # standing / sitting / lying, with skeleton
python tools/fall_demo.py --show --still 4 --timeout 8  # the whole UC3 fall check with short timers
python -m streamlit run demo_fall.py --server.port 8503 # section H as a one-page browser demo
```

The first run downloads the pretrained `yolo11s-pose.pt` (~20 MB) into `models/`.
Put the camera a few metres away so it sees the whole body.

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
7. In section H, click **▶ Start fall check**, step back so the whole body is visible, lie down on the floor and keep still. Watch the card go `POSSIBLE EVENT` → `CHECKING`; then press **✅ Yes**, or wait for `ALARM` and press **🙋 Acknowledge**. Starting H stops C and G (and starting C or G stops H).

The history lives in memory only. It is lost when you refresh the page or restart the app. Reminders, however, are written to `reminders.json` and persist across restarts.

---

## Project structure

```
SCENE-DETECT/
├── app.py               # The Streamlit dashboard (config, styles, UI, camera preview)
├── reminders.py         # Reminder logic: transcription, parsing, storage, speech
├── demo_items.py        # Section G as a standalone page (streamlit run demo_items.py)
├── demo_fall.py         # Section H as a standalone page (streamlit run demo_fall.py)
├── config.yaml          # Thresholds, model name and room zones for the real chain
├── lifeaware/           # The real (non-simulated) chain — see sections G and H above
│   ├── schema.py  config.py  camera.py  objects.py  zones.py  memory.py  item_finder.py
│   ├── pose.py  fall_panel.py   # pose detection + section H
│   └── usecases/
│       ├── lost_items.py          # UC2: find lost items
│       └── fall_check.py          # UC3: fall check state machine
├── tools/
│   ├── zone_setup.py    # Click on a camera image to find zone coordinates
│   ├── item_demo.py     # Run the UC2 chain in a terminal window and print fps
│   ├── pose_demo.py     # Postures from the webcam, with skeleton and fps
│   ├── fall_demo.py     # The UC3 fall check in a window, answers from the keyboard
│   └── speed_check.py   # Measure model speed (main thread / thread / busy thread)
├── tests/               # pytest tests (fake data only: no camera, mic or model download)
├── .github/workflows/tests.yml   # Runs the tests on GitHub for every push
├── pytest.ini
├── requirements.txt     # Python dependencies
├── models/              # Vosk speech model + YOLO weights (downloaded, gitignored)
├── reminders.json       # Saved reminders (gitignored)
├── events/              # fall_events.csv: UC3 event log (gitignored)
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
| **Page** | Renders sections A to F, section G (one call: `render_item_finder()` from `lifeaware/item_finder.py`), section H (one call: `render_fall_check()` from `lifeaware/fall_panel.py`) and the footer. |

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

The fall check (section H) is configured in `config.yaml` too:

- **Pose model and rules** (`pose:`): `model` (`yolo11n-pose.pt` is faster, `yolo11m-pose.pt` more accurate), `every_n_frames`, `lying_torso_angle` (degrees from vertical that count as lying), `sit_knee_ratio`.
- **Fall-check timers** (`fall:`): `min_lying_s`, `still_s`, `answer_timeout_s`, `cooldown_s`, `still_movement`, and `floor_zones` (which zones count as "the floor"). Set `microphone: false` when no microphone is connected — then a missing answer gives `UNKNOWN` instead of an alarm.

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
| `streamlit.exe` is blocked ("Application Control policy") | Run `python -m streamlit run app.py` instead. |
| Section H: always *"unclear"* | The camera only sees part of the body (e.g. sitting at a laptop). Move the camera a few metres away so it sees head to knees at least. |
| Section H: lying on the floor never leads to a question | Check what the caption says under the picture: the zone must be `floor` (or no zone). The `zones` in `config.yaml` are placeholders — set them with `tools/zone_setup.py`. |
| Section H: *"Opening the camera …"* for ~30 s | Normal on some Windows laptops. If it takes longer than 40 s, another program or section holds the webcam. |
| Section H is slow (~1–2 s per check) | Known issue in the dashboard (see section H). Close sections C and G and other camera programs; the terminal tools (`tools/fall_demo.py`) run at full speed. |
| Section G says the model could not be downloaded | The first start needs internet to fetch `yolo11m.pt`. Alternatively copy `yolo11m.pt` from a teammate into `models/`. |

---

## Roadmap

Planned for later versions. Items marked ✅ have a first working version.

- ✅ **Real-time human pose detection** (section H). Standing, sitting and lying from 17 body points. Still to do: walking and bending, and tuning the thresholds on labelled clips.
- ✅ **Fall and inactivity detection** (section H). Lying on the floor and still across several frames → question → alarm. Still to do: speed in the dashboard (separate process), and evaluation (correct / missed / false alarms, reaction time) on test scenarios.
- 🔊 **Voice interaction.** The question *"Are you okay?"* is already spoken; listening for the answer with offline speech recognition is the next step (answers come from buttons for now).
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

*SCENE-DETECT prototype. A university project at OTH Amberg-Weiden. Simulated scene demonstrations; real phone detection in section G; real pose detection and fall check in section H (answers simulated).*
