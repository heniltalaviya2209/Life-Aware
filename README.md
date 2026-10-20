# SCENE-DETECT — Intelligent Scene Recognition

A user-interface prototype for an AI-based situation-recognition system for the home, built as a university project at **OTH Amberg-Weiden**.

The long-term idea is a small home-assistant robot that watches over people at home, especially elderly people (60+). It recognises the situation in a room and gets a human helper when something looks wrong, such as a fall. This repository holds the **first prototype**. It is a Streamlit dashboard that shows the workflow and how the system would *explain* its decisions.

> [!IMPORTANT]
> **This prototype does not run a real AI model.**
> It does not detect people, poses, emotions or falls. All demonstration results are pre-written simulations for fixed scenarios. The camera section shows a live video preview only, and the frames are not analysed.
> It is not a medical device and does not give a medical diagnosis.

---

## Table of contents

- [What the app shows](#what-the-app-shows)
- [The four system states](#the-four-system-states)
- [Demonstration scenarios](#demonstration-scenarios)
- [Getting started](#getting-started)
- [Using the app](#using-the-app)
- [Project structure](#project-structure)
- [Customising the content](#customising-the-content)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Design principles](#design-principles)

---

## What the app shows

The dashboard is a single page with five sections:

| Section | What it does |
|---|---|
| **A · Main Dashboard** | Four cards that explain the states the system can report: `NO PERSON`, `NORMAL`, `CHECK PERSON`, `ALERT`. |
| **B · Scene Demonstration** | Pick one of seven predefined scenarios and press **Run demonstration**. You get a simulated result with a state, an illustrative score, body features, the evidence used, and a plain-language reason for the label. |
| **C · Camera Preview** *(optional)* | Shows your live webcam feed through OpenCV, marked "PREVIEW ONLY – NO AI DETECTION". It is there to show where a real camera would plug in. |
| **D · Event History** | A table of every demonstration you ran in this session (time, scenario, state, explanation), with a **Clear history** button. |
| **E · Future Features** | Planned features for later versions. None of them is implemented yet. |

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

---

## Getting started

### Requirements

- **Python 3.10 or newer**
- A webcam *(optional; only needed for section C)*
- No GPU, no cloud services and no paid APIs. Everything runs on a normal CPU.

Dependencies (from [`requirements.txt`](requirements.txt)):

| Package | Why |
|---|---|
| `streamlit` | The web UI |
| `pandas` | The event history table |
| `numpy` | Used by pandas and OpenCV |
| `opencv-python` | Webcam preview only. The app still works without it. |

### 1. Clone the repository

```bash
git clone https://github.com/heniltalaviya2209/Scene-Detection-Prototype.git
cd Scene-Detection-Prototype
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

---

## Using the app

1. **Read the four cards** in section A to learn what each state means.
2. In section B, **choose a scenario** from the dropdown and click **▶ Run demonstration**. The result panel shows:
   - the resulting **state** (colour-coded),
   - an **illustrative score** (simulated, not a real confidence value),
   - the **simulated body features** (orientation and motion),
   - the **evidence** used and **why this label was chosen**.
3. *(Optional)* In section C, click **📷 Start camera preview** to see your webcam, and **⏹ Stop camera preview** to turn it off.
4. Check section D to see the **history** of all demonstrations you ran. Click **🗑 Clear history** to reset it.

The history lives in memory only. It is lost when you refresh the page or restart the app.

---

## Project structure

```
SCENE-DETECT/
├── app.py               # The whole Streamlit app (config, styles, UI, camera helpers)
├── requirements.txt     # Python dependencies
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
| **Session state** | Keeps the history, the last result and the camera status between Streamlit reruns. |
| **Camera helpers** | Opens the webcam (with a Windows `CAP_DSHOW` fallback) and renders **one frame per rerun**, so the Stop button always stays responsive. If it can't open the camera, it shows a friendly error message instead of crashing. |
| **Page** | Renders sections A to E and the footer. |

OpenCV is imported **lazily**, only when the camera is started. If it isn't installed, everything except the camera preview still works.

---

## Customising the content

All texts and simulated data are at the top of `app.py`, so you can change the demo without touching the UI code:

- **Add a scenario:** add a new entry to the `SCENARIOS` dictionary with `state`, `score`, `features`, `evidence` and `explanation`. The `state` must be one of the keys in `SITUATIONS`.
- **Change a state description:** edit `SITUATIONS`.
- **Change the roadmap cards:** edit `FUTURE_FEATURES`.

Streamlit reloads the app automatically when you save the file.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `streamlit: command not found` | The virtual environment is not active. Activate it (step 2), or run `python -m streamlit run app.py`. |
| "No camera could be opened" | Check that a webcam is connected and not being used by another app (Teams, Zoom, …). On Windows, check **Settings → Privacy & security → Camera** and allow desktop apps to use the camera. |
| "OpenCV is not installed" | Run `pip install opencv-python`, or ignore it. The simulated demos work without it. |
| DLL / import errors for pandas or numpy on Windows | Keep the versions pinned in `requirements.txt` (`pandas<3.0`, `numpy<2.3`). Newer builds were blocked by a Windows Code Integrity policy during development. |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502`. |

---

## Roadmap

Planned for later versions. **None of this is implemented yet.**

- 🧍 **Real-time human pose detection.** Estimate body key points (shoulders, hips, knees, ankles) and classify standing, sitting, walking, bending and lying.
- 🧎 **Fall and inactivity detection.** Watch movement across several frames and only raise an alert when strong evidence persists.
- 🔊 **Voice interaction.** Ask the person *"Are you okay?"* and listen for an answer before escalating.
- 📞 **Human assistance alerts.** Notify a caregiver or emergency contact with the scene details.
- 🔑 **Finding lost objects.** Remember where everyday items like keys, phone or glasses were last seen.
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

*SCENE-DETECT prototype. A university project at OTH Amberg-Weiden. Simulated demonstrations only.*
