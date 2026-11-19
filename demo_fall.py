"""Standalone page for the fall check (same section as app.py section H).

    streamlit run demo_fall.py --server.port 8503

Handy for the mid-term demo, or for testing with a video file instead of the
webcam (type the path into the "Camera" box). All logic lives in
``lifeaware/fall_panel.py`` and ``lifeaware/usecases/fall_check.py``.
"""

import streamlit as st

from lifeaware.fall_panel import render_fall_check


def main() -> None:
    st.set_page_config(page_title="LifeAware – Fall check", page_icon="🧍", layout="wide")
    st.title("🧍 Fall Check")
    st.caption("Use case UC3 · real webcam + pretrained YOLO pose, running locally. "
               "Answers are simulated until the microphone part is connected. "
               "Student project, not a medical device.")
    render_fall_check(show_source_input=True)


if __name__ == "__main__":
    main()
