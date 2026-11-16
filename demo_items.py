"""Standalone page for the item finder (same section as app.py section G).

    streamlit run demo_items.py --server.port 8502

Handy for testing the finder on its own, or with a video file instead of the
webcam (type the path into the "Camera" box). All logic lives in
``lifeaware/item_finder.py``.
"""

import streamlit as st

from lifeaware.item_finder import render_item_finder


def main() -> None:
    st.set_page_config(page_title="LifeAware – Item finder", page_icon="📱", layout="wide")
    st.title("📱 Find My Phone or Keys")
    st.caption("Use case UC2 · real webcam + pretrained YOLO, running locally. "
               "Student project, not a medical device.")
    render_item_finder(show_source_input=True)


if __name__ == "__main__":
    main()
