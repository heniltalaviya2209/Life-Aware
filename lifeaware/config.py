"""Load ``config.yaml`` and fill in defaults for anything that is missing.

Owner: Yash. A missing or empty file is fine: you simply get the defaults.
"""

from __future__ import annotations

import copy
import os

DEFAULT_CONFIG_PATH = os.environ.get("LIFEAWARE_CONFIG", "config.yaml")

DEFAULTS: dict = {
    "camera": {"source": 0, "max_fps": 10, "max_failures": 15},
    "objects": {
        "model": "models/yolo11n.pt",
        "every_n_frames": 3,
        "min_conf": 0.4,
        "image_size": 640,
        "classes": {"cell phone": "phone"},
    },
    "items": {"min_conf": 0.4, "sure_conf": 0.6},
    "pose": {
        "model": "models/yolo11s-pose.pt",
        "every_n_frames": 3,
        "min_conf": 0.4,
        "min_keypoint_conf": 0.3,
        "image_size": 640,
        "max_persons": 1,
        "lying_torso_angle": 60.0,
        "sit_knee_ratio": 0.6,
        "lying_aspect": 1.4,
    },
    "fall": {
        "floor_zones": ["floor"],
        "unknown_zone_counts": True,
        "min_conf": 0.3,
        "min_lying_s": 3.0,
        "still_s": 10.0,
        "still_movement": 0.15,
        "max_gap_s": 3.0,
        "answer_timeout_s": 15.0,
        "cooldown_s": 60.0,
        "microphone": True,
    },
    "zones": {},
}

# Sections that are replaced as a whole instead of merged key by key
# (otherwise a default zone could sneak into your own zone list).
_REPLACE_WHOLE = {"zones", "classes", "floor_zones"}


def _merge(defaults: dict, overrides: dict) -> dict:
    """Return ``defaults`` updated with ``overrides`` (nested dicts merged)."""
    result = copy.deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict) and key not in _REPLACE_WHOLE:
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | os.PathLike = DEFAULT_CONFIG_PATH) -> dict:
    """Read the YAML config at ``path`` and merge it over ``DEFAULTS``."""
    import yaml  # lazy import (PyYAML)

    if not os.path.exists(path):
        return copy.deepcopy(DEFAULTS)
    with open(path, encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return _merge(DEFAULTS, loaded)
