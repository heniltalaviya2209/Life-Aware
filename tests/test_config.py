"""Tests for lifeaware.config.load_config."""

import pytest

from lifeaware.config import DEFAULTS, load_config


def test_missing_file_gives_defaults(tmp_path):
    assert load_config(tmp_path / "nope.yaml") == DEFAULTS


def test_values_are_merged_over_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("camera:\n  max_fps: 5\n", encoding="utf-8")

    config = load_config(path)

    assert config["camera"]["max_fps"] == 5
    assert config["camera"]["max_failures"] == 15  # default kept
    assert config["objects"]["min_conf"] == 0.4


def test_zones_are_replaced_not_merged(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("zones:\n  bed: [[0, 0], [10, 0], [10, 10]]\n", encoding="utf-8")

    assert list(load_config(path)["zones"]) == ["bed"]


def test_defaults_are_not_changed_by_loading(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("camera:\n  max_fps: 1\n", encoding="utf-8")

    load_config(path)

    assert DEFAULTS["camera"]["max_fps"] == 10


def test_non_mapping_file_is_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(path)


def test_repo_config_loads():
    """The real config.yaml in the repo must stay valid."""
    config = load_config("config.yaml")

    assert config["objects"]["classes"]["cell phone"] == "phone"
    assert config["zones"]
