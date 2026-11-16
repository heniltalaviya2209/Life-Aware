"""Tests for lifeaware.zones.ZoneMap (fake polygons)."""

import numpy as np
import pytest

from lifeaware.zones import ZoneMap

SQUARE_ZONES = {
    "sofa": [[0, 0], [100, 0], [100, 100], [0, 100]],
    "floor": [[0, 100], [200, 100], [200, 200], [0, 200]],
}


def test_point_inside_a_zone():
    assert ZoneMap(SQUARE_ZONES).zone_of(50, 50) == "sofa"
    assert ZoneMap(SQUARE_ZONES).zone_of(150, 150) == "floor"


def test_point_outside_all_zones_is_none():
    assert ZoneMap(SQUARE_ZONES).zone_of(150, 50) is None


def test_point_on_an_edge_counts_as_inside():
    assert ZoneMap(SQUARE_ZONES).zone_of(100, 50) == "sofa"


def test_first_zone_wins_when_zones_overlap():
    zones = ZoneMap({"bed": [[0, 0], [10, 0], [10, 10], [0, 10]],
                     "floor": [[0, 0], [20, 0], [20, 20], [0, 20]]})

    assert zones.zone_of(5, 5) == "bed"
    assert zones.zone_of(15, 15) == "floor"


def test_non_rectangular_polygon():
    triangle = ZoneMap({"corner": [[0, 0], [100, 0], [0, 100]]})

    assert triangle.zone_of(10, 10) == "corner"
    assert triangle.zone_of(90, 90) is None


def test_no_zones_means_unknown():
    assert ZoneMap().zone_of(1, 1) is None
    assert ZoneMap(None).zone_of(1, 1) is None


def test_zone_needs_three_points():
    with pytest.raises(ValueError):
        ZoneMap({"line": [[0, 0], [10, 10]]})


def test_draw_returns_a_marked_copy():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    drawn = ZoneMap(SQUARE_ZONES).draw(frame)

    assert drawn.shape == frame.shape
    assert drawn.any()  # something was drawn
    assert not frame.any()  # the original frame is untouched
