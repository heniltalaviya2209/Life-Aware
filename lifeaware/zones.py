"""Zones: map a pixel position to a named area ("sofa", "floor", ...).

Zones are polygons in pixel coordinates of the camera image, loaded from the
``zones`` section of ``config.yaml``. Use ``tools/zone_setup.py`` to find the
coordinates for our fixed camera.
"""

from __future__ import annotations

import numpy as np

# BGR colours used when drawing zones (cycled if there are more zones).
_COLOURS = [(0, 200, 255), (0, 220, 0), (255, 120, 0), (200, 0, 200), (0, 0, 255)]


class ZoneMap:
    """A set of named polygons. If zones overlap, the first one listed wins."""

    def __init__(self, zones: dict[str, list[list[float]]] | None = None) -> None:
        self.zones: dict[str, np.ndarray] = {}
        for name, points in (zones or {}).items():
            polygon = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
            if len(polygon) < 3:
                raise ValueError(f"Zone '{name}' needs at least 3 points")
            self.zones[name] = polygon

    def zone_of(self, x: float, y: float) -> str | None:
        """Name of the zone containing the point (edges count as inside), or None."""
        import cv2  # lazy import

        for name, polygon in self.zones.items():
            if cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0:
                return name
        return None

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Return a copy of ``frame`` with every zone outlined and labelled."""
        import cv2  # lazy import

        canvas = frame.copy()
        for i, (name, polygon) in enumerate(self.zones.items()):
            colour = _COLOURS[i % len(_COLOURS)]
            points = polygon.astype(np.int32)
            cv2.polylines(canvas, [points], isClosed=True, color=colour, thickness=2)
            x, y = points[0][0]
            cv2.putText(
                canvas, name, (int(x) + 4, int(y) + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA,
            )
        return canvas
