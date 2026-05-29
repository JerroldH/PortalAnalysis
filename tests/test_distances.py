"""Unit tests for DistanceCalculator landmark indexing."""

import pytest

from portal_analysis.preprocessing.distances import DistanceCalculator
from portal_analysis.preprocessing import hand_landmarks as hl


def _landmarks_with_wrist(x: float = 0.1, y: float = 0.2, z: float = 0.3) -> list:
    landmarks = [[0.0, 0.0, 0.0] for _ in range(21)]
    landmarks[hl.WRIST] = [x, y, z]
    landmarks[hl.THUMB_TIP] = [0.2, 0.2, 0.0]
    landmarks[hl.INDEX_FINGER_MCP] = [0.15, 0.15, 0.0]
    landmarks[hl.INDEX_FINGER_TIP] = [0.18, 0.18, 0.0]
    return landmarks


def test_wrist_coordinates_scaled():
    calc = DistanceCalculator(width=100, height=200)
    x, y, z = calc.wrist_coordinates(_landmarks_with_wrist())
    assert x == pytest.approx(10.0)
    assert y == pytest.approx(40.0)
    assert z == pytest.approx(30.0)


def test_finger_distance_nonzero_with_valid_landmarks():
    calc = DistanceCalculator()
    dist = calc.finger_distance(_landmarks_with_wrist())
    assert dist > 0.0
