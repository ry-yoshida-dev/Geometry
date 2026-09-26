"""Tests for Points2D row-wise membership."""

from __future__ import annotations

import numpy as np
import pytest
from geometry.planar import Points2D


@pytest.fixture
def points() -> Points2D:
    """Two distinct points."""
    return Points2D(value=np.array([[1.0, 2.0], [3.0, 4.0]]))


class TestPoints2DContains:
    """Membership must match whole rows, not individual components."""

    def test_existing_row_is_contained(self, points: Points2D) -> None:
        assert (3.0, 4.0) in points

    def test_array_coordinate_is_contained(self, points: Points2D) -> None:
        assert np.array([1.0, 2.0]) in points

    def test_partial_component_match_is_not_contained(self, points: Points2D) -> None:
        assert (1.0, 99.0) not in points

    def test_cross_row_components_are_not_contained(self, points: Points2D) -> None:
        assert (1.0, 4.0) not in points

    def test_invalid_shape_raises(self, points: Points2D) -> None:
        with pytest.raises(ValueError):
            _ = np.array([1.0, 2.0, 3.0]) in points
