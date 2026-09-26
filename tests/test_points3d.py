"""Tests for Point3D and Points3D."""

from __future__ import annotations

import numpy as np
import pytest
from geometry.spatial import Point3D, Points3D, Vector3D


@pytest.fixture
def points() -> Points3D:
    """Origin and two axis-aligned points."""
    return Points3D(value=np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 2.0]]))


class TestPoint3D:
    """Single-point accessors and arithmetic."""

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            Point3D(value=np.array([1.0, 2.0]))

    def test_components(self) -> None:
        point = Point3D(value=np.array([1.0, 2.0, 3.0]))
        assert (point.x, point.y, point.z) == (1.0, 2.0, 3.0)
        assert point.tuple == (1.0, 2.0, 3.0)
        assert point.list == [1.0, 2.0, 3.0]

    def test_addition_returns_point(self) -> None:
        total = Point3D(value=np.array([1.0, 2.0, 3.0])) + Point3D(value=np.array([1.0, 1.0, 1.0]))
        assert isinstance(total, Point3D)
        assert total.tuple == (2.0, 3.0, 4.0)

    def test_subtraction_returns_vector(self) -> None:
        difference = Point3D(value=np.array([4.0, 6.0, 3.0])) - Point3D(value=np.array([1.0, 2.0, 3.0]))
        assert isinstance(difference, Vector3D)
        np.testing.assert_allclose(difference.value, [3.0, 4.0, 0.0])


class TestPoints3D:
    """Batch points."""

    @pytest.mark.parametrize("shape", [(3,), (2, 2), (2, 3, 1)])
    def test_invalid_shape_raises(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError):
            Points3D(value=np.zeros(shape))

    def test_components(self, points: Points3D) -> None:
        np.testing.assert_allclose(points.x, [0.0, 3.0, 0.0])
        np.testing.assert_allclose(points.y, [0.0, 4.0, 0.0])
        np.testing.assert_allclose(points.z, [0.0, 0.0, 2.0])

    def test_distance_matrix(self, points: Points3D) -> None:
        distances = points.distance_matrix
        assert distances.shape == (3, 3)
        assert distances[0, 1] == pytest.approx(5.0)
        assert distances[0, 2] == pytest.approx(2.0)
        np.testing.assert_allclose(distances, distances.T)

    def test_arithmetic(self, points: Points3D) -> None:
        np.testing.assert_allclose((points + points).value, points.value * 2)
        np.testing.assert_allclose((points - points).value, 0.0)
        np.testing.assert_allclose((points * 3.0).value, points.value * 3)
        np.testing.assert_allclose((points / 2.0).value, points.value / 2)
        np.testing.assert_allclose(points.scale(0.5).value, points.value / 2)

    def test_indexing(self, points: Points3D) -> None:
        item = points[1]
        assert isinstance(item, Point3D)
        assert item.tuple == (3.0, 4.0, 0.0)
        sliced = points[:2]
        assert isinstance(sliced, Points3D)
        assert len(sliced) == 2

    def test_iteration(self, points: Points3D) -> None:
        assert [point.z for point in points] == [0.0, 0.0, 2.0]
