"""Tests for Point2D arithmetic and Points2D batch operations."""

from __future__ import annotations

import numpy as np
import pytest
from geometry.planar import Geometry2DMeasure, Line2D, Point2D, Points2D, Vector2D


@pytest.fixture
def unit_square() -> Points2D:
    """Four corners of the unit square in counter-clockwise order."""
    return Points2D(value=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))


class TestPoint2D:
    """Single-point accessors and arithmetic."""

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            Point2D(value=np.array([1.0, 2.0, 3.0]))

    def test_components(self) -> None:
        point = Point2D(value=np.array([3.0, 4.0]))
        assert point.x == 3.0
        assert point.y == 4.0
        assert point.tuple == (3.0, 4.0)
        assert point.list == [3.0, 4.0]

    def test_integer_components_stay_integers(self) -> None:
        point = Point2D(value=np.array([3, 4]))
        assert isinstance(point.x, int)
        assert isinstance(point.y, int)

    def test_array_is_a_copy(self) -> None:
        point = Point2D(value=np.array([3.0, 4.0]))
        copied = point.array
        copied[0] = 100.0
        assert point.x == 3.0

    def test_subtraction_returns_vector(self) -> None:
        difference = Point2D(value=np.array([5.0, 7.0])) - Point2D(value=np.array([2.0, 3.0]))
        assert isinstance(difference, Vector2D)
        np.testing.assert_allclose(difference.value, [3.0, 4.0])

    def test_addition_and_scaling(self) -> None:
        point = Point2D(value=np.array([1.0, 2.0]))
        np.testing.assert_allclose(point + Point2D(value=np.array([3.0, 4.0])), [4.0, 6.0])
        np.testing.assert_allclose(point * 2.0, [2.0, 4.0])

    def test_shapely(self) -> None:
        shapely_point = Point2D(value=np.array([1.0, 2.0])).shapely
        assert (shapely_point.x, shapely_point.y) == (1.0, 2.0)


class TestPoints2DConstruction:
    """Shape validation."""

    @pytest.mark.parametrize("shape", [(2,), (3, 3), (2, 2, 2)])
    def test_invalid_shape_raises(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError):
            Points2D(value=np.zeros(shape))


class TestPoints2DGeometry:
    """Components, area, center, and distances."""

    def test_components(self, unit_square: Points2D) -> None:
        np.testing.assert_allclose(unit_square.x, [0.0, 1.0, 1.0, 0.0])
        np.testing.assert_allclose(unit_square.y, [0.0, 0.0, 1.0, 1.0])

    def test_area(self, unit_square: Points2D) -> None:
        assert unit_square.area == pytest.approx(1.0)

    def test_center(self, unit_square: Points2D) -> None:
        np.testing.assert_allclose(unit_square.center.value, [0.5, 0.5])

    def test_distance_matrix(self, unit_square: Points2D) -> None:
        distances = unit_square.distance_matrix
        assert distances.shape == (4, 4)
        np.testing.assert_allclose(np.diag(distances), 0.0)
        np.testing.assert_allclose(distances, distances.T)
        assert distances[0, 2] == pytest.approx(np.sqrt(2.0))

    def test_convex_hull_drops_interior_point(self, unit_square: Points2D) -> None:
        with_interior = unit_square.concat(Points2D(value=np.array([[0.5, 0.5]])))
        hull_rows = {tuple(row) for row in with_interior.convex_hull_points.tolist()}
        assert hull_rows == {tuple(row) for row in unit_square.value.tolist()}

    def test_convex_hull_flag_repairs_self_intersecting_order(self) -> None:
        bowtie_order = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        assert not Points2D(value=bowtie_order).shapely.is_valid
        hull = Points2D(value=bowtie_order, is_convex_hull=True).shapely
        assert hull.is_valid
        assert hull.area == pytest.approx(1.0)


class TestPoints2DCollection:
    """Indexing, iteration, and in-place mutation."""

    def test_len_and_iter(self, unit_square: Points2D) -> None:
        assert len(unit_square) == 4
        assert [point.tuple for point in unit_square][2] == (1.0, 1.0)

    def test_integer_index_returns_point(self, unit_square: Points2D) -> None:
        item = unit_square[1]
        assert isinstance(item, Point2D)
        assert item.tuple == (1.0, 0.0)

    def test_slice_keeps_convex_hull_flag(self) -> None:
        points = Points2D(value=np.zeros((3, 2)), is_convex_hull=True)
        sliced = points[:2]
        assert isinstance(sliced, Points2D)
        assert len(sliced) == 2
        assert sliced.is_convex_hull

    def test_append_and_delete(self, unit_square: Points2D) -> None:
        unit_square.append((2.0, 2.0))
        assert len(unit_square) == 5
        assert (2.0, 2.0) in unit_square
        unit_square.delete(0)
        assert len(unit_square) == 4
        assert (0.0, 0.0) not in unit_square

    def test_append_invalid_shape_raises(self, unit_square: Points2D) -> None:
        with pytest.raises(ValueError):
            unit_square.append(np.array([1.0, 2.0, 3.0]))

    def test_concat_is_non_destructive(self, unit_square: Points2D) -> None:
        combined = unit_square.concat(unit_square)
        assert len(combined) == 8
        assert len(unit_square) == 4


class TestGeometry2DMeasure:
    """Shapely-backed distances between planar shapes."""

    def test_point_to_point(self) -> None:
        first = Point2D(value=np.array([0.0, 0.0]))
        second = Point2D(value=np.array([3.0, 4.0]))
        assert Geometry2DMeasure.measure_distance(first, second) == pytest.approx(5.0)

    def test_point_to_line(self) -> None:
        point = Point2D(value=np.array([5.0, 3.0]))
        line = Line2D(value=np.array([[0.0, 0.0], [10.0, 0.0]]))
        assert Geometry2DMeasure.measure_distance(point, line) == pytest.approx(3.0)

    def test_point_inside_polygon_is_zero(self, unit_square: Points2D) -> None:
        point = Point2D(value=np.array([0.5, 0.5]))
        assert Geometry2DMeasure.measure_distance(point, unit_square) == pytest.approx(0.0)
