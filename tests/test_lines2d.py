"""Tests for Line2D and Lines2D segments."""

from __future__ import annotations

import numpy as np
import pytest
from geometry.planar import Line2D, Lines2D, Point2D, Points2D, Vector2D


@pytest.fixture
def line() -> Line2D:
    """Segment from (1, 2) to (4, 6) with length 5."""
    return Line2D(value=np.array([[1.0, 2.0], [4.0, 6.0]]))


@pytest.fixture
def lines() -> Lines2D:
    """Three segments with lengths 5, 1, and 0."""
    return Lines2D(
        value=np.array(
            [
                [[0.0, 0.0], [3.0, 4.0]],
                [[1.0, 1.0], [1.0, 2.0]],
                [[2.0, 2.0], [2.0, 2.0]],
            ]
        )
    )


class TestLine2D:
    """Single-segment geometry."""

    @pytest.mark.parametrize("shape", [(2,), (3, 2), (2, 3), (1, 2, 2)])
    def test_invalid_shape_raises(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError):
            Line2D(value=np.zeros(shape))

    def test_endpoints(self, line: Line2D) -> None:
        assert line.start.tuple == (1.0, 2.0)
        assert line.end.tuple == (4.0, 6.0)
        np.testing.assert_allclose(line.x, [1.0, 4.0])
        np.testing.assert_allclose(line.y, [2.0, 6.0])

    def test_length(self, line: Line2D) -> None:
        assert line.length == pytest.approx(5.0)

    def test_center(self, line: Line2D) -> None:
        np.testing.assert_allclose(line.center.value, [2.5, 4.0])

    def test_vector2d(self, line: Line2D) -> None:
        vector = line.vector2d
        assert isinstance(vector, Vector2D)
        np.testing.assert_allclose(vector.value, [3.0, 4.0])

    def test_points2d(self, line: Line2D) -> None:
        points = line.points2d
        assert isinstance(points, Points2D)
        np.testing.assert_allclose(points.value, line.value)

    def test_shapely(self, line: Line2D) -> None:
        line_string = line.shapely
        assert list(line_string.coords) == [(1.0, 2.0), (4.0, 6.0)]
        assert line_string.length == pytest.approx(5.0)

    def test_from_two_points(self, line: Line2D) -> None:
        rebuilt = Line2D.from_two_points(
            start_point=Point2D(value=np.array([1.0, 2.0])),
            end_point=Point2D(value=np.array([4.0, 6.0])),
        )
        np.testing.assert_allclose(rebuilt.value, line.value)


class TestLines2D:
    """Batch segment geometry and indexing."""

    @pytest.mark.parametrize("shape", [(2, 2), (3, 2, 3), (3, 3, 2)])
    def test_invalid_shape_raises(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError):
            Lines2D(value=np.zeros(shape))

    def test_empty_batch_is_allowed(self) -> None:
        assert len(Lines2D(value=np.zeros((0, 2, 2)))) == 0

    def test_length(self, lines: Lines2D) -> None:
        np.testing.assert_allclose(lines.length, [5.0, 1.0, 0.0])

    def test_start_end_center(self, lines: Lines2D) -> None:
        np.testing.assert_allclose(lines.start.value, [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        np.testing.assert_allclose(lines.end.value, [[3.0, 4.0], [1.0, 2.0], [2.0, 2.0]])
        np.testing.assert_allclose(lines.center.value, [[1.5, 2.0], [1.0, 1.5], [2.0, 2.0]])

    def test_vectors(self, lines: Lines2D) -> None:
        np.testing.assert_allclose(lines.vectors, [[3.0, 4.0], [0.0, 1.0], [0.0, 0.0]])

    def test_shapely(self, lines: Lines2D) -> None:
        line_strings = lines.shapely
        assert len(line_strings) == 3
        assert line_strings[0].length == pytest.approx(5.0)

    def test_integer_index_returns_line(self, lines: Lines2D) -> None:
        item = lines[0]
        assert isinstance(item, Line2D)
        assert item.length == pytest.approx(5.0)

    def test_slice_returns_lines(self, lines: Lines2D) -> None:
        sliced = lines[1:]
        assert isinstance(sliced, Lines2D)
        np.testing.assert_allclose(sliced.length, [1.0, 0.0])

    def test_length_matches_single_line(self, lines: Lines2D) -> None:
        for index in range(len(lines)):
            item = lines[index]
            assert isinstance(item, Line2D)
            assert item.length == pytest.approx(float(lines.length[index]))
