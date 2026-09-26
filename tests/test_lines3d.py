"""Tests for Line3D and Lines3D segments."""

from __future__ import annotations

import numpy as np
import pytest
from geometry.spatial import Line3D, Lines3D, Point3D, Points3D, Vector3D, Vectors3D


@pytest.fixture
def lines() -> Lines3D:
    """Two segments with lengths 3 and 0."""
    return Lines3D(
        value=np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]],
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            ]
        )
    )


class TestLine3D:
    """Single-segment geometry."""

    @pytest.mark.parametrize("shape", [(3,), (2, 2), (3, 3)])
    def test_invalid_shape_raises(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError):
            Line3D(value=np.zeros(shape))

    def test_endpoints_vector_and_length(self) -> None:
        line = Line3D(value=np.array([[1.0, 1.0, 1.0], [2.0, 3.0, 3.0]]))
        assert line.start.tuple == (1.0, 1.0, 1.0)
        assert line.end.tuple == (2.0, 3.0, 3.0)
        assert isinstance(line.vector3d, Vector3D)
        np.testing.assert_allclose(line.vector3d.value, [1.0, 2.0, 2.0])
        assert line.length == pytest.approx(3.0)

    def test_from_start_and_end(self) -> None:
        line = Line3D.from_start_and_end(
            start=Point3D(value=np.array([0.0, 0.0, 0.0])),
            end=Point3D(value=np.array([0.0, 0.0, 4.0])),
        )
        assert line.length == pytest.approx(4.0)


class TestLines3D:
    """Batch segment geometry and indexing."""

    @pytest.mark.parametrize("shape", [(2, 3), (1, 2, 2), (1, 3, 3)])
    def test_invalid_shape_raises(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError):
            Lines3D(value=np.zeros(shape))

    def test_length(self, lines: Lines3D) -> None:
        np.testing.assert_allclose(lines.length, [3.0, 0.0])

    def test_start_end_center(self, lines: Lines3D) -> None:
        assert isinstance(lines.start, Points3D)
        np.testing.assert_allclose(lines.start.value, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        np.testing.assert_allclose(lines.end.value, [[1.0, 2.0, 2.0], [1.0, 1.0, 1.0]])
        np.testing.assert_allclose(lines.center.value, [[0.5, 1.0, 1.0], [1.0, 1.0, 1.0]])

    def test_vectors(self, lines: Lines3D) -> None:
        vectors = lines.vectors
        assert isinstance(vectors, Vectors3D)
        np.testing.assert_allclose(vectors.value, [[1.0, 2.0, 2.0], [0.0, 0.0, 0.0]])

    def test_indexing(self, lines: Lines3D) -> None:
        assert len(lines) == 2
        item = lines[0]
        assert isinstance(item, Line3D)
        assert item.length == pytest.approx(3.0)
        sliced = lines[1:]
        assert isinstance(sliced, Lines3D)
        assert len(sliced) == 1
