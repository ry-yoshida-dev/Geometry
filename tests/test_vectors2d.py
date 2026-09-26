"""Tests for Vector2D, Vectors2D, and Vector2DPair."""

from __future__ import annotations

import numpy as np
import pytest
from geometry.planar import Line2D, Point2D, Vector2D, Vector2DPair, Vectors2D


class TestVector2D:
    """Single-vector components, norm, and relations."""

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            Vector2D(value=np.array([1.0, 2.0, 3.0]))

    def test_components_and_norm(self) -> None:
        vector = Vector2D(value=np.array([3.0, 4.0]))
        assert (vector.x, vector.y) == (3.0, 4.0)
        assert vector.norm == pytest.approx(5.0)

    def test_unit_vector(self) -> None:
        unit = Vector2D(value=np.array([3.0, 4.0])).unit_vector
        np.testing.assert_allclose(unit, [0.6, 0.8])
        assert np.linalg.norm(unit) == pytest.approx(1.0)

    def test_zero_unit_vector_raises(self) -> None:
        with pytest.raises(ValueError):
            _ = Vector2D(value=np.array([0.0, 0.0])).unit_vector

    @pytest.mark.parametrize(
        ("first", "second", "is_expected_orthogonal"),
        [
            ([1.0, 0.0], [0.0, 2.0], True),
            ([1.0, 1.0], [1.0, -1.0], True),
            ([1.0, 0.0], [1.0, 1.0], False),
        ],
    )
    def test_is_orthogonal(self, first: list[float], second: list[float], is_expected_orthogonal: bool) -> None:
        first_vector = Vector2D(value=np.array(first))
        second_vector = Vector2D(value=np.array(second))
        assert first_vector.is_orthogonal(second_vector) is is_expected_orthogonal

    @pytest.mark.parametrize(
        ("first", "second", "is_expected_parallel"),
        [
            ([1.0, 2.0], [2.0, 4.0], True),
            ([1.0, 2.0], [-1.0, -2.0], True),
            ([1.0, 0.0], [0.0, 1.0], False),
        ],
    )
    def test_is_parallel(self, first: list[float], second: list[float], is_expected_parallel: bool) -> None:
        first_vector = Vector2D(value=np.array(first))
        second_vector = Vector2D(value=np.array(second))
        assert first_vector.is_parallel(second_vector) is is_expected_parallel

    def test_from_two_points(self) -> None:
        vector = Vector2D.from_two_points(
            start_point=Point2D(value=np.array([1.0, 1.0])),
            end_point=Point2D(value=np.array([4.0, 5.0])),
        )
        np.testing.assert_allclose(vector.value, [3.0, 4.0])

    def test_to_line(self) -> None:
        line = Vector2D(value=np.array([3.0, 4.0])).to_line(Point2D(value=np.array([1.0, 1.0])))
        assert isinstance(line, Line2D)
        np.testing.assert_allclose(line.value, [[1.0, 1.0], [4.0, 5.0]])
        assert line.length == pytest.approx(5.0)


class TestVectors2D:
    """Batch vectors."""

    @pytest.fixture
    def vectors(self) -> Vectors2D:
        """Three vectors with norms 5, 1, and 0."""
        return Vectors2D(value=np.array([[3.0, 4.0], [0.0, 1.0], [0.0, 0.0]]))

    @pytest.mark.parametrize("shape", [(2,), (3, 3)])
    def test_invalid_shape_raises(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError):
            Vectors2D(value=np.zeros(shape))

    def test_components_and_norm(self, vectors: Vectors2D) -> None:
        np.testing.assert_allclose(vectors.x, [3.0, 0.0, 0.0])
        np.testing.assert_allclose(vectors.y, [4.0, 1.0, 0.0])
        np.testing.assert_allclose(vectors.norm, [5.0, 1.0, 0.0])

    def test_indexing(self, vectors: Vectors2D) -> None:
        item = vectors[0]
        assert isinstance(item, Vector2D)
        assert item.norm == pytest.approx(5.0)
        sliced = vectors[1:]
        assert isinstance(sliced, Vectors2D)
        assert len(sliced) == 2

    def test_iteration(self, vectors: Vectors2D) -> None:
        norms = [vector.norm for vector in vectors]
        np.testing.assert_allclose(norms, vectors.norm)


class TestVector2DPair:
    """Pairwise relations between two vectors."""

    def test_orthogonal_pair(self) -> None:
        pair = Vector2DPair(
            vector1=Vector2D(value=np.array([2.0, 0.0])),
            vector2=Vector2D(value=np.array([0.0, 3.0])),
        )
        assert pair.is_orthogonal
        assert not pair.is_parallel
        assert pair.dot_product == pytest.approx(0.0)
        assert pair.cross_product == pytest.approx(6.0)
        np.testing.assert_allclose(pair.angle.degree, 90.0)

    def test_antiparallel_angle(self) -> None:
        pair = Vector2DPair(
            vector1=Vector2D(value=np.array([1.0, 0.0])),
            vector2=Vector2D(value=np.array([-2.0, 0.0])),
        )
        assert pair.is_parallel
        np.testing.assert_allclose(pair.angle.degree, 180.0)

    def test_angle_is_unsigned(self) -> None:
        counter_clockwise = Vector2DPair(
            vector1=Vector2D(value=np.array([1.0, 0.0])),
            vector2=Vector2D(value=np.array([1.0, 1.0])),
        )
        clockwise = Vector2DPair(
            vector1=Vector2D(value=np.array([1.0, 0.0])),
            vector2=Vector2D(value=np.array([1.0, -1.0])),
        )
        np.testing.assert_allclose(counter_clockwise.angle.degree, 45.0)
        np.testing.assert_allclose(clockwise.angle.degree, 45.0)
