"""Tests for Vector3D, Vectors3D, Vector3DPair, and OrthonormalBasis."""

from __future__ import annotations

import numpy as np
import pytest
from geometry.spatial import OrthonormalBasis, Point3D, Vector3D, Vector3DPair, Vectors3D


def make_vector(x: float, y: float, z: float) -> Vector3D:
    """Build a Vector3D from three components."""
    return Vector3D(value=np.array([x, y, z]))


class TestVector3D:
    """Single-vector components, norm, and relations."""

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            Vector3D(value=np.array([1.0, 2.0]))

    def test_components_and_norm(self) -> None:
        vector = make_vector(1.0, 2.0, 2.0)
        assert (vector.x, vector.y, vector.z) == (1.0, 2.0, 2.0)
        assert vector.norm == pytest.approx(3.0)

    def test_unit_vector(self) -> None:
        np.testing.assert_allclose(make_vector(0.0, 0.0, 5.0).unit_vector, [0.0, 0.0, 1.0])

    def test_zero_unit_vector_raises(self) -> None:
        with pytest.raises(ValueError):
            _ = make_vector(0.0, 0.0, 0.0).unit_vector

    def test_is_unit(self) -> None:
        assert make_vector(0.0, 1.0, 0.0).is_unit
        assert not make_vector(0.0, 2.0, 0.0).is_unit

    def test_cross_follows_right_hand_rule(self) -> None:
        cross = make_vector(1.0, 0.0, 0.0).cross(make_vector(0.0, 1.0, 0.0))
        np.testing.assert_allclose(cross.value, [0.0, 0.0, 1.0])

    def test_matmul_is_dot_product(self) -> None:
        assert make_vector(1.0, 2.0, 3.0) @ make_vector(4.0, 5.0, 6.0) == pytest.approx(32.0)

    def test_relations(self) -> None:
        x_axis = make_vector(1.0, 0.0, 0.0)
        assert x_axis.is_parallel(make_vector(-3.0, 0.0, 0.0))
        assert not x_axis.is_parallel(make_vector(1.0, 1.0, 0.0))
        assert x_axis.is_orthogonal(make_vector(0.0, 2.0, 3.0))
        assert not x_axis.is_orthogonal(make_vector(1.0, 1.0, 0.0))

    def test_from_two_points(self) -> None:
        vector = Vector3D.from_two_points(
            start_point=Point3D(value=np.array([1.0, 1.0, 1.0])),
            end_point=Point3D(value=np.array([2.0, 3.0, 4.0])),
        )
        np.testing.assert_allclose(vector.value, [1.0, 2.0, 3.0])


class TestGramSchmidt:
    """Orthonormal basis construction."""

    def test_produces_right_handed_orthonormal_basis(self) -> None:
        basis = make_vector(2.0, 0.0, 0.0).gram_schmidt(make_vector(1.0, 3.0, 0.0))
        np.testing.assert_allclose(basis.e1.value, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(basis.e2.value, [0.0, 1.0, 0.0])
        np.testing.assert_allclose(basis.e3.value, [0.0, 0.0, 1.0])

    def test_general_input_is_orthonormal(self) -> None:
        basis = make_vector(1.0, 2.0, 3.0).gram_schmidt(make_vector(-2.0, 0.5, 4.0))
        matrix = np.stack([vector.value for vector in basis])
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(matrix) == pytest.approx(1.0)

    def test_parallel_input_raises(self) -> None:
        with pytest.raises(ValueError):
            make_vector(1.0, 2.0, 3.0).gram_schmidt(make_vector(2.0, 4.0, 6.0))


class TestOrthonormalBasis:
    """Validation and sequence protocol."""

    @pytest.fixture
    def standard_basis(self) -> OrthonormalBasis:
        """Standard basis of R^3."""
        return OrthonormalBasis(
            e1=make_vector(1.0, 0.0, 0.0),
            e2=make_vector(0.0, 1.0, 0.0),
            e3=make_vector(0.0, 0.0, 1.0),
        )

    def test_non_unit_raises(self) -> None:
        with pytest.raises(ValueError):
            OrthonormalBasis(
                e1=make_vector(2.0, 0.0, 0.0),
                e2=make_vector(0.0, 1.0, 0.0),
                e3=make_vector(0.0, 0.0, 1.0),
            )

    def test_non_orthogonal_raises(self) -> None:
        diagonal = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        with pytest.raises(ValueError):
            OrthonormalBasis(
                e1=make_vector(1.0, 0.0, 0.0),
                e2=Vector3D(value=diagonal),
                e3=make_vector(0.0, 0.0, 1.0),
            )

    def test_sequence_protocol(self, standard_basis: OrthonormalBasis) -> None:
        assert len(standard_basis) == 3
        assert standard_basis[0] is standard_basis.e1
        assert standard_basis[-1] is standard_basis.e3
        assert list(standard_basis) == [standard_basis.e1, standard_basis.e2, standard_basis.e3]

    def test_out_of_range_index_raises(self, standard_basis: OrthonormalBasis) -> None:
        with pytest.raises(IndexError):
            _ = standard_basis[3]


class TestVector3DPair:
    """Pair validation and relations."""

    def test_parallel_pair_raises(self) -> None:
        with pytest.raises(ValueError):
            Vector3DPair(vector1=make_vector(1.0, 0.0, 0.0), vector2=make_vector(2.0, 0.0, 0.0))

    def test_orthogonal_pair(self) -> None:
        pair = Vector3DPair(vector1=make_vector(1.0, 0.0, 0.0), vector2=make_vector(0.0, 1.0, 0.0))
        assert pair.is_orthogonal
        assert not pair.is_parallel


class TestVectors3D:
    """Batch vectors and angle conversion."""

    @pytest.fixture
    def vectors(self) -> Vectors3D:
        """Unit axes plus a scaled diagonal in the xy plane."""
        return Vectors3D(
            value=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 2.0],
                    [3.0, 3.0, 0.0],
                ]
            )
        )

    @pytest.mark.parametrize("shape", [(3,), (2, 2)])
    def test_invalid_shape_raises(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError):
            Vectors3D(value=np.zeros(shape))

    def test_components(self, vectors: Vectors3D) -> None:
        np.testing.assert_allclose(vectors.x, [1.0, 0.0, 0.0, 3.0])
        np.testing.assert_allclose(vectors.y, [0.0, 1.0, 0.0, 3.0])
        np.testing.assert_allclose(vectors.z, [0.0, 0.0, 2.0, 0.0])

    def test_unit_vectors(self, vectors: Vectors3D) -> None:
        np.testing.assert_allclose(np.linalg.norm(vectors.unit_vectors, axis=1), 1.0)

    def test_zero_row_unit_vectors_raises(self) -> None:
        with pytest.raises(ValueError):
            _ = Vectors3D(value=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])).unit_vectors

    def test_azimuthal_angles(self, vectors: Vectors3D) -> None:
        angles = vectors.to_azimuthal_angles(up_index=2)
        np.testing.assert_allclose(angles.degree, [90.0, 90.0, 0.0, 90.0])

    def test_polar_angles(self, vectors: Vectors3D) -> None:
        angles = vectors.to_polar_angles(forward_index=0, right_index=1)
        np.testing.assert_allclose(angles.degree, [0.0, 90.0, 0.0, 45.0])

    @pytest.mark.parametrize("up_index", [-1, 3])
    def test_invalid_up_index_raises(self, vectors: Vectors3D, up_index: int) -> None:
        with pytest.raises(ValueError):
            vectors.to_azimuthal_angles(up_index=up_index)

    @pytest.mark.parametrize(("forward_index", "right_index"), [(0, 0), (3, 1), (0, -1)])
    def test_invalid_polar_indices_raise(self, vectors: Vectors3D, forward_index: int, right_index: int) -> None:
        with pytest.raises(ValueError):
            vectors.to_polar_angles(forward_index=forward_index, right_index=right_index)

    def test_sequence_protocol(self, vectors: Vectors3D) -> None:
        assert len(vectors) == 4
        assert isinstance(vectors[2], Vector3D)
        assert vectors[2].norm == pytest.approx(2.0)
        assert [vector.norm for vector in vectors] == pytest.approx([1.0, 1.0, 2.0, np.sqrt(18.0)])
