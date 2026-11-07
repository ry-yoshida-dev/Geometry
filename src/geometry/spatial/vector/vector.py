from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from ...array_types import NumericArray, NumericScalar

if TYPE_CHECKING:
    from .normalized_orthogonal_vectors import OrthonormalBasis
from ..point import Point3D
from .base import Vector3D as Vector3DBase


@dataclass
class Vector3D(Vector3DBase[NumericScalar]):
    """
    A 3D vector stored as a (3,) array.
    """

    value: NumericArray

    def __post_init__(self) -> None:
        if self.value.shape != (3,):
            raise ValueError("value must have shape (3,)")

    @property
    def x(self) -> NumericScalar:
        """
        Get the X component.

        Returns
        -------
        NumericScalar
            The X component.
        """
        return cast(NumericScalar, self.value[0].item())

    @property
    def y(self) -> NumericScalar:
        """
        Get the Y component.

        Returns
        -------
        NumericScalar
            The Y component.
        """
        return cast(NumericScalar, self.value[1].item())

    @property
    def z(self) -> NumericScalar:
        """
        Get the Z component.

        Returns
        -------
        NumericScalar
            The Z component.
        """
        return cast(NumericScalar, self.value[2].item())

    @property
    def norm(self) -> float:
        """
        Get the Euclidean norm of the vector.

        Returns
        -------
        float
            The length of the vector.
        """
        return float(np.linalg.norm(self.value))

    @property
    def unit_vector(self) -> NumericArray:
        """
        Get the vector scaled to unit length.

        Returns
        -------
        NumericArray
            The unit vector, with shape (3,).

        Raises
        ------
        ValueError
            If the vector has zero length.
        """
        norm = np.linalg.norm(self.value)
        if norm == 0:
            raise ValueError("Cannot compute a unit vector from a zero-length vector.")
        return cast(NumericArray, self.value / norm)

    def cross(self, other_vector: Vector3D) -> Vector3D:
        """
        Compute the cross product of the two vectors.

        Parameters
        ----------
        other_vector: Vector3D
            The other vector.

        Returns
        -------
        Vector3D:
            The cross product of the two vectors.
        """
        normal = np.cross(self.value, other_vector.value)
        return Vector3D(value=normal)

    def gram_schmidt(self, other_vector: Vector3D) -> OrthonormalBasis:
        """
        Perform Gram-Schmidt orthogonalization to create an orthonormal basis.

        Given self (u) and other_vector (v), returns two orthonormal vectors (e2, e3)
        such that (e1, e2, e3) form an orthonormal basis where e1 = u/||u||.

        Parameters
        ----------
        other_vector: Vector3D
            The second vector to orthogonalize.

        Returns
        -------
        tuple[Vector3D, Vector3D]:
            The orthonormal vectors (e2, e3).

        Raises
        ------
        ValueError:
            If the two vectors are nearly parallel (cannot form a basis).
        """
        from .normalized_orthogonal_vectors import OrthonormalBasis

        e1 = Vector3D(value=self.unit_vector)

        n_orth = other_vector.value - e1.value * (e1 @ other_vector)
        norm_n_orth = np.linalg.norm(n_orth)
        if norm_n_orth < 1e-10:
            raise ValueError(
                "The two vectors are nearly parallel; orthogonalization would lead to division by zero during normalization."
            )

        e2 = Vector3D(value=n_orth / norm_n_orth)
        e3 = e1.cross(e2)
        e3 = Vector3D(value=e3.value / np.linalg.norm(e3.value))

        return OrthonormalBasis(e1=e1, e2=e2, e3=e3)

    def is_parallel(self, other_vector: Vector3D) -> bool:
        """
        Check whether the vector is parallel to another vector, with an absolute tolerance of 1e-4.

        Parameters
        ----------
        other_vector : Vector3D
            The other vector.

        Returns
        -------
        bool
            True if the cross product of the two vectors is close to zero.
        """
        cross = np.cross(self.value, other_vector.value)
        return np.allclose(cross, 0.0, atol=1e-4)

    def is_orthogonal(self, other_vector: Vector3D) -> bool:
        """
        Check whether the vector is orthogonal to another vector, with an absolute tolerance of 1e-4.

        Parameters
        ----------
        other_vector : Vector3D
            The other vector.

        Returns
        -------
        bool
            True if the dot product of the two vectors is close to zero.
        """
        return bool(np.isclose(np.dot(self.value, other_vector.value), 0.0, atol=1e-4))

    @property
    def is_unit(self) -> bool:
        """
        Check whether the vector has unit length, with an absolute tolerance of 1e-4.

        Returns
        -------
        bool
            True if the norm is close to one.
        """
        return bool(np.isclose(np.linalg.norm(self.value), 1.0, atol=1e-4))

    def __matmul__(self, other_vector: Vector3D) -> float:
        return float(np.dot(self.value, other_vector.value))

    @classmethod
    def from_two_points(cls, start_point: Point3D, end_point: Point3D) -> Vector3D:
        """
        Create a vector from two points.

        Parameters
        ----------
        point1: Point3D
            The first point.
        point2: Point3D
            The second point.

        Returns
        -------
        Vector3D:
            The vector from the two points.
        """
        return end_point - start_point
