"""
Single 2D displacement vector extending the abstract base in base.

Validates storage shape (2,); exposes scalar components, magnitude, unit
direction; and helpers for orthogonality, parallelism, and conversion to Line2D.
"""
from __future__ import annotations

from ...array_types import FloatArray
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..point import Point2D
from .base import Vector2D as Vector2DBase

if TYPE_CHECKING:
    from ..line import Line2D


@dataclass
class Vector2D(Vector2DBase[float]):
    """
    One 2D vector (dx, dy) in the same coordinate frame as Point2D / Line2D.

    Attributes
    ----------
    value : FloatArray
        Shape (2,). Index 0 is x (horizontal), index 1 is y (vertical).
    """

    def __post_init__(self) -> None:
        """
        Validate type and shape of the storage array.

        Raises
        ------
        TypeError
            If the storage is not a NumPy ndarray.
        ValueError
            If the storage does not have shape (2,).
        """
        if self.value.shape != (2,):
            raise ValueError("value must have shape (2,)")

    @property
    def x(self) -> float:
        """
        Horizontal component (first entry of value).

        Returns
        -------
        float
            dx in coordinate units.
        """
        return float(self.value[0])

    @property
    def y(self) -> float:
        """
        Vertical component (second entry of value).

        Returns
        -------
        float
            dy in coordinate units.
        """
        return float(self.value[1])

    @property
    def norm(self) -> float:
        """
        Euclidean length of the vector.

        Returns
        -------
        float
            sqrt(dx**2 + dy**2) in the same units as components.
        """
        return float(np.linalg.norm(self.value))

    @property
    def unit_vector(self) -> FloatArray:
        """
        Unit direction with the same orientation as value.

        Returns
        -------
        FloatArray
            Shape (2,); value / norm.

        Raises
        ------
        ValueError
            If norm is zero.
        """
        n = self.norm
        if n == 0:
            raise ValueError("Cannot compute a unit vector from a zero-length vector.")
        return self.value / n

    def __repr__(self) -> str:
        return f"Vector2D(x={self.x}, y={self.y})"

    def is_orthogonal(
        self,
        other_vector: Vector2D,
        atol: float = 1e-4,
    ) -> bool:
        """
        Whether this vector and other_vector are orthogonal (dot product ~ 0).

        Parameters
        ----------
        other_vector : Vector2D
            Second operand.
        atol : float
            Absolute tolerance on the dot product.

        Returns
        -------
        bool
            True if |dot(self, other)| <= atol in the sense of np.isclose.
        """
        return bool(np.isclose(np.dot(self.value, other_vector.value), 0.0, atol=atol))

    def is_parallel(
        self,
        other_vector: Vector2D,
        atol: float = 1e-4,
    ) -> bool:
        """
        Whether this vector and other_vector are parallel (2D cross ~ 0).

        Parameters
        ----------
        other_vector : Vector2D
            Second operand.
        atol : float
            Absolute tolerance on the scalar cross product.

        Returns
        -------
        bool
            True if the 2D cross product is close to zero.
        """
        return bool(np.allclose(np.cross(self.value, other_vector.value), 0.0, atol=atol))

    @classmethod
    def from_two_points(
        cls,
        start_point: Point2D,
        end_point: Point2D,
    ) -> Vector2D:
        """
        Displacement from start_point to end_point.

        Parameters
        ----------
        start_point : Point2D
            Tail of the arrow.
        end_point : Point2D
            Tip of the arrow.

        Returns
        -------
        Vector2D
            end_point - start_point as a vector.
        """
        return end_point - start_point

    def to_line(self, origin: Point2D) -> Line2D:
        """
        Line segment from origin along this vector (length = norm).

        Parameters
        ----------
        origin : Point2D
            Start of the segment.

        Returns
        -------
        Line2D
            From origin to origin + value.
        """
        from ..line import Line2D

        start_point = origin
        end_point = Point2D(value=origin.value + self.value)
        return Line2D.from_two_points(start_point=start_point, end_point=end_point)
