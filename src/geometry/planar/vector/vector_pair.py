from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from units import Angle, AngleUnit

from .vector import Vector2D


@dataclass
class Vector2DPair:
    """
    Vector2D Pair.

    Attributes
    ----------
    vector1: Vector2D
        The first vector.
    vector2: Vector2D
        The second vector.
    """
    vector1: Vector2D
    vector2: Vector2D

    @property
    def is_parallel(self) -> bool:
        """
        Check if the two vectors are parallel.

        Returns
        -------
        bool: True if the two vectors are parallel, False otherwise.
        """
        return self.vector1.is_parallel(self.vector2)

    @property
    def is_orthogonal(self) -> bool:
        """
        Check if the two vectors are orthogonal.

        Returns
        -------
        bool: True if the two vectors are orthogonal, False otherwise.
        """
        return self.vector1.is_orthogonal(self.vector2)

    @property
    def angle(self) -> Angle:
        """
        Return the angle between the two vectors.

        Returns
        -------
        Angle: The angle between the two vectors.
        """
        u1 = self.vector1.unit_vector
        u2 = self.vector2.unit_vector
        cos_theta = np.clip(np.dot(u1, u2), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        return Angle(
            value=np.array([theta]),
            unit=AngleUnit.RADIAN
            )

    @property
    def dot_product(self) -> float:
        """
        Return the dot product of the two vectors.

        Returns
        -------
        float: The dot product of the two vectors.
        """
        return float(np.dot(self.vector1.value, self.vector2.value))

    @property
    def cross_product(self) -> float:
        """
        Return the scalar 2D cross product of the two vectors.

        Returns
        -------
        float: The z-component of vector1 x vector2.
        """
        return self.vector1.cross(self.vector2)



