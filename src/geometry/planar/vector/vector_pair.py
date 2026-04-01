from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from .vector import Vector2D
from units import Angle, AngleUnit

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
            value=theta, 
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
        return np.dot(self.vector1.value, self.vector2.value)

    @property
    def cross_product(self) -> np.ndarray:
        """
        Return the cross product of the two vectors.

        Returns
        -------
        np.ndarray: The cross product of the two vectors.
        """
        return np.cross(self.vector1.value, self.vector2.value)



