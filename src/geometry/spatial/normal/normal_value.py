from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from cartesian_axis import CartesianCoordinateSystem

@dataclass
class NormalValue:
    """
    A class representing a normal value.

    Parameters
    ----------
    value: np.ndarray
        The normal value represented 3D vector in [-1, 1] value range.
    coordinate_system: CartesianCoordinateSystem
        The coordinate system.
    """
    value: np.ndarray
    coordinate_system: CartesianCoordinateSystem

    def __post_init__(self):
        if self.value.shape != (3,):
            raise ValueError(f"Normal value must have shape (3,), got {self.value.shape}")
        if self.value.min() < -1 or self.value.max() > 1:
            raise ValueError(f"Normal value must be in [-1, 1] value range, got {self.value.min()} and {self.value.max()}")

    def __str__(self) -> str:
        return f"NormalValue(value={self.value}, coordinate_handedness={self.coordinate_system.coordinate_handedness}, axis_orientation={self.coordinate_system.axis_orientation})"


