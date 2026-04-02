from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import TYPE_CHECKING
from .base import Point3D as Point3DBase

if TYPE_CHECKING:
    from ..vector import Vector3D

@dataclass
class Point3D(Point3DBase[float]):
    value: np.ndarray

    def __post_init__(self):
        if self.value.shape != (3,):
            raise ValueError("value must have shape (3,)")
    
    @property
    def x(self) -> float:
        return self.value[0]
    
    @property
    def y(self) -> float:
        return self.value[1]
    
    @property
    def z(self) -> float:
        return self.value[2]
    
    @property
    def array(self) -> np.ndarray:
        return self.value
    
    @property
    def tuple(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z
    
    @property
    def list(self) -> list[float]:
        return [self.x, self.y, self.z]

    def __add__(self, other: Point3D) -> Point3D:
        return Point3D(value=self.value + other.value)

    def __sub__(
        self, 
        other: Point3D
        ) -> "Vector3D":
        """
        Subtract two points.

        Parameters
        ----------
        other: Point3D
            The other point.
        """
        from ..vector import Vector3D
        subtraction = self.value - other.value
        return Vector3D(value=subtraction)