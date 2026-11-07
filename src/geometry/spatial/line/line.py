from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...array_types import NumericArray
from ..point import Point3D
from ..vector import Vector3D
from .base import Line3D as Line3DBase


@dataclass
class Line3D(Line3DBase[float]):
    """
    A 3D line segment stored as its start and end points in a (2, 3) array.
    """

    value: NumericArray

    def __post_init__(self) -> None:
        if self.value.shape != (2, 3):
            raise ValueError("value must have shape (2, 3)")

    @property
    def start(self) -> Point3D:
        """
        Get the start point of the line.

        Returns
        -------
        Point3D
            The first row of `value`.
        """
        return Point3D(value=self.value[0])

    @property
    def end(self) -> Point3D:
        """
        Get the end point of the line.

        Returns
        -------
        Point3D
            The second row of `value`.
        """
        return Point3D(value=self.value[1])

    @property
    def vector3d(self) -> Vector3D:
        """
        Get the vector of the line.

        Returns
        -------
        Vector3D: The vector of the line.
        """
        return self.end - self.start

    @classmethod
    def from_start_and_end(cls, start: Point3D, end: Point3D) -> Line3D:
        """
        Create a line from the start and end points.

        Parameters
        ----------
        start: Point3D
            The start point of the line.
        end: Point3D
            The end point of the line.

        Returns
        -------
        Line3D: The line from the start and end points.
        """
        return cls(value=np.array([start.value, end.value]))

    @property
    def length(self) -> float:
        """
        Get the Euclidean length of the line.

        Returns
        -------
        float
            The distance from the start point to the end point.
        """
        return float(np.linalg.norm(self.vector3d.value))
