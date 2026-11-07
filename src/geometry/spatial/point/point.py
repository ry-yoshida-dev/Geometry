from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from ...array_types import NumericArray, NumericScalar
from .base import Point3D as Point3DBase

if TYPE_CHECKING:
    from ..vector import Vector3D


@dataclass
class Point3D(Point3DBase[NumericScalar]):
    """
    A 3D point stored as a (3,) array.
    """

    value: NumericArray

    def __post_init__(self) -> None:
        if self.value.shape != (3,):
            raise ValueError("value must have shape (3,)")

    @property
    def x(self) -> NumericScalar:
        """
        Get the X coordinate.

        Returns
        -------
        NumericScalar
            The X coordinate.
        """
        return cast(NumericScalar, self.value[0].item())

    @property
    def y(self) -> NumericScalar:
        """
        Get the Y coordinate.

        Returns
        -------
        NumericScalar
            The Y coordinate.
        """
        return cast(NumericScalar, self.value[1].item())

    @property
    def z(self) -> NumericScalar:
        """
        Get the Z coordinate.

        Returns
        -------
        NumericScalar
            The Z coordinate.
        """
        return cast(NumericScalar, self.value[2].item())

    @property
    def array(self) -> NumericArray:
        """
        Get the coordinates as an array.

        Returns
        -------
        NumericArray
            The underlying (3,) array.
        """
        return self.value

    @property
    def tuple(self) -> tuple[float, float, float]:
        """
        Get the coordinates as a tuple.

        Returns
        -------
        tuple[float, float, float]
            The (x, y, z) coordinates.
        """
        return self.x, self.y, self.z

    @property
    def list(self) -> list[float]:
        """
        Get the coordinates as a list.

        Returns
        -------
        list[float]
            The [x, y, z] coordinates.
        """
        return [self.x, self.y, self.z]

    def __add__(self, other: Point3D) -> Point3D:
        return Point3D(value=self.value + other.value)

    def __sub__(self, other: Point3D) -> Vector3D:
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
