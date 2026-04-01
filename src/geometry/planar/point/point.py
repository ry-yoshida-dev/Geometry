"""
Single 2D point API extending the abstract base in base.

This module validates the storage vector with shape (2,) and exposes scalar
x, y; helpers for tuple/list copies; and integration with Shapely Point and
Vector2D for displacement between points.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point as ShapelyPoint

from .base import Point2D as Point2DBase


@dataclass
class Point2D(Point2DBase[float]):
    """
    One (x, y) pair in absolute pixel coordinates stored as a length-2 vector.

    Convention matches the rest of the planar package: x increases to the
    right, y increases downward (image / row-major indexing).

    Attributes
    ----------
    value : np.ndarray
        Shape (2,). Index 0 is x, index 1 is y.
    """

    def __post_init__(self) -> None:
        """
        Validate type and shape of the storage array after initialization.

        Raises
        ------
        TypeError
            If the storage is not a NumPy ndarray.
        ValueError
            If the storage does not have shape (2,).
        """
        if not isinstance(self.value, np.ndarray):
            raise TypeError("value must be a numpy array")
        if self.value.shape != (2,):
            raise ValueError(
                f"value must have shape (2,), got shape {self.value.shape}"
            )

    @property
    def array(self) -> np.ndarray:
        """
        A copy of the underlying coordinate vector.

        Returns
        -------
        np.ndarray
            Shape (2,); modifying the copy does not affect this point.
        """
        return self.value.copy()

    @property
    def tuple(self) -> tuple[float, float]:
        """
        The point as a plain Python (x, y) pair.

        Returns
        -------
        tuple[float, float]
            (x, y) in the same units as the value array.
        """
        return self.x, self.y

    @property
    def list(self) -> list[float]:
        """
        The point as a two-element list [x, y].

        Returns
        -------
        list[float]
            Mutable list copy of the coordinates.
        """
        return [self.x, self.y]

    @property
    def x(self) -> float:
        """
        Horizontal coordinate (first component of the value vector).

        Returns
        -------
        float
            x in pixel coordinates.
        """
        return float(self.value[0])

    @property
    def y(self) -> float:
        """
        Vertical coordinate (second component of the value vector).

        Returns
        -------
        float
            y in pixel coordinates.
        """
        return float(self.value[1])

    @property
    def shapely(self) -> ShapelyPoint:
        """
        This location as a Shapely Point (zero-dimensional geometry).

        Returns
        -------
        ShapelyPoint
            Same (x, y) as this instance.
        """
        return ShapelyPoint(self.x, self.y)

    def __repr__(self) -> str:
        return f"Point2D(x={self.x}, y={self.y})"

    def __add__(self, other: Point2D) -> np.ndarray:
        """
        Element-wise sum of coordinate vectors (returns a raw ndarray).

        Parameters
        ----------
        other : Point2D
            Second operand; same shape semantics as self.

        Returns
        -------
        np.ndarray
            Shape (2,); sum of the two value arrays.
        """
        return self.value + other.value

    def __sub__(self, other: Point2D) -> "Vector2D":
        """
        Displacement from other to self as a 2D vector.

        Parameters
        ----------
        other : Point2D
            Reference point; result is self minus other.

        Returns
        -------
        Vector2D
            Difference in the same coordinate frame as the value arrays.
        """
        from ..vector import Vector2D

        return Vector2D(value=self.value - other.value)

    def __mul__(self, other: float) -> np.ndarray:
        """
        Scale coordinates by a scalar (returns a raw ndarray).

        Parameters
        ----------
        other : float
            Multiplier applied to both x and y.

        Returns
        -------
        np.ndarray
            Shape (2,); scaled coordinates.
        """
        return self.value * other
