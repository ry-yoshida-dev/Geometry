"""
Single 2D line segment API extending the abstract base in base.

This module validates the storage array with shape (2, 2), with start and end
as (x, y) rows; exposes endpoints as Point2D; and integrates with Shapely
LineString and Vector2D for displacement along the segment.
"""
from __future__ import annotations

from ...array_types import FloatArray
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString

from ..point import Point2D, Points2D
from ..vector import Vector2D
from .base import Line2D as Line2DBase


@dataclass
class Line2D(Line2DBase[float, LineString]):
    """
    One line segment stored as two (x, y) rows in absolute pixel coordinates.

    The first row is the start point, the second row is the end point.
    Convention matches the rest of the planar package: x increases to the
    right, y increases downward (image / row-major indexing).

    Attributes
    ----------
    value : FloatArray
        Shape (2, 2). Row 0 is (x_start, y_start), row 1 is (x_end, y_end).
    """

    def __post_init__(self) -> None:
        """
        Validate type and shape of the storage array after initialization.

        Raises
        ------
        TypeError
            If the storage is not a NumPy ndarray.
        ValueError
            If the storage does not have shape (2, 2).
        """
        if self.value.shape != (2, 2):
            raise ValueError(f"value must have shape (2, 2), got {self.value.shape}")

    @property
    def start(self) -> Point2D:
        """
        First endpoint of the segment (row 0 of the value array).

        Returns
        -------
        Point2D
            Start point in pixel coordinates.
        """
        return Point2D(value=self.value[0])

    @property
    def end(self) -> Point2D:
        """
        Second endpoint of the segment (row 1 of the value array).

        Returns
        -------
        Point2D
            End point in pixel coordinates.
        """
        return Point2D(value=self.value[1])

    @property
    def length(self) -> float:
        """
        Euclidean length of the segment (distance from start to end).

        Returns
        -------
        float
            Length in the same units as coordinates (pixels).
        """
        return float(np.linalg.norm(self.end.value - self.start.value))

    @property
    def shapely(self) -> LineString:
        """
        This segment as a Shapely LineString (two vertices, not closed).

        Returns
        -------
        LineString
            Open line from the start point to the end point.
        """
        return LineString([(self.start.x, self.start.y), (self.end.x, self.end.y)])

    @property
    def center(self) -> Point2D:
        """
        Midpoint of the segment (arithmetic mean of the two endpoints).

        Returns
        -------
        Point2D
            Center (x, y) as a point.
        """
        return Point2D(value=np.mean(self.value, axis=0))

    @property
    def vector2d(self) -> Vector2D:
        """
        Displacement from start to end as a 2D vector.

        Returns
        -------
        Vector2D
            End minus start, in the same coordinate frame as the value array.
        """
        return self.end - self.start

    @property
    def points2d(self) -> Points2D:
        """
        Both endpoints as a Points2D collection.

        Returns
        -------
        Points2D
            Two rows, shape (2, 2); row order is start, then end.
        """
        return Points2D(value=np.array([self.start.value, self.end.value]))

    @property
    def x(self) -> FloatArray:
        """
        x-coordinates of the start and end points.

        Returns
        -------
        FloatArray
            Length 2: x_start, then x_end.
        """
        return self.value[:, 0]

    @property
    def y(self) -> FloatArray:
        """
        y-coordinates of the start and end points.

        Returns
        -------
        FloatArray
            Length 2: y_start, then y_end.
        """
        return self.value[:, 1]

    @classmethod
    def from_two_points(cls, start_point: Point2D, end_point: Point2D) -> Line2D:
        """
        Build a segment from two Point2D instances.

        Parameters
        ----------
        start_point : Point2D
            First endpoint; becomes row 0 of the value array.
        end_point : Point2D
            Second endpoint; becomes row 1 of the value array.

        Returns
        -------
        Line2D
            Segment from start_point to end_point.
        """
        return cls(value=np.array([start_point.value, end_point.value]))
