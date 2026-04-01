"""
Batch of 2D points (N rows, two columns) extending base.Point2D.

Validates shape (N, 2), provides column-wise x/y, convex hull helpers, and
Shapely polygon access when the collection is interpreted as a boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Union

import numpy as np
from scipy.spatial import ConvexHull # type: ignore
from shapely.geometry import Polygon

from .base import Point2D as Point2DBase
from .point import Point2D


@dataclass
class Points2D(Point2DBase[np.ndarray, Polygon]):
    """
    N points with value of shape (N, 2).

    Each row is (x, y) in pixel coordinates. Optional is_convex_hull selects
    how shapely builds geometry from the rows.

    Attributes
    ----------
    value : np.ndarray
        Shape (N, 2) in pixel coordinates.
    is_convex_hull : bool
        If True, shapely uses the convex hull polygon; otherwise vertices
        are used in row order as a ring (see Shapely Polygon semantics).
    """

    is_convex_hull: bool = False

    def __post_init__(self) -> None:
        """
        Coerce to ndarray and validate rank and trailing dimensions.

        Raises
        ------
        TypeError
            If the storage is not a NumPy ndarray.
        ValueError
            If the array is not two-dimensional with two columns.
        """
        v = np.asarray(self.value)
        if v.ndim != 2:
            raise ValueError("coordinates must have shape (n, 2)")
        if v.shape[1] != 2:
            raise ValueError("coordinates must have shape (n, 2)")
        self.value = v

    @property
    def x(self) -> np.ndarray:
        """
        All x-coordinates (first column of the value array).

        Returns
        -------
        np.ndarray
            Shape (N,).
        """
        return self.value[:, 0]

    @property
    def y(self) -> np.ndarray:
        """
        All y-coordinates (second column of the value array).

        Returns
        -------
        np.ndarray
            Shape (N,).
        """
        return self.value[:, 1]

    @property
    def area(self) -> float:
        """
        Area of the polygon returned by shapely (Shapely's signed area).

        Returns
        -------
        float
            Geometric area in coordinate units squared.
        """
        return float(self.shapely.area)

    @property
    def center(self) -> Point2D:
        """
        Centroid of the rows (arithmetic mean along axis 0).

        Returns
        -------
        Point2D
            Mean (x, y) as a single point.
        """
        return Point2D(value=np.mean(self.value, axis=0))

    def __repr__(self) -> str:
        return f"Points2D(n={len(self)}, shape={self.value.shape})"

    def __len__(self) -> int:
        return len(self.value)

    def __getitem__(
        self,
        index: Union[int, slice],
    ) -> Union[Point2D, "Points2D"]:
        """
        Index or slice rows of the batch.

        Parameters
        ----------
        index : int | slice
            Row index for one Point2D, or slice for a sub-batch.

        Returns
        -------
        Point2D | Points2D
            One point or a new Points2D sharing the same is_convex_hull flag.
        """
        result = self.value[index]
        if isinstance(index, slice):
            return Points2D(value=result, is_convex_hull=self.is_convex_hull)
        return Point2D(value=result)

    def append(
        self,
        coord: Union[np.ndarray, tuple[float, float]],
    ) -> None:
        """
        Append one (x, y) row in place.

        Parameters
        ----------
        coord : np.ndarray | tuple[float, float]
            Length-2 coordinate in the same dtype as value.

        Raises
        ------
        ValueError
            If coord does not resolve to shape (2,).
        """
        coord = np.asarray(coord, dtype=self.value.dtype)
        if coord.shape != (2,):
            raise ValueError("A coordinate must be a length-2 array or tuple.")
        self.value = np.vstack([self.value, coord])

    def delete(self, index: int) -> None:
        """
        Remove the point at index along axis 0.

        Parameters
        ----------
        index : int
            Row index to delete.
        """
        self.value = np.delete(self.value, index, axis=0)

    def concat(self, other: Points2D) -> Points2D:
        """
        Stack this batch with another along the row axis (non-destructive).

        Parameters
        ----------
        other : Points2D
            Second collection; is_convex_hull is taken from self.

        Returns
        -------
        Points2D
            Rows of self followed by rows of other.
        """
        return Points2D(
            value=np.vstack([self.value, other.value]),
            is_convex_hull=self.is_convex_hull,
        )

    def __iter__(self) -> Iterator[Point2D]:
        for row in self.value:
            yield Point2D(value=row)

    def __contains__(
        self,
        coord: Union[np.ndarray, tuple[float, float]],
    ) -> bool:
        """
        Whether coord appears as a row (exact match via in on ndarray).

        Parameters
        ----------
        coord : np.ndarray | tuple[float, float]
            Candidate (x, y).

        Returns
        -------
        bool
            True if a row equals coord under NumPy membership rules.
        """
        return coord in self.value

    @property
    def convex_hull_points(self) -> np.ndarray:
        """
        Vertices of the 2D convex hull in SciPy order.

        Returns
        -------
        np.ndarray
            Shape (K, 2) where K is the number of hull vertices.
        """
        convex_hull = ConvexHull(self.value)
        return self.value[convex_hull.vertices]

    @property
    def convex_hull_polygon(self) -> Polygon:
        """
        Convex hull as a Shapely Polygon.

        Returns
        -------
        Polygon
            Closed ring from hull vertices.
        """
        return Polygon(self.convex_hull_points)

    @property
    def shapely(self) -> Polygon:
        """
        Polygon built from these points or from their convex hull.

        When is_convex_hull is True, the hull polygon is returned; otherwise
        rows define the polygon ring in order (see Shapely for degenerate cases).

        Returns
        -------
        Polygon
            Possibly invalid or empty for collinear or duplicate vertices.
        """
        if self.is_convex_hull:
            return self.convex_hull_polygon
        return Polygon(self.value)
