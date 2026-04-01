"""
Batch of 2D line segments (N segments, two endpoints each) extending base.Line2D.

Validates shape (N, 2, 2) and provides per-segment length, endpoints, and Shapely access.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
from shapely.geometry import LineString

from ..point import Points2D
from .base import Line2D as Line2DBase
from .line import Line2D


@dataclass
class Lines2D(Line2DBase[np.ndarray, list[LineString]]):
    """
    N line segments with value of shape (N, 2, 2).

    For each segment, row 0 is start (x, y) and row 1 is end (x, y).

    Attributes
    ----------
    value : np.ndarray
        Shape (N, 2, 2) in pixel coordinates.
    """

    def __post_init__(self) -> None:
        """
        Coerce to ndarray and validate rank and trailing dimensions.

        Raises
        ------
        ValueError
            If the array is not three-dimensional with trailing shape (2, 2).
        """
        v = np.asarray(self.value)
        if v.ndim != 3 or v.shape[1:] != (2, 2):
            raise ValueError(
                f"lines must have shape (N, 2, 2), but got {v.shape}"
            )
        self.value = v

    @property
    def start(self) -> Points2D:
        """
        Start point of each segment as Points2D with shape (N, 2).

        Returns
        -------
        Points2D
            All start (x, y) pairs, one row per segment.
        """
        return Points2D(value=self.value[:, 0, :])

    @property
    def end(self) -> Points2D:
        """
        End point of each segment as Points2D with shape (N, 2).

        Returns
        -------
        Points2D
            All end (x, y) pairs, one row per segment.
        """
        return Points2D(value=self.value[:, 1, :])

    @property
    def length(self) -> np.ndarray:
        """
        Euclidean length of each segment.

        Returns
        -------
        np.ndarray
            Shape (N,): one length per segment.
        """
        d = self.value[:, 1, :] - self.value[:, 0, :]
        return np.linalg.norm(d, axis=1)

    @property
    def center(self) -> Points2D:
        """
        Midpoint of each segment.

        Returns
        -------
        Points2D
            Shape (N, 2): center (x, y) per segment.
        """
        return Points2D(value=np.mean(self.value, axis=1))

    @property
    def vectors(self) -> np.ndarray:
        """
        Displacement from start to end for each segment.

        Returns
        -------
        np.ndarray
            Shape (N, 2): end minus start per row.
        """
        return self.value[:, 1, :] - self.value[:, 0, :]

    @property
    def shapely(self) -> list[LineString]:
        """
        One Shapely LineString per segment.

        Returns
        -------
        list[LineString]
            Length N.
        """
        out: list[LineString] = []
        for i in range(len(self)):
            seg = self.value[i]
            out.append(
                LineString([(seg[0, 0], seg[0, 1]), (seg[1, 0], seg[1, 1])])
            )
        return out

    def __len__(self) -> int:
        return self.value.shape[0]

    def __getitem__(self, index: Union[int, slice]) -> Union[Line2D, Lines2D]:
        result = self.value[index]
        if isinstance(index, slice):
            return Lines2D(value=result)
        return Line2D(value=result)
