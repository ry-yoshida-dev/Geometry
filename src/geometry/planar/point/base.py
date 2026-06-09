"""
Abstract 2D point protocol for scalar or batch storage.

The type parameter T is either float (single coordinate component) or
FloatArray (per-row components for a batch). Coordinates follow image space
(x right, y down).
"""
from __future__ import annotations

from ...array_types import FloatArray
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
from shapely.geometry.base import BaseGeometry

T = TypeVar("T", float, FloatArray)
G = TypeVar("G", bound=BaseGeometry)


@dataclass
class Point2D(ABC, Generic[T, G]):
    """
    Abstract single point or batch stored as a NumPy array.

    A single point has shape (2,): (x, y).
    A batch has shape (N, 2): one row per point.

    Attributes
    ----------
    value : FloatArray
        One point: shape (2,). A batch: shape (N, 2).
    """

    value: FloatArray

    @property
    @abstractmethod
    def x(self) -> T:
        """
        x-coordinate(s) of the point(s).

        Returns
        -------
        T
            Scalar x for one point, or an array of shape (N,) for a batch.
        """

    @property
    @abstractmethod
    def y(self) -> T:
        """
        y-coordinate(s) of the point(s).

        Returns
        -------
        T
            Scalar y for one point, or an array of shape (N,) for a batch.
        """

    @property
    @abstractmethod
    def shapely(self) -> G:
        """
        Shapely geometry for this point or point set.

        Returns
        -------
        G
            Concrete geometry type of the subclass (e.g. Point or Polygon).
        """

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        arr_str = np.array2string(self.value, precision=2, separator=", ")
        return f"{cls_name}(value={arr_str})"
