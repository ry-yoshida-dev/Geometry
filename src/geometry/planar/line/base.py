"""
Abstract 2D line segment protocol (two endpoints) for scalar or batch storage.

The type parameter T is either float (single-segment length) or NumericArray
(per-row lengths for a batch). Coordinates follow image space (x right, y down).
"""
from __future__ import annotations

from ...array_types import NumericArray
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T", float, NumericArray)
S = TypeVar("S")


@dataclass
class Line2D(ABC, Generic[T, S]):
    """
    Abstract line segment or batch stored as a NumPy array.

    A single segment has shape (2, 2): start and end as (x, y) rows.
    A batch has shape (N, 2, 2).

    Attributes
    ----------
    value : NumericArray
        One segment: shape (2, 2). A batch: shape (N, 2, 2).
    """

    value: NumericArray

    @property
    @abstractmethod
    def length(self) -> T:
        """
        Euclidean length of the segment or each segment.

        Returns
        -------
        T
            Scalar length, or an array of shape (N,) for a batch.
        """

    @property
    @abstractmethod
    def shapely(self) -> S:
        """
        Shapely representation: one LineString, or a list of LineStrings.

        Returns
        -------
        S
            Concrete representation chosen by the subclass.
        """

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        arr_str = np.array2string(self.value, precision=2, separator=", ")
        return f"{cls_name}(value={arr_str})"
