"""
Abstract 3D line segment protocol for scalar or batch storage.

The type parameter T is either float (single-segment length) or NumericArray
(per-row lengths for a batch).
"""
from __future__ import annotations

from ...array_types import NumericArray, NumericScalar
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T", float, NumericArray)


@dataclass
class Line3D(ABC, Generic[T]):
    """
    Abstract line segment or batch stored as a NumPy array.

    A single segment has shape (2, 3): start and end as (x, y, z) rows.
    A batch has shape (N, 2, 3).
    """

    value: NumericArray

    @property
    @abstractmethod
    def length(self) -> T:
        """Euclidean length of the segment(s)."""

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        arr_str = np.array2string(self.value, precision=2, separator=", ")
        return f"{cls_name}(value={arr_str})"
