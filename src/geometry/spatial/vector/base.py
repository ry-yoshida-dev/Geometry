"""
Abstract 3D vector protocol for scalar or batch storage.

The type parameter T is either NumericScalar (single component) or
NumericArray (per-row values for a batch).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ...array_types import FloatArray, NumericArray, NumericScalar


@dataclass
class Vector3D[T: (NumericScalar, NumericArray)](ABC):
    """
    Abstract single vector or batch stored as a NumPy array.

    A single vector has shape (3,): (dx, dy, dz).
    A batch has shape (N, 3): one row per vector.
    """

    value: NumericArray

    @property
    @abstractmethod
    def x(self) -> T:
        """x-component(s) of the vector(s)."""

    @property
    @abstractmethod
    def y(self) -> T:
        """y-component(s) of the vector(s)."""

    @property
    @abstractmethod
    def z(self) -> T:
        """z-component(s) of the vector(s)."""

    @property
    @abstractmethod
    def norm(self) -> float | FloatArray:
        """Euclidean length (magnitude) of the vector(s)."""

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        arr_str = np.array2string(self.value, precision=2, separator=", ")
        return f"{cls_name}(value={arr_str})"
