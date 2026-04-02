"""
Abstract 3D vector protocol for scalar or batch storage.

The type parameter T is either float (single-component or scalar magnitude) or
np.ndarray (per-row values for a batch).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T", float, np.ndarray)


@dataclass
class Vector3D(ABC, Generic[T]):
    """
    Abstract single vector or batch stored as a NumPy array.

    A single vector has shape (3,): (dx, dy, dz).
    A batch has shape (N, 3): one row per vector.
    """

    value: np.ndarray

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
    def norm(self) -> T:
        """Euclidean length (magnitude) of the vector(s)."""

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        arr_str = np.array2string(self.value, precision=2, separator=", ")
        return f"{cls_name}(value={arr_str})"
