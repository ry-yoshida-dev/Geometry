"""
Abstract 3D point protocol for scalar or batch storage.

The type parameter T is either float (single coordinate component) or
np.ndarray (per-row components for a batch).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T", float, np.ndarray)


@dataclass
class Point3D(ABC, Generic[T]):
    """
    Abstract single point or batch stored as a NumPy array.

    A single point has shape (3,): (x, y, z).
    A batch has shape (N, 3): one row per point.
    """

    value: np.ndarray

    @property
    @abstractmethod
    def x(self) -> T:
        """x-coordinate(s) of the point(s)."""

    @property
    @abstractmethod
    def y(self) -> T:
        """y-coordinate(s) of the point(s)."""

    @property
    @abstractmethod
    def z(self) -> T:
        """z-coordinate(s) of the point(s)."""

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        arr_str = np.array2string(self.value, precision=2, separator=", ")
        return f"{cls_name}(value={arr_str})"
