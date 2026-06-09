"""
Abstract 2D vector protocol for scalar or batch storage.

The type parameter T is either float (single-component or scalar magnitude) or
FloatArray (per-row values for a batch). Components follow image space
(x right, y down), consistent with Point2D and Line2D.
"""
from __future__ import annotations

from ...array_types import FloatArray
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T", float, FloatArray)


@dataclass
class Vector2D(ABC, Generic[T]):
    """
    Abstract single vector or batch stored as a NumPy array.

    A single vector has shape (2,): (dx, dy).
    A batch has shape (N, 2): one row per vector.

    Attributes
    ----------
    value : FloatArray
        One vector: shape (2,). A batch: shape (N, 2).
    """

    value: FloatArray

    @property
    @abstractmethod
    def x(self) -> T:
        """
        x-component(s) of the vector(s).

        Returns
        -------
        T
            Scalar dx for one vector, or an array of shape (N,) for a batch.
        """

    @property
    @abstractmethod
    def y(self) -> T:
        """
        y-component(s) of the vector(s).

        Returns
        -------
        T
            Scalar dy for one vector, or an array of shape (N,) for a batch.
        """

    @property
    @abstractmethod
    def norm(self) -> T:
        """
        Euclidean length (magnitude) of the vector or each row.

        Returns
        -------
        T
            Scalar norm for one vector, or an array of shape (N,) for a batch.
        """

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        arr_str = np.array2string(self.value, precision=2, separator=", ")
        return f"{cls_name}(value={arr_str})"
