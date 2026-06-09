"""
Batch of 2D vectors (N rows, two columns) extending base.Vector2D.

Validates shape (N, 2) and provides per-row components and Euclidean norms.
"""
from __future__ import annotations

from ...array_types import FloatArray, NumericArray
from dataclasses import dataclass
from typing import Iterator, Union

import numpy as np

from .base import Vector2D as Vector2DBase
from .vector import Vector2D


@dataclass
class Vectors2D(Vector2DBase[NumericArray]):
    """
    N displacement vectors with value of shape (N, 2).

    Each row is (dx, dy) in the same frame as :class:`Vector2D`.

    Attributes
    ----------
    value : NumericArray
        Shape (N, 2).
    """

    def __post_init__(self) -> None:
        """
        Coerce to ndarray and validate rank and trailing dimensions.

        Raises
        ------
        ValueError
            If the array is not two-dimensional with two columns.
        """
        v = np.asarray(self.value)
        if v.ndim != 2 or v.shape[1] != 2:
            raise ValueError(
                f"vectors must have shape (N, 2), but got {v.shape}"
            )
        self.value = v

    @property
    def x(self) -> NumericArray:
        """
        All x-components (first column).

        Returns
        -------
        NumericArray
            Shape (N,).
        """
        return self.value[:, 0]

    @property
    def y(self) -> NumericArray:
        """
        All y-components (second column).

        Returns
        -------
        NumericArray
            Shape (N,).
        """
        return self.value[:, 1]

    @property
    def norm(self) -> FloatArray:
        """
        Euclidean length of each row.

        Returns
        -------
        FloatArray
            Shape (N,).
        """
        return np.linalg.norm(self.value, axis=1)

    def __repr__(self) -> str:
        return f"Vectors2D(n={len(self)}, shape={self.value.shape})"

    def __len__(self) -> int:
        return int(self.value.shape[0])

    def __getitem__(
        self,
        index: Union[int, slice],
    ) -> Union[Vector2D, "Vectors2D"]:
        """
        Index or slice rows.

        Parameters
        ----------
        index : int | slice
            Row index for one :class:`Vector2D`, or slice for a sub-batch.

        Returns
        -------
        Vector2D | Vectors2D
            One vector or a new batch.
        """
        result = self.value[index]
        if isinstance(index, slice):
            return Vectors2D(value=result)
        return Vector2D(value=result)

    def __iter__(self) -> Iterator[Vector2D]:
        for row in self.value:
            yield Vector2D(value=row)
