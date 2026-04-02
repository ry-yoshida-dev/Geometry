from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

from ..point import Points3D
from ..vector import Vectors3D
from .base import Line3D as Line3DBase
from .line import Line3D


@dataclass
class Lines3D(Line3DBase[np.ndarray]):
    value: np.ndarray

    def __post_init__(self) -> None:
        v = np.asarray(self.value)
        if v.ndim != 3 or v.shape[1:] != (2, 3):
            raise ValueError(f"lines must have shape (N, 2, 3), but got {v.shape}")
        self.value = v

    @property
    def start(self) -> Points3D:
        return Points3D(value=self.value[:, 0, :])

    @property
    def end(self) -> Points3D:
        return Points3D(value=self.value[:, 1, :])

    @property
    def length(self) -> np.ndarray:
        d = self.value[:, 1, :] - self.value[:, 0, :]
        return np.linalg.norm(d, axis=1)

    @property
    def center(self) -> Points3D:
        return Points3D(value=np.mean(self.value, axis=1))

    @property
    def vectors(self) -> Vectors3D:
        return Vectors3D(value=self.value[:, 1, :] - self.value[:, 0, :])

    def __len__(self) -> int:
        return self.value.shape[0]

    def __getitem__(self, index: Union[int, slice]) -> Union[Line3D, "Lines3D"]:
        result = self.value[index]
        if isinstance(index, slice):
            return Lines3D(value=result)
        return Line3D(value=result)
