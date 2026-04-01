from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from .vector import Vector3D

@dataclass
class OrthonormalBasis:
    """
    Orthonormal basis vectors.

    Attributes:
    ----------
    e1: Vector3D
        The first orthonormal basis vector.
    e2: Vector3D
        The second orthonormal basis vector.
    e3: Vector3D
        The third orthonormal basis vector.

    Raises:
    ------
    ValueError:
        If the vectors are not orthonormal.
        - If the vectors are not unit vectors.
        - If the vectors are not orthogonal.
    """
    e1: "Vector3D"
    e2: "Vector3D"
    e3: "Vector3D"

    def __post_init__(self):
        if not self.e1.is_unit:
            raise ValueError("e1 is not a unit vector")
        if not self.e2.is_unit:
            raise ValueError("e2 is not a unit vector")
        if not self.e3.is_unit:
            raise ValueError("e3 is not a unit vector")
        if not self.e1.is_orthogonal(self.e2):
            raise ValueError("e1 and e2 are not orthogonal")
        if not self.e1.is_orthogonal(self.e3):
            raise ValueError("e1 and e3 are not orthogonal")
        if not self.e2.is_orthogonal(self.e3):
            raise ValueError("e2 and e3 are not orthogonal")

    def __getitem__(self, index: int) -> "Vector3D":
        if index not in (-3, -2, -1, 0, 1, 2):
            raise IndexError(f"index {index} out of range")
        return  (self.e1, self.e2, self.e3)[index]

    def __len__(self) -> int:
        return 3

    def __iter__(self) -> Iterator["Vector3D"]:
        return iter((self.e1, self.e2, self.e3))

