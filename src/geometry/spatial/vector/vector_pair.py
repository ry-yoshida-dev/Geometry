from .vector import Vector3D
from dataclasses import dataclass

@dataclass
class Vector3DPair:
    """
    Vector3D Pair.

    Attributes
    ----------
    vector1: Vector3D
        The first vector.
    vector2: Vector3D
        The second vector.

    Raises
    ------
    ValueError:
        If the two vectors are parallel.
    """
    vector1: Vector3D
    vector2: Vector3D

    def __post_init__(self):
        if self.vector1.is_parallel(self.vector2):
            raise ValueError("The two vectors are parallel. Input non-parallel vectors.")

    @property
    def is_parallel(self) -> bool:
        """
        Check if the two vectors are parallel.

        Returns
        -------
        bool: True if the two vectors are parallel, False otherwise.
        """
        return self.vector1.is_parallel(self.vector2)

    @property
    def is_orthogonal(self) -> bool:
        """
        Check if the two vectors are orthogonal.

        Returns
        -------
        bool: True if the two vectors are orthogonal, False otherwise.
        """
        return self.vector1.is_orthogonal(self.vector2)