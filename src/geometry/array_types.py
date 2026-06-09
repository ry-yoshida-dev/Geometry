"""
Shared NumPy array type aliases for the geometry package.

Dtype is expressed through ``numpy.typing.NDArray``. Shape constraints are
enforced at runtime in dataclass validators and described in class docstrings.
"""

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

NumericScalar: TypeAlias = int | float
NumericArray: TypeAlias = NDArray[np.integer[Any] | np.floating[Any]]
FloatArray: TypeAlias = NDArray[np.floating[Any]]
BoolArray: TypeAlias = NDArray[np.bool_]
