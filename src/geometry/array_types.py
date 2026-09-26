"""
Shared NumPy array type aliases for the geometry package.

Dtype is expressed through ``numpy.typing.NDArray``. Shape constraints are
enforced at runtime in dataclass validators and described in class docstrings.
"""

import numpy as np
from numpy.typing import NDArray

type NumericScalar = int | float
type NumericArray = NDArray[np.integer | np.floating]
type FloatArray = NDArray[np.floating]
type BoolArray = NDArray[np.bool_]
