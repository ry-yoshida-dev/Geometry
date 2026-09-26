# vector

## Overview

2D vectors in image space (x right, y down). Single and batched vectors share an abstract base parameterized by scalar or array component type.

## Components

| Component | Description |
|-----------|-------------|
| [base.py](./base.py) | Abstract `Vector2D` over `NumericScalar` or `NumericArray`. |
| [vector.py](./vector.py) | Single vector — `value` shape `(2,)`; norm, unit vector, orthogonality/parallelism, conversion to `Line2D`. |
| [vectors.py](./vectors.py) | Batch — `value` shape `(N, 2)`; per-row components and norms. |
| [vector_pair.py](./vector_pair.py) | `Vector2DPair` for angle, dot/cross product, and orthogonality/parallelism between two vectors. |

## Examples

```python
import numpy as np
from geometry.planar.vector import Vector2D, Vector2DPair

first = Vector2D(np.array([1.0, 0.0]))
second = Vector2D(np.array([0.0, 1.0]))
Vector2DPair(first, second).is_orthogonal
```
