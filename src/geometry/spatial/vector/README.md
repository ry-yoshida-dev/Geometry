# vector

## Overview

3D vectors with single and batched representations, pairwise relations, and orthonormal basis construction.

## Components

| Component | Description |
|-----------|-------------|
| [base.py](./base.py) | Abstract `Vector3D` over `NumericScalar` or `NumericArray`. |
| [vector.py](./vector.py) | Single vector — `value` shape `(3,)`; norm, unit vector, cross product, Gram–Schmidt. |
| [vectors.py](./vectors.py) | Batch — `value` shape `(N, 3)`; unit vectors and azimuthal/polar angle conversion. |
| [vector_pair.py](./vector_pair.py) | `Vector3DPair` for orthogonality/parallelism between two vectors. |
| [normalized_orthogonal_vectors.py](./normalized_orthogonal_vectors.py) | `OrthonormalBasis` holding three orthonormal vectors. |
