# Geometry

## Overview

Geometry is a Python package for 2D and 3D geometry: points, lines, vectors, bounding boxes, polar coordinates, Cartesian axes, normals, and related helpers.  
For module-level detail, see [src/geometry/planar/README.md](src/geometry/planar/README.md) and [src/geometry/spatial/README.md](src/geometry/spatial/README.md).

## Installation

From the package root (the directory containing `pyproject.toml`):

```bash
pip install .
```

For development, install in editable mode so changes to the source take effect immediately:

```bash
pip install -e ".[dev]"
```

The `dev` extra installs `mypy`, `pytest`, `scipy-stubs`, and `types-shapely`.

Dependencies (`shapely`, `units` from the linked repository) are installed automatically.  
To install only the dependencies without the package, use:

```bash
pip install -r requirements.txt
```

## Example

After installing the package, import subpackages from any directory:

```python
import numpy as np
from geometry.planar import Point2D, Vector2D
from geometry.spatial import Point3D

point = Point2D(np.array([1.0, 2.0]))
vector = Vector2D(np.array([0.0, 1.0]))
point_3d = Point3D(np.array([1.0, 2.0, 3.0]))
```

See the README files under `src/geometry/planar/` and `src/geometry/spatial/` for component-specific usage.
