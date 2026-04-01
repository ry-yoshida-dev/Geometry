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
pip install -e .
```

Dependencies (`shapely`, `units` from the linked repository) are installed automatically.  
To install only the dependencies without the package, use:

```bash
pip install -r requirements.txt
```

## Example

After installing the package, import subpackages from any directory:

```python
from geometry.planar import Point2D, Vector2D, Box2D, Geometry2DMeasure
from geometry.spatial import Point3D, Vector3D, CartesianCoordinateSystem

p = Point2D(1.0, 2.0)
v = Vector2D(0.0, 1.0)
```

See the README files under `src/geometry/planar/` and `src/geometry/spatial/` for component-specific usage.
