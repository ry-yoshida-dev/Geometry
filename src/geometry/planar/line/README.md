# line

2D line segments (start and end points) in image pixel coordinates (x right, y down). Single-segment and batched APIs share an abstract base; storage is always two (x, y) rows per segment—no alternate layouts like the box module.

## Components

| Component | Description |
|-----------|-------------|
| [base.py](./base.py) | Abstract `Line2D` with length typed as `float` or `np.ndarray` per batch. |
| [line.py](./line.py) | One segment — `value` shape `(2, 2)`. |
| [lines.py](./lines.py) | Batch — `value` shape `(N, 2, 2)`. |

Package exports (`__init__.py`): `Line2D`, `Lines2D`.
