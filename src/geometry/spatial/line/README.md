# line

## Overview

3D line segments defined by start and end points. Single and batched segments share an abstract base parameterized by length type.

## Components

| Component | Description |
|-----------|-------------|
| [base.py](./base.py) | Abstract `Line3D` over `float` or `NumericArray`. |
| [line.py](./line.py) | Single segment — `value` shape `(2, 3)`; start/end points, direction vector, length. |
| [lines.py](./lines.py) | Batch — `value` shape `(N, 2, 3)`; start/end points, lengths, centers, direction vectors. |
