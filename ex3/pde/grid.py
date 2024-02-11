from enum import Enum, auto
from typing import Iterable

import numpy as np
from numpy.typing import NDArray


class Boundary(Enum):
    Periodic = auto()
    Outflow = auto()


def to_array(a_slice, n) -> NDArray:
    if isinstance(a_slice, slice):
        stop = a_slice.stop or n
        if stop < 0:
            stop += n
        return np.arange(a_slice.start or 0, stop, a_slice.step, dtype=int)
    else:
        return np.array([a_slice], dtype=int)


def one_shot_dim_vector(size, dim, total):
    v = np.ones(total, dtype=int)
    v[dim] = size
    return v


class Grid:
    def __init__(self, dimensions, boundary: Boundary):
        if isinstance(dimensions, np.ndarray):
            self.grid = dimensions
        else:
            self.grid = np.zeros(dimensions)
        self.boundary = boundary
        self.num_dimensions = len(self.grid.shape)

    def apply_boundary(self, item):
        if not isinstance(item, Iterable):
            item = [item]
        d = self.num_dimensions
        item = [to_array(v, s) for v, s in zip(item, self.grid.shape)]
        item = [v.reshape(one_shot_dim_vector(len(v), i, d)) for i, v in enumerate(item)]
        match self.boundary:
            case Boundary.Periodic:
                return tuple(v % size for v, size in zip(item, self.grid.shape))
            case Boundary.Outflow:
                return tuple(np.clip(v, 0, size - 1) for v, size in zip(item, self.grid.shape))

    def __getitem__(self, item):
        return self.grid[self.apply_boundary(item)]

    def __setitem__(self, key, value):
        self.grid[self.apply_boundary(key)] = value

    def __repr__(self):
        return repr(self.grid)
