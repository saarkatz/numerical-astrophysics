from numpy.typing import NDArray

from ex2.alignment import Alignment


def series_midpoint_measure(x: NDArray, y: NDArray, alignment: Alignment, h=None):
    if alignment == Alignment.Forward:
        ys = y[1:]
    else:
        ys = y[:-1]

    if h:
        delta_x = h
    else:
        delta_x = x[1:] - x[:-1]
    return ys * delta_x
