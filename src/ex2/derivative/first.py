from typing import Callable, Iterable

import numpy as np
from numpy.typing import NDArray

from ex2.alignment import Alignment


def series_derivative(x: NDArray, y: NDArray) -> NDArray:
    """calculate the derivative of a series of points

    Preferably the points should be with equal step size.
    The alignment of the return value to the x values will depend on the alignment you wish to use.
    With forward alignment the first value corresponds to the first x and the last to the second to last x.
    With backward alignment the first value corresponds to the second x and the last to last x.
    Either way the calculation is the same with the only difference being the way the result will be used.

    :param x: The x values of the points, should be of equal step size.
    :param y: The y values of the points, should be of the same size as y.
    :return: An array of size len(x) - 1
    """
    y_diff = y[1:] - y[:-1]
    x_diff = x[1:] - x[:-1]
    return y_diff / x_diff


def series_central_derivative(x: NDArray, y: NDArray) -> NDArray:
    """calculate the central derivative of a series of points

    Preferably the points should be with equal step size.
    The result will not include values corresponding to the first and last points in the series.

    :param x: The x values of the points, should be of equal step size.
    :param y: The y values of the points, should be of the same size as y.
    :return: An array of size len(x) - 2
    """
    y_diff = y[2:] - y[:-2]
    x_diff = x[2:] - x[:-2]
    return y_diff / x_diff


def function_derivative(func: Callable[[NDArray], NDArray], points: NDArray, alignment: Alignment, h=None) -> NDArray:
    """calculate the derivative of a function at a set of points

    The derivative step size can be specified using h. If it is not specified then an optimal value is estimated from
    the precision of the size of the items returned by func which should be float values of some size.

    :param func: The function take the derivative from
    :param points: The set of points on which to the derivative
    :param h: The step size of the derivative
    :param alignment: The alignment of the derivative. Either forward of backward.
    :return: An array of size len(points) with the derivatives at these points
    """
    fs = func(points)

    if not h:  # Estimate optimal step size
        h = np.power(np.finfo(fs.dtype).resolution, 1 / 2, dtype=fs.dtype)

    direction = 1 if alignment == Alignment.Forward else -1
    fs_off = func(points + direction * h)

    if alignment == Alignment.Forward:
        fs, fs_off = fs_off, fs

    derivative = (fs - fs_off) / h
    return derivative


def function_center_derivative(func: Callable[[NDArray], NDArray], points: NDArray, h=None) -> NDArray:
    """calculate the center derivative of a function at a set of points

    The derivative step size can be specified using h. If it is not specified then an optimal value is estimated from
    the precision of the size of the items returned by func which should be float values of some size.

    :param func: The function take the derivative from
    :param points: The set of points on which to the derivative
    :param h: The step size of the derivative
    :return: An array of size len(points) with the derivatives at these points
    """
    if not h:  # Estimate optimal step size
        if isinstance(points, np.ndarray):
            point = points[0]
        else:
            point = points
        test_f = func(point)
        h = np.power(np.finfo(test_f.dtype).resolution, 1 / 3, dtype=test_f.dtype)

    fs = func(points + h)
    fs_off = func(points - h)

    derivative = (fs - fs_off) / (2*h)
    return derivative


def _test():
    import matplotlib.pyplot as plt

    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x)
    z = np.cos(x)

    sd = series_derivative(x, y)
    scd = series_central_derivative(x, y)
    fdf = function_derivative(np.sin, x, alignment=Alignment.Forward)
    fdb = function_derivative(np.sin, x, alignment=Alignment.Backward)
    fcd = function_center_derivative(np.sin, x)

    fcdo = function_center_derivative(np.sin, x, h=1e-12)
    fcdu = function_center_derivative(np.sin, x, h=1e-2)

    plt.plot(x, y, label="Sin(x)")
    plt.plot(x, z, label="Cos(x)")
    plt.plot(x[:-1], sd, label="series derivative")
    plt.plot(x[1:-1], scd, label="series central derivative")
    plt.plot(x, fdf, label="function derivative forward")
    plt.plot(x, fdb, label="function derivative backward")
    plt.plot(x, fcd, label="function central derivative")
    plt.plot(x, fcdo, label="function central derivative - smaller h")
    plt.plot(x, fcdu, label="function central derivative - larger h ")

    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    _test()
