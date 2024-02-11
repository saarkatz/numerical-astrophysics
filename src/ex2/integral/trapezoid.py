import numpy as np

from typing import Callable
from numpy.typing import NDArray


def series_trapezoid_measure(x: NDArray, y: NDArray, h=None):
    y_midpoints = (y[1:] + y[:-1]) / 2
    if h:
        delta_x = h
    else:
        delta_x = x[1:] - x[:-1]
    return y_midpoints * delta_x


def function_trapezoid_measure(func: Callable[[NDArray], NDArray], start, end, h):
    steps = int((end - start) / h) + 1  # we want to include the end too
    x = np.linspace(start, end, steps)
    return series_trapezoid_measure(x, func(x))


def adaptive_trapezoid_generator(func: Callable[[NDArray], NDArray], start, end):
    steps = 1
    step_size = (end - start) / 2
    points = np.array([start, end])
    integ_res = np.sum(series_trapezoid_measure(points, func(points)))

    iteration = 0
    yield integ_res, float("inf"), step_size, iteration
    while True:
        iteration += 1

        points = np.linspace(start + step_size, end - step_size, steps)
        new_integ_res = integ_res / 2 + step_size * np.sum(func(points))
        error = np.abs((new_integ_res - integ_res) / 3)
        yield new_integ_res, error, step_size, iteration

        steps *= 2
        step_size /= 2
        integ_res = new_integ_res


def adaptive_trapezoid_integral(func: Callable[[NDArray], NDArray], start, end, resolution, limit=100):
    func_dtype = func([start]).dtype
    start, end = func_dtype.type(start), func_dtype.type(end)

    integ_res = None
    error = None

    integ_gen = adaptive_trapezoid_generator(func, start, end)
    for integ_res, _ in zip(integ_gen, range(limit)):
        _, error, _, _ = integ_res

        if error < resolution:
            break

    return integ_res


def _test():
    import matplotlib.pyplot as plt

    x = np.linspace(0, np.pi / 2, 200)
    y = np.sin(x)
    z = 1-np.cos(x)
    step_size = x[1] - x[0]

    stm = series_trapezoid_measure(x, y)
    ftm = function_trapezoid_measure(np.sin, 0, np.pi / 2, step_size)
    sti = np.sum(stm)
    fti = np.sum(ftm)
    ati, err, _, _ = adaptive_trapezoid_integral(np.sin, 0, np.pi / 2, 1e-6)

    diff = np.abs(sti - fti)
    diff2 = np.abs(sti - ati)
    print(f"{sti=}, {fti=}, {ati=}±{err}")
    print(f"|st - ft| = {diff}")
    print(f"|st - at| = {diff2}")

    plt.plot(x, y, label="Sin(x)")
    plt.plot(x, z, label="1 - Cos(x)")
    plt.plot(x[:-1] + step_size, np.cumsum(stm), label="series trapezoid")
    plt.plot(x[:-1] + step_size, np.cumsum(ftm), label="function trapezoid")

    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # _test()
    def func(x):
        a = np.sin(np.sqrt(100 * x))
        return a * a
    print(adaptive_trapezoid_integral(func, 0, 1, 1e-6))
