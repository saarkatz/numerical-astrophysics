import numpy as np

from ex2.root.bisection import root_bisection
from ex2.root.raphson import root_raphson
from ex2.root.secant import root_secant


def raphson_shim(func, initial, dummy, resolution, limit):
    return root_raphson(func, initial, resolution, limit=limit)


def find_roots(func, method: str, start, end, frequency, resolution, limit=1000):
    """Find the roots of a function in a given interval

    :param func: The function to explore
    :param method: The method to use for exploring the function (from 'bisect', 'raphson' or 'secant')
    :param start: Beginning of the interval on which to search
    :param end: End of the interval on which to search
    :param frequency: An estimate to the frequency of roots of the function on the interval. If the function has roots
     appearing more frequently than the frequency some of them might be missed by the function.
    :param resolution: The resolution to which we wish to find the root
    :param limit: Limit on the number of iterations
    :return: A tuple with an array of the x value and an array of the iteration counts
    """
    xs = []
    counts = []

    method_func = None
    match method:
        case "bisect":
            method_func = root_bisection
        case "raphson":
            method_func = raphson_shim
        case "secant":
            method_func = root_secant
    if method_func is None:
        raise ValueError(f"{method} is not a valid value for method")

    # Split the interval based on the frequency
    x = np.linspace(start, end, int(frequency * (end - start)))
    for a, b in zip(x, x[1:]):
        if func(a) * func(b) > 0:
            # There is no root (up to frequency) in this interval
            continue
        r, c = method_func(func, a, b, resolution, limit=limit)
        xs.append(r)
        counts.append(c)

    return xs, counts


def _test():
    import matplotlib.pyplot as plt

    def foo(x):
        return np.sin(x - x**2) / x

    roots, counts = find_roots(foo, "secant", 0.5, 10, 7, 1e-6)
    for root, count in zip(roots, counts):
        print(f"{root=}, {count=}")

    x = np.linspace(0.5, 10, 300)
    y = foo(x)

    plt.plot(x, y, label="foo")
    plt.plot(np.array(roots), np.zeros(len(roots)), "ro", label="roots")
    plt.tight_layout()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    _test()
