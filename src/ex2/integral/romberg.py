import numpy as np

from typing import Callable
from numpy.typing import NDArray

from ex2.integral.trapezoid import adaptive_trapezoid_generator


def romberg_generator(func: Callable[[NDArray], NDArray], start, end):
    """generate the romberg integral values in successive order
    The values first increase in the index m from 0 to n and than in index n to n+1 and back to m, starting from (0,0)

    :param func: The integrand
    :param start: The start of the integration interval
    :param end: The end of the integration interval
    :return: The value of the integral and the estimated error of the calculation
    """
    trapezoid_gen = adaptive_trapezoid_generator(func, start, end)

    # We need to keep two rows of the integration - the current one and the previous one.
    coeff = []
    prev = []
    current = []
    index = 1  # this is n
    for trapezoid in trapezoid_gen:  # loop for n
        yield trapezoid
        trapezoid, err, step_size, iteration = trapezoid
        current.append(trapezoid)
        for i in range(1, iteration):  # loop for m
            current.append(current[i - 1] + (current[i - 1] - prev[i - 1]) / coeff[i - 1])

            # We don't know how to calculate the error of the R_m,m value so we will assign to it the previous error value
            if i >= len(prev):
                i = i - 1

            error = np.abs((current[i] - prev[i]) / coeff[i])
            yield current[-1], error, step_size, iteration

        # Moving to the next row
        prev, current = current, []
        coeff.append(4**index - 1)
        index += 1


def romberg_integral(func: Callable[[NDArray], NDArray], start, end, resolution, limit=100):
    """Calculate the integral using romberg method to the specified resolution

    :param func: The integrand
    :param start: The start of the integration interval
    :param end: The end of the integration interval
    :param resolution: The requested resolution of the result
    :param limit: A limit on the number of iterations - the value of n
    :return: The value of the integral, the estimated error of the calculation, the final step size and the iteration count
    """
    func_dtype = func([start]).dtype
    start, end = func_dtype.type(start), func_dtype.type(end)

    integ_res = None

    integ_gen = romberg_generator(func, start, end)
    for integ_res in integ_gen:
        _, error, _, iterations = integ_res

        if error < resolution or iterations >= limit:
            break

    return integ_res


def _test():
    romberg = romberg_integral(np.sin, 0, np.pi / 2, 1e-6)
    print(romberg)


if __name__ == '__main__':
    _test()
    def func(x):
        a = np.sin(np.sqrt(100 * x))
        return a * a
    print(romberg_integral(func, 0, 1, 1e-30))

