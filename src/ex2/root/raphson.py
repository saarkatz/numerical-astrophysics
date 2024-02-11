from functools import partial

from ex2.derivative.first import function_center_derivative


def root_raphson(func, initial, resolution, func_derivative=None, limit=100):
    """find a root of a func using Newton-Raphson method

    The derivative of the function at the initial point must not be 0.

    :param func: The function to explore
    :param initial: Initial guess to start from
    :param resolution: The resolution to which we wish to find the root
    :param func_derivative: The derivative of the function. If this is None, numerical derivative will be used.
    :param limit: Limit on the number of iterations
    :return: The x value of the root along with the number of iterations to reach it
    """
    if func_derivative is None:
        func_derivative = partial(function_center_derivative, func)

    i = 0
    point = initial
    for i in range(limit):
        f = func(point)
        df = func_derivative(point)
        if df == 0:
            # There is no way to continue - exit
            break
        new = point - f / df

        x_resolution = resolution if point == 0 else resolution * point
        if abs(f) < resolution or abs(new - point) < x_resolution:
            break

        point = new

    return point, i
