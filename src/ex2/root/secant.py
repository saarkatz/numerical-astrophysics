def root_secant(func, initial1, initial2, resolution, limit=100):
    """find a root of a func using the secant method

    The function must not assume the same value at the two initial guesses (that is func(initial1) != func(initial2))

    :param func: The function to explore
    :param initial1: Guess for the derivative calculation
    :param initial2: Initial guess to start from
    :param resolution: The resolution to which we wish to find the root
    :param limit: Limit on the number of iterations
    :return: The x value of the root along with the number of iterations to reach it
    """
    i = 0
    point1 = initial1
    point2 = initial2
    for i in range(limit):
        f1 = func(point1)
        f2 = func(point2)
        if f1 - f2 == 0:
            # There is no way to continue - exit
            break
        new = point2 - f2 * (point1 - point2) / (f1 - f2)

        x_resolution = resolution if point2 == 0 else resolution * point2
        if abs(f2) < resolution or abs(new - point2) < x_resolution:
            break

        point1, point2 = point2, new

    return point2, i
