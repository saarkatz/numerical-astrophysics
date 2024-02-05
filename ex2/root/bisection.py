def root_bisection(func, start, end, resolution, limit=100):
    """find a root of a func using the bisection method

    The start and end should satisfy func(start)*func(end) < 0

    :param func: The function to explore
    :param start: Beginning of the interval on which to search
    :param end: End of the interval on which to search
    :param resolution: The resolution to which we wish to find the root
    :param limit: Limit on the number of iterations
    :return: The x value of the root along with the number of iterations to reach it
    """
    assert func(start) * func(end) <= 0

    if func(start) > func(end):
        low = end
        high = start
    else:
        low = start
        high = end

    mid, i = None, 0
    for i in range(limit):
        mid = (low + high) / 2
        value = func(mid)
        if abs(value) < resolution:
            break
        if value > 0:
            high = mid
        else:
            low = mid

    return mid, i
