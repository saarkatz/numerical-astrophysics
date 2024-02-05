from numpy.typing import NDArray
import numpy as np
from scipy.special import comb


class RawSpline:
    def __init__(self, points: NDArray):
        self.points = points

    def assert_input(self, x):
        assert np.all(x >= self.points[0][0]) and np.all(x <= self.points[0][-1])

    def __call__(self, x: NDArray) -> NDArray:
        splines = self.points[1][np.newaxis, :] * np.ones(shape=(x.shape[0], self.points.shape[1]))
        t = (x - self.points[0][0]) / (self.points[0][-1] - self.points[0][0])
        t = t[:, np.newaxis]
        while splines.shape[1] > 1:
            y_0 = splines[:, :-1]
            y_1 = splines[:, 1:]
            splines = (y_1 - y_0) * t + y_0
        return splines.reshape(x.shape)

    def derivative(self, x: NDArray) -> NDArray:
        interval_length = self.points[0][-1] - self.points[0][0]
        splines = self.points[1][np.newaxis, :] * np.ones(shape=(x.shape[0], self.points.shape[1]))
        divs = np.zeros(shape=splines.shape)

        t = (x - self.points[0][0]) / interval_length
        t = t[:, np.newaxis]

        while divs.shape[1] > 1:
            y_0 = splines[:, :-1]
            y_1 = splines[:, 1:]
            dy_0 = divs[:, :-1]
            dy_1 = divs[:, 1:]
            divs = (y_1 - y_0) + (dy_1 - dy_0) * t + dy_0
            splines = (y_1 - y_0) * t + y_0
        return divs.reshape(x.shape) / interval_length

    def second_derivative(self, x: NDArray) -> NDArray:
        interval_length = self.points[0][-1] - self.points[0][0]
        splines = self.points[1][np.newaxis, :] * np.ones(shape=(x.shape[0], self.points.shape[1]))
        divs = np.zeros(shape=splines.shape)
        second_divs = np.zeros(shape=splines.shape)

        t = (x - self.points[0][0]) / interval_length
        t = t[:, np.newaxis]

        while second_divs.shape[1] > 1:
            y_0 = splines[:, :-1]
            y_1 = splines[:, 1:]
            dy_0 = divs[:, :-1]
            dy_1 = divs[:, 1:]
            ddy_0 = second_divs[:, :-1]
            ddy_1 = second_divs[:, 1:]
            second_divs = 2 * (dy_1 - dy_0) + (ddy_1 - ddy_0) * t + ddy_0
            divs = (y_1 - y_0) + (dy_1 - dy_0) * t + dy_0
            splines = (y_1 - y_0) * t + y_0
        return second_divs.reshape(x.shape) / (interval_length * interval_length)


class PolySpline:
    def __init__(self, coeffs, interval, start):
        self.coeffs = coeffs
        self.interval = interval
        self.start = start

    def assert_input(self, x):
        assert np.all(x >= self.start) and np.all(x <= self.start + self.interval)

    @classmethod
    def from_points(cls, points: NDArray):
        n = points.shape[1]
        y = points[1]
        coeffs = [comb(n - 1, i) * np.sum([(-1) ** (i - j) * comb(i, j) * y[j] for j in range(i + 1)]) for i in
                  range(n)]
        return cls(coeffs, points[0][-1] - points[0][0], points[0][0])

    def __call__(self, x: NDArray) -> NDArray:
        if len(self.coeffs) < 1:
            return 0
        t = (x - self.start) / self.interval
        factor = 1
        result = self.coeffs[0]
        for coeff in self.coeffs[1:]:
            factor *= t
            result += coeff * factor
        return result

    def derivative(self, x: NDArray) -> NDArray:
        if len(self.coeffs) < 2:
            return 0
        t = (x - self.start) / self.interval
        factor = 1
        result = self.coeffs[1]
        for i, coeff in enumerate(self.coeffs[2:]):
            factor *= t
            result += coeff * factor * (i + 2)
        return result / self.interval

    def second_derivative(self, x: NDArray) -> NDArray:
        if len(self.coeffs) < 3:
            return 0
        t = (x - self.start) / self.interval
        factor = 1
        result = self.coeffs[2] * 2
        for i, coeff in enumerate(self.coeffs[3:]):
            factor *= t
            result += coeff * factor * (i + 2) * (i + 3)
        return result / (self.interval * self.interval)


def make_spline_set(points: NDArray) -> list[PolySpline]:
    # Reference: https://math.libretexts.org/Bookshelves/Applied_Mathematics/Numerical_Methods_(Chasnov)/05%3A_Interpolation/5.03%3A_Cubic_Spline_Interpolation
    # To calculate the coefficients for the splines we will solve the matrix of the conditions
    # Here I chose to use the not-a-knot boundary condition

    x, y = points[0], points[1]
    n = x.shape[0]
    matrix = np.zeros(shape=(n, n))
    b_vector = np.zeros(shape=n)

    f = y[1:] - y[:-1]
    h = x[1:] - x[:-1]

    b_vector[1:-1] = f[1:] / h[1:] - f[:-1] / h[:-1]

    for i in range(1, n - 1):
        matrix[i, i - 1] = h[i - 1] / 3
        matrix[i, i] = 2 * (h[i - 1] + h[i]) / 3
        matrix[i, i + 1] = h[i] / 3

    # not-a-knot condition
    matrix[0, 0] = h[1]
    matrix[0, 1] = -(h[0] + h[1])
    matrix[0, 2] = h[0]
    matrix[-1, -3] = h[-1]
    matrix[-1, -2] = -(h[-1] + h[-2])
    matrix[-1, -1] = h[-2]

    # Solve for MATRIX*x = B
    b_coeffs = np.linalg.solve(matrix, b_vector)

    # My coefficients are defined a bit differently than the reference
    ah = h * h
    a_coeffs = ah * (b_coeffs[1:] - b_coeffs[:-1]) / 3
    c_coeffs = f - ah * (b_coeffs[1:] + 2 * b_coeffs[:-1]) / 3
    b_coeffs = b_coeffs[:-1] * ah

    return [PolySpline([y[i], c_coeffs[i], b_coeffs[i], a_coeffs[i]], h[i], x[i]) for i in range(len(a_coeffs))]


def make_spline_set_v2(points: NDArray) -> list[PolySpline]:
    # Reference: https://en.wikipedia.org/wiki/Spline_(mathematics)
    # This method uses the natural cubic boundary condition (second derivative at the boundary is 0)
    x, y = points[0], points[1]
    n = x.shape[0]
    h = x[1:] - x[:-1]
    f = y[1:] - y[:-1]
    alpha = 3 * (f[1:] / h[1:] - f[:-1] / h[:-1])

    a_coeffs = y
    c_coeffs = np.zeros(n)

    l_arr = np.zeros(n)
    mu_arr = np.zeros(n)
    z_arr = np.zeros(n)
    l_arr[0] = 1
    l_arr[n - 1] = 1
    for i in range(1, n - 1):
        l_arr[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu_arr[i - 1]
        mu_arr[i] = h[i] / l_arr[i]
        z_arr[i] = (alpha[i - 1] - h[i - 1] * z_arr[i - 1]) / l_arr[i]
    for j in range(n - 2, -1, -1):
        c_coeffs[j] = z_arr[j] - mu_arr[j] * c_coeffs[j + 1]

    b_coeffs = (a_coeffs[1:] - a_coeffs[:-1]) / h - h * (c_coeffs[1:] + 2 * c_coeffs[:-1]) / 3
    d_coeffs = (c_coeffs[1:] - c_coeffs[:-1]) / h / 3

    b_coeffs = b_coeffs * h
    c_coeffs = c_coeffs[:-1] * h * h
    d_coeffs = d_coeffs * h * h * h

    return [PolySpline([a_coeffs[i], b_coeffs[i], c_coeffs[i], d_coeffs[i]], h[i], x[i]) for i in range(len(d_coeffs))]


class CubicSplineInterpolator:
    def __init__(self, points, spline_func=make_spline_set):
        self.x = points[0]
        self.bounds = (self.x[0], self.x[-1])
        self.splines = spline_func(points)

    def assert_input(self, x):
        assert np.all(x >= self.bounds[0]) and np.all(x <= self.bounds[1])

    def _gen_x_splits(self, x: NDArray) -> NDArray:
        indexes = np.searchsorted(x, self.x[1:], side="left")
        last_index = 0
        for i in indexes:
            yield x[last_index:i]
            last_index = i

        # Get all the rest of the points
        yield x[last_index:]

    def __call__(self, x: NDArray) -> NDArray:
        # Find the spline that should be used for the interpolation
        y_parts = [np.array([])]
        for split, spline in zip(self._gen_x_splits(x), self.splines + [self.splines[-1]]):
            if len(split) == 0:
                continue
            y_parts.append(spline(split))
        return np.concatenate(y_parts)

    def derivative(self, x: NDArray) -> NDArray:
        y_parts = [np.array([])]
        for split, spline in zip(self._gen_x_splits(x), self.splines + [self.splines[-1]]):
            if len(split) == 0:
                continue
            y_parts.append(spline.derivative(split))
        return np.concatenate(y_parts)

    def second_derivative(self, x: NDArray) -> NDArray:
        y_parts = [np.array([])]
        for split, spline in zip(self._gen_x_splits(x), self.splines + [self.splines[-1]]):
            if len(split) == 0:
                continue
            y_parts.append(spline.second_derivative(split))
        return np.concatenate(y_parts)
