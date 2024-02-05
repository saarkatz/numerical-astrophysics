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
        coeffs = [comb(n - 1, i) * np.sum([(-1)**(i - j) * comb(i, j) * y[j] for j in range(i + 1)]) for i in range(n)]
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


class CubicSplineInterpolator:
    def __init__(self, points):
        self.x = points[0]
        self.splines = []

        # To calculate the coefficients for the splines we will solve the matrix of the conditions
        # We will use the not-a-knot condition
        # Reference: https://math.libretexts.org/Bookshelves/Applied_Mathematics/Numerical_Methods_(Chasnov)/05%3A_Interpolation/5.03%3A_Cubic_Spline_Interpolation
        n = self.x.shape[0]
        x, y = self.x, points[1]
        matrix = np.zeros(shape=(n, n+1))
        b_vector = np.zeros(shape=(n, 1))

        f = y[1:] - y[:-1]
        h = x[1:] - x[:-1]

        b_vector[1:-1] = f[1:] / h[1:] - f[:-1] / h[:-1]

        for i in range(1, n - 1):
            matrix[h[0] / 3 ]




    def __call__(self, x: NDArray) -> NDArray:
        pass

    def derivative(self, x: NDArray) -> NDArray:
        pass

    def second_derivative(self, x: NDArray) -> NDArray:
        pass