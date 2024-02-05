"""In this module the implementation of the questions of the assignment
"""
import numpy as np
import matplotlib.pyplot as plt

from ex2.integral.trapezoid import adaptive_trapezoid_integral
from ex2.integral.romberg import romberg_integral
from ex2.derivative.first import series_central_derivative
from ex2.interpolate.spline import CubicSplineInterpolator
from ex2.noise import apply_uniform_noise
from ex2.root.multi import find_roots


def q1():
    def integrand(x):
        return np.sin(np.sqrt(100 * x)) ** 2

    real_integral_value = (201 - np.cos(20) - 20 * np.sin(20)) / 400
    trapezoid = adaptive_trapezoid_integral(integrand, 0, 1, 1e-6)
    romberg = romberg_integral(integrand, 0, 1, 1e-6)

    print(f"""\
############################ Q1 ############################
    
The results of the integraion methods are as follows:
The real value: {real_integral_value}
The trapezoid method: {trapezoid[0]} ± {trapezoid[1]} at step size: {trapezoid[2]} after {trapezoid[3]} iterations
The romberg method: {romberg[0]} ± {romberg[1]} at step size: {romberg[2]} after {romberg[3]} iterations""")

    x = np.linspace(0, 1, 200)
    plt.plot(x, integrand(x))
    plt.title(r"$\sin(\sqrt{100x})$")
    plt.tight_layout()
    plt.show()


def q2():
    # We will keep the noise separate to analyse it a bit more
    def foo(x):
        return np.sin(x - x ** 2) / x

    def real_derivative(x):
        x_squere = x ** 2
        sine = np.sin(x - x_squere)
        cosine = np.cos(x - x_squere)
        return - sine / x_squere + cosine / x - 2 * cosine

    x = np.linspace(0.5, 10, 100)
    y = foo(x)
    dy = real_derivative(x)
    noise = apply_uniform_noise(np.zeros(x.shape), 1)
    cubic_spline = CubicSplineInterpolator(np.vstack([x, y + 0.1 * noise]))
    der_noisy = series_central_derivative(x, y + 0.1 * noise)

    lambda_series = np.linspace(0, 0.3, 100)
    noise_in_func = np.std(noise[np.newaxis, :] * lambda_series[:, np.newaxis], axis=1)

    derivatives = []
    spline_derivatives = []
    for value in lambda_series:
        derivatives.append(series_central_derivative(x, y + value * noise))
        cs = CubicSplineInterpolator(np.vstack([x, y + value * noise]))
        spline_derivatives.append(cs.derivative(x))
    derivatives = np.vstack(derivatives)
    spline_derivatives = np.vstack(spline_derivatives)
    noise_in_derivative = np.std(dy[np.newaxis, 1:-1] - derivatives, axis=1)
    spline_noise_derivative = np.std(dy[np.newaxis] - spline_derivatives, axis=1)

    print("""\
############################ Q2 ############################

From the graphs we can see that the noise in the derivatives is match greater and grows much faster than that of the
function itself. We also see that the noise in the splines grow much faster than that of the raw derivative.""")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(x, y, label="clean")
    axes[0, 0].plot(x, y + 0.1 * noise, label="noisy")
    axes[0, 0].plot(x, cubic_spline(x), label="spline")
    axes[0, 0].set_title(r"$\frac{\sin(x - x^2)}{x}$ with $\lambda=0.1$")
    axes[0, 0].legend()

    axes[0, 1].plot(x[1:-1], dy[1:-1] - der_noisy, label="numeric")
    axes[0, 1].set_title(r"$f'(x) - f'_{noisy}(x)$ with $\lambda=0.1$")

    axes[1, 0].plot(x, dy - cubic_spline.derivative(x), label="spline")
    axes[1, 0].set_title(r"$f'(x) - f'_{spline}(x)$ with $\lambda=0.1$")

    axes[1, 1].plot(lambda_series, noise_in_func, label="noise in function")
    axes[1, 1].plot(lambda_series, noise_in_derivative, label="noise in derivative")
    axes[1, 1].plot(lambda_series, spline_noise_derivative, label="spline derivative noise")
    axes[1, 1].plot(lambda_series, lambda_series, "--", label="$y=x$")
    axes[1, 1].set_title(r"$\sigma$ of the noise in the function and derivative vs $\lambda$")
    axes[1, 1].set_xlabel(r"$\lambda$")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


def q3():
    def foo(x):
        return np.sin(x - x ** 2) / x

    x = np.linspace(0.5, 10, 400)
    y = foo(x)

    b_roots, b_counts = find_roots(foo, "bisect", 0.5, 10, 7, 1e-6)
    r_roots, r_counts = find_roots(foo, "raphson", 0.5, 10, 7, 1e-6)
    s_roots, s_counts = find_roots(foo, "secant", 0.5, 10, 7, 1e-6)

    b_convergence = np.mean(b_counts)
    r_convergence = np.mean(r_counts)
    s_convergence = np.mean(s_counts)

    print(f"""\
############################ Q3 ############################
    
The convergence of the three methods on the interval is as follows:
Bisection - mean: {b_convergence}
Newton-Raphson - mean: {r_convergence}
Secant - mean: {s_convergence}

We can see that Bisection takes much more iterations to converge. On the other hand, some of the roots found by 
the other methods were outside the initial interval and in one case rather faraway (as illustrated in the figure)""")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].plot(x, y, label="foo")
    axes[0].set_title(r"$\frac{\sin(x - x^2)}{x}$ - Bisection")
    axes[0].plot(np.array(b_roots), np.zeros(len(b_roots)), "ro", label="roots")
    axes[0].grid()

    axes[1].plot(x, y, label="foo")
    axes[1].set_title(r"$\frac{\sin(x - x^2)}{x}$ - Secant")
    axes[1].plot(np.array(s_roots), np.zeros(len(r_counts)), "ro", label="roots")
    axes[1].grid()

    plt.tight_layout()
    plt.show()


def main():
    q1()
    print("\n" * 2)
    q2()
    print("\n" * 2)
    q3()


if __name__ == '__main__':
    main()
