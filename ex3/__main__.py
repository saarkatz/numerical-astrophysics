"""In this module the implementation of the questions of the assignment
"""
import numpy as np
import matplotlib.pyplot as plt

from ex3.ode.system import System
from ex3.ode.euler import EulerSolver
from ex3.ode.runge import RK4Solver
from ex3.pde.grid import Grid, Boundary
from ex3.pde.advection import step_upwind, step_leapfrog, step_lax_wendorff, step_finite_volume_fo, \
    step_finite_volume_so


def top_hat(size, width):
    hat = np.zeros(size)
    center = int(size // 2)
    half = int(width // 2)
    hat[center - half:center + half] = 1
    return hat


def epsilon_diff(s1, s2, h):
    diff = s2 - s1
    return np.sqrt(h * np.sum(diff ** 2))


def q1():
    # Taken from https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
    a = 1.00000011
    e = 0.01671022

    gm = 4 * np.pi ** 2
    odes = {
        "x": lambda vx, **k: vx,
        "y": lambda vy, **k: vy,
        "vx": lambda x, y, **k: -gm * x / np.power(x ** 2 + y ** 2, 3 / 2),
        "vy": lambda x, y, **k: -gm * y / np.power(x ** 2 + y ** 2, 3 / 2)
    }
    s = System(odes)
    initial_conditions = {
        "x": 0,
        "y": a * (1 - e),
        "vx": -np.sqrt(gm / a * (1 + e) / (1 - e)),
        "vy": 0,
    }
    initial_radius = initial_conditions["y"]

    # We will take the change in radius after a single orbit for a number of dt values
    dts = np.logspace(-3, -1, 50)
    euler_r = []
    runge_r = []

    es = EulerSolver(s)
    rk = RK4Solver(s)
    for dt in dts:
        e_sol = es.solve(initial_conditions, 0, 1, dt)
        er_end = np.sqrt(e_sol["x"][-1]**2 + e_sol["y"][-1]**2)
        euler_r.append(er_end)

        rk_sol = rk.solve(initial_conditions, 0, 1, dt)
        rk_end = np.sqrt(rk_sol["x"][-1] ** 2 + rk_sol["y"][-1] ** 2)
        runge_r.append(rk_end)

    euler_best = es.solve(initial_conditions, 0, 1, dts[0])
    euler_worst = es.solve(initial_conditions, 0, 1, dts[-1])

    runge_best = rk.solve(initial_conditions, 0, 1, dts[0])
    runge_worst = rk.solve(initial_conditions, 0, 1, dts[-1])

    fig, axis = plt.subplots(2, 2, figsize=(10, 8))
    axis[0, 0].plot(euler_worst["x"], euler_worst["y"], label=f"{dts[-1]}")
    axis[0, 0].plot(euler_best["x"], euler_best["y"], label=f"{dts[0]}")
    axis[0, 0].set_title("Euler")
    axis[0, 0].set_aspect(1)
    axis[0, 0].legend()
    axis[1, 0].grid()

    axis[0, 1].plot(dts, np.array(euler_r) - initial_radius)
    axis[0, 1].set_yscale('log')
    axis[0, 1].set_xscale('log')
    axis[0, 1].set_title(r"Euler diff in distance from center vs $\delta t$")

    axis[1, 0].plot(runge_worst["x"], runge_worst["y"], label=f"{dts[-1]}")
    axis[1, 0].plot(runge_best["x"], runge_best["y"], label=f"{dts[0]}")
    axis[1, 0].set_title("Runge")
    axis[1, 0].set_aspect(1)
    axis[1, 0].legend()
    axis[1, 0].grid()

    axis[1, 1].plot(dts, np.array(runge_r) - initial_radius)
    axis[1, 1].set_yscale('log')
    axis[1, 1].set_xscale('log')
    axis[1, 1].set_title(r"Runge diff in distance from center vs $\delta t$")

    plt.tight_layout()
    plt.show()


def q23():
    ns = np.arange(64, 1024 + 1, 8)
    cfl = 0.5
    upwind = []
    leapfrog = []
    fv_fo = []
    fv_so = []

    for n in ns:
        grid = Grid(top_hat(n, n/4), Boundary.Periodic)

        result_upwind = step_upwind(grid, cfl, int(n / cfl))
        result_leapfrog = step_leapfrog(grid, step_lax_wendorff(grid, cfl, 1), cfl, int(n / cfl))
        result_finite_volume_fo = step_finite_volume_fo(grid, cfl, int(n / cfl))
        result_finite_volume_so = step_finite_volume_so(grid, cfl, int(n / cfl))

        upwind.append(epsilon_diff(grid.grid, result_upwind.grid, 1/n))
        leapfrog.append(epsilon_diff(grid.grid, result_leapfrog.grid, 1 / n))
        fv_fo.append(epsilon_diff(grid.grid, result_finite_volume_fo.grid, 1 / n))
        fv_so.append(epsilon_diff(grid.grid, result_finite_volume_so.grid, 1 / n))

    n = 64
    x = np.linspace(0, 1, n)
    grid = Grid(top_hat(n, n / 4), Boundary.Periodic)

    result_upwind = step_upwind(grid, cfl, int(n / cfl))
    result_leapfrog = step_leapfrog(grid, step_lax_wendorff(grid, cfl, 1), cfl, int(n / cfl))
    result_finite_volume_fo = step_finite_volume_fo(grid, cfl, int(n / cfl))
    result_finite_volume_so = step_finite_volume_so(grid, cfl, int(n / cfl))

    fig, axis = plt.subplots(2, 3, figsize=(10, 12))
    axis[0, 0].plot(x, grid.grid, label="initial")
    axis[0, 0].plot(x, result_upwind.grid, label="final")
    axis[0, 0].set_title("Upwind")

    axis[0, 1].plot(x, grid.grid, label="initial")
    axis[0, 1].plot(x, result_leapfrog.grid, label="final")
    axis[0, 1].set_title("Leapfrog")

    axis[0, 2].plot(ns, upwind, label="upwind")
    axis[0, 2].set_title(r"$\epsilon$ vs $N$")
    axis[0, 2].plot(ns, leapfrog, label="leapfrog")
    axis[0, 2].set_title(r"$\epsilon$ vs $N$")
    axis[0, 2].set_yscale('log')
    axis[0, 2].set_xscale('log')
    axis[0, 2].legend()

    axis[1, 0].plot(x, grid.grid, label="initial")
    axis[1, 0].plot(x, result_finite_volume_fo.grid, label="final")
    axis[1, 0].set_title("First Order")

    axis[1, 1].plot(x, grid.grid, label="initial")
    axis[1, 1].plot(x, result_finite_volume_so.grid, label="final")
    axis[1, 1].set_title("Second Order")

    axis[1, 2].plot(ns, fv_fo, label="finite volume first order")
    axis[1, 2].set_title(r"$\epsilon$ vs $N$")
    axis[1, 2].plot(ns, fv_so, label="finite volume second order")
    axis[1, 2].set_title(r"$\epsilon$ vs $N$")
    axis[1, 2].set_yscale('log')
    axis[1, 2].set_xscale('log')
    axis[1, 2].legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # q1()
    q23()
