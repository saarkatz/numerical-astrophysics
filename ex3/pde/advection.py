import numpy as np

from ex3.pde.grid import Grid


def step_upwind(grid, cfl, steps):
    prev_grid = Grid(grid.grid, boundary=grid.boundary)
    next_grid = Grid(grid.grid.shape, boundary=grid.boundary)
    for _ in range(steps):
        next_grid.grid = prev_grid.grid - cfl * (prev_grid.grid - prev_grid[-1:-1])
        prev_grid, next_grid = next_grid, prev_grid
    return prev_grid


def step_lax(grid, cfl, steps):
    prev_grid = Grid(grid.grid, boundary=grid.boundary)
    next_grid = Grid(grid.grid.shape, boundary=grid.boundary)
    for _ in range(steps):
        forward = prev_grid[1:len(prev_grid.grid) + 1] / 2
        backward = prev_grid[-1:-1] / 2
        next_grid.grid = forward + backward - cfl * (forward - backward)
        prev_grid, next_grid = next_grid, prev_grid
    return prev_grid


def step_lax_wendorff(grid, cfl, steps):
    prev_grid = Grid(grid.grid, boundary=grid.boundary)
    next_grid = Grid(grid.grid.shape, boundary=grid.boundary)
    for _ in range(steps):
        forward = prev_grid[1:len(prev_grid.grid) + 1] / 2
        backward = prev_grid[-1:-1] / 2
        next_grid.grid = prev_grid.grid - cfl * (forward - backward) + cfl * cfl * (forward + backward - prev_grid.grid)
        prev_grid, next_grid = next_grid, prev_grid
    return prev_grid


def step_leapfrog(grid: Grid, next_step: Grid, cfl, steps):
    two_prev_grid = Grid(grid.grid, boundary=grid.boundary)
    prev_grid = Grid(next_step.grid, boundary=grid.boundary)
    next_grid = Grid(grid.grid.shape, boundary=grid.boundary)
    for _ in range(steps):
        forward = prev_grid[1:len(prev_grid.grid) + 1]
        backward = prev_grid[-1:-1]
        next_grid.grid = two_prev_grid.grid - cfl * (forward - backward)
        two_prev_grid, prev_grid, next_grid = prev_grid, next_grid, two_prev_grid
    return prev_grid


def step_finite_volume_fo(grid: Grid, cfl, steps):
    """finite volume with first order riemann"""
    sign_u = np.sign(cfl)  # If u>0 take left side of boundaries and if u<0 take the right side
    dir_u = (1 + sign_u) / 2

    prev_grid = Grid(grid.grid, boundary=grid.boundary)
    next_grid = Grid(grid.grid.shape, boundary=grid.boundary)
    for _ in range(steps):
        forward = prev_grid[1 - dir_u:len(prev_grid.grid) + 1 - dir_u]  # i+1/2
        backward = prev_grid[-dir_u:len(prev_grid.grid) - dir_u]  # i-1/2
        next_grid.grid = prev_grid.grid - cfl * (forward - backward)
        prev_grid, next_grid = next_grid, prev_grid
    return prev_grid


def step_finite_volume_so(grid: Grid, cfl, steps):
    """finite volume with second order riemann"""
    sign_u = np.sign(cfl)  # If u>0 take left side of boundaries and if u<0 take the right side
    dir_u = int((1 + sign_u) / 2)

    prev_grid = Grid(grid.grid, boundary=grid.boundary)
    next_grid = Grid(grid.grid.shape, boundary=grid.boundary)
    for _ in range(steps):
        da = (prev_grid[1:len(prev_grid.grid) + 1] - prev_grid[-1:-1]) / 2
        edges = prev_grid.grid + sign_u * da / 2
        forward = np.roll(edges, 1 - dir_u)  # i+1/2
        backward = np.roll(edges, -dir_u)  # i-1/2
        next_grid.grid = prev_grid.grid - cfl * (forward - backward)
        prev_grid, next_grid = next_grid, prev_grid
    return prev_grid


def _test():
    import numpy as np
    import matplotlib.pyplot as plt
    from ex3.pde.grid import Boundary

    size = 128
    x = np.linspace(0, 1, size)
    cfl = 0.5

    grid = Grid(size, Boundary.Periodic)
    grid[size / 2 - size / 8:size / 2 + size / 8] = 1

    result_lax = step_lax(grid, cfl, int(size / cfl))
    result_upwind = step_upwind(grid, cfl, int(size / cfl))
    result_leapfrog = step_leapfrog(grid, step_lax(grid, cfl, 1), cfl, int(size / cfl))
    result_lax_wendorff = step_lax_wendorff(grid, cfl, int(size / cfl))

    result_finite_volume_fo = step_finite_volume_fo(grid, cfl, int(size / cfl))
    result_finite_volume_so = step_finite_volume_so(grid, cfl, int(size / cfl))

    plt.plot(x, grid.grid, label="original")
    plt.plot(x, result_lax.grid, label="lax")
    plt.plot(x, result_lax_wendorff.grid, label="wendorff")
    plt.plot(x, result_upwind.grid, label="upwind")
    plt.plot(x, result_leapfrog.grid, label="leapfrog")
    plt.plot(x, result_finite_volume_fo.grid, label="finite volume FO")
    plt.plot(x, result_finite_volume_so.grid, label="finite volume SO")

    plt.legend()
    plt.show()


if __name__ == '__main__':
    _test()
