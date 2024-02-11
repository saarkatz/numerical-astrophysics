import numpy as np
from ex3.ode.solver import Solver, Solution


class EulerSolver(Solver):
    def solve(self, initial_conditions: dict[str, float], start, end, resolution) -> Solution:
        h = resolution
        steps = int((end - start) / resolution)
        initial_conditions = dict(t=start, **initial_conditions)
        variables = ["t"] + self.system.variables
        relations = [lambda **k: 1] + [self.system.relations[k] for k in self.system.variables]

        points = [np.array([initial_conditions[k] for k in variables])]
        last_point = points[0]
        for _ in range(steps):
            point_dict = {name: v for name, v in zip(variables, last_point)}
            step = np.array([r(**point_dict) for r in relations])
            new_point = last_point + h * step
            points.append(new_point)
            last_point = new_point

        return Solution(self.system, np.vstack(points).transpose())


def _test():
    import matplotlib.pyplot as plt
    from ex3.ode.system import System
    gm = 4 * np.pi ** 2
    odes = {
        "x": lambda t, x, y, vx, vy: vx,
        "y": lambda t, x, y, vx, vy: vy,
        "vx": lambda t, x, y, vx, vy: -gm * x / np.power(x ** 2 + y ** 2, 3 / 2),
        "vy": lambda t, x, y, vx, vy: -gm * y / np.power(x ** 2 + y ** 2, 3 / 2)
    }
    s = System(odes)
    initial_conditions = {
        "x": 0,
        "y": 1,
        "vx": -np.sqrt(gm),
        "vy": 0,
    }
    es = EulerSolver(s)
    solution1 = es.solve(initial_conditions, 0, 1, 1e-2)
    solution2 = es.solve(initial_conditions, 0, 1, 1e-3)
    solution3 = es.solve(initial_conditions, 0, 1, 1e-4)

    plt.plot(solution1["x"], solution1["y"], label="1e-2")
    plt.plot(solution2["x"], solution2["y"], label="1e-3")
    plt.plot(solution3["x"], solution3["y"], label="1e-4")
    plt.gca().set_aspect(1)
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    _test()
