import numpy as np
from ex3.ode.solver import Solver, Solution


def as_dict(names, values):
    return {name: v for name, v in zip(names, values)}


class RK4Solver(Solver):
    def solve(self, initial_conditions: dict[str, float], start, end, resolution) -> Solution:
        h = resolution
        steps = int((end - start) / resolution)
        initial_conditions = dict(t=start, **initial_conditions)
        variables = ["t"] + self.system.variables
        relations = [lambda **k: 1] + [self.system.relations[k] for k in self.system.variables]

        points = [np.array([initial_conditions[k] for k in variables])]
        last_point = points[0]

        for _ in range(steps):

            p = as_dict(variables, last_point)
            k1 = h * np.array([r(**p) for r in relations])

            p_k1 = as_dict(variables, last_point + k1 / 2)
            k2 = h * np.array([r(**p_k1) for r in relations])

            p_k2 = as_dict(variables, last_point + k2 / 2)
            k3 = h * np.array([r(**p_k2) for r in relations])

            p_k3 = as_dict(variables, last_point + k3)
            k4 = h * np.array([r(**p_k3) for r in relations])

            kt = (k1 + 2 * k2 + 2 * k3 + k4) / 6
            new_point = last_point + kt
            points.append(new_point)
            last_point = new_point

        return Solution(self.system, np.vstack(points).transpose())


def _test():
    import matplotlib.pyplot as plt
    from ex3.ode.system import System

    odes = {
        "y": lambda t, y: -y**3 + np.sin(t),
    }
    s = System(odes)
    initial_conditions = {
        "y": 0,
    }

    es = RK4Solver(s)
    solution1 = es.solve(initial_conditions, 0, 10, 1)
    solution2 = es.solve(initial_conditions, 0, 10, 0.5)
    solution3 = es.solve(initial_conditions, 0, 10, 0.2)
    solution4 = es.solve(initial_conditions, 0, 10, 0.1)

    plt.plot(solution1.time, solution1["y"], label="1")
    plt.plot(solution2.time, solution2["y"], label="0.5")
    plt.plot(solution3.time, solution3["y"], label="0.2")
    plt.plot(solution4.time, solution4["y"], label="0.1")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    _test()
