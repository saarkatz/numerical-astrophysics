from abc import ABC, abstractmethod

from numpy.typing import NDArray

from ex3.ode.system import System


class Solution:
    def __init__(self, system: System, solution: NDArray):
        self.system = system
        self.time = solution[0]
        self.solutions = {v: s for v, s in zip(system.variables, solution[1:])}

    def __getitem__(self, item):
        return self.solutions[item]


class Solver(ABC):
    def __init__(self, system: System):
        self.system = system

    @abstractmethod
    def solve(self, initial_conditions: dict[str, float], start, end, resolution) -> Solution:
        pass
