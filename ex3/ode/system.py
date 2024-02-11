from typing import Callable


class System:
    """define a system of variables and first order differential relations between them

    The system contains the variables as a list of names and a relations dictionary that relates each variable
    to the function for its derivative.
    """

    def __init__(self, odes: dict[str, Callable]):
        self.variables = list(odes.keys())
        self.relations = odes
