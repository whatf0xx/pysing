import numpy as np


class Spin:
    """
    A spin within an Ising model. Holds a value, +/- 1; a list of its nearest
    neighbours in the model, and an ID, for debugging.
    """
    def __init__(self, value: np.float64, _id=0):
        assert value in (1., -1.), "Spins can only have value +/- 1."
        self.value = value
        self.id = _id
        self.initialised = False
        self.nearest_neighbours = None

    def __repr__(self):
        d = "up" if self.value == 1. else "down"
        uninit = "UNINITIALISED" if not self.initialised else ""
        return f"{uninit}spin-{d}@{self.id}"

    def initialise(self, nearest_neighbours: np.ndarray):
        """
        Take an `ndarray` of Spins and add them as the nearest neighbours of
        the Spin. This initialises the spin, so it can be used in the model.
        """
        self.nearest_neighbours = nearest_neighbours
        self.initialised = True

    def flip(self):
        """
        Invert the value/direction of the spin.
        """
        self.value *= -1

    def nn_sum(self) ->  np.float64:
        """
        Sum the spins of the nearest neighbours of `self`.
        """
        vals = (spin.value for spin in self.nearest_neighbours)
        return sum(vals)
