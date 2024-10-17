from typing import Tuple, Iterator
from random import choice
from itertools import pairwise
import numpy as np
import matplotlib.pyplot as plt

class Spin:
    def __init__(self, value: np.float32, _id=0):
        assert value in (1., -1.), "Spins can only have value +/- 1."
        self.value = value
        self.id = _id
        self.initialised = False
        self.nearest_neighbours = None

    def __repr__(self):
        d = "up" if self.value == 1. else "down"
        return f"spin-{d}@{self.id}"

    def initialise(self, nearest_neighbours: np.ndarray):
        self.nearest_neighbours = nearest_neighbours
        self.initialised = True

    def flip(self):
        self.value *= -1


class Model:
    """
    An instance of the Ising model, defined on a square lattice with
    `lattice_length` spins in each direction.
    """
    def __init__(self,
                 lattice_length: int,
                 temperature: np.float32,
                 field: np.float32,
                 coupling: np.float32
             ):
        self.lattice_length = lattice_length
        self.spins = np.array(
                            [Spin(choice((1., -1.)), _id=i) for i in range(lattice_length ** 2)]
                        )
        self.T = temperature
        self.H = field
        self.J = coupling

    @property
    def energy(self) -> np.float32:
        """
        Calculate the energy of the system, according to the standard
        'nearest-neightbours' approach of the Ising model. For the square
        lattice, this means the neighbours along the x-y axes, without
        diagonals and without periodic boundary conditions.
        """
        nn_pairs = self.get_nn_pairs()
        pairs_product = np.array([s_i * s_j for s_i, s_j in nn_pairs])
        spins = np.array([spin.value for spin in self.spins])
        return -self.J * np.sum(pairs_product) - self.H * np.sum(spins)

    def get_nn_pairs(self) -> Iterator[Tuple[Spin, Spin]]:
        """
        Iterate over nearest-neighbour pairs in the model. Currently, the order
        in which pairs are returned is arbitrary and defined by the
        implementation; here, all of the row-wise pairs are defined first,
        followed by the column wise pairs, in both cases working left to right
        and then top to bottom.
        """
        l = self.lattice_length
        for row in range(l):
            for s_i, s_j in pairwise(self.spins[row*l:(row+1)*l]):
                yield s_i, s_j

        for i in range(l * (l - 1)):
            s_i = self.spins[i]
            s_j = self.spins[i + l]
            yield s_i, s_j

    def define_nn_pairs(self):
        """
        Initialise the model by setting the nearest neighbour for each spin, so
        that they can quickly be retrieved later.
        """
        l = self.lattice_length
        if l == 1:  # model has no nearest neighbours
            s = self.spins[0]
            s.initialised = True
            return

        # In the top left corner, we only have two neighbours
        s = self.spins[0]
        neighbours = [self.spins[1], self.spins[l]]
        s.nearest_neighbours = neighbours
        s.initialised = True

        # for each spin in the top row there are three neighbours...
        for col in range(1, l-1):
            s = self.spins[col]
            neighbours = [
                self.spins[col-1],
                self.spins[col+1],
                self.spins[col+l]
            ]
            s.nearest_neighbours = neighbours
            s.intialised = True

        # ...except the top right, with only two neighbours, again
        s = self.spins[l-1]
        neighbours = [self.spins[l-2], self.spins[2*l - 1]]
        s.nearest_neighbours = neighbours
        s.initialised = True

        for row in range(1, l-1):
            s = self.spins[l * row]
            neighbours = [
                self.spins[l * (row-1)],  # above
                self.spins[l * row + 1],  # right
                self.spins[l * (row+1)]   # below
            ]
            s.nearest_neighbours = neighbours
            s.initialised = True

            for col in range(1, l-1):
                s = self.spins[l * row + col]
                neighbours = [
                    self.spins[l * (row-1) + col],  # above
                    self.spins[l * row + col + 1],  # right
                    self.spins[l * (row+1) + col],  # below
                    self.spins[l * row + col - 1]   # left
                ]
                s.nearest_neighbours = neighbours
                s.initialised = True

            s = self.spins[l * (row+1) - 1]
            neighbours = [
                self.spins[l * row - 1],      # above
                self.spins[l * (row+1) - 2],  # left
                self.spins[l * (row+2) - 1]   # below
            ]
            s.nearest_neighbours = neighbours
            s.initialised = True

        # In the bottom left corner, we only have two neighbours
        s = self.spins[l * (l-1)]
        neighbours = [
            self.spins[l * (l-2)],     # above
            self.spins[l * (l-1) + 1]  # right
        ]
        s.nearest_neighbours = neighbours
        s.initialised = True

        # for each spin in the bottom row there are three neighbours...
        for col in range(1, l-1):
            s = self.spins[l * (l-1) + col]
            neighbours = [
                self.spins[l * (l-1) + col-1],  # left
                self.spins[l * (l-1) + col+1],  # right
                self.spins[l * (l-2) + col]     # above
            ]
            s.nearest_neighbours = neighbours
            s.intialised = True

        # ...except the bottom right, with only two neighbours, again
        s = self.spins[l**2 - 1]
        neighbours = [self.spins[l**2-2], self.spins[l * (l-1) - 1]]
        s.nearest_neighbours = neighbours
        s.initialised = True
            

    def plot(self, filename: str | None=None):
        """
        For debugging or demonstration purposes, plot the current state of the
        model as a bitmap. If `filename` is passed as an argument, save it to
        the corresponding location, otherwise show the figure.
        """
        _ = plt.figure()
        ax = plt.axes()

        data = np.array([(spin.value + 1) / 2 for spin in self.spins])
        bitmap = data.reshape((self.lattice_length, self.lattice_length))

        ax.imshow(bitmap, cmap="magma")
        T, H, J = self.T, self.H, self.J
        ax.set_title(f"{T=}, {H=}, {J=}")
        if filename is not None:
            plt.savefig(filename=filename, dpi=320)
        else:
            plt.show()


if __name__ == "__main__":
    m = Model(4, 5, 6, 7)
    m.define_nn_pairs()
    for spin in m.spins:
        print(spin, spin.nearest_neighbours)
    m.plot()
