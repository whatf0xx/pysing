from typing import Tuple, Iterator
from random import choice
from itertools import pairwise
import numpy as np
import matplotlib.pyplot as plt
from spin import Spin


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
        return -self.J * self.nn_sum - self.H * self.magnetisation

    @property
    def magnetisation(self) -> np.float32:
        """
        Calculate the magnetisation of the system, equivalent to summing all
        the spins in the system. Useful for calculations of probabilities and
        energies for the model.
        """
        vals = (spin.value for spin in self.spins)
        return sum(vals)

    @property
    def nn_sum(self) -> np.float32:
        """
        Calculate the sum of spin products over nearest neighbour pairs for
        the model.
        """
        pairs = self.get_nn_pairs()
        prod = (s_i.value * s_j.value for s_i, s_j in pairs)
        return sum(prod)

    @property
    def inverse_temp(self) -> np.float32:
        """
        Calculate the inverse temperature of the system.
        """
        k_b = 1.380649e-23
        return 1.0 / (self.T * k_b)

    @property
    def z_prob(self) -> np.float32:
        """
        Calculate the (not normalized) probability of finding the model in the
        current microstate. This is not normalized because this is just the
        numerator of the Boltzmann distribution probability; in other words,
        this is the value of the probaility multiplied through by the partition
        function.
        """
        return np.exp(-self.inverse_temp * self.energy)

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
        s.initialise(np.array(neighbours))

        # for each spin in the top row there are three neighbours...
        for col in range(1, l-1):
            s = self.spins[col]
            neighbours = [
                self.spins[col-1],
                self.spins[col+1],
                self.spins[col+l]
            ]
            s.initialise(np.array(neighbours))

        # ...except the top right, with only two neighbours, again
        s = self.spins[l-1]
        neighbours = [self.spins[l-2], self.spins[2*l - 1]]
        s.initialise(np.array(neighbours))

        for row in range(1, l-1):
            s = self.spins[l * row]
            neighbours = [
                self.spins[l * (row-1)],  # above
                self.spins[l * row + 1],  # right
                self.spins[l * (row+1)]   # below
            ]
            s.initialise(np.array(neighbours))

            for col in range(1, l-1):
                s = self.spins[l * row + col]
                neighbours = [
                    self.spins[l * (row-1) + col],  # above
                    self.spins[l * row + col + 1],  # right
                    self.spins[l * (row+1) + col],  # below
                    self.spins[l * row + col - 1]   # left
                ]
                s.initialise(np.array(neighbours))

            s = self.spins[l * (row+1) - 1]
            neighbours = [
                self.spins[l * row - 1],      # above
                self.spins[l * (row+1) - 2],  # left
                self.spins[l * (row+2) - 1]   # below
            ]
            s.initialise(np.array(neighbours))

        # In the bottom left corner, we only have two neighbours
        s = self.spins[l * (l-1)]
        neighbours = [
            self.spins[l * (l-2)],     # above
            self.spins[l * (l-1) + 1]  # right
        ]
        s.initialise(np.array(neighbours))

        # for each spin in the bottom row there are three neighbours...
        for col in range(1, l-1):
            s = self.spins[l * (l-1) + col]
            neighbours = [
                self.spins[l * (l-1) + col-1],  # left
                self.spins[l * (l-1) + col+1],  # right
                self.spins[l * (l-2) + col]     # above
            ]
            s.initialise(np.array(neighbours))

        # ...except the bottom right, with only two neighbours, again
        s = self.spins[l**2 - 1]
        neighbours = [self.spins[l**2-2], self.spins[l * (l-1) - 1]]
        s.initialise(np.array(neighbours))

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
    m = Model(40, 5, 6, 7)
    m.define_nn_pairs()
    print(m.magnetisation)
    print(m.nn_sum)
    print(m.energy)
    print(m.inverse_temp)
    print(m.z_prob)
    m.plot()
