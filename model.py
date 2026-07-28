import numpy as np
import matplotlib.pyplot as plt


BOUNDARIES = ("open", "periodic")

#: ln(1 + sqrt(2)); the Onsager critical point is at k_B T_c / J = 2 / this.
_LN1P_SQRT2 = float(np.log(1.0 + np.sqrt(2.0)))


class Model:
    """
    An instance of the Ising model, defined on a square lattice with
    `lattice_length` spins in each direction.

    Reduced units are used throughout: k_B = 1 and the coupling J sets the
    energy scale, so the temperature is measured in units of J / k_B and the
    critical point of the infinite lattice sits at T = 2.269... J. Only the
    dimensionless groups beta*J and beta*H carry any physics.

    The state is a single `(L, L)` array of `int8` spins, values +/- 1, indexed
    `[row, col]`. Boundaries are open by default; pass `boundary="periodic"`
    for wraparound, which requires an even lattice length of at least 4 (see
    `evolve`).
    """
    def __init__(self,
                 lattice_length: int,
                 temperature: float=2.0,
                 field: float=0.,
                 coupling: float=1.0,
                 boundary: str="open",
                 seed: int | None=None
             ):
        if lattice_length < 1:
            raise ValueError(
                f"lattice_length must be at least 1, got {lattice_length}."
            )
        if boundary not in BOUNDARIES:
            raise ValueError(
                f"boundary must be one of {BOUNDARIES}, got {boundary!r}."
            )
        if boundary == "periodic":
            if lattice_length < 3:
                raise ValueError(
                    "Periodic boundaries need lattice_length >= 3; below that a "
                    "site is its own neighbour twice over and every bond is "
                    f"double counted (got {lattice_length})."
                )
            if lattice_length % 2 != 0:
                raise ValueError(
                    "Periodic boundaries need an even lattice_length: the wrap "
                    "closes odd-length cycles, the lattice stops being "
                    "bipartite, and the checkerboard update in `evolve` is no "
                    f"longer exact (got {lattice_length})."
                )
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}.")

        self.lattice_length = lattice_length
        self.boundary = boundary
        self.rng = np.random.default_rng(seed)
        self.spins = self.rng.choice(
            np.array([-1, 1], dtype=np.int8), size=(lattice_length, lattice_length)
        )

        # Scratch buffer for neighbour sums, allocated once. The halo stays
        # zero for open boundaries and is refilled from the edges for periodic
        # ones, so both cases share a single code path.
        self._pad = np.zeros((lattice_length + 2, lattice_length + 2), dtype=np.int8)

        # Checkerboard colouring: every black site's neighbours are all white.
        rows, cols = np.indices((lattice_length, lattice_length))
        self._black = (rows + cols) % 2 == 0
        self._white = ~self._black

        self.T = temperature
        self.H = field
        self.J = coupling

    @property
    def beta(self) -> float:
        """
        Inverse temperature, 1 / T. With k_B = 1 this is measured in units of
        1 / J.
        """
        if self.T <= 0:
            raise ValueError(f"temperature must be positive, got {self.T}.")
        return 1.0 / self.T

    @property
    def beta_J(self) -> float:
        """
        The dimensionless coupling beta*J. Together with `beta_H` this is the
        only thing the dynamics actually depends on.
        """
        return self.J / self.T

    @property
    def beta_H(self) -> float:
        """
        The dimensionless field beta*H.
        """
        return self.H / self.T

    @property
    def critical_temperature(self) -> float:
        """
        The Onsager critical temperature, T_c = 2 J / ln(1 + sqrt(2)).

        Onsager's exact solution assumes an infinite lattice, zero field and
        periodic boundaries, none of which hold strictly here: a finite lattice
        has no true transition, and open boundaries suppress order near the
        free surface. Treat this as a guide to where you are in parameter
        space, not as a prediction for this model.
        """
        return 2.0 * self.J / _LN1P_SQRT2

    @property
    def reduced_temperature(self) -> float:
        """
        The temperature of the system in units of the critical temperature,
        T / T_c, i.e. how close the system is to criticality. Values below 1
        are ordered, above 1 disordered. See `critical_temperature` for the
        caveats.
        """
        if self.J == 0:
            raise ValueError(
                "The reduced temperature is undefined at zero coupling: with "
                "J = 0 there is no critical point to measure against."
            )
        return self.T / self.critical_temperature

    @property
    def energy(self) -> float:
        """
        Calculate the energy of the system, according to the standard
        'nearest-neighbours' approach of the Ising model. For the square
        lattice, this means the neighbours along the x-y axes, without
        diagonals; whether the lattice wraps is set by `boundary`.

        Each bond appears twice in the site-wise sum of spin times neighbour
        sum, hence the factor of one half.
        """
        bond_sum = (self.spins * self._neighbour_sum()).sum(dtype=np.int64)
        return -0.5 * self.J * float(bond_sum) - self.H * self.magnetisation

    @property
    def energy_per_spin(self) -> float:
        """
        The energy divided by the number of spins, which stays finite as the
        lattice grows.
        """
        return self.energy / self.lattice_length ** 2

    @property
    def magnetisation(self) -> int:
        """
        Calculate the magnetisation of the system, equivalent to summing all
        the spins in the system. Useful for calculations of probabilities and
        energies for the model.
        """
        return int(self.spins.sum(dtype=np.int64))

    @property
    def magnetisation_per_spin(self) -> float:
        """
        The magnetisation divided by the number of spins, so it lies in
        [-1, 1] regardless of lattice size. This is the order parameter.
        """
        return self.magnetisation / self.lattice_length ** 2

    @property
    def log_weight(self) -> float:
        """
        The logarithm of the unnormalised Boltzmann weight of the current
        microstate, -beta * E. This is the log of the numerator of the
        Boltzmann distribution; the partition function is not included.

        It is returned in log form deliberately: exp(-beta * E) overflows to
        infinity above roughly a 20x20 lattice. Any ratio of weights should be
        built from a difference of energies, never by exponentiating this. The
        sampler itself never needs the quantity at all.
        """
        return -self.beta * self.energy

    def _neighbour_sum(self) -> np.ndarray:
        """
        Sum of the four nearest-neighbour spins, per site, as an `(L, L)`
        array. Absent neighbours contribute zero, so open boundaries fall out
        of a zero halo and periodic ones out of a wrapped halo. Values stay in
        [-4, 4], well inside `int8`.
        """
        spins = self.spins
        pad = self._pad
        pad[1:-1, 1:-1] = spins
        if self.boundary == "periodic":
            pad[0, 1:-1] = spins[-1, :]
            pad[-1, 1:-1] = spins[0, :]
            pad[1:-1, 0] = spins[:, -1]
            pad[1:-1, -1] = spins[:, 0]
        # The halo corners are never read.
        return (pad[:-2, 1:-1] + pad[2:, 1:-1]
                + pad[1:-1, :-2] + pad[1:-1, 2:])

    def _local_field(self) -> np.ndarray:
        """
        The local field H + J * sum_nn s_j at every site. This is an energy,
        and beta times it is the argument of the heat-bath rule below.
        """
        return self.H + self.J * self._neighbour_sum()

    def heat_bath_probs(self) -> np.ndarray:
        """
        The probability of each site taking the value +1 when it is next
        resampled, given the current state of its neighbours:

            p_i = (1 + tanh(beta * (H + J * sum_nn s_j))) / 2

        which is exactly the heat-bath (Glauber) conditional
        exp(beta h_i) / (exp(beta h_i) + exp(-beta h_i)).

        Note that `evolve` does not use this array wholesale: it recomputes
        the probabilities for each sublattice in turn, because the second
        sublattice must see the freshly updated first one.
        """
        return 0.5 * (1.0 + np.tanh(self.beta * self._local_field()))

    def evolve(self):
        """
        Advance the model by one sweep, resampling every spin exactly once.

        The lattice is bipartite, so colouring it like a chessboard makes every
        black site's neighbours white and vice versa. A whole sublattice can
        therefore be redrawn simultaneously from its exact conditional
        distribution, and the two half-sweeps together leave the Boltzmann
        distribution invariant. Updating *all* spins from the same stale state
        would not: that is parallel dynamics, which breaks detailed balance.

        The probabilities and random numbers are computed over the full
        lattice each half-sweep and then half discarded. Fancy-indexing the
        active sublattice instead costs more than the wasted vectorised work
        at any lattice size worth simulating.
        """
        beta = self.beta
        for sublattice in (self._black, self._white):
            p = 0.5 * (1.0 + np.tanh(beta * self._local_field()))
            new = np.where(self.rng.random(p.shape) < p, np.int8(1), np.int8(-1))
            np.copyto(self.spins, new, where=sublattice)

    def _check_state(self) -> bool:
        """
        Confirm that the spin array is still a well-formed state: the right
        shape and dtype, with every entry +/- 1. Intended for tests, not for
        the hot path, where the invariant is structural.
        """
        length = self.lattice_length
        return (self.spins.shape == (length, length)
                and self.spins.dtype == np.int8
                and bool(np.all(np.abs(self.spins) == 1)))

    def plot(self, filename: str | None=None, dpi: int=150):
        """
        For debugging or demonstration purposes, plot the current state of the
        model as a bitmap. If `filename` is passed as an argument, save it to
        the corresponding location, otherwise show the figure.
        """
        fig, ax = plt.subplots()

        self.plot_to_axes(ax)
        T, H, J = self.T, self.H, self.J
        ax.set_title(f"{T=}, {H=}, {J=}, boundary={self.boundary!r}")

        if filename is not None:
            fig.savefig(filename, dpi=dpi)
            plt.close(fig)
        else:
            plt.show()

    def plot_to_axes(self, ax: plt.Axes):
        """
        Plot the current state of the model as a bitmap on an existing set of
        axes, for multi-panel figures.
        """
        ax.imshow((self.spins + 1) / 2, cmap="magma", vmin=0, vmax=1)
