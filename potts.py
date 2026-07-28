"""
The q-state Potts model: q states with no geometry between them, coupled only
by whether neighbours agree.

    E = -J sum_<ij> delta(s_i, s_j) - sum_i H[s_i]

Where the clock model's states live on a circle, Potts states are a bare set:
every pair of distinct states is equally distinct. That has two consequences
worth stating up front.

The first is visual, and is a cost: the colour map is *arbitrary*. Any
assignment of q hues to q states is as good as any other, because the model
has no notion of one state being nearer another. What comes out is hard-edged
blocks of flat colour, with no soft walls anywhere -- which is a perfectly
good picture, but it is not the state space rendering itself the way the
clock model's is.

The second is physical, and is a gain: the transition is exactly located for
every q by self-duality, at `T_c = J / ln(1 + sqrt(q))`, and it *changes
order* with q -- continuous for `q <= 4`, first order for `q >= 5`.

The implementation is a small delta from `clock.ClockModel`: only the local
field and the log-weights differ. The neighbour "field" becomes a one-hot
occupancy count, which means the ordinary neighbour-sum stencil works
unchanged, and open boundaries need no special case at all -- with a zero
halo, an absent neighbour is simply in no state.
"""
import numpy as np
import matplotlib.pyplot as plt

from lattice import Lattice
from palette import oklch_ring, render
from sampling import sample_categorical


INITIALISATIONS = ("hot", "cold")


class PottsModel:
    """
    A q-state Potts model on a square lattice, sampled with an exact
    checkerboard heat bath.

    Reduced units throughout: k_B = 1 and J sets the energy scale.

    The state is an `(L, L)` array of `uint8` labels in `0 .. q-1`. `field`
    is a q-vector biasing each state independently; a scalar is taken to mean
    `H[0] = H`, the usual symmetry-breaking field that favours a single
    state.
    """
    def __init__(self,
                 lattice_length: int,
                 q: int=3,
                 temperature: float=1.0,
                 field=0.0,
                 coupling: float=1.0,
                 boundary: str="open",
                 init: str="hot",
                 seed: int | None=None
             ):
        self.lattice = Lattice(lattice_length, boundary)
        if q < 2:
            raise ValueError(f"q must be at least 2, got {q}.")
        if q > 255:
            raise ValueError(
                f"q must fit in the uint8 label array, so at most 255, got {q}."
            )
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        if init not in INITIALISATIONS:
            raise ValueError(
                f"init must be one of {INITIALISATIONS}, got {init!r}."
            )

        self.q = q
        self.rng = np.random.default_rng(seed)
        self._states = np.arange(q, dtype=np.uint8)[:, None, None]

        if init == "hot":
            self.labels = self.rng.integers(
                q, size=(lattice_length, lattice_length), dtype=np.uint8
            )
        else:
            self.labels = np.zeros((lattice_length, lattice_length), dtype=np.uint8)

        self.T = temperature
        self.H = field
        self.J = coupling

    # ----------------------------------------------------------------------
    # Geometry, forwarded from the lattice
    # ----------------------------------------------------------------------

    @property
    def lattice_length(self) -> int:
        return self.lattice.lattice_length

    @property
    def boundary(self) -> str:
        return self.lattice.boundary

    # ----------------------------------------------------------------------
    # Parameters
    # ----------------------------------------------------------------------

    @property
    def H(self) -> np.ndarray:
        """The external field, as a `(q,)` array, one bias per state."""
        return self._H

    @H.setter
    def H(self, value):
        field = np.asarray(value, dtype=float)
        if field.ndim == 0:
            biases = np.zeros(self.q)
            biases[0] = float(field)
            field = biases
        if field.shape != (self.q,):
            raise ValueError(
                f"field must be a scalar or a {self.q}-vector, got shape "
                f"{field.shape}."
            )
        self._H = field

    @property
    def beta(self) -> float:
        if self.T <= 0:
            raise ValueError(f"temperature must be positive, got {self.T}.")
        return 1.0 / self.T

    @property
    def beta_J(self) -> float:
        return self.J / self.T

    @property
    def critical_temperature(self) -> float:
        """
        The exact critical temperature `T_c = J / ln(1 + sqrt(q))`, for every
        q, from the self-duality of the Potts model on the square lattice.

        Exact for the location, but not for the *nature* of what happens
        there: the transition is continuous for `q <= 4` and first order for
        `q >= 5`. As with Onsager, the derivation assumes an infinite lattice
        and zero field, so treat it as a guide to where you are in parameter
        space rather than a prediction for a finite lattice with open edges.
        """
        return self.J / float(np.log(1.0 + np.sqrt(self.q)))

    @property
    def reduced_temperature(self) -> float:
        if self.J == 0:
            raise ValueError(
                "The reduced temperature is undefined at zero coupling: with "
                "J = 0 there is no critical point to measure against."
            )
        return self.T / self.critical_temperature

    @property
    def is_first_order(self) -> bool:
        """
        Whether the transition is first order, which it is for `q >= 5`.

        Worth checking before trusting any equilibrium average near `T_c`:
        the two phases coexist there, single-site dynamics tunnels between
        them exponentially slowly in the lattice size, and a run that looks
        converged may simply be stuck in one of them.
        """
        return self.q >= 5

    # ----------------------------------------------------------------------
    # State, energy and order parameters
    # ----------------------------------------------------------------------

    def _onehot(self) -> np.ndarray:
        """`(q, L, L)` indicator: is site (i, j) in state k?"""
        return (self.labels[None, :, :] == self._states).astype(np.int8)

    def _counts(self) -> np.ndarray:
        """
        `(q, L, L)`: how many of a site's neighbours are in each state.

        Values lie in [0, 4], so `int8` is ample and no promotion happens in
        the stencil. This is the Potts analogue of the Ising neighbour sum,
        and it is literally the same function call.
        """
        return self.lattice.neighbour_sum(self._onehot())

    def log_weights(self) -> np.ndarray:
        """`(q, L, L)` unnormalised log conditional weights."""
        return self.beta * (self.J * self._counts() + self.H[:, None, None])

    def conditional_probabilities(self) -> np.ndarray:
        """The normalised `(q, L, L)` site conditionals, for tests."""
        weights = self.log_weights()
        weights = np.exp(weights - weights.max(axis=0, keepdims=True))
        return weights / weights.sum(axis=0, keepdims=True)

    @property
    def agreeing_bonds(self) -> int:
        """
        The number of bonds whose two ends are in the same state. The whole
        of the interaction energy is `-J` times this.
        """
        onehot = self._onehot()
        # Picking out each site's own row of the neighbour counts and summing
        # counts every agreeing bond once from each end.
        return int((onehot.astype(np.int64) * self._counts()).sum()) // 2

    @property
    def energy(self) -> float:
        return (-self.J * self.agreeing_bonds
                - float(self.H @ (self.populations * self.lattice.n_sites)))

    @property
    def energy_per_spin(self) -> float:
        return self.energy / self.lattice.n_sites

    @property
    def populations(self) -> np.ndarray:
        """The fraction of sites in each state, `(q,)`, summing to 1."""
        return (np.bincount(self.labels.ravel(), minlength=self.q)
                / self.lattice.n_sites)

    @property
    def order_parameter(self) -> float:
        """
        `(q * max_k rho_k - 1) / (q - 1)`, where `rho_k` is the fraction of
        sites in state k.

        Magnetisation does not survive the loss of the +-1 alphabet, so this
        replaces it: it is 0 when every state is equally populated and 1 when
        one state has taken the whole lattice, for any q.

        Note it is positive-definite and biased upwards on a finite lattice,
        since the largest of q fluctuating populations exceeds 1/q even in a
        completely disordered state. The bias falls as 1/L.
        """
        return (self.q * float(self.populations.max()) - 1.0) / (self.q - 1)

    @property
    def log_weight(self) -> float:
        """`-beta * E`; see `clock.ClockModel.log_weight`."""
        return -self.beta * self.energy

    # ----------------------------------------------------------------------
    # Dynamics
    # ----------------------------------------------------------------------

    def evolve(self):
        """
        Advance the model by one sweep, resampling every site exactly once,
        one checkerboard sublattice at a time.

        Near a first-order transition (`q >= 5`) this is not enough on its
        own to equilibrate: see `is_first_order`.
        """
        for sublattice in self.lattice.sublattices:
            drawn = sample_categorical(self.log_weights(), self.rng, dtype=np.uint8)
            np.copyto(self.labels, drawn, where=sublattice)

    def _check_state(self) -> bool:
        length = self.lattice_length
        return (self.labels.shape == (length, length)
                and self.labels.dtype == np.uint8
                and bool(np.all(self.labels < self.q)))

    # ----------------------------------------------------------------------
    # Rendering
    # ----------------------------------------------------------------------

    def palette(self, **kwargs) -> np.ndarray:
        """
        A `(q, 3)` sRGB ring for the states.

        Unlike the clock model, this assignment is *arbitrary*: Potts states
        have no ordering, so hue carries no meaning beyond distinguishing
        them. The equal-lightness ring is still the right choice, because it
        is the one that adds no spurious contrast between states that are
        physically equivalent.
        """
        return oklch_ring(self.q, **kwargs)

    def to_rgb(self, **kwargs) -> np.ndarray:
        return render(self.labels, self.palette(**kwargs))

    def plot_to_axes(self, ax: plt.Axes, **kwargs):
        ax.imshow(self.to_rgb(**kwargs), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])

    def plot(self, filename: str | None=None, dpi: int=150, **kwargs):
        fig, ax = plt.subplots()
        self.plot_to_axes(ax, **kwargs)
        ax.set_title(f"q={self.q}, T={self.T}, J={self.J}, "
                     f"boundary={self.boundary!r}")
        if filename is not None:
            fig.savefig(filename, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def __repr__(self) -> str:
        return (f"{type(self).__name__}({self.lattice_length}, q={self.q}, "
                f"temperature={self.T}, coupling={self.J}, "
                f"boundary={self.boundary!r})")
