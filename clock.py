"""
The q-state clock model: spins confined to q equally spaced directions on a
circle.

    E = -J sum_<ij> cos(theta_i - theta_j) - H . sum_i n_i,
    theta_k = 2*pi*k/q,  n_k = (cos theta_k, sin theta_k)

This is the model this package's colour work is built around, because its
state space *is* the hue wheel. The map from state to colour is forced rather
than chosen, and states that are close in energy are close in colour: domain
walls between adjacent hues are soft, walls between opposite hues are hard,
and that reads correctly with no tuning at all.

The trick that makes it reuse the Ising machinery rather than sit beside it
is that `cos(theta_i - theta_j) = n_i . n_j`, so the local field is a
two-component vector

    h_i = H + J sum_nn n_j

and the site conditional is `p(k) proportional to exp(beta * h_i . n_k)` --
the same "dot the local field with the candidate state" shape as the Ising
heat bath, with a two-component field and q candidates in place of a
one-component field and two.
"""
import numpy as np
import matplotlib.pyplot as plt

from lattice import Lattice
from palette import oklch_ring, render
from sampling import sample_categorical


INITIALISATIONS = ("hot", "cold")

#: ln(1 + sqrt(2)) and ln(1 + sqrt(3)); the exact critical points at q = 2, 4
#: and q = 3 respectively. See `ClockModel.critical_temperature`.
_LN1P_SQRT2 = float(np.log(1.0 + np.sqrt(2.0)))
_LN1P_SQRT3 = float(np.log(1.0 + np.sqrt(3.0)))

#: The Kosterlitz-Thouless temperature of the 2D XY model, in units of J.
#: The upper transition of the q-state clock model approaches it from below
#: as q grows.
XY_BKT_TEMPERATURE = 0.8929


class ClockModel:
    """
    A q-state clock model on a square lattice, sampled with an exact
    checkerboard heat bath.

    Reduced units throughout: k_B = 1 and J sets the energy scale, so
    temperatures are in units of J / k_B.

    The state is an `(L, L)` array of `uint8` *labels* in `0 .. q-1`, not
    angles. The state space is discrete, and labels keep it exactly so;
    angles and unit vectors are derived on demand through a length-q lookup
    table.

    `field` is a two-vector, which biases a direction and breaks the Z_q
    symmetry. A scalar is taken to mean a field along theta = 0, matching the
    scalar field of the Ising model. It is worth using: at H = 0 the ordered
    state picks one of q colours at random, so panels of a multi-panel figure
    come out in unrelated colours; a small field pins them and makes the
    panels comparable.
    """
    def __init__(self,
                 lattice_length: int,
                 q: int=6,
                 temperature: float=1.0,
                 field=(0.0, 0.0),
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

        angles = 2.0 * np.pi * np.arange(q) / q
        #: `(2, q)`: the unit vector of each state, as columns.
        self._unit = np.stack([np.cos(angles), np.sin(angles)])

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
        """The external field, as a `(2,)` array."""
        return self._H

    @H.setter
    def H(self, value):
        field = np.asarray(value, dtype=float)
        if field.ndim == 0:
            field = np.array([float(field), 0.0])
        if field.shape != (2,):
            raise ValueError(
                f"field must be a scalar or a 2-vector, got shape {field.shape}."
            )
        self._H = field

    @property
    def beta(self) -> float:
        """Inverse temperature, 1 / T, in units of 1 / J."""
        if self.T <= 0:
            raise ValueError(f"temperature must be positive, got {self.T}.")
        return 1.0 / self.T

    @property
    def beta_J(self) -> float:
        return self.J / self.T

    @property
    def angles(self) -> np.ndarray:
        """The angle of every site, `(L, L)` in `[0, 2*pi)`."""
        return 2.0 * np.pi * self.labels / self.q

    @property
    def critical_temperature(self) -> float:
        """
        The exact critical temperature, where one exists in closed form.

        Three values of q reduce to models that are solved:

        - `q = 2` is the Ising model at the *same* coupling, since
          `theta` in `{0, pi}` makes `cos(theta_i - theta_j) = s_i s_j`. So
          `T_c = 2J / ln(1 + sqrt(2))`.
        - `q = 3` maps onto the 3-state Potts model at `J_Potts = 3J/2`,
          because `cos` takes only the values `{1, -1/2}` on the allowed
          angle differences, an affine function of the Kronecker delta.
          Potts is self-dual, giving `T_c = 3J / (2 ln(1 + sqrt(3)))`.
        - `q = 4` is *two decoupled Ising models at J/2*: rotate the axes by
          45 degrees and `n = (s, t)/sqrt(2)`, so
          `n_i . n_j = (s_i s_j + t_i t_j) / 2`. So `T_c = J / ln(1+sqrt(2))`.

        For `q >= 5` there is no single critical temperature to return: the
        model has *two* Kosterlitz-Thouless transitions with a
        quasi-long-range-ordered phase between them, and neither is known in
        closed form. Raises, rather than returning a plausible number.
        """
        if self.q == 2:
            return 2.0 * self.J / _LN1P_SQRT2
        if self.q == 3:
            return 1.5 * self.J / _LN1P_SQRT3
        if self.q == 4:
            return self.J / _LN1P_SQRT2
        raise ValueError(
            f"The q = {self.q} clock model has no single critical temperature: "
            "for q >= 5 there are two Kosterlitz-Thouless transitions with a "
            "quasi-long-range-ordered phase between them. The lower one falls "
            "roughly as 1/q**2; the upper one approaches the XY value "
            f"T ~ {XY_BKT_TEMPERATURE} J from below. Neither is exact, so "
            "neither is returned."
        )

    @property
    def reduced_temperature(self) -> float:
        """T / T_c; see `critical_temperature` for when that exists."""
        if self.J == 0:
            raise ValueError(
                "The reduced temperature is undefined at zero coupling: with "
                "J = 0 there is no critical point to measure against."
            )
        return self.T / self.critical_temperature

    # ----------------------------------------------------------------------
    # State, energy and order parameters
    # ----------------------------------------------------------------------

    def _vectors(self) -> np.ndarray:
        """The unit vector of every site, `(2, L, L)`."""
        return self._unit[:, self.labels]

    def _local_field(self) -> np.ndarray:
        """
        The local field `H + J sum_nn n_j` at every site, `(2, L, L)`. This
        is an energy per unit spin, and beta times its dot product with a
        candidate state is the log-weight of that state.
        """
        return (self.H[:, None, None]
                + self.J * self.lattice.neighbour_sum(self._vectors()))

    def log_weights(self) -> np.ndarray:
        """
        `(q, L, L)` unnormalised log conditional weights, `beta * h_i . n_k`.

        Note that `evolve` does not use this array wholesale: it recomputes
        the weights for each sublattice in turn, because the second
        sublattice must see the freshly updated first one.
        """
        return self.beta * np.einsum("ck,cij->kij", self._unit, self._local_field())

    def conditional_probabilities(self) -> np.ndarray:
        """
        The normalised `(q, L, L)` site conditionals. The sampler never needs
        these -- it works from the log-weights -- but tests and the q = 2
        comparison against `model.Model` do.
        """
        weights = self.log_weights()
        weights = np.exp(weights - weights.max(axis=0, keepdims=True))
        return weights / weights.sum(axis=0, keepdims=True)

    @property
    def energy(self) -> float:
        """
        The energy of the current microstate. Each bond appears twice in the
        site-wise sum of a spin against its neighbour sum, hence the half.
        """
        vectors = self._vectors()
        bond_sum = float((vectors * self.lattice.neighbour_sum(vectors)).sum())
        return -0.5 * self.J * bond_sum - float(self.H @ self.magnetisation)

    @property
    def energy_per_spin(self) -> float:
        return self.energy / self.lattice.n_sites

    @property
    def magnetisation(self) -> np.ndarray:
        """
        The total magnetisation, a `(2,)` vector. Unlike the Ising model's
        scalar: the order parameter of a clock model lives in the plane, and
        its direction is which of the q colours the system chose.
        """
        return self._vectors().sum(axis=(1, 2))

    @property
    def magnetisation_per_spin(self) -> np.ndarray:
        """The `(2,)` magnetisation per site; its length is at most 1."""
        return self.magnetisation / self.lattice.n_sites

    @property
    def order_parameter(self) -> float:
        """
        `|m|`, the scalar to plot: 1 for a fully aligned lattice, and O(1/L)
        for a disordered one.

        In the intermediate phase of a `q >= 5` model this is neither: it
        decays as a power of L rather than saturating or vanishing, which is
        what quasi-long-range order means.
        """
        return float(np.hypot(*self.magnetisation_per_spin))

    @property
    def symmetry_breaking(self) -> float:
        """
        `cos(q * phi)`, where `phi` is the direction of the magnetisation.

        This is what separates the two ordered regimes of a `q >= 5` model,
        and neither `energy` nor `order_parameter` can do it. In the locked
        low-temperature phase the magnetisation points at one of the q
        allowed directions, so this sits near +1. In the intermediate
        quasi-long-range-ordered phase the direction is free to wander round
        the circle and this averages to zero, even though `|m|` is still
        appreciable.
        """
        x, y = self.magnetisation_per_spin
        return float(np.cos(self.q * np.arctan2(y, x)))

    @property
    def populations(self) -> np.ndarray:
        """The fraction of sites in each state, `(q,)`, summing to 1."""
        return (np.bincount(self.labels.ravel(), minlength=self.q)
                / self.lattice.n_sites)

    def phase_field(self, width: int) -> tuple[np.ndarray, np.ndarray]:
        """
        The local mean direction, as `(angle, coherence)` arrays of the full
        `(L, L)` shape, from averaging the spin *vectors* over a
        `width` x `width` window centred on each site.

        What is interesting about the intermediate phase of a `q >= 5` clock
        model is *long-wavelength*: the direction winds slowly across the
        lattice while individual sites still fluctuate hard between
        neighbouring states. At the site level that structure is buried in
        thermal speckle -- at `T = 0.65 J` and `q = 8` fewer than half the
        sites are in the most popular state -- so a raw microstate does not
        show it. Averaging cancels the fluctuation and leaves the field
        underneath.

        Averaging the vectors is the only thing that makes sense here.
        Averaging labels would be meaningless, because the labels wrap:
        states 0 and `q-1` are adjacent, and their mean label is the state
        diametrically opposite to both.

        `coherence` is the length of the local mean, in `[0, 1]`, and it is
        what stops this being a machine for inventing structure. Where the
        spins are genuinely uncorrelated the mean is short and the angle it
        reports is meaningless; a renderer should show that by desaturating
        rather than by drawing a confident colour. Uncorrelated directions
        average to a length of roughly `1/width`, so a window wide enough to
        push that below the eye's threshold is the right one to pick.
        """
        if width < 1 or width > self.lattice_length:
            raise ValueError(
                f"width must be between 1 and {self.lattice_length}, got {width}."
            )
        averaged = self._window_mean(self._vectors(), width)
        return (np.arctan2(averaged[1], averaged[0]) % (2 * np.pi),
                np.hypot(averaged[0], averaged[1]))

    def _window_mean(self, field: np.ndarray, width: int) -> np.ndarray:
        """
        Separable box average of a `(C, L, L)` field over a `width`-wide
        window, wrapping or clamping at the edge to match the boundary
        condition. Two passes of a 1D sum rather than one `width**2` pass.
        """
        half = width // 2
        mode = "wrap" if self.boundary == "periodic" else "edge"
        length = self.lattice_length
        padded = np.pad(field, ((0, 0), (half, width - 1 - half),
                                (half, width - 1 - half)), mode=mode)
        rows = sum(padded[:, offset:offset + length, :] for offset in range(width))
        return sum(rows[:, :, offset:offset + length]
                   for offset in range(width)) / width ** 2

    @property
    def log_weight(self) -> float:
        """
        `-beta * E`: the log of the unnormalised Boltzmann weight of the
        current microstate. Returned in log form because `exp(-beta * E)`
        overflows well before any lattice size worth simulating.
        """
        return -self.beta * self.energy

    # ----------------------------------------------------------------------
    # Dynamics
    # ----------------------------------------------------------------------

    def evolve(self):
        """
        Advance the model by one sweep, resampling every site exactly once.

        Identical in structure to `model.Model.evolve`: the lattice is
        bipartite, so one whole sublattice can be redrawn simultaneously from
        its exact conditional, and the two half-sweeps together leave the
        Boltzmann distribution invariant. Only the alphabet differs -- a
        q-way categorical draw where the Ising model has a coin flip.
        """
        for sublattice in self.lattice.sublattices:
            drawn = sample_categorical(self.log_weights(), self.rng, dtype=np.uint8)
            np.copyto(self.labels, drawn, where=sublattice)

    def _check_state(self) -> bool:
        """
        Confirm the label array is still a well-formed state: right shape and
        dtype, every entry a valid label. For tests, not the hot path.
        """
        length = self.lattice_length
        return (self.labels.shape == (length, length)
                and self.labels.dtype == np.uint8
                and bool(np.all(self.labels < self.q)))

    # ----------------------------------------------------------------------
    # Rendering
    # ----------------------------------------------------------------------

    def palette(self, **kwargs) -> np.ndarray:
        """
        The `(q, 3)` sRGB ring this model's states map onto. Keyword
        arguments go to `palette.oklch_ring`.

        State k sits at hue `2*pi*k/q`, the same angle as the spin itself, so
        the colour wheel and the state space are the same circle.
        """
        return oklch_ring(self.q, **kwargs)

    def to_rgb(self, **kwargs) -> np.ndarray:
        """
        The current state as an `(L, L, 3)` image. Kept separate from the
        plotting calls so an animation can consume frames without a figure.
        """
        return render(self.labels, self.palette(**kwargs))

    def plot_to_axes(self, ax: plt.Axes, **kwargs):
        """Draw the current state on existing axes, for multi-panel figures."""
        ax.imshow(self.to_rgb(**kwargs), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])

    def plot(self, filename: str | None=None, dpi: int=150, **kwargs):
        """
        Plot the current state as a bitmap, saving to `filename` if given and
        showing it otherwise.
        """
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
