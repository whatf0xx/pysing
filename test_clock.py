"""
Correctness tests for the q-state clock model in `clock.py`.

The clock model has an unusually good test story, and this module is mostly
built around it. Three values of q collapse onto models that are already
implemented or already solved:

- `q = 2` is the Ising model *at the same coupling*. Not an approximation, an
  identity: the conditional distributions must agree to machine precision.
- `q = 3` is the 3-state Potts model under an affine map of the energy.
- `q = 4` is two decoupled Ising models at `J/2`.

Everything else here follows the ground rule of `test_model.py`: reference
quantities come from naive Python loops that touch none of the code under
test.
"""
import itertools
import math

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")  # noqa: E402

from clock import ClockModel
from model import Model
from potts import PottsModel
from test_model import ref_bonds, ref_neighbours


# --------------------------------------------------------------------------
# Reference implementations
# --------------------------------------------------------------------------

def ref_angles(labels, q):
    return 2.0 * np.pi * np.asarray(labels, dtype=float) / q


def ref_clock_energy(labels, q, coupling, field, boundary):
    """`-J sum_<ij> cos(dtheta) - H . sum n`, by explicit loop over bonds."""
    length = labels.shape[0]
    bond_term = sum(
        math.cos(2 * math.pi * (int(labels[a]) - int(labels[b])) / q)
        for a, b in ref_bonds(length, boundary)
    )
    field_term = sum(
        field[0] * math.cos(2 * math.pi * int(labels[i, j]) / q)
        + field[1] * math.sin(2 * math.pi * int(labels[i, j]) / q)
        for i in range(length) for j in range(length)
    )
    return -coupling * bond_term - field_term


def ref_conditional(labels, site, q, beta, coupling, field, neighbours):
    """The exact site conditional, as a length-q list of probabilities."""
    weights = []
    for k in range(q):
        angle = 2 * math.pi * k / q
        energy = field[0] * math.cos(angle) + field[1] * math.sin(angle)
        for neighbour in neighbours[site]:
            other = 2 * math.pi * int(labels[neighbour]) / q
            energy += coupling * math.cos(angle - other)
        weights.append(math.exp(beta * energy))
    total = sum(weights)
    return [w / total for w in weights]


def all_label_states(length, q):
    """Every one of the q**(L*L) label configurations."""
    return np.array(
        list(itertools.product(range(q), repeat=length * length)), dtype=np.uint8
    ).reshape(-1, length, length)


def half_sweep_matrix(states, index, sites, q, beta, coupling, field, neighbours):
    """
    Exact transition matrix for resampling `sites` simultaneously. They are
    one sublattice, so conditionally independent given the other, and the
    joint conditional factorises.
    """
    matrix = np.zeros((len(states), len(states)))
    for row, state in enumerate(states):
        conditionals = [
            ref_conditional(state, site, q, beta, coupling, field, neighbours)
            for site in sites
        ]
        target = state.copy()
        for outcome in itertools.product(range(q), repeat=len(sites)):
            weight = 1.0
            for site, conditional, k in zip(sites, conditionals, outcome):
                target[site] = k
                weight *= conditional[k]
            matrix[row, index[target.tobytes()]] += weight
    return matrix


def boltzmann(states, q, beta, coupling, field, boundary):
    energies = np.array([
        ref_clock_energy(s, q, coupling, field, boundary) for s in states
    ])
    weights = np.exp(-beta * (energies - energies.min()))
    return weights / weights.sum(), energies


# --------------------------------------------------------------------------
# The exact checkpoints
# --------------------------------------------------------------------------

@pytest.mark.parametrize("boundary,length", [("open", 5), ("periodic", 6)])
@pytest.mark.parametrize("field", [0.0, 0.4, -0.7])
def test_q2_conditional_is_the_ising_heat_bath(boundary, length, field):
    """
    The strongest test in the suite. At `q = 2` the clock model *is* the
    Ising model at the same coupling: `theta` in `{0, pi}` gives
    `cos(theta_i - theta_j) = s_i s_j` with no rescaling anywhere. So the
    q-way categorical conditional and the `(1 + tanh)/2` conditional are two
    expressions for one number, and must agree to machine precision.

    Label 0 is `theta = 0`, i.e. spin +1; label 1 is `theta = pi`, spin -1.
    """
    rng = np.random.default_rng(7)
    temperature, coupling = 1.6, 0.9

    ising = Model(length, temperature=temperature, field=field,
                  coupling=coupling, boundary=boundary, seed=0)
    ising.spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=(length, length))

    clock = ClockModel(length, q=2, temperature=temperature, field=field,
                       coupling=coupling, boundary=boundary, seed=0)
    clock.labels = ((1 - ising.spins) // 2).astype(np.uint8)

    probabilities = clock.conditional_probabilities()
    assert np.abs(probabilities[0] - ising.heat_bath_probs()).max() < 1e-15
    assert np.abs(probabilities.sum(axis=0) - 1.0).max() < 1e-15


@pytest.mark.parametrize("boundary,length", [("open", 5), ("periodic", 6)])
def test_q2_energy_is_the_ising_energy(boundary, length):
    rng = np.random.default_rng(8)
    for _ in range(4):
        spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=(length, length))
        ising = Model(length, coupling=1.3, field=0.0, boundary=boundary, seed=0)
        ising.spins = spins
        clock = ClockModel(length, q=2, coupling=1.3, boundary=boundary, seed=0)
        clock.labels = ((1 - spins) // 2).astype(np.uint8)
        assert clock.energy == pytest.approx(ising.energy)


@pytest.mark.parametrize("boundary,length", [("open", 5), ("periodic", 6)])
def test_q4_is_two_ising_models_at_half_coupling(boundary, length):
    """
    Rotate the four clock directions by 45 degrees -- which changes no dot
    product -- and each becomes `(s, t)/sqrt(2)` with `s, t = +-1`. Then
    `n_i . n_j = (s_i s_j + t_i t_j)/2`, so a q = 4 clock model at coupling J
    is exactly two independent Ising models at J/2 sharing a lattice.

    A model that looks like it has four coupled states turns out to be two
    Ising models wearing a hat, and it falls out as an exact test rather than
    a remark.
    """
    rng = np.random.default_rng(9)
    coupling = 1.1
    # k = 0, 1, 2, 3 sit at 45, 135, 225, 315 degrees after the rotation.
    sigma_of = np.array([1, -1, -1, 1], dtype=np.int8)
    tau_of = np.array([1, 1, -1, -1], dtype=np.int8)

    for _ in range(4):
        labels = rng.integers(4, size=(length, length)).astype(np.uint8)
        clock = ClockModel(length, q=4, coupling=coupling, boundary=boundary, seed=0)
        clock.labels = labels

        total = 0.0
        for spins in (sigma_of[labels], tau_of[labels]):
            ising = Model(length, coupling=coupling / 2, field=0.0,
                          boundary=boundary, seed=0)
            ising.spins = spins
            total += ising.energy
        assert clock.energy == pytest.approx(total)


@pytest.mark.parametrize("boundary,length", [("open", 5), ("periodic", 6)])
def test_q3_energy_maps_onto_potts(boundary, length):
    """
    At `q = 3` the angle differences are 0 and +-2*pi/3, so `cos` takes only
    the values 1 and -1/2 -- an affine function of the Kronecker delta,
    `cos = (3*delta - 1)/2`. Hence

        E_clock = 1.5 * E_Potts + n_bonds / 2

    at unit coupling. The additive constant is not optional: it is what
    `cos = -1/2` contributes on every disagreeing bond, and dropping it gives
    an energy that is wrong by a lattice-size-dependent amount while still
    ordering states correctly.
    """
    rng = np.random.default_rng(10)
    for _ in range(4):
        labels = rng.integers(3, size=(length, length)).astype(np.uint8)

        clock = ClockModel(length, q=3, coupling=1.0, boundary=boundary, seed=0)
        clock.labels = labels
        potts = PottsModel(length, q=3, coupling=1.0, boundary=boundary, seed=0)
        potts.labels = labels

        expected = 1.5 * potts.energy + clock.lattice.n_bonds / 2
        assert clock.energy == pytest.approx(expected)


@pytest.mark.parametrize("boundary,length", [("open", 5), ("periodic", 6)])
def test_q3_conditional_matches_potts_at_three_halves_coupling(boundary, length):
    """
    The same affine map, applied to the conditional rather than the energy.
    The constant shifts every candidate state equally, so it cancels out of a
    normalised distribution -- leaving the two models with *identical*
    dynamics once the coupling is rescaled by 3/2.
    """
    rng = np.random.default_rng(11)
    labels = rng.integers(3, size=(length, length)).astype(np.uint8)

    clock = ClockModel(length, q=3, temperature=1.4, coupling=1.0,
                       boundary=boundary, seed=0)
    clock.labels = labels
    potts = PottsModel(length, q=3, temperature=1.4, coupling=1.5,
                       boundary=boundary, seed=0)
    potts.labels = labels

    assert np.abs(clock.conditional_probabilities()
                  - potts.conditional_probabilities()).max() < 1e-14


def test_critical_temperatures():
    """The three exactly-known values, and a refusal everywhere else."""
    assert ClockModel(4, q=2, coupling=1.0).critical_temperature == pytest.approx(
        2.0 / math.log(1 + math.sqrt(2))
    )
    assert ClockModel(4, q=3, coupling=1.0).critical_temperature == pytest.approx(
        1.5 / math.log(1 + math.sqrt(3))
    )
    assert ClockModel(4, q=4, coupling=1.0).critical_temperature == pytest.approx(
        1.0 / math.log(1 + math.sqrt(2))
    )
    # q = 2 must agree with the Ising model it *is*.
    assert (ClockModel(4, q=2, coupling=1.0).critical_temperature
            == pytest.approx(Model(4, coupling=1.0).critical_temperature))

    for q in (5, 6, 12):
        with pytest.raises(ValueError):
            ClockModel(4, q=q).critical_temperature


# --------------------------------------------------------------------------
# Sampler correctness
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sweep_2x2_q3():
    """Exact half-sweep matrices and the Boltzmann law for a 2x2 open q=3 lattice."""
    length, q, beta, coupling, field = 2, 3, 0.8, 1.0, (0.3, -0.2)
    states = all_label_states(length, q)
    index = {s.tobytes(): k for k, s in enumerate(states)}
    neighbours = ref_neighbours(length, "open")

    black = [(0, 0), (1, 1)]
    white = [(0, 1), (1, 0)]
    p_black = half_sweep_matrix(
        states, index, black, q, beta, coupling, field, neighbours
    )
    p_white = half_sweep_matrix(
        states, index, white, q, beta, coupling, field, neighbours
    )
    pi, _ = boltzmann(states, q, beta, coupling, field, "open")
    return pi, p_black, p_white


def test_half_sweep_is_reversible(sweep_2x2_q3):
    """
    Each sublattice update satisfies detailed balance with respect to the
    Boltzmann distribution -- the generalisation of the Ising suite's
    strongest structural test to a q-state alphabet. Exact: no sampling, no
    statistical tolerance.
    """
    pi, p_black, p_white = sweep_2x2_q3
    for matrix in (p_black, p_white):
        assert np.allclose(matrix.sum(axis=1), 1.0)
        flux = pi[:, None] * matrix
        assert np.allclose(flux, flux.T, atol=1e-14)


def test_full_sweep_is_stationary(sweep_2x2_q3):
    """
    The composed sweep leaves the Boltzmann distribution invariant but is not
    itself reversible. The second assertion pins that distinction down:
    asserting detailed balance on the full sweep would be wrong.
    """
    pi, p_black, p_white = sweep_2x2_q3
    full = p_black @ p_white
    assert np.allclose(pi @ full, pi, atol=1e-14)

    flux = pi[:, None] * full
    assert not np.allclose(flux, flux.T, atol=1e-14)


@pytest.mark.parametrize("boundary,length", [("open", 3), ("periodic", 4)])
def test_evolve_matches_reference_sweep(boundary, length):
    """
    `ClockModel.evolve` implements the checkerboard sweep analysed above.
    The generator is replaced with scripted uniforms so the update is
    deterministic and can be compared site by site against an independent
    inverse-CDF over the reference conditionals.
    """
    rng = np.random.default_rng(12)
    q, beta, coupling, field = 5, 1 / 1.3, 1.0, (0.25, 0.1)
    model = ClockModel(length, q=q, temperature=1.3, field=field,
                       coupling=coupling, boundary=boundary, seed=0)
    start = rng.integers(q, size=(length, length)).astype(np.uint8)
    model.labels = start.copy()

    draws = [rng.random((length, length)), rng.random((length, length))]

    class ScriptedRNG:
        def __init__(self):
            self.calls = 0

        def random(self, shape):
            assert shape == (length, length)
            self.calls += 1
            return draws[self.calls - 1]

    model.rng = ScriptedRNG()
    model.evolve()

    neighbours = ref_neighbours(length, boundary)
    expected = start.copy()
    for mask_parity, uniforms in ((0, draws[0]), (1, draws[1])):
        updated = expected.copy()
        for i in range(length):
            for j in range(length):
                if (i + j) % 2 != mask_parity:
                    continue
                conditional = ref_conditional(
                    expected, (i, j), q, beta, coupling, field, neighbours
                )
                total, chosen = 0.0, q - 1
                for k in range(q):
                    total += conditional[k]
                    if uniforms[i, j] < total:
                        chosen = k
                        break
                updated[i, j] = chosen
        expected = updated

    assert model.rng.calls == 2
    assert np.array_equal(model.labels, expected)


@pytest.mark.slow
@pytest.mark.parametrize("field", [(0.0, 0.0), (0.35, 0.15)])
def test_matches_exact_enumeration(field):
    """
    Sampled averages against brute-force enumeration of all 3**9 states of a
    3x3 lattice. The end-to-end backstop: it is sensitive to the sign of the
    field term, to the coupling convention, and to any residual bias in the
    update schedule.
    """
    length, q, beta, coupling = 3, 3, 0.55, 1.0
    states = all_label_states(length, q)
    pi, energies = boltzmann(states, q, beta, coupling, field, "open")

    angles = 2 * np.pi * states.astype(float) / q
    abs_m = np.hypot(np.cos(angles).sum(axis=(1, 2)),
                     np.sin(angles).sum(axis=(1, 2))) / length ** 2
    exact_abs_m = float(pi @ abs_m)
    exact_energy = float(pi @ energies) / length ** 2

    model = ClockModel(length, q=q, temperature=1 / beta, field=field,
                       coupling=coupling, boundary="open", seed=20260728)
    for _ in range(500):
        model.evolve()

    samples = 20000
    order = np.empty(samples)
    energy = np.empty(samples)
    for k in range(samples):
        model.evolve()
        order[k] = model.order_parameter
        energy[k] = model.energy_per_spin

    assert order.mean() == pytest.approx(exact_abs_m, abs=0.01)
    assert energy.mean() == pytest.approx(exact_energy, abs=0.02)


# --------------------------------------------------------------------------
# Analytic limits
# --------------------------------------------------------------------------

def test_infinite_temperature_is_uniform_over_states():
    """
    As `T -> infinity` every state is equally likely, so the populations are
    flat to within counting noise and the order parameter is O(1/L).
    """
    model = ClockModel(64, q=6, temperature=1e12, field=(1.0, 1.0),
                       coupling=1.0, seed=13)
    model.evolve()
    assert np.abs(model.populations - 1 / 6).max() < 5 * math.sqrt(1 / 6 * 5 / 6) / 64
    assert model.order_parameter < 5.0 / 64


@pytest.mark.parametrize("q,target", [(6, 0), (6, 2), (8, 5), (3, 1)])
def test_saturating_field_pins_the_nearest_state(q, target):
    """
    A field this strong beats any neighbour configuration, so a single sweep
    puts every site in the state closest in angle to the field direction.
    """
    angle = 2 * np.pi * target / q
    field = 200.0 * np.array([np.cos(angle), np.sin(angle)])
    model = ClockModel(16, q=q, temperature=1.0, field=field, coupling=1.0, seed=14)
    model.evolve()
    assert np.all(model.labels == target)


def test_zero_coupling_gives_the_single_site_distribution():
    """
    With `J = 0` the sites are independent, so the populations must match
    `exp(beta * H . n_k)` normalised -- a closed-form check of the field term
    with no lattice physics in the way.
    """
    temperature, q = 1.5, 6
    field = np.array([0.8, -0.3])
    angles = 2 * np.pi * np.arange(q) / q
    exact = np.exp((field[0] * np.cos(angles) + field[1] * np.sin(angles))
                   / temperature)
    exact /= exact.sum()

    model = ClockModel(64, q=q, temperature=temperature, field=field,
                       coupling=0.0, seed=15)
    pooled = np.zeros(q)
    for _ in range(100):
        model.evolve()
        pooled += model.populations
    assert np.abs(pooled / 100 - exact).max() < 0.005


@pytest.mark.slow
def test_ordering_below_tc_at_q2():
    """
    The q = 2 model must order at the Ising critical temperature, since it is
    the Ising model. Loose bounds -- this is a smoke test, not a measurement.
    """
    critical = 2.0 / math.log(1 + math.sqrt(2))
    cold = ClockModel(32, q=2, temperature=0.7 * critical, coupling=1.0,
                      boundary="periodic", seed=1)
    for _ in range(2000):
        cold.evolve()
    assert cold.order_parameter > 0.9

    hot = ClockModel(32, q=2, temperature=2.0 * critical, coupling=1.0,
                     boundary="periodic", seed=1)
    for _ in range(2000):
        hot.evolve()
    assert hot.order_parameter < 0.2


# --------------------------------------------------------------------------
# Observables and plumbing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("boundary,length", [("open", 5), ("periodic", 4)])
def test_energy_matches_bond_loop(boundary, length):
    rng = np.random.default_rng(16)
    for _ in range(5):
        q = int(rng.integers(2, 9))
        coupling = float(rng.uniform(-2, 2))
        field = rng.uniform(-1, 1, size=2)
        model = ClockModel(length, q=q, coupling=coupling, field=field,
                           boundary=boundary, seed=0)
        model.labels = rng.integers(q, size=(length, length)).astype(np.uint8)
        assert model.energy == pytest.approx(
            ref_clock_energy(model.labels, q, coupling, field, boundary)
        )


def test_order_parameter_limits():
    cold = ClockModel(8, q=6, init="cold", seed=0)
    assert cold.order_parameter == pytest.approx(1.0)
    assert cold.populations[0] == pytest.approx(1.0)
    # A cold start sits on state 0, so the magnetisation points along +x.
    assert cold.magnetisation_per_spin == pytest.approx([1.0, 0.0])
    assert cold.symmetry_breaking == pytest.approx(1.0)


def test_symmetry_breaking_separates_locked_from_free_directions():
    """
    `symmetry_breaking` must be +1 when the magnetisation lies on an allowed
    clock direction and -1 when it lies exactly between two of them. This is
    the observable that tells the locked phase from the intermediate one, so
    its two extremes are worth pinning down.
    """
    model = ClockModel(4, q=8, seed=0)
    model.labels = np.zeros((4, 4), dtype=np.uint8)
    assert model.symmetry_breaking == pytest.approx(1.0)

    # Half the lattice in state 0 and half in state 1 points the
    # magnetisation exactly between two allowed directions.
    model.labels[:2] = 1
    assert model.symmetry_breaking == pytest.approx(-1.0)


def test_phase_field_width_one_is_the_identity():
    model = ClockModel(8, q=4, seed=0)
    model.labels = (np.arange(64).reshape(8, 8) % 4).astype(np.uint8)
    angle, coherence = model.phase_field(1)
    assert np.allclose(angle, model.angles)
    assert np.allclose(coherence, 1.0)


@pytest.mark.parametrize("boundary,length", [("open", 8), ("periodic", 8)])
@pytest.mark.parametrize("width", [2, 3, 5])
def test_phase_field_against_explicit_window(boundary, length, width):
    """
    The separable box average, against a naive loop over the window with the
    edge handled by hand: wrapping for periodic boundaries, clamping to the
    edge for open ones.
    """
    rng = np.random.default_rng(20)
    model = ClockModel(length, q=7, boundary=boundary, seed=0)
    model.labels = rng.integers(7, size=(length, length)).astype(np.uint8)
    vectors = model._vectors()

    half = width // 2
    expected = np.zeros((2, length, length))
    for i in range(length):
        for j in range(length):
            for di in range(-half, width - half):
                for dj in range(-half, width - half):
                    if boundary == "periodic":
                        ii, jj = (i + di) % length, (j + dj) % length
                    else:
                        ii = min(max(i + di, 0), length - 1)
                        jj = min(max(j + dj, 0), length - 1)
                    expected[:, i, j] += vectors[:, ii, jj]
    expected /= width ** 2

    angle, coherence = model.phase_field(width)
    assert np.allclose(angle, np.arctan2(expected[1], expected[0]) % (2 * np.pi))
    assert np.allclose(coherence, np.hypot(expected[0], expected[1]))


def test_phase_field_coherence_reports_incoherence():
    """
    The property that stops this inventing structure: a uniform lattice is
    fully coherent, and a state whose neighbours cancel is not coherent at
    all, whatever angle the arctangent happens to return.
    """
    model = ClockModel(8, q=4, seed=0)
    model.labels = np.full((8, 8), 3, dtype=np.uint8)
    angle, coherence = model.phase_field(3)
    assert np.allclose(coherence, 1.0)
    assert np.allclose(angle, 3 * np.pi / 2)

    # States 0 and 2 are back to back at q = 4, so a 2-wide window over
    # alternating columns cancels exactly -- everywhere, once the lattice
    # wraps. With open boundaries the edge column clamps onto itself instead
    # and stays coherent, which is correct and is why this uses periodic.
    wrapped = ClockModel(8, q=4, boundary="periodic", seed=0)
    wrapped.labels = np.tile(np.array([[0, 2]], dtype=np.uint8), (8, 4))
    _, coherence = wrapped.phase_field(2)
    assert np.allclose(coherence, 0.0, atol=1e-15)


def test_phase_field_rejects_bad_widths():
    model = ClockModel(8, q=6, seed=0)
    for width in (0, -1, 9):
        with pytest.raises(ValueError):
            model.phase_field(width)
    assert model.phase_field(5)[0].shape == (8, 8)


def test_angles_and_vectors_agree():
    model = ClockModel(6, q=7, seed=0)
    for _ in range(3):
        model.evolve()
    assert np.allclose(model.angles, 2 * np.pi * model.labels / 7)
    assert np.allclose(model._vectors()[0], np.cos(model.angles))
    assert np.allclose(model._vectors()[1], np.sin(model.angles))


def test_per_spin_observables():
    model = ClockModel(8, q=5, temperature=1.2, field=(0.1, 0.0), seed=0)
    model.evolve()
    assert model.magnetisation_per_spin == pytest.approx(model.magnetisation / 64)
    assert model.energy_per_spin == pytest.approx(model.energy / 64)
    assert model.log_weight == pytest.approx(-model.energy / model.T)


def test_scalar_field_means_a_field_along_state_zero():
    model = ClockModel(4, q=6, field=0.7, seed=0)
    assert model.H == pytest.approx([0.7, 0.0])
    model.H = (0.0, -0.2)
    assert model.H == pytest.approx([0.0, -0.2])
    with pytest.raises(ValueError):
        model.H = (1.0, 2.0, 3.0)


def test_parameters_are_mutable_midrun():
    model = ClockModel(4, q=6, temperature=2.0, field=1.0, coupling=1.0, seed=0)
    model.H = (0.0, 0.0)
    model.T = 4.0
    assert model.beta == pytest.approx(0.25)
    assert model.beta_J == pytest.approx(0.25)
    model.evolve()
    assert model._check_state()


def test_state_invariants_and_reproducibility():
    def trajectory(seed):
        model = ClockModel(8, q=7, temperature=1.0, field=(0.1, 0.0), seed=seed)
        states = []
        for _ in range(10):
            model.evolve()
            assert model._check_state()
            states.append(model.labels.copy())
        return np.array(states)

    assert np.array_equal(trajectory(42), trajectory(42))
    assert not np.array_equal(trajectory(42), trajectory(43))


def test_cold_and_hot_starts_differ():
    cold = ClockModel(8, q=6, init="cold", seed=0)
    hot = ClockModel(8, q=6, init="hot", seed=0)
    assert np.all(cold.labels == 0)
    assert hot.labels.max() > 0


@pytest.mark.parametrize("kwargs", [
    {"lattice_length": 5, "boundary": "periodic"},
    {"lattice_length": 4, "boundary": "toroidal"},
    {"lattice_length": 4, "temperature": 0.0},
    {"lattice_length": 0},
    {"lattice_length": 4, "q": 1},
    {"lattice_length": 4, "q": 256},
    {"lattice_length": 4, "init": "lukewarm"},
])
def test_constructor_validation(kwargs):
    with pytest.raises(ValueError):
        ClockModel(**kwargs)


def test_rendering():
    model = ClockModel(8, q=6, seed=0)
    image = model.to_rgb()
    assert image.shape == (8, 8, 3)
    assert image.min() >= 0.0 and image.max() <= 1.0
    assert np.array_equal(image[0, 0], model.palette()[model.labels[0, 0]])


def test_plot_writes_file(tmp_path):
    model = ClockModel(8, q=6, seed=0)
    target = tmp_path / "clock.png"
    model.plot(filename=str(target))
    assert target.exists() and target.stat().st_size > 0
