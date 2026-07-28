"""
Correctness tests for the Ising model in `model.py`.

Ground rule: every reference quantity in this module is computed independently
of `model.py`, with naive Python loops over explicit indices. A test that
reuses `Model._neighbour_sum` to check `Model.energy` proves nothing.

Run the fast suite with `pytest -m "not slow"`, everything with `pytest`.
"""
import math
import warnings

import matplotlib
matplotlib.use("Agg")  # noqa: E402  - must precede the pyplot import in model

import numpy as np
import pytest

from model import Model


# --------------------------------------------------------------------------
# Reference implementations (deliberately naive; no numpy tricks, no model.py)
# --------------------------------------------------------------------------

def ref_neighbours(length, boundary):
    """Map every site to the list of its nearest-neighbour sites."""
    neighbours = {}
    for i in range(length):
        for j in range(length):
            sites = []
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ii, jj = i + di, j + dj
                if boundary == "periodic":
                    sites.append(((ii + length) % length, (jj + length) % length))
                elif 0 <= ii < length and 0 <= jj < length:
                    sites.append((ii, jj))
            neighbours[(i, j)] = sites
    return neighbours


def ref_bonds(length, boundary):
    """Every nearest-neighbour bond, counted exactly once."""
    bonds = []
    for i in range(length):
        for j in range(length):
            if boundary == "periodic" or j + 1 < length:
                bonds.append(((i, j), (i, (j + 1) % length)))
            if boundary == "periodic" or i + 1 < length:
                bonds.append(((i, j), ((i + 1) % length, j)))
    return bonds


def ref_neighbour_sum(state, boundary):
    length = state.shape[0]
    out = np.zeros((length, length), dtype=int)
    for site, sites in ref_neighbours(length, boundary).items():
        out[site] = sum(int(state[s]) for s in sites)
    return out


def ref_energy(state, coupling, field, boundary):
    bond_term = sum(
        int(state[a]) * int(state[b]) for a, b in ref_bonds(state.shape[0], boundary)
    )
    return -coupling * bond_term - field * int(state.sum())


def ref_black_mask(length):
    mask = np.zeros((length, length), dtype=bool)
    for i in range(length):
        for j in range(length):
            mask[i, j] = (i + j) % 2 == 0
    return mask


def ref_sweep(state, u_black, u_white, beta, coupling, field, boundary):
    """One checkerboard sweep, black sublattice first, driven by given uniforms."""
    spins = state.copy()
    black = ref_black_mask(state.shape[0])
    for mask, uniforms in ((black, u_black), (~black, u_white)):
        probs = 0.5 * (1.0 + np.tanh(
            beta * (field + coupling * ref_neighbour_sum(spins, boundary))
        ))
        drawn = np.where(uniforms < probs, 1, -1)
        spins = np.where(mask, drawn, spins).astype(np.int8)
    return spins


def all_states(length):
    """Every one of the 2**(L*L) configurations, as an (n_states, L, L) array."""
    n_spins = length * length
    codes = np.arange(2 ** n_spins, dtype=np.int64)
    bits = (codes[:, None] >> np.arange(n_spins)) & 1
    return (2 * bits - 1).astype(np.int8).reshape(-1, length, length)


def boltzmann(states, beta, coupling, field, boundary):
    energies = np.array([ref_energy(s, coupling, field, boundary) for s in states])
    weights = np.exp(-beta * (energies - energies.min()))
    return weights / weights.sum(), energies


def half_sweep_matrix(states, index, sites, beta, coupling, field, neighbours):
    """
    Exact transition matrix for resampling `sites` simultaneously.

    The sites of one sublattice are conditionally independent given the other,
    so the joint conditional factorises and each row is enumerated exactly.
    """
    matrix = np.zeros((len(states), len(states)))
    for k, state in enumerate(states):
        probs = [
            0.5 * (1.0 + math.tanh(beta * (
                field + coupling * sum(int(state[s]) for s in neighbours[site])
            )))
            for site in sites
        ]
        target = state.copy()
        for outcome in range(1 << len(sites)):
            weight = 1.0
            for bit, site in enumerate(sites):
                up = (outcome >> bit) & 1
                target[site] = 1 if up else -1
                weight *= probs[bit] if up else 1.0 - probs[bit]
            matrix[k, index[target.tobytes()]] = weight
    return matrix


@pytest.fixture(scope="module")
def sweep_3x3():
    """Exact half-sweep matrices and the Boltzmann law for a 3x3 open lattice."""
    length, beta, coupling, field = 3, 0.6, 1.0, 0.3
    states = all_states(length)
    index = {s.tobytes(): k for k, s in enumerate(states)}
    neighbours = ref_neighbours(length, "open")
    black = ref_black_mask(length)

    sites = [(i, j) for i in range(length) for j in range(length)]
    p_black = half_sweep_matrix(
        states, index, [s for s in sites if black[s]], beta, coupling, field, neighbours
    )
    p_white = half_sweep_matrix(
        states, index, [s for s in sites if not black[s]], beta, coupling, field, neighbours
    )
    pi, _ = boltzmann(states, beta, coupling, field, "open")
    return pi, p_black, p_white


# --------------------------------------------------------------------------
# 10a. Sampler correctness
# --------------------------------------------------------------------------

def test_half_sweep_is_reversible(sweep_3x3):
    """
    Each sublattice update satisfies detailed balance with respect to the
    Boltzmann distribution. This is the property parallel ("Little") dynamics
    violates, and it is checked exactly -- no sampling, no statistical
    tolerance.
    """
    pi, p_black, p_white = sweep_3x3
    for matrix in (p_black, p_white):
        assert np.allclose(matrix.sum(axis=1), 1.0)
        flux = pi[:, None] * matrix
        assert np.allclose(flux, flux.T, atol=1e-14)


def test_full_sweep_is_stationary(sweep_3x3):
    """
    The full sweep leaves the Boltzmann distribution invariant.

    Composing two reversible kernels gives a stationary kernel but not, in
    general, a reversible one -- so stationarity is the correct claim here.
    The second assertion pins that distinction down: asserting detailed
    balance on the full sweep would be wrong, and would fail.
    """
    pi, p_black, p_white = sweep_3x3
    full = p_black @ p_white
    assert np.allclose(pi @ full, pi, atol=1e-14)

    flux = pi[:, None] * full
    assert not np.allclose(flux, flux.T, atol=1e-14)


@pytest.mark.parametrize("boundary,length", [("open", 3), ("periodic", 4)])
def test_evolve_matches_reference_sweep(boundary, length):
    """
    `Model.evolve` implements the checkerboard sweep the tests above analyse.

    The model's generator is replaced with scripted uniforms so the update is
    deterministic and can be compared element by element against an
    independent implementation.
    """
    rng = np.random.default_rng(4)
    model = Model(length, temperature=1.7, field=0.3, coupling=1.0,
                  boundary=boundary, seed=0)
    start = rng.choice(np.array([-1, 1], dtype=np.int8), size=(length, length))
    model.spins = start.copy()

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

    expected = ref_sweep(start, draws[0], draws[1], 1 / 1.7, 1.0, 0.3, boundary)
    assert model.rng.calls == 2
    assert np.array_equal(model.spins, expected)


@pytest.mark.slow
@pytest.mark.parametrize("boundary", ["open", "periodic"])
@pytest.mark.parametrize("beta,field", [(0.35, 0.0), (0.35, 0.2), (0.2, 0.0)])
def test_matches_exact_enumeration(boundary, beta, field):
    """
    Sampled averages agree with brute-force enumeration of all 2**16 states of
    a 4x4 lattice. This is the end-to-end backstop: the original synchronous
    update failed it by 16 % in the order parameter.

    The field != 0 cases break the s -> -s symmetry, and are the only check on
    the sign of the field term in the sampler.
    """
    length = 4
    states = all_states(length)
    pi, energies = boltzmann(states, beta, 1.0, field, boundary)
    exact_abs_m = float(pi @ np.abs(states.sum(axis=(1, 2)))) / length ** 2
    exact_energy = float(pi @ energies) / length ** 2

    model = Model(length, temperature=1 / beta, field=field, coupling=1.0,
                  boundary=boundary, seed=20240728)
    for _ in range(1000):
        model.evolve()

    n_samples = 30000
    abs_m = np.empty(n_samples)
    energy = np.empty(n_samples)
    for k in range(n_samples):
        model.evolve()
        abs_m[k] = abs(model.magnetisation_per_spin)
        energy[k] = model.energy_per_spin

    # Tolerances are five times the scatter observed over eight seeds in a
    # pilot run (0.003 in |m|, 0.007 in the energy). That scatter is wider than
    # the naive sqrt(N) error because successive sweeps are autocorrelated, so
    # it has to be measured rather than derived. The bias these tests exist to
    # catch was 0.067 in |m| -- four times the tolerance below.
    assert abs_m.mean() == pytest.approx(exact_abs_m, abs=0.015)
    assert energy.mean() == pytest.approx(exact_energy, abs=0.035)


# --------------------------------------------------------------------------
# 10b. Lattice geometry
# --------------------------------------------------------------------------

def test_coordination_numbers():
    """On an all-up lattice the neighbour sum is the coordination number."""
    model = Model(5, boundary="open", seed=0)
    model.spins = np.ones((5, 5), dtype=np.int8)
    expected = np.array([
        [2, 3, 3, 3, 2],
        [3, 4, 4, 4, 3],
        [3, 4, 4, 4, 3],
        [3, 4, 4, 4, 3],
        [2, 3, 3, 3, 2],
    ])
    assert np.array_equal(model._neighbour_sum(), expected)

    periodic = Model(4, boundary="periodic", seed=0)
    periodic.spins = np.ones((4, 4), dtype=np.int8)
    assert np.array_equal(periodic._neighbour_sum(), np.full((4, 4), 4))


@pytest.mark.parametrize("boundary,length", [("open", 5), ("open", 2), ("periodic", 6)])
def test_neighbour_sum_against_reference(boundary, length):
    rng = np.random.default_rng(11)
    model = Model(length, boundary=boundary, seed=0)
    for _ in range(5):
        state = rng.choice(np.array([-1, 1], dtype=np.int8), size=(length, length))
        model.spins = state
        assert np.array_equal(model._neighbour_sum(), ref_neighbour_sum(state, boundary))


def test_open_bc_topology_unchanged():
    """
    Regression against the pre-refactor implementation.

    These neighbour sums were captured by running the original
    `Model.define_nn_pairs` / `Spin.nn_sum` on the state below, before that
    code was deleted. If the rewrite silently changed the open-boundary
    geometry, this is what catches it.
    """
    state = np.array([
        [1, -1, 1, -1, -1],
        [1, 1, 1, 1, -1],
        [1, -1, 1, 1, -1],
        [-1, -1, 1, 1, 1],
        [1, -1, 1, 1, 1],
    ], dtype=np.int8)
    golden = np.array([
        [0, 3, -1, 1, -2],
        [3, 0, 4, 0, -1],
        [-1, 2, 2, 2, 1],
        [1, -2, 2, 4, 1],
        [-2, 1, 1, 3, 2],
    ])
    model = Model(5, boundary="open", seed=0)
    model.spins = state
    assert np.array_equal(model._neighbour_sum(), golden)


@pytest.mark.parametrize("boundary,length", [("open", 5), ("open", 4), ("periodic", 6)])
def test_sublattice_partition(boundary, length):
    """
    The two masks tile the lattice, and no two same-colour sites are
    neighbours. This is the assumption the checkerboard update rests on -- and
    the one that fails for odd lattice lengths with periodic boundaries, which
    is why the constructor forbids that combination.
    """
    model = Model(length, boundary=boundary, seed=0)
    assert np.array_equal(model._white, ~model._black)
    assert (model._black.sum() + model._white.sum()) == length ** 2

    for site, sites in ref_neighbours(length, boundary).items():
        for neighbour in sites:
            assert model._black[site] != model._black[neighbour]


# --------------------------------------------------------------------------
# 10c. Analytic limits
# --------------------------------------------------------------------------

def test_zero_coupling_gives_tanh():
    """
    With J = 0 the sites are independent, so <m> = tanh(beta*H) exactly. A
    closed-form check of the field term and of the (1 + tanh)/2 normalisation,
    with no lattice physics in the way.
    """
    temperature, field = 2.0, 0.5
    model = Model(64, temperature=temperature, field=field, coupling=0.0, seed=3)
    samples = []
    for _ in range(200):
        model.evolve()
        samples.append(model.magnetisation_per_spin)
    assert np.mean(samples) == pytest.approx(math.tanh(field / temperature), abs=0.01)


@pytest.mark.parametrize("field", [50.0, -50.0])
def test_saturating_field(field):
    """A field this strong aligns every spin within a single sweep."""
    model = Model(16, temperature=1.0, field=field, coupling=1.0, seed=5)
    model.evolve()
    assert np.all(model.spins == np.sign(field))


def test_infinite_temperature_is_unbiased():
    """
    As T -> infinity every site is +1 with probability 1/2, so the
    magnetisation per spin is O(N**-0.5). Five standard errors: the false
    failure rate is below one in a million.
    """
    length = 64
    model = Model(length, temperature=1e12, field=1.0, coupling=1.0, seed=7)
    model.evolve()
    assert abs(model.magnetisation_per_spin) < 5.0 / length


@pytest.mark.slow
def test_ordering_below_tc():
    """
    Smoke test that the model does Ising things: it orders below T_c and does
    not above it. Bounds are loose -- this is not a measurement of anything.
    """
    critical = 2.0 / math.log(1.0 + math.sqrt(2.0))

    cold = Model(32, temperature=0.7 * critical, coupling=1.0,
                 boundary="periodic", seed=1)
    for _ in range(2000):
        cold.evolve()
    assert abs(cold.magnetisation_per_spin) > 0.9

    hot = Model(32, temperature=2.0 * critical, coupling=1.0,
                boundary="periodic", seed=1)
    for _ in range(2000):
        hot.evolve()
    assert abs(hot.magnetisation_per_spin) < 0.2


# --------------------------------------------------------------------------
# 10d. Observables and plumbing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("boundary,length", [("open", 5), ("periodic", 4)])
def test_energy_matches_bond_loop(boundary, length):
    rng = np.random.default_rng(13)
    for _ in range(5):
        field, coupling = rng.uniform(-1, 1), rng.uniform(-2, 2)
        model = Model(length, temperature=1.5, field=field, coupling=coupling,
                      boundary=boundary, seed=0)
        state = rng.choice(np.array([-1, 1], dtype=np.int8), size=(length, length))
        model.spins = state
        assert model.energy == pytest.approx(ref_energy(state, coupling, field, boundary))


def test_heat_bath_probs_against_reference():
    model = Model(5, temperature=1.3, field=0.4, coupling=0.9, seed=2)
    expected = 0.5 * (1.0 + np.tanh(
        (0.4 + 0.9 * ref_neighbour_sum(model.spins, "open")) / 1.3
    ))
    assert np.allclose(model.heat_bath_probs(), expected)


def test_log_weight_finite_at_large_lattice():
    """
    Regression for the old `z_prob`, which returned exp(-beta*E) directly and
    overflowed to infinity above roughly a 20x20 lattice.
    """
    model = Model(100, temperature=1.0, coupling=1.0, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        weight = model.log_weight
    assert math.isfinite(weight)
    assert weight == pytest.approx(-model.energy / model.T)


def test_per_spin_observables():
    model = Model(8, temperature=2.0, field=0.1, coupling=1.0, seed=0)
    model.evolve()
    assert model.magnetisation_per_spin == pytest.approx(model.magnetisation / 64)
    assert model.energy_per_spin == pytest.approx(model.energy / 64)


def test_reduced_temperature():
    critical = 2.0 / math.log(1.0 + math.sqrt(2.0))
    model = Model(4, temperature=critical, coupling=1.0, seed=0)
    assert model.critical_temperature == pytest.approx(critical)
    assert model.reduced_temperature == pytest.approx(1.0)

    model.T = 2 * critical
    assert model.reduced_temperature == pytest.approx(2.0)

    model.J = 0.0
    with pytest.raises(ValueError):
        model.reduced_temperature


def test_parameters_are_mutable_midrun():
    """
    `run.py` switches the field off part way through, so nothing may cache
    beta, beta_H or beta_J at construction time.
    """
    model = Model(4, temperature=2.0, field=1.0, coupling=1.0, seed=0)
    assert model.beta_H == pytest.approx(0.5)

    model.H = 0.0
    assert model.beta_H == 0.0

    model.T = 4.0
    assert model.beta == pytest.approx(0.25)
    assert model.beta_J == pytest.approx(0.25)

    model.evolve()
    assert model._check_state()


def test_state_invariants():
    model = Model(9, temperature=1.5, field=0.2, coupling=1.0, seed=0)
    for _ in range(50):
        model.evolve()
    assert model._check_state()


def test_reproducible_with_seed():
    def trajectory(seed):
        model = Model(8, temperature=2.0, field=0.1, coupling=1.0, seed=seed)
        states = []
        for _ in range(10):
            model.evolve()
            states.append(model.spins.copy())
        return np.array(states)

    assert np.array_equal(trajectory(42), trajectory(42))
    assert not np.array_equal(trajectory(42), trajectory(43))


@pytest.mark.parametrize("kwargs", [
    {"lattice_length": 5, "boundary": "periodic"},    # odd length, not bipartite
    {"lattice_length": 2, "boundary": "periodic"},    # bonds double counted
    {"lattice_length": 4, "boundary": "toroidal"},    # unknown boundary
    {"lattice_length": 4, "temperature": 0.0},        # beta = infinity
    {"lattice_length": 4, "temperature": -1.0},
    {"lattice_length": 0},                            # empty lattice
])
def test_constructor_validation(kwargs):
    with pytest.raises(ValueError):
        Model(**kwargs)


def test_plot_writes_file(tmp_path):
    model = Model(8, seed=0)
    target = tmp_path / "state.png"
    model.plot(filename=str(target))
    assert target.exists() and target.stat().st_size > 0
