"""
Correctness tests for the q-state Potts model in `potts.py`.

Potts has its own set of exact reductions:

- `q = 2` is the Ising model at *twice* the coupling, up to an additive
  constant, since `delta(s, s') = (1 + s s')/2`.
- `q = 3` has the same conditional distribution as the `q = 3` clock model at
  two thirds the coupling -- the affine map recorded in `test_clock.py`, seen
  from the other side.
- The critical point is exact for every q by self-duality.

Reference quantities come from naive loops that touch none of the code under
test, as in `test_model.py`.
"""
import itertools
import math

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")  # noqa: E402

from model import Model
from potts import PottsModel
from test_model import ref_bonds, ref_neighbours


# --------------------------------------------------------------------------
# Reference implementations
# --------------------------------------------------------------------------

def ref_potts_energy(labels, coupling, field, boundary):
    """`-J sum_<ij> delta(s_i, s_j) - sum_i H[s_i]`, by explicit loop."""
    length = labels.shape[0]
    agreeing = sum(
        1 for a, b in ref_bonds(length, boundary) if labels[a] == labels[b]
    )
    bias = sum(
        field[int(labels[i, j])] for i in range(length) for j in range(length)
    )
    return -coupling * agreeing - bias


def ref_conditional(labels, site, q, beta, coupling, field, neighbours):
    weights = []
    for k in range(q):
        energy = field[k] + coupling * sum(
            1 for n in neighbours[site] if int(labels[n]) == k
        )
        weights.append(math.exp(beta * energy))
    total = sum(weights)
    return [w / total for w in weights]


def all_label_states(length, q):
    return np.array(
        list(itertools.product(range(q), repeat=length * length)), dtype=np.uint8
    ).reshape(-1, length, length)


def boltzmann(states, coupling, field, beta, boundary):
    energies = np.array([
        ref_potts_energy(s, coupling, field, boundary) for s in states
    ])
    weights = np.exp(-beta * (energies - energies.min()))
    return weights / weights.sum(), energies


# --------------------------------------------------------------------------
# The exact reductions
# --------------------------------------------------------------------------

@pytest.mark.parametrize("boundary,length", [("open", 5), ("periodic", 6)])
def test_q2_is_ising_at_double_coupling(boundary, length):
    """
    `delta(s, s') = (1 + s s')/2`, so a 2-state Potts model at `J_P` is an
    Ising model at `J_I = J_P/2` shifted by a constant:

        E_Potts = E_Ising - J_I * n_bonds

    The constant depends on the lattice, not the state, so it changes no
    dynamics -- but it does change every energy, which is exactly why it has
    to be in a test rather than in a comment.
    """
    rng = np.random.default_rng(0)
    ising_coupling = 0.8
    for _ in range(4):
        spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=(length, length))

        ising = Model(length, coupling=ising_coupling, field=0.0,
                      boundary=boundary, seed=0)
        ising.spins = spins
        potts = PottsModel(length, q=2, coupling=2 * ising_coupling,
                           boundary=boundary, seed=0)
        potts.labels = ((1 - spins) // 2).astype(np.uint8)

        expected = ising.energy - ising_coupling * potts.lattice.n_bonds
        assert potts.energy == pytest.approx(expected)


@pytest.mark.parametrize("boundary,length", [("open", 5), ("periodic", 6)])
def test_q2_conditional_is_the_ising_heat_bath(boundary, length):
    """
    The additive constant cancels out of the normalised conditional, so a
    2-state Potts model at `2 J_I` has *exactly* the Ising site conditional
    at `J_I`. Machine precision, not statistics.
    """
    rng = np.random.default_rng(1)
    temperature, ising_coupling = 1.7, 0.85
    spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=(length, length))

    ising = Model(length, temperature=temperature, field=0.0,
                  coupling=ising_coupling, boundary=boundary, seed=0)
    ising.spins = spins
    potts = PottsModel(length, q=2, temperature=temperature,
                       coupling=2 * ising_coupling, boundary=boundary, seed=0)
    potts.labels = ((1 - spins) // 2).astype(np.uint8)

    assert np.abs(potts.conditional_probabilities()[0]
                  - ising.heat_bath_probs()).max() < 1e-15


@pytest.mark.parametrize("q", [2, 3, 4, 5, 8])
def test_critical_temperature_is_self_dual(q):
    model = PottsModel(4, q=q, coupling=1.0)
    assert model.critical_temperature == pytest.approx(1 / math.log(1 + math.sqrt(q)))
    assert model.is_first_order == (q >= 5)

    # q = 2 at J_P = 2 J_I must land on the Onsager point of the Ising model
    # it reduces to.
    if q == 2:
        assert PottsModel(4, q=2, coupling=2.0).critical_temperature == pytest.approx(
            Model(4, coupling=1.0).critical_temperature
        )


# --------------------------------------------------------------------------
# Sampler correctness
# --------------------------------------------------------------------------

@pytest.mark.parametrize("boundary,length", [("open", 3), ("periodic", 4)])
def test_evolve_matches_reference_sweep(boundary, length):
    """
    Scripted uniforms, then an independent inverse-CDF over the reference
    conditionals, site by site in the same checkerboard order.
    """
    rng = np.random.default_rng(2)
    q, coupling = 4, 1.0
    temperature = 1.1
    field = np.array([0.3, 0.0, -0.2, 0.1])
    model = PottsModel(length, q=q, temperature=temperature, field=field,
                       coupling=coupling, boundary=boundary, seed=0)
    start = rng.integers(q, size=(length, length)).astype(np.uint8)
    model.labels = start.copy()

    draws = [rng.random((length, length)), rng.random((length, length))]

    class ScriptedRNG:
        def __init__(self):
            self.calls = 0

        def random(self, shape):
            self.calls += 1
            return draws[self.calls - 1]

    model.rng = ScriptedRNG()
    model.evolve()

    neighbours = ref_neighbours(length, boundary)
    expected = start.copy()
    for parity, uniforms in ((0, draws[0]), (1, draws[1])):
        updated = expected.copy()
        for i in range(length):
            for j in range(length):
                if (i + j) % 2 != parity:
                    continue
                conditional = ref_conditional(
                    expected, (i, j), q, 1 / temperature, coupling, field, neighbours
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
def test_matches_exact_enumeration():
    """
    Sampled averages against brute-force enumeration of all 3**9 states of a
    3x3 lattice, with a field that breaks the permutation symmetry -- the
    only check on the sign and indexing of the per-state bias.
    """
    length, q, beta, coupling = 3, 3, 0.7, 1.0
    field = np.array([0.4, 0.0, -0.25])
    states = all_label_states(length, q)
    pi, energies = boltzmann(states, coupling, field, beta, "open")

    populations = np.stack([(states == k).mean(axis=(1, 2)) for k in range(q)], -1)
    exact_order = float(pi @ ((q * populations.max(axis=1) - 1) / (q - 1)))
    exact_energy = float(pi @ energies) / length ** 2

    model = PottsModel(length, q=q, temperature=1 / beta, field=field,
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

    assert order.mean() == pytest.approx(exact_order, abs=0.01)
    assert energy.mean() == pytest.approx(exact_energy, abs=0.02)


# --------------------------------------------------------------------------
# Analytic limits
# --------------------------------------------------------------------------

def test_infinite_temperature_is_uniform_over_states():
    model = PottsModel(64, q=5, temperature=1e12, field=1.0, coupling=1.0, seed=3)
    model.evolve()
    assert np.abs(model.populations - 1 / 5).max() < 5 * math.sqrt(1 / 5 * 4 / 5) / 64


@pytest.mark.parametrize("target", [0, 2, 4])
def test_saturating_field_pins_a_state(target):
    q = 5
    field = np.zeros(q)
    field[target] = 200.0
    model = PottsModel(16, q=q, temperature=1.0, field=field, coupling=1.0, seed=4)
    model.evolve()
    assert np.all(model.labels == target)


def test_zero_coupling_gives_the_single_site_distribution():
    """With J = 0 the sites are independent, so populations follow exp(beta*H_k)."""
    temperature, q = 1.5, 4
    field = np.array([0.9, 0.0, -0.4, 0.2])
    exact = np.exp(field / temperature)
    exact /= exact.sum()

    model = PottsModel(64, q=q, temperature=temperature, field=field,
                       coupling=0.0, seed=5)
    pooled = np.zeros(q)
    for _ in range(100):
        model.evolve()
        pooled += model.populations
    assert np.abs(pooled / 100 - exact).max() < 0.005


@pytest.mark.slow
def test_ordering_across_tc():
    """
    Smoke test that the q = 3 model orders below its self-dual point and does
    not above it. Loose bounds: not a measurement.
    """
    critical = 1 / math.log(1 + math.sqrt(3))

    cold = PottsModel(32, q=3, temperature=0.7 * critical, coupling=1.0,
                      boundary="periodic", seed=6)
    for _ in range(2000):
        cold.evolve()
    assert cold.order_parameter > 0.9

    hot = PottsModel(32, q=3, temperature=1.6 * critical, coupling=1.0,
                     boundary="periodic", seed=6)
    for _ in range(2000):
        hot.evolve()
    assert hot.order_parameter < 0.2


# --------------------------------------------------------------------------
# Observables and plumbing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("boundary,length", [("open", 5), ("periodic", 4)])
def test_energy_matches_bond_loop(boundary, length):
    rng = np.random.default_rng(7)
    for _ in range(5):
        q = int(rng.integers(2, 9))
        coupling = float(rng.uniform(-2, 2))
        field = rng.uniform(-1, 1, size=q)
        model = PottsModel(length, q=q, coupling=coupling, field=field,
                           boundary=boundary, seed=0)
        model.labels = rng.integers(q, size=(length, length)).astype(np.uint8)
        assert model.energy == pytest.approx(
            ref_potts_energy(model.labels, coupling, field, boundary)
        )


@pytest.mark.parametrize("boundary,length", [("open", 5), ("periodic", 6)])
def test_agreeing_bonds_counts_each_bond_once(boundary, length):
    rng = np.random.default_rng(8)
    model = PottsModel(length, q=3, boundary=boundary, seed=0)
    for _ in range(4):
        model.labels = rng.integers(3, size=(length, length)).astype(np.uint8)
        expected = sum(
            1 for a, b in ref_bonds(length, boundary)
            if model.labels[a] == model.labels[b]
        )
        assert model.agreeing_bonds == expected

    model.labels = np.zeros((length, length), dtype=np.uint8)
    assert model.agreeing_bonds == model.lattice.n_bonds


def test_counts_are_the_neighbour_occupancy():
    """
    The one-hot occupancy count is what lets the ordinary stencil serve
    Potts, and it makes open boundaries automatically right: with a zero
    halo, an absent neighbour is in no state at all, so the counts at an edge
    site sum to its coordination number rather than to four.
    """
    rng = np.random.default_rng(9)
    model = PottsModel(5, q=3, boundary="open", seed=0)
    model.labels = rng.integers(3, size=(5, 5)).astype(np.uint8)
    counts = model._counts()

    neighbours = ref_neighbours(5, "open")
    for (i, j), sites in neighbours.items():
        assert counts[:, i, j].sum() == len(sites)
        for k in range(3):
            assert counts[k, i, j] == sum(1 for s in sites if model.labels[s] == k)


def test_order_parameter_limits():
    cold = PottsModel(8, q=4, init="cold", seed=0)
    assert cold.order_parameter == pytest.approx(1.0)
    assert cold.populations[0] == pytest.approx(1.0)

    even = PottsModel(8, q=4, seed=0)
    even.labels = (np.arange(64).reshape(8, 8) % 4).astype(np.uint8)
    assert even.order_parameter == pytest.approx(0.0)


def test_scalar_field_biases_state_zero():
    model = PottsModel(4, q=3, field=0.7, seed=0)
    assert model.H == pytest.approx([0.7, 0.0, 0.0])
    model.H = (0.1, 0.2, 0.3)
    assert model.H == pytest.approx([0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        model.H = (1.0, 2.0)


def test_per_spin_observables():
    model = PottsModel(8, q=3, temperature=1.2, field=0.1, seed=0)
    model.evolve()
    assert model.energy_per_spin == pytest.approx(model.energy / 64)
    assert model.log_weight == pytest.approx(-model.energy / model.T)


def test_state_invariants_and_reproducibility():
    def trajectory(seed):
        model = PottsModel(8, q=5, temperature=1.0, seed=seed)
        states = []
        for _ in range(10):
            model.evolve()
            assert model._check_state()
            states.append(model.labels.copy())
        return np.array(states)

    assert np.array_equal(trajectory(42), trajectory(42))
    assert not np.array_equal(trajectory(42), trajectory(43))


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
        PottsModel(**kwargs)


def test_rendering_and_plot(tmp_path):
    model = PottsModel(8, q=3, seed=0)
    image = model.to_rgb()
    assert image.shape == (8, 8, 3)
    assert np.array_equal(image[0, 0], model.palette()[model.labels[0, 0]])

    target = tmp_path / "potts.png"
    model.plot(filename=str(target))
    assert target.exists() and target.stat().st_size > 0
