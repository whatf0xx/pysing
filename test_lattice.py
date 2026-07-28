"""
Correctness tests for the square-lattice geometry in `lattice.py`.

Same ground rule as `test_model.py`: every reference quantity here is
computed independently of the code under test, either with naive Python loops
over explicit indices or from values captured before the code existed.

`ref_neighbours` and friends are imported from `test_model` rather than
duplicated -- they are naive loop implementations that touch neither
`lattice.py` nor `model.py`, so reusing them keeps the independence the rule
is about.
"""
import numpy as np
import pytest

from lattice import Lattice
from test_model import ref_neighbours, ref_neighbour_sum, ref_black_mask


# --------------------------------------------------------------------------
# Golden data: `Model._neighbour_sum` output captured *before* the geometry
# was extracted into `Lattice`. Without this the test below would degenerate
# into checking the new code against itself.
# --------------------------------------------------------------------------

GOLDEN = {
    ("open", 5): (
        np.array([
            [1, 1, 1, -1, 1],
            [-1, 1, 1, -1, -1],
            [1, -1, 1, 1, 1],
            [-1, 1, -1, 1, -1],
            [-1, 1, 1, 1, 1],
        ], dtype=np.int8),
        np.array([
            [0, 3, 1, 1, -2],
            [3, 0, 2, 0, 1],
            [-3, 4, 0, 2, -1],
            [1, -2, 4, 0, 3],
            [0, 1, 1, 3, 0],
        ]),
    ),
    ("open", 4): (
        np.array([
            [1, 1, 1, 1],
            [-1, -1, 1, -1],
            [1, -1, -1, -1],
            [1, -1, 1, 1],
        ], dtype=np.int8),
        np.array([
            [0, 1, 3, 0],
            [1, 0, -2, 1],
            [-1, -2, 0, -1],
            [0, 1, -1, 0],
        ]),
    ),
    ("periodic", 6): (
        np.array([
            [1, 1, -1, 1, 1, 1],
            [-1, 1, -1, -1, -1, 1],
            [1, -1, 1, -1, 1, -1],
            [1, 1, 1, -1, -1, 1],
            [1, -1, 1, 1, 1, 1],
            [1, 1, -1, 1, -1, 1],
        ], dtype=np.int8),
        np.array([
            [2, 2, 0, 0, 0, 4],
            [4, -2, 0, -2, 2, -2],
            [-2, 4, -2, 0, -4, 4],
            [4, 0, 2, 0, 2, 0],
            [2, 4, 0, 2, 0, 4],
            [4, 0, 2, 0, 4, 2],
        ]),
    ),
    ("periodic", 4): (
        np.array([
            [1, -1, 1, 1],
            [1, -1, -1, -1],
            [-1, -1, 1, 1],
            [-1, -1, 1, 1],
        ], dtype=np.int8),
        np.array([
            [0, 0, 0, 2],
            [-2, -2, 0, 2],
            [0, -2, 0, 0],
            [0, -2, 2, 2],
        ]),
    ),
}


@pytest.mark.parametrize("key", list(GOLDEN))
def test_neighbour_sum_matches_pre_refactor_model(key):
    """
    Regression against the implementation that lived on `Model` before the
    geometry was extracted. If the extraction silently changed the boundary
    handling for either boundary, this is what catches it.
    """
    boundary, length = key
    state, golden = GOLDEN[key]
    assert np.array_equal(Lattice(length, boundary).neighbour_sum(state), golden)


# --------------------------------------------------------------------------
# Geometry against the naive reference
# --------------------------------------------------------------------------

@pytest.mark.parametrize("boundary,length", [
    ("open", 1), ("open", 2), ("open", 5), ("periodic", 4), ("periodic", 6),
])
def test_neighbour_sum_against_reference(boundary, length):
    rng = np.random.default_rng(11)
    lattice = Lattice(length, boundary)
    for _ in range(5):
        state = rng.choice(np.array([-1, 1], dtype=np.int8), size=(length, length))
        assert np.array_equal(
            lattice.neighbour_sum(state), ref_neighbour_sum(state, boundary)
        )


def test_coordination_numbers():
    """On an all-ones lattice the neighbour sum is the coordination number."""
    ones = np.ones((5, 5), dtype=np.int8)
    assert np.array_equal(Lattice(5, "open").neighbour_sum(ones), np.array([
        [2, 3, 3, 3, 2],
        [3, 4, 4, 4, 3],
        [3, 4, 4, 4, 3],
        [3, 4, 4, 4, 3],
        [2, 3, 3, 3, 2],
    ]))
    ones4 = np.ones((4, 4), dtype=np.int8)
    assert np.array_equal(
        Lattice(4, "periodic").neighbour_sum(ones4), np.full((4, 4), 4)
    )


@pytest.mark.parametrize("boundary,length", [
    ("open", 4), ("open", 5), ("periodic", 6),
])
def test_sublattice_partition(boundary, length):
    """
    The two masks tile the lattice and no two same-colour sites are
    neighbours. This is the assumption every checkerboard update rests on --
    and the one that fails for odd lengths with periodic boundaries, which is
    why the constructor forbids that combination.
    """
    lattice = Lattice(length, boundary)
    assert np.array_equal(lattice.white, ~lattice.black)
    assert lattice.black.sum() + lattice.white.sum() == length ** 2
    assert np.array_equal(lattice.black, ref_black_mask(length))
    assert lattice.sublattices == (lattice.black, lattice.white)

    for site, sites in ref_neighbours(length, boundary).items():
        for neighbour in sites:
            assert lattice.black[site] != lattice.black[neighbour]


# --------------------------------------------------------------------------
# The one-code-path claims: many shapes and dtypes, one buffer cache
# --------------------------------------------------------------------------

@pytest.mark.parametrize("boundary,length", [("open", 5), ("periodic", 6)])
def test_leading_axes_are_summed_independently(boundary, length):
    """
    A `(C, L, L)` field is C independent 2D neighbour sums. This is what lets
    a clock model's `(2, L, L)` vectors and a Potts model's `(q, L, L)`
    one-hot counts reuse the Ising stencil verbatim.
    """
    rng = np.random.default_rng(5)
    lattice = Lattice(length, boundary)
    field = rng.normal(size=(3, length, length))
    summed = lattice.neighbour_sum(field)
    assert summed.shape == field.shape
    for channel in range(3):
        assert np.allclose(
            summed[channel], lattice.neighbour_sum(field[channel].copy())
        )


@pytest.mark.parametrize("boundary", ["open", "periodic"])
def test_dtype_is_preserved_and_not_promoted(boundary):
    lattice = Lattice(4, boundary)
    for dtype in (np.int8, np.int64, np.float32, np.float64):
        field = np.ones((4, 4), dtype=dtype)
        assert lattice.neighbour_sum(field).dtype == dtype


def test_open_halo_stays_zero_across_calls():
    """
    The open-boundary result must not depend on what was in the buffer last
    time. Feeding a saturated field and then a zero field is the sharpest
    version of that: any leaked halo shows up immediately.
    """
    lattice = Lattice(4, "open")
    lattice.neighbour_sum(np.full((4, 4), 100, dtype=np.int64))
    assert np.array_equal(
        lattice.neighbour_sum(np.zeros((4, 4), dtype=np.int64)), np.zeros((4, 4))
    )


def test_buffers_are_cached_per_shape_and_dtype():
    """
    Interleaving requests of different shape and dtype must not corrupt each
    other's buffers, and repeat requests must not reallocate.
    """
    rng = np.random.default_rng(2)
    lattice = Lattice(4, "periodic")
    scalar = rng.integers(-1, 2, size=(4, 4)).astype(np.int8)
    vector = rng.normal(size=(2, 4, 4))

    expected_scalar = lattice.neighbour_sum(scalar).copy()
    expected_vector = lattice.neighbour_sum(vector).copy()
    for _ in range(3):
        assert np.array_equal(lattice.neighbour_sum(scalar), expected_scalar)
        assert np.allclose(lattice.neighbour_sum(vector), expected_vector)

    assert len(lattice._pads) == 2
    ids = {key: id(pad) for key, pad in lattice._pads.items()}
    lattice.neighbour_sum(scalar)
    assert {key: id(pad) for key, pad in lattice._pads.items()} == ids


def test_wrong_shape_rejected():
    lattice = Lattice(4, "open")
    for shape in [(3, 3), (4, 5), (2, 4, 5)]:
        with pytest.raises(ValueError):
            lattice.neighbour_sum(np.zeros(shape))


# --------------------------------------------------------------------------
# Bond enumeration and counting
# --------------------------------------------------------------------------

@pytest.mark.parametrize("boundary,length", [
    ("open", 1), ("open", 2), ("open", 5), ("periodic", 4), ("periodic", 6),
])
def test_bonds_cover_every_bond_once(boundary, length):
    lattice = Lattice(length, boundary)
    found = []
    for rows_a, cols_a, rows_b, cols_b in lattice.bonds():
        for a, b, c, d in zip(rows_a.ravel(), cols_a.ravel(),
                              rows_b.ravel(), cols_b.ravel()):
            found.append(frozenset({(int(a), int(b)), (int(c), int(d))}))

    expected = [
        frozenset({a, b})
        for a, sites in ref_neighbours(length, boundary).items()
        for b in sites
    ]
    assert sorted(map(sorted, found)) == sorted(map(sorted, set(expected)))
    assert len(found) == len(set(found)) == lattice.n_bonds


@pytest.mark.parametrize("boundary,length", [
    ("open", 1), ("open", 5), ("periodic", 4), ("periodic", 6),
])
def test_site_and_bond_counts(boundary, length):
    lattice = Lattice(length, boundary)
    assert lattice.n_sites == length ** 2
    # Every bond has two ends, so the neighbour count sums to twice the bonds.
    ends = sum(len(s) for s in ref_neighbours(length, boundary).values())
    assert lattice.n_bonds == ends // 2


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"lattice_length": 5, "boundary": "periodic"},    # odd length, not bipartite
    {"lattice_length": 2, "boundary": "periodic"},    # bonds double counted
    {"lattice_length": 4, "boundary": "toroidal"},    # unknown boundary
    {"lattice_length": 0},                            # empty lattice
])
def test_constructor_validation(kwargs):
    with pytest.raises(ValueError):
        Lattice(**kwargs)


def test_repr_round_trips():
    lattice = Lattice(6, "periodic")
    assert repr(lattice) == "Lattice(6, boundary='periodic')"
