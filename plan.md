# Refactor plan

Rewrite `pysing` around an array-backed lattice with a correct (detailed-balance-satisfying)
checkerboard heat-bath update, selectable boundary conditions, and reduced units.

The **update rule itself is not changing** — `p(s_i = +1) = (1 + tanh(β(H + J·Σ_nn s_j)))/2`
is already exactly heat-bath/Glauber. What changes is *when* it is applied, what the state is
stored in, and how the parameters are expressed. See `method.md` for the physics-level diff.

---

## Target file layout

| File | Status | Notes |
|---|---|---|
| `model.py` | rewritten | `Model` becomes array-backed; `Spin` dependency gone |
| `run.py` | updated | reduced units, per-spin magnetisation, seeded |
| `test_model.py` | **new** | pytest suite, see §10 |
| `pytest.ini` | **new** | register the `slow` marker; nothing else |
| `requirements-dev.txt` | **new** | `pytest` only — kept out of `requirements.txt` so the README's install flow stays minimal for people who just want to run the model |
| `method.md` | new | approach comparison |
| `README.md` | touched | document the `boundary` argument, link `method.md`, note how to run tests |
| `spin.py` | **deleted** | see Appendix A |
| `prelude.py` | **deleted** | see Appendix A |

---

## 1. Replace the `Spin` object graph with an `(L, L)` array

**Problem.** `Model.spins` is a flat `ndarray` of 90 000 `Spin` instances, each holding an
*object-dtype* `ndarray` of neighbour references (`spin.py:26`) — numpy's per-element overhead
with none of its vectorisation. One `evolve()` at L=100 costs 103 ms across 128 k function calls;
`Spin.nn_sum` alone is 42 ms of generator machinery over boxed Python floats.

**Change.** `self.spins` becomes a single `np.ndarray` of shape `(L, L)`, `dtype=np.int8`,
values ±1. Row-major indexing `[row, col]` matches the current flat convention `i = row*L + col`,
so `plot` and any saved data stay consistent.

```python
self.spins = self.rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))
```

**Consequence.** `magnetisation` becomes `int(self.spins.sum(dtype=np.int64))`;
`plot` no longer needs the `(spin.value + 1) / 2` list comprehension or the `.reshape`.

**Expected result:** ~54× at L=300 (452 ms/step → 8.3 ms/step, measured against a prototype).

---

## 2. Neighbour sums via a padded halo — one code path for both boundary conditions

**Problem.** `define_nn_pairs` (`model.py:130-209`) is 80 lines of hand-unrolled corner/edge
casework. It is correct (I verified it, including L=2), but it hardcodes open boundaries and
crashes with `IndexError` for `lattice_length=0`.

**Change.** A single private method builds a zero-padded `(L+2, L+2)` buffer and takes four
shifted slices. The boundary condition is *only* how the halo is filled:

```python
def _neighbour_sum(self) -> np.ndarray:
    """Sum of the four nearest-neighbour spins, per site."""
    pad = self._pad                      # cached (L+2, L+2) int8 buffer
    pad[1:-1, 1:-1] = self.spins
    if self.boundary == "periodic":
        pad[0, 1:-1]  = self.spins[-1, :]
        pad[-1, 1:-1] = self.spins[0, :]
        pad[1:-1, 0]  = self.spins[:, -1]
        pad[1:-1, -1] = self.spins[:, 0]
    # halo corners are never read
    return pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
```

Open BC = zero halo, so absent neighbours contribute 0 — **identical** to the current
topology, not an approximation of it. Values stay in [-4, 4], so `int8` does not overflow.
The buffer is allocated once in `__init__` (`self._pad`) and reused, so `_neighbour_sum` does
no allocation beyond the returned array.

---

## 3. Boundary conditions as a constructor option

**Change.** `Model(..., boundary: str = "open")`, accepting `"open"` or `"periodic"`.
Stored as `self.boundary`; validated in `__init__`.

**Validation rules (both must raise `ValueError` with an explanatory message):**

- `boundary` not in `{"open", "periodic"}`.
- `boundary == "periodic"` **and `L` odd** — with wraparound the lattice is no longer bipartite
  (the wrap closes odd-length cycles), so the checkerboard update in §4 would no longer be exact.
  This is a real constraint, not a convenience; it must be enforced rather than documented.
- `boundary == "periodic"` **and `L < 3`** — at L=2 a site's left and right neighbours are the
  same site, so each bond is counted twice and the energy is wrong. Standard PBC degeneracy.
- `lattice_length < 1` (fixes the `IndexError` at L=0).

Open BC has no parity or size restriction; it stays the default so existing behaviour is
opt-out, not opt-in.

---

## 4. Checkerboard update — the correctness fix

**Problem.** `evolve` (`model.py:249`) computes every probability from the state at the start of
the step, then writes all spins, so each site reads stale neighbours. That is parallel ("Little")
dynamics, which does not satisfy detailed balance with respect to the Ising Gibbs measure.
Measured on a 4×4 lattice against exact enumeration of all 65 536 states (βJ = 0.35, H = 0):
⟨|m|⟩ = 0.3496 versus the exact 0.4166 — a 16 % systematic error in the order parameter.

**Change.** Colour the lattice by `(i + j) % 2`. Every black site's neighbours are all white, so
an entire sublattice can be resampled simultaneously *and exactly* — it is a block Gibbs sampler
and detailed balance holds. One `evolve()` = both sublattices = one full sweep.

```python
def evolve(self):
    """One sweep: resample each sublattice in turn from its exact conditional."""
    beta = self.beta
    for sublattice in (self._black, self._white):
        h = beta * (self.H + self.J * self._neighbour_sum())
        p = 0.5 * (1.0 + np.tanh(h))
        new = np.where(self.rng.random(p.shape) < p, np.int8(1), np.int8(-1))
        np.copyto(self.spins, new, where=sublattice)
```

`self._black = (i + j) % 2 == 0` and `self._white = ~self._black` are built once in `__init__`
via `np.indices`.

**Delete the `random() > 0.2` damping.** It was suppressing the checkerboard blinking caused by
synchronous updates; with sublattice updates the oscillation cannot occur, so the hack is not
just unnecessary, it is a second source of bias.

**Known inefficiency, deliberately accepted:** `h` and the random field are computed over the
full lattice each half-sweep and then half-discarded, so the sweep does 2× the strictly necessary
`tanh` and RNG work. Avoiding it needs boolean fancy-indexing, which in practice costs more than
the wasted vectorised work at these sizes. Noted here so it is a choice, not an oversight.

---

## 5. Fold `probability_gradient` into the update

**Problem.** `probability_gradient` (`model.py:92`) is documented as a gradient of the Boltzmann
numerator with the common factor `p` dropped. The maths is right, but the framing is a detour:
passed through `tanh` the quantity is just β times the local field, and the result is the exact
heat-bath conditional. Dropping `p` is also *required*, not merely convenient — keeping it would
overflow exactly as `z_prob` does (§7).

**Change.** Remove the public property. The local field becomes a private helper used by
`evolve` and, if wanted, by observables:

```python
def _local_field(self) -> np.ndarray:
    """H + J·Σ_nn s_j at every site (an energy, not a gradient)."""
    return self.H + self.J * self._neighbour_sum()
```

`evolution_probs` is likewise absorbed; if a public hook is still wanted for plotting, expose
`heat_bath_probs` with a docstring that names the rule correctly.

---

## 6. Reduced units: set `k_B = 1`

**Problem.** `temperature=7.2429705e+22` chosen against a hardcoded `k_b = 1.380649e-23`
(`model.py:78`) so that β ≈ 1. Only the dimensionless combinations βJ and βH carry physics.
This is the blocker for the "recast this in terms of T, not J" commit.

**Change.**

- Delete `inverse_temp` and the `k_b` constant. Add `beta` → `1.0 / self.T`.
- New signature: `Model(lattice_length, temperature=2.0, field=0.0, coupling=1.0, ...)`,
  i.e. J is the energy unit and T is measured in units of J/k_B.
- Add read-only convenience properties `beta_J` (`self.J / self.T`) and `beta_H`.
- Guard `T <= 0` with a `ValueError` (β = ∞ is not representable and `tanh` would give a
  zero-temperature quench, which deserves its own code path if ever wanted).

`T`, `H`, `J` stay plain mutable attributes so `run.py`'s mid-run `m.H = 0` keeps working;
`beta` and friends are derived properties, never cached.

---

## 7. `z_prob` → `log_weight`

**Problem.** `z_prob` (`model.py:82`) returns `inf` for anything above a toy lattice — at L=100
it overflows silently but for a `RuntimeWarning` (βE ≈ −15 000). It is unusable as written.

**Change.** Replace with

```python
@property
def log_weight(self) -> float:
    """log of the unnormalised Boltzmann weight, −βE. Use differences, never exp() of this."""
    return -self.beta * self.energy
```

Docstring states explicitly that any ratio of weights must be formed from ΔE, and that the
sampler itself never needs this quantity.

---

## 8. `critical_temp` → `reduced_temperature`

**Problem.** `critical_temp` (`model.py:31`) is named as a temperature but returns the
dimensionless ratio T/T_c. (The constant 0.440687 = ln(1+√2)/2 is correct.)

**Change.** Rename to `reduced_temperature`, returning `T / T_c` with
`T_c = 2 J / ln(1 + √2)` exposed as a separate `critical_temperature` property so the name
finally means what it says. Docstring must state the three conditions under which Onsager's
result applies: infinite lattice, zero field, periodic boundaries — none of which strictly hold
here, so it is a guide to where you are in parameter space, not a prediction.

Guard `J == 0` (division by zero).

---

## 9. Observables, RNG, annotations, plotting

**9a. Per-spin observables.** `run.py` currently plots raw sums up to ±90 000. Keep
`magnetisation` and `energy` extensive (that is what the words mean), and add
`magnetisation_per_spin` and `energy_per_spin`. `run.py` plots the per-spin quantity.

**9b. Vectorised `energy`.** Each bond appears twice in `Σ_i s_i · nn_sum_i`, so:

```python
@property
def energy(self) -> float:
    bond_sum = (self.spins * self._neighbour_sum()).sum(dtype=np.int64)
    return -0.5 * self.J * float(bond_sum) - self.H * self.magnetisation
```

Correct for open BC (zero halo contributes nothing) and for periodic BC at L ≥ 3.

**9c. Seeded RNG.** Drop `from random import choice, choices, random` entirely. `Model` takes
`seed: int | None = None` and holds `self.rng = np.random.default_rng(seed)`, used for both the
initial state and every update. `run.py` passes a fixed seed.

**9d. Annotations.** Replace `np.float64` with `float` on every scalar parameter and return
(the values are Python floats, and `sum()` of Python floats was never an `np.float64` anyway).
Drop the now-unused `from typing import Tuple, Iterator` and `from itertools import pairwise`.

**9e. Direct-assignment validation.** `Spin.__init__` asserted `value in (1., -1.)` but
`model.py:257` bypassed it. With an `int8` array the invariant is structural; add a
`_check_state()` helper used only by the tests rather than a runtime guard on the hot path.

**9f. Plotting.** `plot` currently duplicates `plot_to_axes` and hardcodes `dpi=320`
(`example.png` is 1.3 MB). Refactor so `plot` creates the figure and *delegates* to
`plot_to_axes` — which turns a dead method into a live one and makes multi-panel figures like
`example.png` easy again. Add a `dpi: int = 150` parameter. Title becomes `T`, `H`, `J` plus the
boundary condition.

---

## 10. Test suite (`pytest`)

`pytest` is not currently in the venv — it goes in a new `requirements-dev.txt`, installed with
`pip install -r requirements-dev.txt`. A three-line `pytest.ini` registers the `slow` marker so
the fast suite is `pytest -m "not slow"` and the full one is plain `pytest`.

**Ground rule:** every reference implementation in the test module is written independently of
`model.py` — naive Python loops over explicit indices. A test that reuses `_neighbour_sum` to
check `energy` proves nothing.

### 10a. Sampler correctness — the tests that justify the whole refactor

| Test | What it asserts | How |
|---|---|---|
| `test_half_sweep_is_reversible` | Each sublattice update satisfies detailed balance, π(s)P(s→s′) = π(s′)P(s′→s) | 3×3 open lattice, 512 states. Build the 512×512 half-sweep transition matrix exactly (the sublattice update factorises over sites, so each entry is a product of per-site conditionals), and compare against the Boltzmann π. **Deterministic — no sampling noise, no tolerance beyond float error.** |
| `test_full_sweep_is_stationary` | πP = π for the two-half-sweep composition | Same machinery. Note the composition of two reversible kernels is *not* generally reversible, so stationarity — not detailed balance — is the correct claim to assert here |
| `test_matches_exact_enumeration` | Sampled ⟨\|m\|⟩ and ⟨E⟩ match brute force | 4×4, enumerate all 2¹⁶ states. Parametrised over (βJ, H) ∈ {(0.35, 0), (0.35, 0.2), (0.2, 0)} × {open, periodic}. The H ≠ 0 case matters: it breaks the ±s symmetry and is the only statistical check on the field term's sign. Marked `slow` |

The first two are the real prize: they pin the exact property the original code violated, in
milliseconds and without a tolerance. The enumeration test is the end-to-end backstop. Both
would have caught the 16 % bias.

Exact transition matrices are only feasible for L ≤ 3 (512 states → a 512² matrix). Periodic
mode needs even L ≥ 4, i.e. ≥ 65 536 states, so periodic is covered by the statistical test only.

### 10b. Lattice geometry

| Test | What it asserts |
|---|---|
| `test_coordination_numbers` | `_neighbour_sum` on an all-up lattice reproduces the coordination number per site: open BC gives 2 at corners, 3 on edges, 4 inside; periodic gives 4 everywhere |
| `test_neighbour_sum_against_reference` | Random states, both BCs, against a naive loop using explicit (and, for periodic, modular) index arithmetic |
| `test_open_bc_topology_unchanged` | **Regression against the current code.** Golden neighbour sums for a fixed 5×5 state, captured from the *existing* `define_nn_pairs`/`Spin.nn_sum` before `spin.py` is deleted, hardcoded into the test. Guards against silently changing the geometry during the rewrite |
| `test_sublattice_partition` | Black and white masks are complementary and cover the lattice, **and no two same-colour sites are neighbours** — checked by shifting the mask in all four directions under the active BC. This is the assumption the checkerboard update rests on, and the one that fails for odd L with periodic BC |

### 10c. Analytic limits

| Test | What it asserts |
|---|---|
| `test_zero_coupling_gives_tanh` | At J = 0 the sites are independent, so ⟨m⟩ = tanh(βH) exactly. Clean closed-form check of the field term and the ½(1+tanh) normalisation, with no lattice physics involved |
| `test_saturating_field` | βH ≫ 1 drives every spin to +1 within one sweep; βH ≪ −1 to −1. Effectively deterministic |
| `test_infinite_temperature_is_unbiased` | T → ∞ gives p = ½ per site, so \|m\| ~ O(N^(−1/2)); assert it sits inside a few standard errors |
| `test_ordering_below_tc` | From a random start at T = 0.7 T_c, periodic BC, \|m\| per spin exceeds ~0.9 after enough sweeps; above T_c it stays small. Loose bounds — this is a smoke test for "the model does Ising things", not a measurement. Marked `slow` |

### 10d. Observables and plumbing

| Test | What it asserts |
|---|---|
| `test_energy_matches_bond_loop` | Vectorised `energy` against an explicit loop over bonds, both BCs, random states and fields |
| `test_log_weight_finite_at_large_lattice` | **Regression for the `z_prob` overflow.** L=100 returns a finite `log_weight`, and `pytest.warns` records no `RuntimeWarning` |
| `test_per_spin_observables` | `magnetisation_per_spin == magnetisation / N`, same for energy |
| `test_reduced_temperature` | Equals 1.0 at T = T_c; scales as expected; raises on J = 0 |
| `test_parameters_are_mutable_midrun` | Setting `m.H = 0` after construction changes `beta_H` — guards against anyone caching β or βH, which is what `run.py` depends on |
| `test_state_invariants` | After many sweeps the array is still `int8` and every entry is ±1 |
| `test_reproducible_with_seed` | Same seed → bit-identical trajectories; different seeds → different ones |
| `test_constructor_validation` | Parametrised `pytest.raises(ValueError)`: odd L periodic, L < 3 periodic, unknown `boundary` string, T ≤ 0, L < 1 |
| `test_plot_writes_file` | `matplotlib.use("Agg")`, `plot(filename=tmp_path/"x.png")` produces a non-empty file — cheap cover for the §9f plotting refactor |

### 10e. Practicalities

- **Tolerances** on the statistical tests will be calibrated from a pilot run with margin, not
  guessed. Successive sweeps are autocorrelated, so the naive √N standard error understates the
  true spread; I will size the tolerance from the observed scatter across seeds.
- **Determinism.** Every stochastic test takes an explicit seed, so a pass is a pass. Where a
  test is inherently statistical I will note the failure probability in a comment.
- **Runtime budget:** `pytest -m "not slow"` under ~5 s; the full suite under ~60 s.

---

## 11. `run.py`

- Parametrise in reduced units. To preserve the current demo's *dimensionless* parameters
  exactly (βJ = 0.3, βH = 1.0): `J = 1.0`, `T = 10/3`, `H = 10/3`. That is T ≈ 1.47 T_c, i.e.
  above criticality, which is consistent with the "relaxation after the field is removed" demo.
- Plot `magnetisation_per_spin`; label axes; label the x-axis "sweeps".
- Pass `seed`, and expose `boundary` as a top-level variable so both modes are one edit away.

The plotted curve will not reproduce the old figure — the dynamics is being corrected and the
time unit changes from "80 % of spins updated" to "one full sweep". That is expected and
correct. I will keep the parameters faithful to the original dimensionless values and, if the
resulting curve reads badly, adjust `field_time`/`relax_time` (the observation window) rather
than the physics.

---

## Appendix A — removed code: what it did, and how to restore the capability

Nothing here is lost information; each item maps to a cheaper array-level equivalent. Recorded
so that reintroducing any of it is a deliberate five-minute job rather than an archaeology
exercise.

### `spin.py` — the whole `Spin` class

| Member | What it did | How to get it back |
|---|---|---|
| `value` | ±1 float, one per site | `model.spins[i, j]` (int8) |
| `id`, `__repr__` | debugging identity, `spin-up@1234` | flat index `i*L + j`; add a `Model.describe(i, j)` helper if a printable form is wanted |
| `initialised`, `initialise` | guarded against using a spin before its neighbours were wired up | no longer meaningful — neighbours are implicit in the array geometry, so the invalid state cannot be constructed |
| `nearest_neighbours` | object-dtype array of neighbour `Spin`s | `_neighbour_sum()` gives the *sum*, which is all any Ising update needs. For the neighbour **identities** (e.g. to build clusters), generate index arrays with `np.indices` and the same four shifts |
| `flip()` | invert one spin in place | `self.spins[i, j] *= -1`. **This is the primitive a Metropolis or Wolff implementation needs** — single-spin proposals and cluster flips both flip rather than resample. It was removed because heat-bath resamples (`np.where`) instead of flipping, not because flipping is wrong |
| `nn_sum()` | sum over one site's neighbours | `_neighbour_sum()[i, j]`, or the four explicit lookups if only one site is needed |

### `prelude.py` — `grid_pairs`

A standalone duplicate of `Model.get_nn_pairs` operating on a flat array of indices, with a
`__main__` block printing the pairs for a 3×3 grid. It was a scratchpad for working out the
bond ordering. Imported nowhere. The bond enumeration it encodes is preserved in the slice
shifts of `_neighbour_sum`; if a printable list of bonds is ever wanted again, it is
`np.stack(np.indices((L, L)))` plus the two shift directions.

### `Model.get_nn_pairs`

Yielded `(Spin, Spin)` for every bond — horizontal bonds row by row, then vertical bonds.
Used only by `nn_sum`, itself used only by `energy`. Both are now O(N) vectorised.

**Restore as** a `bonds()` generator yielding *pairs of index arrays* rather than pairs of
objects:

```python
def bonds(self):
    """Yield (i0, j0, i1, j1) index arrays, one entry per bond direction."""
    ii, jj = np.indices((self.lattice_length, self.lattice_length))
    yield ii[:, :-1], jj[:, :-1], ii[:, 1:], jj[:, 1:]   # horizontal
    yield ii[:-1, :], jj[:-1, :], ii[1:, :], jj[1:, :]   # vertical
```

This is the form worth having for **bond-resolved observables** — two-point correlation
functions ⟨s_0 s_r⟩, bond energy histograms, or the edge set for a Wolff/Swendsen–Wang cluster
build. Add it when one of those is actually needed; it is not needed by the dynamics.

### `Model.define_nn_pairs`

Built the neighbour lists for every site at construction. Wholly replaced by §2 — the
geometry is now implicit, so there is nothing to precompute and nothing to get wrong at the
corners.

### `Model.probability_gradient` and `Model.evolution_probs`

See §5. The local field survives as `_local_field()`; the probabilities are computed inline in
`evolve`. Restore either as a public property in one line if you want to plot the acceptance
field.

### `Model.z_prob`

See §7 — replaced by `log_weight`, which is the same quantity in a representable form.

### `Model.plot_to_axes` — **retained, not removed**

Dead in the current tree, but it is the right primitive for multi-panel figures (which is
presumably how `example.png` was made). §9f makes `plot` call it, so it becomes live code.

---

## Work order

0. **Capture golden data first.** Run the *existing* `define_nn_pairs`/`Spin.nn_sum` on a fixed
   5×5 state and record the neighbour sums. This has to happen before `spin.py` is deleted, or
   `test_open_bc_topology_unchanged` (§10b) degenerates into testing the new code against itself.
1. `model.py` rewrite: array state, `_neighbour_sum`, boundary validation (§1–§3).
2. Checkerboard `evolve`, delete the damping (§4–§5).
3. Units, `log_weight`, `reduced_temperature`, observables, RNG, annotations, plotting (§6–§9).
4. Delete `spin.py`, `prelude.py`; clear stale `__pycache__`.
5. `requirements-dev.txt`, `pytest.ini`, `test_model.py` (§10). Install pytest, run the suite.
   The reversibility and stationarity tests gate everything above — if they fail, nothing else
   matters.
6. `run.py` (§11), then execute it end to end to confirm the figure renders.
7. `README.md`: document `boundary`, link `method.md`, add the test command.
