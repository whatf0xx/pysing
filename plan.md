# Colour plan

Extend `pysing` from the two-state Ising model to models whose state space *is* a colour space,
so that the pictures are a direct rendering of the microstate rather than a false-colour map of
±1.

The organising idea is that the existing `Model` splits cleanly into two halves — **lattice
geometry plus update schedule**, which every model on this list shares verbatim, and **the site
alphabet plus its conditional distribution**, which is different in every case and shares no
code at all. Everything below follows from taking that split seriously: extract the first half,
reimplement the second half per model, and resist the urge to unify the conditionals.

The primary target is the **q-state clock model**, whose state space is literally the hue wheel:
`θ_k = 2πk/q` maps to hue with no design decision, and states that are close in energy are close
in colour. It also happens to have the best test story of anything here, with three independent
exact checkpoints (§5.5).

---

## Scope

| Item | Status | Why |
|---|---|---|
| §1 `Lattice` extraction | **DONE** | the shared half; everything else depends on it |
| §2 `sample_categorical` | **DONE** | shared by clock, Potts, and any discrete model |
| §3 Oklch palettes | **DONE** | this is what makes the figures pop rather than look like clip art |
| §4 Clock model, heat bath | **DONE** | primary deliverable |
| §5 Potts | **DONE** | promoted from SKETCH: wanted for its own sake, and it is a small delta |
| §6 XY | **GO (next)** | clock at q→∞; numpy samples the conditional natively |
| §7 Heisenberg, Metropolis | **GO (after XY)** | n=3; simplest sampler first, exact one deferred |
| §8 Observables | **DONE** for the discrete models | per model, as each lands |
| §9 Tests | **DONE** for the discrete models | gates every phase |
| §10 Static figures | **DONE**, as `demo.py` | the deliverable for now |
| §11 Animation | **DEFERRED** | planned only; static figures first |
| §12 Everything else | **DEFERRED** | clusters, over-relaxation, exact O(3), Ashkin–Teller |

"SKETCH" means the design is recorded here in enough detail to build from, and deliberately not
built. "DEFERRED" means not now, with the reason recorded so the decision is revisitable.

**What landed, against what was planned.** Phases 0–5 plus Potts. Three deviations, all recorded
where they belong:

- **Potts was built**, not left as a sketch (§5). It was wanted for its own sake, and it turned
  out to pay for itself as a test: the `q = 3` clock and Potts conditionals are identical at
  `J_Potts = 3J/2`, which cross-checks both.
- **`oklch_ring` grew a `uniform_chroma` switch** (§3). The uniform-chroma ring the plan called
  for is genuinely washed out — the whole ring gets throttled to whatever chroma the blues can
  manage — so per-hue maxima are now available and are what `demo.py` uses. Equal lightness is
  the claim that matters and it is unchanged; see §3.
- **`ClockModel.phase_field` was added**, which the plan did not anticipate. The intermediate
  phase is long-wavelength and a raw microstate buries it in site-level speckle; nothing in the
  plan would have shown it. It also turned out to be the natural home for the vortex work in §6.

One planned claim did not survive contact: the plan assumed a static figure could be a plain run
at temperature. It cannot, below `T_c` — a quench from disorder has to coarsen through `O(L**2)`
sweeps and freezes into a picture of its own history. `demo.py` starts cold instead, and its
docstring records the hot-versus-cold comparison that establishes the difference.

---

## Target file layout

| File | Status | Notes |
|---|---|---|
| `lattice.py` | **new** | `Lattice`: geometry, boundary, checkerboard, `neighbour_sum` |
| `sampling.py` | **new** | `sample_categorical` and nothing else |
| `palette.py` | **new** | Oklch → sRGB, equal-luminance hue rings |
| `clock.py` | **new** | `ClockModel` |
| `spinvector.py` | new, phase 3 | `VectorModel` base: unit-vector state, energy, order parameter |
| `xy.py` | new, phase 3 | `XYModel(VectorModel)` |
| `heisenberg.py` | new, phase 4 | `HeisenbergModel(VectorModel)` |
| `potts.py` | **new** | built after all; see the scope note above |
| `demo.py` | **new** | interactive multi-panel figures, §10 |
| `model.py` | **touched** | `Model` delegates to `Lattice`; public API byte-identical |
| `test_model.py` | **untouched** | all 30 tests must still pass — that is the refactor's acceptance criterion |
| `test_lattice.py`, `test_clock.py`, `test_palette.py` | **new** | §9 |
| `run.py` | touched | multi-panel static figures, §10 |

---

## 1. Extract `Lattice` — the shared geometry

**Problem.** `Model` currently owns the boundary logic, the halo buffer and the checkerboard
masks. Three more models need all three, and the periodic halo fill is exactly the kind of
four-line block where a copy-pasted bug would live undetected.

**Design.** A `Lattice` owns geometry and knows nothing about spins. `Model` *holds* one and
delegates, so its public surface does not change.

```python
class Lattice:
    """Square lattice geometry: boundary handling, neighbour sums, checkerboard colouring.

    Knows nothing about what lives on the sites. `neighbour_sum` works on any
    `(..., L, L)` array, so a scalar Ising spin, a `(2, L, L)` clock vector and a
    `(q, L, L)` one-hot occupancy all use the same code path.
    """
    def __init__(self, lattice_length: int, boundary: str = "open"):
        # validation moves here verbatim from Model.__init__: L >= 1, boundary in
        # BOUNDARIES, and for periodic L >= 3 and even (bipartiteness).
        ...
        rows, cols = np.indices((lattice_length, lattice_length))
        self.black = (rows + cols) % 2 == 0
        self.white = ~self.black
        self._pads = {}

    @property
    def sublattices(self):
        """The two halves, in update order. Every model's `evolve` loops over this."""
        return (self.black, self.white)

    def _pad_for(self, field):
        """Halo buffer matching `field`'s leading shape and dtype, allocated once each."""
        key = (field.shape[:-2], field.dtype)
        if key not in self._pads:
            L = self.lattice_length
            self._pads[key] = np.zeros(field.shape[:-2] + (L + 2, L + 2), field.dtype)
        return self._pads[key]

    def neighbour_sum(self, field):
        pad = self._pad_for(field)
        pad[..., 1:-1, 1:-1] = field
        if self.boundary == "periodic":
            pad[..., 0, 1:-1] = field[..., -1, :]
            pad[..., -1, 1:-1] = field[..., 0, :]
            pad[..., 1:-1, 0] = field[..., :, -1]
            pad[..., 1:-1, -1] = field[..., :, 0]
        return (pad[..., :-2, 1:-1] + pad[..., 2:, 1:-1]
                + pad[..., 1:-1, :-2] + pad[..., 1:-1, 2:])
```

Points of care:

- **The open-boundary halo must stay zero across calls.** Writing `pad[..., 1:-1, 1:-1]` never
  touches the halo, and `boundary` is fixed for the lifetime of a `Lattice`, so a zeroed buffer
  stays zeroed. Keep the existing comment to that effect — it is the invariant the whole
  one-code-path trick rests on.
- **Caching by `(leading shape, dtype)`** rather than allocating once in `__init__`, because a
  single `Lattice` may be asked for `(L,L) int8` and `(2,L,L) float64` sums by different callers.
- **dtype is the caller's problem.** `int8` holds Ising sums (range [-4,4]) and Potts one-hot
  counts (range [0,4]); the vector models pass float64. No promotion happens inside.
- `(L, L)` sublattice masks broadcast correctly against `(C, L, L)` state in
  `np.copyto(dst, src, where=mask)` — verified.

**`Model` after the change.** `_neighbour_sum` becomes `return self.lattice.neighbour_sum(self.spins)`;
`_black`/`_white` become properties forwarding to the lattice, or the `evolve` loop reads
`self.lattice.sublattices` directly. `lattice_length` and `boundary` stay as forwarding
properties so nothing downstream notices. **Acceptance criterion: `test_model.py` passes
unmodified.**

---

## 2. `sample_categorical` — the shared discrete sampler

The one piece of *sampling* code that genuinely is shared, between the clock model, Potts, and
any future multi-state model.

```python
def sample_categorical(log_weights, rng):
    """Draw one label per site from `(q, L, L)` unnormalised log-weights.

    Returns `(L, L)` labels in `0..q-1`. The max-subtraction is the same log-space
    discipline as `Model.log_weight`: the weights are exponentials of energies and
    overflow at low temperature if formed directly.
    """
    lw = log_weights - log_weights.max(axis=0, keepdims=True)
    cdf = np.cumsum(np.exp(lw), axis=0)
    return (cdf < rng.random(cdf.shape[1:]) * cdf[-1]).sum(axis=0)
```

This has already been checked at q=8: it reproduces the independent per-channel
`½(1 + tanh(β(H + JΣ)))` conditional to a maximum absolute difference of **4.4e-16**. That is
the guarantee that the categorical route is a strict generalisation of the current `evolve`
rather than a parallel implementation that might drift from it.

Cost is O(q) per site. `np.cumsum` plus a comparison beats `searchsorted` here because the
weights differ per site, so there is no shared table to search.

---

## 3. Palettes — making the colour pop

The difference between "pretty" and "clip art" is almost entirely whether the colours are
**equal in perceptual lightness**. `tab10`, `Set1` and friends are not: their yellow reads as
foreground and their blue as background, so a viewer sees structure that is an artifact of the
palette rather than of the physics. sRGB's own luminance weights (0.2126 R, 0.7152 G, 0.0722 B)
show how severe the imbalance is — pure blue carries a fourteenth of the luminance of pure green.

**Design.** A `palette.py` exposing a hue ring at fixed Oklch lightness and chroma:

```python
def oklch_ring(q, lightness=0.72, chroma=None, phase=0.0):
    """`(q, 3)` sRGB array: q hues equally spaced round the Oklch circle at constant
    lightness and chroma, so no state is visually privileged.

    With `chroma=None`, pick the largest chroma that keeps *every* hue in gamut, so the
    ring stays uniform instead of clipping the hues that happen to run out first.
    """
```

- **Why Oklch and not HSL/HSV.** HSV's "value" is not lightness; a full-saturation HSV ring
  swings wildly in perceived brightness. Oklab is designed so that equal `L` reads as equal
  lightness, which is precisely the property wanted.
- **Gamut.** At `L = 0.72` the maximum in-gamut chroma varies by hue (blues run out first).
  Taking the per-hue maximum gives a lumpy ring; take the **minimum over hues** of the per-hue
  maximum instead, found by bisection on "do all channels land in [0, 1]". Slightly duller, but
  uniform, which matters more.
- **Transform chain.** Oklch → Oklab (`a = C cos h`, `b = C sin h`) → LMS' → cube → LMS → linear
  sRGB via Ottosson's matrices → sRGB gamma encode
  (`x ≤ 0.0031308 ? 12.92x : 1.055 x^(1/2.4) − 0.055`). About twenty lines with two fixed 3×3
  matrices. **Spot-check the coefficients against a reference implementation when writing them**
  — they are transcribed constants and a transposed matrix would produce plausible-looking but
  wrong colours.
- **The clock model needs no palette lookup at all** in principle, since hue is continuous and
  `θ` is the hue angle directly; but going through the same `oklch_ring` keeps lightness constant,
  which the naive `hsv` colormap would not.
- **Rendering** is then `ax.imshow(palette[labels])`, `(q,3)` indexed by `(L,L)` giving
  `(L,L,3)` — verified.
- Provide a **cyclic** variant for the clock/XY models (hue wraps, which is correct: state `q-1`
  is adjacent to state `0`) and note that a *sequential* colormap would be wrong there, because
  it would put a false seam in a system with no seam.

---

## 4. The clock model — primary deliverable

$$E = -J\sum_{\langle ij\rangle}\cos(\theta_i - \theta_j) \;-\; \vec H\cdot\sum_i \hat n_i,
\qquad \theta_k = \frac{2\pi k}{q},\quad \hat n_k = (\cos\theta_k, \sin\theta_k)$$

### 4.1 Why this one

- The state space **is** the hue wheel, so the colour map is forced rather than chosen, and
  energetically-similar states are perceptually-similar colours. Domain walls between adjacent
  hues are soft; between opposite hues, hard. That reads correctly without any tuning.
- Three distinct visual regimes from one model. For `q ≥ 5` the 2D clock model has **two**
  BKT transitions: low `T` locks to `q` discrete colours, an intermediate quasi-long-range-ordered
  phase gives smooth hue swirls with visible vortices, high `T` is noise.
- Best test story on this list — see §4.5.

### 4.2 State and the local field

Store **labels**, `(L, L) uint8`, not angles: the state space is discrete and labels keep it
exactly so. Angles are derived on demand through a length-`q` lookup table.

The key move, which is what makes this reuse the existing machinery rather than sit beside it:
since `cos(θ_i − θ_j) = n̂_i · n̂_j`, the local field is a **2-vector**,

$$\vec h_i = \vec H + J\sum_{nn}\hat n_j$$

and the conditional is `p(k) ∝ exp(β h⃗_i · n̂_k)`. That is the same "dot the local field with the
candidate state" shape as Ising's `tanh`, with a two-component field and `q` candidates instead
of a one-component field and two candidates.

```python
class ClockModel:
    def __init__(self, lattice_length, q=6, temperature=1.0, field=(0., 0.),
                 coupling=1.0, boundary="open", init="hot", seed=None):
        ...
        theta = 2 * np.pi * np.arange(q) / q
        self._unit = np.stack([np.cos(theta), np.sin(theta)])      # (2, q)

    def _vectors(self):
        return self._unit[:, self.labels]                          # (2, L, L)

    def _local_field(self):
        return self.H[:, None, None] + self.J * self.lattice.neighbour_sum(self._vectors())

    def evolve(self):
        for sublattice in self.lattice.sublattices:
            lw = self.beta * np.einsum("ck,cij->kij", self._unit, self._local_field())
            new = sample_categorical(lw, self.rng)                 # (L, L)
            np.copyto(self.labels, new, where=sublattice)
    ```

`_local_field` must be recomputed inside the loop, for the same reason as in `Model.evolve`: the
second sublattice has to see the freshly updated first one.

### 4.3 Energy and order parameter

Both keep the shape of the Ising versions, with the scalar product replaced by a dot product:

```python
@property
def energy(self):
    n = self._vectors()
    bond = (n * self.lattice.neighbour_sum(n)).sum()      # each bond counted twice
    return -0.5 * self.J * float(bond) - float(self.H @ n.sum(axis=(1, 2)))

@property
def magnetisation_per_spin(self):
    """The order parameter is a 2-vector; `|m|` is the scalar to plot."""
    return self._vectors().mean(axis=(1, 2))              # (2,)
```

`magnetisation` returning a `(2,)` array rather than an `int` is a deliberate break from `Model`
— these are different classes, not a subtype relationship, precisely so that this can differ.

### 4.4 The field argument

`field` becomes a 2-vector, which both biases a direction and *breaks the `Z_q` symmetry*.
Worth exposing because at `H = 0` the low-temperature state picks one of `q` colours at random,
which makes a multi-panel figure look arbitrary; a small field pins it and makes panels
comparable.

### 4.5 Exact checkpoints — why the test story is good

| q | Equivalent to | Relation |
|---|---|---|
| 2 | **Ising, same J** | `θ ∈ {0, π}` ⇒ `cos(θ_i − θ_j) = σ_i σ_j` exactly. No coupling rescale. |
| 3 | 3-state Potts | `cos Δθ ∈ {1, −½}`, an affine map of `δ`: `cos = (3δ − 1)/2`, giving `E_clock = 1.5·E_Potts + N_bonds/2` (the constant is not optional — a test must encode it) |
| 4 | **two decoupled Ising models at `J/2`** | rotate 45°: `n̂ = (σ, τ)/√2` ⇒ `n̂_i·n̂_j = ½(σ_iσ_j + τ_iτ_j)` ⇒ `β_c J = 2 × 0.4407 = 0.8814` |
| ≥5 | — | two BKT transitions |
| ∞ | XY | single BKT at `β_c J ≈ 1.1199` |

The `q = 2` case is an *identity*, not an approximation: `ClockModel(q=2)` and `Model` must
produce the same conditional distribution at the same `J`. That is a far stronger regression
test than anything statistical.

The `q = 4` case is the "decoupled channels" observation from the RGB discussion reappearing —
a model that looks like it has four coupled states is two independent Ising models wearing a
hat. Nice to have it fall out as an exact test rather than a remark.

---

## 5. Potts — sketch only

$$E = -J\sum_{\langle ij\rangle}\delta(s_i, s_j)$$

Recorded because it is a small delta from the clock model and may be wanted for its own sake
(hard-edged colour blocks, and first-order behaviour that clock does not have).

**Difference from clock.** Only `_local_field` and the log-weights change. The neighbour
"field" becomes a one-hot occupancy count:

```python
def _counts(self):
    """`(q, L, L)`: how many neighbours are in each state."""
    onehot = (self.labels[None, :, :] == np.arange(self.q)[:, None, None])
    return self.lattice.neighbour_sum(onehot.astype(np.int8))

def evolve(self):
    for sublattice in self.lattice.sublattices:
        lw = self.beta * (self.J * self._counts() + self.H[:, None, None])
        np.copyto(self.labels, sample_categorical(lw, self.rng), where=sublattice)
```

Note that one-hot encoding makes the **existing stencil work unchanged**, and that open
boundaries then mean "the absent neighbour is in no state at all", which is exactly right with a
zero halo. No boundary special-casing.

**What Potts has that clock does not.**

- **Exact critical point for every q** by self-duality: `β_c J = ln(1 + √q)`.
- **Transition order changes with q**: continuous for `q ≤ 4`, **first-order for `q ≥ 5`**. That
  buys phase coexistence, metastable droplets, and hysteresis between hot and cold starts — the
  most dramatic pictures available anywhere in this plan.
- A **bimodal energy histogram** at `T_c` for `q ≥ 5`: a plot that is simultaneously pretty and a
  direct measurement of first-order character.

**What it costs.** The colour map is arbitrary again (states have no ordering, so any assignment
of `q` hues is as good as any other), and equilibrating a first-order transition properly needs
multicanonical/Wang–Landau sampling, which is a much larger project than anything else here.

**Order parameter**, since magnetisation does not survive: `m = (q·max_k ρ_k − 1)/(q − 1)` with
`ρ_k` the population fraction, running 0 (disordered) to 1 (ordered) for any q.

**Field** generalises to a `q`-vector `H_k` biasing each state; `H_k = H δ_{k0}` recovers the
usual symmetry-breaking field.

---

## 6. XY — clock at q → ∞

$$E = -J\sum_{\langle ij\rangle}\cos(\theta_i - \theta_j) - \vec H\cdot\sum_i\hat n_i,
\qquad \theta_i \in [0, 2\pi)$$

**Everything in §4 carries over except the sampler.** `_local_field` is unchanged; the state
becomes `(L, L) float64` angles, or equivalently `(2, L, L)` unit vectors.

**The conditional is exactly von Mises.** With `h⃗` the local field, `φ = atan2(h_y, h_x)` and
`a = β|h⃗|`,

$$p(\theta) \propto e^{\beta \vec h\cdot\hat n(\theta)} = e^{a\cos(\theta - \varphi)}$$

which is the von Mises distribution with mean `φ` and concentration `a` — **and numpy samples it
natively**. The entire exact heat-bath update is:

```python
def evolve(self):
    for sublattice in self.lattice.sublattices:
        h = self._local_field()
        new = self.rng.vonmises(np.arctan2(h[1], h[0]), self.beta * np.hypot(*h))
        np.copyto(self.theta, new, where=sublattice)
```

This is *less* code than a Metropolis update and it is exact, so XY should use it rather than
the Metropolis route recommended for Heisenberg. `rng.vonmises` returns angles in `(−π, π]`;
wrap consistently with however `_vectors` is written (it doesn't matter for `cos`/`sin`, only
for plotting the raw angle).

**Physics worth showing.** A genuine BKT transition at `β_c J ≈ 1.1199` (`T_c ≈ 0.893 J`), and
**vortices** — points where all hues meet in a pinwheel. They are strikingly visible under a
cyclic hue map and are the single best argument for this whole colour project. Detect them by
summing the wrapped phase difference round each unit plaquette; the result is `2π` times the
winding number, so a `(L, L)` integer array of `−1/0/+1` marks antivortices, nothing, vortices.
Overplot as scatter points on the hue field.

---

## 7. Heisenberg — Metropolis, n = 3

$$E = -J\sum_{\langle ij\rangle}\vec S_i\cdot\vec S_j - \vec H\cdot\sum_i\vec S_i,
\qquad |\vec S_i| = 1$$

**State layout `(3, L, L)` float64**, not `(L, L, 3)`: it matches the verified broadcast pattern
against the `(L, L)` sublattice masks, and lets the stencil run per-component untouched.

**Sampler: Gaussian-perturbation Metropolis**, as recommended — frame-free, no special functions,
no orthonormal-basis construction.

```python
def evolve(self):
    for sublattice in self.lattice.sublattices:
        h = self.H[:, None, None] + self.J * self.lattice.neighbour_sum(self.spins)
        trial = self.spins + self.step * self.rng.normal(size=self.spins.shape)
        trial /= np.linalg.norm(trial, axis=0, keepdims=True)
        dE = -(h * (trial - self.spins)).sum(axis=0)
        # exp(-beta * max(dE, 0)) is 1 for downhill moves, so they are always accepted,
        # and the exponential never overflows.
        accept = self.rng.random(dE.shape) < np.exp(-self.beta * np.maximum(dE, 0.0))
        np.copyto(self.spins, trial, where=sublattice & accept)
```

Three things to get right:

- **The proposal is symmetric, so plain Metropolis is valid** (no Hastings ratio). `S' =
  normalise(S + εg)` with isotropic Gaussian `g` is rotationally covariant, so `q(S'|S)` depends
  only on `S·S'`, which is symmetric under exchange. Worth stating explicitly in the docstring —
  a non-symmetric proposal used with a bare Metropolis test is exactly the kind of error that
  produces plausible pictures and wrong physics.
- **`step` needs tuning** to roughly 50% acceptance, and acceptance is temperature-dependent.
  Expose it as a constructor argument and expose the running acceptance rate as a property, so
  it is at least visible; automatic tuning during burn-in breaks detailed balance and must not
  be done during measurement.
- **Renormalise defensively.** Floating-point drift in `|S| = 1` accumulates over long runs;
  `_check_state` should assert `|S| = 1` to `~1e-12` and the state should be renormalised
  wholesale if it ever drifts.

**Two honest limitations, to be documented rather than worked around.**

1. **Mermin–Wagner**: continuous symmetry, 2D, short-range interactions ⇒ no spontaneous
   magnetisation at any `T > 0`, and **no finite-temperature transition at all**. The correlation
   length grows exponentially as `T → 0`, so low-`T` pictures still show large smooth swirls, but
   nothing critical can be demonstrated. `critical_temperature` must **raise**, not return a
   number.
2. **S² does not map to colour.** A 3-component unit vector needs two angles, and there is no way
   to lay a sphere on a perceptually uniform colour space without a seam or a degeneracy. Two
   options, both compromised, both worth offering:
   - `(S⃗ + 1)/2` as raw sRGB — lands on the sphere inscribed in the RGB cube, so *the spin is the
     colour*, no palette at all. Honest and elegant; half of it is dark and lightness varies
     wildly.
   - Hue = azimuth, chroma ∝ `sin θ`, lightness fixed — perceptually much better, but sends both
     poles to grey, so the two *most opposed* states render identically.

   Ship the first as default, offer the second, and say plainly in the docstring that the
   ambiguity is geometric and not a rendering bug.

**`spinvector.py`.** XY and Heisenberg share unit-vector state, `_local_field`, `energy`, and the
vector order parameter; they differ only in `n`, the sampler, and the colour map. A small
`VectorModel` base carrying the shared four is justified. It is *not* justified to extend that
base to cover the clock model — clock stores labels, and forcing it into a vector representation
would lose the exactness of the discrete state space for no gain.

---

## 8. Observables

| Quantity | Ising | Clock / XY | Potts | Heisenberg |
|---|---|---|---|---|
| order parameter | `Σs/N` | `\|Σ n̂/N\|` | `(q·max ρ_k − 1)/(q−1)` | `\|Σ S⃗/N\|` (→0 always) |
| energy per spin | as now | as now, dot product | δ-count | dot product |
| critical point | Onsager 2.269 J | q=2,4 exact; else BKT | `ln(1+√q)` | **raises** |
| special | — | vortex winding number (§6) | bimodal `E` histogram, q≥5 | — |

The vortex winding number is the one worth building early: it is cheap, it is a genuine
topological observable, and it turns the XY figure from "nice texture" into "here is the defect
structure that drives the transition".

---

## 9. Tests

Following the existing suite's structure: exact tests first, statistical tests marked `slow`.

**9a. Refactor guard (gates everything).**
- `test_model.py` passes **unmodified**. Non-negotiable.
- `Lattice.neighbour_sum` on `(L,L) int8` matches the pre-refactor `Model._neighbour_sum`
  element-for-element, both boundaries, several `L`. Capture the golden values *before* the
  refactor lands, exactly as `test_open_bc_topology_unchanged` did for the last one.

**9b. Shared sampler.**
- `sample_categorical` with equal weights is uniform (chi-square, `slow`).
- With known unequal weights, empirical frequencies match (chi-square, `slow`).
- Degenerate `q = 1` returns all zeros; extreme log-weights (`±700`) do not overflow or NaN.

**9c. Clock — the exact checkpoints of §4.5.**
- `q = 2` conditional probabilities equal `Model.heat_bath_probs()` at the same `J`, to machine
  precision, on random states, both boundaries. *The strongest test in the suite.*
- `q = 4` energy equals the sum of two Ising energies at `J/2` under the 45° relabelling, on
  random states.
- `q = 3` energy satisfies `E_clock = 1.5·E_Potts + N_bonds/2`.

All three of these have been checked numerically on random 6×6 states and hold to float
precision, so they are known-good before a line of the implementation exists.
- Exact enumeration on a 2×2 open lattice at `q = 3` (81 states) — sampled distribution matches
  Boltzmann weights; and the half-sweep transition matrix is reversible w.r.t. the Boltzmann
  distribution, generalising `test_half_sweep_is_reversible`.
- `β → 0` gives a uniform label histogram; `β → ∞` with a field pins every site to the nearest
  state to `H⃗`.

**9d. Palette.**
- Every entry in `[0, 1]` and in gamut for all `q` in `2..16`.
- Round-tripping the ring back through the forward Oklab transform gives constant `L` to `~1e-6`
  — this is the property being claimed, so it should be the property tested.
- Hues are distinct and evenly spaced; the ring is cyclic (entry `q−1` adjacent to `0`).

**9e. Continuous models.**
- `|S| = 1` invariant holds to `1e-12` after many sweeps (both XY and Heisenberg).
- **Single spin in a field**: `L = 1`, no neighbours ⇒ `⟨S_z⟩ = coth(βH) − 1/(βH)` exactly
  (the Langevin function) for Heisenberg, and `I₁(βH)/I₀(βH)` for XY. Closed-form, decisive,
  `slow`.
- High `T` is uniform on the sphere/circle: mean → 0, and `cos θ` flat for n=3.
- Heisenberg proposal symmetry, numerically: histogram `S·S'` from `S → S'` and from `S' → S`
  and check they agree.
- XY: the `q → ∞` limit of `ClockModel` reproduces `XYModel` energy statistics at large `q`
  (`q = 64` should be indistinguishable), `slow`.

Note explicitly in `method.md` that **the transition-matrix reversibility tests do not generalise
to continuous spins** — XY and Heisenberg lose that safety net, which is exactly why the
closed-form Langevin/Bessel checks matter more there than they would otherwise.

---

## 10. Static figures

`run.py` grows a small set of figure builders, each producing one multi-panel PNG. All static;
no animation yet.

1. **Clock temperature sweep**, `q = 6`, one panel per `T` spanning both BKT transitions —
   locked colours, hue swirls, noise. The headline figure.
2. **Clock q sweep** at fixed reduced temperature, `q = 2, 3, 4, 6, 12` — shows the crossover
   from Ising-like blocks to continuous hue.
3. **XY with vortices**, hue field with winding-number overplot at three temperatures.
4. **Heisenberg**, both colour maps side by side at low `T`, honestly labelled, with the
   Mermin–Wagner caveat in the caption.

Add an **`init` argument** (`"hot"` random, `"cold"` uniform) to every new model while building
these. `Model` currently always hot-starts; cold-vs-hot from the same temperature is what
*shows* hysteresis, and it is needed for equilibration checks regardless.

Each builder takes a seed and writes a named file, so figures are reproducible and regenerable.

---

## 11. Animation — DEFERRED

Planned, not built. Recorded now so the static-figure code does not accidentally foreclose it.

- The natural interface is a `frames(model, n, stride)` generator yielding rendered `(L, L, 3)`
  arrays; `plot_to_axes` already gives the per-frame render, so this is thin.
- `matplotlib.animation.FuncAnimation` → GIF via `pillow`, or mp4 via `ffmpeg` (an external
  binary, so it must not become a hard dependency — keep it out of `requirements.txt` the same
  way `pytest` is).
- The two sequences worth animating: clock-model **domain coarsening** after a quench, and XY
  **vortex pair annihilation**.
- Constraint on the static work: keep rendering (state → RGB array) separate from plotting
  (RGB array → axes), so the animation path can reuse the first without the second.

---

## 12. Deferred, with reasons

| Idea | Why deferred |
|---|---|
| **Over-relaxation** for XY/Heisenberg — reflect `S⃗` through its local field, `S⃗ → 2(S⃗·h⃗)h⃗/\|h⃗\|² − S⃗`. Energy-conserving, deterministic, three vectorised lines, and interleaving it with the sampler dramatically cuts autocorrelation time. | Best effort-to-reward ratio on this list and *should be the first thing added after phase 4*. Deferred only because it is an accelerator, not a capability, and it needs a working sampler to interleave with. Comes with a perfect test: energy conserved to float tolerance. |
| **Exact O(3) heat bath** for Heisenberg: `a = β\|h⃗\|`, azimuth uniform, `u = cos θ = 1 + ln[r + (1−r)e^{−2a}]/a` (written that way for low-`T` stability). Verified against the Langevin function to ~4 decimals over `a ∈ [0.1, 30]`. | Needs a per-site orthonormal basis aligned to `h⃗` (branchless: cross `ĥ` with `x̂`, or with `ẑ` where `\|h_x\|` is large) and an `a → 0` guard falling back to uniform-on-sphere. Metropolis first; this replaces it once the model is otherwise trusted. |
| **Cluster algorithms** — Swendsen–Wang / Wolff for Potts and clock (Fortuin–Kasteleyn, bond probability `1 − e^{−βJ}`), embedded-Wolff for O(n) (project onto a random reflection plane, run Ising Wolff on the signs). | A real complexity jump: needs connected-component labelling (`scipy.sparse.csgraph`) or a BFS that numpy is bad at, and a new dependency. Worth knowing *before* designing it that **one cluster engine serves Ising, Potts, clock and O(n)** — design for that when the time comes. Needs the bond enumeration in Appendix A. |
| **Multicanonical / Wang–Landau** for `q ≥ 5` Potts | Only needed to equilibrate a first-order transition properly. Much larger than anything else here, and unnecessary unless Potts is actually built. |
| **Ashkin–Teller coupled RGB** — three Ising layers with a bond term `−K Σ_⟨ij⟩ Σ_{c<c'} s^c_i s^c_j s^{c'}_i s^{c'}_j`, measured to give cross-channel domain-wall coincidence with excess ratio 1.0 → 3.7 → 12.7 as `K` goes 0 → 0.15 → 0.30, and a Baxter phase where the composite orders while the individual channels do not. | Superseded by the clock model for the *colour* goal, but it is the more interesting *physics* of the two and the measurements are already done. The 8-state site conditional it needs is exactly `sample_categorical`, so §2 keeps the door open at zero cost. |

---

## Appendix A — bond enumeration

Carried over from the previous plan, unchanged; `method.md` references it for the cluster
algorithms in §12.

The original `Model.get_nn_pairs` yielded `(Spin, Spin)` for every bond — horizontal bonds row by
row, then vertical bonds. The enumeration is preserved implicitly in the slice shifts of
`neighbour_sum`. The explicit form, when bond identities are needed rather than bond sums:

```python
def bonds(self):
    """Yield (i0, j0, i1, j1) index arrays, one entry per bond direction."""
    ii, jj = np.indices((self.lattice_length, self.lattice_length))
    yield ii[:, :-1], jj[:, :-1], ii[:, 1:], jj[:, 1:]   # horizontal
    yield ii[:-1, :], jj[:-1, :], ii[1:, :], jj[1:, :]   # vertical
```

(Add the wrapping bonds for periodic boundaries.) This is the form worth having for
**bond-resolved observables** — two-point correlation functions, bond energy histograms, the
cross-channel wall-coincidence statistic of §12 — and for the **edge set of a Wolff/Swendsen–Wang
cluster build**. It belongs on `Lattice`, not on any model, since it is pure geometry.

Also relevant to §12: the single-site `flip()` primitive removed with the `Spin` class is what a
Metropolis or cluster update needs, since both flip rather than resample. It was dropped because
heat-bath *resamples*, not because flipping is wrong; §7 reintroduces the pattern.

---

## Work order

Each numbered step is one commit, and each is independently revertible.

0. **Capture golden data first.** Record `Model._neighbour_sum` output on fixed states at several
   `L` and both boundaries, *before* touching `model.py`, or the `Lattice` test in §9a degenerates
   into testing the new code against itself.
1. **`lattice.py`** plus `test_lattice.py`. `Model` delegates to it. **Gate: `test_model.py`
   passes unmodified.** No behaviour change lands in this commit.
2. **`sampling.py`** plus its tests. Tiny, standalone, no dependants yet.
3. **`palette.py`** plus its tests. Also standalone — deliberately before the first model, so the
   first clock figure is well-coloured on its first render rather than retrofitted.
4. **`clock.py`** plus `test_clock.py`. **Gate: the `q = 2` identity against `Model`, and the 2×2
   `q = 3` exact enumeration.** Nothing proceeds past a failure there.
5. **Static figures 1 and 2** in `run.py`; add `init=` while doing it.
6. **`spinvector.py` + `xy.py`** plus tests, and the vortex winding-number observable. Figure 3.
7. **`heisenberg.py`** plus tests. Figure 4.
8. **`method.md`** update: the geometry/alphabet split, the loss of the transition-matrix tests
   for continuous spins, and the colour-mapping compromises of §7. **`README.md`**: the new
   models, the palette argument, the new figures.

Phases 1–5 are the committed scope. 6–7 follow if 1–5 land cleanly. §11 and §12 are not in scope.
