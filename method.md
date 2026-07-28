# Method: what changed, and why

A comparison of the original `pysing` dynamics with the implementation described in `plan.md`.

## The rule is the same

The original `evolve` sets each spin to +1 with probability

$$P(s_i = +1) = \tfrac{1}{2}\left[1 + \tanh\!\big(\beta (H + J\textstyle\sum_{j \in nn(i)} s_j)\big)\right]$$

arrived at by treating $\beta(H + J\sum_{nn} s_j)$ as a "probability gradient" and squashing it
through a `tanh`. That expression is identically the **heat-bath (Glauber) conditional**

$$P(s_i = +1 \mid \{s_j\}) = \frac{e^{\beta h_i}}{e^{\beta h_i} + e^{-\beta h_i}}, \qquad h_i = H + J\sum_{j \in nn(i)} s_j$$

There is no missing factor of 2 and no approximation: the two are the same function. Dropping the
common factor $p$ from the gradient is also exactly right — it is a positive constant across
spins and cancels in the normalisation. So the acceptance rule is not being changed. Everything
below is about the *schedule*, the *representation*, and the *parameters*.

## Difference 1: update schedule — parallel vs. checkerboard

**Original.** All $N$ probabilities are computed from the state at the start of the step, then
every spin is written. Each site therefore reads neighbour values that its own update has already
invalidated. This is parallel, or "Little", dynamics.

Parallel dynamics is a perfectly well-defined Markov chain, but it does **not** satisfy detailed
balance with respect to the Ising Gibbs measure $\pi(s) \propto e^{-\beta E(s)}$. Its stationary
distribution is a different object (formally, the marginal of a two-layer Ising model), so
equilibrium averages measured from it are biased, and its relaxation times are not Glauber
relaxation times.

It also has a characteristic pathology: a ferromagnetic lattice updated in parallel can fall into
a period-2 checkerboard oscillation, every spin responding to the configuration it just left.
The original code's `if random() > 0.2` — updating only ~80 % of spins per step — is a damping
term that suppresses that oscillation. It reduces the bias but does not remove it; only the
$p_{\text{update}} \to 0$ limit (one spin at a time) recovers true Glauber dynamics.

**New.** The square lattice is bipartite. Colour it like a chessboard by $(i+j) \bmod 2$; every
black site's neighbours are all white. So the whole black sublattice can be resampled at once
from its **exact** conditional distribution, then the whole white sublattice. This is a block
Gibbs sampler: detailed balance holds exactly, and the update is still fully vectorised. The
damping is deleted, because the oscillation it defended against cannot occur.

**Measured.** 4×4 open lattice, $\beta J = 0.35$, $H = 0$, against brute-force enumeration of all
65 536 states:

| Sampler | ⟨\|m\|⟩ |
|---|---|
| exact Boltzmann | 0.4166 |
| original: parallel, 80 % damping | 0.3496 |
| new: checkerboard | 0.4174 |

A 16 % systematic error in the order parameter, removed.

**Caveat this introduces.** With periodic boundaries the wraparound closes odd-length cycles when
$L$ is odd, and the lattice stops being bipartite — so the checkerboard update would no longer be
exact. Periodic mode therefore requires even $L$, enforced in the constructor.

## Difference 2: state representation

**Original.** `Model.spins` is a flat array of $N$ `Spin` objects, each holding an
*object-dtype* `ndarray` of references to its neighbours, wired up by 80 lines of explicit
corner-and-edge casework. The neighbour geometry is data.

**New.** A single `(L, L)` `int8` array. The neighbour geometry is implicit in the array
indexing: neighbour sums are four shifted slices of a zero-padded copy. The boundary condition
becomes nothing more than *how the one-cell halo is filled* — zeros for open, wrapped edges for
periodic — so both modes share one code path.

Measured, same rule, same lattice:

| | original | array-backed |
|---|---|---|
| L=100 | 50 ms/step | 1.2 ms/sweep |
| L=300 | 452 ms/step | 9.8 ms/sweep |

(Measured on the delivered implementation. Note the units differ slightly in the
original's favour — see Difference 5: an old "step" touched ~80 % of the spins, a new
sweep touches all of them.)

The original's cost was almost entirely Python object overhead: at L=100 one step made 128 000
function calls, with 42 ms in generator machinery summing boxed floats over neighbour lists and
39 ms in `random.choices` drawing single Bernoulli variates through a cumulative-weight bisect.

## Difference 3: units

**Original.** `temperature = 7.2429705e+22` chosen against a hardcoded
`k_b = 1.380649e-23` so that $\beta \approx 1$, with `coupling = 1e-2` in implied joules.

**New.** $k_B = 1$ and $J$ is the unit of energy, so temperature is measured in $J/k_B$ and the
critical point sits at the familiar $T_c \approx 2.269\,J$. Only the dimensionless groups
$\beta J$ and $\beta H$ ever had physical content; this makes that explicit rather than encoding
it in a 23-digit default.

A related consequence: `z_prob` returned $e^{-\beta E}$ directly, which overflows to `inf` above
about a 20×20 lattice ($\beta E \approx -15\,000$ at L=100). It becomes `log_weight` = $-\beta E$;
any ratio of weights should be built from $\Delta E$.

## Difference 4: boundary conditions

**Original.** Open boundaries only — edge sites have 3 neighbours, corners 2. A deliberate
choice, documented in the `energy` docstring.

**New.** `boundary="open"` (still the default) or `boundary="periodic"`. Open boundaries impose
a free surface that suppresses order near the edges and shifts the apparent critical point;
periodic boundaries are the standard choice when extracting $T_c$, critical exponents, or
correlation lengths, because they remove the surface entirely. Having both makes the finite-size
effect something you can measure rather than something you inherit.

## Difference 5: unit of time

One call to `evolve()` used to update roughly 80 % of the spins once. It now updates 100 % of
them exactly once — one full sweep, the conventional Monte Carlo time unit. Any relaxation time
measured in "steps" therefore rescales by about $1/0.8$ **on top of** the change from correcting
the dynamics. The two effects are not separable; a τ extracted under the old scheme should not
be compared to one extracted under the new.

## Difference 6: the alphabet is now pluggable, and the geometry is shared

Everything above concerns one model. Adding the Potts and clock models forced the question of what
they actually share with it, and the answer is cleaner than expected: a lattice model splits into
a **geometry** half and an **alphabet** half, and the two have nothing to say to each other.

The geometry half — which sites are neighbours, how the boundary closes, which sites may be
updated simultaneously — is *identical* for Ising, Potts, clock, XY and Heisenberg. It is now
`Lattice`, and it knows nothing about spins. The one trick that makes it shared rather than merely
similar is that `neighbour_sum` takes any `(..., L, L)` array and sums over the trailing two axes,
so a scalar Ising spin, a `(2, L, L)` clock vector and a `(q, L, L)` one-hot occupancy all go
through the same four shifted slices. `Model` holds one and delegates; its public surface did not
change, and the acceptance criterion for the extraction was that `test_model.py` passed unmodified.

The alphabet half — what lives on a site, and the conditional it is redrawn from — shares *no*
code, and deliberately so. The three conditionals are a `tanh`, a q-way softmax over a dot product
with a two-vector field, and a q-way softmax over neighbour counts. Unifying them would mean
inventing an abstraction over three expressions that are already one line each. The only genuinely
shared piece is `sample_categorical`, which draws a label per site from `(q, L, L)` log-weights,
and which reproduces the Ising `tanh` conditional to 4.4e-16 when handed two states.

### Why the clock model, and why its tests are unusually strong

The clock model confines spins to `q` equally spaced directions on a circle. Because
`cos(θ_i − θ_j) = n̂_i · n̂_j`, the local field is a two-vector `h = H + J Σ_nn n̂_j` and the
conditional is `p(k) ∝ exp(β h · n̂_k)` — the same "dot the local field with the candidate" shape
as the Ising heat bath, with two components and `q` candidates instead of one and two.

Its state space *is* the hue wheel, so the colour map is forced rather than chosen. It also
collapses onto solved models at three values of `q`, which is what the test suite is built around:

| q | reduces to | relation |
|---|---|---|
| 2 | **Ising at the same J** | `θ ∈ {0, π}` ⇒ `cos(θ_i − θ_j) = s_i s_j`. An identity, checked to machine precision against `Model.heat_bath_probs` |
| 3 | 3-state Potts | `cos` takes only `{1, −½}`, an affine map of `δ`: `E_clock = 1.5 E_Potts + n_bonds/2` |
| 4 | **two Ising models at J/2** | rotate 45°, `n̂ = (s, t)/√2` ⇒ `n̂_i · n̂_j = (s_i s_j + t_i t_j)/2` |

The additive constant at `q = 3` is not cosmetic. It cancels out of the normalised conditional —
so a `q = 3` clock model and a `q = 3` Potts model at `1.5 J` have *identical dynamics*, which is
itself a test — but it does not cancel out of the energy, and omitting it gives an energy wrong by
a lattice-dependent amount that still orders states correctly.

Potts contributes the other exact anchor: self-duality puts its critical point at
`T_c = J / ln(1 + √q)` for every `q`, and at `q = 2` it must agree with Onsager once the coupling
is doubled, since `δ(s, s') = (1 + s s')/2`.

### What the pictures need that the physics does not

Two additions exist purely so that a figure shows what is actually there.

**Equal-lightness palettes.** `hsv`, `tab10` and friends vary wildly in perceived lightness around
the wheel — their yellow reads as foreground and their blue as background — so a viewer sees
contrast that is an artifact of the palette. `palette.py` builds hue rings at fixed Oklab
lightness instead, where equal `L` really does read as equal lightness. Chroma is a separate
question: insisting on one chroma for the whole ring costs a lot of saturation, because the sRGB
gamut is lopsided and the whole ring gets throttled to whatever blue can manage, so
`oklch_ring(uniform_chroma=False)` gives each hue its own maximum. Lightness uniformity is the
claim that matters; chroma uniformity is a nicety.

**A phase field.** The intermediate phase of a `q ≥ 5` clock model is a long-wavelength
phenomenon: the direction winds slowly across the lattice while individual sites still fluctuate
hard between neighbouring states. At `T = 0.65 J`, `q = 8`, fewer than half the sites are in the
most popular state, so a raw microstate buries the winding in speckle. `ClockModel.phase_field`
averages the spin *vectors* over a sliding window and returns the local mean direction together
with its length. Reporting that length is what keeps this honest: where the spins are genuinely
uncorrelated the mean is short, and a renderer should desaturate rather than draw a confident
colour.

### What the test suite loses, and what replaces it

The strongest tests on the Ising side are structural: the exact half-sweep transition matrix on a
3×3 lattice is checked to be reversible with respect to the Boltzmann distribution, with no
sampling and no tolerance. That generalises to the clock and Potts models — `test_clock.py` builds
the same matrix over all 81 label configurations of a 2×2 lattice at `q = 3` — but it will **not**
generalise to XY or Heisenberg, whose state spaces are continuous and have no transition matrix to
enumerate. Those models will need closed-form single-site limits (the Langevin function for
Heisenberg, a ratio of Bessel functions for XY) to carry the same weight, which is worth knowing
before they are written rather than after.

## What is deliberately *not* changed

- **No Metropolis.** Heat-bath is already what the code does, and it is a fine algorithm; there
  is no reason to swap in a different acceptance rule.
- **No cluster algorithms.** Wolff or Swendsen–Wang would beat single-site heat-bath badly near
  $T_c$, where critical slowing down makes local updates crawl. That is a genuine future
  improvement, not a correction, and it needs the bond enumeration sketched in `plan.md`
  Appendix A.
- **Open boundaries remain the default**, so existing scripts keep their current geometry unless
  they ask otherwise.
- **The spin-resample formulation stays.** Heat-bath sets an absolute value rather than proposing
  a flip; that is why `Spin.flip` disappears. A flip primitive is what Metropolis and cluster
  moves need, and it is one line to reinstate if either is added.
