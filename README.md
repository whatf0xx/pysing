# pysing

*A pure Python generator for lattice model microstates.*

## About

Are you studying a course in statistical mechanics and are sick of being told that the individual microstates don't matter,
that the bulk behaviour can be found analytically? Just want to look at some sweet scale-free patterns that you can tell yourself
have physical significance without melting your brain with a fractal dive? Teaching a course in statistical mechanics and in need
of some cute visualisations of the magic going on behind the master equation?

Then look no further, this is the repository for you!

### How it works

See [this blog entry](https://whatf0xx.github.io/simulation/physics/thermodynamics/ising/2024/10/21/pysing-model.html).

Spins are updated with the heat-bath (Glauber) rule,
`p(s_i = +1) = (1 + tanh(β(H + J Σ_nn s_j))) / 2`, applied to one sublattice of the
chessboard colouring at a time. One call to `evolve()` resamples every spin exactly
once — one sweep. See [`method.md`](method.md) for how this differs from the original
synchronous update, and why the difference matters.

Reduced units are used throughout: `k_B = 1` and the coupling `J` sets the energy scale,
so temperatures are in units of `J / k_B` and the Onsager critical point sits at
`T = 2.269 J`.

### Boundary conditions

`Model(..., boundary="open")` is the default: edge sites have three neighbours and
corners two, so a free surface suppresses order near the edges.

`boundary="periodic"` wraps the lattice instead, removing the surface entirely. This is
the usual choice for extracting `T_c`, critical exponents or correlation lengths. It
requires an **even** lattice length of at least 4: the wraparound closes odd-length
cycles when the length is odd, the lattice stops being bipartite, and the sublattice
update is no longer exact.

## Models with more than two states

`Model` is the Ising model: two states per site, rendered as a false-colour map of ±1. Two more
models put a *colour space* on the sites instead, so the picture is a direct rendering of the
microstate rather than a recolouring of it.

**`PottsModel`** — `q` states with no geometry between them, coupled only by whether neighbours
agree: `E = −J Σ δ(s_i, s_j)`. Every pair of states is equally distinct, so the colours are
arbitrary and every domain wall is equally hard. In exchange the critical point is exact for every
`q` by self-duality, at `T_c = J / ln(1 + √q)`, and the transition changes order at `q = 5`.

**`ClockModel`** — `q` states equally spaced on a circle, `E = −J Σ cos(θ_i − θ_j)` with
`θ_k = 2πk/q`. The state space *is* the hue wheel, so states close in energy are close in colour:
walls between adjacent hues come out soft and walls between opposite hues hard, with no tuning.
For `q ≥ 5` it has **three** phases separated by two Kosterlitz–Thouless transitions — a
low-temperature phase locked to the `q` discrete directions, an intermediate phase where the hue
winds smoothly and the overall direction is free to rotate, and a disordered phase.

Both are sampled with the same exact checkerboard heat bath as `Model`, and share its lattice
geometry through `Lattice`. `ClockModel(q=2)` is `Model` at the same coupling — not approximately,
identically — which is the strongest regression test in the suite. See [`method.md`](method.md).

Colours come from `palette.py`, which builds hue rings at constant Oklab lightness so that no
state is visually privileged. Stock colormaps are not built that way: at full saturation `hsv`
swings by more than a factor of ten in luminance around the wheel, which invents contrast the
physics does not have.

## Example

![Ising model microstates](https://github.com/whatf0xx/pysing/blob/main/example.png?raw=true)

## How to

```
git clone git@github.com:whatf0xx/pysing.git
cd pysing
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
python -m run     # the Ising relaxation demo
python demo.py    # the Potts and clock models, one panel per phase
```

`demo.py` pops up two windows: the 3-state Potts model either side of its exact critical point,
and the 8-state clock model across all three of its phases. `python demo.py --help` for lattice
size, sweep count, `q`, `--smoothed` to add a coarse-grained row under the clock microstates, and
`--save DIR` to write PNGs instead.

An ordered phase is degenerate — the `q` states are related by a symmetry, so which one a lattice
settles into is decided once and never revisited, and no amount of extra sweeps will move it. The
panels are therefore started in *different* states, chosen without replacement; starting them all
in state 0 would paint every ordered panel the same colour and make state 0 look special, which is
the exact artifact the equal-lightness palette exists to prevent. `--init anneal` instead cools in
from disorder and lets the dynamics do the choosing, which is honest but slow: it needs a few
thousand sweeps and stops working above about `L = 96`, because breaking the symmetry from a hot
start means coarsening domains, and that takes of order `L²` sweeps. See the `demo.py` docstring
for the measured numbers.

Popping up a window needs a GUI toolkit, which `matplotlib` does not bring with it — without one
it falls back to the `agg` backend, which renders fine but has nothing to render *into*, and
`plt.show()` warns and returns. `pip install PyQt6` if you hit that; the package is `PyQt6`, not
`pyqt`, and `pyside6` or the system `python3-tk` do just as well. It is deliberately not in
`requirements.txt`: it is an 86 MB download that only the demo needs, and `--save DIR` does not
need it at all.

### Tests

```
pip install -r requirements-dev.txt
pytest -m "not slow"   # ~5 s
pytest                 # ~40 s, adds the statistical checks
```
