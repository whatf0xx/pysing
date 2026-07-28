# pysing

*A pure Python generator for Ising model microstates.*

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

## Example

![Ising model microstates](https://github.com/whatf0xx/pysing/blob/main/example.png?raw=true)

## How to

```
git clone git@github.com:whatf0xx/pysing.git
cd pysing
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
python -m run
```

### Tests

```
pip install -r requirements-dev.txt
pytest -m "not slow"   # ~1 s
pytest                 # ~20 s, adds the statistical checks
```
