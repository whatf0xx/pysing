"""
Square-lattice geometry, shared by every model in the package.

The organising idea of this package is that a lattice model splits into two
halves that have nothing to say to each other: the *geometry* -- which sites
are neighbours, how the boundary closes, which sites may be updated together
-- and the *site alphabet* -- what lives on a site and how it is resampled.
The first half is identical for the Ising, Potts, clock, XY and Heisenberg
models. This module is that half, and it knows nothing about spins.
"""
import numpy as np


BOUNDARIES = ("open", "periodic")


class Lattice:
    """
    A square lattice of `lattice_length` sites in each direction.

    Boundaries are open by default; pass `boundary="periodic"` for wraparound,
    which requires an even lattice length of at least 4 so that the lattice
    stays bipartite and no bond is double counted.

    `neighbour_sum` works on any `(..., L, L)` array, so a scalar Ising spin,
    a `(2, L, L)` clock vector and a `(q, L, L)` one-hot occupancy all go
    through one code path. That is the whole point of the class: the
    four-line halo fill below is exactly the kind of block where a
    copy-pasted bug would live undetected.
    """
    def __init__(self, lattice_length: int, boundary: str="open"):
        if lattice_length < 1:
            raise ValueError(
                f"lattice_length must be at least 1, got {lattice_length}."
            )
        if boundary not in BOUNDARIES:
            raise ValueError(
                f"boundary must be one of {BOUNDARIES}, got {boundary!r}."
            )
        if boundary == "periodic":
            if lattice_length < 3:
                raise ValueError(
                    "Periodic boundaries need lattice_length >= 3; below that a "
                    "site is its own neighbour twice over and every bond is "
                    f"double counted (got {lattice_length})."
                )
            if lattice_length % 2 != 0:
                raise ValueError(
                    "Periodic boundaries need an even lattice_length: the wrap "
                    "closes odd-length cycles, the lattice stops being "
                    "bipartite, and the checkerboard update is no longer exact "
                    f"(got {lattice_length})."
                )

        self.lattice_length = lattice_length
        self.boundary = boundary

        # Checkerboard colouring: every black site's neighbours are all white.
        rows, cols = np.indices((lattice_length, lattice_length))
        self.black = (rows + cols) % 2 == 0
        self.white = ~self.black

        # Scratch halo buffers, one per (leading shape, dtype) a caller asks
        # for, allocated on first use. A single Lattice may serve `(L, L)`
        # int8 spins and `(2, L, L)` float64 vectors in the same run.
        self._pads = {}

    @property
    def n_sites(self) -> int:
        return self.lattice_length ** 2

    @property
    def n_bonds(self) -> int:
        """
        The number of nearest-neighbour bonds, counted once each. Needed by
        the models whose energy zero shifts with the bond count -- the Potts
        and clock delta/cosine relation is one such (see `clock.ClockModel`).
        """
        length = self.lattice_length
        if self.boundary == "periodic":
            return 2 * length ** 2
        return 2 * length * (length - 1)

    @property
    def sublattices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        The two halves of the checkerboard, in update order. Every model's
        `evolve` loops over this and resamples one half at a time: the
        lattice is bipartite, so each half is conditionally independent given
        the other, and the two half-sweeps together leave the Boltzmann
        distribution invariant. Updating *all* sites from the same stale
        state would not -- that is parallel dynamics, which breaks detailed
        balance.
        """
        return (self.black, self.white)

    def _pad_for(self, field: np.ndarray) -> np.ndarray:
        """
        A zero-initialised halo buffer matching `field`'s leading shape and
        dtype, allocated once per distinct request and reused thereafter.

        The open-boundary halo must stay zero across calls. `neighbour_sum`
        only ever writes the interior, and `boundary` is fixed for the
        lifetime of a Lattice, so a buffer that starts zeroed stays zeroed.
        That invariant is what lets open and periodic boundaries share a
        single code path.
        """
        key = (field.shape[:-2], field.dtype)
        if key not in self._pads:
            length = self.lattice_length
            self._pads[key] = np.zeros(
                field.shape[:-2] + (length + 2, length + 2), dtype=field.dtype
            )
        return self._pads[key]

    def neighbour_sum(self, field: np.ndarray) -> np.ndarray:
        """
        Sum the four nearest-neighbour values of `field` at every site.

        `field` is `(..., L, L)`; the result has the same shape and dtype.
        Absent neighbours contribute zero, so open boundaries fall out of a
        zero halo and periodic ones out of a wrapped halo.

        The dtype is deliberately the caller's problem and nothing is
        promoted here: `int8` holds Ising sums (range [-4, 4]) and one-hot
        occupancy counts (range [0, 4]), while the vector models pass float64.
        """
        if field.shape[-2:] != (self.lattice_length,) * 2:
            raise ValueError(
                f"field must be (..., {self.lattice_length}, "
                f"{self.lattice_length}), got shape {field.shape}."
            )
        pad = self._pad_for(field)
        pad[..., 1:-1, 1:-1] = field
        if self.boundary == "periodic":
            pad[..., 0, 1:-1] = field[..., -1, :]
            pad[..., -1, 1:-1] = field[..., 0, :]
            pad[..., 1:-1, 0] = field[..., :, -1]
            pad[..., 1:-1, -1] = field[..., :, 0]
        # The halo corners are never read.
        return (pad[..., :-2, 1:-1] + pad[..., 2:, 1:-1]
                + pad[..., 1:-1, :-2] + pad[..., 1:-1, 2:])

    def bonds(self):
        """
        Yield `(rows_a, cols_a, rows_b, cols_b)` index arrays, one tuple per
        bond direction, covering every bond exactly once.

        `neighbour_sum` keeps this enumeration implicit in its slice shifts,
        which is all the samplers need. The explicit form is for observables
        that resolve individual bonds -- correlation functions, bond-energy
        histograms -- and for the edge set of a cluster update.
        """
        length = self.lattice_length
        rows, cols = np.indices((length, length))
        if self.boundary == "periodic":
            yield rows, cols, rows, (cols + 1) % length
            yield rows, cols, (rows + 1) % length, cols
        else:
            yield rows[:, :-1], cols[:, :-1], rows[:, 1:], cols[:, 1:]
            yield rows[:-1, :], cols[:-1, :], rows[1:, :], cols[1:, :]

    def __repr__(self) -> str:
        return (f"{type(self).__name__}({self.lattice_length}, "
                f"boundary={self.boundary!r})")
