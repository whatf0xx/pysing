"""
The one piece of *sampling* code shared between models.

The site conditionals of the discrete models here -- Potts, clock, and any
future multi-state model -- all reduce to "draw one label per site from a
`(q, L, L)` array of unnormalised log-weights". The weights themselves are
different in every case and are deliberately not abstracted; only the draw is
shared.
"""
import numpy as np


def sample_categorical(log_weights: np.ndarray,
                       rng: np.random.Generator,
                       dtype=np.intp) -> np.ndarray:
    """
    Draw one label per site from `(q, L, L)` unnormalised log-weights,
    returning an `(L, L)` array of labels in `0 .. q-1`.

    The weights are exponentials of energies and overflow if formed directly:
    at low temperature `beta * E` runs to hundreds. Subtracting the per-site
    maximum first is the same log-space discipline as `Model.log_weight`, and
    it is exact -- a common factor cancels out of a normalised categorical
    distribution.

    `dtype` is the accumulator and output dtype. Pass `np.uint8` when
    `q <= 256` to write straight into a `uint8` label array; the count being
    accumulated is at most `q - 1`, so it cannot overflow.

    Cost is O(q) per site. `np.cumsum` plus a comparison beats `searchsorted`
    here because the weights differ from site to site, so there is no shared
    table to search.
    """
    shifted = log_weights - log_weights.max(axis=0, keepdims=True)
    cdf = np.cumsum(np.exp(shifted), axis=0)
    draw = rng.random(cdf.shape[1:]) * cdf[-1]
    # The number of cumulative-weight bins strictly below the draw is the
    # index of the bin it lands in, i.e. inverse-CDF sampling done in
    # parallel over every site at once.
    return (cdf < draw).sum(axis=0, dtype=dtype)
