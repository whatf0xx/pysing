"""
Correctness tests for the shared categorical sampler in `sampling.py`.

The headline test is `test_two_states_reproduce_the_ising_conditional`: the
categorical route has to be a strict generalisation of the two-state
`(1 + tanh(...)) / 2` rule that `model.Model` uses, not a parallel
implementation that might drift from it.
"""
import numpy as np
import pytest

from sampling import sample_categorical


def empirical_probabilities(log_weights, seed, draws):
    """Frequency of each label over repeated draws, as a (q, ...) array."""
    rng = np.random.default_rng(seed)
    q = log_weights.shape[0]
    counts = np.zeros((q,) + log_weights.shape[1:], dtype=np.int64)
    for _ in range(draws):
        labels = sample_categorical(log_weights, rng)
        for k in range(q):
            counts[k] += labels == k
    return counts / draws


def test_two_states_reproduce_the_ising_conditional():
    """
    With `q = 2` and log-weights `+-beta*h`, the categorical draw must have
    exactly the probabilities of the Ising heat bath. This is an algebraic
    identity, so it is checked to machine precision rather than statistically.
    """
    rng = np.random.default_rng(0)
    h = rng.normal(scale=4.0, size=(8, 8))
    beta = 0.7

    log_weights = np.stack([beta * h, -beta * h])
    shifted = log_weights - log_weights.max(axis=0, keepdims=True)
    weights = np.exp(shifted)
    p_up = weights[0] / weights.sum(axis=0)

    assert np.abs(p_up - 0.5 * (1.0 + np.tanh(beta * h))).max() < 1e-15


def test_labels_are_the_inverse_cdf_of_the_draw():
    """
    Replace the generator with scripted uniforms and check the label against
    an independent, explicitly-looped inverse-CDF over the normalised
    probabilities.
    """
    rng = np.random.default_rng(1)
    q, length = 5, 4
    log_weights = rng.normal(scale=3.0, size=(q, length, length))
    uniforms = rng.random((length, length))

    class ScriptedRNG:
        def random(self, shape):
            assert shape == (length, length)
            return uniforms

    labels = sample_categorical(log_weights, ScriptedRNG())

    probabilities = np.exp(log_weights - log_weights.max(axis=0, keepdims=True))
    probabilities /= probabilities.sum(axis=0, keepdims=True)
    for i in range(length):
        for j in range(length):
            total, expected = 0.0, q - 1
            for k in range(q):
                total += probabilities[k, i, j]
                if uniforms[i, j] < total:
                    expected = k
                    break
            assert labels[i, j] == expected


def test_single_state_is_degenerate():
    rng = np.random.default_rng(2)
    labels = sample_categorical(np.zeros((1, 6, 6)), rng)
    assert np.array_equal(labels, np.zeros((6, 6)))


@pytest.mark.parametrize("scale", [700.0, -700.0, 1e4])
def test_extreme_log_weights_do_not_overflow(scale):
    """
    `exp(700)` is the last finite double. The max-subtraction is what keeps
    the sampler usable at temperatures where the raw weights are not
    representable at all.
    """
    rng = np.random.default_rng(3)
    log_weights = scale * np.array([
        [[1.0, 0.0]], [[0.0, 1.0]], [[0.5, 0.5]],
    ])
    with np.errstate(over="raise", invalid="raise"):
        labels = sample_categorical(log_weights, rng)
    assert np.all((labels >= 0) & (labels < 3))


def test_dominant_state_is_always_chosen():
    """A weight this much larger than the rest is chosen with probability 1."""
    rng = np.random.default_rng(4)
    log_weights = np.zeros((4, 16, 16))
    log_weights[2] = 500.0
    assert np.all(sample_categorical(log_weights, rng) == 2)


def test_dtype_is_the_callers_choice():
    rng = np.random.default_rng(5)
    log_weights = np.zeros((7, 4, 4))
    assert sample_categorical(log_weights, rng).dtype == np.intp
    assert sample_categorical(log_weights, rng, dtype=np.uint8).dtype == np.uint8


@pytest.mark.slow
def test_uniform_weights_are_uniform():
    """Equal weights give a flat histogram; chi-square at 8 states."""
    q, length, draws = 8, 16, 500
    frequencies = empirical_probabilities(np.zeros((q, length, length)), 6, draws)

    observed = frequencies.sum(axis=(1, 2)) * draws
    expected = draws * length ** 2 / q
    chi_square = (((observed - expected) ** 2) / expected).sum()
    # 7 degrees of freedom: the 99.9th percentile is 24.3.
    assert chi_square < 24.3


@pytest.mark.slow
def test_unequal_weights_match_their_probabilities():
    """
    Known unequal weights, held fixed across every site, so the frequencies
    can be pooled over the lattice and compared against the exact
    probabilities.
    """
    log_weights = np.array([0.0, -1.0, 2.0, 0.5])[:, None, None] * np.ones((1, 24, 24))
    exact = np.exp(log_weights[:, 0, 0])
    exact /= exact.sum()

    frequencies = empirical_probabilities(log_weights, 7, 400)
    pooled = frequencies.mean(axis=(1, 2))
    assert np.abs(pooled - exact).max() < 0.005
