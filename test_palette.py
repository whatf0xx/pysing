"""
Correctness tests for the colour transforms in `palette.py`.

The transform constants are transcribed, and a transposed matrix would
produce colours that look plausible and are wrong. So the first tests here
are *anchors*: independently published Oklab values for specific sRGB
colours, which no amount of internally-consistent transcription error can
satisfy. Only after those do the round-trips and the uniformity claims run.
"""
import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")  # noqa: E402

from palette import (
    DEFAULT_LIGHTNESS,
    cyclic_colormap,
    full_circle_chroma,
    in_gamut,
    linear_srgb_to_oklab,
    linear_to_srgb,
    max_chroma_per_hue,
    max_uniform_chroma,
    oklab_to_linear_srgb,
    oklch_ring,
    oklch_to_srgb,
    render,
    srgb_to_linear,
)


# --------------------------------------------------------------------------
# Anchors: externally known values, not round-trips
# --------------------------------------------------------------------------

@pytest.mark.parametrize("srgb,oklab", [
    ((1.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((1.0, 0.0, 0.0), (0.6279, 0.2249, 0.1258)),
    ((0.0, 1.0, 0.0), (0.8664, -0.2339, 0.1795)),
    ((0.0, 0.0, 1.0), (0.4520, -0.0325, -0.3115)),
])
def test_oklab_anchors(srgb, oklab):
    """
    Published Oklab coordinates of the sRGB primaries and of white. White in
    particular pins the whole L row: any error in the LMS scaling shows up as
    a lightness that is not exactly 1.
    """
    computed = linear_srgb_to_oklab(srgb_to_linear(np.array(srgb)))
    assert np.allclose(computed, oklab, atol=5e-5)


def test_grey_axis_has_no_chroma():
    """
    Every neutral grey must land on a = b = 0. Exactly zero in exact
    arithmetic -- the `a` and `b` rows of the matrix sum to zero by
    construction -- but the published constants are rounded to ten places, so
    a grey lands within about 1e-8 of the axis rather than on it.
    """
    greys = np.linspace(0.0, 1.0, 17)[:, None] * np.ones(3)
    lab = linear_srgb_to_oklab(srgb_to_linear(greys))
    assert np.abs(lab[:, 1:]).max() < 1e-6
    assert np.all(np.diff(lab[:, 0]) > 0)


# --------------------------------------------------------------------------
# Round trips
# --------------------------------------------------------------------------

def test_transfer_function_round_trip():
    values = np.linspace(0.0, 1.0, 257)
    assert np.allclose(linear_to_srgb(srgb_to_linear(values)), values, atol=1e-12)
    assert np.allclose(srgb_to_linear(linear_to_srgb(values)), values, atol=1e-12)


def test_transfer_function_is_continuous_at_the_knee():
    """
    The sRGB curve switches from linear to a power law at 0.0031308. A wrong
    constant on either side leaves a visible step there; the standard's own
    rounded constants leave one of about 3e-8, far below a display's least
    significant bit.
    """
    below = linear_to_srgb(0.0031308 - 1e-12)
    above = linear_to_srgb(0.0031308 + 1e-12)
    assert abs(below - above) < 1e-6


def test_oklab_round_trip():
    """
    Forward and inverse agree to about 3e-7. That floor is set by the
    published matrices being rounded inverses of each other, not by anything
    in this module -- it is two orders of magnitude below an 8-bit colour
    step, so it is invisible, but it is why the tolerance is not 1e-12.
    """
    rng = np.random.default_rng(0)
    linear = rng.random((500, 3))
    assert np.allclose(
        oklab_to_linear_srgb(linear_srgb_to_oklab(linear)), linear, atol=1e-6
    )


# --------------------------------------------------------------------------
# The uniformity claims -- these are the reason the module exists
# --------------------------------------------------------------------------

@pytest.mark.parametrize("q", range(2, 17))
def test_ring_is_in_gamut(q):
    ring = oklch_ring(q)
    assert ring.shape == (q, 3)
    assert ring.min() >= 0.0 and ring.max() <= 1.0


@pytest.mark.parametrize("q", range(2, 17))
def test_ring_has_constant_lightness(q):
    """
    The claim is equal perceptual lightness, so the test transforms the
    finished sRGB colours *forward* into Oklab and checks the lightness
    there. Reading back the value that was passed in would prove nothing.
    """
    lab = linear_srgb_to_oklab(srgb_to_linear(oklch_ring(q)))
    assert np.allclose(lab[:, 0], DEFAULT_LIGHTNESS, atol=1e-6)


@pytest.mark.parametrize("q", range(2, 17))
def test_ring_hues_are_evenly_spaced_and_cyclic(q):
    """
    Hue steps are equal all the way round, *including* the step from the last
    entry back to the first. That last step is the one a sequential colormap
    gets wrong, and it is the one that matters for a Z_q state space.
    """
    lab = linear_srgb_to_oklab(srgb_to_linear(oklch_ring(q)))
    hues = np.arctan2(lab[:, 2], lab[:, 1])
    steps = np.diff(np.concatenate([hues, hues[:1]])) % (2 * np.pi)
    assert np.allclose(steps, 2 * np.pi / q, atol=1e-6)

    chroma = np.hypot(lab[:, 1], lab[:, 2])
    assert np.allclose(chroma, chroma[0], atol=1e-6)


@pytest.mark.parametrize("q", [3, 6, 12])
def test_ring_luminance_is_far_flatter_than_hsv(q):
    """
    The concrete complaint about `hsv`: at full saturation its luminance
    swings by more than an order of magnitude round the circle, so it invents
    contrast the physics does not have.

    sRGB relative luminance is a cruder model of lightness than Oklab's `L`,
    so an equal-`L` ring is not exactly equal-luminance -- it spreads by
    about 15 %. The claim being tested is the comparison, not flatness:
    `hsv` is an order of magnitude worse on its own preferred measure.
    `test_ring_has_constant_lightness` is where flatness itself is checked.
    """
    hues = np.linspace(0.0, 1.0, q, endpoint=False)
    hsv = matplotlib.colors.hsv_to_rgb(np.stack([hues, np.ones(q), np.ones(q)], -1))
    weights = np.array([0.2126, 0.7152, 0.0722])

    ring_spread = np.ptp(srgb_to_linear(oklch_ring(q)) @ weights)
    hsv_spread = np.ptp(srgb_to_linear(hsv) @ weights)
    assert hsv_spread > 10.0 * ring_spread


def test_phase_rotates_the_ring():
    plain = oklch_ring(6)
    rotated = oklch_ring(6, phase=2 * np.pi / 6)
    assert np.allclose(rotated[:-1], plain[1:], atol=1e-12)


# --------------------------------------------------------------------------
# The gamut search
# --------------------------------------------------------------------------

def test_max_uniform_chroma_is_maximal():
    """In gamut at the value returned, out of gamut just above it."""
    hues = np.linspace(0.0, 2 * np.pi, 64, endpoint=False)
    chroma = max_uniform_chroma(hues, DEFAULT_LIGHTNESS)
    assert np.all(in_gamut(DEFAULT_LIGHTNESS, chroma, hues))
    assert not np.all(in_gamut(DEFAULT_LIGHTNESS, chroma + 1e-3, hues))


def test_max_chroma_per_hue_is_maximal_everywhere():
    """
    Each hue's own ceiling, elementwise -- and it must be the ceiling, not
    merely a value that happens to be inside.
    """
    hues = np.linspace(0.0, 2 * np.pi, 97, endpoint=False).reshape(97, 1)
    chroma = max_chroma_per_hue(hues, DEFAULT_LIGHTNESS)
    assert chroma.shape == hues.shape
    assert np.all(in_gamut(DEFAULT_LIGHTNESS, chroma, hues))
    assert not np.any(in_gamut(DEFAULT_LIGHTNESS, chroma + 1e-3, hues))
    assert max_uniform_chroma(hues, DEFAULT_LIGHTNESS) == pytest.approx(chroma.min())


@pytest.mark.parametrize("q", [3, 6, 8, 12])
def test_per_hue_ring_is_more_saturated_but_still_equal_lightness(q):
    """
    The point of `uniform_chroma=False`: give up chroma uniformity, keep
    lightness uniformity, gain saturation. All three halves of that sentence
    are checked here.
    """
    uniform = oklch_ring(q)
    vivid = oklch_ring(q, uniform_chroma=False)
    assert vivid.min() >= 0.0 and vivid.max() <= 1.0

    uniform_lab = linear_srgb_to_oklab(srgb_to_linear(uniform))
    vivid_lab = linear_srgb_to_oklab(srgb_to_linear(vivid))
    assert np.allclose(vivid_lab[:, 0], DEFAULT_LIGHTNESS, atol=1e-6)

    uniform_chroma = np.hypot(uniform_lab[:, 1], uniform_lab[:, 2])
    vivid_chroma = np.hypot(vivid_lab[:, 1], vivid_lab[:, 2])
    assert np.all(vivid_chroma >= uniform_chroma - 1e-9)
    assert vivid_chroma.mean() > 1.05 * uniform_chroma.mean()


def test_full_circle_chroma_is_the_binding_constraint():
    """
    A ring on a few hues can be more chromatic than one that has to survive
    every hue, which is exactly why the two functions are separate.
    """
    circle = full_circle_chroma()
    triad = max_uniform_chroma(np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3]))
    assert triad > circle
    assert np.all(in_gamut(DEFAULT_LIGHTNESS, circle,
                           np.linspace(0, 2 * np.pi, 360, endpoint=False)))


def test_explicit_chroma_is_honoured():
    ring = oklch_ring(6, chroma=0.05)
    lab = linear_srgb_to_oklab(srgb_to_linear(ring))
    assert np.allclose(np.hypot(lab[:, 1], lab[:, 2]), 0.05, atol=1e-6)


def test_out_of_gamut_is_clipped_only_on_request():
    far = oklch_to_srgb(DEFAULT_LIGHTNESS, 0.6, 0.0, clip=False)
    assert far.max() > 1.0
    assert oklch_to_srgb(DEFAULT_LIGHTNESS, 0.6, 0.0).max() <= 1.0


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def test_render_indexes_the_palette():
    labels = np.array([[0, 1], [2, 0]])
    palette = oklch_ring(3)
    image = render(labels, palette)
    assert image.shape == (2, 2, 3)
    assert np.array_equal(image[0, 0], palette[0])
    assert np.array_equal(image[1, 0], palette[2])


def test_render_rejects_labels_outside_the_palette():
    with pytest.raises(ValueError):
        render(np.array([[0, 3]]), oklch_ring(3))


def test_cyclic_colormap_wraps():
    cmap = cyclic_colormap(samples=256)
    colours = cmap(np.linspace(0.0, 1.0, 256, endpoint=False))[:, :3]
    steps = np.linalg.norm(np.diff(colours, axis=0, append=colours[:1]), axis=1)
    # The wrap step is the last one; it must be no larger than the others.
    assert steps[-1] < 1.5 * steps[:-1].mean()


def test_ring_rejects_empty():
    with pytest.raises(ValueError):
        oklch_ring(0)
