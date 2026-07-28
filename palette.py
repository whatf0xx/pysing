"""
Perceptually uniform colour rings, for models whose state space is a circle.

The difference between a figure that pops and one that looks like clip art is
almost entirely whether the colours are equal in *perceptual lightness*.
`tab10`, `Set1`, `hsv` and friends are not: their yellow reads as foreground
and their blue as background, so a viewer sees structure that is an artifact
of the palette rather than of the physics. sRGB's own luminance weights make
the scale of the problem obvious -- pure blue carries about a fourteenth of
the luminance of pure green.

Oklab is built so that equal `L` reads as equal lightness, which is exactly
the property wanted, and Oklch is its polar form: `L` lightness, `C` chroma,
`h` hue angle. A ring of `q` hues at fixed `L` and `C` therefore privileges no
state, which is the right default for a model whose states are related by a
symmetry.

Transform constants are Bjorn Ottosson's; `test_palette.py` anchors them
against independently known values (sRGB white, sRGB red) rather than
trusting the transcription.
"""
import numpy as np


#: linear sRGB -> LMS, and its inverse, in Ottosson's Oklab.
_RGB_TO_LMS = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])
_LMS_TO_RGB = np.array([
    [4.0767416621, -3.3077115913, 0.2309699292],
    [-1.2684380046, 2.6097574011, -0.3413193965],
    [-0.0041960863, -0.7034186147, 1.7076147010],
])

#: cube-rooted LMS -> Oklab, and its inverse.
_LMS_TO_LAB = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
])
_LAB_TO_LMS = np.array([
    [1.0, 0.3963377774, 0.2158037573],
    [1.0, -0.1055613458, -0.0638541728],
    [1.0, -0.0894841775, -1.2914855480],
])

#: A lightness that leaves room for chroma on every hue. Much above this the
#: blues desaturate; much below, the yellows do.
DEFAULT_LIGHTNESS = 0.72


def _apply(matrix: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Apply a 3x3 matrix along the last axis of a `(..., 3)` array."""
    return values @ matrix.T


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Encode linear light with the sRGB transfer function."""
    linear = np.asarray(linear, dtype=float)
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.abs(linear) ** (1 / 2.4) * np.sign(linear) - 0.055,
    )


def srgb_to_linear(encoded: np.ndarray) -> np.ndarray:
    """Decode sRGB values to linear light."""
    encoded = np.asarray(encoded, dtype=float)
    return np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        ((np.abs(encoded) + 0.055) / 1.055) ** 2.4 * np.sign(encoded),
    )


def linear_srgb_to_oklab(linear: np.ndarray) -> np.ndarray:
    """`(..., 3)` linear sRGB to `(..., 3)` Oklab."""
    lms = _apply(_RGB_TO_LMS, np.asarray(linear, dtype=float))
    return _apply(_LMS_TO_LAB, np.cbrt(lms))


def oklab_to_linear_srgb(lab: np.ndarray) -> np.ndarray:
    """`(..., 3)` Oklab to `(..., 3)` linear sRGB, which may be out of gamut."""
    lms = _apply(_LAB_TO_LMS, np.asarray(lab, dtype=float)) ** 3
    return _apply(_LMS_TO_RGB, lms)


def oklch_to_oklab(lightness, chroma, hue) -> np.ndarray:
    """Polar to cartesian in the a-b plane; `hue` is in radians."""
    lightness, chroma, hue = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (lightness, chroma, hue))
    )
    return np.stack(
        [lightness, chroma * np.cos(hue), chroma * np.sin(hue)], axis=-1
    )


def oklch_to_srgb(lightness, chroma, hue, clip: bool=True) -> np.ndarray:
    """
    Oklch to gamma-encoded sRGB in `[0, 1]`. Out-of-gamut colours are clipped
    unless `clip=False`, which is what the gamut search below needs.
    """
    linear = oklab_to_linear_srgb(oklch_to_oklab(lightness, chroma, hue))
    if clip:
        linear = np.clip(linear, 0.0, 1.0)
    return linear_to_srgb(linear)


def in_gamut(lightness, chroma, hue, tolerance: float=1e-9) -> np.ndarray:
    """Whether each Oklch colour lands inside the sRGB cube."""
    linear = oklab_to_linear_srgb(oklch_to_oklab(lightness, chroma, hue))
    return np.all(
        (linear >= -tolerance) & (linear <= 1.0 + tolerance), axis=-1
    )


def max_chroma_per_hue(hues, lightness: float=DEFAULT_LIGHTNESS,
                       upper: float=0.5, steps: int=40) -> np.ndarray:
    """
    The largest in-gamut chroma at each hue, as an array shaped like `hues`.

    Bisection on "are all three channels inside [0, 1]", which is monotonic
    in chroma at fixed lightness and hue. The whole search is vectorised, so
    this is equally usable on the q hues of a palette and on a full image of
    per-pixel hues.
    """
    hues = np.asarray(hues, dtype=float)
    low = np.zeros(hues.shape)
    high = np.full(hues.shape, float(upper))
    for _ in range(steps):
        middle = 0.5 * (low + high)
        inside = in_gamut(lightness, middle, hues)
        low = np.where(inside, middle, low)
        high = np.where(inside, high, middle)
    return low


def max_uniform_chroma(hues, lightness: float=DEFAULT_LIGHTNESS,
                       upper: float=0.5, steps: int=40) -> float:
    """
    The largest chroma that keeps *every* hue in `hues` inside the sRGB
    gamut: the minimum over hues of `max_chroma_per_hue`.

    Taking each hue's own maximum instead gives a ring that is lumpy in
    chroma -- at fixed lightness the ceiling varies strongly with hue, blues
    running out well before yellows. Which of the two to use is a real trade
    and `oklch_ring` exposes both; see its docstring.
    """
    return float(max_chroma_per_hue(hues, lightness, upper, steps).min())


def oklch_ring(q: int, lightness: float=DEFAULT_LIGHTNESS,
               chroma: float | None=None, phase: float=0.0,
               uniform_chroma: bool=True) -> np.ndarray:
    """
    A `(q, 3)` sRGB array: `q` hues equally spaced round the Oklch circle at
    constant lightness, so no state is visually privileged.

    With `chroma=None` the chroma is chosen automatically, in one of two
    ways:

    - `uniform_chroma=True` (the default) gives every hue the *same* chroma,
      the largest that keeps all of them in gamut. Maximally conservative.
    - `uniform_chroma=False` gives every hue its *own* maximum. The ring is
      then no longer uniform in chroma, but it is still uniform in lightness,
      and it is markedly more vivid -- up to twice the chroma on the hues
      that the uniform ring has to throttle.

    The second is usually the better trade for a figure. Lightness is what
    the eye reads as figure-versus-ground, and holding it fixed is the whole
    point of using Oklch; chroma variation is far weaker as a cue, and the
    sRGB gamut is lopsided enough that insisting on uniform chroma costs a
    lot of saturation to buy very little.

    Pass an explicit `chroma` -- or `full_circle_chroma(lightness)` -- when
    rings of different `q` have to be directly comparable.

    `phase` rotates the ring, which is worth using to put a chosen hue on
    state 0.
    """
    if q < 1:
        raise ValueError(f"q must be at least 1, got {q}.")
    hues = 2.0 * np.pi * np.arange(q) / q + phase
    if chroma is None:
        if uniform_chroma:
            chroma = max_uniform_chroma(hues, lightness)
        else:
            chroma = max_chroma_per_hue(hues, lightness)
    return oklch_to_srgb(lightness, chroma, hues)


def full_circle_chroma(lightness: float=DEFAULT_LIGHTNESS,
                       samples: int=720) -> float:
    """
    The largest chroma in gamut for *every* hue, not just the ones a
    particular ring happens to land on. Use this to make rings of different
    `q` -- or a ring and a continuous colormap -- mutually comparable.
    """
    return max_uniform_chroma(
        np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False), lightness
    )


def cyclic_colormap(lightness: float=DEFAULT_LIGHTNESS,
                    chroma: float | None=None, samples: int=512):
    """
    A `matplotlib` colormap for a continuous angle, wrapping smoothly from
    `2*pi` back to `0`.

    Cyclic is not a stylistic preference here: for the clock and XY models
    state `q-1` really is adjacent to state `0`, and a sequential colormap
    would draw a seam across a system that has none.
    """
    from matplotlib.colors import ListedColormap

    if chroma is None:
        chroma = full_circle_chroma(lightness)
    hues = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    return ListedColormap(oklch_to_srgb(lightness, chroma, hues), name="oklch")


def render(labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """
    `(L, L)` integer labels and a `(q, 3)` palette to an `(L, L, 3)` image.

    Kept separate from any plotting call so the same render can be written to
    axes now and pushed into an animation frame later.
    """
    palette = np.asarray(palette, dtype=float)
    if labels.max(initial=0) >= len(palette):
        raise ValueError(
            f"label {int(labels.max())} is outside a palette of {len(palette)}."
        )
    return palette[labels]
