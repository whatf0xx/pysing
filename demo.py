"""
Interactive demonstrations of the colour models.

    python demo.py            # both figures
    python demo.py potts      # 3-state Potts, either side of its exact T_c
    python demo.py clock      # 8-state clock, all three phases
    python demo.py --init hot # quench from disorder instead: see the difference
    python demo.py --save figures/     # write PNGs instead of popping windows

Each figure is one model at three temperatures, one panel per phase.

A note on the protocol, because it changes what the pictures mean. Panels
start from a *uniform* lattice and run at the target temperature. That is the
cheap way into the low-temperature phases: a quench from disorder has to
coarsen its domains, which takes of order L**2 sweeps, so on any lattice
worth looking at it freezes into a picture of its own history instead of the
phase. Above the transition the memory of the start is gone within a few
sweeps and the choice does not matter. Both were checked here rather than
assumed -- at every temperature in these figures except the Potts critical
point, hot and cold starts land on the same order parameter, and it stops
moving between 500 and 1500 sweeps.

The exception is real and is left in: at `T_c` the two starts *bracket* the
answer instead of meeting, however long they are run. That is critical
slowing down, and it is what cluster algorithms exist to fix. Run with
`--init hot` to watch it happen from the other side.
"""
import argparse
import os
import re
import textwrap

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from clock import ClockModel, XY_BKT_TEMPERATURE
from palette import DEFAULT_LIGHTNESS, max_chroma_per_hue, oklch_to_srgb
from potts import PottsModel


#: A dark figure background. The Oklch rings sit at a fixed lightness of
#: 0.72, bright enough to separate cleanly from this and not so bright as to
#: glare.
BACKGROUND = "#111214"
FOREGROUND = "#e8e6e3"
MUTED = "#9aa0a6"

#: Per-hue maximum chroma rather than one chroma for the whole ring: still
#: uniform in lightness, which is the property that does the work, but
#: markedly more saturated. See `palette.oklch_ring`.
PALETTE = {"uniform_chroma": False}


def run(model, sweeps, label):
    for _ in tqdm(range(sweeps), desc=label, leave=False):
        model.evolve()
    return model


def panel(ax, model, title, subtitle):
    ax.set_facecolor(BACKGROUND)
    model.plot_to_axes(ax, **PALETTE)
    ax.set_title(title, color=FOREGROUND, fontsize=11, pad=8)
    ax.set_xlabel(subtitle, color=MUTED, fontsize=9, labelpad=6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a3d42")


def phase_field_rgb(model, width):
    """
    Render a smoothed clock state: hue is the local mean direction, chroma
    is the local coherence, lightness is fixed.

    Fixing lightness and modulating chroma is the right way round. Lightness
    is what the eye reads as figure against ground, so letting it vary would
    manufacture structure; chroma reads as confidence, which is what the
    coherence measures. Where the spins are uncorrelated the local mean is
    short and the pixel renders grey, so a disordered lattice cannot
    masquerade as a smooth phase field.

    The hue map is continuous here rather than the q-step ring, because
    after averaging the direction is genuinely continuous.
    """
    angle, coherence = model.phase_field(width)
    chroma = max_chroma_per_hue(angle, DEFAULT_LIGHTNESS) * coherence
    return oklch_to_srgb(DEFAULT_LIGHTNESS, chroma, angle)


def swatch(fig, palette, label):
    """The model's palette as a strip, so the state space itself is visible."""
    ax = fig.add_axes([0.36, 0.125, 0.28, 0.021])
    ax.imshow(palette[None, :, :], aspect="auto", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(label, color=MUTED, fontsize=8, labelpad=4)
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a3d42")


#: A mathtext span. Used only to keep the wrapper's hands off it.
_MATHTEXT = re.compile(r"\$[^$]*\$")


def wrap(caption, columns=145):
    """
    Hard-wrap a caption. `matplotlib`'s own `wrap=True` measures against the
    figure width, which lets a centred line run past the axes and off the
    edge of the canvas.

    Spaces inside a `$...$` span are hidden from the wrapper first. Breaking
    a line inside mathtext leaves matplotlib unable to parse either half, and
    it renders the raw TeX instead of failing.
    """
    protected = _MATHTEXT.sub(lambda span: span.group().replace(" ", "\0"),
                              caption)
    return textwrap.fill(protected, columns).replace("\0", " ")


def figure(panels, suptitle, caption, palette, palette_label):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.8), facecolor=BACKGROUND)
    for ax, (model, title, subtitle) in zip(axes, panels):
        panel(ax, model, title, subtitle)
    fig.suptitle(suptitle, color=FOREGROUND, fontsize=14, y=0.965)
    fig.text(0.5, 0.052, wrap(caption), color=MUTED, fontsize=8.5,
             ha="center", va="center")
    fig.subplots_adjust(top=0.85, bottom=0.21, left=0.03, right=0.97, wspace=0.06)
    swatch(fig, palette, palette_label)
    return fig


def potts_figure(length, sweeps, seed, init, q=3):
    """
    The q-state Potts model either side of its exactly-known critical point.

    Potts states are a bare set with no geometry between them, so the hues
    are arbitrary and every domain wall is equally hard -- flat blocks of
    colour with nothing soft anywhere. What the model has instead is an exact
    transition temperature for every q, `T_c = J / ln(1 + sqrt(q))`, from
    self-duality.
    """
    critical = PottsModel(4, q=q, coupling=1.0).critical_temperature

    settings = [
        (0.85, "ordered"),
        (1.00, "critical"),
        (1.35, "disordered"),
    ]

    panels = []
    for ratio, name in settings:
        model = PottsModel(length, q=q, temperature=ratio * critical,
                           coupling=1.0, boundary="periodic", init=init,
                           seed=seed)
        run(model, sweeps, f"potts {name}")
        panels.append((
            model,
            f"{name}     $T = {ratio:.2f}\\,T_c$",
            f"order parameter $= {model.order_parameter:.2f}$,   "
            f"largest population $= {model.populations.max():.2f}$",
        ))

    return figure(
        panels,
        f"{q}-state Potts model, $L = {length}$,   "
        f"$T_c = J/\\ln(1+\\sqrt{{{q}}}) = {critical:.3f}\\,J$",
        f"{sweeps} sweeps from a {init} start. One state takes the lattice "
        "below $T_c$; above it, colour is uncorrelated beyond a couple of "
        "lattice spacings. The middle panel is the honest exception: at "
        "$T_c$ hot and cold starts bracket the order parameter rather than "
        "meeting, so it is a snapshot of the critical region and not an "
        f"equilibrium sample. The {q} colours are interchangeable -- Potts "
        "states have no ordering, so the palette only has to avoid "
        "privileging one, which an equal-lightness ring does and a stock "
        "colormap does not.",
        PottsModel(4, q=q).palette(**PALETTE),
        f"the {q} states (assignment is arbitrary)",
    )


def clock_figure(length, sweeps, seed, init, q=8):
    """
    The q-state clock model, which for `q >= 5` has *three* phases.

    Its state space is the hue wheel -- state k is the angle `2*pi*k/q`, and
    that is the same angle that picks its colour -- so the picture is a
    direct rendering of the microstate rather than a false-colour map of it.

    The temperatures straddle both Kosterlitz-Thouless transitions of the
    q = 8 model, near `0.4 J` and `0.9 J`. Neither is known exactly, which is
    why they are given as plain temperatures rather than as ratios:
    `ClockModel.critical_temperature` deliberately raises for `q >= 5` rather
    than return a number it cannot justify.
    """
    settings = [
        (0.35, "locked"),
        (0.65, "quasi-long-range"),
        (1.25, "disordered"),
    ]
    # Wide enough that uncorrelated spins average away -- a window of n**2
    # random directions has mean length about 1/n, so 13 puts the disordered
    # phase at a coherence the eye reads as grey -- and narrow enough to keep
    # the long-wavelength structure. Odd, so the window is centred.
    width = max(3, (length // 16) | 1)

    models = []
    for temperature, name in settings:
        model = ClockModel(length, q=q, temperature=temperature, coupling=1.0,
                           boundary="periodic", init=init, seed=seed)
        run(model, sweeps, f"clock {name}")
        models.append(model)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.6), facecolor=BACKGROUND)
    for column, (model, (temperature, name)) in enumerate(zip(models, settings)):
        top, bottom = axes[0, column], axes[1, column]
        panel(
            top, model,
            f"{name}     $T = {temperature:.2f}\\,J$",
            f"$|m| = {model.order_parameter:.2f}$,   "
            f"largest population $= {model.populations.max():.2f}$",
        )
        bottom.set_facecolor(BACKGROUND)
        bottom.imshow(phase_field_rgb(model, width), interpolation="nearest")
        bottom.set_xticks([])
        bottom.set_yticks([])
        bottom.set_xlabel(
            f"mean coherence $= {model.phase_field(width)[1].mean():.2f}$",
            color=MUTED, fontsize=9, labelpad=6,
        )
        for spine in bottom.spines.values():
            spine.set_edgecolor("#3a3d42")

    fig.suptitle(
        f"{q}-state clock model, $L = {length}$:   three phases, "
        "two Kosterlitz-Thouless transitions",
        color=FOREGROUND, fontsize=14, y=0.975,
    )
    fig.text(0.012, 0.70, "microstate", color=FOREGROUND, fontsize=10,
             rotation=90, ha="center", va="center")
    fig.text(0.012, 0.34, f"smoothed  ({width}x{width})",
             color=FOREGROUND, fontsize=10, rotation=90, ha="center", va="center")
    fig.text(
        0.5, 0.048,
        wrap(
            f"{sweeps} sweeps from a {init} start. The two ordered phases are "
            "not told apart by $|m|$, which is large in both -- the "
            "populations are. Locked, the lattice sits in a single state and "
            "hue is flat; quasi-long-range, hue winds smoothly through "
            f"neighbouring states, so several are occupied at once; "
            f"disordered, all {q} are equally likely. That winding is a "
            "long-wavelength feature and the top row buries it in site-level "
            "speckle, so the bottom row averages the spin vectors over a "
            "sliding window: hue is the local mean direction, chroma is the "
            "local coherence, and it collapses to grey where there is nothing "
            "coherent to report. The upper transition approaches the XY value "
            f"$T \\approx {XY_BKT_TEMPERATURE}\\,J$ from below as $q$ grows.",
            columns=170,
        ),
        color=MUTED, fontsize=8.5, ha="center", va="center",
    )
    fig.subplots_adjust(top=0.90, bottom=0.175, left=0.035, right=0.985,
                        wspace=0.06, hspace=0.16)
    swatch_ax = fig.add_axes([0.36, 0.122, 0.28, 0.013])
    swatch_ax.imshow(ClockModel(4, q=q).palette(**PALETTE)[None, :, :],
                     aspect="auto", interpolation="nearest")
    swatch_ax.set_xticks([])
    swatch_ax.set_yticks([])
    swatch_ax.set_xlabel(f"the {q} states, at their own angles on the hue wheel",
                         color=MUTED, fontsize=8, labelpad=4)
    for spine in swatch_ax.spines.values():
        spine.set_edgecolor("#3a3d42")
    return fig


def check_backend():
    """Fail before the simulation, not after it.

    With no GUI toolkit installed matplotlib falls back to `agg`, which draws
    perfectly well but has no window to draw into: `plt.show()` warns and
    returns. Since the figures take about a minute to generate, finding that
    out at the end is the worst possible time, so check up front.
    """
    from matplotlib.backends import BackendFilter, backend_registry
    if matplotlib.get_backend() in backend_registry.list_builtin(
            BackendFilter.INTERACTIVE):
        return
    raise SystemExit(
        f"matplotlib is using the non-interactive '{matplotlib.get_backend()}' "
        "backend, so there is no window to pop up.\n"
        "Install a GUI toolkit:  pip install PyQt6\n"
        "(the pip package is 'PyQt6', not 'pyqt'; 'pyside6' also works, as "
        "does the system python3-tk)\n"
        "Or use --save DIR to write PNGs instead."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Interactive demos of the Potts and clock models."
    )
    parser.add_argument("which", nargs="?", default="all",
                        choices=["all", "potts", "clock"])
    parser.add_argument("--size", type=int, default=192,
                        help="lattice length (default: 192)")
    parser.add_argument("--sweeps", type=int, default=500,
                        help="sweeps per panel (default: 500)")
    parser.add_argument("--init", default="cold", choices=["cold", "hot"],
                        help="starting configuration (default: cold; see the "
                             "module docstring for why)")
    parser.add_argument("--q", type=int, default=8,
                        help="number of clock states (default: 8)")
    parser.add_argument("--potts-q", type=int, default=3,
                        help="number of Potts states (default: 3)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save", metavar="DIR",
                        help="write PNGs here instead of showing windows")
    args = parser.parse_args()

    if not args.save:
        check_backend()

    figures = {}
    if args.which in ("all", "potts"):
        figures["potts"] = potts_figure(args.size, args.sweeps, args.seed,
                                        args.init, q=args.potts_q)
    if args.which in ("all", "clock"):
        figures["clock"] = clock_figure(args.size, args.sweeps, args.seed,
                                        args.init, q=args.q)

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        for name, fig in figures.items():
            path = os.path.join(args.save, f"{name}.png")
            fig.savefig(path, dpi=140, facecolor=BACKGROUND)
            print(f"wrote {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
