"""
Interactive demonstrations of the colour models.

    python demo.py             # both figures
    python demo.py potts       # 3-state Potts, either side of its exact T_c
    python demo.py clock       # 8-state clock, all three phases
    python demo.py --smoothed  # add the coarse-grained clock row
    python demo.py --init anneal --size 96   # cool in, and let the model choose
    python demo.py --save figures/           # write PNGs, don't pop windows

Each figure is one model at three temperatures, one panel per phase.

A note on the protocol, because it decides what the pictures mean.

An ordered phase is *degenerate*. The q states are related by a symmetry, so
each is as good a ground state as the next, and which one a given lattice
settles into is not something more sweeps will change -- that is exactly what
spontaneous symmetry breaking is. Somebody has to choose. Panels here start
from a uniform lattice and run at the target temperature, which chooses by
hand and is the cheap way in: a quench from disorder has to coarsen its
domains, which takes of order L**2 sweeps, so on a lattice worth looking at
it freezes into a picture of its own history instead of the phase.

The state each panel is started in is drawn without replacement rather than
being state 0 every time. That is the whole reason the panels come out in
different colours. Starting them all in state 0 -- the earlier behaviour --
made every ordered panel pink, which reads as "state 0 is special" and is
precisely the artifact the equal-lightness palette exists to prevent. Above
the transition none of it matters; memory of the start is gone within a few
sweeps, which was checked rather than assumed.

`--init anneal` does it the honest way instead: start disordered, cool
geometrically through the transition over the first half of the sweeps, and
let the dynamics break the symmetry. Whether that finishes is a question
about coarsening, and the answer was measured rather than guessed -- Potts
q = 3 into `0.85 T_c`, order parameter after annealing:

    L    500 sweeps   2000 sweeps   8000 sweeps
    48      0.12         0.95          0.94
    96      0.45         0.95          0.95
    192     0.17         0.12          0.23

So it works up to about L = 96 given a few thousand sweeps, and at the
default L = 192 it does not work at all -- 8000 sweeps take 70 s a panel and
still leave three domains. Use it at `--size 96 --sweeps 2000` to watch a
model pick its own ground state; at the default size it shows coarsening,
which is worth seeing once but is not a picture of a phase.

One further exception is real and is left in: at `T_c` hot and cold starts
*bracket* the answer instead of meeting, however long they are run. That is
critical slowing down, and it is what cluster algorithms exist to fix.
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


def starting_states(q, panels, seed):
    """
    One state per panel to start a cold run in: `panels` of the q states,
    spread as evenly round them as q allows, from a random offset.

    An ordered phase is degenerate, so which of the q states a panel sits in
    is an input, not a result. Starting every panel in state 0 -- what a bare
    `init="cold"` does -- therefore paints every ordered panel in state 0's
    colour, which is exactly the "one state looks special" artifact the
    equal-lightness palette exists to avoid. Choosing a different state per
    panel costs nothing and shows the degeneracy instead of hiding it.

    Spread rather than a random permutation, because for the clock model the
    labels *are* the hue wheel: neighbouring labels are neighbouring colours,
    and a permutation will cheerfully hand two panels states 0 and 1, which
    read as the same colour. Stepping by q/panels puts them as far apart on
    the wheel as there is room for. The offset is the only random part, and
    it is what stops state 0 being the one that always turns up.
    """
    offset = np.random.default_rng(seed).integers(q)
    return (offset + np.arange(panels) * q // panels) % q


def protocol_note(init):
    """
    One caption sentence saying where an ordered panel's colour came from,
    since the three protocols answer that completely differently and the
    figure would otherwise claim whichever one was written down.
    """
    if init == "cold":
        return ("Which state an ordered panel lands in is degenerate and is "
                "set by its start, so the three are started in different "
                "states on purpose.")
    if init == "anneal":
        return ("Each panel was cooled in from disorder, so the state it "
                "locked into was chosen by the dynamics rather than handed "
                "to it -- if it finished coarsening.")
    return ("Each panel was quenched from disorder, so an ordered one shows "
            "domains still coarsening rather than a settled phase.")


def run(model, sweeps, label, init, state=0, hot=None):
    """
    Equilibrate `model` at its own temperature and leave it there.

    Three protocols, because the choice changes what the picture means:

    - `cold` fills the lattice with `state` and runs at the target. Fast, and
      an equilibrium sample *within* one symmetry sector, but the sector was
      chosen by hand.
    - `hot` quenches: a random lattice dropped straight to the target.
    - `anneal` starts random and cools geometrically from `hot` into the
      target over the first half of the sweeps, so the dynamics break the
      symmetry rather than being handed the answer. This is the honest
      protocol and the slow one -- see the module docstring for the sweeps
      it needs at a given lattice size.

    The cold fill is done here rather than through the model's own
    `init="cold"` because that always means state 0, and choosing the state
    is the entire point.
    """
    target = model.T
    schedule = np.full(sweeps, target)
    if init == "cold":
        model.labels.fill(state)
    elif init == "anneal" and sweeps:
        cooling = max(1, sweeps // 2)
        schedule[:cooling] = np.geomspace(max(hot, target), target, cooling)

    for temperature in tqdm(schedule, desc=label, leave=False):
        model.T = float(temperature)
        model.evolve()
    model.T = target
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


def swatch(fig, palette, label, rectangle):
    """The model's palette as a strip, so the state space itself is visible."""
    ax = fig.add_axes(rectangle)
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


#: Geometry by row count. A two-row figure is two thirds taller, so the same
#: fractional margins would not read as the same white space.
LAYOUT = {
    1: dict(figsize=(13.5, 5.8), suptitle_y=0.965, columns=145,
            adjust=dict(top=0.85, left=0.03, right=0.97, wspace=0.06)),
    2: dict(figsize=(13.5, 9.6), suptitle_y=0.975, columns=170,
            adjust=dict(top=0.90, left=0.035, right=0.985, wspace=0.06,
                        hspace=0.16)),
}

#: The strip below the panels, in points, stacked from the bottom edge up:
#: caption, then the swatch's label, then the swatch itself, then the panels'
#: own labels. Measured rather than pinned to fractions of the figure height,
#: because the caption changes length with `--q`, the protocol and the sweep
#: count -- at a fixed anchor a fifth line either clips off the canvas or
#: shoves the swatch through the panel labels, and both happened.
CAPTION_SIZE = 8.5
CAPTION_LEADING = 1.25
CAPTION_MARGIN = 7.0
SWATCH_LABEL_SPACE = 12.0
SWATCH_HEIGHT = 9.0
PANEL_LABEL_SPACE = 21.0


def figure(panels, suptitle, caption, palette, palette_label, second_row=None):
    """
    A row of microstates, titled and captioned, with the palette below it.

    `second_row` is an optional `(axes, model) -> None` drawing a derived
    view of the same three models underneath.
    """
    rows = 1 if second_row is None else 2
    layout = LAYOUT[rows]
    points = 72.0 * layout["figsize"][1]

    text = wrap(caption, layout["columns"])
    lines = text.count("\n") + 1
    swatch_bottom = (CAPTION_MARGIN + lines * CAPTION_SIZE * CAPTION_LEADING
                     + SWATCH_LABEL_SPACE)
    panels_bottom = swatch_bottom + SWATCH_HEIGHT + PANEL_LABEL_SPACE

    fig, axes = plt.subplots(rows, 3, figsize=layout["figsize"],
                             facecolor=BACKGROUND, squeeze=False)
    for column, (model, title, subtitle) in enumerate(panels):
        panel(axes[0, column], model, title, subtitle)
        if second_row is not None:
            second_row(axes[1, column], model)
    fig.suptitle(suptitle, color=FOREGROUND, fontsize=14, y=layout["suptitle_y"])
    fig.subplots_adjust(bottom=panels_bottom / points, **layout["adjust"])
    fig.text(0.5, CAPTION_MARGIN / points, text, color=MUTED,
             fontsize=CAPTION_SIZE, ha="center", va="bottom")
    swatch(fig, palette, palette_label,
           [0.36, swatch_bottom / points, 0.28, SWATCH_HEIGHT / points])
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
    for (ratio, name), state in zip(settings,
                                    starting_states(q, len(settings), seed)):
        model = PottsModel(length, q=q, temperature=ratio * critical,
                           coupling=1.0, boundary="periodic", init="hot",
                           seed=seed)
        run(model, sweeps, f"potts {name}", init, state=state,
            hot=2.0 * critical)
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
        f"{sweeps} sweeps, {init}. One state takes the lattice below $T_c$; "
        "above it, colour is uncorrelated beyond a couple of lattice "
        f"spacings. The {q} colours are interchangeable -- Potts states have "
        f"no ordering. {protocol_note(init)} The middle panel is the honest "
        "exception: at $T_c$ hot and cold starts bracket the order parameter "
        "rather than meeting, so it is a snapshot of the critical region and "
        "not an equilibrium sample.",
        PottsModel(4, q=q).palette(**PALETTE),
        f"the {q} states (assignment is arbitrary)",
    )


def clock_figure(length, sweeps, seed, init, q=8, smoothed=False):
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

    `smoothed` adds a second row of coarse-grained fields under the
    microstates. It is off by default: it trades away every bit of site-level
    detail for one long-wavelength feature, which is a bad trade unless that
    feature is what you came for.
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

    panels = []
    for (temperature, name), state in zip(
            settings, starting_states(q, len(settings), seed)):
        model = ClockModel(length, q=q, temperature=temperature, coupling=1.0,
                           boundary="periodic", init="hot", seed=seed)
        run(model, sweeps, f"clock {name}", init, state=state,
            hot=2.0 * XY_BKT_TEMPERATURE)
        panels.append((
            model,
            f"{name}     $T = {temperature:.2f}\\,J$",
            f"$|m| = {model.order_parameter:.2f}$,   "
            f"largest population $= {model.populations.max():.2f}$",
        ))

    def phase_field_row(ax, model):
        ax.set_facecolor(BACKGROUND)
        ax.imshow(phase_field_rgb(model, width), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"mean coherence $= {model.phase_field(width)[1].mean():.2f}$",
                      color=MUTED, fontsize=9, labelpad=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#3a3d42")

    caption = (
        f"{sweeps} sweeps, {init}. The two ordered phases are not told apart "
        "by $|m|$, which is large in both -- the populations are. Locked, the "
        "lattice sits in a single state and hue is flat; quasi-long-range, "
        "hue winds smoothly through neighbouring states, so several are "
        f"occupied at once; disordered, all {q} are equally likely. "
        f"{protocol_note(init)} The upper transition approaches the XY value "
        f"$T \\approx {XY_BKT_TEMPERATURE}\\,J$ from below as $q$ grows."
    )
    if smoothed:
        caption += (
            " The winding is long-wavelength and the top row buries it in "
            "site-level speckle, so the bottom row averages the spin vectors "
            "over a sliding window: hue is the local mean direction, chroma "
            "is the local coherence, and it collapses to grey where there is "
            "nothing coherent to report."
        )

    fig = figure(
        panels,
        f"{q}-state clock model, $L = {length}$:   three phases, "
        "two Kosterlitz-Thouless transitions",
        caption,
        ClockModel(4, q=q).palette(**PALETTE),
        f"the {q} states, at their own angles on the hue wheel",
        second_row=phase_field_row if smoothed else None,
    )
    if smoothed:
        fig.text(0.012, 0.70, "microstate", color=FOREGROUND, fontsize=10,
                 rotation=90, ha="center", va="center")
        fig.text(0.012, 0.34, f"smoothed  ({width}x{width})", color=FOREGROUND,
                 fontsize=10, rotation=90, ha="center", va="center")
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
    parser.add_argument("--init", default="cold",
                        choices=["cold", "hot", "anneal"],
                        help="equilibration protocol (default: cold; see the "
                             "module docstring for why)")
    parser.add_argument("--smoothed", action="store_true",
                        help="add a coarse-grained row under the clock "
                             "microstates")
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
                                        args.init, q=args.q,
                                        smoothed=args.smoothed)

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
