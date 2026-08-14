#!/usr/bin/env python3
r"""
kelp_lowpass.py -- a kelp forest, 16:9, with the sea sorting itself by depth.

THE CLAIM
    The top of a kelp forest feels every wave. The bottom feels almost only
    the longest one. Deep water is a low-pass filter, and the filtering is
    not a metaphor -- it falls straight out of the wave equations.

WHY WIDESCREEN IS THE POINT, NOT A CROP
    @Strigon-4WarrenalsoRivulet asked for the kelp in widescreen, twice.
    A crop would be a worse portrait piece. What a wide frame actually buys
    is HORIZONTAL ROOM, and a wave is a thing that travels horizontally.
    At this scale the frame holds ~2 wavelengths of the wind sea, so you can
    watch a wave cross. The old 1080x1920 version showed 0.55 of one, which
    is why its swell looked like the whole forest breathing in unison. It
    wasn't a swell. It was a pulse. This version has room for the wave to
    actually go somewhere, and a kelp forest with a surface canopy is a
    horizontal object anyway -- 16:9 is the shape this subject wanted.

THE PHYSICS, WHICH IS DOING ALL THE WORK
    Finite depth, not deep water. Kelp lives in 10-30 m and a 9 s swell has
    a 100+ m wavelength, so the "deep water" shortcut (depth > lambda/2) is
    simply false here and would give the wrong answer. Full relation:

        omega^2 = g k tanh(k h)

    solved numerically per component. A water particle at height zeta above
    the bed traces an ELLIPSE with semi-axes

        A_x = a cosh(k zeta) / sinh(k h)        (horizontal)
        A_z = a sinh(k zeta) / sinh(k h)        (vertical)

    Two things fall out of that, and they are the whole piece:

    1. At the bed, zeta = 0, so A_z is EXACTLY zero and A_x is not. The
       orbits flatten from circles at the surface into pure back-and-forth
       lines on the bottom. The water down there has nowhere to go but
       sideways.
    2. A_x at the bed is a / sinh(k h). sinh blows up fast, so a short wave
       (big k) is annihilated before it gets down and a long wave is barely
       touched. At 20 m the 9 s swell keeps 47% of its surface sway and the
       3.6 s chop keeps 0.7%.

    Marine snow is drawn honestly too: real aggregates sink of order tens of
    metres per DAY, which is under a centimetre in this whole clip. So the
    flecks do not fall. They orbit and come back, which is also why the loop
    closes perfectly.

WHAT TO WATCH (the description says this too)
    One coral fleck near the surface, then one near the bed. The top one
    goes round. The bottom one slides on a line.

SEAMLESS
    Every period divides the run exactly: 18.0 s = 2 x 9.0 = 4 x 4.5
    = 5 x 3.6. Last frame and first frame are the same water.

Colourway: jade kelp and hot coral snow on near-black teal. Fresh -- the
2026-08-02 kelp was warm ochre on cold slate, and warm-on-dark is spent.

    python3 pieces/kelp_lowpass.py check
    python3 pieces/kelp_lowpass.py still
    python3 pieces/kelp_lowpass.py
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asciilib import Grid, Frame, Encoder, INK          # noqa: E402

OUT = "/tmp/kelp_lowpass.mp4"
FPS = 30
DUR = 18.0
FRAMES = int(round(FPS * DUR))

G = 9.81
DEPTH = 20.0                    # m of water. giant kelp: 10-30 m.
FREEBOARD = 3.4                 # m of air kept above still water
BELOW = 1.6                     # m of seabed kept in frame below the holdfasts

# (period s, amplitude m). Every period divides DUR exactly -> seamless.
WAVES = [(9.0, 1.10),           # ground swell, feels the bottom
         (4.5, 0.70),           # wind sea
         (3.6, 0.40)]           # chop

N_STIPE = 15
OVERGROW = 4.2                  # m of stipe beyond the surface -> canopy mat
FROND_EVERY = 0.62              # m of stipe between blades
FROND_LEN = 1.35                # m
N_SNOW = 420

K_ATTEN = 0.155                 # 1/m, diffuse attenuation of daylight

# ------------------------------------------------------------------ colour --
AIR = (0.045, 0.043, 0.070)     # dark violet slate
SEA_TOP = (0.055, 0.148, 0.152)
SEA_BED = (0.014, 0.042, 0.058)
SURF = (1.00, 0.88, 0.74)       # pale peach, the lit surface
KELP_LIT = (0.44, 0.88, 0.67)   # jade
KELP_DEEP = (0.17, 0.46, 0.52)
SNOW = (1.00, 0.47, 0.34)       # hot coral
BEDROCK = (0.15, 0.23, 0.26)


def solve_k(period, h):
    """omega^2 = g k tanh(k h), by bisection.

    The deep-water value w^2/g is a lower BOUND, not an upper one: tanh < 1
    always, so finite depth needs a LARGER k than deep water does, and a
    shorter wavelength. Bracketing the other way round (which I did first)
    silently pins every component at its deep-water value and quietly
    deletes the entire subject of the piece.
    """
    w = 2.0 * math.pi / period
    lo = w * w / G
    hi = lo
    while G * hi * math.tanh(hi * h) < w * w:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if G * mid * math.tanh(mid * h) < w * w:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class Sea(object):
    """The wave field. Ask it where a bit of water is at time t."""

    def __init__(self, waves, h):
        self.h = h
        self.w = [(2.0 * math.pi / T, solve_k(T, h), a) for T, a in waves]

    def orbit(self, zeta):
        """Semi-axes (ax, az) per component at height zeta above the bed."""
        out = []
        for om, k, a in self.w:
            s = math.sinh(k * self.h)
            out.append((a * np.cosh(k * zeta) / s,
                        a * np.sinh(k * zeta) / s))
        return out

    def move(self, x, zeta, t):
        """Displacement (dx, dz) of water at mean position (x, zeta)."""
        dx = np.zeros_like(np.asarray(x, float))
        dz = np.zeros_like(dx)
        for (om, k, a), (ax, az) in zip(self.w, self.orbit(zeta)):
            ph = k * x - om * t
            dx = dx + ax * np.sin(ph)
            dz = dz + az * np.cos(ph)
        return dx, dz

    def eta(self, x, t):
        """Surface elevation."""
        e = np.zeros_like(np.asarray(x, float))
        for om, k, a in self.w:
            e = e + a * np.cos(k * x - om * t)
        return e


class View(object):
    """World metres <-> character cells. Square cells, so no exaggeration."""

    def __init__(self, g):
        self.g = g
        self.span = DEPTH + FREEBOARD + BELOW
        self.mpr = self.span / g.rows              # metres per row
        self.width = g.cols * self.mpr
        self.row0 = FREEBOARD / self.mpr           # row of still water

    def col(self, x):
        return x / self.mpr

    def row(self, y):                              # y measured up from water
        return self.row0 - y / self.mpr


def build_stipes(view):
    """Holdfast x, arc length, and how far back in the forest it stands.

    Evenly spaced stipes at one distance read as a comb, or worse, as a
    chain-link fence once the blades go on. What makes it a forest is that
    some of them are NEARER than others: near ones brighter, fatter and
    with longer blades, far ones sunk toward the colour of the water.
    A wide frame is what gives that anywhere to happen.
    """
    rng = np.random.default_rng(7)
    xs = np.linspace(-1.5, view.width + 1.5, N_STIPE)
    xs = xs + rng.uniform(-1.1, 1.1, N_STIPE)
    lens = DEPTH + OVERGROW * rng.uniform(0.5, 1.0, N_STIPE)
    far = rng.uniform(0.0, 1.0, N_STIPE) ** 0.8       # 0 near, 1 far
    order = np.argsort(-far)                          # paint back to front
    return xs[order], lens[order], far[order]


def stipe_points(x0, length, n=170):
    """Mean shape: straight up to the surface, then lying along it.

    Returns (x, zeta, s). zeta is height above the bed; the canopy runs
    off horizontally in the direction the plant happens to lean.
    """
    s = np.linspace(0.0, length, n)
    up = np.minimum(s, DEPTH)
    over = np.maximum(s - DEPTH, 0.0)
    side = 1.0 if (int(round(x0 * 7.0)) % 2) else -1.0
    return x0 + side * over, up, s


def check():
    g = Grid(1920, 1080, font_size=28)
    v = View(g)
    sea = Sea(WAVES, DEPTH)
    print(g)
    print("view: %.1f m wide x %.1f m tall, %.3f m per cell"
          % (v.width, v.span, v.mpr))

    surf = sea.orbit(np.array([DEPTH]))
    bed = sea.orbit(np.array([0.0]))
    tot_s = tot_b = 0.0
    print("  wave      lambda     k*h    surface sway    bed sway   kept")
    for i, ((om, k, a), (asx, _), (abx, abz)) in enumerate(
            zip(sea.w, surf, bed)):
        lam = 2.0 * math.pi / k
        print("  %4.1f s  %7.1f m  %6.2f   %8.2f m   %9.3f m  %5.1f%%"
              % (2 * math.pi / om, lam, k * DEPTH,
                 asx[0], abx[0], 100.0 * abx[0] / asx[0]))
        assert abz[0] < 1e-12, "vertical motion at the bed must be zero"
        tot_s += asx[0]
        tot_b += abx[0]
    swell_bed = bed[0][0][0] / tot_b
    swell_surf = surf[0][0][0] / tot_s
    print("  total sway  surface %.2f m   bed %.2f m  (bed is %.0f%%)"
          % (tot_s, tot_b, 100.0 * tot_b / tot_s))
    print("  the 9 s swell is %.0f%% of the motion at the surface"
          " and %.0f%% of it on the bed" % (100 * swell_surf, 100 * swell_bed))

    # the whole claim, as assertions
    assert swell_bed > 0.90, "swell must dominate the bed"
    assert swell_surf < 0.65, "surface must be a mix, not the swell alone"
    assert bed[2][0][0] / surf[2][0][0] < 0.02, "chop must not reach bottom"
    assert 0.20 < tot_b / tot_s < 0.45, "bed sway must be visible but small"

    # it has to fit, and it has to loop
    peak = sum(a for _, a in WAVES)
    print("  worst-case crest %.2f m against %.2f m of freeboard"
          % (peak, FREEBOARD))
    assert peak < FREEBOARD, "waves would leave the top of the frame"
    for T, _ in WAVES:
        assert abs(DUR / T - round(DUR / T)) < 1e-9, "period must divide run"
        assert abs(FPS * T - round(FPS * T)) < 1e-9, "period must land on a frame"
    assert INK[BAND_CH] == 1.00, "surface band needs the solid glyph"

    # a wave must visibly CROSS, which is the reason for widescreen at all
    for T, _ in WAVES:
        k = solve_k(T, DEPTH)
        lam = 2.0 * math.pi / k
        c = (2 * math.pi / T) / k
        print("  %.1f s wave: %.2f wavelengths across frame, crosses in %.1f s"
              % (T, v.width / lam, v.width / c))
    assert v.width / (2 * math.pi / solve_k(WAVES[1][0], DEPTH)) > 1.0, \
        "frame must hold at least one full wind-sea wavelength"

    # the payoff has to be BIG ENOUGH TO SEE, in cells, not just in metres
    print("  what one fleck of marine snow does, in character cells:")
    worst = None
    for name, zeta in (("just under the surface", DEPTH - 0.5),
                       ("mid-water", DEPTH * 0.5),
                       ("on the bed", 0.0)):
        ax = sum(o[0][0] for o in sea.orbit(np.array([zeta]))) / v.mpr
        az = sum(o[1][0] for o in sea.orbit(np.array([zeta]))) / v.mpr
        print("    %-22s %4.1f cells wide x %4.1f cells tall  (%s)"
              % (name, 2 * ax, 2 * az,
                 "circle" if az > 0.4 * ax else "a flat line"))
        if name == "on the bed":
            worst = (ax, az)
    top = sea.orbit(np.array([DEPTH - 0.5]))
    assert 2 * sum(o[1][0] for o in top) / v.mpr > 4.0, \
        "surface orbit too small to read as a circle on screen"
    assert 2 * worst[0] / v.mpr > 2.5, "bed sway must still be visible"
    assert worst[1] < 1e-9, "bed orbit must be perfectly flat"

    print("loop closes: %.1f s, %d frames" % (DUR, FRAMES))


BAND_CH = "#"


def depth_shade(zeta):
    """How much daylight is left at this height above the bed.

    K_ATTEN is a real coastal number, and taken raw it leaves 4% of the
    light on the bed, which is honest and unwatchable. The exponent below
    is a gamma on the DISPLAY, not a change to the physics -- the same move
    your eye makes, which is why 4% of the light does not look like 4%.
    """
    return np.exp(-K_ATTEN * (DEPTH - zeta))


def kelp_colour(zeta, far=0.0):
    f = float(np.clip(depth_shade(zeta), 0.0, 1.0)) ** 0.38
    rgb = [KELP_DEEP[i] + (KELP_LIT[i] - KELP_DEEP[i]) * f for i in range(3)]
    haze = SEA_TOP[0] + (SEA_BED[0] - SEA_TOP[0]) * (1.0 - f)
    m = 0.62 * far                                   # water in between
    return tuple(rgb[i] * (1.0 - m) + (haze, haze * 1.9, haze * 2.1)[i] * m
                 for i in range(3))


def paint(fr, g, v, sea, stipes, snow, t):
    import cairo

    # --- water column: fill the body first, texture goes on top -------------
    grad = cairo.LinearGradient(0, v.row0 * g.cell, 0, g.h_px)
    grad.add_color_stop_rgb(0.0, *SEA_TOP)
    grad.add_color_stop_rgb(1.0, *SEA_BED)
    fr.ctx.set_source(grad)
    fr.ctx.rectangle(0, v.row0 * g.cell, g.w_px, g.h_px)
    fr.ctx.fill()

    # --- the bed ------------------------------------------------------------
    bed_row = int(v.row(-DEPTH))
    for r in range(bed_row, g.rows):
        k = (r - bed_row) / max(1.0, g.rows - bed_row)
        fr.put_run(0, r, BAND_CH * g.cols,
                   tuple(c * (1.0 - 0.45 * k) for c in BEDROCK))

    # --- marine snow: it does not fall, it goes round ------------------------
    sx, sz, sfar = snow
    dx, dz = sea.move(sx, sz, t)
    cols = v.col(sx + dx)
    rows = v.row(sz + dz - DEPTH)
    lit = depth_shade(sz) ** 0.4
    for c, r, l, fdist in zip(cols, rows, lit, sfar):
        a = (0.30 + 0.70 * float(l)) * (1.0 - 0.55 * float(fdist))
        fr.put(int(c), int(r), "." if fdist > 0.45 else "*", SNOW,
               alpha=min(1.0, a))

    # --- kelp, back of the forest first --------------------------------------
    xs, lens, fars = stipes
    fl_px = FROND_LEN / v.mpr
    for i in range(N_STIPE):
        far = float(fars[i])
        px, zeta, s = stipe_points(xs[i], lens[i])
        taper = np.clip(s / 1.6, 0.0, 1.0)          # holdfast is bolted down
        dx, dz = sea.move(px, zeta, t)
        cc = v.col(px + dx * taper)
        rr = v.row(zeta + dz * taper - DEPTH)
        cols_i = kelp_colour  # local
        fat = far < 0.35                            # near stipes are 2 wide

        # A giant kelp blade runs up to about 80 cm. One cell here is 40 cm.
        # So blades CANNOT meaningfully be drawn
        # here -- the first attempt drew them anyway and the forest came out
        # looking like a barcode, rungs stacked every other row. What a
        # bladed stipe actually looks like from a few metres off is a rope
        # with a ragged edge, so that is what gets built: a width profile,
        # filled solid down the middle and fraying at the sides.
        ph = float(xs[i])
        halfw = (2.5 - 1.9 * far) * (
            1.0 + 0.30 * np.sin(s * 2.1 + ph) + 0.22 * np.sin(s * 5.7 - ph))
        halfw = np.clip(halfw, 0.5, 4.0)

        tc = np.gradient(cc)
        tr = np.gradient(rr)
        tn = np.hypot(tc, tr) + 1e-9
        nc, nr = -tr / tn, tc / tn                  # unit normal to the stipe

        drawn = set()
        for j in range(len(s)):
            rgb = cols_i(zeta[j], far)
            hw = float(halfw[j]) * min(1.0, s[j] / 0.8)
            n_off = max(3, int(hw * 4.5))
            for u in np.linspace(-hw, hw, n_off):
                cell = (int(cc[j] + nc[j] * u), int(rr[j] + nr[j] * u))
                if cell in drawn:
                    continue
                drawn.add(cell)
                e = abs(u) / (hw + 1e-9)
                ch = BAND_CH if e < 0.45 else ("%" if e < 0.78 else "=")
                fr.put(cell[0], cell[1], ch, rgb)

    # --- the surface, two rows so it has no holes in it ---------------------
    xw = np.arange(g.cols) * v.mpr
    e = sea.eta(xw, t)
    top = v.row(e)
    for c in range(g.cols):
        r = int(top[c])
        fr.put(c, r, BAND_CH, SURF)
        fr.put(c, r + 1, BAND_CH, SURF)


def render(still=False):
    g = Grid(1920, 1080, font_size=28)
    v = View(g)
    sea = Sea(WAVES, DEPTH)
    stipes = build_stipes(v)
    rng = np.random.default_rng(11)
    snow = (rng.uniform(0.0, v.width, N_SNOW),
            rng.uniform(0.4, DEPTH - 0.3, N_SNOW),
            rng.uniform(0.0, 1.0, N_SNOW))

    if still:
        fr = Frame(g, AIR)
        paint(fr, g, v, sea, stipes, snow, 3.7)
        fr.surface.write_to_png("/tmp/kelp_still.png")
        print("wrote /tmp/kelp_still.png")
        return

    with Encoder(OUT, g, fps=FPS) as enc:
        for f in range(FRAMES):
            fr = Frame(g, AIR)
            paint(fr, g, v, sea, stipes, snow, f / float(FPS))
            enc.write(fr)
            if f % 60 == 0:
                print("  %d/%d" % (f, FRAMES))
    print("wrote", OUT)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "render"
    if cmd == "check":
        check()
    elif cmd == "still":
        check()
        render(still=True)
    else:
        check()
        render()
