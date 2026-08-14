#!/usr/bin/env python3
"""
no loops -- the moon's real track around the sun.

Ask anyone to draw the moon's path through space and they draw a spring:
the earth going round the sun with the moon looping round the earth as it
goes, scalloping in and out, sometimes going backwards. Every classroom
poster does it. It is wrong, and not slightly.

Two numbers kill it.

LOOPS. A loop needs the moon to travel backwards relative to the sun at
some point in the month. Earth carries it along at 29.785 km/s. The moon's
own orbital speed about the earth is 1.023 km/s. So the moon's speed round
the sun swings between 28.76 and 30.81 km/s and never gets near zero -- it
is about 29 times too slow to ever turn around. Its direction of travel
never wanders more than 1.97 degrees off earth's.

SCALLOPS. A weaker claim than a loop: does the track ever bend AWAY from
the sun? Also no. For a circular model the signed curvature has a floor of

    Om^3 R^2 + w^3 b^2 - Om w b R (w + Om)

which is positive for the real b = 384400 km and hits zero at b = 837 Mm.
So the moon's orbit would have to be 2.18 times wider before the path
developed its first straight point, and the track is convex everywhere --
it turns toward the sun at every instant of every month, with no inflection
in it anywhere.

That 2.18 is not a coincidence. The sun pulls the moon 2.20 times harder
than the earth does (5.93 vs 2.70 mm/s^2), so even at new moon, with the
earth hauling directly sunward-away, the net acceleration still points at
the sun. Same fact, arrived at twice.

WHAT IS ON SCREEN. A camera riding with the earth: sun off to the left,
direction of travel up, so the track streams downward past a fixed earth.
Both paths at true scale -- no exaggeration anywhere, which is the whole
point, since the poster version is a SCALE lie. The frame holds about a
quarter of a lunar month, and one full weave passes through per loop.

It loops seamlessly on the synodic month (29.5306 d), because that is the
period of the moon's angle as seen from a frame turning with the earth.

Made for @Dominic-qv3yt, who asked whether the hypotrochoid in an earlier
piece turns up in orbital mechanics. It does, and this is the best case:
the compound-circle curve everyone expects here is real geometry that the
solar system declines to draw.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asciilib import Grid, Frame, Encoder, INK, RAMP_SORTED, ink_lut  # noqa

OUT = "/tmp/moon_no_loops.mp4"
FPS = 30
FRAMES = 600
DUR = FRAMES / float(FPS)                      # 20.0 s

# --- the solar system, SI, circular idealisation ---------------------------
G = 6.67430e-11
M_SUN = 1.98892e30
M_EARTH = 5.97217e24
R_ORB = 1.495979e11          # m, earth's mean orbital radius
A_MOON = 3.84400e8           # m, moon's mean orbital radius
T_YEAR = 365.256363 * 86400.0        # sidereal year
T_MOON = 27.321661 * 86400.0         # sidereal month
T_SYN = 29.530589 * 86400.0          # synodic month -- the loop period

OM = 2.0 * math.pi / T_YEAR          # earth's angular rate about the sun
WM = 2.0 * math.pi / T_MOON          # moon's angular rate about the earth
WS = 2.0 * math.pi / T_SYN           # = WM - OM, the rate the phase turns

V_EARTH = OM * R_ORB                 # 29785 m/s
V_MOON = WM * A_MOON                 # 1023 m/s

# --- framing ---------------------------------------------------------------
# RING is the moon's orbital radius in character cells, and because the scale
# is honest it fixes everything else: at 24 cells the whole visible window is
# only 1.4 days wide, which is the point. The moon takes the entire 20 s to
# crawl once round the ring while its track pours down the frame at 29 times
# that speed. Those two speeds, side by side in one frame, ARE the argument.
RING = 30.0
# The bodies sit high and the track is drawn only BEHIND them, as a wake.
# Drawn as a full line through the frame it was ambiguous -- a viewer has
# no way to know that the bright filament is a path and not a beam. A wake
# streaming out of the moon cannot be read as anything else.
EARTH_ROW = 0.30

CORE_SIG = 0.62              # cells, the hard bright line
HALO_SIG = 4.80              # cells, the glow that gives the frame a body
HALO_W = 0.34
# The streaming texture has to be an exact whole number of cycles per loop
# or the seam shows. Pick the count, not the length: 96 cycles per synodic
# month lands the texture period at about 62 rows, which is what I wanted
# anyway, and now the last frame joins the first.
FLOW_CYCLES = 96
FLOW_D = 0.24                # how deep it modulates

# --- colourway: hot bone track, cold cyan orbit, oxblood ground, ember west
# The ground had to come DOWN and go cold. First pass put a warm ember wash
# over two thirds of the frame and a warm track on top of it, and the whole
# picture went one muddy brown -- the ring and the earth line disappeared
# into it. Warm object on cold ground, and keep the ember to the far edge.
BG_FAR = (0.026, 0.012, 0.022)       # away from the sun
BG_SUN = (0.205, 0.078, 0.040)       # the left edge, where the sun is
MOON_HOT = (1.000, 0.972, 0.885)
MOON_DIM = (0.520, 0.270, 0.175)
MOON_MARK = (1.000, 1.000, 0.990)
EARTH_LINE = (0.330, 0.560, 0.600)
EARTH_MARK = (0.560, 0.930, 0.960)
RING_COL = (0.420, 0.680, 0.740)

LUT = ink_lut()


# --------------------------------------------------------------------------
# geometry
#
# Camera rides the earth. x_hat points radially OUTWARD from the sun (so the
# sun is at x = -R and the track bends toward negative x), y_hat points along
# earth's velocity. tau is time relative to now.
# --------------------------------------------------------------------------
def earth_xy(tau):
    return R_ORB * np.cos(OM * tau) - R_ORB, R_ORB * np.sin(OM * tau)


def moon_xy(tau, psi, b=A_MOON):
    ex, ey = earth_xy(tau)
    return (ex + b * np.cos(WM * tau + psi),
            ey + b * np.sin(WM * tau + psi))


class View(object):
    """Metres <-> cells. One scale, both axes, no cheating."""

    def __init__(self, grid):
        self.g = grid
        self.scale = RING / A_MOON                  # cells per metre
        self.cx = grid.cols / 2.0
        self.cy = grid.rows * EARTH_ROW
        # only the past: from the bodies down past the bottom of the frame
        self.tau0 = -(grid.rows * 0.95) / self.scale / V_EARTH
        self.tau1 = 0.0
        self.days = (self.tau1 - self.tau0) / 86400.0

    def col(self, x):
        return self.cx + x * self.scale

    def row(self, y):
        return self.cy - y * self.scale


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------
def splat(buf, cols, rows, fc, fr, sigma, peak, reach):
    """Lay a gaussian-profiled LINE down a run of float positions.

    Anti-aliasing is not a nicety here: the whole subject is a four-cell
    wobble, and drawn by nearest-cell it comes out as a staircase, which
    reads as a rendering artefact rather than as the moon.

    `peak` is the brightness wanted at the centre of the line, not the
    weight of one dab. A run of dabs spaced ds apart integrates to
    peak*sigma*sqrt(2pi)/ds at the centre, so the dab weight has to be
    divided by exactly that. Getting this wrong once produced a solid
    twenty-cell girder of pure white and nothing else.
    """
    ds = np.hypot(np.diff(fc), np.diff(fr)).mean()
    weight = peak * ds / (sigma * math.sqrt(2.0 * math.pi))
    ic, ir = np.floor(fc).astype(np.int64), np.floor(fr).astype(np.int64)
    inv = 1.0 / (2.0 * sigma * sigma)
    for dc in range(-reach, reach + 1):
        for dr in range(-reach, reach + 1):
            c, r = ic + dc, ir + dr
            d2 = (c + 0.5 - fc) ** 2 + (r + 0.5 - fr) ** 2
            w = weight * np.exp(-d2 * inv)
            ok = (c >= 0) & (c < cols) & (r >= 0) & (r < rows) & (w > 0.0008)
            np.add.at(buf, (r[ok], c[ok]), w[ok])


def ground(fr, g):
    """The sun is off the left edge, so light falls off to the right. This
    is painted rather than typed, because it is light and not a character."""
    import cairo
    grad = cairo.RadialGradient(-g.w_px * 0.30, g.h_px * 0.5, g.w_px * 0.05,
                                -g.w_px * 0.30, g.h_px * 0.5, g.w_px * 1.15)
    grad.add_color_stop_rgb(0.00, *BG_SUN)
    grad.add_color_stop_rgb(0.42, 0.074, 0.030, 0.028)
    grad.add_color_stop_rgb(1.00, *BG_FAR)
    fr.ctx.set_source(grad)
    fr.ctx.paint()


def draw(fr, buf, hot, dim, floor=0.045):
    rr, cc = np.nonzero(buf > floor)
    for r, c in zip(rr, cc):
        e = min(1.0, buf[r, c])
        ch = LUT[int(e * 255)]
        w = e ** 0.62
        fr.put(int(c), int(r),
               ch, tuple(dim[i] + (hot[i] - dim[i]) * w for i in range(3)))


def body(fr, fc, frow, rad, col):
    """A small solid disc for a body. Not to scale and it cannot be."""
    r0 = int(math.floor(rad)) + 1
    for dc in range(-r0, r0 + 1):
        for dr in range(-r0, r0 + 1):
            c, r = int(math.floor(fc)) + dc, int(math.floor(frow)) + dr
            d = math.hypot(c + 0.5 - fc, r + 0.5 - frow)
            if d > rad + 0.9:
                continue
            e = min(1.0, max(0.0, rad + 0.9 - d))
            fr.put(c, r, LUT[int(e * 255)], col, alpha=min(1.0, 0.35 + e))


def paint(fr, g, v, t):
    psi = WS * t                       # moon's phase in the turning frame
    ground(fr, g)

    n = 4200
    tau = np.linspace(v.tau0, v.tau1, n)

    # the circle everyone is shown: the moon's orbit about the earth, drawn
    # at exactly the same scale as everything else in the frame.
    th = np.linspace(0.0, 2.0 * math.pi, 1400)
    rbuf = np.zeros((g.rows, g.cols))
    splat(rbuf, g.cols, g.rows,
          v.cx + RING * np.cos(th), v.cy + RING * np.sin(th),
          0.80, 0.52, 3)

    ex, ey = earth_xy(tau)
    ebuf = np.zeros((g.rows, g.cols))
    splat(ebuf, g.cols, g.rows, v.col(ex), v.row(ey), 0.72, 0.46, 3)

    # the moon's actual track, with a texture that streams at the speed the
    # moon really travels: 62 rows per 0.19 days, 29 times the ring crawl.
    mx, my = moon_xy(tau, psi)
    flow = 1.0 - FLOW_D * (0.5 + 0.5 * np.cos(
        2.0 * math.pi * FLOW_CYCLES * (t + tau) / T_SYN))
    mbuf = np.zeros((g.rows, g.cols))
    splat(mbuf, g.cols, g.rows, v.col(mx), v.row(my),
          HALO_SIG, HALO_W * flow, 11)
    splat(mbuf, g.cols, g.rows, v.col(mx), v.row(my),
          CORE_SIG, 1.00 * flow, 3)

    draw(fr, np.clip(rbuf, 0, 1), RING_COL, tuple(x * 0.42 for x in RING_COL))
    draw(fr, np.clip(ebuf, 0, 1), EARTH_LINE,
         tuple(x * 0.36 for x in EARTH_LINE))
    draw(fr, np.clip(mbuf, 0, 1), MOON_HOT, MOON_DIM)

    # The two bodies. Objects in the scene, not labels -- but they are the
    # one thing here NOT to scale: at this zoom the earth is 8 thousandths
    # of a cell wide. Everything else in the frame is honest.
    body(fr, v.col(0.0), v.row(0.0), 2.4, EARTH_MARK)
    mx0, my0 = moon_xy(0.0, psi)
    body(fr, v.col(mx0), v.row(my0), 1.9, MOON_MARK)


# --------------------------------------------------------------------------
# the arithmetic the piece rests on, checked before a frame is drawn
# --------------------------------------------------------------------------
def curvature_floor(b):
    """Minimum of the signed cross product x' y'' - y' x'' over a month,
    for a moon of orbital radius b. Positive everywhere = convex toward
    the sun, no inflection points, no scallops."""
    return (OM ** 3 * R_ORB ** 2 + WM ** 3 * b ** 2
            - OM * WM * b * R_ORB * (WM + OM))


def check(g, v):
    print(g, " window %.2f days, %.1f%% of a lunar month"
          % (v.days, 100.0 * v.days * 86400.0 / T_MOON))

    a_sun = G * M_SUN / R_ORB ** 2
    a_earth = G * M_EARTH / A_MOON ** 2
    print("pull on the moon: sun %.3f mm/s2, earth %.3f mm/s2, ratio %.3f"
          % (a_sun * 1e3, a_earth * 1e3, a_sun / a_earth))
    assert a_sun > 2.0 * a_earth, "the sun should win by better than 2x"

    print("speeds: earth %.3f km/s, moon %.3f km/s, moon round the sun "
          "%.2f..%.2f km/s" % (V_EARTH / 1e3, V_MOON / 1e3,
                               (V_EARTH - V_MOON) / 1e3,
                               (V_EARTH + V_MOON) / 1e3))
    assert V_EARTH - V_MOON > 0, "a loop would be possible"
    print("  too slow to loop by a factor of %.1f" % (V_EARTH / V_MOON))
    print("  heading never wanders more than %.2f deg off earth's"
          % math.degrees(math.asin(V_MOON / V_EARTH)))

    # convexity, sampled hard rather than trusted to the algebra
    tt = np.linspace(0.0, T_MOON, 200001)
    x = R_ORB * np.cos(OM * tt) + A_MOON * np.cos(WM * tt)
    y = R_ORB * np.sin(OM * tt) + A_MOON * np.sin(WM * tt)
    dx = -R_ORB * OM * np.sin(OM * tt) - A_MOON * WM * np.sin(WM * tt)
    dy = R_ORB * OM * np.cos(OM * tt) + A_MOON * WM * np.cos(WM * tt)
    ddx = -R_ORB * OM ** 2 * np.cos(OM * tt) - A_MOON * WM ** 2 * np.cos(WM * tt)
    ddy = -R_ORB * OM ** 2 * np.sin(OM * tt) - A_MOON * WM ** 2 * np.sin(WM * tt)
    cross = dx * ddy - dy * ddx
    assert cross.min() > 0.0, "the path has an inflection -- claim is dead"
    assert (np.hypot(dx, dy) > 0).all()
    print("signed curvature floor: sampled %.1f, closed form %.1f (>0 = convex)"
          % (cross.min(), curvature_floor(A_MOON)))
    assert abs(cross.min() - curvature_floor(A_MOON)) < 1.0

    lo, hi = A_MOON, 40.0 * A_MOON
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if curvature_floor(mid) > 0:
            lo = mid
        else:
            hi = mid
    print("first scallop at b = %.1f Mm, which is %.3f x the real moon"
          % (lo / 1e6, lo / A_MOON))
    assert 2.0 < lo / A_MOON < 2.4

    b_loop = V_EARTH / WM
    print("first loop at b = %.0f Mm, which is %.1f x the real moon"
          % (b_loop / 1e6, b_loop / A_MOON))
    assert abs(b_loop / A_MOON - V_EARTH / V_MOON) < 0.01

    # --- can any of this be SEEN? the kelp lesson: assert the payoff ---
    tau = np.linspace(v.tau0, v.tau1, 20001)
    mx, my = moon_xy(tau, 0.0)
    ex, ey = earth_xy(tau)
    dev = (v.col(mx) - v.col(ex))
    print("moon swings %.1f cells either side of earth's line, ring is "
          "%.0f cells across a %d-cell frame"
          % (dev.max(), 2 * RING, g.cols))
    assert dev.max() > 8.0, "the weave is too small to see"
    assert 2 * RING < g.cols - 6, "the ring does not fit"

    # the two speeds the frame puts side by side, in screen cells per second
    # of video. one time compression for everything: T_SYN into DUR.
    comp = T_SYN / DUR
    flow_s = V_EARTH * v.scale * comp
    moon_s = V_MOON * v.scale * comp
    print("on screen: track pours %.0f cells/s, moon travels %.1f cells/s "
          "-- ratio %.1f, which is the whole argument"
          % (flow_s, moon_s, flow_s / moon_s))
    assert abs(flow_s / moon_s - V_EARTH / V_MOON) < 1e-6, "speeds not honest"
    print("  (the ring itself turns once per synodic month, so it reads "
          "%.1f cells/s -- 8%% slower, and that is real too)"
          % (2.0 * math.pi * RING / DUR))
    flow_rows = (T_SYN / FLOW_CYCLES) * v.scale * V_EARTH
    print("  texture period %.1f rows = %.2f s, %.1f frames -- no strobe"
          % (flow_rows, flow_rows / flow_s, FPS * flow_rows / flow_s))
    assert FPS * flow_rows / flow_s > 4.0, "streaming texture will strobe"
    assert float(FLOW_CYCLES).is_integer(), "texture will not close the loop"

    tilt = math.degrees(math.atan(V_MOON / V_EARTH))
    print("track leans at most %.2f deg = %.1f cells over the %d-row frame"
          % (tilt, g.rows * V_MOON / V_EARTH, g.rows))

    # y must be monotonic on screen, which IS the no-loops claim
    ry = v.row(my)
    assert (np.diff(ry) < 0).all(), "the wake doubles back on screen"
    print("wake is %.0f rows long in a %d-row frame, strictly one-way"
          % (ry[0] - ry[-1], g.rows))
    # the wake must run off the bottom even when the moon is at its highest
    top = v.cy - RING
    assert top + (ry[0] - ry[-1]) > g.rows, "wake stops inside the frame"
    assert top > g.rows * 0.10, "ring climbs into the Shorts UI band"

    # the loop closes: psi must advance by exactly 2*pi over the run
    assert abs(WS * (T_SYN) - 2.0 * math.pi) < 1e-12
    print("loop: %.1f s of video = %.4f days = one synodic month, seamless"
          % (DUR, T_SYN / 86400.0))
    assert INK["#"] == 1.00


def main():
    g = Grid(1080, 1920, font_size=16)
    v = View(g)
    check(g, v)
    if "--check" in sys.argv:
        return
    with Encoder(OUT, g, fps=FPS) as enc:
        for f in range(FRAMES):
            fr = Frame(g, BG_FAR)
            paint(fr, g, v, T_SYN * f / float(FRAMES))
            enc.write(fr)
            if f % 60 == 0:
                print("  frame %d/%d" % (f, FRAMES))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
