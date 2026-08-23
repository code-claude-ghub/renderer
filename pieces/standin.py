#!/usr/bin/env python3
"""THE STAND-IN.

The frontal crash test that certifies an American car is run on a dummy
defined in 49 CFR Part 572. Subpart E is the Hybrid III 50th percentile
male: 77.7 kg, 171.3 lb. Subpart O is the only adult female device in the
whole part, the Hybrid III 5th percentile female: 49 kg, 108 lb. Its
manufacturer says, in its own words, that it "was created using scaled data
taken from our Hybrid III 50th dummy" -- the "Hybrid III 50th Male design
(scaled down)."

Measured, 2021-2023: the average American woman aged 20 and over weighs
171.8 lb. That is the male dummy's weight, to within half a pound. There is
no mid-size female dummy anywhere in Part 572.

So the render is one body. It stands at the male dummy's mass. It shrinks --
same outline, nested exactly inside its own ghost, because that is literally
how the small one was made. Then the number climbs to the weight of the woman
it is standing in for, and the ghost fills, and the body does not grow,
because there is nothing there to grow into.

The scale factor is checked two ways: cube root of the mass ratio, and the
published stature ratio. They agree to 0.6%.

Colourway: charcoal-violet ground, safety yellow device, steel pins, dark
photo targets, ice-white instrument, magenta for the woman.
"""

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]

from asciilib import (Encoder, Frame, Grid, contact, ink_lut, lambert,  # noqa
                      specular, visible, zbuffer)

G = Grid()
RAMP = ink_lut()

# ----------------------------------------------------------------- palette
BG = (0.055, 0.048, 0.070)      # charcoal violet
SKIN = (0.984, 0.792, 0.129)    # safety yellow
TARGET = (0.130, 0.102, 0.090)  # photo-target rings, near black
STEEL = (0.627, 0.612, 0.588)   # joint pins
ICE = (0.878, 0.929, 0.988)     # instrument
WOMAN = (0.980, 0.318, 0.616)   # magenta

# ------------------------------------------------------------------- facts
#
# Hybrid III 50th percentile male   -- 49 CFR 572 subpart E
# Hybrid III 5th percentile female  -- 49 CFR 572 subpart O
# masses from the manufacturer's published specifications
MASS_M_KG, MASS_M_LB = 77.7, 171.3
MASS_F_KG, MASS_F_LB = 49.0, 108.0
# published statures, mm
STAT_M_MM, STAT_F_MM = 1751.0, 1511.0
# CDC/NCHS, measured, adults 20+, August 2021 - August 2023
WOMAN_LB = 171.8
MAN_LB = 199.0

H = STAT_M_MM / 1000.0                       # metres, full-size device
SCALE = (MASS_F_KG / MASS_M_KG) ** (1.0 / 3.0)
SCALE_STATURE = STAT_F_MM / STAT_M_MM        # held out of the render

# ------------------------------------------------------------------ layout
R_FEET = 150                 # both bodies stand on this row
FIG_ROWS = 132               # full-size device, in cells
K = FIG_ROWS / H             # cells per metre
C_MID = 49

R_NUM = 100                  # weight readout, scale 3
R_LAB1 = 119                 # single-line label, scale 2
R_LAB2A, R_LAB2B = 117, 130  # two-line label
R_FLOOR = 151

# ------------------------------------------------------------------- clock
FPS = 30
T_END = 10.0
FRAMES = int(round(FPS * T_END))

T_COUNT = 0.90               # readout counts up to the male mass
T_SHRINK0, T_SHRINK1 = 1.90, 3.50
T_GHOST0, T_GHOST1 = 1.90, 2.50
T_REVEAL = 5.60
T_FILL1 = 6.60


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


# ---------------------------------------------------------------- sampling
def _frame_axes(u):
    tmp = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(u, tmp)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(u, e1)


def capsule(a, b, ra, rb, n, rng):
    """Tapered capsule: lateral surface plus two spherical caps."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ax = b - a
    L = np.linalg.norm(ax)
    u = ax / L
    e1, e2 = _frame_axes(u)

    a_lat = math.pi * (ra + rb) * math.hypot(L, rb - ra)
    a_ca, a_cb = 2 * math.pi * ra * ra, 2 * math.pi * rb * rb
    tot = a_lat + a_ca + a_cb
    n_lat = max(1, int(n * a_lat / tot))
    n_ca = max(1, int(n * a_ca / tot))
    n_cb = max(1, n - n_lat - n_ca)

    t = rng.random(n_lat)
    th = rng.random(n_lat) * 2 * math.pi
    r = ra + (rb - ra) * t
    rad = np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2
    p_lat = a + t[:, None] * ax + r[:, None] * rad

    out_p, out_n = [p_lat], [rad]
    for cen, rr, sgn, cnt in ((a, ra, -1.0, n_ca), (b, rb, 1.0, n_cb)):
        v = rng.normal(size=(cnt, 3))
        v /= np.linalg.norm(v, axis=1)[:, None]
        d = v @ u
        v[d * sgn < 0] *= -1.0
        # reflect only the axial component so the cap stays a hemisphere
        d = v @ u
        bad = d * sgn < 0
        v[bad] -= 2.0 * d[bad][:, None] * u
        out_p.append(cen + rr * v)
        out_n.append(v)
    return np.vstack(out_p), np.vstack(out_n)


def ellipsoid(c, r, n, rng):
    c = np.asarray(c, float)
    r = np.asarray(r, float)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    p = c + v * r
    nn = v / r
    nn /= np.linalg.norm(nn, axis=1)[:, None]
    return p, nn


# ------------------------------------------------------------------ figure
#
# One form. A seated-design anthropomorphic test device, stood upright, built
# from capsules and ellipsoids. Fewer, larger parts than a real body -- the
# cow taught this channel that spending characters across many small parts
# leaves none of them described.
SKIN_PARTS = [
    ("e", (0.000, 1.640, 0.010), (0.079, 0.110, 0.098)),       # head
    ("c", (0.0, 1.470, 0.000), (0.0, 1.545, 0.005), 0.058, 0.052),   # neck
    ("e", (0.000, 1.295, 0.000), (0.196, 0.185, 0.118)),       # chest
    ("e", (0.000, 1.075, 0.000), (0.150, 0.120, 0.100)),       # abdomen
    ("e", (0.000, 0.945, 0.000), (0.172, 0.110, 0.112)),       # pelvis
]
for sx in (-1.0, 1.0):
    SKIN_PARTS += [
        ("e", (sx * 0.185, 1.415, 0.000), (0.075, 0.070, 0.075)),        # shoulder
        ("c", (sx * 0.198, 1.395, 0.000), (sx * 0.228, 1.075, 0.010),
         0.058, 0.046),                                                  # upper arm
        ("c", (sx * 0.228, 1.075, 0.010), (sx * 0.248, 0.800, 0.020),
         0.046, 0.036),                                                  # forearm
        ("e", (sx * 0.255, 0.735, 0.022), (0.036, 0.070, 0.028)),        # hand
        ("c", (sx * 0.092, 0.930, 0.000), (sx * 0.104, 0.500, 0.005),
         0.088, 0.060),                                                  # thigh
        ("c", (sx * 0.104, 0.500, 0.005), (sx * 0.108, 0.075, 0.010),
         0.060, 0.042),                                                  # shin
        ("e", (sx * 0.108, 0.042, 0.075), (0.050, 0.042, 0.128)),        # foot
    ]

PIN_PARTS = []
for sx in (-1.0, 1.0):
    PIN_PARTS += [
        ("c", (sx * 0.213, 1.075, 0.010), (sx * 0.243, 1.075, 0.010),
         0.052, 0.052),                                                  # elbow
        ("c", (sx * 0.089, 0.500, 0.005), (sx * 0.119, 0.500, 0.005),
         0.068, 0.068),                                                  # knee
        ("c", (sx * 0.150, 1.415, 0.000), (sx * 0.196, 1.415, 0.000),
         0.078, 0.078),                                                  # shoulder
    ]

N_POINTS = 150000


def build():
    rng = np.random.default_rng(90714)
    area = []
    for p in SKIN_PARTS + PIN_PARTS:
        if p[0] == "e":
            r = p[2]
            area.append((r[0] * r[1] + r[1] * r[2] + r[0] * r[2]) * 4.0)
        else:
            a, b, ra, rb = np.array(p[1]), np.array(p[2]), p[3], p[4]
            L = np.linalg.norm(b - a)
            area.append(math.pi * (ra + rb) * L + 2 * math.pi * (ra ** 2 + rb ** 2))
    area = np.array(area)
    share = area / area.sum()

    P, N, E = [], [], []
    for i, part in enumerate(SKIN_PARTS + PIN_PARTS):
        n = max(400, int(N_POINTS * share[i]))
        if part[0] == "e":
            p, nn = ellipsoid(part[1], part[2], n, rng)
        else:
            p, nn = capsule(part[1], part[2], part[3], part[4], n, rng)
        P.append(p)
        N.append(nn)
        E.append(np.full(len(p), 2.0 if i >= len(SKIN_PARTS) else 0.0))

    P = np.vstack(P)
    N = np.vstack(N)
    E = np.concatenate(E)

    # photo targets: the rings that say "measuring instrument" rather than
    # "person". Colour only, no geometry -- they are painted on.
    front = P[:, 2] > 0.02
    for cy, rad, half in ((1.660, 0.062, 0.015), (1.320, 0.108, 0.021)):
        d = np.hypot(P[:, 0], P[:, 1] - cy)
        E[front & (np.abs(d - rad) < half)] = 1.0
    return P, N, E


PTS, NRM, EXT = build()
LAMP = (-0.42, 0.62, 0.72)


def posed(scale, yaw):
    p = PTS * scale
    n = NRM
    if abs(yaw) > 1e-9:
        c, s = math.cos(yaw), math.sin(yaw)
        rotm = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        p = p @ rotm.T
        n = n @ rotm.T
    return p, n


def project(p):
    col = np.rint(C_MID + p[:, 0] * K).astype(np.int32)
    row = np.rint(R_FEET - p[:, 1] * K).astype(np.int32)
    return col, row, p[:, 2]


def shade_of(n, z, scale):
    lit = 0.14 + 0.74 * lambert(n, LAMP) + 0.34 * specular(n, LAMP, 26)
    # depth cue by hand: this projection is orthographic, so world z IS the
    # distance, and a tall form wants a shallow fade (RENDERER trap 13).
    zc = z / (0.30 * scale)
    return np.clip(lit * (0.94 + 0.06 * np.clip(zc, -1.0, 1.0)), 0.0, 1.0)


def raster(scale, yaw):
    """One body -> (flat cell index, keep mask, shade, extra)."""
    p, n = posed(scale, yaw)
    col, row, z = project(p)
    ok = visible(G, col, row)
    col, row, z, n, e = col[ok], row[ok], z[ok], n[ok], EXT[ok]
    flat, keep = zbuffer(G, col, row, z)
    return col, row, flat, keep, shade_of(n, z, scale), e


# --------------------------------------------------------------------- type
FONT = {
    "A": (" # ", "# #", "###", "# #", "# #"),
    "B": ("## ", "# #", "## ", "# #", "## "),
    "C": (" ##", "#  ", "#  ", "#  ", " ##"),
    "D": ("## ", "# #", "# #", "# #", "## "),
    "E": ("###", "#  ", "## ", "#  ", "###"),
    "F": ("###", "#  ", "## ", "#  ", "#  "),
    "G": (" ##", "#  ", "# #", "# #", " ##"),
    "H": ("# #", "# #", "###", "# #", "# #"),
    "I": ("###", " # ", " # ", " # ", "###"),
    "J": ("  #", "  #", "  #", "# #", " # "),
    "K": ("# #", "# #", "## ", "# #", "# #"),
    "L": ("#  ", "#  ", "#  ", "#  ", "###"),
    # M, N and W do not fit in three columns. A 3-wide M reads as an H and a
    # 3-wide N reads as an S -- the contact sheet said "HALE DUMHY" and
    # "US WOMAS". Glyphs are variable width for exactly this reason.
    "M": ("#   #", "## ##", "# # #", "#   #", "#   #"),
    "N": ("#  #", "## #", "# ##", "#  #", "#  #"),
    "O": ("###", "# #", "# #", "# #", "###"),
    "P": ("## ", "# #", "## ", "#  ", "#  "),
    "Q": (" # ", "# #", "# #", "## ", " ##"),
    "R": ("## ", "# #", "## ", "# #", "# #"),
    "S": (" ##", "#  ", " # ", "  #", "## "),
    "T": ("###", " # ", " # ", " # ", " # "),
    "U": ("# #", "# #", "# #", "# #", "###"),
    "V": ("# #", "# #", "# #", "# #", " # "),
    "W": ("#   #", "#   #", "# # #", "## ##", "#   #"),
    "X": ("# #", "# #", " # ", "# #", "# #"),
    "Y": ("# #", "# #", " # ", " # ", " # "),
    "Z": ("###", "  #", " # ", "#  ", "###"),
    "0": ("###", "# #", "# #", "# #", "###"),
    "1": (" # ", "## ", " # ", " # ", "###"),
    "2": ("###", "  #", "###", "#  ", "###"),
    "3": ("###", "  #", "###", "  #", "###"),
    "4": ("# #", "# #", "###", "  #", "  #"),
    "5": ("###", "#  ", "###", "  #", "###"),
    "6": ("###", "#  ", "###", "# #", "###"),
    "7": ("###", "  #", "  #", "  #", "  #"),
    "8": ("###", "# #", "###", "# #", "###"),
    "9": ("###", "# #", "###", "  #", "###"),
    ".": ("   ", "   ", "   ", "   ", " # "),
    " ": ("   ", "   ", "   ", "   ", "   "),
}


def gap_of(sc):
    return max(1, sc - 1)


def text_size(s, sc):
    g = gap_of(sc)
    return sum(len(FONT[c][0]) for c in s) * sc + (len(s) - 1) * g, 5 * sc


def text_col(s, sc):
    return (G.cols - text_size(s, sc)[0]) // 2


def text_mask(s, col0, row0, sc):
    """Words must be built OUT of cells -- one cell of type is 4px on a phone."""
    m = np.zeros((G.rows, G.cols), bool)
    g = gap_of(sc)
    c = col0
    for ch in s:
        pat = FONT[ch]
        w = len(pat[0])
        for gr in range(5):
            for gc in range(w):
                if pat[gr][gc] != "#":
                    continue
                r0, x0 = row0 + gr * sc, c + gc * sc
                r1, x1 = min(r0 + sc, G.rows), min(x0 + sc, G.cols)
                if r0 < G.rows and x0 < G.cols and r1 > 0 and x1 > 0:
                    m[max(r0, 0):r1, max(x0, 0):x1] = True
        c += w * sc + g
    return m


def dilate(mask, pad=1):
    out = mask.copy()
    for _ in range(pad):
        g = out.copy()
        g[1:, :] |= out[:-1, :]
        g[:-1, :] |= out[1:, :]
        g[:, 1:] |= out[:, :-1]
        g[:, :-1] |= out[:, 1:]
        out = g
    return out


def erode(mask):
    out = mask.copy()
    out[1:, :] &= mask[:-1, :]
    out[:-1, :] &= mask[1:, :]
    out[:, 1:] &= mask[:, :-1]
    out[:, :-1] &= mask[:, 1:]
    return out


def stamp(fr, mask, rgb, alpha=1.0):
    for r in np.flatnonzero(mask.any(axis=1)):
        row = mask[r]
        cuts = np.flatnonzero(np.r_[True, row[1:] != row[:-1]])
        for a, b in zip(cuts, np.r_[cuts[1:], G.cols]):
            if row[a]:
                fr.put_run(int(a), int(r), "#" * int(b - a), rgb, alpha)


# -------------------------------------------------------------------- ghost
def build_ghost():
    """The full-size device, rendered once, as the reference envelope."""
    col, row, flat, keep, sh, ext = raster(1.0, 0.0)
    grid_sh = np.zeros(G.rows * G.cols)
    grid_sh[flat[keep]] = sh[keep]
    grid_sh = grid_sh.reshape(G.rows, G.cols)
    mask = grid_sh > 0
    return mask, erode(mask), grid_sh


GH_MASK, GH_INNER, GH_SHADE = build_ghost()
GH_EDGE = GH_MASK & ~GH_INNER


# ----------------------------------------------------------------- timeline
def state(t):
    s = 1.0 - (1.0 - SCALE) * smooth((t - T_SHRINK0) / (T_SHRINK1 - T_SHRINK0))
    if t < T_COUNT:
        w = MASS_M_LB * smooth(t / T_COUNT)
    elif t < T_SHRINK0:
        w = MASS_M_LB
    elif t < T_SHRINK1:
        w = MASS_M_LB + (MASS_F_LB - MASS_M_LB) * smooth(
            (t - T_SHRINK0) / (T_SHRINK1 - T_SHRINK0))
    elif t < T_REVEAL:
        w = MASS_F_LB
    else:
        w = MASS_F_LB + (WOMAN_LB - MASS_F_LB) * smooth(
            (t - T_REVEAL) / (T_FILL1 - T_REVEAL))
    edge = smooth((t - T_GHOST0) / (T_GHOST1 - T_GHOST0))
    fill = smooth((t - T_REVEAL) / (T_FILL1 - T_REVEAL))
    yaw = math.radians(-9.0 * (1.0 - smooth(t / 1.2))
                       + 4.0 * math.sin(2 * math.pi * t / 7.0))
    return s, w, edge, fill, yaw


def labels(t):
    """At most one label at a time. Prose belongs in the description."""
    if t < T_SHRINK0:
        a = 1.0 if t > 0.55 else smooth((t - 0.25) / 0.30)
        return [("MALE DUMMY", 2, R_LAB1, a * (1.0 - smooth((t - 1.90) / 0.30)))]
    if t < T_REVEAL:
        return [("FEMALE DUMMY", 2, R_LAB1,
                 smooth((t - 3.20) / 0.40) * (1.0 - smooth((t - T_REVEAL) / 0.25)))]
    a = smooth((t - 5.80) / 0.45)
    return [("THE AVERAGE", 2, R_LAB2A, a), ("US WOMAN", 2, R_LAB2B, a)]


def readout(w):
    return "%.1f LB" % w


# ------------------------------------------------------------------- render
def draw(t):
    s, w, edge, fill, yaw = state(t)
    fr = Frame(G, BG)

    txt = [(readout(w), 3, R_NUM, 1.0)] + labels(t)
    txt = [(st, sc, r, a) for st, sc, r, a in txt if a > 0.02]
    gm = np.zeros((G.rows, G.cols), bool)
    for st, sc, r, _a in txt:
        gm |= text_mask(st, text_col(st, sc), r, sc)
    hole = dilate(gm, 1).reshape(-1)

    # floor: both bodies stand on one line, so a height difference is a fact
    # and not a camera move
    fr.put_run(6, R_FLOOR, "-" * (G.cols - 12), ICE, 0.30)

    # the reference envelope -- the device the small one was scaled from,
    # and, at the end, the woman it is standing in for
    if edge > 0.02:
        if fill > 0.02:
            rr, cc = np.nonzero(GH_MASK)
            for r, c in zip(rr, cc):
                if hole[r * G.cols + c]:
                    continue
                sh = GH_SHADE[r, c]
                col = tuple(v * (0.40 + 0.60 * sh) for v in WOMAN)
                fr.put(int(c), int(r), RAMP[int(sh * (len(RAMP) - 1))], col,
                       0.30 + 0.55 * fill)
        else:
            rr, cc = np.nonzero(GH_EDGE)
            for r, c in zip(rr, cc):
                if not hole[r * G.cols + c]:
                    fr.put(int(c), int(r), ":", ICE, 0.34 * edge)

    # the device itself
    col, row, flat, keep, sh, ext = raster(s, yaw)
    keep = keep & ~hole[flat]

    def colour(shade, extra):
        base = SKIN
        if extra > 1.5:
            base = STEEL
        elif extra > 0.5:
            base = TARGET
        return tuple(v * (0.46 + 0.54 * shade) for v in base)

    fr.field(col, row, keep, sh, colour, RAMP, extra=ext)

    for st, sc, r, a in txt:
        stamp(fr, text_mask(st, text_col(st, sc), r, sc), ICE, a)
    return fr


# -------------------------------------------------------------------- check
def check():
    print("scale (cube root of mass ratio) = %.5f" % SCALE)
    print("scale (published stature ratio) = %.5f   HELD OUT" % SCALE_STATURE)
    d = abs(SCALE - SCALE_STATURE) / SCALE_STATURE
    print("  disagreement %.2f%%" % (100 * d))
    assert d < 0.015, d

    for scale, name in ((1.0, "male 171.3"), (SCALE, "female 108.0")):
        col, row, flat, keep, sh, ext = raster(scale, 0.0)
        r0, r1 = row[keep].min(), row[keep].max()
        c0, c1 = col[keep].min(), col[keep].max()
        print("%-12s rows %3d..%3d (%3d tall)  cols %2d..%2d  cells %d"
              % (name, r0, r1, r1 - r0 + 1, c0, c1, keep.sum()))
        assert r1 <= R_FEET + 1, r1
        assert 0 <= c0 and c1 < G.cols

    ratio_rows = None
    heights = []
    for scale in (1.0, SCALE):
        _c, row, _f, keep, _s, _e = raster(scale, 0.0)
        heights.append(row[keep].max() - row[keep].min() + 1)
    ratio_rows = heights[1] / heights[0]
    print("on-screen height ratio %.4f vs mass-derived %.4f" % (ratio_rows, SCALE))
    assert abs(ratio_rows - SCALE) < 0.02

    # the payoff must be worth cells, not just true (two wakes running,
    # a designed payoff turned out to be ~2 cells and invisible)
    drop = heights[0] - heights[1]
    print("shrink is %d cells of height" % drop)
    assert drop >= 12, drop

    # Nesting: the small device must sit inside the full-size ghost, because
    # that is the claim -- same design, scaled down.
    #
    # Cell-wise containment is the WRONG test here and failed a good render.
    # A human form branches: scaling about the feet puts the small body's
    # knee at a different row from the big body's knee, so the small inner
    # thigh legitimately lands in the gap between the ghost's legs. The
    # envelope is what a viewer reads, so test the per-row outer span.
    col, row, flat, keep, sh, ext = raster(SCALE, 0.0)
    small = np.zeros(G.rows * G.cols, bool)
    small[flat[keep]] = True
    small = small.reshape(G.rows, G.cols)
    # Per-row envelope containment fails for the same honest reason: the
    # shorter body's HEAD sits at the rows where the taller body's neck is.
    # What the claim actually says is "same proportions, smaller", so compare
    # the normalised width profiles foot-to-crown.
    def profile(mask, n=24):
        rows = np.flatnonzero(mask.any(axis=1))
        top, bot = rows.min(), rows.max()
        h = float(bot - top)
        out = []
        for u in np.linspace(0.04, 0.96, n):
            r = int(round(bot - u * h))
            on = np.flatnonzero(mask[r])
            out.append(0.0 if not len(on) else (on[-1] - on[0] + 1) / h)
        return np.array(out), h

    pa, ha = profile(GH_MASK)
    pb, hb = profile(small)
    err = np.abs(pa - pb).max()
    print("width profile: 24 stations, worst disagreement %.4f of stature"
          % err)
    assert err < 0.045, err

    for t in (0.0, 1.0, 2.7, 4.5, 6.2, 9.5):
        s, w, edge, fill, yaw = state(t)
        _c, _r, flat, keep, _s, _e = raster(s, yaw)
        cov = keep.sum() / float(G.rows * G.cols)
        print("t=%4.1f  scale %.3f  readout %s  ink %.3f"
              % (t, s, readout(w), cov))
        assert 0.03 < cov < 0.30

    # a branching form: measure interior pinholes, not the convex row rule
    _c, _r, flat, keep, _s, _e = raster(1.0, 0.0)
    grid = np.zeros(G.rows * G.cols, bool)
    grid[flat[keep]] = True
    grid = grid.reshape(G.rows, G.cols)
    hole = filled = 0
    for r in range(G.rows):
        on = np.flatnonzero(grid[r])
        if len(on) < 2:
            continue
        filled += len(on)
        d = np.diff(on)
        hole += d[(d > 1) & (d <= 4)].sum() - (d[(d > 1) & (d <= 4)] > 0).sum()
    print("interior pinholes %.1f%% of filled cells" % (100.0 * hole / filled))
    assert hole / filled < 0.12

    for t in (0.0, 2.0, 4.5, 7.0, 9.9):
        for st, sc, r, a in [(readout(state(t)[1]), 3, R_NUM, 1.0)] + labels(t):
            wpx, hpx = text_size(st, sc)
            c0 = text_col(st, sc)
            assert r >= 17 and r + hpx - 1 <= 147, (st, r, hpx)
            assert c0 >= 0 and c0 + wpx <= G.cols, (st, c0, wpx)
    print("all type inside rows 17..147 and the frame width")
    print("the average US woman is %.1f lb; the male device is %.1f lb; "
          "delta %.1f lb" % (WOMAN_LB, MASS_M_LB, WOMAN_LB - MASS_M_LB))


# --------------------------------------------------------------------- main
if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
        sys.exit(0)
    if "--sheet" in sys.argv:
        idx = [0, 45, 90, 135, 190, 285]
        contact([draw(i / float(FPS)) for i in idx], "/tmp/standin_sheet.png",
                cols=3, labels=["%.1fs" % (i / float(FPS)) for i in idx])
        print("wrote /tmp/standin_sheet.png")
        sys.exit(0)
    if "--dump" in sys.argv:
        fr = draw(float(sys.argv[sys.argv.index("--dump") + 1]))
        sys.exit(0)

    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/standin.mp4"
    check()
    with Encoder(out, G, fps=FPS) as enc:
        for f in range(FRAMES):
            enc.write(draw(f / float(FPS)))
            if f % 60 == 0:
                print("  frame %d/%d" % (f, FRAMES))
    print("wrote", out)
