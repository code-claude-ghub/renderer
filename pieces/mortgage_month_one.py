#!/usr/bin/env python3
"""MONTH ONE — the median American mortgage, drawn as height.

One composite form. The top of it is a house: its full height is the loan
on the median US home. It starts solid, drains to a ghost in one second
(you own none of it yet), then refills as principal is retired, month by
month, for thirty years.

Beneath it, a block grows downward. That block is the interest, at the same
scale — a dollar of interest is exactly as tall as a dollar of principal.
It is banded once per year, so the slabs are a calendar you can count, and
they thin as they fall because the machine takes the most first.

At the end the block is taller than the house.

Numbers (all computed here, none typed in):
  price   $434,100   NAR median existing-home, July 2026 (rel. 11 Aug 2026)
  rate      6.67%    Freddie Mac PMMS 30-yr fixed, week of 13 Aug 2026
  20% down, 360 months, fully amortizing.
"""

import math
import os
import sys

import cairo
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asciilib import (Camera, Encoder, Frame, Grid, depth_cue, ink_lut,
                      lambert, specular, visible, zbuffer)

OUT = os.path.expanduser("~/projects/active/youtube/youtube-channel/"
                         "renders/mortgage_month_one.mp4")

# ---------------------------------------------------------------- the sum

PRICE = 434100.0            # NAR median existing-home price, July 2026
DOWN = 0.20
LOAN = PRICE * (1.0 - DOWN)
RATE = 0.0667               # Freddie Mac PMMS 30-year fixed, 13 Aug 2026
NMON = 360
_r = RATE / 12.0
PMT = LOAN * _r / (1.0 - (1.0 + _r) ** -NMON)

_bal = LOAN
CUMI = [0.0]                 # cumulative interest paid after k payments
PRIN = [0.0]                 # principal retired after k payments
for _k in range(NMON):
    _i = _bal * _r
    _p = PMT - _i
    _bal -= _p
    CUMI.append(CUMI[-1] + _i)
    PRIN.append(LOAN - _bal)
CUMI = np.array(CUMI)
PRIN = np.array(PRIN)
TOT_I = CUMI[-1]
I1 = LOAN * _r
P1 = PMT - I1

# ------------------------------------------------------------- the shape
# world: x across, y into the picture, z up. house height 1.0 == the loan.

HOUSE_H = 1.0
W, D = 1.30, 1.55           # footprint; the ridge runs along y, at the camera
HW = 0.46                   # eaves. A shallow roof at a low camera reads as a
RISE = HOUSE_H - HW         # bevelled box -- 40 degrees reads as a house.
COL_H = TOT_I / LOAN        # 1.316 house-heights of interest

ELEV = math.radians(21.0)
SPIN0 = math.radians(31.0)
SPIN_A = math.radians(9.0)

FPS, DUR = 30, 21.0
FRAMES = int(FPS * DUR)

T_DRAIN0, T_DRAIN1 = 0.45, 1.35
T_RUN0, T_RUN1 = 1.35, 15.30
T_LAB, T_PUNCH = 15.90, 18.30

STEP = 0.0090               # surface sample spacing, world units
LINE_T = 0.0110             # half-thickness of the fill line
GHOST_KEEP = 0.26

BG = (0.026, 0.047, 0.039)          # bottle green, almost black
IVORY = (0.985, 0.955, 0.885)       # the house, owned
SAGE = (0.44, 0.60, 0.50)           # the house, not yours yet
EMBER = (1.00, 0.42, 0.13)          # the interest
SEAM = (0.30, 0.09, 0.03)           # the groove between two years
GOLD = (1.00, 0.855, 0.42)          # the line where ownership stops
INK = (0.985, 0.965, 0.925)         # words

M_HOUSE, M_COL, M_GHOST, M_SEAM, M_LINE = range(5)
GAIN = np.array([1.00, 0.92, 0.78, 0.80, 1.00])

LAMP = (-0.58, -0.52, 0.62)         # camera space: -y is up the screen

G = Grid()
RAMP = ink_lut()
RND = np.random.RandomState(70414)

# ----------------------------------------------------------- the surface


def _quad(o, u, v, n, step):
    """Sample a parallelogram o + a*u + b*v, jittered off the fixed seed."""
    lu = max(2, int(np.linalg.norm(u) / step))
    lv = max(2, int(np.linalg.norm(v) / step))
    a, b = np.meshgrid(np.linspace(0, 1, lu), np.linspace(0, 1, lv))
    a = a.ravel() + RND.uniform(-0.5, 0.5, a.size) / lu
    b = b.ravel() + RND.uniform(-0.5, 0.5, b.size) / lv
    np.clip(a, 0.0, 1.0, out=a)
    np.clip(b, 0.0, 1.0, out=b)
    p = np.asarray(o) + a[:, None] * np.asarray(u) + b[:, None] * np.asarray(v)
    nn = np.asarray(n, float)
    nn = nn / np.linalg.norm(nn)
    return p, np.repeat(nn[None, :], p.shape[0], 0)


def _tri(o, u, v, n, step):
    p, nn = _quad(o, u, v, n, step)
    # fold the far half of the parallelogram back onto the triangle
    d = p - np.asarray(o)
    # barycentric a + b <= 1 test in (u, v)
    uu, vv = np.asarray(u), np.asarray(v)
    m = np.linalg.inv(np.array([[uu @ uu, uu @ vv], [uu @ vv, vv @ vv]]))
    ab = (np.stack([d @ uu, d @ vv], 1)) @ m.T
    keep = (ab[:, 0] + ab[:, 1]) <= 1.0
    return p[keep], nn[keep]


def build():
    hx, hy = W / 2.0, D / 2.0
    P, N, M = [], [], []

    def add(p, n, m):
        P.append(p)
        N.append(n)
        M.append(np.full(p.shape[0], m, np.int8))

    # --- house: four walls to the eaves
    add(*_quad((-hx, -hy, 0), (W, 0, 0), (0, 0, HW), (0, -1, 0), STEP), M_HOUSE)
    add(*_quad((-hx, hy, 0), (W, 0, 0), (0, 0, HW), (0, 1, 0), STEP), M_HOUSE)
    add(*_quad((-hx, -hy, 0), (0, D, 0), (0, 0, HW), (-1, 0, 0), STEP), M_HOUSE)
    add(*_quad((hx, -hy, 0), (0, D, 0), (0, 0, HW), (1, 0, 0), STEP), M_HOUSE)

    # --- gable ends, facing the camera (ridge runs along y)
    for sy, ny in ((-hy, -1.0), (hy, 1.0)):
        for sx in (-hx, hx):
            add(*_tri((sx, sy, HW), (-sx, 0, 0), (-sx, 0, RISE),
                      (0, ny, 0), STEP), M_HOUSE)

    # --- two roof planes
    add(*_quad((-hx, -hy, HW), (hx, 0, RISE), (0, D, 0),
               (-RISE, 0, hx), STEP), M_HOUSE)
    add(*_quad((hx, -hy, HW), (-hx, 0, RISE), (0, D, 0),
               (RISE, 0, hx), STEP), M_HOUSE)

    # --- the interest block, hanging under the house
    add(*_quad((-hx, -hy, -COL_H), (W, 0, 0), (0, 0, COL_H),
               (0, -1, 0), STEP), M_COL)
    add(*_quad((-hx, hy, -COL_H), (W, 0, 0), (0, 0, COL_H),
               (0, 1, 0), STEP), M_COL)
    add(*_quad((-hx, -hy, -COL_H), (0, D, 0), (0, 0, COL_H),
               (-1, 0, 0), STEP), M_COL)
    add(*_quad((hx, -hy, -COL_H), (0, D, 0), (0, 0, COL_H),
               (1, 0, 0), STEP), M_COL)

    return (np.concatenate(P), np.concatenate(N), np.concatenate(M))


PTS, NRM, MAT = build()
ZW = PTS[:, 2].copy()
IS_H = MAT == M_HOUSE
IS_C = MAT == M_COL

# A groove every five years, so the slabs can be counted. Yearly grooves came
# out three cells apart and read as wood grain rather than as a calendar.
BAND_YR = 5
YEAR_Z = -CUMI[12 * BAND_YR:360:12 * BAND_YR] / LOAN
SEAMED = np.zeros(PTS.shape[0], bool)
for _z in YEAR_Z:
    SEAMED |= IS_C & (np.abs(ZW - _z) < 0.0125)
NBAND = len(YEAR_Z)

GKEEP = RND.rand(PTS.shape[0]) < GHOST_KEEP

# ------------------------------------------------------------- the motion


def smooth(a, b, u):
    u = min(1.0, max(0.0, u))
    return a + (b - a) * (u * u * (3.0 - 2.0 * u))


def month(t):
    if t <= T_RUN0:
        return 0.0
    return min(1.0, (t - T_RUN0) / (T_RUN1 - T_RUN0)) * NMON


def _lerp(arr, m):
    k = int(m)
    if k >= NMON:
        return arr[NMON]
    return arr[k] + (arr[k + 1] - arr[k]) * (m - k)


def fill_z(t):
    if t < T_DRAIN0:
        return HOUSE_H
    if t < T_DRAIN1:
        return smooth(HOUSE_H, 0.0, (t - T_DRAIN0) / (T_DRAIN1 - T_DRAIN0))
    return _lerp(PRIN, month(t)) / LOAN * HOUSE_H


def col_z(t):
    return _lerp(CUMI, month(t)) / LOAN


def _rot_z(p, n, a):
    c, s = math.cos(a), math.sin(a)
    r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return p @ r.T, n @ r.T


def _cam(p, n):
    s, c = math.sin(ELEV), math.cos(ELEV)
    r = np.array([[1.0, 0.0, 0.0], [0.0, -s, -c], [0.0, -c, s]])
    return p @ r.T, n @ r.T


def place(p, n, t):
    p, n = _rot_z(p, n, SPIN0 + SPIN_A * math.sin(2.0 * math.pi * t / DUR))
    return _cam(p, n)


def pose(t):
    """Exactly what draw() draws, at full extent — the camera must see it."""
    return place(PTS, NRM, t)[0]


CAM = Camera(G).fit([pose(t) for t in np.linspace(0.0, DUR, 9)], margin=1.03)

# ---------------------------------------------------------------- letters

_TXT = {}


def text_cells(text, h_cells, ss=8, thresh=0.40):
    key = (text, h_cells)
    if key in _TXT:
        return _TXT[key]
    hp = h_cells * ss
    probe = cairo.Context(cairo.ImageSurface(cairo.FORMAT_A8, 8, 8))
    probe.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL,
                           cairo.FONT_WEIGHT_BOLD)
    fs = float(hp)
    for _ in range(4):
        probe.set_font_size(fs)
        e = probe.text_extents(text)
        if e.height <= 0:
            break
        fs *= hp / e.height
    probe.set_font_size(fs)
    e = probe.text_extents(text)
    wp = int(math.ceil(e.width))
    wc = max(1, int(math.ceil(wp / float(ss))))
    surf = cairo.ImageSurface(cairo.FORMAT_A8, wc * ss, hp)
    ctx = cairo.Context(surf)
    ctx.select_font_face("sans-serif", cairo.FONT_SLANT_NORMAL,
                         cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(fs)
    ctx.set_source_rgba(1, 1, 1, 1)
    ctx.move_to(-e.x_bearing, -e.y_bearing)
    ctx.show_text(text)
    surf.flush()
    buf = np.frombuffer(surf.get_data(), np.uint8)
    buf = buf.reshape(hp, surf.get_stride())[:, :wc * ss] / 255.0
    cells = buf.reshape(h_cells, ss, wc, ss).mean(axis=(1, 3))
    _TXT[key] = cells > thresh
    return _TXT[key]


def cells(text, h):
    """Shrink until it fits the frame. A later, longer line must not run off."""
    for hh in range(h, 3, -1):
        m = text_cells(text, hh)
        if m.shape[1] + 6 <= G.cols:
            return m
    raise AssertionError("%r will not fit at 4 cells" % text)


def stamp(fr, mask, row0, rgb, halo=2):
    h, w = mask.shape
    col0 = (G.cols - w) // 2
    if halo:
        pad = np.zeros((h + 2 * halo, w + 2 * halo), bool)
        for dr in range(-halo, halo + 1):
            for dc in range(-halo, halo + 1):
                pad[halo + dr:halo + dr + h, halo + dc:halo + dc + w] |= mask
        pad[halo:halo + h, halo:halo + w] &= ~mask
        for r, c in zip(*np.nonzero(pad)):
            fr.put(col0 - halo + c, row0 - halo + r, "#", BG)
    for r, c in zip(*np.nonzero(mask)):
        fr.put(col0 + c, row0 + r, "#", rgb)


def money(v):
    return "$%s" % format(int(round(v)), ",d")


# ------------------------------------------------------------------ paint


def colour(shade, extra):
    m = int(extra)
    if m == M_COL:
        base = EMBER
    elif m == M_GHOST:
        base = SAGE
    elif m == M_SEAM:
        base = SEAM
    elif m == M_LINE:
        base = GOLD
    else:
        base = IVORY
    # The glyph already carries the shading. Multiplying the tint by shade as
    # well darkens everything twice and every colour collapses toward the
    # background -- burnt orange came out as dried blood. Keep a floor.
    k = 0.46 + 0.54 * shade
    return (base[0] * k, base[1] * k, base[2] * k)


def draw(f):
    t = f / float(FPS)
    fz, cz = fill_z(t), col_z(t)

    live = IS_H | (IS_C & (ZW >= -cz))
    ghost = IS_H & (ZW > fz)
    live &= ~(ghost & ~GKEEP)

    p, n = place(PTS[live], NRM[live], t)
    front = n[:, 2] > 0.015                      # backface cull
    p, n = p[front], n[front]

    mat = np.where(ghost[live][front], M_GHOST, MAT[live][front]).astype(np.int8)
    mat[SEAMED[live][front] & (mat == M_COL)] = M_SEAM
    if T_DRAIN0 <= t <= T_RUN1 + 0.4:
        onl = IS_H[live][front] & (np.abs(ZW[live][front] - fz) < LINE_T)
        mat[onl] = M_LINE

    col, row, z = CAM.project(p)
    ok = visible(G, col, row)
    col, row, z, n, mat = col[ok], row[ok], z[ok], n[ok], mat[ok]
    _, keep = zbuffer(G, col, row, z)

    lit = (0.36 + 0.55 * lambert(n, LAMP)
           + 0.22 * specular(n, LAMP, 24)) * depth_cue(z, 1.0, 0.94)
    shade = np.clip(lit * np.take(GAIN, mat), 0.0, 1.0)
    shade[mat == M_LINE] = 0.97
    shade[mat == M_SEAM] = 0.26

    fr = Frame(G, BG)
    fr.field(col, row, keep, shade, colour, RAMP, extra=mat.astype(float))

    if T_LAB <= t < T_PUNCH:
        stamp(fr, cells(money(LOAN), 11), 43, INK)
        stamp(fr, cells(money(TOT_I), 11), 101, INK)
    elif t >= T_PUNCH:
        stamp(fr, cells("MONTH 1 BUYS", 8), 60, INK)
        stamp(fr, cells(money(P1) + " OF HOUSE", 8), 74, INK)
    return fr


# ------------------------------------------------------------------ check


def check():
    print("loan       %s   payment %s/mo" % (money(LOAN), money(PMT)))
    print("month 1    interest %s   principal %s  (%.1f%% of the payment)"
          % (money(I1), money(P1), 100.0 * P1 / PMT))
    cross = next(k for k in range(1, NMON + 1)
                 if (PMT - (LOAN - PRIN[k - 1]) * _r) > (LOAN - PRIN[k - 1]) * _r)
    print("half-and-half at payment %d = %d yr %d mo"
          % (cross, cross // 12, cross % 12))
    print("5 years    paid %s, principal retired %s (%.1f%%)"
          % (money(PMT * 60), money(PRIN[60]), 100.0 * PRIN[60] / LOAN))
    print("total      paid %s, interest %s = %.3f x the loan"
          % (money(PMT * NMON), money(TOT_I), TOT_I / LOAN))
    print("points     %d  (house %d, block %d)"
          % (PTS.shape[0], IS_H.sum(), IS_C.sum()))
    print("block      %.3f house-heights, %d five-year grooves" % (COL_H, NBAND))
    lo_r, hi_r, lo_c, hi_c = 1e9, -1e9, 1e9, -1e9
    for t in np.linspace(0.0, DUR, 13):
        c, r, _ = CAM.project(place(PTS, NRM, t)[0])
        lo_r, hi_r = min(lo_r, r.min()), max(hi_r, r.max())
        lo_c, hi_c = min(lo_c, c.min()), max(hi_c, c.max())
    print("rows %d..%d of 0..%d   cols %d..%d of 0..%d"
          % (lo_r, hi_r, G.rows - 1, lo_c, hi_c, G.cols - 1))
    assert lo_r >= 1 and hi_r < G.rows - 1, "clips vertically"
    assert lo_c >= 0 and hi_c < G.cols, "clips horizontally"
    for s, h in ((money(LOAN), 11), (money(TOT_I), 11),
                 ("MONTH 1 BUYS", 8), (money(P1) + " OF HOUSE", 8)):
        m = cells(s, h)
        print("  text %-18r %d x %d cells" % (s, m.shape[1], m.shape[0]))
        assert m.shape[1] + 6 <= G.cols


if __name__ == "__main__":
    check()
    if "--check" in sys.argv:
        sys.exit(0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with Encoder(OUT, G, fps=FPS) as enc:
        for f in range(FRAMES):
            enc.write(draw(f))
            if f % 60 == 0:
                print("  %d/%d" % (f, FRAMES), flush=True)
    print("wrote", OUT)
