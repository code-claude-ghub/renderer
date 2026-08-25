#!/usr/bin/env python3
"""
ONE LAP.

An escalator, cut open, running its whole chain once.  The video is exactly
one circuit of one step and it loops seamlessly, so the machine's period and
the step's period are both on screen at the same time and they are wildly
different numbers:

    the machine repeats every 0.8 seconds.
    one step repeats every 40 seconds.

Everything on screen comes off four measured facts and some trigonometry:

    incline            30 degrees          (the common installation angle)
    step pitch         400 mm              (EN 115 floor is 380; 400 is what
                                            is actually supplied)
    nominal speed      0.5 m/s             (EN 115 nominal; max 0.75)
    flat steps         >= 2 at each end    (EN 115, for speeds <= 0.5 m/s)

and from those the rise per step is NOT a free choice.  The step plate is
rigid and the chain pitch is constant, so on the incline

    rise = pitch * sin(theta)  =  400 * sin(30) = 200.0 mm
    run  = pitch * cos(theta)  =  400 * cos(30) = 346.4 mm

which is why the tread overhangs the step below it by 53.6 mm -- the nosing.
That relation is the held-out check: published escalator rise figures are
"about 200 mm at 30 degrees" and "about 230 mm at 35 degrees", and neither
number is used anywhere in the construction.  See check().

Steps invert on the return run.  That is not a stylistic choice either: the
step chain carries the front roller round the turnaround sprocket and the
trailer track takes the back roller with it, so the plate turns over and
rides home upside down.  It is the half of an escalator nobody has seen.

    python3 scripts/escalator.py --check
    python3 scripts/escalator.py
"""

import math
import os
import sys

import cairo
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]

from asciilib import (Camera, Encoder, Frame, Grid, contact, depth_cue,
                      ink_lut, lambert, visible, zbuffer)

OUT = os.path.join(_HERE, "out")
os.makedirs(OUT, exist_ok=True)

G = Grid()
RAMP = ink_lut()
FPS = 30

BG = (0.030, 0.038, 0.055)
STEEL = (0.68, 0.76, 0.87)      # the step plates
TRUSS = (0.24, 0.30, 0.40)      # the frame they run inside
FLOOR = (0.17, 0.20, 0.26)      # the two storeys
GLASS = (0.24, 0.36, 0.47)      # balustrade, far side only
RAIL = (0.40, 0.46, 0.56)       # the handrail, its own separate loop
MARK = (1.00, 0.66, 0.16)       # the one step we are following
SKIN = (0.90, 0.80, 0.62)       # the person who is on it for a while

M_STEEL, M_TRUSS, M_FLOOR, M_GLASS, M_RAIL, M_MARK, M_SKIN = 1, 2, 3, 4, 5, 6, 7
M_NEAR = 8   # the near truss: in front of everything, so it gets held back
PAL = {M_STEEL: STEEL, M_TRUSS: TRUSS, M_FLOOR: FLOOR, M_GLASS: GLASS,
       M_RAIL: RAIL, M_MARK: MARK, M_SKIN: SKIN,
       M_NEAR: (0.13, 0.17, 0.24)}

LAMP = np.array([0.30, 0.86, 0.42])
LAMP = LAMP / np.linalg.norm(LAMP)

# ------------------------------------------------------------------- spec
THETA = math.radians(30.0)      # installation angle
PITCH = 0.400                   # step pitch along the chain = tread depth, m
SPEED = 0.500                   # m/s, nominal
WIDTH = 1.000                   # step width, m
N_STEP = 50                     # steps in the chain -> 20.0 m -> 40.0 s

R_SPR = 0.42                    # turnaround sprocket pitch radius, m
R_TRN = 2.40                    # track transition radius at each landing, m
FLAT = 3 * PITCH                # flat run at each landing: three steps

RISE = PITCH * math.sin(THETA)  # 0.2000 m -- derived, never assumed
RUN = PITCH * math.cos(THETA)   # 0.3464 m
NOSE = PITCH - RUN              # 0.0536 m of overhang

CHAIN = N_STEP * PITCH          # 20.0 m
LAP = CHAIN / SPEED             # 40.0 s
BEAT = PITCH / SPEED            # 0.8 s -- a step arrives this often


# --------------------------------------------------------------- the path
# A closed chain loop in the x-y plane.  Built as a list of primitives so
# the arc length is exact rather than sampled, because the whole piece hangs
# on the loop closing on a step boundary.
def arclen(s):
    """A straight carries its length at [3]; an arc carries radius * sweep."""
    return s[3] if s[0] == "L" else abs(s[2] * s[4])


def build_path(incline):
    """(segments, total_length).  Each segment carries its own arc length."""
    seg = []
    p = np.array([0.0, R_SPR])          # top of the lower sprocket
    h = 0.0                             # heading, radians

    def straight(length):
        nonlocal p, h
        d = np.array([math.cos(h), math.sin(h)])
        seg.append(("L", p.copy(), d, length))
        p = p + d * length

    def arc(sweep, radius):
        """sweep > 0 turns left (concave up)."""
        nonlocal p, h
        s = 1.0 if sweep > 0 else -1.0
        nrm = np.array([-math.sin(h), math.cos(h)]) * s
        c = p + nrm * radius
        a0 = math.atan2(p[1] - c[1], p[0] - c[0])
        seg.append(("A", c, radius, a0, sweep))
        h += sweep
        a1 = a0 + sweep
        p = c + radius * np.array([math.cos(a1), math.sin(a1)])

    straight(FLAT)                      # lower landing, flat
    arc(THETA, R_TRN)                   # sag curve into the incline
    straight(incline)                   # the climb
    arc(-THETA, R_TRN)                  # crest curve out of it
    straight(FLAT)                      # upper landing, flat
    top = p.copy()                      # top of the upper sprocket

    ctr_t = top - np.array([0.0, R_SPR])
    ctr_b = np.array([0.0, 0.0])
    u = ctr_b - ctr_t
    u = u / np.linalg.norm(u)
    nrm = np.array([-u[1], u[0]])       # points down, out of the truss
    if nrm[1] > 0:
        nrm = -nrm
    a_ret = math.atan2(nrm[1], nrm[0])

    def wrap(centre, a0, a1):
        """Clockwise around a sprocket from a0 to a1."""
        sweep = (a1 - a0) % (2 * math.pi) - 2 * math.pi
        seg.append(("A", centre, R_SPR, a0, sweep))

    wrap(ctr_t, math.pi / 2, a_ret)                       # over the top
    seg.append(("L", ctr_t + nrm * R_SPR, u,
                float(np.linalg.norm(ctr_b - ctr_t))))     # home, inverted
    wrap(ctr_b, a_ret, math.pi / 2)                       # round the bottom

    return seg, float(sum(arclen(s) for s in seg))


def solve_incline():
    """Length of the inclined straight that closes the chain on 50 steps."""
    lo, hi = 0.2, 12.0
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if build_path(mid)[1] < CHAIN:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


INCLINE = solve_incline()
PATH, PATH_LEN = build_path(INCLINE)
CUM = np.cumsum([0.0] + [arclen(s) for s in PATH])
S_TOP = CUM[5]                          # arc position of the upper comb
RIDE = S_TOP / SPEED                    # seconds a passenger is carried


def at(s):
    """Arc position -> (point, heading). Heading is the chain's direction."""
    s = s % PATH_LEN
    i = int(np.searchsorted(CUM, s, "right") - 1)
    i = min(max(i, 0), len(PATH) - 1)
    d = s - CUM[i]
    q = PATH[i]
    if q[0] == "L":
        return q[1] + q[2] * d, math.atan2(q[2][1], q[2][0])
    c, r, a0, sweep = q[1], q[2], q[3], q[4]
    a = a0 + math.copysign(d / r, sweep)
    p = c + r * np.array([math.cos(a), math.sin(a)])
    tan = a + (math.pi / 2 if sweep > 0 else -math.pi / 2)
    return p, tan


def tilt(s):
    """Step plate angle at arc position s.

    Level the whole way up -- that is the trailer track's entire job -- then
    turned over by the sprockets and level again, upside down, going home.
    """
    s = s % PATH_LEN
    if s <= CUM[5]:
        return 0.0
    if s <= CUM[6]:                                     # over the top
        return -math.pi * (s - CUM[5]) / (CUM[6] - CUM[5])
    if s <= CUM[7]:                                     # the hidden half
        return -math.pi
    return -math.pi - math.pi * (s - CUM[7]) / (CUM[8] - CUM[7])


# ------------------------------------------------------------- primitives
def box(hx, hy, hz, step=0.045, faces="all"):
    """Surface samples of an axis-aligned box, with outward normals."""
    P, N = [], []

    def face(u0, u1, ax, sign, h):
        a = np.arange(-u0 + step / 2, u0, step)
        b = np.arange(-u1 + step / 2, u1, step)
        if not len(a):
            a = np.array([0.0])
        if not len(b):
            b = np.array([0.0])
        A, B = np.meshgrid(a, b, indexing="ij")
        z = np.full(A.size, sign * h)
        cols = {0: (z, A.ravel(), B.ravel()),
                1: (A.ravel(), z, B.ravel()),
                2: (A.ravel(), B.ravel(), z)}[ax]
        P.append(np.stack(cols, 1))
        n = np.zeros((A.size, 3))
        n[:, ax] = sign
        N.append(n)

    face(hy, hz, 0, +1, hx)
    face(hy, hz, 0, -1, hx)
    face(hx, hz, 1, +1, hy)
    if faces == "all":
        face(hx, hz, 1, -1, hy)
    face(hx, hy, 2, +1, hz)
    face(hx, hy, 2, -1, hz)
    return np.vstack(P).astype(np.float32), np.vstack(N).astype(np.float32)


def slab(cx, cy, cz, hx, hy, hz, step=0.09):
    p, n = box(hx, hy, hz, step)
    return p + np.array([cx, cy, cz], np.float32), n


def tube(pts, r, around=8, step=0.09):
    """A round bar swept along a polyline, for handrails and chords."""
    P, N = [], []
    pts = np.asarray(pts, float)
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        d = b - a
        ln = float(np.linalg.norm(d))
        if ln < 1e-9:
            continue
        d = d / ln
        up = np.array([0.0, 0.0, 1.0])
        if abs(d @ up) > 0.9:
            up = np.array([0.0, 1.0, 0.0])
        e1 = np.cross(d, up)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(d, e1)
        ts = np.arange(0.0, ln, step)
        for k in range(around):
            a_ = 2 * math.pi * k / around
            off = math.cos(a_) * e1 + math.sin(a_) * e2
            P.append(a + np.outer(ts, d) + r * off)
            N.append(np.tile(off, (len(ts), 1)))
    if not P:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    return np.vstack(P).astype(np.float32), np.vstack(N).astype(np.float32)


# ------------------------------------------------------------- a step unit
# Local frame: origin at the chain roller.  Tread plate above and behind it,
# riser plate hanging off the back edge down to the tread of the step below.
_TREAD_Y = 0.115
_TREAD_T = 0.015
_RISE_TOP = _TREAD_Y - _TREAD_T
_RISE_BOT = _TREAD_Y + _TREAD_T - RISE


def step_unit():
    p1, n1 = box(PITCH / 2, _TREAD_T, WIDTH / 2, 0.038)
    p1 = p1 + np.array([-PITCH / 2, _TREAD_Y, 0.0])
    hy = (_RISE_TOP - _RISE_BOT) / 2.0
    p2, n2 = box(0.014, hy, WIDTH / 2, 0.042)
    p2 = p2 + np.array([-PITCH, (_RISE_TOP + _RISE_BOT) / 2.0, 0.0])
    # the two chain rollers, because they are the reason for all of it
    p3, n3 = tube([(0.0, 0.0, -WIDTH / 2 - 0.05),
                   (0.0, 0.0, WIDTH / 2 + 0.05)], 0.045, 7, 0.06)
    return (np.vstack([p1, p2, p3]).astype(np.float32),
            np.vstack([n1, n2, n3]).astype(np.float32))


STEP_P, STEP_N = step_unit()


def place(s):
    """Step at arc position s -> world points and normals."""
    p, _ = at(s)
    a = tilt(s)
    c, sn = math.cos(a), math.sin(a)
    R = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]], np.float32)
    return (STEP_P @ R.T + np.array([p[0], p[1], 0.0], np.float32),
            STEP_N @ R.T)


# ------------------------------------------------------------ the building
def carry_poly(n=160, dy=0.0):
    return np.array([at(CUM[5] * i / n)[0] + np.array([0.0, dy])
                     for i in range(n + 1)])


def return_poly(dy=0.0):
    a, _ = at(CUM[6] + 1e-6)
    b, _ = at(CUM[7] - 1e-6)
    return np.array([a + np.array([0.0, dy]), b + np.array([0.0, dy])])


DECK = R_SPR + _TREAD_Y + _TREAD_T      # the tread surface at a landing
X_BOT = 0.0                             # lower comb
X_TOP = float(at(CUM[5])[0][0])         # upper comb
Y_TOP = float(at(CUM[5])[0][1]) + _TREAD_Y + _TREAD_T


def build_static():
    P, N, M = [], [], []

    def add(pn, mat):
        if len(pn[0]):
            P.append(pn[0])
            N.append(pn[1])
            M.append(np.full(len(pn[0]), mat, np.float32))

    # the two storeys.  the tread is flush with the floor at each landing.
    add(slab(X_BOT - 5.0, DECK - 0.20, 0.0, 5.0, 0.18, 2.30, 0.105), M_FLOOR)
    add(slab(X_TOP + 5.0, Y_TOP - 0.20, 0.0, 5.0, 0.18, 2.30, 0.105), M_FLOOR)

    # truss: top chord under the carrying run, bottom chord under the return,
    # with verticals between.  it is a lattice, so you can see through it.
    top = carry_poly(70, -0.34)
    bot = return_poly(-0.30)
    for zz, mat in ((-0.78, M_TRUSS), (0.78, M_NEAR)):
        add(tube([(p[0], p[1], zz) for p in top], 0.075, 7, 0.085), mat)
        add(tube([(bot[0][0], bot[0][1], zz), (bot[1][0], bot[1][1], zz)],
                 0.075, 7, 0.085), mat)
        d = bot[1] - bot[0]
        for k in range(0 if mat is M_TRUSS else 9, 9):
            t = (k + 0.5) / 9.0
            b = bot[0] + d * t
            i = int(round(t * 70))
            i = min(max(i, 0), len(top) - 1)
            a_ = top[i]
            add(tube([(a_[0], a_[1], zz), (b[0], b[1], zz)],
                     0.05, 6, 0.10), mat)

    # far balustrade only.  the near one is removed -- that is the cutaway.
    glass = carry_poly(70, 0.0)
    for k in range(2, 71, 9):
        a_ = glass[k]
        add(tube([(a_[0], a_[1] + 0.12, -0.62),
                  (a_[0], a_[1] + 0.95, -0.62)], 0.030, 5, 0.10), M_GLASS)

    # the handrail is a second endless loop with its own period, and it is
    # the only part of an escalator most people could draw.
    rail = [(p[0], p[1] + 1.00, -0.66) for p in carry_poly(70, 0.0)]
    add(tube(rail, 0.045, 7, 0.07), M_RAIL)
    for end, sgn in ((rail[0], -1.0), (rail[-1], 1.0)):
        newel = [(end[0] + sgn * 0.30 * math.sin(t), end[1] - 0.30 * (1 - math.cos(t)),
                  -0.66) for t in np.linspace(0.0, math.pi, 12)]
        add(tube(newel, 0.045, 7, 0.07), M_RAIL)

    # comb plates: where the steps go into and out of the floor
    add(slab(X_BOT + 0.12, DECK - 0.02, 0.0, 0.13, 0.02, WIDTH / 2, 0.05),
        M_FLOOR)
    add(slab(X_TOP - 0.12, Y_TOP - 0.02, 0.0, 0.13, 0.02, WIDTH / 2, 0.05),
        M_FLOOR)

    return (np.vstack(P).astype(np.float32), np.vstack(N).astype(np.float32),
            np.concatenate(M).astype(np.float32))


STATIC = build_static()


# ------------------------------------------------------------- a passenger
def rider(t):
    """Where the person is at time t, or None once they have gone."""
    board = BEAT * 3.0                      # steps on here
    off = board + RIDE                      # steps off here
    if t < board - 2.2:
        return None
    if t > off + 2.6:
        return None
    if t < board:                           # walking in along the floor
        u = (t - (board - 2.2)) / 2.2
        return (X_BOT - 2.5 + 2.5 * u, DECK, 0.0)
    if t <= off:                            # carried
        p, _ = at(SPEED * (t - board))
        return (float(p[0]) - PITCH * 0.55, float(p[1]) + _TREAD_Y + _TREAD_T,
                0.0)
    u = (t - off) / 2.6                     # walking away
    return (X_TOP + 2.6 * u, Y_TOP, 0.0)


def rider_cloud(pos):
    x, y, z = pos
    P, N, keep = [], [], []

    def add(pn):
        P.append(pn[0])
        N.append(pn[1])

    add(slab(x, y + 0.42, z, 0.10, 0.42, 0.13, 0.055))       # legs
    add(slab(x, y + 1.10, z, 0.14, 0.28, 0.19, 0.055))       # body
    add(slab(x + 0.02, y + 1.52, z, 0.11, 0.14, 0.11, 0.05))  # head
    add(tube([(x + 0.10, y + 1.22, z), (x + 0.22, y + 1.02, z - 0.55)],
             0.05, 6, 0.07))                                  # arm to rail
    del keep
    return (np.vstack(P).astype(np.float32), np.vstack(N).astype(np.float32))


# ------------------------------------------------------------- the camera
YAW, PITCH_CAM = math.radians(28.0), math.radians(12.0)
_CX = (X_BOT + X_TOP) / 2.0
_CY = (DECK + Y_TOP) / 2.0


def pose(p):
    x, y, z = p[:, 0] - _CX, p[:, 1] - _CY, p[:, 2]
    cy_, sy_ = math.cos(YAW), math.sin(YAW)
    x1, z1 = x * cy_ + z * sy_, -x * sy_ + z * cy_
    cx_, sx_ = math.cos(PITCH_CAM), math.sin(PITCH_CAM)
    y1, z2 = y * cx_ - z1 * sx_, y * sx_ + z1 * cx_
    return np.stack([x1, -y1, z2], 1).astype(np.float32)


# Fit the MACHINE, not the building.  The floors are deliberately wider
# than the frame -- they are the two storeys, and a storey that stops
# inside the picture is a plank.
_MACH = STATIC[0][np.abs(STATIC[2] - M_FLOOR) > 0.5]
_ALL = [pose(_MACH)]
_ALL += [pose(place(k * PITCH)[0]) for k in range(N_STEP)]
CAM = Camera(G).fit(_ALL, margin=1.04)


# --------------------------------------------------------------- painting
def colour(shade, extra):
    base = PAL.get(int(round(extra)), STEEL)
    k = 0.42 + 0.58 * shade
    return (base[0] * k, base[1] * k, base[2] * k)


def project(P, N, M, cue=(1.0, 0.90), gain=1.0, floor=0.16):
    scr = pose(P)
    col, row, z = CAM.project(scr)
    ok = visible(G, col, row)
    if not ok.any():
        return None
    col, row, z, n, m = col[ok], row[ok], z[ok], N[ok], M[ok]
    sh = (floor + gain * lambert(n, LAMP)) * depth_cue(z, *cue)
    return col, row, z, np.clip(sh, 0.05, 1.0), m


_S_CACHE = project(*STATIC, cue=(1.0, 0.88), gain=0.62)


def draw(f, plain=False):
    """plain=True draws the machine only -- no amber step, nobody on it.

    That is what makes the two-clocks check mean anything: strip out the
    two things that are ABOUT one step, and what is left must repeat every
    single beat.
    """
    t = (f / float(FPS)) % LAP
    fr = Frame(G, BG)

    cols = [_S_CACHE[0]]
    rows = [_S_CACHE[1]]
    zs = [_S_CACHE[2]]
    shs = [_S_CACHE[3]]
    ms = [_S_CACHE[4]]

    marked = (SPEED * t) % PATH_LEN
    for k in range(N_STEP):
        s = (marked + k * PITCH) % PATH_LEN
        P, N = place(s)
        hero = (k == 0) and not plain
        M = np.full(len(P), M_MARK if hero else M_STEEL, np.float32)
        q = project(P, N, M, cue=(1.0, 0.90), gain=0.95,
                    floor=0.50 if hero else 0.22)
        if q is None:
            continue
        for lst, v in zip((cols, rows, zs, shs, ms), q):
            lst.append(v)

    who = None if plain else rider(t)
    if who is not None:
        P, N = rider_cloud(who)
        q = project(P, N, np.full(len(P), M_SKIN, np.float32),
                    cue=(1.0, 0.92), gain=0.80, floor=0.26)
        if q is not None:
            for lst, v in zip((cols, rows, zs, shs, ms), q):
                lst.append(v)

    col = np.concatenate(cols)
    row = np.concatenate(rows)
    z = np.concatenate(zs)
    sh = np.concatenate(shs)
    m = np.concatenate(ms)
    _, keep = zbuffer(G, col, row, z)
    idx, _v = fr.field(col, row, keep, sh, colour, RAMP, extra=m)
    return fr, idx, m, col, row, keep


# ---------------------------------------------------------------- caption
def text_cells(s, cell_h):
    F = 8
    fs = cell_h * F
    probe = cairo.ImageSurface(cairo.FORMAT_A8, 8, 8)
    pc = cairo.Context(probe)
    pc.select_font_face("monospace", cairo.FONT_SLANT_NORMAL,
                        cairo.FONT_WEIGHT_BOLD)
    pc.set_font_size(fs)
    ext = pc.text_extents(s)
    w, h = int(ext.x_advance) + F * 2, int(fs * 1.6)
    surf = cairo.ImageSurface(cairo.FORMAT_A8, w, h)
    ctx = cairo.Context(surf)
    ctx.select_font_face("monospace", cairo.FONT_SLANT_NORMAL,
                         cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(fs)
    ctx.move_to(F, h * 0.72)
    ctx.show_text(s)
    surf.flush()
    buf = np.frombuffer(surf.get_data(), np.uint8)
    buf = buf.reshape(h, surf.get_stride())[:, :w]
    ys, xs = np.nonzero(buf > 40)
    if not len(ys):
        return np.zeros((1, 1), bool)
    buf = buf[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    hh, ww = buf.shape
    buf = np.pad(buf, ((0, (-hh) % F), (0, (-ww) % F)))
    hh, ww = buf.shape
    return buf.reshape(hh // F, F, ww // F, F).mean((1, 3)) > 46.0


def stamp(fr, s, cell_h, ccen, rcen, rgb, back=BG):
    m = text_cells(s, cell_h)
    while m.shape[1] + 2 > G.cols and cell_h > 2:
        cell_h -= 1
        m = text_cells(s, cell_h)
    h, w = m.shape
    hm = np.zeros((h + 2, w + 2), bool)
    for dy in range(3):
        for dx in range(3):
            hm[dy:dy + h, dx:dx + w] |= m
    c0, r0 = int(round(ccen - w / 2.0)), int(round(rcen - h / 2.0))
    for j in range(h + 2):
        for i in range(w + 2):
            if not hm[j, i]:
                continue
            on = (0 <= j - 1 < h and 0 <= i - 1 < w and m[j - 1, i - 1])
            fr.put(c0 - 1 + i, r0 - 1 + j, "#", rgb if on else back)
    return c0, r0, w, h


LINE = "40 seconds a lap"


# ------------------------------------------------------------------ check
def check():
    print("ONE LAP -- an escalator, whole chain")
    print("  incline            %.1f deg" % math.degrees(THETA))
    print("  step pitch         %.0f mm" % (PITCH * 1000))
    print("  rise per step      %.1f mm   (pitch * sin)" % (RISE * 1000))
    print("  run per step       %.1f mm   (pitch * cos)" % (RUN * 1000))
    print("  nosing overhang    %.1f mm" % (NOSE * 1000))
    print("  chain              %.3f m in %d steps" % (PATH_LEN, N_STEP))
    print("  inclined straight  %.3f m" % INCLINE)
    print("  rise, floor-floor  %.3f m" % (Y_TOP - DECK))
    print("  lap                %.2f s     (a step every %.2f s)"
          % (LAP, BEAT))
    print("  carried for        %.2f s" % RIDE)

    # HELD OUT 1 -- the rise per step is forced by pitch and angle alone.
    # Published figures for real machines: about 200 mm at 30 degrees and
    # about 230 mm at 35.  Neither number appears anywhere above.
    assert abs(RISE * 1000 - 200.0) < 1.0, RISE
    r35 = PITCH * math.sin(math.radians(35.0)) * 1000
    print("  same maths at 35   %.1f mm  (published: about 230)" % r35)
    assert abs(r35 - 230.0) < 2.0, r35

    # HELD OUT 2 -- the chain has to close.  Walk it as 50 separate steps
    # and land back inside a micron of where the first one stands.
    a0, _ = at(0.0)
    a1, _ = at(N_STEP * PITCH)
    assert float(np.linalg.norm(a1 - a0)) < 1e-6, (a0, a1)
    assert abs(PATH_LEN - CHAIN) < 1e-6, PATH_LEN

    # HELD OUT 3 -- EN 115 wants at least two flat steps at each landing for
    # 0.5 m/s.  Count them off the placed geometry, not off the design.
    lo = hi = 0
    for k in range(N_STEP):
        p, _ = at(k * PITCH)
        q, _ = at((k + 1) * PITCH)
        if abs(p[1] - q[1]) < 1e-4:
            if p[0] < X_TOP / 2:
                lo += 1
            else:
                hi += 1
    print("  flat steps         %d bottom, %d top   (EN 115 wants 2)"
          % (lo, hi))
    assert lo >= 2 and hi >= 2, (lo, hi)

    # the rise between consecutive steps ON the incline, measured off the
    # placed points rather than computed.
    mids = []
    for k in range(N_STEP - 1):
        p, _ = at(k * PITCH)
        q, _ = at((k + 1) * PITCH)
        d = q - p
        if abs(math.atan2(d[1], d[0]) - THETA) < 1e-3:
            mids.append(d[1])
    assert len(mids) >= 6, len(mids)
    assert abs(np.mean(mids) - RISE) < 1e-6, np.mean(mids)

    # the plate turns over exactly once and comes back level.
    assert abs(tilt(0.0)) < 1e-9
    assert abs(tilt(CUM[7] - 0.01) + math.pi) < 1e-3
    assert abs(tilt(PATH_LEN - 1e-9) + 2 * math.pi) < 1e-3

    fr, idx, mat, col, row, keep = draw(0)
    ink = int((idx > 0).sum())
    rr, cc = np.nonzero(idx)
    print("  frame 0            %d cells, cols %d..%d rows %d..%d"
          % (ink, cc.min(), cc.max(), rr.min(), rr.max()))
    assert ink > 1400, ink
    assert rr.min() >= 1 and rr.max() < G.rows - 1

    # HELD OUT 4 -- the two clocks.  Ignoring which step is painted amber,
    # the picture must repeat every BEAT and only every BEAT.  The amber one
    # must NOT come back until the whole lap is done.
    def sig(f, plain=True):
        _fr, ix, _m, _c, _r, _k = draw(f, plain)
        return ix

    base = sig(0)
    per = None
    for f in range(1, 40):
        if np.array_equal(sig(f), base):
            per = f
            break
    print("  picture repeats    every %s frames (expected %d)"
          % (per, int(round(BEAT * FPS))))
    assert per == int(round(BEAT * FPS)), per

    def amber(f):
        _fr, ix, m, c, r, k = draw(f)
        sel = np.abs(m - M_MARK) < 0.5
        if not (sel & k).any():
            return None
        return (float(c[sel & k].mean()), float(r[sel & k].mean()))

    a_now, a_beat = amber(0), amber(per)
    assert a_now is not None and a_beat is not None
    moved = math.dist(a_now, a_beat)
    print("  amber step moves   %.1f cells in one beat" % moved)
    assert moved > 2.0, moved
    a_lap = amber(int(round(LAP * FPS)))
    assert a_lap is not None
    print("  amber after a lap  %.3f cells from where it started"
          % math.dist(a_now, a_lap))
    assert math.dist(a_now, a_lap) < 1e-6

    # the loop is seamless, and this one is checked on the WHOLE frame --
    # amber step, passenger and all -- because that is what gets uploaded.
    assert np.array_equal(sig(int(round(LAP * FPS)), False), sig(0, False))
    print("  seamless           yes (full frame, %d == 0)"
          % int(round(LAP * FPS)))

    shots = []
    for f in (0, 90, 300, 560, 640, 900, 1150):
        fr, _i, _m, _c, _r, _k = draw(f)
        stamp(fr, LINE, 9, G.cols / 2.0, G.safe_bot - 10, (0.95, 0.95, 0.95))
        shots.append(fr)
    contact(shots, os.path.join(OUT, "escalator_sheet.png"), cols=3,
            labels=["%.1fs" % (f / 30.0)
                    for f in (0, 90, 300, 560, 640, 900, 1150)])
    print("  sheet              out/escalator_sheet.png")


def render():
    n = int(round(LAP * FPS))
    path = os.path.join(OUT, "escalator.mp4")
    with Encoder(path, G, fps=FPS, crf=20, preset="medium") as enc:
        for f in range(n):
            fr, _i, _m, _c, _r, _k = draw(f)
            stamp(fr, LINE, 9, G.cols / 2.0, G.safe_bot - 10, (0.95, 0.95, 0.95))
            enc.write(fr)
            if f % 120 == 0:
                print("  %4d / %d" % (f, n), flush=True)
    print(path)


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        render()
