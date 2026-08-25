#!/usr/bin/env python3
"""
THE CATHEDRAL — a serial.  Part I: the foundation.

Not an explainer.  There is no fact here to hand anybody.  It is a building,
and it is not finished, and the only way to see it finished is to come back.

The whole cathedral exists in this file from day one as a GHOST: a line
drawing of every mass, hanging in the air where it will be.  Each part moves
from ghost to stone, and the camera never moves, so any two episodes can be
laid on top of each other and read.

    THE CAMERA IS FIXED FOREVER.  It is fitted to the finished building, not
    to whatever is built yet.  Do not re-fit it, do not "improve" the view,
    and do not edit MASSES once part I has shipped -- both would break the
    only thing that makes the series legible.

To continue the series: implement the next entry in STAGES and run with
--stage N.  Everything up to and including N is stone.  Everything after is
still a drawing.

    python3 scripts/cathedral.py --check --stage 0
    python3 scripts/cathedral.py --stage 0
"""

import argparse
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]

import cairo  # noqa: E402
from asciilib import (Camera, Encoder, Frame, Grid, contact, depth_cue,  # noqa
                      ink_lut, lambert, visible, zbuffer)

# ---------------------------------------------------------------- palette
BG = (0.043, 0.051, 0.106)     # evening, an hour after the light goes
GHOST = (0.298, 0.373, 0.541)  # the drawing of what it will be
STONE = (0.902, 0.827, 0.663)  # limestone, lit low from the south-west
EARTH = (0.259, 0.180, 0.129)  # the trench
GOLD = (0.949, 0.749, 0.325)   # the only warm thing that is not stone

OLD = (0.639, 0.612, 0.549)    # part I, a season later: weathered, colder
CRYPT = (0.906, 0.733, 0.443)  # the one room that will never see daylight

ROUGH = (0.470, 0.412, 0.361)  # part III: rubble. the wall nobody dressed.

M_GHOST, M_STONE, M_EARTH, M_OLD, M_CRYPT, M_SLAB, M_CWALL = 1, 2, 3, 4, 5, 6, 7
M_WALL3, M_PART = 8, 9

# the crypt wall does not get buried -- it keeps going up and becomes the
# outside of the choir.  so it stops being warm when the room is sealed.
CW = {"rgb": CRYPT}

G = Grid()
RAMP = ink_lut()
FPS = 30
RNG = np.random.default_rng(1163)   # notre-dame de paris, begun

# ---------------------------------------------------------------- the plan
# metres.  x runs west to east, y is up, z runs north to south.
# a cruciform gothic cathedral: west front and towers, nave with aisles,
# transept, crossing tower and spire, choir with aisles, apse.
NAVE_Z, AISLE_Z = 8.0, 15.0
NAVE_Y, AISLE_Y = 36.0, 19.0
X_WEST, X_NAVE, X_TRAN, X_CHOIR, X_APSE = -4.0, 0.0, 62.0, 80.0, 106.0

MASSES = [
    # (name, kind, args)
    ("westfront", "box", (X_WEST, X_NAVE, 0.0, 46.0, -AISLE_Z, AISLE_Z)),
    ("tower_n", "box", (X_WEST, 8.0, 0.0, 64.0, -AISLE_Z, -5.0)),
    ("tower_s", "box", (X_WEST, 8.0, 0.0, 64.0, 5.0, AISLE_Z)),
    ("nave", "box", (X_NAVE, X_TRAN, 0.0, NAVE_Y, -NAVE_Z, NAVE_Z)),
    ("aisle_n", "box", (X_NAVE, X_TRAN, 0.0, AISLE_Y, -AISLE_Z, -NAVE_Z)),
    ("aisle_s", "box", (X_NAVE, X_TRAN, 0.0, AISLE_Y, NAVE_Z, AISLE_Z)),
    ("transept", "box", (X_TRAN, X_CHOIR, 0.0, NAVE_Y, -26.0, 26.0)),
    ("choir", "box", (X_CHOIR, X_APSE, 0.0, NAVE_Y, -NAVE_Z, NAVE_Z)),
    ("choir_n", "box", (X_CHOIR, X_APSE, 0.0, AISLE_Y, -AISLE_Z, -NAVE_Z)),
    ("choir_s", "box", (X_CHOIR, X_APSE, 0.0, AISLE_Y, NAVE_Z, AISLE_Z)),
    ("apse", "apse", (X_APSE, 0.0, AISLE_Z, 0.0, NAVE_Y)),
    ("crossing", "box", (X_TRAN, X_CHOIR, NAVE_Y, 58.0, -9.0, 9.0)),
    ("spire", "pyr", (71.0, 0.0, 9.0, 58.0, 86.0)),
    ("roof_nave", "roof", (X_NAVE, X_TRAN, -NAVE_Z, NAVE_Z, NAVE_Y, 46.0)),
    ("roof_choir", "roof", (X_CHOIR, X_APSE, -NAVE_Z, NAVE_Z, NAVE_Y, 46.0)),
    ("roof_tran", "roofz", (X_TRAN, X_CHOIR, -26.0, 26.0, NAVE_Y, 46.0)),
    ("tower_cap_n", "pyr", (2.0, -10.0, 6.0, 64.0, 78.0)),
    ("tower_cap_s", "pyr", (2.0, 10.0, 6.0, 64.0, 78.0)),
]

# the outer wall line of the whole footprint, walked as a closed loop.
# this is what gets dug in part I.
FOOTPRINT = [
    (X_WEST, -AISLE_Z), (X_TRAN, -AISLE_Z), (X_TRAN, -26.0),
    (X_CHOIR, -26.0), (X_CHOIR, -AISLE_Z), (X_APSE, -AISLE_Z),
]


def _apse_arc(n=26):
    a = np.linspace(-math.pi / 2.0, math.pi / 2.0, n)
    return [(X_APSE + AISLE_Z * math.cos(t), AISLE_Z * math.sin(t))
            for t in a]


FOOTPRINT = (FOOTPRINT + _apse_arc()
             + [(X_APSE, AISLE_Z), (X_CHOIR, AISLE_Z), (X_CHOIR, 26.0),
                (X_TRAN, 26.0), (X_TRAN, AISLE_Z), (X_WEST, AISLE_Z),
                (X_WEST, -AISLE_Z)])


# ---------------------------------------------------------------- samplers
def _edge(a, b, step=1.7):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(2, int(np.linalg.norm(b - a) / step))
    t = np.linspace(0.0, 1.0, n)[:, None]
    return a + (b - a) * t


def box_edges(x0, x1, y0, y1, z0, z1):
    """The twelve edges, minus the bottom rectangle when the mass sits on
    the ground -- those four lines land in the ground plane and fight the
    trench for the same cells."""
    c = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    segs = []
    for i in range(8):
        for j in range(i + 1, 8):
            if sum(abs(c[i][k] - c[j][k]) > 1e-9 for k in range(3)) != 1:
                continue
            if y0 < 1e-6 and abs(c[i][1]) < 1e-6 and abs(c[j][1]) < 1e-6:
                continue
            segs.append(_edge(c[i], c[j]))
    return np.vstack(segs)


def pyr_edges(cx, cz, half, y0, y1):
    base = [(cx - half, y0, cz - half), (cx + half, y0, cz - half),
            (cx + half, y0, cz + half), (cx - half, y0, cz + half)]
    tip = (cx, y1, cz)
    segs = [_edge(base[i], base[(i + 1) % 4]) for i in range(4)]
    segs += [_edge(b, tip) for b in base]
    return np.vstack(segs)


def roof_edges(x0, x1, z0, z1, yb, yr, along_x=True):
    zm, xm = (z0 + z1) / 2.0, (x0 + x1) / 2.0
    if along_x:
        ridge = [(x0, yr, zm), (x1, yr, zm)]
        eaves = [(x0, yb, z0), (x1, yb, z0), (x0, yb, z1), (x1, yb, z1)]
    else:
        ridge = [(xm, yr, z0), (xm, yr, z1)]
        eaves = [(x0, yb, z0), (x0, yb, z1), (x1, yb, z0), (x1, yb, z1)]
    segs = [_edge(*ridge)]
    for e in eaves:
        segs.append(_edge(e, ridge[0] if (e[0] == ridge[0][0] or
                                          e[2] == ridge[0][2]) else ridge[1]))
    segs.append(_edge(eaves[0], eaves[1]))
    segs.append(_edge(eaves[2], eaves[3]))
    return np.vstack(segs)


def apse_edges(cx, cz, r, y0, y1):
    a = np.linspace(-math.pi / 2.0, math.pi / 2.0, 34)
    ring = np.stack([cx + r * np.cos(a), np.zeros_like(a), cz + r * np.sin(a)],
                    1)
    segs = []
    for y in (y0, y1):
        rr = ring.copy()
        rr[:, 1] = y
        segs.append(rr)
    for k in range(0, len(a), 6):
        segs.append(_edge((ring[k][0], y0, ring[k][2]),
                          (ring[k][0], y1, ring[k][2])))
    return np.vstack(segs)


def ghost_points():
    out = []
    for name, kind, args in MASSES:
        if kind == "box":
            out.append(box_edges(*args))
        elif kind == "pyr":
            out.append(pyr_edges(*args))
        elif kind == "roof":
            out.append(roof_edges(*args, along_x=True))
        elif kind == "roofz":
            out.append(roof_edges(*args, along_x=False))
        elif kind == "apse":
            out.append(apse_edges(*args))
    return np.vstack(out).astype(np.float32)


def block(cx, cy, cz, hx, hy, hz, step=0.62):
    """A dressed stone: the five faces you can see, with normals."""
    pts, nrm = [], []
    for ax, hi, n in ((0, hx, (1, 0, 0)), (1, hy, (0, 1, 0)),
                      (2, hz, (0, 0, 1))):
        for s in (-1, 1):
            if ax == 1 and s < 0:
                continue                       # no underside
            u_ax, v_ax = [k for k in (0, 1, 2) if k != ax]
            hu = (hx, hy, hz)[u_ax]
            hv = (hx, hy, hz)[v_ax]
            uu = np.arange(-hu, hu + 1e-6, step)
            vv = np.arange(-hv, hv + 1e-6, step)
            U, V = np.meshgrid(uu, vv)
            p = np.zeros((U.size, 3))
            p[:, ax] = s * hi
            p[:, u_ax] = U.ravel()
            p[:, v_ax] = V.ravel()
            p += np.array([cx, cy, cz])
            pts.append(p)
            nrm.append(np.tile(np.array(n, float) * s, (U.size, 1)))
    return np.vstack(pts), np.vstack(nrm)


def _walk(spacing):
    """Even spacing along the WHOLE loop, by cumulative arc length.

    Doing it per segment instead put one stone on every segment however
    short, and the 25-segment apse arc -- 47 m of curve -- came out with 25
    footings on it while the 66 m nave flank got 10.  The held-out perimeter
    check caught it: 74 stones placed where 53 were implied.
    """
    pts = np.array(FOOTPRINT, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    n = int(round(total / spacing))
    out = []
    for k in range(n):
        d = (k + 0.5) * total / n
        i = int(np.searchsorted(cum, d) - 1)
        i = min(max(i, 0), len(seg) - 1)
        t = (d - cum[i]) / seg[i]
        out.append(pts[i] + (pts[i + 1] - pts[i]) * t)
    return out


def foundation_stones(spacing=6.4):
    """Footing blocks set along the trench, in the order they are laid."""
    path = _walk(spacing)
    out = []
    for j, (x, z) in enumerate(path):
        wob = RNG.uniform(-0.16, 0.16, 3)
        out.append(block(x + wob[0], 1.35 + wob[1] * 0.4, z + wob[2],
                         2.05, 1.35, 2.05))
    return out


def trench_points(step=0.85):
    """The cut itself: a dark channel following the wall line."""
    pts = []
    for i in range(len(FOOTPRINT) - 1):
        a = np.array(FOOTPRINT[i], float)
        b = np.array(FOOTPRINT[i + 1], float)
        n = max(2, int(np.linalg.norm(b - a) / step))
        for t in np.linspace(0.0, 1.0, n):
            p = a + (b - a) * t
            for off in (-2.6, -1.3, 0.0, 1.3, 2.6):
                d = b - a
                nn = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
                q = p + nn * off
                pts.append((q[0], 0.05, q[1]))
    return np.array(pts, np.float32)


# ------------------------------------------------------------ part II: crypt
# The crypt is the undercroft under the choir and the apse.  Two things are
# true about it and the episode is both of them: it is the first ROOM, and it
# is the only part of a cathedral built to be buried.  Its ceiling is the
# choir floor.  Nobody standing in the finished building ever sees it again.
Y_FOOT, Y_SPRING, Y_CROWN = 2.7, 6.4, 8.6

# the crypt wall follows the part I footprint round the east end.  no west
# wall: you go DOWN into a crypt from the choir, and a wall there would stand
# on nothing -- part I never dug a footing across the middle of the building.
CRYPT_PATH = ([(X_CHOIR, -AISLE_Z), (X_APSE, -AISLE_Z)] + _apse_arc()[1:]
              + [(X_CHOIR, AISLE_Z)])

N_COURSE = 5
PIER_X = [81.0, 88.0, 95.0, 102.0]
PIER_Z = [-6.5, 6.5]
APSE_R = 8.0
APSE_A = [math.radians(a) for a in (-72.0, -36.0, 0.0, 36.0, 72.0)]
APSE_PIER = [(X_APSE + APSE_R * math.cos(a), APSE_R * math.sin(a))
             for a in APSE_A]


def _dedupe(path):
    out = [np.array(path[0], float)]
    for p in path[1:]:
        p = np.array(p, float)
        if np.linalg.norm(p - out[-1]) > 1e-9:
            out.append(p)
    return np.array(out)


def _walk_ang(path, spacing, off=0.0):
    """Arc-length walk that also hands back the tangent bearing.

    Same lesson as _walk: space by cumulative length over the WHOLE path or
    the 25-chord apse arc eats a quarter of the stones.
    """
    pts = _dedupe(path)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    n = max(1, int(round(total / spacing)))
    out = []
    for k in range(n):
        d = ((k + 0.5 + off) % n) * total / n
        i = min(max(int(np.searchsorted(cum, d) - 1), 0), len(seg) - 1)
        u = (d - cum[i]) / seg[i]
        p = pts[i] + (pts[i + 1] - pts[i]) * u
        dv = pts[i + 1] - pts[i]
        out.append((p[0], p[1], math.atan2(-dv[1], dv[0])))
    return out, total


_BOXES = {}


def _local(hx, hy, hz, step):
    key = (round(hx, 3), round(hy, 3), round(hz, 3), step)
    if key not in _BOXES:
        _BOXES[key] = block(0.0, 0.0, 0.0, hx, hy, hz, step)
    return _BOXES[key]


def stone(cx, cy, cz, hx, hy, hz, ang=0.0, step=0.5):
    """One dressed block, turned ang about the vertical."""
    p, n = _local(hx, hy, hz, step)
    if abs(ang) < 1e-9:
        return p + np.array([cx, cy, cz]), n
    c, s = math.cos(ang), math.sin(ang)
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return p @ R.T + np.array([cx, cy, cz]), n @ R.T


def assemble(units):
    """A list of blocks in laying order -> points, normals, and a 0..1 clock
    saying when each point has been set."""
    if not units:
        z = np.zeros((0, 3), np.float32)
        return z, z, np.zeros(0, np.float32)
    P = np.vstack([u[0] for u in units]).astype(np.float32)
    N = np.vstack([u[1] for u in units]).astype(np.float32)
    d = max(1, len(units) - 1)
    O = np.concatenate([np.full(len(u[0]), i / float(d))
                        for i, u in enumerate(units)]).astype(np.float32)
    return P, N, O


def crypt_floor(step=1.25):
    """Paving, laid west to east.  Flat, so it is the plan of the room
    arriving before anything stands up."""
    xs = np.arange(X_CHOIR + 0.9, X_APSE + AISLE_Z, step)
    zs = np.arange(-AISLE_Z + 0.9, AISLE_Z, step)
    X, Z = np.meshgrid(xs, zs)
    X, Z = X.ravel(), Z.ravel()
    keep = inside_crypt(X, Z, inset=1.9)
    X, Z = X[keep], Z[keep]
    o = np.argsort(X)
    X, Z = X[o], Z[o]
    P = np.stack([X, np.full(len(X), Y_FOOT), Z], 1).astype(np.float32)
    N = np.tile(np.array([0.0, 1.0, 0.0]), (len(X), 1)).astype(np.float32)
    O = np.linspace(0.0, 1.0, len(X)).astype(np.float32)
    return P, N, O


# the direction the fixed camera looks FROM, read straight off _pose: screen
# depth is y*sin(pitch) + (-x*sin(yaw) + z*cos(yaw))*cos(pitch), and bigger
# is nearer, so the gradient of that is the way out of the screen.
_VIEW = np.array([-math.sin(math.radians(-58.0)) * math.cos(math.radians(28.0)),
                  math.sin(math.radians(28.0)),
                  math.cos(math.radians(-58.0)) * math.cos(math.radians(28.0))])
CRYPT_MID = (X_CHOIR + X_APSE + AISLE_Z) / 2.0


def crypt_wall(spacing=2.4):
    """Five courses, bonded: every other course starts half a stone along, so
    the vertical joints break the way masonry actually does.

    Split into the far half and the near half.  The near half is the only
    thing between this camera and the room, and the camera is not allowed to
    move, so while the room is being built the near wall is drawn instead of
    laid -- the same line-drawing convention the whole series already runs
    on.  It becomes stone when the room is sealed."""
    hh = (Y_SPRING - Y_FOOT) / (2.0 * N_COURSE)
    far, near, total = [], [], 0.0
    for k in range(N_COURSE):
        y = Y_FOOT + (2 * k + 1) * hh
        walk, total = _walk_ang(CRYPT_PATH, spacing, off=0.5 * (k % 2))
        for (x, z, ang) in walk:
            u = stone(x, y, z, spacing * 0.45, hh * 0.86, 0.95, ang)
            d = (x - CRYPT_MID) * _VIEW[0] + z * _VIEW[2]
            (near if d > 0.0 else far).append(u)
    return (assemble(far), assemble(near),
            len(far) + len(near), total)


def crypt_piers():
    """Four bays of the arcade plus a ring of five in the apse.  Drums, from
    the floor up, because that is the order they go on."""
    hh = (Y_SPRING - Y_FOOT) / 8.0
    seats = [(x, z, 0.0) for x in PIER_X for z in PIER_Z]
    seats += [(x, z, math.atan2(-z, x - X_APSE)) for (x, z) in APSE_PIER]
    units = []
    for d in range(4):
        y = Y_FOOT + (2 * d + 1) * hh
        for (x, z, ang) in seats:
            w = 0.95 if d < 3 else 1.18       # the capital spreads
            units.append(stone(x, y, z, w, hh * 0.88, w, ang))
    return assemble(units), len(seats)


RIB_H = 0.44
# the crown of the vault plus the depth of its own stone IS the floor above.
RISE = Y_CROWN - Y_SPRING - RIB_H


def _rib(a, b, rise=RISE, step=1.0):
    """Voussoirs along one arch, set from both springings inward so the
    keystone is the last stone in.  Level crowns: the rise is the same
    whatever the span, which makes the short arches steep and the long ones
    flat, which is what a groin vault over unequal bays has to do."""
    d = math.dist(a, b)
    n = max(3, int(round(d / step)))
    ang = math.atan2(-(b[1] - a[1]), b[0] - a[0])
    out = []
    for i in range(n):
        t = (i + 0.5) / n
        x = a[0] + (b[0] - a[0]) * t
        z = a[1] + (b[1] - a[1]) * t
        y = Y_SPRING + rise * math.sin(math.pi * t)
        out.append((min(t, 1.0 - t), stone(x, y, z, 0.58, RIB_H, 0.62, ang)))
    out.sort(key=lambda r: r[0])
    return [u for (_, u) in out]


def crypt_vault():
    """The ribs.  Transverse across each bay, longitudinal down the arcade,
    then the apse: a ring between the five, and five radiating out to the
    wall.  Arch by arch, in the order the centring would be struck."""
    arches = []
    for x in PIER_X:                                   # across
        arches += [((x, -AISLE_Z + 1.0), (x, PIER_Z[0])),
                   ((x, PIER_Z[0]), (x, PIER_Z[1])),
                   ((x, PIER_Z[1]), (x, AISLE_Z - 1.0))]
    for z in PIER_Z:                                   # along
        for i in range(len(PIER_X) - 1):
            arches.append(((PIER_X[i], z), (PIER_X[i + 1], z)))
    for i in range(len(APSE_PIER) - 1):                # the apse ring
        arches.append((APSE_PIER[i], APSE_PIER[i + 1]))
    for (x, z) in APSE_PIER:                           # and out to the wall
        a = math.atan2(z, x - X_APSE)
        arches.append(((x, z), (X_APSE + (AISLE_Z - 1.0) * math.cos(a),
                                (AISLE_Z - 1.0) * math.sin(a))))
    arches.append(((PIER_X[-1], PIER_Z[0]), APSE_PIER[0]))
    arches.append(((PIER_X[-1], PIER_Z[1]), APSE_PIER[-1]))
    units = []
    for (a, b) in arches:
        units += _rib(a, b)
    return assemble(units), len(arches)


def inside_crypt(x, z, inset=0.0):
    r = AISLE_Z - inset
    return (((x >= X_CHOIR + inset) & (x <= X_APSE) & (np.abs(z) <= r))
            | ((x > X_APSE) & ((x - X_APSE) ** 2 + z * z <= r * r)))


def crypt_slab(step=0.85):
    """The choir floor.  It goes on west to east and everything under it is
    gone.  This is the whole point of the episode, so it is deliberately
    featureless: one flat plane, no joints, nothing to look at."""
    xs = np.arange(X_CHOIR, X_APSE + AISLE_Z + step, step)
    zs = np.arange(-AISLE_Z, AISLE_Z + step, step)
    X, Z = np.meshgrid(xs, zs)
    X, Z = X.ravel(), Z.ravel()
    keep = inside_crypt(X, Z, inset=-0.1)
    X, Z = X[keep], Z[keep]
    o = np.argsort(X + 0.02 * np.abs(Z))
    X, Z = X[o], Z[o]
    P = np.stack([X, np.full(len(X), Y_CROWN), Z], 1).astype(np.float32)
    N = np.tile(np.array([0.0, 1.0, 0.0]), (len(X), 1)).astype(np.float32)
    O = np.linspace(0.0, 1.0, len(X)).astype(np.float32)
    return P, N, O


# ------------------------------------------------- part III: the choir walls
# The east end goes up first and gets used first.  Cologne: the choir was
# consecrated in 1322 and sealed off with a wall that was meant to be
# temporary, so services could be held in the finished part while the rest
# was a building site.  That wall came down in 1863.  It stood 541 years.
#
# So this episode builds a wall around a room, and then builds a worse wall
# across the open end, and the worse wall is the one with the story in it.
Y_ARCADE = AISLE_Y                 # 19.0 -- where the aisle roof will land
COURSE3 = 0.74                     # the course height the crypt was laid at
N_COURSE3 = int(round((Y_ARCADE - Y_SPRING) / COURSE3))      # 17
WIN_SILL, WIN_HEAD = 11.6, 17.5
WIN_W = 3.1
N_BAY = 14
BUT_PROJ = 1.30
SILL_K = int(round((WIN_SILL - Y_SPRING) / COURSE3 - 0.5))


def _on_path(pts, seg, cum, d):
    """Point, tangent bearing and OUTWARD normal at arc position d.

    Outward, not inward: stone() maps its local +z to (-dz, dx), which for
    this path points into the building, so the normal handed back here is
    the negative of that.  A buttress on the wrong side is a buttress in
    the aisle.
    """
    i = min(max(int(np.searchsorted(cum, d) - 1), 0), len(seg) - 1)
    u = (d - cum[i]) / seg[i]
    p = pts[i] + (pts[i + 1] - pts[i]) * u
    dv = pts[i + 1] - pts[i]
    L = float(np.linalg.norm(dv))
    return (p[0], p[1], math.atan2(-dv[1], dv[0]), dv[1] / L, -dv[0] / L)


_CP = _dedupe(CRYPT_PATH)
_CS = np.linalg.norm(np.diff(_CP, axis=0), axis=1)
_CC = np.concatenate([[0.0], np.cumsum(_CS)])
PERIM3 = float(_CC[-1])
BAY = PERIM3 / N_BAY


def _is_window(s, y):
    """A lancet in the middle of each bay: parallel sides, pointed head."""
    if not (WIN_SILL <= y <= WIN_HEAD):
        return False
    tt = (y - WIN_SILL) / (WIN_HEAD - WIN_SILL)
    w = WIN_W * (1.0 if tt < 0.62 else max(0.0, 1.0 - (tt - 0.62) / 0.38))
    off = (s % BAY) - 0.5 * BAY
    return abs(off) < 0.5 * w


def choir_wall(spacing=2.4):
    """The outer wall of the east end, off the crypt wall it stands on,
    seventeen courses to the height of the aisle roof.

    The buttresses and the holes go up together because they are the same
    decision.  A gothic wall is not a wall with windows cut into it -- it is
    a row of piers with the gaps left open, and the load from the roof goes
    down the piers, because it cannot go down a hole.
    """
    hh = COURSE3 / 2.0
    units, nw, nb, nslot = [], 0, 0, 0
    for k in range(N_COURSE3):
        y = Y_SPRING + (2 * k + 1) * hh
        proj = BUT_PROJ * (1.0 - 0.45 * k / float(N_COURSE3 - 1))
        for j in range(N_BAY):                       # the piers, first
            x, z, ang, ox, oz = _on_path(_CP, _CS, _CC, (j * BAY) % PERIM3)
            d = 0.95 + 0.5 * proj
            units.append(stone(x + ox * d, y, z + oz * d,
                               1.15, hh * 0.86, 0.5 * proj, ang))
            nb += 1
        n = max(1, int(round(PERIM3 / spacing)))     # then the wall between
        for i in range(n):
            s = ((i + 0.5 + 0.5 * (k % 2)) % n) * PERIM3 / n
            x, z, ang, ox, oz = _on_path(_CP, _CS, _CC, s)
            nslot += 1
            if _is_window(s, y):
                continue
            t = 0.95 + (0.30 if k == SILL_K else 0.0)   # the string course
            e = 0.16 if k == SILL_K else 0.0
            units.append(stone(x + ox * e, y, z + oz * e,
                               spacing * 0.45, hh * 0.86, t, ang))
            nw += 1
    return assemble(units), nw, nb, nslot


def choir_partition(step=1.05):
    """The temporary wall.

    Rubble, undressed, no window, no buttress, a ragged top, laid across the
    open west end of the choir as fast as it can be laid.  It is not meant to
    be there long.  The one at Cologne stood for five hundred and forty-one
    years.
    """
    zs = np.arange(-AISLE_Z + 0.55, AISLE_Z, step)
    ys = np.arange(Y_CROWN + 0.45, Y_ARCADE, step)
    units = []
    for r, y in enumerate(ys):
        # nobody courses the top of a wall that is coming down again
        drop = (0.36, 0.13)[min(len(ys) - 1 - r, 1)] if r >= len(ys) - 2 else 0.0
        for z in zs:
            if RNG.random() < drop:
                continue
            j = RNG.uniform(-0.14, 0.14, 3)
            units.append(stone(X_CHOIR + j[0], y + j[1], z + j[2],
                               0.55, 0.50 * step, 0.5 * step))
    return assemble(units), len(ys), len(zs)


# ---------------------------------------------------------------- stages
STAGES = [
    "THE FOUNDATION",
    "THE CRYPT",
    "THE CHOIR WALLS",
    "THE TRANSEPT",
    "THE NAVE PIERS",
    "THE AISLES",
    "THE TRIFORIUM",
    "THE CLERESTORY",
    "THE BUTTRESSES",
    "THE HIGH VAULT",
    "THE ROOF",
    "THE WEST FRONT",
    "THE ROSE WINDOW",
    "THE TOWERS",
    "THE SPIRE",
]

# ---------------------------------------------------------------- camera
GHOST = np.asarray(ghost_points())
_GHOST_RGB = (0.298, 0.373, 0.541)


def _pose(p):
    """The one view.  Fixed for the life of the series.

    Trap 1: the projector puts +y DOWN the screen, so the world's up axis is
    negated here.  The first render had the cathedral hanging by its spire.
    """
    yaw, pitch = math.radians(-58.0), math.radians(28.0)
    x, y, z = p[:, 0] - 51.0, p[:, 1] - 30.0, p[:, 2]
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    x1, z1 = x * cy_ + z * sy_, -x * sy_ + z * cy_
    cx_, sx_ = math.cos(pitch), math.sin(pitch)
    y1, z2 = y * cx_ - z1 * sx_, y * sx_ + z1 * cx_
    return np.stack([x1, -y1, z2], 1).astype(np.float32)


CAM = Camera(G).fit([_pose(GHOST)], margin=1.06)
LAMP = np.array([-0.52, 0.62, -0.59])
LAMP = LAMP / np.linalg.norm(LAMP)

STONES = foundation_stones()
TRENCH = trench_points()
NSTONE = len(STONES)

FLOOR = crypt_floor()
(WALL, WALL_N, NWALL, CRYPT_PERIM) = crypt_wall()
(PIERS, NPIER) = crypt_piers()
(VAULT, NARCH) = crypt_vault()
SLAB = crypt_slab()

# THE CLOSE SHOT.  Same yaw, same pitch, same _pose -- fitted to the crypt
# instead of the cathedral, so the cut at the end is a pure change of scale
# and nothing else.  That is the episode: this room, then this room in the
# building.  The fixed frame is still the last thing you see and two episodes
# still lay on top of each other, which was the whole reason for the rule.
_CRYPT_PTS = np.vstack([WALL[0], WALL_N[0], PIERS[0], VAULT[0], FLOOR[0]])
_pad = _CRYPT_PTS.copy()
_pad[:, 1] = _CRYPT_PTS[:, 1].min() - 5.2      # keep the caption clear
CAM_A = Camera(G).fit([_pose(np.vstack([_CRYPT_PTS, _pad]))], margin=1.10)

# a crypt is lit by lamps and nothing else, forever.  low, warm, from inside.
LAMP_C = np.array([0.44, 0.44, -0.78])
LAMP_C = LAMP_C / np.linalg.norm(LAMP_C)

# --- part III
(WALL3, NW3, NB3, NSLOT3) = choir_wall()
(PART, PART_ROWS, PART_COLS) = choir_partition()

# everything that was already standing when this episode opens.  no fade-in
# for any of it: part I's footings and part II's crypt have been there since
# the last two videos, and the slab is the floor this wall is built off.
_LEG_P = np.vstack([np.vstack([s[0] for s in STONES]),
                    WALL[0], WALL_N[0], SLAB[0]]).astype(np.float32)
_LEG_N = np.vstack([np.vstack([s[1] for s in STONES]),
                    WALL[1], WALL_N[1], SLAB[1]]).astype(np.float32)

# THE CLOSE SHOT, part III.  Same yaw, same pitch, same _pose -- fitted to
# the east end.  Same rule as part II: the cut is a pure change of scale.
#
# Fitted to the WALL, not to the room.  The choir floor is 41 m by 30 m and
# the wall on it is 12.6 m tall, so from an elevation of 28 degrees a camera
# that frames the floor is a camera pointed at a car park with a kerb round
# it.  Let the floor run off the bottom of the picture instead.
_E_PTS = np.vstack([WALL3[0], PART[0]])
_epad = _E_PTS.copy()
_epad[:, 1] = _E_PTS[:, 1].min() - 5.0         # keep the caption clear
CAM_B = Camera(G).fit([_pose(np.vstack([_E_PTS, _epad]))], margin=1.06)

# ---------------------------------------------------------------- timeline
T_GHOST, T_HOLD, T_DIG, T_LAY, T_END = 1.5, 2.4, 3.6, 9.9, 12.4

# part II
C_GHOST, C_PAVE, C_WALL = 1.0, (0.9, 2.7), (2.7, 5.8)
C_PIER, C_VAULT = (5.8, 7.3), (7.3, 10.0)
C_CUT = 11.1
C_SLAB, C_END = (12.0, 14.4), 15.6

# part III.  The cut moved to the END.  It was in the middle, matching part
# II, and the partition -- the whole point of the episode -- went up in the
# wide frame where the entire east end is 45 cells across and a 30 m wall
# added 200 lit cells nobody would notice.  So: build it all close, watch the
# cheap wall seal the good room, and only THEN pull out and find out what the
# good room is a corner of.
H_GHOST = 1.0
H_WALL = (1.1, 9.4)
H_PART = (10.3, 13.7)
H_CUT = 14.4
H_END = 17.8

T_ENDS = [T_END, C_END, H_END]
LAST = {}


def frames_for(stage):
    return int(round(T_ENDS[stage] * FPS))


def _put(buf, col, row, z, sh, mat, cover):
    ok = visible(G, col, row)
    if not ok.any():
        return
    col, row, z, sh = col[ok], row[ok], z[ok], sh[ok]
    flat, keep = zbuffer(G, col, row, z)
    c, r, s = col[keep], row[keep], sh[keep]
    idx = r * G.cols + c
    better = s > buf["sh"].ravel()[idx] * (0.0 if cover else 1.0)
    idx = idx[better]
    buf["sh"].ravel()[idx] = s[better]
    buf["mat"].ravel()[idx] = mat


def draw(f, stage):
    return (draw_foundation, draw_crypt, draw_choir)[stage](f, stage)


def _label(fr, t, stage, t0=0.8):
    """The numeral, then the stage.  Two lines from part III on.

    It was one line for I and II.  By III the string was 21 characters, the
    fitter shrank it to 4.2 cells a letter, and at that width the 3x3 halos
    of neighbouring letters merge -- CHOIR came out QHQOIII.  Split, the
    name gets 6.0 cells a letter and still fits at XIII . THE ROSE WINDOW,
    which is the longest this series will ever have to set.
    """
    boxes = []
    if t > t0:
        a = min(1.0, (t - t0) / 0.7)
        boxes.append(stamp(fr, roman(stage + 1), 6, 49, 127,
                           blend(BG, GOLD, a * 0.72)))
        boxes.append(stamp(fr, STAGES[stage], 10, 49, 139, blend(BG, GOLD, a)))
    LAST["boxes"] = boxes


def _paint(buf):
    fr = Frame(G, BG)
    on = buf["mat"] > 0
    cc, rr = np.meshgrid(np.arange(G.cols), np.arange(G.rows))
    fr.field(cc[on].ravel(), rr[on].ravel(), np.ones(on.sum(), bool),
             buf["sh"][on].ravel(), colour, RAMP,
             extra=buf["mat"][on].ravel().astype(float))
    LAST["ink"] = on
    LAST["mat"] = buf["mat"]
    return fr


def _grow(buf, part, u, mat, lamp, amb, gain, cam=None, near=1.0, far=0.86):
    """Draw the fraction u of an assembled element that has been set."""
    P, N, O = part
    m = O <= u
    if not m.any():
        return 0
    col, row, z = (cam or CAM).project(_pose(P[m]))
    sh = (amb + gain * lambert(N[m], lamp)) * depth_cue(z, near, far)
    _put(buf, col, row, z, np.clip(sh, 0.06, 1.0), mat, True)
    return int(m.sum())


def draw_crypt(f, stage):
    """Part II.  Two shots.  Up close the room gets built and lit; then one
    cut to the fixed frame of the whole cathedral, where it turns out to be
    a hand's width of warm stone, and the choir floor goes over it."""
    t = f / float(FPS)
    wide = t >= C_CUT
    cam = CAM if wide else CAM_A
    buf = {"sh": np.zeros((G.rows, G.cols)),
           "mat": np.zeros((G.rows, G.cols), np.int16)}

    gfade = min(1.0, t / C_GHOST)
    n = int(len(GHOST) * gfade)
    if n > 8:
        col, row, z = cam.project(_pose(GHOST[:n]))
        lift = 1.0 + 0.55 * min(1.0, max(0.0, (t - C_SLAB[1]) / 1.0))
        sh = ((0.20 + 0.34 * depth_cue(z, 1.0, 0.30))
              * (0.72 + 0.28 * gfade) * lift)
        _put(buf, col, row, z + 4000.0, sh, M_GHOST, False)

    # part I, standing.  no fade-in: it has been there since the last video.
    pts = np.vstack([s[0] for s in STONES])
    nrm = np.vstack([s[1] for s in STONES])
    col, row, z = cam.project(_pose(pts))
    sh = (0.24 + 0.62 * lambert(nrm, LAMP)) * depth_cue(z, 1.0, 0.86)
    _put(buf, col, row, z, np.clip(sh, 0.06, 1.0), M_OLD, True)

    def win(w):
        return min(1.0, max(0.0, (t - w[0]) / (w[1] - w[0])))

    us = win(C_SLAB)
    CW["rgb"] = blend(CRYPT, STONE, us)
    uw = win(C_WALL)
    lit = [_grow(buf, WALL, uw, M_CWALL, LAMP_C, 0.30, 0.74, cam)]
    for part, w in ((FLOOR, C_PAVE), (PIERS, C_PIER), (VAULT, C_VAULT)):
        lit.append(_grow(buf, part, win(w), M_CRYPT, LAMP_C, 0.30, 0.74, cam))
    LAST["crypt"] = lit

    # the near wall: drawn while you need to see past it, laid at the end
    P, N, O = WALL_N
    m = O <= uw
    if m.any():
        thin = np.zeros(len(P), bool)
        thin[::5 if wide else 9] = True
        sel = m & (thin | (us > 0.0))
        if sel.any():
            col, row, z = cam.project(_pose(P[sel]))
            if us > 0.02:
                sh = ((0.30 + 0.74 * lambert(N[sel], LAMP_C))
                      * depth_cue(z, 1.0, 0.86))
                _put(buf, col, row, z, np.clip(sh, 0.06, 1.0), M_CWALL, True)
            else:
                _put(buf, col, row, z + 4000.0, np.full(int(sel.sum()), 0.30),
                     M_GHOST, False)

    LAST["slab"] = _grow(buf, SLAB, us, M_SLAB, LAMP, 0.50, 0.46, cam,
                         far=0.9) if us > 0 else 0

    fr = _paint(buf)
    _label(fr, t, stage)
    return fr


def draw_choir(f, stage):
    """Part III.  Two shots again, and the cut is the argument.

    Close up, the east end is a finished church: a wall, buttresses, a row of
    lancets, a floor.  Then one cut to the fixed frame, where it turns out to
    be the far corner of something enormous that does not exist yet -- and
    the last thing that goes up is the cheap wall that closes it off, so the
    finished corner can be used while the rest of it is a drawing.
    """
    t = f / float(FPS)
    wide = t >= H_CUT
    cam = CAM if wide else CAM_B
    buf = {"sh": np.zeros((G.rows, G.cols)),
           "mat": np.zeros((G.rows, G.cols), np.int16)}

    gfade = min(1.0, t / H_GHOST)
    n = int(len(GHOST) * gfade)
    if n > 8:
        col, row, z = cam.project(_pose(GHOST[:n]))
        lift = 1.0 + 0.55 * min(1.0, max(0.0, (t - H_CUT - 0.9) / 1.1))
        sh = ((0.20 + 0.34 * depth_cue(z, 1.0, 0.30))
              * (0.72 + 0.28 * gfade) * lift)
        _put(buf, col, row, z + 4000.0, sh, M_GHOST, False)

    # parts I and II, weathered, already standing.  Held well back: the choir
    # floor alone is 1,100 m2 of flat pale plane and lit at part II's levels
    # it simply outshouts the thing this episode is about.
    col, row, z = cam.project(_pose(_LEG_P))
    sh = (0.17 + 0.44 * lambert(_LEG_N, LAMP)) * depth_cue(z, 1.0, 0.86)
    _put(buf, col, row, z, np.clip(sh, 0.05, 1.0), M_OLD, True)

    def win(w):
        return min(1.0, max(0.0, (t - w[0]) / (w[1] - w[0])))

    LAST["wall3"] = _grow(buf, WALL3, win(H_WALL), M_WALL3, LAMP,
                          0.26, 0.72, cam)
    up = win(H_PART)
    LAST["part"] = _grow(buf, PART, up, M_PART, LAMP, 0.26, 0.44, cam) \
        if up > 0 else 0

    fr = _paint(buf)
    _label(fr, t, stage)
    return fr


def draw_foundation(f, stage):
    t = f / float(FPS)
    buf = {"sh": np.zeros((G.rows, G.cols)),
           "mat": np.zeros((G.rows, G.cols), np.int16)}

    # --- the drawing of what it will be
    gfade = min(1.0, t / T_GHOST)
    n = int(len(GHOST) * gfade)
    if n > 8:
        wp = _pose(GHOST[:n])
        col, row, z = CAM.project(wp)
        # after the last stone is down, the drawing comes up a little: the
        # only ending this series can honestly have is the rest of it.
        lift = 1.0 + 0.55 * min(1.0, max(0.0, (t - T_LAY) / 1.4))
        sh = ((0.20 + 0.34 * depth_cue(z, 1.0, 0.30))
              * (0.72 + 0.28 * gfade) * lift)
        _put(buf, col, row, z + 4000.0, sh, M_GHOST, False)

    # --- the cut
    if t > T_HOLD:
        u = min(1.0, (t - T_HOLD) / (T_DIG - T_HOLD))
        n = int(len(TRENCH) * u)
        if n > 4:
            col, row, z = CAM.project(_pose(TRENCH[:n]))
            _put(buf, col, row, z, np.full(n, 0.62), M_EARTH, True)

    # --- the stones, one at a time, in the order a mason would walk it
    if t > T_DIG:
        u = min(1.0, (t - T_DIG) / (T_LAY - T_DIG))
        k = int(round(u * NSTONE))
        LAST["laid"] = k
        if k:
            pts = np.vstack([s[0] for s in STONES[:k]])
            nrm = np.vstack([s[1] for s in STONES[:k]])
            wp = _pose(pts)
            col, row, z = CAM.project(wp)
            sh = (0.28 + 0.78 * lambert(nrm, LAMP)) * depth_cue(z, 1.0, 0.86)
            _put(buf, col, row, z, np.clip(sh, 0.06, 1.0), M_STONE, True)
    else:
        LAST["laid"] = 0

    fr = Frame(G, BG)
    on = buf["mat"] > 0
    cc, rr = np.meshgrid(np.arange(G.cols), np.arange(G.rows))
    fr.field(cc[on].ravel(), rr[on].ravel(), np.ones(on.sum(), bool),
             buf["sh"][on].ravel(), colour, RAMP,
             extra=buf["mat"][on].ravel().astype(float))
    LAST["ink"] = on
    LAST["mat"] = buf["mat"]

    _label(fr, t, stage)
    return fr


def blend(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def colour(v, m):
    base = {M_GHOST: GHOST_RGB, M_STONE: STONE, M_EARTH: EARTH,
            M_OLD: OLD, M_CRYPT: CRYPT, M_SLAB: STONE,
            M_CWALL: CW["rgb"], M_WALL3: STONE, M_PART: ROUGH}[int(m)]
    t = np.clip(0.22 + 0.78 * v, 0.0, 1.0)
    return blend(BG, base, t)


GHOST_RGB = _GHOST_RGB


def roman(n):
    vals = ((100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"),
            (9, "IX"), (5, "V"), (4, "IV"), (1, "I"))
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out


# ---------------------------------------------------------------- lettering
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


def stamp(fr, s, cell_h, ccen, rcen, rgb, halo=BG):
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
            fr.put(c0 - 1 + i, r0 - 1 + j, "#", rgb if on else halo)
    return (c0, r0, w, h)


# ---------------------------------------------------------------- check
def check_crypt(stage):
    print("THE CATHEDRAL — part %s, %s" % (roman(stage + 1), STAGES[stage]))
    print("  wall blocks          %d in %d courses" % (NWALL, N_COURSE))
    print("  piers                %d" % NPIER)
    print("  vault arches         %d" % NARCH)
    print("  slab samples         %d" % len(SLAB[0]))

    # HELD OUT 1: the block count is never derived from the perimeter.  do it
    # the other way -- length over spacing, times courses -- and agree.
    implied = N_COURSE * CRYPT_PERIM / 2.4
    print("  crypt wall run       %.1f m -> %.0f blocks at 2.4 m x %d courses"
          % (CRYPT_PERIM, implied, N_COURSE))
    assert abs(implied - NWALL) / NWALL < 0.04, (implied, NWALL)

    # HELD OUT 2: part II must stand on part I.  every wall block has to sit
    # over a footing laid in the last video -- half a footing spacing is the
    # worst case, since the footings are 6.4 m apart on the same line.
    foot = np.array(_walk(6.4))
    off = []
    for k in range(N_COURSE):
        for (x, z, _) in _walk_ang(CRYPT_PATH, 2.4, off=0.5 * (k % 2))[0]:
            off.append(np.min(np.hypot(foot[:, 0] - x, foot[:, 1] - z)))
    off = np.array(off)
    print("  wall block to nearest part I footing: max %.2f m (footings are "
          "6.4 m apart, so 3.2 m is the worst legal case)" % off.max())
    assert off.max() < 3.3, off.max()

    # the room may not stick out through the wall of the building above it
    P = np.vstack([WALL[0], PIERS[0], VAULT[0], FLOOR[0]])
    out = ~inside_crypt(P[:, 0].astype(float), P[:, 2].astype(float),
                        inset=-1.2)
    print("  crypt points outside the choir/apse footprint: %d" % out.sum())
    assert out.sum() == 0, out.sum()
    print("  crypt top %.2f m, choir floor %.2f m, nave ghost %.0f m"
          % (float(P[:, 1].max()), Y_CROWN, NAVE_Y))
    assert float(P[:, 1].max()) <= Y_CROWN + 1e-3

    # the close shot has to actually contain the room
    col, row, _ = CAM_A.project(_pose(_CRYPT_PTS))
    print("  close frame          c%d..%d  r%d..%d of %dx%d"
          % (col.min(), col.max(), row.min(), row.max(), G.cols, G.rows))
    assert col.min() >= 0 and col.max() < G.cols, (col.min(), col.max())
    assert row.min() >= 0 and row.max() < G.rows, (row.min(), row.max())
    assert row.max() < 128, ("room runs into the caption", row.max())

    sheet, buried, wallpk = [], None, None
    for t in (1.6, 3.4, 5.0, 6.8, 8.6, 10.4, 11.6, 13.4, 15.2):
        fr = draw(int(t * FPS), stage)
        ink, mat = LAST["ink"], LAST["mat"]
        ncry, nslab = int((mat == M_CRYPT).sum()), int((mat == M_SLAB).sum())
        print("  t=%4.1f cov %.3f  ghost %5d old %4d wall %4d room %4d "
              "slab %4d" % (t, ink.mean(), (mat == M_GHOST).sum(),
                            (mat == M_OLD).sum(), (mat == M_CWALL).sum(),
                            ncry, nslab))
        assert 0.02 < ink.mean() < 0.60, ink.mean()
        for (c0, r0, w, h) in LAST["boxes"]:
            assert r0 - 1 >= G.safe_top, ("text above safe", r0)
            assert r0 + h + 1 <= G.safe_bot, ("text below safe", r0 + h)
            assert c0 - 1 >= 0 and c0 + w + 1 <= G.cols, ("width", c0, w)
        if abs(t - 11.6) < 1e-6:
            buried, wallpk = ncry, int((mat == M_CWALL).sum())
        sheet.append(fr)

    # HELD OUT 3: the episode's whole claim is that the choir floor buries
    # the ROOM -- floor, piers, vault -- while the wall survives as the
    # outside of the building.  measure both in the finished frame, not in
    # the code that placed them.
    draw(int(C_SLAB[1] * FPS + 6), stage)
    left = int((LAST["mat"] == M_CRYPT).sum())
    wall = int((LAST["mat"] == M_CWALL).sum())
    print("  room cells: %d lit -> %d under the floor (%.0f%% gone), wall "
          "survives %d -> %d" % (buried, left, 100.0 * (1 - left /
                                 float(buried)), wallpk, wall))
    assert buried > 120, buried
    assert left < 0.15 * buried, (buried, left)
    assert wall > 0.80 * wallpk, (wallpk, wall)

    contact(sheet, os.path.join(_HERE, "..", "content", "cath_sheet.png"),
            cols=3, labels=["1.6 paving", "3.4 wall", "5.0", "6.8 piers",
                            "8.6 vault", "10.4 lit", "11.6 CUT", "13.4 floor",
                            "15.2"])


def _excess(x, z):
    """How far outside the building's footprint line a point sits."""
    x, z = np.asarray(x, float), np.asarray(z, float)
    straight = np.abs(z) - AISLE_Z
    round_ = np.hypot(x - X_APSE, z) - AISLE_Z
    return np.where(x > X_APSE, round_, straight)


def check_choir(stage):
    print("THE CATHEDRAL — part %s, %s" % (roman(stage + 1), STAGES[stage]))
    print("  wall run             %.1f m in %d bays of %.2f m"
          % (PERIM3, N_BAY, BAY))
    print("  courses              %d of %.2f m, %.1f m to %.1f m"
          % (N_COURSE3, COURSE3, Y_SPRING, Y_ARCADE))
    print("  wall blocks          %d set, %d slots, %d buttress blocks"
          % (NW3, NSLOT3, NB3))
    print("  partition            %d courses x %d, rubble" % (PART_ROWS,
                                                              PART_COLS))

    # HELD OUT 1 -- how much of this wall is hole.  The blocks were dropped
    # one slot at a time by a boolean test.  Get the same number the other
    # way, from areas alone: a lancet is WIN_W wide for the bottom 62% of its
    # height and tapers to a point over the rest, so its area is
    # WIN_W * H * (0.62 + 0.38/2).  The two must agree or the window shape
    # on screen is not the window shape in the arithmetic.
    H = WIN_HEAD - WIN_SILL
    open_area = N_BAY * WIN_W * H * (0.62 + 0.38 / 2.0)
    wall_area = PERIM3 * N_COURSE3 * COURSE3
    pred = open_area / wall_area
    meas = (NSLOT3 - NW3) / float(NSLOT3)
    print("  glazed: %.1f m2 of %.0f m2 -> predicted %.1f%%, built %.1f%%"
          % (open_area, wall_area, 100 * pred, 100 * meas))
    assert abs(pred - meas) / pred < 0.15, (pred, meas)

    # HELD OUT 2 -- the buttresses have to land BETWEEN the windows.  Nothing
    # in the code guarantees that: the piers are placed at j*BAY and the
    # holes come out of a modulo on a walk that is offset half a stone on
    # alternate courses.  So measure the holes off the mask itself -- sample
    # the wall line finely at mid-window height, find the runs -- and check
    # what comes back against where the piers actually went.
    ss = np.arange(0.0, PERIM3, 0.02)
    mask = np.array([_is_window(s, 0.5 * (WIN_SILL + WIN_HEAD)) for s in ss])
    runs, cur = [], None
    for i, m in enumerate(mask):
        if m and cur is None:
            cur = i
        elif not m and cur is not None:
            runs.append((ss[cur], ss[i - 1]))
            cur = None
    if cur is not None:
        runs.append((ss[cur], ss[-1]))
    widths = [b - a for (a, b) in runs]
    cents = np.array([0.5 * (a + b) for (a, b) in runs])
    print("  holes measured off the mask: %d, mean width %.2f m (drawn %.2f)"
          % (len(runs), float(np.mean(widths)), WIN_W))
    assert len(runs) == N_BAY, len(runs)
    assert abs(np.mean(widths) - WIN_W) < 0.1, np.mean(widths)
    piers = np.array([(j * BAY) % PERIM3 for j in range(N_BAY)])
    gap = np.abs(piers[:, None] - cents[None, :])
    gap = np.minimum(gap, PERIM3 - gap).min(1)
    print("  nearest hole to a buttress: %.2f m (half a bay is %.2f)"
          % (gap.min(), 0.5 * BAY))
    assert gap.min() > 0.35 * BAY, gap.min()

    # HELD OUT 3 -- the temporary wall has to actually close the hole, or the
    # whole episode is about nothing.  Grid the opening and ask, of each
    # point, whether there is rubble on it.
    zz, yy = np.meshgrid(np.linspace(-AISLE_Z + 0.6, AISLE_Z - 0.6, 40),
                         np.linspace(Y_CROWN + 0.6, Y_ARCADE - 1.2, 18))
    P = PART[0]
    covered = 0
    for (y, z) in zip(yy.ravel(), zz.ravel()):
        d = np.hypot(P[:, 1] - y, P[:, 2] - z)
        covered += int(d.min() < 0.85)
    frac = covered / float(yy.size)
    print("  opening sealed       %.1f%% of the gap has rubble on it"
          % (100 * frac))
    assert frac > 0.97, frac
    assert abs(float(P[:, 0].mean()) - X_CHOIR) < 0.1
    assert float(P[:, 2].min()) < -AISLE_Z + 1.5
    assert float(P[:, 2].max()) > AISLE_Z - 1.5

    # a buttress is MEANT to stand outside the mass line -- that is what it
    # is -- but only by its own projection, and nothing else may.
    A = np.vstack([WALL3[0], PART[0]])
    ex = _excess(A[:, 0], A[:, 2])
    print("  furthest outside the mass line: %.2f m (wall half-thickness "
          "0.95 + buttress %.2f = %.2f)" % (ex.max(), BUT_PROJ,
                                            0.95 + BUT_PROJ))
    assert ex.max() < 0.95 + BUT_PROJ + 0.35, ex.max()
    assert float(A[:, 1].max()) <= Y_ARCADE + 0.4, float(A[:, 1].max())

    # both frames must hold it, and the cut has to be worth making
    for nm, c in (("fixed", CAM), ("close", CAM_B)):
        col, row, _ = c.project(_pose(np.vstack([WALL3[0], PART[0]])))
        w = col.max() - col.min() + 1
        print("  %-5s frame  c%d..%d r%d..%d  (%d cells wide)"
              % (nm, col.min(), col.max(), row.min(), row.max(), w))
        assert col.min() >= 0 and col.max() < G.cols, (col.min(), col.max())
        assert row.min() >= 0 and row.max() < G.rows, (row.min(), row.max())
        if nm == "fixed":
            wide_w = w
        else:
            assert row.max() < 128, ("wall runs into the caption", row.max())
            print("  the cut magnifies    %.1fx" % (w / float(wide_w)))
            assert w / float(wide_w) > 1.9, w / float(wide_w)

    sheet = []
    for t in (0.8, 2.6, 4.6, 6.6, 9.0, 11.2, 12.8, 14.9, 17.2):
        fr = draw(int(t * FPS), stage)
        ink, mat = LAST["ink"], LAST["mat"]
        print("  t=%4.1f cov %.3f  ghost %5d old %5d wall %5d rubble %4d"
              % (t, ink.mean(), (mat == M_GHOST).sum(), (mat == M_OLD).sum(),
                 (mat == M_WALL3).sum(), (mat == M_PART).sum()))
        assert 0.02 < ink.mean() < 0.60, ink.mean()
        for (c0, r0, w, h) in LAST["boxes"]:
            assert r0 - 1 >= G.safe_top, ("text above safe", r0)
            assert r0 + h + 1 <= G.safe_bot, ("text below safe", r0 + h)
            assert c0 - 1 >= 0 and c0 + w + 1 <= G.cols, ("width", c0, w)
        sheet.append(fr)

    # HELD OUT 4 -- the wall has to be visibly full of holes in the finished
    # close shot.  Count ink in the window band on screen against the band
    # just under the sill, which is solid.  A wall that came out solid would
    # pass every number above and fail here.
    draw(int((H_WALL[1] + 0.5) * FPS), stage)
    m = LAST["mat"] == M_WALL3
    rows = np.nonzero(m.any(1))[0]
    top, bot = rows.min(), rows.max()
    band = int(top + 0.30 * (bot - top)), int(top + 0.55 * (bot - top))
    solid = int(top + 0.68 * (bot - top)), int(top + 0.90 * (bot - top))
    a = m[band[0]:band[1]].mean()
    b = m[solid[0]:solid[1]].mean()
    print("  window band %.3f ink vs solid band %.3f -> %.0f%% lighter"
          % (a, b, 100 * (1 - a / b)))
    assert a < 0.86 * b, (a, b)

    contact(sheet, os.path.join(_HERE, "..", "content", "cath_sheet.png"),
            cols=3, labels=["0.8 ghost", "2.6 wall", "4.6", "6.6 sill",
                            "9.0 windows", "11.2 rubble", "12.8", "14.9 CUT",
                            "17.2"])


def check(stage):
    if stage == 1:
        return check_crypt(stage)
    if stage == 2:
        return check_choir(stage)
    print("THE CATHEDRAL — part %s, %s" % (roman(stage + 1), STAGES[stage]))
    print("  masses in the ghost  %d" % len(MASSES))
    print("  ghost points         %d" % len(GHOST))
    print("  footing stones       %d" % NSTONE)
    print("  stages defined       %d" % len(STAGES))

    # the ghost must FIT: the fixed camera is fitted to the finished
    # building, so every future part is already guaranteed to be in frame.
    col, row, _ = CAM.project(_pose(GHOST))
    print("  ghost bbox           c%d..%d  r%d..%d"
          % (col.min(), col.max(), row.min(), row.max()))
    assert col.min() >= 0 and col.max() < G.cols, (col.min(), col.max())
    assert row.min() >= 0 and row.max() < G.rows, (row.min(), row.max())
    wfill = (col.max() - col.min() + 1) / float(G.cols)
    hfill = (row.max() - row.min() + 1) / float(G.rows)
    print("  fills                %.0f%% wide  %.0f%% tall"
          % (100 * wfill, 100 * hfill))
    assert wfill > 0.78, wfill
    assert hfill > 0.62, hfill

    # HELD OUT: the stones are laid along FOOTPRINT at a fixed spacing, but
    # the count is never computed from the perimeter.  Do it the other way
    # and the two must agree.
    per = sum(math.dist(FOOTPRINT[i], FOOTPRINT[i + 1])
              for i in range(len(FOOTPRINT) - 1))
    implied = per / 6.4
    print("  footprint perimeter  %.1f m -> %.1f stones at 6.4 m"
          % (per, implied))
    assert abs(implied - NSTONE) / NSTONE < 0.04, (implied, NSTONE)

    # and independently of the count: consecutive footings must actually be
    # a footing apart.  a right count with a bad distribution passes the
    # line above and fails this one.
    walk = np.array(_walk(6.4))
    gaps = np.linalg.norm(np.diff(walk, axis=0), axis=1)
    close = float(np.linalg.norm(walk[0] - walk[-1]))
    print("  chord between footings %.2f .. %.2f m  (arc is 6.44; a chord "
          "across a corner is legitimately shorter)" % (gaps.min(),
                                                        gaps.max()))
    print("  loop closes at       %.2f m" % close)
    # a chord can never EXCEED the arc it subtends, so anything over 6.44
    # means a skipped stone.  the floor catches a duplicate (~0) while
    # letting the square corners of a cruciform plan be what they are.
    assert gaps.max() <= 6.45, gaps.max()
    assert gaps.min() > 4.2, gaps.min()
    assert 4.2 < close < 6.45, close

    sheet = []
    for t in (0.6, 1.6, 2.9, 4.4, 6.2, 8.0, 9.6, 11.0, 12.2):
        fr = draw(int(t * FPS), stage)
        ink, mat = LAST["ink"], LAST["mat"]
        rr = np.nonzero(ink.any(1))[0]
        cc = np.nonzero(ink.any(0))[0]
        print("  t=%4.1f cov %.3f  r%d..%d c%d..%d  ghost %5d stone %5d "
              "earth %4d  laid %d"
              % (t, ink.mean(), rr.min(), rr.max(), cc.min(), cc.max(),
                 (mat == M_GHOST).sum(), (mat == M_STONE).sum(),
                 (mat == M_EARTH).sum(), LAST["laid"]))
        assert 0.02 < ink.mean() < 0.60, ink.mean()
        for (c0, r0, w, h) in LAST["boxes"]:
            assert r0 - 1 >= G.safe_top, ("text above safe", r0)
            assert r0 + h + 1 <= G.safe_bot, ("text below safe", r0 + h)
            assert c0 - 1 >= 0 and c0 + w + 1 <= G.cols, ("width", c0, w)
        sheet.append(fr)

    # by the end, every stone is down and the stone must OUTWEIGH the ghost
    assert LAST["laid"] == NSTONE, LAST["laid"]
    contact(sheet, os.path.join(_HERE, "..", "content", "cath_sheet.png"),
            cols=3, labels=["0.6", "1.6 ghost", "2.9 dig", "4.4", "6.2",
                            "8.0", "9.6", "11.0", "12.2 set"])


def main(stage, out):
    FRAMES = frames_for(stage)
    with Encoder(out, G, fps=FPS) as enc:
        for f in range(FRAMES):
            enc.write(draw(f, stage))
            if f % 60 == 0:
                print("  %d/%d" % (f, FRAMES), flush=True)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    path = os.path.join(_HERE, "..", "content",
                        "cathedral_%02d.mp4" % (a.stage + 1))
    check(a.stage) if a.check else main(a.stage, path)
