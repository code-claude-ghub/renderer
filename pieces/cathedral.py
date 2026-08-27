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
sys.path.insert(0, _HERE)

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
M_TRAN, M_PIER = 10, 11        # part IV: the arms, and the four that carry it
M_NAVE = 12                    # part V: the arcade

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


# ------------------------------------------------------ part IV: transept
# The crossbar.  Two arms reaching 26 m either side of the centreline, and
# the four piers standing where they cross the nave.
#
# What is true of THIS part and no other: the transept is the thing that
# makes the plan a cross.  And the plan is the one view of a cathedral that
# nobody in it can get.  Part I dug the whole footprint -- so the cross has
# been lying on the ground since the first video in this series and has not
# been visible in a single frame of it.  The fixed camera cannot show it.
# That is not a limitation of the camera.  It is the subject.
#
# The transept has no crypt under it, so its wall starts at the footings and
# not at the crypt crown: 22 courses instead of the choir's 17.  The sill and
# the head stay at the same ABSOLUTE height as part III, because a string
# course runs round a building at one level and does not step.
Y_TFOOT = Y_FOOT
N_COURSE4 = int(round((Y_ARCADE - Y_TFOOT) / COURSE3))          # 22
CROSS_Z = 9.0                      # the crossing square, straight off MASSES
TRAN_Z = 26.0                      # how far the arms reach, ditto
PIER_HW = 1.5                      # 3.0 m square.  sized in check_transept.
PIER_TOP = 22.0                    # they do not stop where the walls stop

# Walked so that _on_path's normal comes out pointing OUT of the building on
# every segment.  The arms are re-entrant -- the nave already occupies
# |z| < 15 -- so "outward" flips sense between the two of them, and the fix
# is to walk the south arm the other way round rather than special-case it.
ARM_N = [(X_TRAN, -AISLE_Z), (X_TRAN, -TRAN_Z),
         (X_CHOIR, -TRAN_Z), (X_CHOIR, -AISLE_Z)]
ARM_S = [(X_CHOIR, AISLE_Z), (X_CHOIR, TRAN_Z),
         (X_TRAN, TRAN_Z), (X_TRAN, AISLE_Z)]


def _arm_piers(corners, total, target=5.8):
    """Buttress positions along one arm.  A pier lands on every corner --
    that is where a wall most needs one -- and the straight runs between are
    divided as near the choir's bay as they will go."""
    out, edges = set(), [0.0] + list(corners) + [total]
    for a, b in zip(edges[:-1], edges[1:]):
        n = max(1, int(round((b - a) / target)))
        for k in range(n + 1):
            out.add(round(a + (b - a) * k / n, 6))
    return sorted(out)


def _arms():
    out = []
    for path in (ARM_N, ARM_S):
        P = _dedupe(path)
        S = np.linalg.norm(np.diff(P, axis=0), axis=1)
        C = np.concatenate([[0.0], np.cumsum(S)])
        tot = float(C[-1])
        out.append((P, S, C, tot, _arm_piers((float(C[1]), float(C[2])), tot)))
    return out


ARMS = _arms()


def _tran_window(s, y, piers):
    """One lancet in the middle of each bay.  The bays are not all the same
    width here -- the corners take a pier wherever they fall -- so the hole
    is found from the bay it is in rather than from a modulo, which also
    guarantees no window is ever cut across a corner."""
    if not (WIN_SILL <= y <= WIN_HEAD):
        return False
    for a, b in zip(piers[:-1], piers[1:]):
        if a <= s <= b:
            tt = (y - WIN_SILL) / (WIN_HEAD - WIN_SILL)
            w = WIN_W * (1.0 if tt < 0.62
                         else max(0.0, 1.0 - (tt - 0.62) / 0.38))
            return abs(s - 0.5 * (a + b)) < 0.5 * w
    return False


def transept_wall(spacing=2.4):
    """Both arms, course by course, north then south so they rise together."""
    hh = COURSE3 / 2.0
    units, nw, nb, nslot = [], 0, 0, 0
    for k in range(N_COURSE4):
        y = Y_TFOOT + (2 * k + 1) * hh
        proj = BUT_PROJ * (1.0 - 0.45 * k / float(N_COURSE4 - 1))
        for (P, S, C, tot, piers) in ARMS:
            for s in piers:
                x, z, ang, ox, oz = _on_path(P, S, C, min(s, tot - 1e-6))
                d = 0.95 + 0.5 * proj
                units.append(stone(x + ox * d, y, z + oz * d,
                                   1.15, hh * 0.86, 0.5 * proj, ang))
                nb += 1
            n = max(1, int(round(tot / spacing)))
            for i in range(n):
                s = min(((i + 0.5 + 0.5 * (k % 2)) % n) * tot / n,
                        tot - 1e-6)
                x, z, ang, ox, oz = _on_path(P, S, C, s)
                nslot += 1
                if _tran_window(s, y, piers):
                    continue
                th = 0.95 + (0.30 if abs(y - WIN_SILL) < 0.5 * COURSE3 else 0.0)
                e = 0.16 if abs(y - WIN_SILL) < 0.5 * COURSE3 else 0.0
                units.append(stone(x + ox * e, y, z + oz * e,
                                   spacing * 0.45, hh * 0.86, th, ang))
                nw += 1
    return assemble(units), nw, nb, nslot


def crossing_piers():
    """Four piers on the corners of an 18 m square, and everything above the
    roof one day lands on them.  They are the thickest thing in the building
    for that reason -- see the bearing-stress check.  They go up first and
    they finish taller than the walls, because they are nowhere near done."""
    units = []
    ys = np.arange(Y_TFOOT, PIER_TOP, COURSE3)
    for k, y in enumerate(ys):
        for x in (X_TRAN, X_CHOIR):
            for z in (-CROSS_Z, CROSS_Z):
                j = RNG.uniform(-0.03, 0.03, 2)
                units.append(stone(x + j[0], y + 0.5 * COURSE3, z + j[1],
                                   PIER_HW, COURSE3 * 0.43, PIER_HW))
    return assemble(units), len(ys)


# ---------------------------------------------------------- part V: arcade
# Twenty piers in two rows, and nothing on top of them.
#
# What is true of THIS part and no other: it is the first repeat.  The
# footings follow the ground, the crypt is one room, the choir wall follows a
# path, the transept is a one-off.  An arcade is one thing, built again, and
# the whole job is that the second one is the same as the first.  It is also
# the first stone in this building that is not part of the outside -- every
# course laid in parts I to IV has been perimeter.  When these are up there
# is a nave and there are aisles, which is to say the building has an inside
# for the first time.
#
# THE BAY IS DECIDED HERE, ONCE, FOR EVERY EPISODE AFTER THIS ONE.  The
# aisle windows, the vault springing, the buttresses and the roof trusses all
# have to land on these lines.  So it is not chosen by eye: the nave is 62 m
# and the module the choir and the transept were already set out on is 5.8 m
# (see _arm_piers), and 11 bays is the division of 62 that lands nearest it.
N_BAY5 = 11
BAY5 = (X_TRAN - X_NAVE) / float(N_BAY5)       # 5.636 m
PIER5_HW = 1.2                                 # 2.4 m square. sized in check.
N_COURSE5 = 15
Y_CAP5 = Y_FOOT + N_COURSE5 * COURSE3          # 13.80 m to the abacus

# Why the capital lands there: the arcade arch has to get across a bay and
# still duck under the aisle roof.  A two-centred arch spanning 5.636 m rises
# sqrt(3)/2 of its span, so its crown sits 4.88 m above the capital, and
# Y_ARCADE is 19.0.  Fifteen courses is the tallest whole number that clears
# it, by 0.32 m.  I wrote fourteen first and the check said fourteen was not
# the tallest, which is the second time this series has been corrected by
# arithmetic it was carrying anyway.  Asserted in check_nave.
ARCH_RISE5 = 0.5 * math.sqrt(3.0) * BAY5


def nave_piers():
    """East to west, a bay at a time, north and south rising together.

    East to west because that is the direction the building is going: the
    choir end gets finished and used while the nave is still a drawing, and
    the arcade's east end is already standing -- it is one of part IV's
    crossing piers, which went up last episode without being told what for.

    k = 0 is left out on purpose.  That one is engaged in the west front and
    the west front is part XII.
    """
    units = []
    xs = [X_NAVE + k * BAY5 for k in range(N_BAY5 - 1, 0, -1)]
    for x in xs:
        for c in range(N_COURSE5):
            # a base at the bottom, an abacus at the top, square between
            hw = PIER5_HW * (1.22 if (c < 2 or c == N_COURSE5 - 1) else 1.0)
            y = Y_FOOT + (c + 0.5) * COURSE3
            for z in (-NAVE_Z, NAVE_Z):
                j = RNG.uniform(-0.025, 0.025, 2)
                units.append(stone(x + j[0], y, z + j[1],
                                   hw, COURSE3 * 0.43, hw))
    return assemble(units), len(xs), N_COURSE5


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

# --- part IV
(WALL4, NW4, NB4, NSLOT4) = transept_wall()
(PIERS4, N_PCOURSE) = crossing_piers()

# Part I's footings, kept SEPARATE from everything else that is standing.
#
# From overhead they are the whole argument -- 350 m of dotted line that has
# been a cross since the first video -- so they get let up as the camera
# rises.  The rest does not, and the reason is the choir slab: it is 1,100 m2
# of flat pale plane whose normal points straight at the lens up there, and
# lit to match the footings it turns the head of the cross into a light bulb.
# Overhead, a floor is not the drawing.  The walls are.
_FOOT_P = np.vstack([s[0] for s in STONES]).astype(np.float32)
_FOOT_N = np.vstack([s[1] for s in STONES]).astype(np.float32)
_REST_P = np.vstack([WALL[0], WALL_N[0], SLAB[0],
                     WALL3[0], PART[0]]).astype(np.float32)
_REST_N = np.vstack([WALL[1], WALL_N[1], SLAB[1],
                     WALL3[1], PART[1]]).astype(np.float32)
_LEG4_P = np.vstack([_FOOT_P, _REST_P]).astype(np.float32)
_LEG4_N = np.vstack([_FOOT_N, _REST_N]).astype(np.float32)

# every point the move has to keep on screen, thinned -- the fit only needs
# the silhouette and this runs once a frame.
_MOVE_ALL = np.vstack([GHOST, WALL4[0][::7], PIERS4[0][::7],
                       _LEG4_P[::7]]).astype(np.float32)




def _pose_at(p, yaw_deg, pitch_deg):
    """_pose, with the two angles let out.  _pose itself is untouched and
    check_transept asserts the two agree at (-58, 28), because rule 1 of this
    series is that the established view never drifts."""
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    x, y, z = p[:, 0] - 51.0, p[:, 1] - 30.0, p[:, 2]
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    x1, z1 = x * cy_ + z * sy_, -x * sy_ + z * cy_
    cx_, sx_ = math.cos(pitch), math.sin(pitch)
    y1, z2 = y * cx_ - z1 * sx_, y * sx_ + z1 * cx_
    return np.stack([x1, -y1, z2], 1).astype(np.float32)


# THE ONE VIEW THIS SERIES IS NOT ALLOWED TO HAVE.
#
# Rule 1 says the camera never moves, and the reason it says so is that every
# episode has to lay on top of every other one.  Part IV needs the plan --
# there is no other way to see a cross -- so the camera leaves, and then it
# comes back, and the episode still opens and closes in the fixed frame.  The
# rule keeps its purpose and gives up its literal wording, which is what part
# II did to it as well.
#
# Straight down, and turned so the apse is at the top of the picture.  A
# latin cross has its long arm BELOW the crossbar: the short way round from
# the established yaw puts the west front at the top instead, which draws an
# inverted cross, which is a thing this piece is not about.  So it turns the
# long way, 148 degrees, and the turn is most of what the move is.
PLAN_YAW, PLAN_PITCH = 90.0, 90.0
PLAN_PAD = 44.0                    # metres of nothing reserved under the west
                                   # front.  It was 21, and at 21 the plan ran
                                   # down over the numeral -- which every check
                                   # passed, because they all asked whether the
                                   # TEXT was in the safe area and none of them
                                   # asked whether anything was on top of it.
_PLAN_PTS = _pose_at(GHOST, PLAN_YAW, PLAN_PITCH)
_ppad = _PLAN_PTS.copy()
_ppad[:, 1] = _PLAN_PTS[:, 1].max() + PLAN_PAD
CAM_P = Camera(G).fit([np.vstack([_PLAN_PTS, _ppad])], margin=1.02)


def _mix_cam(u):
    """The camera during the move.

    Lerping offset and scale between the two ends does NOT frame the poses in
    between: half way up, at 59 degrees of pitch and 16 of yaw, the cathedral
    is 125 cells wide across a 98 cell grid and both ends of it are off the
    picture.  A camera that is correct at both ends of a move and wrong in
    the middle is the whole trap.

    So the lerp is pulled toward an actual fit of THIS pose, weighted to zero
    at both ends -- which pins the two views the series is allowed to have --
    and hardest in the middle, where the lerp is worst.  What it looks like
    is a camera drawing back as it swings, which is what you would do.
    """
    yaw = -58.0 + (PLAN_YAW + 58.0) * u
    pitch = 28.0 + (PLAN_PITCH - 28.0) * u
    off = CAM.off * (1.0 - u) + CAM_P.off * u
    scale = CAM.scale * (1.0 - u) + CAM_P.scale * u

    w = (4.0 * u * (1.0 - u)) ** 0.35
    if w > 1e-6:
        P = _pose_at(_MOVE_ALL, yaw, pitch)
        pad = P.copy()
        pad[:, 1] = P[:, 1].max() + PLAN_PAD * u      # keep the caption clear
        f = Camera(G).fit([np.vstack([P, pad])], margin=1.06)
        off = off * (1.0 - w) + f.off * w
        scale = scale * (1.0 - w) + min(scale, f.scale) * w

    c = Camera(G)
    c.off, c.scale = off, scale
    return c


# --- part V
(PIERS5, N_PIER5, N_PC5) = nave_piers()

# Everything standing when this episode opens: four videos of stone.  Held
# back at part III's levels for the same reason -- by now the new work is
# twenty slim piers against four episodes of wall, and if the legacy is lit
# to match, the episode is invisible inside its own building.
_LEG5_P = np.vstack([_LEG4_P, WALL4[0], PIERS4[0]]).astype(np.float32)
_LEG5_N = np.vstack([_LEG4_N, WALL4[1], PIERS4[1]]).astype(np.float32)

# THE CLOSE SHOT, part V -- and it is the first one in this series that is
# not the established view at a different scale.
#
# Parts II and III cut to a close shot at the SAME yaw and pitch, so the cut
# was a pure change of scale.  That cannot work here, and the reason is
# arithmetic rather than taste.  Adjacent piers separate on screen only when
# the horizontal step between them beats the width one pier reads as, and a
# square pier turned 58 degrees to the camera shows two faces at once:
#
#     step  = BAY5 * cos(yaw)                  5.64 m at yaw 0
#     reads = 2.93 * (cos(yaw) + sin(yaw))     4.08 m at yaw 58
#
# At the established yaw of 58 degrees the step is 3.23 m and one pier reads
# 4.08 m wide, so neighbours overlap by 0.85 m and the arcade is a solid
# band AT EVERY SCALE.  Zooming in does not help, because both quantities
# scale together.  They come apart below about 38 degrees.
#
# I found this the expensive way.  A still at the established view looked
# fine to me -- I could see vertical stripes and read them as piers.  They
# were the glyphs of the character ramp.  The check that counts separate
# runs of stone said ONE, twice, and it was right both times.
#
# THE YAW IS NOT CHOSEN EITHER.  There are two rows of piers 16 m apart, and
# at a general angle the far row lands in the near row's gaps and fills them
# in -- which is what the first sweep found, and it is why 14, 28 and 34
# degrees all measured as ONE run while 20 measured as ten.  The far row
# hides behind the near one exactly when its sideways offset is a whole
# number of bays:
#
#     2 * NAVE_Z * tan(yaw) = BAY5      ->  yaw = 19.406 degrees
#
# So the angle this episode is shot at is the angle at which a cathedral's
# two arcades line up, and there is only one of those under 38 degrees.
P_YAW = -math.degrees(math.atan(BAY5 / (2.0 * NAVE_Z)))
P_PITCH = 18.0


def _pose_n(p):
    return _pose_at(p, P_YAW, P_PITCH)


# Fitted to the piers AND to the drawing of what stands on them: the nave,
# the aisles and the roof over them, up to the ridge.  Fitted to the piers
# alone it is a low band of stone in an empty frame, which is true and dull.
# With the nave ghost in, the top of the picture is twenty-five metres of
# building that does not exist and the piers are holding all of it.
_NAVE_GH = GHOST[(GHOST[:, 0] > -1.0) & (GHOST[:, 0] < 63.0)
                 & (GHOST[:, 1] <= 46.5)]
_N_PTS = np.vstack([PIERS5[0], _NAVE_GH]).astype(np.float32)
_npad = PIERS5[0].copy()
_npad[:, 1] = PIERS5[0][:, 1].min() - 5.0      # keep the caption clear
CAM_N = Camera(G).fit([_pose_n(np.vstack([_N_PTS, _npad]))], margin=1.05)


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

# part IV.  The shortest episode so far, on purpose: this channel measured
# 200 videos and retention falls monotonically with length, so the series
# creeping 12.4 -> 15.6 -> 17.8 was going the wrong way and nobody had said
# so.  The build is quick and the plan is held long, because the plan is the
# episode and the stone going up is only how you get there.
Q_GHOST = 0.9
Q_PIER = (0.8, 2.5)
Q_WALL = (2.5, 6.0)
Q_UP = (6.1, 7.5)
Q_DOWN = (9.7, 10.9)
Q_END = 11.8

# part V.  Shorter again -- 12.4, 15.6, 17.8, 11.8, and now 8.8.  The one
# thing this episode has to do is let you COUNT them, so the piers get the
# whole middle of it and there is no cut, no second camera and no excursion.
# It is the first episode since part I that never leaves the established
# view, and that is not a rule being obeyed, it is that nothing here needs
# another angle.
P_GHOST = 0.9
P_CUT = 1.8
P_PIER = (0.9, 7.0)
P_END = 8.8

T_ENDS = [T_END, C_END, H_END, Q_END, P_END]
LAST = {}


def _smooth(t, a, b):
    u = min(1.0, max(0.0, (t - a) / (b - a)))
    return u * u * (3.0 - 2.0 * u)


def _u_at(t):
    """0 in the fixed view, 1 in the plan.  Up, hold, down."""
    if t < Q_UP[0]:
        return 0.0
    if t < Q_DOWN[0]:
        return _smooth(t, *Q_UP)
    return 1.0 - _smooth(t, *Q_DOWN)


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
    return (draw_foundation, draw_crypt, draw_choir,
            draw_transept, draw_nave)[stage](f, stage)


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
    LAST["sh"] = buf["sh"]
    return fr


def _grow(buf, part, u, mat, lamp, amb, gain, cam=None, near=1.0, far=0.86,
          pose=None):
    """Draw the fraction u of an assembled element that has been set."""
    P, N, O = part
    m = O <= u
    if not m.any():
        return 0
    col, row, z = (cam or CAM).project((pose or _pose)(P[m]))
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
        col, row, z = cam.project(pose(GHOST[:n]))
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
        col, row, z = cam.project(pose(GHOST[:n]))
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


def draw_transept(f, stage):
    """Part IV.  One shot, and the camera leaves the series for 4.8 seconds.

    The four crossing piers, then both arms, in the view this series has had
    since January.  From there the transept is a wide bit -- more wall, on a
    building already made of wall.  Then straight up, and the reason for it
    is on the ground and has been since part I: the footings are a cross.
    Then back down to the same stubborn view, which still cannot show it.
    """
    t = f / float(FPS)
    u = _u_at(t)
    yaw = -58.0 + (PLAN_YAW + 58.0) * u
    pitch = 28.0 + (PLAN_PITCH - 28.0) * u
    cam = CAM if u <= 0.0 else _mix_cam(u)

    def pose(P):
        return _pose(P) if u <= 0.0 else _pose_at(P, yaw, pitch)

    buf = {"sh": np.zeros((G.rows, G.cols)),
           "mat": np.zeros((G.rows, G.cols), np.int16)}

    gfade = min(1.0, t / Q_GHOST)
    n = int(len(GHOST) * gfade)
    if n > 8:
        col, row, z = cam.project(pose(GHOST[:n]))
        # overhead, the ghost stops being scaffolding and becomes the
        # drawing.  it is the only thing up there that knows the shape.
        sh = ((0.20 + 0.34 * depth_cue(z, 1.0, 0.30))
              * (0.72 + 0.28 * gfade) * (1.0 + 0.62 * u))
        _put(buf, col, row, z + 4000.0, sh, M_GHOST, False)

    # parts I to III.  On the ground both of these sit exactly where part III
    # put them, so the episodes still lay on top of each other.  Overhead the
    # footings come up and the rest does not -- see _FOOT_P.
    for P, N, amb, gain in ((_REST_P, _REST_N, 0.17, 0.44),
                            (_FOOT_P, _FOOT_N, 0.17 + 0.26 * u,
                             0.44 + 0.30 * u)):
        col, row, z = cam.project(pose(P))
        sh = (amb + gain * lambert(N, LAMP)) * depth_cue(z, 1.0, 0.86)
        _put(buf, col, row, z, np.clip(sh, 0.05, 1.0), M_OLD, True)

    def win(w):
        return min(1.0, max(0.0, (t - w[0]) / (w[1] - w[0])))

    LAST["piers4"] = _grow(buf, PIERS4, win(Q_PIER), M_PIER, LAMP,
                           0.26, 0.72, cam, pose=pose)
    LAST["wall4"] = _grow(buf, WALL4, win(Q_WALL), M_TRAN, LAMP,
                          0.26, 0.72, cam, pose=pose)
    LAST["u"] = u

    fr = _paint(buf)
    _label(fr, t, stage)
    return fr


def draw_nave(f, stage):
    """Part V.  One shot, the established view, and twenty piers.

    Nothing is revealed and nothing is cut to.  A row of identical things
    arrives one at a time, from the crossing westward, and stops in mid-air
    where the arches will start.  The building gets an inside.
    """
    t = f / float(FPS)
    close = t >= P_CUT
    cam = CAM_N if close else CAM
    pose = _pose_n if close else _pose
    buf = {"sh": np.zeros((G.rows, G.cols)),
           "mat": np.zeros((G.rows, G.cols), np.int16)}

    gfade = min(1.0, t / P_GHOST)
    n = int(len(GHOST) * gfade)
    if n > 8:
        col, row, z = cam.project(pose(GHOST[:n]))
        lift = 1.0 + 0.55 * min(1.0, max(0.0, (t - P_PIER[1] - 0.3) / 1.1))
        sh = ((0.20 + 0.34 * depth_cue(z, 1.0, 0.30))
              * (0.72 + 0.28 * gfade) * lift)
        _put(buf, col, row, z + 4000.0, sh, M_GHOST, False)

    # parts I to IV, standing, at the level part III set.
    col, row, z = cam.project(pose(_LEG5_P))
    sh = (0.17 + 0.44 * lambert(_LEG5_N, LAMP)) * depth_cue(z, 1.0, 0.86)
    _put(buf, col, row, z, np.clip(sh, 0.05, 1.0), M_OLD, True)

    u = min(1.0, max(0.0, (t - P_PIER[0]) / (P_PIER[1] - P_PIER[0])))
    LAST["nave"] = _grow(buf, PIERS5, u, M_NAVE, LAMP, 0.28, 0.78, cam,
                         pose=pose)
    LAST["u5"] = u
    LAST["close"] = close

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
    LAST["sh"] = buf["sh"]

    _label(fr, t, stage)
    return fr


def blend(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def colour(v, m):
    base = {M_GHOST: GHOST_RGB, M_STONE: STONE, M_EARTH: EARTH,
            M_OLD: OLD, M_CRYPT: CRYPT, M_SLAB: STONE,
            M_CWALL: CW["rgb"], M_WALL3: STONE, M_PART: ROUGH,
            M_TRAN: STONE, M_PIER: STONE, M_NAVE: STONE}[int(m)]
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


def _plan_row(x):
    p = np.array([[x, 0.0, 0.0]], np.float32)
    _, row, _ = CAM_P.project(_pose_at(p, PLAN_YAW, PLAN_PITCH))
    return int(row[0])


def check_transept(stage):
    print("THE CATHEDRAL — part %s, %s" % (roman(stage + 1), STAGES[stage]))
    print("  arms                 2 x %.0f m of wall, reaching z = %+.0f"
          % (ARMS[0][3], TRAN_Z))
    print("  courses              %d of %.2f m, %.1f m to %.1f m"
          % (N_COURSE4, COURSE3, Y_TFOOT, Y_ARCADE))
    print("  wall blocks          %d set, %d slots, %d buttress blocks"
          % (NW4, NSLOT4, NB4))
    print("  crossing piers       4 of %.1f m square, %d courses to %.1f m"
          % (2 * PIER_HW, N_PCOURSE, PIER_TOP))

    # RULE 1.  _pose_at is a generalisation, not a replacement.  If these two
    # ever disagree the established view has drifted and three episodes stop
    # laying on top of each other.
    d = np.abs(_pose_at(GHOST, -58.0, 28.0) - _pose(GHOST)).max()
    print("  established view unchanged: max disagreement %.2e m" % d)
    assert d < 1e-3, d

    # HELD OUT 1 -- the bearing stress under a crossing pier.  Everything
    # above the roof lands on four piers, and nothing in the render knows
    # that.  Take the tower and the spire straight off MASSES as hollow
    # masonry, weigh them, and divide by the four pier tops.  Gothic
    # cathedrals work at roughly 1 N/mm2 (Heyman, The Stone Skeleton) -- a
    # few per cent of what the stone can take.  This is a sizing constraint
    # rather than a blind prediction, and it is stated as one: PIER_HW was
    # chosen to land here.  What it checks is that the model's proportions
    # still do, after the geometry moved.
    TWALL, SWALL, RHO, G0 = 1.6, 0.35, 2300.0, 9.81
    side, h_t = X_CHOIR - X_TRAN, 58.0 - NAVE_Y
    v_tower = (side ** 2 - (side - 2 * TWALL) ** 2) * h_t
    half, y0, y1 = 9.0, 58.0, 86.0
    slant = math.hypot(y1 - y0, half)
    v_spire = 4 * 0.5 * (2 * half) * slant * SWALL
    load = (v_tower + v_spire) * RHO * G0
    area = 4 * (2 * PIER_HW) ** 2
    sigma = load / area / 1e6
    print("  tower %.0f m3 + spire %.0f m3 of masonry = %.0f tonnes"
          % (v_tower, v_spire, (v_tower + v_spire) * RHO / 1000.0))
    print("  on %.1f m2 of pier -> %.2f MPa, about %.0f%% of limestone's "
          "50 MPa" % (area, sigma, 100 * sigma / 50.0))
    assert 0.5 < sigma < 3.0, sigma

    # nothing may leave the picture on ANY frame of the move.  A lerped
    # camera is not guaranteed to frame the poses between its two ends.
    A = np.vstack([GHOST, WALL4[0], PIERS4[0], _LEG4_P])
    worst = (0, 1e9, -1e9, 1e9, -1e9)
    for f in range(int(Q_END * FPS)):
        t = f / float(FPS)
        u = _u_at(t)
        if u <= 0.0:
            continue
        cam = _mix_cam(u)
        col, row, _ = cam.project(_pose_at(A, -58.0 + 148.0 * u,
                                           28.0 + 62.0 * u))
        if (col.min() < worst[1] or col.max() > worst[2]
                or row.min() < worst[3] or row.max() > worst[4]):
            worst = (f, min(worst[1], col.min()), max(worst[2], col.max()),
                     min(worst[3], row.min()), max(worst[4], row.max()))
    print("  over the whole move  c%d..%d  r%d..%d"
          % (worst[1], worst[2], worst[3], worst[4]))
    assert worst[1] >= 0 and worst[2] < G.cols, worst
    assert worst[3] >= 0 and worst[4] < G.rows, worst

    # HELD OUT 2 -- IS IT A CROSS?  Measure it off the finished plan frame:
    # rasterised, z-buffered pixels, not the model.  A cruciform plan has one
    # band of rows much wider than the rest, and that band has to be where
    # the transept is.  The arithmetic route: 2 x 26 m across the arms
    # against 2 x 15 m across the body.
    draw(int(8.7 * FPS), stage)
    mat = LAST["mat"]
    # STONE only, no ghost.  The ghost is the whole finished outline and
    # would draw a perfect cross whatever was built, so measuring the ink
    # would be measuring the drawing.  The claim is about what is ON THE
    # GROUND.  And measure the EXTENT of each row, not how many cells are
    # lit: most of the nave is part I's footings, which are a dotted line
    # down each flank with 6.4 m of nothing between them, so a count says
    # sixteen cells where the building is thirty metres across.
    st = (mat == M_OLD) | (mat == M_TRAN) | (mat == M_PIER)
    span = np.zeros(G.rows)
    for i, r in enumerate(st):
        c = np.nonzero(r)[0]
        if len(c):
            span[i] = c.max() - c.min() + 1
    rows = np.nonzero(span > 0)[0]
    r0, r1 = _plan_row(X_CHOIR), _plan_row(X_TRAN)
    r0, r1 = min(r0, r1), max(r0, r1)
    body = np.median(span[[i for i in rows if not (r0 - 2 <= i <= r1 + 2)]])
    pk = span.max()
    hit = np.nonzero(span >= 0.92 * pk)[0]
    print("  plan: %d rows of stone, body %.0f cells, widest %.0f cells"
          % (len(rows), body, pk))
    print("  measured cross ratio %.2f   from the model %.2f"
          % (pk / body, TRAN_Z / AISLE_Z))
    assert abs(pk / body - TRAN_Z / AISLE_Z) < 0.30, (pk / body)
    print("  widest rows %d..%d, the transept is rows %d..%d"
          % (hit.min(), hit.max(), r0, r1))
    assert hit.min() >= r0 - 3 and hit.max() <= r1 + 3, (hit.min(), hit.max())

    # and the long arm must be BELOW the crossbar or it is not this shape.
    print("  long arm runs to row %d, crossbar centre row %d"
          % (rows.max(), (r0 + r1) // 2))
    assert rows.max() > r1 + 12, (rows.max(), r1)

    # HELD OUT 3 -- the arms have to be full of holes like part III's wall.
    ss = np.arange(0.0, ARMS[0][3], 0.02)
    piers = ARMS[0][4]
    mask = np.array([_tran_window(s, 0.5 * (WIN_SILL + WIN_HEAD), piers)
                     for s in ss])
    runs, cur = [], None
    for i, m in enumerate(mask):
        if m and cur is None:
            cur = i
        elif not m and cur is not None:
            runs.append((ss[cur], ss[i - 1]))
            cur = None
    if cur is not None:
        runs.append((ss[cur], ss[-1]))
    print("  holes measured off one arm: %d in %d bays, mean width %.2f m "
          "(drawn %.2f)" % (len(runs), len(piers) - 1,
                            float(np.mean([b - a for a, b in runs])), WIN_W))
    assert len(runs) == len(piers) - 1, (len(runs), len(piers))
    for (a, b) in runs:
        assert min(abs(np.array(piers) - 0.5 * (a + b))) > 1.4, (a, b)

    sheet, caps = [], []
    for t in (0.6, 1.8, 3.4, 5.4, 6.8, 7.7, 8.7, 10.4, 11.5):
        fr = draw(int(t * FPS), stage)
        ink, mat = LAST["ink"], LAST["mat"]
        print("  t=%4.1f u=%.2f cov %.3f  ghost %5d old %5d wall %5d "
              "pier %4d" % (t, LAST["u"], ink.mean(),
                            (mat == M_GHOST).sum(), (mat == M_OLD).sum(),
                            (mat == M_TRAN).sum(), (mat == M_PIER).sum()))
        assert 0.02 < ink.mean() < 0.60, ink.mean()
        for (c0, r0b, w, h) in LAST["boxes"]:
            assert r0b - 1 >= G.safe_top, ("text above safe", r0b)
            assert r0b + h + 1 <= G.safe_bot, ("text below safe", r0b + h)
            assert c0 - 1 >= 0 and c0 + w + 1 <= G.cols, ("width", c0, w)
        # THE PLAN MUST NOT TOUCH THE LETTERING.
        #
        # Two wrong versions of this check came first, and both of them were
        # wrong the same way: they asserted a quantity instead of naming the
        # defect.  "No ink under the text" fires on the fixed view, where the
        # building has stood behind the numeral since part I.  "Nothing
        # bright under the text" fires there too.  Both are fine, because
        # stamp paints a BG halo round every glyph and gold on a dark outline
        # reads over anything.
        #
        # What actually broke was narrower: overhead, the drawing is FLAT and
        # reads as a diagram, and a numeral sitting inside a diagram becomes
        # part of it -- you cannot tell the IV from a chapel.  In perspective
        # that never happens, because the text is obviously in front.  So the
        # rule is about the plan and it is stated as such.
        if LAST["u"] > 0.9:
            top = min(b[1] for b in LAST["boxes"])
            low = np.nonzero(ink.any(1))[0].max()
            caps.append((t, low, top))
        sheet.append(fr)

    assert caps, "no plan frame was sampled"
    for (t, low, top) in caps:
        print("  plan t=%.1f: drawing ends row %d, lettering starts row %d"
              % (t, low, top))
        assert low < top - 2, (t, low, top)

    print("  runtime              %.1f s, %d frames  (III was %.1f s)"
          % (Q_END, int(Q_END * FPS), H_END))
    contact(sheet, os.path.join(_HERE, "..", "content", "cath_sheet.png"),
            cols=3, labels=["0.6 ghost", "1.8 piers", "3.4 arms", "5.4",
                            "6.8 RISING", "7.7 plan", "8.7 the cross",
                            "10.4 falling", "11.5 back"])


def check_nave(stage):
    print("THE CATHEDRAL — part %s, %s" % (roman(stage + 1), STAGES[stage]))
    print("  nave                 %.0f m in %d bays of %.3f m"
          % (X_TRAN - X_NAVE, N_BAY5, BAY5))
    print("  piers                %d per row, %d courses of %.2f m, "
          "%.1f m to %.2f m" % (N_PIER5, N_PC5, COURSE3, Y_FOOT, Y_CAP5))
    print("  section              %.1f m square = %.2f m2"
          % (2 * PIER5_HW, (2 * PIER5_HW) ** 2))

    # RULE 1.  The one thing this series still publishes, checked every
    # episode: the established view has not drifted.
    d = np.abs(_pose_at(GHOST, -58.0, 28.0) - _pose(GHOST)).max()
    print("  established view unchanged: max disagreement %.2e m" % d)
    assert d < 1e-3, d

    # THE BAY.  11 is not a taste.  The choir and the transept were set out
    # on a 5.8 m module (_arm_piers), the nave is 62 m, and of every whole
    # number of bays 11 is the one that lands nearest that module.  Check it
    # against all the others rather than asserting the winner.
    cand = [(abs((X_TRAN - X_NAVE) / n - 5.8), n) for n in range(6, 17)]
    cand.sort()
    print("  module 5.80 m -> best division of %.0f m is %d bays at %.3f m "
          "(next best %d at %.3f)"
          % (X_TRAN - X_NAVE, cand[0][1], (X_TRAN - X_NAVE) / cand[0][1],
             cand[1][1], (X_TRAN - X_NAVE) / cand[1][1]))
    assert cand[0][1] == N_BAY5, cand[:2]
    assert abs(N_BAY5 * BAY5 - (X_TRAN - X_NAVE)) < 1e-9

    # THE CAPITAL HEIGHT.  A two-centred arch across one bay rises
    # sqrt(3)/2 of its span, and it has to stay under the aisle roof.
    # Fifteen courses is the tallest whole number that does -- so check
    # that sixteen does NOT, or the number is unexplained.
    crown = Y_CAP5 + ARCH_RISE5
    crown16 = Y_FOOT + (N_COURSE5 + 1) * COURSE3 + ARCH_RISE5
    print("  arch crown           %.2f m, aisle roof at %.2f m "
          "(clears by %.2f m)" % (crown, Y_ARCADE, Y_ARCADE - crown))
    print("  one more course      %.2f m -> %s"
          % (crown16, "clears" if crown16 < Y_ARCADE else "does not clear"))
    assert crown < Y_ARCADE, crown
    assert crown16 >= Y_ARCADE, crown16

    # HELD OUT 1 -- the bearing stress at the foot of a nave pier.  Nothing
    # in the render knows what these carry: everything above the capital is
    # still a drawing.  Take it off MASSES -- one bay of nave wall from the
    # capital to the wall head, half the nave vault, and the pier's own
    # weight -- and divide by the section.  Gothic cathedrals work at about
    # 1 N/mm2 (Heyman, The Stone Skeleton).  Like part IV this is a sizing
    # constraint and is stated as one: PIER5_HW was chosen to land here.
    NWALL_T, VAULT_T, RHO, G0 = 1.2, 0.30, 2300.0, 9.81
    v_wall = (NAVE_Y - Y_CAP5) * BAY5 * NWALL_T
    v_vault = NAVE_Z * BAY5 * VAULT_T
    v_self = (Y_CAP5 - Y_FOOT) * (2 * PIER5_HW) ** 2
    load = (v_wall + v_vault + v_self) * RHO * G0
    sigma = load / ((2 * PIER5_HW) ** 2) / 1e6
    print("  carries              %.0f m3 wall + %.0f m3 vault + %.0f m3 "
          "of itself = %.0f tonnes"
          % (v_wall, v_vault, v_self,
             (v_wall + v_vault + v_self) * RHO / 1000.0))
    # Wikipedia's limestone article: "dense limestone can have a crushing
    # strength of up to 180 MPa".  Building limestone is a lot softer than
    # that, so the ratio is quoted against both ends rather than one number
    # I cannot source.
    print("  bearing stress       %.2f MPa = %.2f%% of a dense limestone at "
          "180 MPa, %.1f%% of a soft one at 30"
          % (sigma, 100 * sigma / 180.0, 100 * sigma / 30.0))
    assert 0.4 < sigma < 2.0, sigma

    # The arcade has to STOP before it hits part IV.  The crossing square is
    # 18 m and the nave is 16, so the east end of this arcade is a crossing
    # pier standing a metre outboard of the arcade line.  If the last bay
    # were carried all the way, two pieces of stone would occupy the same
    # place and nobody would see it from here.
    east = X_NAVE + (N_BAY5 - 1) * BAY5
    gap = (X_TRAN - PIER_HW) - (east + PIER5_HW * 1.22)
    print("  last free pier at x=%.2f, crossing pier face at x=%.2f "
          "-> %.2f m clear" % (east, X_TRAN - PIER_HW, gap))
    assert gap > 0.8, gap
    print("  arcade line z=%.0f, crossing pier line z=%.0f -> %.0f m outboard"
          % (NAVE_Z, CROSS_Z, CROSS_Z - NAVE_Z))

    # HELD OUT 2 -- CAN YOU COUNT THEM?  That is the entire episode, so
    # measure it off the finished frame rather than the model: rasterised,
    # z-buffered cells of new stone only.  A row of piers that has merged
    # into a band of wall passes every other check in this file.
    #
    # The row band is taken from the MODEL -- project the pier tops and
    # bottoms -- and not from a fraction of the frame.
    xs5 = [X_NAVE + k * BAY5 for k in range(1, N_BAY5)]

    def _read(cam, pose, t, force_wide=False):
        """How many separate things does a row of piers read as, in the
        frame that is actually on screen?  Two ways of getting this wrong
        have already been paid for.  One: collapsing both rows into a single
        column profile, which fills every gap because the far row sits in
        the near row's gaps at almost every angle.  Two: measuring a frame
        rendered through one camera with the projection of another.  So the
        camera and the pose are passed in together and never assumed.
        """
        global P_CUT
        keep = P_CUT
        if force_wide:
            P_CUT = 1e9                       # render the wide view instead
        try:
            draw(int(t * FPS), stage)
        finally:
            P_CUT = keep
        m = LAST["mat"]
        out = []
        for z in (-NAVE_Z, NAVE_Z):
            top = np.array([[x, Y_CAP5, z] for x in xs5], np.float32)
            bot = top.copy()
            bot[:, 1] = Y_FOOT
            ct, rt, _ = cam.project(pose(top))
            cb, rb, _ = cam.project(pose(bot))
            assert ct.min() >= 0 and ct.max() < G.cols, (ct.min(), ct.max())
            assert rt.min() >= 0 and rb.max() < G.rows, (rt.min(), rb.max())
            r0, r1 = int(rt.min()), int(rb.max())
            c0, c1 = int(ct.min()), int(ct.max())
            lit = (m[r0:r1 + 1, c0:c1 + 1] == M_NAVE).any(0)
            runs, cur = 0, False
            for v in list(lit) + [False]:
                if v and not cur:
                    runs += 1
                cur = v
            out.append((runs, int((~lit).sum()), len(lit), r0, r1, c0, c1))
        return out

    def _show(tag, rows):
        print("  %s" % tag)
        for (runs, air, n, r0, r1, c0, c1) in rows:
            print("    rows %3d..%3d cols %2d..%2d -> %2d separate runs, "
                  "%2d/%2d columns of air" % (r0, r1, c0, c1, runs, air, n))

    wide = _read(CAM, _pose, 7.6, force_wide=True)
    _show("in the ESTABLISHED view -- this is why the episode cuts:", wide)
    got = _read(CAM_N, _pose_n, 7.6)
    _show("in the CLOSE view, which is the one that ships:", got)
    # the established view must FAIL: if it ever stops merging, the cut is
    # unjustified and this episode should not have taken a second camera.
    for (runs, _a, _n, _r0, _r1, _c0, _c1) in wide:
        assert runs <= 3, ("the established view resolves them after all",
                           runs)
    # ten piers a row.  Nine is a merge somewhere and it is not countable.
    for (runs, air, n, _r0, _r1, _c0, _c1) in got:
        assert runs >= N_PIER5, (runs, N_PIER5)
        assert air >= 0.12 * n, (air, n)

    sheet = []
    for t in (0.5, 1.2, 2.2, 3.2, 4.3, 5.4, 6.4, 7.6, 8.6):
        fr = draw(int(t * FPS), stage)
        ink, mat = LAST["ink"], LAST["mat"]
        print("  t=%4.1f u=%.2f cov %.3f  ghost %5d old %5d nave %5d  "
              "set %d" % (t, LAST["u5"], ink.mean(),
                          (mat == M_GHOST).sum(), (mat == M_OLD).sum(),
                          (mat == M_NAVE).sum(), LAST["nave"]))
        assert 0.02 < ink.mean() < 0.60, ink.mean()
        for (bc0, br0, w, h) in LAST["boxes"]:
            assert br0 - 1 >= G.safe_top, ("text above safe", br0)
            assert br0 + h + 1 <= G.safe_bot, ("text below safe", br0 + h)
            assert bc0 - 1 >= 0 and bc0 + w + 1 <= G.cols, ("width", bc0, w)
        sheet.append(fr)

    assert LAST["u5"] >= 1.0, LAST["u5"]
    print("  runtime              %.1f s, %d frames  (IV was %.1f s)"
          % (P_END, int(P_END * FPS), Q_END))
    contact(sheet, os.path.join(_HERE, "..", "content", "cath_sheet.png"),
            cols=3, labels=["0.5 ghost", "1.2", "2.2", "3.2", "4.3", "5.4",
                            "6.4", "7.6 all twenty", "8.6"])


def check(stage):
    if stage == 4:
        return check_nave(stage)
    if stage == 1:
        return check_crypt(stage)
    if stage == 2:
        return check_choir(stage)
    if stage == 3:
        return check_transept(stage)
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
