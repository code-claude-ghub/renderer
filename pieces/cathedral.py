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
M_AISLE = 13                   # part VI: the aisle walls
M_ARCH, M_TRIF, M_TRIFB = 14, 15, 16   # part VII: arches and spandrels /
                               # the screen facing the nave / the back skin
M_CAP8, M_CLER = 17, 18        # part VIII: the passage ceiling / the wall
                               # that is mostly window
M_BUT9, M_FLY9, M_COP9 = 19, 20, 21    # part IX: pier+pinnacle / the arc /
                               # the straight coping -- two load paths, so
                               # two materials the checks can tell apart
M_RIB10, M_WEB10 = 22, 23      # part X: the ribs (the skeleton the thrust
                               # travels along) / the web (the shell that
                               # only has to reach the nearest rib)
INNER = (0.694, 0.633, 0.506)  # stone seen through an opening: the
                               # passage's own shadow, not a new material

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


# ------------------------------------------------------ part VI: the aisles
# The two outer walls of the nave aisles, and nothing else.
#
# What is true of THIS part and no other: it is the episode that hides an
# episode.  Part V stood twenty piers where anyone at the fixed camera could
# try to count them.  This wall goes up between the camera and all of them.
# Every part so far ADDED something to the established frame; this one is the
# first that takes something out of it.  A building gets an inside by taking
# it away from everyone outside -- and from here to the end of the series,
# the arcade is something you would have to walk in to see.  The lancets are
# the concession: eleven slots a side where the outside is still allowed to
# look in.
#
# Everything dimensional is inherited, none of it chosen here:
#   height    Y_FOOT -> Y_ARCADE, like the transept wall (no crypt under it)
#   courses   22 of 0.74 m, same arithmetic as part IV, asserted in check
#   bays      the 11 x 5.636 m grid part V froze; a lancet mid-bay, a
#             buttress on each interior bay line
#   sill/head WIN_SILL and WIN_HEAD, unchanged since part III: a string
#             course runs round a building at one level and does not step
# The bay-line buttresses at the two ENDS are left out on purpose: x = 0 is
# engaged in the west front (part XII), and the corner at x = 62 already has
# part IV's pier on it -- _arm_piers put one on every corner of the arms.
N_COURSE6 = int(round((Y_ARCADE - Y_FOOT) / COURSE3))           # 22

# Walked so _on_path's outward normal points OUT of the building: north wall
# west to east, south wall east to west, same fix as the transept arms.
AISLE_PATHS = ([(X_NAVE, -AISLE_Z), (X_TRAN, -AISLE_Z)],
               [(X_TRAN, AISLE_Z), (X_NAVE, AISLE_Z)])


def _lancet_w(y):
    """Lancet width at height y: parallel sides, pointed head.  Part III's
    profile on part V's grid."""
    if not (WIN_SILL <= y <= WIN_HEAD):
        return 0.0
    tt = (y - WIN_SILL) / (WIN_HEAD - WIN_SILL)
    return WIN_W * (1.0 if tt < 0.62 else max(0.0, 1.0 - (tt - 0.62) / 0.38))


def _clip_slot(s, hx, y):
    """A slot stone that meets a lancet gets CUT AT THE JAMB, not skipped.

    The choir and transept walls skip any slot whose centre is inside the
    window, which leaves stones poking up to a metre into the opening from
    both sides -- the 2.4 m lattice does not care where the jambs are.  On
    those walls nobody measured it.  Here the lancet count is a held-out
    check, and the worst window narrowed to under a metre of true opening
    and read closed at the shipping yaw.  Masons dress a jamb straight;
    this returns the pieces of [s-hx, s+hx] left after the windows are cut,
    which is the same thing."""
    w = _lancet_w(y)
    pieces, a, b = [], s - hx, s + hx
    if w <= 0.0:
        return [(s, hx)]
    lo = a
    j0, j1 = int(a // BAY5), int(b // BAY5) + 1
    for j in range(j0, j1 + 1):
        wc = (j + 0.5) * BAY5
        wa, wb = wc - 0.5 * w, wc + 0.5 * w
        if wb <= lo or wa >= b:
            continue
        if wa > lo:
            pieces.append((0.5 * (lo + wa), 0.5 * (wa - lo)))
        lo = max(lo, wb)
    if lo < b:
        pieces.append((0.5 * (lo + b), 0.5 * (b - lo)))
    return [(c, h) for (c, h) in pieces if h > 0.12]


def aisle_wall(spacing=2.4):
    """Both walls, course by course, north and south rising together."""
    hh = COURSE3 / 2.0
    paths = []
    for path in AISLE_PATHS:
        P = _dedupe(path)
        S = np.linalg.norm(np.diff(P, axis=0), axis=1)
        C = np.concatenate([[0.0], np.cumsum(S)])
        paths.append((P, S, C, float(C[-1])))
    units, nw, nb, nslot = [], 0, 0, 0
    for k in range(N_COURSE6):
        y = Y_FOOT + (2 * k + 1) * hh
        proj = BUT_PROJ * (1.0 - 0.45 * k / float(N_COURSE6 - 1))
        for (P, S, C, tot) in paths:
            for j in range(1, N_BAY5):           # interior bay lines only
                x, z, ang, ox, oz = _on_path(P, S, C, j * BAY5)
                d = 0.95 + 0.5 * proj
                units.append(stone(x + ox * d, y, z + oz * d,
                                   1.15, hh * 0.86, 0.5 * proj, ang))
                nb += 1
            n = max(1, int(round(tot / spacing)))
            for i in range(n):
                s = min(((i + 0.5 + 0.5 * (k % 2)) % n) * tot / n,
                        tot - 1e-6)
                nslot += 1
                th = 0.95 + (0.30 if abs(y - WIN_SILL) < 0.5 * COURSE3
                             else 0.0)
                e = 0.16 if abs(y - WIN_SILL) < 0.5 * COURSE3 else 0.0
                for (sc, sh_) in _clip_slot(s, spacing * 0.45, y):
                    x, z, ang, ox, oz = _on_path(P, S, C,
                                                 min(max(sc, 0.0),
                                                     tot - 1e-6))
                    units.append(stone(x + ox * e, y, z + oz * e,
                                       sh_, hh * 0.86, th, ang))
                    nw += 1
    return assemble(units), nw, nb, nslot


# --------------------------------------------- part VII: the triforium
# The middle storey of the nave elevation: the arcade arches turned across
# part V's piers, the spandrels levelled off, and above them the storey
# that faces the wrong way.  A triforium is a passage INSIDE the wall's
# thickness -- arcaded toward the nave, blank toward the sky -- and it is
# not decoration: the wall up here carries nothing but itself and the
# clerestory to come, so the masons hollow it and save the piers the
# weight.  The wall was 2.4 m thick when it was solid pier.  It is still
# 2.4 m from face to face.  Most of it is now air.
#
# NOTHING dimensional is chosen here.  The screen facing the nave is the
# arcade below it built again at EXACTLY quarter scale: four openings to
# the bay (span BAY5/4), the same two-centred arch (rise/span = sqrt(3)/2
# whatever the span, so the small arch IS the big arch scaled), colonnette
# height (Y_CAP5 - Y_FOOT)/4.  All three ratios are 4 to the last bit, by
# construction, and every fourth colonnette stands on a pier centreline
# exactly, because 4 divides the bay grid it inherited.
#
# The wall's cross-section is the pier's width spent three ways:
#     0.70 m back skin + 1.35 m passage + 0.35 m screen  =  2.40 m
# The passage width is not chosen either -- it is what is LEFT of the
# pier width after the two skins.  It has no ceiling this episode: the
# clerestory sill caps it in part VIII.
RING5 = 0.29                       # arcade voussoir depth.  0.29 and not
                                   # 0.30, because the check said so: at
                                   # 0.30 the extrados crown pokes 1.2 mm
                                   # above course 22.  asserted.
RING_T = 0.10                      # screen voussoir depth
Y_SPAN_TOP = Y_FOOT + N_COURSE6 * COURSE3   # 18.98 -- course 22.  the
                                   # spandrels level off where the aisle
                                   # walls topped out: the whole building
                                   # reaches course 22 together.
SPAN_T = BAY5 / 4.0                # 1.409 -- exactly a quarter bay
RISE_T = 0.5 * math.sqrt(3.0) * SPAN_T      # exactly ARCH_RISE5 / 4
SHAFT_T = (Y_CAP5 - Y_FOOT) / 4.0  # 2.775 -- exactly the pier height / 4
Y_SILL7 = Y_SPAN_TOP + COURSE3     # 19.72 -- course 23, the passage floor
Y_SHAFT_TOP = Y_SILL7 + SHAFT_T    # 22.495 -- where the small arches spring
CROWN_T = Y_SHAFT_TOP + RISE_T     # 23.715
K_TOP7 = 29                        # least course count clearing the small
                                   # crowns; 28 does not.  asserted.
Y_TOP7 = Y_FOOT + K_TOP7 * COURSE3          # 24.16
SKIN_TH, SCREEN_TH = 0.70, 0.35
PASSAGE7 = 2.0 * PIER5_HW - SKIN_TH - SCREEN_TH     # 1.35, by remainder
Z_SKIN = NAVE_Z + PIER5_HW - 0.5 * SKIN_TH          # 8.85
Z_SCREEN = NAVE_Z - PIER5_HW + 0.5 * SCREEN_TH      # 6.975
X_A7, X_B7 = BAY5, X_TRAN          # the built run.  bay 0 has no west
                                   # support until part XII; raw end.
N_OPEN7 = 4 * (N_BAY5 - 1)         # 40 screen openings per row


def _intr(x, xa, xb):
    """Intrados height of a two-centred arch over [xa, xb], above its
    springing.  Two arcs of radius = span, each centred on the opposite
    springing point, meeting at rise sqrt(3)/2 * span."""
    s = xb - xa
    x = min(max(x, xa), xb)
    d = (xb - x) if x <= 0.5 * (xa + xb) else (x - xa)
    return math.sqrt(max(0.0, s * s - d * d))


def _arch7(xa, xb, y0, zc, hz, hx, hy, n):
    """One two-centred arch: voussoirs from both springings inward,
    alternating sides, keystone last -- same order _rib used in the crypt,
    because that is the order an arch can be built in at all."""
    s = xb - xa
    units = []
    for i in range(n):
        th = (i + 0.5) / n * (math.pi / 3.0)
        for left in (True, False):
            x = xb - s * math.cos(th) if left else xa + s * math.cos(th)
            units.append(stone(x, y0 + s * math.sin(th), zc, hx, hy, hz))
    units.append(stone(0.5 * (xa + xb), y0 + 0.5 * math.sqrt(3.0) * s,
                       zc, hx, hy, hz))
    return units


def arcade_arches():
    """Ten arches per row, east to west, both rows rising together.  The
    east arch lands on part IV's crossing pier, which went up two episodes
    ago without being told it was a springing."""
    units = []
    for k in range(N_BAY5 - 1, 0, -1):
        xa, xb = k * BAY5, (k + 1) * BAY5
        for zc in (-NAVE_Z, NAVE_Z):
            units += _arch7(xa, xb, Y_CAP5, zc, PIER5_HW, 0.30, 0.16, 10)
    return assemble(units), 2 * (N_BAY5 - 1)


def spandrel7():
    """The fill above the arches up to course 22, cut AT the extrados --
    part VI's jamb lesson, applied to a curve: a stone that meets the arch
    is dressed to it, not skipped."""
    units = []
    n = int(round((X_B7 - X_A7) / 0.9))
    w = (X_B7 - X_A7) / n
    for i in range(n):                          # east to west
        x = X_B7 - (i + 0.5) * w
        k = min(N_BAY5 - 1, max(1, int(x // BAY5)))
        dy = _intr(x, k * BAY5, (k + 1) * BAY5)
        y0 = Y_CAP5 + (dy + RING5 if dy > 0.05 else 0.0)
        if Y_SPAN_TOP - y0 < 0.08:
            continue
        for zc in (-NAVE_Z, NAVE_Z):
            units.append(stone(x, 0.5 * (y0 + Y_SPAN_TOP), zc,
                               0.47 * w, 0.5 * (Y_SPAN_TOP - y0), PIER5_HW))
    return assemble(units), n


def sill7():
    """Course 23, full thickness: the passage floor."""
    units = []
    n = int(round((X_B7 - X_A7) / 1.9))
    w = (X_B7 - X_A7) / n
    for i in range(n):                          # east to west
        x = X_B7 - (i + 0.5) * w
        for zc in (-NAVE_Z, NAVE_Z):
            units.append(stone(x, Y_SPAN_TOP + 0.5 * COURSE3, zc,
                               0.47 * w, COURSE3 * 0.43, PIER5_HW))
    return assemble(units), n


def skin7():
    """The back skin: courses 24 to 29, 0.70 m thick, outboard.  This is
    the only part of the storey the fixed view will ever read."""
    units = []
    nc = K_TOP7 - 23                            # 6 courses
    n = int(round((X_B7 - X_A7) / 2.0))
    w = (X_B7 - X_A7) / n
    for c in range(nc):
        y = Y_SILL7 + (c + 0.5) * COURSE3
        for i in range(n):                      # east to west, bonded
            x = X_B7 - ((i + 0.5 + 0.5 * (c % 2)) % n) * w
            for o in (-1.0, 1.0):
                units.append(stone(x, y, o * Z_SKIN,
                                   0.47 * w, COURSE3 * 0.43, 0.5 * SKIN_TH))
    return assemble(units), nc


def colonnettes():
    """41 per row: base, monolithic shaft, cap.  A colonnette is
    turned, not coursed -- one stone tall.  Every fourth stands over a
    pier centreline."""
    units = []
    b = 0.20
    for m in range(N_OPEN7, -1, -1):            # east to west
        x = X_A7 + m * SPAN_T
        for o in (-1.0, 1.0):
            zc = o * Z_SCREEN
            units.append(stone(x, Y_SILL7 + 0.5 * b, zc,
                               0.24, 0.5 * b, 0.5 * SCREEN_TH))
            units.append(stone(x, 0.5 * (Y_SILL7 + Y_SHAFT_TOP), zc,
                               0.16, 0.5 * (SHAFT_T - 2 * b), 0.16))
            units.append(stone(x, Y_SHAFT_TOP - 0.5 * b, zc,
                               0.24, 0.5 * b, 0.5 * SCREEN_TH))
    return assemble(units), N_OPEN7 + 1


def screen_arches():
    """80 small arches, quarter-scale twins of the arcade below, keystone
    last like their parents."""
    units = []
    for m in range(N_OPEN7 - 1, -1, -1):        # east to west
        xa = X_A7 + m * SPAN_T
        for o in (-1.0, 1.0):
            units += _arch7(xa, xa + SPAN_T, Y_SHAFT_TOP, o * Z_SCREEN,
                            0.5 * SCREEN_TH, 0.10, 0.075, 4)
    return assemble(units), 2 * N_OPEN7


def screen_fill():
    """The screen's own spandrels, up to the band top -- cut at the small
    extrados exactly as the big spandrels were cut at the big one."""
    units = []
    n = int(round((X_B7 - X_A7) / 0.55))
    w = (X_B7 - X_A7) / n
    for i in range(n):                          # east to west
        x = X_B7 - (i + 0.5) * w
        m = min(N_OPEN7 - 1, max(0, int((x - X_A7) // SPAN_T)))
        dy = _intr(x, X_A7 + m * SPAN_T, X_A7 + (m + 1) * SPAN_T)
        y0 = Y_SHAFT_TOP + (dy + RING_T if dy > 0.02 else 0.0)
        if Y_TOP7 - y0 < 0.08:
            continue
        for o in (-1.0, 1.0):
            units.append(stone(x, 0.5 * (y0 + Y_TOP7), o * Z_SCREEN,
                               0.47 * w, 0.5 * (Y_TOP7 - y0),
                               0.5 * SCREEN_TH))
    return assemble(units), n


# --------------------------------------------- part VIII: the clerestory
# The top storey of the nave wall: the one whose entire job is windows.
# "Clerestory" is "clear storey" -- the storey that clears the aisle roofs
# and lights the nave from above them.  Part VII's description promised
# that its open passage gets a ceiling here, and the ceiling is the first
# stone laid: course 30, full pier thickness, is at once the triforium's
# lid and the clerestory's sill.
#
# NOTHING dimensional is chosen here either.
#   - The window is the arcade arch AT EXACTLY HALF SCALE: span BAY5/2,
#     the same two-centred rise.  With part VII's screen at quarter scale
#     the finished elevation is ONE arch at three sizes -- 1, 1/4, 1/2 --
#     and every ratio is a power of two, exact in floats.
#   - The jambs sit on the quarter-bay lines, which ARE the triforium's
#     colonnette lines.  The storeys share a grid without being told to.
#   - The wall is 1.2 m thick because three older numbers agree: it is
#     the pier's half-width; it is the 1.2 m of overhead wall part V's
#     stress check assumed when it sized the piers (NWALL_T); and it puts
#     the outer face EXACTLY on the frozen mass face |z| = NAVE_Z while
#     the inner face continues the screen's own line upward.
#   - The wall tops out at course 45, and Y_FOOT + 45 * COURSE3 is 36.0
#     -- NAVE_Y, frozen in MASSES since part I -- with a float difference
#     of literally zero.  The crypt's course height and the first video's
#     silhouette agree, and nobody planned that in part II.
SPAN_8 = BAY5 / 2.0                # window clear span: exactly half a bay
RISE_8 = 0.5 * math.sqrt(3.0) * SPAN_8      # exactly ARCH_RISE5 / 2
RING8 = RING5 / 2.0                # 0.145 -- the ring scales with its arch
K_CAP8 = 30                        # the cap course: the passage's ceiling
Y_CAP8 = Y_FOOT + K_CAP8 * COURSE3          # 24.90 -- also the window sill
K_SPRING8 = 41                     # the GREATEST course whose arch crown
                                   # still clears the wall top; 42 overtops
                                   # by 0.37 m.  asserted both ways.
Y_SPRING8 = Y_FOOT + K_SPRING8 * COURSE3    # 33.04
K_TOP8 = 45                        # NAVE_Y, on the crypt's grid, exactly
Y_TOP8 = NAVE_Y                    # 36.0 -- the line part I drew
WALL8_TH = PIER5_HW                # 1.2 -- see the three agreements above
Z_WALL8 = NAVE_Z - PIER5_HW + 0.5 * WALL8_TH        # 7.4; outer face 8.0
N_WIN8 = N_BAY5 - 1                # ten windows a side, bays 1..10


def _jambs8(k):
    """Window k lives in bay k, between the quarter-bay lines."""
    return k * BAY5 + 0.25 * BAY5, k * BAY5 + 0.75 * BAY5


def cap8():
    """Course 30, full pier thickness, the whole run, both rows: the
    passage stops being open to the sky.  East to west, like everything
    in this series, because the choir end is the end in use."""
    units = []
    n = int(round((X_B7 - X_A7) / 1.9))
    w = (X_B7 - X_A7) / n
    for i in range(n):
        x = X_B7 - (i + 0.5) * w
        for zc in (-NAVE_Z, NAVE_Z):
            units.append(stone(x, Y_TOP7 + 0.5 * COURSE3, zc,
                               0.47 * w, COURSE3 * 0.43, PIER5_HW))
    return assemble(units), n


def strips8():
    """The solid wall between the windows: a half-bay strip centred on
    every pier line, courses 31 to 45, stones dressed AT the jamb --
    part VI's lesson, built in from the start this time.  The two raw
    ends get their half strips."""
    hh = COURSE3 / 2.0
    strips = [(X_A7, X_A7 + 0.25 * BAY5)]
    for m in range(2, N_BAY5):
        strips.append((m * BAY5 - 0.25 * BAY5, m * BAY5 + 0.25 * BAY5))
    strips.append((X_B7 - 0.25 * BAY5, X_B7))
    units = []
    for c in range(K_CAP8, K_TOP8):             # courses 31..45
        y = Y_FOOT + (2 * c + 1) * hh
        for (a, b) in reversed(strips):         # east to west
            n = max(1, int(round((b - a) / 1.35)))
            w = (b - a) / n
            for i in range(n + 1):
                x0 = a + (i - 0.5 * ((c - K_CAP8) % 2)) * w
                x1 = min(b, x0 + w)
                x0 = max(a, x0)
                if x1 - x0 < 0.10:
                    continue
                for o in (-1.0, 1.0):
                    units.append(stone(0.5 * (x0 + x1), y, o * Z_WALL8,
                                       0.47 * (x1 - x0), hh * 0.86,
                                       0.5 * WALL8_TH))
    return assemble(units), K_TOP8 - K_CAP8, len(strips)


def win_arches8():
    """Twenty window heads: the arcade arch at half scale, voussoirs from
    both springings inward, keystone last, like both of its parents."""
    units = []
    for k in range(N_WIN8, 0, -1):              # east to west
        xa, xb = _jambs8(k)
        for o in (-1.0, 1.0):
            units += _arch7(xa, xb, Y_SPRING8, o * Z_WALL8,
                            0.5 * WALL8_TH, 0.21, 0.08, 7)
    return assemble(units), 2 * N_WIN8


def spandrel8():
    """The fill over each window head up to the wall top, cut at the
    extrados -- spandrel7's pattern at half scale.  The last stones of
    the last full-height wall this building will ever need."""
    units = []
    for k in range(N_WIN8, 0, -1):              # east to west
        xa, xb = _jambs8(k)
        n = int(round((xb - xa) / 0.52))
        w = (xb - xa) / n
        for i in range(n):
            x = xb - (i + 0.5) * w
            y0 = Y_SPRING8 + _intr(x, xa, xb) + RING8
            if Y_TOP8 - y0 < 0.08:
                continue
            for o in (-1.0, 1.0):
                units.append(stone(x, 0.5 * (y0 + Y_TOP8), o * Z_WALL8,
                                   0.47 * w, 0.5 * (Y_TOP8 - y0),
                                   0.5 * WALL8_TH))
    return assemble(units), N_WIN8


# --------------------------------------------- part IX: the buttresses
# The answer to part VIII's question: a wall that is 44% hole does not
# hold a vault's push.  It hires help that leaps the aisle.  Ten outer
# piers a side rise from part VI's bay-line buttresses -- which went up
# three episodes ago without being told what for, like the crossing pier
# before them -- and a flyer springs from each one to the clerestory
# wall.  Part V's comment, the day the bay was frozen: "the buttresses
# and the roof trusses all have to land on these lines."  They land on
# these lines.
#
# NOTHING dimensional is chosen here either.
#   - The leap R9 is the pier's inner face to the wall's outer face:
#     AISLE_Z - PIER5_HW - NAVE_Z = 5.8 m.  All three terms are frozen.
#   - The flyer head arrives at the LEAST course that reaches over the
#     clerestory springing line (course 41's, part VIII's) -- course 34;
#     course 33 falls 0.12 m short.  Asserted both ways.
#   - The chord from tail to head is 45 degrees EXACTLY: rise = run = R9
#     by construction, because the head course was chosen on the same
#     grid the tail springs from.
#   - The intrados is a 60-degree arc, and 60 degrees is not a style: it
#     is the ONLY sweep for which the radius equals the chord -- the
#     equilateral fact this series has built every arch out of since the
#     crypt.  R_seg = sqrt(2) * R9.  Tangents fall at 75 and 15 degrees.
#   - The flyer is as wide as the wall it props is thick: PIER5_HW.
#     The pier below it keeps part VI's buttress width (2.3 m) and the
#     arcade pier's depth (2.4 m): the arcade pier, moved outside.
#   - The pinnacle is the series' own triangle made solid: 60-degree
#     faces, height PIER5_HW * sqrt(3).
#
# THE SHAPE WAS CORRECTED BY ITS OWN ARITHMETIC, TWICE.  The first
# draft was the open flyer the books draw -- arc band, coping, nothing
# between -- at the pier's own 2.3 m width.  The thrust-line walk in
# check_buttress refused it: below ~26 t of thrust there is NO path
# through that shape (the line falls through the open spandrel), and a
# flyer that cannot stand until a vault pushes on it is a flyer that
# falls the day the centering drops.  So (1) the spandrel got a web,
# masonry's own fix, and (2) the width was SIZED IN CHECK, like the
# arcade piers were: at 2.3 m the flyer leans ~26 t on a wall that can
# lend ~6; at half the wall's thickness it stands on a few.  The check
# asserts both halves of that sentence.
RHO9 = RING5 / BAY5                # voussoir depth per span, the series'
                                   # own ratio (0.0514, set in part VII)
R9 = AISLE_Z - PIER5_HW - NAVE_Z   # 5.8 -- the leap, all terms frozen
D9 = RHO9 * (2.0 * R9)             # 0.597 -- ring depth for that span
K_SPR9 = 34                        # tail springing course: LEAST whose
                                   # head reaches over Y_SPRING8
Y_SPR9 = Y_FOOT + K_SPR9 * COURSE3          # 27.86
Y_HEAD9 = Y_SPR9 + R9              # 33.66 -- 45-degree chord, rise = run
K_PIER9 = 35                       # pier top: LEAST course over the
                                   # coping's tail arrival.  34 is under.
Y_TOP9 = Y_FOOT + K_PIER9 * COURSE3         # 28.60
Y_BASE9 = Y_FOOT + N_COURSE6 * COURSE3      # 18.98 -- part VI's wall top
Z_PIER9 = AISLE_Z                  # 15.0 -- the aisle wall line
PIER9_HX, PIER9_HZ = 1.15, PIER5_HW         # 2.3 x 2.4: the arcade pier,
                                   # moved outside the building
W9X = 0.5 * PIER5_HW               # 0.6 -- flyer width, SIZED IN CHECK:
                                   # the least width whose standing
                                   # thrust the wall can carry.  see the
                                   # header comment and check_buttress.
R_SEG9 = math.sqrt(2.0) * R9       # 8.202 -- radius = chord: 60 degrees
PIN9_H = PIER5_HW * math.sqrt(3.0)          # 2.078 -- 60-degree faces
N_BUT9 = N_BAY5 - 1                # ten a side, bay lines 1..10


def _arc9(o):
    """Centre and end angles of the intrados arc, side o = +-1.
    Chord runs tail (13.8, Y_SPR9) -> head (8.0, Y_HEAD9); the centre
    sits R_seg*cos(30) off the chord midpoint, away from the bulge."""
    zt, zh = Z_PIER9 - PIER9_HZ, NAVE_Z
    mz, my = 0.5 * (zt + zh), 0.5 * (Y_SPR9 + Y_HEAD9)
    d = R_SEG9 * math.cos(math.pi / 6.0)
    n = 1.0 / math.sqrt(2.0)
    cz, cy = mz + d * n, my + d * n
    at = math.atan2(Y_SPR9 - cy, zt - cz)
    ah = math.atan2(Y_HEAD9 - cy, zh - cz)
    return cz, cy, at, ah


def piers9():
    """Ten outer piers a side, courses 23..35, standing on part VI's
    bay-line buttresses.  East to west, both sides rising together."""
    units = []
    for c in range(N_COURSE6, K_PIER9):
        y = Y_FOOT + (c + 0.5) * COURSE3
        for m in range(N_BUT9, 0, -1):
            x = m * BAY5
            for o in (-1.0, 1.0):
                j = RNG.uniform(-0.02, 0.02, 2)
                units.append(stone(x + j[0], y, o * Z_PIER9 + j[1],
                                   PIER9_HX, COURSE3 * 0.43, PIER9_HZ))
    return assemble(units), N_BUT9, K_PIER9 - N_COURSE6


def flyers9():
    """All twenty arcs rise together, voussoir by voussoir from the
    tail up -- the way the whole run has risen since the aisle walls.
    The head stones, the ones that touch the wall, go in last, in one
    moment, on every bay line at once.  Then the spandrel webs.  The
    web is not decoration: without it the thrust line has no path
    (see the header comment)."""
    units = []
    n = int(round((R_SEG9 * math.pi / 3.0) / 0.62))
    cz, cy, at, ah = _arc9(1.0)
    r_in = R_SEG9 - D9
    for i in range(n):                  # all twenty arcs rise together
        a = at + (ah - at) * (i + 0.5) / n
        r = R_SEG9 - 0.5 * D9
        z, y = cz + r * math.cos(a), cy + r * math.sin(a)
        for m in range(N_BUT9, 0, -1):
            for o in (-1.0, 1.0):
                units.append(stone(m * BAY5, y, o * z,
                                   0.48 * W9X, 0.52 * D9, 0.52 * D9))
    nw = 9
    for i in range(nw):                 # then the webs, tail to head
        z = NAVE_Z + 0.35 + (nw - 1 - i + 0.5) / nw * (
            Z_PIER9 - PIER9_HZ - NAVE_Z - 0.7)
        dz2 = (z - cz) ** 2
        yb = (cy - math.sqrt(r_in * r_in - dz2)
              if r_in * r_in > dz2 else Y_HEAD9 - (z - NAVE_Z))
        yt = Y_HEAD9 - (z - NAVE_Z)
        if yt - yb < 0.12:
            continue
        for m in range(N_BUT9, 0, -1):
            for o in (-1.0, 1.0):
                units.append(stone(m * BAY5, 0.5 * (yb + yt), o * z,
                                   0.44 * W9X, 0.5 * (yt - yb), 0.26))
    return assemble(units), n


def copings9():
    """The straight band on the 45-degree chord: the strut part X's
    thrust will actually travel down.  Vertical thickness D9."""
    units = []
    zt, zh = Z_PIER9 - PIER9_HZ, NAVE_Z
    n = int(round((zt - zh) * math.sqrt(2.0) / 0.62))
    for i in range(n):                  # tail to head, together; the
        u = 1.0 - (i + 0.5) / n         # last stone touches the wall
        z = zh + (zt - zh) * u
        y = Y_HEAD9 - R9 * u + 0.5 * D9
        for m in range(N_BUT9, 0, -1):
            for o in (-1.0, 1.0):
                units.append(stone(m * BAY5, y, o * z,
                                   0.48 * W9X, 0.55 * D9, 0.36))
    return assemble(units), n


def pinnacles9():
    """Ballast last: the pinnacle goes on after the flyer exists to
    need it.  Five shrinking tiers to a point."""
    units = []
    nt = 5
    for j in range(nt):                 # all twenty tips rise together
        f = (j + 0.5) / nt
        y = Y_TOP9 + f * PIN9_H
        for m in range(N_BUT9, 0, -1):
            for o in (-1.0, 1.0):
                units.append(stone(m * BAY5, y, o * Z_PIER9,
                                   max(0.10, PIER9_HX * (1.0 - f)),
                                   0.45 * PIN9_H / nt,
                                   max(0.10, PIER9_HZ * (1.0 - f))))
    return assemble(units), nt


# --------------------------------------------- part X: the high vault
# The episode the last one wrote a cheque for.  Part IX's statics
# published a budget -- the vault must arrive pushing less than
# H_MAX t a bay -- and this is the vault arriving.  Quadripartite rib
# vaults over the ten clerestory bays: a transverse arch on every bay
# line, two diagonal ribs crossing at a boss, a wall rib against each
# clerestory wall, and the thin stone web spanning between them.
#
# NOTHING dimensional is chosen here either -- and this time the two
# oldest decisions in the series do the sizing between them:
#   - The span is the clerestory's clear width, 2 * (NAVE_Z - WALL8_TH)
#     = 13.6 m; the springing is course 41, the line part VIII NAMED
#     the springing line.  Both frozen.
#   - The series' own equilateral arch, on that span, is REFUSED BY
#     THE ROOF: part I's frozen roof plane (eaves 36, ridge 46) cuts
#     its intrados 0.31 m deep at z = 3.8.  The first time in ten
#     episodes the equilateral loses.
#   - The mason's answer, which is THE historical answer: the diagonal
#     of the bay gets a SEMICIRCLE -- the only round arch this building
#     will ever hold -- radius RHO10 = hypot(S10, BAY5/2).  Every other
#     rib must rise from a shorter span to the semicircle's crown, and
#     an arch forced higher than its span wants is a POINTED arch.
#     The centre offsets have closed forms, and they are symmetric:
#     transverse q_t = sb^2 / 2s, wall rib q_w = s^2 / 2sb (a lancet:
#     its centres lie outside its own span).  The pointed arch is not
#     a style here.  It is what a round arch forces on its neighbours.
#   - Ring depths use the series' ratio RHO9 = RING5/BAY5, and the wall
#     rib's comes out at RHO9 * BAY5 = RING5 exactly -- the arcade's own
#     voussoir depth, because its span IS the bay.  The web is one such
#     voussoir thick.
#   - Every ridge is LEVEL at Y_SPR10 + RHO10: each rib was built to
#     reach the same crown, so the crowns agree by construction.
#
# THE BUDGET IS THE EPISODE.  check_vault walks one thrust line from
# the keystone through the springing, across the wall, into part IX's
# flyer head and down the pier to the ground -- the vault's own walk
# spliced onto last episode's, one H threading both.  The vault's
# feasible band and the buttress system's window overlap on a few
# tonnes, and the low end of the vault's band -- the thrust a masonry
# arch actually settles to as its abutments give -- lands under the
# published number.  The check asserts it.
S10 = NAVE_Z - WALL8_TH            # 6.8 -- half the clear span, the
                                   # clerestory's inner faces.  frozen.
SB10 = 0.5 * BAY5                  # 2.818 -- half the bay
RHO10 = math.hypot(S10, SB10)      # 7.361 -- half the bay diagonal:
                                   # the semicircle's radius, and the
                                   # rise of every rib in the vault
Y_SPR10 = Y_SPRING8                # 33.04 -- part VIII's springing
                                   # line, doing what its name promised
Y_CROWN10 = Y_SPR10 + RHO10        # 40.40 -- every ridge, level
QT10 = SB10 * SB10 / (2.0 * S10)   # 0.584 -- transverse centre offset
RT10 = S10 + QT10                  # 7.384 -- transverse radius
QW10 = S10 * S10 / (2.0 * SB10)    # 8.204 -- wall-rib offset: centres
                                   # OUTSIDE the span.  a lancet.
RW10 = SB10 + QW10                 # 11.022 -- wall-rib radius
D10T = RHO9 * 2.0 * S10            # 0.700 -- transverse ring depth
D10D = RHO9 * 2.0 * RHO10          # 0.757 -- diagonal ring depth
D10W = RHO9 * 2.0 * SB10           # 0.290 == RING5: the arcade
                                   # voussoir, back at its own scale
TW10 = D10W                        # the web: one wall-rib voussoir thick
N_VBAY10 = N_BAY5 - 1              # ten vaulted bays, 1..10


def _y10(xi, z):
    """Web intrados over bay-local (xi in [0, BAY5], z in [-S10, S10]).
    Four cells cut by the plan diagonals a = b.  In each cell the
    course from the diagonal seam to the level ridge is its own
    bounding rib's profile, scaled -- the series' one arch, at every
    size from full to nothing."""
    a = min(1.0, abs(z) / S10)
    b = min(1.0, abs(2.0 * xi / BAY5 - 1.0))
    t = max(a, b)
    d = RHO10 * math.sqrt(max(0.0, 1.0 - t * t))
    if t < 1e-9:
        return Y_SPR10 + RHO10
    if a >= b:                      # wall cell: the lancet, scaled
        pr = math.sqrt(max(0.0, RW10 * RW10
                           - ((b / a) * SB10 + QW10) ** 2)) / RHO10
    else:                           # arch cell: the pointed arch, scaled
        pr = math.sqrt(max(0.0, RT10 * RT10
                           - ((a / b) * S10 + QT10) ** 2)) / RHO10
    return Y_SPR10 + d + (RHO10 - d) * pr


def tarches10():
    """Eleven transverse arches, lines 1..11 (line 11 leans on the
    transept, which was built four episodes before it was needed).
    Voussoir angle outer, lines inner: all eleven rise together and
    the crown pairs meet last -- part IX's stall lesson, kept."""
    units = []
    phimax = math.atan2(RHO10, QT10)
    n = int(round(RT10 * phimax / 0.62))
    r_mid = RT10 - 0.5 * D10T
    for i in range(n):
        phi = (i + 0.5) / n * phimax
        for m in range(N_VBAY10 + 1, 0, -1):        # east to west
            for o in (-1.0, 1.0):
                z = o * (r_mid * math.cos(phi) - QT10)
                y = Y_SPR10 + r_mid * math.sin(phi)
                units.append(stone(m * BAY5, y, z,
                                   0.48 * W9X, 0.52 * D10T, 0.52 * D10T))
    return assemble(units), n


def diags10():
    """Twenty diagonal ribs -- the only round arches in the building.
    Each stops one voussoir short of its crossing; the boss closes
    both at once.  Level outer (both springings rise toward the
    middle), bays inner."""
    units = []
    n = int(round(math.pi * RHO10 / 0.66))
    if n % 2 == 0:
        n += 1                      # odd: a single crown slot, kept open
    A = math.atan2(2.0 * S10, BAY5)
    r_mid = RHO10 - 0.5 * D10D
    for l in range(n // 2):
        for m in range(N_VBAY10, 0, -1):
            xc = (m + 0.5) * BAY5
            for sg in (-1.0, 1.0):
                for e in (l, n - 1 - l):
                    th = (e + 0.5) / n * math.pi
                    dd = r_mid * math.cos(th)
                    units.append(stone(xc + dd * BAY5 / (2.0 * RHO10),
                                       Y_SPR10 + r_mid * math.sin(th),
                                       sg * dd * S10 / RHO10,
                                       0.31, 0.52 * D10D, 0.45 * W9X,
                                       ang=-sg * A))
    return assemble(units), n


def wribs10():
    """Wall ribs against both clerestory walls: the lancet, one per
    bay per side, centres outside its own span.  Its derived ring
    depth is RING5 to the bit."""
    units = []
    psimax = math.atan2(RHO10, QW10)
    n = int(round(RW10 * psimax / 0.66))
    r_mid = RW10 - 0.5 * D10W
    for i in range(n):
        psi = (i + 0.5) / n * psimax
        for m in range(N_VBAY10, 0, -1):
            xm = (m + 0.5) * BAY5
            for o in (-1.0, 1.0):
                for h in (-1.0, 1.0):
                    x = xm + h * (r_mid * math.cos(psi) - QW10)
                    units.append(stone(x, Y_SPR10 + r_mid * math.sin(psi),
                                       o * (S10 - 0.5 * D10W),
                                       0.33, 0.55 * D10W, 0.55 * D10W))
    return assemble(units), n


def bosses10():
    """Ten bosses, east to west: each one is the keystone of four
    ribs at once.  The only stone in the vault with no rib of its
    own."""
    units = []
    for m in range(N_VBAY10, 0, -1):
        units.append(stone((m + 0.5) * BAY5,
                           Y_SPR10 + RHO10 - 0.5 * D10D, 0.0,
                           0.50, 0.45, 0.50))
    return assemble(units), N_VBAY10


def web10():
    """The shell between the ribs, one voussoir thick.  Course outer,
    bays inner: sixteen courses rise from the springing on every bay
    at once and close around the bosses.  Each course is an arc --
    its cell's own rib profile, scaled -- so the web is the series'
    one arch repeated at every size down to nothing."""
    units = []
    KW = 16
    for l in range(KW):
        t = 1.0 - (l + 0.5) / KW
        tn = max(0.02, t - 1.0 / KW)
        for m in range(N_VBAY10, 0, -1):
            xb = m * BAY5
            # wall cells: a course along x at |z| = t * S10, both flanks
            half = 0.90 * t * SB10
            ns = max(2, int(round(2.0 * half / 0.72)))
            for i in range(ns):
                dxi = -half + (i + 0.5) / ns * 2.0 * half
                y = _y10(SB10 + dxi, t * S10)
                yn = _y10(SB10 + dxi * tn / t, tn * S10)
                hy = 0.55 * max(TW10, abs(yn - y))
                for o in (-1.0, 1.0):
                    units.append(stone(xb + SB10 + dxi, y + 0.5 * TW10,
                                       o * t * S10,
                                       0.55 * 2.0 * half / ns, hy, 0.30))
            # arch cells: two courses at xi = mid +- t * SB10, along z
            zh = 0.90 * t * S10
            ns = max(2, int(round(2.0 * zh / 0.72)))
            for i in range(ns):
                z = -zh + (i + 0.5) / ns * 2.0 * zh
                for h in (-1.0, 1.0):
                    xi = SB10 + h * t * SB10
                    y = _y10(xi, z)
                    yn = _y10(SB10 + h * tn * SB10, z * tn / t)
                    hy = 0.55 * max(TW10, abs(yn - y))
                    units.append(stone(xb + xi, y + 0.5 * TW10, z,
                                       0.30, hy, 0.55 * 2.0 * zh / ns))
    return assemble(units), KW


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

# --- part VI
(WALL6, NW6, NB6, NSLOT6) = aisle_wall()

# Parts I to IV stay legacy.  Part V is drawn SEPARATELY, at the same held-
# back level but with its own material, because this episode's central claim
# -- the wall takes the arcade out of the frame -- is measured by finding
# arcade cells in the finished picture, and a pier merged into M_OLD cannot
# be found.
_LEG6_P, _LEG6_N = _LEG5_P, _LEG5_N

# THE CLOSE SHOT, part VI: part V's close camera, and the same fit.  The
# lancets sit mid-bay on the exact grid the piers sit on, so the arithmetic
# that merged the piers at the established yaw merges the windows too, for
# the same reason, and the angle that resolves one resolves the other --
# nothing about this cut is a new decision.  Whether CAM_N's frame actually
# contains the new wall (it stands 7 m outboard of the piers, plus 1.3 m of
# buttress) is asserted in check_aisles, not assumed here.
CAM_A6 = CAM_N

# --- part VII
(ARCH7, N_ARCH7) = arcade_arches()
(SPAN7, N_SPAN7) = spandrel7()
(SILL7, N_SILL7) = sill7()
(SKIN7, N_SKIN7C) = skin7()
(COL7, N_COL7) = colonnettes()
(SARC7, N_SARC7) = screen_arches()
(FILL7, N_FILL7) = screen_fill()

# Parts I to VI stay legacy; part VI's walls join the pile.  Part V's piers
# keep their own material one more episode -- the checks still have to find
# an arcade cell to prove the arches landed on something.
_LEG7_P = np.vstack([_LEG6_P, WALL6[0]]).astype(np.float32)
_LEG7_N = np.vstack([_LEG6_N, WALL6[1]]).astype(np.float32)

# THE SECTION.  The first one in the series, and it is forced, not chosen.
# The triforium's face is on the INSIDE of the wall.  From the fixed
# camera the near band shows its blank back, and the far band's face --
# which the camera's 28 degrees can see clean over everything, the same
# altitude that saved part V's far row -- crosses the frame at under one
# glyph column per opening: forty openings you can see and cannot count.
# The check derives that number.  So the episode cuts to what a mason
# would draw instead: the building cut open, the north side alone, the
# east three bays, from inside the nave.  Everything south of the cut is
# simply not drawn.  That is what a section IS, and it is declared in the
# description as one.
T_YAW7, T_PITCH7 = -14.0, 8.0


def _pose_t(p):
    return _pose_at(p, T_YAW7, T_PITCH7)


def _nfilt(part):
    m = part[0][:, 2] < 0.0
    return (part[0][m], part[1][m], part[2][m])


ARCH7S, SPAN7S, SILL7S = _nfilt(ARCH7), _nfilt(SPAN7), _nfilt(SILL7)
SKIN7S, COL7S = _nfilt(SKIN7), _nfilt(COL7)
SARC7S, FILL7S = _nfilt(SARC7), _nfilt(FILL7)

_X_SECT = X_TRAN - 3.0 * BAY5                  # 45.1 -- the east three bays
_m7 = ((_LEG7_P[:, 2] < -4.5) & (_LEG7_P[:, 0] > _X_SECT - 2.0)
       & (_LEG7_P[:, 0] < 66.0))
_LEG7T_P, _LEG7T_N = _LEG7_P[_m7], _LEG7_N[_m7]
_mp7 = (PIERS5[0][:, 2] < 0.0) & (PIERS5[0][:, 0] > _X_SECT - 1.5)
PIERS5S = (PIERS5[0][_mp7], PIERS5[1][_mp7], PIERS5[2][_mp7])

_S_NEW = np.vstack([q[0] for q in
                    (ARCH7S, SPAN7S, SILL7S, SKIN7S, COL7S, SARC7S, FILL7S)])
_S_NEW = _S_NEW[_S_NEW[:, 0] > _X_SECT]
GHOST_T = GHOST[(GHOST[:, 2] < -6.0) & (GHOST[:, 0] > _X_SECT - 1.2)
                & (GHOST[:, 0] < 63.5) & (GHOST[:, 1] < 37.0)]
_S_PTS = np.vstack([_S_NEW, PIERS5S[0], GHOST_T]).astype(np.float32)
_spad = _S_PTS.copy()
_spad[:, 1] = _S_PTS[:, 1].min() - 4.5         # keep the caption clear
CAM_T = Camera(G).fit([_pose_t(np.vstack([_S_PTS, _spad]))], margin=1.05)

# inside the nave the light is the nave's own: from the south, high, the
# way clerestory light will actually fall on this screen for centuries.
LAMP7 = np.array([-0.30, 0.52, 0.80])
LAMP7 = LAMP7 / np.linalg.norm(LAMP7)

# --- part VIII
(CAP8, N_CAP8) = cap8()
(STRIP8, N_CRS8, N_STRIP8) = strips8()
(WARC8, N_WARC8) = win_arches8()
(SPAN8, N_SPAN8) = spandrel8()

# Parts I to VI stay legacy and part V's piers join them: the arches
# landed two episodes ago and the checks that needed to see them land are
# closed.  Part VII keeps its own materials one more episode, because this
# episode's held-out check reads the colonnette rhythm off the pixels and
# a colonnette merged into M_OLD cannot be found.
_LEG8_P = np.vstack([_LEG7_P, PIERS5[0]]).astype(np.float32)
_LEG8_N = np.vstack([_LEG7_N, PIERS5[1]]).astype(np.float32)
_m8 = ((_LEG8_P[:, 2] < -4.5) & (_LEG8_P[:, 0] > _X_SECT - 2.0)
       & (_LEG8_P[:, 0] < 66.0))
_LEG8T_P, _LEG8T_N = _LEG8_P[_m8], _LEG8_N[_m8]

# THE SECTION, AGAIN -- and this time reusing it is the point.  CAM_T was
# fitted in part VII to the east three bays WITH the ghost up to the roof
# line, so the frame that watched the wall get cut open already contains
# every course this episode lays.  The camera that showed you the passage
# open to the sky is the camera that watches the sky get shut out.  Zero
# new camera decisions; check_clerestory asserts the new wall top actually
# lands in CAM_T's frame instead of trusting this comment.
CAP8S, STRIP8S = _nfilt(CAP8), _nfilt(STRIP8)
WARC8S, SPAN8S = _nfilt(WARC8), _nfilt(SPAN8)

# --- part IX
(PIER9, N_PIER9, N_PC9) = piers9()
(FLY9, N_VOUS9) = flyers9()
(COP9, N_COP9) = copings9()
(PIN9, N_TIER9) = pinnacles9()

# Part VII joins the legacy pile -- its held-out checks closed last
# episode.  Part VIII keeps its own materials one more episode, because
# this episode's statics press on the mullion strips and the probes have
# to find a strip to prove the flyer head landed on solid wall.
_LEG9_P = np.vstack([_LEG8_P, ARCH7[0], SPAN7[0], SILL7[0], SKIN7[0],
                     COL7[0], SARC7[0], FILL7[0]]).astype(np.float32)
_LEG9_N = np.vstack([_LEG8_N, ARCH7[1], SPAN7[1], SILL7[1], SKIN7[1],
                     COL7[1], SARC7[1], FILL7[1]]).astype(np.float32)

# THE TRANSVERSE SECTION -- the third camera this series' close work has
# ever taken, and like the other two it is forced, not chosen.  The merge
# theorem runs a FOURTH time and comes back worse than ever: a buttress
# is 8.2 m deep along z, the first element deeper than its own bay pitch,
# so at the established yaw ten of them read as one solid corridor.  And
# the section camera parts VII and VIII shared cannot help either: the
# flyers land on the bay lines, which is exactly where the wall is solid,
# so from inside the nave the machine holding the windows open hides
# behind the very stone it presses on.  What is left is the drawing every
# book about these buildings opens with: the HALF-SECTION -- the nave
# spine to the outer pier of one flank, cut across the building.  The
# south half alone: everything north of the spine is simply not drawn,
# exactly as part VII did not draw the south.  A full transverse slice
# was tried first and refused by looking at it: with both flanks in, the
# far flyer hides behind the far wall and the near one drowns between
# two towers of masonry.  Angles: -90 for the axis, VII's own 14 for the
# lean, VII's own 8 of pitch.  Zero new angle decisions.
X_YAW9, X_PITCH9 = -90.0 - T_YAW7, T_PITCH7          # -76, 8


def _pose_x9(p):
    return _pose_at(p, X_YAW9, X_PITCH9)


def _x9(part):
    m = ((part[0][:, 0] > _X_SECT - 2.0) & (part[0][:, 0] < 63.5)
         & (part[0][:, 2] > -1.0))
    return (part[0][m], part[1][m], part[2][m])


_m9 = ((_LEG9_P[:, 0] > _X_SECT - 2.0) & (_LEG9_P[:, 0] < 63.5)
       & (_LEG9_P[:, 2] > -1.0))
_LEG9X_P, _LEG9X_N = _LEG9_P[_m9], _LEG9_N[_m9]
CAP8X, STRIP8X = _x9(CAP8), _x9(STRIP8)
WARC8X, SPAN8X = _x9(WARC8), _x9(SPAN8)
PIER9X, FLY9X = _x9(PIER9), _x9(FLY9)
COP9X, PIN9X = _x9(COP9), _x9(PIN9)

GHOST_X = GHOST[(GHOST[:, 0] > _X_SECT - 1.2) & (GHOST[:, 0] < 63.5)
                & (GHOST[:, 1] < 47.0) & (GHOST[:, 2] > -1.0)]
_X_NEW = np.vstack([PIER9X[0], FLY9X[0], COP9X[0], PIN9X[0]])
_X_PTS = np.vstack([_X_NEW, _LEG9X_P[::5], GHOST_X]).astype(np.float32)
_xpad = _X_PTS.copy()
_xpad[:, 1] = _X_PTS[:, 1].min() - 11.0        # keep the caption clear:
                                   # this section runs ground to roof
                                   # line, so it needs a deeper reserve
                                   # than VII's storey-high frames
CAM_X9 = Camera(G).fit([_pose_x9(np.vstack([_X_PTS, _xpad]))],
                       margin=1.05)

# --- part X
(TARCH10, N_TV10) = tarches10()
(DIAG10, N_DV10) = diags10()
(WRIB10, N_WV10) = wribs10()
(BOSS10, N_BOSS10) = bosses10()
(WEB10, N_WCRS10) = web10()

# Part VIII joins the legacy pile -- its strips held their material one
# episode so part IX's probes could find them, and that check is closed.
# Part IX keeps ALL its materials one more, because this episode's whole
# point is a force arriving at the flyer: the probes have to find a
# flyer to prove the system is in the frame, and the thread walk is
# spliced onto part IX's own.
_LEG10_P = np.vstack([_LEG9_P, CAP8[0], STRIP8[0], WARC8[0],
                      SPAN8[0]]).astype(np.float32)
_LEG10_N = np.vstack([_LEG9_N, CAP8[1], STRIP8[1], WARC8[1],
                      SPAN8[1]]).astype(np.float32)

# THE SAME HALF-SECTION, RAISED TO THE WORK.  The angles are part IX's
# unchanged (which were part VII's, negated) -- but reusing IX's FIT
# was tried first and refused by looking at it: ground-to-roof, the
# vault is the top 15% of the frame and the episode happens in a
# corner.  So the fit does what part VII's did the day sections were
# invented here: frame the STOREY the masons are on.  Everything from
# the triforium cap up -- the clerestory, part IX's flyers waiting at
# the wall, the new vault, the ghost's roof over it.  The ground
# leaves the picture because nothing happens there this episode.
# check_vault asserts the crown and the flyer both project inside the
# grid instead of trusting this comment.
# And the wide view keeps a surprise this episode owes to the NEXT
# one: the crown rises 4.4 m proud of the wall top (the springing is
# frozen at 33.04 and the semicircle's rise is derived, so nobody
# chose this), which means the ceiling shows from OUTSIDE -- a low
# stone hill in the attic, under the ghost's roof line.  Part XI's
# roof will bury it forever.  This is the only episode that ever sees
# the building's sky from above.
_m10 = ((_LEG10_P[:, 0] > _X_SECT - 2.0) & (_LEG10_P[:, 0] < 63.5)
        & (_LEG10_P[:, 2] > -1.0))
_LEG10X_P, _LEG10X_N = _LEG10_P[_m10], _LEG10_N[_m10]
TARCH10X, DIAG10X = _x9(TARCH10), _x9(DIAG10)
WRIB10X, BOSS10X = _x9(WRIB10), _x9(BOSS10)
WEB10X = _x9(WEB10)

_IX_STAND = ((PIER9, PIER9X, M_BUT9), (FLY9, FLY9X, M_FLY9),
             (COP9, COP9X, M_COP9), (PIN9, PIN9X, M_BUT9))

_X10_NEW = np.vstack([TARCH10X[0], DIAG10X[0], WRIB10X[0], BOSS10X[0],
                      WEB10X[0]])
_IX_HI = np.vstack([PIER9X[0], FLY9X[0], COP9X[0], PIN9X[0]])
_IX_HI = _IX_HI[_IX_HI[:, 1] > Y_CAP8]
_X10_PTS = np.vstack([_X10_NEW, _LEG10X_P[_LEG10X_P[:, 1] > Y_CAP8][::4],
                      _IX_HI,
                      GHOST_X[GHOST_X[:, 1] > Y_CAP8]]).astype(np.float32)
_x10pad = _X10_PTS.copy()
_x10pad[:, 1] = _X10_PTS[:, 1].min() - 6.0     # caption reserve: this
                                   # frame is two storeys, not the
                                   # whole building, so the -11 pad of
                                   # part IX would waste half the grid
CAM_X10 = Camera(G).fit([_pose_x9(np.vstack([_X10_PTS, _x10pad]))],
                        margin=1.05)


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

# part VI.  Two cuts, both derived.  OUT at 1.6 because the lancets live on
# the bay grid and merge at the established yaw exactly as the piers did --
# part V's arithmetic, inherited, not re-decided.  BACK at 8.3 because the
# episode's payoff is a fact OF the established frame: the arcade that was
# there at t = 0 is not there at the end.  Part V ended in its close view;
# this one has to come home to be true.
A_GHOST = 0.9
A_CUT = 1.6
A_WALL = (1.7, 7.8)
A_BACK = 8.3
A_END = 10.4

# part VII.  Cut OUT early for VI's reason (nothing this small reads at
# the established yaw -- the check derives under one column per opening)
# and BACK at the end for VI's reason too: the payoff of the wide frame is
# a new blank storey riding above the aisle wall, showing you its back.
# In between, the masonry in the order the masonry has to go in: arches
# (keystones last), spandrels, the passage floor, the back skin, then the
# screen -- colonnettes, arches, fill -- east to west, both rows,
# the camera holding the east three bays of the north side, in section.
V_GHOST = 0.9
V_CUT = 1.5
V_ARCH = (1.6, 3.5)
V_SPAN = (3.5, 4.5)
V_SILL = (4.5, 5.0)
V_SKIN = (5.0, 6.3)
V_COL = (6.3, 7.8)
V_SARC = (7.8, 9.2)
V_FILL = (9.2, 9.8)
V_BACK = 10.3
V_END = 11.8

# part VIII.  Same shape as VII and for VII's reasons -- out early
# (windows on the bay grid merge at the established yaw exactly as the
# piers and the lancets did; the check runs part V's overlap arithmetic a
# third time), home at the end (the payoff of the wide frame is stone
# arriving at the ghost's own line).  The promise is kept FIRST: the cap
# slides over the passage before anything rises above it, because a mason
# cannot stand on a wall that is not there and this series builds in the
# order the stone demands.
W_GHOST = 0.9
W_CUT = 1.5
W_CAP = (1.6, 3.2)
W_STRIP = (3.2, 5.9)
W_ARCH = (5.9, 7.9)
W_SPAN = (7.9, 9.0)
W_BACK = 9.5
W_END = 11.6

# part IX.  Same shape as VII and VIII, one axis over: out early (the
# merge theorem, fourth appearance, worst case yet), home at the end
# (the payoff of the wide frame is the first stone the ghost never
# drew).  The masonry order is the structural order: piers first (a
# flyer needs both ends), then the arcs -- tail to head, the head
# touching the wall last -- then the coping strut, then the ballast.
X_GHOST = 0.9
X_CUT = 1.5
X_PIER = (1.6, 4.6)
X_FLY = (4.6, 7.4)
X_COP = (7.4, 8.5)
X_PIN = (8.5, 9.4)
X_BACK = 9.9
X_END = 12.2

# part X.  Same shape one more time: out early (the vault is interior
# work and the section is the only view that can watch a rib rise),
# home at the end -- where the payoff is a hump of stone cresting the
# wall tops, the back of the new ceiling standing in the open attic,
# one episode before the roof buries it.  The masonry order is the
# mason's: transverse
# arches first (each bay needs its frame), then the diagonals rising
# toward an open crown slot, the wall ribs, the ten bosses closing
# four ribs each in one moment, and only then the web -- the shell can
# only exist after the skeleton it spans between.
Z_GHOST = 0.9
Z_CUT = 1.5
Z_TARCH = (1.6, 3.8)
Z_DIAG = (3.8, 5.9)
Z_WRIB = (5.9, 6.8)
Z_BOSS = (6.8, 7.3)
Z_WEB = (7.3, 10.4)
Z_BACK = 10.9
Z_END = 13.0

T_ENDS = [T_END, C_END, H_END, Q_END, P_END, A_END, V_END, W_END, X_END,
          Z_END]
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
    return (draw_foundation, draw_crypt, draw_choir, draw_transept,
            draw_nave, draw_aisles, draw_triforium,
            draw_clerestory, draw_buttress, draw_vault)[stage](f, stage)


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


def _put7(buf, col, row, z, sh, mat):
    """Part VII draws its stone through a real z-buffer.  Every episode
    before it could order painter's calls back-to-front by hand; VII is
    the first where the work interleaves in depth both ways at once -- a
    far screen seen OVER a near band, a near skin hiding a far screen --
    so the depth decides, not the calling order.  Nearer is larger
    projected z: that is the convention zbuffer() keeps, and part VI's
    occlusion probes established it empirically."""
    ok = visible(G, col, row)
    if not ok.any():
        return
    col, row, z, sh, mat = col[ok], row[ok], z[ok], sh[ok], mat[ok]
    flat, keep = zbuffer(G, col, row, z)
    c, r, zz, s, mt = col[keep], row[keep], z[keep], sh[keep], mat[keep]
    idx = r * G.cols + c
    better = zz > buf["z"].ravel()[idx]
    idx = idx[better]
    buf["z"].ravel()[idx] = zz[better]
    buf["sh"].ravel()[idx] = s[better]
    buf["mat"].ravel()[idx] = mt[better]


def _grow7(buf, part, u, mat, lamp, amb, gain, cam, pose):
    """_grow for part VII: z-buffered, and u < 0 draws nothing at all
    (assemble gives its first unit O = 0, so a clamped u of 0 would lay
    the east stone of every element in frame one)."""
    if u < 0.0:
        return 0
    P, N, O = part
    m = O <= min(1.0, u)
    if not m.any():
        return 0
    col, row, z = cam.project(pose(P[m]))
    sh = (amb + gain * lambert(N[m], lamp)) * depth_cue(z, 1.0, 0.86)
    _put7(buf, col, row, z,
          np.clip(sh, 0.06, 1.0), np.full(int(m.sum()), mat, np.int16))
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


def draw_aisles(f, stage):
    """Part VI.  Open in the established frame with five episodes standing;
    cut to part V's close view; two walls rise course by course, the piers
    sink behind them and come back as stripes through the lancets; then home
    to the fixed frame, where the arcade is no longer in the picture."""
    t = f / float(FPS)
    close = A_CUT <= t < A_BACK
    cam = CAM_A6 if close else CAM
    pose = _pose_n if close else _pose
    buf = {"sh": np.zeros((G.rows, G.cols)),
           "mat": np.zeros((G.rows, G.cols), np.int16)}

    gfade = min(1.0, t / A_GHOST)
    n = int(len(GHOST) * gfade)
    if n > 8:
        col, row, z = cam.project(pose(GHOST[:n]))
        lift = 1.0 + 0.55 * min(1.0, max(0.0, (t - A_WALL[1] - 0.3) / 1.1))
        sh = ((0.20 + 0.34 * depth_cue(z, 1.0, 0.30))
              * (0.72 + 0.28 * gfade) * lift)
        _put(buf, col, row, z + 4000.0, sh, M_GHOST, False)

    # parts I to IV, standing, at the level part III set.
    col, row, z = cam.project(pose(_LEG6_P))
    sh = (0.17 + 0.44 * lambert(_LEG6_N, LAMP)) * depth_cue(z, 1.0, 0.86)
    _put(buf, col, row, z, np.clip(sh, 0.05, 1.0), M_OLD, True)

    # part V, same held-back level, its own material -- the checks have to
    # be able to find an arcade cell in the finished frame to count it.
    col, row, z = cam.project(pose(PIERS5[0]))
    sh = (0.17 + 0.44 * lambert(PIERS5[1], LAMP)) * depth_cue(z, 1.0, 0.86)
    _put(buf, col, row, z, np.clip(sh, 0.05, 1.0), M_NAVE, True)

    u = min(1.0, max(0.0, (t - A_WALL[0]) / (A_WALL[1] - A_WALL[0])))
    LAST["aisle"] = _grow(buf, WALL6, u, M_AISLE, LAMP, 0.28, 0.78, cam,
                          pose=pose)
    LAST["u6"] = u
    LAST["close"] = close

    fr = _paint(buf)
    _label(fr, t, stage)
    return fr


def draw_triforium(f, stage):
    """Part VII.  Open home with six episodes standing; cut to the section
    -- the north side alone, the east three bays, from inside the nave;
    the middle storey goes up in the order the stone demands; then home,
    where the new storey rides above the aisle wall and shows the fixed
    frame nothing but its back."""
    t = f / float(FPS)
    close = V_CUT <= t < V_BACK
    cam = CAM_T if close else CAM
    pose = _pose_t if close else _pose
    lamp = LAMP7 if close else LAMP
    buf = {"sh": np.zeros((G.rows, G.cols)),
           "mat": np.zeros((G.rows, G.cols), np.int16),
           "z": np.full((G.rows, G.cols), -1e9)}

    gfade = min(1.0, t / V_GHOST)
    n = int(len(GHOST) * gfade)
    if n > 8:
        col, row, z = cam.project(pose(GHOST[:n]))
        lift = 1.0 + 0.55 * min(1.0, max(0.0, (t - V_FILL[1] - 0.3) / 1.1))
        sh = ((0.20 + 0.34 * depth_cue(z, 1.0, 0.30))
              * (0.72 + 0.28 * gfade) * lift)
        _put(buf, col, row, z + 4000.0, sh, M_GHOST, False)

    # parts I to VI, standing, at the level part III set.  In the section
    # everything south of the cut simply is not drawn.
    lp, ln = (_LEG7T_P, _LEG7T_N) if close else (_LEG7_P, _LEG7_N)
    col, row, z = cam.project(pose(lp))
    sh = (0.17 + 0.44 * lambert(ln, lamp)) * depth_cue(z, 1.0, 0.86)
    _put7(buf, col, row, z, np.clip(sh, 0.05, 1.0),
          np.full(len(z), M_OLD, np.int16))

    # part V, its own material one more episode: the arches have to be
    # seen landing on something the checks can name.
    pp = PIERS5S if close else PIERS5
    col, row, z = cam.project(pose(pp[0]))
    sh = (0.17 + 0.44 * lambert(pp[1], lamp)) * depth_cue(z, 1.0, 0.86)
    _put7(buf, col, row, z, np.clip(sh, 0.05, 1.0),
          np.full(len(z), M_NAVE, np.int16))

    def win(w):
        return (t - w[0]) / (w[1] - w[0])

    for full, sect, w, mat, amb, gain in (
            (ARCH7, ARCH7S, V_ARCH, M_ARCH, 0.28, 0.78),
            (SPAN7, SPAN7S, V_SPAN, M_ARCH, 0.28, 0.78),
            (SILL7, SILL7S, V_SILL, M_ARCH, 0.28, 0.78),
            (SKIN7, SKIN7S, V_SKIN, M_TRIFB, 0.22, 0.62),
            (COL7, COL7S, V_COL, M_TRIF, 0.30, 0.80),
            (SARC7, SARC7S, V_SARC, M_TRIF, 0.30, 0.80),
            (FILL7, FILL7S, V_FILL, M_TRIF, 0.30, 0.80)):
        _grow7(buf, sect if close else full, win(w), mat, lamp, amb, gain,
               cam, pose)

    LAST["u7"] = min(1.0, max(0.0, win(V_FILL)))
    LAST["close"] = close

    fr = _paint(buf)
    _label(fr, t, stage)
    return fr


# the seven elements of part VII, standing, in both framings.  they keep
# their own materials one more episode -- see _LEG8_P for why.
_VII_STAND = ((ARCH7, ARCH7S, M_ARCH), (SPAN7, SPAN7S, M_ARCH),
              (SILL7, SILL7S, M_ARCH), (SKIN7, SKIN7S, M_TRIFB),
              (COL7, COL7S, M_TRIF), (SARC7, SARC7S, M_TRIF),
              (FILL7, FILL7S, M_TRIF))


def draw_clerestory(f, stage):
    """Part VIII.  Open home with seven episodes standing; cut to the
    section part VII established; the promise first -- the cap course
    closes the passage -- then the strips, the window heads (keystones
    last) and the spandrels; then home, where the wall's outer face has
    arrived exactly on the line the ghost has drawn since part I."""
    t = f / float(FPS)
    close = W_CUT <= t < W_BACK
    cam = CAM_T if close else CAM
    pose = _pose_t if close else _pose
    lamp = LAMP7 if close else LAMP
    buf = {"sh": np.zeros((G.rows, G.cols)),
           "mat": np.zeros((G.rows, G.cols), np.int16),
           "z": np.full((G.rows, G.cols), -1e9)}

    gfade = min(1.0, t / W_GHOST)
    n = int(len(GHOST) * gfade)
    if n > 8:
        col, row, z = cam.project(pose(GHOST[:n]))
        lift = 1.0 + 0.55 * min(1.0, max(0.0, (t - W_SPAN[1] - 0.3) / 1.1))
        sh = ((0.20 + 0.34 * depth_cue(z, 1.0, 0.30))
              * (0.72 + 0.28 * gfade) * lift)
        _put(buf, col, row, z + 4000.0, sh, M_GHOST, False)

    # parts I to VI and the piers, standing, at the level part III set.
    lp, ln = (_LEG8T_P, _LEG8T_N) if close else (_LEG8_P, _LEG8_N)
    col, row, z = cam.project(pose(lp))
    sh = (0.17 + 0.44 * lambert(ln, lamp)) * depth_cue(z, 1.0, 0.86)
    _put7(buf, col, row, z, np.clip(sh, 0.05, 1.0),
          np.full(len(z), M_OLD, np.int16))

    # part VII, standing, held back, own materials.
    for full, sect, mat in _VII_STAND:
        _grow7(buf, sect if close else full, 1.0, mat, lamp, 0.17, 0.44,
               cam, pose)

    def win(w):
        return (t - w[0]) / (w[1] - w[0])

    for full, sect, w, mat in ((CAP8, CAP8S, W_CAP, M_CAP8),
                               (STRIP8, STRIP8S, W_STRIP, M_CLER),
                               (WARC8, WARC8S, W_ARCH, M_CLER),
                               (SPAN8, SPAN8S, W_SPAN, M_CLER)):
        _grow7(buf, sect if close else full, win(w), mat, lamp, 0.28, 0.78,
               cam, pose)

    LAST["u8"] = min(1.0, max(0.0, win(W_SPAN)))
    LAST["close"] = close

    fr = _paint(buf)
    _label(fr, t, stage)
    return fr


# the four elements of part VIII, standing, in both framings of THIS
# episode (full / transverse slab).  own materials one more episode --
# see _LEG9_P for why.
_VIII_STAND = ((CAP8, CAP8X, M_CAP8), (STRIP8, STRIP8X, M_CLER),
               (WARC8, WARC8X, M_CLER), (SPAN8, SPAN8X, M_CLER))


def draw_buttress(f, stage):
    """Part IX.  Open home with eight episodes standing; cut ACROSS the
    building -- the transverse section, the diagram -- and watch the
    piers rise, the arcs leap, the coping land, the ballast go on; then
    home, where the south flank has grown a comb of stone standing 1.2 m
    outside every line the ghost has drawn since part I."""
    t = f / float(FPS)
    close = X_CUT <= t < X_BACK
    cam = CAM_X9 if close else CAM
    pose = _pose_x9 if close else _pose
    lamp = LAMP7 if close else LAMP
    buf = {"sh": np.zeros((G.rows, G.cols)),
           "mat": np.zeros((G.rows, G.cols), np.int16),
           "z": np.full((G.rows, G.cols), -1e9)}

    gfade = min(1.0, t / X_GHOST)
    gsrc = GHOST_X if close else GHOST
    n = int(len(gsrc) * gfade) if not close else len(gsrc)
    if n > 8:
        col, row, z = cam.project(pose(gsrc[:n]))
        lift = 1.0 + 0.55 * min(1.0, max(0.0, (t - X_PIN[1] - 0.3) / 1.1))
        sh = ((0.20 + 0.34 * depth_cue(z, 1.0, 0.30))
              * (0.72 + 0.28 * gfade) * lift)
        _put(buf, col, row, z + 4000.0, sh, M_GHOST, False)

    # parts I to VII, standing, at the level part III set.
    lp, ln = (_LEG9X_P, _LEG9X_N) if close else (_LEG9_P, _LEG9_N)
    col, row, z = cam.project(pose(lp))
    sh = (0.17 + 0.44 * lambert(ln, lamp)) * depth_cue(z, 1.0, 0.86)
    _put7(buf, col, row, z, np.clip(sh, 0.05, 1.0),
          np.full(len(z), M_OLD, np.int16))

    # part VIII, standing, held back, own materials.
    for full, slab, mat in _VIII_STAND:
        _grow7(buf, slab if close else full, 1.0, mat, lamp, 0.17, 0.44,
               cam, pose)

    def win(w):
        return (t - w[0]) / (w[1] - w[0])

    for full, slab, w, mat in ((PIER9, PIER9X, X_PIER, M_BUT9),
                               (FLY9, FLY9X, X_FLY, M_FLY9),
                               (COP9, COP9X, X_COP, M_COP9),
                               (PIN9, PIN9X, X_PIN, M_BUT9)):
        _grow7(buf, slab if close else full, win(w), mat, lamp, 0.28, 0.78,
               cam, pose)

    LAST["u9"] = min(1.0, max(0.0, win(X_PIN)))
    LAST["close"] = close

    fr = _paint(buf)
    _label(fr, t, stage)
    return fr


def draw_vault(f, stage):
    """Part X.  Open home with nine episodes standing; cut to the
    half-section and watch the skeleton go up -- arches, diagonals,
    wall ribs, ten bosses in one moment -- then the web close over
    all of it; then home, where the new ceiling crests the wall tops
    as a low stone hill in the open attic: the only view of the
    building's sky from above there will ever be."""
    t = f / float(FPS)
    close = Z_CUT <= t < Z_BACK
    cam = CAM_X10 if close else CAM
    pose = _pose_x9 if close else _pose
    lamp = LAMP7 if close else LAMP
    buf = {"sh": np.zeros((G.rows, G.cols)),
           "mat": np.zeros((G.rows, G.cols), np.int16),
           "z": np.full((G.rows, G.cols), -1e9)}

    gfade = min(1.0, t / Z_GHOST)
    gsrc = GHOST_X if close else GHOST
    n = int(len(gsrc) * gfade) if not close else len(gsrc)
    if n > 8:
        col, row, z = cam.project(pose(gsrc[:n]))
        lift = 1.0 + 0.55 * min(1.0, max(0.0, (t - Z_WEB[1] - 0.3) / 1.1))
        sh = ((0.20 + 0.34 * depth_cue(z, 1.0, 0.30))
              * (0.72 + 0.28 * gfade) * lift)
        _put(buf, col, row, z + 4000.0, sh, M_GHOST, False)

    # parts I to VIII, standing, at the level part III set.
    lp, ln = (_LEG10X_P, _LEG10X_N) if close else (_LEG10_P, _LEG10_N)
    col, row, z = cam.project(pose(lp))
    sh = (0.17 + 0.44 * lambert(ln, lamp)) * depth_cue(z, 1.0, 0.86)
    _put7(buf, col, row, z, np.clip(sh, 0.05, 1.0),
          np.full(len(z), M_OLD, np.int16))

    # part IX, standing, held back, own materials: the machine that has
    # been waiting one whole episode for what happens in this one.
    for full, slab, mat in _IX_STAND:
        _grow7(buf, slab if close else full, 1.0, mat, lamp, 0.17, 0.44,
               cam, pose)

    def win(w):
        return (t - w[0]) / (w[1] - w[0])

    for full, slab, w, mat in ((TARCH10, TARCH10X, Z_TARCH, M_RIB10),
                               (DIAG10, DIAG10X, Z_DIAG, M_RIB10),
                               (WRIB10, WRIB10X, Z_WRIB, M_RIB10),
                               (BOSS10, BOSS10X, Z_BOSS, M_RIB10),
                               (WEB10, WEB10X, Z_WEB, M_WEB10)):
        _grow7(buf, slab if close else full, win(w), mat, lamp, 0.28, 0.78,
               cam, pose)

    LAST["u10"] = min(1.0, max(0.0, win(Z_WEB)))
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
            M_TRAN: STONE, M_PIER: STONE, M_NAVE: STONE,
            M_AISLE: STONE, M_ARCH: STONE, M_TRIF: STONE,
            M_TRIFB: INNER, M_CAP8: STONE, M_CLER: STONE,
            M_BUT9: STONE, M_FLY9: STONE, M_COP9: STONE,
            M_RIB10: STONE, M_WEB10: STONE}[int(m)]
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


def check_aisles(stage):
    print("THE CATHEDRAL — part %s, %s" % (roman(stage + 1), STAGES[stage]))
    print("  walls                2 x %.0f m, %d courses of %.2f m, %.1f m "
          "to %.2f m" % (X_TRAN - X_NAVE, N_COURSE6, COURSE3, Y_FOOT,
                         Y_FOOT + N_COURSE6 * COURSE3))
    print("  stones               %d wall + %d buttress in %d slots"
          % (NW6, NB6, NSLOT6))

    # RULE 1.  The established view has not drifted.
    d = np.abs(_pose_at(GHOST, -58.0, 28.0) - _pose(GHOST)).max()
    print("  established view unchanged: max disagreement %.2e m" % d)
    assert d < 1e-3, d

    # EVERYTHING DIMENSIONAL IS INHERITED.  22 courses is the transept's
    # arithmetic on this wall's interval: the tallest whole number that stays
    # under the aisle roof line, so 23 must NOT fit or the count is
    # unexplained.  The sill and head are part III's constants, shared, and
    # the windows and buttresses land on part V's bay grid by construction --
    # print the grid so a reader can lay VI over V.
    top22 = Y_FOOT + N_COURSE6 * COURSE3
    top23 = Y_FOOT + (N_COURSE6 + 1) * COURSE3
    print("  course 22 tops out   %.2f m (roof line %.1f), course 23 would "
          "be %.2f -> %s" % (top22, Y_ARCADE, top23,
                             "over" if top23 > Y_ARCADE else "UNDER?"))
    assert top22 <= Y_ARCADE < top23, (top22, top23)
    assert N_COURSE6 == N_COURSE4, (N_COURSE6, N_COURSE4)
    print("  bay grid             %d bays of %.3f m (part V's), lancet "
          "mid-bay, buttress on lines 1..%d" % (N_BAY5, BAY5, N_BAY5 - 1))
    print("  sill %.1f head %.1f  unchanged since part III -- a string "
          "course does not step" % (WIN_SILL, WIN_HEAD))
    assert NB6 == 2 * (N_BAY5 - 1) * N_COURSE6, NB6

    # THE CLOSE CAMERA IS PART V'S, so the claim "no new framing decision"
    # is only true if the wall actually fits in that frame -- it stands 7 m
    # outboard of the piers it was fitted to, plus 1.3 m of buttress.
    c6, r6, _ = CAM_A6.project(_pose_n(WALL6[0]))
    print("  wall in part V's close frame: c%d..%d of %d, r%d..%d of %d"
          % (c6.min(), c6.max(), G.cols, r6.min(), r6.max(), G.rows))
    assert c6.min() >= 0 and c6.max() < G.cols, (c6.min(), c6.max())
    assert r6.min() >= 0 and r6.max() < G.rows, (r6.min(), r6.max())

    # HELD OUT 1 -- THE EPISODE'S CLAIM, measured off the finished frames.
    # First written as "the arcade leaves the picture," and the render said
    # no: 801 cells before, 300 after, and a classifier I wrote to explain
    # the 300 was wrong twice.  So the probes went to the model instead.
    # What is actually true, point by point: the NEAR row's shafts are gone
    # -- every sample on them now reads the new wall's stone, occluded by
    # this episode specifically and not by luck -- and the FAR row survives,
    # because this camera sits 28 degrees above the ground and its sight
    # lines clear a 19 m wall on the way to a pier 23 m behind it.  Stand a
    # person at the door instead and both rows are gone.  One row deleted,
    # one row saved by the altitude of the camera: that is the fact, and it
    # is asserted in all three parts.
    draw(int(1.4 * FPS), stage)
    before = int((LAST["mat"] == M_NAVE).sum())
    draw(int((A_END - 0.2) * FPS), stage)
    m = LAST["mat"]
    after = int((m == M_NAVE).sum())

    def _probe(zrow, lo=4.0, hi=10.0):
        pts = np.array([[X_NAVE + k * BAY5, y, zrow]
                        for k in range(3, 9)
                        for y in np.linspace(lo, hi, 4)], np.float32)
        c, r, _ = CAM.project(_pose(pts))
        vals = [int(m[rr, cc]) for rr, cc in zip(r, c)]
        return (sum(v == M_AISLE for v in vals),
                sum(v == M_NAVE for v in vals), len(vals))

    s_wall, s_pier, s_n = _probe(NAVE_Z)      # near row (same side as camera)
    n_wall, n_pier, n_n = _probe(-NAVE_Z)     # far row
    print("  arcade cells in the established frame: %d before, %d after"
          % (before, after))
    print("  near-row shaft probes: %d/%d read the new wall, %d still pier"
          % (s_wall, s_n, s_pier))
    print("  far-row shaft probes:  %d/%d still pier -- saved by the "
          "camera's 28 degrees, not by the builder" % (n_pier, n_n))
    assert before > 120, before
    assert after < 0.45 * before, (before, after)
    assert s_wall == s_n and s_pier == 0, (s_wall, s_pier, s_n)
    assert n_pier > 0, n_pier

    # HELD OUT 2 -- CAN YOU COUNT THE LANCETS?  A first version measured gap
    # runs across a projected row band and got zero: at this yaw the wall
    # recedes, so a band of rows that brackets the lancets at the near end
    # sweeps solid lower courses at the far end.  Perspective broke the
    # instrument, not the wall.  So, point probes at positions the model
    # dictates: the centre of every lancet must NOT read wall, every
    # interior bay line between them MUST, and the two sets have to
    # alternate as separate columns on screen or the count is not visible.
    draw(int((A_BACK - 0.3) * FPS), stage)
    m = LAST["mat"]
    # The near wall in this view is the NORTH one: a probe of the south
    # wall's lancets came back reading a pier, which can only mean the
    # arcade stood between the camera and that wall.  The wide camera is a
    # southern one, part V's close camera is a northern one, and this check
    # now knows that because it asked, not because I assumed it.
    yb = WIN_SILL + 0.5 * 0.62 * (WIN_HEAD - WIN_SILL)
    # A lancet is carved by SKIPPING 2.4 m slots, so its opening is ragged
    # against the ideal centre line -- a stone edge can sit on the exact
    # centre of a window that reads wide open.  A single centre probe failed
    # 4 of 11 that way.  The claim is "eleven windows read open on screen,"
    # so each window is sampled across its width and across the lancet's
    # height, and counts open if ANY sample shows through.
    wvals = []
    for j in range(N_BAY5):
        xc = X_NAVE + (j + 0.5) * BAY5
        pts = np.array([[xc + dx, y, -AISLE_Z]
                        for dx in (-1.0, -0.5, 0.0, 0.5, 1.0)
                        for y in (WIN_SILL + 0.8, yb, yb + 0.8)], np.float32)
        c, r, _ = CAM_A6.project(_pose_n(pts))
        wvals.append(min(int(m[rr, cc]) for rr, cc in zip(r, c)
                         if 0 <= rr < G.rows and 0 <= cc < G.cols))
    cw, rw, _ = CAM_A6.project(_pose_n(np.array(
        [[X_NAVE + (j + 0.5) * BAY5, yb, -AISLE_Z]
         for j in range(N_BAY5)], np.float32)))
    bpts = np.array([[X_NAVE + j * BAY5, yb, -AISLE_Z]
                     for j in range(1, N_BAY5)], np.float32)
    cb, rb, _ = CAM_A6.project(_pose_n(bpts))
    bvals = [int(m[r, c]) for r, c in zip(rb, cb)]
    holes = sum(v != M_AISLE for v in wvals)
    bars = sum(v == M_AISLE for v in bvals)
    cols = sorted([(int(c), "w") for c in cw] + [(int(c), "b") for c in cb])
    alt = all(a[1] != b[1] for a, b in zip(cols[:-1], cols[1:]))
    sep = min(b[0] - a[0] for a, b in zip(cols[:-1], cols[1:]))
    thru = int((m == M_NAVE).sum())
    print("  lancet centres open %d/%d, bay lines walled %d/%d, "
          "alternating on screen: %s, min separation %d col(s)"
          % (holes, N_BAY5, bars, N_BAY5 - 1, alt, sep))
    print("  arcade cells visible through the lancets, close view: %d"
          % thru)
    assert holes == N_BAY5, (holes, wvals)
    assert bars == N_BAY5 - 1, (bars, bvals)
    assert alt and sep >= 1, (alt, sep)

    sheet = []
    for t in (0.6, 1.5, 2.4, 3.6, 4.8, 6.0, 7.2, 8.0, 10.0):
        fr = draw(int(t * FPS), stage)
        ink, mat = LAST["ink"], LAST["mat"]
        print("  t=%4.1f u=%.2f cov %.3f  ghost %5d old %5d nave %5d "
              "aisle %5d  %s" % (t, LAST["u6"], ink.mean(),
                                 (mat == M_GHOST).sum(),
                                 (mat == M_OLD).sum(),
                                 (mat == M_NAVE).sum(),
                                 (mat == M_AISLE).sum(),
                                 "close" if LAST["close"] else "wide"))
        assert 0.02 < ink.mean() < 0.60, ink.mean()
        for (c0b, r0b, w, h) in LAST["boxes"]:
            assert r0b - 1 >= G.safe_top, ("text above safe", r0b)
            assert r0b + h + 1 <= G.safe_bot, ("text below safe", r0b + h)
            assert c0b - 1 >= 0 and c0b + w + 1 <= G.cols, ("width", c0b, w)
        sheet.append(fr)

    assert LAST["u6"] >= 1.0, LAST["u6"]
    print("  runtime              %.1f s, %d frames  (V was %.1f s)"
          % (A_END, int(A_END * FPS), P_END))
    contact(sheet, os.path.join(_HERE, "..", "content", "cath_sheet.png"),
            cols=3, labels=["0.6 ghost", "1.5 wide", "2.4 close", "3.6",
                            "4.8", "6.0 sill", "7.2 lancets", "8.0",
                            "10.0 home"])


def check_triforium(stage):
    print("THE CATHEDRAL — part %s, %s" % (roman(stage + 1), STAGES[stage]))
    print("  arches               %d of span %.3f m rising %.3f m, "
          "crown %.3f m" % (N_ARCH7, BAY5, ARCH_RISE5, Y_CAP5 + ARCH_RISE5))
    print("  screen               %d colonnettes and %d arches per row, "
          "%d openings" % (N_COL7, N_SARC7 // 2, N_OPEN7))
    print("  band                 %.2f m to %.2f m, courses 23..%d on the "
          "crypt's grid" % (Y_SPAN_TOP, Y_TOP7, K_TOP7))

    # RULE 1.  The established view has not drifted.
    d = np.abs(_pose_at(GHOST, -58.0, 28.0) - _pose(GHOST)).max()
    print("  established view unchanged: max disagreement %.2e m" % d)
    assert d < 1e-3, d

    # THE QUARTER.  The screen is the arcade below it at 1:4, and all
    # three ratios are 4 to the last bit -- not approximately, because
    # each small dimension is the big one divided by 4 and nothing else.
    r_span = BAY5 / SPAN_T
    r_rise = ARCH_RISE5 / RISE_T
    r_shaft = (Y_CAP5 - Y_FOOT) / SHAFT_T
    print("  span %.4f/%.4f  rise %.4f/%.4f  shaft %.4f/%.4f -> "
          "ratios %r %r %r" % (BAY5, SPAN_T, ARCH_RISE5, RISE_T,
                               Y_CAP5 - Y_FOOT, SHAFT_T,
                               r_span, r_rise, r_shaft))
    assert r_span == 4.0 and r_rise == 4.0 and r_shaft == 4.0
    # and 4 divides the bay, so every fourth colonnette stands on a pier
    # centreline -- to within one float ulp, which is under a picometre.
    worst = max(abs(X_A7 + (4 * (k - 1)) * SPAN_T - k * BAY5)
                for k in range(1, N_BAY5))
    print("  every 4th colonnette on a pier centreline: worst error "
          "%.1e m, all %d" % (worst, N_BAY5 - 1))
    assert worst < 1e-12, worst

    # THE COURSE GRID.  Spandrels level off at course 22 -- the height the
    # aisle walls topped out at, so the whole building reaches course 22
    # together.  The voussoir ring is 0.30 m and not more: the extrados
    # crown clears the spandrel top by millimetres.
    extr = Y_CAP5 + ARCH_RISE5 + RING5
    print("  course 22 at %.4f m, arch extrados crown %.4f m -> clears "
          "by %.1f mm" % (Y_SPAN_TOP, extr, 1000 * (Y_SPAN_TOP - extr)))
    assert extr < Y_SPAN_TOP, (extr, Y_SPAN_TOP)
    assert N_COURSE6 * COURSE3 + Y_FOOT == Y_SPAN_TOP
    # and 29 is the least course count that covers the small crowns.
    low = Y_FOOT + (K_TOP7 - 1) * COURSE3
    print("  small crowns %.3f m + %.2f ring: course 28 tops at %.2f "
          "(under), 29 at %.2f (over)" % (CROWN_T, RING_T, low, Y_TOP7))
    assert low < CROWN_T, (low, CROWN_T)
    assert Y_TOP7 >= CROWN_T + RING_T, (Y_TOP7, CROWN_T)

    # THE CROSS-SECTION.  The pier width spent three ways; the passage is
    # the remainder, not a choice.
    print("  %.2f skin + %.2f passage + %.2f screen = %.2f m = the pier"
          % (SKIN_TH, PASSAGE7, SCREEN_TH, SKIN_TH + PASSAGE7 + SCREEN_TH))
    assert abs(SKIN_TH + PASSAGE7 + SCREEN_TH - 2 * PIER5_HW) < 1e-12
    assert PASSAGE7 >= 1.0, PASSAGE7
    head = Y_TOP7 - Y_SILL7
    print("  passage              %.2f m wide, %.2f m of headroom, "
          "open to the sky until part VIII" % (PASSAGE7, head))
    assert head > 2.0, head

    # THE MASS LEDGER.  What hollowing the storey saves, integrated from
    # the same curves the stones were cut to.  Part V sized its piers
    # assuming 1.2 m of solid wall above the capitals (NWALL_T in
    # check_nave); a solid storey at the pier's own 2.4 m would have
    # doubled that.  Hollow, the average comes back to the assumption.
    xs = np.linspace(X_A7, X_B7, 4001)
    op = np.zeros(len(xs))
    for i, x in enumerate(xs):
        m = min(N_OPEN7 - 1, max(0, int((x - X_A7) // SPAN_T)))
        xa = X_A7 + m * SPAN_T
        dx = min(x - xa, xa + SPAN_T - x)
        if dx > 0.16:                       # outside the colonnette
            op[i] = min(_intr(x, xa, xa + SPAN_T) + (SHAFT_T - 0.2),
                        head)
    open_frac = float(np.mean(op)) / head
    t_avg = (0.74 * 2 * PIER5_HW
             + head * (SKIN_TH + SCREEN_TH * (1.0 - open_frac))) / (
                 0.74 + head)
    run = X_B7 - X_A7
    saved = ((2 * PIER5_HW - t_avg) * (0.74 + head) * run * 2
             * 2300.0 / 1000.0)
    print("  screen zone          %.0f%% open; band averages %.2f m of "
          "solid stone in a %.2f m wall" % (100 * open_frac, t_avg,
                                            2 * PIER5_HW))
    print("  part V assumed %.1f m of wall above the capitals; hollow "
          "delivers %.2f.  solid would be %.1f" % (1.2, t_avg,
                                                   2 * PIER5_HW))
    print("  the passage spares the piers %.0f tonnes of stone" % saved)
    assert 1.0 < t_avg < 1.45, t_avg
    assert saved > 1000.0, saved

    # THE RESOLUTION THEOREM -- why this episode needs a section.  The
    # fixed camera's 28 degrees see the FAR screen's face clean over the
    # near band (the altitude that saved part V's far row saves it), but
    # at the established yaw one opening crosses the frame in under a
    # column.  Forty openings you can see and cannot count.
    pa = np.array([[20.0, 21.0, -Z_SCREEN], [20.0 + 10 * SPAN_T, 21.0,
                                             -Z_SCREEN]], np.float32)
    ca, _, _ = CAM.project(_pose(pa))
    ppc = abs(float(ca[1]) - float(ca[0])) / 10.0
    clear = 21.0 + 2 * NAVE_Z * math.tan(math.radians(28.0))
    print("  far screen from the fixed view: sight line clears the near "
          "band at %.1f m (band tops at %.2f)" % (clear, Y_TOP7))
    print("  one opening = %.2f columns in the fixed frame -> the face "
          "is visible and cannot be read" % ppc)
    assert clear > Y_TOP7, (clear, Y_TOP7)
    assert ppc < 1.0, ppc

    # FRAME FACTS, wide, end of episode: the storey is in the picture,
    # its back is what you get, and it is the tallest stone yet laid.
    draw(int(1.4 * FPS), stage)
    before = int((LAST["mat"] == M_TRIFB).sum())
    draw(int((V_END - 0.2) * FPS), stage)
    m = LAST["mat"]
    after = int((m == M_TRIFB).sum())
    pts = np.array([[k * BAY5 + 2.8, 21.5, NAVE_Z + PIER5_HW]
                    for k in range(2, 9)], np.float32)
    c, r, _ = CAM.project(_pose(pts))
    bvals = [int(m[rr, cc]) for rr, cc in zip(r, c)
             if 0 <= rr < G.rows and 0 <= cc < G.cols]
    hits = sum(v == M_TRIFB for v in bvals)
    far = int((m == M_TRIF).sum())
    y_old = float(max(_LEG7_P[:, 1].max(), PIERS5[0][:, 1].max()))
    print("  back-skin cells wide: %d before, %d after; outer-face probes "
          "%d/%d; far screen's face: %d cells, seen and unreadable"
          % (before, after, hits, len(bvals), far))
    print("  tallest stone before this episode %.2f m; the band tops at "
          "%.2f -- the storey is the new high point" % (y_old, Y_TOP7))
    assert before == 0, before
    assert after > 60, after
    assert hits >= 5, (hits, bvals)
    assert far > 20, far
    assert Y_TOP7 > y_old, (Y_TOP7, y_old)

    # SECTION FACTS.  Through every sampled opening the back skin shows --
    # the storey is hollow ON SCREEN, not just in the model -- and the
    # colonnettes read as their own material where the model puts them.
    draw(int(9.9 * FPS), stage)
    m = LAST["mat"]
    # An opening is 1.1 m clear; a single centre pixel can land on a
    # colonnette edge or a bond gap (part VI's lancet lesson, again), so
    # each opening is sampled across its width and counts hollow if ANY
    # sample reads the back skin through it.
    hole, hn = 0, 0
    for mm in range(29, 38):
        p = np.array([[X_A7 + (mm + 0.5) * SPAN_T + dx, y0,
                       -(NAVE_Z + PIER5_HW - SKIN_TH)]
                      for dx in (-0.35, 0.0, 0.35)
                      for y0 in (20.9, 21.3, 21.7)], np.float32)
        c, r, _ = CAM_T.project(_pose_t(p))
        vals = [int(m[rr, cc]) for rr, cc in zip(r, c)
                if 0 <= rr < G.rows and 0 <= cc < G.cols]
        if vals:
            hn += 1
            hole += any(v == M_TRIFB for v in vals)
    # and a colonnette is 0.32 m -- under two columns -- so it too gets
    # sampled along its height rather than trusted to one pixel.
    colh, cn = 0, 0
    for mm in range(30, 40):
        p = np.array([[X_A7 + mm * SPAN_T, y0, -Z_SCREEN]
                      for y0 in (20.3, 20.9, 21.5, 22.1)], np.float32)
        c, r, _ = CAM_T.project(_pose_t(p))
        vals = [int(m[rr, cc]) for rr, cc in zip(r, c)
                if 0 <= rr < G.rows and 0 <= cc < G.cols]
        if vals:
            cn += 1
            colh += any(v == M_TRIF for v in vals)
    print("  openings showing the back skin: %d/%d, colonnettes reading "
          "as screen: %d/%d" % (hole, hn, colh, cn))
    assert hn >= 6 and hole >= hn - 2, (hole, hn)
    assert cn >= 6 and colh >= cn - 2, (colh, cn)

    # HELD OUT -- THE QUARTER, read back off the pixels.  Nothing below
    # measures the model: pier repeats and colonnette repeats are taken
    # from the material buffer of the finished section frame, and their
    # ratio has to hand back the 4 the geometry was built from.
    def runs(row, mat_id):
        cs = np.nonzero(m[row] == mat_id)[0]
        if len(cs) == 0:
            return []
        out, start, prev = [], cs[0], cs[0]
        for c0 in cs[1:]:
            if c0 > prev + 1:
                out.append(0.5 * (start + prev))
                start = c0
            prev = c0
        out.append(0.5 * (start + prev))
        return out

    pr = CAM_T.project(_pose_t(np.array([[50.7, 8.0, -(NAVE_Z - PIER5_HW)]],
                                        np.float32)))[1][0]
    cr = CAM_T.project(_pose_t(np.array([[50.7, 21.0, -Z_SCREEN]],
                                        np.float32)))[1][0]
    pc = runs(int(pr), M_NAVE)
    cc0 = runs(int(cr), M_TRIF)
    pg = np.diff(pc)
    cg = np.diff(cc0)
    cg = cg[(cg > 0.4 * np.median(cg)) & (cg < 1.6 * np.median(cg))]
    pg = pg[(pg > 0.4 * np.median(pg)) & (pg < 1.6 * np.median(pg))]
    ratio = float(np.mean(pg)) / float(np.mean(cg))
    print("  pier repeat %.1f cols over %d gaps, colonnette repeat %.2f "
          "over %d -> ratio %.2f (built from 4)"
          % (float(np.mean(pg)), len(pg), float(np.mean(cg)), len(cg),
             ratio))
    assert len(pg) >= 2 and len(cg) >= 6, (len(pg), len(cg))
    assert 3.5 < ratio < 4.5, ratio

    sheet = []
    for t in (0.6, 1.4, 2.6, 4.1, 5.6, 7.2, 8.6, 9.9, 11.4):
        fr = draw(int(t * FPS), stage)
        ink, mat = LAST["ink"], LAST["mat"]
        print("  t=%4.1f u=%.2f cov %.3f  ghost %5d old %5d arch %5d "
              "skin %4d screen %4d  %s"
              % (t, LAST["u7"], ink.mean(), (mat == M_GHOST).sum(),
                 (mat == M_OLD).sum(), (mat == M_ARCH).sum(),
                 (mat == M_TRIFB).sum(), (mat == M_TRIF).sum(),
                 "close" if LAST["close"] else "wide"))
        assert 0.02 < ink.mean() < 0.60, ink.mean()
        for (c0b, r0b, w, h) in LAST["boxes"]:
            assert r0b - 1 >= G.safe_top, ("text above safe", r0b)
            assert r0b + h + 1 <= G.safe_bot, ("text below safe", r0b + h)
            assert c0b - 1 >= 0 and c0b + w + 1 <= G.cols, ("width", c0b, w)
        sheet.append(fr)

    assert LAST["u7"] >= 1.0, LAST["u7"]
    print("  runtime              %.1f s, %d frames  (VI was %.1f s)"
          % (V_END, int(V_END * FPS), A_END))
    contact(sheet, os.path.join(_HERE, "..", "content", "cath_sheet.png"),
            cols=3, labels=["0.6 ghost", "1.4 wide", "2.6 arches",
                            "4.1 spandrel", "5.6 skin", "7.2 colonnettes",
                            "8.6 arches again", "9.9 screen", "11.4 home"])


def check_clerestory(stage):
    print("THE CATHEDRAL — part %s, %s" % (roman(stage + 1), STAGES[stage]))
    print("  windows              %d a side, span %.3f m rising %.3f m"
          % (N_WIN8, SPAN_8, RISE_8))
    print("  storey               %.2f m to %.2f m, courses %d..%d on the "
          "crypt's grid" % (Y_CAP8, Y_TOP8, K_CAP8 + 1, K_TOP8))

    # RULE 1.  The established view has not drifted.
    d = np.abs(_pose_at(GHOST, -58.0, 28.0) - _pose(GHOST)).max()
    print("  established view unchanged: max disagreement %.2e m" % d)
    assert d < 1e-3, d

    # THE HALVES.  The window is the arcade arch at exactly 1:2, so the
    # finished elevation is ONE arch at three scales -- 1, 1/4, 1/2 --
    # and every ratio is a power of two, exact in floats.
    r_span = BAY5 / SPAN_8
    r_rise = ARCH_RISE5 / RISE_8
    r_ring = RING5 / RING8
    r_trif = SPAN_8 / SPAN_T
    print("  span %.4f/%.4f rise %.4f/%.4f ring %.3f/%.3f -> ratios "
          "%r %r %r" % (BAY5, SPAN_8, ARCH_RISE5, RISE_8, RING5, RING8,
                        r_span, r_rise, r_ring))
    print("  one arch, three storeys: arcade 1, triforium 1/4, "
          "clerestory 1/2 (window/screen-opening = %r)" % r_trif)
    assert r_span == 2.0 and r_rise == 2.0 and r_ring == 2.0
    assert r_trif == 2.0
    # and the jambs stand on the triforium's colonnette lines -- the
    # quarter-bay grid, shared without being told to.  one ulp again.
    worst = 0.0
    for k in range(1, N_WIN8 + 1):
        xa, xb = _jambs8(k)
        worst = max(worst,
                    abs(xa - (X_A7 + (4 * (k - 1) + 1) * SPAN_T)),
                    abs(xb - (X_A7 + (4 * (k - 1) + 3) * SPAN_T)))
    print("  every jamb on a colonnette line: worst error %.1e m, "
          "all %d windows" % (worst, N_WIN8))
    assert worst < 1e-12, worst

    # THE COURSE GRID.  The wall tops out at course 45, and course 45 IS
    # the frozen mass top from part I -- the crypt's course height and
    # the first video's silhouette agree, to the bit.
    dtop = Y_FOOT + K_TOP8 * COURSE3 - NAVE_Y
    print("  Y_FOOT + %d courses = %.10f; NAVE_Y = %.1f; diff %r"
          % (K_TOP8, Y_FOOT + K_TOP8 * COURSE3, NAVE_Y, dtop))
    print("  the storey is %d courses exactly" % (K_TOP8 - K_CAP8))
    assert abs(dtop) < 1e-12, dtop
    assert K_TOP8 - K_CAP8 == 15
    # and course 41 is the GREATEST springing whose arch clears the top.
    over = Y_FOOT + (K_SPRING8 + 1) * COURSE3 + RISE_8 + RING8
    crown = Y_SPRING8 + RISE_8 + RING8
    print("  extrados crown %.3f m from course %d (clears %.2f); from "
          "course %d it is %.3f (overtops)" % (crown, K_SPRING8,
                                               NAVE_Y - crown,
                                               K_SPRING8 + 1, over))
    assert crown <= NAVE_Y, crown
    assert over > NAVE_Y, over

    # THE PROMISE.  Part VII's description: the passage is open to the
    # sky until part VIII, and the clerestory sill is its ceiling.  The
    # cap is course 30 at FULL pier thickness -- one course that is both
    # the triforium's lid and the window sill.
    print("  cap: course %d, %.2f m thick = the whole pier; the passage "
          "keeps %.2f m of headroom under it" % (K_CAP8, 2 * PIER5_HW,
                                                 Y_TOP7 - Y_SILL7))
    assert abs(Y_CAP8 - Y_TOP7 - COURSE3) < 1e-12
    assert Y_TOP7 - Y_SILL7 > 2.0

    # THE WALL'S THICKNESS -- three old numbers agree, none of them new.
    NWALL_T = 1.2                      # what part V's stress check assumed
    zi = NAVE_Z - PIER5_HW             # the screen's own inner line
    print("  wall %.2f m thick: = pier half-width %.2f, = part V's "
          "assumed %.1f, and %.2f + %.2f = %.2f = the frozen mass face"
          % (WALL8_TH, PIER5_HW, NWALL_T, zi, WALL8_TH, zi + WALL8_TH))
    assert WALL8_TH == PIER5_HW == NWALL_T
    assert zi + WALL8_TH == NAVE_Z

    # THE LIGHT LEDGER.  How much of the storey is sky, integrated from
    # the same curves the stones were cut to.
    xs = np.linspace(X_A7, X_B7, 4001)
    hgt = Y_TOP8 - Y_CAP8
    op = np.zeros(len(xs))
    for i, x in enumerate(xs):
        k = int(x // BAY5)
        if 1 <= k <= N_WIN8:
            xa, xb = _jambs8(k)
            if xa < x < xb:
                op[i] = (Y_SPRING8 - Y_CAP8) + _intr(x, xa, xb)
    open_frac = float(np.mean(op)) / hgt
    area = float(np.mean(op)) * (X_B7 - X_A7)
    print("  the storey is %.0f%% window; each side admits %.0f m2 of "
          "sky" % (100 * open_frac, area))
    assert 0.35 < open_frac < 0.55, open_frac
    # and the mass ledger keeps running: against a solid storey at the
    # pier's own thickness, cap included.
    v_solid = 2 * PIER5_HW * (Y_TOP8 - Y_TOP7) * (X_B7 - X_A7)
    v_built = (2 * PIER5_HW * COURSE3
               + WALL8_TH * hgt * (1.0 - open_frac)) * (X_B7 - X_A7)
    saved = (v_solid - v_built) * 2 * 2300.0 / 1000.0
    print("  built %.0f m3 a side where solid would be %.0f -> the "
          "windows spare the piers %.0f tonnes" % (v_built, v_solid,
                                                   saved))
    assert saved > 2000.0, saved

    # THE MERGE THEOREM, third appearance.  The windows sit on the bay
    # grid, so at the established yaw the mullion strips overlap exactly
    # as the piers did in part V and the lancets in part VI -- the wall
    # reads solid and the windows cannot be counted.  Hence the section.
    step = BAY5 * math.cos(math.radians(58.0))
    reads = SPAN_8 * (math.cos(math.radians(58.0))
                      + math.sin(math.radians(58.0)))
    print("  at the established yaw a mullion reads %.2f m against a "
          "%.2f m step -> neighbours overlap %.2f m; the fixed view "
          "cannot count ten windows" % (reads, step, reads - step))
    assert reads > step, (reads, step)

    # FRAME FACTS, wide, end of episode: the storey is in the picture,
    # and the wall has arrived at the ghost's line.
    draw(int((W_END - 0.2) * FPS), stage)
    m = LAST["mat"]
    cler = int((m == M_CLER).sum())
    hits, hn = 0, 0
    for mm in range(3, 10):
        p = np.array([[mm * BAY5 + dx, y0, NAVE_Z - 0.5 * WALL8_TH]
                      for dx in (-0.5, 0.0, 0.5)
                      for y0 in (28.5, 30.0, 31.5)], np.float32)
        c, r, _ = CAM.project(_pose(p))
        vals = [int(m[rr, cc]) for rr, cc in zip(r, c)
                if 0 <= rr < G.rows and 0 <= cc < G.cols]
        if vals:
            hn += 1
            hits += any(v == M_CLER for v in vals)
    top_new = float(max(STRIP8[0][:, 1].max(), SPAN8[0][:, 1].max()))
    print("  wide: %d clerestory cells; mullion probes %d/%d; stone "
          "reaches %.2f m of the frozen %.1f (a joint short, like every "
          "course)" % (cler, hits, hn, top_new, NAVE_Y))
    assert cler > 150, cler
    assert hn >= 5 and hits >= hn - 1, (hits, hn)
    assert top_new > NAVE_Y - 0.08, top_new
    # reusing CAM_T is asserted, not trusted: the new wall top lands in
    # the section frame.
    tp = np.array([[46.0, Y_TOP8, -NAVE_Z], [61.5, Y_TOP8, -NAVE_Z]],
                  np.float32)
    c, r, _ = CAM_T.project(_pose_t(tp))
    print("  section frame holds the new top: rows %s cols %s "
          "(grid %dx%d)" % (list(r), list(c), G.rows, G.cols))
    assert all(0 <= rr < G.rows for rr in r), r
    assert all(0 <= cc < G.cols for cc in c), c

    # SECTION FACTS.  The windows are HOLES on screen -- through every
    # sampled aperture the sky shows -- and the cap reads as its own
    # stone above the passage.  Multi-sample, part VI's lesson.
    draw(int(9.3 * FPS), stage)
    m = LAST["mat"]
    holes, sn = 0, 0
    for k in (8, 9, 10):
        xa, xb = _jambs8(k)
        p = np.array([[0.5 * (xa + xb) + dx, y0, -Z_WALL8]
                      for dx in (-0.8, 0.0, 0.8)
                      for y0 in (26.5, 29.0, 31.5)], np.float32)
        c, r, _ = CAM_T.project(_pose_t(p))
        vals = [int(m[rr, cc]) for rr, cc in zip(r, c)
                if 0 <= rr < G.rows and 0 <= cc < G.cols]
        if vals:
            sn += 1
            empty = sum(v in (0, M_GHOST) for v in vals)
            holes += empty >= max(1, len(vals) - 3)
    caps, cn = 0, 0
    for x0 in (47.0, 50.0, 53.0, 56.0, 59.0):
        p = np.array([[x0, y0, -(NAVE_Z - PIER5_HW)]
                      for y0 in (24.30, 24.55, 24.80)], np.float32)
        c, r, _ = CAM_T.project(_pose_t(p))
        vals = [int(m[rr, cc]) for rr, cc in zip(r, c)
                if 0 <= rr < G.rows and 0 <= cc < G.cols]
        if vals:
            cn += 1
            caps += any(v == M_CAP8 for v in vals)
    print("  apertures reading sky: %d/%d; cap probes reading the new "
          "ceiling: %d/%d" % (holes, sn, caps, cn))
    assert sn == 3 and holes >= 2, (holes, sn)
    assert cn >= 4 and caps >= cn - 1, (caps, cn)

    # HELD OUT -- read back off the pixels, nothing from the model.
    # (a) mullion pitch / colonnette pitch = 4: the storeys share the bay.
    # (b) window pitch / window opening = 2: the wall gives half of
    #     itself to the sky.  Both from the finished section frame.
    def runs(row, mat_id):
        cs = np.nonzero(m[row] == mat_id)[0]
        if len(cs) == 0:
            return []
        out, start, prev = [], cs[0], cs[0]
        for c0 in cs[1:]:
            if c0 > prev + 1:
                out.append((0.5 * (start + prev), prev - start + 1))
                start = c0
            prev = c0
        out.append((0.5 * (start + prev), prev - start + 1))
        return out

    jr = CAM_T.project(_pose_t(np.array([[53.0, 29.0,
                                          -(NAVE_Z - PIER5_HW)]],
                                        np.float32)))[1][0]
    cr = CAM_T.project(_pose_t(np.array([[53.0, 21.0, -Z_SCREEN]],
                                        np.float32)))[1][0]
    mruns = runs(int(jr), M_CLER)
    truns = runs(int(cr), M_TRIF)
    mc = [c0 for (c0, _w) in mruns]
    tc = [c0 for (c0, _w) in truns]
    mg, tg = np.diff(mc), np.diff(tc)
    tg = tg[(tg > 0.4 * np.median(tg)) & (tg < 1.6 * np.median(tg))]
    pitch = float(np.mean(mg))
    r4 = pitch / float(np.mean(tg))
    gaps = [mc[i + 1] - mc[i] - 0.5 * (mruns[i][1] + mruns[i + 1][1])
            for i in range(len(mc) - 1)]
    # the raw pitch/opening ratio does NOT come back as 2, and the reason
    # is worth the check it broke: the aperture is seen THROUGH a 1.2 m
    # wall at 14 degrees off its normal, so the jamb's own thickness
    # shades tan(14) * 1.2 = 0.30 m of the opening -- a window is
    # narrower than its span from anywhere but straight on -- and the
    # inclusive pixel runs shave about a cell more.  Correct for both and
    # the pixels agree with the model; leave them out and they honestly
    # cannot.
    theta = math.radians(abs(T_YAW7))
    exp_gap = ((SPAN_8 - WALL8_TH * math.tan(theta)) * pitch / BAY5) - 1.0
    r2 = float(np.mean(gaps)) / exp_gap
    print("  held out: mullion pitch %.1f cols / colonnette %.2f -> "
          "%.2f (built from 4); opening %.1f cols vs %.1f predicted "
          "through the wall's own thickness -> %.2f"
          % (pitch, float(np.mean(tg)), r4, float(np.mean(gaps)),
             exp_gap, r2))
    assert len(mg) >= 2 and len(tg) >= 6, (len(mg), len(tg))
    assert 3.4 < r4 < 4.6, r4
    assert 0.8 < r2 < 1.2, r2

    sheet = []
    for t in (0.6, 1.4, 2.4, 4.4, 6.8, 8.4, 9.3, 10.2, 11.4):
        fr = draw(int(t * FPS), stage)
        ink, mat = LAST["ink"], LAST["mat"]
        print("  t=%4.1f u=%.2f cov %.3f  ghost %5d old %5d trif %5d "
              "cap %4d cler %5d  %s"
              % (t, LAST["u8"], ink.mean(), (mat == M_GHOST).sum(),
                 (mat == M_OLD).sum(), (mat == M_TRIF).sum(),
                 (mat == M_CAP8).sum(), (mat == M_CLER).sum(),
                 "close" if LAST["close"] else "wide"))
        assert 0.02 < ink.mean() < 0.60, ink.mean()
        for (c0b, r0b, w, h) in LAST["boxes"]:
            assert r0b - 1 >= G.safe_top, ("text above safe", r0b)
            assert r0b + h + 1 <= G.safe_bot, ("text below safe", r0b + h)
            assert c0b - 1 >= 0 and c0b + w + 1 <= G.cols, ("width", c0b, w)
        sheet.append(fr)

    assert LAST["u8"] >= 1.0, LAST["u8"]
    print("  runtime              %.1f s, %d frames  (VII was %.1f s)"
          % (W_END, int(W_END * FPS), V_END))
    contact(sheet, os.path.join(_HERE, "..", "content", "cath_sheet.png"),
            cols=3, labels=["0.6 ghost", "1.4 wide", "2.4 the cap",
                            "4.4 strips", "6.8 window heads",
                            "8.4 spandrels", "9.3 done", "10.2 home",
                            "11.4 the line"])


def check_buttress(stage):
    print("THE CATHEDRAL — part %s, %s" % (roman(stage + 1), STAGES[stage]))
    print("  buttresses           %d a side on the bay lines, leap %.2f m,"
          " arc radius %.3f m" % (N_BUT9, R9, R_SEG9))
    print("  pier                 courses %d..%d on part VI's buttresses; "
          "pinnacle %.2f m" % (N_COURSE6 + 1, K_PIER9, PIN9_H))

    # RULE 1.  The established view has not drifted.
    d = np.abs(_pose_at(GHOST, -58.0, 28.0) - _pose(GHOST)).max()
    print("  established view unchanged: max disagreement %.2e m" % d)
    assert d < 1e-3, d

    # THE DERIVATIONS.  Every dimension is older than this episode.
    print("  leap: AISLE_Z - PIER5_HW - NAVE_Z = %.1f - %.1f - %.1f = %.1f"
          % (AISLE_Z, PIER5_HW, NAVE_Z, R9))
    assert R9 == AISLE_Z - PIER5_HW - NAVE_Z
    dr = abs(D9 / (2.0 * R9) - RING5 / BAY5)
    print("  ring depth %.3f m: the series' own ratio ring/span = %.5f "
          "(part VII's), diff %.1e" % (D9, RING5 / BAY5, dr))
    assert dr < 1e-15, dr
    # the chord is 45 degrees: rise = run because head and tail were set
    # on the same grid, R9 apart.  (One ulp of re-addition allowed --
    # part VII's lesson: ratios with ==, positions with < 1e-12.)
    assert abs((Y_HEAD9 - Y_SPR9) - R9) < 1e-12
    chord = math.hypot((Z_PIER9 - PIER9_HZ) - NAVE_Z, Y_HEAD9 - Y_SPR9)
    print("  chord: rise %.1f = run %.1f -> 45 degrees; length %.4f = "
          "radius %.4f (diff %.1e)" % (Y_HEAD9 - Y_SPR9, R9, chord,
                                       R_SEG9, abs(chord - R_SEG9)))
    assert abs(chord - R_SEG9) < 1e-12
    # radius = chord happens at ONE sweep only: 60 degrees, the
    # equilateral angle every arch in this series is built at.
    cz, cy, at, ah = _arc9(1.0)
    sweep = math.degrees(abs(ah - at))
    print("  sweep %.4f degrees (radius = chord <-> equilateral): the "
          "flyer springs at 75 degrees and arrives at 15" % sweep)
    assert abs(sweep - 60.0) < 1e-9, sweep
    # pinnacle: the equilateral triangle made solid.
    assert PIN9_H / PIER5_HW == math.sqrt(3.0)
    print("  pinnacle/half-width = sqrt(3) exactly: 60-degree faces")

    # THE COURSES.  Head and pier top both chosen by the grid, both
    # asserted both ways.
    print("  head %.2f m: least course reaching over the springing line "
          "%.2f (course %d gives %.2f, short by %.2f)"
          % (Y_HEAD9, Y_SPRING8, K_SPR9 - 1,
             Y_FOOT + (K_SPR9 - 1) * COURSE3 + R9,
             Y_SPRING8 - (Y_FOOT + (K_SPR9 - 1) * COURSE3 + R9)))
    assert Y_HEAD9 >= Y_SPRING8
    assert Y_FOOT + (K_SPR9 - 1) * COURSE3 + R9 < Y_SPRING8
    cop_top = Y_SPR9 + D9
    print("  pier top %.2f m: least course over the coping's tail "
          "arrival %.3f (course %d is %.2f, under it)"
          % (Y_TOP9, cop_top, K_PIER9 - 1,
             Y_FOOT + (K_PIER9 - 1) * COURSE3))
    assert Y_TOP9 >= cop_top
    assert Y_FOOT + (K_PIER9 - 1) * COURSE3 < cop_top

    # THE PROMISE, four episodes old.  Part V, the day the bay was
    # frozen: "the buttresses ... have to land on these lines."
    print("  every pier centred on a bay line, x = m * %.3f -- part V's "
          "promise, kept" % BAY5)
    # and the head presses only on solid wall: the flyer is narrower
    # than the mullion strip it lands on, so no window is touched.
    print("  flyer half-width %.2f < strip half-width %.3f: the head "
          "touches no window -- and for the same reason the old section "
          "camera cannot see a flyer through one" % (0.5 * W9X,
                                                     0.25 * BAY5))
    assert 0.5 * W9X < 0.25 * BAY5

    # OUTSIDE THE DRAWING.  MASSES has no buttresses: these are the
    # first stones of the series standing outside the ghost.
    gm = GHOST[(GHOST[:, 0] > 1.0) & (GHOST[:, 0] < 61.0)
               & (GHOST[:, 1] < 30.0)]
    gz = float(np.abs(gm[:, 2]).max())
    print("  ghost's widest line along the nave: |z| = %.1f; pier face "
          "at %.1f -- %.1f m proud of the drawing" % (gz,
              Z_PIER9 + PIER9_HZ, Z_PIER9 + PIER9_HZ - gz))
    assert Z_PIER9 + PIER9_HZ > gz
    print("  the flyer crosses the aisle %.2f m above its roof line"
          % (Y_SPR9 - AISLE_Y))
    assert Y_SPR9 > AISLE_Y

    # THE MERGE THEOREM, fourth appearance, worst case: the first
    # element DEEPER than its own bay pitch.
    dep = Z_PIER9 + PIER9_HZ - NAVE_Z
    step = BAY5 * math.cos(math.radians(58.0))
    reads = (dep * math.sin(math.radians(58.0))
             + 2.0 * PIER9_HX * math.cos(math.radians(58.0)))
    print("  at the established yaw a buttress reads %.2f m against a "
          "%.2f m step -> overlap %.2f m, the deepest yet; ten of them "
          "are one corridor" % (reads, step, reads - step))
    assert reads > step

    # THE STATICS.  Walk the thrust line: force (H, Vw) entering at the
    # head, weights added as the cut moves outward, the line kept inside
    # masonry all the way to the pier base.  Tonnes, metres, stone at
    # 2.3 t/m3.  The masonry is one piece from intrados to coping top
    # because the web fills the spandrel -- the first draft left it
    # open, and this walk is what refused it.
    rho = 2.3
    zt = Z_PIER9 - PIER9_HZ                     # 13.8, the tail face
    r_in, r_mid = R_SEG9 - D9, R_SEG9 - 0.5 * D9
    K = 40

    def y_chord(z):
        return Y_HEAD9 - (z - NAVE_Z)

    def y_intra(z):
        return cy - math.sqrt(max(R_SEG9 ** 2 - (z - cz) ** 2, 0.0))

    # weight per z-slab for a given flyer width: arc by its own
    # parametrisation, web by the gap it fills, coping uniform.
    aa = np.linspace(at, ah, 800)
    az = cz + r_mid * np.cos(aa)

    def slabs(width):
        w_arc = (R_SEG9 * math.pi / 3.0) * D9 * width * rho
        ab = (np.histogram(az, bins=K, range=(NAVE_Z, zt))[0]
              / 800.0 * w_arc)
        wb = np.zeros(K)
        dz = (zt - NAVE_Z) / K
        for k in range(K):
            z = NAVE_Z + (k + 0.5) * dz
            d2 = (z - cz) ** 2
            yb = (cy - math.sqrt(r_in * r_in - d2) if r_in * r_in > d2
                  else y_chord(z))
            wb[k] = max(0.0, y_chord(z) - yb) * dz * width * rho
        cb = R9 * D9 * width * rho / K
        return ab + wb + cb

    w_pier = (2 * PIER9_HX) * (2 * PIER9_HZ) * (Y_TOP9 - Y_BASE9) * rho
    w_pin = (2 * PIER9_HX) * (2 * PIER9_HZ) * PIN9_H / 3.0 * rho
    wbin = slabs(W9X)
    w_fly = float(wbin.sum())
    print("  weights per bay line: flyer %.1f t (arc + web + coping), "
          "pier %.1f t, pinnacle %.1f t" % (w_fly, w_pier, w_pin))

    def walk(H, Vw, y0, pin, wb):
        if H < 1e-6:
            return False
        Fz, Fy = H, Vw
        M = NAVE_Z * Fy - y0 * Fz
        for k in range(K):
            z1 = NAVE_Z + (zt - NAVE_Z) * (k + 1) / K
            zm = NAVE_Z + (zt - NAVE_Z) * (k + 0.5) / K
            Fy -= wb[k]
            M -= zm * wb[k]
            y = (z1 * Fy - M) / Fz
            if not (y_intra(z1) - 0.06 <= y <= y_chord(z1) + D9 + 0.06):
                return False
        y_arr = (zt * Fy - M) / Fz
        if not (Y_SPR9 - 0.5 <= y_arr <= Y_TOP9):
            return False
        if pin:
            Fy -= w_pin
            M -= Z_PIER9 * w_pin
        nc = K_PIER9 - N_COURSE6
        for c in range(nc):
            Fy -= w_pier / nc
            M -= Z_PIER9 * (w_pier / nc)
            yc_ = Y_TOP9 - (c + 1) * COURSE3
            z_ = (M + yc_ * Fz) / Fy
            if not (zt - 0.05 <= z_ <= Z_PIER9 + PIER9_HZ + 0.05):
                return False
        return True

    def feasible(H, pin, vlo, wb):
        for Vw in np.linspace(vlo, 10.0, 81):
            for y0 in np.linspace(Y_HEAD9, Y_HEAD9 + D9, 7):
                if walk(H, Vw, y0, pin, wb):
                    return True
        return False

    Hs = np.arange(0.5, 60.0, 0.25)

    def find_min(pin, vlo, wb):
        for h in Hs:
            if feasible(h, pin, vlo, wb):
                return float(h)
        return float("inf")

    def find_max(pin, vlo, wb):
        for h in Hs[::-1]:
            if feasible(h, pin, vlo, wb):
                return float(h)
        return 0.0

    # today: no vault.  the only vertical the wall can lend the head is
    # the spandrel standing above the contact.
    v_spandrel = 0.5 * BAY5 * WALL8_TH * (NAVE_Y - Y_HEAD9) * rho
    H_self = find_min(True, -v_spandrel, wbin)
    # part X: the vault's springing weight arrives at the head too; the
    # walk lets up to 30 t of it bear down there.  stated assumption.
    H_max = find_max(True, -30.0, wbin)
    H_max0 = find_max(False, -30.0, wbin)
    # the counterfactual that sized the width: the same flyer at part
    # VI's buttress width.
    H_wide = find_min(True, -v_spandrel, slabs(2.0 * PIER9_HX))
    w_strip = 0.5 * BAY5 * WALL8_TH * (NAVE_Y - Y_CAP8) * rho
    cap_strip = w_strip * (NAVE_Z - 0.6 - (NAVE_Z - WALL8_TH)) / \
        (Y_HEAD9 - Y_CAP8)
    print("  today, vault-less, the flyer leans on the wall with at "
          "least H = %.2f t; one mullion strip alone hinges at %.1f t"
          % (H_self, cap_strip))
    if math.isinf(H_wide):
        print("  the width was sized by this: at the pier's own 2.3 m "
              "there is NO thrust at which a line fits the shape -- it "
              "cannot stand at any push.  at %.1f m it stands." % W9X)
    else:
        print("  the width was sized by this: at the pier's own 2.3 m "
              "the least standing thrust is %.2f t.  at %.1f m it is "
              "%.2f" % (H_wide, W9X, H_self))
    print("  with the vault pushing, the system takes up to H = %.2f t "
          "a bay before the line leaves the pier; without the pinnacle "
          "%.2f -> the %.1f t of ballast buys %.2f t"
          % (H_max, H_max0, w_pin, H_max - H_max0))
    print("  PART X'S BUDGET: the vault must arrive under %.1f t a bay "
          "(head entry on its face, up to 30 t of springing weight)"
          % H_max)
    assert 0.5 <= H_self <= 20.0, H_self
    assert H_wide > 2.5 * H_self, (H_wide, H_self)
    assert 12.0 <= H_max <= 60.0, H_max
    assert H_max0 <= H_max + 1e-9, (H_max0, H_max)
    assert H_self < H_max

    # FRAME FACTS, wide, end of episode.
    draw(int((X_END - 0.2) * FPS), stage)
    m = LAST["mat"]
    new = int(((m == M_BUT9) | (m == M_FLY9) | (m == M_COP9)).sum())
    hits, hn = 0, 0
    for mm in range(3, 10):
        p = np.array([[mm * BAY5 + dx, y0, Z_PIER9 + 0.8]
                      for dx in (-0.8, 0.0, 0.8)
                      for y0 in (20.5, 23.0, 25.5)], np.float32)
        c, r, _ = CAM.project(_pose(p))
        vals = [int(m[rr, cc]) for rr, cc in zip(r, c)
                if 0 <= rr < G.rows and 0 <= cc < G.cols]
        if vals:
            hn += 1
            hits += any(v == M_BUT9 for v in vals)
    tp = np.array([[mm * BAY5, Y_TOP9 + PIN9_H, o * Z_PIER9]
                   for mm in (1, 10) for o in (-1.0, 1.0)], np.float32)
    c, r, _ = CAM.project(_pose(tp))
    print("  wide: %d new cells; pier probes %d/%d; all four corner "
          "pinnacle tips project inside the frame" % (new, hits, hn))
    assert new > 150, new
    assert hn >= 5 and hits >= hn - 1, (hits, hn)
    assert all(0 <= rr < G.rows for rr in r), r
    assert all(0 <= cc < G.cols for cc in c), c

    # SECTION FACTS, everything up.  Multi-sample, part VI's lesson.
    draw(int(9.6 * FPS), stage)
    m = LAST["mat"]
    amid = 0.5 * (at + ah)

    def probe(pts, mat_id):
        c, r, _ = CAM_X9.project(_pose_x9(np.asarray(pts, np.float32)))
        vals = [int(m[rr, cc]) for rr, cc in zip(r, c)
                if 0 <= rr < G.rows and 0 <= cc < G.cols]
        return vals and any(v == mat_id for v in vals)

    fh, ch_, ph, pnh, tot = 0, 0, 0, 0, 0
    for mm in (8, 9, 10):
        x = mm * BAY5
        tot += 1
        fh += probe([[x + dx, cy + r_mid * math.sin(a),
                      cz + r_mid * math.cos(a)]
                     for dx in (-0.12, 0.0, 0.12)
                     for a in (amid - 0.35, amid, amid + 0.35)], M_FLY9)
        zc_ = 0.5 * (NAVE_Z + zt)
        ch_ += probe([[x + dx, y_chord(z_) + 0.5 * D9, z_]
                      for dx in (-0.12, 0.0, 0.12)
                      for z_ in (zc_ - 1.0, zc_, zc_ + 1.0)], M_COP9)
        ph += probe([[x + dx, 24.0, Z_PIER9]
                     for dx in (-0.6, 0.0, 0.6)], M_BUT9)
        pnh += probe([[x + dx, Y_TOP9 + 0.5, Z_PIER9]
                      for dx in (-0.4, 0.0, 0.4)], M_BUT9)
    print("  section probes (south flank; the north is not drawn) -- "
          "arc %d/%d, coping %d/%d, pier %d/%d, pinnacle %d/%d"
          % (fh, tot, ch_, tot, ph, tot, pnh, tot))
    assert fh >= tot - 1, (fh, tot)
    assert ch_ >= tot - 1, (ch_, tot)
    assert ph == tot, (ph, tot)
    assert pnh >= tot - 1, (pnh, tot)

    # HELD OUT: the 45-degree chord, read back off the pixels.  Fit a
    # line through the coping cells of one side of the frame and compare
    # its slope with the slope the camera says a 45-degree line should
    # project at.  Nothing about the render feeds the prediction.
    # (First draft regressed over ALL coping cells and got slope -0.33
    # against -1.09 predicted: three parallel copings pooled into one
    # fit, and the cluster centres lie along the depth axis, which is
    # nearly horizontal here.  An instrument that pools parallel lines
    # measures the line BETWEEN them.  So: one coping, the nearest.)
    rows_, cols_ = np.nonzero(m == M_COP9)
    p2 = CAM_X9.project(_pose_x9(np.array(
        [[10 * BAY5, Y_HEAD9 + 0.3, NAVE_Z],
         [10 * BAY5, Y_SPR9 + 0.3, zt]], np.float32)))
    cA, rA = p2[0].astype(float), p2[1].astype(float)
    sl_pred = (rA[1] - rA[0]) / (cA[1] - cA[0])
    b_pred = rA[0] - sl_pred * cA[0]
    near = np.abs(rows_ - (sl_pred * cols_ + b_pred)) < 2.5
    used = int(near.sum())
    sl_meas = float(np.polyfit(cols_[near], rows_[near], 1)[0])
    print("  held out: nearest coping slope %.3f rows/col over %d "
          "cells vs %.3f predicted for 45 degrees -> ratio %.2f"
          % (sl_meas, used, sl_pred, sl_meas / sl_pred))
    assert used >= 25, used
    assert 0.8 < sl_meas / sl_pred < 1.25, (sl_meas, sl_pred)

    sheet = []
    for t in (0.6, 1.4, 2.8, 4.4, 5.9, 7.9, 9.6, 10.4, 11.9):
        fr = draw(int(t * FPS), stage)
        ink, mat = LAST["ink"], LAST["mat"]
        print("  t=%4.1f u=%.2f cov %.3f  ghost %5d old %5d cler %5d "
              "but %5d fly %4d cop %4d  %s"
              % (t, LAST["u9"], ink.mean(), (mat == M_GHOST).sum(),
                 (mat == M_OLD).sum(), (mat == M_CLER).sum(),
                 (mat == M_BUT9).sum(), (mat == M_FLY9).sum(),
                 (mat == M_COP9).sum(),
                 "close" if LAST["close"] else "wide"))
        assert 0.02 < ink.mean() < 0.60, ink.mean()
        for (c0b, r0b, w, h) in LAST["boxes"]:
            assert r0b - 1 >= G.safe_top, ("text above safe", r0b)
            assert r0b + h + 1 <= G.safe_bot, ("text below safe", r0b + h)
            assert c0b - 1 >= 0 and c0b + w + 1 <= G.cols, ("width", c0b, w)
        sheet.append(fr)

    assert LAST["u9"] >= 1.0, LAST["u9"]
    print("  runtime              %.1f s, %d frames  (VIII was %.1f s)"
          % (X_END, int(X_END * FPS), W_END))
    contact(sheet, os.path.join(_HERE, "..", "content", "cath_sheet.png"),
            cols=3, labels=["0.6 ghost", "1.4 wide", "2.8 piers",
                            "4.4 piers done", "5.9 the leap",
                            "7.9 coping", "9.6 ballast", "10.4 home",
                            "11.9 outside the line"])


def check_vault(stage):
    print("THE CATHEDRAL — part %s, %s" % (roman(stage + 1), STAGES[stage]))
    print("  vault                %d bays, span %.1f m, springing %.2f "
          "(course 41 -- the line part VIII named)"
          % (N_VBAY10, 2 * S10, Y_SPR10))

    # RULE 1.  The established view has not drifted.
    d = np.abs(_pose_at(GHOST, -58.0, 28.0) - _pose(GHOST)).max()
    print("  established view unchanged: max disagreement %.2e m" % d)
    assert d < 1e-3, d

    # THE DERIVATIONS.  Every dimension is older than this episode, and
    # the arch shapes are FORCED, not styled.
    #  (ratios with ==, positions with < 1e-12 -- part VII's lesson.)
    print("  diagonal semicircle: radius = hypot(%.1f, %.4f) = %.4f -- "
          "the only round arch in the building" % (S10, SB10, RHO10))
    assert RHO10 == math.hypot(S10, SB10)
    rise_t = math.sqrt(S10 * S10 + 2.0 * S10 * QT10)
    rise_w = math.sqrt(SB10 * SB10 + 2.0 * SB10 * QW10)
    print("  every rib rises to the semicircle's crown: transverse "
          "%.6f, wall rib %.6f, semicircle %.6f (diffs %.1e, %.1e)"
          % (rise_t, rise_w, RHO10, abs(rise_t - RHO10),
             abs(rise_w - RHO10)))
    assert abs(rise_t - RHO10) < 1e-12
    assert abs(rise_w - RHO10) < 1e-12
    print("  centre offsets, symmetric closed forms: q_t = sb^2/2s = "
          "%.4f, q_w = s^2/2sb = %.4f" % (QT10, QW10))
    print("  the wall rib is a LANCET: its centres sit %.2f m outside "
          "its own %.2f m half-span" % (QW10 - SB10, SB10))
    assert QW10 > SB10
    print("  wall-rib ring depth: RHO9 * BAY5 = %.5f = RING5 %.5f "
          "(diff %.1e) -- the arcade voussoir, back at its own scale, "
          "because the wall rib's span IS the bay"
          % (D10W, RING5, abs(D10W - RING5)))
    assert abs(D10W - RING5) < 1e-15
    # an accident that ISN'T one, checked so nobody wonders: q_w lands
    # 1.4 mm from part IX's arc radius.  Not equal, no theorem.
    print("  (q_w %.4f vs part IX's R_seg %.4f: %.1f mm apart and "
          "unrelated -- not every agreement is a theorem)"
          % (QW10, R_SEG9, 1000 * abs(QW10 - R_SEG9)))
    assert abs(QW10 - R_SEG9) > 1e-4

    # THE REFUSAL.  The series' own equilateral, on this span, breaks
    # part I's roof: the frozen plane from eaves 36 to ridge 46.
    def roofpl(z):
        return 46.0 - (46.0 - NAVE_Y) / NAVE_Z * abs(z)

    viol, zv = -1e9, 0.0
    clear, zc = 1e9, 0.0
    for z in np.linspace(0.0, S10, 400):
        y_eq = Y_SPR10 + math.sqrt(max(0.0, (2 * S10) ** 2
                                       - (z + S10) ** 2))
        if y_eq - roofpl(z) > viol:
            viol, zv = y_eq - roofpl(z), z
        g = Y_SPR10 + math.sqrt(max(0.0, RT10 * RT10
                                    - (z + QT10) ** 2)) + TW10
        if roofpl(z) - g < clear:
            clear, zc = roofpl(z) - g, z
    print("  the equilateral is REFUSED: its intrados stands %.2f m "
          "through the frozen roof at z = %.1f (before any web goes "
          "on).  first loss in ten episodes" % (viol, zv))
    assert viol > 0.25, viol
    print("  the derived vault clears the roof by %.2f m at its "
          "tightest (z = %.1f)" % (clear, zc))
    assert clear > 1.0, clear
    # and the ridges are level, everywhere, by construction.
    for xi, z in ((SB10, 0.0), (0.3 * SB10, 0.0), (SB10, 0.5 * S10),
                  (SB10, 0.9 * S10)):
        assert abs(_y10(xi, z) - Y_CROWN10) < 1e-9 or z > 0 or xi != SB10
    rrr = [abs(_y10(SB10, zz) - Y_CROWN10) for zz in (0.0, 2.0, 5.0)]
    rrr += [abs(_y10(x_, 0.0) - Y_CROWN10) for x_ in (0.5, 2.0, 4.0)]
    print("  ridges level at %.3f: max deviation along both ridge "
          "lines %.1e m" % (Y_CROWN10, max(rrr)))
    assert max(rrr) < 1e-9

    # THE STATICS.  One thrust line, keystone to ground.  The vault's
    # own walk (crown to springing, weights binned off the real
    # surface) splices onto part IX's (head, flyer, pier) through an
    # equilibrium node at the wall.  Tonnes, metres, stone at 2.3.
    rho = 2.3
    K = 40
    dz10 = S10 / K

    def web_bins():
        wb_ = np.zeros(K)
        nx, nz = 60, 240
        dxi, dzz = BAY5 / nx, S10 / nz
        for i in range(nx):
            xi = (i + 0.5) * dxi
            for j in range(nz):
                z = (j + 0.5) * dzz
                y0_ = _y10(xi, z)
                yx = (_y10(xi + 0.01, z) - y0_) / 0.01
                yz = (_y10(xi, z + 0.01) - y0_) / 0.01
                dA = math.sqrt(1.0 + yx * yx + yz * yz) * dxi * dzz
                wb_[min(K - 1, int(z / dz10))] += dA * TW10 * rho
        return wb_

    phimax = math.atan2(RHO10, QT10)
    psimax = math.atan2(RHO10, QW10)

    def rib_bins():
        wb_ = np.zeros(K)
        n = 800
        w_ta = RT10 * phimax * D10T * (0.96 * W9X) * rho
        for i in range(n):
            phi = (i + 0.5) / n * phimax
            z = max(0.0, RT10 * math.cos(phi) - QT10)
            wb_[min(K - 1, int(z / dz10))] += w_ta / n
        w_dg = math.pi * RHO10 * D10D * (0.90 * W9X) * rho
        for i in range(n):
            th = (i + 0.5) / n * math.pi
            z = abs(RHO10 * math.cos(th)) * S10 / RHO10
            wb_[min(K - 1, int(z / dz10))] += w_dg / n
        w_wr = 2.0 * RW10 * psimax * (1.10 * D10W) * (1.10 * D10W) * rho
        wb_[K - 1] += w_wr
        wb_[0] += 0.5 * 1.0 * 0.9 * 1.0 * rho     # half a boss
        return wb_

    wb_web, wb_rib = web_bins(), rib_bins()
    wb10 = wb_web + wb_rib
    w_flank = float(wb10.sum())
    print("  weights, one flank of one bay: web %.1f t + ribs %.1f t "
          "= %.1f t; the whole vault %.0f t"
          % (wb_web.sum(), wb_rib.sum(), w_flank, 20.0 * w_flank))
    assert 1200.0 <= 20.0 * w_flank <= 2000.0

    def lo10(z):
        v = (RT10 - D10T) ** 2 - (z + QT10) ** 2
        return Y_SPR10 + math.sqrt(v) if v > 0 else Y_SPR10 - 0.4

    def hi10(z):
        g = RT10 * RT10 - (z + QT10) ** 2
        return Y_SPR10 + (math.sqrt(g) if g > 0 else 0.0) + TW10

    def vwalk(H, y0):
        Fz, Fy, M = H, 0.0, -y0 * H
        for k in range(K):
            z1 = (k + 1) * dz10
            zm = (k + 0.5) * dz10
            Fy -= wb10[k]
            M -= zm * wb10[k]
            y = (z1 * Fy - M) / Fz
            if not (lo10(z1) - 0.06 <= y <= hi10(z1) + 0.06):
                return None
        y_arr = (S10 * Fy - M) / Fz
        if not (Y_SPR10 - 0.5 <= y_arr <= Y_TOP8):
            return None
        return y_arr, -Fy

    # part IX's walk, verbatim in structure: the flyer and the pier.
    cz9, cy9, at9, ah9 = _arc9(1.0)
    zt9 = Z_PIER9 - PIER9_HZ
    r_in9, r_mid9 = R_SEG9 - D9, R_SEG9 - 0.5 * D9
    K9 = 40

    def y_chord9(z):
        return Y_HEAD9 - (z - NAVE_Z)

    def y_intra9(z):
        return cy9 - math.sqrt(max(R_SEG9 ** 2 - (z - cz9) ** 2, 0.0))

    aa = np.linspace(at9, ah9, 800)
    az = cz9 + r_mid9 * np.cos(aa)
    w_arc = (R_SEG9 * math.pi / 3.0) * D9 * W9X * rho
    ab = np.histogram(az, bins=K9, range=(NAVE_Z, zt9))[0] / 800.0 * w_arc
    wbn = np.zeros(K9)
    dz9 = (zt9 - NAVE_Z) / K9
    for k in range(K9):
        z = NAVE_Z + (k + 0.5) * dz9
        d2 = (z - cz9) ** 2
        yb = (cy9 - math.sqrt(r_in9 * r_in9 - d2) if r_in9 * r_in9 > d2
              else y_chord9(z))
        wbn[k] = max(0.0, y_chord9(z) - yb) * dz9 * W9X * rho
    wbin9 = ab + wbn + R9 * D9 * W9X * rho / K9
    w_pier = (2 * PIER9_HX) * (2 * PIER9_HZ) * (Y_TOP9 - Y_BASE9) * rho
    w_pin = (2 * PIER9_HX) * (2 * PIER9_HZ) * PIN9_H / 3.0 * rho

    def walk9(H, Vw, y0):
        if H < 1e-6:
            return False
        Fz, Fy = H, Vw
        M = NAVE_Z * Fy - y0 * Fz
        for k in range(K9):
            z1 = NAVE_Z + (zt9 - NAVE_Z) * (k + 1) / K9
            zm = NAVE_Z + (zt9 - NAVE_Z) * (k + 0.5) / K9
            Fy -= wbin9[k]
            M -= zm * wbin9[k]
            y = (z1 * Fy - M) / Fz
            if not (y_intra9(z1) - 0.06 <= y <= y_chord9(z1) + D9 + 0.06):
                return False
        y_arr = (zt9 * Fy - M) / Fz
        if not (Y_SPR9 - 0.5 <= y_arr <= Y_TOP9):
            return False
        Fy -= w_pin
        M -= Z_PIER9 * w_pin
        nc = K_PIER9 - N_COURSE6
        for c in range(nc):
            Fy -= w_pier / nc
            M -= Z_PIER9 * (w_pier / nc)
            yc_ = Y_TOP9 - (c + 1) * COURSE3
            z_ = (M + yc_ * Fz) / Fy
            if not (zt9 - 0.05 <= z_ <= Z_PIER9 + PIER9_HZ + 0.05):
                return False
        return True

    # the budget, recomputed today by the same walk that published it.
    def feasible9(H):
        for Vw in np.linspace(-30.0, 10.0, 81):
            for y0 in np.linspace(Y_HEAD9, Y_HEAD9 + D9, 7):
                if walk9(H, Vw, y0):
                    return True
        return False

    Hs = np.arange(0.5, 60.0, 0.25)
    H_budget = 0.0
    for h in Hs[::-1]:
        if feasible9(float(h)):
            H_budget = float(h)
            break
    print("  the budget, recomputed by part IX's own walk: %.2f t a bay"
          % H_budget)
    assert 12.0 <= H_budget <= 60.0, H_budget

    # the vault alone: the band it can push in.  a masonry arch on
    # abutments that give settles to the LOW end of its band.
    y0s = np.linspace(Y_CROWN10 - 0.70, Y_CROWN10 + TW10, 9)

    def vfeas(H):
        for y0 in y0s:
            if vwalk(H, y0) is not None:
                return True
        return False

    Hv = [float(h) for h in np.arange(2.0, 60.0, 0.25) if vfeas(float(h))]
    assert Hv, "no thrust fits the vault"
    Hv_lo, Hv_hi = Hv[0], Hv[-1]
    print("  the vault's own band: it can push %.2f to %.2f t a bay; "
          "settling on giving abutments, it ARRIVES at %.2f"
          % (Hv_lo, Hv_hi, Hv_lo))
    print("  THE CHEQUE CLEARS: %.2f arrives under the %.2f budget "
          "with %.2f t to spare -- the budget is %.0f%% spent"
          % (Hv_lo, H_budget, H_budget - Hv_lo,
             100.0 * Hv_lo / H_budget))
    assert Hv_lo <= H_budget, (Hv_lo, H_budget)
    assert Hv_lo >= 12.0, Hv_lo

    # the THREAD: one H through the whole system.  the vault line
    # arrives at the wall; an equilibrium node hands H to the flyer
    # head (the strip takes the vertical the head does not); part IX's
    # walk carries it to the ground.
    w_strip = 0.5 * BAY5 * WALL8_TH * (Y_TOP8 - Y_CAP8) * rho
    zn_c = 0.5 * (S10 + NAVE_Z)

    def node_ok(H, y_a, V_v, Vw, y_h):
        T_head = -Vw
        V_wall = V_v - T_head + w_strip
        if V_wall <= 0.0:
            return False
        M = H * (y_a - y_h) + V_v * (S10 - zn_c) - T_head * (NAVE_Z - zn_c)
        zstar = zn_c + M / V_wall
        return (S10 - 0.10) <= zstar <= (NAVE_Z + 0.10)

    def thread(H):
        for y0 in y0s:
            r = vwalk(H, y0)
            if r is None:
                continue
            y_a, V_v = r
            for Vw in np.linspace(-30.0, 10.0, 81):
                for y_h in np.linspace(Y_HEAD9, Y_HEAD9 + D9, 7):
                    if node_ok(H, y_a, V_v, Vw, y_h) and walk9(H, Vw, y_h):
                        return y0, y_a, V_v, Vw, y_h
        return None

    Ht = [float(h) for h in np.arange(2.0, 60.0, 0.25)
          if thread(float(h)) is not None]
    assert Ht, "no single thrust threads the system"
    sol = thread(Ht[0])
    print("  ONE LINE, KEYSTONE TO GROUND: the system threads at any "
          "H in [%.2f, %.2f].  at %.2f t: leaves the keystone at "
          "y %.1f, crosses the springing at %.1f, enters the flyer "
          "head at %.1f with %.1f t bearing down (30 allowed), and "
          "walks part IX's pier to the base"
          % (Ht[0], Ht[-1], Ht[0], sol[0], sol[1], sol[4], -sol[3]))
    assert Ht[-1] <= H_budget + 1e-9, (Ht[-1], H_budget)
    assert -sol[3] <= 30.0 + 1e-9, sol[3]
    print("  the springing carries %.1f t a bay down the wall; the "
          "arcade piers were sized for overhead wall in part V" % sol[2])

    # FRAME FACTS, wide, end of episode.  The first draft of this
    # check asserted the vault was INVISIBLE from outside -- interior
    # work, sealed box.  The frame refused it: 523 cells of ceiling.
    # The frame was right.  The crown stands proud of the wall top
    # (both numbers frozen or derived, nobody chose it), so the vault
    # CRESTS the walls -- a stone hill in the open attic, under the
    # ghost's roof line, visible for exactly one episode until part
    # XI's roof buries it.
    print("  the crown crests the wall top by %.2f m: Y_CROWN10 + web "
          "%.2f > NAVE_Y %.1f" % (Y_CROWN10 + TW10 - NAVE_Y,
                                  Y_CROWN10 + TW10, NAVE_Y))
    assert Y_CROWN10 + TW10 > NAVE_Y
    draw(int((Z_END - 0.2) * FPS), stage)
    m = LAST["mat"]
    new = int(((m == M_RIB10) | (m == M_WEB10)).sum())
    print("  wide, everything up: %d cells of the ceiling's BACK "
          "visible from outside -- the only episode that ever sees "
          "the building's sky from above" % new)
    assert new > 150, new
    # and the hump must sit under the ghost's roof: crown projects
    # BELOW the ghost ridge in the frame (rows grow downward).
    pr_ = CAM.project(_pose(np.array([[30.0, Y_CROWN10 + TW10, 0.0],
                                      [30.0, 46.0, 0.0]], np.float32)))
    assert pr_[1][0] > pr_[1][1], pr_
    print("  and the hill keeps the drawing's word: crown row %d sits "
          "below the ghost ridge row %d" % (pr_[1][0], pr_[1][1]))

    # the raised fit holds what its comment promised: the crown and
    # part IX's flyer both project inside the grid.
    amid9 = 0.5 * (at9 + ah9)
    chk = CAM_X10.project(_pose_x9(np.array(
        [[9.5 * BAY5, Y_CROWN10, 0.0],
         [9.0 * BAY5, cy9 + r_mid9 * math.sin(amid9),
          cz9 + r_mid9 * math.cos(amid9)]], np.float32)))
    assert all(0 <= cc < G.cols for cc in chk[0]), chk[0]
    assert all(0 <= rr < G.rows for rr in chk[1]), chk[1]
    print("  raised fit: crown and flyer both inside the grid "
          "(cols %s, rows %s)" % (list(chk[0]), list(chk[1])))

    # SECTION FACTS.  The skeleton is probed BEFORE the web seals it
    # in (t just after the bosses); the web, boss cut-face and part
    # IX's flyer -- the whole system in one frame -- after.
    def probe(m_, pts, mat_id):
        c, r, _ = CAM_X10.project(_pose_x9(np.asarray(pts, np.float32)))
        vals = [int(m_[rr, cc]) for rr, cc in zip(r, c)
                if 0 <= rr < G.rows and 0 <= cc < G.cols]
        return vals and any(v == mat_id for v in vals)

    draw(int(7.25 * FPS), stage)
    m_sk = LAST["mat"]
    ta_h, dg_h, wr_h, tot = 0, 0, 0, 0
    r_mid_t = RT10 - 0.5 * D10T
    for mm in (9, 10, 11):
        tot += 1
        ta_h += probe(m_sk, [[mm * BAY5 + dx,
                              Y_SPR10 + r_mid_t * math.sin(phi),
                              r_mid_t * math.cos(phi) - QT10]
                             for dx in (-0.12, 0.0, 0.12)
                             for phi in (0.55 * phimax, 0.75 * phimax,
                                         0.95 * phimax)], M_RIB10)
    for mm in (8, 9, 10):
        xc = (mm + 0.5) * BAY5
        r_mid_d = RHO10 - 0.5 * D10D
        dg_h += probe(m_sk, [[xc + r_mid_d * math.cos(th) * BAY5
                              / (2 * RHO10) + dx,
                              Y_SPR10 + r_mid_d * math.sin(th),
                              r_mid_d * math.cos(th) * S10 / RHO10]
                             for dx in (-0.12, 0.0, 0.12)
                             for th in (0.30 * math.pi, 0.40 * math.pi,
                                        0.60 * math.pi)], M_RIB10)
        wr_h += probe(m_sk, [[xc + dx, Y_SPR10 + (RW10 - 0.5 * D10W)
                              * math.sin(0.7 * psimax)
                              - 0 * dx, (S10 - 0.5 * D10W)]
                             for dx in (-0.3, 0.0, 0.3)], M_RIB10)
    print("  skeleton probes (before the web) -- transverse %d/%d, "
          "diagonal %d/%d, wall rib %d/%d"
          % (ta_h, tot, dg_h, tot, wr_h, tot))
    assert ta_h >= tot - 1, (ta_h, tot)
    assert dg_h >= tot - 1, (dg_h, tot)
    assert wr_h >= tot - 1, (wr_h, tot)

    draw(int(10.6 * FPS), stage)
    m_fn = LAST["mat"]
    wb_h, fl_h, tot2 = 0, 0, 0
    for mm in (8, 9, 10):
        xb = mm * BAY5
        tot2 += 1
        wb_h += probe(m_fn, [[xb + SB10 + dx,
                              _y10(SB10 + dx, zz) + 0.5 * TW10, zz]
                             for dx in (-0.4, 0.0, 0.4)
                             for zz in (0.35 * S10, 0.55 * S10,
                                        0.75 * S10)], M_WEB10)
        amid9 = 0.5 * (at9 + ah9)
        fl_h += probe(m_fn, [[xb + dx, cy9 + r_mid9 * math.sin(a),
                              cz9 + r_mid9 * math.cos(a)]
                             for dx in (-0.12, 0.0, 0.12)
                             for a in (amid9 - 0.35, amid9,
                                       amid9 + 0.35)], M_FLY9)
    print("  finished-frame probes -- web %d/%d; part IX's flyer "
          "still in frame %d/%d: the push and the thing that catches "
          "it, one picture" % (wb_h, tot2, fl_h, tot2))
    assert wb_h >= tot2 - 1, (wb_h, tot2)
    assert fl_h >= tot2 - 1, (fl_h, tot2)

    # HELD OUT: the level ridge, read off the pixels.  Predict the
    # projected line of y = crown + web along the nave axis at z = 0
    # from the camera alone; select web cells NEAR that line (one
    # line, part IX's pooling lesson); fit; compare.  Near-horizontal
    # lines make slope RATIOS unstable, so compare row error at the
    # two ends instead.
    rows_, cols_ = np.nonzero(m_fn == M_WEB10)
    p2 = CAM_X10.project(_pose_x9(np.array(
        [[8.0 * BAY5, Y_CROWN10 + TW10, 0.0],
         [11.0 * BAY5, Y_CROWN10 + TW10, 0.0]], np.float32)))
    cA, rA = p2[0].astype(float), p2[1].astype(float)
    sl_p = (rA[1] - rA[0]) / (cA[1] - cA[0])
    b_p = rA[0] - sl_p * cA[0]
    near = np.abs(rows_ - (sl_p * cols_ + b_p)) < 2.5
    used = int(near.sum())
    fit = np.polyfit(cols_[near], rows_[near], 1)
    e0 = abs((fit[0] * cA[0] + fit[1]) - rA[0])
    e1 = abs((fit[0] * cA[1] + fit[1]) - rA[1])
    print("  held out: ridge line off %d web cells -- row error %.2f "
          "and %.2f at the frame's two ends (slope %.3f vs %.3f "
          "predicted for a LEVEL ridge)" % (used, e0, e1, fit[0], sl_p))
    assert used >= 25, used
    assert e0 < 3.0 and e1 < 3.0, (e0, e1)

    sheet = []
    for t in (0.6, 1.4, 2.9, 5.0, 6.6, 7.1, 9.0, 10.6, 12.8):
        fr = draw(int(t * FPS), stage)
        ink, mat = LAST["ink"], LAST["mat"]
        print("  t=%4.1f u=%.2f cov %.3f  ghost %5d old %5d fly %4d "
              "rib %5d web %5d  %s"
              % (t, LAST["u10"], ink.mean(), (mat == M_GHOST).sum(),
                 (mat == M_OLD).sum(), (mat == M_FLY9).sum(),
                 (mat == M_RIB10).sum(), (mat == M_WEB10).sum(),
                 "close" if LAST["close"] else "wide"))
        assert 0.02 < ink.mean() < 0.60, ink.mean()
        for (c0b, r0b, w, h) in LAST["boxes"]:
            assert r0b - 1 >= G.safe_top, ("text above safe", r0b)
            assert r0b + h + 1 <= G.safe_bot, ("text below safe", r0b + h)
            assert c0b - 1 >= 0 and c0b + w + 1 <= G.cols, ("width", c0b, w)
        sheet.append(fr)

    assert LAST["u10"] >= 1.0, LAST["u10"]
    print("  runtime              %.1f s, %d frames  (IX was %.1f s)"
          % (Z_END, int(Z_END * FPS), X_END))
    contact(sheet, os.path.join(_HERE, "..", "content", "cath_sheet.png"),
            cols=3, labels=["0.6 ghost", "1.4 wide", "2.9 arches",
                            "5.0 diagonals", "6.6 wall ribs",
                            "7.1 bosses", "9.0 the web", "10.6 sealed",
                            "12.8 the stone hill"])


def check(stage):
    if stage == 9:
        return check_vault(stage)
    if stage == 8:
        return check_buttress(stage)
    if stage == 7:
        return check_clerestory(stage)
    if stage == 6:
        return check_triforium(stage)
    if stage == 5:
        return check_aisles(stage)
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
