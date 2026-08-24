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

M_GHOST, M_STONE, M_EARTH = 1, 2, 3

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

# ---------------------------------------------------------------- timeline
T_GHOST, T_HOLD, T_DIG, T_LAY, T_END = 1.5, 2.4, 3.6, 9.9, 12.4
FRAMES = int(round(T_END * FPS))
LAST = {}


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

    boxes = []
    if t > 0.8:
        a = min(1.0, (t - 0.8) / 0.7)
        g = blend(BG, GOLD, a)
        boxes.append(stamp(fr, "%s . %s" % (roman(stage + 1), STAGES[stage]),
                           8, 49, 139, g))
    LAST["boxes"] = boxes
    return fr


def blend(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def colour(v, m):
    base = {M_GHOST: GHOST_RGB, M_STONE: STONE, M_EARTH: EARTH}[int(m)]
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
def check(stage):
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
