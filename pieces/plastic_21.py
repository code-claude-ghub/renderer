#!/usr/bin/env python3
"""THE SLOW SPIRAL -- twenty-one equilateral triangles, sides 1..200.

Fibonacci's ratio is the root of x^2 = x + 1.  Its spiral is made of squares
and it grows by 1.618 a step.  Padovan's ratio is the root of x^3 = x + 1 --
the same question asked one dimension deeper -- and its spiral is made of
equilateral triangles that grow by 1.3247 a step.  After twenty steps
Fibonacci is at 6,765 and Padovan is still at 200.

The tiling is not decorative and it is not approximate: each new triangle
sits exactly on an edge of the existing outline, because the sequence obeys
P(n) = P(n-1) + P(n-5), so the last triangle's outer side and a piece left
exposed five steps ago are collinear and add up.  build() asserts that at
every step -- if the identity failed the attach edge would be the wrong
length and the render would stop.

Asked for by @Dominic-qv3yt.  The number is older than the sequence: Gerard
Cordonnier found it in 1924 at seventeen and called it the radiant number,
and Dom Hans van der Laan derived it again in 1928 and built an abbey out
of it.

Render:  python3 plastic_21.py            (writes out/plastic_21.mp4)
Check:   python3 plastic_21.py --check    (numbers only, no frames)
Sheet:   python3 plastic_21.py --sheet    (contact sheet of eight stills)
"""

import math
import os
import sys

import cairo
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# asciilib sits beside this file in the working tree and one level up in
# the public repo, where pieces live in pieces/. Insert both.
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]

from asciilib import (Camera, Encoder, Frame, Grid, contact, depth_cue,
                      ink_lut, lambert, rot, specular, visible, zbuffer)

# ---------------------------------------------------------------- numbers

N_TILES = 21                       # sides 1 .. 200
RHO = 1.3247179572447458           # real root of x^3 = x + 1
PHI = 1.6180339887498949           # real root of x^2 = x + 1

FPS = 30
HOLD_A = 24                        # the first unit triangle, alone
STEP = 20                          # frames per placement
RISE = 9                           # of those, spent rising
HOLD_B = 76                        # sit on 200
FRAMES = HOLD_A + (N_TILES - 1) * STEP + HOLD_B      # 500 -> 16.67 s

# ---------------------------------------------------------------- the form

TH = 0.40                          # slab thickness as a fraction of its side
EDGE_IN = 0.014                    # seam inset, fraction of side
EDGE_UP = 0.006                    # seam lifted proud (trap 8)
PITCH = 0.63                       # ~36 deg: enough tilt to see the slabs
FILL = 1.18                        # fill the width, bleed a little
AGE = 3.0                          # steps a tile stays warm after it lands

LAMP = np.array([-0.52, -0.66, 0.56])       # y is DOWN: upper-left, front

BG = (0.031, 0.071, 0.059)         # deep pine, almost black
CHALK = (0.933, 0.906, 0.824)      # warm limestone
OCHRE = (0.918, 0.549, 0.153)      # the tile being laid

N_TOP, N_WALL, N_EDGE = 20000, 4200, 4000

TXT_FRAC = 0.20                    # digit height as a fraction of tile side
TXT_MIN, TXT_MAX = 7, 30

RNG = np.random.default_rng(19280000 + 1324)
H3 = math.sqrt(3.0) / 2.0

G = Grid()
RAMP = ink_lut()
OUT = os.path.join(os.getcwd(), "out")


# ---------------------------------------------------------------- tiling

def padovan(n):
    p = [1, 1, 1]
    while len(p) < n:
        p.append(p[-2] + p[-3])
    return p[:n]


def _dir(p, q):
    dx, dy = q[0] - p[0], q[1] - p[1]
    L = math.hypot(dx, dy)
    return (dx / L, dy / L)


def tiling(P):
    """Grow the spiral. Returns the triangles in placement order.

    The outline is kept as a CCW polygon; interior is to the LEFT of every
    directed edge, so a triangle attached to the right of A->B lands
    outside.  After placing one, the next attach edge starts at the new
    apex and swallows however many following outline edges are collinear
    with it -- which is exactly where P(n) = P(n-1) + P(n-5) shows up as
    geometry.  The assert is the proof.
    """
    poly = [(0.0, 0.0), (1.0, 0.0), (0.5, H3)]           # CCW
    tris = [tuple(poly)]
    i, j = 0, 1
    for n in range(1, len(P)):
        poly = poly[i:] + poly[:i]                        # rotate so i == 0
        j, i = (j - i) % len(poly), 0
        A, B = poly[0], poly[j]
        L = math.hypot(B[0] - A[0], B[1] - A[1])
        assert abs(L - P[n]) < 1e-7, ("attach edge %.6f != P(%d)=%d"
                                      % (L, n, P[n]))
        vx, vy = B[0] - A[0], B[1] - A[1]                 # rotate -60 = out
        C = (A[0] + vx * 0.5 + vy * H3, A[1] - vx * H3 + vy * 0.5)
        tris.append((A, B, C))
        new = [A, C] + poly[j:]
        base, e = _dir(new[1], new[2]), 2
        while e + 1 < len(new):
            d = _dir(new[e], new[e + 1])
            if abs(d[0] - base[0]) < 1e-9 and abs(d[1] - base[1]) < 1e-9:
                e += 1
            else:
                break
        poly, i, j = new, 1, e
    return tris


# ---------------------------------------------------------------- geometry

def _bary(n):
    """n uniform points inside the unit triangle, as barycentric weights."""
    u = RNG.random(n)
    v = RNG.random(n)
    flip = u + v > 1.0
    u = np.where(flip, 1.0 - u, u)
    v = np.where(flip, 1.0 - v, v)
    return np.stack([1.0 - u - v, u, v], -1)


def slab(tri, side):
    """One triangular slab: top face, three walls, and a seam on the rim.

    Built with +y up and +z out of the tiling plane. The flip to screen
    coordinates happens once, at the end of build().
    """
    v = np.array([[tri[0][0], tri[0][1]],
                  [tri[1][0], tri[1][1]],
                  [tri[2][0], tri[2][1]]])
    t = TH * side
    pts, nrm, mat = [], [], []

    w = _bary(N_TOP)
    xy = w @ v
    z = np.full(N_TOP, t)
    pts.append(np.stack([xy[:, 0], xy[:, 1], z], -1))
    nrm.append(np.tile([0.0, 0.0, 1.0], (N_TOP, 1)))
    mat.append(np.zeros(N_TOP))

    cen = v.mean(0)
    for a in range(3):
        p0, p1 = v[a], v[(a + 1) % 3]
        s = RNG.random(N_WALL)[:, None]
        h = RNG.random(N_WALL) * t
        base = p0 + (p1 - p0) * s
        pts.append(np.stack([base[:, 0], base[:, 1], h], -1))
        e = p1 - p0
        nn = np.array([e[1], -e[0]]) / (np.hypot(*e) + 1e-12)
        if np.dot(nn, cen - (p0 + p1) / 2.0) > 0:         # point it outward
            nn = -nn
        nrm.append(np.tile([nn[0], nn[1], 0.0], (N_WALL, 1)))
        mat.append(np.zeros(N_WALL))

    # the seam. A drawn line at a fixed shade (trap 9) so it survives on
    # the lit face and the dark one alike; without it two neighbouring
    # tiles of equal side merge into one shape.
    per = N_EDGE // 3
    for a in range(3):
        p0 = v[a] + (cen - v[a]) * (EDGE_IN * 3.0)
        p1 = v[(a + 1) % 3] + (cen - v[(a + 1) % 3]) * (EDGE_IN * 3.0)
        s = RNG.random(per)[:, None]
        base = p0 + (p1 - p0) * s
        pts.append(np.stack([base[:, 0], base[:, 1],
                             np.full(per, t + EDGE_UP * side)], -1))
        nrm.append(np.tile([0.0, 0.0, 1.0], (per, 1)))
        mat.append(np.full(per, 2.0))

    return (np.concatenate(pts), np.concatenate(nrm), np.concatenate(mat))


def build(P, tris):
    """All slabs, concatenated, plus the index ranges of each one.

    Jittered once off a fixed seed (trap 10) -- every surface here is a
    plane, and planes sampled on a lattice beat against the character grid.
    """
    pts, nrm, mat, span, tops, who, tint = [], [], [], [], [], [], []
    k = 0
    for i, tri in enumerate(tris):
        p, n, m = slab(tri, P[i])
        p[:, :2] += RNG.normal(0.0, 0.0022 * P[i], (len(p), 2))
        pts.append(p)
        nrm.append(n)
        mat.append(m)
        who.append(np.full(len(p), float(i)))
        # every top face shares one normal, so two neighbours of the same
        # side land on the same glyph and merge into one shape. A fixed
        # per-tile nudge of a few percent separates them like laid stone.
        tint.append(np.full(len(p), 0.93 + 0.14 * ((i * 7) % 5) / 4.0))
        span.append((k, k + len(p)))
        k += len(p)
        tops.append(np.array([[tri[a][0], tri[a][1], zz]
                              for a in range(3) for zz in (0.0, TH * P[i])]))
    P3 = np.concatenate(pts)
    N3 = np.concatenate(nrm)
    # trap 1: negative y is screen-UP, and the tiling is drawn with +y up.
    P3 = np.stack([P3[:, 0], -P3[:, 1], P3[:, 2]], -1)
    N3 = np.stack([N3[:, 0], -N3[:, 1], N3[:, 2]], -1)
    tops = [np.stack([h[:, 0], -h[:, 1], h[:, 2]], -1) for h in tops]
    return (P3, N3, np.concatenate(mat), span, tops,
            np.concatenate(who), np.concatenate(tint))


# ---------------------------------------------------------------- camera

def framing(tops):
    """One (centre, scale) per step, from the slab corners only.

    The point cloud lives inside the convex hull of these corners, so a box
    around them is a box around every sample -- and it costs 126 points
    instead of two million.
    """
    out = []
    for k in range(len(tops)):
        h = np.concatenate(tops[:k + 1])
        n = np.zeros_like(h)
        h, _ = rot(h, n, PITCH, 0.0, 0.0)
        x0, x1 = h[:, 0].min(), h[:, 0].max()
        y0, y1 = h[:, 1].min(), h[:, 1].max()
        hw, hh = (x1 - x0) / 2.0, (y1 - y0) / 2.0
        s = min((G.cx - 1.0) / (hw + 1e-9), 80.0 / (hh + 1e-9)) * FILL
        out.append((np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0]), s))
    return out


def phase(f):
    """(step index, fraction through the placement) for a frame number."""
    if f < HOLD_A:
        return 0, 0.0
    g = f - HOLD_A
    k = 1 + g // STEP
    if k >= N_TILES:
        return N_TILES - 1, 1.0
    return k, (g % STEP) / float(STEP)


def _smooth(t):
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------- big type

def raster(text, cw, ch, ss=8):
    """Words have to be built out of cells (trap 11): draw at 8x into an
    alpha surface, area-average onto the grid, hand back coverage."""
    W, H = cw * ss, ch * ss
    surf = cairo.ImageSurface(cairo.FORMAT_A8, W, H)
    c = cairo.Context(surf)
    c.select_font_face("monospace", cairo.FONT_SLANT_NORMAL,
                       cairo.FONT_WEIGHT_BOLD)
    size = float(H)
    for _ in range(60):                       # fit loop, never overflow
        c.set_font_size(size)
        e = c.text_extents(text)
        if e.width <= W * 0.94 and e.height <= H * 0.90:
            break
        size *= 0.94
    c.set_font_size(size)
    e = c.text_extents(text)
    c.move_to((W - e.width) / 2.0 - e.x_bearing,
              (H - e.height) / 2.0 - e.y_bearing)
    c.show_text(text)
    surf.flush()
    stride = surf.get_stride()
    buf = np.frombuffer(surf.get_data(), np.uint8).reshape(H, stride)[:, :W]
    return buf.reshape(ch, ss, cw, ss).mean(axis=(1, 3)) / 255.0


_TXT = {}


def digits(n, ch):
    s = str(n)
    cw = max(4, int(round(ch * 0.72 * len(s))))
    key = (s, ch)
    if key not in _TXT:
        _TXT[key] = raster(s, cw, ch) > 0.34
    return _TXT[key]


# ---------------------------------------------------------------- shading

def colour(shade, extra):
    """extra is how freshly laid the tile is, 1 -> 0 over AGE steps.

    A hard on/off would mark only the newest tile and the picture is very
    nearly self-similar, so a single ochre patch jumping about is the only
    motion there is. Fading it back over three steps leaves a warm tail
    that shows which way the spiral winds.
    """
    e = 0.0 if extra < 0.0 else (1.0 if extra > 1.0 else extra)
    ink = (CHALK[0] + (OCHRE[0] - CHALK[0]) * e,
           CHALK[1] + (OCHRE[1] - CHALK[1]) * e,
           CHALK[2] + (OCHRE[2] - CHALK[2]) * e)
    a = 0.46 + 0.54 * shade                   # tint with a floor (trap 12)
    return (BG[0] + (ink[0] - BG[0]) * a,
            BG[1] + (ink[1] - BG[1]) * a,
            BG[2] + (ink[2] - BG[2]) * a)


P = padovan(N_TILES)
TRIS = tiling(P)
PTS, NRM, MAT, SPAN, TOPS, WHO, TINT = build(P, TRIS)
FRAME = framing(TOPS)
# the centroid has to travel through exactly the rotation the samples do,
# or the count gets stamped where the tile used to be. It did, first try.
_C = np.array([[sum(t[a][0] for a in range(3)) / 3.0,
                -sum(t[a][1] for a in range(3)) / 3.0,
                TH * P[i]] for i, t in enumerate(TRIS)])
CAM = Camera(G)


def _live(k, grow):
    """Index of every sample to draw this frame.

    Old tiles are prefix-subsampled: a tile five steps back is a fifth the
    size on screen and needs a twenty-fifth of the points. The samples are
    already in random order, so a prefix is a fair subsample.
    """
    keep = []
    for i in range(k + 1):
        a, b = SPAN[i]
        f = min(1.0, 4.0 * (P[i] / float(P[k])) ** 2)
        keep.append(np.arange(a, a + int((b - a) * f)))
    return np.concatenate(keep)


def draw(f, want_pts=False):
    k, u = phase(f)
    idx = _live(k, u)
    p = PTS[idx].copy()
    n = NRM[idx]
    m = MAT[idx]
    age = (k + u) - WHO[idx]

    # the new slab rises into place rather than popping
    r = 1.0
    if k > 0 and u < RISE / float(STEP):
        a, b = SPAN[k]
        hit = (idx >= a) & (idx < b)
        r = _smooth(min(1.0, u * STEP / float(RISE)))
        p[hit, 2] *= r

    p, n = rot(p, n, PITCH, 0.0, 0.0)
    cpt = _C[k].copy()
    cpt[2] *= r
    cpt, _ = rot(cpt[None, :], np.zeros((1, 3)), PITCH, 0.0, 0.0)

    c0, s0 = FRAME[k]
    c1, s1 = FRAME[min(k + 1, N_TILES - 1)]
    t = _smooth(u) if k + 1 < N_TILES else 1.0
    CAM.off = c0 + (c1 - c0) * t
    CAM.scale = math.exp(math.log(s0) + (math.log(s1) - math.log(s0)) * t)

    col, row, z = CAM.project(p)
    ok = visible(G, col, row)
    col, row, z, n, m = col[ok], row[ok], z[ok], n[ok], m[ok]
    age, tnt = age[ok], TINT[idx][ok]
    new = np.zeros(len(idx), bool)
    a, b = SPAN[k]
    new[(idx >= a) & (idx < b)] = True
    new = new[ok]
    _, keep = zbuffer(G, col, row, z)

    lam = lambert(n, LAMP)
    spc = specular(n, LAMP, 26)
    cue = depth_cue(z, near=1.0, far=0.90)
    # a 0.20 floor put the unlit walls down where the ramp is single dots,
    # and the big shadowed face under the first slab read as dirt, not as
    # a surface. 0.30 is the point at which a dark wall is still a wall.
    body = (0.30 + 0.72 * lam + 0.22 * spc) * cue * tnt
    seam = np.full(len(z), 0.06)               # fixed-shade drawn line
    shade = np.where(m > 1.5, seam, body)

    # the count is cut INTO the slab being laid, not written over the top
    ch = int(round(np.clip(TXT_FRAC * P[k] * CAM.scale, TXT_MIN, TXT_MAX)))
    msk = digits(P[k], ch)
    cc, rr, _ = CAM.project(cpt)
    r0 = int(rr[0]) - msk.shape[0] // 2
    c0i = int(cc[0]) - msk.shape[1] // 2
    inb = ((row >= r0) & (row < r0 + msk.shape[0])
           & (col >= c0i) & (col < c0i + msk.shape[1]) & new)
    if inb.any():
        hit = msk[row[inb] - r0, col[inb] - c0i]
        sel = np.flatnonzero(inb)[hit]
        shade[sel] = 0.0

    fr = Frame(G, BG)
    extra = np.where(m > 1.5, 0.0, np.clip(1.0 - age / AGE, 0.0, 1.0))
    fr.field(col, row, keep, np.clip(shade, 0.0, 1.0), colour, RAMP, extra)
    if want_pts:
        return fr, col[keep], row[keep], (r0, c0i, msk.shape, ch)
    return fr


# ---------------------------------------------------------------- checks

def check():
    print(G)
    print("padovan  :", P)
    fib = [1, 1]
    while len(fib) < N_TILES:
        fib.append(fib[-1] + fib[-2])
    print("fibonacci:", fib)
    print("ratios   : rho %.10f  phi %.10f" % (RHO, PHI))
    print("last ratio %.8f -> rho, err %.2e"
          % (P[-1] / P[-2], abs(P[-1] / P[-2] - RHO)))
    print("rho^3 - rho - 1 = %.2e" % (RHO ** 3 - RHO - 1.0))
    print("after %d steps: fib %d, padovan %d  (%.1fx apart)"
          % (N_TILES - 1, fib[-1], P[-1], fib[-1] / float(P[-1])))
    area = sum(math.sqrt(3) / 4.0 * s * s for s in P)
    print("tiles %d  total area %.1f  samples %d" % (len(P), area, len(PTS)))
    print("frames %d = %.2f s at %d fps" % (FRAMES, FRAMES / float(FPS), FPS))

    for f in (0, HOLD_A, HOLD_A + 5 * STEP, HOLD_A + 12 * STEP,
              FRAMES - HOLD_B, FRAMES - 1):
        fr, c, r, (r0, c0i, sh, ch) = draw(f, want_pts=True)
        k, u = phase(f)
        print("f%-4d tile %-2d side %-4d cells %6d  col %3d..%-3d "
              "row %3d..%-3d  digits %dx%d at r%d c%d"
              % (f, k, P[k], len(c), c.min(), c.max(), r.min(), r.max(),
                 sh[0], sh[1], r0, c0i))
        assert len(c) > 2500, "frame %d is too thin: %d cells" % (f, len(c))
        assert r0 >= G.safe_top and r0 + sh[0] <= G.safe_bot, \
            "digits leave the safe band at frame %d (rows %d..%d, safe %d..%d)" \
            % (f, r0, r0 + sh[0], G.safe_top, G.safe_bot)
        assert c0i >= 0 and c0i + sh[1] <= G.cols, \
            "digits leave the frame at %d" % f
    print("checks pass")


def sheet():
    os.makedirs(OUT, exist_ok=True)
    if "--motion" in sys.argv:
        # eight frames inside two placements: does anything actually move?
        b = HOLD_A + 13 * STEP
        marks = [b + i * 5 for i in range(8)]
    else:
        marks = [0, HOLD_A + 2 * STEP, HOLD_A + 5 * STEP, HOLD_A + 8 * STEP,
                 HOLD_A + 11 * STEP, HOLD_A + 14 * STEP, HOLD_A + 17 * STEP,
                 FRAMES - 1]
    frames = [draw(f) for f in marks]
    path = os.path.join(OUT, "plastic_21_sheet.png")
    contact(frames, path, cols=4, width=1560,
            labels=["f%d s%d" % (f, P[phase(f)[0]]) for f in marks])
    print("wrote", path)


def render():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "plastic_21.mp4")
    with Encoder(path, G, fps=FPS, crf=17, preset="medium") as enc:
        for f in range(FRAMES):
            enc.write(draw(f))
            if f % 50 == 0:
                print("  %d/%d" % (f, FRAMES), flush=True)
    print("wrote", path)


if __name__ == "__main__":
    check()
    if "--sheet" in sys.argv:
        sheet()
    elif "--check" not in sys.argv:
        render()
