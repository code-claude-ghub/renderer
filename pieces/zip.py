#!/usr/bin/env python3
"""ZIP -- a zip fastener at macro scale, closing, and then opening.

The slider never moves. The camera is locked to it and the chain flows THROUGH
it: down, and the teeth mesh; then up, and they come apart. Nothing else in the
frame changes, because nothing else CAN. A zip slider is a solid lump of metal
with a Y-shaped channel cut through it and no moving parts whatsoever. Which
way you pull is the entire mechanism.

So the second half of this video is the first half reversed, frame for frame,
and `--check` proves it by rendering both halves independently and comparing
every pixel. That is not an editing trick. A zip is a reversible machine, so a
picture of one closing IS a picture of one opening, run backwards.

Verified (en.wikipedia.org/wiki/Zipper, en.wikipedia.org/wiki/Gideon_Sundback):
  - "The slider, usually operated by hand, contains a Y-shaped channel that, by
    moving along the rows of teeth, meshes or separates them, depending on the
    direction of the slider's movement."
  - the teeth are also called ELEMENTS, and may be discrete or cut from a coil
  - Sundback's machine cut scoops from a special Y-shaped wire, punched the
    DIMPLE and the NIB into each one, and clamped it to the tape -- so the
    bump on top of every tooth and the hollow underneath are the whole joint
  - when a zip fails it is usually the slider, worn until it no longer aligns
    and joins the teeth. The teeth are almost never the problem.

Invented: this particular zip, the cloth, the light.
Real: the mechanism, and the fact that it runs equally well both ways.

Not ASCII. Brushed metal is a continuous value across a curved surface, which
is exactly what a ten-level glyph ramp turns into flat banded patches.

    python3 scripts/zip.py --check      structure + palindrome, renders little
    python3 scripts/zip.py --stills     full-resolution PNGs at key moments
    python3 scripts/zip.py --out x.mp4  render
"""

import argparse
import math
import os
import subprocess
import sys

import numpy as np
from scipy.ndimage import binary_closing, gaussian_filter, uniform_filter

# ------------------------------------------------------------------ output
W, H = 1080, 1920
FPS = 30
N_FRAME = 216                      # 7.2 s
HALF_F = N_FRAME // 2              # the turn

# ---------------------------------------------------------- the zip, in mm
# Everything below is millimetres. A #5 metal zip is about this size.
PITCH = 5.00                       # tooth spacing along ONE tape
STEP = PITCH / 2.0                 # spacing along the CLOSED chain, alternating
GAP_X = 1.95                       # half-distance between the two tape edges
HEAD_IN = 3.90                     # how far a tooth reaches in from its edge
TIP_X = HEAD_IN - GAP_X            # 1.95 -- how far it crosses the centreline

SPREAD_DEG = 13.5                  # half-angle the tapes open at
SPREAD = math.tan(math.radians(SPREAD_DEG))

SL_TOP, SL_BOT = 6.6, -6.6         # the slider body, along the chain
SL_W_TOP, SL_W_BOT = 7.4, 3.9      # half-width at each end -- it tapers
PLATE_IN, PLATE_OUT = 2.10, 2.90   # top plate sits between these in z
TRAVEL = 45.0                      # mm of chain pulled through, one way

Y_LO, Y_HI = -34.0, 40.0           # the run of chain that gets built

# ------------------------------------------------------------- the camera
EYE = np.array([0.0, -21.0, 47.0])
TARGET = np.array([0.0, 8.5, 0.0])
FIELD_MM = 52.0                    # vertical field at the target's depth
FOCUS_AT = np.array([0.0, -3.0, 1.4])  # just under the slider, where it joins
COC = 0.29                         # blur px per mm of defocus at the plate
COC_MAX = 11.0

# ----------------------------------------------------------------- lights
KEY_P = np.array([-27.0, 17.0, 33.0])
KEY_C = np.array([1.00, 0.94, 0.86]) * 395.0
FILL_P = np.array([23.0, -9.0, 26.0])
FILL_C = np.array([0.62, 0.72, 1.00]) * 155.0
AMBIENT = np.array([0.030, 0.035, 0.048])

Z_NEAR, Z_SPAN = 20.0, 90.0        # depth quantisation window, mm
Z_BIAS = 0.35                      # mm a later part must win by

METAL_RGB = np.array([0.300, 0.315, 0.345])
TAB_RGB = np.array([0.268, 0.280, 0.310])
CLOTH_RGB = np.array([0.031, 0.036, 0.052])


# ================================================================ geometry
def sq(n, half, e, rng, over=6):
    """n points spread EVENLY OVER THE AREA of a superquadric, with normals.
    e=2 is an ellipsoid, e>=5 is nearly a box.

    Casting uniform random directions and keeping where they pierce the
    surface is the obvious way to do this and it is wrong, because equal
    solid angle is not equal area.  On a 14.8 x 13.2 x 1.1 mm plate the rim
    and its shoulder are 17% of the surface and catch 7% of the directions,
    so the samples there run thin, the gaps CLUSTER instead of scattering,
    and the render grows a stipple round every edge that no amount of extra
    density, splat size or hole filling will touch -- all of which were tried
    first.

    For any surface written as a radius along a direction, the area element
    is r^2 / cos(angle between the direction and the normal) per unit solid
    angle.  So oversample directions, weight by exactly that, and resample."""
    m = int(n * over)
    d = rng.normal(size=(m, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    a = np.asarray(half, float)
    t = np.sum(np.abs(d / a) ** e, axis=1) ** (-1.0 / e)
    p = d * t[:, None]
    g = (e / a) * np.abs(p / a) ** (e - 1.0) * np.sign(p)
    g /= np.linalg.norm(g, axis=1, keepdims=True)

    cosang = np.abs(np.sum(d * g, axis=1))
    w = t * t / np.maximum(cosang, 1e-3)
    c = np.cumsum(w)
    idx = np.searchsorted(c, rng.random(n) * c[-1])
    idx = np.clip(idx, 0, m - 1)
    return p[idx].astype(np.float32), g[idx].astype(np.float32)


def taper_x(p, n, v0, v1, s0, s1):
    """Squeeze x by a factor that ramps with y from s0 at v0 to s1 at v1.
    Normals get the inverse transpose, or the slider's sides shade like a box
    it is not."""
    u = np.clip((p[:, 1] - v0) / (v1 - v0), 0.0, 1.0)
    s = s0 + (s1 - s0) * u
    inside = (p[:, 1] > v0) & (p[:, 1] < v1)
    ds = np.where(inside, (s1 - s0) / (v1 - v0), 0.0)
    q = p.copy()
    q[:, 0] = p[:, 0] * s
    m = n.copy()
    m[:, 0] = n[:, 0] / s
    m[:, 1] = n[:, 1] - p[:, 0] * ds / s * n[:, 0]
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    return q, m


def build_tooth(dens=1.0):
    """One tooth (a 'scoop') in local coords: u runs INWARD from the tape edge,
    v along the chain, w is thickness.  Origin at the tape edge."""
    rng = np.random.default_rng(7)
    parts = []

    def add(n, half, e, at):
        p, g = sq(int(n * dens), half, e, rng)
        p = p + np.asarray(at, np.float32)
        parts.append((p, g))

    add(7500, (0.98, 1.34, 0.78), 4.6, (-0.95, 0.0, -0.16))   # clamp, on the tape
    add(4200, (0.92, 0.86, 0.66), 4.0, (0.98, 0.0, 0.0))      # the waist
    add(19000, (1.28, 1.12, 0.94), 4.2, (2.62, 0.0, 0.0))     # head
    add(4000, (0.60, 0.60, 0.44), 2.0, (2.72, 0.0, 0.76))     # the nib

    p = np.vstack([q for q, _ in parts])
    g = np.vstack([m for _, m in parts])
    return p, g


TOOTH_P, TOOTH_N = build_tooth()


def build_slider(dens=1.0):
    rng = np.random.default_rng(11)
    parts = []

    def add(n, half, e, at, tap=None):
        p, g = sq(int(n * dens), half, e, rng)
        if tap is not None:
            p, g = taper_x(p, g, *tap)
        p = p + np.asarray(at, np.float32)
        parts.append((p, g))

    hz = (PLATE_OUT - PLATE_IN) / 2.0
    mz = (PLATE_OUT + PLATE_IN) / 2.0
    # the underside of the top plate.  NOTHING inside the slider may reach
    # it: two surfaces sharing a depth stipple against each other, and
    # that stipple is indistinguishable from sampling holes.
    inner = mz - hz * 1.15 - 0.05
    tap = (SL_BOT, SL_TOP, SL_W_BOT / SL_W_TOP, 1.0)

    add(300000, (SL_W_TOP, 6.6, hz * 1.15), 6.4, (0.0, 0.0, mz), tap)  # top plate
    add(110000, (SL_W_TOP, 6.6, hz * 1.15), 6.4, (0.0, 0.0, -mz), tap) # bottom
    # the two side flanges, which are what actually steers the teeth
    for sgn in (-1.0, 1.0):
        add(42000, (0.62, 6.6, inner), 5.0,
            (sgn * (SL_W_BOT - 0.62), 0.0, 0.0),
            (SL_BOT, SL_TOP, 1.0, SL_W_TOP / SL_W_BOT))
    # the wedge at the fork of the Y.  Splits one channel into two.
    add(26000, (1.28, 3.15, inner), 3.4, (0.0, 4.5, 0.0))
    add(26000, (1.35, 1.05, 0.62), 3.0, (0.0, -6.4, mz))            # the lug

    p = np.vstack([q for q, _ in parts])
    g = np.vstack([m for _, m in parts])
    return p, g


SLIDER_P, SLIDER_N = build_slider()


TAB_SWING = math.radians(76.0)


def build_tab(dens=1.0):
    """The pull tab.  Hangs off the lug, then swung aside about that lug --
    which is what a real one does, and which keeps it off the closed chain."""
    rng = np.random.default_rng(13)
    p, g = sq(int(88000 * dens), (2.95, 4.9, 0.40), 4.2, rng)
    p[:, 1] -= 10.9
    p2, g2 = sq(int(13000 * dens), (0.92, 1.35, 0.40), 3.0, rng)
    p2[:, 1] -= 6.15
    p = np.vstack([p, p2])
    g = np.vstack([g, g2])

    pv = np.array([0.0, -5.5, 0.0], np.float32)
    q = p - pv
    c, sn = math.cos(TAB_SWING), math.sin(TAB_SWING)
    qx = q[:, 0] * c - q[:, 1] * sn
    qy = q[:, 0] * sn + q[:, 1] * c
    gx = g[:, 0] * c - g[:, 1] * sn
    gy = g[:, 0] * sn + g[:, 1] * c
    out = np.stack([qx, qy, q[:, 2]], 1) + pv
    out[:, 2] += 1.62
    return (out.astype(np.float32),
            np.stack([gx, gy, g[:, 2]], 1).astype(np.float32))


TAB_P, TAB_N = build_tab()


# --------------------------------------------------------- where teeth are
def edge_at(y):
    """Half-gap between the tape edges at height y.  GAP_X where the chain is
    closed, opening at a constant angle above the slider."""
    y = np.asarray(y, float)
    over = np.clip(y - SL_TOP, 0.0, None)
    return GAP_X + SPREAD * over


def yaw_at(y):
    """How far a tooth has been swung by the channel, radians.  Zero below the
    slider, the full tape angle above it.  The turn happens INSIDE the slider,
    which is exactly where you cannot see it."""
    y = np.asarray(y, float)
    u = np.clip((y - SL_BOT) / (SL_TOP - SL_BOT), 0.0, 1.0)
    return math.radians(SPREAD_DEG) * (u * u * (3.0 - 2.0 * u))


def tooth_indices(scroll):
    """Every tooth whose centre is in the built run, given the chain offset."""
    k0 = int(math.floor((Y_LO - scroll) / STEP)) - 1
    k1 = int(math.ceil((Y_HI - scroll) / STEP)) + 1
    k = np.arange(k0, k1 + 1)
    return k, k * STEP + scroll


def chain(scroll):
    """The whole tooth chain for one frame: points, normals."""
    k, y = tooth_indices(scroll)
    side = np.where(k % 2 == 0, -1.0, 1.0)          # -1 left tape, +1 right
    ex = edge_at(y)
    yw = yaw_at(y) * side                            # swing away from centre

    P, N = TOOTH_P, TOOTH_N
    m = P.shape[0]
    n = k.size

    u = P[None, :, 0] * side[:, None]                # inward becomes +x or -x
    v = np.broadcast_to(P[None, :, 1], (n, m))
    w = np.broadcast_to(P[None, :, 2], (n, m))
    nu = N[None, :, 0] * side[:, None]
    nv = np.broadcast_to(N[None, :, 1], (n, m))
    nw = np.broadcast_to(N[None, :, 2], (n, m))

    c, s = np.cos(yw)[:, None], np.sin(yw)[:, None]
    x = u * c - v * s
    yy = u * s + v * c
    nx = nu * c - nv * s
    ny = nu * s + nv * c

    x = x + (-side * ex)[:, None]                    # sit on the tape edge
    yy = yy + y[:, None]

    p = np.stack([x, yy, np.broadcast_to(w, x.shape)], -1).reshape(-1, 3)
    g = np.stack([nx, ny, np.broadcast_to(nw, x.shape)], -1).reshape(-1, 3)
    return p.astype(np.float32), g.astype(np.float32)


# ------------------------------------------------------------- projection
def _basis():
    fwd = TARGET - EYE
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return right, up, fwd


RIGHT, UP, FWD = _basis()
DIST_T = float(np.linalg.norm(TARGET - EYE))
FOCAL = H * DIST_T / FIELD_MM


def project(p):
    """world mm -> (px, py, depth).  depth is distance along the view axis."""
    d = p - EYE.astype(np.float32)
    vx = d @ RIGHT.astype(np.float32)
    vy = d @ UP.astype(np.float32)
    vz = d @ FWD.astype(np.float32)
    vz = np.maximum(vz, 1e-3)
    px = W * 0.5 + FOCAL * vx / vz
    py = H * 0.5 - FOCAL * vy / vz
    return px, py, vz


DEPTH_FOCUS = float((FOCUS_AT - EYE) @ FWD)


# --------------------------------------------------------------- materials
def shade(p, n, base, spec_k, rough):
    """One point light plus one fill, both with inverse-square falloff, plus a
    cheap environment term.  Metal without an environment reads as plastic."""
    view = EYE.astype(np.float32) - p
    view /= np.linalg.norm(view, axis=1, keepdims=True)
    out = np.repeat(AMBIENT.astype(np.float32)[None, :], p.shape[0], 0) * base

    for lp, lc in ((KEY_P, KEY_C), (FILL_P, FILL_C)):
        ld = lp.astype(np.float32) - p
        r2 = np.sum(ld * ld, axis=1, keepdims=True)
        ld /= np.sqrt(r2)
        lam = np.clip(np.sum(ld * n, axis=1, keepdims=True), 0.0, None)
        hv = ld + view
        hv /= np.maximum(np.linalg.norm(hv, axis=1, keepdims=True), 1e-6)
        sp = np.clip(np.sum(hv * n, axis=1, keepdims=True), 0.0, None) ** rough
        e = lc.astype(np.float32)[None, :] / r2
        out += e * (base * lam * 0.58 + spec_k * sp)

    # environment: a bright strip overhead, dark below.  Faked off the
    # reflected direction, which is all a curved metal surface really shows.
    refl = 2.0 * np.sum(view * n, axis=1, keepdims=True) * n - view
    t = np.clip(refl[:, 2:3] * 0.5 + 0.5, 0.0, 1.0)
    env = (0.020 + 0.55 * t ** 2.6) * spec_k * 1.22
    out += env * np.array([0.93, 0.96, 1.00], np.float32)[None, :]
    return out


def cloth_texture(x, y):
    """A woven tape.  Two thread directions at slightly different pitch, so it
    never quite lines up, plus enough noise to stop it being a screen door."""
    a = np.sin(x * (2.0 * math.pi / 1.15))
    b = np.sin(y * (2.0 * math.pi / 0.98) + 1.1)
    weave = 0.5 + 0.5 * (0.62 * a * b + 0.38 * np.sin((x + y) * 1.9))
    fx = np.floor(x * 1.7).astype(np.int64)
    fy = np.floor(y * 1.7).astype(np.int64)
    h = (fx * 374761393 + fy * 668265263) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0xFFFFFFFF
    grain = (h >> 8).astype(np.float32) / float(1 << 24)
    return 0.84 + 0.20 * weave + 0.13 * grain


# ---------------------------------------------------------------- the fill
def fill_holes(img, dep, mask, kernel=9, rounds=(5, 9)):
    """A splatted point cloud leaves holes, and they are not scattered -- they
    come in PATCHES.  Sampling a superquadric by random directions starves the
    high-curvature shoulder, because that whole band of the surface answers to
    a sliver of the sphere of directions.  So the gaps cluster exactly where a
    plate turns over its edge.

    A density threshold cannot fix that: raise it and the patches stay, lower
    it and the object grows a fringe, because a patch interior and a genuine
    silhouette look identical to a neighbour count.  Morphological CLOSING can
    -- it fills anything enclosed and leaves the outline where it was.  Then
    only the enclosed pixels get painted, from their real neighbours."""
    inside = binary_closing(mask, np.ones((kernel, kernel), bool))
    m = mask.astype(np.float32)
    out = img.copy()
    dz = np.where(mask, dep, 0.0).astype(np.float32)
    todo = inside & ~mask
    for k in rounds:
        if not todo.any():
            break
        den = uniform_filter(m, size=k)
        need = todo & (den > 1e-4)
        if not need.any():
            break
        num = uniform_filter(out * m[..., None], size=(k, k, 1))
        numz = uniform_filter(dz, size=k)
        out[need] = num[need] / den[need][:, None]
        dz[need] = numz[need] / den[need]
        m = np.where(need, 1.0, m)
        todo = todo & ~need
    return out, dz, m > 0.5


# ----------------------------------------------------------------- a frame
_PIX = None
LAST = {}


def _pixel_rays():
    global _PIX
    if _PIX is None:
        xs = (np.arange(W, dtype=np.float32) + 0.5 - W * 0.5) / FOCAL
        ys = -(np.arange(H, dtype=np.float32) + 0.5 - H * 0.5) / FOCAL
        gx, gy = np.meshgrid(xs, ys)
        d = (RIGHT[None, None, :] * gx[..., None]
             + UP[None, None, :] * gy[..., None]
             + FWD[None, None, :])
        _PIX = d.astype(np.float32)
    return _PIX


def backdrop(scroll):
    """The two tapes, drawn analytically: every pixel's ray meets the plane
    z = 0 exactly once, so there is no sampling to get wrong."""
    d = _pixel_rays()
    dz = d[..., 2]
    t = np.where(np.abs(dz) > 1e-6, -EYE[2] / dz, -1.0)
    hit = t > 0.0
    t = np.where(hit, t, 1.0)
    x = EYE[0] + t * d[..., 0]
    y = EYE[1] + t * d[..., 1]

    on_tape = np.abs(x) > edge_at(y)
    lit = np.zeros((H, W, 3), np.float32)

    px = np.stack([x, y, np.zeros_like(x)], -1).reshape(-1, 3)
    nn = np.zeros_like(px)
    nn[:, 2] = 1.0
    col = shade(px, nn, CLOTH_RGB.astype(np.float32)[None, :], 0.030, 22.0)
    tex = cloth_texture(x - 0.0, y - scroll).reshape(-1)   # the cloth moves too
    col *= tex[:, None]
    lit = col.reshape(H, W, 3)

    # the gap between the tapes, above the slider: a dark recess
    lit = np.where((hit & on_tape)[..., None], lit, lit * 0.10)
    # d is built as RIGHT*gx + UP*gy + FWD on an orthonormal basis, so
    # d . FWD == 1 and the ray parameter t IS the view depth.  No conversion.
    depth = np.where(hit, t, 1e6)
    return lit, depth.astype(np.float32)


def scroll_at(f):
    """Raised cosine: starts still, one stroke down, still, one stroke back.
    scroll(f) == scroll(N-f) exactly, which is the palindrome, and
    scroll(0) == scroll(N) == 0, which is the loop."""
    return TRAVEL * 0.5 * (1.0 - math.cos(2.0 * math.pi * f / N_FRAME))


def render_frame(f, dof=True):
    scroll = scroll_at(f)
    cloth, cloth_z = backdrop(scroll)

    img = np.zeros((H, W, 3), np.float32)
    depth = np.full((H, W), 1e6, np.float32)
    cp, cn = chain(scroll)
    # Solids first, chain last, and everything after the first has to win
    # by a MARGIN.  A splat hands its own depth to all 25 of its pixels,
    # so a sample two pixels off carries a depth error with it, and a
    # tooth 0.3 mm behind a plate was punching through it in 8% of the
    # slider.  Genuine occlusion here is a millimetre or more, so a
    # 0.35 mm bias costs nothing real.
    parts = [(SLIDER_P, SLIDER_N, METAL_RGB, 0.88, 26.0, 2),
             (TAB_P, TAB_N, TAB_RGB, 0.80, 19.0, 2),
             (cp, cn, METAL_RGB, 0.95, 34.0, 1)]

    for p, n, base, sk, rough, sp_n in parts:
        px, py, vz = project(p)
        ix = np.round(px).astype(np.int64)
        iy = np.round(py).astype(np.int64)
        ok = (ix >= -sp_n) & (ix < W + sp_n) & (iy >= -sp_n) & (iy < H + sp_n)
        if not ok.any():
            continue
        ix, iy, vz = ix[ok], iy[ok], vz[ok]
        pp, nn = p[ok], n[ok]
        # facing away from the camera cannot be seen, and dropping it early
        # keeps the z-buffer honest about thin plates
        vd = EYE.astype(np.float32) - pp
        face = np.sum(vd * nn, axis=1) > 0.0
        ix, iy, vz = ix[face], iy[face], vz[face]
        pp, nn = pp[face], nn[face]
        nsrc = ix.size
        if nsrc == 0:
            continue

        # Painter's algorithm, sorted far to near, in chunks.
        #
        # The obvious way to splat is to build every (sample, offset) pair at
        # once and sort by pixel then depth.  At 25 offsets that is eighteen
        # million rows for the slider alone and it peaked over a gigabyte --
        # the first full render was killed at frame 36 with no traceback,
        # which is what being OOM-killed looks like.
        #
        # Instead: order the samples once, furthest first, and scatter.  Numpy
        # fancy-index assignment keeps the LAST write to a repeated index, so
        # the nearest sample wins with no sort at all.  Laying each sample's
        # offsets out contiguously (repeat, not tile) keeps the whole array in
        # depth order, and chunking keeps it small, because chunk k is
        # entirely behind chunk k+1.
        order = np.argsort(-vz)
        ixs, iys, vzs = ix[order], iy[order], vz[order]

        idxbuf = np.full(H * W, -1, np.int32)
        zbuf = np.empty(H * W, np.float32)
        oa = np.array([a for a in range(-sp_n, sp_n + 1)
                       for _ in range(-sp_n, sp_n + 1)], np.int64)
        ob = np.array([b for _ in range(-sp_n, sp_n + 1)
                       for b in range(-sp_n, sp_n + 1)], np.int64)
        k = oa.size
        CH = 60000
        for s0 in range(0, nsrc, CH):
            sl = slice(s0, min(s0 + CH, nsrc))
            bx = np.repeat(ixs[sl], k) + np.tile(oa, sl.stop - sl.start)
            by = np.repeat(iys[sl], k) + np.tile(ob, sl.stop - sl.start)
            m = (bx >= 0) & (bx < W) & (by >= 0) & (by < H)
            if not m.any():
                continue
            flat = by[m] * W + bx[m]
            idxbuf[flat] = np.repeat(
                np.arange(sl.start, sl.stop, dtype=np.int32), k)[m]
            zbuf[flat] = np.repeat(vzs[sl], k)[m]
            del bx, by, m, flat

        cell = np.flatnonzero(idxbuf >= 0)
        if cell.size == 0:
            continue
        sel = order[idxbuf[cell]]
        near = zbuf[cell] < depth.reshape(-1)[cell] - Z_BIAS
        cell, sel = cell[near], sel[near]
        if cell.size == 0:
            continue
        col = shade(pp[sel], nn[sel], base.astype(np.float32)[None, :],
                    sk, rough)
        img.reshape(-1, 3)[cell] = col
        depth.reshape(-1)[cell] = zbuf[cell]
        del idxbuf, zbuf

    img, depth, solid = fill_holes(img, depth, depth < 1e5)
    depth = np.where(solid, depth, 1e6)
    front = solid & (depth < cloth_z)
    img = np.where(front[..., None], img, cloth)
    depth = np.where(front, depth, cloth_z)

    LAST['depth'] = depth
    if dof:
        img = defocus(img, depth)
    return img


def defocus(img, depth):
    """Macro depth of field.  Three blur levels blended by circle of
    confusion, which is cheap and, at these radii, indistinguishable."""
    coc = np.clip(np.abs(depth - DEPTH_FOCUS) * COC, 0.0, COC_MAX)
    lv = [img,
          gaussian_filter(img, (2.6, 2.6, 0.0)),
          gaussian_filter(img, (7.4, 7.4, 0.0)),
          gaussian_filter(img, (16.0, 16.0, 0.0))]
    r = np.array([0.0, 2.6, 7.4, 16.0], np.float32)
    out = np.zeros_like(img)
    acc = np.zeros(depth.shape, np.float32)
    for i in range(3):
        lo, hi = r[i], r[i + 1]
        u = np.clip((coc - lo) / (hi - lo), 0.0, 1.0)
        band = (coc >= lo) & (coc < hi)
        wgt = band.astype(np.float32)
        out += wgt[..., None] * (lv[i] * (1.0 - u)[..., None]
                                 + lv[i + 1] * u[..., None])
        acc += wgt
    top = (coc >= r[3]).astype(np.float32)
    out += top[..., None] * lv[3]
    acc += top
    return out / np.maximum(acc, 1e-6)[..., None]


def to_bytes(img):
    """Float -> rgb24, with a shoulder rather than a clip, so a specular
    highlight rolls off instead of turning into a white hole."""
    x = np.maximum(img, 0.0)
    x = x / (1.0 + x * 0.88)
    x = np.clip(x * 1.32, 0.0, 1.0) ** (1.0 / 2.25)
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8).tobytes()


# ------------------------------------------------------------------ checks
def check():
    ok = True

    def say(good, label, detail):
        nonlocal ok
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  {label:<26} {detail}")

    print("ZIP -- a zip fastener, closing and opening\n")

    # 1. the interlock is physically possible: heads must not collide
    left_v = 0.0
    right_v = STEP
    hv = 1.12
    clear = (right_v - hv) - (left_v + hv)
    say(clear > 0.05, "teeth clear each other",
        f"{clear:.2f} mm between a head and the next, pitch {PITCH:.2f}")

    # ... and they must actually MESH, ie cross the centreline
    say(TIP_X > 0.5, "teeth interlock",
        f"each crosses the centreline by {TIP_X:.2f} mm")

    # 2. the chain closed is one column, alternating tapes
    k, y = tooth_indices(0.0)
    closed = k[y < SL_BOT - 2.0]
    sides = closed % 2
    alt = np.all(sides[1:] != sides[:-1])
    say(bool(alt), "closed chain alternates",
        f"{closed.size} teeth below the slider, every other one flips tape")

    # 3. the palindrome, from the timing alone
    worst = max(abs(scroll_at(f) - scroll_at(N_FRAME - f))
                for f in range(1, N_FRAME))
    say(worst < 1e-9, "scroll is a palindrome",
        f"max |s(f) - s(N-f)| = {worst:.2e} mm over {N_FRAME} frames")
    say(abs(scroll_at(0)) < 1e-9 and abs(scroll_at(N_FRAME)) < 1e-9,
        "and it loops", f"s(0) = s({N_FRAME}) = 0, travel {TRAVEL:.0f} mm")

    # 4. the palindrome, from the PIXELS -- rendered independently
    worstpx = 0
    for f in (17, 53, 91):
        a = render_frame(f)
        b = render_frame(N_FRAME - f)
        worstpx = max(worstpx, int(np.abs(a - b).max() * 255.0))
    say(worstpx == 0, "frames mirror exactly",
        f"f and {N_FRAME}-f identical at f=17,53,91: max diff {worstpx}/255")

    # 5. HELD OUT.  Work out where every tooth's nib should land on screen,
    #    straight from the model and the camera, then ask the RENDERED pixels
    #    whether a tooth is there -- each predicted nib must be brighter than
    #    the two gaps either side of it.  A first attempt measured pitch by
    #    FFT and could not: the visible closed chain holds about three periods,
    #    which quantises the answer to 100 px or 134 px with nothing between.
    #    A second measured a strip that was mostly slider.  Perspective means
    #    there is no single pitch to compare anyway -- the spacing grows all
    #    the way down the frame -- so the thing to check is POSITIONS.
    img = render_frame(0, dof=False)
    lum = img.mean(2)

    k, y = tooth_indices(0.0)
    side = np.where(k % 2 == 0, -1.0, 1.0)
    ex = edge_at(y)
    yw = yaw_at(y) * side
    au = 2.72 * side
    apex = np.stack([au * np.cos(yw) - side * ex,
                     au * np.sin(yw) + y,
                     np.full(y.shape, 0.76 + 0.44)], 1).astype(np.float32)
    acol, arow, _ = project(apex)

    order = np.argsort(arow)
    acol, arow = acol[order], arow[order]
    mid_c = 0.5 * (acol[1:] + acol[:-1])
    mid_r = 0.5 * (arow[1:] + arow[:-1])

    def sample(c, r):
        ci = np.round(c).astype(int)
        ri = np.round(r).astype(int)
        ok = (ci > 2) & (ci < W - 3) & (ri > 2) & (ri < H - 3)
        return lum[ri[ok], ci[ok]], ok

    # only teeth the render can actually SEE.  A nib parked under the slider
    # has the slider in front of it, so both it and the gaps either side of it
    # sample the same lump of metal and the comparison means nothing.
    dep = LAST['depth']
    adep = project(apex)[2][order]
    vis = np.zeros(acol.size, bool)
    for i in range(acol.size):
        ci, ri = int(round(acol[i])), int(round(arow[i]))
        if 2 < ci < W - 3 and 2 < ri < H - 3:
            vis[i] = abs(float(dep[ri, ci]) - float(adep[i])) < 0.9
    acol, arow = acol[vis], arow[vis]
    mid_c = 0.5 * (acol[1:] + acol[:-1])
    mid_r = 0.5 * (arow[1:] + arow[:-1])

    tooth_v, tok = sample(acol, arow)
    gap_v, gok = sample(mid_c, mid_r)
    ratio = float(tooth_v.mean() / max(gap_v.mean(), 1e-6))
    say(tooth_v.size >= 12 and ratio > 1.6, "teeth land where predicted",
        f"{tooth_v.size} nibs visible, mean {tooth_v.mean():.3f} against "
        f"{gap_v.mean():.3f} in the gaps between them ({ratio:.2f}x)")

    # and each one individually, against its own two neighbours
    wins = 0
    tot = 0
    for i in range(1, acol.size - 1):
        if not (2 < acol[i] < W - 3 and 2 < arow[i] < H - 3):
            continue
        a0 = lum[int(round(arow[i])), int(round(acol[i]))]
        g0 = lum[int(round(mid_r[i - 1])), int(round(mid_c[i - 1]))] \
            if 2 < mid_c[i - 1] < W - 3 and 2 < mid_r[i - 1] < H - 3 else 0.0
        g1 = lum[int(round(mid_r[i])), int(round(mid_c[i]))] \
            if 2 < mid_c[i] < W - 3 and 2 < mid_r[i] < H - 3 else 0.0
        tot += 1
        wins += int(a0 > g0 and a0 > g1)
    say(tot >= 10 and wins >= int(tot * 0.85), "and each beats its neighbours",
        f"{wins} of {tot} nibs brighter than the gap on both sides")

    # 6. nothing important is clipped
    p, _ = chain(scroll_at(HALF_F))
    px, py, _ = project(np.vstack([p, SLIDER_P, TAB_P]))
    say(py.max() > H * 0.99, "chain runs off the bottom",
        f"lowest drawn row {py.max():.0f} of {H}")
    sx, sy, _ = project(SLIDER_P)
    say(sx.min() > 0 and sx.max() < W and sy.min() > 0 and sy.max() < H,
        "slider is fully in frame",
        f"c{sx.min():.0f}..{sx.max():.0f}  r{sy.min():.0f}..{sy.max():.0f}")

    # 7. the picture is not static: something must actually move
    d = np.abs(render_frame(HALF_F // 2) - render_frame(HALF_F // 2 + 3))
    moved = float((d.mean(2) > 0.02).mean())
    say(moved > 0.05, "the chain visibly moves",
        f"{moved * 100:.1f}% of pixels change over 3 frames (0.1 s)")

    print(f"\n  runtime            {N_FRAME / FPS:.1f} s, {N_FRAME} frames, "
          f"silent")
    print(f"  points             {chain(0.0)[0].shape[0] + SLIDER_P.shape[0] + TAB_P.shape[0]:,}")
    print("\n" + ("ALL PASS" if ok else "SOMETHING FAILED"))
    return 0 if ok else 1


def stills(prefix):
    from PIL import Image
    for f in (0, 22, HALF_F // 2, HALF_F, HALF_F + 40, N_FRAME - 22):
        img = render_frame(f)
        px = np.frombuffer(to_bytes(img), np.uint8).reshape(H, W, 3)
        p = f"{prefix}_{f:04d}.png"
        Image.fromarray(px).save(p)
        print("wrote", p)


def render(path):
    cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
           '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '17',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in range(N_FRAME):
        pr.stdin.write(to_bytes(render_frame(f)))
        if f % 12 == 0:
            print(f"  {f}/{N_FRAME}", flush=True)
    pr.stdin.close()
    pr.wait()
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--stills')
    ap.add_argument('--out')
    a = ap.parse_args()
    if a.check:
        return check()
    if a.stills:
        return stills(a.stills)
    render(a.out or os.path.join(os.path.dirname(_HERE_), 'zip.mp4'))
    return 0


_HERE_ = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    sys.exit(main())
