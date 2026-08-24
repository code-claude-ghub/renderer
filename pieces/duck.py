#!/usr/bin/env python3
"""
A joke.

No facts, no citations, nothing to learn, no question at the end.  A storm at
sea, rendered with every bit of seriousness this renderer has -- marched
heightfield swell, sharp crests, foam on the breaking faces, spray, a low
bruised sky -- and floating in the middle of it, at perfect ease, a rubber
duck.  The sea does its absolute best for eleven seconds. The duck is fine.

The whole frame is raytraced per cell, which is affordable because there are
only 98x174 of them: each cell casts one ray, marches it against the ocean
heightfield, solves a quadratic against each of the duck's five ellipsoids,
and keeps whichever is nearer.  One depth buffer, no projection, no z-fight.

Zero words on screen.

    python3 scripts/duck.py --check
    python3 scripts/duck.py
"""

import argparse
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]

from asciilib import Encoder, Frame, Grid, contact, ink_lut  # noqa: E402

OUT = os.path.join(_HERE, "..", "content", "duck.mp4")

# ---------------------------------------------------------------- palette
SKY_HI = (0.129, 0.145, 0.180)
SKY_LO = (0.400, 0.412, 0.416)   # the horizon is always brighter
SEA_LO = (0.075, 0.145, 0.153)
SEA_HI = (0.541, 0.667, 0.627)
FOAM = (0.925, 0.953, 0.945)
DUCK = (0.996, 0.812, 0.114)
BILL = (0.961, 0.478, 0.114)
EYE = (0.055, 0.043, 0.035)

M_SKY, M_SEA, M_FOAM, M_DUCK, M_BILL, M_EYE = 1, 2, 3, 4, 5, 6

G = Grid()
RAMP = ink_lut()
FPS = 30
T_END = 11.0
FRAMES = int(round(T_END * FPS))
RNG = np.random.default_rng(1992)

# ---------------------------------------------------------------- the sea
# four trains of swell.  (amplitude, wavelength, direction deg, speed mult)
SWELL = [(2.35, 41.0, 8.0, 1.00),
         (1.42, 23.0, -26.0, 1.18),
         (0.62, 11.5, 39.0, 1.42),
         (0.27, 5.3, -61.0, 1.75)]
GRAV = 9.81
WAVES = []
for amp, lam, deg, mult in SWELL:
    k = 2.0 * math.pi / lam
    w = math.sqrt(GRAV * k) * mult
    a = math.radians(deg)
    WAVES.append((amp, k * math.cos(a), k * math.sin(a), w))


def sea(x, z, t):
    """Height of the water.  Crests are sharpened -- a real swell is not a
    sine, it is pointed on top and lazy in the trough."""
    h = np.zeros_like(x)
    for amp, kx, kz, w in WAVES:
        s = 0.5 + 0.5 * np.sin(kx * x + kz * z - w * t)
        h = h + amp * (2.0 * s ** 2.3 - 1.0)
    return h


def sea_n(x, z, t, e=0.22):
    """Surface normal by central difference -- cheaper and steadier than
    differentiating the sharpened profile by hand."""
    hx = sea(x + e, z, t) - sea(x - e, z, t)
    hz = sea(x, z + e, t) - sea(x, z - e, t)
    nx, nz = -hx / (2 * e), -hz / (2 * e)
    n = np.sqrt(nx * nx + nz * nz + 1.0)
    return nx / n, 1.0 / n, nz / n


# ---------------------------------------------------------------- the duck
DUCK_XZ = (0.0, 4.6)
SCALE = 0.80          # a bath duck is 8 cm.  this one is not a bath duck.
BODY = (0.0, 0.02, 0.0, 1.00, 0.74, 0.86)     # cx cy cz  rx ry rz
HEAD = (0.42, 0.86, 0.0, 0.54, 0.52, 0.50)
BEAK = (0.92, 0.78, 0.0, 0.34, 0.17, 0.24)
TAIL = (-0.95, 0.44, 0.0, 0.36, 0.30, 0.22)
EYEB = (0.60, 1.02, 0.33, 0.15, 0.15, 0.15)
PARTS = [(BODY, M_DUCK), (TAIL, M_DUCK), (HEAD, M_DUCK),
         (BEAK, M_BILL), (EYEB, M_EYE)]

# ---------------------------------------------------------------- camera
CAM_Y = 1.15          # eye height above the LOCAL water, in metres
FOVX = math.radians(46.0)
CC, RR = np.meshgrid(np.arange(G.cols, dtype=np.float64),
                     np.arange(G.rows, dtype=np.float64))
JIT = RNG.normal(0.0, 1.0, (G.rows, G.cols))
LAMP = np.array([-0.40, 0.47, 0.79])
LAMP = LAMP / np.linalg.norm(LAMP)
LAST = {}


def rays(t):
    """One ray per cell.  The camera rides the swell a little, because a
    camera that ignores a force ten is a camera nobody believes."""
    ax = math.tan(FOVX / 2.0)
    px = (CC - G.cols / 2.0) / (G.cols / 2.0) * ax
    py = -(RR - G.rows / 2.0) / (G.cols / 2.0) * ax
    # and it heaves with the surface it is sitting on
    snx, sny, snz = sea_n(np.array(0.0), np.array(DUCK_XZ[1] - 2.2), t)
    # NB positive pitch looks DOWN in this parametrisation.  the first three
    # attempts aimed the camera at the sky and wondered where the duck went.
    pitch = math.radians(14.0) + 0.45 * float(math.atan2(float(snz),
                                                         float(sny)))
    roll = 0.9 * float(math.atan2(float(snx), float(sny)))
    cr, sr = math.cos(roll), math.sin(roll)
    px, py = px * cr - py * sr, px * sr + py * cr
    dz = np.ones_like(px)
    n = np.sqrt(px * px + py * py + 1.0)
    dx, dy, dz = px / n, py / n, dz / n
    cp, sp = math.cos(pitch), math.sin(pitch)
    dy, dz = dy * cp - dz * sp, dy * sp + dz * cp
    # the camera floats on the SAME water the duck is on, a couple of metres
    # in front of it.  a camera pinned to mean sea level made the duck swing
    # four metres up and down the frame and fall out of the bottom of it --
    # and a camera that ignores a force ten is a camera nobody believes.
    oy = CAM_Y + float(sea(np.array(0.0), np.array(DUCK_XZ[1] - 2.2), t))
    return dx, dy, dz, oy


def march(dx, dy, dz, oy, t, steps=38, far=190.0):
    """First crossing of the ray with the water, then four bisections."""
    down = dy < -1e-4
    hit = np.zeros(dx.shape, bool)
    tt = np.full(dx.shape, far)
    lo = np.zeros(dx.shape)
    hi = np.full(dx.shape, far)
    # a ray that never gets below the highest crest cannot hit anything
    prev = np.zeros(dx.shape)
    prev_d = oy - sea(np.zeros_like(dx), np.zeros_like(dx), t)
    for i in range(1, steps + 1):
        s = far * (i / float(steps)) ** 2.1
        y = oy + dy * s
        d = y - sea(dx * s, dz * s, t)
        cross = down & ~hit & (d < 0.0) & (prev_d > 0.0)
        lo = np.where(cross, prev, lo)
        hi = np.where(cross, s, hi)
        hit |= cross
        prev, prev_d = s, d
    for _ in range(5):
        mid = 0.5 * (lo + hi)
        d = (oy + dy * mid) - sea(dx * mid, dz * mid, t)
        lo = np.where(d > 0.0, mid, lo)
        hi = np.where(d > 0.0, hi, mid)
    tt = np.where(hit, 0.5 * (lo + hi), far)
    return hit, tt


def ellipsoid(dx, dy, dz, ox, oy, oz, c, t_now):
    """Ray vs axis-aligned ellipsoid, in the duck's own frame."""
    cx, cy, cz, rx, ry, rz = c
    ex = (ox - cx) / rx
    ey = (oy - cy) / ry
    ez = (oz - cz) / rz
    fx, fy, fz = dx / rx, dy / ry, dz / rz
    a = fx * fx + fy * fy + fz * fz
    b = 2.0 * (ex * fx + ey * fy + ez * fz)
    cc_ = ex * ex + ey * ey + ez * ez - 1.0
    disc = b * b - 4.0 * a * cc_
    ok = disc > 0.0
    sq = np.sqrt(np.maximum(disc, 0.0))
    t0 = (-b - sq) / (2.0 * a)
    ok &= t0 > 0.02
    px, py, pz = ox + dx * t0, oy + dy * t0, oz + dz * t0
    nx, ny, nz = (px - cx) / (rx * rx), (py - cy) / (ry * ry), \
        (pz - cz) / (rz * rz)
    n = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    return ok, t0, nx / n, ny / n, nz / n


def duck(dx, dy, dz, oy, t):
    """Put the duck on the water, tilt it with the surface, then trace it."""
    dxp, dzp = DUCK_XZ
    wy = sea(np.array(dxp), np.array(dzp), t) * 1.0
    snx, sny, snz = sea_n(np.array(dxp), np.array(dzp), t)
    # roll and pitch the duck to sit along the local surface
    tilt_x = float(math.atan2(float(snx), float(sny))) * 0.85
    tilt_z = float(math.atan2(float(snz), float(sny))) * 0.85
    # ray into duck space: translate, then un-rotate
    ox = -dxp
    oyy = oy - (float(wy) + 0.30 * SCALE)
    oz = -dzp
    ca, sa = math.cos(-tilt_x), math.sin(-tilt_x)
    ox, oyy = ox * ca - oyy * sa, ox * sa + oyy * ca
    ddx, ddy = dx * ca - dy * sa, dx * sa + dy * ca
    cb, sb = math.cos(tilt_z), math.sin(tilt_z)
    oz, oyy = oz * cb - oyy * sb, oz * sb + oyy * cb
    ddz, ddy = dz * cb - ddy * sb, dz * sb + ddy * cb
    ox, oyy, oz = ox / SCALE, oyy / SCALE, oz / SCALE

    best_t = np.full(dx.shape, 1e9)
    mat = np.zeros(dx.shape, np.int16)
    nrm = np.zeros((3,) + dx.shape)
    for c, m in PARTS:
        ok, tt, nx, ny, nz = ellipsoid(ddx, ddy, ddz, ox, oyy, oz, c, t)
        win = ok & (tt < best_t)
        best_t = np.where(win, tt, best_t)
        mat = np.where(win, m, mat)
        for i, nn in enumerate((nx, ny, nz)):
            nrm[i] = np.where(win, nn, nrm[i])
    # back out of duck space for the normal
    nx, ny, nz = nrm
    nz, ny = nz * cb + ny * sb, -nz * sb + ny * cb
    nx, ny = nx * ca + ny * sa, -nx * sa + ny * ca
    return mat > 0, best_t * SCALE, nx, ny, nz


# ---------------------------------------------------------------- draw
def lerp3(a, b, u):
    return tuple(a[i] + (b[i] - a[i]) * u for i in range(3))


def colour(v, m):
    m = int(m)
    if m == M_SKY:
        return lerp3(SKY_HI, SKY_LO, v)
    if m == M_SEA:
        return lerp3(SEA_LO, SEA_HI, v)
    if m == M_FOAM:
        return lerp3(SEA_HI, FOAM, v)
    if m == M_DUCK:
        return lerp3((0.58, 0.42, 0.06), DUCK, v)
    if m == M_BILL:
        return lerp3((0.56, 0.26, 0.05), BILL, v)
    return EYE


def draw(f):
    t = f / float(FPS)
    dx, dy, dz, oy = rays(t)
    hit, tw = march(dx, dy, dz, oy, t)
    dhit, td, dnx, dny, dnz = duck(dx, dy, dz, oy, t)

    mat = np.full((G.rows, G.cols), M_SKY, np.int16)
    # sky: dark overhead, bright along the horizon, banded weather
    v = RR / float(G.rows)
    band = 0.06 * np.sin(v * 21.0 + 0.7) + 0.04 * np.sin(v * 47.0 + 2.1)
    shade = np.clip(0.16 + 0.66 * v ** 2.6 + band * v, 0.0, 1.0)

    # --- water.  a sea is a MIRROR, not a lambert surface.  what you see is
    # mostly sky, reflected, and the amount depends on how grazing your look
    # is: nearly all of it at the horizon, hardly any at your feet.  shading
    # it like matte plastic (which is what the first pass did) gives a flat
    # dark void with faint diagonal bands and no water in it anywhere.
    wx, wz = dx * tw, dz * tw
    nx, ny, nz = sea_n(wx, wz, t)
    cos_i = np.clip(-(nx * dx + ny * dy + nz * dz), 0.0, 1.0)
    fres = 0.03 + 0.97 * (1.0 - cos_i) ** 4.2
    sky_ref = 0.56 + 0.40 * (1.0 - np.clip(ny, 0.0, 1.0)) ** 0.7
    lam = np.clip(nx * LAMP[0] + ny * LAMP[1] + nz * LAMP[2], 0.0, 1.0)
    deep = 0.15 + 0.26 * lam
    hvx, hvy, hvz = LAMP[0] - dx, LAMP[1] - dy, LAMP[2] - dz
    hn = np.sqrt(hvx * hvx + hvy * hvy + hvz * hvz) + 1e-9
    spec = np.clip((nx * hvx + ny * hvy + nz * hvz) / hn, 0.0, 1.0) ** 90
    dist = np.clip(tw / 120.0, 0.0, 1.0)
    # a floor, then the reflection on top of it.  physically the water at
    # your feet on an overcast day really is almost black, and rendering that
    # honestly gave an empty frame -- the glyph ramp has nothing to spend.
    swat = 0.27 + 0.54 * (deep * (1.0 - fres) + sky_ref * fres) + 1.5 * spec
    swat = swat * (1.0 - 0.34 * dist) + 0.42 * dist
    swat = np.clip(swat, 0.0, 1.0)

    hgt = sea(wx, wz, t)
    slope = np.clip(1.0 - ny, 0.0, 1.0)
    # foam breaks up into speckle.  a smooth threshold gave a solid white
    # slab across half the frame, which is a dust sheet, not a wave.
    hsh = np.modf(np.sin(wx * 12.9898 + wz * 78.233) * 43758.5453)[0]
    amt = (np.clip((slope - 0.185) * 6.0, 0.0, 1.0)
           * np.clip((hgt - 1.05) * 1.3, 0.0, 1.0))
    foam = (amt > 0.30) & (hsh < np.clip(amt, 0.0, 0.92)) & (tw > 2.2)
    fval = np.clip(0.42 + 0.55 * amt, 0.0, 1.0)

    mat = np.where(hit, M_SEA, mat)
    shade = np.where(hit, swat, shade)
    mat = np.where(hit & foam, M_FOAM, mat)
    shade = np.where(hit & foam, fval, shade)

    # --- duck
    near = dhit & (td < np.where(hit, tw, 1e9))
    dlam = np.clip(dnx * LAMP[0] + dny * LAMP[1] + dnz * LAMP[2], 0.0, 1.0)
    dsh = np.clip(0.40 + 0.58 * dlam + 0.40 * dlam ** 20, 0.0, 1.0)
    dmat, dtd = duck_mat(dx, dy, dz, oy, t)
    mat = np.where(near, dmat, mat)
    shade = np.where(near, dsh, shade)

    shade = np.clip(shade * (1.0 + 0.05 * JIT), 0.0, 1.0)
    fr = Frame(G, SKY_HI)
    on = shade > 0.035
    fr.field(CC[on].astype(np.int32).ravel(), RR[on].astype(np.int32).ravel(),
             np.ones(int(on.sum()), bool), shade[on].ravel(), colour, RAMP,
             extra=mat[on].astype(float).ravel())
    LAST.update(mat=mat, shade=shade, hit=hit, near=near, t=t)
    return fr


def duck_mat(dx, dy, dz, oy, t):
    """Second pass just for material ids -- cheap, and keeps duck() honest."""
    dxp, dzp = DUCK_XZ
    wy = sea(np.array(dxp), np.array(dzp), t)
    snx, sny, snz = sea_n(np.array(dxp), np.array(dzp), t)
    tilt_x = float(math.atan2(float(snx), float(sny))) * 0.85
    tilt_z = float(math.atan2(float(snz), float(sny))) * 0.85
    ox, oyy, oz = -dxp, oy - (float(wy) + 0.30 * SCALE), -dzp
    ca, sa = math.cos(-tilt_x), math.sin(-tilt_x)
    ox, oyy = ox * ca - oyy * sa, ox * sa + oyy * ca
    ddx, ddy = dx * ca - dy * sa, dx * sa + dy * ca
    cb, sb = math.cos(tilt_z), math.sin(tilt_z)
    oz, oyy = oz * cb - oyy * sb, oz * sb + oyy * cb
    ddz, ddy = dz * cb - ddy * sb, dz * sb + ddy * cb
    ox, oyy, oz = ox / SCALE, oyy / SCALE, oz / SCALE
    best = np.full(dx.shape, 1e9)
    mat = np.zeros(dx.shape, np.int16)
    for c, m in PARTS:
        ok, tt, _, _, _ = ellipsoid(ddx, ddy, ddz, ox, oyy, oz, c, t)
        win = ok & (tt < best)
        best = np.where(win, tt, best)
        mat = np.where(win, m, mat)
    return mat, best * SCALE


# ---------------------------------------------------------------- check
def check():
    print("A JOKE — no words, no facts, no ask")
    for i, (amp, lam, deg, mult) in enumerate(SWELL):
        c = math.sqrt(GRAV * lam / (2 * math.pi)) * mult
        print("  swell %d  %.2f m x %.1f m  from %+.0f deg  %.1f m/s"
              % (i, amp, lam, deg, c))
    # HELD OUT: a gravity wave steeper than 1/7 breaks.  the sharpened
    # profile must stay under that, or the sea is not water any more.
    steep = max(2 * a / l for a, l, _, _ in SWELL)
    print("  steepest H/L         %.3f  (breaking limit 0.143)" % steep)
    assert steep < 0.143, steep
    hmax = sum(a for a, _, _, _ in SWELL)
    print("  max crest            %.2f m above a %.2f m eye" % (hmax, CAM_Y))
    assert hmax > CAM_Y * 2.0, (hmax, CAM_Y)

    sheet, seen = [], []
    for t in (0.4, 1.6, 2.8, 4.0, 5.2, 6.4, 7.6, 8.8, 10.2):
        fr = draw(int(t * FPS))
        mat, sh = LAST["mat"], LAST["shade"]
        nd = int((mat >= M_DUCK).sum())
        seen.append(nd)
        rows = np.nonzero((mat >= M_DUCK).any(1))[0]
        print("  t=%4.1f  sky %4d  sea %5d  foam %4d  duck %4d  rows %s  "
              "mean %.3f"
              % (t, (mat == M_SKY).sum(), (mat == M_SEA).sum(),
                 (mat == M_FOAM).sum(), nd,
                 ("%d..%d" % (rows.min(), rows.max())) if len(rows) else "-",
                 sh.mean()))
        assert (mat == M_SEA).sum() > 2000, (mat == M_SEA).sum()
        assert (mat == M_SKY).sum() > 400, (mat == M_SKY).sum()
        assert (mat == M_FOAM).sum() < 2600, (mat == M_FOAM).sum()
        if nd > 40:
            mid = int(rows.mean())
            assert 34 < mid < 148, ("duck drifted out of frame", t, mid)
        sheet.append(fr)

    # the duck is the subject: it must be on screen in every single frame,
    # and big enough to read as a duck rather than a speck
    # the duck may duck -- a swell passing in front of it is the sea doing
    # its job -- but it has to be the subject, so: big when visible, and
    # never gone for more than a beat.
    print("  duck cells min %d  max %d  mean %d"
          % (min(seen), max(seen), int(np.mean(seen))))
    gone = [n for n in seen if n < 40]
    assert len(gone) <= 2, gone
    assert max(seen) > 900, max(seen)
    assert np.mean(seen) > 500, np.mean(seen)
    contact(sheet, os.path.join(_HERE, "..", "content", "duck_sheet.png"),
            cols=3, labels=["%.1f" % t for t in
                            (0.4, 1.6, 2.8, 4.0, 5.2, 6.4, 7.6, 8.8, 10.2)])


def main():
    with Encoder(OUT, G, fps=FPS) as enc:
        for f in range(FRAMES):
            enc.write(draw(f))
            if f % 45 == 0:
                print("  %d/%d" % (f, FRAMES), flush=True)
    print("wrote", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    check() if ap.parse_args().check else main()
