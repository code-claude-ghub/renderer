#!/usr/bin/env python3
"""THE POTATO RADIUS — grow a lump past ~500 km and gravity rounds it off.

For @Lost_Warden, who wanted to know where to buy cosmically large plates.

One body, big, tumbling, always the same size on screen. The diameter
counter carries the scale. As it climbs through the potato radius the
mountains sink — not eroding, SINKING, because at that size rock flows —
and the form relaxes into a sphere. At the end one mountain tries to
rise again and is crushed. You cannot keep a corner.

Physics (verified 2026-08-21 against Lineweaver & Norman 2010,
arXiv:1004.1091): potato radius ~200 km for icy bodies, ~300 km for
rocky (i.e. ~400 / ~600 km diameter). Mimas is round at 396 km
diameter, Proteus is a lumpy box at 420. The line is soft.

Colorway: peach/rust body on deep violet-charcoal. New this piece.
"""
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# Works from scripts/ (asciilib alongside) and from
# the public repo, where pieces live in pieces/. Insert both.
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
from asciilib import (Camera, Encoder, Frame, Grid, contact, depth_cue,
                      ink_lut, lambert, rot, specular, visible, zbuffer)

G = Grid()
RAMP = ink_lut()

# ---------------------------------------------------------------- colours
BG = (19 / 255, 13 / 255, 28 / 255)          # deep violet-charcoal
RUST = np.array([0.62, 0.33, 0.25])          # dim end of the body tint
PEACH = np.array([1.00, 0.80, 0.58])         # lit end
COUNT_RGB = (0.96, 0.90, 0.82)               # the instrument reading

FPS = 30
T_END = 19.0
FRAMES = int(T_END * FPS)                    # 570
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', 'renders', 'potato_radius.mp4')

# ---------------------------------------------------------------- timeline
T_GROW0, T_GROW1 = 2.5, 13.0                 # diameter 120 -> 640 km
T_MTN0, T_MTN1, T_MTN2 = 15.3, 16.1, 17.6    # late mountain: rise, crush
D_LO, D_HI = 120.0, 640.0                    # km, the counter
SLUMP0, SLUMP1 = 380.0, 580.0                # slump window in km (soft line)
A_MAX, A_MIN = 0.36, 0.012                   # lumpiness amplitude
M_MAX = 0.28                                 # late-mountain amplitude
STRETCH = np.array([1.38, 0.84, 1.06])       # potato elongation, relaxes too


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def diameter(t):
    if t <= T_GROW0:
        return D_LO
    if t >= T_GROW1:
        return D_HI
    u = (t - T_GROW0) / (T_GROW1 - T_GROW0)
    return D_LO + (D_HI - D_LO) * u ** 1.4


def lump_amp(d_km):
    """Material strength vs gravity: full potato below the window,
    near-sphere above it. The window straddles the icy potato radius."""
    s = smoothstep((d_km - SLUMP0) / (SLUMP1 - SLUMP0))
    return A_MAX + (A_MIN - A_MAX) * s


def mountain_amp(t):
    if t <= T_MTN0 or t >= T_MTN2:
        return 0.0
    if t < T_MTN1:                            # rise
        return M_MAX * smoothstep((t - T_MTN0) / (T_MTN1 - T_MTN0))
    return M_MAX * (1.0 - smoothstep((t - T_MTN1) / (T_MTN2 - T_MTN1)))


# ---------------------------------------------------------------- geometry
N = 120_000
RNG = np.random.default_rng(11)


def fib_sphere(n):
    i = np.arange(n, dtype=np.float64)
    phi = math.pi * (3.0 - math.sqrt(5.0)) * i
    y = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    d = np.stack([np.cos(phi) * r, y, np.sin(phi) * r], -1)
    # Trap 10: a quasi-regular lattice spun against the cell grid moires.
    # Jitter once, fixed seed, renormalise.
    d = d + 0.004 * RNG.standard_normal(d.shape)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


DIRS = fib_sphere(N)

# The potato: a fixed sum of cosine plane waves (smooth, seeded) plus a
# few broad gaussian massifs for character. Depends only on direction, so
# every per-point value is computed exactly once.
NWAVE = 14
WK = RNG.standard_normal((NWAVE, 3))
WK = WK / np.linalg.norm(WK, axis=1, keepdims=True) \
     * RNG.uniform(1.6, 4.4, NWAVE)[:, None]
WA = RNG.uniform(0.5, 1.0, NWAVE) * (1.6 / np.abs(WK).max(1))
WP = RNG.uniform(0.0, 2 * math.pi, NWAVE)

MASSIF_C = np.array([[0.8, -0.5, 0.3], [-0.6, 0.6, 0.5], [0.1, 0.3, -0.9]])
MASSIF_C = MASSIF_C / np.linalg.norm(MASSIF_C, axis=1, keepdims=True)
MASSIF_A = np.array([0.9, -0.7, 0.8])
MASSIF_W = np.array([0.55, 0.65, 0.5])


def bump(d):
    b = np.zeros(len(d))
    for k in range(NWAVE):
        b += WA[k] * np.cos(d @ WK[k] + WP[k])
    for c, a, w in zip(MASSIF_C, MASSIF_A, MASSIF_W):
        ang = np.arccos(np.clip(d @ c, -1.0, 1.0))
        b += a * np.exp(-(ang / w) ** 2)
    return b


_B_RAW = bump(DIRS)
B_SCALE = 1.0 / np.abs(_B_RAW).max()
B = _B_RAW * B_SCALE                          # in [-1, 1]

# The late mountain: place it on the upper-left RIM at its peak, so it
# reads in silhouette against the background (a bump aimed at the viewer
# changes nothing a phone can see). We pre-rotate the desired on-screen
# direction back through the body rotation at that moment.
VIEW_DIR = np.array([[-0.62, -0.72, 0.31]])
VIEW_DIR = VIEW_DIR / np.linalg.norm(VIEW_DIR)


def pose_angles(t):
    ax = 0.45 + 0.15 * math.sin(0.23 * t)
    ay = 0.30 * t
    az = 0.10 * math.sin(0.11 * t)
    return ax, ay, az


def unrotate(v, t):
    """Inverse of rot(ax, ay, az) applied at time t (for one vector)."""
    ax, ay, az = pose_angles(t)
    n = np.zeros_like(v)
    v, _ = rot(v, n, az=-az)
    v, _ = rot(v, n, ay=-ay)
    v, _ = rot(v, n, ax=-ax)
    return v


MTN_C = unrotate(VIEW_DIR.copy(), T_MTN1 + 0.35)[0]
MTN_W = 0.38


def mtn_field(d):
    ang = np.arccos(np.clip(d @ MTN_C, -1.0, 1.0))
    return np.exp(-(ang / MTN_W) ** 2)


MTN = mtn_field(DIRS)


def tangential_grad(fn, d, eps=1e-3):
    """Numerical surface gradient of a scalar field on the unit sphere."""
    up = np.where(np.abs(d[:, 1:2]) < 0.9,
                  np.tile([0.0, 1.0, 0.0], (len(d), 1)),
                  np.tile([1.0, 0.0, 0.0], (len(d), 1)))
    t1 = np.cross(d, up)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(d, t1)
    f0 = fn(d)

    def at(dd):
        return fn(dd / np.linalg.norm(dd, axis=1, keepdims=True))

    g1 = (at(d + eps * t1) - f0) / eps
    g2 = (at(d + eps * t2) - f0) / eps
    return g1[:, None] * t1 + g2[:, None] * t2


GRAD_B = tangential_grad(lambda d: bump(d) * B_SCALE, DIRS)
GRAD_M = tangential_grad(mtn_field, DIRS)

LAMP = np.array([-0.55, -0.62, 0.56])
LAMP /= np.linalg.norm(LAMP)

# Trap 18: smooth analytic shading on a near-sphere contours into rings.
# Stipple once, fixed seed, attached to the surface so it does not crawl.
STIPPLE = 1.0 + 0.09 * RNG.uniform(-1.0, 1.0, N)


def body(t):
    """Points and normals in world space at time t."""
    a = lump_amp(diameter(t))
    m = mountain_amp(t)
    h = a * B + m * MTN
    pts = DIRS * (1.0 + h)[:, None]
    nrm = DIRS - (a * GRAD_B + m * GRAD_M)
    # the elongation is a lump too — it relaxes on the same curve
    q = (a - A_MIN) / (A_MAX - A_MIN)
    s = 1.0 + q * (STRETCH - 1.0)
    pts = pts * s
    nrm = nrm / s                    # normals transform by the inverse scale
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    # slumping conserves volume: the sphere is what the potato weighs
    vol = s.prod() * np.mean((1.0 + h) ** 3)
    pts = pts * (1.0 / vol) ** (1.0 / 3.0)
    ax, ay, az = pose_angles(t)
    return rot(pts, nrm, ax=ax, ay=ay, az=az)


# Fit over the widest thing the animation ever shows: the full potato at
# many rotations (trap 2 / trap 7).
_poses = []
for tt in np.linspace(0.0, T_GROW0 + 2.0, 9):
    p, _ = body(tt)
    _poses.append(p)
CAM = Camera(G).fit(_poses, margin=1.10)


# ---------------------------------------------------------------- counter
# 3x5 pixel font, drawn at 2x -> each glyph 6x10 cells of '@'. Instrument
# reading only: the diameter in km. Built out of cells (trap 11), placed
# below the top-10% safe line and well clear of the body.
FONT = {
    '0': ['###', '#.#', '#.#', '#.#', '###'],
    '1': ['.#.', '##.', '.#.', '.#.', '###'],
    '2': ['###', '..#', '###', '#..', '###'],
    '3': ['###', '..#', '.##', '..#', '###'],
    '4': ['#.#', '#.#', '###', '..#', '..#'],
    '5': ['###', '#..', '###', '..#', '###'],
    '6': ['###', '#..', '###', '#.#', '###'],
    '7': ['###', '..#', '.#.', '.#.', '.#.'],
    '8': ['###', '#.#', '###', '#.#', '###'],
    '9': ['###', '#.#', '###', '..#', '###'],
    'K': ['#.#', '#.#', '##.', '#.#', '#.#'],
    'M': ['#.#', '###', '###', '#.#', '#.#'],
    ' ': ['...', '...', '...', '...', '...'],
}
CNT_ROW = 21          # top of digits; safe_top ends at row 17
CNT_SCALE = 2


def draw_counter(fr, d_km):
    txt = '%d KM' % round(d_km)
    w = len(txt) * (3 * CNT_SCALE + CNT_SCALE) - CNT_SCALE
    c0 = (G.cols - w) // 2
    for ch in txt:
        pat = FONT[ch]
        for r in range(5):
            for c in range(3):
                if pat[r][c] == '#':
                    for dr in range(CNT_SCALE):
                        for dc in range(CNT_SCALE):
                            fr.put(c0 + c * CNT_SCALE + dc,
                                   CNT_ROW + r * CNT_SCALE + dr,
                                   '@', COUNT_RGB)
        c0 += 3 * CNT_SCALE + CNT_SCALE


# ---------------------------------------------------------------- drawing
def colour(s, _):
    # Trap 12: the glyph carries the light; tint with a floor.
    c = RUST + (PEACH - RUST) * s
    return (c[0], c[1], c[2])


def draw_frame(t):
    pts, nrm = body(t)
    col, row, z = CAM.project(pts)
    ok = visible(G, col, row)
    col, row, z, n = col[ok], row[ok], z[ok], nrm[ok]
    front = n[:, 2] > -0.05
    col, row, z, n = col[front], row[front], z[front], n[front]
    stip = STIPPLE[ok][front]
    _, keep = zbuffer(G, col, row, z)
    shade = (0.22 + 0.62 * lambert(n, LAMP) +
             0.28 * specular(n, LAMP, 14)) * depth_cue(z, far=0.92) * stip
    fr = Frame(G, BG)
    fr.field(col, row, keep, shade, colour, RAMP)
    draw_counter(fr, diameter(t))
    return fr


# ---------------------------------------------------------------- checks
def check():
    print('frames %d  (%.1f s at %d fps)' % (FRAMES, T_END, FPS))
    print('diameter / lumpiness through the run:')
    for tt in [0.0, 3.0, 6.0, 8.0, 10.0, 11.5, 13.0, 16.1, 18.5]:
        d = diameter(tt)
        print('  t=%5.1f  D=%4.0f km  A=%.3f  mtn=%.3f'
              % (tt, d, lump_amp(d), mountain_amp(tt)))
    lo_r, hi_r, lo_c, hi_c = 1e9, -1e9, 1e9, -1e9
    worst_gap = 0
    for tt in np.linspace(0.0, T_END, 25):
        pts, _ = body(tt)
        col, row, z = CAM.project(pts)
        ok = visible(G, col, row)
        col, row = col[ok], row[ok]
        lo_r, hi_r = min(lo_r, row.min()), max(hi_r, row.max())
        lo_c, hi_c = min(lo_c, col.min()), max(hi_c, col.max())
        # convex-ish silhouette: per-row min..max gap check is valid here
        filled = np.zeros((G.rows, G.cols), bool)
        filled[row, col] = True
        for r in range(G.rows):
            cc = np.nonzero(filled[r])[0]
            if len(cc) > 2:
                gaps = np.diff(cc) - 1
                worst_gap = max(worst_gap, int(gaps.max()))
    print('body rows %d..%d  cols %d..%d  worst interior gap %d'
          % (lo_r, hi_r, lo_c, hi_c, worst_gap))
    assert lo_r > CNT_ROW + 10 + 2, 'body collides with the counter'
    assert hi_r < G.rows - 2 and lo_c >= 0 and hi_c < G.cols, 'clipping'
    assert worst_gap <= 2, 'holes in the body'
    print('counter rows %d..%d — clear of body and of top safe area'
          % (CNT_ROW, CNT_ROW + 9))
    print('OK')


def stills():
    ts = [0.5, 5.0, 9.0, 11.0, 12.2, 14.0, 16.1, 17.0]
    frames = [draw_frame(t) for t in ts]
    sheet = os.path.join(os.path.dirname(OUT), 'potato_sheet.png')
    contact(frames, sheet, cols=4,
            labels=['t=%.1f D=%dkm' % (t, round(diameter(t))) for t in ts])
    print(sheet)


def render():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with Encoder(OUT, G, fps=FPS) as enc:
        for f in range(FRAMES):
            enc.write(draw_frame(f / FPS))
            if f % 60 == 0:
                print('  frame %d/%d' % (f, FRAMES))
    print(OUT)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'check'
    if mode == 'check':
        check()
    elif mode == 'stills':
        stills()
    elif mode == 'render':
        render()
