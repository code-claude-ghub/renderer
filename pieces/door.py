#!/usr/bin/env python3
"""THE DOOR -- four people in a revolving door, going round.

Six seconds, which is exactly one revolution, because an eight-foot manual
revolving door is not permitted to turn faster than 10 rpm (IBC Table
1010.3.1(1) / BHMA A156.27). The video is the length the code makes it.

Nothing here is explained and there is not one word on screen. It is silent.

A four-wing door in a two-opening drum is symmetric under a HALF turn, so the
picture at t and the picture at t+3.000 s are not similar, they are the same
picture. That is not an editing trick -- it is the machine's own symmetry, and
it means the piece has no beginning and no end. `--check` proves it by
rendering the two frames independently and comparing every cell.

The same symmetry has a price, and the price is the joke: for the loop to
close, the person in compartment 1 must be identical to the person in
compartment 3. There are four bodies in this door and two people.

Invented: the door, the lobby, the street, everyone in it.
Real: 10 rpm, 8 ft, the geometry that keeps a revolving door sealed.

    python3 scripts/door.py --check
    python3 scripts/door.py
"""

import argparse
import math
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]

from asciilib import (Camera, Encoder, Frame, Grid, contact, ink_lut,  # noqa
                      lambert, specular, visible, zbuffer)

RNG = np.random.default_rng(20260825)

# ---------------------------------------------------------------- the door
#
# IBC Table 1010.3.1(1), Maximum Door Speed, Manual Revolving Doors:
#     6-0 ft -> 12 rpm    7-0 -> 11    8-0 -> 10    9-0 -> 9    10-0 -> 8
# We build the 8-0 and run it at the top of its band.
DIAM_FT = 8.0
RPM = 10.0
R_IN = DIAM_FT * 0.3048 / 2.0            # 1.2192 m inside radius
SEC_PER_REV = 60.0 / RPM                 # 6.000 s
FPS = 30
N_WING = 4
N_FRAME = int(round(SEC_PER_REV * FPS))  # 180
HALF = N_FRAME // 2                      # 90 -- the actual loop period

# Drum: two fixed walls, two openings. A revolving door only stays sealed if
# each wall arc is WIDER than one compartment (360/n), so that a wing is
# always in contact with it. 120 > 90, so this door is always closed.
WALL_ARC = math.radians(120.0)
OPEN_ARC = math.radians(60.0)
WALL_MID = (math.radians(180.0), math.radians(0.0))   # west wall, east wall
THROAT = 2.0 * R_IN * math.sin(OPEN_ARC / 2.0)        # 1.2192 m chord

H_WING = 2.10
H_WALL = 2.28
R_SHAFT = 0.048
W_STILE = 0.050
R_PERSON = 0.615                          # how far out people stand

# Framing: a tall corridor -- pavement, door, lobby.
X_LO, X_HI = -1.92, 1.92
Y_LO, Y_HI = -3.55, 3.05

# Two shots were wrong before this one. At 66 deg with the drum filling half
# the width, the picture was a bright flat lobby slab over a black void with
# some blades in between. Dropping to 38 deg fixed nothing and cost the
# silhouette: from outside, a revolving door is glass in front of glass, and
# rendered as stipple over stipple it is mush. It reads from ABOVE, where the
# circle and the turning cross are unmistakable -- so: go back up, crop hard
# enough that the drum fills the frame, and give the drum a FLOOR so there is
# one solid thing for the cross and the people to sit on.
YAW = math.radians(12.0)
EL = math.radians(70.0)

BG = (0.052, 0.057, 0.072)

# Evening. The lobby is lit from inside and above; there is one streetlight
# out on the pavement, low and to the south-west, throwing everything
# north-east across the floor.
SUN = np.array([-0.646, -0.646, 0.407])          # toward the streetlight
SUN = SUN / np.linalg.norm(SUN)
LOBBY = np.array([0.10, 0.34, 0.93])             # toward the lobby ceiling
LOBBY = LOBBY / np.linalg.norm(LOBBY)
C_STREET = np.array([0.86, 0.90, 1.00])
C_LOBBY = np.array([1.00, 0.83, 0.60])

# materials
M_PAVE, M_FLOOR, M_GLASS, M_FRAME, M_WALL = 0, 1, 2, 3, 4
M_SKIN, M_COAT_A, M_COAT_B, M_COAT_W, M_TROUS, M_HAIR, M_BAG = 5, 6, 7, 8, 9, 10, 11
M_DRUM, M_JOINT, M_HAIR2 = 12, 13, 14

MAT_RGB = {
    M_PAVE:   (0.40, 0.42, 0.47),
    M_FLOOR:  (0.56, 0.50, 0.44),
    M_GLASS:  (0.62, 0.74, 0.86),
    M_FRAME:  (0.80, 0.82, 0.86),
    M_WALL:   (0.44, 0.44, 0.48),
    M_SKIN:   (0.86, 0.66, 0.53),
    M_COAT_A: (0.24, 0.44, 0.31),      # dark green
    M_COAT_B: (0.74, 0.35, 0.18),      # rust
    M_COAT_W: (0.24, 0.46, 0.78),      # the one outside. Pale grey read as
                                       # a bollard; blue reads as a coat.
    M_TROUS:  (0.22, 0.24, 0.31),
    M_HAIR:   (0.20, 0.16, 0.15),
    M_BAG:    (0.52, 0.20, 0.22),
    M_DRUM:   (0.64, 0.58, 0.50),
    M_JOINT:  (0.26, 0.27, 0.31),
    M_HAIR2:  (0.80, 0.70, 0.46),
}
MAT_GAIN = {
    M_PAVE: 0.60, M_FLOOR: 0.62, M_DRUM: 1.05, M_JOINT: 0.42, M_HAIR2: 1.15, M_GLASS: 0.66, M_FRAME: 1.34, M_WALL: 0.82,
    M_SKIN: 1.10, M_COAT_A: 1.02, M_COAT_B: 1.06, M_COAT_W: 1.12,
    M_TROUS: 0.92, M_HAIR: 0.80, M_BAG: 1.00,
}
# Only these take a specular highlight -- glass and metal and the polished
# lobby floor. Trap 12: tint with a floor, never multiply the light twice.
MAT_SPEC = {M_GLASS: 0.55, M_FRAME: 0.40, M_FLOOR: 0.22}

STEP_FINE = 0.0165
STEP_MED = 0.024
STEP_COARSE = 0.016       # the GROUND. At 0.036 it was sparser than a cell
STEP_GLASS = 0.056        # sparser than a cell, so glass reads as a hole

G = Grid()
# No text anywhere in this piece, so the safe area does not bind and the
# picture may bleed to all four edges.
G.cy = G.rows / 2.0
G.room_c = G.cols / 2.0 + 1.0
G.room_r = G.rows / 2.0 + 1.0
RAMP = ink_lut()


# ------------------------------------------------------------- projection
def pose(p):
    """World (x east, y north, z up) -> screen (right, DOWN, nearness)."""
    cc, ss = math.cos(YAW), math.sin(YAW)
    se, ce = math.sin(EL), math.cos(EL)
    x1 = p[:, 0] * cc + p[:, 1] * ss
    y1 = -p[:, 0] * ss + p[:, 1] * cc
    return np.stack([x1, -y1 * se - p[:, 2] * ce, p[:, 2] * se - y1 * ce], -1)


# --------------------------------------------------------------- samplers
def _jit(p, step, k=0.38):
    """Trap 10: jitter EVERY sampled surface or it beats against the cells."""
    p = p + RNG.uniform(-k * step, k * step, p.shape)
    return p


def slab(x0, x1, y0, y1, z, step, mat, up=1.0):
    nx = max(2, int((x1 - x0) / step))
    ny = max(2, int((y1 - y0) / step))
    X, Y = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny),
                       indexing="ij")
    p = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], -1)
    p = _jit(p, step)
    p[:, 2] = z
    n = np.zeros_like(p)
    n[:, 2] = up
    return p, n, np.full(len(p), mat, np.int16)


def vplane(x0, x1, y, z0, z1, step, mat, thick=0.030):
    """A flat vertical pane in the x-z plane. Normal points along y."""
    out = []
    nx = max(2, int((x1 - x0) / step))
    nz = max(2, int((z1 - z0) / step))
    X, Z = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(z0, z1, nz),
                       indexing="ij")
    for sgn in (1.0, -1.0):
        p = np.stack([X.ravel(), np.full(X.size, y + sgn * thick / 2.0),
                      Z.ravel()], -1)
        p = _jit(p, step)
        n = np.zeros_like(p)
        n[:, 1] = sgn
        out.append((p, n, np.full(len(p), mat, np.int16)))
    return _cat(out)


def joints(part, pitch, width, mat=None):
    """Retag the samples that fall on a joint line of a paved grid."""
    p, n, m = part
    if mat is None:
        mat = M_JOINT
    dx = np.abs(((p[:, 0] + pitch / 2) % pitch) - pitch / 2)
    dy = np.abs(((p[:, 1] + pitch / 2) % pitch) - pitch / 2)
    m = np.where((dx < width) | (dy < width), mat, m).astype(np.int16)
    return p, n, m


def disc(r, z, step, mat):
    """A flat horizontal disc, evenly sampled by area."""
    n = max(400, int(3.0 * math.pi * r * r / (step * step)))
    rr = r * np.sqrt(RNG.random(n))
    th = RNG.random(n) * 2 * math.pi
    p = np.stack([rr * np.cos(th), rr * np.sin(th), np.full(n, z)], -1)
    nn = np.zeros_like(p)
    nn[:, 2] = 1.0
    return p, nn, np.full(n, mat, np.int16)


def arcwall(r, th0, th1, z0, z1, step, mat, thick=0.055, cap_step=STEP_MED):
    """A curved glass wall: both faces plus the top edge."""
    out = []
    for sgn, rr in ((1.0, r + thick), (-1.0, r)):
        nth = max(3, int(abs(th1 - th0) * rr / step))
        nz = max(2, int((z1 - z0) / step))
        T, Z = np.meshgrid(np.linspace(th0, th1, nth),
                           np.linspace(z0, z1, nz), indexing="ij")
        t, zz = T.ravel(), Z.ravel()
        p = np.stack([rr * np.cos(t), rr * np.sin(t), zz], -1)
        p = _jit(p, step)
        n = np.stack([sgn * np.cos(t), sgn * np.sin(t), np.zeros_like(t)], -1)
        out.append((p, n, np.full(len(p), mat, np.int16)))
    nth = max(3, int(abs(th1 - th0) * r / cap_step))
    nr = max(2, int(thick / cap_step) + 1)
    T, RR = np.meshgrid(np.linspace(th0, th1, nth),
                        np.linspace(r, r + thick, nr), indexing="ij")
    t, rr = T.ravel(), RR.ravel()
    p = np.stack([rr * np.cos(t), rr * np.sin(t), np.full(t.size, z1)], -1)
    p = _jit(p, cap_step)
    p[:, 2] = z1
    n = np.zeros_like(p)
    n[:, 2] = 1.0
    out.append((p, n, np.full(len(p), M_FRAME, np.int16)))
    return _cat(out)


def panel(th, r0, r1, z0, z1, step, mat, thick=0.030):
    """A flat radial panel at angle th -- one wing leaf. Normal tangential."""
    out = []
    tx, ty = -math.sin(th), math.cos(th)          # tangent
    cx, cy = math.cos(th), math.sin(th)           # radial
    nr = max(2, int((r1 - r0) / step))
    nz = max(2, int((z1 - z0) / step))
    R, Z = np.meshgrid(np.linspace(r0, r1, nr), np.linspace(z0, z1, nz),
                       indexing="ij")
    rr, zz = R.ravel(), Z.ravel()
    for sgn in (1.0, -1.0):
        off = sgn * thick / 2.0
        p = np.stack([rr * cx + off * tx, rr * cy + off * ty, zz], -1)
        p = _jit(p, step)
        n = np.tile([sgn * tx, sgn * ty, 0.0], (len(p), 1))
        out.append((p, n, np.full(len(p), mat, np.int16)))
    return _cat(out)


def box(c, half, step, mat):
    """Axis-aligned box surface."""
    c = np.asarray(c, float)
    half = np.asarray(half, float)
    out = []
    for ax in range(3):
        a, b = (ax + 1) % 3, (ax + 2) % 3
        na = max(2, int(2 * half[a] / step))
        nb = max(2, int(2 * half[b] / step))
        A, B = np.meshgrid(np.linspace(-half[a], half[a], na),
                           np.linspace(-half[b], half[b], nb), indexing="ij")
        for sgn in (1.0, -1.0):
            p = np.zeros((A.size, 3))
            p[:, a] = A.ravel()
            p[:, b] = B.ravel()
            p[:, ax] = sgn * half[ax]
            n = np.zeros_like(p)
            n[:, ax] = sgn
            p = _jit(p + c, step)
            out.append((p, n, np.full(len(p), mat, np.int16)))
    return _cat(out)


def taper(a, b, ra, rb, step, mat, wide=1.0):
    """Tapered capsule from a to b, with hemispherical caps.

    `wide` stretches the cross-section along local x, which is what turns a
    tube into a torso -- people are wider than they are deep.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ax = b - a
    L = np.linalg.norm(ax)
    ax = ax / L
    up = np.array([0.0, 0.0, 1.0])
    if abs(ax @ up) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(ax, up)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax, e1)
    nt = max(6, int(2 * math.pi * max(ra, rb) * wide / step))
    nl = max(2, int(L / step))
    T, S = np.meshgrid(np.linspace(0, 2 * math.pi, nt, endpoint=False),
                       np.linspace(0.0, 1.0, nl), indexing="ij")
    t, s = T.ravel(), S.ravel()
    r = ra + (rb - ra) * s
    # local offset, stretched along e1
    ox = np.cos(t) * wide
    oy = np.sin(t)
    d = ox[:, None] * e1 + oy[:, None] * e2
    p = a + (L * s)[:, None] * ax + r[:, None] * d
    n = ox[:, None] / (wide ** 2) * e1 + oy[:, None] * e2
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    p = _jit(p, step)
    out = [(p, n, np.full(len(p), mat, np.int16))]
    for end, rad, sgn in ((a, ra, -1.0), (b, rb, 1.0)):
        pc, nc = _dome(rad, step, wide)
        pw = end + pc[:, 0:1] * e1 + pc[:, 1:2] * e2 + sgn * pc[:, 2:3] * ax
        nw = nc[:, 0:1] * e1 + nc[:, 1:2] * e2 + sgn * nc[:, 2:3] * ax
        out.append((_jit(pw, step), nw, np.full(len(pw), mat, np.int16)))
    return _cat(out)


def _dome(r, step, wide=1.0):
    n = max(60, int(2.2 * (2 * math.pi * r / step) ** 2))
    u = RNG.random(n)
    v = RNG.random(n)
    th = 2 * math.pi * u
    ph = np.arccos(v)                      # upper hemisphere only
    x = np.sin(ph) * np.cos(th)
    y = np.sin(ph) * np.sin(th)
    z = np.cos(ph)
    p = np.stack([r * wide * x, r * y, r * z], -1)
    nn = np.stack([x / wide, y, z], -1)
    nn /= np.linalg.norm(nn, axis=1, keepdims=True)
    return p, nn


def sphere(c, r, step, mat, wide=1.0, squash=1.0):
    n = max(120, int(3.0 * (2 * math.pi * r / step) ** 2))
    u = RNG.random(n) * 2 - 1
    th = RNG.random(n) * 2 * math.pi
    s = np.sqrt(1 - u * u)
    x, y, z = s * np.cos(th), s * np.sin(th), u
    p = np.stack([c[0] + r * wide * x, c[1] + r * y, c[2] + r * squash * z], -1)
    nn = np.stack([x / wide, y, z / squash], -1)
    nn /= np.linalg.norm(nn, axis=1, keepdims=True)
    return _jit(p, step), nn, np.full(n, mat, np.int16)


def _cat(parts):
    p = np.concatenate([q[0] for q in parts])
    n = np.concatenate([q[1] for q in parts])
    m = np.concatenate([q[2] for q in parts])
    return p, n, m


# ----------------------------------------------------------------- people
def person(coat, hair, bag, height=1.72, build=1.0):
    """One body, built in local coords: x = right, y = facing, z = up."""
    h = height
    parts = []
    # legs as one mass -- from directly above two legs read as one anyway
    parts.append(taper((0, 0, 0.02), (0, 0, 0.56 * h), 0.115 * build,
                       0.135 * build, STEP_MED, M_TROUS, wide=1.30))
    # torso
    parts.append(taper((0, 0, 0.55 * h), (0, 0, 0.82 * h), 0.140 * build,
                       0.168 * build, STEP_FINE, coat, wide=1.38))
    # shoulders
    parts.append(taper((-0.185 * build, 0, 0.815 * h),
                       (0.185 * build, 0, 0.815 * h),
                       0.078, 0.078, STEP_FINE, coat, wide=1.0))
    # arms
    for sgn in (-1.0, 1.0):
        parts.append(taper((sgn * 0.205 * build, 0.005, 0.805 * h),
                           (sgn * 0.185 * build, 0.055, 0.545 * h),
                           0.055, 0.043, STEP_FINE, coat, wide=1.0))
    # neck, head, hair cap
    parts.append(taper((0, 0, 0.83 * h), (0, 0.005, 0.875 * h),
                       0.052, 0.050, STEP_FINE, M_SKIN, wide=1.0))
    parts.append(sphere((0, 0.004, 0.925 * h), 0.098, STEP_FINE, M_SKIN,
                        wide=1.02, squash=1.16))
    pc, nc = _dome(0.101, STEP_FINE)
    hp = np.stack([pc[:, 0], pc[:, 1], pc[:, 2] * 1.16], -1)
    hp = hp + np.array([0.0, 0.004, 0.925 * h])
    keep = hp[:, 2] > 0.925 * h + 0.010
    parts.append((_jit(hp[keep], STEP_FINE), nc[keep],
                  np.full(keep.sum(), hair, np.int16)))
    if bag:
        parts.append(box((0.20 * build, -0.055, 0.60 * h),
                         (0.075, 0.048, 0.105), STEP_FINE, M_BAG))
        parts.append(taper((0.115 * build, -0.02, 0.80 * h),
                           (0.205 * build, -0.05, 0.66 * h),
                           0.018, 0.018, STEP_FINE, M_BAG, wide=1.0))
    return _cat(parts)


def place(part, cx, cy, facing, lift=0.0):
    """Put a locally-built body into the world facing `facing` radians."""
    p, n, m = part
    f = np.array([math.cos(facing), math.sin(facing), 0.0])
    r = np.array([f[1], -f[0], 0.0])          # right of facing
    z = np.array([0.0, 0.0, 1.0])
    M = np.stack([r, f, z])                   # rows: local x,y,z -> world
    return (p @ M + np.array([cx, cy, lift]), n @ M, m)


# ---------------------------------------------------------- the assemblies
#
# THE TWO PEOPLE. Compartment k gets TYPE[k]. For the half-turn symmetry to
# hold -- and it is the only reason this loops -- compartment k and
# compartment k+2 must be the same person. So there are two.
TYPE = (0, 1, 0, 1)
BODY = (dict(coat=M_COAT_A, hair=M_HAIR, bag=True, height=1.74, build=1.00),
        dict(coat=M_COAT_B, hair=M_HAIR2, bag=False, height=1.63, build=0.94))


def mirror(part):
    """Exact half turn: (x, y) -> (-x, -y). No trig, so no rounding.

    This is why it exists. Every sampled surface here is dithered off a
    random seed (trap 10), and a random dither is NOT symmetric -- so a door
    whose four compartments were each generated independently would not
    survive its own half turn, and the loop would show a seam. Build half
    the machine and negate it, and the symmetry is exact down to the noise.
    """
    p, n, m = part
    return (np.stack([-p[:, 0], -p[:, 1], p[:, 2]], -1),
            np.stack([-n[:, 0], -n[:, 1], n[:, 2]], -1), m)


def build_rotor_half():
    """Two wings and two bodies -- one half turn's worth of door.

    Returns a fourth array, the body id: -1 for the machine, k for the body
    riding compartment k. The checks use it to ask exactly how much of each
    person survives the z-buffer, which beats guessing with a cylinder.
    """
    bodies = {}
    parts = [taper((0, 0, 0.0), (0, 0, H_WING), R_SHAFT, R_SHAFT,
                   STEP_MED, M_FRAME, wide=1.0)]
    for k in range(N_WING // 2):
        th = k * 2 * math.pi / N_WING
        # glass leaf, sparse -- a revolving door is mostly nothing
        parts.append(panel(th, R_SHAFT + 0.02, R_IN - 0.012, 0.30,
                           H_WING - 0.09, STEP_GLASS, M_GLASS))
        # frame: outer stile, bottom rail, top rail, one mid rail
        parts.append(panel(th, R_IN - 0.012 - W_STILE, R_IN - 0.012,
                           0.03, H_WING, STEP_MED, M_FRAME, thick=0.042))
        for z0, z1 in ((0.03, 0.30), (H_WING - 0.09, H_WING), (1.02, 1.06)):
            parts.append(panel(th, R_SHAFT + 0.02, R_IN - 0.012, z0, z1,
                               STEP_MED, M_FRAME, thick=0.042))
        phi = (k + 0.5) * 2 * math.pi / N_WING
        parts.append(place(person(**BODY[TYPE[k]]),
                           R_PERSON * math.cos(phi),
                           R_PERSON * math.sin(phi), phi + math.pi / 2.0))
        bodies[len(parts) - 1] = k
    p, n, m = _cat(parts)
    bid = np.concatenate([np.full(len(q[0]), bodies.get(i, -1), np.int8)
                          for i, q in enumerate(parts)])
    return p, n, m, bid


def build_rotor():
    p, n, m, b = build_rotor_half()
    mp, mn, _ = mirror((p, n, m))
    mb = np.where(b >= 0, b + N_WING // 2, -1).astype(np.int8)
    return (np.concatenate([p, mp]), np.concatenate([n, mn]),
            np.concatenate([m, m]), np.concatenate([b, mb]))


def build_static():
    """Everything that does not turn."""
    parts = []
    # Pavement and lobby floor, generated past the frame so no edge shows.
    # Both get their joints marked as a separate material: a bare horizontal
    # plane under a high lamp returns one glyph across its whole face and
    # reads as gauze (trap 9), and the joints also give the sweeping shadows
    # something to be measured against. 600 mm is a normal paving slab.
    parts.append(joints(slab(GX0, GX1, GY0, -0.02, 0.0, STEP_COARSE, M_PAVE),
                        0.600, 0.013))
    parts.append(joints(slab(GX0, GX1, 0.02, GY1, 0.0, STEP_COARSE, M_FLOOR),
                        0.900, 0.010))
    # the drum: two glass walls, each 120 deg. Sampled SPARSER than a cell so
    # you can see the people through them -- at STEP_MED the walls were
    # opaque and ate whichever body was behind them.
    for mid in WALL_MID:
        parts.append(arcwall(R_IN, mid - WALL_ARC / 2.0, mid + WALL_ARC / 2.0,
                             0.0, H_WALL, STEP_GLASS, M_GLASS))
    # facade: a plinth and mullions either side of the drum, running off frame
    for sgn in (-1.0, 1.0):
        x0 = sgn * (R_IN + 0.10)
        x1 = sgn * (max(abs(GX0), abs(GX1)) + 0.4)
        lo, hi = min(x0, x1), max(x0, x1)
        parts.append(box(((lo + hi) / 2.0, 0.0, 0.19),
                         ((hi - lo) / 2.0, 0.075, 0.19), STEP_MED, M_WALL))
        parts.append(box(((lo + hi) / 2.0, 0.0, 2.44),
                         ((hi - lo) / 2.0, 0.075, 0.09), STEP_MED, M_WALL))
        n_mull = 3
        for i in range(n_mull):
            xm = lo + (hi - lo) * (i + 0.5) / n_mull
            parts.append(box((xm, 0.0, 1.30), (0.042, 0.070, 0.92),
                             STEP_MED, M_FRAME))
        # shopfront glazing: a VERTICAL pane. The first version of this was a
        # horizontal slab at z=1.30 -- a sheet of glass lying in mid-air.
        parts.append(vplane(lo, hi, 0.0, 0.38, 2.35, STEP_GLASS, M_GLASS))
    # THE DRUM FLOOR. One solid disc for the cross and the four people to
    # stand on. Without it the door had no form at all -- just stipple in
    # front of stipple -- and the whole frame read as murk. Lifted 5 mm clear
    # of the pavement it sits on, per trap 8, or the two planes z-fight per
    # cell and the disc renders as dashes.
    parts.append(disc(R_IN - 0.005, 0.005, STEP_COARSE, M_DRUM))
    return _cat(parts)


# The one outside. Static except for a weight shift whose period is exactly
# the loop, so they are alive without breaking it.
WAIT_XY = (1.02, -1.78)
WAIT_BODY = dict(coat=M_COAT_W, hair=M_HAIR, bag=False, height=1.79,
                 build=1.05)


_WAIT_REST = None


def build_waiter(f):
    """Built ONCE and then moved. Rebuilding it per frame re-rolled the
    dither on 44,000 points every frame, which put random noise across the
    whole figure and broke the loop before anything else could."""
    global _WAIT_REST
    if _WAIT_REST is None:
        _WAIT_REST = place(person(**WAIT_BODY), 0.0, WAIT_XY[1],
                           math.pi / 2.0)
    ph = 2 * math.pi * (f % HALF) / HALF
    lean = 0.052 * math.sin(ph)
    shift = 0.035 * math.sin(ph)
    p, n, m = _WAIT_REST
    p = p + np.array([WAIT_XY[0] + shift, 0.0, 0.0])
    # rock about the feet
    c, s = math.cos(lean), math.sin(lean)
    x, z = p[:, 0] - (WAIT_XY[0] + shift), p[:, 2]
    p = np.stack([(WAIT_XY[0] + shift) + x * c - z * s, p[:, 1],
                  x * s + z * c], -1)
    n = np.stack([n[:, 0] * c - n[:, 2] * s, n[:, 1],
                  n[:, 0] * s + n[:, 2] * c], -1)
    return p, n, m


def _cs(ang):
    """cos/sin, snapped at the quarter turns.

    math.sin(math.pi) is 1.22e-16, not zero, so a half turn built with raw
    trig lands every point about 1e-16 off where it belongs. That is
    physically nothing and it still flipped exactly one cell of 17,052 across
    a rounding boundary -- enough to break the one claim this piece makes. A
    half turn is a half turn.
    """
    q = ang / (math.pi / 2.0)
    if abs(q - round(q)) < 1e-9:
        c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][int(round(q)) % 4]
        return float(c), float(s)
    return math.cos(ang), math.sin(ang)


def spin(part, ang):
    p, n, m = part
    c, s = _cs(ang)
    return (np.stack([p[:, 0] * c - p[:, 1] * s,
                      p[:, 0] * s + p[:, 1] * c, p[:, 2]], -1),
            np.stack([n[:, 0] * c - n[:, 1] * s,
                      n[:, 0] * s + n[:, 1] * c, n[:, 2]], -1),
            m)


# ------------------------------------------------------------------ camera
#
# Set by hand rather than fitted. `fit` on a bounding box was width-limited
# and left twelve rows of slack at top and bottom, which is a visible edge --
# the picture floated in a rectangle of background. SCALE is chosen so the
# 2.44 m drum fills about 72 of 98 columns.
SCALE = 34.0
CAM = Camera(G)
CAM.off = np.array([0.0, -0.205])
CAM.scale = SCALE


def frame_box(pad=1.2):
    """Invert the projection at z=0 to find the world rectangle the camera
    can see, so the ground can be generated with no edge in shot."""
    cc, ss = math.cos(YAW), math.sin(YAW)
    se = math.sin(EL)
    xs, ys = [], []
    for c in (-2.0, G.cols + 2.0):
        for r in (-2.0, G.rows + 2.0):
            sx = (c - G.cx) / CAM.scale + CAM.off[0]
            sy = (r - G.cy) / CAM.scale + CAM.off[1]
            x1, y1 = sx, -sy / se
            xs.append(x1 * cc - y1 * ss)
            ys.append(x1 * ss + y1 * cc)
    return (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)


GX0, GX1, GY0, GY1 = frame_box()

ROTOR = build_rotor()
STATIC = build_static()

# Trap 24: anchor the depth cue to FIXED bounds, or anything entering or
# leaving silently rescales the brightness of the whole frame.
_allz = np.concatenate([pose(STATIC[0])[:, 2], pose(ROTOR[0])[:, 2],
                        pose(build_waiter(0)[0])[:, 2]])
Z_LO, Z_HI = float(_allz.min()), float(_allz.max())


def fixed_cue(z, near=1.0, far=0.94):
    return far + (near - far) * ((z - Z_LO) / (Z_HI - Z_LO))


# ----------------------------------------------------------------- shadows
SH_STEP = 0.030
SH_X0, SH_Y0 = GX0 - 0.6, GY0 - 0.6
SH_NX = int((GX1 - GX0 + 1.2) / SH_STEP)
SH_NY = int((GY1 - GY0 + 1.2) / SH_STEP)


def shadow_map(casters):
    g = np.zeros((SH_NX, SH_NY), bool)
    for p in casters:
        # NO SUBSAMPLING. `[::2]` here took every second point IN ARRAY ORDER,
        # and a half turn is a permutation of the same points -- so the two
        # halves of the revolution got shadows cast from different halves of
        # the door. That, not rounding, was the seam.
        q = p[p[:, 2] > 0.05]
        t = q[:, 2] / SUN[2]
        gx = q[:, 0] - t * SUN[0]
        gy = q[:, 1] - t * SUN[1]
        i = ((gx - SH_X0) / SH_STEP).astype(np.int32)
        j = ((gy - SH_Y0) / SH_STEP).astype(np.int32)
        ok = (i >= 0) & (i < SH_NX) & (j >= 0) & (j < SH_NY)
        g[i[ok], j[ok]] = True
    for _ in range(2):
        g = np.maximum.reduce([g, np.roll(g, 1, 0), np.roll(g, -1, 0),
                               np.roll(g, 1, 1), np.roll(g, -1, 1)])
    return g


def in_shadow(g, p):
    i = np.clip(((p[:, 0] - SH_X0) / SH_STEP).astype(np.int32), 0, SH_NX - 1)
    j = np.clip(((p[:, 1] - SH_Y0) / SH_STEP).astype(np.int32), 0, SH_NY - 1)
    return g[i, j]


# -------------------------------------------------------------------- draw
def colour(sh, mat):
    base = MAT_RGB[int(mat)]
    k = 0.42 + 0.58 * sh
    return (base[0] * k, base[1] * k, base[2] * k)


def scene(f, with_waiter=True, with_person=None, fold=True):
    """Assemble the world for frame f. `with_person` drops one body for the
    controlled visibility diff (trap 25: same frame, one variable).

    `fold=False` advances the door by the honest angle instead of folding it
    back into the first half turn. The render folds, because the second half
    of the revolution is the first half; the check does NOT, because that is
    the thing being tested."""
    ang = 2 * math.pi * ((f % HALF) if fold else f) / N_FRAME
    rp, rn, rm = spin(ROTOR[:3], ang)
    parts = [(rp, rn, rm), STATIC]
    bids = [ROTOR[3], np.full(len(STATIC[0]), -1, np.int8)]
    if with_waiter:
        w = build_waiter(f)
        parts.append(w)
        bids.append(np.full(len(w[0]), -1, np.int8))
    p, n, m = _cat(parts)
    b = np.concatenate(bids)
    if with_person is not None:
        keep = b != with_person
        p, n, m, b = p[keep], n[keep], m[keep], b[keep]
    return p, n, m, b


def draw(f, sm_override=None, want_vis=False, **kw):
    p, n, m, bid = scene(f, **kw)
    sm = (shadow_map([p[(m != M_PAVE) & (m != M_FLOOR)]])
          if sm_override is None else sm_override)
    lit = ~in_shadow(sm, p) | (p[:, 2] > 0.06)

    key = lambert(n, SUN) * lit
    fill = lambert(n, LOBBY)
    # the lobby glows; the street does not. north of the facade is inside.
    inside = np.clip((p[:, 1] + 0.15) * 1.9, 0.0, 1.0)
    amb = 0.175 + 0.135 * inside

    # The lobby fill used to run to 0.78 on an up-facing floor, which put
    # the brightest thing in the picture on the emptiest surface in it.
    shade = amb + 0.60 * key + (0.12 + 0.34 * inside) * fill
    spec = np.zeros(len(p))
    for mat, g in MAT_SPEC.items():
        sel = m == mat
        if sel.any():
            spec[sel] = g * specular(n[sel], SUN, 26) * lit[sel]
    shade = shade + spec
    gain = np.array([MAT_GAIN[int(x)] for x in m])
    sp = pose(p)
    shade = shade * gain * fixed_cue(sp[:, 2])

    col, row, z = CAM.project(sp)
    ok = visible(G, col, row)
    col, row, z, shade, m, bid = (col[ok], row[ok], z[ok], shade[ok],
                                  m[ok], bid[ok])
    flat, keep = zbuffer(G, col, row, z)

    # CANONICAL TIE-BREAK. `field` writes cells in array order, so when two
    # samples land in one cell at exactly equal z the winner depends on where
    # they happen to sit in the array. The half turn is a PERMUTATION of the
    # same point set, so five cells came out different purely from that.
    # Pick the winner by value, not by position, and the order stops mattering.
    ki = np.nonzero(keep)[0]
    order = np.lexsort((m[ki], shade[ki], z[ki], flat[ki]))
    ki = ki[order]
    fk = flat[ki]
    last = np.ones(len(ki), bool)
    last[:-1] = fk[1:] != fk[:-1]
    keep = np.zeros(len(z), bool)
    keep[ki[last]] = True

    if want_vis:
        # How much of each body survives the z-buffer, as a fraction of the
        # cells it would cover with nothing in front of it. Geometry only,
        # no rendering -- cheap enough to sweep the whole loop.
        out = {}
        for k in range(N_WING):
            s = bid == k
            if not s.any():
                continue
            foot = len(set(zip(col[s].tolist(), row[s].tolist())))
            won = len(set(zip(col[s & keep].tolist(), row[s & keep].tolist())))
            out[k] = (won, foot)
        return out

    fr = Frame(G, BG)
    idx, val = fr.field(col, row, keep, shade, colour, RAMP,
                        extra=m.astype(float))
    return fr, idx, val, keep, bid, col, row


# ------------------------------------------------------------------- check
def check():
    ok = True

    def say(name, good, detail=""):
        nonlocal ok
        ok = ok and good
        print(("  PASS  " if good else "  FAIL  ") + name +
              (("   " + detail) if detail else ""))

    print("THE DOOR -- checks")

    # 1. the runtime is not a choice
    say("8-0 ft manual door tops out at 10 rpm (IBC 1010.3.1(1))",
        RPM == 10.0 and abs(DIAM_FT * 0.3048 - 2.4384) < 1e-9,
        "%.4f m diameter" % (DIAM_FT * 0.3048))
    say("one revolution == the whole video",
        abs(N_FRAME / FPS - SEC_PER_REV) < 1e-9,
        "%d frames / %d fps = %.3f s" % (N_FRAME, FPS, N_FRAME / FPS))

    # 2. the door is always sealed: every 120 deg wall always holds a wing
    worst = 99
    for i in range(2000):
        a = 2 * math.pi * i / 2000
        for mid in WALL_MID:
            cnt = 0
            for k in range(N_WING):
                d = (a + k * math.pi / 2 - mid + math.pi) % (2 * math.pi) - math.pi
                if abs(d) <= WALL_ARC / 2.0:
                    cnt += 1
            worst = min(worst, cnt)
    say("a wing is in contact with each drum wall at every angle",
        worst >= 1, "worst case %d wing(s), throat %.3f m" % (worst, THROAT))

    # 3. the half-turn is a real symmetry of the WHOLE scene
    say("compartment k and k+2 are the same person",
        all(TYPE[k] == TYPE[(k + 2) % N_WING] for k in range(N_WING)),
        "types " + str(TYPE) + " -> %d distinct" % len(set(TYPE)))

    # 4. THE LOOP. Advance the door the honest half turn -- no folding -- and
    #    compare every cell of both the glyph plane and the value plane.
    _, i0, v0 = draw(0, fold=False)[:3]
    _, i1, v1 = draw(HALF, fold=False)[:3]
    dif = int((i0 != i1).sum())
    say("frame 0 and frame 90 are the same picture, cell for cell",
        dif == 0, "%d of %d cells differ" % (dif, i0.size))
    say("...and the same brightness", float(np.abs(v0 - v1).max()) < 1e-12,
        "max delta %.2e" % float(np.abs(v0 - v1).max()))

    # 5. the loop is not trivial -- consecutive frames must actually differ
    _, ia, _ = draw(0)[:3]
    _, ib, _ = draw(1)[:3]
    _, ic, _ = draw(HALF // 2)[:3]
    d1 = int((ia != ib).sum())
    d2 = int((ia != ic).sum())
    say("consecutive frames differ", d1 > 300, "%d cells" % d1)
    # A quarter turn maps the four wings onto themselves, so the only thing
    # that can change is which of the two people is where. If this number is
    # small, the two people are not distinguishable and the joke is invisible.
    say("a quarter turn is NOT a symmetry -- the two people differ",
        d2 > 800, "%d cells (%.0f%% of the frame)"
        % (d2, 100.0 * d2 / ia.size))

    # 6. the one outside is alive and still loops
    wa = build_waiter(0)[0]
    wb = build_waiter(HALF // 4)[0]
    wc = build_waiter(HALF)[0]
    say("the waiting figure moves", float(np.abs(wa - wb).max()) > 0.012,
        "%.1f cm" % (100 * float(np.abs(wa - wb).max())))
    say("the waiting figure returns exactly",
        float(np.abs(wa - wc).max()) < 1e-12)

    # 7. every drawable cell carries geometry -- no visible edge anywhere
    fr, idx, _ = draw(11)[:3]
    bare = int((idx[1:G.rows - 1] == 0).sum())
    say("no bare background", bare == 0,
        "%d bare of %d" % (bare, idx[1:G.rows - 1].size))

    # 8. the subject is inside the frame
    p, _, m, _ = scene(11)
    sel = (np.hypot(p[:, 0], p[:, 1]) < R_IN + 0.12) & (p[:, 2] > 0.02)
    c, r, _ = CAM.project(pose(p[sel]))
    say("the whole door is in frame",
        int(c.min()) >= 0 and int(c.max()) < G.cols and int(r.min()) >= 1
        and int(r.max()) < G.rows - 1,
        "cols %d..%d rows %d..%d" % (c.min(), c.max(), r.min(), r.max()))

    # 9. CONTROLLED VISIBILITY (trap 25): same frame, drop one body only.
    #    A person you cannot see is the failure mode this piece cannot
    #    afford, and it is the one no other check can see. Measured against
    #    the body's OWN projected footprint, not a number I picked.
    # 9a. OCCLUSION SWEEP, geometry only. The far half of the door is behind
    #     the shaft and two wings, so nobody is 100% visible all the way
    #     round -- but nobody may vanish either, and each person has to come
    #     fully into the clear at some point in the loop.
    seen = {k: [] for k in range(N_WING)}
    for f in range(0, HALF, 6):
        for k, (won, foot) in draw(f, want_vis=True).items():
            seen[k].append(won / max(foot, 1))
    for k in range(N_WING):
        lo, hi = min(seen[k]), max(seen[k])
        say("body %d never disappears" % k, lo > 0.28,
            "visible %.0f%%..%.0f%% of its own footprint over the loop"
            % (100 * lo, 100 * hi))
    for t in sorted(set(TYPE)):
        best = max(max(seen[k]) for k in range(N_WING) if TYPE[k] == t)
        say("person %d comes fully into the clear" % t, best > 0.85,
            "peaks at %.0f%%" % (100 * best))

    # 9b. CONTRAST (trap 25): the body must actually CHANGE the cells it
    #     wins. A body that z-buffers correctly and renders the same shade as
    #     what is behind it passes every other check and is invisible -- that
    #     is exactly how fifty cars came out darker than their own car park.
    #     Shadow map held FIXED across the pair, or dropping a body also
    #     deletes its long shadow and the diff scores 198% with the centroid
    #     twenty cells adrift.
    sc = scene(24)
    SM = shadow_map([sc[0][(sc[2] != M_PAVE) & (sc[2] != M_FLOOR)]])
    vis = draw(24, sm_override=SM, want_vis=True)
    _, ion, _ = draw(24, sm_override=SM)[:3]
    for k in range(N_WING):
        _, ioff, _ = draw(24, sm_override=SM, with_person=k)[:3]
        d = (ion != ioff)
        n_ch = int(d.sum())
        won, foot = vis[k]
        rr, cc = np.nonzero(d)
        ang = 2 * math.pi * 24 / N_FRAME
        phi = (k + 0.5) * 2 * math.pi / N_WING + ang
        ec, er, _ = CAM.project(pose(np.array(
            [[R_PERSON * math.cos(phi), R_PERSON * math.sin(phi), 0.9]])))
        off = math.hypot(cc.mean() - ec[0], rr.mean() - er[0])
        say("body %d reads against what is behind it" % k,
            n_ch >= 0.80 * won and off < 9.0,
            "changes %d of the %d cells it wins (%.0f%%), centroid %.1f off"
            % (n_ch, won, 100.0 * n_ch / max(won, 1), off))

    # 10. the lobby must read brighter than the street
    _, _, val = draw(11)[:3]
    half = int(G.rows * 0.42)
    top = val[1:half][val[1:half] > 0].mean()
    bot = val[half:G.rows - 1][val[half:G.rows - 1] > 0].mean()
    say("the lobby is brighter than the pavement", top > 1.25 * bot,
        "%.3f vs %.3f = %.2fx" % (top, bot, top / bot))

    print("\n" + ("ALL CHECKS PASS" if ok else "CHECKS FAILED"))
    return ok


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--out", default="/tmp/door.mp4")
    a = ap.parse_args()

    if a.check:
        sys.exit(0 if check() else 1)

    if a.sheet:
        fs = [draw(f)[0] for f in (0, 11, 22, 34, 45, 56, 68, 79, 89)]
        contact(fs, "/tmp/door_sheet.png", cols=3,
                labels=["f%d" % f for f in (0, 11, 22, 34, 45, 56, 68, 79, 89)])
        print("/tmp/door_sheet.png")
        return

    # The second half of the revolution is the first half again -- that is the
    # whole claim of the piece -- so there is nothing to compute for it. Hold
    # the raw frame BYTES rather than 90 live cairo surfaces (which is 750 MB)
    # and push them through the encoder a second time.
    print("rendering %d distinct frames, writing %d" % (HALF, N_FRAME))
    buf = []
    with Encoder(a.out, G, fps=FPS, crf=18) as enc:
        for f in range(HALF):
            fr = draw(f)[0]
            raw = bytes(fr.surface.get_data())
            buf.append(raw)
            enc.proc.stdin.write(raw)
            if f % 10 == 0:
                print("  %d/%d" % (f, HALF), flush=True)
        for raw in buf:
            enc.proc.stdin.write(raw)
    print(a.out, os.path.getsize(a.out), "bytes")


if __name__ == "__main__":
    main()
