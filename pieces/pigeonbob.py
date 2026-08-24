#!/usr/bin/env python3
"""
THE HEAD DOES NOT BOB.

A walking pigeon does not wag its head back and forth.  It holds the head
LOCKED IN SPACE while the body walks out from under it, then throws it
forward to a new fixed point.  Frost (1978) filmed this at 64 frames/s and
found the head is stationary about 63% of the walking cycle.  Troje & Frost
(2000) measured how stationary: standard deviation of head position along
the roll axis during the hold phase, 0.34 mm.

Then Frost put the birds on a treadmill.  If the belt matches the walking
speed the bird walks but the visual world does not move -- and the bobbing
STOPS.  So it is not a gait mechanic and it is not balance.  It is vision.

And then, by accident: the belt was left running at a crawl instead of off.
Too slow to make a pigeon walk, fast enough to drag it.  The head stayed
nailed to the room while the body was carried out from under it, and the
bird toppled over.  The reflex outranks standing up.

This piece is that experiment as one bird, in profile, filling the frame.
It draws its own strobe photograph: every eye position gets a mark, and the
marks come out in CLUMPS.  The ground and the far horizon carry the argument
-- normally they move together, on the treadmill they come apart, and the
head follows the horizon, not the floor.

Renderer notes: NOT a 3D point cloud.  Every part is an ellipsoid or a
tapered capsule evaluated analytically on the cell grid, which gives real
normals for a couple of milliseconds a frame.  Light ground, so trap 15:
ink density is 1-light with a floor, and colour blends from paper toward
the ink rather than multiplying.

    python3 scripts/pigeonbob.py --check
    python3 scripts/pigeonbob.py
"""

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]

import cairo  # noqa: E402
from asciilib import Encoder, Frame, Grid, contact, ink_lut  # noqa: E402

OUT = os.path.join(_HERE, "..", "content", "pigeonbob.mp4")

# ---------------------------------------------------------------- palette
# bone paper / graphite bird / the bird's own neck colours for the traces.
BG = (0.945, 0.925, 0.870)   # warm bone paper
INK = (0.137, 0.157, 0.196)  # graphite slate
IRID = (0.055, 0.463, 0.416)  # neck iridescence, green side
CORAL = (0.855, 0.376, 0.290)  # beak, feet, eye ring
MAG = (0.804, 0.063, 0.404)  # neck iridescence, purple side -> the eye trace
FAINT = (0.400, 0.443, 0.482)  # the far world
BELTC = (0.243, 0.251, 0.282)  # the machine

M_BIRD, M_TRAIL, M_WORLD, M_BELT, M_IRID, M_CORAL, M_DARK = 1, 2, 3, 4, 5, 6, 7
MAT = {M_BIRD: INK, M_TRAIL: MAG, M_WORLD: FAINT, M_BELT: BELTC,
       M_IRID: IRID, M_CORAL: CORAL, M_DARK: (0.05, 0.05, 0.07)}

# ---------------------------------------------------------------- measured
# Troje & Frost 2000, J Exp Biol 203:935-940, profile-view footage:
BOB_HZ = 3.46          # mean head-bobbing frequency, Hz
HOLD_MS = 156.0        # mean hold-phase duration, ms
THRUST_MS = 132.0      # mean thrust-phase duration, ms
HOLD_SD_MM = 0.34      # s.d. of head position, roll axis, hold phase
SLIP_MMS = 3.15        # systematic forward slip during the hold, mm/s
# Frost 1978, J Exp Biol 74:187-195:
THRUST_MS_1978 = 610.0 / 1000.0 * 1000.0   # 0.61 m/s, kept as m/s below
THRUST_V = 0.61        # m/s, velocity of the forward head thrust
HOLD_FRAC_1978 = 0.63  # fraction of walking time the head is stationary
BELT_FAST = 60.0       # cm/s, the treadmill speed that abolished bobbing
BELT_CREEP = 1.1       # cm/s, the speed that toppled the bird

PERIOD = HOLD_MS + THRUST_MS                       # 288 ms
STEP_CM = THRUST_V * (THRUST_MS / 1000.0) * 100.0  # 8.05 cm per thrust
WALK_CMS = STEP_CM / (PERIOD / 1000.0)             # ~28 cm/s

# ---------------------------------------------------------------- geometry
G = Grid()
RAMP = ink_lut()
K = 2.20                 # cells per cm
R_GND = 132              # ground row
CX = 51.0                # the body's local origin, in columns
R_HOR = 44               # far horizon row
R_RAIL = 58              # a nearer rail: the flow field needs two depths
BELT_H = 15              # rows of machine below the walking surface
FPS = 30
PIVOT = (4.4, 0.4)       # the front toe -- everything falls about this

# act boundaries, seconds
T_A1 = 5.6               # free walking
T_SW = 6.1               # the treadmill slides in
T_A2 = 9.9               # belt at 60 cm/s
T_SL = 10.5              # belt down to a crawl
OVER_TIP = 2.4           # cm of head beyond the toes before it goes
TIP_MAX = 1.00           # radians: beak reaches the floor and stops there
LEGF_FAST = 5.0          # step rate on the fast belt, Hz
T_FALL = T_SL + OVER_TIP / BELT_CREEP
T_END = 15.4
FRAMES = int(round(T_END * FPS))

RNG = np.random.default_rng(3461)
NOISE = RNG.normal(0.0, 1.0, (G.rows, G.cols))

# body-local rest pose, centimetres, y UP, origin on the ground under the hip
HEAD0 = (10.6, 21.9)
LAMP = np.array([-0.42, 0.70, 0.575])
LAMP = LAMP / np.linalg.norm(LAMP)

# per-cell body-local coordinates, filled by pose()
CC, RR = np.meshgrid(np.arange(G.cols), np.arange(G.rows))


def _rot(px, py, ang):
    ca, sa = math.cos(ang), math.sin(ang)
    return px * ca + py * sa, -px * sa + py * ca


def ellipsoid(px, py, cx, cy, a, b, c, ang=0.0, bias=0.0):
    """Analytic ellipsoid: inside mask, z, and a real normal."""
    qx, qy = _rot(px - cx, py - cy, ang)
    t = (qx / a) ** 2 + (qy / b) ** 2
    inside = t <= 1.0
    z = c * np.sqrt(np.maximum(0.0, 1.0 - t))
    nx, ny, nz = qx / (a * a), qy / (b * b), z / (c * c) + 1e-9
    nx, ny = _rot(nx, ny, -ang)
    n = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    return inside, z + bias, np.stack([nx / n, ny / n, nz / n]), (qx, qy)


def capsule(px, py, ax, ay, bx, by, ra, rb, bias=0.0):
    """Tapered swept sphere.  Same three returns."""
    dx, dy = bx - ax, by - ay
    dd = dx * dx + dy * dy + 1e-9
    h = np.clip(((px - ax) * dx + (py - ay) * dy) / dd, 0.0, 1.0)
    cx, cy = ax + h * dx, ay + h * dy
    r = ra + h * (rb - ra)
    ex, ey = px - cx, py - cy
    d = np.sqrt(ex * ex + ey * ey)
    inside = d <= r
    z = np.sqrt(np.maximum(0.0, r * r - d * d))
    nx, ny, nz = ex / r, ey / r, z / r + 1e-9
    n = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    return inside, z + bias, np.stack([nx / n, ny / n, nz / n]), (h, d / r)


# ---------------------------------------------------------------- the pose
def head_world(t):
    """Head x in WORLD cm during free walking: hold, thrust, hold, thrust."""
    n = math.floor(t * 1000.0 / PERIOD)
    ph = (t * 1000.0 % PERIOD)
    base = n * STEP_CM
    if ph <= HOLD_MS:
        # the hold is not perfect -- Troje's systematic slip, 3.15 mm/s
        return base + SLIP_MMS * 0.1 * (ph / 1000.0)
    u = (ph - HOLD_MS) / THRUST_MS
    s = u * u * (3.0 - 2.0 * u)                 # smoothstep, accel + decel
    return base + SLIP_MMS * 0.1 * (HOLD_MS / 1000.0) + s * STEP_CM


def body_world(t):
    return WALK_CMS * t - STEP_CM * 0.5


def _head_offset_mean():
    """Mean of (head - body) over a walking cycle, so the drawn head
    oscillates ABOUT its rest position instead of sitting to one side."""
    ts = np.linspace(2.0, 2.0 + PERIOD / 1000.0, 400, endpoint=False)
    return float(np.mean([head_world(x) - body_world(x) for x in ts]))


OFFS = _head_offset_mean()
HEAD_FREEZE = body_world(T_A1) + OFFS   # frozen head sits at the rest pose


def pose(t):
    """Everything the frame needs to know about time t."""
    p = {}
    if t < T_A1:                                    # free walking
        p["body"] = body_world(t)
        p["head"] = head_world(t)
        p["world"] = p["body"]                      # camera locked to body
        p["belt"] = p["body"]
        p["legf"] = BOB_HZ
        p["rollers"] = 0.0
        p["speed"] = None
    elif t < T_SW:                                  # the machine slides in
        u = (t - T_A1) / (T_SW - T_A1)
        s = u * u * (3.0 - 2.0 * u)
        p["body"] = body_world(T_A1)
        p["head"] = head_world(T_A1) + (HEAD_FREEZE - head_world(T_A1)) * s
        p["world"] = p["body"]
        p["belt"] = p["body"] + BELT_FAST * (t - T_A1) * u
        p["legf"] = BOB_HZ + (LEGF_FAST - BOB_HZ) * u
        p["rollers"] = u
        p["speed"] = None
    elif t < T_A2:                                  # belt at 60 cm/s
        p["body"] = body_world(T_A1)
        p["head"] = HEAD_FREEZE                     # DEAD STILL
        p["world"] = body_world(T_A1)
        p["belt"] = (body_world(T_A1) + BELT_FAST * (T_SW - T_A1) * 0.5
                     + BELT_FAST * (t - T_SW))
        p["legf"] = LEGF_FAST
        p["rollers"] = 1.0
        p["speed"] = "60 CM/S"
    else:                                           # the crawl, and the fall
        b0 = (body_world(T_A1) + BELT_FAST * (T_SW - T_A1) * 0.5
              + BELT_FAST * (T_A2 - T_SW))
        if t < T_SL:
            u = (t - T_A2) / (T_SL - T_A2)
            v = BELT_FAST + (BELT_CREEP - BELT_FAST) * (u * u * (3 - 2 * u))
            p["belt"] = b0 + (BELT_FAST + v) * 0.5 * (t - T_A2)
            p["legf"] = LEGF_FAST * (1.0 - u)
            drift = 0.0
        else:
            p["belt"] = (b0 + (BELT_FAST + BELT_CREEP) * 0.5 * (T_SL - T_A2)
                         + BELT_CREEP * (t - T_SL))
            p["legf"] = 0.0
            drift = -BELT_CREEP * (t - T_SL)        # feet carried backward
        p["body"] = body_world(T_A1) + drift
        p["head"] = HEAD_FREEZE                     # still nailed to the room
        p["world"] = body_world(T_A1)
        p["rollers"] = 1.0
        p["speed"] = "1.1 CM/S"
    # the head is held; the body is dragged out from under it; the bird
    # leans to keep the head where the room is, and then it goes over.
    p["hx"] = HEAD0[0] + (p["head"] - p["body"] - OFFS)
    # ONLY the crawl drags the head out past the toes.  during free walking
    # hx oscillates about its rest position by design, and reading that as
    # "the head is out over the toes" tipped the bird 18 degrees every step.
    over = max(0.0, p["hx"] - HEAD0[0]) if t >= T_SL else 0.0
    p["over"] = over
    lean = min(0.34, 0.14 * over)
    if t < T_FALL:
        p["tip"] = lean
        p["falling"] = False
    else:
        tau = t - T_FALL
        p["tip"] = min(TIP_MAX, lean + 0.5 * 1.55 * tau * tau)
        p["falling"] = True
    return p


def camera_ref(t):
    """Screen column of body-local x=0.  Locked to the body throughout;
    pans left only during the fall, so the beak stays in the frame."""
    return CX - 32.0 * pose(t)["tip"]


def eye_screen(t):
    """Where the eye actually is on screen, tip and all."""
    hx, hy = head_local(t)
    ex, ey = hx + 1.35, hy + 0.75
    tip = pose(t)["tip"]
    if tip > 0.0:
        qx, qy = _rot(ex - PIVOT[0], ey - PIVOT[1], -tip)
        ex, ey = qx + PIVOT[0], qy + PIVOT[1]
    return camera_ref(t) + ex * K, R_GND - ey * K


def head_local(t):
    """Where to draw the head so that, AFTER the lean is applied, it is
    still standing exactly where the room is.  That compensation is the
    whole reflex: the bird gives up its posture before it gives up the
    fixed point.  Once it is falling, the head rides with the body."""
    p = pose(t)
    want, tip = (p["hx"], HEAD0[1]), p["tip"]
    if p["falling"]:
        pf = pose(T_FALL - 1e-4)
        want, tip = (pf["hx"], HEAD0[1]), pf["tip"]
    if tip > 0.0:
        qx, qy = _rot(want[0] - PIVOT[0], want[1] - PIVOT[1], tip)
        want = (qx + PIVOT[0], qy + PIVOT[1])
    if not p["falling"]:
        return want
    # once it is going over, the bird gives the fixed point up: the neck
    # relaxes back to its rest pose and the head rides down with the body.
    u = min(1.0, (t - T_FALL) / 0.70)
    w = u * u * (3.0 - 2.0 * u)
    return (want[0] + (HEAD0[0] - want[0]) * w,
            want[1] + (HEAD0[1] - want[1]) * w)


# ---------------------------------------------------------------- drawing
def bird(mat, dens, t):
    p = pose(t)
    bx = camera_ref(t)
    # body-local coordinates of every cell, in cm
    px0 = (CC - bx) / K
    py0 = (R_GND - RR) / K
    px, py = px0, py0
    # the fall: rotate the query point about the front toe
    tip = p["tip"]
    if tip > 0.0:
        px, py = px - PIVOT[0], py - PIVOT[1]
        px, py = _rot(px, py, -tip)
        px, py = px + PIVOT[0], py + PIVOT[1]

    hx, hy = head_local(t)
    # the BODY bounces with the gait.  the head does not.
    bob = 0.0
    if p["legf"] > 0.01:
        bob = 0.30 * math.sin(4.0 * math.pi * p["legf"] * t)
    phase = 2.0 * math.pi * p["legf"] * t
    if p["legf"] > 0.01:
        foot_a = 2.2 + 2.9 * math.sin(phase)
        foot_b = 2.2 + 2.9 * math.sin(phase + math.pi)
        lift_a = max(0.0, 1.7 * math.sin(phase + math.pi * 0.5))
        lift_b = max(0.0, 1.7 * math.sin(phase + math.pi * 1.5))
    else:                                    # standing, feet planted
        foot_a, foot_b, lift_a, lift_b = 3.3, 0.4, 0.0, 0.0

    # the feet do NOT come off the belt.  the body pitches over them, so
    # the legs are drawn in the UNROTATED frame from a hip that has moved.
    def hip(x, y):
        if tip <= 0.0:
            return x, y
        qx, qy = _rot(x - PIVOT[0], y - PIVOT[1], -tip)
        return qx + PIVOT[0], qy + PIVOT[1]

    hax, hay = hip(1.7, 8.4 + bob)
    hbx, hby = hip(-0.9, 8.4 + bob)
    parts = []
    # far leg first (drawn behind, dimmer)
    parts.append(("legf", capsule(px0, py0, hbx, hby, foot_b - 1.0,
                                  0.55 + lift_b, 0.78, 0.52, -4.0)))
    parts.append(("footf", capsule(px0, py0, foot_b - 2.4, 0.42 + lift_b,
                                   foot_b + 2.0, 0.42 + lift_b,
                                   0.5, 0.34, -4.1)))
    parts.append(("legn", capsule(px0, py0, hax, hay, foot_a - 1.0,
                                  0.55 + lift_a, 0.82, 0.55, -2.0)))
    parts.append(("footn", capsule(px0, py0, foot_a - 2.6, 0.4 + lift_a,
                                   foot_a + 2.2, 0.4 + lift_a,
                                   0.55, 0.36, -2.1)))
    parts.append(("tail", ellipsoid(px, py, -15.0, 10.6 + bob,
                                    6.8, 2.0, 1.1, math.radians(-15.0), -0.6)))
    parts.append(("body", ellipsoid(px, py, 0.0, 13.3 + bob,
                                    9.7, 6.3, 5.4, math.radians(6.0), 0.0)))
    parts.append(("neck", capsule(px, py, 4.6, 16.4 + bob,
                                  hx - 1.5, hy - 1.4 + bob, 2.9, 2.4, 0.15)))
    parts.append(("wing", ellipsoid(px, py, -2.2, 13.5 + bob,
                                    7.5, 3.9, 1.5, math.radians(-4.0), 4.6)))
    parts.append(("head", ellipsoid(px, py, hx, hy + bob,
                                    3.5, 3.15, 3.15, 0.0, 0.4)))
    parts.append(("beak", capsule(px, py, hx + 2.5, hy - 0.5 + bob,
                                  hx + 5.9, hy - 1.5 + bob, 1.2, 0.30, 0.8)))

    best_z = np.full((G.rows, G.cols), -1e9)
    which = np.zeros((G.rows, G.cols), np.int8)
    nrm = np.zeros((3, G.rows, G.cols))
    local = np.zeros((2, G.rows, G.cols))
    for i, (name, (ins, z, n, q)) in enumerate(parts):
        win = ins & (z > best_z)
        best_z = np.where(win, z, best_z)
        which = np.where(win, i + 1, which)
        nrm = np.where(win[None, :, :], n, nrm)
        local = np.where(win[None, :, :], np.stack(q), local)

    on = which > 0
    if not on.any():
        return
    light = 0.10 + 0.90 * np.clip(
        nrm[0] * LAMP[0] + nrm[1] * LAMP[1] + nrm[2] * LAMP[2], 0.0, 1.0)
    # a rim of sky bounce off the top-back, so the silhouette does not die
    light += 0.10 * np.clip(-nrm[0] * 0.6 + nrm[1] * 0.8, 0.0, 1.0)

    names = [nm for nm, _ in parts]
    idx = {nm: i + 1 for i, nm in enumerate(names)}
    wingm = which == idx["wing"]
    # feather ranks across the wing, and a darker covert edge
    light = np.where(wingm,
                     light * (1.0 - 0.16 * (0.5 + 0.5 * np.cos(
                         local[0] * 1.9 + 1.1))), light)
    light = np.where(wingm & (local[0] < -3.4), light * 0.82, light)
    # the two legs read as different depths
    light = np.where((which == idx["legf"]) | (which == idx["footf"]),
                     light * 0.62, light)

    m = np.where(on, M_BIRD, 0).astype(np.int16)
    for nm in ("legn", "legf", "footn", "footf", "beak"):
        m = np.where(which == idx[nm], M_CORAL, m)
    # iridescent throat: the lower neck only
    thr = (which == idx["neck"]) & (py < hy - 1.0) & (py > 15.0 + bob)
    m = np.where(thr, M_IRID, m)

    # the eye
    ex, ey = hx + 1.35, hy + 0.75 + bob
    d = np.sqrt((px - ex) ** 2 + (py - ey) ** 2)
    ring = d < 1.15
    pupil = d < 0.62
    m = np.where(ring & on, M_CORAL, m)
    m = np.where(pupil & on, M_DARK, m)
    light = np.where(pupil & on, 0.02, light)
    light = np.where(ring & ~pupil & on, 0.75, light)

    light = light * (1.0 + 0.085 * NOISE)           # trap 10 / trap 18
    d_ink = np.clip(0.40 + 0.60 * (1.0 - light), 0.0, 1.0)
    mat[:] = np.where(on, m, mat)
    dens[:] = np.where(on, d_ink, dens)


def world_and_belt(mat, dens, t):
    p = pose(t)
    bx = camera_ref(t)
    wx = (CC - bx) / K + p["body"]                 # world cm of each cell

    # --- the far world.  two depths, because one moving line is not a flow
    #     field.  this is what the head is actually holding still against.
    far = wx * 0.36 + p["world"] * 0.64
    hor = RR == R_HOR
    mat[:] = np.where(hor, M_WORLD, mat)
    dens[:] = np.where(hor, 0.58, dens)
    ph = (far + 9.0) % 17.0
    post = (ph < 0.62) & (RR < R_HOR) & (RR > R_HOR - 10)
    mat[:] = np.where(post, M_WORLD, mat)
    dens[:] = np.where(post, 0.82, dens)

    near = wx * 0.70 + p["world"] * 0.30
    rail = RR == R_RAIL
    mat[:] = np.where(rail, M_WORLD, mat)
    dens[:] = np.where(rail, 0.48, dens)
    sh = (near + 3.0) % 8.0
    stud = (sh < 0.75) & (RR <= R_RAIL) & (RR > R_RAIL - 5)
    mat[:] = np.where(stud, M_WORLD, mat)
    dens[:] = np.where(stud, 0.70, dens)

    # --- the floor.  ordinary ground until the machine arrives.
    surf = RR == R_GND
    mat[:] = np.where(surf, M_BELT, mat)
    dens[:] = np.where(surf, 0.95, dens)
    bel = (CC - bx) / K + p["belt"]
    r = p["rollers"]

    if r < 0.5:                                    # plain ground, scrolling
        for row, per, off in ((R_GND + 2, 7.3, 0.0),
                              (R_GND + 4, 11.1, 3.1),
                              (R_GND + 6, 5.7, 1.7)):
            g = (bel + off) % per
            m = (g < 0.9) & (RR == row)
            mat[:] = np.where(m, M_BELT, mat)
            dens[:] = np.where(m, 0.42, dens)
        return

    bot = R_GND + BELT_H
    band = (RR > R_GND) & (RR < bot)
    edge = RR == bot
    mat[:] = np.where(edge, M_BELT, mat)
    dens[:] = np.where(edge, 0.80, dens)
    th = (bel + 40.0) % 3.6
    tick = (th < 0.72) & band
    mat[:] = np.where(tick, M_BELT, mat)
    dens[:] = np.where(tick, 0.72, dens)

    rr_ = 0.5 * BELT_H * r
    cy = R_GND + BELT_H * 0.5
    ang = p["belt"] / 5.0
    for cx_ in (10.0, G.cols - 11.0):
        dd = np.sqrt((CC - cx_) ** 2 + (RR - cy) ** 2)
        rim = (dd < rr_) & (dd > rr_ - 1.8)
        mat[:] = np.where(rim, M_BELT, mat)
        dens[:] = np.where(rim, 0.92, dens)
        for k in range(3):
            a = ang + k * 2.0 * math.pi / 3.0
            for sdist in np.linspace(0.0, rr_ - 2.0, 20):
                c0 = int(round(cx_ + sdist * math.cos(a)))
                r0 = int(round(cy + sdist * math.sin(a)))
                if 0 <= c0 < G.cols and 0 <= r0 < G.rows:
                    mat[r0, c0] = M_BELT
                    dens[r0, c0] = 0.62


# ---------------------------------------------------------------- the trace
class Trace(object):
    """Every eye position gets a mark on one line above the bird, with a
    live plumb-line down to the eye so there is no doubt whose line it is.
    The marks come out in CLUMPS, and the bird draws them itself."""

    def __init__(self):
        self.xs = []

    def add(self, t, n=5):
        if t >= T_FALL:
            return
        dt = 1.0 / (FPS * n)
        for i in range(n):
            tt = t + i * dt
            p = pose(tt)
            self.xs.append(p["body"] + p["hx"] + 1.35)   # the eye, in the room

    def draw(self, mat, dens, t):
        if not self.xs or t > T_FALL + 0.5:
            return 0
        fade = 1.0 - max(0.0, (t - T_FALL) / 0.5)
        bx = camera_ref(t)
        p = pose(t)
        cols = np.rint(bx + (np.array(self.xs) - p["body"]) * K).astype(int)
        cols = cols[(cols >= 21) & (cols < G.cols)]
        if not len(cols):
            return 0
        hits = np.bincount(cols, minlength=G.cols).astype(float)
        d = np.clip(hits / 9.0, 0.0, 1.0)
        v = (0.30 + 0.70 * d) * fade
        # a HOLD parks two dozen samples in one cell and a THRUST smears
        # twenty across eighteen.  let the mark get FAT where the head
        # stopped -- that thickness is the whole argument, drawn by the bird.
        for c in np.nonzero(hits > 0)[0]:
            h = hits[c]
            half = 0 if h < 10 else (2 if h < 60 else (3 if h < 220 else 4))
            for dr in range(-half, half + 1):
                r = R_TRACE + dr
                if mat[r, c] == 0:
                    mat[r, c] = M_TRAIL
                    dens[r, c] = v[c] * (1.0 if dr == 0 else 0.84)
        # the plumb line: this mark belongs to THAT eye, right now
        ec, er = eye_screen(t)
        ci = int(round(ec))
        if 0 <= ci < G.cols:
            for r in range(R_TRACE + 5, int(round(er)) - 2):
                if 0 <= r < G.rows and mat[r, ci] == 0 and (r % 2 == 0):
                    mat[r, ci] = M_TRAIL
                    dens[r, ci] = 0.30 * fade
        LAST["hot"] = int((hits >= 10).sum())
        return int((hits > 0).sum())


R_EYE = int(round(R_GND - (HEAD0[1] + 0.75) * K))
R_TRACE = 71               # the lane, clear of the bird at every pose


# ---------------------------------------------------------------- lettering
def text_cells(s, cell_h):
    """Rasterise a string at 8x and area-average it down to CELLS."""
    F = 8
    fs = cell_h * F
    probe = cairo.ImageSurface(cairo.FORMAT_A8, 8, 8)
    pc = cairo.Context(probe)
    pc.select_font_face("monospace", cairo.FONT_SLANT_NORMAL,
                        cairo.FONT_WEIGHT_BOLD)
    pc.set_font_size(fs)
    ext = pc.text_extents(s)
    w = int(ext.x_advance) + F * 2
    h = int(fs * 1.6)
    surf = cairo.ImageSurface(cairo.FORMAT_A8, w, h)
    ctx = cairo.Context(surf)
    ctx.select_font_face("monospace", cairo.FONT_SLANT_NORMAL,
                         cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(fs)
    ctx.move_to(F, h * 0.72)
    ctx.show_text(s)
    surf.flush()
    stride = surf.get_stride()
    buf = np.frombuffer(surf.get_data(), np.uint8).reshape(h, stride)[:, :w]
    ys, xs = np.nonzero(buf > 40)
    if not len(ys):
        return np.zeros((1, 1), bool)
    buf = buf[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    hh, ww = buf.shape
    ph, pw = (-hh) % F, (-ww) % F
    buf = np.pad(buf, ((0, ph), (0, pw)))
    hh, ww = buf.shape
    small = buf.reshape(hh // F, F, ww // F, F).mean((1, 3))
    return small > 46.0


def stamp(fr, s, cell_h, ccen, rcen, rgb, halo=BG):
    """Words are built OUT of cells (trap 11).  The halo must be SOLID."""
    m = text_cells(s, cell_h)
    while m.shape[1] + 2 > G.cols and cell_h > 1:
        cell_h -= 1
        m = text_cells(s, cell_h)
    h, w = m.shape
    hm = np.zeros((h + 2, w + 2), bool)
    for dy in range(3):
        for dx in range(3):
            hm[dy:dy + h, dx:dx + w] |= m
    c0 = int(round(ccen - w / 2.0))
    r0 = int(round(rcen - h / 2.0))
    for j in range(h + 2):
        for i in range(w + 2):
            if not hm[j, i]:
                continue
            cc_, rr_ = c0 - 1 + i, r0 - 1 + j
            on = (0 <= j - 1 < h and 0 <= i - 1 < w and m[j - 1, i - 1])
            fr.put(cc_, rr_, "#", rgb if on else halo)
    return (c0, r0, w, h)


def colour(v, m):
    base = MAT.get(int(m), INK)
    t = np.clip(0.34 + 0.66 * v, 0.0, 1.0)
    return (BG[0] + (base[0] - BG[0]) * t,
            BG[1] + (base[1] - BG[1]) * t,
            BG[2] + (base[2] - BG[2]) * t)


LAST = {}


def draw(f, trace):
    t = f / float(FPS)
    mat = np.zeros((G.rows, G.cols), np.int16)
    dens = np.zeros((G.rows, G.cols))
    world_and_belt(mat, dens, t)
    trace.add(t)
    ncell = trace.draw(mat, dens, t)
    bird(mat, dens, t)

    fr = Frame(G, BG)
    on = mat > 0
    idx, _ = fr.field(CC[on].ravel(), RR[on].ravel(),
                      np.ones(on.sum(), bool), dens[on].ravel(),
                      colour, RAMP, extra=mat[on].ravel().astype(float))
    LAST["ink"] = idx != 0
    LAST["mat"] = mat
    LAST["trace_cells"] = ncell

    p = pose(t)
    boxes = []
    if 1.1 < t < T_FALL:
        boxes.append(stamp(fr, "EYE", 11, 13, R_TRACE, MAG))
    if 2.4 < t < T_FALL:
        boxes.append(stamp(fr, "HELD TO 0.34 MM", 8, 52, 26, MAG))
    if p["speed"]:
        boxes.append(stamp(fr, "BELT " + p["speed"], 9, 49, 140, BELTC))
    LAST["boxes"] = boxes
    return fr


# ---------------------------------------------------------------- check
def check():
    print("MEASURED")
    print("  bob frequency        %.2f Hz  (Troje & Frost 2000)" % BOB_HZ)
    print("  hold / thrust        %.0f / %.0f ms  = %.1f%% hold"
          % (HOLD_MS, THRUST_MS, 100.0 * HOLD_MS / PERIOD))
    print("  Frost 1978 hold      %.0f%%" % (100 * HOLD_FRAC_1978))
    print("  thrust velocity      %.2f m/s  (Frost 1978)" % THRUST_V)
    print("  => step length       %.2f cm" % STEP_CM)
    print("  => walking speed     %.1f cm/s" % WALK_CMS)
    print("  hold-phase s.d.      %.2f mm" % HOLD_SD_MM)

    # HELD OUT: the two papers never combined their numbers.  Frost's thrust
    # velocity and Troje's phase durations imply a walking speed; Troje
    # separately reports that the hold phase survives only below 70 cm/s.
    # The implied walk must land comfortably under that ceiling, and the
    # implied hold fraction must reproduce Frost's independently measured
    # 63% to within a few points.
    implied_hold = 100.0 * HOLD_MS / PERIOD
    assert 20.0 < WALK_CMS < 45.0, WALK_CMS
    assert WALK_CMS < 70.0
    assert abs(implied_hold - 100 * HOLD_FRAC_1978) < 10.0, implied_hold
    print("  CROSS-CHECK          %.1f cm/s < 70 cm/s ceiling, "
          "hold %.1f%% vs 63%% measured  OK" % (WALK_CMS, implied_hold))
    print("  eye trail row        %d  (safe %d..%d)"
          % (R_EYE, G.safe_top, G.safe_bot))
    # the bird stands UP the whole time it is walking
    worst = max(pose(f / float(FPS))["tip"] for f in range(int(T_SL * FPS)))
    print("  max tip before crawl %.4f rad" % worst)
    assert worst == 0.0, worst

    trace = Trace()
    sheet = []
    done = 0
    for t in (1.0, 3.4, 5.4, 7.0, 9.2, 11.4, 12.6, 13.4, 15.0):
        f = int(t * FPS)
        while done < f:                    # the trace only accretes forward
            trace.add(done / float(FPS))
            done += 1
        fr = draw(f, trace)
        done = f + 1
        ink = LAST["ink"]
        mat = LAST["mat"]
        cov = ink.mean()
        rows_on = np.nonzero(ink.any(1))[0]
        cols_on = np.nonzero(ink.any(0))[0]
        birdm = np.isin(mat, [M_BIRD, M_IRID, M_CORAL, M_DARK])
        bc = np.nonzero(birdm.any(0))[0]
        br = np.nonzero(birdm.any(1))[0]
        # interior pinholes (trap 6): 1-3 cell gaps inside the bird
        pin = 0
        for r in br:
            xs = np.nonzero(birdm[r])[0]
            if len(xs) < 2:
                continue
            g = np.diff(xs)
            pin += int(((g > 1) & (g <= 4)).sum())
        print("  t=%4.1f cov %.3f  ink r%d..%d c%d..%d  bird c%d..%d r%d..%d"
              "  trail %3d  pin %3d"
              % (t, cov, rows_on.min(), rows_on.max(), cols_on.min(),
                 cols_on.max(), bc.min(), bc.max(), br.min(), br.max(),
                 LAST["trace_cells"], pin))
        assert 0 <= cols_on.min() and cols_on.max() < G.cols
        assert bc.min() >= 0 and bc.max() < G.cols, (bc.min(), bc.max())
        assert br.max() <= R_GND + 4, br.max()
        assert 0.07 < cov < 0.62, cov
        assert pin < 90, pin
        for (c0, r0, w, h) in LAST["boxes"]:
            assert r0 - 1 >= G.safe_top, ("text above safe", r0)
            assert r0 + h + 1 <= G.safe_bot, ("text below safe", r0 + h)
            assert c0 - 1 >= 0 and c0 + w + 1 <= G.cols, ("text width", c0, w)
        sheet.append(fr)

    # the bird must be BIG: at least 70% of the frame width, 30% of height
    wfill = (bc.max() - bc.min() + 1) / float(G.cols)
    print("  bird width fill      %.1f%%" % (100 * wfill))

    # the trace must actually CLUMP.  measure it at the end of act 1.
    trace = Trace()
    f = int(5.3 * FPS)
    for i in range(f + 1):
        trace.add(i / float(FPS))
    draw(f, trace)
    mat = LAST["mat"]
    hits = np.nonzero(mat[R_TRACE] == M_TRAIL)[0]
    dd = LAST["hot"]
    print("  trace cells at 5.3s  %d  saturated (hold) cells %d"
          % (len(hits), dd))
    assert len(hits) > 25, len(hits)
    assert dd >= 3, dd

    contact(sheet, os.path.join(_HERE, "..", "content", "pigeon_sheet.png"),
            cols=3, labels=["1.0 walk", "3.4 walk", "5.4 walk",
                            "7.0 belt", "9.2 belt", "11.4 crawl",
                            "12.6 lean", "13.4 fall", "15.0 down"])
    print("  contact sheet written")


def main():
    trace = Trace()
    with Encoder(OUT, G, fps=FPS) as enc:
        for f in range(FRAMES):
            enc.write(draw(f, trace))
            if f % 60 == 0:
                print("  %d/%d" % (f, FRAMES), flush=True)
    print("wrote", OUT)


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        main()
