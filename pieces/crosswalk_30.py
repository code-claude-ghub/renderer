#!/usr/bin/env python3
"""THE CROSSING — a crosswalk erased at the legal walking speed.

https://youtube.com/shorts/H-CXEUknyQc

The picture is one crossing, seen from behind a walker, receding up the
frame. The paint is eaten from the near end forward at exactly 3.0 ft/s,
which is the walking speed the MUTCD requires a signalised crossing to
serve (walk interval + pedestrian clearance, from a point 6 ft behind the
curb to the far side). The walker moves at 0.8 m/s = 2.62 ft/s, the mean
walking speed of women aged 65+ in the Health Survey for England sample
(Asher et al. 2012).

So the gap between the erasure front and her feet IS the shortfall, in
feet, growing every second. When the paint reaches the far curb the time
is up. She is 6.75 ft short. Nothing on screen is drawn by hand; the
front and the walker are both integrations of a constant speed.

Geometry: x = travel (ft), y = up, z = lateral. Camera is orthographic,
fixed elevation, tracking the walker so she holds one size and one row.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asciilib import (Camera, Encoder, Frame, Grid, RAMP_SORTED, contact,
                      ink_lut, lambert, visible, zbuffer)

OUT = os.path.expanduser("~/projects/active/youtube/youtube-channel/"
                         "out/crossing.mp4")

# ---------------------------------------------------------------- the numbers
FT_PER_M = 1.0 / 0.3048
V_LEGAL = 3.0                      # MUTCD walk+clearance design speed, ft/s
V_WALK = 0.8 * FT_PER_M            # 2.6247 ft/s — mean, women 65+, HSE 2005
X_START = 0.0                      # 6 ft behind the near curb, per MUTCD
X_CURB_NEAR = 6.0
ROADWAY = 48.0                     # four 12 ft lanes
X_CURB_FAR = X_CURB_NEAR + ROADWAY # 54.0
T_TOTAL = (X_CURB_FAR - X_START) / V_LEGAL          # 18.0 s
X_END = X_START + V_WALK * T_TOTAL                  # 47.25 ft
SHORTFALL = X_CURB_FAR - X_END                      # 6.75 ft

FPS = 30
FRAMES = int(round(T_TOTAL * FPS))                  # 540
HOLD = 24                                           # the beat after

# stripes: 1.5 ft of paint on 3 ft centres, so one stripe is one second
# of budget at the legal speed.
STRIPE_W = 1.5
STRIPE_PITCH = 3.0
STRIPE_HALF_Z = 5.2
STRIPE_X = np.arange(X_CURB_NEAR + 1.5, X_CURB_FAR - 0.5, STRIPE_PITCH)

# ---------------------------------------------------------------- the camera
PHI = math.radians(40.0)
SIN_P, COS_P = math.sin(PHI), math.cos(PHI)
HALF_WIDTH = 4.35                  # ft of road across the frame
FEET_ROW_FRAC = 0.74               # where her heels sit in the frame

# ---------------------------------------------------------------- the palette
BG = (0.949, 0.722, 0.078)         # road-sign yellow
INK_BODY = (0.043, 0.039, 0.075)   # near-black, cool
INK_PAINT = (0.322, 0.196, 0.031)  # warm dark brown
INK_KERB = (0.157, 0.114, 0.031)

LAMP = np.array([-0.45, 0.80, 0.40])

RNG = np.random.default_rng(20260817)
G = Grid()
RAMP = ink_lut(RAMP_SORTED)


# ------------------------------------------------------------------ sampling
def capsule(p0, p1, r0, r1, n, ecc=1.0):
    """Surface points + normals on a tapered capsule. ecc squashes z."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    ax = p1 - p0
    L = np.linalg.norm(ax)
    ax = ax / max(L, 1e-9)
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(ax @ tmp) > 0.95:
        tmp = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(ax, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax, e1)
    u = RNG.random(n)
    th = RNG.random(n) * 2.0 * math.pi
    r = r0 + (r1 - r0) * u
    ca, sa = np.cos(th), np.sin(th)
    off = (e1[None, :] * (ca * r)[:, None]
           + e2[None, :] * (sa * r * ecc)[:, None])
    pts = p0[None, :] + ax[None, :] * (u * L)[:, None] + off
    nrm = off / (np.linalg.norm(off, axis=1, keepdims=True) + 1e-9)
    return pts, nrm


def ball(c, r, n, sy=1.0):
    c = np.asarray(c, float)
    v = RNG.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    p = c[None, :] + v * r * np.array([1.0, sy, 1.0])[None, :]
    return p, v


def slab(x0, x1, y, z0, z1, n):
    """A flat painted patch lying on the road."""
    x = RNG.uniform(x0, x1, n)
    z = RNG.uniform(z0, z1, n)
    p = np.stack([x, np.full(n, y), z], -1)
    nrm = np.tile(np.array([0.0, 1.0, 0.0]), (n, 1))
    return p, nrm


# ------------------------------------------------------------------- the walk
CADENCE = 1.70                     # steps per second
T_STRIDE = 2.0 / CADENCE
DUTY = 0.62
HIP_Y = 3.10
THIGH, SHANK, ANKLE_Y = 1.45, 1.40, 0.25
HIP_Z = 0.30
SHOULDER_Y, SHOULDER_Z = 4.62, 0.55
HEAD_Y, HEAD_R = 5.20, 0.38
EXCURSION = DUTY * V_WALK * T_STRIDE / 2.0          # +-0.96 ft
SWING_LIFT = 0.24


def foot_state(ph):
    """Foot position relative to the pelvis, and its pitch, at cycle phase."""
    ph = ph % 1.0
    if ph < DUTY:                                   # planted
        s = ph / DUTY
        fx = EXCURSION - 2.0 * EXCURSION * s
        fy = 0.0
        pitch = math.radians(-14.0) * max(0.0, (s - 0.72) / 0.28) ** 2 * 3.2
        pitch += math.radians(9.0) * max(0.0, (0.16 - s) / 0.16)
    else:                                           # swinging
        s = (ph - DUTY) / (1.0 - DUTY)
        fx = -EXCURSION + 2.0 * EXCURSION * (s - math.sin(2 * math.pi * s)
                                             / (2 * math.pi))
        fy = SWING_LIFT * math.sin(math.pi * s) ** 0.8
        pitch = math.radians(-26.0) * math.exp(-8.0 * s) \
            + math.radians(11.0) * s ** 2
    return fx, fy, pitch


def two_link(hip, ankle, a, b):
    """Sagittal IK. Returns the knee. Knee bends forward (+x)."""
    d = ankle - hip
    L = float(np.linalg.norm(d))
    L = min(max(L, 0.20), a + b - 0.015)
    uD = d / max(np.linalg.norm(d), 1e-9)
    uP = np.array([-uD[1], uD[0]])
    if uP[0] < 0:
        uP = -uP
    ca = np.clip((a * a + L * L - b * b) / (2.0 * a * L), -1.0, 1.0)
    al = math.acos(ca)
    return hip + a * (math.cos(al) * uD + math.sin(al) * uP)


def walker(t):
    """Point cloud of the figure, in coordinates relative to her own feet."""
    ph = (t / T_STRIDE) % 1.0
    bob = -0.055 * math.cos(4.0 * math.pi * ph) - 0.02
    sway = 0.075 * math.sin(2.0 * math.pi * ph)
    lean = math.radians(3.0)

    pelvis = np.array([0.0, HIP_Y + bob, sway])
    P, N, M = [], [], []

    def add(p, n, m):
        P.append(p)
        N.append(n)
        M.append(np.full(len(p), m, np.float32))

    for side, phase in ((+1.0, 0.0), (-1.0, 0.5)):
        fx, fy, pitch = foot_state(ph + phase)
        z = pelvis[2] + side * HIP_Z
        hip2 = np.array([pelvis[0], pelvis[1]])
        ank2 = np.array([fx, fy + ANKLE_Y])
        knee2 = two_link(hip2, ank2, THIGH, SHANK)
        hip = np.array([hip2[0], hip2[1], z])
        knee = np.array([knee2[0], knee2[1], z])
        ank = np.array([ank2[0], ank2[1], z])
        add(*capsule(hip, knee, 0.34, 0.235, 1900), 1.0)
        add(*capsule(knee, ank, 0.235, 0.145, 1700), 1.0)
        # foot: heel to toe, pitched
        cp, sp = math.cos(pitch), math.sin(pitch)
        heel = ank + np.array([-0.28 * cp, -ANKLE_Y + 0.28 * sp, 0.0])
        toe = ank + np.array([0.62 * cp, -ANKLE_Y - 0.62 * sp, 0.0])
        add(*capsule(heel, toe, 0.17, 0.115, 1000, ecc=0.82), 1.0)

    # torso, leaning very slightly forward
    hipc = np.array([pelvis[0] + 0.02, pelvis[1], pelvis[2]])
    sho = np.array([pelvis[0] + math.sin(lean) * (SHOULDER_Y - HIP_Y),
                    SHOULDER_Y + bob, pelvis[2] * 0.55])
    add(*capsule(hipc, sho, 0.40, 0.56, 3800, ecc=0.62), 1.0)
    add(*ball([sho[0] + 0.05, HEAD_Y + bob, sho[2]], HEAD_R, 1500, sy=1.12),
        1.0)

    for side, phase in ((+1.0, 0.5), (-1.0, 0.0)):
        aph = (ph + phase) % 1.0
        th = math.radians(21.0) * math.cos(2.0 * math.pi * aph)
        sz = sho[2] + side * SHOULDER_Z
        s0 = np.array([sho[0] - 0.02, SHOULDER_Y - 0.12 + bob, sz])
        elb = s0 + np.array([math.sin(th) * 1.02, -math.cos(th) * 1.02,
                             side * 0.05])
        fl = math.radians(30.0 + 22.0 * max(0.0, math.cos(2 * math.pi * aph)))
        hnd = elb + np.array([math.sin(th + fl) * 0.94,
                              -math.cos(th + fl) * 0.94, side * 0.04])
        add(*capsule(s0, elb, 0.20, 0.150, 1000), 1.0)
        add(*capsule(elb, hnd, 0.150, 0.110, 900), 1.0)

    return np.concatenate(P), np.concatenate(N), np.concatenate(M)


# ------------------------------------------------------------------ the road
def road(t, x_cam):
    """Paint still on the ground, plus the two kerbs. World coords."""
    front = X_START + V_LEGAL * t          # the legal walker's position
    P, N, M = [], [], []

    def add(p, n, m):
        P.append(p)
        N.append(n)
        M.append(np.full(len(p), m, np.float32))

    for cx in STRIPE_X:
        if cx + STRIPE_W / 2 < front:      # spent
            continue
        if not (-11.0 < cx - x_cam < 24.0):
            continue
        x0 = max(cx - STRIPE_W / 2, front)
        add(*slab(x0, cx + STRIPE_W / 2, 0.012,
                  -STRIPE_HALF_Z, STRIPE_HALF_Z,
                  int(7200 * (cx + STRIPE_W / 2 - x0) / STRIPE_W) + 60), 2.0)

    for kx in (X_CURB_NEAR, X_CURB_FAR):
        if not (-11.0 < kx - x_cam < 24.0):
            continue
        top, nt = slab(kx, kx + 2.6, 0.52, -STRIPE_HALF_Z, STRIPE_HALF_Z, 13000)
        if kx == X_CURB_NEAR:
            top[:, 0] = kx - 2.6 + (top[:, 0] - kx)
        add(top, nt, 3.0)
        n = 4200
        face = np.stack([np.full(n, kx + (0.0 if kx == X_CURB_FAR else -0.02)),
                         RNG.uniform(0.0, 0.52, n),
                         RNG.uniform(-STRIPE_HALF_Z, STRIPE_HALF_Z, n)], -1)
        fn = np.tile(np.array([-1.0, 0.25, 0.0]), (n, 1))
        fn /= np.linalg.norm(fn, axis=1, keepdims=True)
        add(face, fn, 3.0)

    if not P:
        return (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, np.float32))
    return np.concatenate(P), np.concatenate(N), np.concatenate(M)


# ------------------------------------------------------------------- project
def to_cam(p, x_cam):
    """World -> asciilib space. Screen x = lateral, screen y = down."""
    x = p[:, 0] - x_cam
    U = x * SIN_P + p[:, 1] * COS_P
    D = x * COS_P - p[:, 1] * SIN_P
    return np.stack([p[:, 2], -U, -D], -1)


CAM = Camera(G)
CAM.scale = G.room_c / HALF_WIDTH
CAM.off = np.array([0.0, (G.cy - FEET_ROW_FRAC * G.rows) / CAM.scale])


def colour(shade, extra):
    if extra > 2.5:
        ink = INK_KERB
    elif extra > 1.5:
        ink = INK_PAINT
    else:
        ink = INK_BODY
    a = 0.74 + 0.26 * shade
    return (BG[0] + (ink[0] - BG[0]) * a,
            BG[1] + (ink[1] - BG[1]) * a,
            BG[2] + (ink[2] - BG[2]) * a)


def draw(f):
    t = f / float(FPS)
    x_cam = X_START + V_WALK * t

    wp, wn, wm = walker(t)
    rp, rn, rm = road(t, x_cam)
    wp = wp + np.array([x_cam, 0.0, 0.0])

    p = np.concatenate([wp, rp]) if len(rp) else wp
    n = np.concatenate([wn, rn]) if len(rp) else wn
    m = np.concatenate([wm, rm]) if len(rp) else wm

    # jitter every sampled surface, once, off a fixed seed
    p = p + RNG.normal(scale=0.012, size=p.shape)

    q = to_cam(p, x_cam)
    col, row, z = CAM.project(q)
    ok = visible(G, col, row)
    col, row, z, n, m = col[ok], row[ok], z[ok], n[ok], m[ok]
    _, keep = zbuffer(G, col, row, z)

    light = 0.16 + 0.84 * lambert(n, LAMP)
    dens = np.where(m > 1.5,
                    0.80 + 0.10 * (1.0 - light),      # flat paint, flat ink
                    0.86 + 0.14 * (1.0 - light))      # the body
    dens = np.where(m > 2.5, 0.70 + 0.30 * (1.0 - light), dens)

    fr = Frame(G, BG)
    fr.field(col, row, keep, dens, colour, RAMP, extra=m)
    return fr


def check():
    print(G)
    print("legal %.3f ft/s  walk %.4f ft/s  T %.1f s  end %.2f ft  "
          "shortfall %.2f ft" % (V_LEGAL, V_WALK, T_TOTAL, X_END, SHORTFALL))
    print("stripes %d  first %.1f last %.1f" % (len(STRIPE_X), STRIPE_X[0],
                                                STRIPE_X[-1]))
    print("scale %.2f cells/ft  frame is %.2f ft wide, %.1f ft of road deep"
          % (CAM.scale, G.cols / CAM.scale, (G.rows / CAM.scale) / SIN_P))
    for f in (0, 120, 300, 480, FRAMES - 1):
        t = f / float(FPS)
        x_cam = X_START + V_WALK * t
        wp, _, _ = walker(t)
        q = to_cam(wp + np.array([x_cam, 0, 0]), x_cam)
        c, r, _ = CAM.project(q)
        print("  f%3d t=%5.2f  x=%6.2f  front=%6.2f  gap=%5.2f  "
              "body cols %d..%d rows %d..%d"
              % (f, t, x_cam, V_LEGAL * t, V_LEGAL * t - x_cam,
                 c.min(), c.max(), r.min(), r.max()))


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
        sys.exit(0)
    if "--sheet" in sys.argv:
        check()
        idx = [0, 90, 180, 270, 360, 450, 510, 539]
        contact([draw(i) for i in idx],
                os.path.expanduser("~/projects/active/youtube/"
                                   "youtube-channel/out/crossing_sheet.png"),
                cols=4, labels=["%.1fs" % (i / float(FPS)) for i in idx])
        sys.exit(0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with Encoder(OUT, G, fps=FPS) as enc:
        for f in range(FRAMES):
            enc.write(draw(f))
            if f % 60 == 0:
                print("  %d/%d" % (f, FRAMES), flush=True)
        last = draw(FRAMES - 1)          # 0.8 s still in the road
        for _ in range(HOLD):
            enc.write(last)
    print("wrote", OUT)
