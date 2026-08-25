#!/usr/bin/env python3
"""
THE LOT -- one day in a car park. 04:40 to 20:00, one frame per minute.

Nothing in this piece is explained and nothing is captioned. It is the first
thing this channel has made that is INVENTED rather than found: the lot does
not exist, the cars do not exist, nobody in it is real.

The sun is real.

Melbourne, 2026-08-25. Late winter in the southern hemisphere. Sunrise 06:50,
sunset 17:54, eleven hours and three minutes of daylight. Solar altitude and
azimuth come out of the NOAA algorithm implemented below, so the shadows sweep
west to east across the asphalt at the rate they actually would, the light
level rises and falls through the real civil twilight, and the day is the
length the day is.

One car arrives at 05:40 and leaves at 18:40. Thirteen hours. Both ends of it
happen in the dark, with about two hours to spare, and that is not overtime --
it is what a normal long shift costs you in August at 37 degrees south.

    THE RATE. 920 minutes of clock, 920 frames, 30 fps. One frame is one
    minute; one second is half an hour; the whole thing runs 1800x.

    At 1800x a car cannot be seen to arrive. A car pulls in over about
    thirty seconds, which is half a frame, so it appears. That is not a
    shortcut, it is what a time-lapse IS, and the alternative -- slowing
    the arrivals so they read -- would have made the picture lie about the
    only thing it is actually measuring. So the cars pop, the way they pop
    in every real time-lapse of a car park ever shot.

    What CAN be seen at one frame a minute is a light going on and going
    off. So that is the only event in the piece with a cause attached.

Run:
    python3 scripts/lot.py --check      # numbers only, no render
    python3 scripts/lot.py              # render
"""

import argparse
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]

from asciilib import (Camera, Encoder, Frame, Grid, contact, ink_lut,
                      visible, zbuffer)

OUT = os.path.join(_HERE, "out")
G = Grid()
RAMP = ink_lut()
FPS = 30
RNG = np.random.default_rng(20260825)

# ---------------------------------------------------------------- the day

LAT = -37.8136          # Melbourne
LON = 144.9631
TZ = 10.0               # AEST. Australia goes to DST in October, not August.
DATE = (2026, 8, 25)

T0_MIN = 4 * 60 + 40    # the picture opens at 04:40
N_FRAME = 920           # ... and closes at 20:00. one frame per minute.
DUR = N_FRAME / float(FPS)
RATE = 60.0 * FPS       # 1800x

# clock minutes of the events
F_ARRIVE = 5 * 60 + 40 - T0_MIN         # 05:40  our car appears
F_WIN_ON = 5 * 60 + 44 - T0_MIN         # 05:44  a window lights
F_FILL = (7 * 60 + 5 - T0_MIN, 10 * 60 + 30 - T0_MIN)
F_EMPTY = (13 * 60 + 0 - T0_MIN, 17 * 60 + 15 - T0_MIN)
F_WIN_OFF = 18 * 60 + 36 - T0_MIN       # 18:36  it goes out
F_DEPART = 18 * 60 + 40 - T0_MIN        # 18:40  our car is gone


def _julian(y, m, d, hours_utc):
    if m <= 2:
        y, m = y - 1, m + 12
    a = y // 100
    b = 2 - a + a // 4
    return (int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b
            - 1524.5 + hours_utc / 24.0)


def _sun_dec_eot(jd):
    """NOAA: solar declination (deg) and the equation of time (minutes)."""
    t = (jd - 2451545.0) / 36525.0
    l0 = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360.0
    m = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    mr = math.radians(m)
    c = (math.sin(mr) * (1.914602 - t * (0.004817 + 0.000014 * t))
         + math.sin(2 * mr) * (0.019993 - 0.000101 * t)
         + math.sin(3 * mr) * 0.000289)
    omega = 125.04 - 1934.136 * t
    app = l0 + c - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    eps0 = 23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059
                                                       - t * 0.001813))) / 60.0) / 60.0
    eps = eps0 + 0.00256 * math.cos(math.radians(omega))
    dec = math.degrees(math.asin(math.sin(math.radians(eps))
                                 * math.sin(math.radians(app))))
    yy = math.tan(math.radians(eps / 2.0)) ** 2
    lr = math.radians(l0)
    eot = 4.0 * math.degrees(
        yy * math.sin(2 * lr) - 2 * e * math.sin(mr)
        + 4 * e * yy * math.sin(mr) * math.cos(2 * lr)
        - 0.5 * yy * yy * math.sin(4 * lr)
        - 1.25 * e * e * math.sin(2 * mr))
    return dec, eot


def sun_at(minute):
    """Local clock minute -> (altitude deg, azimuth deg from north, clockwise).

    Azimuth is the compass bearing of the sun. In Melbourne the noon sun sits
    at bearing 0 -- due NORTH -- which is why the building on the far side of
    this lot is backlit all day and every car throws its shadow toward the
    camera.
    """
    jd = _julian(DATE[0], DATE[1], DATE[2], (minute / 60.0) - TZ)
    dec, eot = _sun_dec_eot(jd)
    tst = (minute + eot + 4.0 * LON - 60.0 * TZ) % 1440.0
    ha = math.radians(tst / 4.0 - 180.0)
    d, p = math.radians(dec), math.radians(LAT)
    alt = math.asin(math.sin(p) * math.sin(d)
                    + math.cos(p) * math.cos(d) * math.cos(ha))
    az = math.atan2(-math.sin(ha) * math.cos(d),
                    math.cos(p) * math.sin(d)
                    - math.sin(p) * math.cos(d) * math.cos(ha))
    return math.degrees(alt), math.degrees(az) % 360.0


def sun_vec(minute):
    """Unit vector from the ground toward the sun, in world axes."""
    alt, az = sun_at(minute)
    a, z = math.radians(alt), math.radians(az)
    return np.array([math.sin(z) * math.cos(a),      # +x east
                     math.cos(z) * math.cos(a),      # +y north
                     math.sin(a)]), alt              # +z up


def rise_set():
    """Sunrise and sunset as local clock minutes, by bisection on altitude.

    Deliberately NOT the hour-angle formula -- check() derives it that way
    too, from figures this function never touches, and asserts they agree.
    """
    out = []
    for lo, hi in ((240, 720), (720, 1200)):
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            up = sun_at(mid)[0] > -0.833     # refraction + solar radius
            if (up == (sun_at(lo)[0] > -0.833)):
                lo = mid
            else:
                hi = mid
        out.append(0.5 * (lo + hi))
    return out[0], out[1]


SUNRISE, SUNSET = rise_set()

# ---------------------------------------------------------------- the lot

# A DEEP, NARROW lot that RUNS OFF EVERY EDGE of the frame.
#
# First version bounded the asphalt at the lot's real edges, and the result
# was a lit trapezoid floating in black -- it read as a rendered plane, not
# as a place. The ground is now generated to cover whatever the camera can
# see (see ground_for_frame), the bay rows run out past both sides, and the
# building spans the top. You are looking at part of a car park, which is
# all anyone ever looks at.
#
# The frame is 98 x 174 and this piece has no text in it anywhere, so
# nothing has to keep out of the Shorts safe band and the picture can bleed
# to all four edges -- but only if its shape matches the frame's. Six bay
# rows deep by ten across is portrait; four by eight was letterboxed.
BAY_W, BAY_D = 2.80, 5.00
BAY_X0 = -1.40
N_BAY = 10
ROW_Y = (7.00, 12.00, 23.00, 28.00, 39.00, 44.00)
ROW_FACE = (+1.0, -1.0, +1.0, -1.0, +1.0, -1.0)   # which way the nose points
AISLE_Y = (2.25, 17.50, 33.50, 49.50)
BLD_Y0, BLD_Y1 = 52.50, 60.00
BLD_H = 6.50
DOOR_X = 11.20                           # dead in line with one bay
WIN_PITCH = 3.85
WIN_X = tuple(DOOR_X + WIN_PITCH * k for k in range(-6, 7))
OUR_WIN = 6                              # the one directly over the door

POLE_X = (-2.50, 8.50, 19.50, 30.50)
POLE_H = 7.50
POLES = [(x, y) for x in POLE_X for y in AISLE_Y]
POOL_R = 6.2
WIN_SILL, WIN_H, WIN_W = 3.60, 1.40, 1.60

def bay_xy(row, i):
    return BAY_X0 + BAY_W * (i + 0.5), ROW_Y[row]


BAYS = [(r, i) for r in range(len(ROW_Y)) for i in range(N_BAY)]
BAYS.sort(key=lambda b: (bay_xy(*b)[0] - DOOR_X) ** 2
          + (bay_xy(*b)[1] - BLD_Y0) ** 2)
# BAYS[0] is now the bay closest to the door, which is the one the first
# person in the car park takes, because that is what people do.

N_CAR = 50

# ---------------------------------------------------------------- sampling


def _plane(x0, x1, y0, y1, z, step, normal, jitter=0.0):
    nx = max(2, int(round((x1 - x0) / step)))
    ny = max(2, int(round((y1 - y0) / step)))
    gx, gy = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    p = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], 1)
    if jitter:
        p[:, :2] += RNG.uniform(-jitter, jitter, (len(p), 2))
    n = np.tile(np.asarray(normal, float), (len(p), 1))
    return p.astype(np.float32), n.astype(np.float32)


def _box(cx, cy, cz, hx, hy, hz, step, faces="tns"):
    """A box, sampled on the faces that a camera above and to the south sees.

    t = top, n = the face toward -y (south, toward us), s = the two sides.
    The bottom and the north face are never visible and cost nothing to omit.
    """
    P, N = [], []
    if "t" in faces:
        p, n = _plane(cx - hx, cx + hx, cy - hy, cy + hy, cz + hz, step,
                      (0, 0, 1))
        P.append(p); N.append(n)
    if "n" in faces:
        nx = max(2, int(round(2 * hx / step)))
        nz = max(2, int(round(2 * hz / step)))
        gx, gz = np.meshgrid(np.linspace(cx - hx, cx + hx, nx),
                             np.linspace(cz - hz, cz + hz, nz))
        p = np.stack([gx.ravel(), np.full(gx.size, cy - hy), gz.ravel()], 1)
        P.append(p.astype(np.float32))
        N.append(np.tile([0.0, -1.0, 0.0], (len(p), 1)).astype(np.float32))
    if "s" in faces:
        ny = max(2, int(round(2 * hy / step)))
        nz = max(2, int(round(2 * hz / step)))
        gy, gz = np.meshgrid(np.linspace(cy - hy, cy + hy, ny),
                             np.linspace(cz - hz, cz + hz, nz))
        for sx in (-1.0, 1.0):
            p = np.stack([np.full(gy.size, cx + sx * hx),
                          gy.ravel(), gz.ravel()], 1)
            P.append(p.astype(np.float32))
            N.append(np.tile([sx, 0.0, 0.0], (len(p), 1)).astype(np.float32))
    return np.vstack(P), np.vstack(N)


def car(cx, cy, length, width, roof, colour_id, step=0.20):
    """One car: body, cabin, and a band of glass. Always axis-aligned --
    it is parked.

    THE CABIN IS BODY COLOUR. The first version built the whole cabin out of
    glass, which from an elevation of 54 degrees laid a dark rectangle over
    more than half of every car's visible area -- a silver car came out
    DARKER than the asphalt it was standing on. Measured across the day, car
    brightness over surrounding brightness ran 0.78 to 1.03, which is to say
    the cars were invisible and the lot looked empty at every hour. Seen from
    above a car is mostly roof, and the glass is a stripe.
    """
    hy, hx = 0.5 * length, 0.5 * width
    ch = 0.5 * max(0.12, roof - 1.12)
    P, N, M = [], [], []
    p, n = _box(cx, cy, 0.70, hx, hy, 0.42, step)                    # body
    P.append(p); N.append(n); M.append(np.full(len(p), colour_id))
    p, n = _box(cx, cy + 0.10, 1.12 + ch, hx * 0.86, hy * 0.40, ch,  # cabin
                step)
    P.append(p); N.append(n); M.append(np.full(len(p), colour_id))
    for sgn in (-1.0, 1.0):                                          # glass
        p, n = _box(cx, cy + 0.10 + sgn * hy * 0.40, 1.12 + ch,
                    hx * 0.80, 0.16, ch * 0.92, step * 0.75)
        P.append(p); N.append(n); M.append(np.full(len(p), M_GLASS))
    return (np.vstack(P).astype(np.float32),
            np.vstack(N).astype(np.float32),
            np.concatenate(M).astype(np.float32))


# materials ------------------------------------------------------------
M_ASPH, M_PAINT, M_WALL, M_ROOF, M_WIN, M_POLE, M_GLASS = 1, 2, 3, 4, 5, 6, 7
M_CAR0 = 10                                     # 10..15, six car colours
N_CAR_COL = 6

BG = (0.021, 0.027, 0.041)
ASPHALT = (0.385, 0.395, 0.415)
PAINT = (0.880, 0.870, 0.790)
WALL = (0.430, 0.420, 0.400)
ROOFC = (0.330, 0.335, 0.350)
WINC = (1.000, 0.800, 0.480)
POLEC = (0.380, 0.400, 0.430)
GLASSC = (0.300, 0.360, 0.420)
SODIUM = (1.000, 0.660, 0.330)                  # what the lamps put down
CARC = [(0.620, 0.630, 0.650),                  # silver
        (0.180, 0.200, 0.240),                  # dark grey
        (0.720, 0.720, 0.720),                  # white
        (0.480, 0.180, 0.170),                  # red
        (0.190, 0.290, 0.430),                  # blue
        (0.330, 0.330, 0.300)]                  # the colour of a work van


def build_lot():
    """Everything whose position is fixed by the LOT: markings and poles.

    Not the ground and not the building -- those are sized to the frame, and
    the frame is not known until the camera has been fitted to this.
    """
    P, N, M = [], [], []
    for r in range(len(ROW_Y)):
        y0, y1 = ROW_Y[r] - 0.5 * BAY_D, ROW_Y[r] + 0.5 * BAY_D
        yf = y1 if ROW_FACE[r] > 0 else y0
        # bay markings, lifted clear of the asphalt so the z-buffer cannot
        # dither them into dashes (RENDERER.md trap 8)
        p, n = _plane(BAY_X0, BAY_X0 + BAY_W * N_BAY, yf - 0.05, yf + 0.05,
                      0.025, 0.075, (0, 0, 1))
        P.append(p); N.append(n); M.append(np.full(len(p), M_PAINT))
        for i in range(N_BAY + 1):
            x = BAY_X0 + BAY_W * i
            p, n = _plane(x - 0.05, x + 0.05, y0, y1, 0.025, 0.075, (0, 0, 1))
            P.append(p); N.append(n); M.append(np.full(len(p), M_PAINT))

    for x, y in POLES:
        p, n = _box(x, y, 0.5 * POLE_H, 0.085, 0.085, 0.5 * POLE_H, 0.17)
        P.append(p); N.append(n); M.append(np.full(len(p), M_POLE))
        p, n = _box(x, y - 0.55, POLE_H, 0.26, 0.62, 0.09, 0.14)
        P.append(p); N.append(n); M.append(np.full(len(p), M_POLE))

    return (np.vstack(P).astype(np.float32), np.vstack(N).astype(np.float32),
            np.concatenate(M).astype(np.float32))


def frame_box(cam, pad=1.6):
    """Invert the projection at z=0 to find the world rectangle the camera
    can see. Used to size the ground and the building so neither has a
    visible edge -- an edge is what made the first version read as a slab
    lying in a void rather than as a piece of somewhere larger."""
    cc, ss = math.cos(YAW), math.sin(YAW)
    se = math.sin(EL)
    xs, ys = [], []
    for c in (-2.0, G.cols + 2.0):
        for r in (-2.0, G.rows + 2.0):
            sx = (c - G.cx) / cam.scale + cam.off[0]
            sy = (r - G.cy) / cam.scale + cam.off[1]
            x1, y1 = sx, -sy / se
            xs.append(x1 * cc + y1 * ss)
            ys.append(-x1 * ss + y1 * cc)
    return (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)


def build_ground(box, want=3.4):
    """Asphalt over the whole visible rectangle, sampled at about `want`
    samples per character cell -- enough to close the surface without
    paying for oversample nobody can see."""
    x0, x1, y0, y1 = box
    y1 = min(y1, BLD_Y0)                    # the wall hides everything past it
    step = math.sqrt((x1 - x0) * (y1 - y0) / (want * G.cols * G.rows))
    return _plane(x0, x1, y0, y1, 0.0, step, (0, 0, 1), jitter=0.28 * step)


def build_building(box, cam):
    # The roof has to reach the top of the frame. Fitting the camera to a
    # 7.5 m deep building left twenty bare rows of void above it, because
    # the fit is limited by the WIDTH and there is vertical slack left over.
    # So the depth is whatever the picture needs, not a number I picked.
    se, ce = math.sin(EL), math.cos(EL)
    sy_top = (-3.0 - G.cy) / cam.scale + cam.off[1]
    far = max(BLD_Y1, (-sy_top - BLD_H * ce) / se + 2.0)
    x0, x1 = box[0] - 4.0, box[1] + 4.0
    p, n = _box(0.5 * (x0 + x1), 0.5 * (BLD_Y0 + far), 0.5 * BLD_H,
                0.5 * (x1 - x0), 0.5 * (far - BLD_Y0), 0.5 * BLD_H,
                0.26, faces="tn")
    keep = ~_window_mask(p) & ~_door_mask(p)
    p, n = p[keep], n[keep]
    m = np.where(p[:, 2] > BLD_H - 1e-3, M_ROOF, M_WALL)
    return p, n, m.astype(np.float32)


def _window_mask(p):
    on_wall = np.abs(p[:, 1] - BLD_Y0) < 1e-3
    m = np.zeros(len(p), bool)
    for x in WIN_X:
        m |= (np.abs(p[:, 0] - x) < 0.5 * WIN_W) & \
             (p[:, 2] > WIN_SILL) & (p[:, 2] < WIN_SILL + WIN_H)
    return m & on_wall


def _door_mask(p):
    on_wall = np.abs(p[:, 1] - BLD_Y0) < 1e-3
    return on_wall & (np.abs(p[:, 0] - DOOR_X) < 0.62) & (p[:, 2] < 2.30)


def build_windows():
    """The glazing, sampled separately so it can be switched on and off."""
    P, N, W = [], [], []
    for k, x in enumerate(WIN_X):
        # _plane() builds in xy; the glazing is upright, so build it here
        nx = max(2, int(round(WIN_W / 0.07)))
        nz = max(2, int(round(WIN_H / 0.07)))
        gx, gz = np.meshgrid(np.linspace(x - 0.5 * WIN_W, x + 0.5 * WIN_W, nx),
                             np.linspace(WIN_SILL, WIN_SILL + WIN_H, nz))
        p = np.stack([gx.ravel(), np.full(gx.size, BLD_Y0 - 0.04),
                      gz.ravel()], 1).astype(np.float32)
        P.append(p)
        N.append(np.tile([0.0, -1.0, 0.0], (len(p), 1)).astype(np.float32))
        W.append(np.full(len(p), k))
    # the doorway, which is glazed too and lights with the window
    nx = max(2, int(round(1.24 / 0.07)))
    nz = max(2, int(round(2.30 / 0.07)))
    gx, gz = np.meshgrid(np.linspace(DOOR_X - 0.62, DOOR_X + 0.62, nx),
                         np.linspace(0.05, 2.30, nz))
    p = np.stack([gx.ravel(), np.full(gx.size, BLD_Y0 - 0.04),
                  gz.ravel()], 1).astype(np.float32)
    P.append(p)
    N.append(np.tile([0.0, -1.0, 0.0], (len(p), 1)).astype(np.float32))
    W.append(np.full(len(p), OUR_WIN))
    return (np.vstack(P), np.vstack(N), np.concatenate(W).astype(np.int32))


# ---------------------------------------------------------------- schedule

def build_cars():
    arrive, depart = np.zeros(N_CAR, int), np.zeros(N_CAR, int)
    arrive[0], depart[0] = F_ARRIVE, F_DEPART

    a = np.sort(RNG.beta(1.9, 2.4, N_CAR - 1))
    arrive[1:] = np.rint(F_FILL[0] + a * (F_FILL[1] - F_FILL[0])).astype(int)
    d = np.sort(RNG.beta(2.6, 1.7, N_CAR - 1))
    depart[1:] = np.rint(F_EMPTY[0] + d * (F_EMPTY[1] - F_EMPTY[0])).astype(int)

    # Who leaves when must not be predictable from who arrived when, or the
    # ending is a sorting artefact and not an observation. A single shuffle
    # of 23 people has a rank correlation with a standard deviation of about
    # 0.21, so one draw lands at 0.47 often enough to matter -- this seed did.
    # Draw until the schedule actually HAS the property, then let check()
    # measure it back off the built schedule.
    for _ in range(400):
        perm = RNG.permutation(N_CAR - 1)
        rho = np.corrcoef(np.argsort(np.argsort(arrive[1:])),
                          np.argsort(np.argsort(depart[1:][perm])))[0, 1]
        if abs(rho) < 0.18:
            break
    depart[1:] = depart[1:][perm]
    depart[1:] = np.clip(depart[1:], arrive[1:] + 25, F_EMPTY[1])

    # bays fill from the door outward, with enough slop that it is not a
    # perfect gradient -- people take the near space, but not religiously
    bay_of = [0] * N_CAR
    pool = list(range(1, len(BAYS)))
    for j in range(len(pool) - 1):
        if RNG.random() < 0.52:
            pool[j], pool[j + 1] = pool[j + 1], pool[j]
    for k in range(1, N_CAR):
        bay_of[k] = pool[k - 1]

    geo = []
    for k in range(N_CAR):
        r, i = BAYS[bay_of[k]]
        cx, cy = bay_xy(r, i)
        if k == 0:
            length, width, roof, col = 4.55, 1.82, 1.48, 0      # silver
        else:
            van = RNG.random() < 0.16
            length = float(RNG.uniform(4.15, 4.95)) + (0.55 if van else 0.0)
            width = float(RNG.uniform(1.74, 1.90))
            roof = float(RNG.uniform(1.42, 1.56)) + (0.62 if van else 0.0)
            col = 5 if van else int(RNG.integers(0, N_CAR_COL - 1))
        cy = cy + ROW_FACE[r] * (0.5 * BAY_D - 0.5 * length - 0.28)
        geo.append(car(cx, cy, length, width, roof, M_CAR0 + col))
    return geo, arrive, depart, bay_of


# ---------------------------------------------------------------- camera

YAW = math.radians(5.0)
EL = math.radians(54.0)


def pose(p):
    """World (x east, y north, z up) -> screen (right, DOWN, nearness).

    The projector puts +y down the picture, so screen-down is -up. And the
    z-buffer keeps the MAXIMUM, so 'nearness' must grow toward the camera:
    tall is near, far north is far.
    """
    c, s = math.cos(YAW), math.sin(YAW)
    x1 = p[:, 0] * c - p[:, 1] * s
    y1 = p[:, 0] * s + p[:, 1] * c
    ce, se = math.cos(EL), math.sin(EL)
    return np.stack([x1,
                     -y1 * se - p[:, 2] * ce,
                     p[:, 2] * se - y1 * ce], 1)


# ---------------------------------------------------------------- lighting

SH_STEP = 0.24          # SH_X0/Y0/W/H are set once the frame box is known


def _splat(pts, sv):
    """Drop a point cloud onto the ground along the sun ray, as a mask."""
    g = np.zeros(SH_H * SH_W, np.float32)
    if sv[2] < 0.06:
        return g.reshape(SH_H, SH_W)
    k = pts[:, 2] / sv[2]
    gx = pts[:, 0] - k * sv[0]
    gy = pts[:, 1] - k * sv[1]
    ci = np.rint((gx - SH_X0) / SH_STEP).astype(np.int32)
    ri = np.rint((gy - SH_Y0) / SH_STEP).astype(np.int32)
    ok = (ci >= 0) & (ci < SH_W) & (ri >= 0) & (ri < SH_H)
    g[ri[ok] * SH_W + ci[ok]] = 1.0
    g = g.reshape(SH_H, SH_W)
    # close the stipple: a sampled cloud lands as dots, a shadow is a shape
    for _ in range(2):
        g = np.maximum.reduce([g,
                               np.roll(g, 1, 0), np.roll(g, -1, 0),
                               np.roll(g, 1, 1), np.roll(g, -1, 1)])
    return g


def _lookup(mask, x, y):
    ci = np.clip(np.rint((x - SH_X0) / SH_STEP).astype(np.int32), 0, SH_W - 1)
    ri = np.clip(np.rint((y - SH_Y0) / SH_STEP).astype(np.int32), 0, SH_H - 1)
    return mask[ri, ci]


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def pools(pts):
    """Standing light on the ground from the lamp posts and the door lamp.

    Static -- neither the poles nor the ground move -- so this is evaluated
    once at import and only scaled per frame.

    Falloff of 2.5, not the 1.0 it started at. With a gentle falloff,
    sixteen poles sum to more than one EVERYWHERE and the night rendered as
    a single sheet of orange with no lamps discernible in it -- which also
    meant the one shot the whole piece rests on, a car alone in the dark,
    had nothing for the car to be alone in. A car park at night is mostly
    black with bright discs in it. Peak to mid-span is now about 10:1.

    Attenuated above 2.2 m as well, or a roof thirty metres from the
    nearest pole lights up. Below that it is left alone, because a car IS
    below that and a car in a pool of light should be in the pool.
    """
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    v = np.zeros(len(pts), np.float32)
    for px, py in POLES:
        d2 = (x - px) ** 2 + (y - py) ** 2
        v += 1.0 / (1.0 + d2 / (POOL_R * POOL_R)) ** 2.5
    d2 = (x - DOOR_X) ** 2 + (y - BLD_Y0) ** 2
    v += 2.40 / (1.0 + d2 / 40.0) ** 2.0
    return v / (1.0 + np.maximum(0.0, z - 2.2) ** 2 / 2.0)


# ---------------------------------------------------------------- assembly

LOT_P, LOT_N, LOT_M = build_lot()
WIN_P, WIN_N, WIN_ID = build_windows()
CARS, ARRIVE, DEPART, BAY_OF = build_cars()

CAR_P = np.vstack([c[0] for c in CARS])
CAR_N = np.vstack([c[1] for c in CARS])
CAR_M = np.concatenate([c[2] for c in CARS])
CAR_SLICE = []
_o = 0
for c in CARS:
    CAR_SLICE.append((_o, _o + len(c[0])))
    _o += len(c[0])

# What must be in frame: the bay field across, and up to the far roof edge.
SUBJ = np.array([[BAY_X0 - 0.3, ROW_Y[0] - 0.5 * BAY_D - 2.5, 0.0],
                 [BAY_X0 + BAY_W * N_BAY + 0.3, ROW_Y[0] - 0.5 * BAY_D - 2.5,
                  0.0],
                 [BAY_X0 - 0.3, BLD_Y1, BLD_H],
                 [BAY_X0 + BAY_W * N_BAY + 0.3, BLD_Y1, BLD_H]], np.float32)

# Fit to the WHOLE frame, not to the safe box. Camera.fit() defaults to the
# text-safe area because almost every piece this channel has made carries a
# caption. This one carries nothing, and graphics are allowed to bleed, so
# the safe box would throw away a quarter of the picture for no reason.
G.room_c = G.cols / 2.0 - 1.0
G.room_r = G.rows / 2.0 - 1.0
G.cy = G.rows / 2.0
CAM = Camera(G).fit([pose(SUBJ)], margin=1.0)
BOX = frame_box(CAM)

SH_X0, SH_Y0 = BOX[0] - 2.0, BOX[2] - 2.0
SH_W = int((BOX[1] + 2.0 - SH_X0) / SH_STEP)
SH_H = int((BLD_Y1 + 2.0 - SH_Y0) / SH_STEP)

_gp, _gn = build_ground(BOX)
_bp, _bn, _bm = build_building(BOX, CAM)
STATIC_P = np.vstack([_gp, LOT_P, _bp]).astype(np.float32)
STATIC_N = np.vstack([_gn, LOT_N, _bn]).astype(np.float32)
STATIC_M = np.concatenate([np.full(len(_gp), M_ASPH), LOT_M, _bm]
                          ).astype(np.float32)
ALL = np.vstack([STATIC_P, WIN_P, CAR_P])

# Depth cue against FIXED world bounds.
#
# asciilib's depth_cue() normalises against the z range of whatever is on
# screen in the frame it is handed, which is right for a solid object and
# wrong here: a car popping into a bay changes the extremes, which rescales
# the cue over every other cell in the picture. Fifty cars arriving and
# fifty leaving is a hundred moments where the whole lot silently changes
# brightness. Found by diffing the two frames either side of the departure
# and discovering that 89% of what changed was nowhere near the car.
Z_LO, Z_HI = float(pose(ALL)[:, 2].min()), float(pose(ALL)[:, 2].max())


def fixed_cue(z, near=1.0, far=0.93):
    return far + (near - far) * ((z - Z_LO) / (Z_HI - Z_LO))


# static projections, computed once
_S_SC = pose(STATIC_P)
S_COL, S_ROW, S_Z = CAM.project(_S_SC)
_W_SC = pose(WIN_P)
W_COL, W_ROW, W_Z = CAM.project(_W_SC)
_C_SC = pose(CAR_P)
C_COL, C_ROW, C_Z = CAM.project(_C_SC)

S_POOL = pools(STATIC_P).astype(np.float32)
C_POOL = pools(CAR_P).astype(np.float32)

# Per-material ink gain. field() turns brightness straight into ink, so a
# surface that should sit BEHIND the subject has to return less of it. The
# first version gave asphalt the same gain as a car and the lot came out as
# one undifferentiated field of noise with cars hidden in it.
MAT_GAIN = np.ones(len(STATIC_M), np.float32)
MAT_GAIN[STATIC_M == M_ASPH] = 0.74
MAT_GAIN[STATIC_M == M_PAINT] = 1.06
MAT_GAIN[STATIC_M == M_WALL] = 1.38
MAT_GAIN[STATIC_M == M_ROOF] = 0.92
MAT_GAIN[STATIC_M == M_POLE] = 1.10

# the building and the poles cast on everything; the cars only cast on the
# ground, or every car would stand in its own shadow
CASTERS = STATIC_P[(STATIC_M == M_WALL) | (STATIC_M == M_ROOF)
                   | (STATIC_M == M_POLE)]


def present(f):
    return (ARRIVE <= f) & (f < DEPART)


# The colour of the light itself, walked through waypoints by solar
# altitude. A straight two-stop lerp from night-blue to daylight passes
# through grey at exactly the altitude half the frames live at
# (RENDERER.md trap 17), so the day is given stops it has to hit.
_TINT_ALT = (-14.0, -4.0, 0.5, 7.0, 20.0, 40.0)
_TINT_COL = ((0.66, 0.78, 1.10),        # night, and the sky is the only lamp
             (0.80, 0.84, 1.08),        # nautical twilight
             (1.24, 0.94, 0.74),        # the sun on the horizon
             (1.18, 1.00, 0.86),        # low morning / late afternoon
             (1.06, 1.02, 0.96),
             (1.00, 1.00, 1.00))        # noon
_TINT = (1.0, 1.0, 1.0)


def day_tint(alt):
    a = np.interp(alt, _TINT_ALT, [c[0] for c in _TINT_COL])
    b = np.interp(alt, _TINT_ALT, [c[1] for c in _TINT_COL])
    c = np.interp(alt, _TINT_ALT, [c[2] for c in _TINT_COL])
    return (float(a), float(b), float(c))


_BASE = {M_ASPH: ASPHALT, M_PAINT: PAINT, M_WALL: WALL, M_ROOF: ROOFC,
         M_POLE: POLEC, M_GLASS: GLASSC}


def colour(shade, extra):
    m = int(extra)
    w = float(extra) - m
    if m == M_WIN:
        return WINC
    base = _BASE.get(m)
    if base is None:
        base = CARC[min(N_CAR_COL - 1, max(0, m - M_CAR0))]
    base = (base[0] * _TINT[0], base[1] * _TINT[1], base[2] * _TINT[2])
    if w > 0.02:
        # sodium wins where sodium is what is actually falling on it, which
        # is why the lamps stay orange while the sky above them goes blue
        base = tuple(base[i] * (1.0 - w) + SODIUM[i] * w for i in range(3))
    # tint with a floor and let the glyph carry the light (trap 12)
    k = 0.62 + 0.38 * shade
    return (min(1.0, base[0] * k), min(1.0, base[1] * k),
            min(1.0, base[2] * k))


def draw(f):
    global _TINT
    minute = T0_MIN + f
    sv, alt = sun_vec(minute)
    direct = max(0.0, math.sin(math.radians(alt))) ** 0.88
    amb = 0.020 + 0.480 * float(smoothstep(-13.0, 9.0, alt))
    night = 1.0 - float(smoothstep(-2.0, 6.0, alt))
    _TINT = day_tint(alt)
    on = present(f)

    fr = Frame(G, BG)
    col, row, z, sh, ex = [], [], [], [], []

    # ---- shadows
    cast = [CASTERS]
    for k in range(N_CAR):
        if on[k]:
            a, b = CAR_SLICE[k]
            cast.append(CAR_P[a:b])
    ground_shadow = _splat(np.vstack(cast), sv)
    solid_shadow = _splat(CASTERS, sv)

    # ---- static
    up = np.clip(STATIC_N[:, 2], 0.0, 1.0)
    lam = np.clip(STATIC_N @ sv, 0.0, 1.0)
    sd = _lookup(ground_shadow, STATIC_P[:, 0], STATIC_P[:, 1])
    sd = np.where(STATIC_P[:, 2] > 0.06,
                  _lookup(solid_shadow, STATIC_P[:, 0], STATIC_P[:, 1]), sd)
    lampy = S_POOL * night * 0.39 * (0.30 + 0.70 * up)
    s = (amb * (0.58 + 0.42 * up) + 0.94 * direct * lam * (1.0 - 0.86 * sd)
         + lampy) * MAT_GAIN
    col.append(S_COL); row.append(S_ROW); z.append(S_Z)
    sh.append(s)
    ex.append(STATIC_M + np.clip(lampy / (s + 1e-4), 0.0, 0.97) * 0.97)

    # ---- windows: two states, on and off. the only event in the piece
    #      that a viewer can actually see happen at one frame a minute.
    lit = np.zeros(len(WIN_P), np.float32)
    ours = (WIN_ID == OUR_WIN) & (F_WIN_ON <= f) & (f < F_WIN_OFF)
    lit[ours] = 1.0
    ws = np.where(lit > 0, 0.97, 0.16 * amb + 0.05 * direct)
    col.append(W_COL); row.append(W_ROW); z.append(W_Z)
    sh.append(ws)
    ex.append(np.where(lit > 0, float(M_WIN), float(M_GLASS)))

    # ---- cars
    if on.any():
        idx = np.concatenate([np.arange(*CAR_SLICE[k])
                              for k in range(N_CAR) if on[k]])
        n = CAR_N[idx]
        up = np.clip(n[:, 2], 0.0, 1.0)
        lam = np.clip(n @ sv, 0.0, 1.0)
        sd = _lookup(solid_shadow, CAR_P[idx, 0], CAR_P[idx, 1])
        # Car paint is glossy and asphalt is not. Without this the roof of
        # a car under a sodium lamp returns the same ink as the tarmac
        # beside it and the whole lot reads as empty after dark -- which is
        # fatal, because after dark is when there is one car in it.
        lampy = C_POOL[idx] * night * (0.42 * (0.30 + 0.70 * up)
                                       + 0.62 * up ** 4)
        s = (amb * (0.58 + 0.42 * up) + 0.72 * direct * lam * (1.0 - 0.70 * sd)
             + 0.30 * direct * lam ** 4 + lampy)
        col.append(C_COL[idx]); row.append(C_ROW[idx]); z.append(C_Z[idx])
        sh.append(s)
        ex.append(CAR_M[idx] + np.clip(lampy / (s + 1e-4), 0.0, 0.97) * 0.97)

    col = np.concatenate(col); row = np.concatenate(row)
    z = np.concatenate(z); sh = np.concatenate(sh); ex = np.concatenate(ex)

    ok = visible(G, col, row)
    col, row, z, sh, ex = col[ok], row[ok], z[ok], sh[ok], ex[ok]
    _, keep = zbuffer(G, col, row, z)
    sh = np.clip(sh, 0.0, 1.0) * fixed_cue(z)
    fr.field(col, row, keep, sh, colour, RAMP, extra=ex)
    return fr


# ---------------------------------------------------------------- checks

def hhmm(minute):
    return "%02d:%02d" % (int(minute) // 60, int(round(minute)) % 60)


def check():
    print("THE LOT -- %s-%02d-%02d, Melbourne (%.4f, %.4f), UTC+%g"
          % (DATE[0], DATE[1], DATE[2], LAT, LON, TZ))

    # 1. the sun, against a service that shares no code with this file
    ref_rise, ref_set = 6 * 60 + 50.65, 17 * 60 + 54.05      # sunrise-sunset.org
    print("  sunrise      %s  (reference %s, %+0.1f min)"
          % (hhmm(SUNRISE), hhmm(ref_rise), SUNRISE - ref_rise))
    print("  sunset       %s  (reference %s, %+0.1f min)"
          % (hhmm(SUNSET), hhmm(ref_set), SUNSET - ref_set))
    assert abs(SUNRISE - ref_rise) < 3.0 and abs(SUNSET - ref_set) < 3.0

    # 2. day length, derived a SECOND way: the hour-angle formula, which
    #    never calls sun_at() and never bisects anything
    jd = _julian(DATE[0], DATE[1], DATE[2], 12.0 - TZ)
    dec = math.radians(_sun_dec_eot(jd)[0])
    p = math.radians(LAT)
    cosH = ((math.cos(math.radians(90.833)) - math.sin(p) * math.sin(dec))
            / (math.cos(p) * math.cos(dec)))
    h_len = 2.0 * math.degrees(math.acos(cosH)) / 15.0
    b_len = (SUNSET - SUNRISE) / 60.0
    print("  day length   %dh %02dm by bisection, %dh %02dm by hour angle"
          % (int(b_len), round((b_len % 1) * 60),
             int(h_len), round((h_len % 1) * 60)))
    assert abs(h_len - b_len) < 0.05, (h_len, b_len)

    # 3. both ends of the shift are genuinely dark -- measured off the solar
    #    code, not asserted by hand
    a_in = sun_at(T0_MIN + F_ARRIVE)[0]
    a_out = sun_at(T0_MIN + F_DEPART)[0]
    print("  arrives      %s, sun %+.1f deg   (%d min before sunrise)"
          % (hhmm(T0_MIN + F_ARRIVE), a_in, SUNRISE - T0_MIN - F_ARRIVE))
    print("  leaves       %s, sun %+.1f deg   (%d min after sunset)"
          % (hhmm(T0_MIN + F_DEPART), a_out, T0_MIN + F_DEPART - SUNSET))
    assert a_in < -6.0 and a_out < -6.0, (a_in, a_out)
    shift = (F_DEPART - F_ARRIVE) / 60.0
    print("  shift        %.2f h, of which %.2f h outside daylight"
          % (shift, shift - b_len))

    # 4. first in, last out -- read off the OCCUPANCY, not off the schedule
    occ = np.array([present(f) for f in range(N_FRAME)])
    any_f = np.where(occ.any(1))[0]
    first, last = any_f[0], any_f[-1]
    assert occ[first].sum() == 1 and occ[first][0], "first car is not car 0"
    assert occ[last].sum() == 1 and occ[last][0], "last car is not car 0"
    alone = int((occ.sum(1) == 1).sum())
    peak = int(occ.sum(1).max())
    print("  occupancy    peak %d of %d bays, %d cars total"
          % (peak, len(BAYS), N_CAR))
    print("  alone        %d frames = %d min (%.1f s of video)"
          % (alone, alone, alone / float(FPS)))
    solo_tail = int((occ[F_DEPART - 200:F_DEPART].sum(1) == 1).sum())
    print("  first in %s, last out %s, and it never moved bay"
          % (hhmm(T0_MIN + first), hhmm(T0_MIN + last)))
    assert solo_tail > 60, solo_tail

    # arriving order must not predict leaving order, or the ending is a
    # sorting artefact rather than an observation
    rk = np.argsort(np.argsort(ARRIVE[1:]))
    rd = np.argsort(np.argsort(DEPART[1:]))
    rho = float(np.corrcoef(rk, rd)[0, 1])
    print("  order        arrival vs departure rank correlation %+.3f" % rho)
    assert abs(rho) < 0.40, rho

    # 5. the frame is a minute -- two ways
    print("  rate         %d frames for %d minutes = %.1f s at %d fps"
          % (N_FRAME, N_FRAME, DUR, FPS))
    assert abs(RATE - 60.0 * FPS) < 1e-6
    assert N_FRAME * 60.0 / DUR == RATE

    # the SUBJECT has to be inside the frame ...
    c, r, _ = CAM.project(pose(SUBJ))
    print("  subject      c%d..%d r%d..%d of %dx%d"
          % (c.min(), c.max(), r.min(), r.max(), G.cols, G.rows))
    assert 0 <= c.min() and c.max() < G.cols
    assert 0 <= r.min() and r.max() < G.rows

    # ... and the ground has to run off every edge of it. Not "the asphalt
    # is big", which is easy to assert and easy to be wrong about -- every
    # single cell of the frame must have static geometry landing in it, or
    # somewhere there is a hard edge with the void behind it.
    # (the glazing counts -- the wall has real holes cut in it where the
    # windows are, and the windows are what fills them. rows 0 and rows-1
    # are never drawable: asciilib clips them.)
    c, r, z = CAM.project(pose(np.vstack([STATIC_P, WIN_P])))
    ok = visible(G, c, r)
    hit = np.zeros(G.rows * G.cols, bool)
    hit[r[ok] * G.cols + c[ok]] = True
    hit = hit.reshape(G.rows, G.cols)[1:G.rows - 1]
    miss = int((~hit).sum())
    print("  coverage     %d of %d drawable cells filled, %d bare"
          % (hit.sum(), hit.size, miss))
    assert miss == 0, miss
    print("  points       %d static, %d glazing, %d car"
          % (len(STATIC_P), len(WIN_P), len(CAR_P)))

    # 6. AND IT IS ACTUALLY VISIBLE. Everything above can pass on a render
    # where the last car is a dark smudge nobody can find -- that is exactly
    # what the first two versions of this were. So: draw the frame before it
    # leaves and a frame after, in the dark, and diff the finished pixels.
    # Whatever changed had better be car-shaped and standing in our bay.
    # Draw the SAME minute twice, once with the car held one minute longer.
    # Comparing consecutive frames does not work: a one-minute change in
    # ambient light nudges a few hundred cells across a glyph boundary, and
    # one ramp step is a big pixel-level change, so the diff came back 89%
    # noise spread over the whole frame. Same light, car present or not, is
    # the only comparison that isolates the car.
    DEPART[0] += 1
    before = _lum(F_DEPART)
    DEPART[0] -= 1
    after = _lum(F_DEPART)
    d = np.abs(before - after) > 0.045
    ys, xs = np.nonzero(d)
    c0, r0, _ = CAM.project(pose(CAR_P[CAR_SLICE[0][0]:CAR_SLICE[0][1]]))
    want = (float(c0.mean()) * G.cell, float(r0.mean()) * G.cell)
    off = math.hypot(xs.mean() - want[0], ys.mean() - want[1]) / G.cell
    print("  visible      %d px change when it leaves, centred %.1f cells "
          "from the bay" % (d.sum(), off))
    # at least a quarter of the car's own footprint has to actually change,
    # or it is a smudge that happens to be in the right place
    a0, b0 = CAR_SLICE[0]
    foot = ((c0.max() - c0.min() + 1) * (r0.max() - r0.min() + 1)
            * G.cell * G.cell)
    print("  visible      %.0f%% of its own footprint" % (100.0 * d.sum() / foot))
    assert d.sum() > 0.25 * foot, (d.sum(), foot)
    assert off < 6.0, off

    noon, night = _lum(462).mean(), _lum(0).mean()
    print("  levels       noon %.4f ink, 04:40 %.4f -> the day is %.1fx"
          % (noon, night, noon / night))
    assert noon > 2.0 * night, (noon, night)
    print("  no text on screen at any frame.")
    return True


def _lum(f):
    a = np.frombuffer(draw(f).surface.get_data(), np.uint8)
    return a.reshape(G.h_px, G.w_px, 4)[:, :, :3].astype(np.float32).mean(2) / 255.0


def ink(f):
    """Lit cells and mean ink of one frame -- for the on-screen check."""
    fr = draw(f)
    buf = np.frombuffer(fr.surface.get_data(), np.uint8).reshape(
        G.h_px, G.w_px, 4)[:, :, :3].astype(np.float32).mean(2) / 255.0
    return float(buf.mean()), int((buf > 0.10).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--sheet", action="store_true")
    a = ap.parse_args()
    check()
    if a.check and not a.sheet:
        return
    marks = [0, F_ARRIVE - 2, F_ARRIVE + 8, 120, 300, 462, 640, 760,
             810, F_DEPART - 2, F_DEPART + 12, 900]
    if a.sheet:
        labels = ["%s  f%d" % (hhmm(T0_MIN + m), m) for m in marks]
        contact([draw(m) for m in marks],
                os.path.join(OUT, "lot_sheet.png"), cols=4, labels=labels)
        return
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "lot.mp4")
    with Encoder(path, G, fps=FPS, crf=19) as enc:
        for f in range(N_FRAME):
            enc.write(draw(f))
            if f % 60 == 0:
                print("  %s  %d/%d" % (hhmm(T0_MIN + f), f, N_FRAME),
                      flush=True)
    print("wrote %s  %d frames  %.3f s" % (path, N_FRAME, DUR))


if __name__ == "__main__":
    main()
