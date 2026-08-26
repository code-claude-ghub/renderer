#!/usr/bin/env python3
"""
RING -- 22 cars on a 230 m loop of road. Nothing is blocking it.

Every car is driving forward. The jam is travelling backward.

The cars are simulated with the Intelligent Driver Model (Treiber, Hennecke &
Helbing 2000), which is collision-free by construction, on a ring road the
size of the one in Sugiyama et al. 2008 -- 230 m, 22 vehicles, drivers asked
for 30 km/h, no bottleneck anywhere. The jam is not placed. It emerges, and
then it holds together and walks upstream.

This one does not loop, and it cannot. A settled stop-and-go wave repeats
itself only up to a rotation: after 3.44 s the cars are arranged exactly as
they were, but eleven metres further back round the ring. The picture would
close only when the wave had gone all the way round, which takes 73 s. To
close it in ten you would need a road about 32 m long, and a traffic jam does
not fit on 32 m of road. So the piece just runs and stops.

    python3 scripts/ring.py --check
    python3 scripts/ring.py --stills /tmp/ring
    python3 scripts/ring.py --out content/ring.mp4

numpy + pycairo + ffmpeg.
"""

import argparse
import math
import os
import subprocess
import sys

import cairo
import numpy as np

# ---------------------------------------------------------------- the road

L_RING = 230.0          # m, circumference -- Sugiyama et al. 2008
N_CAR = 22              # vehicles          -- ditto (their critical density)
CAR_L = 4.5             # m, car length
CAR_W = 1.85            # m, car width
TRACK_W = 4.8           # m, paved width -- one lane, as on the day

# ------------------------------------------------------- the drivers (IDM)

V0 = 8.333              # m/s desired speed = 30 km/h, as instructed on the day
T_HW = 1.2              # s  desired time headway
S_MIN = 2.0             # m  gap kept at a standstill
A_MAX = 0.7             # m/s^2 acceleration
B_CMF = 1.5             # m/s^2 comfortable braking
DELTA = 4.0

DT = 0.001              # s, integration step for the recorded window
DT_BURN = 0.005         # s, coarser step while the jam is still forming
BURN = 2000.0           # s, let the jam find itself
TARGET_S = 10.0         # s, aim the duration at this and land on a repeat

# ------------------------------------------------------------- the picture

W, H = 1080, 1920
FPS = 30
SS = 2                  # supersample factor

MARGIN_X, MARGIN_Y = 40, 60

GRASS = np.array([0.296, 0.352, 0.212])
ASPHALT = np.array([0.300, 0.292, 0.298])
LINE = np.array([0.895, 0.885, 0.840])
BODY = np.array([0.880, 0.878, 0.860])
GLASS = np.array([0.150, 0.170, 0.205])
BRAKE = np.array([0.960, 0.130, 0.070])

SUN = (0.42, 0.60)      # shadow offset direction, in car-lengths, screen space


# ---------------------------------------------------------------- geometry
# A stadium: two straights up the sides, two semicircles top and bottom.
# Solve R and Ls so the outline just fills the frame and the perimeter is
# exactly L_RING.

def solve_track():
    """Return (scale px/m, R metres, Ls metres)."""
    usable_w = W - 2 * MARGIN_X
    usable_h = H - 2 * MARGIN_Y
    # 2R + TRACK_W = usable_w / k ;  Ls = (usable_h - usable_w) / k
    # 2*pi*R + 2*Ls = L_RING
    num = math.pi * usable_w + 2.0 * (usable_h - usable_w)
    k = num / (L_RING + math.pi * TRACK_W)
    R = (usable_w / k - TRACK_W) / 2.0
    Ls = (usable_h - usable_w) / k
    return k, R, Ls


K_PX, R_M, LS_M = solve_track()
assert abs(2 * math.pi * R_M + 2 * LS_M - L_RING) < 1e-9

ARC_TOP = math.pi * R_M


def pos_head(s):
    """Arc length -> (x, y) in metres (y up, origin centre) and heading rad."""
    s = np.asarray(s, dtype=np.float64) % L_RING
    x = np.empty_like(s)
    y = np.empty_like(s)
    th = np.empty_like(s)

    a = s < LS_M                                        # right straight, up
    x[a] = R_M
    y[a] = -LS_M / 2.0 + s[a]
    th[a] = math.pi / 2.0

    b = (s >= LS_M) & (s < LS_M + ARC_TOP)              # over the top
    t = (s[b] - LS_M) / R_M
    x[b] = R_M * np.cos(t)
    y[b] = LS_M / 2.0 + R_M * np.sin(t)
    th[b] = t + math.pi / 2.0

    c = (s >= LS_M + ARC_TOP) & (s < 2 * LS_M + ARC_TOP)  # left straight, down
    x[c] = -R_M
    y[c] = LS_M / 2.0 - (s[c] - LS_M - ARC_TOP)
    th[c] = -math.pi / 2.0

    d = s >= 2 * LS_M + ARC_TOP                         # under the bottom
    t = math.pi + (s[d] - 2 * LS_M - ARC_TOP) / R_M
    x[d] = R_M * np.cos(t)
    y[d] = -LS_M / 2.0 + R_M * np.sin(t)
    th[d] = t + math.pi / 2.0

    return x, y, th


def to_px(x, y):
    """Metres (y up, centred) -> pixels (y down)."""
    return W / 2.0 + x * K_PX, H / 2.0 - y * K_PX


def dist_to_centreline(px, py):
    """Pixel grid -> metres from the centreline of the road."""
    x = (px - W / 2.0) / K_PX
    y = (H / 2.0 - py) / K_PX
    flat = np.abs(y) <= LS_M / 2.0
    d = np.where(
        flat,
        np.abs(np.abs(x) - R_M),
        np.abs(np.hypot(x, np.maximum(np.abs(y) - LS_M / 2.0, 0.0)) - R_M),
    )
    return d


# ---------------------------------------------------------------- the sim

def idm_step(x, v, dt):
    s = (np.roll(x, -1) - x) % L_RING - CAR_L
    dv = v - np.roll(v, -1)
    s_star = S_MIN + np.maximum(
        0.0, v * T_HW + v * dv / (2.0 * math.sqrt(A_MAX * B_CMF))
    )
    acc = A_MAX * (1.0 - (v / V0) ** DELTA - (s_star / np.maximum(s, 0.05)) ** 2)
    v_new = np.maximum(0.0, v + acc * dt)
    return v_new, acc, s


def simulate(record_s):
    """Burn in, then record `record_s` seconds. Returns unwrapped x, v, acc."""
    x = np.arange(N_CAR, dtype=np.float64) * (L_RING / N_CAR)
    x += np.random.default_rng(5).normal(0.0, 0.10, N_CAR)
    v = np.full(N_CAR, 1.0)

    for _ in range(int(BURN / DT_BURN)):
        v, _, _ = idm_step(x, v, DT_BURN)
        x = (x + v * DT_BURN) % L_RING
    for _ in range(int(20.0 / DT)):          # settle again at the fine step
        v, _, _ = idm_step(x, v, DT)
        x = (x + v * DT) % L_RING

    n = int(record_s / DT) + 2
    X = np.empty((n, N_CAR))
    V = np.empty((n, N_CAR))
    A = np.empty((n, N_CAR))
    xu = x.copy()
    for i in range(n):
        v, acc, _ = idm_step(x, v, DT)
        X[i] = xu
        V[i] = v
        A[i] = acc
        step = v * DT
        x = (x + step) % L_RING
        xu = xu + step
    return X, V, A


def find_period(X, tol=0.004, tmin=1.0, tmax=8.0):
    """
    The FUNDAMENTAL period: the smallest t at which the cars are arranged as
    they were, with the labels rotated -- but sitting `slide` metres further
    round the ring. It is a repeat of the pattern, not of the picture.
    Returns (err_metres, tau_seconds, index_shift, slide_metres).
    """
    base = np.sort(X[0] % L_RING)
    best = None
    for i in range(int(tmin / DT), min(int(tmax / DT), len(X))):
        cur = np.sort(X[i] % L_RING)
        for sh in range(N_CAR):
            d = (np.roll(cur, sh) - base + L_RING / 2.0) % L_RING - L_RING / 2.0
            err = float(np.abs(d - np.median(d)).max())
            if best is None or err < best[0]:
                best = (err, i * DT, sh, float(np.median(d)))
        if best[0] < tol and i * DT > best[1] + 0.5:
            break                              # first good repeat wins
    return best


def sample(X, V, A, t):
    """Linear interpolation of the recorded state at time t."""
    f = t / DT
    i = int(f)
    w = f - i
    i = min(i, len(X) - 2)
    return (
        X[i] * (1 - w) + X[i + 1] * w,
        V[i] * (1 - w) + V[i + 1] * w,
        A[i] * (1 - w) + A[i + 1] * w,
    )


# ------------------------------------------------------------- the drawing

def rounded(cr, cx, cy, w, h, r):
    cr.new_sub_path()
    cr.arc(cx + w / 2 - r, cy - h / 2 + r, r, -math.pi / 2, 0)
    cr.arc(cx + w / 2 - r, cy + h / 2 - r, r, 0, math.pi / 2)
    cr.arc(cx - w / 2 + r, cy + h / 2 - r, r, math.pi / 2, math.pi)
    cr.arc(cx - w / 2 + r, cy - h / 2 + r, r, math.pi, 3 * math.pi / 2)
    cr.close_path()


def stadium_path(cr, radius):
    """Centreline stadium, offset outward by (radius - R_M) metres."""
    r = radius * K_PX
    hl = LS_M / 2.0 * K_PX
    cx, cy = W / 2.0 * SS, H / 2.0 * SS
    r *= SS
    hl *= SS
    cr.new_path()
    cr.arc(cx, cy - hl, r, math.pi, 2 * math.pi)      # top semicircle
    cr.line_to(cx + r, cy + hl)
    cr.arc(cx, cy + hl, r, 0, math.pi)                # bottom semicircle
    cr.close_path()


def draw_car(cr, px, py, heading, braking):
    cl = CAR_L * K_PX * SS
    cw = CAR_W * K_PX * SS
    cr.save()
    cr.translate(px * SS, py * SS)
    cr.rotate(-heading)                                # screen y is flipped
    r = cw * 0.30

    if braking:
        g = cairo.RadialGradient(-cl * 0.46, 0, 0, -cl * 0.46, 0, cw * 1.15)
        g.add_color_stop_rgba(0.0, *BRAKE, 0.55)
        g.add_color_stop_rgba(1.0, *BRAKE, 0.0)
        cr.set_source(g)
        cr.arc(-cl * 0.46, 0, cw * 1.15, 0, 2 * math.pi)
        cr.fill()

    cr.set_source_rgb(*BODY)
    rounded(cr, 0, 0, cl, cw, r)
    cr.fill()

    # a shallow crease down the bonnet so the body is not one flat value
    cr.set_source_rgba(1.0, 1.0, 0.98, 0.55)
    rounded(cr, cl * 0.06, 0, cl * 0.80, cw * 0.30, cw * 0.14)
    cr.fill()

    cr.set_source_rgb(*GLASS)
    rounded(cr, -cl * 0.02, 0, cl * 0.44, cw * 0.78, cw * 0.20)
    cr.fill()
    cr.set_source_rgba(0.62, 0.70, 0.78, 0.60)         # sky in the windscreen
    rounded(cr, cl * 0.13, 0, cl * 0.07, cw * 0.68, cw * 0.12)
    cr.fill()

    if braking:
        cr.set_source_rgb(*BRAKE)
        for sgn in (-1, 1):
            rounded(cr, -cl * 0.455, sgn * cw * 0.30, cl * 0.05, cw * 0.26,
                    cw * 0.05)
            cr.fill()
    cr.restore()


def draw_shadow(cr, px, py, heading):
    cl = CAR_L * K_PX * SS
    cw = CAR_W * K_PX * SS
    cr.save()
    cr.translate(px * SS + SUN[0] * cw, py * SS + SUN[1] * cw)
    cr.rotate(-heading)
    for grow, alpha in ((1.22, 0.055), (1.11, 0.075), (1.0, 0.170)):
        cr.set_source_rgba(0.06, 0.07, 0.05, alpha)
        rounded(cr, 0, 0, cl * grow, cw * grow, cw * 0.30 * grow)
        cr.fill()
    cr.restore()


def render_frame(surf, cr, s_pos, headings, braking):
    cr.set_source_rgb(*GRASS)
    cr.paint()

    # the road
    cr.set_line_cap(cairo.LINE_CAP_BUTT)
    cr.set_source_rgb(*ASPHALT)
    cr.set_line_width(TRACK_W * K_PX * SS)
    stadium_path(cr, R_M)
    cr.stroke()

    # distance posts every 10 m on the grass -- the only fixed marks in the
    # frame, and the thing the drift of the jam can be read against
    sm = np.arange(0.0, L_RING, 10.0)
    xs, ys, ths = pos_head(sm)
    for x, y, th in zip(xs, ys, ths):
        nx, ny = math.sin(th), -math.cos(th)           # outward normal
        ax, ay = to_px(x + nx * (TRACK_W / 2 + 0.7), y + ny * (TRACK_W / 2 + 0.7))
        bx, by = to_px(x + nx * (TRACK_W / 2 + 2.1), y + ny * (TRACK_W / 2 + 2.1))
        cr.set_source_rgba(0.10, 0.12, 0.07, 0.30)     # its shadow
        cr.set_line_width(0.34 * K_PX * SS)
        cr.move_to(ax * SS + 5, ay * SS + 7)
        cr.line_to(bx * SS + 5, by * SS + 7)
        cr.stroke()
        cr.set_source_rgb(0.86, 0.85, 0.80)
        cr.set_line_width(0.30 * K_PX * SS)
        cr.move_to(ax * SS, ay * SS)
        cr.line_to(bx * SS, by * SS)
        cr.stroke()

    # edge lines
    cr.set_source_rgb(*LINE)
    cr.set_line_width(0.16 * K_PX * SS)
    stadium_path(cr, R_M + TRACK_W / 2 - 0.20)
    cr.stroke()
    stadium_path(cr, R_M - TRACK_W / 2 + 0.20)
    cr.stroke()

    xs, ys, ths = pos_head(s_pos)
    pxs, pys = to_px(xs, ys)
    for px, py, th in zip(pxs, pys, ths):
        draw_shadow(cr, px, py, th)
    for px, py, th, br in zip(pxs, pys, ths, braking):
        draw_car(cr, px, py, th, br)

    surf.flush()
    buf = np.ndarray(shape=(H * SS, W * SS, 4), dtype=np.uint8,
                     buffer=surf.get_data())
    img = buf[:, :, [2, 1, 0]].astype(np.float32) / 255.0
    img = img.reshape(H, SS, W, SS, 3).mean(axis=(1, 3))
    return img


# ------------------------------------------------------- light and texture

def value_noise(cell, rng):
    """Smooth bilinear/smoothstep value noise -- no visible lattice blocks."""
    gh, gw = H // cell + 2, W // cell + 2
    g = rng.normal(0.0, 1.0, (gh, gw))
    ys = np.arange(H) / cell
    xs = np.arange(W) / cell
    y0 = ys.astype(int)
    x0 = xs.astype(int)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    fy = fy * fy * (3.0 - 2.0 * fy)
    fx = fx * fx * (3.0 - 2.0 * fx)
    a = g[y0][:, x0]
    b = g[y0][:, x0 + 1]
    c = g[y0 + 1][:, x0]
    d = g[y0 + 1][:, x0 + 1]
    return ((a * (1 - fx) + b * fx) * (1 - fy)
            + (c * (1 - fx) + d * fx) * fy)


def make_grain():
    """Static ground texture + a light gradient, so no region is one value."""
    rng = np.random.default_rng(11)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    on_road = dist_to_centreline(xx, yy) < TRACK_W / 2.0

    # grass: broad mown unevenness, then a little tuft detail
    grass = (value_noise(150, rng) * 0.028
             + value_noise(46, rng) * 0.017
             + value_noise(13, rng) * 0.009
             + rng.normal(0.0, 1.0, (H, W)) * 0.006)
    # asphalt: patchy wear plus a fine aggregate speckle
    road = (value_noise(90, rng) * 0.013
            + value_noise(11, rng) * 0.007
            + rng.normal(0.0, 1.0, (H, W)) * 0.009)

    grain = np.where(on_road, road, grass)

    # low sun from the top-left, plus a soft falloff into the corners
    ramp = 0.055 * (1.0 - (xx / W) * 0.55 - (yy / H) * 0.85)
    rad = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2))
    vig = -0.075 * np.clip(rad - 0.55, 0.0, None) ** 2 * 2.2
    return (grain + ramp + vig).astype(np.float32)


GRAIN = None


def finish(img):
    global GRAIN
    if GRAIN is None:
        GRAIN = make_grain()
    x = img + GRAIN[:, :, None]
    x = np.maximum(x, 0.0)
    x = 0.022 + x * 0.978           # lift the black point off the floor
    x = x / (1.0 + x * 0.16)
    x = np.clip(x * 1.14, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)


# -------------------------------------------------------------------- runs

def build():
    X, V, A = simulate(16.0)
    err, tau1, shift, slide1 = find_period(X)
    reps = max(1, int(round(TARGET_S / tau1)))
    tau = tau1 * reps
    n_frame = int(round(tau * FPS))
    return X, V, A, tau1, tau, reps, shift, slide1, err, n_frame


def frames(X, V, A, tau, n_frame):
    for i in range(n_frame):
        x, v, a = sample(X, V, A, i * tau / n_frame)
        yield i, x, v, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--stills')
    ap.add_argument('--out')
    args = ap.parse_args()

    X, V, A, tau1, tau, reps, shift, slide1, err, n_frame = build()
    wave = slide1 / tau1
    print(f"road {L_RING:.0f} m, {N_CAR} cars, {K_PX:.2f} px/m "
          f"(R {R_M:.1f} m, straights {LS_M:.1f} m)")
    print(f"pattern repeats every {tau1:.3f} s, {slide1:+.2f} m round the ring "
          f"({shift} car, closure {err * 1000:.2f} mm)")
    print(f"piece is {reps} of those -> {n_frame} frames "
          f"({n_frame / FPS:.2f} s at {FPS} fps)")
    print(f"wave {wave:+.3f} m/s ({wave * 3.6:+.2f} km/h)")

    if args.check:
        return run_checks(X, V, A, tau1, tau, reps, shift, slide1, err, n_frame)

    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W * SS, H * SS)
    cr = cairo.Context(surf)
    cr.set_antialias(cairo.ANTIALIAS_BEST)

    if args.stills:
        from PIL import Image
        marks = [0, n_frame // 8, n_frame // 4, n_frame // 2,
                 3 * n_frame // 4, n_frame - 1]
        for i, x, v, a in frames(X, V, A, tau, n_frame):
            if i not in marks:
                continue
            img = render_frame(surf, cr, x % L_RING, None,
                               (a < -0.10) | (v < 0.20))
            Image.fromarray(finish(img)).save(f"{args.stills}_{i:04d}.png")
            print(f"  still {i}")
        return 0

    if not args.out:
        print("nothing to do -- pass --check, --stills or --out")
        return 1

    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{W}x{H}', '-r', str(FPS), '-i', 'pipe:0',
        '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '17',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', args.out,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for i, x, v, a in frames(X, V, A, tau, n_frame):
        img = render_frame(surf, cr, x % L_RING, None, (a < -0.10) | (v < 0.20))
        p.stdin.write(finish(img).tobytes())
        if i % 30 == 0:
            print(f"  frame {i}/{n_frame}", flush=True)
    p.stdin.close()
    p.wait()
    print(f"wrote {args.out}")
    return 0


# ------------------------------------------------------------------ checks

def run_checks(X, V, A, tau1, tau, reps, shift, slide1, err, n_frame):
    ok = True

    def t(cond, msg):
        nonlocal ok
        print(("ok  " if cond else "FAIL ") + msg)
        ok = ok and bool(cond)

    xs = np.array([sample(X, V, A, i * tau / n_frame)[0]
                   for i in range(n_frame + 1)])
    vs = np.array([sample(X, V, A, i * tau / n_frame)[1]
                   for i in range(n_frame + 1)])
    dur = n_frame / FPS
    wave = slide1 / tau1

    step = np.diff(xs, axis=0)
    t(step.min() > -1e-9,
      f"every car only ever moves forward -- worst step {step.min():+.1e} m, "
      f"which is float noise on a car that is standing still")

    t(wave_backward(X, tau, n_frame),
      "and the jam only ever moves backward -- monotone over every frame")

    t(wave < 0,
      f"the jam travels against the traffic at {abs(wave):.2f} m/s "
      f"({abs(wave) * 3.6:.1f} km/h)")

    mean_v = vs.mean()
    t(mean_v > 0 and abs(wave) > mean_v,
      f"and it travels backward faster than the cars manage forward -- "
      f"cars average {mean_v:.2f} m/s, the jam does {abs(wave):.2f}")

    gaps = np.array([(np.roll(x % L_RING, -1) - (x % L_RING)) % L_RING - CAR_L
                     for x in xs])
    t(gaps.min() > 0.0, f"nobody hits anybody -- closest gap {gaps.min():.2f} m")

    stopped = (vs < 0.20).sum(axis=1)
    t(stopped.min() >= 8,
      f"there is always a solid block of stopped cars -- never fewer than "
      f"{stopped.min()} of {N_CAR} at a dead stop")

    t(vs.max() > 6.0,
      f"and cars out in the open get properly up to speed -- "
      f"fastest {vs.max():.2f} m/s ({vs.max() * 3.6:.1f} km/h)")

    travelled = xs[-1] - xs[0]
    t(travelled.min() < 0.5,
      f"at least one car does not move at all for the whole piece -- "
      f"stillest car covers {travelled.min():.2f} m while the jam covers "
      f"{abs(wave) * dur:.1f} m")
    t(travelled.max() < L_RING,
      f"and nobody laps the ring inside it -- busiest car covers "
      f"{travelled.max():.1f} m of {L_RING:.0f} m")

    v_uni = uniform_speed()
    t(mean_v < v_uni,
      f"the jam costs the road real throughput -- spaced evenly these same "
      f"cars would all sit at {v_uni:.2f} m/s, jammed they average {mean_v:.2f}")

    # held out: three separately measured numbers that have to close a loop.
    # in the wave's own frame a car advances by exactly one car spacing per
    # period, so (mean speed + wave speed) * period must equal L / N.
    closes = (mean_v + abs(wave)) * tau1
    spacing = L_RING / N_CAR
    t(abs(closes - spacing) < 0.05,
      f"held out -- cars {mean_v:.3f} m/s and jam {abs(wave):.3f} m/s and "
      f"period {tau1:.3f} s were each measured on their own, and "
      f"(cars+jam)*period = {closes:.3f} m, which is the {spacing:.3f} m "
      f"spacing of {N_CAR} cars on {L_RING:.0f} m")

    # the reason this piece has no seamless loop, asserted rather than claimed
    t(err < 0.004 and abs(slide1) > 5.0,
      f"the pattern repeats every {tau1:.2f} s to within {err * 1000:.1f} mm -- "
      f"but {abs(slide1):.1f} m further round the ring, so it is a repeat of "
      f"the arrangement and not of the picture")
    lap = L_RING / abs(wave)
    t(lap > 3 * dur,
      f"the picture would only close once the wave had gone right round, "
      f"which takes {lap:.0f} s -- this piece is {dur:.1f} s")
    need = abs(wave) * dur
    jam_m = jam_length(xs, vs)
    t(need < jam_m,
      f"and shortening the road to make it close is not available either: it "
      f"would have to be {need:.0f} m, and the jam alone is {jam_m:.0f} m long")

    # ---- and now the picture itself
    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W * SS, H * SS)
    cr = cairo.Context(surf)
    cr.set_antialias(cairo.ANTIALIAS_BEST)
    x0, v0, a0 = sample(X, V, A, 0.0)
    f0 = finish(render_frame(surf, cr, x0 % L_RING, None,
                             (a0 < -0.10) | (v0 < 0.20)))

    lv, cnt = np.unique(f0[:, :, 1], return_counts=True)
    t(len(lv) > 120 and cnt.max() / f0[:, :, 1].size < 0.14,
      f"the frame is a gradient, not a few flat tones -- {len(lv)} levels, "
      f"commonest holds {100 * cnt.max() / f0[:, :, 1].size:.1f}%")
    t((f0 == 255).mean() < 0.002,
      f"highlights are not a clipped plateau -- "
      f"{100 * (f0 == 255).mean():.3f}% of pixels at 255")
    t(f0.min() > 4, f"and the shadows are not crushed -- darkest {f0.min()}")

    red = (f0[:, :, 0].astype(int) - f0[:, :, 1].astype(int)) > 40
    t(0.0004 < red.mean() < 0.02,
      f"the brake lights read as the only red in the frame -- "
      f"{100 * red.mean():.3f}% of pixels")

    print("\nALL CHECKS PASS" if ok else "\nSOMETHING FAILED")
    return 0 if ok else 1


def jam_length(xs, vs):
    """Metres of road occupied by the contiguous block of stopped cars."""
    best = 0.0
    for x, v in zip(xs, vs):
        p = x % L_RING
        o = np.argsort(p)
        p, st = p[o], v[o] < 0.20
        runs, cur, start = [], 0, None
        for k in range(2 * N_CAR):
            i = k % N_CAR
            if st[i]:
                if cur == 0:
                    start = i
                cur += 1
            elif cur:
                runs.append((cur, start, (i - 1) % N_CAR))
                cur = 0
        if runs:
            n, a, b = max(runs)
            best = max(best, (p[b] - p[a]) % L_RING + CAR_L)
    return best


def wave_backward(X, tau, n_frame):
    """Circular cross-correlation of the occupancy profile, frame to frame."""
    nb = 460
    prof = []
    for i in range(n_frame + 1):
        x = X[int(i * tau / n_frame / DT)] % L_RING
        p = np.zeros(nb)
        np.add.at(p, (x / L_RING * nb).astype(int) % nb, 1.0)
        prof.append(np.fft.rfft(p))
    shifts = []
    for i in range(0, n_frame - 15):
        c = np.fft.irfft(np.conj(prof[i]) * prof[i + 15], nb)
        j = int(np.argmax(c))
        shifts.append(j - nb if j > nb // 2 else j)
    return max(shifts) <= 0


def uniform_speed():
    """IDM equilibrium speed at even spacing -- what the road could do."""
    s = L_RING / N_CAR - CAR_L
    lo, hi = 0.0, V0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        se = (S_MIN + mid * T_HW) / math.sqrt(max(1e-12, 1.0 - (mid / V0) ** DELTA))
        if se < s:
            lo = mid
        else:
            hi = mid
    return lo


if __name__ == '__main__':
    sys.exit(main())
