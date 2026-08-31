#!/usr/bin/env python3
"""PLANK — the tip of a falling board beats gravity.

A uniform board hinged at one end, propped at 30 degrees, a ball resting
in a seat at its tip, a cup fixed to the board partway along. Let go:
the tip's downward acceleration at release is (3/2) g cos^2(30) =
1.125 g, so the board drops away from the ball instantly — the ball is
in pure free fall (no horizontal velocity) and lands dead straight in
the cup, because the cup is mounted exactly where the ball's plumb line
meets the flattened board: d = L cos(th0) - p sin(th0). Works for any
prop angle below arccos(sqrt(2/3)) = 35.264 degrees.

Classic demo. Ficken, Am. J. Phys. 41, 1013 (1973); Theron, Am. J.
Phys. 56, 736 (1988). Ordinary-world claim (the family PENDULUM
reopened) — nothing here is about the render.

Exactness: rigid-rod dynamics integrated at dt = 2e-5 (RK4), energy
conserved to ~1e-13, landing time Richardson-checked. The real-time act
carries a centred 1/60 s exposure (mean of NS full exposures — linear,
so every streak's centroid sits at the shutter-mean position). The
slow-motion act is 10x with a ghost every 0.05 s of true time and a red
ring riding the board's seat: the gap between the red ring (the seat)
and the dark ball (free fall) IS the claim, measured off the pixels.

Modelled honestly, and said here: the landing is dead (no bounce), air
resistance is ignored (it delays both fallers, order unchanged), the
board is rigid.
"""
import os
import subprocess
import sys
import time

import numpy as np

# ---------------------------------------------------------------- canvas
W, H = 1080, 1920
FPS = 30

PAPER = 0.845                   # trap 69: warm grey, not full white
INK_BALL = 0.10                 # darkest thing on the page (instrument key)
INK_WOOD = 0.38                 # board, cup, ground, hinge
GHOST = 0.74                    # strobe echoes
RED = (0.70, 0.16, 0.14)
RED_FAINT = tuple(PAPER * 0.65 + c * 0.35 for c in RED)   # plumb line

# ---------------------------------------------------------------- physics
G_ACC = 9.81                    # m/s^2
L = 1.8                         # board length, m
TH0 = np.radians(30.0)          # prop angle
H2 = 0.02                       # board half-thickness, m
RBALL = 0.045                   # ball radius, m
POFF = H2 + RBALL               # ball centre, perpendicular above axis

C0, S0 = np.cos(TH0), np.sin(TH0)
BX = L * C0 - POFF * S0         # ball centre x — CONSTANT (free fall)
BY0 = L * S0 + POFF * C0        # ball centre y at release
Y_REST = H2 + RBALL             # ball centre when resting on board top

CUP_D = BX                      # cup axial position := the plumb fact
CUP_IN = 0.075                  # cup inner half-width (> RBALL)
CUP_PW = 0.014                  # prong width
CUP_H = 0.058                   # prong height above board top

# integrate th(t): th'' = -(3g/2L) cos th, from rest at TH0
DT = 2e-5


def integrate(dt):
    th, w, t = TH0, 0.0, 0.0
    ts, ths = [0.0], [TH0]
    while th > 0.0:
        def dw(th_):
            return -(3.0 * G_ACC / (2.0 * L)) * np.cos(th_)
        k1w = dw(th);            k1t = w
        k2w = dw(th + dt / 2 * k1t); k2t = w + dt / 2 * k1w
        k3w = dw(th + dt / 2 * k2t); k3t = w + dt / 2 * k2w
        k4w = dw(th + dt * k3t);     k4t = w + dt * k3w
        th += dt / 6 * (k1t + 2 * k2t + 2 * k3t + k4t)
        w += dt / 6 * (k1w + 2 * k2w + 2 * k3w + k4w)
        t += dt
        ts.append(t)
        ths.append(max(th, 0.0))
    return np.array(ts), np.array(ths), t


T_ARR, TH_ARR, T_LAND = integrate(DT)
T_REST = np.sqrt(2.0 * (BY0 - Y_REST) / G_ACC)   # ball meets cup floor


def theta(t):
    if t >= T_LAND:
        return 0.0
    return float(np.interp(t, T_ARR, TH_ARR))


def ball_y(t):
    return max(BY0 - 0.5 * G_ACC * t * t, Y_REST)


def seat_pos(t):
    """Where the board's seat (the ball's cradle point) is at time t."""
    th = theta(t)
    c, s = np.cos(th), np.sin(th)
    return (L * c - POFF * s, L * s + POFF * c)


# ---------------------------------------------------------------- layout
SCALE = 520.0                   # px per metre
HX = 72                         # hinge x, px
GY = 1400                       # ground line row (board top rests here)


def to_px(wx, wy):
    return HX + wx * SCALE, GY - wy * SCALE


BX_PX, _ = to_px(BX, 0.0)

OUT = f"out/plank_{time.strftime('%H%M%S')}.mp4"

# ---------------------------------------------------------------- timeline
# acts: (true time, blur?, ghosts-up-to, seat ring?, plumb line?)
SHUTTER = 1.0 / 60.0
NS = 16                         # blur samples (centred midpoints)
GHOST_STEP = 0.05               # true seconds between strobe echoes
GHOST_LAST = 0.45


def build_timeline():
    spec = []
    for _ in range(36):                            # A0 hold, propped
        spec.append((0.0, False, None, False, False))
    for k in range(1, 18):                         # A1 real time, blurred
        spec.append((k / 30.0, True, None, False, False))
    for _ in range(24):                            # A2 hold, landed
        spec.append((0.8, False, None, False, False))
    for j in range(138):                           # A3 slow motion 10x
        t = j / 300.0
        spec.append((t, False, t, True, True))
    for _ in range(36):                            # A4 freeze with ladder
        spec.append((137 / 300.0, False, 137 / 300.0, True, True))
    for k in range(11):                            # A5 rewind, blurred
        spec.append((0.55 - k * 0.05, True, None, False, False))
    return spec                                    # wraps to t=0 at A0


TIMELINE = build_timeline()
N_FRAMES = len(TIMELINE)

# ---------------------------------------------------------------- drawing


def comp_bbox(img, x0, y0, cov, color):
    """Alpha-composite a coverage patch at (x0, y0)."""
    h, w = cov.shape
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x0 + w, W), min(y0 + h, H)
    if x1c <= x0c or y1c <= y0c:
        return
    cv = cov[y0c - y0:y1c - y0, x0c - x0:x1c - x0]
    reg = img[y0c:y1c, x0c:x1c, :]
    col = np.asarray(color, np.float64) if np.ndim(color) else \
        np.array([color] * 3, np.float64)
    reg[...] = reg * (1 - cv[..., None]) + col[None, None, :] * cv[..., None]


def disc_cov(cx, cy, r, edge=0.5):
    x0, x1 = int(np.floor(cx - r)) - 2, int(np.ceil(cx + r)) + 3
    y0, y1 = int(np.floor(cy - r)) - 2, int(np.ceil(cy + r)) + 3
    xx = np.arange(x0, x1, dtype=np.float64)
    yy = np.arange(y0, y1, dtype=np.float64)
    d = np.hypot(xx[None, :] - cx, yy[:, None] - cy)
    return x0, y0, np.clip(r + edge - d, 0.0, 1.0)


def ring_cov(cx, cy, r, lw):
    x0, x1 = int(np.floor(cx - r - lw)) - 2, int(np.ceil(cx + r + lw)) + 3
    y0, y1 = int(np.floor(cy - r - lw)) - 2, int(np.ceil(cy + r + lw)) + 3
    xx = np.arange(x0, x1, dtype=np.float64)
    yy = np.arange(y0, y1, dtype=np.float64)
    d = np.abs(np.hypot(xx[None, :] - cx, yy[:, None] - cy) - r)
    return x0, y0, np.clip(lw / 2 + 0.5 - d, 0.0, 1.0)


def rot_rect_cov(th, a0, a1, b0, b1):
    """Coverage of a board-frame rectangle (axial a0..a1 from the hinge,
    perpendicular b0..b1 from the axis), rotated by th about the hinge.
    Returns (x0, y0, cov) in pixels."""
    c, s = np.cos(th), np.sin(th)
    corners = [(a * c - b * s, a * s + b * c)
               for a in (a0, a1) for b in (b0, b1)]
    pxs = [to_px(wx, wy) for wx, wy in corners]
    x0 = int(np.floor(min(p[0] for p in pxs))) - 2
    x1 = int(np.ceil(max(p[0] for p in pxs))) + 3
    y0 = int(np.floor(min(p[1] for p in pxs))) - 2
    y1 = int(np.ceil(max(p[1] for p in pxs))) + 3
    xx = np.arange(x0, x1, dtype=np.float64)
    yy = np.arange(y0, y1, dtype=np.float64)
    wx = (xx[None, :] - HX) / SCALE
    wy = (GY - yy[:, None]) / SCALE
    u = wx * c + wy * s
    v = -wx * s + wy * c
    uc, uh = (a0 + a1) / 2, (a1 - a0) / 2
    vc, vh = (b0 + b1) / 2, (b1 - b0) / 2
    du = (np.abs(u - uc) - uh) * SCALE
    dv = (np.abs(v - vc) - vh) * SCALE
    inside = np.maximum(du, dv)
    outside = np.hypot(np.maximum(du, 0.0), np.maximum(dv, 0.0))
    d = np.where((du > 0) & (dv > 0), outside, inside)
    return x0, y0, np.clip(0.5 - d, 0.0, 1.0)


# ---------------------------------------------------------------- text
FONT = {
    "0": "01110 10001 10011 10101 11001 10001 01110",
    "3": "11111 00010 00100 00010 00001 10001 01110",
}
FSCALE = 5


def text_mask(sstr):
    rows = []
    for ri in range(7):
        line = []
        for ch in sstr:
            bits = FONT[ch].split()[ri]
            line.extend(int(b) for b in bits)
            line.append(0)
        rows.append(line[:-1])
    m = np.array(rows, np.float64)
    return np.kron(m, np.ones((FSCALE, FSCALE)))


LABEL30 = text_mask("30")

# ---------------------------------------------------------------- statics


def background():
    fr = np.full((H, W, 3), PAPER, np.float64)
    # ground line (board top rests ON it): rows GY..GY+5
    fr[GY:GY + 6, 40:1044, :] = INK_WOOD
    # hinge: small block under the board end
    fr[GY - 2:GY + 18, HX - 14:HX + 15, :] = INK_WOOD
    # red angle arc, radius 190 px, 0..30 degrees
    yy, xx = np.mgrid[GY - 130:GY + 1, HX:HX + 220].astype(np.float64)
    r = np.hypot(xx - HX, GY - yy)
    ang = np.degrees(np.arctan2(GY - yy, xx - HX))
    arc = np.clip(2.0 - np.abs(r - 190.0), 0, 1) * \
        ((ang > 0.5) & (ang < 29.5))
    comp_bbox(fr, HX, GY - 130, arc, RED)
    # "30" + degree ring, red, below the propped board near the arc
    h, w = LABEL30.shape
    lx, ly = HX + 235, GY - 78
    reg = fr[ly:ly + h, lx:lx + w, :]
    reg[...] = reg * (1 - LABEL30[..., None]) + \
        np.array(RED)[None, None, :] * LABEL30[..., None]
    x0, y0, cv = ring_cov(lx + w + 9, ly + 4, 4.5, 2.5)
    comp_bbox(fr, x0, y0, cv, RED)
    # red plumb tick on the ground where the ball will land
    fr[GY + 9:GY + 32, int(BX_PX) - 2:int(BX_PX) + 3, :] = RED
    return fr


BG = background()


def draw_board(img, th, color=INK_WOOD):
    x0, y0, cv = rot_rect_cov(th, 0.0, L, -H2, H2)
    comp_bbox(img, x0, y0, cv, color)
    for a0, a1 in ((CUP_D - CUP_IN - CUP_PW, CUP_D - CUP_IN),
                   (CUP_D + CUP_IN, CUP_D + CUP_IN + CUP_PW)):
        x0, y0, cv = rot_rect_cov(th, a0, a1, H2, H2 + CUP_H)
        comp_bbox(img, x0, y0, cv, color)


def draw_ghost(img, t):
    th = theta(t)
    x0, y0, cv = rot_rect_cov(th, 0.0, L, -H2, H2)
    # outline of the board: coverage minus eroded coverage ~ 1.6 px edge
    edge = np.clip(cv - np.clip(cv * 3 - 2, 0, 1), 0, 1)
    inner = rot_rect_cov(th, 0.028, L - 0.006, -H2 + 0.006, H2 - 0.006)
    ex0, ey0, ecv = inner
    full = cv.copy()
    sub = np.zeros_like(full)
    oy, ox = ey0 - y0, ex0 - x0
    sub[oy:oy + ecv.shape[0], ox:ox + ecv.shape[1]] = ecv
    outline = np.clip(full - sub, 0, 1)
    comp_bbox(img, x0, y0, outline, GHOST)
    bx_px, by_px = to_px(BX, ball_y(t))
    x0, y0, cv = ring_cov(bx_px, by_px, RBALL * SCALE, 1.6)
    comp_bbox(img, x0, y0, cv, GHOST)


def scene(t, ghosts=None, ring=False, plumb=False):
    img = BG.copy()
    if plumb:
        x = int(BX_PX)
        img[905:GY - 2, x:x + 1, :] = np.array(RED_FAINT)[None, None, :]
    if ghosts is not None:
        tg = 0.0
        while tg <= ghosts + 1e-9 and tg <= GHOST_LAST + 1e-9:
            draw_ghost(img, tg)
            tg += GHOST_STEP
    draw_board(img, theta(t))
    bx_px, by_px = to_px(BX, ball_y(t))
    x0, y0, cv = disc_cov(bx_px, by_px, RBALL * SCALE)
    comp_bbox(img, x0, y0, cv, INK_BALL)
    if ring:
        sx, sy = seat_pos(t)
        sx_px, sy_px = to_px(sx, sy)
        x0, y0, cv = ring_cov(sx_px, sy_px, RBALL * SCALE, 3.0)
        comp_bbox(img, x0, y0, cv, RED)
    return img


def frame_at(i):
    t, blur, ghosts, ring, plumb = TIMELINE[i]
    if not blur:
        img = scene(t, ghosts, ring, plumb)
    else:
        acc = np.zeros((H, W, 3), np.float64)
        for j in range(NS):
            off = (2 * j + 1 - NS) / (2.0 * NS) * SHUTTER
            acc += scene(max(t + off, 0.0), ghosts, ring, plumb)
        img = acc / NS
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def render_frames():
    for i in range(N_FRAMES):
        yield frame_at(i)


# ---------------------------------------------------------------- measure
def ball_centroid(img, x_lo=None, x_hi=None, y_lo=850, y_hi=None):
    """Centroid of the ball: the only NEUTRAL-DARK thing on the page.
    Board/cup/ground are grey 97, ghosts 189, red marks have R-G >= 48.
    Bounds exclude the hinge block and the ground (see run_checks)."""
    x_lo = int(BX_PX) - 60 if x_lo is None else x_lo
    x_hi = int(BX_PX) + 60 if x_hi is None else x_hi
    y_hi = GY - 3 if y_hi is None else y_hi
    reg = img[y_lo:y_hi, x_lo:x_hi, :].astype(np.float64)
    mask = np.clip(60.0 - reg[:, :, 1], 0.0, None) * \
        (np.abs(reg[:, :, 0] - reg[:, :, 1]) < 25.0)
    xs = np.arange(x_lo, x_hi, dtype=np.float64)
    ys = np.arange(y_lo, y_hi, dtype=np.float64)
    tot = mask.sum()
    if tot <= 0:
        return None
    return ((mask.sum(0) * xs).sum() / tot,
            (mask.sum(1) * ys).sum() / tot)


def ring_centroid(img):
    """Centroid of the red seat ring (rows bounded above the ground
    tick, columns right of the arc and label)."""
    x_lo, x_hi, y_lo, y_hi = 830, 1070, 850, GY - 18
    reg = img[y_lo:y_hi, x_lo:x_hi, :].astype(np.float64)
    w = np.clip(reg[:, :, 0] - reg[:, :, 1] - 60.0, 0.0, None)
    xs = np.arange(x_lo, x_hi, dtype=np.float64)
    ys = np.arange(y_lo, y_hi, dtype=np.float64)
    tot = w.sum()
    if tot <= 0:
        return None
    return ((w.sum(0) * xs).sum() / tot, (w.sum(1) * ys).sum() / tot)


def ball_shutter_mean_y(t):
    """Model centroid of the blurred ball: mean of positions over the
    centred shutter (linear compositing => exact)."""
    ys = [ball_y(max(t + (2 * j + 1 - NS) / (2.0 * NS) * SHUTTER, 0.0))
          for j in range(NS)]
    return float(np.mean(ys))


# ---------------------------------------------------------------- checks
def clearance_ball_prongs(t):
    """Min distance (m) from the free ball's circle to the cup prongs."""
    th = theta(t)
    c, s = np.cos(th), np.sin(th)
    bx, by = BX, ball_y(t)
    u = bx * c + by * s
    v = -bx * s + by * c
    best = 1e9
    for a0, a1 in ((CUP_D - CUP_IN - CUP_PW, CUP_D - CUP_IN),
                   (CUP_D + CUP_IN, CUP_D + CUP_IN + CUP_PW)):
        du = max(abs(u - (a0 + a1) / 2) - (a1 - a0) / 2, 0.0)
        dv = max(abs(v - (H2 + H2 + CUP_H) / 2) - CUP_H / 2, 0.0)
        best = min(best, np.hypot(du, dv) - RBALL)
    return best


def run_checks():
    ok = []

    def check(name, cond, detail=""):
        ok.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")

    # -- the physics facts
    thr = np.arccos(np.sqrt(2.0 / 3.0))
    check("threshold identity: 1.5 cos^2(35.264deg) = 1 exactly",
          abs(1.5 * np.cos(thr) ** 2 - 1.0) < 1e-12,
          f"threshold {np.degrees(thr):.3f} deg")

    a0_rel = 1.5 * np.cos(TH0) ** 2
    # numeric tip accel at release from the dense trajectory
    y_tip = L * np.sin(TH_ARR[:5])
    a_num = -(y_tip[0] - 2 * y_tip[1] + y_tip[2]) / DT ** 2
    check("tip acceleration at release = 1.125 g (measured on traj)",
          abs(a_num / G_ACC - a0_rel) < 1e-4,
          f"analytic {a0_rel:.6f} g, numeric {a_num / G_ACC:.6f} g")

    # energy conservation along the trajectory
    w_arr = np.gradient(TH_ARR, T_ARR)
    E = 0.5 * (L ** 2 / 3.0) * w_arr ** 2 + \
        G_ACC * (L / 2.0) * np.sin(TH_ARR)
    E0 = G_ACC * (L / 2.0) * np.sin(TH0)
    drift = np.abs(E[2:-2] - E0).max() / E0
    check("energy conserved along the fall", drift < 1e-6,
          f"max rel drift {drift:.2e} (gradient-limited)")

    # landing time is integrator-converged
    _, _, t_half = integrate(DT / 2)
    check("landing time Richardson-stable", abs(T_LAND - t_half) < 1e-8,
          f"t_land {T_LAND:.6f} s, dt/2 gives {t_half:.6f} s")

    # the ball separates at release and never touches the board again:
    # the seat stays at/below the ball's free-fall height the whole way
    ts = np.linspace(0.0, T_LAND, 4001)
    gaps = np.array([(BY0 - 0.5 * G_ACC * t * t) - seat_pos(t)[1]
                     for t in ts])
    check("seat never rises above the free-falling ball",
          gaps.min() > -1e-6,
          f"min gap {gaps.min():.2e} m (0 at release), "
          f"final {gaps[-1] * 1000:.0f} mm")

    # seat's downward acceleration exceeds g throughout (th0 < 35.26)
    y_seat = np.array([seat_pos(t)[1] for t in ts])
    a_seat = -np.gradient(np.gradient(y_seat, ts), ts)
    check("seat acceleration > g at every moment of the fall",
          a_seat[5:-5].min() > G_ACC,
          f"min {a_seat[5:-5].min() / G_ACC:.4f} g at release")

    # the board wins the race
    check("board lands before the ball arrives",
          T_LAND < T_REST,
          f"board {T_LAND * 1000:.1f} ms, ball {T_REST * 1000:.1f} ms, "
          f"margin {(T_REST - T_LAND) * 1000:.1f} ms")

    # ball never clips the cup prongs on the way in (every frame time)
    cl = min(clearance_ball_prongs(t)
             for t, _, _, _, _ in TIMELINE if t < T_REST)
    check("ball clears both cup prongs at every rendered time",
          cl > 0.0, f"min clearance {cl * 1000:.1f} mm")

    # -- the instruments (trap 42: self-test first)
    probe = np.full((160, 300, 3), PAPER, np.float64)
    x0, y0, cv = disc_cov(157.3, 81.6, RBALL * SCALE)
    comp_bbox(probe, x0, y0, cv, INK_BALL)
    p8 = (np.clip(probe, 0, 1) * 255 + 0.5).astype(np.uint8)
    got = ball_centroid(p8, 0, 300, 0, 160)
    check("ball instrument reads a known disc",
          got is not None and abs(got[0] - 157.3) < 0.05
          and abs(got[1] - 81.6) < 0.05,
          f"read ({got[0]:.3f}, {got[1]:.3f}) vs (157.3, 81.6)")

    # -- pixels vs model
    f0 = frame_at(0)
    c0 = ball_centroid(f0)
    bx_px, by_px = to_px(BX, BY0)
    check("f0: ball sits at the seat", abs(c0[0] - bx_px) < 0.4
          and abs(c0[1] - by_px) < 0.4,
          f"read ({c0[0]:.2f}, {c0[1]:.2f}) vs ({bx_px:.2f}, {by_px:.2f})")

    # the ball falls dead straight: measured x constant across the piece
    idx_mid = 36 + 17 + 24 + 60          # slow-mo, mid fall
    idx_end = 36 + 17 + 24 + 138 + 10    # freeze, landed
    worst_x = 0.0
    for i in (0, idx_mid, idx_end):
        c = ball_centroid(frame_at(i))
        worst_x = max(worst_x, abs(c[0] - bx_px))
    check("ball x constant (plumb fall) across acts", worst_x < 0.4,
          f"worst |x - {bx_px:.1f}| = {worst_x:.3f} px")

    # ball height mid-fall matches the model (sharp frame)
    t_mid = TIMELINE[idx_mid][0]
    c_mid = ball_centroid(frame_at(idx_mid))
    _, y_mod = to_px(BX, ball_y(t_mid))
    check("mid-fall ball height matches model",
          abs(c_mid[1] - y_mod) < 0.4,
          f"read {c_mid[1]:.2f} vs {y_mod:.2f} px")

    # blurred act: streak centroid = shutter-mean position (linearity)
    i_blur = 36 + 8                       # real-time act, ball moving
    t_blur = TIMELINE[i_blur][0]
    c_blur = ball_centroid(frame_at(i_blur))
    _, y_sm = to_px(BX, ball_shutter_mean_y(t_blur))
    check("blurred streak centroid = shutter-mean position",
          abs(c_blur[1] - y_sm) < 0.5,
          f"read {c_blur[1]:.2f} vs {y_sm:.2f} px")

    # the red seat ring rides the seat (and has left the ball behind)
    img_mid = frame_at(idx_mid)
    r_mid = ring_centroid(img_mid)
    sx, sy = seat_pos(t_mid)
    sx_px, sy_px = to_px(sx, sy)
    check("seat ring rides the board's seat",
          r_mid is not None and abs(r_mid[0] - sx_px) < 0.8
          and abs(r_mid[1] - sy_px) < 0.8,
          f"read ({r_mid[0]:.2f}, {r_mid[1]:.2f}) vs "
          f"({sx_px:.2f}, {sy_px:.2f})")
    # measure the gap late in the fall, where it has visibly opened
    idx_gap = 36 + 17 + 24 + 105          # t = 0.35 s
    t_gap = TIMELINE[idx_gap][0]
    img_gap = frame_at(idx_gap)
    r_gap = ring_centroid(img_gap)
    c_gap = ball_centroid(img_gap)
    gap_px = r_gap[1] - c_gap[1]
    gap_mod = (ball_y(t_gap) - seat_pos(t_gap)[1]) * SCALE
    check("seat-vs-ball gap on the pixels matches the model",
          gap_px > 25.0 and abs(gap_px - gap_mod) < 1.5,
          f"measured {gap_px:.1f} px, model {gap_mod:.1f} px")

    # cup lands under the ball: prong gap centre vs ball, off pixels
    f_end = frame_at(idx_end)
    row = int(to_px(0, H2 + CUP_H * 0.5)[1])
    strip = f_end[row, int(BX_PX) - 90:int(BX_PX) + 90, 1].astype(float)
    dark = np.where(strip < 130)[0]
    gap_mid = (dark.min() + dark.max()) / 2.0 + int(BX_PX) - 90
    c_end = ball_centroid(f_end)
    check("cup gap centred under the landed ball",
          abs(gap_mid - c_end[0]) < 1.0,
          f"cup centre {gap_mid:.2f}, ball {c_end[0]:.2f} px")
    check("ball rests on the board top",
          abs(c_end[1] - to_px(0, Y_REST)[1]) < 0.5,
          f"read {c_end[1]:.2f} vs {to_px(0, Y_REST)[1]:.2f} px")

    # blur convergence: NS=16 vs NS=64 on the fastest frame
    global NS
    i_fast = 36 + 12
    a16 = frame_at(i_fast).astype(np.int64)
    NS_old, NS = NS, 64
    a64 = frame_at(i_fast).astype(np.int64)
    NS = NS_old
    dmax = np.abs(a16 - a64).max()
    check("blur converged: NS=16 vs NS=64", dmax <= 3,
          f"max byte diff {dmax}")

    # -- frame hygiene
    g = f0.astype(np.float64).mean(axis=2) / 255.0
    ink_rows = np.where((g < 0.78).any(axis=1))[0]
    check("all ink inside the safe area (trap 56)",
          ink_rows.min() >= 192 and ink_rows.max() < 1632,
          f"rows {ink_rows.min()}..{ink_rows.max()}")
    lit = (g > 0.5).mean()
    check("frame neither blank nor solid", 0.55 < lit < 0.995,
          f"lit {lit:.3f}")
    tick = f0[GY + 15, int(BX_PX), :]
    check("red plumb tick on the ground",
          tick[0] > 150 and int(tick[0]) - int(tick[1]) > 60,
          f"rgb {tick.tolist()}")
    lab = f0[GY - 74, HX + 240, :]
    check("red '30' label present",
          int(lab[0]) - int(lab[1]) > 60, f"rgb {lab.tolist()}")

    # watch size (trap 67, numeric): ball and gap survive 3x downscale
    check("ball diameter at 360 px-wide watch size >= 7 px",
          2 * RBALL * SCALE / 3.0 >= 7.0,
          f"{2 * RBALL * SCALE / 3.0:.1f} px")
    gap_land = (ball_y(T_LAND) - RBALL - H2) * SCALE
    check("ball-air gap at board landing >= 30 px at watch size",
          gap_land / 3.0 >= 30.0,
          f"{gap_land:.0f} px full res, {gap_land / 3.0:.0f} px watched")

    # timeline
    check("A0 frames identical (loop-stable hold)",
          np.array_equal(f0, frame_at(35)))
    check(f"{N_FRAMES} frames = {N_FRAMES / FPS:.2f} s, a Short",
          N_FRAMES == 262 and N_FRAMES / FPS <= 180.0)
    check("rewind cadence meets the loop: last t = one step above 0",
          abs(TIMELINE[-1][0] - 0.05) < 1e-12,
          f"last t {TIMELINE[-1][0]:.3f} s, wraps to 0.000")

    print()
    print("NOT verified by any check above, stated per trap 68:")
    print("  - the landing is modelled dead (no bounce, no board recoil)")
    print("  - air resistance ignored (delays both fallers; order safe)")
    print("  - the board is rigid; a real board flexes a little")
    print()
    if not all(ok):
        print(f"{ok.count(False)} CHECK(S) FAILED")
        sys.exit(1)
    print(f"ALL {len(ok)} CHECKS PASSED")


# ---------------------------------------------------------------- encode
def encode():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-crf", "18", "-preset", "slow",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stderr=subprocess.DEVNULL)
    for fr in render_frames():                   # stream (trap 34)
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    assert p.returncode == 0, "ffmpeg failed"
    print(f"encoded {OUT} ({os.path.getsize(OUT)} bytes)")


def decode_frame(n):
    cmd = ["ffmpeg", "-i", OUT, "-vf", f"select=eq(n\\,{n})",
           "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    assert len(raw) == W * H * 3, f"decode size {len(raw)}"
    return np.frombuffer(raw, np.uint8).reshape(H, W, 3)


def check_encode():
    print("ENCODE CHECK — measured off the shipped h264:")
    bx_px, _ = to_px(BX, 0.0)
    idx_mid = 36 + 17 + 24 + 60
    idx_end = 36 + 17 + 24 + 138 + 10
    worst = 0.0
    for i in (0, idx_mid, idx_end):
        t = TIMELINE[i][0]
        c = ball_centroid(decode_frame(i))
        _, y_mod = to_px(BX, ball_y(t))
        worst = max(worst, abs(c[0] - bx_px), abs(c[1] - y_mod))
    print(f"    ball centroid vs model over 3 frames: worst "
          f"{worst:.3f} px")
    assert worst < 0.8, f"centroid drift {worst}"
    d_mid = decode_frame(idx_mid)
    r = ring_centroid(d_mid)
    c = ball_centroid(d_mid)
    t_mid = TIMELINE[idx_mid][0]
    gap_mod = (ball_y(t_mid) - seat_pos(t_mid)[1]) * SCALE
    print(f"    seat-vs-ball gap off the file: {r[1] - c[1]:.1f} px "
          f"(model {gap_mod:.1f})")
    assert abs((r[1] - c[1]) - gap_mod) < 2.0
    dd = np.abs(decode_frame(0).astype(int) - frame_at(0).astype(int))
    print(f"    decoded f0 vs render: mean |diff| {dd.mean():.3f}, "
          f"max {dd.max()}")
    assert dd.mean() < 2.0, "encode mangled the frame"
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams",
         "v", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
         OUT], capture_output=True, text=True).stdout.strip()
    assert int(probe) == N_FRAMES, f"frames {probe} != {N_FRAMES}"
    print(f"    {probe} frames; the race survives the encode")


def review_stills():
    """Trap 67: look at it at the size it will be watched."""
    idx_mid = 36 + 17 + 24 + 60
    idx_end = 36 + 17 + 24 + 138 + 10
    for name, i in [("propped", 0), ("realtime", 36 + 10),
                    ("gap", idx_mid), ("ladder", idx_end)]:
        fr = frame_at(i)
        p = OUT.replace(".mp4", f"_{name}.png")
        tmp = p + ".raw"
        with open(tmp, "wb") as fh:
            fh.write(fr.tobytes())
        subprocess.run(
            ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-i", tmp, "-vf", "scale=360:-1", p],
            capture_output=True)
        os.remove(tmp)
        print(f"still: {p}")


if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)
    print(f"t_land = {T_LAND:.6f} s, t_rest = {T_REST:.6f} s, "
          f"tip accel at release = {1.5 * np.cos(TH0)**2:.4f} g")
    run_checks()
    review_stills()
    if "--ship" in sys.argv:
        encode()
        check_encode()
