#!/usr/bin/env python3
"""TAUTO — the tautochrone. Two bowls, same width, same depth, same four
release heights, released at the same instant.

Top bowl: a circular arc. The four balls' periods differ (elliptic
integral), so they drift apart and never line up again.
Bottom bowl: a cycloid. Arc-length motion is EXACT simple harmonic motion
(s'' = -(g/4R) s, an algebraic identity of the curve), so every ball
reaches the bottom at the same instant, and keeps doing so forever
(Huygens, Horologium Oscillatorium, 1673).

Design (from the physics, not tuned by eye):
  - cycloid period T = 3.2 s exactly (96 frames), so R = g/(4 w^2)
  - circle radius rho = pi*R/sin(alpha) with alpha = 2*atan(2/pi): the
    unique circular arc with the cycloid's width AND depth (fair control)
  - release fractions of depth: 0.93, 0.60, 0.32, 0.12; same hue = same
    height in both bowls
  - run = release hold 0.8 s + 4 cycloid periods (12.8 s) + 0.8 s hold.
    At the end the cycloid balls are back ON their release marks; the
    circle balls are scattered (spread measured at 488 px).
  - a lamp under each bowl lights only when ALL FOUR balls are within
    8 px of the bottom at one instant. The cycloid lamp fires 8 times;
    the circle lamp never fires (its best instant still has a ball 18 px
    out — measured, margin 2.2x).

The render integrates the full EOM in the curve parameter by RK4; the
closed form s(t) = s0 cos(wt) is the cross-check. For the cycloid the
deviation must be INTEGRATOR-LEVEL (~1e-11 px), because exactness is the
claim (trap 62). The circle balls are checked against the elliptic
integral sqrt(rho/g)*K(sin(th0/2)) — the control that must come out the
other way (trap 59).

What is NOT verified is printed at the end (trap 68).
"""
import os
import subprocess
import sys
import time

import numpy as np

# ---------------------------------------------------------------- canvas
W, H = 1080, 1920
FPS = 30
PAPER = 0.845                   # trap 69
INK = 0.10
INK_WIRE = 0.30
GHOST = 0.58
C_BALLS = [(0.55, 0.10, 0.10),  # red     f=0.93 (deepest)
           (0.12, 0.16, 0.52),  # blue    f=0.60
           (0.10, 0.40, 0.12),  # green   f=0.32
           (0.52, 0.10, 0.48)]  # magenta f=0.12

# ---------------------------------------------------------------- physics
G_ACC = 9.81
T_CYC = 3.2                     # cycloid full period, s = 96 frames
OMEGA = 2 * np.pi / T_CYC
R = G_ACC / (4 * OMEGA**2)      # 0.63613 m
ALPHA = 2 * np.arctan(2 / np.pi)
RHO = np.pi * R / np.sin(ALPHA)  # 2.20573 m
WIDTH = 2 * np.pi * R
DEPTH = 2 * R
FRACS = [0.93, 0.60, 0.32, 0.12]
N_B = len(FRACS)

PRE = 24                        # frames held at release
T_RUN = 4 * T_CYC               # 12.8 s of motion
POST = 24                       # frames held at the end
N_FRAMES = PRE + int(T_RUN * FPS) + POST      # 432 = 14.4 s
T_REL = PRE / FPS               # 0.8 s

DT = 1e-3

PHI0 = [2 * np.arccos(np.sqrt(f)) for f in FRACS]          # cycloid starts
TH0 = [-np.arccos(1 - f * DEPTH / RHO) for f in FRACS]     # circle starts


def acc_cyc(phi, phid):
    return np.cos(phi / 2) * (G_ACC - R * phid**2) / (2 * R * np.sin(phi / 2))


def acc_circ(th, thd):
    return -(G_ACC / RHO) * np.sin(th)


def rk4_table(accf, q0, t_end, dt):
    n = int(round(t_end / dt))
    q, qd = q0, 0.0
    traj = np.empty((n + 1, 2))
    traj[0] = (q, qd)
    for i in range(n):
        k1v, k1a = qd, accf(q, qd)
        k2v, k2a = qd + 0.5 * dt * k1a, accf(q + 0.5 * dt * k1v,
                                             qd + 0.5 * dt * k1a)
        k3v, k3a = qd + 0.5 * dt * k2a, accf(q + 0.5 * dt * k2v,
                                             qd + 0.5 * dt * k2a)
        k4v, k4a = qd + dt * k3a, accf(q + dt * k3v, qd + dt * k3a)
        q = q + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        qd = qd + dt / 6 * (k1a + 2 * k2a + 2 * k3a + k4a)
        traj[i + 1] = (q, qd)
    return traj


TT = np.arange(int(round(T_RUN / DT)) + 1) * DT
TR_CYC = [rk4_table(acc_cyc, p0, T_RUN, DT) for p0 in PHI0]
TR_CIR = [rk4_table(acc_circ, t0, T_RUN, DT) for t0 in TH0]


def q_at(traj, t):
    """Curve parameter at wall-clock t (held before release and after)."""
    tp = min(max(t - T_REL, 0.0), T_RUN)
    return float(np.interp(tp, TT, traj[:, 0]))


# ---------------------------------------------------------------- layout
SCALE = 980.0 / WIDTH           # 245.2 px/m
X0 = (W - 980.0) / 2            # 50: left cusp of both bowls
XC = W / 2.0                    # 540: both bottoms
Y_RIM_A = 560.0                 # circle bowl rim
Y_RIM_B = 1120.0                # cycloid bowl rim
DEPTH_PX = DEPTH * SCALE        # 311.9
Y_BOT_A = Y_RIM_A + DEPTH_PX
Y_BOT_B = Y_RIM_B + DEPTH_PX
CY_A = Y_BOT_A - RHO * SCALE    # circle centre (above the rim)
R_BALL = 20.0
LAMP_A = (XC, Y_BOT_A + 52.0)
LAMP_B = (XC, Y_BOT_B + 52.0)
LAMP_R = 17.0

SH_RT = 1.0 / 60.0
NS = 24                         # streak up to 19.7 px -> 0.82 px/sample

OUT = f"out/tauto_{time.strftime('%H%M%S')}.mp4"

FLASH_TOL = 8.0                 # px; circle's best instant is 18 px out
FLASH_SIG = 0.12                # s, lamp glow decay


def cyc_px(phi):
    return (X0 + R * (phi - np.sin(phi)) * SCALE,
            Y_RIM_B + R * (1 - np.cos(phi)) * SCALE)


def cir_px(th):
    return (XC + RHO * np.sin(th) * SCALE, CY_A + RHO * np.cos(th) * SCALE)


BOWLS = [("circle", TR_CIR, cir_px, TH0),
         ("cycloid", TR_CYC, cyc_px, PHI0)]


def ball_px(bowl, i, t):
    _, trs, pxf, _ = BOWLS[bowl]
    return pxf(q_at(trs[i], t))


# ---------------------------------------------------------------- flash
def flash_times(bowl):
    """Instants when ALL balls are within FLASH_TOL px of the bottom,
    measured on the trajectory table (interval midpoints)."""
    _, trs, pxf, _ = BOWLS[bowl]
    m = np.max(np.abs(np.stack(
        [pxf(tr[:, 0])[0] for tr in trs]) - XC), axis=0)
    inside = m < FLASH_TOL
    times = []
    i = 0
    while i < len(inside):
        if inside[i]:
            j = i
            while j + 1 < len(inside) and inside[j + 1]:
                j += 1
            times.append(T_REL + 0.5 * (TT[i] + TT[j]))
            i = j + 1
        else:
            i += 1
    return times


FLASH = [flash_times(0), flash_times(1)]


def lamp_glow(bowl, t):
    ts = FLASH[bowl]
    if not ts:
        return 0.0
    d = min(abs(t - tk) for tk in ts)
    return float(np.exp(-(d / FLASH_SIG)**2))


# ---------------------------------------------------------------- drawing
def comp_bbox(img, x0, y0, cov, color):
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


def polyseg_cov(pts, lw):
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    pad = lw / 2 + 2
    x0 = int(np.floor(xs.min() - pad))
    y0 = int(np.floor(ys.min() - pad))
    x1 = int(np.ceil(xs.max() + pad)) + 1
    y1 = int(np.ceil(ys.max() + pad)) + 1
    cov = np.zeros((y1 - y0, x1 - x0), np.float64)
    for i in range(len(pts) - 1):
        ax, ay = xs[i], ys[i]
        bx, by = xs[i + 1], ys[i + 1]
        sx0 = int(np.floor(min(ax, bx) - pad)) - x0
        sy0 = int(np.floor(min(ay, by) - pad)) - y0
        sx1 = int(np.ceil(max(ax, bx) + pad)) + 1 - x0
        sy1 = int(np.ceil(max(ay, by) + pad)) + 1 - y0
        gx = np.arange(sx0, sx1, dtype=np.float64) + x0
        gy = np.arange(sy0, sy1, dtype=np.float64) + y0
        px = gx[None, :] - ax
        py = gy[:, None] - ay
        ux, uy = bx - ax, by - ay
        ll = ux * ux + uy * uy
        tpar = np.clip((px * ux + py * uy) / max(ll, 1e-12), 0.0, 1.0)
        d = np.hypot(px - tpar * ux, py - tpar * uy)
        st = np.clip(lw / 2 + 0.5 - d, 0.0, 1.0)
        reg = cov[sy0:sy1, sx0:sx1]
        np.maximum(reg, st, out=reg)
    return x0, y0, cov


# ---------------------------------------------------------------- text
FONT = {
    "c": "00000 00000 01110 10000 10000 10001 01110",
    "i": "00100 00000 01100 00100 00100 00100 01110",
    "r": "00000 00000 10110 11001 10000 10000 10000",
    "l": "01100 00100 00100 00100 00100 00100 01110",
    "e": "00000 00000 01110 10001 11111 10000 01110",
    "y": "00000 00000 10001 10001 01111 00001 01110",
    "o": "00000 00000 01110 10001 10001 10001 01110",
    "d": "00001 00001 01101 10011 10001 10011 01101",
}
FSCALE = 6


def text_mask(sstr):
    rows = []
    for ri in range(7):
        line = []
        for ch in sstr:
            line.extend(int(b) for b in FONT[ch].split()[ri])
            line.append(0)
        rows.append(line[:-1])
    return np.kron(np.array(rows, np.float64),
                   np.ones((FSCALE, FSCALE)))


def stamp_center(img, sstr, cx, ytop, color):
    m = text_mask(sstr)
    h, w = m.shape
    x0 = int(round(cx - w / 2))
    col = np.asarray(color, np.float64) if np.ndim(color) else \
        np.array([color] * 3, np.float64)
    reg = img[ytop:ytop + h, x0:x0 + w, :]
    reg[...] = reg * (1 - m[..., None]) + col[None, None, :] * m[..., None]


LBL_A_Y, LBL_B_Y = 480, 1040    # label tops (7*6=42 px tall)


def curve_pts(pxf, q_lo, q_hi):
    qs = np.linspace(q_lo, q_hi, 3000)
    xs, ys = pxf(qs)
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    u = np.arange(0, s[-1], 2.0)
    return list(zip(np.interp(u, s, xs), np.interp(u, s, ys)))


def background():
    fr = np.full((H, W, 3), PAPER, np.float64)
    # bowls
    x0, y0, cv = polyseg_cov(curve_pts(cir_px, -ALPHA, ALPHA), 3.5)
    comp_bbox(fr, x0, y0, cv, INK_WIRE)
    x0, y0, cv = polyseg_cov(curve_pts(cyc_px, 1e-4, 2 * np.pi - 1e-4), 3.5)
    comp_bbox(fr, x0, y0, cv, INK_WIRE)
    # release ticks (ghost): short normal dashes at each release point
    for bowl in (0, 1):
        _, trs, pxf, q0s = BOWLS[bowl]
        for q0 in q0s:
            x, y = pxf(q0)
            dq = 1e-4
            tx, ty = np.subtract(pxf(q0 + dq), pxf(q0 - dq))
            nrm = np.hypot(tx, ty)
            nx, ny = -ty / nrm, tx / nrm
            x0, y0, cv = polyseg_cov(
                [(x - 14 * nx, y - 14 * ny), (x + 14 * nx, y + 14 * ny)], 3.0)
            comp_bbox(fr, x0, y0, cv, GHOST)
    # lamps (outline only; the glow is per-frame)
    for lx, ly in (LAMP_A, LAMP_B):
        x0, y0, cv = ring_cov(lx, ly, LAMP_R, 3.0)
        comp_bbox(fr, x0, y0, cv, INK)
    stamp_center(fr, "circle", XC, LBL_A_Y, INK)
    stamp_center(fr, "cycloid", XC, LBL_B_Y, INK)
    return fr


BG = background()


def scene(t):
    img = BG.copy()
    for bowl in (0, 1):
        for i in (3, 2, 1, 0):          # deepest (red) drawn last, on top
            x, y = ball_px(bowl, i, t)
            x0, y0, cv = disc_cov(x, y, R_BALL)
            comp_bbox(img, x0, y0, cv, C_BALLS[i])
    return img


def lamps(img, t):
    for bowl, (lx, ly) in ((0, LAMP_A), (1, LAMP_B)):
        g = lamp_glow(bowl, t)
        if g > 0.01:
            x0, y0, cv = disc_cov(lx, ly, LAMP_R - 4.0)
            comp_bbox(img, x0, y0, cv * g, INK)


def frame_at(i, ns=NS):
    t = i / FPS
    acc = np.zeros((H, W, 3), np.float64)
    for j in range(ns):
        off = (2 * j + 1 - ns) / (2.0 * ns) * SH_RT
        acc += scene(t + off)
    img = acc / ns
    lamps(img, t)                       # instrument overlay, sharp
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def render_frames():
    for i in range(N_FRAMES):
        yield frame_at(i)


# ---------------------------------------------------------------- measure
def mask_centroid(img, mask_fn, x_lo, x_hi, y_lo, y_hi):
    x_lo, y_lo = max(x_lo, 0), max(y_lo, 0)
    x_hi, y_hi = min(x_hi, W), min(y_hi, H)
    reg = img[y_lo:y_hi, x_lo:x_hi, :].astype(np.float64)
    wgt = mask_fn(reg)
    xs = np.arange(x_lo, x_hi, dtype=np.float64)
    ys = np.arange(y_lo, y_hi, dtype=np.float64)
    tot = wgt.sum()
    if tot <= 0:
        return None
    return ((wgt.sum(0) * xs).sum() / tot, (wgt.sum(1) * ys).sum() / tot)


def m_red(reg):
    return np.clip(reg[:, :, 0] - reg[:, :, 1] - 60, 0, None) * \
        (reg[:, :, 2] - reg[:, :, 1] < 40)


def m_blue(reg):
    return np.clip(reg[:, :, 2] - reg[:, :, 1] - 60, 0, None) * \
        (reg[:, :, 0] - reg[:, :, 1] < 20)


def m_grn(reg):
    return np.minimum(np.clip(reg[:, :, 1] - reg[:, :, 0] - 40, 0, None),
                      np.clip(reg[:, :, 1] - reg[:, :, 2] - 40, 0, None))


def m_mag(reg):
    return np.minimum(np.clip(reg[:, :, 0] - reg[:, :, 1] - 60, 0, None),
                      np.clip(reg[:, :, 2] - reg[:, :, 1] - 40, 0, None))


M_FNS = [m_red, m_blue, m_grn, m_mag]
ROWBAND = [(int(Y_RIM_A) - 30, int(Y_BOT_A) + 30),
           (int(Y_RIM_B) - 30, int(Y_BOT_B) + 30)]


def ball_centroid(img, bowl, i, t, box=60):
    """Centroid by hue, bounded to a box round the MODEL position and to
    the bowl's own row band (traps 58/64)."""
    mx, my = ball_px(bowl, i, t)
    y_lo, y_hi = ROWBAND[bowl]
    return mask_centroid(img, M_FNS[i], int(mx) - box, int(mx) + box,
                         max(int(my) - box, y_lo), min(int(my) + box, y_hi))


def shutter_mean_px(bowl, i, t):
    xs, ys = [], []
    for k in range(NS):
        off = (2 * k + 1 - NS) / (2.0 * NS) * SH_RT
        x, y = ball_px(bowl, i, t + off)
        xs.append(x)
        ys.append(y)
    return float(np.mean(xs)), float(np.mean(ys))


def sat_cols(img, bowl):
    """Columns holding saturated (ball-coloured) pixels in a bowl's band.
    Paper, wire ink, ghost ticks, lamp and labels are all grey ->
    excluded by the saturation test itself; the row band excludes the
    other bowl (trap 58)."""
    y_lo, y_hi = ROWBAND[bowl]
    reg = img[y_lo:y_hi, :, :].astype(np.float64)
    sat = reg.max(2) - reg.min(2) > 40
    return np.where(sat.any(0))[0]


def lamp_mean(img, which):
    lx, ly = (LAMP_A, LAMP_B)[which]
    x0, y0, cv = disc_cov(lx, ly, LAMP_R - 6.0)
    reg = img[y0:y0 + cv.shape[0], x0:x0 + cv.shape[1], :].astype(np.float64)
    return float((reg.mean(2) * cv).sum() / cv.sum())


# ---------------------------------------------------------------- checks
def ellipK(k):
    a, b = 1.0, np.sqrt(1 - k * k)
    for _ in range(60):
        a, b = (a + b) / 2, np.sqrt(a * b)
    return np.pi / (2 * a)


def run_checks():
    ok = 0

    def ck(name, cond, detail=""):
        nonlocal ok
        assert cond, f"CHECK FAILED: {name} {detail}"
        ok += 1
        print(f"  ok {ok:2d}  {name}{('  ' + detail) if detail else ''}")

    print("CHECKS")
    # -- construction
    ck("circle bowl has the cycloid's width and depth exactly",
       abs(RHO * (1 - np.cos(ALPHA)) - DEPTH) < 1e-12 and
       abs(2 * RHO * np.sin(ALPHA) - WIDTH) < 1e-12,
       f"rho={RHO:.5f} m, alpha={np.degrees(ALPHA):.2f} deg")
    ck("cycloid period is exactly 96 frames",
       abs(2 * np.pi / OMEGA * FPS - 96) < 1e-9, f"T={T_CYC} s")

    # -- the algebraic heart: cycloid EOM implies s'' = -(g/4R) s
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(200):
        phi = rng.uniform(0.4, 2 * np.pi - 0.4)
        phid = rng.uniform(-3, 3)
        sdd = R * np.cos(phi / 2) * phid**2 + \
            2 * R * np.sin(phi / 2) * acc_cyc(phi, phid)
        worst = max(worst, abs(sdd + (G_ACC / (4 * R)) *
                               (-4 * R * np.cos(phi / 2))))
    ck("cycloid EOM implies EXACT SHM in arc length (identity)",
       worst < 1e-10, f"worst residual {worst:.1e} over 200 random states")

    # -- integrator health: energy per ball
    for bowl, name in ((0, "circle"), (1, "cycloid")):
        _, trs, pxf, _ = BOWLS[bowl]
        wde = 0.0
        for tr in trs:
            if bowl == 0:
                h = RHO * (1 - np.cos(tr[:, 0]))
                v2 = (RHO * tr[:, 1])**2
            else:
                h = 2 * R * np.cos(tr[:, 0] / 2)**2
                v2 = (2 * R * np.sin(tr[:, 0] / 2) * tr[:, 1])**2
            e = 0.5 * v2 + G_ACC * h
            wde = max(wde, np.max(np.abs(e - e[0])) / e[0])
        ck(f"{name}: energy conserved for all four balls", wde < 1e-9,
           f"worst dE/E {wde:.1e}")

    # -- the claim (trap 62: exactness IS the formula)
    wdev = 0.0
    for i, f in enumerate(FRACS):
        s_sim = -4 * R * np.cos(TR_CYC[i][:, 0] / 2)
        s0 = -4 * R * np.cos(PHI0[i] / 2)
        wdev = max(wdev, np.max(np.abs(s_sim - s0 * np.cos(OMEGA * TT))))
    ck("cycloid sim == s0*cos(wt) at INTEGRATOR level (all four)",
       wdev * SCALE < 1e-6, f"worst {wdev * SCALE:.1e} px")

    # -- times to bottom
    tb_cyc = []
    for i in range(N_B):
        s_sim = -4 * R * np.cos(TR_CYC[i][:, 0] / 2)
        j = int(np.argmax(np.diff(np.sign(s_sim)) != 0))
        tb_cyc.append(TT[j] + DT * s_sim[j] / (s_sim[j] - s_sim[j + 1]))
    spread_c = max(tb_cyc) - min(tb_cyc)
    ck("cycloid: all four reach bottom at t = pi*sqrt(R/g), together",
       spread_c < 2 * DT and
       all(abs(tb - np.pi * np.sqrt(R / G_ACC)) < 2 * DT for tb in tb_cyc),
       f"t={tb_cyc[0]:.4f} s, spread {spread_c * 1e3:.2f} ms")

    tb_cir = []
    for i in range(N_B):
        th = TR_CIR[i][:, 0]
        j = int(np.argmax(np.diff(np.sign(th)) != 0))
        tm = TT[j] + DT * th[j] / (th[j] - th[j + 1])
        tp = np.sqrt(RHO / G_ACC) * ellipK(np.sin(abs(TH0[i]) / 2))
        assert abs(tm / tp - 1) < 1e-4, f"circle ball {i}: {tm} vs {tp}"
        tb_cir.append(tm)
    spread = max(tb_cir) - min(tb_cir)
    ck("circle CONTROL: times match sqrt(rho/g)*K(sin(th0/2)); NOT equal",
       spread > 0.05, f"spread {spread * 1e3:.1f} ms "
       f"({tb_cir[-1]:.4f}..{tb_cir[0]:.4f} s)")

    # -- return at end of run
    wret = 0.0
    for i in range(N_B):
        x0p, y0p = cyc_px(PHI0[i])
        x1p, y1p = cyc_px(TR_CYC[i][-1, 0])
        wret = max(wret, np.hypot(x1p - x0p, y1p - y0p))
    ck("cycloid balls end EXACTLY on their release marks",
       wret < 1e-6, f"worst {wret:.1e} px after 4 periods")
    dev = max(abs(cir_px(TR_CIR[i][-1, 0])[0] - cir_px(TH0[i])[0])
              for i in range(N_B))
    ck("circle balls do NOT return (scattered at the end)",
       dev > 200, f"max |x_end - x_release| = {dev:.0f} px")

    # -- lamp events
    ck("cycloid lamp fires 8 times, at t_rel + 0.8 + 1.6k",
       len(FLASH[1]) == 8 and
       all(abs(FLASH[1][k] - (T_REL + 0.8 + 1.6 * k)) < 2e-3
           for k in range(8)),
       f"times {['%.3f' % t for t in FLASH[1]]}")
    ck("circle lamp NEVER fires", len(FLASH[0]) == 0)

    # -- instrument self-test (trap 42), incl. cross-talk
    test = np.full((H, W, 3), PAPER, np.float64)
    spots = [(200.25, 700.75), (500.5, 700.25), (700.75, 700.5),
             (900.25, 700.75)]
    for (sx, sy), c in zip(spots, C_BALLS):
        x0, y0, cv = disc_cov(sx, sy, R_BALL)
        comp_bbox(test, x0, y0, cv, c)
    t8 = (np.clip(test, 0, 1) * 255 + 0.5).astype(np.uint8)
    werr = 0.0
    for i, (sx, sy) in enumerate(spots):
        c = mask_centroid(t8, M_FNS[i], int(sx) - 60, int(sx) + 60,
                          int(sy) - 60, int(sy) + 60)
        werr = max(werr, np.hypot(c[0] - sx, c[1] - sy))
        for jj, (ox, oy) in enumerate(spots):
            if jj != i:
                reg = t8[int(oy) - 30:int(oy) + 30,
                         int(ox) - 30:int(ox) + 30].astype(np.float64)
                assert M_FNS[i](reg).sum() == 0.0, \
                    f"mask {i} sees ball {jj}"
    ck("hue masks read known discs, zero cross-talk", werr < 0.05,
       f"worst centroid err {werr:.3f} px")

    # -- pixel checks on rendered frames
    f_rel = frame_at(PRE)
    worst = 0.0
    for bowl in (0, 1):
        for i in range(N_B):
            c = ball_centroid(f_rel, bowl, i, PRE / FPS)
            mx, my = shutter_mean_px(bowl, i, PRE / FPS)
            worst = max(worst, np.hypot(c[0] - mx, c[1] - my))
    ck("release frame: all 8 centroids == model release points",
       worst < 0.4, f"worst {worst:.3f} px")

    t_mid = T_REL + 0.55            # balls spread mid-fall, all moving
    f_mid = frame_at(int(round(t_mid * FPS)))
    worst = 0.0
    for bowl in (0, 1):
        for i in range(N_B):
            c = ball_centroid(f_mid, bowl, i,
                              int(round(t_mid * FPS)) / FPS)
            mx, my = shutter_mean_px(bowl, i, int(round(t_mid * FPS)) / FPS)
            worst = max(worst, np.hypot(c[0] - mx, c[1] - my))
    ck("mid-fall: blurred centroids == shutter-mean model (all 8)",
       worst < 0.4, f"worst {worst:.3f} px")

    # beads ride ON the wire (trap 66): measured centroid sits on curve
    wd = 0.0
    for bowl in (0, 1):
        _, trs, pxf, _ = BOWLS[bowl]
        for i in range(N_B):
            c = ball_centroid(f_mid, bowl, i, int(round(t_mid * FPS)) / FPS)
            qs = np.linspace(*((-ALPHA, ALPHA) if bowl == 0 else
                               (1e-3, 2 * np.pi - 1e-3)), 4000)
            xs, ys = pxf(qs)
            wd = max(wd, np.min(np.hypot(xs - c[0], ys - c[1])))
    ck("measured centroids lie ON their wire (coupling, trap 66)",
       wd < 1.0, f"worst distance to curve {wd:.2f} px")

    # the money frame: 8th simultaneous crossing at t = 13.6 s (frame 408)
    # -> during POST? no: t_rel+12.0 = 12.8 s -> frame 384.
    f384 = frame_at(384)
    cols_cy = sat_cols(f384, 1)
    cols_ci = sat_cols(f384, 0)
    ck("last crossing: cycloid balls MERGED at the bottom",
       cols_cy.min() > XC - 60 and cols_cy.max() < XC + 60,
       f"cols {cols_cy.min()}..{cols_cy.max()}")
    ck("last crossing: circle balls SCATTERED",
       cols_ci.max() - cols_ci.min() > 300,
       f"span {cols_ci.max() - cols_ci.min()} px")
    ck("last crossing: cycloid lamp lit, circle lamp dark",
       lamp_mean(f384, 1) < 0.45 * 255 and lamp_mean(f384, 0) > 0.7 * 255,
       f"B {lamp_mean(f384, 1) / 255:.2f}, A {lamp_mean(f384, 0) / 255:.2f}")

    # final frame: cycloid back on marks, circle scattered
    f_end = frame_at(N_FRAMES - 1)
    worst = 0.0
    for i in range(N_B):
        c = ball_centroid(f_end, 1, i, (N_FRAMES - 1) / FPS)
        mx, my = cyc_px(PHI0[i])
        worst = max(worst, np.hypot(c[0] - mx, c[1] - my))
    ck("final frame: cycloid balls back ON their release marks",
       worst < 0.5, f"worst {worst:.3f} px")

    # -- framing
    ink = np.abs(f_mid.astype(np.float64) - PAPER * 255).max(2) > 14
    rows = np.where(ink.any(1))[0]
    frac = ink.mean()
    ck("ink rows inside the safe area (192..1632)",
       rows[0] >= 192 and rows[-1] <= 1632,
       f"rows {rows[0]}..{rows[-1]}, lit {frac:.3f}")
    ck("lit fraction in a sane band", 0.01 < frac < 0.30, f"{frac:.3f}")

    # -- blur convergence
    fb = frame_at(int(round(t_mid * FPS)), ns=72)
    db = np.abs(fb.astype(int) - f_mid.astype(int))
    ck("NS=24 blur converged (vs NS=72)", db.max() <= 2,
       f"max byte diff {db.max()}")

    k = 360.0 / W
    print(f"        watch-size: ball {2 * R_BALL * k:.1f} px, bowl depth "
          f"{DEPTH_PX * k:.0f} px, end-scatter "
          f"{488 * k:.0f} px, lamp {2 * LAMP_R * k:.1f} px")
    ck("ball diameter at watch size >= 12 px", 2 * R_BALL * k >= 12)
    ck("432 frames = 14.4 s at 30 fps", N_FRAMES == 432)

    print(f"ALL {ok} CHECKS PASSED")
    print()
    print("NOT verified (trap 68) — the model's idealizations:")
    print("  - beads slide on frictionless wires; they do not roll")
    print("    (a rolling ball's contact point rides a parallel curve,")
    print("    which is not a cycloid — real ball-in-bowl demos are close,")
    print("    not exact)")
    print("  - no air drag; a real pair of bowls damps out in minutes")
    print("  - Huygens' pendulum clocks used cycloidal cheeks for exactly")
    print("    this isochronism; in practice the cheeks added friction")
    print("    that cost more than the isochronism bought")


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
    d = decode_frame(384)
    cols_cy = sat_cols(d, 1)
    cols_ci = sat_cols(d, 0)
    print(f"    last crossing off the file: cycloid cols "
          f"{cols_cy.min()}..{cols_cy.max()}, circle span "
          f"{cols_ci.max() - cols_ci.min()} px")
    assert cols_cy.min() > XC - 65 and cols_cy.max() < XC + 65
    assert cols_ci.max() - cols_ci.min() > 300
    d_end = decode_frame(N_FRAMES - 1)
    worst = 0.0
    for i in range(N_B):
        c = ball_centroid(d_end, 1, i, (N_FRAMES - 1) / FPS)
        mx, my = cyc_px(PHI0[i])
        worst = max(worst, np.hypot(c[0] - mx, c[1] - my))
    print(f"    final frame off the file: cycloid return worst "
          f"{worst:.3f} px")
    assert worst < 0.8
    dd = np.abs(decode_frame(0).astype(int) - frame_at(0).astype(int))
    print(f"    decoded f0 vs render: mean |diff| {dd.mean():.3f}, "
          f"max {dd.max()}")
    assert dd.mean() < 2.0
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams",
         "v", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
         OUT], capture_output=True, text=True).stdout.strip()
    assert int(probe) == N_FRAMES, f"frames {probe} != {N_FRAMES}"
    print(f"    {probe} frames; the isochronism survives the encode")


def review_stills():
    """Trap 67: look at it at the size it will be watched."""
    for name, i in [("release", PRE), ("first", 48), ("desync", 300),
                    ("final", N_FRAMES - 1)]:
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
    print(f"R={R:.5f} m  rho={RHO:.5f} m  T_cyc={T_CYC} s  "
          f"balls at {FRACS} of depth  {N_FRAMES} frames")
    run_checks()
    review_stills()
    if "--ship" in sys.argv:
        encode()
        check_encode()
