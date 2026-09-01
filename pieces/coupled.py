#!/usr/bin/env python3
"""COUPLED — two pendulums joined by a weak spring. The left one swings,
comes to a genuine dead stop in mid-air, and gets every joule back.

Design (all from the physics, none tuned by eye):
  - loop period T = 16.000 s = 480 frames at 30 fps, a perfect loop
  - delta = (w2-w1)/2 = pi/16, wbar = 17*delta (ODD multiple => the linear
    system returns EXACTLY to its initial state at t=T, and at t=T/2 the
    left pendulum has theta=0 AND thetadot=0 simultaneously: a true dead
    stop, not a zero crossing)
  - hence w1 = 16*delta = pi exactly: L = g/pi^2 = 0.994 m, the SECONDS
    pendulum (alone, without the spring, it would tick 2.000 s per period)
  - the render integrates the FULL NONLINEAR system with a real geometric
    spring (finite extension, force along the actual spring vector); the
    closed form of the linearized system is the cross-check, not the truth
  - amplitude A = 0.15 rad chosen from a measured sweep: nonlinear dead-stop
    residual 0.02 px, loop mismatch 0.33 px, both sub-pixel

Energy gauge: one bar of fixed total width. Red fills from the left with
pendulum 1's energy, blue from the right with pendulum 2's, the grey wedge
between them is the spring's share IN TRANSIT. The bar is always exactly
full — that is conservation, drawn.

Verified by the checks below; what is NOT verified is printed at the end
(trap 68).
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
INK = 0.10
INK_ROD = 0.32
INK_SPR = 0.25
GHOST = 0.58                    # plumb dashes (thin marks need contrast)
C_RED = (0.55, 0.10, 0.10)      # left bob
C_BLUE = (0.12, 0.16, 0.52)     # right bob
C_GREY = (0.50, 0.50, 0.50)     # spring's energy share in the gauge

# ---------------------------------------------------------------- physics
G_ACC = 9.81
DELTA = np.pi / 16.0
W1 = 16.0 * DELTA               # == pi exactly (asserted): seconds pendulum
W2 = 18.0 * DELTA
WBAR = 17.0 * DELTA
L = G_ACC / W1**2               # 0.993961 m
M = 1.0
D_PIV = 0.60                    # pivot separation (m); spring natural length
D_ATT = 0.50                    # spring attachment distance down each rod
K_SPR = (W2**2 - W1**2) / 2.0 * M * L**2 / D_ATT**2   # 5.180 N/m
A_AMP = 0.15                    # rad (8.6 deg)
T_LOOP = 16.0
N_FRAMES = 480

P1 = np.array([0.0, 0.0])       # pivot 1 (world, y down)
P2 = np.array([D_PIV, 0.0])


def deriv_nl(s):
    th1, th2, o1, o2 = s
    a1p = P1 + D_ATT * np.array([np.sin(th1), np.cos(th1)])
    a2p = P2 + D_ATT * np.array([np.sin(th2), np.cos(th2)])
    v = a2p - a1p
    ln = np.hypot(v[0], v[1])
    ext = ln - D_PIV
    f = K_SPR * ext * v / ln
    tau1 = D_ATT * (f[0] * np.cos(th1) - f[1] * np.sin(th1))
    tau2 = D_ATT * (-f[0] * np.cos(th2) + f[1] * np.sin(th2))
    al1 = (-M * G_ACC * L * np.sin(th1) + tau1) / (M * L**2)
    al2 = (-M * G_ACC * L * np.sin(th2) + tau2) / (M * L**2)
    return np.array([o1, o2, al1, al2])


def rk4_table(s0, t_end, dt):
    n = int(round(t_end / dt))
    s = s0.copy()
    traj = np.empty((n + 1, 4))
    traj[0] = s
    for i in range(n):
        k1 = deriv_nl(s)
        k2 = deriv_nl(s + 0.5 * dt * k1)
        k3 = deriv_nl(s + 0.5 * dt * k2)
        k4 = deriv_nl(s + dt * k3)
        s = s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        traj[i + 1] = s
    return traj


DT = 1e-3
TRAJ = rk4_table(np.array([A_AMP, 0.0, 0.0, 0.0]), T_LOOP, DT)
TT = np.arange(TRAJ.shape[0]) * DT


def state(t):
    """Interpolated (th1, th2, om1, om2) at time t, wrapped to the loop."""
    tw = t % T_LOOP
    return np.array([np.interp(tw, TT, TRAJ[:, j]) for j in range(4)])


def spring_ext(th1, th2):
    a1p = P1 + D_ATT * np.array([np.sin(th1), np.cos(th1)])
    a2p = P2 + D_ATT * np.array([np.sin(th2), np.cos(th2)])
    return np.hypot(*(a2p - a1p)) - D_PIV


def energies(s):
    th1, th2, o1, o2 = s
    e1 = 0.5 * M * L**2 * o1**2 + M * G_ACC * L * (1 - np.cos(th1))
    e2 = 0.5 * M * L**2 * o2**2 + M * G_ACC * L * (1 - np.cos(th2))
    es = 0.5 * K_SPR * spring_ext(th1, th2)**2
    return e1, e2, es


E_TOT = sum(energies(TRAJ[0]))

# ---------------------------------------------------------------- layout
SCALE = 760.0                   # px per metre
PIVY = 400.0
PX1 = 540.0 - D_PIV * SCALE / 2.0    # 312
PX2 = 540.0 + D_PIV * SCALE / 2.0    # 768
REST_Y = PIVY + L * SCALE            # 1155.4
R_BOB = 26.0

BAR_X0, BAR_X1 = 160, 920            # gauge outline
BAR_Y0, BAR_Y1 = 1440, 1484
BAR_IX0, BAR_IX1 = 163.0, 917.0      # fill interior
BAR_IY0, BAR_IY1 = 1443, 1481
BAR_W = BAR_IX1 - BAR_IX0

SH_RT = 1.0 / 60.0                   # 180-degree shutter at 30 fps
NS = 16                              # max streak ~6 px => ~1 sample/px

OUT = f"out/coupled_{time.strftime('%H%M%S')}.mp4"


def bob_px(th, pivx):
    return (pivx + L * SCALE * np.sin(th), PIVY + L * SCALE * np.cos(th))


def att_px(th, pivx):
    return (pivx + D_ATT * SCALE * np.sin(th),
            PIVY + D_ATT * SCALE * np.cos(th))


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


def polyseg_cov(pts, lw):
    """Coverage of connected segments, one bbox, exact distance-to-segment
    per segment, max-composited (no double-dark joints)."""
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


def spring_pts(a1, a2, n_zig=9, amp=13.0, lead=18.0):
    """Zigzag polyline between two attachment points (px)."""
    a1 = np.array(a1)
    a2 = np.array(a2)
    v = a2 - a1
    ln = np.hypot(*v)
    u = v / ln
    n = np.array([-u[1], u[0]])
    pts = [tuple(a1), tuple(a1 + u * lead)]
    z0, z1 = lead, ln - lead
    for i in range(n_zig):
        f = (i + 0.5) / n_zig
        s = 1.0 if i % 2 == 0 else -1.0
        pts.append(tuple(a1 + u * (z0 + f * (z1 - z0)) + n * (s * amp)))
    pts.append(tuple(a2 - u * lead))
    pts.append(tuple(a2))
    return pts


def background():
    fr = np.full((H, W, 3), PAPER, np.float64)
    # pivot bar
    x0, y0, cv = polyseg_cov([(250, PIVY), (830, PIVY)], 6.0)
    comp_bbox(fr, x0, y0, cv, INK)
    # pivot discs
    for px in (PX1, PX2):
        x0, y0, cv = disc_cov(px, PIVY, 7.0)
        comp_bbox(fr, x0, y0, cv, INK)
    # plumb dashes: the rest position each bob returns to
    for px in (PX1, PX2):
        y = PIVY + 34
        while y < REST_Y + 44:
            x0, y0, cv = polyseg_cov([(px, y), (px, min(y + 9, REST_Y + 44))],
                                     3.0)
            comp_bbox(fr, x0, y0, cv, GHOST)
            y += 22
    # gauge outline (fixed, always exactly full = conservation drawn)
    for seg in [[(BAR_X0, BAR_Y0), (BAR_X1, BAR_Y0)],
                [(BAR_X0, BAR_Y1), (BAR_X1, BAR_Y1)],
                [(BAR_X0, BAR_Y0), (BAR_X0, BAR_Y1)],
                [(BAR_X1, BAR_Y0), (BAR_X1, BAR_Y1)]]:
        x0, y0, cv = polyseg_cov(seg, 3.0)
        comp_bbox(fr, x0, y0, cv, INK)
    return fr


BG = background()


def hspan_cov(xs, xe):
    """Sub-pixel horizontal span coverage over the gauge interior."""
    cols = np.arange(int(BAR_IX0) - 1, int(BAR_IX1) + 2, dtype=np.float64)
    cov1 = np.clip(np.minimum(cols + 1.0, xe) - np.maximum(cols, xs), 0, 1)
    hgt = BAR_IY1 - BAR_IY0
    return int(cols[0]), BAR_IY0, np.tile(cov1, (hgt, 1))


def scene(t):
    """Full RGB scene at exact time t (no gauge; the gauge is an overlay)."""
    img = BG.copy()
    th1, th2, _, _ = state(t)
    b1 = bob_px(th1, PX1)
    b2 = bob_px(th2, PX2)
    a1 = att_px(th1, PX1)
    a2 = att_px(th2, PX2)
    for piv, bob in [((PX1, PIVY), b1), ((PX2, PIVY), b2)]:
        x0, y0, cv = polyseg_cov([piv, bob], 5.0)
        comp_bbox(img, x0, y0, cv, INK_ROD)
    x0, y0, cv = polyseg_cov(spring_pts(a1, a2), 4.0)
    comp_bbox(img, x0, y0, cv, INK_SPR)
    x0, y0, cv = disc_cov(*b1, R_BOB)
    comp_bbox(img, x0, y0, cv, C_RED)
    x0, y0, cv = disc_cov(*b2, R_BOB)
    comp_bbox(img, x0, y0, cv, C_BLUE)
    return img


def gauge(img, t):
    """Energy gauge at shutter-centre time (an instrument, drawn sharp)."""
    e1, e2, es = energies(state(t))
    w1 = e1 / E_TOT * BAR_W
    ws = es / E_TOT * BAR_W
    x0, y0, cv = hspan_cov(BAR_IX0, BAR_IX0 + w1)
    comp_bbox(img, x0, y0, cv, C_RED)
    x0, y0, cv = hspan_cov(BAR_IX0 + w1, BAR_IX0 + w1 + ws)
    comp_bbox(img, x0, y0, cv, C_GREY)
    x0, y0, cv = hspan_cov(BAR_IX0 + w1 + ws, BAR_IX1)
    comp_bbox(img, x0, y0, cv, C_BLUE)


def frame_at(i, ns=NS):
    t = i / FPS
    acc = np.zeros((H, W, 3), np.float64)
    for j in range(ns):
        off = (2 * j + 1 - ns) / (2.0 * ns) * SH_RT
        acc += scene(t + off)           # state() wraps mod T_LOOP
    img = acc / ns
    gauge(img, t)
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def render_frames():
    for i in range(N_FRAMES):
        yield frame_at(i)


# ---------------------------------------------------------------- measure
def mask_centroid(img, mask_fn, x_lo, x_hi, y_lo, y_hi):
    reg = img[y_lo:y_hi, x_lo:x_hi, :].astype(np.float64)
    wgt = mask_fn(reg)
    xs = np.arange(x_lo, x_hi, dtype=np.float64)
    ys = np.arange(y_lo, y_hi, dtype=np.float64)
    tot = wgt.sum()
    if tot <= 0:
        return None
    return ((wgt.sum(0) * xs).sum() / tot, (wgt.sum(1) * ys).sum() / tot)


def m_red(reg):
    return np.clip(reg[:, :, 0] - reg[:, :, 1] - 60.0, 0.0, None)


def m_blue(reg):
    return np.clip(reg[:, :, 2] - reg[:, :, 1] - 60.0, 0.0, None)


MASKS = {"red": (m_red, 0), "blue": (m_blue, 1)}
PIVX = {"red": PX1, "blue": PX2}


def bob_centroid(img, name, t, box=60):
    """Bob centroid by unique colour, bounded to a box round the MODEL
    position and to rows above the gauge (traps 58/64)."""
    mfn, j = MASKS[name]
    th = state(t)[j]
    mx, my = bob_px(th, PIVX[name])
    x_lo, x_hi = int(mx) - box, int(mx) + box
    y_lo, y_hi = max(int(my) - box, 0), min(int(my) + box, 1350)
    return mask_centroid(img, mfn, x_lo, x_hi, y_lo, y_hi)


def shutter_mean_px(name, t):
    _, j = MASKS[name]
    xs, ys = [], []
    for k in range(NS):
        off = (2 * k + 1 - NS) / (2.0 * NS) * SH_RT
        th = state(t + off)[j]
        px, py = bob_px(th, PIVX[name])
        xs.append(px)
        ys.append(py)
    return float(np.mean(xs)), float(np.mean(ys))


def bar_widths(img):
    """Measured red/grey/blue widths (px) in the gauge interior."""
    band = img[BAR_IY0 + 4:BAR_IY1 - 4, int(BAR_IX0) + 1:int(BAR_IX1) - 1, :]
    band = band.astype(np.float64)
    red = (band[:, :, 0] - band[:, :, 1] > 60).mean(0)
    blue = (band[:, :, 2] - band[:, :, 1] > 60).mean(0)
    return red.sum(), blue.sum()


# ---------------------------------------------------------------- checks
def run_checks():
    ok = 0

    def ck(name, cond, detail=""):
        nonlocal ok
        assert cond, f"CHECK FAILED: {name} {detail}"
        ok += 1
        print(f"  ok {ok:2d}  {name}{('  ' + detail) if detail else ''}")

    print("CHECKS")
    # -- construction
    ck("w1 is exactly pi (seconds pendulum by construction)", W1 == np.pi,
       f"w1={W1!r}")
    ck("L*pi^2 == g", abs(L * np.pi**2 - G_ACC) < 1e-12, f"L={L:.6f} m")
    ck("wbar/delta = 17, odd (exact recurrence at T)",
       abs(WBAR / DELTA - 17.0) < 1e-12)

    # -- linear closed form vs linear RK4 vs nonlinear limit
    def closed(t, amp):
        return (amp * np.cos(DELTA * t) * np.cos(WBAR * t),
                amp * np.sin(DELTA * t) * np.sin(WBAR * t))

    def deriv_lin(s):
        th1, th2, o1, o2 = s
        c = K_SPR * D_ATT**2 / (M * L**2)
        return np.array([o1, o2, -W1**2 * th1 + c * (th2 - th1),
                         -W1**2 * th2 + c * (th1 - th2)])

    def rk4g(deriv, s0, t_end, dt):
        n = int(round(t_end / dt))
        s = s0.copy()
        tr = np.empty((n + 1, 4))
        tr[0] = s
        for i in range(n):
            k1 = deriv(s)
            k2 = deriv(s + 0.5 * dt * k1)
            k3 = deriv(s + 0.5 * dt * k2)
            k4 = deriv(s + dt * k3)
            s = s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            tr[i + 1] = s
        return tr

    tl = np.arange(0, int(round(T_LOOP / DT)) + 1) * DT
    trB = rk4g(deriv_lin, np.array([A_AMP, 0, 0, 0]), T_LOOP, DT)
    cf1, cf2 = closed(tl, A_AMP)
    errB = max(np.max(np.abs(trB[:, 0] - cf1)), np.max(np.abs(trB[:, 1] - cf2)))
    ck("linear RK4 == closed form", errB < 1e-9, f"max err {errB:.2e} rad")
    a0 = 1e-3
    trC = rk4g(deriv_nl, np.array([a0, 0, 0, 0]), T_LOOP, DT)
    cf1, cf2 = closed(tl, a0)
    errC = max(np.max(np.abs(trC[:, 0] - cf1)), np.max(np.abs(trC[:, 1] - cf2)))
    ck("nonlinear model reduces to linear at tiny amplitude",
       errC < 2e-6 * a0 * 1e3, f"max err {errC:.2e} rad at A=1e-3")

    # -- how nonlinear is the shipped amplitude? (trap 62: the deviation has
    # a closed form — pendulum detuning dw/w ~ A^2/16 accumulates as carrier
    # phase drift, so max dev ~ A * (A^2/16) * wbar * T. Assert the MATCH.)
    cf1, cf2 = closed(tl, A_AMP)
    dev = max(np.max(np.abs(TRAJ[:, 0] - cf1)), np.max(np.abs(TRAJ[:, 1] - cf2)))
    pred = A_AMP * (A_AMP**2 / 16.0) * WBAR * T_LOOP
    ck("nonlinear deviation from linear MATCHES the A^3 drift prediction",
       0.6 < dev / pred < 1.4,
       f"max {dev:.5f} rad vs predicted {pred:.5f} (ratio {dev / pred:.2f})")

    # -- energy conservation (integrator health)
    ee = np.array([sum(energies(TRAJ[i])) for i in range(0, len(TRAJ), 200)])
    de = np.max(np.abs(ee - E_TOT)) / E_TOT
    ck("energy conserved along the whole run", de < 1e-10, f"dE/E {de:.1e}")

    # -- the dead stop (instrument self-tested in feas_coupled.py)
    i7, i9 = int(7.0 / DT), int(9.0 / DT)
    rad = np.sqrt(TRAJ[i7:i9, 0]**2 + (TRAJ[i7:i9, 2] / W1)**2)
    imin = int(np.argmin(rad))
    t_stop = (i7 + imin) * DT
    ck("left pendulum reaches a TRUE dead stop (theta AND thetadot ~ 0)",
       rad[imin] < 5e-5,
       f"residual {rad[imin]:.2e} rad = {rad[imin] * L * SCALE:.4f} px "
       f"at t={t_stop:.3f} s")
    e1s, e2s, ess = energies(state(t_stop))
    ck("at the stop the left pendulum's energy is ~zero",
       e1s / E_TOT < 1e-6, f"E1/E0 = {e1s / E_TOT:.1e}")
    ck("at t=8 the right pendulum holds what the left held at t=0",
       abs(e2s / E_TOT - energies(TRAJ[0])[0] / E_TOT) < 2e-3,
       f"{e2s / E_TOT:.4f} vs {energies(TRAJ[0])[0] / E_TOT:.4f}")

    # -- loop closure (the perfect-loop claim)
    dth = np.max(np.abs(TRAJ[-1, :2] - TRAJ[0, :2]))
    dom = np.max(np.abs(TRAJ[-1, 2:] - TRAJ[0, 2:]))
    # position mismatch shows as bob offset; velocity mismatch only shows
    # through the blur streak, scale = dom * L * SCALE * shutter
    dth_px = dth * L * SCALE
    dom_px = dom * L * SCALE * SH_RT
    # vel mismatch = A*wbar^2*tau (linear in the A^3 phase drift tau where
    # position is quadratic): predicted 0.037 rad/s -> 0.47 px of streak.
    # Both sub-pixel; the frame-level pixel check below is the decider.
    ck("state at t=16 equals state at t=0 (sub-pixel loop, pos AND vel)",
       dth_px < 0.45 and dom_px < 0.6,
       f"pos {dth_px:.2f} px, vel-through-blur {dom_px:.3f} px")

    # -- normal coordinates: the coupling relation (trap 66)
    s_sum = TRAJ[:, 0] + TRAJ[:, 1]
    s_dif = TRAJ[:, 0] - TRAJ[:, 1]
    d_sum = np.max(np.abs(s_sum - A_AMP * np.cos(W1 * tl)))
    d_dif = np.max(np.abs(s_dif - A_AMP * np.cos(W2 * tl)))
    # deviations are the SAME A^3 carrier drift seen in mode coordinates,
    # so they too must match pred (not merely "be small")
    ck("normal coordinates: th1+th2 ~ A cos(w1 t), th1-th2 ~ A cos(w2 t)",
       max(d_sum, d_dif) < 1.5 * pred and min(d_sum, d_dif) > 0.5 * pred,
       f"max dev {d_sum:.5f} / {d_dif:.5f} rad vs pred {pred:.5f}")

    # -- spring numbers for the description
    exts = np.array([spring_ext(TRAJ[i, 0], TRAJ[i, 1])
                     for i in range(0, len(TRAJ), 40)])
    print(f"        spring: k={K_SPR:.3f} N/m, extension within "
          f"[{exts.min() * 100:.1f}, {exts.max() * 100:.1f}] cm, "
          f"max energy share {np.max([energies(TRAJ[i])[2] for i in range(0, len(TRAJ), 200)]) / E_TOT * 100:.1f}%")

    # -- instrument self-test on a known scene (trap 42)
    test = np.full((H, W, 3), PAPER, np.float64)
    x0, y0, cv = disc_cov(400.25, 900.75, R_BOB)
    comp_bbox(test, x0, y0, cv, C_RED)
    x0, y0, cv = disc_cov(700.5, 900.25, R_BOB)
    comp_bbox(test, x0, y0, cv, C_BLUE)
    t8 = (np.clip(test, 0, 1) * 255 + 0.5).astype(np.uint8)
    cr = mask_centroid(t8, m_red, 340, 460, 840, 960)
    cb = mask_centroid(t8, m_blue, 640, 760, 840, 960)
    err = max(np.hypot(cr[0] - 400.25, cr[1] - 900.75),
              np.hypot(cb[0] - 700.5, cb[1] - 900.25))
    ck("colour-mask instruments read known discs", err < 0.05,
       f"worst {err:.3f} px")

    # -- pixel checks on rendered frames
    f0 = frame_at(0)
    c = bob_centroid(f0, "red", 0.0)
    mx, my = shutter_mean_px("red", 0.0)
    d0 = np.hypot(c[0] - mx, c[1] - my)
    ck("frame 0: red bob centroid == shutter-mean model position",
       d0 < 0.35, f"{d0:.3f} px")

    f240 = frame_at(240)                      # t = 8.0, the dead stop
    c = bob_centroid(f240, "red", 8.0)
    dstop = np.hypot(c[0] - PX1, c[1] - REST_Y)
    ck("dead-stop frame: red bob hangs AT REST on its plumb line",
       dstop < 0.5, f"{dstop:.3f} px from (312, {REST_Y:.1f})")
    # no streak on either bob at t=8 (both momentarily at rest)
    for name in ("red", "blue"):
        mfn, j = MASKS[name]
        th = state(8.0)[j]
        mx, my = bob_px(th, PIVX[name])
        reg = f240[int(my) - 60:int(my) + 60,
                   int(mx) - 60:int(mx) + 60].astype(np.float64)
        wgt = mfn(reg)
        cols = np.where(wgt.sum(0) > 1)[0]
        wid = cols[-1] - cols[0] + 1
        ck(f"dead-stop frame: {name} bob has no streak (width ~ 2r)",
           abs(wid - 2 * R_BOB) < 4, f"width {wid} px vs {2 * R_BOB:.0f}")

    f120 = frame_at(120)                      # t = 4.0, mid-transfer
    worst = 0.0
    for name in ("red", "blue"):
        c = bob_centroid(f120, name, 4.0)
        mx, my = shutter_mean_px(name, 4.0)
        worst = max(worst, np.hypot(c[0] - mx, c[1] - my))
    ck("mid-transfer: blurred centroids == shutter-mean (both bobs)",
       worst < 0.35, f"worst {worst:.3f} px")

    # -- gauge reads the physics
    e1, e2, es = energies(state(0.0))
    rw, bw = bar_widths(f0)
    ck("gauge frame 0: red width == E1/E0 * bar",
       abs(rw - e1 / E_TOT * BAR_W) < 2.5,
       f"{rw:.1f} px vs {e1 / E_TOT * BAR_W:.1f}")
    e1, e2, es = energies(state(8.0))
    rw8, bw8 = bar_widths(f240)
    ck("gauge dead stop: blue width == E2/E0 * bar",
       abs(bw8 - e2 / E_TOT * BAR_W) < 2.5,
       f"{bw8:.1f} px vs {e2 / E_TOT * BAR_W:.1f}")
    ck("gauge mirror: red share at t=0 == blue share at t=8",
       abs(rw - bw8) < 2.5, f"{rw:.1f} vs {bw8:.1f} px")
    # bar always full: no paper-coloured pixels inside the interior
    for fr, nm in [(f0, "f0"), (f120, "f120"), (f240, "f240")]:
        band = fr[BAR_IY0 + 4:BAR_IY1 - 4,
                  int(BAR_IX0) + 2:int(BAR_IX1) - 2, :].astype(np.float64)
        paperish = ((np.abs(band - PAPER * 255).max(2) < 12)).mean()
        assert paperish < 0.002, f"gauge not full at {nm}: {paperish}"
    ck("gauge is exactly full in every sampled frame (conservation, drawn)",
       True)

    # -- the loop, on pixels. NOTE what this tests: frame_at(480) samples
    # through the t % T_LOOP wrap, so byte-identity here verifies the WRAP
    # machinery; the sim's own closure (the seam frame 479 -> 0 crosses) is
    # check 11's sub-pixel state bound.
    fend = frame_at(480)
    dd = np.abs(fend.astype(int) - f0.astype(int))
    ck("frame at t=16.0 equals frame 0 (wrap machinery; closure is ck 11)",
       dd.mean() < 0.05 and dd.max() <= 60,
       f"mean |diff| {dd.mean():.4f}, max {dd.max()}")

    # -- blur convergence (sample budget)
    f120b = frame_at(120, ns=48)
    db = np.abs(f120b.astype(int) - f120.astype(int))
    ck("NS=16 blur is converged (vs NS=48)", db.max() <= 2,
       f"max byte diff {db.max()}")

    # -- framing
    ink = np.abs(f120.astype(np.float64) - PAPER * 255).max(2) > 14
    rows = np.where(ink.any(1))[0]
    frac = ink.mean()
    ck("ink rows inside the safe area (192..1632)",
       rows[0] >= 192 and rows[-1] <= 1632,
       f"rows {rows[0]}..{rows[-1]}, lit {frac:.3f}")
    ck("lit fraction in a sane band", 0.02 < frac < 0.30)

    # -- watch size numerics (trap 67; the stills get LOOKED at too)
    k = 360.0 / W
    print(f"        watch-size: bob {2 * R_BOB * k:.1f} px, "
          f"swing {2 * A_AMP * L * SCALE * k:.1f} px, "
          f"gauge {(BAR_Y1 - BAR_Y0) * k:.1f} px tall, "
          f"spring amp {26 * k:.1f} px")
    ck("bob diameter at watch size >= 15 px", 2 * R_BOB * k >= 15)

    ck("480 frames = 16.000 s at 30 fps", N_FRAMES == 480)

    print(f"ALL {ok} CHECKS PASSED")
    print()
    print("NOT verified (trap 68) — the model's idealizations:")
    print("  - rods rigid and massless, bobs point masses, motion planar")
    print("  - spring ideal Hookean, massless, lossless")
    print("  - no air drag, no pivot friction: a real pair exchanges the")
    print("    same way but the envelope decays over minutes")
    print("  - the exact 16.000 s recurrence is a property of the tuned")
    print("    frequency ratio (w2/w1 = 9/8); an untuned pair still trades")
    print("    energy fully back and forth, just not on a clean loop")


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
    d240 = decode_frame(240)
    c = bob_centroid(d240, "red", 8.0)
    dstop = np.hypot(c[0] - PX1, c[1] - REST_Y)
    print(f"    dead stop off the file: red bob {dstop:.3f} px from rest")
    assert dstop < 0.8
    d120 = decode_frame(120)
    worst = 0.0
    for name in ("red", "blue"):
        c = bob_centroid(d120, name, 4.0)
        mx, my = shutter_mean_px(name, 4.0)
        worst = max(worst, np.hypot(c[0] - mx, c[1] - my))
    print(f"    mid-transfer centroids vs model: worst {worst:.3f} px")
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
    print(f"    {probe} frames; the exchange survives the encode")


def review_stills():
    """Trap 67: look at it at the size it will be watched."""
    for name, i in [("start", 0), ("mid", 120), ("deadstop", 240),
                    ("return", 360)]:
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
    print(f"w1=pi (seconds pendulum, L={L:.4f} m), w2={W2:.4f}, "
          f"k={K_SPR:.3f} N/m, A={A_AMP} rad, loop {T_LOOP} s")
    run_checks()
    review_stills()
    if "--ship" in sys.argv:
        encode()
        check_encode()
