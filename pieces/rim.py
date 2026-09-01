#!/usr/bin/env python3
"""RIM — the cycloid's genesis: a pen on a rolling wheel.

A wheel rolls at constant speed along a road. A red pen fixed to its rim
draws the cycloid — and stops dead once per turn, at the exact instant it
becomes the wheel's contact point. Two identical lights, one on the pen
and one on the hub, glow when their point is slow: the pen's fires at
every cusp, the hub's never comes on (the wheel never slows). The ink
pools dark at the cusps because dwell time scales as 1/speed. Then the
first arch rotates a half turn down onto a ghost outline waiting below
and lands exactly: the bowl from the last two videos (brachistochrone,
tautochrone). One parameter: the wheel's radius.

Third piece of the cycloid family, closing the trilogy. Not a loop, has
on-canvas words (COUPLED's loop + wordless experiments stay single-
instance). Silent.
"""
import os
import subprocess
import time

import numpy as np

# ---------------------------------------------------------------- canvas
W, H = 1080, 1920
FPS = 30
PAPER = 0.845                   # trap 69
INK = 0.10
GHOST = 0.58
C_RED = (0.55, 0.10, 0.10)

# ---------------------------------------------------------------- model
X0 = 50.0
R_PX = 980.0 / (4 * np.pi)      # 77.986 px: two arches exactly span 980
ARCH = 2 * np.pi * R_PX         # 490.0
ROAD = 620.0
T_TURN = 1.6                    # s per wheel turn (48 frames)
OM = 2 * np.pi / T_TURN
V_C = ARCH / T_TURN             # hub speed, 306.25 px/s
V_TOL = 0.35 * V_C              # light glow scale

PRE = 24                        # hold at start
ROLL = 96                       # 2 turns
HOLD = 24                       # hold on the finished trail
FLIP = 48                       # arch 1 rotates down onto the bowl
POST = 66                       # final hold
N_FRAMES = PRE + ROLL + HOLD + FLIP + POST      # 258 = 8.6 s
T_REL = PRE / FPS               # 0.8
T_END = T_REL + ROLL / FPS      # 4.0 roll ends
T_FLIP0 = T_END + HOLD / FPS    # 4.8
T_FLIP1 = T_FLIP0 + FLIP / FPS  # 6.4
LIGHT_FADE = 0.45               # s, lights fade out after the roll

SH_RT = 1.0 / 60.0
NS = 16                         # peak streak 10.2 px -> 0.64 px/sample

OUT = f"out/rim_{time.strftime('%H%M%S')}.mp4"

# bowl target (arch 1 lands here)
Y_RIM_BOWL = 1290.0
PIV0 = np.array([X0 + np.pi * R_PX, ROAD])       # arch 1 chord midpoint
PIV1 = np.array([W / 2.0, Y_RIM_BOWL])
LBL_Y = 1560                    # "cycloid" label top row

R_PEN = 16.0
R_HUB = 13.0                    # hub ring radius
RING_PEN = 30.0                 # pen light ring radius (bumped 24->30
                                # after the trap-67 watch-size look: the
                                # lit annulus was ~2 px on a phone)


def phi_of(t):
    return OM * np.clip(t - T_REL, 0.0, 2 * T_TURN)


def hub_px(t):
    ph = phi_of(t)
    return X0 + R_PX * ph, ROAD - R_PX


def pen_px(t):
    ph = phi_of(t)
    return (X0 + R_PX * (ph - np.sin(ph)),
            ROAD - R_PX * (1 - np.cos(ph)))


def pen_speed(t):
    if t <= T_REL or t >= T_END:
        return 0.0
    return 2 * V_C * abs(np.sin(phi_of(t) / 2))


def glow_of(v):
    return float(np.exp(-(v / V_TOL) ** 2))


def flip_ease(t):
    u = np.clip((t - T_FLIP0) / (T_FLIP1 - T_FLIP0), 0.0, 1.0)
    return 3 * u * u - 2 * u ** 3


def flip_xform(pts, t):
    """Rigid: rotate by pi*e about the arch chord midpoint, carried to
    the moving pivot. e=0 identity, e=1 lands on the bowl."""
    e = flip_ease(t)
    th = np.pi * e
    piv = PIV0 + (PIV1 - PIV0) * e
    c, s = np.cos(th), np.sin(th)
    d = pts - PIV0
    return np.stack([c * d[:, 0] - s * d[:, 1] + piv[0],
                     s * d[:, 0] + c * d[:, 1] + piv[1]], 1)


# ---------------------------------------------------------------- trail
DT_S = 1 / 240.0                # stamp interval
R_S = 3.2                       # stamp radius
A_S = 0.16                      # per-stamp alpha

T_ST = np.arange(0.0, 2 * T_TURN + DT_S / 2, DT_S)      # roll-relative
PH_ST = OM * T_ST
ST_X = X0 + R_PX * (PH_ST - np.sin(PH_ST))
ST_Y = ROAD - R_PX * (1 - np.cos(PH_ST))
ARCH1 = PH_ST <= 2 * np.pi + 1e-12       # first-turn stamps flip later


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


def alpha_bbox(buf, x0, y0, cov, a):
    """Over-composite a stamp into a scalar alpha buffer."""
    h, w = cov.shape
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x0 + w, W), min(y0 + h, H)
    if x1c <= x0c or y1c <= y0c:
        return
    cv = cov[y0c - y0:y1c - y0, x0c - x0:x1c - x0]
    reg = buf[y0c:y1c, x0c:x1c]
    reg += a * cv * (1 - reg)


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
    "l": "01100 00100 00100 00100 00100 00100 01110",
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


def stamp_center(img, sstr, cx, ytop, color, fade=1.0):
    m = text_mask(sstr) * fade
    h, w = m.shape
    x0 = int(round(cx - w / 2))
    col = np.asarray(color, np.float64) if np.ndim(color) else \
        np.array([color] * 3, np.float64)
    reg = img[ytop:ytop + h, x0:x0 + w, :]
    reg[...] = reg * (1 - m[..., None]) + col[None, None, :] * m[..., None]


# ---------------------------------------------------------------- curves
def bowl_curve():
    ph = np.linspace(0, 2 * np.pi, 2001)
    bx = PIV1[0] - np.pi * R_PX + R_PX * (ph - np.sin(ph))
    by = Y_RIM_BOWL + R_PX * (1 - np.cos(ph))
    return np.stack([bx, by], 1)


def resample(pts, step=2.0):
    s = np.concatenate([[0], np.cumsum(
        np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])))])
    u = np.arange(0, s[-1], step)
    return np.stack([np.interp(u, s, pts[:, 0]),
                     np.interp(u, s, pts[:, 1])], 1)


def background():
    fr = np.full((H, W, 3), PAPER, np.float64)
    # the road
    x0, y0, cv = polyseg_cov([(28.0, ROAD), (W - 28.0, ROAD)], 4.0)
    comp_bbox(fr, x0, y0, cv, INK)
    # ghost bowl: dashed outline of the landing curve
    bp = resample(bowl_curve(), 2.0)
    for k in range(len(bp)):
        if (k // 7) % 2 == 0:
            x0, y0, cv = disc_cov(bp[k, 0], bp[k, 1], 1.6)
            comp_bbox(fr, x0, y0, cv, GHOST)
    return fr


BG = background()

# static trail alpha buffers, built once
def build_trail(idx):
    buf = np.zeros((H, W), np.float64)
    for k in idx:
        x0, y0, cv = disc_cov(ST_X[k], ST_Y[k], R_S)
        alpha_bbox(buf, x0, y0, cv, A_S)
    return buf


TRAIL_FULL = None               # filled lazily
TRAIL_A2 = None


def trail_alpha(t):
    """Trail alpha buffer at time t (cached full/partial builds)."""
    global TRAIL_FULL, TRAIL_A2
    if t >= T_FLIP0:
        if TRAIL_A2 is None:
            TRAIL_A2 = build_trail(np.where(~ARCH1)[0])
        return TRAIL_A2
    n = int(np.clip((t - T_REL) / DT_S, 0, len(T_ST) - 1)) + 1 \
        if t > T_REL else 0
    if n >= len(T_ST):
        if TRAIL_FULL is None:
            TRAIL_FULL = build_trail(range(len(T_ST)))
        return TRAIL_FULL
    return build_trail(range(n))


# during the render frames come in order; keep an incremental buffer
class TrailInc:
    def __init__(self):
        self.buf = np.zeros((H, W), np.float64)
        self.n = 0

    def upto(self, t):
        n = int(np.clip((t - T_REL) / DT_S, 0, len(T_ST) - 1)) + 1 \
            if t > T_REL else 0
        if n < self.n:
            self.buf[:] = 0.0
            self.n = 0
        for k in range(self.n, n):
            x0, y0, cv = disc_cov(ST_X[k], ST_Y[k], R_S)
            alpha_bbox(self.buf, x0, y0, cv, A_S)
        self.n = n
        return self.buf


TINC = TrailInc()


def compose_trail(img, a):
    img[...] = img * (1 - a[..., None]) + \
        np.array(C_RED)[None, None, :] * a[..., None]


def draw_wheel(img, t):
    cx, cy = hub_px(t)
    ph = phi_of(t)
    x0, y0, cv = ring_cov(cx, cy, R_PX, 5.0)
    comp_bbox(img, x0, y0, cv, INK)
    for k in range(4):
        th = ph + k * np.pi / 2
        p0 = (cx + R_HUB * np.sin(th), cy - R_HUB * np.cos(th))
        p1 = (cx + (R_PX - 3) * np.sin(th), cy - (R_PX - 3) * np.cos(th))
        x0, y0, cv = polyseg_cov([p0, p1], 3.5)
        comp_bbox(img, x0, y0, cv, INK)
    # solid hub plate: occludes the trail when the hub crosses it, and
    # keeps the hub light's interior honest (only glow ink can be there)
    x0, y0, cv = disc_cov(cx, cy, R_HUB + 0.5)
    comp_bbox(img, x0, y0, cv, PAPER)
    x0, y0, cv = ring_cov(cx, cy, R_HUB, 3.0)
    comp_bbox(img, x0, y0, cv, INK)
    px, py = pen_px(t)
    x0, y0, cv = disc_cov(px, py, R_PEN)
    comp_bbox(img, x0, y0, cv, C_RED)


def light_fade(t):
    if t < T_REL:
        return 0.0
    if t <= T_END:
        return 1.0
    return float(np.clip(1 - (t - T_END) / LIGHT_FADE, 0.0, 1.0))


def lights(img, t):
    """Sharp overlay: identical rule for both — glow when your point is
    slow. Reading freezes when the roll ends, then fades with the ring."""
    fade = light_fade(t)
    if fade <= 0.0:
        return
    teff = float(np.clip(t, T_REL + 1e-9, T_END - 1e-9))
    g_pen = glow_of(2 * V_C * abs(np.sin(phi_of(teff) / 2)))
    g_hub = glow_of(V_C)
    px, py = pen_px(t)
    cx, cy = hub_px(t)
    x0, y0, cv = ring_cov(px, py, RING_PEN, 3.0)
    comp_bbox(img, x0, y0, cv * fade, INK)
    x0, y0, cv = ring_cov(px, py, (RING_PEN + R_PEN + 1) / 2,
                          RING_PEN - R_PEN - 3)      # gap annulus fills
    comp_bbox(img, x0, y0, cv * fade * g_pen, INK)
    # hub light: the hub ring's own interior fills (plate sits under it)
    x0, y0, cv = disc_cov(cx, cy, R_HUB - 2.0)
    comp_bbox(img, x0, y0, cv * fade * g_hub, INK)


def label(img, t):
    born = T_REL + T_TURN                    # arch 1 complete
    fade = float(np.clip((t - born) / 0.4, 0.0, 1.0))
    if fade > 0:
        stamp_center(img, "cycloid", W / 2, LBL_Y, INK, fade)


def scene(t, trail_buf):
    img = BG.copy()
    compose_trail(img, trail_buf)
    if t >= T_FLIP0:
        # arch 1 stamps under the rigid flip transform, sharp
        pts = flip_xform(np.stack([ST_X[ARCH1], ST_Y[ARCH1]], 1), t)
        fb = np.zeros((H, W), np.float64)
        for k in range(pts.shape[0]):
            x0, y0, cv = disc_cov(pts[k, 0], pts[k, 1], R_S)
            alpha_bbox(fb, x0, y0, cv, A_S)
        compose_trail(img, fb)
    draw_wheel(img, t)
    return img


def frame_at(i, inc=None):
    t = i / FPS
    moving = T_REL - SH_RT < t < T_END + SH_RT
    ns = NS if moving else 1
    tb = (inc.upto(t) if inc is not None else trail_alpha(t)) \
        if t < T_FLIP0 else trail_alpha(t)
    if ns == 1:
        img = scene(t, tb)
    else:
        img = np.zeros((H, W, 3), np.float64)
        for j in range(ns):
            off = (2 * j + 1 - ns) / (2.0 * ns) * SH_RT
            img += scene(t + off, tb)
        img /= ns
    lights(img, t)
    label(img, t)
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def render_frames():
    inc = TrailInc()
    for i in range(N_FRAMES):
        yield frame_at(i, inc)


# ---------------------------------------------------------------- measure
def red_mask(reg):
    return np.clip(reg[:, :, 0] - reg[:, :, 1] - 60, 0, None) * \
        (reg[:, :, 2] - reg[:, :, 1] < 40)


def red_alpha_at(img, x, y):
    """Ink alpha from the green channel: paper 0.845 -> red 0.10."""
    g = img[int(round(y)), int(round(x)), 1] / 255.0
    return (PAPER - g) / (PAPER - C_RED[1])


def centerline_alpha(img, ph_lo, ph_hi, n=25):
    ph = np.linspace(ph_lo, ph_hi, n)
    xs = X0 + R_PX * (ph - np.sin(ph))
    ys = ROAD - R_PX * (1 - np.cos(ph))
    return float(np.mean([red_alpha_at(img, x, y)
                          for x, y in zip(xs, ys)]))


def red_extent_x(img, cx, cy, box=40, a_thresh=0.15):
    """Horizontal extent of red TINT (alpha > a_thresh off the green
    channel). red_mask's implicit alpha cut (~0.52) would erase most of
    a shutter streak, whose ends are pale by construction."""
    x_lo, x_hi = int(cx) - box, int(cx) + box
    y_lo, y_hi = int(cy) - box, int(cy) + box
    reg = img[y_lo:y_hi, x_lo:x_hi, :].astype(np.float64) / 255.0
    a = (PAPER - reg[:, :, 1]) / (PAPER - C_RED[1])
    cols = np.where((a > a_thresh).any(0))[0]
    return (cols.max() - cols.min() + 1) if len(cols) else 0


def annulus_mean(img, cx, cy, r, lw):
    x0, y0, cv = ring_cov(cx, cy, r, lw)
    reg = img[y0:y0 + cv.shape[0], x0:x0 + cv.shape[1], :] \
        .astype(np.float64)
    return float((reg.mean(2) * cv).sum() / cv.sum())


def disc_mean(img, cx, cy, r):
    x0, y0, cv = disc_cov(cx, cy, r)
    reg = img[y0:y0 + cv.shape[0], x0:x0 + cv.shape[1], :] \
        .astype(np.float64)
    return float((reg.mean(2) * cv).sum() / cv.sum())


# ---------------------------------------------------------------- checks
def run_checks():
    ok = 0

    def ck(name, cond, detail=""):
        nonlocal ok
        assert cond, f"CHECK FAILED: {name} {detail}"
        ok += 1
        print(f"  ok {ok:2d}  {name}{('  ' + detail) if detail else ''}")

    print("CHECKS")
    # -- model
    ck("two arches exactly span the 980 px budget",
       abs(2 * ARCH - 980.0) < 1e-9, f"R={R_PX:.4f} px")

    ts = np.linspace(T_REL + 0.01, T_END - 0.01, 2000)
    eps = 1e-6
    vx = np.array([(pen_px(t + eps)[0] - pen_px(t - eps)[0]) / (2 * eps)
                   for t in ts])
    vy = np.array([(pen_px(t + eps)[1] - pen_px(t - eps)[1]) / (2 * eps)
                   for t in ts])
    verr = np.max(np.abs(np.hypot(vx, vy) -
                         [pen_speed(t) for t in ts])) / (2 * V_C)
    ck("pen speed law |v| = 2 v_c |sin(phi/2)|", verr < 1e-6,
       f"rel err {verr:.1e}")

    hx = np.array([(hub_px(t + eps)[0] - hub_px(t - eps)[0]) / (2 * eps)
                   for t in ts])
    herr = np.max(np.abs(hx - V_C)) / V_C
    ck("hub speed constant at v_c", herr < 1e-6, f"rel err {herr:.1e}")

    worst = 0.0
    for k in (0, 1, 2):
        t = T_REL + k * T_TURN
        px, py = pen_px(t)
        cxx = X0 + R_PX * phi_of(t)          # contact point x
        worst = max(worst, abs(px - cxx), abs(py - ROAD), pen_speed(t))
    ck("cusp coupling: pen IS the contact point, v = 0 (trap 66)",
       worst < 1e-9, f"worst {worst:.1e}")

    g_hub = glow_of(V_C)
    g_q = glow_of(pen_speed(T_REL + T_TURN / 4))
    ck("light margins: hub never fires, pen fires only at cusps",
       g_hub < 1e-3 and g_q < 1e-6 and
       glow_of(pen_speed(T_REL + T_TURN)) == 1.0,
       f"hub {g_hub:.1e}, quarter-turn {g_q:.1e}, cusp 1.0")

    pts = np.stack([ST_X[ARCH1], ST_Y[ARCH1]], 1)
    landed = flip_xform(pts, T_FLIP1 + 1.0)
    ph = OM * T_ST[ARCH1]
    bx = PIV1[0] + np.pi * R_PX - R_PX * (ph - np.sin(ph))
    by = Y_RIM_BOWL + R_PX * (1 - np.cos(ph))
    ferr = max(np.max(np.abs(landed[:, 0] - bx)),
               np.max(np.abs(landed[:, 1] - by)))
    ck("flip lands the arch on the bowl curve exactly", ferr < 1e-9,
       f"max err {ferr:.1e} px")

    b = {"x0": 1e9, "x1": -1e9, "y0": 1e9, "y1": -1e9}
    for i in range(FLIP + 1):
        q = flip_xform(pts, T_FLIP0 + i / FPS)
        b["x0"] = min(b["x0"], q[:, 0].min())
        b["x1"] = max(b["x1"], q[:, 0].max())
        b["y0"] = min(b["y0"], q[:, 1].min())
        b["y1"] = max(b["y1"], q[:, 1].max())
    ck("flip framed on EVERY frame of the move (trap 37)",
       b["x0"] > 4 and b["x1"] < W - 4 and b["y0"] > 192 and
       b["y1"] < LBL_Y - 10,
       f"x {b['x0']:.0f}..{b['x1']:.0f} y {b['y0']:.0f}..{b['y1']:.0f}")

    ph_all = OM * T_ST
    rx = X0 + R_PX * (ph_all - np.sin(ph_all))
    ry = ROAD - R_PX * (1 - np.cos(ph_all))
    serr = max(np.max(np.abs(rx - ST_X)), np.max(np.abs(ry - ST_Y)))
    step = np.max(np.hypot(np.diff(ST_X), np.diff(ST_Y)))
    ck("trail stamps on the analytic cycloid, always overlapping",
       serr < 1e-9 and step < 2 * R_S,
       f"residual {serr:.1e}, max step {step:.2f} < {2*R_S:.1f}")

    a_peak = 1 - (1 - A_S) ** (2 * R_S / (2 * V_C * DT_S))
    ck("ink model leaves pools at the cusps", a_peak < 0.55,
       f"alpha at peak speed {a_peak:.3f}, at cusp -> 1")

    ck("timeline: 258 frames = 8.6 s",
       N_FRAMES == 258 and N_FRAMES / FPS <= 180)

    # -- pixels (rendered frames)
    f_hold = frame_at(int((T_END + 0.6) * FPS))          # trail complete
    ink_frac = float((f_hold.astype(int).sum(2) < 3 * 180).mean())
    ck("ink fraction sane on the hold frame (trap 56)",
       0.005 < ink_frac < 0.30, f"{ink_frac:.3f}")

    # every red pixel in the trail band lies on the analytic curve
    band = f_hold[int(ROAD - 2 * R_PX - 12):int(ROAD + 10),
                  38:1042, :].astype(np.float64)
    m = red_mask(band) > 0
    ph_d = np.linspace(0, 4 * np.pi, 4000)
    cpts = list(zip(X0 + R_PX * (ph_d - np.sin(ph_d)) - 38,
                    ROAD - R_PX * (1 - np.cos(ph_d)) -
                    (ROAD - 2 * R_PX - 12)))
    x0, y0, cv = polyseg_cov(cpts, 2 * (R_S + 2.5))
    covfull = np.zeros(m.shape)
    x1c, y1c = min(x0 + cv.shape[1], m.shape[1]), \
        min(y0 + cv.shape[0], m.shape[0])
    covfull[max(y0, 0):y1c, max(x0, 0):x1c] = \
        cv[max(-y0, 0):y1c - y0, max(-x0, 0):x1c - x0]
    pxr, pyr = pen_px(T_END + 0.6)
    keep = np.ones(m.shape, bool)          # exclude the pen's own disc
    yy, xx = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    keep &= np.hypot(xx + 38 - pxr, yy + (ROAD - 2 * R_PX - 12) - pyr) \
        > R_PEN + 6
    stray = int((m & keep & (covfull < 0.01)).sum())
    ck("every trail pixel sits on the analytic cycloid", stray == 0,
       f"{stray} stray red px")

    a_cusp = centerline_alpha(f_hold, 2 * np.pi - 0.10, 2 * np.pi + 0.10)
    a_pk = centerline_alpha(f_hold, np.pi - 0.25, np.pi + 0.25)
    ck("the ink pools where the pen was slow",
       a_cusp > 0.85 and 0.18 < a_pk < 0.60 and a_cusp / a_pk > 1.8,
       f"cusp {a_cusp:.3f} vs arch top {a_pk:.3f} "
       f"({a_cusp/a_pk:.1f}x)")

    probes = [(2 * np.pi - 1.6, 2 * np.pi - 1.4),
              (2 * np.pi - 0.8, 2 * np.pi - 0.6),
              (2 * np.pi - 0.4, 2 * np.pi - 0.3),
              (2 * np.pi - 0.2, 2 * np.pi - 0.1)]
    vals = [centerline_alpha(f_hold, a, bnd) for a, bnd in probes]
    ck("ink darkens monotonically as the pen slows",
       all(vals[i] < vals[i + 1] + 0.02 for i in range(3)),
       " -> ".join(f"{v:.2f}" for v in vals))

    # pen streak: same frame twice, one variable (trap 25) — bare pen
    zero = np.zeros((H, W), np.float64)

    def pen_only(t, ns):
        acc = np.zeros((H, W, 3), np.float64)
        for j in range(ns):
            off = (2 * j + 1 - ns) / (2.0 * ns) * SH_RT
            img = np.full((H, W, 3), PAPER, np.float64)
            px, py = pen_px(t + off)
            x0, y0, cv = disc_cov(px, py, R_PEN)
            comp_bbox(img, x0, y0, cv, C_RED)
            acc += img
        return (np.clip(acc / ns, 0, 1) * 255 + 0.5).astype(np.uint8)

    t_pk = T_REL + T_TURN / 2
    t_cu = T_REL + T_TURN
    e_pk = red_extent_x(pen_only(t_pk, NS), *pen_px(t_pk))
    e_cu = red_extent_x(pen_only(t_cu, NS), *pen_px(t_cu))
    streak = 2 * V_C * SH_RT
    # tint>0.15 extent: 2*(R_PEN + L/2 - 0.15 L) streaked, 2*R_PEN sharp
    p_pk = 2 * (R_PEN + 0.35 * streak)
    ck("motion blur: pen streaked at the top, sharp at the cusp",
       abs(e_pk - p_pk) < 4 and abs(e_cu - 2 * R_PEN) < 4 and
       e_pk - e_cu > 3,
       f"top {e_pk} px (pred {p_pk:.1f}), cusp {e_cu} px "
       f"(pred {2*R_PEN:.0f})")

    # lights off the actual frames. The pen annulus legitimately holds
    # rim-ring, one spoke tip and fresh trail ink, so the honest read is
    # lit-vs-unlit CONTRAST at the same annulus, not an absolute.
    f_cusp = frame_at(int(t_cu * FPS))
    i_q = int((T_REL + T_TURN * 0.75) * FPS)
    f_q = frame_at(i_q)
    px1, py1 = pen_px(int(t_cu * FPS) / FPS)
    r_mid = (RING_PEN + R_PEN + 1) / 2
    lit = annulus_mean(f_cusp, px1, py1, r_mid, 4.0)
    pxq, pyq = pen_px(i_q / FPS)
    unlit = annulus_mean(f_q, pxq, pyq, r_mid, 4.0)
    ck("pen light fires at the cusp, dark elsewhere",
       lit < 110 and unlit - lit > 80,
       f"annulus {lit:.0f} at cusp vs {unlit:.0f} at 3/4 turn")

    hub_vals = []
    for tf in np.linspace(T_REL + 0.05, T_END - 0.05, 9):
        i_f = int(round(tf * FPS))
        fr = frame_at(i_f)
        cx, cy = hub_px(i_f / FPS)
        hub_vals.append(disc_mean(fr, cx, cy, 7.0))
    ck("hub light NEVER fires across the roll (the control, trap 59)",
       min(hub_vals) > 195,
       f"min interior {min(hub_vals):.0f} (unlit paper ~{PAPER*255:.0f})")

    # nothing red below the road region before the flip
    low = f_hold[820:, :, :].astype(np.float64)
    ck("no red below the road before the flip (trap 58 region bound)",
       int((red_mask(low) > 0).sum()) == 0)

    # final frame: the arch sits ON the ghost bowl
    f_end = frame_at(N_FRAMES - 1)
    lowreg = f_end[1100:1700, :, :].astype(np.float64)
    mlow = red_mask(lowreg) > 0
    cols = np.where(mlow.any(0))[0]
    bc = bowl_curve()
    cpts2 = [(p[0], p[1] - 1100) for p in resample(bc, 3.0)]
    x0, y0, cv = polyseg_cov(cpts2, 2 * (R_S + 2.5))
    covb = np.zeros(mlow.shape)
    x1c, y1c = min(x0 + cv.shape[1], mlow.shape[1]), \
        min(y0 + cv.shape[0], mlow.shape[0])
    covb[max(y0, 0):y1c, max(x0, 0):x1c] = \
        cv[max(-y0, 0):y1c - y0, max(-x0, 0):x1c - x0]
    stray2 = int((mlow & (covb < 0.01)).sum())
    ck("final frame: the arch sits exactly on the ghost bowl",
       stray2 == 0 and abs(cols.min() - (PIV1[0] - np.pi * R_PX)) < 6
       and abs(cols.max() - (PIV1[0] + np.pi * R_PX)) < 6,
       f"red span {cols.min()}..{cols.max()} "
       f"(bowl {PIV1[0]-np.pi*R_PX:.0f}..{PIV1[0]+np.pi*R_PX:.0f}), "
       f"{stray2} stray px")

    lbl = f_end[LBL_Y:LBL_Y + 42, :, :]
    ck("label present and inside the text-safe area",
       (lbl.astype(int).sum(2) < 3 * 120).sum() > 200 and
       LBL_Y + 42 < int(0.85 * H))

    ck("holds are truly static (byte-equal frames)",
       np.array_equal(frame_at(2), frame_at(20)) and
       np.array_equal(frame_at(N_FRAMES - 8), frame_at(N_FRAMES - 1)))

    print(f"ALL {ok} CHECKS PASSED")
    print("NOT verified by any check above (trap 68):")
    print("  - a real taped pen has width; its tip rides a curtate/"
          "prolate trochoid unless it sits exactly ON the rim radius")
    print("  - rolling without slipping is assumed exact (no skid)")
    print("  - the flip is presentation, not physics: a rigid half-turn"
          " in the plane, drawn sharp on purpose")
    print("  - 'stops dead' is instantaneous: zero velocity for zero"
          " time; the light fires because speed is LOW nearby, the ink"
          " pools because dwell scales as 1/speed")


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
    return np.frombuffer(raw, np.uint8).reshape(H, W, 3).copy()


def check_encode():
    print("ENCODE CHECK — measured off the shipped h264:")
    d = decode_frame(N_FRAMES - 1)
    lowreg = d[1100:1700, :, :].astype(np.float64)
    mlow = red_mask(lowreg) > 0
    cols = np.where(mlow.any(0))[0]
    assert abs(cols.min() - (PIV1[0] - np.pi * R_PX)) < 8
    assert abs(cols.max() - (PIV1[0] + np.pi * R_PX)) < 8
    print(f"    final frame: seated arch spans cols "
          f"{cols.min()}..{cols.max()} (bowl "
          f"{PIV1[0]-np.pi*R_PX:.0f}..{PIV1[0]+np.pi*R_PX:.0f})")
    d_hold = decode_frame(int((T_END + 0.6) * FPS))
    a_cusp = centerline_alpha(d_hold, 2 * np.pi - 0.10, 2 * np.pi + 0.10)
    a_pk = centerline_alpha(d_hold, np.pi - 0.25, np.pi + 0.25)
    assert a_cusp / a_pk > 1.5
    print(f"    ink pools survive the encode: cusp {a_cusp:.2f} vs "
          f"arch top {a_pk:.2f} ({a_cusp/a_pk:.1f}x)")
    dd = np.abs(decode_frame(0).astype(int) - frame_at(0).astype(int))
    print(f"    decoded f0 vs render: mean |diff| {dd.mean():.3f}, "
          f"max {dd.max()}")
    assert dd.mean() < 2.0
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams",
         "v", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
         OUT], capture_output=True, text=True).stdout.strip()
    assert int(probe) == N_FRAMES, f"frames {probe} != {N_FRAMES}"
    print(f"    {probe} frames; the cusps survive the encode")


def review_stills():
    base = OUT[:-4]
    picks = [("release", PRE), ("cusp", int((T_REL + T_TURN) * FPS)),
             ("trail", int((T_END + 0.6) * FPS)),
             ("flip", int((T_FLIP0 + 0.8) * FPS)),
             ("final", N_FRAMES - 1)]
    for name, n in picks:
        fr = frame_at(n)
        p = f"{base}_{name}.png"
        subprocess.run(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt",
                        "rgb24", "-s", f"{W}x{H}", "-i", "-", "-vf",
                        "scale=360:-1", p],
                       input=fr.tobytes(), stderr=subprocess.DEVNULL)
        print(f"    still: {p}")


if __name__ == "__main__":
    run_checks()
    encode()
    check_encode()
    review_stills()
    print("DONE")
