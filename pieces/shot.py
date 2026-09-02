#!/usr/bin/env python3
"""SHOT — the monkey and the hunter, run both ways.

A dart launcher is aimed dead at a hanging can — the dashed line runs
straight through its center. Pass 1: the can holds still and the
perfectly aimed dart misses UNDER it, by exactly half g t squared; the
miss is drawn as a red bar hanging from the can. Pass 2: the hang line
is cut the instant the dart fires. The can falls straight down the miss
bar — because the bar's length is what gravity steals from everything,
every time — and the dart, still climbing, meets it at the bar's lower
end. Strobe ghosts pin the coupling: the dart's sag below the aim line
equals the can's drop at every instant (bars in 1:4:9). The classic
lecture demonstration of the independence of projectile components
(documented at least to Sutton, 1938). Units are pixels; every checked
claim is scale-invariant. Silent.

Seventh ordinary-world classic. Not a loop, has on-canvas words
(COUPLED's loop + wordless experiments stay single-instance).
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
STRIPE = 0.62                   # can stripes

# ---------------------------------------------------------------- model
M = np.array([150.0, 1480.0])   # muzzle
C0 = np.array([930.0, 480.0])   # can center at the hold
T_FL = 1.2                      # flight time, s (36 frames)
SAG = 170.0                     # 0.5 g T_FL^2
G = 2 * SAG / T_FL ** 2         # 236.11 px/s^2 (+y is DOWN, trap 1)
D = C0 - M
L = float(np.hypot(*D))
U = D / L                       # unit aim direction
V = L / T_FL                    # launch speed: reaches the can line at T_FL

CAN_W, CAN_H = 84.0, 104.0
R_DART = 15.0
LW_BAR = 7.0
FLOOR = 1600.0
LBL_Y = 1470                    # clear of the floor line, the aim line,
                                # the barrel and both trajectories
IMPACT = np.array([C0[0], C0[1] + SAG])

# timeline (frames)
PRE = 30
FL = 36                         # pass-1 flight (t = 0..1.2)
CONT = 10                       # dart continues off frame
HOLD1 = 32                      # the miss, held
PRE2 = 24                       # reset: dart back, faint history
FL2 = 36                        # pass-2 flight, can falling
FLASH = 8                       # impact ring
POST = 58
N_FRAMES = PRE + FL + CONT + HOLD1 + PRE2 + FL2 + FLASH + POST  # 234
I_F1 = PRE                      # fire 1
I_MISS = PRE + FL               # 66: dart crosses the can column
I_H1 = I_MISS + CONT            # 76: hold starts
I_F2 = I_H1 + HOLD1 + PRE2      # 132: fire 2 (line cut)
I_HIT = I_F2 + FL2              # 168: impact
I_POST = I_HIT + FLASH          # 176

SH_RT = 1.0 / 60.0
NS = 16                         # 17.6 px peak streak -> 1.1 px/sample

STROBE_T = (0.4, 0.8)           # bars 18.9 / 75.6 px (1:4; miss bar = 9)
W_OLD = 0.35                    # pass-1 history weight after the reset

OUT = f"out/shot_{time.strftime('%H%M%S')}.mp4"


def dart_pos(t):
    t = float(np.clip(t, 0.0, T_FL + CONT / FPS + SH_RT))
    return M + V * t * U + np.array([0.0, 0.5 * G * t * t])


def can_pos(t2):
    t2 = float(np.clip(t2, 0.0, T_FL))
    return C0 + np.array([0.0, 0.5 * G * t2 * t2])


def nograv(t):
    return M + V * t * U


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


def rect_cov(cx, cy, w, h):
    x0, x1 = int(np.floor(cx - w / 2)) - 2, int(np.ceil(cx + w / 2)) + 3
    y0, y1 = int(np.floor(cy - h / 2)) - 2, int(np.ceil(cy + h / 2)) + 3
    xx = np.arange(x0, x1, dtype=np.float64)
    yy = np.arange(y0, y1, dtype=np.float64)
    cvx = np.clip(np.minimum(xx - (cx - w / 2), (cx + w / 2) - xx) + 0.5,
                  0.0, 1.0)
    cvy = np.clip(np.minimum(yy - (cy - h / 2), (cy + h / 2) - yy) + 0.5,
                  0.0, 1.0)
    return x0, y0, cvy[:, None] * cvx[None, :]


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
    "m": "00000 00000 11010 10101 10101 10101 10101",
    "i": "00100 00000 01100 00100 00100 00100 01110",
    "s": "00000 00000 01111 10000 01110 00001 11110",
    "h": "10000 10000 10110 11001 10001 10001 10001",
    "t": "01000 01000 11100 01000 01000 01001 00110",
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
    col = np.array([color] * 3, np.float64)
    reg = img[ytop:ytop + h, x0:x0 + w, :]
    reg[...] = reg * (1 - m[..., None]) + col[None, None, :] * m[..., None]


# ---------------------------------------------------------------- static
def background():
    fr = np.full((H, W, 3), PAPER, np.float64)
    # floor
    x0, y0, cv = polyseg_cov([(28.0, FLOOR), (W - 28.0, FLOOR)], 4.0)
    comp_bbox(fr, x0, y0, cv, INK)
    # aim line: dashed ghost from muzzle dead through the can center
    n = int(L // 2)
    for k in range(n):
        if (k // 7) % 2 == 0:
            p = M + U * (k * 2.0)
            x0, y0, cv = disc_cov(p[0], p[1], 1.8)
            comp_bbox(fr, x0, y0, cv, GHOST)
    # launcher: barrel along the aim line + pedestal to the floor
    x0, y0, cv = polyseg_cov([tuple(M - 95 * U), tuple(M)], 18.0)
    comp_bbox(fr, x0, y0, cv, INK)
    x0, y0, cv = polyseg_cov([(95.0, 1552.0), (95.0, FLOOR)], 8.0)
    comp_bbox(fr, x0, y0, cv, INK)
    return fr


BG = background()

# ---------------------------------------------------------------- trail
DT_S = 1 / 240.0
R_S = 3.0
A_S = 0.20

TS1 = np.arange(0.0, T_FL + CONT / FPS + DT_S / 2, DT_S)
P1 = np.array([M + V * t * U + [0, 0.5 * G * t * t] for t in TS1])
KEEP1 = P1[:, 0] < W + R_S + 2
TS2 = np.arange(0.0, T_FL + DT_S / 2, DT_S)
P2 = np.array([M + V * t * U + [0, 0.5 * G * t * t] for t in TS2])


def build_trail(P, n):
    buf = np.zeros((H, W), np.float64)
    for k in range(n):
        x0, y0, cv = disc_cov(P[k, 0], P[k, 1], R_S)
        alpha_bbox(buf, x0, y0, cv, A_S)
    return buf


class TrailInc:
    def __init__(self, P):
        self.P = P
        self.buf = np.zeros((H, W), np.float64)
        self.n = 0

    def upto(self, n):
        n = int(np.clip(n, 0, len(self.P)))
        if n < self.n:
            self.buf[:] = 0.0
            self.n = 0
        for k in range(self.n, n):
            x0, y0, cv = disc_cov(self.P[k, 0], self.P[k, 1], R_S)
            alpha_bbox(self.buf, x0, y0, cv, A_S)
        self.n = n
        return self.buf


TRAIL1_FULL = None


def n_of(t, TS):
    return int(np.clip(np.searchsorted(TS, t + 1e-9), 0, len(TS)))


def compose_alpha(img, a, color=C_RED, w=1.0):
    aw = a * w
    img[...] = img * (1 - aw[..., None]) + \
        np.array(color)[None, None, :] * aw[..., None]


# ---------------------------------------------------------------- parts
def draw_can(img, cx, cy):
    x0, y0, cv = rect_cov(cx, cy, CAN_W, CAN_H)
    comp_bbox(img, x0, y0, cv, INK)
    for dy in (-20.0, 20.0):
        x0, y0, cv = rect_cov(cx, cy + dy, CAN_W - 14, 7.0)
        comp_bbox(img, x0, y0, cv, STRIPE)


def draw_can_ghost(img, cx, cy):
    hw, hh = CAN_W / 2, CAN_H / 2
    corners = [(cx - hw, cy - hh), (cx + hw, cy - hh),
               (cx + hw, cy + hh), (cx - hw, cy + hh), (cx - hw, cy - hh)]
    x0, y0, cv = polyseg_cov(corners, 3.0)
    comp_bbox(img, x0, y0, cv, GHOST)


def draw_dart(img, cx, cy):
    x0, y0, cv = disc_cov(cx, cy, R_DART)
    comp_bbox(img, x0, y0, cv, C_RED)


def draw_bar(img, x, y_top, y_bot, w=1.0):
    x0, y0, cv = polyseg_cov([(x, y_top), (x, y_bot)], LW_BAR)
    comp_bbox(img, x0, y0, cv * w, C_RED)


def draw_dart_ghost(img, cx, cy, w=1.0):
    x0, y0, cv = ring_cov(cx, cy, R_DART, 3.0)
    comp_bbox(img, x0, y0, cv * 0.45 * w, C_RED)


def hang_line(img):
    x0, y0, cv = polyseg_cov([(C0[0], 6.0), (C0[0], C0[1] - CAN_H / 2)],
                             4.0)
    comp_bbox(img, x0, y0, cv, INK)


def strobes(img, upto_t, w=1.0, skip_leq=None):
    """Dart ghosts + drop bars for every strobe time <= upto_t.
    skip_leq: in pass 2, the faint pass-1 set must NOT be drawn under a
    strobe pass 2 has already re-stamped — the positions are identical
    (that is the point), and overprinting stacks the ghost ring's alpha
    past the strict-red cut, which stretched a measured bar by 11 px."""
    for ts in STROBE_T:
        if upto_t + 1e-9 >= ts and \
                (skip_leq is None or ts > skip_leq + 1e-9):
            ng, dp = nograv(ts), dart_pos(ts)
            draw_bar(img, dp[0], ng[1], dp[1], w)
            draw_dart_ghost(img, dp[0], dp[1], w)


def flash_ring(img, cx, cy, u, r0, dr, lw):
    x0, y0, cv = ring_cov(cx, cy, r0 + dr * u, lw)
    comp_bbox(img, x0, y0, cv * (1 - u), INK)


# ---------------------------------------------------------------- frames
def static_layer(i, inc1=None, inc2=None):
    """Everything except the dart and (in pass 2) the falling can."""
    global TRAIL1_FULL
    img = BG.copy()
    t1 = (i - I_F1) / FPS
    t2 = (i - I_F2) / FPS
    in2 = i >= I_F2 - PRE2          # reset happened

    # trails
    if not in2:
        n1 = n_of(t1, TS1)
        a1 = inc1.upto(n1) if inc1 is not None else build_trail(P1, n1)
        compose_alpha(img, a1)
    else:
        if TRAIL1_FULL is None:
            TRAIL1_FULL = build_trail(P1, len(TS1))
        compose_alpha(img, TRAIL1_FULL, w=W_OLD)
        n2 = n_of(t2, TS2)
        a2 = inc2.upto(n2) if inc2 is not None else build_trail(P2, n2)
        compose_alpha(img, a2)

    # miss bar: hangs from the can center to the crossing point.
    # Full strength from the moment of the miss, forever — in pass 2 it
    # is the rail the can falls down.
    if i >= I_MISS:
        draw_bar(img, C0[0], C0[1], IMPACT[1])
        draw_dart_ghost(img, IMPACT[0], IMPACT[1],
                        1.0 if not in2 else W_OLD)

    # strobe ghosts
    if not in2:
        strobes(img, t1, 1.0)
    else:
        strobes(img, T_FL, W_OLD, skip_leq=t2)   # pass-1 set, faint,
        strobes(img, t2, 1.0)               # replaced as pass 2 arrives
        if i >= I_F2:
            draw_can_ghost(img, C0[0], C0[1])
            for ts in STROBE_T:
                if t2 + 1e-9 >= ts:
                    cp = can_pos(ts)
                    draw_can_ghost(img, cp[0], cp[1])

    # the can, when it is static; the hang line while it holds
    if i < I_F2:
        hang_line(img)
        draw_can(img, C0[0], C0[1])
    elif i >= I_HIT:
        draw_can(img, IMPACT[0], IMPACT[1])

    return img


def moving_layer(img, i, t_off):
    """Dart (both flights) and the falling can (pass 2), at frame time
    plus a shutter offset."""
    if I_F1 <= i < I_F1 + FL + CONT:
        t = (i - I_F1) / FPS + t_off
        p = dart_pos(t)
        if p[0] < W + R_DART + 2:
            draw_dart(img, p[0], p[1])
    elif I_F2 <= i < I_HIT:
        t = (i - I_F2) / FPS + t_off
        cp = can_pos(t)
        draw_can(img, cp[0], cp[1])
        p = dart_pos(min(t, T_FL))
        draw_dart(img, p[0], p[1])


def sharp_layer(img, i):
    # dart parked at the muzzle before each fire; at the can after impact
    if i < I_F1 or I_H1 + HOLD1 <= i < I_F2:
        draw_dart(img, M[0], M[1])
    if i >= I_HIT:
        draw_dart(img, IMPACT[0], IMPACT[1])
    # muzzle flash on each fire
    for fire in (I_F1, I_F2):
        if fire <= i < fire + 4:
            flash_ring(img, M[0], M[1], (i - fire) / 4.0, 14, 55, 4.0)
    # impact ring
    if I_HIT <= i < I_HIT + FLASH:
        flash_ring(img, IMPACT[0], IMPACT[1], (i - I_HIT) / FLASH,
                   18, 70, 5.0)
    # labels
    if I_H1 <= i < I_F2 - PRE2:
        fade = float(np.clip((i - I_H1) / 12.0, 0.0, 1.0))
        stamp_center(img, "miss", W / 2, LBL_Y, INK, fade)
    if i >= I_POST:
        fade = float(np.clip((i - I_POST) / 12.0, 0.0, 1.0))
        stamp_center(img, "hit", W / 2, LBL_Y, INK, fade)


def frame_at(i, inc1=None, inc2=None):
    base = static_layer(i, inc1, inc2)
    moving = (I_F1 <= i < I_F1 + FL + CONT) or (I_F2 <= i < I_HIT)
    if moving:
        acc = np.zeros((H, W, 3), np.float64)
        for j in range(NS):
            off = (2 * j + 1 - NS) / (2.0 * NS) * SH_RT
            img = base.copy()
            moving_layer(img, i, off)
            acc += img
        img = acc / NS
    else:
        img = base
    sharp_layer(img, i)
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def render_frames():
    inc1, inc2 = TrailInc(P1), TrailInc(P2)
    for i in range(N_FRAMES):
        yield frame_at(i, inc1, inc2)


# ---------------------------------------------------------------- measure
def red_strict(reg):
    """Strict red mask: only alpha > 0.52 red survives (dart disc, bars).
    Trail (max alpha 0.39) and ghost rings (0.45) are invisible to it."""
    return (np.clip(reg[:, :, 0] - reg[:, :, 1] - 60, 0, None) *
            (reg[:, :, 2] - reg[:, :, 1] < 40)) > 0


def tint_alpha(reg):
    return (PAPER - reg[:, :, 1] / 255.0) / (PAPER - C_RED[1])


def ink_mask(reg):
    return reg.max(2) < 100


def centroid(mask, x_off=0, y_off=0):
    ys, xs = np.nonzero(mask)
    assert len(xs) > 0, "empty mask"
    return xs.mean() + x_off, ys.mean() + y_off


def red_centroid_in(img, cx, cy, half=26):
    x0, y0 = int(cx) - half, int(cy) - half
    reg = img[y0:y0 + 2 * half, x0:x0 + 2 * half, :].astype(np.float64)
    return centroid(red_strict(reg), x0, y0)


def bar_extent(img, x, y_lo, y_hi):
    """Vertical extent of strict red in a bounded column (trap 58: rows
    AND cols fenced; excludes trail/ghosts by the alpha cut)."""
    reg = img[y_lo:y_hi, int(x) - 4:int(x) + 5, :].astype(np.float64)
    rows = np.where(red_strict(reg).any(1))[0]
    return (rows.max() - rows.min() + 1) if len(rows) else 0


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
    err = np.abs(M + V * T_FL * U + [0, 0.5 * G * T_FL ** 2] -
                 (C0 + [0, 0.5 * G * T_FL ** 2])).max()
    ck("impact identity from two independent formulas", err < 1e-9,
       f"|ballistic - free-fall| = {err:.1e} px")

    miss = dart_pos(T_FL) - C0
    ck("control law: held can is missed by exactly 0.5 g T^2 (trap 59)",
       abs(miss[0]) < 1e-9 and abs(miss[1] - SAG) < 1e-9,
       f"miss ({miss[0]:.1e}, {miss[1]:.4f}) px, SAG {SAG}")

    ts = np.linspace(0, T_FL, 481)
    dcup = np.max([abs((dart_pos(t)[1] - nograv(t)[1]) -
                       (can_pos(t)[1] - C0[1])) for t in ts])
    ck("coupling: dart sag below the line == can drop, all t (trap 66)",
       dcup < 1e-9, f"max diff {dcup:.1e} px")

    ck("aim line arithmetic: v T = |can - muzzle| exactly",
       abs(V * T_FL - L) < 1e-9 and abs(np.hypot(*U) - 1) < 1e-12)

    bars = np.array([0.5 * G * t * t for t in (*STROBE_T, T_FL)])
    ck("strobe bars are 1 : 4 : 9 exactly",
       np.allclose(bars / bars[0], [1, 4, 9], atol=1e-12),
       f"{bars.round(1)} px")

    vy = V * U[1] + G * T_FL
    ck("the dart is still climbing when it hits", vy < 0,
       f"vy at impact {vy:.1f} px/s")

    # framing over every frame incl. shutter spill (trap 37 habit)
    lo = np.array([1e9, 1e9])
    hi = -lo.copy()
    for i in range(I_F1, I_F1 + FL):
        for off in (-SH_RT / 2, 0, SH_RT / 2):
            p = dart_pos((i - I_F1) / FPS + off)
            lo, hi = np.minimum(lo, p), np.maximum(hi, p)
    exit_x = dart_pos((I_F1 + FL + CONT - 1 - I_F1) / FPS)[0]
    ck("flight framed; pass-1 dart fully exits during CONT",
       lo[0] > 40 and hi[0] < W - 40 and lo[1] > 220 and hi[1] < H - 260
       and exit_x > W + R_DART,
       f"box x {lo[0]:.0f}..{hi[0]:.0f} y {lo[1]:.0f}..{hi[1]:.0f}, "
       f"exit x {exit_x:.0f}")

    ck("timeline: 234 frames = 7.8 s",
       N_FRAMES == 234 and N_FRAMES / FPS <= 180)

    # -- pixels
    f_end = frame_at(N_FRAMES - 1)
    ink_frac = float((f_end.astype(int).sum(2) < 3 * 180).mean())
    ck("ink fraction sane on the final frame (trap 56)",
       0.004 < ink_frac < 0.30, f"{ink_frac:.3f}")

    for i_t in (40, 52, 64):
        t = (i_t - I_F1) / FPS
        p = dart_pos(t)
        fx, fy = red_centroid_in(frame_at(i_t), p[0], p[1])
        derr = np.hypot(fx - p[0], fy - p[1])
        assert derr < 2.5, f"dart centroid off {derr:.1f} px at f{i_t}"
    ck("dart rides the model parabola (centred exposure, 3 frames)",
       True, f"last err {derr:.2f} px")

    f_hold = frame_at(I_H1 + 20)
    e_miss = bar_extent(f_hold, C0[0], int(C0[1]), int(IMPACT[1]) + 20)
    p_miss = (IMPACT[1] + LW_BAR / 2) - (C0[1] + CAN_H / 2)
    ck("the miss bar emerges from under the can, length as drawn",
       abs(e_miss - p_miss) < 5,
       f"{e_miss} px visible (model {p_miss:.0f}; full bar {SAG:.0f})")

    e1 = bar_extent(f_hold, nograv(0.4)[0], 1100, 1220)
    e2 = bar_extent(f_hold, nograv(0.8)[0], 790, 940)
    r = (e2 - LW_BAR - 1) / (e1 - LW_BAR - 1)
    ck("strobe bars measure 1 : 4 off the pixels",
       abs(e1 - (bars[0] + LW_BAR + 1)) < 4 and
       abs(e2 - (bars[1] + LW_BAR + 1)) < 4 and abs(r - 4) < 0.8,
       f"{e1} px and {e2} px -> ratio {r:.2f}")

    # gravity only ever pulls DOWN off the line: topmost RED tint never
    # rises above the aim line (trap 1 as a pixel fact). A red-only mask
    # (R-G channel), because tint off the green channel alone would read
    # the ink can and labels as 'red' (trap 61's lesson: say WHAT it is).
    top_ok = True
    for x in range(250, 871, 25):
        tx = (x - M[0]) / (V * U[0])
        line_y = nograv(tx)[1]
        col = f_hold[:, x - 1:x + 2, :].astype(np.float64)
        red = (col[:, :, 0] - col[:, :, 1] > 17) & \
              (col[:, :, 2] - col[:, :, 1] < 40)
        rows = np.where(red.any(1))[0]
        if len(rows) and rows.min() < line_y - 8:
            top_ok = False
    ck("nothing red ever sits above the aim line", top_ok)

    # pass 2: the can's fall equals the dart's sag, off the pixels.
    # The can is read at the strobe frame; the BAR is read at the impact
    # frame, because at the strobe instant the dart sits at the bar's
    # lower end BY CONSTRUCTION (that is the coupling) and its disc
    # stretches the column's extent. The bar is the stamped record of
    # the dart's drop at t=0.8; the dart has left that column by I_HIT.
    f_s2 = frame_at(I_F2 + 24)          # t2 = 0.8
    f_hit = frame_at(I_HIT)
    cp = can_pos(0.8)
    cx, cy = centroid(ink_mask(
        f_s2[int(cp[1]) - 60:int(cp[1]) + 60,
             int(cp[0]) - 50:int(cp[0]) + 50, :].astype(np.float64)),
        int(cp[0]) - 50, int(cp[1]) - 60)
    can_drop = cy - C0[1]
    e2b = bar_extent(f_hit, nograv(0.8)[0], 790, 940)
    bar_len = e2b - LW_BAR - 1
    ck("COUPLING off the pixels: can drop == dart drop bar (trap 66)",
       abs(can_drop - bars[1]) < 4 and abs(bar_len - bars[1]) < 4,
       f"can {can_drop:.1f} px, bar {bar_len:.0f} px, model {bars[1]:.1f}")

    # impact: dart centroid == can centroid, one frame, no flash yet
    rx, ry = red_centroid_in(f_hit, IMPACT[0], IMPACT[1], 24)
    kx, ky = centroid(ink_mask(
        f_hit[int(IMPACT[1]) - 60:int(IMPACT[1]) + 60,
              int(IMPACT[0]) - 50:int(IMPACT[0]) + 50, :]
        .astype(np.float64)), int(IMPACT[0]) - 50, int(IMPACT[1]) - 60)
    herr = np.hypot(rx - kx, ry - ky)
    merr = np.hypot(rx - IMPACT[0], ry - IMPACT[1])
    ck("impact: dart and can centroids coincide on the frame",
       herr < 3.0 and merr < 3.0,
       f"dart-can {herr:.1f} px, dart-model {merr:.1f} px")

    # the hang line: present through pass 1, gone from the fire
    def line_ink(i):
        fr = frame_at(i)
        return int(ink_mask(fr[120:400, int(C0[0]) - 6:int(C0[0]) + 7, :]
                            .astype(np.float64)).sum())
    ck("the hang line holds all of pass 1 and is cut at fire 2",
       line_ink(I_MISS) > 500 and line_ink(I_F2 - 1) > 500 and
       line_ink(I_F2 + 1) == 0,
       f"ink {line_ink(I_F2 - 1)} before, {line_ink(I_F2 + 1)} after")

    # label region bounded in x too (trap 58): the barrel and the aim
    # line cross these rows outside 400..680
    lbl = f_end[LBL_Y:LBL_Y + 42, 400:680, :]
    ck("labels: 'hit' on the final frame, inside the text-safe area",
       (lbl.astype(int).sum(2) < 3 * 120).sum() > 150 and
       LBL_Y + 42 < int(0.85 * H))
    lbl_m = frame_at(I_H1 + 20)[LBL_Y:LBL_Y + 42, 400:680, :]
    ck("labels: 'miss' during the hold, gone before pass 2",
       (lbl_m.astype(int).sum(2) < 3 * 120).sum() > 150 and
       (frame_at(I_F2 - 4)[LBL_Y:LBL_Y + 42, 400:680, :].astype(int)
        .sum(2) < 3 * 120).sum() == 0)

    ck("holds are truly static (byte-equal frames)",
       np.array_equal(frame_at(I_H1 + 16), frame_at(I_H1 + HOLD1 - 2)) and
       np.array_equal(frame_at(I_POST + 16), frame_at(N_FRAMES - 1)) and
       np.array_equal(frame_at(2), frame_at(PRE - 2)))

    print(f"ALL {ok} CHECKS PASSED")
    print("NOT verified by any check above (trap 68):")
    print("  - units are pixels, g chosen for the frame; every claim"
          " checked (the coupling, the equal drops, the hit) is"
          " scale-invariant, but the speeds are not real-world")
    print("  - the freeze after impact is presentation: really both"
          " would keep falling together")
    print("  - drag, spin and the dart's length are ignored; a real"
          " dart is not a point")
    print("  - the release is instantaneous: a real electromagnet trips"
          " ~ms after the muzzle photogate")


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
    rx, ry = red_centroid_in(d, IMPACT[0], IMPACT[1], 24)
    err = np.hypot(rx - IMPACT[0], ry - IMPACT[1])
    assert err < 3.0, err
    print(f"    dart sits on the impact point to {err:.1f} px")
    e1 = bar_extent(d, nograv(0.4)[0], 1100, 1220)
    e2 = bar_extent(d, nograv(0.8)[0], 790, 940)
    r = (e2 - LW_BAR - 1) / (e1 - LW_BAR - 1)
    assert abs(r - 4) < 1.0, r
    print(f"    strobe bars survive the encode: {e1} / {e2} px "
          f"(ratio {r:.2f}, model 4)")
    dd = np.abs(decode_frame(0).astype(int) - frame_at(0).astype(int))
    print(f"    decoded f0 vs render: mean |diff| {dd.mean():.3f}, "
          f"max {dd.max()}")
    assert dd.mean() < 2.0
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams",
         "v", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
         OUT], capture_output=True, text=True).stdout.strip()
    assert int(probe) == N_FRAMES, f"frames {probe} != {N_FRAMES}"
    print(f"    {probe} frames; the equality survives the encode")


def review_stills():
    base = OUT[:-4]
    picks = [("aim", 10), ("flight1", 52), ("miss", I_H1 + 20),
             ("fall", I_F2 + 24), ("hit", I_HIT + 3),
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
