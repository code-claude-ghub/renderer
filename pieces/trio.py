#!/usr/bin/env python3
"""TRIO — three darts, three speeds, one falling can.

Sequel to SHOT (the monkey and the hunter). The can from the last video
is hung at the same spot, the launcher aimed dead at it. At the muzzle
flash the hang line is cut and THREE darts leave at once — fast, medium,
slow — all down the same aim line. Every one hits the falling can: the
fast one high (still climbing), the middle one lower (still climbing),
the slow lob near the bottom, on its way down. The impact points sit at
drops of exactly 1:4:9 (t = 0.5, 1.0, 1.5 s — Galileo's odd numbers),
ticked and numbered on the fall line. The deep fact, drawn as a ghost
line: gravity translates the WHOLE picture down by half g t squared —
at every instant the three darts and the can are collinear on the aim
line's falling copy, so in the falling frame there is no gravity at all
and three straight shots meet a stationary can. Muzzle speed cannot
matter. The classic demonstration's standard extension (vary the speed,
the hit survives — documented in lecture-demo catalogues). Units are
pixels; every checked claim is scale-invariant. Silent.

Eighth ordinary-world classic. Title points at the previous video —
the deliberate second instance of RIM's pointing-title experiment.
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
STRIPE = 0.62

# ---------------------------------------------------------------- model
M = np.array([150.0, 1480.0])   # muzzle (same scene as shot.py)
C0 = np.array([930.0, 480.0])   # can center at the hold
T_IMP = (0.5, 1.0, 1.5)         # impact times -> drops exactly 1:4:9
DROP3 = 700.0                   # slowest dart's drop at t = 1.5
G = 2 * DROP3 / T_IMP[2] ** 2   # 622.22 px/s^2 (+y is DOWN, trap 1)
D = C0 - M
L = float(np.hypot(*D))
U = D / L
V = [L / t for t in T_IMP]      # 2536.5 / 1268.2 / 845.5 px/s
DROPS = np.array([0.5 * G * t * t for t in T_IMP])   # 77.8/311.1/700
IMP_Y = C0[1] + DROPS

CAN_W, CAN_H = 84.0, 104.0
R_DART = 15.0
LW_BAR = 7.0
FLOOR = 1600.0
TICK_X0, TICK_X1 = 976.0, 1016.0    # right of the can (right edge 972)
DIGIT_CX = 1044

# timeline (frames)
PRE = 36
FL = 45                          # t = 0 .. 1.5
FLASH = 8
POST = 55
N_FRAMES = PRE + FL + FLASH + POST          # 144 = 4.8 s
I_F = PRE                                    # fire + line cut, f36
I_IMP = [I_F + int(round(t * FPS)) for t in T_IMP]   # f51 f66 f81
I_FRZ = I_IMP[2]                             # freeze at the third hit

SH_RT = 1.0 / 60.0
NS = 20                          # fast dart: 42 px streak, 2.1 px/sample

OUT = f"out/trio_{time.strftime('%H%M%S')}.mp4"


def dart(k, t):
    t = float(np.clip(t, 0.0, T_IMP[k]))
    return M + V[k] * t * U + np.array([0.0, 0.5 * G * t * t])


def can_pos(t):
    t = float(np.clip(t, 0.0, T_IMP[2]))
    return C0 + np.array([0.0, 0.5 * G * t * t])


def nograv(k, t):
    return M + V[k] * t * U


def line_y(x):
    """Static aim line's y at column x."""
    return M[1] + (x - M[0]) * U[1] / U[0]


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
    "1": "00100 01100 00100 00100 00100 00100 01110",
    "4": "00010 00110 01010 10010 11111 00010 00010",
    "9": "01110 10001 10001 01111 00001 00010 01100",
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

# ------------------------------------------------------------------ trail
# Equal-ARCLENGTH stamps (ds = 3 px), not equal-time: three trails of
# identical texture, and the single-trail plateau (~0.41) stays under
# the strict-red cut (0.523). The three paths coincide within a disc
# width for the first ~83 px from the muzzle, where the stack goes
# strict-visible — accepted, and every red measurement sits at x >= 250.
DS = 3.0
R_S = 3.0
A_S = 0.16


def build_stamps():
    stamps = []                      # (t, x, y) merged over all darts
    for k in range(3):
        tt = np.linspace(0.0, T_IMP[k], 4000)
        pp = np.array([M + V[k] * t * U + [0, 0.5 * G * t * t]
                       for t in tt])
        seg = np.hypot(*np.diff(pp, axis=0).T)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        s_pick = np.arange(0.0, s[-1], DS)
        t_pick = np.interp(s_pick, s, tt)
        for t in t_pick:
            p = M + V[k] * t * U + np.array([0, 0.5 * G * t * t])
            stamps.append((t, p[0], p[1]))
    stamps.sort(key=lambda z: z[0])
    return np.array(stamps)


STAMPS = build_stamps()


class TrailInc:
    def __init__(self):
        self.buf = np.zeros((H, W), np.float64)
        self.n = 0

    def upto(self, t):
        n = int(np.searchsorted(STAMPS[:, 0], t + 1e-9))
        if n < self.n:
            self.buf[:] = 0.0
            self.n = 0
        for k in range(self.n, n):
            _, x, y = STAMPS[k]
            x0, y0, cv = disc_cov(x, y, R_S)
            alpha_bbox(self.buf, x0, y0, cv, A_S)
        self.n = n
        return self.buf


def compose_alpha(img, a, color=C_RED):
    img[...] = img * (1 - a[..., None]) + \
        np.array(color)[None, None, :] * a[..., None]


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


def draw_dart(img, cx, cy, r=R_DART):
    x0, y0, cv = disc_cov(cx, cy, r)
    comp_bbox(img, x0, y0, cv, C_RED)


def draw_ring_ghost(img, cx, cy):
    x0, y0, cv = ring_cov(cx, cy, R_DART, 3.0)
    comp_bbox(img, x0, y0, cv * 0.45, C_RED)


def hang_line(img):
    x0, y0, cv = polyseg_cov([(C0[0], 6.0), (C0[0], C0[1] - CAN_H / 2)],
                             4.0)
    comp_bbox(img, x0, y0, cv, INK)


def draw_bar(img, y_top, y_bot):
    if y_bot - y_top < 2:
        return
    x0, y0, cv = polyseg_cov([(C0[0], y_top), (C0[0], y_bot)], LW_BAR)
    comp_bbox(img, x0, y0, cv, C_RED)


def falling_line(img, drop):
    """The aim line's rigid falling copy: solid thin ghost, clipped at
    the floor. Every live dart and the can ride this line — that is the
    piece's deep claim, checked to 1e-13 in the model and read off the
    pixels at the freeze."""
    ya = M[1] + drop
    yb = line_y(C0[0]) + drop            # at the can column
    xa, xb = M[0], C0[0]
    y_cut = FLOOR - 6.0
    if ya > y_cut:                        # clip the low (left) end
        if yb > y_cut:
            return
        f = (ya - y_cut) / (ya - yb)
        xa = xa + f * (xb - xa)
        ya = y_cut
    x0, y0, cv = polyseg_cov([(xa, ya), (xb, yb)], 2.5)
    comp_bbox(img, x0, y0, cv, GHOST)


def flash_ring(img, cx, cy, u, r0, dr, lw):
    x0, y0, cv = ring_cov(cx, cy, r0 + dr * u, lw)
    comp_bbox(img, x0, y0, cv * (1 - u), INK)


# ---------------------------------------------------------------- frames
def static_layer(i, inc=None):
    img = BG.copy()
    t = (i - I_F) / FPS

    # trails
    if i >= I_F:
        a = inc.upto(t) if inc is not None else TrailInc().upto(t)
        compose_alpha(img, a)

    # frozen scene extras
    if i >= I_FRZ:
        falling_line(img, DROPS[2])

    # the drop bar: from the hold point down to the can (the fall's
    # own ruler; the can covers its lower end). The hold ghost is drawn
    # FIRST — its bottom edge crosses the bar's column, and a neutral
    # stroke over the bar would gap the strict-red continuity read.
    if i >= I_F:
        draw_can_ghost(img, C0[0], C0[1])       # where it hung
        cp = can_pos(t)
        draw_bar(img, C0[1], cp[1])

    # impact ghost rings (permanent record) — delayed until each flash
    # ends: a ring lands on its own trail's tail end, and 0.45 over the
    # trail's 0.41 stacks past the strict cut (SHOT's trap-62b) right
    # inside the impact-coincidence boxes measured at I_IMP[k]
    for k in range(3):
        if i >= I_IMP[k] + FLASH:
            draw_ring_ghost(img, C0[0], IMP_Y[k])

    # the can, when static
    if i < I_F:
        hang_line(img)
        draw_can(img, C0[0], C0[1])
    elif i >= I_FRZ:
        draw_can(img, C0[0], IMP_Y[2])
        draw_dart(img, C0[0], IMP_Y[2])         # the stuck darts

    return img


def moving_layer(img, i, t_off):
    """Falling line, falling can, stuck + live darts, at frame time
    plus a shutter offset."""
    t = max((i - I_F) / FPS + t_off, 0.0)
    falling_line(img, 0.5 * G * min(t, T_IMP[2]) ** 2)
    cp = can_pos(t)
    draw_can(img, cp[0], cp[1])
    if t > T_IMP[0]:
        draw_dart(img, cp[0], cp[1])            # stuck darts ride the can
    for k in range(3):
        if t <= T_IMP[k]:
            p = dart(k, t)
            draw_dart(img, p[0], p[1])


def sharp_layer(img, i):
    # three loaded darts on the barrel before the fire
    if i < I_F:
        for kk in (28.0, 58.0, 88.0):
            p = M - kk * U
            draw_dart(img, p[0], p[1], r=11.0)
    # muzzle flash
    if I_F <= i < I_F + 4:
        flash_ring(img, M[0], M[1], (i - I_F) / 4.0, 14, 55, 4.0)
    # impact flashes, ticks and digits
    for k in range(3):
        if I_IMP[k] <= i < I_IMP[k] + FLASH:
            flash_ring(img, C0[0], IMP_Y[k], (i - I_IMP[k]) / FLASH,
                       18, 70, 5.0)
        if i >= I_IMP[k]:
            x0, y0, cv = polyseg_cov([(TICK_X0, IMP_Y[k]),
                                      (TICK_X1, IMP_Y[k])], 5.0)
            comp_bbox(img, x0, y0, cv, C_RED)
            fade = float(np.clip((i - I_IMP[k]) / 12.0, 0.0, 1.0))
            stamp_center(img, "149"[k], DIGIT_CX,
                         int(IMP_Y[k]) - 21, INK, fade)


def frame_at(i, inc=None):
    base = static_layer(i, inc)
    if I_F <= i < I_FRZ:
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
    inc = TrailInc()
    for i in range(N_FRAMES):
        yield frame_at(i, inc)


# ---------------------------------------------------------------- measure
def red_strict(reg):
    """Only alpha > 0.52 red survives: dart discs, the bar, the ticks.
    Trails (plateau ~0.41) and ghost rings (0.45) are invisible to it —
    except the 3-trail overlap zone within ~83 px of the muzzle, which
    every fence below excludes (x >= 250)."""
    return (np.clip(reg[:, :, 0] - reg[:, :, 1] - 60, 0, None) *
            (reg[:, :, 2] - reg[:, :, 1] < 40)) > 0


def red_tint(reg):
    return (reg[:, :, 0] - reg[:, :, 1] > 17) & \
           (reg[:, :, 2] - reg[:, :, 1] < 40)


def ink_mask(reg):
    return reg.max(2) < 100


def ghost_mask(reg):
    """Neutral mid-grey: the ghost lines. Excludes paper (0.845), ink
    (0.10), and anything red-tinted."""
    r = reg[:, :, 0].astype(np.float64)
    g = reg[:, :, 1].astype(np.float64)
    b = reg[:, :, 2].astype(np.float64)
    v = (r + g + b) / (3 * 255.0)
    return (np.abs(r - g) < 12) & (np.abs(b - g) < 12) & \
           (v > 0.40) & (v < 0.74)


def centroid(mask, x_off=0, y_off=0):
    ys, xs = np.nonzero(mask)
    assert len(xs) > 0, "empty mask"
    return xs.mean() + x_off, ys.mean() + y_off


def red_centroid_in(img, cx, cy, half=26):
    x0, y0 = int(cx) - half, int(cy) - half
    reg = img[y0:y0 + 2 * half, x0:x0 + 2 * half, :].astype(np.float64)
    return centroid(red_strict(reg), x0, y0)


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
    err = max(np.abs(M + V[k] * T_IMP[k] * U +
                     [0, 0.5 * G * T_IMP[k] ** 2] -
                     (C0 + [0, 0.5 * G * T_IMP[k] ** 2])).max()
              for k in range(3))
    ck("three impact identities, two independent formulas each",
       err < 1e-9, f"worst |ballistic - free-fall| = {err:.1e} px")

    err = max(np.abs(nograv(k, T_IMP[k]) - C0).max() for k in range(3))
    ck("no-gravity frame: every dart arrives exactly at the hold point",
       err < 1e-9, f"worst {err:.1e} px")

    ck("impact drops are 1 : 4 : 9 exactly (binary-exact t^2)",
       np.all(DROPS / DROPS[0] == np.array([1.0, 4.0, 9.0])),
       f"{DROPS.round(1)} px")

    worst = 0.0
    for t in np.linspace(0, T_IMP[2], 481):
        sag = 0.5 * G * t * t
        base = M + np.array([0.0, sag])
        for p in [dart(k, t) for k in range(3) if t <= T_IMP[k]] + \
                 [can_pos(t)]:
            rel = p - base
            worst = max(worst, abs(rel[0] * U[1] - rel[1] * U[0]))
    ck("THE RIGID LINE: live darts + can collinear on the falling aim "
       "line, all t (trap 66)", worst < 1e-9, f"max {worst:.1e} px")

    vys = [V[k] * U[1] + G * T_IMP[k] for k in range(3)]
    ck("two darts hit while climbing, the slow lob on its way down",
       vys[0] < 0 and vys[1] < 0 and vys[2] > 0,
       f"vy at impact {[round(v) for v in vys]} px/s")

    lo = np.array([1e9, 1e9])
    hi = -lo.copy()
    for k in range(3):
        for tq in np.linspace(0, T_IMP[k], 200):
            for off in (-SH_RT / 2, 0, SH_RT / 2):
                p = dart(k, min(max(tq + off, 0), T_IMP[k]))
                lo, hi = np.minimum(lo, p), np.maximum(hi, p)
    ck("every flight framed incl. shutter spill; can lands clear of "
       "the floor",
       lo[0] > 40 and hi[0] < W - 40 and lo[1] > 220 and hi[1] < H - 260
       and IMP_Y[2] + CAN_H / 2 < FLOOR - 40,
       f"box x {lo[0]:.0f}..{hi[0]:.0f} y {lo[1]:.0f}..{hi[1]:.0f}")

    ck("timeline: 144 frames = 4.8 s", N_FRAMES == 144
       and N_FRAMES / FPS <= 180)

    # -- pixels
    f_end = frame_at(N_FRAMES - 1)
    ink_frac = float((f_end.astype(int).sum(2) < 3 * 180).mean())
    ck("ink fraction sane on the final frame (trap 56)",
       0.004 < ink_frac < 0.30, f"{ink_frac:.3f}")

    # each dart rides its own parabola, at a frame where it is the only
    # red thing in its box (separations asserted in the model first)
    picks = [(0, 44), (1, 58), (2, 74)]
    worst_d = 0.0
    for k, fi in picks:
        t = (fi - I_F) / FPS
        p = dart(k, t)
        others = [dart(j, t) for j in range(3)
                  if j != k and t <= T_IMP[j]] + [can_pos(t)]
        assert min(np.hypot(*(p - q)) for q in others) > 70, (k, fi)
        fx, fy = red_centroid_in(frame_at(fi), p[0], p[1])
        worst_d = max(worst_d, np.hypot(fx - p[0], fy - p[1]))
    ck("each dart rides its model parabola (centred exposure)",
       worst_d < 2.5, f"worst err {worst_d:.2f} px")

    # impacts: dart centroid == can centroid == model, at all three
    worst_dc, worst_dm = 0.0, 0.0
    for k in range(3):
        fr = frame_at(I_IMP[k])
        rx, ry = red_centroid_in(fr, C0[0], IMP_Y[k], 24)
        kx, ky = centroid(ink_mask(
            fr[int(IMP_Y[k]) - 60:int(IMP_Y[k]) + 60,
               int(C0[0]) - 50:int(C0[0]) + 50, :].astype(np.float64)),
            int(C0[0]) - 50, int(IMP_Y[k]) - 60)
        worst_dc = max(worst_dc, np.hypot(rx - kx, ry - ky))
        worst_dm = max(worst_dm, np.hypot(rx - C0[0], ry - IMP_Y[k]))
    ck("three impacts: dart and can centroids coincide on the frames",
       worst_dc < 3.5 and worst_dm < 3.5,
       f"worst dart-can {worst_dc:.1f} px, dart-model {worst_dm:.1f} px")

    # the ruler: three strict-red ticks in a box nothing else can enter
    reg = f_end[:, int(TICK_X0):int(TICK_X1) + 1, :].astype(np.float64)
    rows = np.where(red_strict(reg).any(1))[0]
    groups = np.split(rows, np.where(np.diff(rows) > 4)[0] + 1)
    cents = [g.mean() for g in groups]
    ratio = (cents[2] - cents[0]) / (cents[1] - cents[0])
    ck("three ticks at the impact heights; spacing ratio 8/3 "
       "(pure pixel read of 1:4:9)",
       len(groups) == 3 and
       all(abs(c - y) < 2.5 for c, y in zip(cents, IMP_Y)) and
       abs(ratio - 8 / 3) < 0.05,
       f"y {[round(c,1) for c in cents]}, ratio {ratio:.4f}")

    # the bar tracks the can: full red from the hold point down to the
    # can's (blurred) top edge, mid-fall and at the freeze. Mid-fall
    # frame is f64: flash 0 has died (f59), flash 1 not yet fired (f66)
    # — an ACTIVE ink flash ring crosses this column and legitimately
    # gaps the red (trap 62: the expectation was wrong, not the bar)
    for fi, t in ((64, (64 - I_F) / FPS), (100, T_IMP[2])):
        fr = frame_at(fi)
        y_lo = int(C0[1]) - 4
        y_hi = int(can_pos(t)[1] - CAN_H / 2) - 14
        col = fr[y_lo:y_hi, int(C0[0]) - 4:int(C0[0]) + 5, :] \
            .astype(np.float64)
        rr = np.where(red_strict(col).any(1))[0]
        assert len(rr) and rr.min() < 6 and \
            rr.max() > (y_hi - y_lo) - 8 and \
            (np.diff(np.where(red_strict(col).any(1))[0]) <= 2).all(), \
            (fi, len(rr))
    ck("the drop bar runs unbroken from the hold point to the can, "
       "mid-fall and at the freeze", True)

    # can rides the model: ink centroid mid-fall (blur is centred).
    # f64, not f70: at f70 flash 1's lower arc reaches into this ink
    # box (same trap-62 lesson as the bar read above)
    t64 = (64 - I_F) / FPS
    cp = can_pos(t64)
    fr64 = frame_at(64)
    kx, ky = centroid(ink_mask(
        fr64[int(cp[1]) - 62:int(cp[1]) + 62,
             int(cp[0]) - 52:int(cp[0]) + 52, :].astype(np.float64)),
        int(cp[0]) - 52, int(cp[1]) - 62)
    ck("the can rides free fall (ink centroid, mid-fall)",
       np.hypot(kx - cp[0], ky - cp[1]) < 4.0,
       f"err {np.hypot(kx - cp[0], ky - cp[1]):.1f} px")

    # nothing red above the static aim line (gravity only pulls DOWN
    # off it) — x fence starts past the 3-trail overlap zone
    top_ok = True
    for x in range(250, 871, 25):
        col = f_end[:, x - 1:x + 2, :].astype(np.float64)
        rows_r = np.where(red_tint(col).any(1))[0]
        if len(rows_r) and rows_r.min() < line_y(x) - 8:
            top_ok = False
    ck("nothing red ever sits above the aim line", top_ok)

    # the falling line off the PIXELS at the freeze: at column x=700
    # the ghost sits exactly DROP3 below the static line (rows fenced
    # to exclude the static dashes; red things fail the neutral mask)
    xq = 700
    y_stat = line_y(xq)
    col = f_end[int(y_stat) + 30:int(FLOOR) - 40, xq - 2:xq + 3, :]
    g_rows = np.where(ghost_mask(col).any(1))[0]
    y_meas = g_rows.mean() + int(y_stat) + 30
    ck("the falling line reads half g t^2 below the aim line off the "
       "pixels", abs(y_meas - (y_stat + DROPS[2])) < 3.0,
       f"gap {y_meas - y_stat:.1f} px, model {DROPS[2]:.1f}")

    # trails are present but strictly sub-cut away from the muzzle
    tq = 0.8654                      # dart 2 passes x=600 here
    p = dart(2, tq)
    box = f_end[int(p[1]) - 10:int(p[1]) + 11,
                int(p[0]) - 10:int(p[0]) + 11, :].astype(np.float64)
    ck("trail: visible tint, invisible to the strict mask (economy "
       "holds, trap 61: say what it is)",
       red_tint(box).sum() > 30 and red_strict(box).sum() == 0,
       f"tint {red_tint(box).sum()} px, strict {red_strict(box).sum()}")

    # hang line: holds through PRE, cut at the fire
    def line_ink(i):
        fr = frame_at(i)
        return int(ink_mask(fr[120:400, int(C0[0]) - 6:int(C0[0]) + 7, :]
                            .astype(np.float64)).sum())
    ck("the hang line holds until the muzzle flash, then is cut",
       line_ink(10) > 500 and line_ink(I_F - 1) > 500 and
       line_ink(I_F + 1) == 0,
       f"ink {line_ink(I_F - 1)} before, {line_ink(I_F + 1)} after")

    # digits present on the final frame, absent before the first impact
    def digit_ink(fr, k):
        y = int(IMP_Y[k])
        return int((fr[y - 21:y + 21, DIGIT_CX - 18:DIGIT_CX + 18, :]
                    .astype(int).sum(2) < 3 * 120).sum())
    f45 = frame_at(45)
    ck("digits 1, 4, 9 on the final frame; none before the first hit",
       all(digit_ink(f_end, k) > 120 for k in range(3)) and
       all(digit_ink(f45, k) == 0 for k in range(3)),
       f"ink {[digit_ink(f_end, k) for k in range(3)]} px")

    ck("digits inside the text-safe area (trap 3)",
       IMP_Y[0] - 21 > 0.10 * H and IMP_Y[2] + 21 < 0.85 * H)

    ck("holds are truly static (byte-equal frames)",
       np.array_equal(frame_at(2), frame_at(PRE - 2)) and
       np.array_equal(frame_at(I_FRZ + 15), frame_at(N_FRAMES - 1)))

    print(f"ALL {ok} CHECKS PASSED")
    print("NOT verified by any check above (trap 68):")
    print("  - units are pixels, g chosen for the frame; the checked"
          " claims (the hits, 1:4:9, the rigid line) are"
          " scale-invariant, but the speeds are not real-world")
    print("  - three darts from one muzzle at one instant is a diagram"
          " convention; the real demo varies the speed across TRIALS")
    print("  - the impacts leave the can's fall unchanged (momentum"
          " ignored) and the darts stick at its center, coinciding")
    print("  - the freeze after the third hit is presentation: really"
          " everything would keep falling")
    print("  - drag, spin and the darts' length are ignored; the"
          " release is instantaneous")


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
    reg = d[:, int(TICK_X0):int(TICK_X1) + 1, :].astype(np.float64)
    rows = np.where(red_strict(reg).any(1))[0]
    groups = np.split(rows, np.where(np.diff(rows) > 4)[0] + 1)
    cents = [g.mean() for g in groups]
    ratio = (cents[2] - cents[0]) / (cents[1] - cents[0])
    assert len(groups) == 3 and abs(ratio - 8 / 3) < 0.06, \
        (len(groups), ratio)
    print(f"    ticks survive the encode at ratio {ratio:.4f} "
          f"(model 8/3 = {8/3:.4f})")
    rx, ry = red_centroid_in(d, C0[0], IMP_Y[2], 24)
    err = np.hypot(rx - C0[0], ry - IMP_Y[2])
    assert err < 3.0, err
    print(f"    stuck darts sit on the third impact point to "
          f"{err:.1f} px")
    dd = np.abs(decode_frame(0).astype(int) - frame_at(0).astype(int))
    print(f"    decoded f0 vs render: mean |diff| {dd.mean():.3f}, "
          f"max {dd.max()}")
    assert dd.mean() < 2.0
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams",
         "v", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
         OUT], capture_output=True, text=True).stdout.strip()
    assert int(probe) == N_FRAMES, f"frames {probe} != {N_FRAMES}"
    print(f"    {probe} frames; 1:4:9 survives the encode")


def review_stills():
    base = OUT[:-4]
    picks = [("aim", 10), ("fire", I_F + 2), ("hit1", I_IMP[0] + 2),
             ("mid", 62), ("lob", 76), ("final", N_FRAMES - 1)]
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
