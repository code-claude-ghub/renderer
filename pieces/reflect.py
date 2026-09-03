#!/usr/bin/env python3
"""REFLECT — a pulse hits a fixed wall and comes back upside down.

Third wave-family piece, sequel to CANCEL and NODES. One raised-cosine
pulse travels right along a rope toward a rigid wall. Beyond the wall,
drawn in ghost on darker paper, lives the MIRROR WORLD: an inverted
twin pulse approaching from the other side. The real rope is the sum
of pulse and twin — the method of images, drawn instead of hidden.

The wall point (red pin) is pulse-minus-twin at the SAME table index,
so it sits at 0.0 BINARY-EXACT on all 137 motion frames. When the
pulse is centred on the wall the twin cancels it EVERYWHERE: the rope
is exactly flat for one frame, its energy 100% kinetic (PE == 0.0,
literally — nothing is stretched), and the freeze shows red velocity
arrows: real side moving down, mirror side up. Then the pulse
re-emerges inverted. Time reversal about the flat frame is an EXACT
vertical flip — the film after the bounce is the film before it,
upside down, byte-checkable off the render.

Deep checks: an independent leapfrog integration with the wall as a
Dirichlet boundary reproduces the drawn model to ~1e-12 px, and a
SECOND leapfrog on the free doubled domain (no wall, ghost twin as
real initial data) matches the walled run EXACTLY — 0.0 — because odd
symmetry is preserved bitwise by the update. Silent.

Eleventh ordinary-world classic. Third wave-family piece.
"""
import os
import subprocess
import time

import numpy as np

# ---------------------------------------------------------------- canvas
W, H = 1080, 1920
FPS = 30
PAPER = 0.845                   # real world (trap 69)
MIRROR = 0.815                  # the mirror world's slightly darker paper
INK = 0.10
GHOST = 0.58
C_RED = (0.55, 0.10, 0.10)

# ---------------------------------------------------------------- model
XW = 540                        # the wall (frame centre)
X0, X1 = 60, 1020               # drawn span: real | mirror
AMP = 160.0                     # pulse height, px
HW = 120                        # pulse half-width, px
S0 = 200                        # pulse centre at t=0
PXF = 5                         # exactly 5 px per frame
C_PXS = PXF * FPS               # 150 px/s (prose only)
T_FLAT = (XW - S0) // PXF       # 68 — pulse centred on the wall
T_END = 2 * T_FLAT              # 136 — back where it started, inverted

Y_M = 960.0                     # rope baseline (frame centre)
LW_ROPE = 5.0
LW_GHOST = 3.0
BAR_W, BAR_H = 10.0, 760.0      # the wall: 580..1340, symmetric about Y_M
PIN_R = 7.5

# timeline (t = motion frames; i = video frames)
PRE = 36
SEG1 = T_FLAT                   # 68: approach
FRZ = 32                        # freeze on the flat frame
SEG2 = T_FLAT                   # 68: departure, inverted
POST = 45
N_FRAMES = PRE + SEG1 + FRZ + SEG2 + POST          # 249 = 8.3 s
I_GO = PRE                                          # f36
I_FRZ = PRE + SEG1                                  # f104, t=68, FLAT
I_R2 = I_FRZ + FRZ                                  # f136
I_END = I_R2 + SEG2                                 # f204

ARROW_K = 0.2                   # arrow px per (px/s)
ARROW_DS = [-90, -60, -30, 30, 60, 90]              # wall offsets

OUT = f"out/reflect_{time.strftime('%H%M%S')}.mp4"


# ------------------------------------------------------- the exact table
def build_pulse():
    """Raised-cosine bump, even symmetry enforced bitwise."""
    p = np.zeros(2 * HW + 1, np.float64)
    for k in range(0, HW + 1):
        q = 0.5 * AMP * (1.0 + np.cos(np.pi * k / HW))
        p[HW + k] = q
        p[HW - k] = q             # the SAME float, mirrored
    return p


PT = build_pulse()
XS = np.arange(X0, X1 + 1)      # integer columns


def pval(idx):
    idx = np.asarray(idx)
    out = np.zeros(idx.shape, np.float64)
    m = np.abs(idx) <= HW
    out[m] = PT[idx[m] + HW]
    return out


def u_at(t):
    """Rope displacement over XS: pulse minus its mirror twin."""
    s = S0 + PXF * t
    return pval(XS - s) - pval(2 * XW - XS - s)


def t_at(i):
    """Motion time at video frame i (hold / run / freeze / run / hold)."""
    if i < I_GO:
        return 0
    if i < I_FRZ:
        return i - I_GO
    if i < I_R2:
        return T_FLAT
    if i < I_END:
        return T_FLAT + (i - I_R2)
    return T_END


def v_flat(delta):
    """Rope velocity (px/s) at wall offset delta on the flat frame."""
    return float(C_PXS * AMP * np.pi / HW * np.sin(np.pi * delta / HW))


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


def draw_curve(img, uvals, x_lo, x_hi, color, lw):
    """The rope between x_lo and x_hi: y = Y_M - u(x), every 3 px."""
    xs = np.arange(float(x_lo), x_hi + 1.0, 3.0)
    ui = uvals[(xs - X0).astype(int)]
    pts = list(zip(xs, Y_M - ui))
    x0, y0, cv = polyseg_cov(pts, lw)
    comp_bbox(img, x0, y0, cv, color)


def draw_arrow(img, x, v, fade):
    """Red velocity arrow at column x. v > 0 = UP."""
    y_tip = Y_M - v * ARROW_K
    x0, y0, cv = polyseg_cov([(x, Y_M), (x, y_tip)], 4.0)
    comp_bbox(img, x0, y0, cv * fade, C_RED)
    barb_y = y_tip + (14.0 if v > 0 else -14.0)
    for dx in (-9.0, 9.0):
        x0, y0, cv = polyseg_cov([(x + dx, barb_y), (x, y_tip)], 4.0)
        comp_bbox(img, x0, y0, cv * fade, C_RED)


def draw_chevron(img, cx, cy, right, color):
    """Direction arrow during the pre-hold."""
    s = 1.0 if right else -1.0
    x0, y0, cv = polyseg_cov([(cx - s * 30, cy), (cx + s * 30, cy)], 4.0)
    comp_bbox(img, x0, y0, cv, color)
    for dy in (-10.0, 10.0):
        x0, y0, cv = polyseg_cov([(cx + s * 14, cy + dy),
                                  (cx + s * 30, cy)], 4.0)
        comp_bbox(img, x0, y0, cv, color)


# ---------------------------------------------------------------- static
def background():
    fr = np.full((H, W, 3), PAPER, np.float64)
    fr[:, XW:, :] = MIRROR          # the mirror world's paper
    # the wall: a rigid ink bar, symmetric about the baseline (the
    # symmetry matters: it makes the time-flip identity byte-checkable)
    x0, y0, cv = rect_cov(float(XW), Y_M, BAR_W, BAR_H)
    comp_bbox(fr, x0, y0, cv, INK)
    # the red pin: the one point the wall holds still — the rope will
    # sit at EXACTLY this height there on every frame
    x0, y0, cv = disc_cov(float(XW), Y_M, PIN_R)
    comp_bbox(fr, x0, y0, cv, C_RED)
    return fr


BG = background()


# ---------------------------------------------------------------- frames
def frame_at(i):
    img = BG.copy()
    t = t_at(i)
    u = u_at(t)
    draw_curve(img, u, XW, X1, GHOST, LW_GHOST)     # mirror world first
    draw_curve(img, u, X0, XW, INK, LW_ROPE)        # the real rope on top
    if i < I_GO:
        draw_chevron(img, 200.0, 740.0, True, INK)
        draw_chevron(img, 880.0, 1180.0, False, GHOST)
    if I_FRZ <= i < I_R2:
        fade = float(np.clip((i - I_FRZ) / 8.0, 0.0, 1.0))
        if fade > 0:
            for d in ARROW_DS:
                draw_arrow(img, float(XW + d), v_flat(d), fade)
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def render_frames():
    for i in range(N_FRAMES):
        yield frame_at(i)                            # stream (trap 34)


# ---------------------------------------------------------------- measure
# FENCE AUDIT (written before the first render):
#   wall bar cols 533..547 (ink!)   pin cols 532..548, rows 952..968
#   arrow cols 439..641, freeze frames only     chevrons: pre-hold only,
#   boxes (160..240, 724..756) ink and (840..920, 1164..1196) ghost
#   rope_row_at band rows 760..1160: chevron boxes are OUTSIDE it
#   profile columns keep >=18 px clear of bar and pin (fenced to
#   x <= 515 real / x >= 565 mirror); flat-frame crops likewise
#   flipud band rows 620..1300 (symmetric about 960); used only on
#   motion frames (no chevrons, no arrows)
def red_strict(reg):
    return (np.clip(reg[:, :, 0] - reg[:, :, 1] - 60, 0, None) *
            (reg[:, :, 2] - reg[:, :, 1] < 40)) > 0


def ink_mask(reg):
    return reg.max(2) < 100


def ghost_mask(reg):
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


def rope_row_at(fr, x, mask_fn=ink_mask):
    y0 = int(Y_M) - 200
    col = fr[y0:int(Y_M) + 200, x - 1:x + 2, :].astype(np.float64)
    _, ry = centroid(mask_fn(col), 0, y0)
    return ry


PIN_BOX = (XW - 14, XW + 15, int(Y_M) - 14, int(Y_M) + 15)


def red_outside_pin(fr):
    m = red_strict(fr.astype(np.float64))
    x0, x1, y0, y1 = PIN_BOX
    m[y0:y1, x0:x1] = False
    return int(m.sum())


CHEV_BOXES = ((160, 240, 724, 756), (840, 920, 1164, 1196))


def mask_chevrons(fr):
    out = fr.copy()
    for x0, x1, y0, y1 in CHEV_BOXES:
        out[y0:y1, x0:x1] = 0
        # the flipped position of rows [y0, y1) under r -> H - r is
        # [H-y1+1, H-y0+1) — the naive [H-y1, H-y0) is off by one at
        # BOTH edges and breaks byte-equality on two background rows
        out[H - y1 + 1:H - y0 + 1, x0:x1] = 0
    return out


# --------------------------------------------------------------- leapfrog
GL = -700


def analytic_on(xs, m):
    s = S0 + m
    return pval(xs - s) - pval(2 * XW - xs - s)


def leapfrog(x_hi, substeps):
    """Courant number exactly 1, Dirichlet at both ends of GL..x_hi."""
    xg = np.arange(GL, x_hi + 1)
    up, u = analytic_on(xg, -1), analytic_on(xg, 0)
    snaps = {0: u.copy()}
    for m in range(1, substeps + 1):
        un = np.zeros(len(xg))
        un[1:-1] = u[:-2] + u[2:] - up[1:-1]
        up, u = u, un
        if m % PXF == 0:
            snaps[m // PXF] = u.copy()
    return snaps


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
    iw = XW - X0
    worst = max(abs(u_at(t)[iw]) for t in range(T_END + 1))
    ck("the wall point sits at 0.0 BINARY-EXACT on all 137 motion "
       "frames", worst == 0.0, f"max |u(wall)| = {worst}")

    ck("the flat frame: u == 0.0 EVERYWHERE, bitwise, at t=68",
       np.all(u_at(T_FLAT) == 0.0))
    ck("one frame either side it is not flat (trap 59 control)",
       np.abs(u_at(T_FLAT - 1)).max() > 10,
       f"max |u| at t=67: {np.abs(u_at(T_FLAT - 1)).max():.1f} px")

    ck("time reversal about the flat frame is an EXACT vertical flip: "
       "u(t_flat+tau) == -u(t_flat-tau), bitwise, every tau",
       all(np.array_equal(u_at(T_FLAT + tau), -u_at(T_FLAT - tau))
           for tau in range(1, T_FLAT + 1)))

    ck("odd symmetry about the wall, bitwise: the mirror world really "
       "is the mirror",
       all(np.array_equal(u_at(t)[::-1], -u_at(t))
           for t in range(0, T_END + 1, 4)))

    u0, ue = u_at(0), u_at(T_END)
    ck("the pulse returns EXACTLY inverted: u(t_end) == -u(0) bitwise; "
       "peak +160.0 exactly, then -160.0 exactly",
       np.array_equal(ue, -u0) and u0.max() == AMP and ue.min() == -AMP)

    SW = leapfrog(XW, PXF * T_END)
    worst = max(np.abs(SW[n][X0 - GL:] - u_at(n)[:XW - X0 + 1]).max()
                for n in range(T_END + 1))
    ck("independent leapfrog PDE (wall as Dirichlet BC, Courant 1) "
       "reproduces the drawn model", worst < 1e-9,
       f"max |num-analytic| {worst:.1e} px")

    SF = leapfrog(2 * XW - GL, PXF * T_END)      # no wall: doubled domain
    worst = max(np.abs(SF[n][:XW - GL + 1] - SW[n]).max()
                for n in range(T_END + 1))
    ck("METHOD OF IMAGES: the free universe with the ghost twin == the "
       "walled universe, EXACTLY (odd symmetry survives bitwise)",
       worst == 0.0, f"max diff {worst:.1e}")

    worst = max(np.abs(SW[n][:50]).max() for n in range(T_END + 1))
    ck("left guard never touched: the cut left edge is honest",
       worst == 0.0, f"max |u| left of {GL + 50}: {worst}")

    # energy: free doubled domain (compact support; no boundary term —
    # cutting at the wall put a rectangle-rule endpoint on a large
    # time-varying u_x and manufactured 8e-5 of fake drift)
    xf = np.arange(GL, 2 * XW - GL + 0.01, 0.01)

    def dp(k):
        k = np.asarray(k, np.float64)
        return np.where(np.abs(k) <= HW,
                        -0.5 * AMP * np.pi / HW * np.sin(np.pi * k / HW),
                        0.0)

    def energies(t):
        ct = float(PXF * t)
        a1 = xf - S0 - ct
        a2 = 2 * XW - xf - S0 - ct
        ut = C_PXS * (-dp(a1) + dp(a2))
        ux = dp(a1) + dp(a2)
        return (0.5 * np.sum(ut ** 2) * 0.01,
                0.5 * C_PXS ** 2 * np.sum(ux ** 2) * 0.01)

    E0 = sum(energies(0))
    drift = max(abs(sum(energies(t)) - E0) / E0
                for t in range(0, T_END + 1, 4))
    keF, peF = energies(T_FLAT)
    ck("energy constant through the bounce; on the flat frame PE == "
       "0.0 EXACTLY — nothing is stretched, everything is moving",
       drift < 1e-9 and peF == 0.0,
       f"drift {drift:.1e}, PE at flat {peF}")

    v_fd = (u_at(T_FLAT + 1) - u_at(T_FLAT - 1)) * FPS / 2
    worst = max(abs(v_flat(d) - v_fd[XW - X0 + d]) for d in ARROW_DS)
    ck("flat-frame velocity: analytic matches finite difference at all "
       "six arrow columns; real side DOWN, mirror side UP",
       worst < 3.0 and v_flat(-60) < -600 and v_flat(60) > 600,
       f"|v|max {abs(v_flat(60)):.0f} px/s, fd dev {worst:.2f}")

    ck("timeline: 249 frames = 8.3 s", N_FRAMES == 249
       and N_FRAMES / FPS <= 180)

    # -- pixels
    f_end = frame_at(N_FRAMES - 1)
    ink_frac = float((f_end.astype(int).sum(2) < 3 * 180).mean())
    ck("ink fraction sane on the final frame (trap 56)",
       0.002 < ink_frac < 0.30, f"{ink_frac:.4f}")

    ck("the two papers differ: real vs mirror background",
       f_end[100, 100, 0] > f_end[100, 980, 0] + 3,
       f"real {f_end[100, 100, 0]}, mirror {f_end[100, 980, 0]}")

    # THE flat frame off the pixels (i=I_FRZ: fade is 0, no arrows yet;
    # crops fenced clear of bar 533..547 and pin 532..548)
    fr_flat = frame_at(I_FRZ)
    band = fr_flat[int(Y_M) - 200:int(Y_M) + 200, 70:520, :] \
        .astype(np.float64)
    rows = np.where(ink_mask(band).any(1))[0] + int(Y_M) - 200
    ck("flat frame off the pixels, real side: all rope ink within "
       "+-4.5 px of the baseline", len(rows) > 0 and
       abs(rows.min() - Y_M) < 4.5 and abs(rows.max() - Y_M) < 4.5,
       f"ink rows {rows.min()}..{rows.max()}")
    band = fr_flat[int(Y_M) - 200:int(Y_M) + 200, 560:1010, :] \
        .astype(np.float64)
    rows = np.where(ghost_mask(band).any(1))[0] + int(Y_M) - 200
    ck("flat frame, mirror side: the ghost line is flat too",
       len(rows) > 0 and
       abs(rows.min() - Y_M) < 4.5 and abs(rows.max() - Y_M) < 4.5,
       f"ghost rows {rows.min()}..{rows.max()}")

    # the pin never moves
    worst_p = 0.0
    for fi in (0, 76, 120, 164, 190, 248):
        fr = frame_at(fi).astype(np.float64)
        x0, x1, y0, y1 = PIN_BOX
        bx, by = centroid(red_strict(fr[y0:y1, x0:x1, :]), x0, y0)
        worst_p = max(worst_p, abs(bx - XW), abs(by - Y_M))
    ck("the red pin sits at the wall point on every sampled frame",
       worst_p < 1.0, f"worst centroid err {worst_p:.2f} px")

    # profile vs model, both worlds, three frames (t=40, 96, 122)
    worst_i = worst_g = 0.0
    for fi in (76, 164, 190):
        fr = frame_at(fi)
        t = t_at(fi)
        u = u_at(t)
        for x in range(75, 516, 40):
            ry = rope_row_at(fr, x)
            worst_i = max(worst_i, abs(ry - (Y_M - u[x - X0])))
        for x in range(565, 1006, 40):
            ry = rope_row_at(fr, x, ghost_mask)
            worst_g = max(worst_g, abs(ry - (Y_M - u[x - X0])))
    ck("the drawn rope IS the model: ink profile, 36 column reads",
       worst_i < 3.0, f"worst {worst_i:.2f} px")
    ck("the mirror world tracks its model too: ghost profile, 36 reads",
       worst_g < 3.0, f"worst {worst_g:.2f} px")

    # red economy
    ck("red pin always; NOTHING else red outside the freeze",
       red_outside_pin(frame_at(0)) == 0 and
       red_outside_pin(frame_at(76)) == 0 and
       red_outside_pin(frame_at(190)) == 0 and
       red_outside_pin(frame_at(248)) == 0 and
       red_outside_pin(frame_at(120)) > 400,
       f"mid-freeze arrows {red_outside_pin(frame_at(120))} px")

    # arrow directions on the freeze
    fr120 = frame_at(120).astype(np.float64)
    dn = red_strict(fr120[:, 440:522, :])
    up = red_strict(fr120[:, 559:641, :])
    _, ry_dn = centroid(dn, 0, 0)
    _, ry_up = centroid(up, 0, 0)
    ck("freeze arrows: real-side red mass BELOW the baseline (moving "
       "down), mirror-side ABOVE (moving up)",
       ry_dn > Y_M + 10 and ry_up < Y_M - 10,
       f"down {ry_dn:.0f}, up {ry_up:.0f}, baseline {Y_M:.0f}")

    # the byte identities off the render
    bA = frame_at(76)[620:1301, :, :]        # t = 68-28 = 40
    bB = frame_at(164)[620:1301, :, :]       # t = 68+28 = 96
    ck("TIME-FLIP, byte-exact: the frame 28 ticks after the bounce is "
       "the frame 28 ticks before it, upside down",
       np.array_equal(bB, bA[::-1, :, :]))

    ck("the video ends as the upside-down of its beginning (final hold "
       "== pre-hold flipped, byte-equal outside the chevron boxes)",
       np.array_equal(mask_chevrons(frame_at(N_FRAMES - 2))[620:1301],
                      mask_chevrons(frame_at(2))[620:1301][::-1]))

    ck("chevrons in the pre-hold only",
       ink_mask(frame_at(10)[724:756, 160:240, :]
                .astype(np.float64)).sum() > 40 and
       ink_mask(frame_at(76)[724:756, 160:240, :]
                .astype(np.float64)).sum() == 0)

    # (fenced: the rope, the ghost line and the pin legally cross the
    # bar around y=960 — the rope is pinned there, that's the piece)
    ck("the wall is there: ink bar solid above and below the rope",
       ink_mask(frame_at(76)[600:920, 536:545, :]
                .astype(np.float64)).all() and
       ink_mask(frame_at(76)[1000:1320, 536:545, :]
                .astype(np.float64)).all())

    ck("holds are truly static (byte-equal frames)",
       np.array_equal(frame_at(2), frame_at(PRE - 2)) and
       np.array_equal(frame_at(I_FRZ + 10), frame_at(I_R2 - 1)) and
       np.array_equal(frame_at(I_END + 2), f_end))

    print(f"ALL {ok} CHECKS PASSED")
    print("NOT verified by any check above (trap 68):")
    print("  - units are pixels; the wave speed is chosen for the"
          " frame. the checked claims (inversion, the flat instant,"
          " the image construction) are scale-invariant")
    print("  - the string is IDEAL: linear, lossless, non-dispersive,"
          " and the wall is PERFECTLY rigid. a real clamp absorbs a"
          " little of every bounce and the returning pulse is smaller")
    print("  - nothing exists beyond the wall. the mirror world is the"
          " method of images — the bookkeeping physicists actually use,"
          " drawn instead of hidden. the checks prove it gives EXACTLY"
          " the walled answer; they cannot make it real")
    print("  - the amplitude is large next to the pulse width (max"
          " slope ~2.1): the linear wave equation is ASSERTED, not"
          " derived, at this steepness")
    print("  - the freeze on the flat frame is presentation: really it"
          " lasts one frame, 1/30 s")


# ---------------------------------------------------------------- encode
def encode():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-crf", "18", "-preset", "slow",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stderr=subprocess.DEVNULL)
    for fr in render_frames():
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
    d = decode_frame(I_FRZ)
    # crops already fenced clear of the bar, the pin and their chroma
    # smear (nodes.py lesson: decode-side fences differ from render-side)
    band = d[int(Y_M) - 200:int(Y_M) + 200, 70:520, :].astype(np.float64)
    rows = np.where(ink_mask(band).any(1))[0] + int(Y_M) - 200
    assert abs(rows.min() - Y_M) < 6 and abs(rows.max() - Y_M) < 6, \
        (rows.min(), rows.max())
    print(f"    flat frame survives the encode: ink rows "
          f"{rows.min()}..{rows.max()} (baseline {Y_M:.0f})")
    n_pin = int(red_strict(decode_frame(76).astype(np.float64)).sum())
    assert n_pin > 80, n_pin
    print(f"    pin survives: {n_pin} strict-red px on a motion frame")
    n_frz = int(red_strict(decode_frame(120).astype(np.float64)).sum())
    assert n_frz > n_pin + 300, (n_frz, n_pin)
    print(f"    arrows on the freeze: {n_frz} strict-red px")
    dd = np.abs(decode_frame(0).astype(int) - frame_at(0).astype(int))
    print(f"    decoded f0 vs render: mean |diff| {dd.mean():.3f}, "
          f"max {dd.max()}")
    assert dd.mean() < 2.0
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams",
         "v", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
         OUT], capture_output=True, text=True).stdout.strip()
    assert int(probe) == N_FRAMES, f"frames {probe} != {N_FRAMES}"
    print(f"    {probe} frames; the bounce survives the encode")


def review_stills():
    base = OUT[:-4]
    picks = [("start", 10), ("approach", 80), ("flat", 120),
             ("emerge", 164), ("return", 190), ("final", N_FRAMES - 1)]
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
