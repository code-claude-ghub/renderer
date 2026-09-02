#!/usr/bin/env python3
"""NODES — two wavetrains race through each other; five points never move.

Sequel to CANCEL, same equation layout: wave 1 alone (ghost, running
right) plus wave 2 alone (ghost, running left) equals the rope (ink).
The sum is a STANDING WAVE. Five interior points — the nodes, marked
as red beads threaded on the rope — sit at displacement 0.0
BINARY-EXACT on every frame of the video, while both component trains
run through them at full speed the whole time. Twice a second the
whole rope passes through flat (CANCEL's flat frame, now recurring);
at that instant the energy is 100% kinetic and red arrows show
adjacent antinodes moving opposite ways.

The construction: both trains index ONE 300-entry table built with its
symmetries enforced bitwise (W[150-i] == W[i]; W[i+150] == -W[i]), and
the speed is exactly 5 px/frame — so at a node the two trains hand the
rope the SAME table entry with opposite signs, every frame. The motion
repeats exactly: frames 60 apart are byte-identical off the render,
and a half-period is a byte-exact mirror about the baseline.

Deep check: an independent leapfrog integration (fixed ends, Courant
number exactly 1) reproduces the drawn solution to ~1e-11 px, and the
two trains integrated SOLO on a ring sum to the joint run. Silent.

Tenth ordinary-world classic. Second wave-family piece.
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
XL, XR = 90.0, 990.0            # rope span: 900 px = 3 wavelengths
L = 300                          # wavelength, px
A = 80.0                         # component amplitude; rope swings +-160
PXF = 5                          # exactly 5 px per frame
C_PXS = PXF * FPS                # 150 px/s (prose only)
K = 2 * np.pi / L

Y_A, Y_C, Y_B = 420.0, 800.0, 1310.0     # lanes: wave1 + wave2 = rope
LW_ROPE = 5.0
LW_LANE = 3.0

NODES_IN = [240.0, 390.0, 540.0, 690.0, 840.0]    # the five red beads
ANTIN_X = [165.0, 315.0, 465.0, 615.0, 765.0, 915.0]
BEAD_R = 7.5

# timeline (t = motion frames; i = video frames)
PRE = 36
SEG1 = 15                        # t 0 -> 15: run to the first flat
FRZ = 32                         # freeze on the flat frame
SEG2 = 105                       # t 15 -> 120: 1.75 more periods
POST = 45                        # hold on the start pattern (t=120)
N_FRAMES = PRE + SEG1 + FRZ + SEG2 + POST         # 233 = 7.77 s
I_GO = PRE                                         # f36
I_FRZ = PRE + SEG1                                 # f51, t=15, FLAT
I_R2 = I_FRZ + FRZ                                 # f83
I_END = I_R2 + SEG2                                # f188
T_END = SEG1 + SEG2                                # 120 motion frames

ARROW_K = 0.2                    # arrow px per (px/s)
ARROW_DS = [75 + 150 * k + s for k in range(6) for s in (-45, 0, 45)]

OUT = f"out/nodes_{time.strftime('%H%M%S')}.mp4"


# ------------------------------------------------------- the exact table
def build_table():
    t = np.zeros(L, np.float64)
    for i in range(0, 76):
        q = A * np.sin(2 * np.pi * i / L)
        t[i] = q
        t[150 - i] = q                    # mirror symmetry, bitwise
    for j in range(150):
        t[150 + j] = -t[j]                # half-period, exact negation
    return t


WT = build_table()
DG = np.arange(0, int(XR - XL) + 1)       # d = x - XL, 0..900


def w1_at(t):
    """Right-moving train on the integer grid at motion frame t."""
    return WT[np.mod(DG - PXF * t, L)]


def w2_at(t):
    """Left-moving train (same table, opposite index direction)."""
    return WT[np.mod(DG + PXF * t, L)]


def t_at(i):
    """Motion time at video frame i (hold / run / freeze / run / hold)."""
    if i < I_GO:
        return 0
    if i < I_FRZ:
        return i - I_GO
    if i < I_R2:
        return SEG1                        # 15 — the flat instant
    if i < I_END:
        return SEG1 + (i - I_R2)
    return T_END                           # 120 — the start pattern again


def v_flat(d):
    """Rope velocity (px/s) at column d on a flat frame (t=15 family)."""
    return -2 * np.pi * A * np.sin(K * d)


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


def draw_curve(img, y_lane, uvals, color, lw):
    """One lane's curve: y = y_lane - u(x), sampled every 3 px."""
    xs = np.arange(XL, XR + 1.0, 3.0)
    ui = np.interp(xs, XL + DG.astype(np.float64), uvals)
    pts = list(zip(xs, y_lane - ui))
    x0, y0, cv = polyseg_cov(pts, lw)
    comp_bbox(img, x0, y0, cv, color)


def draw_arrow(img, x, v, fade):
    """Red velocity arrow at column x on the rope lane. v > 0 = UP."""
    y_tip = Y_B - v * ARROW_K
    x0, y0, cv = polyseg_cov([(x, Y_B), (x, y_tip)], 4.0)
    comp_bbox(img, x0, y0, cv * fade, C_RED)
    barb_y = y_tip + (14.0 if v > 0 else -14.0)
    for dx in (-9.0, 9.0):
        x0, y0, cv = polyseg_cov([(x + dx, barb_y), (x, y_tip)], 4.0)
        comp_bbox(img, x0, y0, cv * fade, C_RED)


def draw_chevron(img, cx, cy, right):
    """Small ink direction arrow during the pre-hold."""
    s = 1.0 if right else -1.0
    x0, y0, cv = polyseg_cov([(cx - s * 30, cy), (cx + s * 30, cy)], 4.0)
    comp_bbox(img, x0, y0, cv, INK)
    for dy in (-10.0, 10.0):
        x0, y0, cv = polyseg_cov([(cx + s * 14, cy + dy),
                                  (cx + s * 30, cy)], 4.0)
        comp_bbox(img, x0, y0, cv, INK)


# ---------------------------------------------------------------- static
def background():
    fr = np.full((H, W, 3), PAPER, np.float64)
    # equation furniture: wave1 + wave2 = rope (same spots as CANCEL)
    for cx, cy, w_, h_ in ((540.0, 610.0, 44.0, 7.0),
                           (540.0, 610.0, 7.0, 44.0),
                           (540.0, 1041.0, 44.0, 7.0),
                           (540.0, 1069.0, 44.0, 7.0)):
        x0, y0, cv = rect_cov(cx, cy, w_, h_)
        comp_bbox(fr, x0, y0, cv, INK)
    # ink pins at the rope's fixed ends (they are nodes too, held by
    # the wall; the five FREE never-movers get the red beads)
    for x in (XL, XR):
        x0, y0, cv = disc_cov(x, Y_B, 5.0)
        comp_bbox(fr, x0, y0, cv, INK)
    # the five red beads at the interior nodes — the rope will thread
    # them on every single frame
    for x in NODES_IN:
        x0, y0, cv = disc_cov(x, Y_B, BEAD_R)
        comp_bbox(fr, x0, y0, cv, C_RED)
    return fr


BG = background()


# ---------------------------------------------------------------- frames
def frame_at(i):
    img = BG.copy()
    t = t_at(i)
    w1, w2 = w1_at(t), w2_at(t)
    draw_curve(img, Y_A, w1, GHOST, LW_LANE)
    draw_curve(img, Y_C, w2, GHOST, LW_LANE)
    draw_curve(img, Y_B, w1 + w2, INK, LW_ROPE)   # the sum, literally
    if i < I_GO:
        draw_chevron(img, 170.0, 294.0, right=True)
        draw_chevron(img, 910.0, 926.0, right=False)
    if I_FRZ <= i < I_R2:
        fade = float(np.clip((i - I_FRZ) / 8.0, 0.0, 1.0))
        if fade > 0:
            for d in ARROW_DS:
                draw_arrow(img, XL + d, v_flat(d), fade)
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def render_frames():
    for i in range(N_FRAMES):
        yield frame_at(i)                      # stream (trap 34)


# ---------------------------------------------------------------- measure
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


def rope_row_at(fr, x):
    """Ink centroid row of the rope in one 3-px column (fenced band;
    beads and arrows are red, not ink, so they cannot pollute it)."""
    y0 = int(Y_B) - 200
    col = fr[y0:int(Y_B) + 200, x - 1:x + 2, :].astype(np.float64)
    _, ry = centroid(ink_mask(col), 0, y0)
    return ry


def bead_boxes():
    return [(int(x) - 14, int(x) + 15, int(Y_B) - 14, int(Y_B) + 15)
            for x in NODES_IN]


def red_outside_beads(fr):
    m = red_strict(fr.astype(np.float64))
    for x0, x1, y0, y1 in bead_boxes():
        m[y0:y1, x0:x1] = False
    return int(m.sum())


# --------------------------------------------------------------- leapfrog
NG = len(DG)


def leapfrog_dirichlet(steps):
    """u_tt = c^2 u_xx, dx = 1 px, Courant number exactly 1, FIXED ends
    (the ends are nodes, so the analytic solution honours u=0 there)."""
    up = WT[np.mod(DG + 1, L)] + WT[np.mod(DG - 1, L)]     # step -1
    u = w1_at(0) + w2_at(0)
    snaps = {0: u.copy()}
    for m in range(1, steps + 1):
        un = np.zeros(NG)
        un[1:-1] = u[:-2] + u[2:] - up[1:-1]
        up, u = u, un
        if m % PXF == 0:
            snaps[m // PXF] = u.copy()
    return snaps


def leapfrog_ring(steps, with1=True, with2=True):
    """Same integrator on a 900-px ring (3 whole wavelengths), which is
    the domain where each train can run ALONE."""
    dr = np.arange(0, 900)

    def st(m):
        a = WT[np.mod(dr - m, L)] if with1 else np.zeros(900)
        b = WT[np.mod(dr + m, L)] if with2 else np.zeros(900)
        return a + b
    up, u = st(-1), st(0)
    snaps = {0: u.copy()}
    for m in range(1, steps + 1):
        un = np.roll(u, 1) + np.roll(u, -1) - up
        up, u = u, un
        if m % PXF == 0:
            snaps[m // PXF] = u.copy()
    return snaps


def ring_full(u):
    return np.concatenate([u, u[:1]])          # d=900 == d=0


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
    nd = np.array([0, 150, 300, 450, 600, 750, 900])
    worst = max(np.abs((w1_at(t) + w2_at(t))[nd]).max()
                for t in range(T_END + 1))
    ck("all 7 nodes at 0.0 BINARY-EXACT on every one of 121 motion "
       "frames", worst == 0.0, f"max |u| at nodes = {worst}")

    for t in (15, 45, 75, 105):
        assert np.all(w1_at(t) + w2_at(t) == 0.0), t
    ck("the whole rope is EXACTLY flat at t = 15, 45, 75, 105 "
       "(four binary-exact flat frames)", True)
    ck("one frame either side of a flat frame it is not flat",
       np.abs(w1_at(14) + w2_at(14)).max() > 10,
       f"max |u| at t=14: {np.abs(w1_at(14) + w2_at(14)).max():.1f} px")

    ck("exact periodicity: u(t+60) == u(t) and u(t+30) == -u(t), "
       "bitwise, all columns",
       all(np.array_equal(w1_at(t + 60) + w2_at(t + 60),
                          w1_at(t) + w2_at(t)) and
           np.array_equal(w1_at(t + 30) + w2_at(t + 30),
                          -(w1_at(t) + w2_at(t)))
           for t in range(0, 61, 5)))

    u0 = w1_at(0) + w2_at(0)
    ck("t=0 is the full pattern (peak exactly 160 px = 2A) and t=120 "
       "repeats it bitwise",
       u0.max() == 2 * A and
       np.array_equal(w1_at(T_END) + w2_at(T_END), u0))

    snaps = leapfrog_dirichlet(PXF * T_END)
    worst = max(np.abs(snaps[n] - (w1_at(n) + w2_at(n))).max()
                for n in range(T_END + 1))
    ck("independent leapfrog PDE (fixed ends, Courant 1) reproduces "
       "the drawn model", worst < 1e-9,
       f"max |num-analytic| {worst:.1e} px")

    jr = leapfrog_ring(PXF * T_END)
    s1 = leapfrog_ring(PXF * T_END, with2=False)
    s2 = leapfrog_ring(PXF * T_END, with1=False)
    worst = max(np.abs(jr[n] - (s1[n] + s2[n])).max()
                for n in range(T_END + 1))
    ck("superposition: each train integrated SOLO on a ring; the solos "
       "SUM to the joint run", worst < 1e-9, f"max {worst:.1e} px")
    worst = max(np.abs(ring_full(jr[n]) - snaps[n]).max()
                for n in range(T_END + 1))
    ck("ring joint == fixed-end joint (the walls never feel a force)",
       worst == 0.0, f"max {worst:.1e} px")

    # energy (continuous model, open-period rectangle rule)
    xf = np.linspace(0.0, 900.0, 36000, endpoint=False)
    dxf = xf[1] - xf[0]

    def energies(t):
        ct = float(PXF * t)
        a1, a2 = K * (xf - ct), K * (xf + ct)
        ut = A * C_PXS * K * (-np.cos(a1) + np.cos(a2))
        ux = A * K * (np.cos(a1) + np.cos(a2))
        return (0.5 * np.sum(ut ** 2) * dxf,
                0.5 * C_PXS ** 2 * np.sum(ux ** 2) * dxf)

    E0 = sum(energies(0))
    drift = max(abs(sum(energies(t)) - E0) / E0 for t in range(0, 121, 3))
    ke15, pe15 = energies(15)
    ke0, _ = energies(0)
    ck("energy constant; flat frames 100% KINETIC; peak frames 100% "
       "POTENTIAL (KE = 0.0 exactly)",
       drift < 1e-9 and pe15 / E0 < 1e-20 and ke0 == 0.0,
       f"drift {drift:.1e}, PE/E at flat {pe15 / E0:.1e}")

    v_ana = np.array([v_flat(x - XL) for x in ANTIN_X])
    v_fd = np.array([(w1_at(16) + w2_at(16) - w1_at(14) - w2_at(14))
                     [int(x - XL)] * FPS / 2 for x in ANTIN_X])
    ck("flat-frame velocity: analytic matches finite difference; "
       "adjacent antinodes OPPOSITE",
       np.abs(v_ana - v_fd).max() < 3.0 and
       np.all(v_ana[::2] < -400) and np.all(v_ana[1::2] > 400),
       f"|v| {np.abs(v_ana).max():.0f} px/s, fd dev "
       f"{np.abs(v_ana - v_fd).max():.2f}")

    ck("timeline: 233 frames = 7.77 s", N_FRAMES == 233
       and N_FRAMES / FPS <= 180)

    # -- pixels
    f_end = frame_at(N_FRAMES - 1)
    ink_frac = float((f_end.astype(int).sum(2) < 3 * 180).mean())
    ck("ink fraction sane on the final frame (trap 56)",
       0.002 < ink_frac < 0.30, f"{ink_frac:.4f}")

    # THE flat frame off the pixels (f51, arrows not yet faded in;
    # x-crop 110..970 excludes the end pins; beads are red, not ink)
    fr_flat = frame_at(I_FRZ)
    band = fr_flat[int(Y_B) - 200:int(Y_B) + 200, 110:970, :] \
        .astype(np.float64)
    rows = np.where(ink_mask(band).any(1))[0] + int(Y_B) - 200
    ck("flat frame off the pixels: all rope ink within +-4.5 px of the "
       "baseline", len(rows) > 0 and
       abs(rows.min() - Y_B) < 4.5 and abs(rows.max() - Y_B) < 4.5,
       f"ink rows {rows.min()}..{rows.max()}, baseline {Y_B:.0f}")

    # THE claim on the pixels: rope ink centroid pinned to the baseline
    # at all 7 node columns across the whole cycle — while antinode
    # columns swing (trap 59 control)
    node_xs = [90, 240, 390, 540, 690, 840, 990]
    worst_n = 0.0
    for fi in (90, 98, 108, 118, 128, 138, 148, 158):
        fr = frame_at(fi)
        for x in node_xs:
            worst_n = max(worst_n, abs(rope_row_at(fr, x) - Y_B))
    fr108 = frame_at(108)
    swing = abs(rope_row_at(fr108, 165) - Y_B)
    ck("rope pinned at ALL 7 node columns across a full cycle (8 frames "
       "x 7 columns) while an antinode column swings",
       worst_n < 2.5 and swing > 40.0,
       f"node worst {worst_n:.2f} px; antinode swing {swing:.0f} px")

    # profile vs model, three frames x 15 columns (no arrows on these)
    worst_p = 0.0
    for fi in (42, 100, 150):
        fr = frame_at(fi)
        t = t_at(fi)
        u = w1_at(t) + w2_at(t)
        for x in range(200, 881, 48):
            ry = rope_row_at(fr, x)
            my = Y_B - u[x - int(XL)]
            worst_p = max(worst_p, abs(ry - my))
    ck("the drawn rope IS the sum: profile vs model, 45 column reads",
       worst_p < 3.0, f"worst {worst_p:.2f} px")

    # component lanes track their trains (windows fenced: chevrons are
    # PRE-only, furniture far away)
    fr100 = frame_at(100)
    t100 = t_at(100)                       # 32: lane A crest at x=625
    crest = fr100[332:358, 575:675, :].astype(np.float64)
    cx1, _ = centroid(ghost_mask(crest), 575, 0)
    trough = fr100[712:738, 555:655, :].astype(np.float64)
    cx2, _ = centroid(ghost_mask(trough), 555, 0)
    x1_model = XL + np.argmax(w1_at(t100)[535 - 50:535 + 51]) + 485
    x2_model = XL + np.argmax(w2_at(t100)[515 - 50:515 + 51]) + 465
    ck("component lanes: each train's crest rides its model position",
       abs(cx1 - x1_model) < 3.0 and abs(cx2 - x2_model) < 3.0,
       f"errs {cx1 - x1_model:+.1f}, {cx2 - x2_model:+.1f} px")

    # red economy: beads always, arrows only in the freeze
    ck("red beads present and NOTHING else red outside the freeze",
       red_outside_beads(frame_at(0)) == 0 and
       red_outside_beads(frame_at(45)) == 0 and
       red_outside_beads(frame_at(150)) == 0 and
       red_outside_beads(frame_at(232)) == 0 and
       red_outside_beads(frame_at(67)) > 400,
       f"mid-freeze arrows {red_outside_beads(frame_at(67))} px")

    # beads never move: red centroid per bead box, whole timeline
    worst_b = 0.0
    for fi in (0, 45, 67, 108, 168, 232):
        fr = frame_at(fi).astype(np.float64)
        for (x0, x1, y0, y1), xn in zip(bead_boxes(), NODES_IN):
            bx, by = centroid(red_strict(fr[y0:y1, x0:x1, :]), x0, y0)
            worst_b = max(worst_b, abs(bx - xn), abs(by - Y_B))
    ck("the five red beads sit at exactly the node positions on every "
       "sampled frame", worst_b < 1.0, f"worst centroid err "
       f"{worst_b:.2f} px")

    # arrow directions at the freeze: antinodes 1,3,5 DOWN, 2,4,6 UP
    fr67 = frame_at(67).astype(np.float64)
    dn = red_strict(fr67[:, 130:200, :])
    up = red_strict(fr67[:, 280:350, :])
    _, ry_dn = centroid(dn, 0, 0)
    _, ry_up = centroid(up, 0, 0)
    ck("freeze arrows: antinode 1 red mass BELOW the baseline, "
       "antinode 2 ABOVE (opposite motion)",
       ry_dn > Y_B + 10 and ry_up < Y_B - 10,
       f"down-centroid {ry_dn:.0f}, up-centroid {ry_up:.0f}, "
       f"baseline {Y_B:.0f}")

    # the byte-level identities
    ck("EXACT LOOP: frames 60 apart are byte-identical off the render "
       "(t=40 vs t=100)",
       np.array_equal(frame_at(108), frame_at(168)))

    b40 = frame_at(108)[1140:1481, :, :]
    b70 = frame_at(138)[1140:1481, :, :]
    ck("half-period is a byte-exact MIRROR about the baseline "
       "(t=40 band vs t=70 band flipped)",
       np.array_equal(b70, b40[::-1, :, :]))

    f2 = frame_at(2)
    fe = frame_at(N_FRAMES - 2).copy()
    f2m = f2.copy()
    for x0, x1, y0, y1 in ((130, 210, 280, 310), (870, 950, 912, 942)):
        f2m[y0:y1, x0:x1] = 0
        fe[y0:y1, x0:x1] = 0
    ck("the video ends EXACTLY where it began (final hold == pre-hold, "
       "byte-equal outside the chevron boxes)",
       np.array_equal(f2m, fe))

    ck("chevrons in the pre-hold only",
       ink_mask(frame_at(10)[280:310, 130:210, :]
                .astype(np.float64)).sum() > 40 and
       ink_mask(fr100[280:310, 130:210, :].astype(np.float64)).sum() == 0)

    ck("equation furniture present: + and = in ink",
       ink_mask(f2[588:632, 518:562, :].astype(np.float64)).sum() > 200
       and ink_mask(f2[1034:1076, 518:562, :]
                    .astype(np.float64)).sum() > 200)

    ck("holds are truly static (byte-equal frames)",
       np.array_equal(f2, frame_at(PRE - 2)) and
       np.array_equal(frame_at(60), frame_at(I_R2 - 1)) and
       np.array_equal(frame_at(I_END + 2), f_end))

    print(f"ALL {ok} CHECKS PASSED")
    print("NOT verified by any check above (trap 68):")
    print("  - units are pixels; the wave speed is chosen for the frame."
          " the checked claims (nodes, superposition, the energy slosh)"
          " are scale-invariant; the speeds are not real-world")
    print("  - the string is IDEAL: linear, lossless, non-dispersive."
          " a real rope's standing wave decays and its nodes wander"
          " slightly")
    print("  - on a real fixed-end rope the left-moving train IS the"
          " right-moving train's reflection off the wall. here they are"
          " drawn as independent trains; the sum is identical, the"
          " bookkeeping is not")
    print("  - the amplitude is large next to the wavelength (max slope"
          " ~3.4): the linear wave equation is ASSERTED, not derived,"
          " at this steepness")
    print("  - the freeze on the flat frame is presentation: really it"
          " lasts one frame, 1/30 s")
    print("  - lanes 1 and 2 are diagram, not rope: one rope exists")


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
    band = d[int(Y_B) - 200:int(Y_B) + 200, 110:970, :].astype(np.float64)
    # trap 73 family: 4:2:0 chroma smear at the rope-through-bead
    # junctions decodes bead-EDGE pixels dark enough to read as ink
    # (render-side the beads are cleanly red, so only this copy of the
    # check needs the fence). The claim is about the ROPE: mask the
    # bead columns and measure the rope everywhere else.
    m = ink_mask(band)
    for xn in NODES_IN:
        m[:, int(xn) - 16 - 110:int(xn) + 17 - 110] = False
    rows = np.where(m.any(1))[0] + int(Y_B) - 200
    assert abs(rows.min() - Y_B) < 6 and abs(rows.max() - Y_B) < 6, \
        (rows.min(), rows.max())
    print(f"    flat frame survives the encode: rope ink rows "
          f"{rows.min()}..{rows.max()} (baseline {Y_B:.0f})")
    n_bead = int(red_strict(decode_frame(150).astype(np.float64)).sum())
    assert n_bead > 200, n_bead
    print(f"    beads survive: {n_bead} strict-red px on a motion frame")
    n_frz = int(red_strict(decode_frame(67).astype(np.float64)).sum())
    assert n_frz > n_bead + 300, (n_frz, n_bead)
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
    print(f"    {probe} frames; the nodes survive the encode")


def review_stills():
    base = OUT[:-4]
    picks = [("start", 10), ("run", 44), ("flat", 67), ("swing", 108),
             ("flip", 138), ("final", N_FRAMES - 1)]
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
