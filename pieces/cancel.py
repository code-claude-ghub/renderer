#!/usr/bin/env python3
"""CANCEL — two waves cross on one rope and, for one frame, vanish.

Drawn as an equation in three lanes: wave 1 alone (ghost), plus wave 2
alone (ghost), equals the rope (ink) — the rope is the sum, pixel for
pixel. An up-pulse runs right, an equal down-pulse runs left. When
their centers coincide the rope is COMPLETELY flat: displacement 0.0,
binary-exact, at every pixel column, for exactly one frame. Both waves
are still there — hidden in the velocity field, drawn as red arrows at
the freeze (v = -2c f'), where 100% of the energy is kinetic. Then the
pulses walk out of each other unchanged.

The deep check: the render draws the d'Alembert solution; an
INDEPENDENT leapfrog integration of the wave equation (Courant number
exactly 1) reproduces it to ~1e-12 at every motion frame, and solo
integrations of each pulse sum to the joint one (superposition,
non-vacuous by code path). Units are pixels; every checked claim is
scale-invariant. Silent.

Ninth ordinary-world classic. First wave-family piece.
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
XL, XR = 90.0, 990.0            # rope span
XC = 540.0                       # crossing point
AMP = 140.0                      # pulse amplitude, px
HW = 110.0                       # pulse half-width, px
X1_0, X2_0 = 290.0, 790.0        # start centers (up ->, down <-)
PXF = 5.0                        # exactly 5 px per frame
C_PXS = PXF * FPS                # 150 px/s

Y_A, Y_C, Y_B = 420.0, 800.0, 1310.0     # lanes: wave1 + wave2 = rope
LW_ROPE = 5.0
LW_LANE = 3.0

# timeline
PRE = 36
PH1 = 50                         # approach: 50 * 5 = 250 px each
FRZ = 32                         # freeze on the flat frame
PH2 = 50                         # separation
POST = 48
N_FRAMES = PRE + PH1 + FRZ + PH2 + POST      # 216 = 7.2 s
I_GO = PRE                                    # motion starts, f36
I_FLAT = PRE + PH1                            # THE flat frame, f86
I_R2 = I_FLAT + FRZ                           # motion resumes, f118
I_END = I_R2 + PH2                            # final hold from f168

ARROW_K = 0.2                    # arrow px per (px/s) of rope speed
ARROW_XS = [XC + 26.0 * k for k in range(-4, 5) if k != 0]

OUT = f"out/cancel_{time.strftime('%H%M%S')}.mp4"


def f(s):
    s = np.asarray(s, np.float64)
    out = np.zeros_like(s)
    m = np.abs(s) < HW
    out[m] = AMP * np.cos(np.pi * s[m] / (2 * HW)) ** 2
    return out


def fp(s):
    s = np.asarray(s, np.float64)
    out = np.zeros_like(s)
    m = np.abs(s) < HW
    out[m] = -AMP * np.pi / (2 * HW) * np.sin(np.pi * s[m] / HW)
    return out


def off_at(i):
    """Pulse advance in px at frame i (piecewise: hold/run/freeze)."""
    if i < I_GO:
        return 0.0
    if i < I_FLAT:
        return (i - I_GO) * PXF
    if i < I_R2:
        return PXF * PH1                       # 250.0 — the flat instant
    if i < I_END:
        return PXF * PH1 + (i - I_R2) * PXF
    return PXF * (PH1 + PH2)                   # 500.0


def u_rope(x, off):
    return f(x - (X1_0 + off)) - f(x - (X2_0 - off))


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
    ui = np.interp(xs, np.arange(XL, XR + 1.0), uvals)
    pts = list(zip(xs, y_lane - ui))
    x0, y0, cv = polyseg_cov(pts, lw)
    comp_bbox(img, x0, y0, cv, color)


def draw_arrow(img, x, v, fade):
    """Red velocity arrow at column x on the rope lane. v > 0 = rope
    moving UP (screen -y)."""
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
    # the equation furniture: wave1 + wave2 = rope
    for cx, cy, w_, h_ in ((XC, 610.0, 44.0, 7.0), (XC, 610.0, 7.0, 44.0),
                           (XC, 1041.0, 44.0, 7.0), (XC, 1069.0, 44.0, 7.0)):
        x0, y0, cv = rect_cov(cx, cy, w_, h_)
        comp_bbox(fr, x0, y0, cv, INK)
    # dashed start-shape outlines in the component lanes: at the end,
    # each pulse can be compared against its own starting shape
    xg = np.arange(XL, XR + 1.0)
    for y_lane, sgn, c0 in ((Y_A, 1.0, X1_0), (Y_C, -1.0, X2_0)):
        prof = sgn * f(xg - c0)
        for x in np.arange(c0 - HW, c0 + HW + 1, 2.0):
            if (int(x) // 8) % 2 == 0:
                y = y_lane - sgn * f(np.array([x - c0]))[0]
                x0, y0, cv = disc_cov(x, y, 1.6)
                comp_bbox(fr, x0, y0, cv, GHOST)
    return fr


BG = background()


# ---------------------------------------------------------------- frames
def frame_at(i):
    img = BG.copy()
    off = off_at(i)
    xg = np.arange(XL, XR + 1.0)
    draw_curve(img, Y_A, f(xg - (X1_0 + off)), GHOST, LW_LANE)
    draw_curve(img, Y_C, -f(xg - (X2_0 - off)), GHOST, LW_LANE)
    draw_curve(img, Y_B, u_rope(xg, off), INK, LW_ROPE)
    if i < I_GO:
        draw_chevron(img, X1_0, Y_A - AMP - 46.0, right=True)
        draw_chevron(img, X2_0, Y_C + AMP + 46.0, right=False)
    if I_FLAT <= i < I_R2:
        fade = float(np.clip((i - I_FLAT) / 8.0, 0.0, 1.0))
        if fade > 0:
            for x in ARROW_XS:
                v = -2 * C_PXS * fp(np.array([x - XC]))[0]
                draw_arrow(img, x, v, fade)
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
    """Ink centroid row of the rope in one 3-px column (fenced band)."""
    y0 = int(Y_B) - 200
    col = fr[y0:int(Y_B) + 200, x - 1:x + 2, :].astype(np.float64)
    _, ry = centroid(ink_mask(col), 0, y0)
    return ry


# --------------------------------------------------------------- leapfrog
NG = int(XR - XL) + 1
XGRID = XL + np.arange(NG, dtype=np.float64)


def leapfrog(with1=True, with2=True):
    """Independent integration of u_tt = c^2 u_xx, dx = 1 px,
    Courant number exactly 1 (1 px per step, 5 steps per frame)."""
    def init(o):
        a = f(XGRID - (X1_0 + o)) if with1 else np.zeros(NG)
        b = f(XGRID - (X2_0 - o)) if with2 else np.zeros(NG)
        return a - b if with2 else a
    u_prev, u = init(-1.0), init(0.0)
    snaps = {0: u.copy()}
    for m in range(1, int(PXF) * (PH1 + PH2) + 1):
        u_next = np.zeros(NG)
        u_next[1:-1] = u[:-2] + u[2:] - u_prev[1:-1]
        u_prev, u = u, u_next
        if m % int(PXF) == 0:
            snaps[m // int(PXF)] = u.copy()
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
    xs = np.arange(0.0, W)

    # -- model
    ck("flat instant: u == 0.0 BINARY-EXACT at every pixel column",
       np.all(u_rope(xs, PXF * PH1) == 0.0))
    ck("one frame either side the rope is NOT flat ('one frame' is "
       "literal)",
       np.abs(u_rope(xs, PXF * (PH1 - 1))).max() > 10 and
       np.abs(u_rope(xs, PXF * (PH1 + 1))).max() > 10,
       f"max |u| at +-1 frame = "
       f"{np.abs(u_rope(xs, PXF * (PH1 - 1))).max():.1f} px")

    joint = leapfrog()
    worst = max(np.abs(joint[n] - u_rope(XGRID, PXF * n)).max()
                for n in range(PH1 + PH2 + 1))
    ck("independent leapfrog PDE reproduces the drawn model, every "
       "motion frame", worst < 1e-9, f"max |num-analytic| {worst:.1e} px")

    solo1, solo2 = leapfrog(with2=False), leapfrog(with1=False)
    worst = max(np.abs(joint[n] - (solo1[n] + solo2[n])).max()
                for n in range(PH1 + PH2 + 1))
    ck("superposition: solo integrations SUM to the joint one "
       "(non-vacuous by code path)", worst < 1e-9,
       f"max deviation {worst:.1e} px")

    worst = max(np.abs(joint[n] + joint[n][::-1]).max()
                for n in range(PH1 + PH2 + 1))
    ck("antisymmetry about the crossing point held by the integrator",
       worst < 1e-9, f"max {worst:.1e} px")

    xf = np.linspace(XL, XR, 36001)
    dxf = xf[1] - xf[0]

    def energies(o):
        ut = C_PXS * (-fp(xf - (X1_0 + o)) - fp(xf - (X2_0 - o)))
        ux = fp(xf - (X1_0 + o)) - fp(xf - (X2_0 - o))
        return (0.5 * np.sum(ut ** 2) * dxf,
                0.5 * C_PXS ** 2 * np.sum(ux ** 2) * dxf)

    E0 = sum(energies(0.0))
    rel = max(abs(sum(energies(PXF * n)) - E0) / E0
              for n in range(0, PH1 + PH2 + 1, 5))
    ke_f, pe_f = energies(PXF * PH1)
    ck("energy constant; at the flat instant 100% KINETIC (PE = 0.0 "
       "exactly)", rel < 1e-9 and pe_f == 0.0 and
       abs(ke_f - E0) / E0 < 1e-12,
       f"drift {rel:.1e}, PE {pe_f}, KE/E {ke_f / E0:.9f}")

    vts = np.array([-2 * C_PXS * fp(np.array([x - XC]))[0]
                    for x in ARROW_XS])
    ck("arrow field: right half rises, left half sinks, antisymmetric",
       np.all(vts[4:] > 0) and np.all(vts[:4] < 0) and
       np.abs(vts + vts[::-1]).max() < 1e-9,
       f"max |v| {np.abs(vts).max():.0f} px/s")

    ck("final rope is the initial rope negated, exactly",
       np.all(u_rope(xs, PXF * (PH1 + PH2)) == -u_rope(xs, 0.0)))

    ck("timeline: 216 frames = 7.2 s", N_FRAMES == 216
       and N_FRAMES / FPS <= 180)

    # -- pixels
    f_end = frame_at(N_FRAMES - 1)
    ink_frac = float((f_end.astype(int).sum(2) < 3 * 180).mean())
    ck("ink fraction sane on the final frame (trap 56)",
       0.002 < ink_frac < 0.30, f"{ink_frac:.4f}")

    # THE flat frame: every rope-ink row within half a linewidth of Y_B
    # (band fenced: '=' ends y1073, arrows have not faded in at f86)
    fr_flat = frame_at(I_FLAT)
    band = fr_flat[int(Y_B) - 200:int(Y_B) + 200,
                   int(XL) + 20:int(XR) - 20, :].astype(np.float64)
    rows = np.where(ink_mask(band).any(1))[0] + int(Y_B) - 200
    ck("the FLAT FRAME off the pixels: all rope ink within +-4.5 px of "
       "the baseline", len(rows) > 0 and
       abs(rows.min() - Y_B) < 4.5 and abs(rows.max() - Y_B) < 4.5,
       f"ink rows {rows.min()}..{rows.max()}, baseline {Y_B:.0f}")

    fr80 = frame_at(80)
    dev = np.abs(u_rope(xs[int(XL) + 20:int(XR) - 20],
                        off_at(80))).max()
    band80 = fr80[int(Y_B) - 200:int(Y_B) + 200,
                  int(XL) + 20:int(XR) - 20, :].astype(np.float64)
    rows80 = np.where(ink_mask(band80).any(1))[0] + int(Y_B) - 200
    meas = max(Y_B - rows80.min(), rows80.max() - Y_B)
    ck("six frames earlier the rope bulges by the model amount",
       abs(meas - (dev + LW_ROPE / 2)) < 3.0,
       f"measured {meas:.1f} px, model {dev + LW_ROPE / 2:.1f}")

    # rope profile vs model at three frames x 15 columns (trap 58
    # fences: no arrows on these frames, lanes far outside the band)
    worst_p = 0.0
    for fi in (60, 82, 140):
        fr = frame_at(fi)
        for x in range(200, 881, 48):
            ry = rope_row_at(fr, x)
            my = Y_B - u_rope(np.array([float(x)]), off_at(fi))[0]
            worst_p = max(worst_p, abs(ry - my))
    ck("the drawn rope IS the superposition: profile vs model, 45 "
       "column reads", worst_p < 3.0, f"worst {worst_p:.2f} px")

    # component lanes track their pulses (columns fenced away from the
    # dashed start outlines)
    fr60 = frame_at(60)
    x1 = X1_0 + off_at(60)
    crest = fr60[int(Y_A - AMP) - 8:int(Y_A - AMP) + 18,
                 int(x1) - 50:int(x1) + 50, :].astype(np.float64)
    cx1, _ = centroid(ghost_mask(crest), int(x1) - 50, 0)
    x2 = X2_0 - off_at(60)
    trough = fr60[int(Y_C + AMP) - 18:int(Y_C + AMP) + 8,
                  int(x2) - 50:int(x2) + 50, :].astype(np.float64)
    cx2, _ = centroid(ghost_mask(trough), int(x2) - 50, 0)
    ck("component lanes: each pulse rides its own model position",
       abs(cx1 - x1) < 3.0 and abs(cx2 - x2) < 3.0,
       f"errs {cx1 - x1:+.1f}, {cx2 - x2:+.1f} px")

    # red economy: arrows exist only in the freeze
    def red_count(i):
        fr = frame_at(i)
        return int(red_strict(fr.astype(np.float64)).sum())
    ck("red arrows present mid-freeze, absent before and after "
       "(strict-red economy)",
       red_count(102) > 400 and red_count(80) == 0 and
       red_count(130) == 0 and red_count(0) == 0,
       f"mid-freeze {red_count(102)} px")

    fr102 = frame_at(102).astype(np.float64)
    right = red_strict(fr102[:, 560:660, :])
    left = red_strict(fr102[:, 420:520, :])
    _, ry_r = centroid(right, 0, 0)
    _, ry_l = centroid(left, 0, 0)
    ck("arrows point the right way: red mass above baseline on the "
       "right, below on the left",
       ry_r < Y_B - 10 and ry_l > Y_B + 10,
       f"right {ry_r:.0f}, left {ry_l:.0f}, baseline {Y_B:.0f}")

    # the mirror identity read off the BYTES: final band = initial band
    # flipped about the baseline row (band holds only rope ink + paper)
    f0 = frame_at(0)
    b0 = f0[int(Y_B) - 170:int(Y_B) + 171, :, :]
    be = f_end[int(Y_B) - 170:int(Y_B) + 171, :, :]
    dd = np.abs(be.astype(int) - b0[::-1, :, :].astype(int))
    ck("final rope band == initial band mirrored about the baseline "
       "(byte-level)", dd.max() <= 1 and dd.mean() < 0.01,
       f"max |diff| {dd.max()}, mean {dd.mean():.4f}")

    ck("chevrons in the pre-hold only",
       ink_mask(frame_at(10)[224:264, 230:350, :]
                .astype(np.float64)).sum() > 40 and
       ink_mask(fr60[224:264, 230:350, :].astype(np.float64)).sum() == 0)

    ck("equation furniture present: + and = in ink",
       ink_mask(f0[588:632, 518:562, :].astype(np.float64)).sum() > 200
       and ink_mask(f0[1034:1076, 518:562, :]
                    .astype(np.float64)).sum() > 200)

    ck("holds are truly static (byte-equal frames)",
       np.array_equal(frame_at(2), frame_at(PRE - 2)) and
       np.array_equal(frame_at(100), frame_at(I_R2 - 1)) and
       np.array_equal(frame_at(I_END + 2), f_end))

    print(f"ALL {ok} CHECKS PASSED")
    print("NOT verified by any check above (trap 68):")
    print("  - units are pixels; c is chosen for the frame. the checked"
          " claims (cancellation, superposition, the energy split) are"
          " scale-invariant; the speeds are not real-world")
    print("  - the string is IDEAL: linear, lossless, non-dispersive."
          " real ropes have stiffness and damping, so real pulses"
          " distort a little as they cross")
    print("  - the drawn amplitude is large next to the pulse width"
          " (max slope ~2). the linear wave equation is ASSERTED here;"
          " for a real rope it is the small-slope limit")
    print("  - the freeze on the flat frame is presentation: really it"
          " lasts one frame, 1/30 s")
    print("  - lanes 1 and 2 are diagram, not rope: one rope exists;"
          " the components are mathematical")


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
    d = decode_frame(I_FLAT)
    band = d[int(Y_B) - 200:int(Y_B) + 200,
             int(XL) + 20:int(XR) - 20, :].astype(np.float64)
    rows = np.where(ink_mask(band).any(1))[0] + int(Y_B) - 200
    assert abs(rows.min() - Y_B) < 6 and abs(rows.max() - Y_B) < 6, \
        (rows.min(), rows.max())
    print(f"    flat frame survives the encode: rope ink rows "
          f"{rows.min()}..{rows.max()} (baseline {Y_B:.0f})")
    d2 = decode_frame(102).astype(np.float64)
    n_red = int(red_strict(d2).sum())
    assert n_red > 300, n_red
    print(f"    velocity arrows on the freeze: {n_red} strict-red px")
    dd = np.abs(decode_frame(0).astype(int) - frame_at(0).astype(int))
    print(f"    decoded f0 vs render: mean |diff| {dd.mean():.3f}, "
          f"max {dd.max()}")
    assert dd.mean() < 2.0
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams",
         "v", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
         OUT], capture_output=True, text=True).stdout.strip()
    assert int(probe) == N_FRAMES, f"frames {probe} != {N_FRAMES}"
    print(f"    {probe} frames; the flat instant survives the encode")


def review_stills():
    base = OUT[:-4]
    picks = [("start", 10), ("approach", 60), ("almost", 84),
             ("flat", 102), ("emerge", 130), ("final", N_FRAMES - 1)]
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
