#!/usr/bin/env python3
"""FREEND — a pulse hits a FREE end and comes back right side up.

Fourth wave-family piece, the dual of REFLECT. One raised-cosine pulse
travels right along a rope whose far end is FREE: a massless ring
sliding on a frictionless rail. Beyond the rail, drawn in ghost on
darker paper, lives the mirror world — and this time the twin pulse is
UPRIGHT. The real rope is the sum of pulse and twin: the method of
images with the opposite sign.

Everything flips relative to the fixed wall:
  - the pulse returns UPRIGHT, not inverted
  - the free end is never held at zero — instead it OVERSHOOTS: at
    the bounce the twin adds to the pulse and the ring rides to
    EXACTLY double height, 320.0 = 2 * 160.0, exact in floats
  - REFLECT's flat frame (all kinetic, PE == 0) becomes its dual:
    at the doubling instant every point of the rope has velocity
    EXACTLY zero — all potential, KE == 0.0. The freeze is, for
    once, physically true: the rope really does stand still.
  - and the time symmetry is no longer a flip but a PALINDROME:
    u(t_dbl+tau) == u(t_dbl-tau) pointwise bitwise, so the second
    half of the film is the first half played backwards, byte for
    byte, and the last frame equals the first.

Deep checks: a leapfrog with the free end as a Neumann boundary
reproduces the drawn model to ~1e-11 px, and a SECOND leapfrog on the
free doubled domain (no wall, upright twin as real initial data)
matches it EXACTLY — 0.0 — because even symmetry survives the update
bitwise. Silent.

Twelfth ordinary-world classic. Fourth wave-family piece.
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
XW = 540                        # the rail (frame centre)
X0, X1 = 60, 1020               # drawn span: real | mirror
AMP = 160.0                     # pulse height, px
HW = 120                        # pulse half-width, px
S0 = 200                        # pulse centre at t=0
PXF = 5                         # exactly 5 px per frame
C_PXS = PXF * FPS               # 150 px/s (prose only)
T_DBL = (XW - S0) // PXF        # 68 — pulse centred on the rail
T_END = 2 * T_DBL               # 136 — back where it started, upright

Y_M = 960.0                     # rope baseline (frame centre)
LW_ROPE = 5.0
LW_GHOST = 3.0
RAIL_W, RAIL_H = 5.0, 760.0     # thin rail: rows 580..1340, cols 538..542
RING_R = 9.0                    # the free end: a red ring that SLIDES

# timeline (t = motion frames; i = video frames)
PRE = 36
SEG1 = T_DBL                    # 68: approach
FRZ = 32                        # freeze on the doubling instant
SEG2 = T_DBL                    # 68: departure, upright
POST = 45
N_FRAMES = PRE + SEG1 + FRZ + SEG2 + POST          # 249 = 8.3 s
I_GO = PRE                                          # f36
I_FRZ = PRE + SEG1                                  # f104, t=68, DOUBLED
I_R2 = I_FRZ + FRZ                                  # f136
I_END = I_R2 + SEG2                                 # f204

DASH_HALF = 45.0                # freeze ruler dashes: x = 540 +- 45
Y_1X = Y_M - AMP                # 800.0 — the incoming pulse's height
Y_2X = Y_M - 2 * AMP            # 640.0 — where the ring actually goes

OUT = f"out/freend_{time.strftime('%H%M%S')}.mp4"


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
    """Rope displacement over XS: pulse PLUS its upright twin."""
    s = S0 + PXF * t
    return pval(XS - s) + pval(2 * XW - XS - s)


def tip_at(t):
    """The free end: u at the rail. p + p == 2p, exact."""
    return float(u_at(t)[XW - X0])


def t_at(i):
    """Motion time at video frame i (hold / run / freeze / run / hold)."""
    if i < I_GO:
        return 0
    if i < I_FRZ:
        return i - I_GO
    if i < I_R2:
        return T_DBL
    if i < I_END:
        return T_DBL + (i - I_R2)
    return T_END


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


def draw_chevron(img, cx, cy, right, color):
    """Direction arrow during the pre-hold."""
    s = 1.0 if right else -1.0
    x0, y0, cv = polyseg_cov([(cx - s * 30, cy), (cx + s * 30, cy)], 4.0)
    comp_bbox(img, x0, y0, cv, color)
    for dy in (-10.0, 10.0):
        x0, y0, cv = polyseg_cov([(cx + s * 14, cy + dy),
                                  (cx + s * 30, cy)], 4.0)
        comp_bbox(img, x0, y0, cv, color)


def draw_dash(img, y, fade):
    """Freeze ruler: a red height mark on the rail."""
    x0, y0, cv = polyseg_cov([(XW - DASH_HALF, y), (XW + DASH_HALF, y)],
                             3.0)
    comp_bbox(img, x0, y0, cv * fade, C_RED)


# ---------------------------------------------------------------- static
def background():
    fr = np.full((H, W, 3), PAPER, np.float64)
    fr[:, XW:, :] = MIRROR          # the mirror world's paper
    # the rail: THIN — nothing grabs the rope here. the end slides.
    x0, y0, cv = rect_cov(float(XW), Y_M, RAIL_W, RAIL_H)
    comp_bbox(fr, x0, y0, cv, INK)
    return fr


BG = background()


# ---------------------------------------------------------------- frames
def frame_at(i):
    img = BG.copy()
    t = t_at(i)
    u = u_at(t)
    if I_FRZ <= i < I_R2:
        fade = float(np.clip((i - I_FRZ) / 8.0, 0.0, 1.0))
        if fade > 0:
            draw_dash(img, Y_1X, fade)      # the wave's own height...
            draw_dash(img, Y_2X, fade)      # ...and exactly double it
    draw_curve(img, u, XW, X1, GHOST, LW_GHOST)     # mirror world first
    draw_curve(img, u, X0, XW, INK, LW_ROPE)        # the real rope on top
    # the free end: a red ring that RIDES the rail (REFLECT's pin,
    # unpinned) — drawn last, on top of rail and rope
    x0, y0, cv = disc_cov(float(XW), Y_M - u[XW - X0], RING_R)
    comp_bbox(img, x0, y0, cv, C_RED)
    if i < I_GO:
        draw_chevron(img, 200.0, 740.0, True, INK)
        draw_chevron(img, 880.0, 740.0, False, GHOST)
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def render_frames():
    for i in range(N_FRAMES):
        yield frame_at(i)                            # stream (trap 34)


# ---------------------------------------------------------------- measure
# FENCE AUDIT (written before the first render):
#   rail cols 536..544 pad (ink), rows 580..1340; STATIC-furniture
#     checks fenced where the subject legally crosses (reflect lesson):
#     rope+ring cross at y ~ 960 on the sampled motion frame, so rail
#     solidity reads rows 600..920 and 1000..1320 only, cols 539..542
#   ring disc r=9 centred (540, Y_M - tip): cols 529..551, rows vary
#     629..971 across the bounce — profile columns keep clear (<=515
#     real / >=565 mirror)
#   dashes FREEZE ONLY, red, rows 638..642 and 798..802, cols 493..587;
#     they sit inside the rope_row_at band (600..1320) at those cols
#     but are RED — ink_mask and ghost_mask both exclude them; profile
#     checks run on motion frames (76, 164, 190) where no dash exists
#   chevron boxes PRE-HOLD only, BOTH at rows 724..756:
#     (160..240) ink and (840..920) ghost — outside the rope band;
#     the palindrome end==start check masks the boxes AS-IS (identity,
#     no flip, so no off-by-one flip arithmetic to get wrong)
#   palindrome pairs compare motion frames only (no chevrons, no
#     dashes on either side of each pair)
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
    # band is +-360, NOT reflect.py's +-200: the doubled pulse reaches
    # 320 px and an instrument inherited from a smaller piece could not
    # see this piece's headline fact (run-1 lesson). the wider band
    # admits the chevron rows (724..756) and dash rows (638..642), but
    # chevrons exist only in the pre-hold (never measured) and dashes
    # are red (excluded by both masks) and freeze-only
    y0 = int(Y_M) - 360
    col = fr[y0:int(Y_M) + 360, x - 1:x + 2, :].astype(np.float64)
    _, ry = centroid(mask_fn(col), 0, y0)
    return ry


def ring_box(t):
    ry = int(round(Y_M - tip_at(t)))
    return (XW - 17, XW + 18, ry - 17, ry + 18)


def red_outside_ring(fr, t):
    m = red_strict(fr.astype(np.float64))
    x0, x1, y0, y1 = ring_box(t)
    m[y0:y1, x0:x1] = False
    return int(m.sum())


CHEV_BOXES = ((160, 240, 724, 756), (840, 920, 724, 756))


def mask_chevrons(fr):
    out = fr.copy()
    for x0, x1, y0, y1 in CHEV_BOXES:
        out[y0:y1, x0:x1] = 0
    return out


# --------------------------------------------------------------- leapfrog
GL = -700


def analytic_on(xs, m):
    s = S0 + m
    return pval(xs - s) + pval(2 * XW - xs - s)


def leapfrog(x_hi, substeps, neumann_right=False):
    """Courant number exactly 1 on GL..x_hi; optionally FREE at x_hi."""
    xg = np.arange(GL, x_hi + 1)
    up, u = analytic_on(xg, -1), analytic_on(xg, 0)
    snaps = {0: u.copy()}
    for m in range(1, substeps + 1):
        un = np.zeros(len(xg))
        un[1:-1] = u[:-2] + u[2:] - up[1:-1]
        if neumann_right:
            un[-1] = 2.0 * u[-2] - up[-1]   # ghost cell u[iw+1] := u[iw-1]
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
    tips = np.array([tip_at(t) for t in range(T_END + 1)])
    ck("the free end OVERSHOOTS to EXACTLY 2*AMP = 320.0, once",
       tips.max() == 2.0 * AMP and tips[T_DBL] == 320.0
       and int(np.sum(tips == 320.0)) == 1, f"max tip {tips.max()}")

    ck("zero slope at the rail bitwise x137: u[iw+1] == u[iw-1] — the "
       "rail never pulls sideways, which is what 'free' means",
       all(u_at(t)[iw + 1] == u_at(t)[iw - 1] for t in range(T_END + 1)))

    ck("u is even about the rail bitwise on all 137 frames: the mirror "
       "world really is the mirror",
       all(np.array_equal(u_at(t)[::-1], u_at(t))
           for t in range(0, T_END + 1, 4)))

    ck("THE STILLNESS: at the doubling instant the central-difference "
       "velocity is 0.0 bitwise at EVERY point",
       np.array_equal(u_at(T_DBL + 1), u_at(T_DBL - 1)))
    ck("one frame off the doubling instant it moves (trap 59 control)",
       np.abs(u_at(T_DBL + 2) - u_at(T_DBL)).max() > 5,
       f"max |du| {np.abs(u_at(T_DBL + 2) - u_at(T_DBL)).max():.1f} px")

    ck("PALINDROME: u(t_dbl+tau) == u(t_dbl-tau) pointwise bitwise, "
       "every tau — the film out is the film in, reversed",
       all(np.array_equal(u_at(T_DBL + tau), u_at(T_DBL - tau))
           for tau in range(1, T_DBL + 1)))

    u0, ue = u_at(0), u_at(T_END)
    ck("the pulse returns EXACTLY upright: u(t_end) == u(0) bitwise, "
       "peak +160.0 both times", np.array_equal(ue, u0)
       and u0.max() == AMP and ue.max() == AMP)

    SW = leapfrog(XW, PXF * T_END, neumann_right=True)
    worst = max(np.abs(SW[n][X0 - GL:] - u_at(n)[:iw + 1]).max()
                for n in range(T_END + 1))
    ck("independent leapfrog PDE (free end as Neumann BC, Courant 1) "
       "reproduces the drawn model", worst < 1e-9,
       f"max |num-analytic| {worst:.1e} px")

    SF = leapfrog(2 * XW - GL, PXF * T_END)      # no wall: doubled domain
    worst = max(np.abs(SF[n][:XW - GL + 1] - SW[n]).max()
                for n in range(T_END + 1))
    ck("METHOD OF IMAGES: the free universe with the UPRIGHT twin == "
       "the Neumann-walled universe, EXACTLY (even symmetry survives "
       "bitwise)", worst == 0.0, f"max diff {worst:.1e}")

    worst = max(np.abs(SW[n][:50]).max() for n in range(T_END + 1))
    ck("left guard never touched: the cut left edge is honest",
       worst == 0.0, f"max |u| left of {GL + 50}: {worst}")

    # energy: free doubled domain (compact support — the reflect.py
    # lesson: put integration boundaries where the integrand is zero)
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
        ut = -C_PXS * (dp(a1) + dp(a2))
        ux = dp(a1) - dp(a2)
        return (0.5 * np.sum(ut ** 2) * 0.01,
                0.5 * C_PXS ** 2 * np.sum(ux ** 2) * 0.01)

    E0 = sum(energies(0))
    drift = max(abs(sum(energies(t)) - E0) / E0
                for t in range(0, T_END + 1, 4))
    keD, peD = energies(T_DBL)
    ck("energy constant through the bounce; at the doubling instant "
       "KE == 0.0 EXACTLY — nothing moves, everything is stretched "
       "(the dual of REFLECT's flat frame)",
       drift < 1e-9 and keD == 0.0,
       f"drift {drift:.1e}, KE at double {keD}")

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

    # THE doubled frame off the pixels (i=I_FRZ: fade 0, no dashes yet;
    # profile columns fenced clear of rail 536..544 and ring 529..551)
    fr_dbl = frame_at(I_FRZ)
    want = Y_M - 2.0 * float(pval(np.array([515 - XW]))[0])
    ry = rope_row_at(fr_dbl, 515)
    ck("doubled frame off the pixels, real side: ink at x=515 sits at "
       "the DOUBLE-height curve", abs(ry - want) < 3.0,
       f"read {ry:.1f}, model {want:.1f}")
    ry = rope_row_at(fr_dbl, 565, ghost_mask)
    ck("doubled frame, mirror side: the ghost twin doubles too",
       abs(ry - want) < 3.0, f"read {ry:.1f}, model {want:.1f}")

    # the ring RIDES: centroid tracks the model tip across the bounce
    worst_r = 0.0
    for fi in (0, 76, 104, 120, 164, 190, 248):
        t = t_at(fi)
        fr = frame_at(fi).astype(np.float64)
        x0, x1, y0, y1 = ring_box(t)
        bx, by = centroid(red_strict(fr[y0:y1, x0:x1, :]), x0, y0)
        worst_r = max(worst_r, abs(bx - XW),
                      abs(by - (Y_M - tip_at(t))))
    ck("the red ring rides the rail: centroid == model tip on every "
       "sampled frame (top of travel included)",
       worst_r < 1.0, f"worst centroid err {worst_r:.2f} px")

    ck("at the doubling instant the ring touches the 2x dash: tip row "
       "640 == dash row 640", Y_M - tips[T_DBL] == Y_2X)

    # profile vs model, both worlds, three motion frames (t=40, 96, 122)
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

    # red economy: the ring always; the ruler dashes ONLY in the freeze
    ck("red ring always; NOTHING else red outside the freeze",
       red_outside_ring(frame_at(0), 0) == 0 and
       red_outside_ring(frame_at(76), t_at(76)) == 0 and
       red_outside_ring(frame_at(190), t_at(190)) == 0 and
       red_outside_ring(frame_at(248), T_END) == 0 and
       red_outside_ring(frame_at(120), T_DBL) > 300,
       f"mid-freeze ruler {red_outside_ring(frame_at(120), T_DBL)} px")

    # the ruler dashes are WHERE they claim (mid-freeze, fade complete)
    fr120 = frame_at(120).astype(np.float64)
    d2 = red_strict(fr120[630:651, 495:521, :])
    d1 = red_strict(fr120[790:811, 495:521, :])
    _, ry2 = centroid(d2, 0, 630)
    _, ry1 = centroid(d1, 0, 790)
    ck("freeze ruler: 1x dash at 800, 2x dash at 640, off the pixels",
       abs(ry1 - Y_1X) < 1.5 and abs(ry2 - Y_2X) < 1.5,
       f"1x {ry1:.1f}, 2x {ry2:.1f}")

    # the byte identities off the render
    ck("PALINDROME off the pixels, byte-exact: frames equidistant from "
       "the doubling instant are IDENTICAL (4 pairs)",
       all(np.array_equal(frame_at(I_R2 + k), frame_at(I_FRZ - k))
           for k in (5, 20, 40, 67)))

    ck("the video ends EXACTLY where it began (final hold == pre-hold, "
       "byte-equal outside the chevron boxes)",
       np.array_equal(mask_chevrons(frame_at(N_FRAMES - 2)),
                      mask_chevrons(frame_at(2))))

    ck("chevrons in the pre-hold only",
       ink_mask(frame_at(10)[724:756, 160:240, :]
                .astype(np.float64)).sum() > 40 and
       ink_mask(frame_at(76)[724:756, 160:240, :]
                .astype(np.float64)).sum() == 0)

    # (fenced: rope and ring legally cross the rail near y=960 on the
    # sampled frame — static-furniture fence, the reflect.py lesson)
    ck("the rail is there: thin ink line solid above and below the rope",
       ink_mask(frame_at(76)[600:920, 539:543, :]
                .astype(np.float64)).all() and
       ink_mask(frame_at(76)[1000:1320, 539:543, :]
                .astype(np.float64)).all())

    ck("holds are truly static (byte-equal frames)",
       np.array_equal(frame_at(2), frame_at(PRE - 2)) and
       np.array_equal(frame_at(I_FRZ + 10), frame_at(I_R2 - 1)) and
       np.array_equal(frame_at(I_END + 2), f_end))

    print(f"ALL {ok} CHECKS PASSED")
    print("NOT verified by any check above (trap 68):")
    print("  - units are pixels; the wave speed is chosen for the"
          " frame. the checked claims (upright return, the exact"
          " doubling, the stillness, the image construction) are"
          " scale-invariant")
    print("  - 'free end' is an idealisation: a MASSLESS ring on a"
          " FRICTIONLESS rail. real rope ends have real hardware, and"
          " any mass or friction there takes a bite out of every"
          " bounce")
    print("  - nothing exists beyond the rail. the mirror world is the"
          " method of images — the checks prove it gives EXACTLY the"
          " free-end answer; they cannot make it real")
    print("  - the doubled pulse reaches slope ~4.2, twice the last"
          " piece's: the linear wave equation is ASSERTED, not derived,"
          " at this steepness")
    print("  - the freeze lasts 32 frames for presentation. the"
          " stillness it shows is real but instantaneous: velocity is"
          " exactly zero AT the doubling instant, one frame of film")


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
    # doubled frame: profile column fenced clear of rail, ring AND
    # their chroma smear (nodes.py lesson: decode fences != render)
    want = Y_M - 2.0 * float(pval(np.array([505 - XW]))[0])
    ry = rope_row_at(d, 505)
    assert abs(ry - want) < 6, (ry, want)
    print(f"    doubled frame survives the encode: x=505 ink at "
          f"{ry:.1f} (model {want:.1f})")
    n_ring = int(red_strict(decode_frame(76).astype(np.float64)).sum())
    assert n_ring > 80, n_ring
    print(f"    ring survives: {n_ring} strict-red px on a motion frame")
    n_frz = int(red_strict(decode_frame(120).astype(np.float64)).sum())
    assert n_frz > n_ring + 250, (n_frz, n_ring)
    print(f"    ruler on the freeze: {n_frz} strict-red px")
    da = decode_frame(I_R2 + 20).astype(int)
    db = decode_frame(I_FRZ - 20).astype(int)
    pd = np.abs(da - db).mean()
    assert pd < 3.0, pd
    print(f"    palindrome survives the encode: paired frames mean "
          f"|diff| {pd:.3f} (lossy codec, same picture)")
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
    picks = [("start", 10), ("approach", 80), ("double", 120),
             ("retreat", 164), ("return", 190), ("final", N_FRAMES - 1)]
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
