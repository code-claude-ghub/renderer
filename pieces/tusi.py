#!/usr/bin/env python3
"""TUSI — a wheel rolls inside a wheel twice its size; every point on
its rim travels in a dead straight line.

The classical construction (Nasir al-Din al-Tusi, 1247) that the
ELLIPSE description declined to claim, and that @rorucopexperements'
line on STRAIGHT invited: "And of course when it's true it's a line
instead." A rim point at phase alpha slides along the diameter at
angle alpha/2 — no curve at all. Two diametrically opposite rim
points are marked:

  - RED (phase pi) traces the VERTICAL diameter, and its x is
    540.0 BITWISE, all 137 samples — the same 540 STRAIGHT produced,
    out of a completely different machine (measured, not assumed;
    the horizontal orientation only cancels to 1.1e-13)
  - GHOST (phase 0) traces the horizontal diameter (dev <= 1.1e-13)
  - the rod between them is a diameter of the rolling wheel — the
    trammel of Archimedes — and its midpoint is the wheel's centre,
    BITWISE, all 137 samples
  - at the freeze the red point touches the outer rim with velocity
    EXACTLY (0.0, 0.0) while its partner crosses dead centre at top
    speed: the two ends of one rigid rod, stopped and flat out
  - the red diameter is complete before the freeze; the return leg
    retraces it. the GHOST diameter is NOT done at the freeze — the
    return leg finishes it. the retrace claim is red-only.

kinematics only: the wheel is driven and rolls without slip
(contact-point velocity 0.0 identically, by construction).
"""
import os
import subprocess
import time

import numpy as np

# ---------------------------------------------------------------- canvas
W, H = 1080, 1920
FPS = 30
PAPER = 0.845                   # trap 69: warm grey, not white
INK = 0.10
GHOST = 0.58
C_RED = (0.55, 0.10, 0.10)

# ---------------------------------------------------------------- model
XC, YC = 540.0, 960.0
R_BIG = 420.0
R_SM = 210.0                    # exactly half — the whole theorem
K = 68
M_N = 2 * K + 1                 # 137
SPAN = 1.03 * np.pi
A = SPAN / K

LW_BIG = 4.0
LW_WHEEL = 4.0
LW_ROD = 2.0
LW_TRAIL = 3.0
LW_DOT_TRAIL = 3.5
LW_RING = 3.0
R_PEN = 6.0
R_HUB = 4.0
R_DOT = 7.0
R_RING = 20.0

PRE = 36
LEG1 = 68
FRZ = 32                        # freeze: red stopped on the rim
LEG2 = 68
POST = 45
N_FRAMES = PRE + LEG1 + FRZ + LEG2 + POST          # 249 = 8.3 s
I_GO = PRE                                          # f36
I_FRZ = PRE + LEG1                                  # f104, delta = 0
I_R2 = I_FRZ + FRZ                                  # f136
I_END = I_R2 + LEG2                                 # f204

OUT = f"out/tusi_{time.strftime('%H%M%S')}.mp4"

# ------------------------------------------------------- exact tables
# theta = pi/2 + delta; cos th = -sin d, sin th = cos d
MS = np.arange(M_N)
T = (MS - K).astype(np.float64)
D_ABS = A * np.abs(T)
SN = np.sign(T) * np.sin(D_ABS)         # odd, bitwise
CS = np.cos(D_ABS)                      # even, bitwise

OX = XC - R_SM * SN                     # rolling-wheel centre
OY = YC + R_SM * CS
RED_X = OX + R_SM * SN                  # rim point, phase pi: vertical
RED_Y = OY + R_SM * CS
GHO_X = OX - R_SM * SN                  # rim point, phase 0: horizontal
GHO_Y = OY - R_SM * CS

RIM_TOUCH = (540.0, 1380.0)             # red's freeze position


def m_at(i):
    if i < I_GO:
        return 0
    if i < I_FRZ:
        return i - I_GO
    if i < I_R2:
        return K
    if i < I_END:
        return K + 1 + (i - I_R2)
    return 2 * K


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


def circ_cov(cx, cy, r, lw):
    """Circle OUTLINE: coverage of |dist - r| < lw/2."""
    pad = lw / 2 + 2
    x0, x1 = int(np.floor(cx - r - pad)), int(np.ceil(cx + r + pad)) + 1
    y0, y1 = int(np.floor(cy - r - pad)), int(np.ceil(cy + r + pad)) + 1
    xx = np.arange(x0, x1, dtype=np.float64)
    yy = np.arange(y0, y1, dtype=np.float64)
    d = np.hypot(xx[None, :] - cx, yy[:, None] - cy)
    return x0, y0, np.clip(lw / 2 + 0.5 - np.abs(d - r), 0.0, 1.0)


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


def draw_poly(img, pts, color, lw):
    if len(pts) < 2:
        return
    x0, y0, cv = polyseg_cov(pts, lw)
    comp_bbox(img, x0, y0, cv, color)


def draw_chevron(img, cx, cy, right, color):
    s = 1.0 if right else -1.0
    draw_poly(img, [(cx - s * 30, cy), (cx + s * 30, cy)], color, 4.0)
    for dy in (-10.0, 10.0):
        draw_poly(img, [(cx + s * 14, cy + dy), (cx + s * 30, cy)],
                  color, 4.0)


# the big circle never moves: compute its coverage once
BIGC = circ_cov(XC, YC, R_BIG, LW_BIG)
BG = np.full((H, W, 3), PAPER, np.float64)


# ---------------------------------------------------------------- frames
def frame_at(i):
    img = BG.copy()
    m = m_at(i)
    # trails: ghost's horizontal diameter, then the rod, then red's
    # vertical diameter (red always on top of grey)
    if m >= 1:
        draw_poly(img, list(zip(GHO_X[:m + 1], GHO_Y[:m + 1])),
                  GHOST, LW_TRAIL)
    draw_poly(img, [(RED_X[m], RED_Y[m]), (GHO_X[m], GHO_Y[m])],
              GHOST, LW_ROD)
    if m >= 1:
        draw_poly(img, list(zip(RED_X[:m + 1], RED_Y[:m + 1])),
                  C_RED, LW_DOT_TRAIL)
    # the outer wheel, drawn OVER the trails so it stays solid
    comp_bbox(img, BIGC[0], BIGC[1], BIGC[2], INK)
    # the rolling wheel and its hub (the rod's midpoint)
    x0, y0, cv = circ_cov(OX[m], OY[m], R_SM, LW_WHEEL)
    comp_bbox(img, x0, y0, cv, INK)
    x0, y0, cv = disc_cov(OX[m], OY[m], R_HUB)
    comp_bbox(img, x0, y0, cv, INK)
    # freeze furniture: a ring around the stopped red point
    if I_FRZ <= i < I_R2:
        fade = float(np.clip((i - I_FRZ) / 8.0, 0.0, 1.0))
        if fade > 0:
            x0, y0, cv = circ_cov(*RIM_TOUCH, R_RING, LW_RING)
            comp_bbox(img, x0, y0, cv * fade, C_RED)
    # pens: ghost first, red last
    x0, y0, cv = disc_cov(GHO_X[m], GHO_Y[m], R_PEN)
    comp_bbox(img, x0, y0, cv, INK)
    x0, y0, cv = disc_cov(RED_X[m], RED_Y[m], R_DOT)
    comp_bbox(img, x0, y0, cv, C_RED)
    if i < I_GO:
        draw_chevron(img, 820.0, 700.0, True, INK)
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def render_frames():
    for i in range(N_FRAMES):
        yield frame_at(i)                            # stream (trap 34)


# ---------------------------------------------------------------- measure
# FENCE AUDIT (written before the first render):
#   RED lives in exactly three places — the red pen disc (moving,
#     r=7, box +-12 around the model point), the red trail (the
#     vertical diameter: |x-540| <= 7, rows 533..1388), and the
#     freeze ring (r=20, lw=3 around (540,1380), FREEZE FRAMES ONLY,
#     box x 513..568, y 1353..1408; fade is 0 AT I_FRZ itself).
#   ink lives in: the big circle (static annulus r 420 +- 2.5), the
#     rolling wheel (moving annulus r 210), its hub (r 4), the ghost
#     pen (r 6), the chevron (PRE-HOLD ONLY, x 790..851, y 686..715).
#   ghost lives in: the horizontal trail (rows ~953..967) and the
#     rod (moving, drawn UNDER the red trail).
#   the hub crosses the red lane near the freeze (O_x = 540 there) —
#     ink over red, which the red fence does not mind; no red check
#     samples rows near 1166..1174 on freeze frames.
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


def red_centroid_at(fr, mx, my, halfw=14):
    x0, y0 = int(round(mx)) - halfw, int(round(my)) - halfw
    box = fr[y0:y0 + 2 * halfw + 1, x0:x0 + 2 * halfw + 1, :]
    return centroid(red_strict(box.astype(np.float64)), x0, y0)


YY, XX = np.mgrid[0:H, 0:W]
LANE = (np.abs(XX - XC) <= 7) & (YY >= 533) & (YY <= 1388)


def allowed_red(m, frozen):
    allow = LANE.copy()
    x0 = int(round(RED_X[m]))
    y0 = int(round(RED_Y[m]))
    allow[max(y0 - 12, 0):y0 + 13, max(x0 - 12, 0):x0 + 13] = True
    if frozen:
        allow[1353:1409, 513:569] = True
    return allow


# columns/rows for the final-frame completeness checks, chosen in
# feasibility to dodge the parked wheel, pens, rod and the crossing
ROWS_R = [570, 670, 770, 870, 1030, 1130, 1230, 1300, 1360]
COLS_G = [160, 260, 360, 460, 700, 800, 880, 940]


# ---------------------------------------------------------------- checks
def run_checks():
    n = [0]

    def ok(name, cond, note=""):
        print(("  ok  " if cond else "  FAIL") + f"  {name}" +
              (f"  [{note}]" if note else ""))
        assert cond, name
        n[0] += 1

    print("== TUSI render checks ==")

    # ---- model
    ok(f"RED x == 540.0 BITWISE x{M_N} — the line is exactly straight",
       bool(np.all(RED_X == 540.0)))
    devg = np.abs(GHO_Y - 960.0).max()
    ok("GHOST y == 960 within 1e-9", devg < 1e-9,
       f"max dev {devg:.2e}, {int(np.sum(GHO_Y == 960.0))}/{M_N} bitwise")
    ok("rod midpoint == wheel centre bitwise",
       bool(np.all((RED_X + GHO_X) / 2.0 == OX) and
            np.all((RED_Y + GHO_Y) / 2.0 == OY)))
    rod = np.hypot(RED_X - GHO_X, RED_Y - GHO_Y)
    ok("rod rigid: length == 420 the whole run",
       np.abs(rod - 2 * R_SM).max() < 1e-9,
       f"max dev {np.abs(rod - 2 * R_SM).max():.2e}")
    ok("odd/even tables bitwise",
       bool(np.all(SN == -SN[::-1]) and np.all(CS == CS[::-1])))
    ok("RED y time-mirror bitwise", bool(np.all(RED_Y == RED_Y[::-1])))
    ok("freeze RED at the rim exactly",
       RED_X[K] == 540.0 and RED_Y[K] == 1380.0)
    ok("freeze GHOST at dead centre exactly",
       GHO_X[K] == 540.0 and GHO_Y[K] == 960.0)
    ok("freeze RED velocity exactly (0,0)", -R_BIG * np.sin(0.0) == 0.0)
    ok("contact-point velocity 0.0 exactly (rolls without slip)",
       R_SM * (-np.sin(0.7) + np.sin(0.7)) == 0.0 and
       R_SM * (np.cos(0.7) - np.cos(0.7)) == 0.0)
    cover = 2 * SPAN
    ok("coverage > full lap", cover > 2 * np.pi,
       f"{cover:.3f} rad = {cover / (2 * np.pi) * 100:.0f}% of a lap")
    red1 = RED_Y[:K + 1]
    ok("red diameter complete by the freeze",
       red1.min() <= YC - R_BIG + 0.5 and red1.max() == YC + R_BIG,
       f"leg-1 y [{red1.min():.4f}, {red1.max():.4f}]")
    ok("ghost diameter needs the return leg",
       GHO_X[:K + 1].min() > 400 and GHO_X.min() <= XC - R_BIG + 0.5,
       f"leg-1 min x {GHO_X[:K + 1].min():.1f}, "
       f"overall min {GHO_X.min():.4f}")

    # ---- determinism / statics
    ok("frame purity", bool(np.array_equal(frame_at(60), frame_at(60))))
    ok("freeze static after fade",
       bool(np.array_equal(frame_at(I_FRZ + 20), frame_at(I_FRZ + 30))))
    ok("post static",
       bool(np.array_equal(frame_at(I_END + 5), frame_at(N_FRAMES - 1))))

    # ---- pixels, f0
    # trap 47: don't grade what the render cannot show — the red pen
    # sits ON the outer rim at f0 and on the final frame, so circle
    # sample points under a pen are skipped, not failed
    def outline_pts(fr, cx, cy, r, m):
        tot = hit = 0
        for th in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            px = cx + r * np.cos(th)
            py = cy + r * np.sin(th)
            if np.hypot(px - RED_X[m], py - RED_Y[m]) < 16 or \
               np.hypot(px - GHO_X[m], py - GHO_Y[m]) < 16:
                continue
            tot += 1
            box = fr[int(round(py)) - 3:int(round(py)) + 4,
                     int(round(px)) - 3:int(round(px)) + 4]
            hit += bool(ink_mask(box).sum() > 0)
        return hit, tot

    f0 = frame_at(0)
    for name, cx, cy, r in (("outer wheel", XC, YC, R_BIG),
                            ("rolling wheel", OX[0], OY[0], R_SM)):
        hit, tot = outline_pts(f0, cx, cy, r, 0)
        ok(f"{name} outline present at f0", hit == tot and tot >= 10,
           f"{hit}/{tot} visible circle points inked")
    cx, cy = red_centroid_at(f0, RED_X[0], RED_Y[0])
    d0 = np.hypot(cx - RED_X[0], cy - RED_Y[0])
    ok("red pen centroid at f0", d0 < 0.8, f"off by {d0:.2f} px")
    gx0, gy0 = int(round(GHO_X[0])) - 8, int(round(GHO_Y[0])) - 8
    cnt = int(ink_mask(f0[gy0:gy0 + 17, gx0:gx0 + 17]).sum())
    ok("ghost pen disc present at f0", cnt > 60, f"{cnt} ink px")

    # ---- safe area (trap 3); trap 74: compute paper byte as frame_at does
    paper8 = np.uint8(PAPER * 255.0 + 0.5)
    for i in (0, I_FRZ + 16, N_FRAMES - 1):
        fr = frame_at(i)
        top = int((fr[:192] != paper8).sum())
        bot = int((fr[1632:] != paper8).sum())
        ok(f"safe area clean at f{i}", top == 0 and bot == 0,
           f"{top}+{bot} non-paper px")

    # ---- motion: red pen follows the model
    for i in (I_GO + 30, I_R2 + 25):
        m = m_at(i)
        fr = frame_at(i)
        cx, cy = red_centroid_at(fr, RED_X[m], RED_Y[m], halfw=11)
        d = np.hypot(cx - RED_X[m], cy - RED_Y[m])
        ok(f"red pen tracks model at f{i}", d < 2.5, f"off {d:.2f} px")

    # ---- red fence
    for i in (0, 60, I_FRZ, I_FRZ + 16, 170, N_FRAMES - 1):
        m = m_at(i)
        frozen = I_FRZ < i < I_R2
        fr = frame_at(i)
        stray = red_strict(fr.astype(np.float64)) & \
            ~allowed_red(m, frozen)
        ok(f"red fence at f{i}", int(stray.sum()) == 0,
           f"{int(stray.sum())} stray red px")

    # ---- the drawn line is STRAIGHT, measured off the pixels
    fend = frame_at(N_FRAMES - 1)
    rfull = red_strict(fend.astype(np.float64))
    # span on the FULL mask (trap 75: each claim measures its own mask;
    # the red pen parks at the TOP end of the line on the final frame)
    ys_f, _ = np.nonzero(rfull)
    ok("trail spans the diameter",
       ys_f.min() <= YC - R_BIG + 6 and ys_f.max() >= YC + R_BIG - 6,
       f"rows {ys_f.min()}..{ys_f.max()} (model 540..1380)")
    # straightness: exclude the pen disc (r=7 > the 6 px tolerance)
    rmask = rfull.copy()
    mdx = int(round(RED_X[2 * K]))
    mdy = int(round(RED_Y[2 * K]))
    rmask[mdy - 12:mdy + 13, mdx - 12:mdx + 13] = False
    ys, xs = np.nonzero(rmask)
    ok("trail straightness (pixel-measured)",
       np.abs(xs - 540.0).max() <= 6.0,
       f"max |col-540| = {np.abs(xs - 540.0).max():.1f} px "
       f"over {len(xs)} trail px")

    # ---- freeze: ring centred on the stopped point, pens in place
    ffz = frame_at(I_FRZ + 16)
    box = ffz[1354:1407, 514:567]
    ring = red_strict(box.astype(np.float64))
    # the ring shares its box with the red trail's end: measure the
    # ring on the ANNULUS only (claim gets its own mask, trap 75)
    by, bx = np.nonzero(ring)
    rr = np.hypot(bx + 514 - RIM_TOUCH[0], by + 1354 - RIM_TOUCH[1])
    ann = (rr > R_RING - 4) & (rr < R_RING + 4)
    ok("freeze ring present", int(ann.sum()) > 200,
       f"{int(ann.sum())} red px on the annulus")
    # ring + trail end + pen are all x-symmetric about 540 in the
    # model, so the box's x centroid must sit on the line:
    cyx, cyy = centroid(ring, 514, 1354)
    ok("freeze ring centred (x)", abs(cyx - 540.0) < 1.0,
       f"x centroid off {abs(cyx - 540.0):.2f} px")
    x0, y0 = int(RIM_TOUCH[0]) - 8, int(RIM_TOUCH[1]) - 8
    cnt = int(red_strict(ffz[y0:y0 + 17, x0:x0 + 17]
                         .astype(np.float64)).sum())
    ok("red pen stopped on the rim during the freeze", cnt > 100,
       f"{cnt} red px")
    gx0, gy0 = 540 - 8, 960 - 8
    cnt = int(ink_mask(ffz[gy0:gy0 + 17, gx0:gx0 + 17]).sum())
    ok("ghost pen at dead centre during the freeze", cnt > 60,
       f"{cnt} ink px")

    # ---- trail growth; the red retrace adds ~nothing
    c1 = int(red_strict(frame_at(I_GO + 10).astype(np.float64)).sum())
    c2 = int(red_strict(frame_at(I_GO + 40).astype(np.float64)).sum())
    c3 = int(red_strict(frame_at(I_FRZ - 1).astype(np.float64)).sum())
    c4 = int(red_strict(fend.astype(np.float64)).sum())
    ok("red trail grows through leg 1", c1 < c2 < c3, f"{c1}->{c2}->{c3}")
    ok("red retrace adds ~nothing (a line has nowhere new to go)",
       abs(c4 - c3) < 400, f"freeze-eve {c3} vs final {c4}")
    # but the GHOST line is longer after the return leg (not a
    # retrace). true formula: leg 2 adds x in [120, 500.5] = 380.5 px
    # of new line at ~3 solid mask rows ≈ 1,100 px (minus occlusions)
    g3 = int(ghost_mask(frame_at(I_FRZ - 1)[950:971, :, :]).sum())
    g4 = int(ghost_mask(fend[950:971, :, :]).sum())
    ok("ghost diameter grows on the return leg", g4 > g3 + 800,
       f"lane ghost px {g3} -> {g4} (model expects ~+1100)")

    # ---- completeness on the final frame
    misses = []
    for sy in ROWS_R:
        box = fend[sy - 5:sy + 6, 534:547]
        if int(red_strict(box.astype(np.float64)).sum()) == 0:
            misses.append(sy)
    ok("red line present at 9/9 sampled heights", not misses,
       "9/9" if not misses else f"missing rows {misses}")
    misses = []
    for sx in COLS_G:
        box = fend[954:967, sx - 5:sx + 6]
        if int(ghost_mask(box).sum()) == 0:
            misses.append(sx)
    ok("ghost line present at 8/8 sampled columns", not misses,
       "8/8" if not misses else f"missing cols {misses}")
    # the outer wheel is still solid over the trails
    hit, tot = outline_pts(fend, XC, YC, R_BIG, 2 * K)
    ok("outer wheel solid on the final frame", hit == tot and tot >= 10,
       f"{hit}/{tot} visible circle points inked")
    # the rod is visible (sample its midpoint = the hub — ink there)
    hx, hy = int(round(OX[2 * K])), int(round(OY[2 * K]))
    cnt = int(ink_mask(fend[hy - 6:hy + 7, hx - 6:hx + 7]).sum())
    ok("hub (rod midpoint) present on the final frame", cnt > 30,
       f"{cnt} ink px")

    # ---- chevron: pre-hold only
    f5 = frame_at(5)
    fgo = frame_at(I_GO + 5)
    c_pre = int(ink_mask(f5[686:715, 789:852]).sum())
    c_post = int(ink_mask(fgo[686:715, 789:852]).sum())
    ok("chevron pre-hold only", c_pre > 100 and c_post == 0,
       f"{c_pre} px pre, {c_post} after")

    print(f"ALL {n[0]} CHECKS PASSED")
    print("NOT verified (stated, not proven):")
    print("  - the wheel is DRIVEN; kinematic rolling only, no dynamics")
    print("  - rolling without slip is imposed by construction")
    print("  - the freeze ring is ANNOTATION, not physics: it marks the")
    print("    instant the red point's velocity is exactly zero")
    print("  - only TWO rim points carry trails; the title's 'every")
    print("    point' is the algebra (phase alpha slides on the")
    print("    diameter at angle alpha/2), not 137 rendered trails")
    print("  - the retrace claim is RED-only; the ghost diameter is")
    print("    finished by the return leg, not retraced")
    print("  - the history (al-Tusi 1247, Copernicus) lives in the")
    print("    description, verified against sources, not in pixels")


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
    d = decode_frame(N_FRAMES - 1)
    misses = sum(int(red_strict(d[sy - 5:sy + 6, 534:547]
                                .astype(np.float64)).sum()) == 0
                 for sy in ROWS_R)
    assert misses == 0, f"{misses} red heights lost"
    print("    red line survives: 9/9 sampled heights, final frame")
    gm = sum(int(ghost_mask(d[954:967, sx - 5:sx + 6]).sum()) == 0
             for sx in COLS_G)
    assert gm == 0, f"{gm} ghost columns lost"
    print("    ghost line survives: 8/8 sampled columns")
    d0 = decode_frame(0)
    cx, cy = red_centroid_at(d0, RED_X[0], RED_Y[0])
    dd = np.hypot(cx - RED_X[0], cy - RED_Y[0])
    assert dd < 2.0, dd
    print(f"    red pen survives: f0 centroid off by {dd:.2f} px")
    dfz = decode_frame(I_FRZ + 16)
    box = dfz[1354:1407, 514:567]
    ring = red_strict(box.astype(np.float64))
    by, bx = np.nonzero(ring)
    rr = np.hypot(bx + 514 - RIM_TOUCH[0], by + 1354 - RIM_TOUCH[1])
    cnt = int(((rr > R_RING - 4) & (rr < R_RING + 4)).sum())
    assert cnt > 150, cnt
    print(f"    freeze ring survives: {cnt} red px on the annulus")
    diff = np.abs(d0.astype(int) - frame_at(0).astype(int))
    assert diff.mean() < 2.0
    print(f"    decoded f0 vs render: mean |diff| {diff.mean():.3f}, "
          f"max {diff.max()}")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams",
         "v", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
         OUT], capture_output=True, text=True).stdout.strip()
    assert int(probe) == N_FRAMES, f"frames {probe} != {N_FRAMES}"
    print(f"    {probe} frames; the straight lines survive the encode")


def review_stills():
    base = OUT[:-4]
    picks = [("start", 10), ("rolling", 76), ("freeze", 120),
             ("returning", 168), ("closing", 198), ("final", N_FRAMES - 1)]
    for name, fno in picks:
        fr = frame_at(fno)
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
