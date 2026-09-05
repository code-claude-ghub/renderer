#!/usr/bin/env python3
"""COLLIDE — the mid-air collision and the hidden point (family #3).

Two balls fly toward each other, perfectly level the whole way (they
share the family's vertical profile). The heavy ball (mass 3) comes
from the left, the light one (mass 1) from the right. The red point
between them — the mass-weighted average, drawn with a spoke to each
ball, three parts to one — rides its own parabola, on neither ball.
At m=84 they touch and STICK: a perfectly inelastic collision.
Momentum conservation makes the merged velocity the mass-weighted
average of the two, so the merged ball's centre IS the red point,
BITWISE, for the remaining 84 frames. Two dashed ghosts show the
flights that never happen.

  ACT A (n 0..83):  two balls close in; spokes to the red point.
  MERGE (n 84..90): ring flash; the dashes appear.
  ACT B (n 84..168): one ball, r=36 (area exactly conserved),
    riding the red arc with the red point at its core.
  ACT C (n 169..228): freeze on the composite.

Physics exact, not schematic (sources verified live, in the
description): objects that stick together form "one single composite
object" that "moves with a velocity dictated by the conservation of
momentum" (OpenStax Univ. Physics 9.4); inelastic collisions "do
obey conservation of momentum" though kinetic energy is lost
(Wikipedia, Inelastic collision — 83.15% of it here, computed in
feasibility). Air resistance neglected, declared.

Exactness (proven in scripts/feas_collide.py, 43 checks):
  xA = 156 + 4m; xB = 960 - 5m; Y = 1480 - 21m + m^2/8 (dyadic;
  second difference BITWISE 0.25 across all 167 triples)
  c = ((3*A) + B)/4 == (357 + 1.75m, Y(m)) BITWISE, all m
  lever law 3*(cx - xA) == xB - cx BITWISE, all m
  merged vx = ((3*4) + (-5))/4 == 1.75 == the red point's, exactly;
  momentum-stepped path == closed form BITWISE, m 85..168
  areas: 972 == 3*324 (mass ratio); 972 + 324 == 1296 == 36^2
"""
import math
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- model
W, H = 1080, 1920
FPS = 30
F = 168
MC = 84
XA0, VXA = 156, 4
XB0, VXB = 960, -5
Y0, VYC = 1480, 21
R_A = math.sqrt(972.0)
R_B = 18.0
R_MERGE = 36.0
R_DOT = 14.0
LW_RED, LW_TR, LW_DASH, LW_SPOKE, LW_RING = 11.0, 6.0, 6.0, 5.0, 7.0

A_HI = 83
RING_LO, RING_HI = 84, 90
C_LO = 169
N = 229
COL_L, COL_R = 220, 720
SPOKE_NF, SPOKE_X = 50, 580

BGC = (0.055, 0.060, 0.078)
C_REDTR = (0.88, 0.18, 0.14)
C_GRTR = (0.36, 0.40, 0.48)
C_DASH = (0.50, 0.53, 0.62)
C_SPOKE = (0.48, 0.44, 0.38)
C_DISC = (0.80, 0.82, 0.86)
C_DOTR = (0.98, 0.25, 0.18)
C_RING = (0.97, 0.97, 0.99)
C_LBL = (0.55, 0.57, 0.62)

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
OUT_DIR = "/home/maroon-beret/projects/active/youtube/youtube-channel/out"
STAMP = time.strftime("%H%M%S")
OUT_MP4 = f"{OUT_DIR}/collide_{STAMP}.mp4"


def Y(m):
    return Y0 - VYC * m + (m * m) / 8


def xA(m):
    return float(XA0 + VXA * m)


def xB(m):
    return float(XB0 + VXB * m)


def cx(m):
    # exactly as the physics says: momentum-weighted average, / total
    return ((3.0 * xA(m)) + xB(m)) / 4.0


def m_of(n):
    return min(n, F)


# ---------------------------------------------------------------- prims
def comp_bbox(img, x0, y0, cov, color):
    h, w = cov.shape
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x0 + w, W), min(y0 + h, H)
    if x1c <= x0c or y1c <= y0c:
        return
    cv = cov[y0c - y0:y1c - y0, x0c - x0:x1c - x0]
    reg = img[y0c:y1c, x0c:x1c, :]
    col = np.asarray(color, np.float64)
    reg[...] = reg * (1 - cv[..., None]) + col[None, None, :] * cv[..., None]


def disc_cov(cx_, cy_, r):
    x0, x1 = int(np.floor(cx_ - r)) - 2, int(np.ceil(cx_ + r)) + 3
    y0, y1 = int(np.floor(cy_ - r)) - 2, int(np.ceil(cy_ + r)) + 3
    xx = np.arange(x0, x1, dtype=np.float64)
    yy = np.arange(y0, y1, dtype=np.float64)
    d = np.hypot(xx[None, :] - cx_, yy[:, None] - cy_)
    return x0, y0, np.clip(r + 0.5 - d, 0.0, 1.0)


def ring_cov(cx_, cy_, r, lw):
    pad = lw / 2 + 2
    x0, x1 = int(np.floor(cx_ - r - pad)), int(np.ceil(cx_ + r + pad)) + 1
    y0, y1 = int(np.floor(cy_ - r - pad)), int(np.ceil(cy_ + r + pad)) + 1
    xx = np.arange(x0, x1, dtype=np.float64)
    yy = np.arange(y0, y1, dtype=np.float64)
    d = np.hypot(xx[None, :] - cx_, yy[:, None] - cy_)
    return x0, y0, np.clip(lw / 2 + 0.5 - np.abs(d - r), 0.0, 1.0)


def seg_cov(xa, ya, xb, yb, lw):
    """Capsule from (xa,ya) to (xb,yb), any angle."""
    pad = lw / 2 + 2
    x0 = int(np.floor(min(xa, xb) - pad))
    x1 = int(np.ceil(max(xa, xb) + pad)) + 1
    y0 = int(np.floor(min(ya, yb) - pad))
    y1 = int(np.ceil(max(ya, yb) + pad)) + 1
    xx = np.arange(x0, x1, dtype=np.float64)[None, :] - xa
    yy = np.arange(y0, y1, dtype=np.float64)[:, None] - ya
    dx, dy = xb - xa, yb - ya
    L2 = dx * dx + dy * dy
    if L2 == 0:
        d = np.hypot(xx, yy)
    else:
        t = np.clip((xx * dx + yy * dy) / L2, 0.0, 1.0)
        d = np.hypot(xx - t * dx, yy - t * dy)
    return x0, y0, np.clip(lw / 2 + 0.5 - d, 0.0, 1.0)


def stamp_max(buf, x0, y0, cov):
    h, w = cov.shape
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x0 + w, W), min(y0 + h, H)
    if x1c <= x0c or y1c <= y0c:
        return
    cv = cov[y0c - y0:y1c - y0, x0c - x0:x1c - x0]
    reg = buf[y0c:y1c, x0c:x1c]
    np.maximum(reg, cv, out=reg)


def text_cov(s, px):
    f = ImageFont.truetype(FONT, px * 4)
    im = Image.new("L", (px * len(s) * 4, px * 8), 0)
    ImageDraw.Draw(im).text((8, 8), s, font=f, fill=255)
    a = np.asarray(im, np.float64) / 255.0
    ys, xs = np.where(a > 0)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h4, w4 = a.shape
    h4 -= h4 % 4
    w4 -= w4 % 4
    a = a[:h4, :w4].reshape(h4 // 4, 4, w4 // 4, 4).mean((1, 3))
    return a


BG = np.empty((H, W, 3), np.float64)
BG[..., 0], BG[..., 1], BG[..., 2] = BGC

LBL_A = text_cov("the weighted average of the pair", 34)
LBL_B1 = text_cov("the merged ball rides the red arc", 34)
LBL_B2 = text_cov("the dashed arcs never happen", 34)
LBL_C = text_cov("the crash never moved it", 34)
LBL_Y_A, LBL_Y_B1, LBL_Y_B2, LBL_Y_C = 230, 230, 292, 356

# static dashed ghosts: both un-collided continuations (m 84..168)
DASH = np.zeros((H, W), np.float64)
for _j in range(MC, F - 2, 7):
    _a, _b = _j, min(_j + 3, F)
    stamp_max(DASH, *seg_cov(xA(_a), Y(_a), xA(_b), Y(_b), LW_DASH))
    stamp_max(DASH, *seg_cov(xB(_a), Y(_a), xB(_b), Y(_b), LW_DASH))


# ---------------------------------------------------------------- state
def new_trails():
    return (np.zeros((H, W), np.float64),      # red weighted-point trail
            np.zeros((H, W), np.float64))      # ball trails (grey)


def append_red(tr_red, m):
    if m >= 1:
        stamp_max(tr_red, *seg_cov(cx(m - 1), Y(m - 1),
                                   cx(m), Y(m), LW_RED))


def append_grey(tr_gr, m):
    if 1 <= m <= MC:
        stamp_max(tr_gr, *seg_cov(xA(m - 1), Y(m - 1),
                                  xA(m), Y(m), LW_TR))
        stamp_max(tr_gr, *seg_cov(xB(m - 1), Y(m - 1),
                                  xB(m), Y(m), LW_TR))


def fade(n, n0, span=10):
    return float(np.clip((n - n0 + 1) / span, 0.0, 1.0))


def composite(n, tr_red, tr_gr):
    img = BG.copy()
    m = m_of(n)
    # dashed ghosts under everything (appear at the merge)
    if n >= MC:
        da = 0.85 * fade(n, MC, 6)
        a = DASH[..., None] * da
        img *= (1 - a)
        img += np.asarray(C_DASH, np.float64)[None, None, :] * a
    for buf, col in ((tr_gr, C_GRTR), (tr_red, C_REDTR)):
        a = buf[..., None]
        img *= (1 - a)
        img += np.asarray(col, np.float64)[None, None, :] * a
    # spokes: ball -> weighted point -> ball (act A only; the merge
    # closes both distances to zero and the spokes die with them)
    if 1 <= n <= A_HI:
        sa = 0.85 * fade(n, 8)
        for xa_, xb_ in ((xA(m), cx(m)), (cx(m), xB(m))):
            x0, y0, cv = seg_cov(xa_, Y(m), xb_, Y(m), LW_SPOKE)
            comp_bbox(img, x0, y0, cv * sa, C_SPOKE)
    # discs: two balls, then the one composite ball (area conserved)
    if n <= A_HI:
        comp_bbox(img, *disc_cov(xA(m), Y(m), R_A), C_DISC)
        comp_bbox(img, *disc_cov(xB(m), Y(m), R_B), C_DISC)
    else:
        comp_bbox(img, *disc_cov(cx(m), Y(m), R_MERGE), C_DISC)
    # merge ring flash: full brightness two frames, then fade (the
    # BURST lesson — a fading ring never reaches mask-white)
    if RING_LO <= n <= RING_HI:
        r = 36.0 + 12.0 * (n - RING_LO)
        al = 1.0 if n <= RING_LO + 1 else 1.0 - (n - RING_LO - 1) / 5.0
        x0, y0, cv = ring_cov(cx(MC), Y(MC), r, LW_RING)
        comp_bbox(img, x0, y0, cv * al, C_RING)
    # the red point — bitwise the weighted average, then the ball core
    comp_bbox(img, *disc_cov(cx(m), Y(m), R_DOT), C_DOTR)
    # labels (safe area: y 210..400, trap 3)
    if 16 <= n <= A_HI:
        cv = LBL_A * fade(n, 16)
        comp_bbox(img, (W - LBL_A.shape[1]) // 2, LBL_Y_A, cv, C_LBL)
    if n >= RING_HI + 2:
        cv1 = LBL_B1 * fade(n, RING_HI + 2)
        cv2 = LBL_B2 * fade(n, RING_HI + 2)
        comp_bbox(img, (W - LBL_B1.shape[1]) // 2, LBL_Y_B1, cv1, C_REDTR)
        comp_bbox(img, (W - LBL_B2.shape[1]) // 2, LBL_Y_B2, cv2, C_LBL)
    if n >= C_LO + 11:
        cv = LBL_C * fade(n, C_LO + 11)
        comp_bbox(img, (W - LBL_C.shape[1]) // 2, LBL_Y_C, cv, C_LBL)
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def frame_at(n):
    """Pure reconstruction — rebuilds trail state from scratch."""
    tr_red, tr_gr = new_trails()
    for m in range(1, m_of(n) + 1):
        append_red(tr_red, m)
        append_grey(tr_gr, m)
    return composite(n, tr_red, tr_gr)


def render_frames():
    """Incremental generator — identical op order to frame_at."""
    tr_red, tr_gr = new_trails()
    for n in range(N):
        if 1 <= n <= F:
            append_red(tr_red, n)
            append_grey(tr_gr, n)
        yield composite(n, tr_red, tr_gr)


# ---------------------------------------------------------------- checks
# FENCE AUDIT (written before the first render):
#   RED (r-g > 60): red trail + dot (rows 400..1700, all acts, from
#     n=1) and label LBL_B1 (rows <400, n >= 92). Row fence 400.
#   WHITE (min >= 225): the merge ring ONLY, n 84..85 (full-alpha
#     frames). Discs are (204,209,219) — min 204 < 225.
#   COOL GREY (b > r+8, 40 < max < 160): ball trails (n >= 1), the
#     dashes (n >= 84), labels A/B2/C. Rows >= 420 exclude labels.
#   WARM (r > b+8 AND g-b >= 8, 40 < max < 160): spokes ONLY, act A
#     (n 1..83). Red AA edges excluded by g-b (BURST's algebra: any
#     red blend with g-b >= 8 has max >= 160).
#   DISC colour (each channel within 10 of (204,209,219)): the two
#     balls (act A) / the merged ball (B, C) only.
CHECKS = {"pass": 0, "fail": 0}


def ok(name, cond, detail=""):
    s = "ok  " if cond else "FAIL"
    CHECKS["pass" if cond else "fail"] += 1
    print(f"{s} {name}" + (f" [{detail}]" if detail else ""), flush=True)


def red_mask(fr):
    return (fr[:, :, 0].astype(np.int64) - fr[:, :, 1].astype(np.int64)) > 60


def white_mask(fr):
    return fr.min(2) >= 225


def grey_mask(fr):
    mx = fr.max(2).astype(np.int64)
    return ((fr[:, :, 2].astype(np.int64) > fr[:, :, 0].astype(np.int64) + 8)
            & (mx > 40) & (mx < 160))


def warm_mask(fr):
    r = fr[:, :, 0].astype(np.int64)
    g = fr[:, :, 1].astype(np.int64)
    b = fr[:, :, 2].astype(np.int64)
    mx = fr.max(2).astype(np.int64)
    return (r > b + 8) & (g - b >= 8) & (mx > 40) & (mx < 160)


DISC8 = np.asarray([np.uint8(v * 255 + 0.5) for v in C_DISC], np.int64)


def disc_mask(fr):
    d = np.abs(fr.astype(np.int64) - DISC8[None, None, :])
    return d.max(2) <= 10


def centroid(mask):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def clusters(mask):
    rows = np.where(mask.any(1))[0]
    if len(rows) == 0:
        return []
    return np.split(rows, np.where(np.diff(rows) > 4)[0] + 1)


def disc_centroid(fr, px, py, half):
    box = disc_mask(fr[int(py) - half:int(py) + half,
                       int(px) - half:int(px) + half])
    cc = centroid(box)
    if cc is None:
        return None
    return cc[0] + int(px) - half, cc[1] + int(py) - half


def run_checks():
    print("== render checks ==", flush=True)
    bg8 = tuple(np.uint8(np.clip(v, 0, 1) * 255.0 + 0.5) for v in BGC)

    f2 = frame_at(2)
    ok("corner pixel is background", tuple(f2[4, 4]) == bg8,
       f"{tuple(f2[4, 4])} vs {bg8}")
    f100 = frame_at(100)
    frac = (f100.max(2) > 40).mean()
    ok("lit fraction sane (not blank, not floodlit)", 0.01 < frac < 0.30,
       f"{frac:.4f}")

    gen = render_frames()
    g50 = g150 = g210 = None
    for k, fr in enumerate(gen):
        if k == 50:
            g50 = fr
        elif k == 150:
            g150 = fr
        elif k == 210:
            g210 = fr
        elif k > 210:
            break
    ok("generator == frame_at, byte-exact (n=50, 150, 210)",
       np.array_equal(g50, frame_at(50))
       and np.array_equal(g150, frame_at(150))
       and np.array_equal(g210, frame_at(210)))

    ok("balls have visibly moved by frame 8",
       not np.array_equal(f2, frame_at(8)))

    # ---- act A: both discs at the model positions (frame 30)
    f30 = frame_at(30)
    ca = disc_centroid(f30, xA(30), Y(30), 46)
    cb = disc_centroid(f30, xB(30), Y(30), 26)
    da = (np.hypot(ca[0] - xA(30), ca[1] - Y(30)) if ca else 99.0)
    db = (np.hypot(cb[0] - xB(30), cb[1] - Y(30)) if cb else 99.0)
    ok("both discs centred on the model (frame 30)",
       max(da, db) < 2.0, f"devs {da:.2f}, {db:.2f} px")

    # ---- the trap-66 coupling check, act A: the weighted average of
    #      the two MEASURED disc centroids lands on the model parabola
    if ca and cb:
        ax = ((3.0 * ca[0]) + cb[0]) / 4.0
        ay = ((3.0 * ca[1]) + cb[1]) / 4.0
        cdev = np.hypot(ax - cx(30), ay - Y(30))
    else:
        cdev = 99.0
    ok("weighted average of the MEASURED discs == model parabola",
       cdev < 2.5, f"dev {cdev:.2f} px")

    # ---- act B: the merged disc centred on the weighted point
    f120 = frame_at(120)
    cm = disc_centroid(f120, cx(120), Y(120), 48)
    dm = (np.hypot(cm[0] - cx(120), cm[1] - Y(120)) if cm else 99.0)
    ok("merged disc centred on the red point (frame 120)", dm < 2.0,
       f"dev {dm:.2f} px")

    # ---- red leading edge before AND after the merge
    for nn in (40, 120, 160):
        fr = frame_at(nn) if nn not in (120,) else f120
        band = red_mask(fr[400:1700, :])
        xs = np.where(band.any(0))[0]
        lead = xs.max() if len(xs) else -1
        ok(f"red leading edge == weighted point + dot radius (frame {nn})",
           abs(lead - (cx(m_of(nn)) + R_DOT)) <= 2.5,
           f"{lead} vs {cx(m_of(nn)) + R_DOT:.0f}")

    # ---- ring flash: mask-white only while it holds full brightness
    f50 = frame_at(50)
    f85 = frame_at(85)
    ok("no white before the merge (frame 50)",
       not white_mask(f50).any())
    ok("ring flash present (frame 85)", white_mask(f85).sum() > 400,
       f"{white_mask(f85).sum()} px")
    ok("no white after the flash (frame 120)",
       not white_mask(f120).any())

    # ---- spokes: warm ink in act A at the feas-proven point, and
    #      NONE after the merge (the distances closed to zero)
    sm = warm_mask(f50[int(Y(SPOKE_NF)) - 3:int(Y(SPOKE_NF)) + 4,
                       SPOKE_X - 3:SPOKE_X + 4])
    ok("spoke present at the feas sample point (frame 50)", sm.any())
    ok("A-side spoke present too (frame 50)",
       warm_mask(f50[int(Y(50)) - 3:int(Y(50)) + 4, 404:425]).any())
    ok("no spoke ink after the merge (frames 120, 210)",
       not warm_mask(f120[420:1700, :]).any()
       and not warm_mask(g210[420:1700, :]).any())

    # ---- act C: red curvature off the pixels (stride 16 -> sd 64)
    f210 = g210
    cents = []
    for m in range(16, 153, 16):
        x = int(cx(m))
        col = red_mask(f210[400:1700, x - 2:x + 3])
        ys = np.where(col.any(1))[0]
        cents.append(ys.mean() + 400 if len(ys) else -1e9)
    # a vertical slice under-reads a steep band: thickness in the
    # slice scales with sqrt(1+slope^2) and the slope VARIES across
    # the slice, so the band is thicker on its steep side and the
    # centroid sits ~+1.5 px toward it at |dy/dx| ~ 10 (measured on
    # the coverage buffer directly, 2026-09-04 diag; the discs, same
    # drawing code, measure 0.00). The columns keep the curvature
    # identity — a slowly-varying bias cancels in second differences —
    # and the TIGHT on-parabola claim is the row-slice check below.
    dev = max(abs(cy_ - Y(m)) for cy_, m in zip(cents, range(16, 153, 16)))
    ok("red columns near the model parabola (slope-bias bounded)",
       dev <= 3.2, f"max dev {dev:.2f} px")
    sds = [cents[i + 1] - 2 * cents[i] + cents[i - 1]
           for i in range(1, len(cents) - 1)]
    ok("curvature from pixels alone: second difference == 64 (0.25*16^2)",
       max(abs(s - 64.0) for s in sds) <= 6.0,
       f"{[round(s, 1) for s in sds]}")

    # ---- red continuity THROUGH the merge column (x 480..530)
    cont = all(
        red_mask(f210[560:680, x:x + 1]).any() for x in range(480, 531))
    ok("red trail unbroken through the merge (x 480..530)", cont)

    # ---- the tight on-parabola claim: ROW slices cut the steep
    #      flanks near-perpendicular (slope in x-per-y ~ 0.1), so the
    #      cluster centroid is unbiased where the columns are not.
    #      Y(m) = 598 + (m-84)^2/8, so row r crosses at
    #      m = 84 -+ sqrt(8*(r-598)), x = 357 + 1.75*m.
    rdev = 0.0
    rows_ok = True
    for r in (800, 1000, 1200, 1400):
        mm = math.sqrt(8.0 * (r - 598))
        exp_x = sorted(357.0 + 1.75 * (84.0 + s * mm) for s in (-1, 1))
        band = red_mask(f210[r - 1:r + 2, 340:670])
        xs = np.where(band.any(0))[0]
        if len(xs) == 0:
            rows_ok = False
            continue
        cl = np.split(xs, np.where(np.diff(xs) > 4)[0] + 1)
        if len(cl) != 2:
            rows_ok = False
            continue
        got = [float(c.mean()) + 340 for c in cl]
        rdev = max(rdev, max(abs(g - e) for g, e in zip(got, exp_x)))
    ok("red flanks on the model parabola (8 row slices)",
       rows_ok and rdev <= 2.0, f"max dev {rdev:.2f} px")

    # ---- grey crossings at the two feas-proven columns
    for col, name in ((COL_L, "COL_L"), (COL_R, "COL_R")):
        exp = sorted((Y((col - XA0) / VXA), Y((XB0 - col) / (-VXB))))
        gm = grey_mask(f210[420:1700, col - 2:col + 3])
        got = sorted(float(c.mean()) + 420 for c in clusters(gm))
        match = (len(got) == 2
                 and max(abs(g - e) for g, e in zip(got, exp)) < 3.0)
        ok(f"two grey crossings at {name}={col}", match,
           f"got {[round(g) for g in got]} vs exp {[round(e) for e in exp]}")

    # ---- the ghosts: dash ink at both curves' m=140 points (frame 100)
    gx, gy = int(xA(140)), int(Y(140))
    hx = int(xB(140))
    ok("ghost-A dash visible (frame 100)",
       grey_mask(f100[gy - 4:gy + 5, gx - 4:gx + 5]).any())
    ok("ghost-B dash visible (frame 100)",
       grey_mask(f100[gy - 4:gy + 5, hx - 4:hx + 5]).any())
    ok("no dash before the merge (frame 50, ghost-A point)",
       not grey_mask(f50[gy - 4:gy + 5, gx - 4:gx + 5]).any())

    # ---- labels per act (colour-matched, trap 61)
    lab = (slice(200, 400), slice(0, W))
    ok("act A label present (grey), no red label yet (frame 30)",
       grey_mask(f30[lab]).sum() > 300 and not red_mask(f30[lab]).any())
    ok("act B labels present (red + grey) (frame 100)",
       red_mask(f100[lab]).sum() > 300
       and grey_mask(f100[lab]).sum() > 300)
    ok("act C third label present (frame 210)",
       grey_mask(f210[lab]).sum() > 300)

    # ---- freeze byte-identity after label C fade completes
    ok("act C frames byte-identical (192 == 220, 200 == 225)",
       np.array_equal(frame_at(192), frame_at(220))
       and np.array_equal(frame_at(200), frame_at(225)))

    # ---- safe areas (trap 3)
    ok("top 192 rows pure background, all sampled frames",
       all((fr[:192] == np.asarray(bg8, np.uint8)).all()
           for fr in (f2, f30, f50, f85, f100, f120, f210)))
    ok("rows >= 1700 pure background, all sampled frames",
       all((fr[1700:] == np.asarray(bg8, np.uint8)).all()
           for fr in (f2, f30, f50, f85, f100, f120, f210)))

    print()
    print("NOT verified by these checks (trap 68):")
    print("  - air resistance is neglected; declared in the description")
    print("  - the collision is drawn at frame resolution: contact")
    print("    falls between m=83 and m=84 and the merge is drawn at 84")
    print("  - 'perfectly inelastic' is a modelling choice; real balls")
    print("    have a coefficient of restitution above zero — declared")
    print("  - the theorem holds until the merged ball lands; no ground")
    print("    is drawn inside the flight")
    print("  - h264 fidelity is checked on fenced regions only")
    print()
    if CHECKS["fail"]:
        print(f"{CHECKS['fail']} CHECK FAILURES")
        sys.exit(1)
    print(f"ALL {CHECKS['pass']} CHECKS PASSED", flush=True)


# ---------------------------------------------------------------- encode
def encode():
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "18",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT_MP4]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    for fr in render_frames():
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        print("ENCODE FAILED", flush=True)
        sys.exit(1)
    print(f"encoded {OUT_MP4} ({os.path.getsize(OUT_MP4)} bytes)",
          flush=True)


def decode_frame(n):
    r = subprocess.run(
        ["ffmpeg", "-i", OUT_MP4, "-vf", f"select=eq(n\\,{n})",
         "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True)
    return np.frombuffer(r.stdout, np.uint8).reshape(H, W, 3)


def check_encode():
    print("== encode checks ==", flush=True)
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-count_frames", "-select_streams",
         "v", "-show_entries",
         "stream=nb_read_frames,width,height,r_frame_rate",
         "-of", "csv=p=0", OUT_MP4], capture_output=True, text=True)
    print("ffprobe:", r.stdout.strip(), flush=True)
    ok("229 frames in the file", f"{N}" in r.stdout)

    d210 = decode_frame(210)
    nred = red_mask(d210[400:1700, :]).sum()
    ok("red parabola survives h264", nred > 3000, f"{nred} red px")
    cents = []
    for m in range(16, 153, 16):
        x = int(cx(m))
        col = red_mask(d210[400:1700, x - 2:x + 3])
        ys = np.where(col.any(1))[0]
        cents.append(ys.mean() + 400 if len(ys) else -1e9)
    # columns only where they cut the band near-perpendicular (the
    # apex, |dy/dx| <= 4). On the steep flanks 4:2:0 chroma extends
    # the red mask ~1 px in one x-direction, which a vertical slice
    # reads as ~|slope|/2 px of centroid shift (measured +7.5 at
    # slope 9.8, sign following the slope). The flanks are graded by
    # the ROW slices below, which cut them near-perpendicular.
    apex = [(cy_, m) for cy_, m in zip(cents, range(16, 153, 16))
            if 64 <= m <= 112]
    dev = max(abs(cy_ - Y(m)) for cy_, m in apex)
    ok("red apex columns on the parabola on the SHIPPED file",
       dev <= 3.5, f"max dev {dev:.2f} px over {len(apex)} columns")
    # the tight claim, same row-slice instrument as the render check
    rdev = 0.0
    rows_ok = True
    for r in (800, 1000, 1200, 1400):
        mm = math.sqrt(8.0 * (r - 598))
        exp_x = sorted(357.0 + 1.75 * (84.0 + s * mm) for s in (-1, 1))
        band = red_mask(d210[r - 1:r + 2, 340:670])
        xs = np.where(band.any(0))[0]
        if len(xs) == 0:
            rows_ok = False
            continue
        cl = np.split(xs, np.where(np.diff(xs) > 4)[0] + 1)
        if len(cl) != 2:
            rows_ok = False
            continue
        got = [float(c.mean()) + 340 for c in cl]
        rdev = max(rdev, max(abs(g - e) for g, e in zip(got, exp_x)))
    ok("red flanks on the parabola on the SHIPPED file",
       rows_ok and rdev <= 2.5, f"max dev {rdev:.2f} px")
    for col, name in ((COL_L, "COL_L"), (COL_R, "COL_R")):
        gm = grey_mask(d210[420:1700, col - 2:col + 3])
        ok(f"two grey crossings survive h264 at {name}",
           len(clusters(gm)) == 2, f"{len(clusters(gm))} clusters")
    d85 = decode_frame(85)
    ok("ring flash survives h264 (frame 85)",
       white_mask(d85).sum() > 300, f"{white_mask(d85).sum()} px")
    a, b = decode_frame(195), decode_frame(220)
    d1 = np.abs(a.astype(np.int64) - b.astype(np.int64)).mean()
    ok("freeze survives h264 (195 vs 220 near-identical)", d1 < 0.5,
       f"mean |diff| {d1:.3f} grey")
    d2 = np.abs(decode_frame(2).astype(np.int64)
                - decode_frame(8).astype(np.int64)).mean()
    ok("early frames genuinely differ (motion by frame 8)", d2 > 0.5,
       f"mean |diff| {d2:.3f} grey")
    print()
    if CHECKS["fail"]:
        print(f"{CHECKS['fail']} FAILURES (incl. render)")
        sys.exit(1)
    print("ENCODE CHECKS PASSED — DONE", flush=True)


def review_stills():
    for n in (30, 60, 85, 120, 168, 210):
        Image.fromarray(frame_at(n)).save(f"{OUT_DIR}/collide_f{n:03d}.png")
    subprocess.run(
        ["ffmpeg", "-y", "-pattern_type", "glob",
         "-i", f"{OUT_DIR}/collide_f*.png",
         "-filter_complex", "scale=270:-1,tile=3x2",
         f"{OUT_DIR}/collide_sheet.png"],
        capture_output=True)
    print("sheet:", f"{OUT_DIR}/collide_sheet.png", flush=True)


if __name__ == "__main__":
    run_checks()
    encode()
    check_encode()
    review_stills()
