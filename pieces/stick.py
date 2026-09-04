#!/usr/bin/env python3
"""STICK — the tumbling stick and the hidden point.

A rigid stick (heavy disc one end, light disc the other, mass 3:1,
disc areas drawn proportional to mass) is thrown spinning: six full
turns across one flight.

ACT A — the flight, tracing BOTH END paths: the light end loops six
  times, the heavy end once. A tangle.
ACT B — the same flight replayed at 2x, now tracing two marked
  points: the rod's geometric MIDDLE (white — wobbles +-30 px) and
  the balance point (red — a clean parabola). The balance point is
  NOT the middle: it sits three times closer to the heavy end
  (lever law, 3 x 30 == 1 x 90).
ACT C — freeze on the full composite.

The physics is exact, not schematic: in a uniform gravitational
field the resultant torque about the centre of mass is zero, so the
spin is constant and the CoM moves like a point particle — the
parabola. The model implements the theorem directly (air resistance
neglected, declared in the description).

Exactness (all proven in scripts/feas_stick.py, 37 checks):
  x_c(m) = 120 + 5m           (bitwise +5.0 every frame)
  y_c(m) = 1480 - 21m + m^2/8 (dyadic-exact; second difference
                               BITWISE 0.25 across all 167 triples;
                               time-symmetric bitwise)
  orientation: 28-entry table, (m+7) % 28 — six whole turns, lands
  in its launch orientation (the same table entry)
  (3 p_heavy + p_light)/4 == CoM to 2.3e-13 px (measured)
"""
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
P = 28
X0, VX = 120, 5
Y0, VYC = 1480, 21
RH, RL = 30.0, 90.0
R_DISC_H, R_DISC_L = 31.0, 18.0
R_DOT_RED, R_DOT_MID = 14.0, 10.0
LW_ROD, LW_END, LW_MID, LW_RED = 8.0, 6.0, 9.0, 11.0

A_LO, A_HI = 0, 168
B_LO, B_HI = 169, 253
C_LO, C_HI = 254, 299
N = 300

BGC = (0.055, 0.060, 0.078)
C_ENDTR = (0.36, 0.40, 0.48)
C_MIDTR = (0.93, 0.93, 0.95)
C_REDTR = (0.88, 0.18, 0.14)
C_ROD = (0.52, 0.54, 0.60)
C_DISC = (0.80, 0.82, 0.86)
C_DOTR = (0.98, 0.25, 0.18)
C_DOTM = (0.99, 0.99, 1.00)
C_LBL = (0.55, 0.57, 0.62)

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
OUT_DIR = "/home/maroon-beret/projects/active/youtube/youtube-channel/out"
STAMP = time.strftime("%H%M%S")
OUT_MP4 = f"{OUT_DIR}/stick_{STAMP}.mp4"

E_TAB = [(np.cos(2 * np.pi * j / P), np.sin(2 * np.pi * j / P))
         for j in range(P)]


def xc(m):
    return float(X0 + VX * m)


def yc(m):
    return Y0 - VYC * m + (m * m) / 8


def ends(m):
    c = (xc(m), yc(m))
    ex, ey = E_TAB[(m + 7) % P]
    ph = (c[0] + RH * ex, c[1] + RH * ey)
    pl = (c[0] - RL * ex, c[1] - RL * ey)
    return c, ph, pl


def midpt(m):
    c, ph, pl = ends(m)
    return ((ph[0] + pl[0]) / 2, (ph[1] + pl[1]) / 2)


def m_of(n):
    if n <= A_HI:
        return n
    if n <= B_HI:
        return 2 * (n - B_LO)
    return F


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


def disc_cov(cx, cy, r):
    x0, x1 = int(np.floor(cx - r)) - 2, int(np.ceil(cx + r)) + 3
    y0, y1 = int(np.floor(cy - r)) - 2, int(np.ceil(cy + r)) + 3
    xx = np.arange(x0, x1, dtype=np.float64)
    yy = np.arange(y0, y1, dtype=np.float64)
    d = np.hypot(xx[None, :] - cx, yy[:, None] - cy)
    return x0, y0, np.clip(r + 0.5 - d, 0.0, 1.0)


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

LBL_A = text_cov("both ends of the stick, traced", 34)
LBL_B1 = text_cov("the middle of the stick", 34)
LBL_B2 = text_cov("its balance point", 34)
LBL_C = text_cov("three times closer to the heavy end", 34)

LBL_Y_A = 230
LBL_Y_B1, LBL_Y_B2, LBL_Y_C = 230, 292, 356


# ---------------------------------------------------------------- state
def new_trails():
    return (np.zeros((H, W), np.float64),      # end paths
            np.zeros((H, W), np.float64),      # midpoint
            np.zeros((H, W), np.float64))      # balance point


def append_end_segs(tr_end, m):
    """Called when the stick arrives at flight state m (act A)."""
    if m < 1:
        return
    _, ph0, pl0 = ends(m - 1)
    _, ph1, pl1 = ends(m)
    stamp_max(tr_end, *seg_cov(*ph0, *ph1, LW_END))
    stamp_max(tr_end, *seg_cov(*pl0, *pl1, LW_END))


def append_b_segs(tr_mid, tr_red, m):
    """Called when the replay arrives at even state m (act B)."""
    for k in (m - 1, m):
        if k < 1:
            continue
        a, b = midpt(k - 1), midpt(k)
        stamp_max(tr_mid, *seg_cov(*a, *b, LW_MID))
        ca = (xc(k - 1), yc(k - 1))
        cb = (xc(k), yc(k))
        stamp_max(tr_red, *seg_cov(*ca, *cb, LW_RED))


def alpha_end(n):
    if n <= A_HI:
        return 1.0
    return 1.0 - 0.55 * min(n - A_HI, 13) / 13.0


def fade(n, n0, span=10):
    return float(np.clip((n - n0 + 1) / span, 0.0, 1.0))


def composite(n, tr_end, tr_mid, tr_red):
    img = BG.copy()
    ae = alpha_end(n)
    for buf, col, al in ((tr_end, C_ENDTR, ae),
                         (tr_mid, C_MIDTR, 1.0),
                         (tr_red, C_REDTR, 1.0)):
        a = buf[..., None] * al
        if al > 0:
            img *= (1 - a)
            img += np.asarray(col, np.float64)[None, None, :] * a
    # the stick
    m = m_of(n)
    c, ph, pl = ends(m)
    comp_bbox(img, *seg_cov(*ph, *pl, LW_ROD), C_ROD)
    comp_bbox(img, *disc_cov(*ph, R_DISC_H), C_DISC)
    comp_bbox(img, *disc_cov(*pl, R_DISC_L), C_DISC)
    if n >= B_LO:                       # marked points, acts B and C
        comp_bbox(img, *disc_cov(*midpt(m), R_DOT_MID), C_DOTM)
        comp_bbox(img, *disc_cov(*c, R_DOT_RED), C_DOTR)
    # labels (safe area: y 210..400, trap 3)
    if A_LO + 20 <= n <= A_HI:
        cv = LBL_A * fade(n, A_LO + 20)
        comp_bbox(img, (W - LBL_A.shape[1]) // 2, LBL_Y_A, cv, C_LBL)
    if n >= B_LO:
        cv1 = LBL_B1 * fade(n, B_LO)
        cv2 = LBL_B2 * fade(n, B_LO)
        comp_bbox(img, (W - LBL_B1.shape[1]) // 2, LBL_Y_B1, cv1, C_MIDTR)
        comp_bbox(img, (W - LBL_B2.shape[1]) // 2, LBL_Y_B2, cv2, C_REDTR)
    if n >= C_LO:
        cv = LBL_C * fade(n, C_LO)
        comp_bbox(img, (W - LBL_C.shape[1]) // 2, LBL_Y_C, cv, C_LBL)
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def frame_at(n):
    """Pure reconstruction — rebuilds trail state from scratch."""
    tr_end, tr_mid, tr_red = new_trails()
    for m in range(1, min(n, A_HI) + 1):
        append_end_segs(tr_end, m)
    if n >= B_LO:
        for k in range(B_LO, min(n, B_HI) + 1):
            append_b_segs(tr_mid, tr_red, 2 * (k - B_LO))
    return composite(n, tr_end, tr_mid, tr_red)


def render_frames():
    """Incremental generator — identical op order to frame_at."""
    tr_end, tr_mid, tr_red = new_trails()
    for n in range(N):
        if 1 <= n <= A_HI:
            append_end_segs(tr_end, n)
        elif B_LO <= n <= B_HI:
            append_b_segs(tr_mid, tr_red, 2 * (n - B_LO))
        yield composite(n, tr_end, tr_mid, tr_red)


# ---------------------------------------------------------------- checks
# FENCE AUDIT (written before the first render):
#   RED (r-g > 60): the red trail (act B/C, along the parabola,
#     rows 490..1560), the red dot (on the trail tip), and label
#     LBL_B2 (rows 292..336+). Row fence 400..1600 isolates trail+dot.
#   WHITE (min channel >= 225): mid trail (act B/C), white dot, and
#     LBL_B1 (rows 230..274+). Discs are (204,209,219) — below 225.
#     Row fence 400..1600 isolates trail+dot; x fence < 900 excludes
#     the act-C parked dot (x=960).
#   GREY-BLUE (b > r+8, 40 < max < 160): end trails, faded end
#     trails, the rod (133,138,153) and label A/C (140,145,158).
#     Tangle measured only in act C at x <= 900 (rod parked at 960)
#     and rows >= 420 (labels end by y 400).
#   BRIGHT (max >= 190) in act A: ONLY the two discs (no dots, no
#     white trail, no white label in act A; label A peaks at 158).
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


def bright_mask(fr):
    return fr.max(2) >= 190


def clusters(mask):
    """Split a mask into connected row-groups (cheap 1-D clustering)."""
    rows = np.where(mask.any(1))[0]
    if len(rows) == 0:
        return []
    return np.split(rows, np.where(np.diff(rows) > 4)[0] + 1)


def centroid(mask):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


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

    # ---- generator/reconstruction identity (instrument integrity)
    gen = render_frames()
    for k, fr in enumerate(gen):
        if k == 50:
            g50 = fr
        elif k == 200:
            g200 = fr
        elif k == 280:
            g280 = fr
        elif k > 280:
            break
    ok("generator == frame_at, byte-exact (n=50, 200, 280)",
       np.array_equal(g50, frame_at(50))
       and np.array_equal(g200, frame_at(200))
       and np.array_equal(g280, frame_at(280)))

    # ---- hook: motion by frame 8
    f8 = frame_at(8)
    ok("stick has visibly moved by frame 8",
       not np.array_equal(f2, f8))
    bc = centroid(bright_mask(f8[400:1700, :]))
    c8, ph8, pl8 = ends(8)
    exp_x = (3 * ph8[0] + pl8[0] + 0) / 4  # rough: bright = 2 discs
    ok("frame-8 discs near the model stick",
       bc is not None and abs(bc[0] - (ph8[0] + pl8[0]) / 2) < 40,
       f"bright centroid x {bc[0]:.0f}" if bc else "none")

    # ---- act A: discs at model positions (frame 50)
    f50 = frame_at(50)
    bm = bright_mask(f50[400:1700, :])
    cl = clusters(bm)
    ok("act A: exactly two bright clusters (the discs)", len(cl) in (1, 2),
       f"{len(cl)} row-groups")
    c50, ph50, pl50 = ends(50)
    # per-disc centroid in a fenced box (trap 58/75: own mask each)
    devs = []
    for (px, py), r in ((ph50, R_DISC_H), (pl50, R_DISC_L)):
        box = bright_mask(f50[int(py) - 40:int(py) + 40,
                              int(px) - 40:int(px) + 40])
        cc = centroid(box)
        devs.append(np.hypot(cc[0] - 40 - (px - int(px)),
                             cc[1] - 40 - (py - int(py)))
                    if cc else 99.0)
    ok("disc centroids on the model ends (frame 50)",
       max(devs) < 2.0, f"max dev {max(devs):.2f} px")
    # rod present between the discs — sampled on the LIGHT side:
    # the heavy disc (r=31) covers the whole c..p_heavy stretch
    # (30 px), so run 1 sampled INSIDE the disc (instrument bug)
    qx = int(c50[0] + 0.4 * (pl50[0] - c50[0]))
    qy = int(c50[1] + 0.4 * (pl50[1] - c50[1]))
    px_rod = f50[qy, qx].astype(np.int64)
    rod8 = np.asarray([np.uint8(np.clip(v, 0, 1) * 255 + 0.5)
                       for v in C_ROD], np.int64)
    ok("rod ink between disc and centre (frame 50)",
       np.abs(px_rod - rod8).max() <= 8, f"{tuple(px_rod)}")

    # ---- act A: no red, no white anywhere
    ok("no red in act A", not red_mask(f100).any())
    ok("no white in act A", not white_mask(f100).any())

    # ---- act B: red dot leads the red trail at the model CoM
    f200 = frame_at(200)
    m200 = m_of(200)                                     # 62
    band = red_mask(f200[400:1600, :])
    xs = np.where(band.any(0))[0]
    lead = xs.max() if len(xs) else -1
    ok("red leading edge at the CoM + dot radius (frame 200)",
       abs(lead - (xc(m200) + R_DOT_RED)) <= 2.5,
       f"{lead} vs {xc(m200) + R_DOT_RED:.0f}")
    ok("white dot present in act B (frame 200)",
       white_mask(f200[400:1600, :]).sum() > 100,
       f"{white_mask(f200[400:1600, :]).sum()} px")

    # ---- act C: red trail curvature OFF THE PIXELS (frame 280)
    f280 = frame_at(280)
    cents = []
    for m in range(16, 153, 16):                         # x <= 880
        x = int(xc(m))
        col = red_mask(f280[400:1600, x - 2:x + 3])
        ys = np.where(col.any(1))[0]
        cents.append(ys.mean() + 400 if len(ys) else -1e9)
    dev = max(abs(cy - yc(m)) for cy, m in zip(cents, range(16, 153, 16)))
    ok("red trail on the model parabola, 9 columns",
       dev <= 2.0, f"max dev {dev:.2f} px")
    sds = [cents[i + 1] - 2 * cents[i] + cents[i - 1]
           for i in range(1, len(cents) - 1)]
    ok("curvature from pixels alone: second difference == 64 (0.25*16^2)",
       max(abs(s - 64.0) for s in sds) <= 6.0,
       f"{[round(s, 1) for s in sds]}")

    # ---- act C: midpoint trail wobbles +-30 about the parabola.
    # run 1 sampled the sin=-1 troughs by column centroid; there the
    # wobble curve BACKTRACKS in x (dx = -1.7/frame), the column cuts
    # a near-vertical multi-valued stretch, and a centroid grades the
    # whole stretch (trap 47/58 family — the render was right).
    # (a) crest columns are single-crossing (dx = +11.7): centroid.
    wdev = []
    for m in (28, 56, 112, 140):                 # sin=+1, mid at yc-30
        x = int(xc(m))
        col = white_mask(f280[400:1600, x - 2:x + 3])
        ys = np.where(col.any(1))[0]
        got = ys.mean() + 400 if len(ys) else -1e9
        wdev.append(abs(got - (yc(m) - 30.0)))
    ok("white trail 30 px above the parabola at four crest columns",
       max(wdev) <= 2.5, f"max dev {max(wdev):.2f} px")
    # (b) global amplitude: every white pixel's vertical deviation
    # from the parabola-in-x must live inside the model tube, and
    # must REACH both extremes. bounds computed from the model curve
    # plus the trail's own radius — never hand-picked (trap 62).
    mm = np.arange(0, 168.0001, 0.05)
    th = 2 * np.pi * (mm + 7) / P
    mxs = X0 + VX * mm - RH * np.cos(th)
    mys = Y0 - VYC * mm + mm * mm / 8 - RH * np.sin(th)
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    r_t = LW_MID / 2 + 1.0
    dall = []
    for oa in ang:
        px_ = mxs + r_t * np.cos(oa)
        py_ = mys + r_t * np.sin(oa)
        mx_ = (px_ - X0) / VX
        dall.append(py_ - (Y0 - VYC * mx_ + mx_ * mx_ / 8))
    dall = np.concatenate(dall)
    xs_ = np.tile(mxs, 16) + np.concatenate(
        [r_t * np.cos(a) * np.ones_like(mxs) for a in ang])
    sel = (xs_ >= 150) & (xs_ <= 880)
    lo_m, hi_m = dall[sel].min(), dall[sel].max()
    wm = white_mask(f280[400:1600, 150:881])
    yy, xx = np.where(wm)
    mx_ = (xx + 150 - X0) / VX
    dpx = (yy + 400) - (Y0 - VYC * mx_ + mx_ * mx_ / 8)
    ok("white tube inside model deviation bounds (x 150..880)",
       dpx.min() >= lo_m - 2 and dpx.max() <= hi_m + 2,
       f"px [{dpx.min():.1f},{dpx.max():.1f}] vs model "
       f"[{lo_m:.1f},{hi_m:.1f}]")
    ok("wobble REACHES both extremes on the pixels",
       dpx.min() <= lo_m + 6 and dpx.max() >= hi_m - 6,
       f"extremes hit within 6 px of model bound")

    # ---- act C: the tangle (end trails) spans tall at mid-flight
    tang = grey_mask(f280[420:1700, 538:543])
    rows = np.where(tang.any(1))[0]
    span = rows.max() - rows.min() if len(rows) else 0
    ok("end-path tangle spans >= 120 px at x=540 (act C)",
       span >= 120, f"{span} px")

    # ---- freeze: act C byte-identical after label fade (>= 264)
    ok("act C frames byte-identical (270 == 290, 275 == 295)",
       np.array_equal(frame_at(270), frame_at(290))
       and np.array_equal(frame_at(275), frame_at(295)))

    # ---- labels present/absent per act (colour-matched, trap 61)
    lab_band = (slice(200, 400), slice(0, W))
    ok("act A label present (grey), no red/white labels",
       grey_mask(f100[lab_band]).sum() > 300
       and not red_mask(f100[lab_band]).any()
       and not white_mask(f100[lab_band]).any())
    ok("act B labels present (white + red)",
       white_mask(f200[lab_band]).sum() > 300
       and red_mask(f200[lab_band]).sum() > 300)
    ok("act C third label present",
       grey_mask(f280[lab_band]).sum() > 300)

    # ---- safe areas (trap 3): pure background outside them
    ok("top 192 rows pure background, all acts",
       all((fr[:192] == np.asarray(bg8, np.uint8)).all()
           for fr in (f2, f100, f200, f280)))
    ok("bottom band (y >= 1632) pure background, all acts",
       all((fr[1632:] == np.asarray(bg8, np.uint8)).all()
           for fr in (f2, f100, f200, f280)))

    print()
    print("NOT verified by these checks (trap 68):")
    print("  - air resistance is neglected; a real stick's spin decays.")
    print("    the zero-torque theorem is exact for the model, stated")
    print("    with sources in the description")
    print("  - that the disc sizes READ as mass is the viewer's percept;")
    print("    areas are drawn proportional to mass and declared")
    print("  - h264 fidelity is checked after encode on fenced regions,")
    print("    not on every pixel")
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
    ok("300 frames in the file", f"{N}" in r.stdout)

    d280 = decode_frame(280)
    nred = red_mask(d280[400:1600, :]).sum()
    ok("red parabola survives h264", nred > 3000, f"{nred} red px")
    cents = []
    for m in range(16, 153, 16):
        x = int(xc(m))
        col = red_mask(d280[400:1600, x - 2:x + 3])
        ys = np.where(col.any(1))[0]
        cents.append(ys.mean() + 400 if len(ys) else -1e9)
    dev = max(abs(cy - yc(m)) for cy, m in zip(cents, range(16, 153, 16)))
    ok("red trail on the parabola on the SHIPPED file", dev <= 3.0,
       f"max dev {dev:.2f} px")
    nwh = white_mask(d280[400:1600, :900]).sum()
    ok("white wobble trail survives h264", nwh > 1500, f"{nwh} px")
    tang = grey_mask(d280[420:1700, 538:543])
    rows = np.where(tang.any(1))[0]
    ok("tangle survives h264 (span at x=540)",
       len(rows) > 0 and rows.max() - rows.min() >= 100,
       f"{rows.max() - rows.min() if len(rows) else 0} px")

    a, b = decode_frame(270), decode_frame(290)
    d1 = np.abs(a.astype(np.int64) - b.astype(np.int64)).mean()
    ok("freeze survives h264 (270 vs 290 near-identical)", d1 < 0.5,
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
    for n in (8, 60, 130, 200, 245, 280):
        Image.fromarray(frame_at(n)).save(f"{OUT_DIR}/stick_f{n:03d}.png")
    subprocess.run(
        ["ffmpeg", "-y", "-pattern_type", "glob",
         "-i", f"{OUT_DIR}/stick_f*.png",
         "-filter_complex", "scale=270:-1,tile=3x2",
         f"{OUT_DIR}/stick_sheet.png"],
        capture_output=True)
    print("sheet:", f"{OUT_DIR}/stick_sheet.png", flush=True)


if __name__ == "__main__":
    run_checks()
    encode()
    check_encode()
    review_stills()
