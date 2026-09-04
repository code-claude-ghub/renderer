#!/usr/bin/env python3
"""BURST — the exploding shell and the hidden point (family #2).

A shell rides a parabola, tracing red (the red point from STICK).
At m=56 it explodes into FOUR EQUAL pieces. Each piece flies its own
new parabola — and the average of the four positions never leaves
the old arc, bitwise, because the explosion's forces are internal
and cancel (Newton III). At the burst a dashed grey curve flashes
up: the path the shell WOULD have taken. The red point — drawn each
frame at the average of the pieces, with grey spokes to show where
it comes from — swallows the dashes one by one.

  ACT A (n 0..55):  one shell, one arc, red trace.
  BURST (n 56..62): ring flash; dashed prediction appears.
  ACT B (n 56..168): four pieces, four grey arcs; spokes from each
    piece to the red average point riding the dashed curve.
  ACT C (n 169..228): freeze on the composite.

Physics exact, not schematic (sources verified live, in the
description): internal forces cannot change the CoM's momentum
(OpenStax Univ. Physics 9.10); the trajectory is Galileo's parabola
(Wikipedia, Projectile motion). Air resistance neglected, declared.

Exactness (proven in scripts/feas_burst.py, 26 checks):
  x_c(m) = 140 + 4m; y_c(m) = 1480 - 21m + m^2/8 (dyadic; second
  difference BITWISE 0.25 across all 167 triples)
  kicks are dyadic quarters summing to exactly (0, 0); the average
  of the four fragment positions == the continued parabola BITWISE
  (not epsilon — equal), m 56..168
  each fragment: its own y second difference bitwise 0.25
  shell area == 4 fragment areas exactly (36^2 == 4*18^2)
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
MB = 56
X0, VX = 140, 4
Y0, VYC = 1480, 21
R_SHELL, R_FRAG = 36.0, 18.0
R_DOT = 14.0
LW_RED, LW_FR, LW_DASH, LW_SPOKE, LW_RING = 11.0, 6.0, 6.0, 5.0, 7.0

K = [(1.75, -1.50),
     (0.50, 1.50),
     (-0.75, -1.00),
     (-1.50, 1.00)]

A_HI = 55
RING_LO, RING_HI = 56, 62
C_LO = 169
N = 229
COL_4TR = 620
SPOKE_MF, SPOKE_T = 160, 0.7

BGC = (0.055, 0.060, 0.078)
C_REDTR = (0.88, 0.18, 0.14)
C_FRTR = (0.36, 0.40, 0.48)
C_DASH = (0.50, 0.53, 0.62)
C_SPOKE = (0.48, 0.44, 0.38)
C_DISC = (0.80, 0.82, 0.86)
C_DOTR = (0.98, 0.25, 0.18)
C_RING = (0.97, 0.97, 0.99)
C_LBL = (0.55, 0.57, 0.62)

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
OUT_DIR = "/home/maroon-beret/projects/active/youtube/youtube-channel/out"
STAMP = time.strftime("%H%M%S")
OUT_MP4 = f"{OUT_DIR}/burst_{STAMP}.mp4"


def xc(m):
    return float(X0 + VX * m)


def yc(m):
    return Y0 - VYC * m + (m * m) / 8


def frag(i, m):
    t = m - MB
    return (xc(m) + K[i][0] * t, yc(m) + K[i][1] * t)


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


def disc_cov(cx, cy, r):
    x0, x1 = int(np.floor(cx - r)) - 2, int(np.ceil(cx + r)) + 3
    y0, y1 = int(np.floor(cy - r)) - 2, int(np.ceil(cy + r)) + 3
    xx = np.arange(x0, x1, dtype=np.float64)
    yy = np.arange(y0, y1, dtype=np.float64)
    d = np.hypot(xx[None, :] - cx, yy[:, None] - cy)
    return x0, y0, np.clip(r + 0.5 - d, 0.0, 1.0)


def ring_cov(cx, cy, r, lw):
    pad = lw / 2 + 2
    x0, x1 = int(np.floor(cx - r - pad)), int(np.ceil(cx + r + pad)) + 1
    y0, y1 = int(np.floor(cy - r - pad)), int(np.ceil(cy + r + pad)) + 1
    xx = np.arange(x0, x1, dtype=np.float64)
    yy = np.arange(y0, y1, dtype=np.float64)
    d = np.hypot(xx[None, :] - cx, yy[:, None] - cy)
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

LBL_A = text_cov("one shell, one arc", 34)
LBL_B1 = text_cov("the average of the four pieces", 34)
LBL_B2 = text_cov("no piece keeps the old arc", 34)
LBL_C = text_cov("the explosion never moved it", 34)
LBL_Y_A, LBL_Y_B1, LBL_Y_B2, LBL_Y_C = 230, 230, 292, 356

# static dashed prediction of the continuation (m 56..168)
DASH = np.zeros((H, W), np.float64)
for _j in range(MB, F - 2, 7):
    _a, _b = _j, min(_j + 3, F)
    stamp_max(DASH, *seg_cov(xc(_a), yc(_a), xc(_b), yc(_b), LW_DASH))


# ---------------------------------------------------------------- state
def new_trails():
    return (np.zeros((H, W), np.float64),      # red CoM trail
            np.zeros((H, W), np.float64))      # fragment trails


def append_red(tr_red, m):
    if m >= 1:
        stamp_max(tr_red, *seg_cov(xc(m - 1), yc(m - 1),
                                   xc(m), yc(m), LW_RED))


def append_frags(tr_fr, m):
    if m >= MB + 1:
        for i in range(4):
            a, b = frag(i, m - 1), frag(i, m)
            stamp_max(tr_fr, *seg_cov(*a, *b, LW_FR))


def fade(n, n0, span=10):
    return float(np.clip((n - n0 + 1) / span, 0.0, 1.0))


def composite(n, tr_red, tr_fr):
    img = BG.copy()
    m = m_of(n)
    # dashed prediction under everything (appears at the burst)
    if n >= MB:
        da = 0.85 * fade(n, MB, 6)
        a = DASH[..., None] * da
        img *= (1 - a)
        img += np.asarray(C_DASH, np.float64)[None, None, :] * a
    for buf, col in ((tr_fr, C_FRTR), (tr_red, C_REDTR)):
        a = buf[..., None]
        img *= (1 - a)
        img += np.asarray(col, np.float64)[None, None, :] * a
    # spokes: piece -> average point (acts B and C, after the flash)
    if n >= RING_HI + 2:
        for i in range(4):
            fx, fy = frag(i, m)
            x0, y0, cv = seg_cov(fx, fy, xc(m), yc(m), LW_SPOKE)
            comp_bbox(img, x0, y0, cv * 0.85, C_SPOKE)
    # shell (act A) or fragments (B/C)
    if n <= A_HI:
        comp_bbox(img, *disc_cov(xc(m), yc(m), R_SHELL), C_DISC)
    else:
        for i in range(4):
            fx, fy = frag(i, m)
            comp_bbox(img, *disc_cov(fx, fy, R_FRAG), C_DISC)
    # burst ring flash: full brightness for two frames (a real pop —
    # run 1's fade never let the ring reach mask-white), then fade
    if RING_LO <= n <= RING_HI:
        r = 36.0 + 12.0 * (n - RING_LO)
        al = 1.0 if n <= RING_LO + 1 else 1.0 - (n - RING_LO - 1) / 5.0
        x0, y0, cv = ring_cov(xc(MB), yc(MB), r, LW_RING)
        comp_bbox(img, x0, y0, cv * al, C_RING)
    # the red point — bitwise the average of the four pieces
    comp_bbox(img, *disc_cov(xc(m), yc(m), R_DOT), C_DOTR)
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
    tr_red, tr_fr = new_trails()
    for m in range(1, m_of(n) + 1):
        append_red(tr_red, m)
        append_frags(tr_fr, m)
    return composite(n, tr_red, tr_fr)


def render_frames():
    """Incremental generator — identical op order to frame_at."""
    tr_red, tr_fr = new_trails()
    for n in range(N):
        if 1 <= n <= F:
            append_red(tr_red, n)
            append_frags(tr_fr, n)
        yield composite(n, tr_red, tr_fr)


# ---------------------------------------------------------------- checks
# FENCE AUDIT (written before the first render):
#   RED (r-g > 60): red trail + dot (rows 400..1700, all acts, from
#     n=1) and label LBL_B1 (rows 230..280, n >= 64). Row fence 400
#     separates them.
#   WHITE (min >= 225): the burst ring ONLY, n 56..62. Discs are
#     (204,209,219) — min 204 < 225. Red dot min channel 46.
#   COOL GREY (b > r+8, 40 < max < 160): fragment trails (n >= 57),
#     the dash (n >= 56), labels A/B2/C. Rows >= 420 exclude labels.
#   WARM (r > b+8 AND g-b >= 8, 40 < max < 160): spokes ONLY
#     (n >= 64). BG fails (b > r); red trail/dot core fails on
#     max >= 160; red AA edge blends fail on g-b (<= 2 until bright
#     enough to be excluded by max — verified algebraically).
#   DISC colour (each channel within 10 of (204,209,219)): the shell
#     and the four fragment discs only.
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
    # r>b alone also matches the red trail's AA edge blends (run 1);
    # the spoke's g sits ABOVE b (g-b ~ 12) where red's does not
    # (g-b <= 2 until the pixel is bright enough for max >= 160).
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

    ok("shell has visibly moved by frame 8",
       not np.array_equal(f2, frame_at(8)))

    # ---- act A: shell disc at the model CoM (colour-matched box)
    f30 = frame_at(30)
    px, py = xc(30), yc(30)
    box = disc_mask(f30[int(py) - 46:int(py) + 46,
                        int(px) - 46:int(px) + 46])
    cc = centroid(box)
    dev = (np.hypot(cc[0] - 46 - (px - int(px)),
                    cc[1] - 46 - (py - int(py))) if cc else 99.0)
    ok("shell disc centred on the model CoM (frame 30)", dev < 2.0,
       f"dev {dev:.2f} px")

    # ---- red leading edge before AND after the burst
    for nn in (40, 120, 160):
        fr = frame_at(nn)
        band = red_mask(fr[400:1700, :])
        xs = np.where(band.any(0))[0]
        lead = xs.max() if len(xs) else -1
        ok(f"red leading edge == CoM + dot radius (frame {nn})",
           abs(lead - (xc(m_of(nn)) + R_DOT)) <= 2.5,
           f"{lead} vs {xc(m_of(nn)) + R_DOT:.0f}")

    # ---- ring flash: mask-white only while the flash holds full
    #      brightness (n 56..57); the fade drops below the threshold
    f50 = frame_at(50)
    f57 = frame_at(57)
    ok("no white before the burst (frame 50)",
       not white_mask(f50).any())
    ok("ring flash present (frame 57)", white_mask(f57).sum() > 400,
       f"{white_mask(f57).sum()} px")
    ok("no white after the flash (frame 100)",
       not white_mask(f100).any())

    # ---- spokes: warm ink only after the flash, at feas-proven pts
    ok("no spoke ink in act A (frame 50)",
       not warm_mask(f50[420:1700, :]).any())
    f160 = frame_at(160)
    hits = 0
    for i in range(4):
        fx, fy = frag(i, SPOKE_MF)
        sx = fx + SPOKE_T * (xc(SPOKE_MF) - fx)
        sy = fy + SPOKE_T * (yc(SPOKE_MF) - fy)
        if warm_mask(f160[int(sy) - 3:int(sy) + 4,
                          int(sx) - 3:int(sx) + 4]).any():
            hits += 1
    ok("all four spokes present at feas sample points (frame 160)",
       hits == 4, f"{hits}/4")

    # ---- fragments at model positions + the trap-66 coupling check:
    #      average of MEASURED disc centroids == the model parabola
    devs, cents = [], []
    for i in range(4):
        fx, fy = frag(i, 160)
        box = disc_mask(f160[int(fy) - 26:int(fy) + 26,
                             int(fx) - 26:int(fx) + 26])
        cc = centroid(box)
        if cc is None:
            devs.append(99.0)
            continue
        gx, gy = cc[0] + int(fx) - 26, cc[1] + int(fy) - 26
        cents.append((gx, gy))
        devs.append(np.hypot(gx - fx, gy - fy))
    ok("fragment discs on their own parabolas (frame 160)",
       max(devs) < 2.0, f"max dev {max(devs):.2f} px")
    if len(cents) == 4:
        ax = sum(c[0] for c in cents) / 4
        ay = sum(c[1] for c in cents) / 4
        cdev = np.hypot(ax - xc(160), ay - yc(160))
    else:
        cdev = 99.0
    ok("average of the four MEASURED discs == model parabola",
       cdev < 2.5, f"dev {cdev:.2f} px")

    # ---- act C: red curvature off the pixels (stride 16 -> sd 64)
    f210 = frame_at(210)
    cents = []
    for m in range(16, 153, 16):
        x = int(xc(m))
        col = red_mask(f210[400:1700, x - 2:x + 3])
        ys = np.where(col.any(1))[0]
        cents.append(ys.mean() + 400 if len(ys) else -1e9)
    dev = max(abs(cy - yc(m)) for cy, m in zip(cents, range(16, 153, 16)))
    ok("red trail on the model parabola, 9 columns", dev <= 2.0,
       f"max dev {dev:.2f} px")
    sds = [cents[i + 1] - 2 * cents[i] + cents[i - 1]
           for i in range(1, len(cents) - 1)]
    ok("curvature from pixels alone: second difference == 64 (0.25*16^2)",
       max(abs(s - 64.0) for s in sds) <= 6.0,
       f"{[round(s, 1) for s in sds]}")

    # ---- red continuity THROUGH the burst column (x 340..390)
    cont = all(
        red_mask(f210[600:800, x:x + 1]).any() for x in range(340, 391))
    ok("red trail unbroken through the burst (x 340..390)", cont)

    # ---- four grey trails at the cluster column (feas-proven gaps)
    gm = grey_mask(f210[420:1700, COL_4TR - 2:COL_4TR + 3])
    cl = clusters(gm)
    exp = []
    for i in range(4):
        ms = (COL_4TR - X0 + MB * K[i][0]) / (VX + K[i][0])
        exp.append(yc(ms) + K[i][1] * (ms - MB))
    exp.sort()
    got = sorted(float(c.mean()) + 420 for c in cl)
    match = (len(got) == len(exp)
             and max(abs(g - e) for g, e in zip(got, exp)) < 3.0)
    ok("four fragment-trail crossings at the check column",
       match, f"got {[round(g) for g in got]} vs "
       f"exp {[round(e) for e in exp]}")

    # ---- dash: ahead of the red point at n=70, consumed by n=210
    f70 = frame_at(70)
    dx_, dy_ = int(xc(140)), int(yc(140))
    ok("dash visible ahead of the red point (frame 70)",
       grey_mask(f70[dy_ - 4:dy_ + 5, dx_ - 4:dx_ + 5]).any())
    ok("red has consumed the dash there by the freeze",
       red_mask(f210[dy_ - 2:dy_ + 3, dx_ - 2:dx_ + 3]).any()
       and not grey_mask(f210[dy_ - 2:dy_ + 3, dx_ - 2:dx_ + 3]).any())

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
           for fr in (f2, f30, f57, f100, f160, f210)))
    ok("rows >= 1700 pure background, all sampled frames",
       all((fr[1700:] == np.asarray(bg8, np.uint8)).all()
           for fr in (f2, f30, f57, f100, f160, f210)))

    print()
    print("NOT verified by these checks (trap 68):")
    print("  - air resistance is neglected; declared in the description")
    print("  - the theorem holds until any fragment lands; no ground is")
    print("    drawn and no fragment reaches one inside the flight —")
    print("    stated honestly in the description")
    print("  - the explosion is an instantaneous velocity change with")
    print("    zero mass loss; equal masses make the plain average the")
    print("    mass-weighted one")
    print("  - that the ring flash READS as an explosion is the")
    print("    viewer's percept")
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
        x = int(xc(m))
        col = red_mask(d210[400:1700, x - 2:x + 3])
        ys = np.where(col.any(1))[0]
        cents.append(ys.mean() + 400 if len(ys) else -1e9)
    dev = max(abs(cy - yc(m)) for cy, m in zip(cents, range(16, 153, 16)))
    ok("red trail on the parabola on the SHIPPED file", dev <= 3.0,
       f"max dev {dev:.2f} px")
    gm = grey_mask(d210[420:1700, COL_4TR - 2:COL_4TR + 3])
    ok("four grey trails survive h264 at the check column",
       len(clusters(gm)) == 4, f"{len(clusters(gm))} clusters")
    d57 = decode_frame(57)
    ok("ring flash survives h264 (frame 57)",
       white_mask(d57).sum() > 300, f"{white_mask(d57).sum()} px")
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
    for n in (30, 58, 100, 140, 168, 210):
        Image.fromarray(frame_at(n)).save(f"{OUT_DIR}/burst_f{n:03d}.png")
    subprocess.run(
        ["ffmpeg", "-y", "-pattern_type", "glob",
         "-i", f"{OUT_DIR}/burst_f*.png",
         "-filter_complex", "scale=270:-1,tile=3x2",
         f"{OUT_DIR}/burst_sheet.png"],
        capture_output=True)
    print("sheet:", f"{OUT_DIR}/burst_sheet.png", flush=True)


if __name__ == "__main__":
    run_checks()
    encode()
    check_encode()
    review_stills()
