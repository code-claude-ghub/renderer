#!/usr/bin/env python3
"""OVAL — the near-ellipse: two cycloid arches, one honest sliver.

A wheel rolls along a line; a point on its rim draws a cycloid arch
(the pen this audience has watched stop dead every turn). It rolls
back along the underside and the mirror arch closes the shape into
an oval 2*pi*R wide and 4R tall. It looks like an ellipse. It is
not: the TRUE ellipse through the same four extreme points contains
the oval entirely, touching it at exactly those four points, and
the sliver of daylight between them has area EXACTLY
2*pi*(pi-3)*R^2 — the ellipse beats the oval by a factor of pi/3,
so the miss is (pi-3)/pi of the ellipse: "pi minus 3 parts in pi",
about 4.5%.

  ACT A (n 0..77):    wheel rolls left->right on the line; the rim
                      pen (red dot) draws the warm top arch
  ACT B (n 78..155):  the mirror pass along the underside; the oval
                      closes at the left kiss
  ACT E (n 162..209): the true ellipse sweeps on in red
  ACT G (n 210..):    four white kiss ticks; the gap fill (dim
                      magenta) fades in; the area labels land
  FREEZE (n 238..281)

Physics/maths exact, sources verified live (in the description):
cycloid definition and arch area 3*pi*r^2 (Wikipedia, Cycloid —
Roberval 1634, Torricelli 1644; Galileo weighed cut-outs ~1599);
ellipse area pi*a*b (Wikipedia). The inscription (oval strictly
inside the ellipse, four contact points) is proven numerically to
machine precision in scripts/feas_oval.py (38 checks), along with
every number on screen.

Born from @Dominic-qv3yt's "two cycloids make an ellipse" via
@rorucopexperements's reading of it as a claim about the SHAPE.
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
R = 140.0
CX, CY = 540.0, 960.0
A_ELL = math.pi * R
B_ELL = 2.0 * R
X_L = CX - A_ELL
X_R = CX + A_ELL

A_HI = 77
B_LO, B_HI = 78, 155
E_LO, E_HI = 162, 209
TICK_N = 210
GAP_LO = 214
FREEZE = 238
N = 282

LW_OVAL, LW_ELL, LW_GROUND, LW_RING, LW_SPOKE, LW_TICK = \
    5.5, 5.5, 4.0, 5.0, 4.0, 3.5
R_PEN = 9.0
R_TICK = 16.0
GAP_PX_PRED = 10335.0        # from feas_oval.py, margin 3.5 px/side

ROWS_ORDER = (770, 820, 1100, 1150)
ROWS_TIGHT = (800, 840, 1080, 1120)
COLS_CREST = (480, 600)

BGC = (0.055, 0.060, 0.078)
C_GROUND = (0.36, 0.40, 0.48)
C_DISC = (0.80, 0.82, 0.86)
C_OVAL = (0.92, 0.72, 0.20)
C_ELL = (0.88, 0.18, 0.14)
C_PEN = (0.98, 0.25, 0.18)
C_GAP = (0.40, 0.08, 0.30)
C_TICK = (0.97, 0.97, 0.99)
C_LBL = (0.55, 0.57, 0.62)

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
OUT_DIR = "/home/maroon-beret/projects/active/youtube/youtube-channel/out"
STAMP = time.strftime("%H%M%S")
OUT_MP4 = f"{OUT_DIR}/oval_{STAMP}.mp4"


def arch_top(t):
    return (CX - A_ELL + R * (t - np.sin(t)),
            CY - R * (1.0 - np.cos(t)))


def mirror(x, y):
    return 2.0 * CX - x, 2.0 * CY - y


def wheel_a(t):
    return (CX - A_ELL + R * t, CY - R)


def ellipse_pt(phi):
    return CX + A_ELL * np.cos(phi), CY - B_ELL * np.sin(phi)


def t_of(n):
    return 2.0 * math.pi * n / 77.0


def u_of(n):
    return 2.0 * math.pi * (n - B_LO) / 77.0


def phi_of(n):
    return 2.0 * math.pi * (n - (E_LO - 1)) / 48.0


def pen_at(n):
    """Model pen position during acts A and B."""
    if n <= A_HI:
        return arch_top(t_of(min(n, A_HI)))
    return mirror(*arch_top(u_of(min(n, B_HI))))


def wheel_at(n):
    if n <= A_HI:
        return wheel_a(t_of(min(n, A_HI)))
    return mirror(*wheel_a(u_of(min(n, B_HI))))


def cyc_x_at(dy):
    d = dy / R
    t = np.arccos(1.0 - d)
    xl = CX - A_ELL + R * (t - np.sin(t))
    xr = CX - A_ELL + R * ((2.0 * np.pi - t) + np.sin(t))
    return xl, xr


def ell_x_at(dy):
    s = np.sqrt(np.clip(1.0 - (dy / B_ELL) ** 2, 0.0, None))
    return CX - A_ELL * s, CX + A_ELL * s


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

LBL_A = text_cov("a wheel rolls — its rim point draws a cycloid", 34)
LBL_B = text_cov("back along the underside — the oval closes", 34)
LBL_E = text_cov("the true ellipse through the same four points", 34)
LBL_G2 = text_cov("they touch at exactly four points", 34)
LBL_G1 = text_cov("areas: oval 6πr² · ellipse 2π²r²", 34)
LBL_C = text_cov("the miss: exactly π−3 parts in π", 34)
LBL_Y_A, LBL_Y_B, LBL_Y_E = 230, 292, 354
LBL_Y_G2, LBL_Y_G1, LBL_Y_C = 1450, 1512, 1574

# ground line, static
GROUND = np.zeros((H, W), np.float64)
stamp_max(GROUND, *seg_cov(80.0, CY, 1000.0, CY, LW_GROUND))

# kiss ticks, static
TICKS = np.zeros((H, W), np.float64)
for _tx, _ty in ((X_L, CY), (X_R, CY), (CX, CY - 2 * R), (CX, CY + 2 * R)):
    stamp_max(TICKS, *ring_cov(_tx, _ty, R_TICK, LW_TICK))

# gap fill, static: per row, the two intervals between the curves.
# Drawn up to the curves; the strokes on top cover the edges.
GAPBUF = np.zeros((H, W), np.float64)
for _yy in range(681, 1240):
    _dy = abs(CY - _yy)
    if _dy <= 0.0 or _dy >= 2 * R:
        continue
    _cl, _cr = (float(v) for v in cyc_x_at(np.float64(_dy)))
    _el, _er = (float(v) for v in ell_x_at(np.float64(_dy)))
    for _a, _b in ((_el, _cl), (_cr, _er)):
        if _b - _a <= 0.0:
            continue
        _x0, _x1 = int(math.floor(_a)), int(math.ceil(_b))
        _xs = np.arange(_x0, _x1 + 1, dtype=np.float64)
        _cov = np.clip(np.minimum(_b, _xs + 1.0) - np.maximum(_a, _xs), 0, 1)
        _sl = GAPBUF[_yy, max(_x0, 0):min(_x1 + 1, W)]
        np.maximum(_sl, _cov[max(_x0, 0) - _x0:len(_cov)
                             - (max(0, _x1 + 1 - W))], out=_sl)


# ---------------------------------------------------------------- state
def new_trails():
    return (np.zeros((H, W), np.float64),      # warm oval trail
            np.zeros((H, W), np.float64))      # red ellipse trail


def _chain(buf, pts, lw):
    for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
        stamp_max(buf, *seg_cov(xa, ya, xb, yb, lw))


def append_oval(tr_o, m):
    if 1 <= m <= A_HI:
        ts = [t_of(m - 1) + k * (t_of(m) - t_of(m - 1)) / 4.0
              for k in range(5)]
        _chain(tr_o, [arch_top(t) for t in ts], LW_OVAL)
    elif B_LO + 1 <= m <= B_HI:
        us = [u_of(m - 1) + k * (u_of(m) - u_of(m - 1)) / 4.0
              for k in range(5)]
        _chain(tr_o, [mirror(*arch_top(u)) for u in us], LW_OVAL)


def append_ell(tr_e, m):
    if E_LO <= m <= E_HI:
        ps = [phi_of(m - 1) + k * (phi_of(m) - phi_of(m - 1)) / 4.0
              for k in range(5)]
        _chain(tr_e, [ellipse_pt(p) for p in ps], LW_ELL)


def fade(n, n0, span=10):
    return float(np.clip((n - n0 + 1) / span, 0.0, 1.0))


def composite(n, tr_o, tr_e):
    img = BG.copy()
    # ground line, always, dim
    a = GROUND[..., None] * 0.8
    img *= (1 - a)
    img += np.asarray(C_GROUND, np.float64)[None, None, :] * a
    # gap fill under everything else (act G on)
    if n >= GAP_LO:
        a = GAPBUF[..., None] * 0.95 * fade(n, GAP_LO, 10)
        img *= (1 - a)
        img += np.asarray(C_GAP, np.float64)[None, None, :] * a
    # trails: warm oval, then red ellipse on top
    for buf, col in ((tr_o, C_OVAL), (tr_e, C_ELL)):
        a = buf[..., None]
        img *= (1 - a)
        img += np.asarray(col, np.float64)[None, None, :] * a
    # the rolling wheel: rim ring + spoke to the pen (acts A, B)
    if n <= B_HI:
        wx, wy = wheel_at(n)
        px, py = pen_at(n)
        # full alpha: the DISC fence measures this ink, and a 0.9
        # blend lands ~19 grey levels outside it (run-1 crash)
        x0, y0, cv = ring_cov(wx, wy, R, LW_RING)
        comp_bbox(img, x0, y0, cv, C_DISC)
        x0, y0, cv = seg_cov(wx, wy, px, py, LW_SPOKE)
        comp_bbox(img, x0, y0, cv, C_DISC)
        # the pen itself — the red rim point
        comp_bbox(img, *disc_cov(px, py, R_PEN), C_PEN)
    # kiss ticks
    if n >= TICK_N:
        x0, y0 = 0, 0
        a = TICKS[..., None] * fade(n, TICK_N, 8)
        img *= (1 - a)
        img += np.asarray(C_TICK, np.float64)[None, None, :] * a
    # labels (safe area: trap 3)
    if n >= 14:
        cv = LBL_A * 0.9 * fade(n, 14)
        comp_bbox(img, (W - LBL_A.shape[1]) // 2, LBL_Y_A, cv, C_LBL)
    if n >= 92:
        cv = LBL_B * 0.9 * fade(n, 92)
        comp_bbox(img, (W - LBL_B.shape[1]) // 2, LBL_Y_B, cv, C_LBL)
    if n >= E_LO + 6:
        cv = LBL_E * fade(n, E_LO + 6)
        comp_bbox(img, (W - LBL_E.shape[1]) // 2, LBL_Y_E, cv, C_ELL)
    if n >= TICK_N + 2:
        cv = LBL_G2 * 0.9 * fade(n, TICK_N + 2)
        comp_bbox(img, (W - LBL_G2.shape[1]) // 2, LBL_Y_G2, cv, C_LBL)
    if n >= GAP_LO + 4:
        cv = LBL_G1 * fade(n, GAP_LO + 4)
        comp_bbox(img, (W - LBL_G1.shape[1]) // 2, LBL_Y_G1, cv, C_OVAL)
    if n >= GAP_LO + 12:
        cv = LBL_C * fade(n, GAP_LO + 12)
        comp_bbox(img, (W - LBL_C.shape[1]) // 2, LBL_Y_C, cv, C_ELL)
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def frame_at(n):
    """Pure reconstruction — rebuilds trail state from scratch."""
    tr_o, tr_e = new_trails()
    for m in range(1, min(n, B_HI) + 1):
        append_oval(tr_o, m)
    for m in range(E_LO, min(n, E_HI) + 1):
        append_ell(tr_e, m)
    return composite(n, tr_o, tr_e)


def render_frames():
    """Incremental generator — identical op order to frame_at."""
    tr_o, tr_e = new_trails()
    for n in range(N):
        if 1 <= n <= B_HI:
            append_oval(tr_o, n)
        if E_LO <= n <= E_HI:
            append_ell(tr_e, n)
        yield composite(n, tr_o, tr_e)


# ---------------------------------------------------------------- checks
# FENCE AUDIT (proven on the palette in feas_oval.py):
#   RED (r-g > 60 AND r > 190): ellipse stroke core + red pen dot
#     (acts A/B only) + labels LBL_E (y 354..~400, n >= 168) and
#     LBL_C (y 1574.., n >= 226). Field checks bound rows 420..1400.
#   WARM (r > 180, g > 130, b < 100): oval trail core + LBL_G1
#     (y 1512.., n >= 218).
#   GAPF (r-g > 60, r < 150, 60 < b < 120): the gap fill interior
#     ONLY (dim magenta; the blue channel is the fence that no red
#     blend can enter).
#   GREY (b > r+8, 40 < max < 160): ground line + grey labels.
#   DISC (within 10 of (204,209,219)): wheel ring + spoke, acts A/B.
#   WHITE (min >= 225): kiss ticks only, n >= 210.
CHECKS = {"pass": 0, "fail": 0}


def ok(name, cond, detail=""):
    s = "ok  " if cond else "FAIL"
    CHECKS["pass" if cond else "fail"] += 1
    print(f"{s} {name}" + (f" [{detail}]" if detail else ""), flush=True)


def red_mask(fr):
    r = fr[:, :, 0].astype(np.int64)
    g = fr[:, :, 1].astype(np.int64)
    return (r - g > 60) & (r > 190)


def warm_mask(fr):
    return ((fr[:, :, 0] > 180) & (fr[:, :, 1] > 130)
            & (fr[:, :, 2] < 100))


def gap_mask(fr):
    r = fr[:, :, 0].astype(np.int64)
    g = fr[:, :, 1].astype(np.int64)
    b = fr[:, :, 2].astype(np.int64)
    return (r - g > 60) & (r < 150) & (b > 60) & (b < 120)


def grey_mask(fr):
    mx = fr.max(2).astype(np.int64)
    return ((fr[:, :, 2].astype(np.int64) > fr[:, :, 0].astype(np.int64) + 8)
            & (mx > 40) & (mx < 160))


def white_mask(fr):
    return fr.min(2) >= 225


DISC8 = np.asarray([np.uint8(v * 255 + 0.5) for v in C_DISC], np.int64)


def disc_mask(fr):
    d = np.abs(fr.astype(np.int64) - DISC8[None, None, :])
    return d.max(2) <= 10


def centroid(mask):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def two_clusters_x(mask, x_off):
    """Column positions of the two stroke clusters in a row-slice
    mask. Clusters narrower than 5 px are dropped before counting:
    4:2:0 chroma synthesises 1-3 px fence-crossing strays at the
    gap-fill boundaries (measured on the decoded file, run-3 diag:
    strays span <= 3 px; a real stroke decodes 7-12 px wide)."""
    xs = np.where(mask.any(0))[0]
    if len(xs) == 0:
        return None
    cl = [c for c in np.split(xs, np.where(np.diff(xs) > 4)[0] + 1)
          if c.max() - c.min() >= 4]
    if len(cl) != 2:
        return None
    return [float(c.mean()) + x_off for c in cl]


def row_slice_devs(fr, rows, mask_fn, exp_fn):
    """Max |cluster - expected| over rows; None on cluster failure."""
    worst = 0.0
    for r_ in rows:
        dy = abs(CY - r_)
        exp = sorted(float(v) for v in exp_fn(np.float64(dy)))
        got = two_clusters_x(mask_fn(fr[r_ - 1:r_ + 2, 60:1020]), 60)
        if got is None:
            return None
        worst = max(worst, max(abs(g - e) for g, e in zip(got, exp)))
    return worst


def run_checks():
    print("== render checks ==", flush=True)
    bg8 = tuple(np.uint8(np.clip(v, 0, 1) * 255.0 + 0.5) for v in BGC)

    f2 = frame_at(2)
    ok("corner pixel is background", tuple(f2[4, 4]) == bg8,
       f"{tuple(f2[4, 4])} vs {bg8}")
    f260 = frame_at(260)
    frac = (f260.max(2) > 40).mean()
    ok("lit fraction sane (not blank, not floodlit)", 0.01 < frac < 0.30,
       f"{frac:.4f}")

    gen = render_frames()
    g50 = g150 = g260 = None
    for k, fr in enumerate(gen):
        if k == 50:
            g50 = fr
        elif k == 150:
            g150 = fr
        elif k == 260:
            g260 = fr
    ok("generator == frame_at, byte-exact (n=50, 150, 260)",
       np.array_equal(g50, frame_at(50))
       and np.array_equal(g150, frame_at(150))
       and np.array_equal(g260, f260))

    ok("wheel has visibly moved by frame 8",
       not np.array_equal(f2, frame_at(8)))

    # ---- act A frame 40: the trap-66 couplings, all from pixels
    f40 = frame_at(40)
    pxm, pym = pen_at(40)
    wxm, wym = wheel_at(40)
    pc = centroid(red_mask(f40[int(pym) - 16:int(pym) + 16,
                               int(pxm) - 16:int(pxm) + 16]))
    pen = ((pc[0] + int(pxm) - 16, pc[1] + int(pym) - 16) if pc else None)
    dev_pen = (math.hypot(pen[0] - pxm, pen[1] - pym) if pen else 99.0)
    ok("red pen on the model rim point (frame 40)", dev_pen < 2.0,
       f"dev {dev_pen:.2f} px")
    box = disc_mask(f40[int(wym) - 150:int(wym) + 150,
                        int(wxm) - 150:int(wxm) + 150])
    ys, xs = np.where(box)
    if len(ys):
        wc = ((xs.min() + xs.max()) / 2.0 + int(wxm) - 150,
              (ys.min() + ys.max()) / 2.0 + int(wym) - 150)
        dev_w = math.hypot(wc[0] - wxm, wc[1] - wym)
    else:
        wc, dev_w = (wxm, wym), 99.0
    ok("wheel ring extent centred on the model centre (frame 40)",
       dev_w < 2.5, f"dev {dev_w:.2f} px, {len(ys)} disc px")
    if pen:
        rr = math.hypot(pen[0] - wc[0], pen[1] - wc[1])
        ok("MEASURED pen sits R from the MEASURED wheel centre",
           abs(rr - R) < 2.5, f"{rr:.2f} vs {R}")
        mx_, my_ = (pen[0] + wc[0]) / 2.0, (pen[1] + wc[1]) / 2.0
        ok("spoke ink on the centre->pen midpoint (the pen is ON the wheel)",
           disc_mask(f40[int(my_) - 4:int(my_) + 5,
                         int(mx_) - 4:int(mx_) + 5]).any())
    ok("warm trail reaches the pen (frame 40)",
       warm_mask(f40[int(pym) - 14:int(pym) + 15,
                     int(pxm) - 14:int(pxm) + 15]).any())

    # ---- act B frame 120: same couplings on the mirror pass
    f120 = frame_at(120)
    pxm2, pym2 = pen_at(120)
    pc2 = centroid(red_mask(f120[int(pym2) - 16:int(pym2) + 16,
                                 int(pxm2) - 16:int(pxm2) + 16]))
    dev2 = (math.hypot(pc2[0] + int(pxm2) - 16 - pxm2,
                       pc2[1] + int(pym2) - 16 - pym2) if pc2 else 99.0)
    ok("red pen on the mirror rim point (frame 120)", dev2 < 2.0,
       f"dev {dev2:.2f} px")
    # the white tick rings' AA edges pass through the disc-grey band
    # on their way down to background ((214,214,219) at ~82% cover —
    # 20 px, all inside the four tick boxes, run-2 diag). The defect
    # this check hunts is a PERSISTED WHEEL, which cannot live inside
    # a 24 px tick box — so those boxes are excluded, and nothing
    # else is.
    nw = disc_mask(f260[420:1400, :]).copy()
    for tx, ty in ((X_L, CY), (X_R, CY), (CX, CY - 2 * R), (CX, CY + 2 * R)):
        nw[int(ty) - 24 - 420:int(ty) + 25 - 420,
           int(tx) - 24:int(tx) + 25] = False
    ok("no wheel ink after act B (frame 260, tick boxes excluded)",
       not nw.any(), f"{int(nw.sum())} px")

    # ---- the ellipse sweep is a sweep (frame 180)
    f180 = frame_at(180)
    ex1, ey1 = ellipse_pt(2.0)
    ex2, ey2 = ellipse_pt(5.0)
    ok("red ellipse present at swept phi=2.0 (frame 180)",
       red_mask(f180[int(ey1) - 6:int(ey1) + 7,
                     int(ex1) - 6:int(ex1) + 7]).any())
    ok("no red yet at unswept phi=5.0 (frame 180)",
       not red_mask(f180[int(ey2) - 6:int(ey2) + 7,
                         int(ex2) - 6:int(ex2) + 7]).any())
    f100 = frame_at(100)
    ex3, ey3 = ellipse_pt(2.2)
    ok("no ellipse before act E (frame 100, top-left arc point)",
       not red_mask(f100[int(ey3) - 5:int(ey3) + 6,
                         int(ex3) - 5:int(ex3) + 6]).any())

    # ---- final frame: the four-cluster ordering (trap 76: row
    #      slices; ROWS_ORDER includes the max-gap rows, loose tol)
    dev_ro = row_slice_devs(f260, ROWS_ORDER, red_mask, ell_x_at)
    ok("red ellipse clusters at the model x (4 order rows)",
       dev_ro is not None and dev_ro <= 3.5,
       f"max dev {dev_ro if dev_ro is None else round(dev_ro, 2)} px")
    dev_wo = row_slice_devs(f260, ROWS_ORDER, warm_mask, cyc_x_at)
    ok("warm oval clusters at the model x (4 order rows)",
       dev_wo is not None and dev_wo <= 3.5,
       f"max dev {dev_wo if dev_wo is None else round(dev_wo, 2)} px")
    ordered = True
    for r_ in ROWS_ORDER:
        dy = np.float64(abs(CY - r_))
        got_r = two_clusters_x(red_mask(f260[r_ - 1:r_ + 2, 60:1020]), 60)
        got_w = two_clusters_x(warm_mask(f260[r_ - 1:r_ + 2, 60:1020]), 60)
        if got_r is None or got_w is None or not (
                got_r[0] < got_w[0] < got_w[1] < got_r[1]):
            ordered = False
    ok("ellipse OUTSIDE the oval on every order row (red<warm<warm<red)",
       ordered)

    # ---- the tight on-curve claim (near-perpendicular rows only)
    dev_rt = row_slice_devs(f260, ROWS_TIGHT, red_mask, ell_x_at)
    ok("red flanks tight on the ellipse (4 rows, |dx/dy|<=1.2)",
       dev_rt is not None and dev_rt <= 2.0,
       f"max dev {dev_rt if dev_rt is None else round(dev_rt, 2)} px")
    dev_wt = row_slice_devs(f260, ROWS_TIGHT, warm_mask, cyc_x_at)
    ok("warm flanks tight on the cycloid (4 rows)",
       dev_wt is not None and dev_wt <= 2.0,
       f"max dev {dev_wt if dev_wt is None else round(dev_wt, 2)} px")

    # ---- crest columns: red on top where the curves hug (feas:
    #      separation 0.61 px there — the warm stroke is honestly
    #      underneath, not asserted)
    dev_c = 0.0
    for c in COLS_CREST:
        yell = CY - B_ELL * math.sqrt(1 - ((c - CX) / A_ELL) ** 2)
        col = red_mask(f260[600:760, c - 2:c + 3])
        ys2 = np.where(col.any(1))[0]
        dev_c = max(dev_c, abs(ys2.mean() + 600 - yell) if len(ys2) else 99)
    ok("red crest on the model ellipse (columns, near-flat)",
       dev_c <= 3.0, f"max dev {dev_c:.2f} px")

    # ---- gap fill: the sliver exists, where the model puts it,
    #      and its pixel count matches the feas prediction
    ngap = int(gap_mask(f260[420:1400, :]).sum())
    ok("gap-fill pixel count matches feas prediction",
       0.6 * GAP_PX_PRED < ngap < 1.3 * GAP_PX_PRED,
       f"{ngap} vs pred {GAP_PX_PRED:.0f}")
    r_ = 770
    el, _ = (float(v) for v in ell_x_at(np.float64(CY - r_)))
    cl, _ = (float(v) for v in cyc_x_at(np.float64(CY - r_)))
    ok("gap ink BETWEEN the curves at the max-gap row",
       gap_mask(f260[r_ - 1:r_ + 2,
                     int(el) + 5:int(cl) - 4]).any())
    ok("no gap ink before act G (frame 180)",
       not gap_mask(f180[420:1400, :]).any())

    # ---- kiss ticks
    ok("no white before the ticks (frame 100)",
       not white_mask(f100).any())
    ok("four kiss ticks present (frame 260)",
       white_mask(f260).sum() > 400, f"{white_mask(f260).sum()} px")

    # ---- labels per act (colour-matched, trap 61)
    top = (slice(200, 410), slice(0, W))
    bot = (slice(1440, 1632), slice(0, W))
    f40top = grey_mask(f40[top]).sum()
    ok("act A label present (grey), no red label yet (frame 40)",
       f40top > 300 and not red_mask(f40[top]).any(), f"{f40top}")
    ok("act E label present in red (frame 180)",
       red_mask(f180[top]).sum() > 300)
    ok("closing labels present: grey + warm + red (frame 260)",
       grey_mask(f260[bot]).sum() > 300
       and warm_mask(f260[bot]).sum() > 300
       and red_mask(f260[bot]).sum() > 300)

    # ---- freeze byte-identity after every fade completes
    ok("freeze frames byte-identical (245 == 270, 250 == 275)",
       np.array_equal(frame_at(245), frame_at(270))
       and np.array_equal(frame_at(250), frame_at(275)))

    # ---- safe areas (trap 3)
    ok("top 192 rows pure background, all sampled frames",
       all((fr[:192] == np.asarray(bg8, np.uint8)).all()
           for fr in (f2, f40, f100, f120, f180, f260)))
    ok("rows >= 1640 pure background, all sampled frames",
       all((fr[1640:] == np.asarray(bg8, np.uint8)).all()
           for fr in (f2, f40, f100, f120, f180, f260)))

    print()
    print("NOT verified by these checks (trap 68):")
    print("  - rolling without slipping is asserted in the model")
    print("    (contact x == R*t, feas); the pixels verify pen-on-rim")
    print("    and pen-at-radius-R, not the no-slip condition itself")
    print("  - near the crests the two curves lie within one stroke")
    print("    width (0.61 px at the checked columns) and the red")
    print("    stroke covers the warm one — the on-screen merge there")
    print("    is honest geometry, not separately measured")
    print("  - the inscription (oval inside ellipse, 4 contacts) is")
    print("    proven numerically on 200k samples, not symbolically")
    print("  - area identities are proven in the model (shoelace vs")
    print("    closed forms); on pixels only via the gap-count band")
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
    ok(f"{N} frames in the file", f"{N}" in r.stdout)

    d260 = decode_frame(260)
    nred = red_mask(d260[420:1400, :]).sum()
    ok("red ellipse survives h264", nred > 3000, f"{nred} red px")
    dev_ro = row_slice_devs(d260, ROWS_ORDER, red_mask, ell_x_at)
    dev_wo = row_slice_devs(d260, ROWS_ORDER, warm_mask, cyc_x_at)
    ok("order rows hold on the SHIPPED file (red + warm)",
       dev_ro is not None and dev_wo is not None
       and max(dev_ro, dev_wo) <= 4.0,
       f"red {dev_ro if dev_ro is None else round(dev_ro, 2)}, "
       f"warm {dev_wo if dev_wo is None else round(dev_wo, 2)} px")
    dev_rt = row_slice_devs(d260, ROWS_TIGHT, red_mask, ell_x_at)
    dev_wt = row_slice_devs(d260, ROWS_TIGHT, warm_mask, cyc_x_at)
    ok("tight rows hold on the SHIPPED file (red + warm)",
       dev_rt is not None and dev_wt is not None
       and max(dev_rt, dev_wt) <= 2.5,
       f"red {dev_rt if dev_rt is None else round(dev_rt, 2)}, "
       f"warm {dev_wt if dev_wt is None else round(dev_wt, 2)} px")
    dev_c = 0.0
    for c in COLS_CREST:
        yell = CY - B_ELL * math.sqrt(1 - ((c - CX) / A_ELL) ** 2)
        col = red_mask(d260[600:760, c - 2:c + 3])
        ys2 = np.where(col.any(1))[0]
        dev_c = max(dev_c, abs(ys2.mean() + 600 - yell) if len(ys2) else 99)
    ok("red crest columns hold on the SHIPPED file (near-flat)",
       dev_c <= 3.5, f"max dev {dev_c:.2f} px")
    ngap_d = int(gap_mask(d260[420:1400, :]).sum())
    ok("gap fill survives h264 (chroma-tolerant band)",
       0.4 * GAP_PX_PRED < ngap_d < 1.4 * GAP_PX_PRED,
       f"{ngap_d} decoded vs pred {GAP_PX_PRED:.0f}")
    ok("kiss ticks survive h264", white_mask(d260).sum() > 250,
       f"{white_mask(d260).sum()} px")
    a, b = decode_frame(245), decode_frame(270)
    d1 = np.abs(a.astype(np.int64) - b.astype(np.int64)).mean()
    ok("freeze survives h264 (245 vs 270 near-identical)", d1 < 0.5,
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
    for n in (40, 77, 120, 185, 215, 260):
        Image.fromarray(frame_at(n)).save(f"{OUT_DIR}/oval_f{n:03d}.png")
    subprocess.run(
        ["ffmpeg", "-y", "-pattern_type", "glob",
         "-i", f"{OUT_DIR}/oval_f*.png",
         "-filter_complex", "scale=270:-1,tile=3x2",
         f"{OUT_DIR}/oval_sheet.png"],
        capture_output=True)
    print("sheet:", f"{OUT_DIR}/oval_sheet.png", flush=True)


if __name__ == "__main__":
    run_checks()
    encode()
    check_encode()
    review_stills()
