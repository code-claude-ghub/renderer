#!/usr/bin/env python3
"""ELLIPSE — two wheels roll opposite ways; a point halfway between
their pens draws an ellipse that never moves.

@Dominic-qv3yt, on the cycloid piece: "Two cycloids make an ellipse."
He is right, and this is the proof by construction. Wheel A rolls
RIGHT along an upper rail with a pen at radius 40 (inside the disc);
wheel B rolls LEFT along a lower rail with a pen at radius 200 (a
spoke past the rim). Opposite rolling means opposite spin. For the
red point midway between the pens:

  - the translations cancel: hub midpoint == 540.0 BITWISE, all frames
  - the counter-rotations superpose into
        M = (540 + 80 sin phi, 960 + 120 cos phi)
    an ellipse with semi-axes (RB-RA)/2 and (RA+RB)/2 that stays
    centred while both wheels drive away  (1:1 Lissajous figure)
  - at the crossing frame hubs, pens and dot are COLLINEAR on x=540
    exactly (sin(0.0) == 0.0)
  - equal pens would flatten it to a straight segment (a_x == 0.0
    exactly) — the Tusi couple, two-wheel edition; a pen at the hub
    would round it to a circle

kinematics only: the wheels are driven at constant speed and roll
without slip (contact-point velocity 0.0 exactly, by construction).
no dynamics is claimed.
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
R = 105.0                       # wheel radius
RA, RB = 40.0, 200.0            # pen radii: curtate / prolate
YA, YB = 560.0, 1360.0          # hub heights; rails at YA+R, YB+R
XC = 540.0
STEP = 5.0                      # px of hub travel per motion sample
K = 68                          # half the samples; s in [-340, 340]
M_N = 2 * K + 1                 # 137
# travel is +-340, not more: the prolate pen reaches RB=200 px past
# its hub, and at +-420 it left the frame. 340 keeps every pen and
# both wheels on-screen for the whole film and still covers 6.476 rad
# = 103% of a lap — the ellipse closes, then retraces its own line.

EAX = (RB - RA) / 2.0           # 80.0   ellipse semi-axis, x
EAY = (RB + RA) / 2.0           # 120.0  ellipse semi-axis, y
EYC = (YA + YB) / 2.0           # 960.0  ellipse centre row

LW_WHEEL = 4.0
LW_SPOKE = 3.0
LW_RAIL = 3.0
LW_TRAIL = 3.0
LW_DOT_TRAIL = 3.5
LW_LINK = 2.0
R_PEN = 6.0
R_HUB = 4.0
R_DOT = 7.0

PRE = 36
SEG1 = 68
FRZ = 32                        # freeze on the crossing: all collinear
SEG2 = 68
POST = 45
N_FRAMES = PRE + SEG1 + FRZ + SEG2 + POST          # 249 = 8.3 s
I_GO = PRE                                          # f36
I_FRZ = PRE + SEG1                                  # f104, s = 0
I_R2 = I_FRZ + FRZ                                  # f136
I_END = I_R2 + SEG2                                 # f204

OUT = f"out/ellipse_{time.strftime('%H%M%S')}.mp4"

# ------------------------------------------------------- exact tables
# sign applied explicitly so odd symmetry is BITWISE, not libm's favour
MS = np.arange(M_N)
S = STEP * (MS - K)                     # exact integer-valued floats
PHI_ABS = np.abs(S) / R
SN = np.sign(S) * np.sin(PHI_ABS)       # odd, bitwise
CS = np.cos(PHI_ABS)                    # even, bitwise

HUB_AX = XC + S
HUB_BX = XC - S
PEN_AX = HUB_AX - RA * SN
PEN_AY = YA + RA * CS
PEN_BX = HUB_BX + RB * SN
PEN_BY = YB + RB * CS
DOT_X = (PEN_AX + PEN_BX) / 2.0
DOT_Y = (PEN_AY + PEN_BY) / 2.0


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


def draw_freeze_line(img, fade):
    """The crossing frame's fact: everything on x = 540. Dashed, red."""
    y = 490.0
    while y < 1560.0:
        x0, y0, cv = polyseg_cov([(XC, y), (XC, min(y + 18.0, 1560.0))],
                                 3.0)
        comp_bbox(img, x0, y0, cv * fade, C_RED)
        y += 32.0


def draw_wheel(img, hx, hy, sn, cs, pen_r, spoke_len):
    x0, y0, cv = circ_cov(hx, hy, R, LW_WHEEL)
    comp_bbox(img, x0, y0, cv, INK)
    dx, dy = -sn if spoke_len is None else sn, cs
    # wheel A's spoke runs hub->rim through its pen; wheel B's runs
    # past the rim to the prolate pen. direction passed via sn's sign.
    ex, ey = hx + dx * (R if spoke_len is None else spoke_len), \
        hy + dy * (R if spoke_len is None else spoke_len)
    draw_poly(img, [(hx, hy), (ex, ey)], INK, LW_SPOKE)
    x0, y0, cv = disc_cov(hx, hy, R_HUB)
    comp_bbox(img, x0, y0, cv, INK)


BG = np.full((H, W, 3), PAPER, np.float64)


# ---------------------------------------------------------------- frames
def frame_at(i):
    img = BG.copy()
    m = m_at(i)
    # freeze furniture first, under everything that moves
    if I_FRZ <= i < I_R2:
        fade = float(np.clip((i - I_FRZ) / 8.0, 0.0, 1.0))
        if fade > 0:
            draw_freeze_line(img, fade)
    # trails: the two pens' cycloids (ghost) and the dot's path (red)
    if m >= 1:
        draw_poly(img, list(zip(PEN_AX[:m + 1], PEN_AY[:m + 1])),
                  GHOST, LW_TRAIL)
        draw_poly(img, list(zip(PEN_BX[:m + 1], PEN_BY[:m + 1])),
                  GHOST, LW_TRAIL)
    # the linkage: pen-to-pen line, midpoint marked
    draw_poly(img, [(PEN_AX[m], PEN_AY[m]), (PEN_BX[m], PEN_BY[m])],
              GHOST, LW_LINK)
    if m >= 1:
        draw_poly(img, list(zip(DOT_X[:m + 1], DOT_Y[:m + 1])),
                  C_RED, LW_DOT_TRAIL)
    # rails drawn OVER the trails so they stay solid (fence audit)
    draw_poly(img, [(0.0, YA + R), (float(W), YA + R)], INK, LW_RAIL)
    draw_poly(img, [(0.0, YB + R), (float(W), YB + R)], INK, LW_RAIL)
    # wheels, spokes, hubs
    draw_wheel(img, HUB_AX[m], YA, SN[m], CS[m], None, None)
    x0, y0, cv = circ_cov(HUB_BX[m], YB, R, LW_WHEEL)
    comp_bbox(img, x0, y0, cv, INK)
    draw_poly(img, [(HUB_BX[m], YB),
                    (HUB_BX[m] + RB * SN[m], YB + RB * CS[m])],
              INK, LW_SPOKE)
    x0, y0, cv = disc_cov(HUB_BX[m], YB, R_HUB)
    comp_bbox(img, x0, y0, cv, INK)
    # pens
    x0, y0, cv = disc_cov(PEN_AX[m], PEN_AY[m], R_PEN)
    comp_bbox(img, x0, y0, cv, INK)
    x0, y0, cv = disc_cov(PEN_BX[m], PEN_BY[m], R_PEN)
    comp_bbox(img, x0, y0, cv, INK)
    # the midpoint, drawn last
    x0, y0, cv = disc_cov(DOT_X[m], DOT_Y[m], R_DOT)
    comp_bbox(img, x0, y0, cv, C_RED)
    if i < I_GO:
        draw_chevron(img, 200.0, 400.0, True, INK)      # above wheel A
        draw_chevron(img, 880.0, 1210.0, False, INK)    # above wheel B
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def render_frames():
    for i in range(N_FRAMES):
        yield frame_at(i)                            # stream (trap 34)


# ---------------------------------------------------------------- measure
# FENCE AUDIT (written before the first render):
#   RED lives in exactly three places — the dot disc (moving, r=7,
#     box +-12 around the model dot), the dot trail (on the analytic
#     ellipse by construction, band |rho-1| <= 0.10 in normalised
#     radius = >= 8 px of slack), and the freeze line (cols 533..547,
#     rows 486..1564, FREEZE FRAMES ONLY, fade starts at 0 so frame
#     I_FRZ itself is line-free). red fence = union of the three.
#   dot centroid checks: the trail tail always ends AT the dot, so on
#     motion frames the centroid is pulled along the path — tolerance
#     2.5 px there, 0.8 px at f0 where no trail exists yet.
#   ink lives in: wheels (moving), spokes, hubs, pens, rails (static
#     rows 663..667 and 1463..1467, drawn OVER the trails so the
#     solidity check needs no crossing exclusions), chevrons (PRE-HOLD
#     ONLY, boxes x 166..234 y 382..418 and x 846..914 y 1192..1228 —
#     both sit ABOVE their wheels' topmost rows (453 and 1253) so no
#     wheel can ever enter either box at any m; the pen-to-pen link
#     stays below y=520 at f0, clear of both).
#   ghost lives in: pen trails and the pen-to-pen link. ghost checks
#     sample model positions strictly behind the current m.
#   ellipse zone rows 840..1080 are clear of both wheels' reach
#     (wheel A bottom 665+2; pen B never above row 1160).
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
RHO = np.hypot((XX - XC) / EAX, (YY - EYC) / EAY)
ELL_BAND = np.abs(RHO - 1.0) <= 0.10


def allowed_red(m, frozen):
    allow = ELL_BAND.copy()
    x0 = int(round(DOT_X[m]))
    y0 = int(round(DOT_Y[m]))
    allow[max(y0 - 12, 0):y0 + 13, max(x0 - 12, 0):x0 + 13] = True
    if frozen:
        allow[486:1565, 533:548] = True
    return allow


# ---------------------------------------------------------------- checks
def run_checks():
    n = [0]

    def ok(name, cond, note=""):
        print(("  ok  " if cond else "  FAIL") + f"  {name}" +
              (f"  [{note}]" if note else ""))
        assert cond, name
        n[0] += 1

    print("== ELLIPSE render checks ==")

    # ---- model
    ok("s exact integers", bool(np.all(S == np.round(S))))
    ok(f"hub midpoint 540.0 bitwise x{M_N}",
       bool(np.all((HUB_AX + HUB_BX) / 2.0 == 540.0)))
    dev = max(np.abs(DOT_X - (XC + EAX * SN)).max(),
              np.abs(DOT_Y - (EYC + EAY * CS)).max())
    ok("dot == analytic ellipse", dev < 1e-9, f"max dev {dev:.2e} px")
    ok("odd/even tables bitwise",
       bool(np.all(SN == -SN[::-1]) and np.all(CS == CS[::-1])))
    ok("dot_y time-mirror bitwise", bool(np.all(DOT_Y == DOT_Y[::-1])))
    col = (HUB_AX[K], PEN_AX[K], DOT_X[K], PEN_BX[K], HUB_BX[K])
    ok("crossing collinear at 540.0 exactly",
       all(c == 540.0 for c in col))
    cover = PHI_ABS[0] + PHI_ABS[-1]
    ok("coverage > 2*pi", cover > 2 * np.pi + 0.15,
       f"{cover:.3f} rad = {cover/(2*np.pi)*100:.0f}% of a lap")
    v = 1.0
    ok("contact velocity exactly 0.0",
       v - R * np.cos(0.0) * (v / R) == 0.0 and
       -R * np.sin(0.0) * (v / R) == 0.0)

    # ---- determinism / statics
    ok("frame purity", bool(np.array_equal(frame_at(60), frame_at(60))))
    ok("freeze static after fade",
       bool(np.array_equal(frame_at(I_FRZ + 20), frame_at(I_FRZ + 30))))
    ok("post static",
       bool(np.array_equal(frame_at(I_END + 5), frame_at(N_FRAMES - 1))))

    # ---- pixels, f0 (no trail yet)
    f0 = frame_at(0)
    for name, hx, hy in (("A", HUB_AX[0], YA), ("B", HUB_BX[0], YB)):
        pts_ok = 0
        for th in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            px = int(round(hx + R * np.cos(th)))
            py = int(round(hy + R * np.sin(th)))
            box = f0[py - 3:py + 4, px - 3:px + 4]
            pts_ok += bool(ink_mask(box).sum() > 0)
        ok(f"wheel {name} outline present at f0", pts_ok == 12,
           f"{pts_ok}/12 circle points inked")
    cx, cy = red_centroid_at(f0, DOT_X[0], DOT_Y[0])
    d0 = np.hypot(cx - DOT_X[0], cy - DOT_Y[0])
    ok("dot centroid at f0", d0 < 0.8, f"off by {d0:.2f} px "
       f"(model {DOT_X[0]:.1f},{DOT_Y[0]:.1f})")
    for name, px, py in (("pen A", PEN_AX[0], PEN_AY[0]),
                         ("pen B", PEN_BX[0], PEN_BY[0])):
        x0, y0 = int(round(px)) - 8, int(round(py)) - 8
        cnt = int(ink_mask(f0[y0:y0 + 17, x0:x0 + 17]).sum())
        ok(f"{name} disc present at f0", cnt > 60, f"{cnt} ink px")

    # ---- safe area (trap 3): pure paper above 10% and below 85%
    for i in (0, I_FRZ, N_FRAMES - 1):
        fr = frame_at(i)
        # paper encodes as int(0.845*255+0.5) TRUNCATED by astype = 215,
        # not round()'s 216 — compute it the way frame_at does
        paper8 = np.uint8(PAPER * 255.0 + 0.5)
        top = int((fr[:192] != paper8).sum())
        bot = int((fr[1632:] != paper8).sum())
        ok(f"safe area clean at f{i}", top == 0 and bot == 0,
           f"{top}+{bot} non-paper px")

    # ---- motion: dot follows the model (trail tail pulls <= 2.5 px)
    for i in (I_GO + 30, I_R2 + 25):
        m = m_at(i)
        fr = frame_at(i)
        cx, cy = red_centroid_at(fr, DOT_X[m], DOT_Y[m], halfw=11)
        d = np.hypot(cx - DOT_X[m], cy - DOT_Y[m])
        ok(f"dot tracks model at f{i}", d < 2.5, f"off {d:.2f} px")

    # ---- red fence: no red anywhere it has no business being
    for i in (0, 60, I_FRZ, I_FRZ + 16, 170, N_FRAMES - 1):
        m = m_at(i)
        frozen = I_FRZ < i < I_R2       # fade is 0 AT I_FRZ itself
        fr = frame_at(i)
        stray = red_strict(fr.astype(np.float64)) & \
            ~allowed_red(m, frozen)
        ok(f"red fence at f{i}", int(stray.sum()) == 0,
           f"{int(stray.sum())} stray red px")

    # ---- freeze: the collinear line, faded in, on x = 540
    ffz = frame_at(I_FRZ + 16)
    line = red_strict(ffz[486:1565, 533:548].astype(np.float64))
    ok("freeze line present", int(line.sum()) > 300,
       f"{int(line.sum())} red px")
    _, lx = np.nonzero(line)
    ok("freeze line on x=540", abs((lx.mean() + 533) - 540.0) < 0.7,
       f"mean col {lx.mean() + 533:.2f}")
    hub_box = ink_mask(ffz[int(YA) - 5:int(YA) + 6, 535:546])
    ok("hub A on the line at the freeze", int(hub_box.sum()) > 20)

    # ---- trail growth across motion frames
    cnts = [int(red_strict(frame_at(i).astype(np.float64)).sum())
            for i in (I_GO + 10, I_FRZ - 1, I_R2 + 30, I_END - 1)]
    ok("dot trail grows", cnts[0] < cnts[1] < cnts[2] < cnts[3],
       "->".join(str(c) for c in cnts))

    # ---- closure: the final frame holds the WHOLE ellipse in red
    fend = frame_at(N_FRAMES - 1)
    misses = []
    for th in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        ex = int(round(XC + EAX * np.sin(th)))
        ey = int(round(EYC + EAY * np.cos(th)))
        box = fend[ey - 5:ey + 6, ex - 5:ex + 6]
        if int(red_strict(box.astype(np.float64)).sum()) == 0:
            misses.append(round(th, 2))
    ok("ellipse closed on final frame", not misses,
       f"12/12 sampled angles red" if not misses else f"missing {misses}")

    # ---- ghost furniture
    m = m_at(170)
    j = m - 30
    gx, gy = int(round(PEN_BX[j])), int(round(PEN_BY[j]))
    cnt = int(ghost_mask(fend[gy - 5:gy + 6, gx - 5:gx + 6]).sum())
    ok("pen B trail present", cnt > 0, f"{cnt} ghost px at sample {j}")
    mm = 2 * K
    lx = (PEN_AX[mm] + DOT_X[mm]) / 2.0
    ly = (PEN_AY[mm] + DOT_Y[mm]) / 2.0
    cnt = int(ghost_mask(fend[int(ly) - 5:int(ly) + 6,
                              int(lx) - 5:int(lx) + 6]).sum())
    ok("pen-to-pen link present", cnt > 0, f"{cnt} ghost px")

    # ---- rails stay solid (drawn over the trails on purpose)
    for row, name in ((int(YA + R), "A"), (int(YB + R), "B")):
        band = fend[row - 3:row + 4, :, :]
        im = ink_mask(band)
        cols_ok = all(im[:, c - 1:c + 2].sum() > 0
                      for c in range(30, 1051, 10))
        ok(f"rail {name} solid", cols_ok)

    # ---- chevrons: pre-hold only
    f5 = frame_at(5)
    fgo = frame_at(I_GO + 5)
    c_pre = int(ink_mask(f5[382:419, 166:235]).sum())
    c_post = int(ink_mask(fgo[382:419, 166:235]).sum())
    ok("chevron A pre-hold only", c_pre > 100 and c_post == 0,
       f"{c_pre} px pre, {c_post} after")

    print(f"ALL {n[0]} CHECKS PASSED")
    print("NOT verified (stated, not proven):")
    print("  - the wheels are DRIVEN at constant speed; no dynamics,")
    print("    no gravity, no friction model — kinematic rolling only")
    print("  - rolling without slip is imposed by construction, not")
    print("    simulated; its exactness is the parametrisation's")
    print("  - the co-rotating control (midpoint = circle) was checked")
    print("    in feasibility, not rendered")
    print("  - prolate pen B dips below its rail line, as a train")
    print("    wheel's flange does; the rail is drawn as a line, not")
    print("    a solid floor, so nothing visually collides")


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
    misses = 0
    for th in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        ex = int(round(XC + EAX * np.sin(th)))
        ey = int(round(EYC + EAY * np.cos(th)))
        box = d[ey - 6:ey + 7, ex - 6:ex + 7]
        misses += int(red_strict(box.astype(np.float64)).sum()) == 0
    assert misses == 0, f"{misses} ellipse angles lost"
    print("    ellipse survives: red at 12/12 sampled angles, final frame")
    d0 = decode_frame(0)
    cx, cy = red_centroid_at(d0, DOT_X[0], DOT_Y[0])
    dd = np.hypot(cx - DOT_X[0], cy - DOT_Y[0])
    assert dd < 2.0, dd
    print(f"    dot survives: f0 centroid off by {dd:.2f} px")
    dfz = decode_frame(I_FRZ + 16)
    line = red_strict(dfz[486:1565, 533:548].astype(np.float64))
    assert int(line.sum()) > 250, int(line.sum())
    print(f"    freeze line survives: {int(line.sum())} red px on x=540")
    diff = np.abs(decode_frame(0).astype(int) - frame_at(0).astype(int))
    assert diff.mean() < 2.0
    print(f"    decoded f0 vs render: mean |diff| {diff.mean():.3f}, "
          f"max {diff.max()}")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams",
         "v", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
         OUT], capture_output=True, text=True).stdout.strip()
    assert int(probe) == N_FRAMES, f"frames {probe} != {N_FRAMES}"
    print(f"    {probe} frames; the ellipse survives the encode")


def review_stills():
    base = OUT[:-4]
    picks = [("start", 10), ("rolling", 76), ("freeze", 120),
             ("closing", 168), ("closed", 198), ("final", N_FRAMES - 1)]
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
