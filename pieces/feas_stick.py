#!/usr/bin/env python3
"""FEASIBILITY — STICK: the tumbling stick and the hidden point.

A rigid stick (heavy disc one end, light disc the other, mass 3:1)
is thrown spinning. Both ends trace loops. The rod's geometric
MIDDLE wobbles. The balance point (centre of mass) draws a clean
parabola — and it is not the middle of the stick.

The physics is NOT schematic this time: in a uniform gravitational
field the resultant torque about the centre of mass is zero
(Wikipedia, Center of mass: "the gravity forces will not cause the
body to rotate"), so omega is constant and the CoM moves like a
point particle — Galileo's parabola (Wikipedia, Projectile motion).
The model implements the theorem directly. Air resistance neglected,
declared.

Exactness scheme:
  x_c(m) = 120 + 5m                       (integers, exact)
  y_c(m) = 1480 - 21m + m*m/8             (dyadic rationals, exact
                                           in float64 -> second
                                           difference BITWISE 0.25)
  orientation from a 28-entry table, index (m+7) % 28
                                          (6 whole turns in 168
                                           frames; landing entry ==
                                           launch entry, same object)
  p_heavy = c + 30 e,  p_light = c - 90 e (lever: 3*30 == 1*90)
  midpoint = c - 30 e                     (wobble amplitude 30 px)
  CoM identity measured: (3 p_h + p_l)/4 vs c  (/4 is exact)

Every claim below is MEASURED here before stick.py exists.
"""
import sys
from fractions import Fraction

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
FPS = 30
F = 168                     # flight: m = 0..168 (169 states)
P = 28                      # frames per revolution
X0, VX = 120, 5
Y0, VYC = 1480, 21          # y = Y0 - VYC*m + m*m/8
MH, ML = 3, 1               # masses
RH, RL = 30.0, 90.0         # offsets from CoM (3*30 == 1*90)
R_DISC_H, R_DISC_L = 31.0, 18.0
R_DOT_RED, R_DOT_MID = 12.0, 9.0
LW_ROD = 7.0

# timeline
A_LO, A_HI = 0, 168         # act A render frames (m = n)
B_LO, B_HI = 169, 253       # act B render frames (m = 2*(n-169))
C_LO, C_HI = 254, 299       # act C freeze
N = 300

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

TITLE = ("the spinning stick's ends loop through the air. one point "
         "draws a perfect parabola — not the middle.")

CHECKS = {"pass": 0, "fail": 0}


def ok(name, cond, detail=""):
    s = "ok  " if cond else "FAIL"
    CHECKS["pass" if cond else "fail"] += 1
    print(f"{s} {name}" + (f" [{detail}]" if detail else ""), flush=True)


# ---------------------------------------------------------------- model
E_TAB = [(np.cos(2 * np.pi * j / P), np.sin(2 * np.pi * j / P))
         for j in range(P)]


def xc(m):
    return float(X0 + VX * m)


def yc(m):
    return Y0 - VYC * m + (m * m) / 8


def ee(m):
    return E_TAB[(m + 7) % P]


def ends(m):
    c = (xc(m), yc(m))
    ex, ey = ee(m)
    ph = (c[0] + RH * ex, c[1] + RH * ey)
    pl = (c[0] - RL * ex, c[1] - RL * ey)
    return c, ph, pl


def midpt(m):
    c, ph, pl = ends(m)
    return ((ph[0] + pl[0]) / 2, (ph[1] + pl[1]) / 2)


# ---------------------------------------------------------- 1 timeline
ok("N = 300 frames = 10.0 s at 30 fps", N == 300 and N / FPS == 10.0)
ok("act A covers m 0..168 one-to-one", A_HI - A_LO == F)
ms = [2 * (n - B_LO) for n in range(B_LO, B_HI + 1)]
ok("act B replay covers even m 0..168", ms[0] == 0 and ms[-1] == 168
   and all(b - a == 2 for a, b in zip(ms, ms[1:])), f"{len(ms)} frames")
ok("act C freeze length", C_HI - C_LO + 1 == 46)
ok("6 whole turns: F/P integer", F % P == 0 and F // P == 6)
ok("landing orientation is the LAUNCH table entry",
   (0 + 7) % P == (F + 7) % P, "same object, bitwise trivially")
ok("launch orientation vertical", abs(ee(0)[0]) < 1e-15 and ee(0)[1] == 1.0,
   f"e(0)=({ee(0)[0]:.1e},{ee(0)[1]})")

# ---------------------------------------------------------- 2 exact CoM
xs = [xc(m) for m in range(F + 1)]
ok("x advance bitwise 5.0 every frame",
   all(xs[m + 1] - xs[m] == 5.0 for m in range(F)))
ok("x span 120..960", xs[0] == 120.0 and xs[-1] == 960.0)

ys = [yc(m) for m in range(F + 1)]
yf = [Fraction(Y0) - VYC * m + Fraction(m * m, 8) for m in range(F + 1)]
ok("y float == exact rational, all 169 states",
   all(Fraction(ys[m]) == yf[m] for m in range(F + 1)),
   "dyadic, exact in float64")
sd = {ys[m + 1] - 2 * ys[m] + ys[m - 1] for m in range(1, F)}
ok("second difference BITWISE 0.25, all 167 triples",
   sd == {0.25}, f"set={sd}")
ok("time-symmetric BITWISE: y(m) == y(168-m)",
   all(ys[m] == ys[F - m] for m in range(F + 1)))
ok("launch == landing height == 1480.0", ys[0] == 1480.0 and ys[-1] == 1480.0)
ok("apex at m=84, y exactly 598.0", ys[84] == 598.0
   and min(ys) == ys[84], f"rise {ys[0] - ys[84]:.0f} px")

# ---------------------------------------------------------- 3 the lever
ok("lever law: MH*RH == ML*RL", MH * RH == ML * RL, f"{MH}*{RH} == {ML}*{RL}")
ok("disc AREAS ~ mass ratio 3", abs(R_DISC_H**2 / R_DISC_L**2 - 3) < 0.05,
   f"{R_DISC_H**2 / R_DISC_L**2:.3f}")

dev = 0.0
for m in range(F + 1):
    c, ph, pl = ends(m)
    gx = (MH * ph[0] + ML * pl[0]) / 4
    gy = (MH * ph[1] + ML * pl[1]) / 4
    dev = max(dev, abs(gx - c[0]), abs(gy - c[1]))
ok("weighted average (3 p_h + p_l)/4 returns the CoM",
   dev < 1e-9, f"max |dev| {dev:.2e} px over 169 states")

wob = [abs(np.hypot(midpt(m)[0] - xc(m), midpt(m)[1] - yc(m)) - RH)
       for m in range(F + 1)]
ok("midpoint sits exactly 30 px from the CoM, every frame",
   max(wob) < 1e-9, f"max |r-30| {max(wob):.2e}")
mdy = [midpt(m)[1] - yc(m) for m in range(F + 1)]
ok("midpoint wobble reaches the full +-30 px vertically",
   min(mdy) == -30.0 and max(mdy) == 30.0,
   f"[{min(mdy):.1f}, {max(mdy):.1f}]")

# ---------------------------------------------------------- 4 the loops
def n_self_int(pts):
    seg = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]

    def cr(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    cnt = 0
    for i in range(len(seg)):
        for j in range(i + 2, len(seg)):
            p, q = seg[i]
            r, s = seg[j]
            d1, d2 = cr(p, q, r), cr(p, q, s)
            d3, d4 = cr(r, s, p), cr(r, s, q)
            if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
                cnt += 1
    return cnt


path_h = [ends(m)[1] for m in range(F + 1)]
path_l = [ends(m)[2] for m in range(F + 1)]
path_c = [(xc(m), yc(m)) for m in range(F + 1)]
nl, nh, nc = n_self_int(path_l), n_self_int(path_h), n_self_int(path_c)
ok("light end path has loops (>=3 self-intersections)", nl >= 3, f"{nl}")
ok("heavy end path loops too (>=1)", nh >= 1, f"{nh}")
ok("the CoM path NEVER crosses itself", nc == 0, f"{nc}")

# relative tip speeds vs CoM speed (why the loops happen where they do)
om = 2 * np.pi / P
ok("light tip rel speed beats CoM speed at apex",
   RL * om > VX, f"{RL * om:.1f} vs {VX} px/f")
ok("heavy tip rel speed beats CoM speed at apex",
   RH * om > VX, f"{RH * om:.2f} vs {VX} px/f")

# ---------------------------------------------------------- 5 framing
lo_x = hi_x = 540.0
lo_y = hi_y = 960.0
for m in range(F + 1):
    c, ph, pl = ends(m)
    for (px, py), r in ((ph, R_DISC_H), (pl, R_DISC_L),
                        (c, R_DOT_RED)):
        lo_x, hi_x = min(lo_x, px - r), max(hi_x, px + r)
        lo_y, hi_y = min(lo_y, py - r), max(hi_y, py + r)
ok("all geometry inside frame with >= 8 px margin",
   lo_x >= 8 and hi_x <= W - 8 and lo_y >= 8 and hi_y <= H - 8,
   f"x [{lo_x:.0f},{hi_x:.0f}] y [{lo_y:.0f},{hi_y:.0f}]")
ok("nothing in the bottom UI zone (y > 1632)", hi_y <= 1632.0,
   f"max y {hi_y:.0f}")
ok("apex clears the top text band (y < 430 stays empty of stick)",
   lo_y >= 430, f"min y {lo_y:.0f}")

# resting pose (act C): stick vertical at x=960
cF, phF, plF = ends(F)
ok("act C resting stick is vertical at x=960",
   abs(phF[0] - 960) < 1e-12 and abs(plF[0] - 960) < 1e-12
   and abs(phF[1] - 1510) < 1e-12 and abs(plF[1] - 1390) < 1e-12,
   f"heavy y {phF[1]:.0f}, light y {plF[1]:.0f}")

# ---------------------------------------------------------- 6 trails
# chord fidelity: drawn trail is per-frame chords; measure sagitta
# against the half-frame true position
def pos_at(t, which):
    x = X0 + VX * t
    y = Y0 - VYC * t + t * t / 8
    th = 2 * np.pi * (t + 7) / P
    ex, ey = np.cos(th), np.sin(th)
    r = {"h": RH, "l": -RL, "m": -RH}[which]
    return (x + r * ex, y + r * ey)


sag = 0.0
for which in ("h", "l", "m"):
    for m in range(F):
        a, b = pos_at(m, which), pos_at(m + 1, which)
        tmid = pos_at(m + 0.5, which)
        sag = max(sag, np.hypot(tmid[0] - (a[0] + b[0]) / 2,
                                tmid[1] - (a[1] + b[1]) / 2))
ok("per-frame chords track the true path (sagitta < 1.5 px)",
   sag < 1.5, f"max {sag:.2f} px vs trail half-width 2.5")

# act B appends model segments (m-2,m-1) and (m-1,m): stitches all
mm = set()
for n in range(B_LO, B_HI + 1):
    m = 2 * (n - B_LO)
    if m >= 2:
        mm.add((m - 2, m - 1))
        mm.add((m - 1, m))
    elif m == 0:
        pass
ok("act B trail segments stitch m 0..168 gap-free",
   mm == {(k, k + 1) for k in range(F)}, f"{len(mm)} segments")

# ---------------------------------------------------------- 7 labels
def text_w(s, px):
    f = ImageFont.truetype(FONT, px * 4)
    im = Image.new("L", (px * len(s) * 5, px * 8), 0)
    ImageDraw.Draw(im).text((8, 8), s, font=f, fill=255)
    a = np.asarray(im)
    xs_ = np.where(a.max(0) > 0)[0]
    return (xs_.max() - xs_.min() + 1) / 4


LBL_A = "both ends of the stick, traced"
LBL_B1 = "the middle of the stick"
LBL_B2 = "its balance point"
LBL_C = "three times closer to the heavy end"
for s, px in ((LBL_A, 34), (LBL_B1, 34), (LBL_B2, 34), (LBL_C, 34)):
    w = text_w(s, px)
    ok(f"label fits ({s[:24]}...)", w <= 900, f"{w:.0f} px at {px}")
ok("label band inside safe area (y 210..330; top UI ends 192)",
   210 > 192 and 330 + 44 < 1632)

# ---------------------------------------------------------- 8 misc
ok("title <= 100 chars", len(TITLE) <= 100, f"{len(TITLE)}")
ok("hook: stick has moved and turned by frame 8",
   xc(8) - xc(0) == 40.0 and (8 + 7) % P != 7,
   f"40 px, {8 * 360 / P:.0f} deg")

print()
if CHECKS["fail"]:
    print(f"{CHECKS['fail']} FEASIBILITY FAILURES")
    sys.exit(1)
print(f"ALL {CHECKS['pass']} FEASIBILITY CHECKS PASSED", flush=True)
