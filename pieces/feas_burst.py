#!/usr/bin/env python3
"""Feasibility for BURST — the exploding shell and the hidden point.

Hidden-point family #2 (after STICK). A shell rides the family's
dyadic parabola, tracing red. At m=56 it explodes into FOUR EQUAL
fragments with velocity kicks that sum to zero. Each fragment flies
its own parabola; the AVERAGE of the four positions is BITWISE the
continued original parabola, because internal forces cancel (Newton
III) and dyadic /4 is exact in float64.

Sources (verified live 2026-09-04):
  - OpenStax University Physics (LibreTexts 9.10, CC BY 4.0):
    "these internal forces cannot change the momentum of the center
    of mass of the (now exploded) shell"; "each fragment is a
    projectile on its own, thus tracing out thousands of glowing
    parabolas."
  - Wikipedia, Center of mass: internal forces "cancel in
    accordance with Newton's third law" -> the CoM moves as if only
    external forces act.
  - Wikipedia, Projectile motion: "Galileo Galilei showed that the
    trajectory of a given projectile is parabolic."

Every claim the render will make is proven here first.
"""
import sys
from fractions import Fraction

import numpy as np

# ---------------------------------------------------------------- model
W, H = 1080, 1920
FPS = 30
F = 168                      # flight frames, m = 0..168
MB = 56                      # burst frame
X0, VX = 140, 4
Y0, VYC = 1480, 21
R_SHELL, R_FRAG = 36, 18
R_DOT = 14.0
LW_RED, LW_FR, LW_DASH, LW_SPOKE = 11.0, 6.0, 6.0, 3.0

# kicks: dyadic quarters, sum exactly zero (vector)
K = [(1.75, -1.50),
     (0.50, 1.50),
     (-0.75, -1.00),
     (-1.50, 1.00)]

N = 229                      # 0..168 flight, 169..228 freeze
COL_4TR = 620                # column for the four-trail cluster check
SPOKE_MF, SPOKE_T = 160, 0.7  # spoke-sample frame and blend toward CoM

TITLE = ("the shell explodes midair. each piece takes a new arc. "
         "their average point never leaves the old one.")


def xc(m):
    return float(X0 + VX * m)


def yc(m):
    return Y0 - VYC * m + (m * m) / 8


def frag(i, m):
    t = m - MB
    return (xc(m) + K[i][0] * t, yc(m) + K[i][1] * t)


CHECKS = {"pass": 0, "fail": 0}


def ok(name, cond, detail=""):
    s = "ok  " if cond else "FAIL"
    CHECKS["pass" if cond else "fail"] += 1
    print(f"{s} {name}" + (f" [{detail}]" if detail else ""), flush=True)


# ---------------------------------------------------------------- checks
def main():
    print("== feasibility: BURST ==", flush=True)

    # 1. title
    ok("title <= 100 chars", len(TITLE) <= 100, f"{len(TITLE)}")

    # 2. y dyadic-exact vs Fraction, all states
    exact = all(
        yc(m) == float(Fraction(Y0) - Fraction(VYC) * m
                       + Fraction(m * m, 8))
        for m in range(F + 1))
    ok("y_c(m) float == Fraction-exact for all 169 states", exact)

    # 3. second difference bitwise 0.25
    ys = [yc(m) for m in range(F + 1)]
    sd = {ys[m + 1] - 2 * ys[m] + ys[m - 1] for m in range(1, F)}
    ok("second difference BITWISE {0.25} across 167 triples",
       sd == {0.25}, f"{sd}")

    # 4. time symmetry, launch/landing, apex
    ok("time-symmetric bitwise y(m) == y(168-m)",
       all(ys[m] == ys[F - m] for m in range(F + 1)))
    ok("launch == landing == 1480.0", ys[0] == 1480.0 and ys[F] == 1480.0)
    ok("apex (m=84, y=598.0)", ys[84] == 598.0 and min(ys) == 598.0)
    ok("x advances bitwise +4.0",
       {xc(m + 1) - xc(m) for m in range(F)} == {4.0})

    # 5. kicks sum to exactly zero (float)
    sx = ((K[0][0] + K[1][0]) + K[2][0]) + K[3][0]
    sy = ((K[0][1] + K[1][1]) + K[2][1]) + K[3][1]
    ok("kick sum exactly (0.0, 0.0)", sx == 0.0 and sy == 0.0,
       f"({sx}, {sy})")

    # 6. THE claim: average of fragments == continued parabola BITWISE
    bit = True
    for m in range(MB, F + 1):
        fx = [frag(i, m)[0] for i in range(4)]
        fy = [frag(i, m)[1] for i in range(4)]
        ax = (((fx[0] + fx[1]) + fx[2]) + fx[3]) / 4
        ay = (((fy[0] + fy[1]) + fy[2]) + fy[3]) / 4
        if ax != xc(m) or ay != yc(m):
            bit = False
            break
    ok("average of the 4 fragments == CoM parabola BITWISE, m 56..168",
       bit)

    # 7. fragments coincide with the shell at the burst instant
    ok("fragments start at the shell centre bitwise (m=56)",
       all(frag(i, MB) == (xc(MB), yc(MB)) for i in range(4)))

    # 8. each fragment is itself a parabola: y sd bitwise 0.25, x sd 0.0
    good = True
    for i in range(4):
        fy = [frag(i, m)[1] for m in range(MB, F + 1)]
        fx = [frag(i, m)[0] for m in range(MB, F + 1)]
        if {fy[j + 1] - 2 * fy[j] + fy[j - 1]
                for j in range(1, len(fy) - 1)} != {0.25}:
            good = False
        if {fx[j + 1] - 2 * fx[j] + fx[j - 1]
                for j in range(1, len(fx) - 1)} != {0.0}:
            good = False
    ok("each fragment: y second difference bitwise 0.25, x bitwise 0.0",
       good)

    # 9. single-valued trails (the STICK lesson, pre-applied):
    #    dx/dm = VX + kx > 0 for every fragment
    dxs = [VX + k[0] for k in K]
    ok("all fragment trails single-valued in x (dx/dm > 0)",
       min(dxs) > 0, f"dx/dm {dxs}")

    # 10. bounds: centres + disc radius + trail halfwidth inside frame,
    #     below the label band (y >= 420), above bottom UI-ish zone
    pad = R_FRAG + LW_FR / 2 + 2
    bx = [frag(i, m)[0] for i in range(4) for m in range(MB, F + 1)]
    by = [frag(i, m)[1] for i in range(4) for m in range(MB, F + 1)]
    ok("fragment x inside [pad, W-pad]",
       min(bx) - pad >= 0 and max(bx) + pad <= W,
       f"x [{min(bx):.1f}, {max(bx):.1f}]")
    ok("fragment y inside [420, 1700-pad]",
       min(by) >= 420 and max(by) + pad <= 1700,
       f"y [{min(by):.1f}, {max(by):.1f}]")

    # 11. shell area == sum of fragment areas (mass drawn as area)
    ok("shell area = 4 fragment areas exactly (36^2 == 4*18^2)",
       R_SHELL ** 2 == 4 * R_FRAG ** 2)

    # 12. fragment discs separate cleanly: pairwise >= 2r+2 from m=76
    sep_ok, worst = True, 1e9
    for m in range(76, F + 1):
        P = [frag(i, m) for i in range(4)]
        for a in range(4):
            for b in range(a + 1, 4):
                d = np.hypot(P[a][0] - P[b][0], P[a][1] - P[b][1])
                worst = min(worst, d)
                if d < 2 * R_FRAG + 2:
                    sep_ok = False
    ok("pairwise fragment separation >= 38 px for m >= 76",
       sep_ok, f"min {worst:.1f} px")

    # 13. red-dot clearance from every fragment disc for m >= 90
    clr, worst = True, 1e9
    for m in range(90, F + 1):
        for i in range(4):
            d = np.hypot(frag(i, m)[0] - xc(m), frag(i, m)[1] - yc(m))
            worst = min(worst, d)
            if d < R_FRAG + R_DOT + 2:
                clr = False
    ok("red dot clear of every fragment disc for m >= 90",
       clr, f"min {worst:.1f} px")

    # 14. end-of-flight spread is readable
    P = [frag(i, F) for i in range(4)]
    spread = max(np.hypot(P[a][0] - P[b][0], P[a][1] - P[b][1])
                 for a in range(4) for b in range(a + 1, 4))
    ok("max pairwise spread at landing >= 350 px", spread >= 350,
       f"{spread:.0f} px")

    # 15. burst ring extent: centre (x(56), y(56)), r_max = 36+12*6
    bxc, byc_ = xc(MB), yc(MB)
    rmax = 36 + 12 * 6
    ok("burst ring inside frame and below the label band",
       bxc - rmax >= 0 and bxc + rmax <= W and byc_ - rmax >= 420
       and byc_ + rmax <= 1700,
       f"centre ({bxc:.0f}, {byc_:.0f}), rmax {rmax}")

    # 16. four-trail cluster column COL_4TR: each trail crosses once;
    #     compute crossing y's, gaps, and clearance from parked discs,
    #     the red parabola and the red dot
    cross = []
    for i in range(4):
        mstar = (COL_4TR - X0 + MB * K[i][0]) / (VX + K[i][0])
        cross.append(yc(mstar) + K[i][1] * (mstar - MB))
    cross.sort()
    gaps = [cross[j + 1] - cross[j] for j in range(3)]
    ok("4 trail crossings at the check column, gaps >= 12 px",
       min(gaps) >= 12, f"y {[round(c) for c in cross]}, gaps "
       f"{[round(g, 1) for g in gaps]}")
    m_red = (COL_4TR - X0) / VX
    y_red = yc(m_red)
    dred = min(abs(c - y_red) for c in cross)
    ok("crossings clear of the red parabola (>= 15 px)", dred >= 15,
       f"min {dred:.1f} px")
    # parked discs (m=168) and dot vs the column window +-6
    clear = True
    for (px, py) in [frag(i, F) for i in range(4)] + [(xc(F), yc(F))]:
        if abs(px - COL_4TR) < R_FRAG + 6:
            clear = False
    ok("column window clear of parked discs and the dot", clear)

    # 17. spoke sample points (frame SPOKE_MF, blend SPOKE_T toward
    #     the CoM): >= 10 px from every trail polyline, >= 22 px from
    #     every current disc centre, >= 18 px from the red dot.
    #     Frame/t chosen by a scan (120/0.5 gave only 6.6 px).
    mS = SPOKE_MF
    segs = [[(xc(m), yc(m)) for m in range(0, mS + 1)]]
    for i in range(4):
        segs.append([frag(i, m) for m in range(MB, mS + 1)])
    good, worst = True, 1e9
    mids = []
    for i in range(4):
        fx, fy = frag(i, mS)
        mx = fx + SPOKE_T * (xc(mS) - fx)
        my = fy + SPOKE_T * (yc(mS) - fy)
        mids.append((round(mx), round(my)))
        for poly in segs:
            arr = np.asarray(poly)
            d = np.hypot(arr[:, 0] - mx, arr[:, 1] - my).min()
            worst = min(worst, d)
            if d < 10:
                good = False
        for j in range(4):
            if np.hypot(frag(j, mS)[0] - mx,
                        frag(j, mS)[1] - my) < R_FRAG + 4:
                good = False
        if np.hypot(xc(mS) - mx, yc(mS) - my) < R_DOT + 4:
            good = False
    ok(f"spoke samples (frame {mS}, t={SPOKE_T}) >= 10 px from every "
       "trail, clear of discs and dot",
       good, f"min trail dist {worst:.1f} px; pts {mids}")

    # 18. dash segments stay on-model and inside bounds
    dash_ok = True
    for j in range(MB, F - 2, 7):
        for mm in (j, j + 3):
            if not (0 <= xc(mm) <= W and 420 <= yc(mm) <= 1600):
                dash_ok = False
    ok("dash segments inside x bounds, y in [420, 1600]", dash_ok)

    # 19. duration
    ok("duration 229 frames = 7.63 s (<= 180 s)", N / FPS < 180,
       f"{N / FPS:.2f} s")

    # 20. fragment-1 apex above the label band with margin
    f1y = [frag(0, m)[1] for m in range(MB, F + 1)]
    ok("highest fragment point >= 420 (labels end at 400)",
       min(f1y) >= 420, f"min y {min(f1y):.1f}")

    print()
    if CHECKS["fail"]:
        print(f"{CHECKS['fail']} CHECK FAILURES")
        sys.exit(1)
    print(f"ALL {CHECKS['pass']} CHECKS PASSED "
          f"({CHECKS['pass']} feasibility)", flush=True)


if __name__ == "__main__":
    main()
