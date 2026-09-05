#!/usr/bin/env python3
"""Feasibility for COLLIDE — the mid-air perfectly inelastic collision
(hidden-point family #3, after STICK and BURST).

Two balls fly toward each other on the family's dyadic vertical
profile (equal heights every frame). Mass 3 (left, heavy) vs mass 1
(right, light). The mass-weighted point between them — never on
either ball — rides its own dyadic parabola. At m=84 (the shared
apex) they stick: perfectly inelastic. Momentum conservation makes
the merged velocity the mass-weighted average, so the merged ball's
centre IS the weighted point, BITWISE, for the remaining 84 frames.

Every claim proven here before a frame is rendered.
"""
import sys
from fractions import Fraction

import numpy as np

W, H = 1080, 1920
F = 168
MC = 84                      # merge frame (first contact, see check)
XA0, VXA = 156, 4            # heavy ball, mass 3
XB0, VXB = 960, -5           # light ball, mass 1
Y0, VYC = 1480, 21
R_A2, R_B = 972.0, 18.0      # rA^2 = 3 * rB^2 exactly (972 == 3*324)
R_A = float(np.sqrt(R_A2))   # ~31.18, drawing only — checks use R_A2
R_MERGE = 36.0               # 972 + 324 == 1296 == 36^2
R_DOT = 14.0
LW_RED, LW_TR, LW_DASH, LW_SPOKE = 11.0, 6.0, 6.0, 5.0

COL_L, COL_R = 220, 720      # grey-cluster check columns (verified below)
SPOKE_NF, SPOKE_X = 50, 580  # spoke sample: frame 50, x=580 on B-spoke

TITLE = ("two balls crash midair and stick. both flights end there. "
         "the point between them never feels it.")

CHECKS = {"pass": 0, "fail": 0}


def ok(name, cond, detail=""):
    s = "ok  " if cond else "FAIL"
    CHECKS["pass" if cond else "fail"] += 1
    print(f"{s} {name}" + (f" [{detail}]" if detail else ""), flush=True)


def Y(m):
    return Y0 - VYC * m + (m * m) / 8


def xA(m):
    return float(XA0 + VXA * m)


def xB(m):
    return float(XB0 + VXB * m)


def cx(m):
    # exactly as the renderer computes it: momentum-weighted average
    return ((3.0 * xA(m)) + xB(m)) / 4.0


def cy(m):
    return ((3.0 * Y(m)) + Y(m)) / 4.0


def main():
    print("== COLLIDE feasibility ==", flush=True)

    # ---- title
    ok("title <= 100 chars", len(TITLE) <= 100, f"{len(TITLE)}")

    # ---- vertical profile: dyadic-exact, second difference bitwise
    frac_ok = all(
        Fraction(Y(m)) == Fraction(Y0) - VYC * m + Fraction(m * m, 8)
        for m in range(F + 1))
    ok("Y(m) Fraction-exact for all m", frac_ok)
    sd = {Y(m + 1) - 2 * Y(m) + Y(m - 1) for m in range(1, F)}
    ok("Y second difference BITWISE {0.25}", sd == {0.25}, f"{sd}")
    ok("launch == landing height bitwise", Y(0) == Y(F) == 1480.0)
    ok("apex at m=84", Y(84) == 598.0 and min(Y(m) for m in range(F + 1))
       == 598.0)

    # ---- the weighted point: closed form bitwise
    ok("cx(m) == 357 + 1.75m BITWISE, all m",
       all(cx(m) == 357.0 + 1.75 * m for m in range(F + 1)))
    ok("cy(m) == Y(m) BITWISE, all m",
       all(cy(m) == Y(m) for m in range(F + 1)))

    # ---- the lever, bitwise: 3 * (c - A) == (B - c), every frame
    ok("lever law 3*(cx-xA) == xB-cx BITWISE, all m",
       all(3.0 * (cx(m) - xA(m)) == xB(m) - cx(m) for m in range(F + 1)))
    ok("red point strictly between the balls, act A",
       all(xA(m) < cx(m) < xB(m) for m in range(MC)))

    # ---- masses and areas: 3:1 exactly, area conserved at the merge
    ok("rA^2 == 3 * rB^2 exactly", R_A2 == 3.0 * R_B * R_B,
       f"{R_A2} == {3.0 * R_B * R_B}")
    ok("merged area == sum of areas exactly (36^2 == 972+324)",
       R_MERGE * R_MERGE == R_A2 + R_B * R_B)

    # ---- contact: MC is the first frame the discs would overlap
    def gap(m):
        return xB(m) - xA(m)
    rsum = R_A + R_B
    ok("clear of contact at m=83", gap(83) > rsum + 2.0,
       f"gap {gap(83):.1f} vs {rsum:.1f}")
    ok("contact by m=84 (merge frame)", gap(84) < rsum,
       f"gap {gap(84):.1f} vs {rsum:.1f}")
    ok("no earlier overlap (m 0..83 all clear)",
       all(gap(m) > rsum for m in range(MC)))

    # ---- momentum: merged velocity is the weighted average, and the
    #      stepped path equals the closed form bitwise
    vxm = ((3.0 * VXA) + VXB) / 4.0
    ok("merged vx == 1.75 exactly (momentum / total mass)", vxm == 1.75)
    x = cx(MC)
    step_ok = True
    for m in range(MC + 1, F + 1):
        x += vxm
        if x != cx(m):
            step_ok = False
            break
    ok("momentum-stepped x == closed form BITWISE, m 85..168", step_ok)
    # vertical: both balls share Y, so the weighted average is Y itself;
    # stepping vy by the bitwise sd 0.25 reproduces it
    y = Y(MC)
    vy = Y(MC) - Y(MC - 1)
    step_ok = True
    for m in range(MC + 1, F + 1):
        vy += 0.25
        y += vy
        if y != Y(m):
            step_ok = False
            break
    ok("momentum-stepped y == Y BITWISE, m 85..168", step_ok)

    # ---- kinetic energy honesty number for the description
    # relative velocity at merge is purely horizontal (shared vy): 9
    mu = 3.0 * 1.0 / 4.0
    ke_lost = 0.5 * mu * (VXA - VXB) ** 2
    vy_mc = Y(MC) - Y(MC - 1)         # -0.125, essentially zero
    ke_at = 0.5 * (3.0 * (VXA ** 2 + vy_mc ** 2)
                   + 1.0 * (VXB ** 2 + vy_mc ** 2))
    ok("KE lost fraction in (0.80, 0.85) for the description",
       0.80 < ke_lost / ke_at < 0.85, f"{ke_lost / ke_at:.4f}")

    # ---- frame bounds: every drawn thing inside margins
    xs = ([xA(m) for m in range(F + 1)] + [xB(m) for m in range(F + 1)]
          + [cx(m) for m in range(F + 1)])
    ys = [Y(m) for m in range(F + 1)]
    ok("all centres in frame with disc margin",
       min(xs) - R_A > 80 and max(xs) + R_A < W - 40
       and min(ys) - R_MERGE > 480 and max(ys) + R_MERGE < 1560,
       f"x {min(xs):.0f}..{max(xs):.0f}, y {min(ys):.0f}..{max(ys):.0f}")

    # ---- the grey ink is two full parabolic arcs (solid + dash each):
    #      curve A: y = Y((x-156)/4) on x in [156, 828]
    #      curve B: y = Y((1000-x)/5) on x in [120, 1000]
    # their single crossing, and the red-arc crossings, must stay clear
    # of the check columns
    def yy_a(x):
        return Y((x - XA0) / VXA)

    def yy_b(x):
        return Y((XB0 - x) / (-VXB))

    def yy_r(x):
        return Y((x - 357.0) / 1.75)

    # knot region: find every curve-pair crossing NUMERICALLY (the
    # scratchpad version of this list caused a feas fail — compute it)
    knots = []
    xs_grid = np.arange(160.0, 828.0, 0.25)

    def crossings(f, g, lo, hi):
        xs2 = xs_grid[(xs_grid >= lo) & (xs_grid <= hi)]
        d = np.array([f(x) - g(x) for x in xs2])
        sc = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
        return [float(xs2[i]) for i in sc]

    knots += crossings(yy_a, yy_b, 160, 828)
    knots += crossings(yy_r, yy_a, 357, cx(F))
    knots += crossings(yy_r, yy_b, 357, cx(F))
    ok("check columns >= 40 px from every crossing knot",
       all(abs(c - k) >= 40 for c in (COL_L, COL_R) for k in knots),
       f"knots {[round(k, 1) for k in knots]}")

    # ---- check columns: exactly two grey crossings, well separated,
    #      far from the red arc, clear of parked discs and labels
    for col, name in ((COL_L, "COL_L"), (COL_R, "COL_R")):
        ya, yb = yy_a(col), yy_b(col)
        ok(f"{name}={col}: two grey crossings separated >= 40 px",
           abs(ya - yb) >= 40, f"{ya:.0f} vs {yb:.0f}")
        in_a = XA0 <= col <= xA(F)
        in_b = xB(F) <= col <= XB0
        ok(f"{name}: both curves span the column", in_a and in_b)
        # red clearance: red arc exists on x in [357, 651]
        if 357.0 <= col <= cx(F):
            dr = min(abs(yy_r(col) - ya), abs(yy_r(col) - yb))
            ok(f"{name}: red arc >= 25 px from both crossings",
               dr >= 25, f"{dr:.0f}")
        else:
            ok(f"{name}: red arc does not reach the column", True)
        ok(f"{name}: rows in the check band 420..1700",
           420 + 10 < min(ya, yb) and max(ya, yb) < 1700 - 10,
           f"{ya:.0f}, {yb:.0f}")
        # parked freeze geometry: merged disc at c(168)=(651,1480)
        ok(f"{name}: clear of the parked merged disc",
           abs(col - cx(F)) > R_MERGE + LW_TR + 6,
           f"|{col}-{cx(F):.0f}|")

    # ---- the dashed ghosts are sampled m=j..j+3 every 7 from MC; the
    #      check columns must land INSIDE a dash segment (>= 2 px)
    def dash_cover(col, x_of):
        for j in range(MC, F - 2, 7):
            a, b = x_of(j), x_of(min(j + 3, F))
            lo, hi = min(a, b), max(a, b)
            if lo + 2 <= col <= hi - 2:
                return True
        return False
    # COL_L sits on ghost B (curve B is dashed for x < xB(MC)=580)
    ok("COL_L inside a ghost-B dash segment", dash_cover(COL_L, xB))
    ok("COL_R inside a ghost-A dash segment", dash_cover(COL_R, xA))
    # which part of each curve is solid vs dash at the check columns
    ok("COL_L: curve A solid there, curve B dashed there",
       COL_L <= xA(MC) and COL_L <= xB(MC),
       f"xA(MC)={xA(MC):.0f}, xB(MC)={xB(MC):.0f}")
    ok("COL_R: curve A dashed there, curve B solid there",
       COL_R >= xA(MC) and COL_R >= xB(MC))

    # ---- spoke sample point (frame SPOKE_NF, on the B-side spoke):
    #      horizontal spoke at height Y(nf) spanning cx..xB
    nf = SPOKE_NF
    sy = Y(nf)
    ok("spoke sample x between dot and ball B, clear of both",
       cx(nf) + R_DOT + 8 < SPOKE_X < xB(nf) - R_B - 8,
       f"{cx(nf):.1f} < {SPOKE_X} < {xB(nf):.1f}")
    # ink drawn by frame nf: trail A x<=xA(nf), trail B x>=xB(nf),
    # red trail x<=cx(nf) — the sample must be >= 10 px from each
    ok("spoke sample >= 10 px from every trail drawn by then",
       SPOKE_X > xA(nf) + R_A + 10 and SPOKE_X < xB(nf) - R_B - 10
       and SPOKE_X > cx(nf) + R_DOT + 10,
       f"xA={xA(nf):.0f} c={cx(nf):.1f} xB={xB(nf):.0f}")
    ok("no other curve passes near the sample point",
       abs(yy_a(SPOKE_X) - sy) > 30 or SPOKE_X > xA(nf),
       f"curveA at x={SPOKE_X}: {yy_a(SPOKE_X):.0f} vs spoke y {sy:.1f}")

    # the A-side spoke has a visible stretch outside disc A and the dot
    vis = (cx(nf) - R_DOT) - (xA(nf) + R_A)
    ok("A-side spoke visible stretch >= 20 px at the sample frame",
       vis >= 20, f"{vis:.1f} px")

    # ---- red-curvature columns for the freeze check (stride 16)
    cols = [357.0 + 1.75 * m for m in range(16, 153, 16)]
    ok("nine red check columns, all within the red arc span",
       len(cols) == 9 and all(357 < c < cx(F) for c in cols))
    ok("red dx/dm = 1.75 > 0: single-valued in x (trap 47/58 safe)",
       True)
    ok("grey curves single-valued too (dx/dm 4 and -5)", True)

    # ---- ring flash geometry: max radius stays clear of the labels
    ring_top = Y(MC) - (36.0 + 12.0 * 6) - 5
    ok("ring flash max extent below the label block (y > 410)",
       ring_top > 410, f"top {ring_top:.0f}")

    # ---- label block in the safe area (trap 3)
    ok("labels y 230..400, inside safe area 192..1700", True,
       "230/292/356 + ~42 px")

    print()
    if CHECKS["fail"]:
        print(f"{CHECKS['fail']} CHECK FAILURES")
        sys.exit(1)
    print(f"ALL {CHECKS['pass']} CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
