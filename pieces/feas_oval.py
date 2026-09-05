#!/usr/bin/env python3
"""Feasibility for OVAL — the near-ellipse (two cycloid arches).

The claim chain, proven numerically before any pixel exists:
  - a wheel of radius R rolling on a line traces a cycloid arch;
    the arch and its mirror close into an oval of width 2*pi*R and
    height 4R
  - the true ellipse through the same four extreme points contains
    the oval ENTIRELY, touching it at exactly those four points
  - areas: oval 6*pi*R^2, ellipse 2*pi^2*R^2, ratio EXACTLY pi/3;
    the gap between them is exactly 2*pi*(pi-3)*R^2 — the miss is
    (pi-3)/pi of the ellipse, "pi minus 3 parts in pi"
  - the layout, the slice rows/columns (trap 76 discipline), the
    colour fences and the gap-fill pixel prediction all clear

Born from @Dominic-qv3yt's claim "two cycloids make an ellipse" via
@rorucopexperements's reading of it as a claim about the SHAPE.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CHECKS = {"pass": 0, "fail": 0}


def ok(name, cond, detail=""):
    s = "ok  " if cond else "FAIL"
    CHECKS["pass" if cond else "fail"] += 1
    print(f"{s} {name}" + (f" [{detail}]" if detail else ""), flush=True)


# ---------------------------------------------------------------- model
W, H = 1080, 1920
FPS = 30
R = 140.0
CX, CY = 540.0, 960.0
A_ELL = math.pi * R          # ellipse semi-axis, horizontal
B_ELL = 2.0 * R              # ellipse semi-axis, vertical
X_L = CX - A_ELL             # left kiss x
X_R = CX + A_ELL             # right kiss x

A_HI = 77                    # act A: n 0..77, t = 2*pi*n/77
B_LO, B_HI = 78, 155         # act B mirror arch
E_LO, E_HI = 162, 209        # ellipse sweep, phi = 2*pi*(n-161)/48
TICK_N = 210                 # kiss ticks appear
GAP_LO = 214                 # gap fill fades in (10 frames)
FREEZE = 238
N = 282

LW_OVAL, LW_ELL, LW_GROUND, LW_RING, LW_SPOKE = 5.5, 5.5, 4.0, 5.0, 4.0
R_PEN = 9.0

TITLE = ("a rolling wheel draws this oval. it is not an ellipse — "
         "it misses by exactly π−3 parts in π.")

ROWS_ORDER = (770, 820, 1100, 1150)   # flank rows: 4 ordered clusters
ROWS_TIGHT = (800, 840, 1080, 1120)   # tight on-curve rows (|dx/dy|<=1.2)
COLS_CREST = (480, 600)               # crest columns (avoid tick at 540)
GAP_MARGIN = 3.5                      # stroke+fence erosion per side


def arch_top(t):
    """Top arch, screen coords. t in [0, 2*pi]."""
    return (CX - A_ELL + R * (t - np.sin(t)),
            CY - R * (1.0 - np.cos(t)))


def wheel_a(t):
    return (CX - A_ELL + R * t, CY - R)


def mirror(x, y):
    return 2.0 * CX - x, 2.0 * CY - y


def ellipse_pt(phi):
    return CX + A_ELL * np.cos(phi), CY - B_ELL * np.sin(phi)


def cyc_x_at(dy):
    """Top-arch crossings at height dy = CY - y in (0, 2R]:
    returns (x_left, x_right)."""
    d = dy / R
    t = np.arccos(1.0 - d)
    xl = CX - A_ELL + R * (t - np.sin(t))
    xr = CX - A_ELL + R * ((2.0 * np.pi - t) + np.sin(t))
    return xl, xr


def ell_x_at(dy):
    s = np.sqrt(np.clip(1.0 - (dy / B_ELL) ** 2, 0.0, None))
    return CX - A_ELL * s, CX + A_ELL * s


# ---------------------------------------------------------------- run
def main():
    print("== feasibility: OVAL ==", flush=True)

    ok("title fits (<= 100 chars)", len(TITLE) <= 100, f"{len(TITLE)}")
    ok("duration sane", N / FPS <= 180.0, f"{N / FPS:.2f} s")

    # ---- geometry endpoints
    x0, y0 = arch_top(0.0)
    x2, y2 = arch_top(2.0 * math.pi)
    xc, yc = arch_top(math.pi)
    ok("arch starts at the left kiss",
       abs(x0 - X_L) < 1e-9 and abs(y0 - CY) < 1e-9,
       f"({x0:.3f}, {y0:.3f})")
    ok("arch ends at the right kiss",
       abs(x2 - X_R) < 1e-9 and abs(y2 - CY) < 1e-9,
       f"({x2:.3f}, {y2:.3f})")
    ok("crest at (540, 680)",
       abs(xc - CX) < 1e-9 and abs(yc - (CY - 2 * R)) < 1e-9,
       f"({xc:.3f}, {yc:.3f})")

    # ---- rolling identity: pen == wheel centre + R*(-sin t, cos t)
    t = np.linspace(0.0, 2.0 * np.pi, 200_001)
    px, py = arch_top(t)
    wx, wy = wheel_a(t)
    off = np.hypot(px - wx, py - wy)
    ok("pen sits exactly on the rim (|pen-centre| == R)",
       float(np.abs(off - R).max()) < 1e-9,
       f"max err {np.abs(off - R).max():.2e}")
    ok("wheel centre height constant == CY - R",
       float(np.abs(wy - (CY - R)).max()) == 0.0)
    ok("wheel bottom on the ground line",
       float(np.abs((wy + R) - CY).max()) == 0.0)
    ok("contact point x == arc length rolled (R*t)",
       float(np.abs((wx - X_L) - R * t).max()) < 1e-9)

    # ---- mirror arch: exact transform (trap 28), continuity
    mx, my = mirror(px, py)
    bx0, by0 = mirror(*arch_top(0.0))
    ok("mirror arch starts where arch A ended (right kiss)",
       abs(bx0 - X_R) < 1e-9 and abs(by0 - CY) < 1e-9)
    ok("mirror is the exact point map (bitwise)",
       np.array_equal(mx, 2.0 * CX - px)
       and np.array_equal(my, 2.0 * CY - py))

    # ---- inscription: the oval lies INSIDE the ellipse
    E = ((px - CX) / A_ELL) ** 2 + ((py - CY) / B_ELL) ** 2
    ok("oval never leaves the ellipse (E <= 1 everywhere)",
       float(E.max()) <= 1.0 + 1e-12, f"max E {E.max():.12f}")
    interior = (t > 0.05) & (t < 2 * np.pi - 0.05) & (np.abs(t - np.pi) > 0.05)
    ok("strictly inside away from the four kisses",
       float(E[interior].max()) < 0.99999,
       f"interior max E {E[interior].max():.7f}")
    kissE = [((arch_top(tt)[0] - CX) / A_ELL) ** 2
             + ((arch_top(tt)[1] - CY) / B_ELL) ** 2
             for tt in (0.0, math.pi)]
    ok("kiss points ON the ellipse (E == 1)",
       max(abs(e - 1.0) for e in kissE) < 1e-12,
       f"devs {[f'{abs(e - 1):.1e}' for e in kissE]}")

    # ---- areas (shoelace on dense samples vs closed forms)
    area_arch = -float(np.trapezoid(py - CY, px))   # screen y is down
    ok("arch area == 3*pi*R^2 (Roberval 1634)",
       abs(area_arch - 3 * math.pi * R * R) / (3 * math.pi * R * R) < 1e-6,
       f"{area_arch:.2f} vs {3 * math.pi * R * R:.2f}")
    area_oval = 2.0 * area_arch
    area_ell_cf = math.pi * A_ELL * B_ELL
    ok("ellipse area closed form == 2*pi^2*R^2",
       abs(area_ell_cf - 2 * math.pi ** 2 * R * R) < 1e-9)
    phi = np.linspace(0.0, 2.0 * np.pi, 400_001)
    ex, ey = ellipse_pt(phi)
    area_ell_num = 0.5 * abs(float(
        np.trapezoid((ex - CX) * np.gradient(-(ey - CY), phi)
                     - (-(ey - CY)) * np.gradient(ex - CX, phi), phi)))
    ok("ellipse area numeric == pi*a*b",
       abs(area_ell_num - area_ell_cf) / area_ell_cf < 1e-6,
       f"{area_ell_num:.2f} vs {area_ell_cf:.2f}")
    gap_cf = 2.0 * math.pi * (math.pi - 3.0) * R * R
    ok("gap == 2*pi*(pi-3)*R^2",
       abs((area_ell_cf - area_oval) - gap_cf) / gap_cf < 1e-5,
       f"{area_ell_cf - area_oval:.2f} vs {gap_cf:.2f} px^2")
    ok("ratio == pi/3 exactly",
       abs(area_ell_cf / (6 * math.pi * R * R) - math.pi / 3.0) < 1e-12,
       f"{area_ell_cf / (6 * math.pi * R * R):.12f}")
    ok("the title's fraction: 1 - 3/pi == (pi-3)/pi",
       abs((1 - 3 / math.pi) - (math.pi - 3) / math.pi) < 1e-15,
       f"{(math.pi - 3) / math.pi:.6f} (= {100 * (math.pi - 3) / math.pi:.3f}%)")

    # ---- gap size: is the sliver visible? (trap 67 arithmetic)
    th = t[(t >= np.pi) & (t <= 2 * np.pi)]
    cxr, cyr = arch_top(th)
    dyr = CY - cyr
    exr = CX + A_ELL * np.sqrt(np.clip(1 - (dyr / B_ELL) ** 2, 0, 1))
    hgap = exr - cxr
    i = int(hgap.argmax())
    ok("max horizontal gap >= 20 px",
       float(hgap[i]) >= 20.0,
       f"{hgap[i]:.1f} px at y {cyr[i]:.0f}")
    # perpendicular gap: min distance from each cycloid pt to ellipse
    d2 = ((cxr[:, None] - ex[None, ::400]) ** 2
          + (cyr[:, None] - ey[None, ::400]) ** 2)
    perp = np.sqrt(d2.min(1))
    ok("max perpendicular gap >= 12 px (sliver readable)",
       float(perp.max()) >= 12.0, f"{perp.max():.1f} px")

    # ---- slice rows: 4 ordered clusters, separations clear
    row_report = []
    rows_good = True
    for r_ in ROWS_ORDER + ROWS_TIGHT:
        dy = abs(CY - r_)
        cl, cr = (float(v) for v in cyc_x_at(np.float64(dy)))
        el, er = (float(v) for v in ell_x_at(np.float64(dy)))
        sep = min(cl - el, er - cr)
        width = cr - cl
        row_report.append(f"y{r_}: sep {sep:.1f}")
        if not (el < cl < cr < er):
            rows_good = False
        if sep < 16.0 or width < 200.0:
            rows_good = False
    ok("slice rows give 4 ordered, separated clusters",
       rows_good, "; ".join(row_report))
    # slopes at the TIGHT rows (trap 76: the tight on-curve claim
    # only lives where the row cuts near-perpendicular; ROWS_ORDER
    # includes the max-gap rows and asserts ordering with a loose
    # tolerance instead)
    worst = 0.0
    for r_ in ROWS_TIGHT:
        dy = abs(CY - r_)
        d = dy / R
        tt = math.acos(1.0 - d)
        cyc_dxdy = abs(math.tan(tt / 2.0))       # dx/dy of the arch
        ell_dxdy = (A_ELL * dy / B_ELL ** 2
                    / math.sqrt(1 - (dy / B_ELL) ** 2))
        worst = max(worst, cyc_dxdy, ell_dxdy)
    ok("tight rows cut both curves near-perpendicular (|dx/dy| <= 1.2)",
       worst <= 1.2, f"worst {worst:.2f}")

    # ---- crest columns: near-flat there (trap 76 columns)
    worst_c = 0.0
    crest_dev = 0.0
    for c in COLS_CREST:
        # invert x for the top arch near the crest
        tt = min((abs(arch_top(v)[0] - c), v)
                 for v in np.linspace(2.0, 4.3, 20001))[1]
        slope = abs(1.0 / math.tan(tt / 2.0))    # |dy/dx|
        worst_c = max(worst_c, slope)
        ycyc = arch_top(tt)[1]
        yell = CY - B_ELL * math.sqrt(1 - ((c - CX) / A_ELL) ** 2)
        crest_dev = max(crest_dev, abs(ycyc - yell))
    ok("crest columns near-flat (|dy/dx| <= 0.55)",
       worst_c <= 0.55, f"worst {worst_c:.2f}")
    ok("curves within one stroke width at the crest columns",
       crest_dev <= 2.6, f"max sep {crest_dev:.2f} px")

    # ---- gap-fill pixel prediction (the render check's constant)
    pred = 0.0
    for yy in range(682, 1239):
        dy = abs(CY - yy)
        if dy >= 2 * R or dy == 0:
            continue
        cl, cr = (float(v) for v in cyc_x_at(np.float64(dy)))
        el, er = (float(v) for v in ell_x_at(np.float64(dy)))
        pred += max(0.0, (cl - GAP_MARGIN) - (el + GAP_MARGIN))
        pred += max(0.0, (er - GAP_MARGIN) - (cr + GAP_MARGIN))
    ok("gap-fill pixel prediction sane", 4000 < pred < 20000,
       f"GAP_PX_PRED = {pred:.0f}")

    # ---- layout / safe areas
    ok("oval fits the frame with margins",
       X_L > 60 and X_R < 1020 and CY - 2 * R > 640 and CY + 2 * R < 1280,
       f"x {X_L:.1f}..{X_R:.1f}, y {CY - 2 * R:.0f}..{CY + 2 * R:.0f}")
    ok("wheels clear both label zones",
       (CY - 2 * R) >= 560 and (CY + 2 * R) <= 1360)
    ok("kiss ticks inside safe area",
       CY - 2 * R - 20 > 420 and CY + 2 * R + 20 < 1632)
    ok("pen never leaves the frame",
       float(px.min()) > 0 and float(px.max()) < W
       and float(py.min()) > 420 and float(py.max()) < 1632,
       f"x {px.min():.0f}..{px.max():.0f}, y {py.min():.0f}..{py.max():.0f}")

    # ---- timing
    ok("acts tile the piece",
       A_HI + 1 == B_LO and B_HI < E_LO and E_HI < TICK_N < GAP_LO < FREEZE,
       f"A..77 B78..155 E162..209 T210 G214 F{FREEZE} N{N}")
    ok("freeze long enough and fades complete before it",
       N - FREEZE >= 40 and GAP_LO + 10 < FREEZE)
    # per-frame chord flatness (x4 subdivision in the piece)
    tt = 2 * np.pi * np.arange(0, A_HI * 4 + 1) / (A_HI * 4)
    qx, qy = arch_top(tt)
    chord = np.hypot(np.diff(qx), np.diff(qy))
    ok("arch chords short (max sub-step)", float(chord.max()) < 8.0,
       f"{chord.max():.2f} px")
    dphi = 2 * np.pi / 48 / 4
    sag = A_ELL * (1 - math.cos(dphi / 2))
    ok("ellipse sweep sagitta < 0.2 px", sag < 0.2, f"{sag:.3f}")

    # ---- colour fences: each palette colour hits only its own mask
    def u8(c):
        return tuple(int(v * 255.0 + 0.5) for v in c)

    pal = {
        "bg": u8((0.055, 0.060, 0.078)),
        "ground": u8((0.36, 0.40, 0.48)),
        "disc": u8((0.80, 0.82, 0.86)),
        "oval": u8((0.92, 0.72, 0.20)),
        "ell": u8((0.88, 0.18, 0.14)),
        "pen": u8((0.98, 0.25, 0.18)),
        "gap": u8((0.40, 0.08, 0.30)),
        "tick": u8((0.97, 0.97, 0.99)),
        "lbl": u8((0.55, 0.57, 0.62)),
    }

    def fences(c):
        r_, g_, b_ = (int(v) for v in c)
        mx = max(c)
        return {
            "red": r_ - g_ > 60 and r_ > 190,
            "warm": r_ > 180 and g_ > 130 and b_ < 100,
            "gapf": r_ - g_ > 60 and r_ < 150 and 60 < b_ < 120,
            "grey": b_ > r_ + 8 and 40 < mx < 160,
            "disc": all(abs(int(v) - w) <= 10
                        for v, w in zip(c, (204, 209, 219))),
            "white": min(c) >= 225,
        }

    want = {"bg": set(), "ground": {"grey"}, "disc": {"disc"},
            "oval": {"warm"}, "ell": {"red"}, "pen": {"red"},
            "gap": {"gapf"}, "tick": {"white"}, "lbl": {"grey"}}
    fence_ok = True
    detail = []
    for name, c in pal.items():
        hit = {k for k, v in fences(c).items() if v}
        if hit != want[name]:
            fence_ok = False
            detail.append(f"{name}: {hit} != {want[name]}")
    ok("palette fence audit (each colour only its own fence)",
       fence_ok, "; ".join(detail) or "clean")
    # the dangerous blends land in NO fence
    blends = [tuple((a + b) // 2 for a, b in zip(pal["oval"], pal["gap"])),
              tuple((a + b) // 2 for a, b in zip(pal["ell"], pal["bg"])),
              tuple((a + b) // 2 for a, b in zip(pal["gap"], pal["bg"])),
              tuple((a + b) // 2 for a, b in zip(pal["ell"], pal["gap"])),
              tuple((a + b) // 2 for a, b in zip(pal["oval"], pal["bg"]))]
    blend_ok = all(not any(fences(c).values()) for c in blends)
    ok("AA midpoint blends (5 dangerous pairs) hit no fence",
       blend_ok, "; ".join(str(c) for c in blends))

    # ---- labels fit
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    labels = [
        "a wheel rolls — its rim point draws a cycloid",
        "back along the underside — the oval closes",
        "the true ellipse through the same four points",
        "they touch at exactly four points",
        "areas: oval 6πr² · ellipse 2π²r²",
        "the miss: exactly π−3 parts in π",
    ]
    f = ImageFont.truetype(font_path, 34 * 4)
    wide = 0
    for s in labels:
        im = Image.new("L", (34 * len(s) * 4, 34 * 8), 0)
        ImageDraw.Draw(im).text((8, 8), s, font=f, fill=255)
        a = np.asarray(im)
        xs = np.where(a.any(0))[0]
        wide = max(wide, (xs.max() - xs.min()) // 4)
    ok("widest label fits the frame", wide <= 1000, f"{wide} px")

    print()
    if CHECKS["fail"]:
        print(f"{CHECKS['fail']} CHECK FAILURES")
        raise SystemExit(1)
    print(f"ALL {CHECKS['pass']} FEASIBILITY CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
