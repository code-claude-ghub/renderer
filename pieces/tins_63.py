#!/usr/bin/env python3
"""THREE CANS — the US poverty line is a food budget times three.

The recipe (Mollie Orshansky, SSA, 1963-64): take the USDA economy food
plan and multiply it by 3. The 3 came from the 1955 Household Food
Consumption Survey, which found families of three or more spent about a
third of their after-tax income on food. The thresholds have only ever
been carried forward on CPI. The multiplier has never been re-based.

Food is now 12.9% of what an American household spends (BLS Consumer
Expenditure Survey, 2024). About an eighth, not a third.

THE FORM: three stacked cans -- the multiplier, made countable. A coral
hoop marks one whole household budget, and at 1963 the stack reaches it
exactly. As the years run, a can is worth a smaller and smaller slice of
that budget, so the cans squash. The stack is always three cans. The
hoop never moves. The gap that opens is the argument.

    python3 scripts/tins_63.py --check
    python3 scripts/tins_63.py --sheet
    python3 scripts/tins_63.py
"""

import math
import os
import sys

import cairo
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# asciilib sits beside this file in the working tree and one level up in
# the public repo, where pieces live in pieces/. Insert both.
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
from asciilib import (Camera, Encoder, Frame, Grid, contact, depth_cue,
                      ink_lut, lambert, rot, specular, visible, zbuffer)

# ---------------------------------------------------------------- the numbers

SHARE_1963 = 1.0 / 3.0      # 1955 Household Food Consumption Survey, 3+ persons
SHARE_2026 = 0.129          # BLS Consumer Expenditure Survey 2024, food share
RATIO = SHARE_2026 / SHARE_1963          # 0.387 -- what three cans are worth now

YEAR_A, YEAR_B = 1963, 2026

FPS = 30
HOLD_A = 45                 # 1.5 s resting on the hoop, so the reference reads
RUN = 420                   # 14.0 s of years
HOLD_B = 75                 # 2.5 s on the gap
FRAMES = HOLD_A + RUN + HOLD_B           # 540 -> 18.0 s

# ------------------------------------------------------------------ the solid

R = 0.62                    # can radius
H0 = 1.00                   # can height in 1963
SEAM = 0.03                 # cans touch rim to rim, as they do on a shelf
DISH = 0.055                # lid recess -- a flat lid cannot be shaded
RIM = 0.045                 # rolled-seam tube radius
RC = R + 0.048              # the rolled seam stands proud of the body:
                            # that waist is what makes THREE cans countable
                            # rather than one drum with grooves (trap 14)
TOP63 = 3.0 * H0 + 2.0 * SEAM            # 3.10 -- where the hoop lives forever

RH, TH = 0.92, 0.055        # hoop major radius, tube radius
SEAM_TH = (0.0, 2.10, 4.30)              # each can stacked at its own angle
SEAM_HALF = 0.028                        # seam half-width in radians
SEAM_OUT = 0.012                         # proud of the wall, so it cannot z-fight

PITCH = -0.28               # negative tilts the lids toward the viewer
SPIN = 0.38                 # rad/s

LAMP = np.array([-0.45, -0.70, 0.55])    # y is DOWN, so this is upper-left-front

BG = (0.086, 0.047, 0.106)          # deep plum
TIN = (0.972, 0.918, 0.784)         # warm bone
HOOP = (0.976, 0.361, 0.286)        # coral

G = Grid()
RAMP = ink_lut()
RNG = np.random.default_rng(19630825)

# rows the composition is pinned to: base of the stack, and how tall 1963 is
ROW_BASE = 136.0
ROWS_TALL = 88.0

N_WALL, N_LID, N_BASE, N_RIM, N_HOOP, N_SEAM = 30000, 8000, 5000, 5000, 20000, 3000

# ------------------------------------------------------------ cached samples
# Random sampling is its own jitter -- no lattice to beat against the cells.

W_TH = RNG.random(N_WALL) * 2 * math.pi
W_U = RNG.random(N_WALL)
L_TH = RNG.random(N_LID) * 2 * math.pi
L_R = R * np.sqrt(RNG.random(N_LID))
B_TH = RNG.random(N_BASE) * 2 * math.pi
B_R = R * np.sqrt(RNG.random(N_BASE))
R_TH = RNG.random(N_RIM) * 2 * math.pi
R_PH = RNG.random(N_RIM) * 2 * math.pi
H_TH = RNG.random(N_HOOP) * 2 * math.pi
H_PH = RNG.random(N_HOOP) * 2 * math.pi
S_DT = (RNG.random(N_SEAM) * 2 - 1) * SEAM_HALF
S_U = RNG.random(N_SEAM)


def _wall(base, hc):
    x, z = R * np.cos(W_TH), R * np.sin(W_TH)
    up = base + W_U * hc
    n = np.stack([np.cos(W_TH), np.zeros(N_WALL), np.sin(W_TH)], -1)
    return np.stack([x, up, z], -1), n


def _seam(base, hc, th0):
    """The one feature that makes a solid of revolution show that it turns.

    A stack of coaxial cylinders spun about its own axis is visually
    identical in every frame. Real cans have a side seam, and stacking them
    at three different angles makes the stack read as three objects rather
    than one column.
    """
    th = th0 + S_DT
    rr = R + SEAM_OUT
    p = np.stack([rr * np.cos(th), base + S_U * hc, rr * np.sin(th)], -1)
    n = np.stack([np.cos(th), np.zeros(N_SEAM), np.sin(th)], -1)
    return p, n


def _lid(base, hc):
    x, z = L_R * np.cos(L_TH), L_R * np.sin(L_TH)
    up = base + hc - DISH * (1.0 - (L_R / R) ** 2)
    g = 2.0 * DISH / R ** 2
    n = np.stack([-g * x, np.ones(N_LID), -g * z], -1)
    return np.stack([x, up, z], -1), n / np.linalg.norm(n, axis=1, keepdims=True)


def _base(base):
    x, z = B_R * np.cos(B_TH), B_R * np.sin(B_TH)
    n = np.stack([np.zeros(N_BASE), -np.ones(N_BASE), np.zeros(N_BASE)], -1)
    return np.stack([x, np.full(N_BASE, base), z], -1), n


def _ring(up, major, tube, th, ph):
    rr = major + tube * np.cos(ph)
    p = np.stack([rr * np.cos(th), up + tube * np.sin(ph), rr * np.sin(th)], -1)
    n = np.stack([np.cos(ph) * np.cos(th), np.sin(ph), np.cos(ph) * np.sin(th)], -1)
    return p, n


def build(hc):
    """Assemble the stack plus the hoop. Returns points, normals, material."""
    P, N, M = [], [], []
    for k in range(3):
        b = k * (hc + SEAM)
        for p, n in (_wall(b, hc), _lid(b, hc), _base(b),
                     _ring(b + RIM, RC, RIM, R_TH, R_PH),
                     _ring(b + hc - RIM, RC, RIM, R_TH, R_PH)):
            P.append(p); N.append(n); M.append(np.zeros(len(p)))
        p, n = _seam(b, hc, SEAM_TH[k])
        P.append(p); N.append(n); M.append(np.full(len(p), 2.0))
    p, n = _ring(TOP63, RH, TH, H_TH, H_PH)
    P.append(p); N.append(n); M.append(np.ones(len(p)))
    P, N, M = np.concatenate(P), np.concatenate(N), np.concatenate(M)
    # up-space -> world: y is DOWN on screen (trap 1)
    P = np.stack([P[:, 0], -P[:, 1], P[:, 2]], -1)
    N = np.stack([N[:, 0], -N[:, 1], N[:, 2]], -1)
    return P, N, M


# ------------------------------------------------------------------- camera
# Set by hand, not fit(): the argument IS the size comparison, so the scale
# must not move, and the frame has to leave rows 20..36 clear for the year.

CAM = Camera(G)
CAM.scale = ROWS_TALL / TOP63
CAM.off = np.array([0.0, (G.cy - ROW_BASE) / CAM.scale])


def can_height(f):
    t = min(max((f - HOLD_A) / float(RUN), 0.0), 1.0)
    return H0 * (1.0 + (RATIO - 1.0) * t), int(round(YEAR_A + (YEAR_B - YEAR_A) * t))


def colour(shade, extra):
    ink = HOOP if 0.5 < extra < 1.5 else TIN
    a = 0.50 + 0.50 * shade
    return (BG[0] + (ink[0] - BG[0]) * a,
            BG[1] + (ink[1] - BG[1]) * a,
            BG[2] + (ink[2] - BG[2]) * a)


# --------------------------------------------------------------- the counter
# Trap 11: a one-cell glyph is ~4 px on a phone. Rasterise at 8x, area-average
# onto the cell grid, stamp it, and punch a background halo behind it.

def raster(text, cell_w, cell_h, ss=8):
    W, H = cell_w * ss, cell_h * ss
    surf = cairo.ImageSurface(cairo.FORMAT_A8, W, H)
    ctx = cairo.Context(surf)
    ctx.select_font_face("monospace", cairo.FONT_SLANT_NORMAL,
                         cairo.FONT_WEIGHT_BOLD)
    size = float(H)
    for _ in range(60):
        ctx.set_font_size(size)
        e = ctx.text_extents(text)
        if e.width <= W * 0.96 and e.height <= H * 0.96:
            break
        size *= 0.94
    e = ctx.text_extents(text)
    ctx.move_to((W - e.width) / 2.0 - e.x_bearing,
                (H - e.height) / 2.0 - e.y_bearing)
    ctx.set_source_rgba(1, 1, 1, 1)
    ctx.show_text(text)
    surf.flush()
    buf = np.frombuffer(surf.get_data(), np.uint8)
    buf = buf.reshape(H, surf.get_stride())[:, :W]
    return buf.reshape(cell_h, ss, cell_w, ss).mean(axis=(1, 3)) / 255.0


def stamp(fr, cov, c0, r0, rgb):
    m = cov > 0.04
    halo = m.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            halo |= np.roll(np.roll(m, dr, 0), dc, 1)
    fr.ctx.set_source_rgb(*BG)
    for r, c in zip(*np.nonzero(halo)):
        fr.ctx.rectangle((c0 + c) * G.cell, (r0 + r) * G.cell, G.cell, G.cell)
    fr.ctx.fill()
    for r, c in zip(*np.nonzero(m)):
        v = float(cov[r, c])
        fr.put(c0 + c, r0 + r,
               RAMP[int(np.clip(v * (len(RAMP) - 1), 1, len(RAMP) - 1))], rgb)


TXT_W, TXT_H, TXT_ROW = 38, 17, 19
_TEXT = {y: raster(str(y), TXT_W, TXT_H) for y in range(YEAR_A, YEAR_B + 1)}


# ---------------------------------------------------------------------- draw

def draw(f):
    hc, year = can_height(f)
    P, N, M = build(hc)
    # Spin FIRST about the cans' own axis, tilt the view SECOND. Doing it the
    # other way round precesses the whole stack -- it leans sideways and the
    # silhouette wanders, which wrecks the one comparison the piece is making.
    P, N = rot(P, N, 0.0, SPIN * f / FPS, 0.0)
    P, N = rot(P, N, PITCH, 0.0, 0.0)

    col, row, z = CAM.project(P)
    ok = visible(G, col, row)
    col, row, z, N, M = col[ok], row[ok], z[ok], N[ok], M[ok]
    _, keep = zbuffer(G, col, row, z)

    lam = lambert(N, LAMP)
    spec = specular(N, LAMP, 26)
    cue = depth_cue(z, near=1.0, far=0.94)          # trap 13: a tall object
    # A 0.10 floor let the unlit half of the body fall to one faint glyph
    # and the right edge of the silhouette dissolved into the ground.
    tin = (0.24 + 0.76 * lam + 0.28 * spec) * cue
    hoop = (0.44 + 0.56 * lam) * cue
    seam = 0.78 * cue           # a drawn line, fixed shade (trap 9)
    shade = np.clip(np.where(M > 1.5, seam,
                             np.where(M > 0.5, hoop, tin)), 0.0, 1.0)

    fr = Frame(G, BG)
    fr.field(col, row, keep, shade, colour, RAMP, extra=M)
    stamp(fr, _TEXT[year], int(round(G.cx - TXT_W / 2.0)), TXT_ROW, TIN)
    return fr


# --------------------------------------------------------------------- check

def check():
    print(G)
    print("share 1963 %.4f  share 2026 %.4f  ratio %.4f" %
          (SHARE_1963, SHARE_2026, RATIO))
    print("multiplier then %.2f  now %.2f" % (1 / SHARE_1963, 1 / SHARE_2026))
    print("stack 1963 %.3f  stack 2026 %.3f  hoop %.3f" %
          (TOP63, 3 * H0 * RATIO + 2 * SEAM, TOP63))
    print("scale %.3f cells/unit   frames %d  = %.1f s" %
          (CAM.scale, FRAMES, FRAMES / float(FPS)))
    print("counter rows %d..%d  (safe_top %d)" %
          (TXT_ROW, TXT_ROW + TXT_H, G.safe_top))
    for f in (0, HOLD_A, HOLD_A + RUN // 2, FRAMES - 1):
        hc, year = can_height(f)
        P, N, M = build(hc)
        P, N = rot(P, N, 0.0, SPIN * f / FPS, 0.0)
        P, N = rot(P, N, PITCH, 0.0, 0.0)
        c, r, _ = CAM.project(P)
        ch, rh = c[M == 1.0], r[M == 1.0]
        cs, rs = c[M != 1.0], r[M != 1.0]
        print("f%-4d %d  can %.3f | stack col %d..%d row %d..%d "
              "| hoop row %d..%d | gap %d rows"
              % (f, year, hc, cs.min(), cs.max(), rs.min(), rs.max(),
                 rh.min(), rh.max(), rs.min() - rh.max()))
        assert cs.min() >= 0 and cs.max() < G.cols, "stack off the sides"
        assert rh.min() > TXT_ROW + TXT_H, "hoop collides with the counter"


def sheet():
    fs = [0, HOLD_A, HOLD_A + 90, HOLD_A + 180,
          HOLD_A + 280, HOLD_A + 380, HOLD_A + RUN, FRAMES - 1]
    contact([draw(f) for f in fs], "out/tins_sheet.png", cols=4,
            labels=["f%d %d" % (f, can_height(f)[1]) for f in fs])
    print("wrote out/tins_sheet.png")


if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)
    if "--check" in sys.argv:
        check()
    elif "--sheet" in sys.argv:
        check(); sheet()
    else:
        check()
        with Encoder("out/tins.mp4", G, fps=FPS) as enc:
            for f in range(FRAMES):
                enc.write(draw(f))
                if f % 60 == 0:
                    print("  %d/%d" % (f, FRAMES), flush=True)
        print("wrote out/tins.mp4")
