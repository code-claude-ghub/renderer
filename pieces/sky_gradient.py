#!/usr/bin/env python3
"""THE GRADIENT WITH NO EDGE — for @intothewildvoid.

The whole frame is cloudless sky. Rows map to elevation, 90 deg (zenith)
at the top to 0 deg (horizon) at the bottom. Air mass by Kasten-Young;
whiteness saturates with the extra path length, so the top half is his
sameness and the fade lives low, which is where it really lives.

One move: a 20x20 chip of zenith sky detaches and rides down the frame
keeping the values of the row it was cut from. It is invisible at the top
because it IS its surroundings. It becomes visible only as the sky pales
behind it. At the bottom it sits above the horizon, a dark saturated
square on pale ground: the same sky, twice. Then it relaxes to local
values and vanishes with no edge. Seamless loop. Wordless. Silent.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# Works from scripts/ (asciilib alongside) and from
# the public repo, where pieces live in pieces/. Insert both.
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]

from asciilib import Encoder, Frame, Grid, ink_lut, contact

G = Grid()
RAMP = ink_lut()
BG = (6 / 255, 8 / 255, 14 / 255)
ZEN = np.array([62, 116, 222]) / 255.0   # zenith: deep saturated blue
HOR = np.array([233, 239, 245]) / 255.0  # horizon: near-white

FPS = 30
T = 18.0                                  # seconds, loops
FRAMES = int(T * FPS)

# phases (seconds)
P_HOLD0 = 2.5                             # pure sky
P_DESC0, P_DESC1 = 2.5, 13.0              # chip descends
P_HOLD1 = 15.5                            # chip parked above horizon
P_DISS1 = 17.5                            # chip dissolved by here

TAU = 5.0                                 # whiteness saturation constant
CHIP = 10                                 # chip half-size in cells
R0, R1 = 26, 140                          # chip centre row, start -> end
                                          # 140: chip bottom edge (150) sits just
                                          # above the shorts UI band (rows >=148)
CC = G.cols // 2

RNG = np.random.default_rng(11)
N1 = RNG.standard_normal((G.rows, G.cols))
N2 = RNG.standard_normal((G.rows, G.cols))
N3 = RNG.standard_normal((G.rows, G.cols))
NC = np.random.default_rng(7).standard_normal((2 * CHIP, 2 * CHIP))

ROWS = np.arange(G.rows)
U = ROWS / (G.rows - 1.0)
EL = 90.0 * (1.0 - U) ** 1.15             # elevation per row, degrees


def airmass(el_deg):
    """Kasten & Young 1989. m(90)=1, m(0)~38."""
    return 1.0 / (np.sin(np.radians(el_deg))
                  + 0.50572 * (el_deg + 6.07995) ** -1.6364)


M = airmass(EL)
W_ROW = 1.0 - np.exp(-(M - 1.0) / TAU)    # whiteness per row, 0 zenith -> ~1 horizon
W_CHIP = float(W_ROW[R0])                 # the shade the chip is cut from
B_ROW = 0.50 + 0.50 * W_ROW               # luminance: horizon ~2x zenith glyph-wise

COL_G, ROW_G = np.meshgrid(np.arange(G.cols), ROWS)
COL_F, ROW_F = COL_G.ravel(), ROW_G.ravel()
KEEP = np.ones(COL_F.size, bool)


def smoothstep(p):
    p = min(max(p, 0.0), 1.0)
    return p * p * (3.0 - 2.0 * p)


def chip_state(t):
    """(centre_row, presence) — presence 1 = frozen zenith values, 0 = local sky."""
    if t < P_DESC0:
        return R0, 1.0
    if t < P_DESC1:
        p = smoothstep((t - P_DESC0) / (P_DESC1 - P_DESC0))
        return int(round(R0 + (R1 - R0) * p)), 1.0
    if t < P_HOLD1:
        return R1, 1.0
    if t < P_DISS1:
        return R1, 1.0 - smoothstep((t - P_HOLD1) / (P_DISS1 - P_HOLD1))
    return R1, 0.0


def shimmer(t):
    """Slow scintillation, exactly periodic in T so the loop is seamless."""
    a1 = np.sin(2 * np.pi * t / T)
    a2 = np.sin(4 * np.pi * t / T + 1.7)
    a3 = np.sin(6 * np.pi * t / T + 3.1)
    return (a1 * N1 + a2 * N2 + a3 * N3) / 1.8


def colour(s, w):
    w = min(max(w, 0.0), 1.0)
    r, g, b = ZEN + (HOR - ZEN) * w
    return (r, g, b)


def draw(f):
    t = f / FPS
    n = shimmer(t)
    w = np.repeat(W_ROW[:, None], G.cols, axis=1) * (1.0 + 0.06 * n)
    b = np.repeat(B_ROW[:, None], G.cols, axis=1) * (1.0 + 0.07 * n)

    cy, a = chip_state(t)
    if a > 0.0:
        r_lo, r_hi = cy - CHIP, cy + CHIP
        c_lo, c_hi = CC - CHIP, CC + CHIP
        chip_w = W_CHIP * (1.0 + 0.06 * NC)
        chip_b = (0.50 + 0.50 * W_CHIP) * (1.0 + 0.07 * NC)
        w[r_lo:r_hi, c_lo:c_hi] = (1 - a) * w[r_lo:r_hi, c_lo:c_hi] + a * chip_w
        b[r_lo:r_hi, c_lo:c_hi] = (1 - a) * b[r_lo:r_hi, c_lo:c_hi] + a * chip_b

    fr = Frame(G, BG)
    fr.field(COL_F, ROW_F, KEEP, np.clip(b, 0.05, 1.0).ravel(),
             colour, RAMP, extra=w.ravel())
    return fr


def check():
    assert R0 - CHIP > 0 and R1 + CHIP < G.rows, "chip clips frame"
    for el in (90, 45, 30, 15, 5, 0.4):
        m = airmass(np.array([el]))[0]
        wv = 1.0 - np.exp(-(m - 1.0) / TAU)
        print(f"el {el:5.1f}  m {m:6.2f}  w {wv:.3f}")
    print(f"chip cut at row {R0} (el {EL[R0]:.1f} deg, w {W_CHIP:.3f})")
    print(f"chip parks row {R1} (el {EL[R1]:.1f} deg, w {W_ROW[R1]:.3f})")
    print(f"grid {G.cols}x{G.rows}  frames {FRAMES}  dur {T}s")
    assert not np.isnan(W_ROW).any()


def stills(path="out/sky_sheet.png"):
    ts = [0.0, 4.0, 7.0, 10.0, 13.0, 14.5, 16.4, 17.9]
    contact([draw(int(t * FPS)) for t in ts], path, cols=4,
            labels=[f"t={t}" for t in ts])
    print("sheet:", path)


if __name__ == "__main__":
    import sys
    check()
    if "--stills" in sys.argv:
        stills()
    else:
        out = "out/sky_gradient.mp4"
        with Encoder(out, G, fps=FPS) as enc:
            for f in range(FRAMES):
                enc.write(draw(f))
        print("wrote", out)
