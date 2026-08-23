#!/usr/bin/env python3
"""your screen has no yellow lamp.

ONE FORM: a circular window onto a lit screen showing solid yellow. The
window magnifies from one character per screen pixel to sixteen, and the
yellow comes apart into red bars and green bars with the blue emitter dark.
Then it goes back, and the eye reassembles it.

Nothing in this renderer ever paints yellow. The only colours emitted are
the three sRGB primaries. The yellow is the AREA AVERAGE, in linear light,
of those emissions over one character cell -- which is the same operation a
retina performs when the stripes fall below its resolution. At x1 a
subpixel is 0.33 of a character and there is no other colour available; at
x16 it is 5.3 characters and there never was any yellow there.

Screen model: RGB stripe, 460 ppi (iPhone 15 density), black matrix on both
axes. Most phone OLEDs are a diamond pentile layout instead -- that is why
the video asks what yours looks like.

Sources, all opened:
  en.wikipedia.org/wiki/Metamerism_(color)   -- three cone types, tristimulus
  en.wikipedia.org/wiki/SRGB                 -- primaries, transfer function
  en.wikipedia.org/wiki/Visual_acuity        -- 1 arcmin (20/20), 0.5 cone
  en.wikipedia.org/wiki/PenTile_matrix_family
  en.wikipedia.org/wiki/IPhone_15            -- ~460 ppi

usage: subpixel.py [check | sheet | render]
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]

from asciilib import Encoder, Frame, Grid, contact, ink_lut  # noqa: E402

G = Grid()
LUT = ink_lut()
BG = (0.008, 0.008, 0.013)          # neutral near-black void
BONE = (0.86, 0.87, 0.91)

FPS = 30
T_END = 13.0
FRAMES = int(round(T_END * FPS))    # 390

# --- the window -----------------------------------------------------------
CX = G.cols / 2.0 - 0.5             # 48.5, the centre of cell 48
CY = 84.5
R_DISC = 42.0                       # cells
RING_B = 0.45

# --- the screen being magnified -------------------------------------------
PPI = 460.0                         # iPhone 15
PIX_UM = 25.4e3 / PPI               # 55.22 um pixel pitch
SUB_UM = PIX_UM / 3.0               # 18.41 um subpixel pitch
EYE_MM = 250.0                      # standard near viewing distance
ARCMIN = 3437.7467707849396         # arcmin per radian

M_MAX = 16.0                        # characters per screen pixel at the top
GAPX = 0.18                         # black matrix, fraction of subpixel width
GAPY = 0.14                         # black matrix, fraction of pixel height
BLUE_ON = 0.05                      # DECISION: a real yellow pixel drives
                                    # blue to zero. Left faintly lit so the
                                    # third lamp is visible as a thing that
                                    # is off, rather than as nothing.
PAN_PX = 4.0                        # lateral drift, screen pixels
PAN_PY = 1.5

# The cell average is computed EXACTLY, not supersampled. A 16-sample
# average of a three-band pattern splits 6/5/5 instead of evenly, which
# tilted the fused colour toward orange and would have been invisible in a
# still. The pattern is piecewise constant and periodic, so integrate it.

LUM = np.array([0.2126, 0.7152, 0.0722])
# sRGB linear -> CIE XYZ (D65), from the primaries + white point.
M_XYZ = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])

TEXT_ROW = 132
SC = 2
FONT = {
    '0': "111101101101111", '1': "010110010010111",
    '2': "111001111100111", '3': "111001111001111",
    '4': "101101111001001", '5': "111100111001111",
    '6': "111100111101111", '7': "111001001001001",
    '8': "111101111101111", '9': "111101111001111",
    'P': "111101111100100", 'X': "101101010101101",
    ' ': "000000000000000",
}


# --- colour ---------------------------------------------------------------
def srgb(x):
    """linear -> sRGB encoded, the exact piecewise curve."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def unsrgb(x):
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


# --- the screen -----------------------------------------------------------
def _cover(a, b, lo, hi):
    """Exact measure of a lit band [lo, hi) of a period-1 pattern inside
    the interval [a, b)."""
    w = hi - lo

    def cum(u):
        n = np.floor(u)
        return n * w + np.clip(u - n - lo, 0.0, w)

    return cum(b) - cum(a)


def column_rgb(m, u0):
    """Linear emission averaged across each character cell, per column.

    Depends only on the column, because the stripes run vertically. Returns
    (cols, 3) of linear light with NO yellow anywhere in it: three bands,
    one per lamp, integrated over the cell.
    """
    c = np.arange(G.cols, dtype=float)
    a = (c - CX) / m + u0
    b = (c + 1.0 - CX) / m + u0
    g = GAPX / 6.0                            # half the matrix, in pixels
    out = np.empty((G.cols, 3))
    lamp = (1.0, 1.0, BLUE_ON)
    for k in range(3):
        out[:, k] = (_cover(a, b, k / 3.0 + g, (k + 1) / 3.0 - g)
                     / (b - a) * lamp[k])
    return out


def row_fill(m, v0):
    """Vertical aperture of the pixel row, per row. Scales all three
    channels equally, so it never touches hue."""
    r = np.arange(G.rows, dtype=float)
    a = (r - CY) / m + v0
    b = (r + 1.0 - CY) / m + v0
    return _cover(a, b, GAPY / 2.0, 1.0 - GAPY / 2.0) / (b - a)


# Exposure: expose on the subject at its honest condition -- the fused
# yellow at x1 -- and let the lit green clip, because a lit subpixel really
# is blazing next to the average of itself and two neighbours. Derived from
# the target, not tuned by eye.
EXPOSE = 0.95
_FUSED1 = column_rgb(1.0, 0.0)[G.cols // 2] * row_fill(1.0, 0.0)[int(CY)]
GAIN = float(unsrgb(EXPOSE) / (_FUSED1 @ LUM))


# --- motion ---------------------------------------------------------------
EASE = 0.55                         # < 1 leaves x1 quickly and dwells at x16


def prog(t):
    """0 -> 1 -> 0, zero derivative at both ends, so the loop closes in
    value AND in velocity. m(0) == m(T_END) exactly. The exponent buys a
    long dwell at the top and a short one at the bottom: a flat yellow disc
    is the first frame, not the first two seconds."""
    c = np.clip((1.0 - np.cos(2.0 * np.pi * t / T_END)) * 0.5, 0.0, 1.0)
    return c ** EASE


def mag(t):
    return M_MAX ** prog(t)


# --- geometry, once -------------------------------------------------------
_R = np.arange(G.rows)[:, None]
_C = np.arange(G.cols)[None, :]
DIST = np.hypot(_C + 0.5 - CX, _R + 0.5 - CY)
ALPHA = np.clip(R_DISC + 0.5 - DIST, 0.0, 1.0)
RING = np.clip(1.0 - np.abs(DIST - (R_DISC + 1.4)) / 1.0, 0.0, 1.0) * RING_B
RING[RING < 0.03] = 0.0


def _text_w(s):
    return len(s) * 3 * SC + (len(s) - 1) * SC


def draw_text(fr, s, row, rgb):
    col = int(round((G.cols - _text_w(s)) / 2.0))
    x = col
    for ch in s:
        pat = FONT[ch]
        for j in range(5):
            for i in range(3):
                if pat[j * 3 + i] == '1':
                    for dy in range(SC):
                        fr.put_run(x + i * SC, row + j * SC + dy,
                                   '#' * SC, rgb)
        x += 3 * SC + SC
    return col


def cell_field(t):
    """Return (bright, hue_rgb, npx, m) for one instant.

    bright: (rows, cols) encoded brightness, drives the glyph.
    hue_rgb: (cols, 3) encoded colour, unit-max so the glyph carries light
             and the colour carries only hue. Never darkened twice.
    """
    m = mag(t)
    p = prog(t)
    crgb = column_rgb(m, PAN_PX * p)          # (cols, 3) linear
    rf = row_fill(m, PAN_PY * p)              # (rows,)
    lin = crgb[None, :, :] * rf[:, None, None] * ALPHA[:, :, None]
    y = lin @ LUM
    bright = srgb(GAIN * y)
    mx = crgb.max(axis=1)
    hue = srgb(crgb / np.where(mx > 1e-12, mx, 1.0)[:, None])
    hue = np.where((mx > 1e-12)[:, None], hue, np.array(BONE)[None, :])
    return bright, hue, int(round(G.cols / m)), m


def draw(t):
    bright, hue, npx, m = cell_field(t)
    fr = Frame(G, BG)
    idx = np.clip((bright * 255.0).astype(np.int64), 0, 255)
    show = bright > 0.02
    key = np.rint(hue * 47.0).astype(np.int64)
    keyid = key[:, 0] * 2304 + key[:, 1] * 48 + key[:, 2]
    n_ink = 0
    for r in range(G.rows):
        sig = np.where(show[r], idx[r] * 300000 + keyid, -1)
        bnd = np.flatnonzero(np.r_[True, sig[1:] != sig[:-1]])
        ends = np.r_[bnd[1:], G.cols]
        for s, e in zip(bnd, ends):
            if sig[s] < 0:
                continue
            fr.put_run(int(s), r, LUT[idx[r, s]] * int(e - s),
                       tuple(hue[s]))
            n_ink += int(e - s)
    # The aperture, drawn in its own colour so it stays an object while
    # everything inside it changes.
    rr, cc = np.nonzero(RING)
    for r, c in zip(rr, cc):
        fr.put(int(c), int(r), LUT[int(RING[r, c] * 255)], BONE)

    lab = "%d PX" % npx
    hot = m >= 3.0                            # one character per subpixel
    col = np.array(BG) + (np.array(BONE) - np.array(BG)) * (1.0 if hot
                                                            else 0.55)
    draw_text(fr, lab, TEXT_ROW, tuple(col))
    fr.n_ink = n_ink
    fr.label = lab
    return fr


# --- the claim, as numbers ------------------------------------------------
def check():
    print(G)
    print("screen  %.0f ppi   pixel %.2f um   subpixel %.2f um"
          % (PPI, PIX_UM, SUB_UM))
    ang_sub = SUB_UM * 1e-3 / EYE_MM * ARCMIN
    ang_pix = PIX_UM * 1e-3 / EYE_MM * ARCMIN
    print("at %.0f mm:  one pixel %.3f arcmin   one subpixel %.3f arcmin"
          % (EYE_MM, ang_pix, ang_sub))
    print("            20/20 resolves 1.000 arcmin  -> subpixel is %.2fx "
          "too small" % (1.0 / ang_sub))
    print("            foveal cone   0.500 arcmin  -> still %.2fx too small"
          % (0.5 / ang_sub))
    assert 3.5 < 1.0 / ang_sub < 4.5
    assert 1.5 < 0.5 / ang_sub < 2.5
    assert ang_pix < 1.0, "a 460 ppi pixel should already be under 20/20"

    # magnification and the loop
    assert abs(mag(0.0) - 1.0) < 1e-12
    assert abs(mag(T_END) - 1.0) < 1e-12
    assert abs(mag(T_END / 2.0) - M_MAX) < 1e-9
    print("magnification  x%.0f -> x%.0f -> x%.0f   over %.1f s, %d frames"
          % (mag(0.0), mag(T_END / 2), mag(T_END), T_END, FRAMES))

    # THE PAYOFF, IN CELLS ON SCREEN
    sub1 = 1.0 / 3.0
    sub16 = M_MAX / 3.0
    print("one subpixel is %.2f characters wide at x1, %.2f at x16"
          % (sub1, sub16))
    assert sub1 < 0.5, "at x1 a subpixel must be unresolvable"
    assert sub16 > 4.0, "at x16 a subpixel must read as a bar"

    # x1: nothing but one colour, and that colour is yellow nobody painted
    c1 = column_rgb(1.0, 0.0)
    inside = np.abs(np.arange(G.cols) - CX) < R_DISC - 2
    mx = c1.max(axis=1)
    hue1 = c1 / mx[:, None]
    spread = float(np.ptp(hue1[inside], axis=0).max())
    print("x1 hue spread across the whole window: %.5f" % spread)
    assert spread < 0.02, "at x1 the window must be one flat colour"
    fused = c1[G.cols // 2]
    enc = srgb(fused / fused.max())
    print("x1 fused colour  linear %s -> sRGB %s"
          % (np.round(fused, 4).tolist(), np.round(enc, 3).tolist()))
    assert enc[0] > 0.95 and enc[1] > 0.90 and enc[2] < 0.30, "that is not yellow"

    # it is the average of the three lamps and nothing else
    manual = np.array([(1.0 - GAPX) / 3.0, (1.0 - GAPX) / 3.0,
                       BLUE_ON * (1.0 - GAPX) / 3.0])
    print("   arithmetic mean of the three lamps: %s"
          % np.round(manual, 4).tolist())
    assert np.abs(fused - manual).max() < 2e-3, "the yellow was painted"

    xyz = M_XYZ @ fused
    xy = xyz[:2] / xyz.sum()
    ref = M_XYZ @ np.array([1.0, 1.0, 0.0])
    print("x1 chromaticity  x %.4f  y %.4f   (sRGB yellow primary mix "
          "%.4f, %.4f)" % (xy[0], xy[1], ref[0] / ref.sum(),
                           ref[1] / ref.sum()))
    assert 0.40 < xy[0] < 0.48 and 0.46 < xy[1] < 0.56

    # x16: pure red bars and pure green bars, wide enough to count
    c16 = column_rgb(M_MAX, PAN_PX)
    mx16 = np.where(c16.max(axis=1) > 1e-12, c16.max(axis=1), 1.0)
    h16 = c16 / mx16[:, None]
    red = (h16[:, 0] > 0.85) & (h16[:, 1] < 0.15)
    grn = (h16[:, 1] > 0.85) & (h16[:, 0] < 0.15)
    print("x16 columns: %d pure red, %d pure green, out of %d"
          % (red.sum(), grn.sum(), G.cols))
    assert red.sum() >= 12 and grn.sum() >= 12
    mixed = ((h16[:, 0] > 0.3) & (h16[:, 1] > 0.3)).sum()
    print("x16 columns that are still a mixture: %d" % mixed)
    assert mixed <= 8, "the bars should be separated, not blended"

    # exposure
    b1, _, npx1, _ = cell_field(0.0)
    b16, _, npx16, _ = cell_field(T_END / 2.0)
    disc = ALPHA > 0.99
    print("exposure gain %.3f  ->  x1 mean brightness %.3f, "
          "x16 range %.3f..%.3f"
          % (GAIN, b1[disc].mean(), b16[disc].min(), b16[disc].max()))
    assert 0.90 <= b1[disc].mean() <= 0.99
    assert b16[disc].max() > 0.98 and b16[disc].min() < 0.06
    print("readout  %d PX -> %d PX   (frame is %.2f mm -> %.3f mm of glass)"
          % (npx1, npx16, npx1 * PIX_UM * 1e-3, npx16 * PIX_UM * 1e-3))
    assert npx1 == G.cols and npx16 == 6

    # window geometry and text
    rr, cc = np.nonzero(ALPHA > 0)
    print("window rows %d..%d  cols %d..%d" % (rr.min(), rr.max(),
                                               cc.min(), cc.max()))
    assert rr.min() > 2 and rr.max() < G.rows - 2
    assert cc.min() >= 0 and cc.max() < G.cols
    w = _text_w("98 PX")
    print("text rows %d..%d  width %d  safe band %d..%d"
          % (TEXT_ROW, TEXT_ROW + 5 * SC - 1, w, G.safe_top, G.safe_bot))
    assert TEXT_ROW >= G.safe_top
    assert TEXT_ROW + 5 * SC - 1 <= G.safe_bot
    assert w < G.cols
    assert TEXT_ROW > CY + R_DISC, "the readout must sit clear of the window"

    # ink, and the seamless loop
    for t in (0.0, 2.0, 4.0, 6.5, 9.0, 11.0, T_END):
        b, _, npx, m = cell_field(t)
        print("  t=%5.2f  x%5.2f  %3d PX  ink %.3f"
              % (t, m, npx, float((b > 0.02).mean())))
    a = draw(0.0)
    b = draw(T_END)
    assert a.surface.get_data() == b.surface.get_data(), "loop has a seam"
    mid = draw(T_END / 2.0)
    assert mid.surface.get_data() != a.surface.get_data()
    print("loop: frame at t=0 and t=%.1f are byte-identical; midpoint is not"
          % T_END)
    print("labels  x1 '%s'   x16 '%s'" % (a.label, mid.label))
    print("OK")


def sheet(path="/tmp/subpixel_sheet.png"):
    ts = [0.0, 1.1, 2.2, 4.0, 6.5, 11.6]
    fr = [draw(t) for t in ts]
    contact(fr, path, cols=3,
            labels=["t=%.1f  x%.1f" % (t, mag(t)) for t in ts])
    print("wrote", path)


def render(path="/tmp/subpixel.mp4"):
    import time
    t0 = time.time()
    with Encoder(path, G, fps=FPS) as enc:
        for f in range(FRAMES):
            enc.write(draw(f / FPS))
            if f % 60 == 0:
                print("  %3d/%d  %.1fs" % (f, FRAMES, time.time() - t0),
                      flush=True)
    print("wrote %s in %.1fs" % (path, time.time() - t0))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    {"check": check, "sheet": sheet, "render": render}[cmd]()
