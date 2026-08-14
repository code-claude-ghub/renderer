#!/usr/bin/env python3
"""
blind spot -- a dot walks into the hole in your eye.

Every eye has a patch of retina with no photoreceptors on it, where the
million-odd nerve fibres gather up and punch out through the back of the
eyeball. Nothing that lands there is seen. It sits about 15 degrees to the
TEMPORAL side of where you are looking and about 1.5 degrees below the
horizontal, and it is roughly 7.6 degrees wide by 8.3 degrees tall -- big
enough to swallow fourteen full moons laid side by side.

You have never noticed it. Not because it is small, but because the visual
system does not report a hole. It reports whatever the surround is doing.
So this piece rules the whole screen with horizontal lines, and when the
dot drops into the hole the lines close over it as if it had never been.

The piece is a demonstration, not an illustration, so the geometry has to
be honest. The dot walks along a track whose angular distance from the
fixation cross is computed for a real phone held at a real distance, and
check() refuses to render unless that track passes cleanly through the
blind spot's measured band (13.6 to 21.2 degrees) with the dot small
enough to fit entirely inside the hole.

Left eye only. The blind spot lies in the TEMPORAL field, which for the
left eye is the left side -- so the cross goes right, the dot goes left.

Sources for the numbers, all in the description:
  eccentricity + 1.5 deg below horizontal ...... Humphrey perimetry norms
  7.6 x 8.3 deg measured extent ................ Vision Research, 2023
  discovery + the reasoning order .............. Mariotte, "Nouvelle
      decouverte touchant la veue", 1668 -- he cut eyes open first, found
      the disc had no receptors, PREDICTED a hole in sight, then went and
      found it.

Shipped: https://youtube.com/watch?v=G9mUwZ14k_E
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asciilib import Grid, Frame, Encoder                      # noqa: E402

OUT = "/tmp/blind_spot.mp4"
FPS = 30

HEAD = 90                  # instruction, 3.0 s
SWEEP = 150                # one traverse, 5.0 s
NSWEEP = 4                 # out, back, out, back -- returns to start
DRAWIN = 18                # rules wipe on over 0.6 s, ending at HEAD
TOTAL = HEAD + SWEEP * NSWEEP

# --- colourway: deep plum ground, lilac rules, bone cross, vermilion dot ---
BG = (0.075, 0.043, 0.106)
RULE = (0.530, 0.451, 0.702)
BONE = (0.949, 0.925, 0.867)
DOT = (0.976, 0.353, 0.153)
DOT_TEX = (0.741, 0.180, 0.055)

# A ruled line one row tall is a row of dashes with gaps between them, and
# gaps are the one thing this piece cannot have: the claim is that the
# pattern closes over the dot with nothing missing. Two rows of '#' -- the
# only glyph measured at full ink -- make a bar with no holes in it.
BAND_ON = 2
BAND_OFF = 3
BAND_STEP = BAND_ON + BAND_OFF
BAND_CH = "#"

DOT_CH = "@"
R_DOT = 3.6                # dot radius in cells

# --- the viewing model the geometry is designed around -------------------
PHONE_W_CM = 7.0           # a 6.1" 19.5:9 handset is 7.0 cm across
DIST_CM = 15.0             # phone held close, as the instruction says.
#                            Set by check(), not by taste. The track only
#                            crosses 15 deg for viewing distances of roughly
#                            15 to 22 cm, so the third line of the
#                            instruction is load-bearing, not decoration.

# --- measured blind spot, in degrees of visual angle ---------------------
BS_INNER = 13.6            # nearest edge to fixation
BS_OUTER = 21.2            # furthest edge
BS_WIDE = 7.6              # horizontal extent
BS_TALL = 8.3
BS_BELOW = 1.5             # centre sits this far below the horizontal

TEXT = ["close your right eye",
        "stare at the +",
        "bring the phone close"]


def deg_per_cell(g):
    """One character cell, in degrees of visual angle, under the model."""
    full = 2.0 * math.degrees(math.atan(PHONE_W_CM / 2.0 / DIST_CM))
    return full / g.cols


def layout(g):
    """Where the cross sits, where the dot walks, and how big it is."""
    fix_c = g.cols - 5
    fix_r = (g.safe_top + g.safe_bot) // 2
    dpc = deg_per_cell(g)
    dot_r = fix_r + int(round(BS_BELOW / dpc))   # 1.5 deg low, like the hole
    c0 = R_DOT + 0.5                             # far end of the walk
    c1 = round(g.cols * 0.56, 1)                 # near end
    return fix_c, fix_r, dot_r, c0, c1, dpc


def dot_col(f, c0, c1):
    """Dot column at frame f. Constant speed, so WHERE it vanishes is
    proportional to WHEN -- which is what makes the viewer's answer mean
    something."""
    if f < HEAD:
        return None
    k, u = divmod(f - HEAD, SWEEP)
    u /= float(SWEEP)
    return c0 + (c1 - c0) * u if k % 2 == 0 else c1 + (c0 - c1) * u


def bands(g):
    """Row indices carrying pattern. Strictly periodic -- that is what lets
    the surround close the gap without inventing anything."""
    return [r for r in range(1, g.rows - 1) if r % BAND_STEP < BAND_ON]


def cross(fr, g, c, r):
    """A crosshair in a small clearing. The clearing is not decoration: a
    bone glyph laid over a lilac bar is hard to hold an eye on, and the
    whole demonstration depends on the eye not wandering. It sits far from
    the dot's track, so it cannot help the surround fill anything in."""
    fr.ctx.set_source_rgb(*BG)
    fr.ctx.rectangle((c - 4) * g.cell, (r - 4) * g.cell,
                     g.w_px - (c - 4) * g.cell, 9 * g.cell)
    fr.ctx.fill()
    fr.put_run(c - 3, r, "-------", BONE)
    for d in (-3, -2, -1, 1, 2, 3):
        fr.put(c, r + d, "|", BONE)
    fr.put(c, r, "+", BONE)


def paint(fr, g, f, lay):
    fix_c, fix_r, dot_r, c0, c1, _ = lay

    # the surround wipes on left to right in the last 0.6 s of instruction
    if f >= HEAD - DRAWIN:
        w = min(1.0, (f - (HEAD - DRAWIN) + 1) / float(DRAWIN))
        n = int(round(g.cols * w))
        if n:
            for r in bands(g):
                fr.put_run(0, r, BAND_CH * n, RULE)

    # The dot: a filled body with characters ON it, not characters floating
    # on the ground. Every glyph leaks background through its holes, and a
    # leaky dot is a dot that half-vanishes on its own -- which would hand
    # the viewer the answer for the wrong reason.
    dc = dot_col(f, c0, c1)
    if dc is not None:
        fr.ctx.set_source_rgb(*DOT)
        fr.ctx.arc((dc + 0.5) * g.cell, (dot_r + 0.5) * g.cell,
                   R_DOT * g.cell, 0.0, 2.0 * math.pi)
        fr.ctx.fill()
        lo, hi = int(math.floor(dc - R_DOT)), int(math.ceil(dc + R_DOT))
        for rr in range(dot_r - int(R_DOT) - 1, dot_r + int(R_DOT) + 2):
            run, start = "", None
            for cc in range(lo, hi + 1):
                if (cc - dc) ** 2 + (rr - dot_r) ** 2 <= (R_DOT - 0.4) ** 2:
                    if start is None:
                        start = cc
                    run += DOT_CH
                elif run:
                    fr.put_run(start, rr, run, DOT_TEX)
                    run, start = "", None
            if run:
                fr.put_run(start, rr, run, DOT_TEX)

    cross(fr, g, fix_c, fix_r)

    # instruction, then gone. no captions ride along with the demonstration.
    if f < HEAD:
        a = 1.0 if f < HEAD - DRAWIN else 1.0 - (f - (HEAD - DRAWIN)) / float(DRAWIN)
        for i, line in enumerate(TEXT):
            r = g.safe_top + 8 + i * 4
            fr.put_run((g.cols - len(line)) // 2, r, line, BONE, alpha=a)


def check(g, lay):
    """Assert the thing the video claims. If the track cannot reach the
    blind spot, the demonstration is a lie and this refuses to render."""
    fix_c, fix_r, dot_r, c0, c1, dpc = lay
    print(g)
    print("model: %.1f cm screen at %.0f cm -> %.3f deg per cell"
          % (PHONE_W_CM, DIST_CM, dpc))

    far = (fix_c - c0) * dpc          # eccentricity at the far end of walk
    near = (fix_c - c1) * dpc         # ...and at the near end
    print("dot walks %.1f deg -> %.1f deg from fixation" % (far, near))
    assert far > BS_OUTER, "walk never reaches the outer edge of the hole"
    assert near < BS_INNER, "walk never leaves the hole on the inside"

    dot_deg = 2 * R_DOT * dpc
    print("dot is %.1f deg across; hole is %.1f x %.1f deg"
          % (dot_deg, BS_WIDE, BS_TALL))
    assert dot_deg < BS_WIDE * 0.6, "dot too big to disappear whole"

    below = (dot_r - fix_r) * dpc
    print("dot rides %.1f deg below the cross (hole sits %.1f deg low)"
          % (below, BS_BELOW))
    assert abs(below - BS_BELOW) < 0.5, "dot not on the hole's meridian"

    # where it vanishes, as a fraction of the walk, at three distances
    for d in (15.0, 18.0, 22.0):
        full = 2.0 * math.degrees(math.atan(PHONE_W_CM / 2.0 / d))
        per = full / g.cols
        col = fix_c - 15.0 / per      # column at 15 deg eccentricity
        print("  at %.0f cm the dot should wink out %3.0f%% along the walk"
              % (d, 100.0 * (col - c0) / (c1 - c0)))

    # The surround is strictly periodic and full width, so whatever the eye
    # puts in the hole is already true everywhere else on that row.
    rs = bands(g)
    per = sorted(set(b - a for a, b in zip(rs, rs[1:])))
    assert per == [1, BAND_OFF + 1], "surround is not periodic: %s" % per
    assert R_DOT * 2 > BAND_STEP, "dot smaller than one period of surround"
    print("%d patterned rows, %d on / %d off, identical across all %d columns"
          % (len(rs), BAND_ON, BAND_OFF, g.cols))

    # the dot never touches the cross, or the fixation target moves
    gap = min(abs(dot_col(f, c0, c1) - fix_c) for f in range(HEAD, TOTAL))
    assert gap > R_DOT + 4, "dot crowds the fixation cross"

    # the walk closes, so the loop does
    assert abs(dot_col(HEAD, c0, c1) - c0) < 1e-9
    assert abs(dot_col(TOTAL - 1, c0, c1) - c0) < 0.5, "walk does not close"
    print("walk closes: %.2f s, %d frames" % (TOTAL / float(FPS), TOTAL))


def main():
    g = Grid(font_size=28)
    lay = layout(g)
    check(g, lay)
    with Encoder(OUT, g, fps=FPS) as enc:
        for f in range(TOTAL):
            fr = Frame(g, BG)
            paint(fr, g, f, lay)
            enc.write(fr)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
