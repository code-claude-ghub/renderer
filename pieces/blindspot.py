#!/usr/bin/env python3
"""
blindspot.py -- a vertical line with a gap in it, and a fixation cross.

WHY THIS EXISTS

Every piece this channel has made in the last month has been verified in the
pixels: pull the finished bytes back out of the mp4, measure the thing the
title claims, refuse to publish until the number comes out right.

This one cannot be checked that way. The event the video is about does not
happen on the screen. It happens about 15 degrees off to one side of a
person's fovea, in the patch of retina where the optic nerve leaves and there
are no photoreceptors at all. I do not have a retina. So everything below
verifies the GEOMETRY -- that the gap is where the standard numbers say a
human blind spot is, at a stated viewing distance -- and then stops, because
the last step belongs to the viewer.

THE SETUP

  close the RIGHT eye. look at the + with the LEFT eye.

The optic disc sits nasal to the fovea on the retina. The retinal image is
inverted, so the corresponding hole in the VISUAL FIELD is temporal -- for the
left eye, off to the left. Hence: cross on the right, line on the left.
It is also slightly superior on the retina, so the hole is slightly BELOW the
horizontal through fixation. Hence: the gap sits a little under the cross.

Numbers built to (Wikipedia, "Blind spot (vision)", sourced there to
MIL-STD-1472F, 1999): 12-15 degrees temporal, 1.5 degrees below the
horizontal, 5.5 degrees wide, 7.5 degrees high. That is a weaker source than
I would like and the description says so out loud.

THE MOTION

The gap does not move. It GROWS, symmetrically, from 1.6 degrees to 9.4 and
back, on a cosine, once every 12 seconds. While both its edges are inside the
blind spot there is nothing to see -- the line looks whole and nothing appears
to be happening. When the gap outgrows the hole it snaps back into existence.
The moment it snaps is a measurement of the viewer's own blind spot, and it is
a measurement I cannot make from here.

Published file is two periods, the second byte-identical to the first, so one
seamless repeat is guaranteed inside the file rather than trusted to a player.

  python3 pieces/blindspot.py --check
  python3 pieces/blindspot.py --stills /tmp/bs
  python3 pieces/blindspot.py --out out/blindspot.mp4

numpy + pycairo + ffmpeg.
"""

import argparse
import math
import subprocess

import cairo
import numpy as np

# ------------------------------------------------------------------- canvas

W, H = 1080, 1920
FPS = 30

# ------------------------------------------------------- the whole geometry
#
# Everything hangs off one decision: the cross and the line are 13.5 degrees
# apart, the middle of the 12-15 band. Every other angle in the piece is
# derived from that, so there is exactly one number to be wrong about.

X_CROSS = 990.0
X_LINE = 100.0                            # far enough in not to read as bezel
SEP_PX = X_CROSS - X_LINE                 # 890 px
SEP_DEG = 13.5
PPD = SEP_PX / SEP_DEG                    # 68.148 px per degree

Y_CROSS = 640.0
BELOW_DEG = 1.5                           # blind spot sits under the meridian
Y_GAP = Y_CROSS + BELOW_DEG * PPD         # 742.2

BS_W_DEG = 5.5                            # published blind spot extent
BS_H_DEG = 7.5

GAP_MIN_DEG = 1.6                         # a real gap, and a small one
GAP_MAX_DEG = 9.4                         # wider than any standard blind spot

T_PER = 12.0
N_PER = 2
NFP = int(round(T_PER * FPS))             # 360 frames in a period
NF = NFP * N_PER                          # 720
DUR = NF / FPS                            # 24.0 s

# ------------------------------------------------------------------- ink
#
# Paper, not the usual black field. Filling-in demos have been done on white
# paper with a black pen for a century and that is the version with the most
# reported successes, so function beats the house palette here. A warm grey
# rather than a full white so it is not painful on a phone at night.

BG = (0.815, 0.805, 0.785)
INK = (0.100, 0.100, 0.110)
TXT = (0.420, 0.420, 0.432)

LINE_W = 15.0
CROSS_ARM = 46.0
CROSS_W = 13.0

FONT = "DejaVu Sans"
TXT_SIZE = 54.0
TXT_LINES = ("close your right eye", "look at the +")
TXT_BASE = (250.0, 326.0)

TWO_PI = 2.0 * math.pi


# ------------------------------------------------------------------ motion

def gap_deg(tv):
    """Symmetric cosine breathe. gap_deg(0) == gap_deg(T_PER) exactly."""
    u = (1.0 - math.cos(TWO_PI * tv / T_PER)) / 2.0
    return GAP_MIN_DEG + (GAP_MAX_DEG - GAP_MIN_DEG) * u


def gap_px(tv):
    return gap_deg(tv) * PPD


# ------------------------------------------------------------------- draw

def draw(i):
    tv = (i % NFP) / FPS
    g = gap_px(tv)
    top = Y_GAP - g / 2.0
    bot = Y_GAP + g / 2.0

    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    cr = cairo.Context(surf)
    cr.set_antialias(cairo.ANTIALIAS_BEST)

    cr.set_source_rgb(*BG)
    cr.rectangle(0, 0, W, H)
    cr.fill()

    cr.set_source_rgb(*INK)
    # the line, in two pieces, with nothing between them
    cr.rectangle(X_LINE - LINE_W / 2.0, 0.0, LINE_W, top)
    cr.fill()
    cr.rectangle(X_LINE - LINE_W / 2.0, bot, LINE_W, H - bot)
    cr.fill()

    # the cross
    cr.rectangle(X_CROSS - CROSS_ARM, Y_CROSS - CROSS_W / 2.0,
                 2.0 * CROSS_ARM, CROSS_W)
    cr.fill()
    cr.rectangle(X_CROSS - CROSS_W / 2.0, Y_CROSS - CROSS_ARM,
                 CROSS_W, 2.0 * CROSS_ARM)
    cr.fill()

    cr.select_font_face(FONT, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    cr.set_font_size(TXT_SIZE)
    cr.set_source_rgb(*TXT)
    for s, yb in zip(TXT_LINES, TXT_BASE):
        e = cr.text_extents(s)
        cr.move_to(W / 2.0 - e.width / 2.0 - e.x_bearing, yb)
        cr.show_text(s)

    surf.flush()
    buf = np.ndarray(shape=(H, W, 4), dtype=np.uint8, buffer=surf.get_data())
    return np.ascontiguousarray(buf[:, :, [2, 1, 0]])


# ------------------------------------------------------------------ checks

OK = True


def t(cond, msg):
    global OK
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        OK = False


def measure_gap(f8):
    """Read the gap height straight off the finished bytes, in pixels.

    Walks the line's own centre column and counts rows that are not ink.
    Anti-aliased edge rows land between, so this is the half-intensity count
    and is what the analytic value should be compared against.
    """
    col = f8[:, int(round(X_LINE)), :].astype(np.float64).mean(axis=1)
    lo, hi = INK[0] * 255.0, BG[0] * 255.0
    mid = (lo + hi) / 2.0
    return float((col > mid).sum())


def runs(f8):
    """Number of separate breaks in the line's centre column."""
    col = f8[:, int(round(X_LINE)), :].astype(np.float64).mean(axis=1)
    mid = (INK[0] * 255.0 + BG[0] * 255.0) / 2.0
    b = (col > mid).astype(np.int8)
    return int((np.diff(b) == 1).sum())


def run_checks():
    print("\n-- geometry")
    t(abs(SEP_PX / PPD - SEP_DEG) < 1e-9,
      f"cross and line sit {SEP_DEG} deg apart ({SEP_PX:.0f} px at "
      f"{PPD:.3f} px/deg)")
    t(12.0 <= SEP_DEG <= 15.0,
      f"{SEP_DEG} deg is inside the published 12-15 deg temporal band")
    t(abs((Y_GAP - Y_CROSS) / PPD - BELOW_DEG) < 1e-9,
      f"gap centre is {BELOW_DEG} deg below the cross "
      f"({Y_GAP - Y_CROSS:.1f} px), the direction the disc actually lies")
    t(X_LINE < X_CROSS,
      "the gap is to the LEFT of the cross, which is the temporal field of "
      "the LEFT eye -- the eye the video asks you to keep open")

    print("\n-- the gap, against the published blind spot")
    t(GAP_MIN_DEG < BS_W_DEG,
      f"smallest gap {GAP_MIN_DEG} deg fits inside a {BS_W_DEG} deg wide hole")
    t(GAP_MAX_DEG > BS_H_DEG,
      f"largest gap {GAP_MAX_DEG} deg is bigger than the {BS_H_DEG} deg tall "
      f"hole, so it has to become visible to everybody")
    t(GAP_MIN_DEG * PPD > 90.0,
      f"smallest gap is {GAP_MIN_DEG * PPD:.0f} px -- an unmistakable break "
      f"when you look straight at it")

    print("\n-- working distance (assumes the video displays 6.8 cm wide)")
    wcm = 6.8
    scm = SEP_PX / W * wcm
    near = scm / math.tan(math.radians(SEP_DEG + BS_W_DEG / 2.0))
    far = scm / math.tan(math.radians(SEP_DEG - BS_W_DEG / 2.0))
    mid = scm / math.tan(math.radians(SEP_DEG))
    print(f"       separation on glass {scm:.2f} cm -> centred at {mid:.1f} cm,"
          f" works {near:.1f}-{far:.1f} cm")
    t(15.0 < mid < 35.0,
      f"the centred distance {mid:.1f} cm is a distance a person actually "
      f"holds a phone at")
    t(far - near > 8.0,
      f"the working window is {far - near:.1f} cm wide, so this does not need "
      f"a ruler")

    print("\n-- motion")
    t(abs(gap_deg(0.0) - gap_deg(T_PER)) < 1e-12,
      f"gap at t=0 equals gap at t={T_PER} exactly "
      f"({abs(gap_deg(0.0) - gap_deg(T_PER)):.1e} deg)")
    half = [gap_deg(i / FPS) for i in range(NFP // 2 + 1)]
    t(all(b >= a - 1e-12 for a, b in zip(half, half[1:])),
      "the gap only ever opens for the first six seconds")
    t(abs(gap_deg(T_PER / 2.0) - GAP_MAX_DEG) < 1e-12,
      f"it is widest exactly halfway ({gap_deg(T_PER / 2.0):.3f} deg)")
    sym = max(abs(gap_deg(tv) - gap_deg(T_PER - tv))
              for tv in np.linspace(0, T_PER, 241))
    t(sym < 1e-12, f"opening and closing are mirror images ({sym:.1e} deg)")

    print("\n-- the finished bytes")
    f0 = draw(0)
    fmid = draw(NFP // 2)
    fend = draw(NFP)
    t(np.array_equal(f0, fend),
      "frame 360 is byte-identical to frame 0 -- the period closes exactly")
    t(np.array_equal(draw(7), draw(NFP + 7)),
      "the second period is byte-identical to the first")
    t(runs(f0) == 1 and runs(fmid) == 1,
      f"there is exactly one break in the line, at both extremes "
      f"({runs(f0)} and {runs(fmid)})")
    for i, lab in ((0, "min"), (NFP // 2, "max"), (NFP // 4, "quarter")):
        m = measure_gap(draw(i))
        a = gap_px(i / FPS)
        t(abs(m - a) < 2.0,
          f"{lab} gap measured off the pixels is {m:.0f} px against "
          f"{a:.1f} px asked for")

    print("\n-- nothing else in the frame moves")
    a, b = draw(NFP // 4), draw(NFP // 4 + 1)
    d = np.argwhere((a != b).any(axis=2))
    if len(d):
        r0, r1 = d[:, 0].min(), d[:, 0].max()
        c0, c1 = d[:, 1].min(), d[:, 1].max()
        t(c0 >= X_LINE - LINE_W and c1 <= X_LINE + LINE_W,
          f"between two frames only columns {c0}-{c1} change, and the line is "
          f"{X_LINE - LINE_W / 2:.0f}-{X_LINE + LINE_W / 2:.0f}")
        t(r0 > 380 and r1 < 1120,
          f"only rows {r0}-{r1} change -- the text, the cross and both ends "
          f"of the line are frozen")
    else:
        t(False, "consecutive frames are identical, which means nothing moves")

    print("\n-- legible in a feed")
    from PIL import Image
    small = np.asarray(Image.fromarray(draw(NFP // 2)).resize(
        (360, 640), Image.LANCZOS)).astype(np.float64)
    lum = small.mean(axis=2)
    t(lum[:, int(round(X_LINE / 3.0))].min() < 120.0,
      f"at 360 px wide the line is still ink "
      f"(darkest {lum[:, int(round(X_LINE / 3.0))].min():.0f} of 255)")
    band = lum[int(TXT_BASE[0] / 3.0) - 14:int(TXT_BASE[1] / 3.0) + 4, :]
    t(band.max() - band.min() > 55.0,
      f"at 360 px wide the instruction still has "
      f"{band.max() - band.min():.0f} levels of contrast")
    cross_band = lum[int(Y_CROSS / 3.0) - 2:int(Y_CROSS / 3.0) + 3,
                     int((X_CROSS - CROSS_ARM) / 3.0):
                     int((X_CROSS + CROSS_ARM) / 3.0)]
    t(cross_band.min() < 120.0,
      f"at 360 px wide the + is still findable "
      f"(darkest {cross_band.min():.0f} of 255)")

    print("\n-- what I am NOT checking")
    print("       whether the gap disappears. that happens in a retina and I")
    print("       do not have one. the geometry is verified. the effect is")
    print("       the viewer's to report.")

    print("\n" + ("ALL CHECKS PASSED" if OK else "SOMETHING FAILED"))
    return 0 if OK else 1


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--stills')
    ap.add_argument('--out')
    args = ap.parse_args()

    print(f"blindspot: {DUR:.1f} s, {NF} frames, {N_PER} x {T_PER} s period")
    print(f"  {PPD:.2f} px/deg, gap {GAP_MIN_DEG}-{GAP_MAX_DEG} deg "
          f"({GAP_MIN_DEG * PPD:.0f}-{GAP_MAX_DEG * PPD:.0f} px)")

    if args.check:
        return run_checks()

    if args.stills:
        from PIL import Image
        for i in (0, NFP // 4, NFP // 2, 3 * NFP // 4, NFP - 1):
            Image.fromarray(draw(i)).save(f"{args.stills}_{i:04d}.png")
            print(f"  still {i}")
        return 0

    if not args.out:
        print("nothing to do -- pass --check, --stills or --out")
        return 1

    cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{W}x{H}', '-r', str(FPS), '-i', 'pipe:0',
           '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '17',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', args.out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for i in range(NF):
        p.stdin.write(draw(i).tobytes())
        if i % 60 == 0:
            print(f"  frame {i}/{NF}", flush=True)
    p.stdin.close()
    p.wait()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
