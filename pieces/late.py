#!/usr/bin/env python3
"""
LATE -- thirty-five lights pulsing in unison, and one of them keeps a
slightly different time.

The many run at period T/12. The odd one runs at T/11, so over the piece's
whole duration it loses EXACTLY one cycle:

    t = 0      every light together
    t = T/2    the odd one is in antiphase -- the only dark light in a bright
               field, which is the moment the piece hands you the answer
    t = T      back together, so frame N wraps onto frame 0

The loop is exact and the loop is the idea. It drifts all the way out and all
the way back, forever, and nothing on screen explains that.

Not ASCII. First piece after the glyph-grid constraint was lifted, and a
smoothly pulsing brightness is precisely the thing a ten-level ramp cannot
hold -- it bands, which is the artifact that spoiled THE DOOR. So this is
drawn as real light: analytic antialiased discs plus a soft glow, composited
in float, encoded straight to x264.

    python3 scripts/late.py --check      verify the structure, render nothing
    python3 scripts/late.py --stills     full-resolution PNGs at key moments
    python3 scripts/late.py --out x.mp4  render
"""

import argparse
import math
import os
import subprocess
import sys

import numpy as np

# ---------------------------------------------------------------- geometry
W, H = 1080, 1920
FPS = 30

COLS, ROWS = 5, 7
N_DOT = COLS * ROWS               # 35
STEP = 216                        # square lattice, px
CX, CY = W / 2.0, H / 2.0

# The odd one. Off-centre and off-axis so the eye does not land on it first,
# and low enough that it is clear of the row your thumb covers.
ODD_COL, ODD_ROW = 1, 4
ODD = ODD_ROW * COLS + ODD_COL    # 21

# ------------------------------------------------------------------ timing
CYC_MANY = 12                     # cycles the many complete over the piece
CYC_ODD = 11                      # the odd one completes one fewer
N_FRAME = 252                     # 8.4 s at 30 fps
T_TOTAL = N_FRAME / FPS

# ------------------------------------------------------------------- light
R_CORE = 46.0                     # disc radius, px
R_GLOW = 132.0                    # halo reach, px
GLOW_AMP = 0.55
SHARP = 1.45                      # >1 spends longer dark, snappier peaks
FLOOR = 0.14                      # never fully out, so the lattice holds

BG = np.array([0.030, 0.034, 0.045])      # dark, faintly blue
COL_CORE = np.array([1.00, 0.72, 0.36])   # warm amber
COL_GLOW = np.array([1.00, 0.42, 0.13])   # redder halo

TILE = 300                        # half-width of a dot's stamp


def centres():
    """Dot centres, row-major."""
    out = []
    for r in range(ROWS):
        for c in range(COLS):
            out.append((CX + (c - (COLS - 1) / 2.0) * STEP,
                        CY + (r - (ROWS - 1) / 2.0) * STEP))
    return out


CENTRES = centres()


def profile():
    """The spatial stamp of one light: core coverage and glow, computed once.

    Every dot is the same shape, so this is built a single time and only
    scaled per frame. That is what keeps 35 x 252 composites cheap.
    """
    ax = np.arange(-TILE, TILE + 1, dtype=np.float64)
    dx, dy = np.meshgrid(ax, ax, indexing='xy')
    d = np.sqrt(dx * dx + dy * dy)
    # analytic 1px antialiased disc -- no sampling, so no stair-stepping
    core = np.clip(0.5 + (R_CORE - d), 0.0, 1.0)
    glow = np.exp(-(d / (R_GLOW * 0.42)) ** 2) * GLOW_AMP
    glow *= (1.0 - core)          # halo lives outside the disc, not under it
    return core, glow


CORE, GLOW = profile()


def phases(f):
    """Phase of every light at frame f, in cycles."""
    u = f / float(N_FRAME)                    # 0..1 over the whole piece
    ph = np.full(N_DOT, u * CYC_MANY)
    ph[ODD] = u * CYC_ODD
    return ph


def brightness(ph):
    """Phase (cycles) -> 0..1 brightness. Phase 0 is bright."""
    p = (0.5 + 0.5 * np.cos(2.0 * math.pi * ph)) ** SHARP
    return FLOOR + (1.0 - FLOOR) * p


def draw(f):
    """Render frame f as a float HxWx3 in scene-linear-ish units."""
    img = np.tile(BG, (H, W, 1))
    b = brightness(phases(f))

    for i, (cx, cy) in enumerate(CENTRES):
        ix, iy = int(round(cx)), int(round(cy))
        x0, x1 = ix - TILE, ix + TILE + 1
        y0, y1 = iy - TILE, iy + TILE + 1
        # clip the stamp against the canvas
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(W, x1), min(H, y1)
        if sx0 >= sx1 or sy0 >= sy1:
            continue
        tx0, ty0 = sx0 - x0, sy0 - y0
        tx1, ty1 = tx0 + (sx1 - sx0), ty0 + (sy1 - sy0)

        c = CORE[ty0:ty1, tx0:tx1, None]
        g = GLOW[ty0:ty1, tx0:tx1, None]
        img[sy0:sy1, sx0:sx1] += b[i] * (c * COL_CORE + g * COL_GLOW)

    return img


def to_bytes(img):
    """Float image -> rgb24 bytes, with a gentle shoulder instead of a clip."""
    x = np.maximum(img, 0.0)
    x = x / (1.0 + x * 0.22)          # soft highlight rolloff, no hard clip
    x = np.clip(x * 1.18, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8).tobytes()


# ------------------------------------------------------------------ checks
def check():
    ok = True

    def say(good, label, detail=''):
        nonlocal ok
        ok = ok and good
        print(('  ok   ' if good else '  FAIL ') + label + (' -- ' + detail if detail else ''))

    print('LATE -- structure')
    print('  %d lights, %d frames, %.2f s at %d fps' % (N_DOT, N_FRAME, T_TOTAL, FPS))
    print('  many: %d cycles (period %.4f s) | odd: %d cycles (period %.4f s)'
          % (CYC_MANY, T_TOTAL / CYC_MANY, CYC_ODD, T_TOTAL / CYC_ODD))
    print('  pulse rate %.2f Hz -- WCAG general flash threshold is 3 Hz' % (CYC_MANY / T_TOTAL))
    print()

    # 1. the loop is exact by construction: phase at the wrap frame == phase at 0
    p0, pN = phases(0), phases(N_FRAME)
    say(np.allclose(p0 % 1.0, pN % 1.0, atol=1e-12),
        'frame N wraps exactly onto frame 0',
        'max phase error %.2e' % np.abs((pN - p0) % 1.0).max())

    # 2. both cycle counts whole, differing by one -- that is what makes it loop
    say(CYC_MANY - CYC_ODD == 1 and isinstance(CYC_MANY, int) and isinstance(CYC_ODD, int),
        'the odd light loses exactly one cycle over the piece')

    # 3. at t=0 every light is identical
    b0 = brightness(phases(0))
    say(float(b0.max() - b0.min()) < 1e-12,
        'at t=0 nothing gives it away', 'spread %.2e' % float(b0.max() - b0.min()))

    # 4. the midpoint is true antiphase -- the whole reveal depends on this
    mid = N_FRAME // 2
    bm = brightness(phases(mid))
    others = np.delete(bm, ODD)
    say(abs(bm[ODD] - FLOOR) < 1e-9 and others.min() > 0.999,
        'at the midpoint the odd light is at its floor and the rest at full',
        'odd %.3f vs others %.3f, contrast %.2fx' % (bm[ODD], others.min(), others.min() / bm[ODD]))

    # 5. every other light stays in perfect unison, always
    worst = 0.0
    for f in range(N_FRAME):
        b = brightness(phases(f))
        o = np.delete(b, ODD)
        worst = max(worst, float(o.max() - o.min()))
    say(worst < 1e-12, 'the other 34 never break unison', 'worst spread %.2e' % worst)

    # 6. The instantaneous gap OSCILLATES -- two things blinking out of step
    #    cross the same brightness twice a cycle no matter how far apart they
    #    are. So the honest measure is the envelope: the worst gap inside each
    #    pulse period. That is what the eye integrates.
    def gap(f):
        b = brightness(phases(f))
        return abs(float(b[ODD] - np.delete(b, ODD)[0]))

    per = N_FRAME / CYC_MANY                      # frames in one pulse
    env = [max(gap(f) for f in range(int(k * per), min(N_FRAME, int((k + 1) * per))))
           for k in range(CYC_MANY)]

    # A Short gets about a second to justify itself. Something must already be
    # off -- but not so off that there is nothing left to look for.
    say(env[0] > 0.03, 'the first pulse already shows something is off',
        'envelope %.3f' % env[0])
    say(max(env) > 0.80, 'and it grows to unmissable', 'peak envelope %.3f' % max(env))

    # This is NOT a spot-the-difference. The other 34 hold perfect unison, so
    # the odd one is the unique brightness in the field at almost every frame
    # and you can always re-find it. The piece is the drifting, not the hunt.
    findable = sum(1 for f in range(N_FRAME) if gap(f) > 0.02) / float(N_FRAME)
    say(findable > 0.80, 'once found it stays findable -- this is not a puzzle',
        'distinguishable in %.0f%% of frames' % (100 * findable))

    # 7. the real structural claim: phase offset grows to exactly half a cycle
    def off(f):
        p = phases(f)
        d = abs(p[ODD] - p[0]) % 1.0
        return min(d, 1.0 - d)                    # distance on the circle

    half = [off(f) for f in range(mid + 1)]
    rising = sum(1 for i in range(1, len(half)) if half[i] >= half[i - 1] - 1e-12)
    say(rising == len(half) - 1, 'phase offset only ever grows in the first half',
        '%d/%d steps' % (rising, len(half) - 1))
    say(abs(off(mid) - 0.5) < 1e-12, 'and reaches exactly half a cycle at the midpoint',
        'offset %.6f' % off(mid))

    # 8. the envelope grows through the first half and shrinks through the
    #    second -- out and back, which is the shape of the whole piece
    up = all(env[i] >= env[i - 1] - 1e-9 for i in range(1, CYC_MANY // 2))
    down = all(env[i] <= env[i - 1] + 1e-9 for i in range(CYC_MANY // 2 + 1, CYC_MANY))
    say(up and down, 'the envelope swells and then settles back',
        ' '.join('%.2f' % e for e in env))

    # 8. pixels: background is not crushed, highlights are not clipped flat
    img = draw(mid)
    px = np.frombuffer(to_bytes(img), np.uint8).reshape(H, W, 3)
    bgpx = px[4, 4]
    say(bool(bgpx.max() > 6), 'background is not crushed to black', 'bg rgb %s' % (tuple(int(v) for v in bgpx),))
    frac_max = float((px.max(2) == 255).mean())
    say(frac_max < 0.06, 'highlights are not a flat clipped plateau',
        '%.2f%% of pixels at 255' % (100 * frac_max))

    # 9. tonal range -- the artifact that spoiled THE DOOR was flat fields
    vals, counts = np.unique(px[..., 0], return_counts=True)
    top = counts.max() / counts.sum()
    say(len(vals) > 60 and top < 0.80,
        'the frame is a gradient, not a few flat tones',
        '%d distinct levels, commonest holds %.0f%%' % (len(vals), 100 * top))

    print()
    print('ALL CHECKS PASS' if ok else 'SOMETHING IS WRONG')
    return 0 if ok else 1


def stills(prefix):
    """Full-resolution PNGs at the moments that matter. Look at these."""
    from PIL import Image
    marks = [(0, 'together'), (N_FRAME // 6, 'drifting'),
             (N_FRAME // 3, 'wide'), (N_FRAME // 2, 'antiphase'),
             (2 * N_FRAME // 3, 'closing'), (N_FRAME - 1, 'home')]
    for f, name in marks:
        px = np.frombuffer(to_bytes(draw(f)), np.uint8).reshape(H, W, 3)
        p = '%s_%02d_%s.png' % (prefix, f, name)
        Image.fromarray(px).save(p)
        print(p)


def render(path):
    cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', '%dx%d' % (W, H), '-r', str(FPS), '-i', 'pipe:0',
           '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '17',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in range(N_FRAME):
        proc.stdin.write(to_bytes(draw(f)))
        if f % 30 == 0:
            print('  frame %d/%d' % (f, N_FRAME), flush=True)
    proc.stdin.close()
    proc.wait()
    print('wrote %s (%.1f KB)' % (path, os.path.getsize(path) / 1024.0))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--stills')
    ap.add_argument('--out')
    a = ap.parse_args()

    if a.check:
        sys.exit(check())
    if a.stills:
        stills(a.stills)
    if a.out:
        render(a.out)
    if not (a.check or a.stills or a.out):
        ap.print_help()
