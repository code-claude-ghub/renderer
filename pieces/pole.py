#!/usr/bin/env python3
"""
POLE -- two barber poles. One is being turned. One is being pushed straight up.

THE WHOLE PIECE IN ONE LINE. A helix is a screw, so turning it and sliding it
along its own axis are THE SAME MOTION. Not similar. The same. Rotate a helical
stripe by an angle a and you get exactly the stripe you would have got by
sliding it a distance PITCH * a / 2pi, because that is what "pitch" means.

So the left pole and the right pole are computed by two functions that share no
arithmetic:

    phase_spin(y, phi, t)  = y/PITCH - (phi + turn_angle(t)) / 2pi
    phase_slide(y, phi, t) = (y - slide_dist(t))/PITCH - phi / 2pi

and they emit the same bytes. Not nearly. The check renders both boxes for all
288 frames and asserts np.array_equal on the uint8 arrays. Zero pixels differ.

WHY THE BACKGROUND IS A FUNCTION OF ROW ONLY. It has to be. The poles are
composited with an alpha edge, so along the silhouette the finished pixel is a
blend of pole and background. Give the background any horizontal structure -- a
vignette, a gradient, a wall -- and the left and right boxes stop being
comparable at the one place the comparison is most interesting. The identity
claim in the title constrained the lighting design. That is the good kind of
constraint.

THEN THE DOT. If the two motions are the same picture, a fair question is
whether they are the same thing at all. They are not, and one spot of paint is
enough to prove it. At 4.8 s a dot appears near the bottom of each pole, in the
same place on both, with no fade -- they start together and then diverge. It is a point on the surface, so it does exactly what the
transform does to a point:

    left  -- it goes ROUND. Sideways, off behind the pole, back again, twice.
    right -- it goes UP. Straight up, the whole length of the glass, and off.

Neither of them moves the way the stripes appear to move. The stripes climb on
both poles for the entire 9.6 seconds, and there is nothing on either pole that
is climbing with them. That is the barber pole. It is not an illusion of the
eye -- the image really is translating upward. It is just not carrying anything.

THE NUMBERS ARE MINE AND THEY ARE A MODEL. Glass 9 cm across and 48 cm long,
stripe wrapping every 24 cm, one turn every 2.4 s. Real barber poles vary and I
have not measured one. What does not depend on the numbers: one full turn moves
the stripes up by exactly one pitch, always, and the only way to make them climb
faster is to turn it faster.

    python3 pieces/pole.py --check
    python3 pieces/pole.py --stills /tmp/pole
    python3 pieces/pole.py --out pole.mp4

numpy + pycairo + ffmpeg.
"""

import argparse
import math
import subprocess

import cairo
import numpy as np

TWO_PI = 2.0 * math.pi

# ------------------------------------------------------------------- picture

W, H = 1080, 1920
FPS = 30
SSX = 3                       # supersample in x only -- see AA note below

SAFE_TOP, SAFE_BOT = 192, 1656

R_M = 0.045                   # glass radius, metres
R_G = 108.0                   # glass radius, pixels
SCALE = R_G / R_M             # 2400 px per metre, exactly
R_C = 118.0                   # cap radius, pixels

CAP_H = 64                    # cap length, pixels
BW, BH = 252, 1280            # the box one whole pole is drawn into
Y0 = 336                      # top of the assembly
GLASS_T = Y0 + CAP_H          # 400
GLASS_B = Y0 + BH - CAP_H     # 1552
GLASS_M = (GLASS_B - GLASS_T) / SCALE   # 0.48 m of visible glass

CX_L, CX_R = 300, 780         # pole centres
X_L = CX_L - BW // 2          # box left edges: 174 and 654
X_R = CX_R - BW // 2

PITCH = 0.24                  # metres the stripe rises in one full wrap
T_TURN = 2.4                  # seconds per revolution
N_TURN = 4
DUR = N_TURN * T_TURN         # 9.6 s
NF = int(round(DUR * FPS)) + 1   # 289: the LAST frame lands exactly on 4 turns

DOT_T = 2.0 * T_TURN          # 4.8 s -- two clean turns first, then two with it
R_DOT = 0.015                 # metres
Y_DOT = GLASS_M - 0.048       # painted just clear of the bottom cap

# cairo and the compositor both want 0..1 floats. 0..255 clamps every channel
# to white and no geometry check will ever notice. (RENDERER.md trap 55.)
RED = (0.762, 0.135, 0.150)
WHT = (0.955, 0.945, 0.930)
CAP_RGB = (0.560, 0.578, 0.612)
DOT_RGB = (0.055, 0.065, 0.150)
LABEL_RGB = (0.700, 0.720, 0.765)

BG_TOP = (0.052, 0.057, 0.068)
BG_BOT = (0.088, 0.094, 0.110)

# light, normalised. n has no y component on a cylinder about y, so only the
# x and z parts of these ever matter.
LX, LZ = -0.5187, 0.8080
HX, HZ = -0.2728, 0.9508      # halfway between the light and the eye
RX, RZ = 0.8000, 0.6000       # a weak rim from the other side, for the glass

AMB, DIF = 0.200, 0.860
SPEC, SPEC_K = 0.440, 48.0
RIM, RIM_K = 0.150, 6.0

FONT = "DejaVu Sans"
LABEL_SIZE = 60.0
LABEL_BASE = 302              # baseline, clear of the boxes at y=336
LABEL_L = "turning"
LABEL_R = "sliding up"


# --------------------------------------------------------------- the motions

def turn_angle(tv):
    """Radians the striped core has been rotated by. Negative so the stripes
    climb, which is the way every barber pole I have ever seen goes."""
    return -TWO_PI * tv / T_TURN


def slide_dist(tv):
    """Metres the striped core has been slid by, positive DOWN the screen.
    This is the whole identity: sliding by pitch * angle / 2pi is rotating by
    angle. Nothing else in this file depends on the two agreeing -- they are
    used by two separate functions that never compare notes."""
    return PITCH * turn_angle(tv) / TWO_PI


def phase_spin(y_m, phi, tv):
    """Stripe phase for a core that is being ROTATED. The stripe is fixed to
    the core, so the pattern seen at surface angle phi is the pattern that was
    painted at phi minus the rotation."""
    a = turn_angle(tv)
    return y_m[:, None] / PITCH - (phi[None, :] + a) / TWO_PI


def phase_slide(y_m, phi, tv):
    """Stripe phase for a core that is being SLID along its axis. The pattern
    seen at height y is the pattern that was painted at y minus the slide."""
    d = slide_dist(tv)
    return (y_m[:, None] - d) / PITCH - phi[None, :] / TWO_PI


def dot_spin(tv):
    """Where the painted dot is on the ROTATED pole: (surface angle, height).
    It goes round. Its height never changes."""
    return turn_angle(tv) - turn_angle(DOT_T), Y_DOT


def dot_slide(tv):
    """Where the painted dot is on the SLID pole. It goes up. Its angle never
    changes, so it stays dead centre and simply leaves the top."""
    return 0.0, Y_DOT + (slide_dist(tv) - slide_dist(DOT_T))


# ----------------------------------------------------------------- rendering

# per-sample geometry, computed once. x is supersampled SSX times; y is not,
# because every edge that runs across the frame is either pixel-aligned (the
# cap joints, at exactly y=400 and y=1552) or analytically antialiased (the
# stripes and the dot). Supersampling y would cost 3x for nothing.
_xs = (np.arange(BW * SSX, dtype=np.float64) + 0.5) / SSX
_ux = (_xs - BW / 2.0) / R_G
_uc = (_xs - BW / 2.0) / R_C
_ys = np.arange(BH, dtype=np.float64) + 0.5

IN_G = np.abs(_ux) < 1.0
IN_C = np.abs(_uc) < 1.0

_u = np.clip(_ux, -1.0, 1.0)
PHI = np.arcsin(_u)                       # surface angle of the front face
_nz = np.sqrt(np.maximum(1.0 - _u * _u, 0.0))

_uc2 = np.clip(_uc, -1.0, 1.0)
_nzc = np.sqrt(np.maximum(1.0 - _uc2 * _uc2, 0.0))


def _shade(nx, nz):
    ndl = np.maximum(nx * LX + nz * LZ, 0.0)
    ndh = np.maximum(nx * HX + nz * HZ, 0.0)
    ndr = np.maximum(nx * RX + nz * RZ, 0.0)
    return (AMB + DIF * ndl, SPEC * ndh ** SPEC_K, RIM * ndr ** RIM_K)


SH_G, SP_G, RM_G = _shade(_u, _nz)
SH_C, SP_C, RM_C = _shade(_uc2, _nzc)

# how fast the stripe phase moves per pixel, for the antialias width.
# d(phi)/dx blows up at the silhouette, so near the rim the width exceeds half
# a stripe and the blend saturates to flat -- which is exactly right, the
# stripes are genuinely unresolvable there.
_dphidx = (1.0 / R_G) / np.sqrt(np.maximum(1.0 - _ux * _ux, 1e-9))
AA_X = (_dphidx / TWO_PI) / SSX
AA_Y = 1.0 / (PITCH * SCALE)
AA = AA_X + AA_Y

Y_M = (_ys - CAP_H) / SCALE               # metres down from the top of glass

# the cap profile: a plain metal cylinder with two turned grooves in it.
_v_top = _ys[:CAP_H] / CAP_H
_v_bot = (BH - _ys[BH - CAP_H:]) / CAP_H


def _cap_ring(v):
    return (1.0
            - 0.42 * np.exp(-((v - 0.80) / 0.055) ** 2)
            - 0.30 * np.exp(-((v - 0.30) / 0.075) ** 2)
            - 0.35 * np.exp(-((v - 0.03) / 0.045) ** 2))


RING_T = _cap_ring(_v_top)
RING_B = _cap_ring(_v_bot)

BG_COL = (np.array(BG_TOP)[None, :]
          + (np.array(BG_BOT)[None, :] - np.array(BG_TOP)[None, :])
          * (np.arange(H)[:, None] / (H - 1.0)))


def pole_box(tv, mode, dot=True):
    """One pole, drawn into its own BH x BW box. -> (rgb, alpha) float arrays.

    `mode` is 'spin' or 'slide' and picks which of the two phase functions
    runs. Nothing else in this function branches on it.
    """
    ph = (phase_spin if mode == "spin" else phase_slide)(Y_M, PHI, tv)

    # triangle wave on the phase: 0 in the middle of a white band, 1 in the
    # middle of a red one, slope 2. So a phase half-width w is a half-width
    # 2w here.
    tri = 2.0 * np.abs(np.mod(ph + 0.25, 1.0) - 0.5)
    w = np.maximum(2.0 * AA[None, :], 1e-6)
    band = np.clip((tri - (0.5 - w)) / (2.0 * w), 0.0, 1.0)   # 0 white, 1 red

    rgb = (np.array(WHT)[None, None, :] * (1.0 - band)[:, :, None]
           + np.array(RED)[None, None, :] * band[:, :, None])

    if dot and tv >= DOT_T:
        phi_d, y_d = (dot_spin if mode == "spin" else dot_slide)(tv)
        dphi = np.mod(PHI - phi_d + math.pi, TWO_PI) - math.pi
        s = R_M * dphi[None, :]
        dy = (Y_M - y_d)[:, None]
        r = np.sqrt(s * s + dy * dy)
        aa = (R_M * _dphidx / SSX + 1.0 / SCALE)[None, :]
        m = np.clip((R_DOT + aa - r) / (2.0 * aa), 0.0, 1.0)
        rgb = rgb * (1.0 - m[:, :, None]) + np.array(DOT_RGB)[None, None, :] * m[:, :, None]

    rgb = rgb * SH_G[None, :, None] + (SP_G + RM_G)[None, :, None]

    cap = (np.array(CAP_RGB)[None, None, :] * SH_C[None, :, None]
           + (SP_C + 0.6 * RM_C)[None, :, None])
    top = cap[:1] * RING_T[:, None, None]
    bot = cap[:1] * RING_B[:, None, None]
    rgb = np.concatenate([np.broadcast_to(top, (CAP_H, BW * SSX, 3)),
                          rgb[CAP_H:BH - CAP_H],
                          np.broadcast_to(bot, (CAP_H, BW * SSX, 3))], axis=0)

    a = np.where(IN_G, 1.0, 0.0)[None, :].repeat(BH, 0)
    a[:CAP_H] = IN_C
    a[BH - CAP_H:] = IN_C

    rgb = rgb.reshape(BH, BW, SSX, 3).mean(axis=2)
    a = a.reshape(BH, BW, SSX).mean(axis=2)
    return rgb, a


def _label_alpha():
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    cr = cairo.Context(surf)
    cr.set_antialias(cairo.ANTIALIAS_BEST)
    cr.select_font_face(FONT, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    cr.set_font_size(LABEL_SIZE)
    cr.set_source_rgba(1, 1, 1, 1)
    for cx, s in ((CX_L, LABEL_L), (CX_R, LABEL_R)):
        e = cr.text_extents(s)
        cr.move_to(cx - e.width / 2 - e.x_bearing, LABEL_BASE)
        cr.show_text(s)
    surf.flush()
    buf = np.ndarray(shape=(H, W, 4), dtype=np.uint8, buffer=surf.get_data())
    return buf[:, :, 3].astype(np.float64) / 255.0


LAB_A = _label_alpha()


def render_frame(i, dot=True):
    tv = i / FPS
    img = np.repeat(BG_COL[:, None, :], W, axis=1).copy()
    for x0, mode in ((X_L, "spin"), (X_R, "slide")):
        rgb, a = pole_box(tv, mode, dot=dot)
        sl = img[Y0:Y0 + BH, x0:x0 + BW]
        img[Y0:Y0 + BH, x0:x0 + BW] = rgb * a[:, :, None] + sl * (1.0 - a[:, :, None])
    img = img * (1.0 - LAB_A[:, :, None]) + np.array(LABEL_RGB)[None, None, :] * LAB_A[:, :, None]
    return img


def to8(img):
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def boxes(f8):
    return f8[Y0:Y0 + BH, X_L:X_L + BW], f8[Y0:Y0 + BH, X_R:X_R + BW]


# -------------------------------------------------------------------- checks

OK = True


def t(cond, msg):
    global OK
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        OK = False


# Bounded in ROWS and COLUMNS to the glass interior, so the check cannot see
# either metal cap, the antialiased silhouette, or the strip of BACKGROUND that
# sits inside the box either side of the pole -- which is what the first version
# of this check counted. The lower half of the background gradient is dark and
# blue-dominant, so it read 2668 "dot" pixels on a frame with no dot in it.
# (RENDERER.md trap 58: bound a pixel check in rows as well as columns, and
# name the exclusions.)
_bcols = np.arange(BW) + 0.5
GLASS_MASK = np.zeros((BH, BW), bool)
GLASS_MASK[CAP_H + 2:BH - CAP_H - 2, :] = np.abs(_bcols - BW / 2.0) < (R_G - 3.0)


def dot_mask(box):
    """Find the DOT and nothing else.

    Counting ink says how much is drawn, never what it is (RENDERER.md trap
    61). Inside the glass the dot is the only blue-dominant thing there is: the
    red stripe has r well above b at every light level, and the white stripe is
    neutral. So: dark, and blue beats red by more than lighting ever varies it.
    """
    f = box.astype(np.float64) / 255.0
    return ((f[:, :, 2] > f[:, :, 0] + 0.05) & (f.max(2) < 0.28) & GLASS_MASK)


def dot_pixels(box):
    return int(dot_mask(box).sum())


def _centroid(m):
    """-> (pixel count, mean row, mean col), or None if there is no dot."""
    n = int(m.sum())
    if n < 40:
        return None
    r, c = np.where(m)
    return n, r.mean(), c.mean()


def run_checks():
    print("\nthe identity, on paper")
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(200):
        tv = float(rng.uniform(0, 400))
        y = rng.uniform(-2, 2, 40)
        p = rng.uniform(-math.pi / 2, math.pi / 2, 40)
        worst = max(worst, float(np.abs(phase_spin(y, p, tv)
                                        - phase_slide(y, p, tv)).max()))
    t(worst < 1e-12, f"the two phase functions agree to {worst:.2e} on 200 random times")
    t(abs(slide_dist(T_TURN) + PITCH) < 1e-12,
      f"one full turn slides the core exactly one pitch ({-slide_dist(T_TURN):.3f} m up)")
    t(abs(abs(slide_dist(DUR)) / DUR - PITCH / T_TURN) < 1e-12,
      f"stripes climb at {abs(slide_dist(DUR)) / DUR:.3f} m/s, "
      f"surface turns at {TWO_PI * R_M / T_TURN:.3f} m/s")

    print("\ngeometry")
    t(X_L >= 0 and X_R + BW <= W, f"boxes span x {X_L}..{X_L + BW} and {X_R}..{X_R + BW}")
    t(X_L + BW < X_R, f"boxes do not touch -- {X_R - (X_L + BW)} px of gap")
    t(Y0 >= SAFE_TOP and Y0 + BH <= SAFE_BOT,
      f"assembly rows {Y0}..{Y0 + BH}, safe area {SAFE_TOP}..{SAFE_BOT}")
    lr = np.where(LAB_A.any(1))[0]
    t(lr.min() >= SAFE_TOP and lr.max() < Y0,
      f"labels sit in rows {lr.min()}..{lr.max()}, above the boxes and inside safe")
    lc = np.where(LAB_A.any(0))[0]
    t(lc.min() >= 0 and lc.max() < W, f"labels span x {lc.min()}..{lc.max()}")
    t(GLASS_M / PITCH == 2.0, f"exactly {GLASS_M / PITCH:.1f} wraps of stripe are visible")

    print("\nthe identity, in pixels")
    bad = []
    for i in range(NF):
        bl, br = boxes(to8(render_frame(i, dot=False)))
        if not np.array_equal(bl, br):
            bad.append((i, int((bl != br).sum())))
    t(not bad, f"turned and slid are byte-identical on all {NF} frames "
               f"({'0 pixels differ' if not bad else bad[:3]})")

    f0 = to8(render_frame(0, dot=False))
    ft = to8(render_frame(int(T_TURN * FPS), dot=False))
    t(np.array_equal(f0, ft), f"frame 0 == frame {int(T_TURN * FPS)}: one turn is an exact loop")
    fq = to8(render_frame(int(0.25 * T_TURN * FPS), dot=False))
    moved = int((np.abs(f0.astype(int) - fq.astype(int)).max(2) > 8).sum())
    t(moved > 120000, f"and it is not a still -- {moved} px change in a quarter turn")

    print("\nthe dot")
    bl, br = boxes(to8(render_frame(int(DOT_T * FPS) - 1)))
    t(dot_pixels(bl) == 0 and dot_pixels(br) == 0,
      f"no dot before {DOT_T:.1f} s (left {dot_pixels(bl)}, right {dot_pixels(br)})")
    j = int((DOT_T + 0.25) * FPS)
    bl, br = boxes(to8(render_frame(j)))
    t(dot_pixels(bl) > 1200 and dot_pixels(br) > 1200,
      f"both dots are up at {j / FPS:.2f} s (left {dot_pixels(bl)}, right {dot_pixels(br)})")


    b0l, b0r = boxes(to8(render_frame(int(DOT_T * FPS))))
    p0l, p0r = _centroid(dot_mask(b0l)), _centroid(dot_mask(b0r))
    t(abs(p0l[1] - p0r[1]) < 1.0 and abs(p0l[2] - p0r[2]) < 1.0,
      f"the two dots are painted in the same place (row {p0l[1]:.1f}/{p0r[1]:.1f}, "
      f"col {p0l[2]:.1f}/{p0r[2]:.1f})")
    t(dot_spin(DOT_T) == dot_slide(DOT_T), "and the two dot functions agree that they are")

    trk_l, trk_r = [], []
    for k in range(int(DOT_T * FPS), NF):
        b1, b2 = boxes(to8(render_frame(k)))
        m1, m2 = dot_mask(b1), dot_mask(b2)
        trk_l.append(_centroid(m1))
        trk_r.append(_centroid(m2))
    sl = [p for p in trk_l if p]
    sr = [p for p in trk_r if p]
    dr_l = max(p[1] for p in sl) - min(p[1] for p in sl)
    dc_l = max(p[2] for p in sl) - min(p[2] for p in sl)
    dr_r = max(p[1] for p in sr) - min(p[1] for p in sr)
    # only while the dot is WHOLE. Once the cap starts cutting it off, the
    # surviving sliver is no longer symmetric about its own centre and the
    # centroid drifts a few px -- that is the check losing sight of it, not the
    # dot moving. (RENDERER.md trap 62: find the true behaviour, do not just
    # loosen the number.)
    whole = [p for p in sr if p[0] >= 1500]
    dc_r = max(p[2] for p in whole) - min(p[2] for p in whole)
    t(dr_l < 1.0 and dc_l > 100,
      f"the turned pole's dot goes SIDEWAYS: {dc_l:.0f} px across, {dr_l:.0f} px up")
    t(dc_r < 1.5 and dr_r > 900,
      f"the slid pole's dot goes UP and only up: {dr_r:.0f} px up, {dc_r:.1f} px "
      f"across over the {len(whole)} frames it is whole")
    t(len(sl) < len(trk_l) - 20,
      f"the turned dot goes round the back -- absent on {len(trk_l) - len(sl)} of "
      f"{len(trk_l)} frames")
    t(trk_r[-1] is None, "the slid dot has left the top of the glass by the last frame")
    lz, rz = boxes(to8(render_frame(NF - 1)))
    t(np.array_equal(lz, b0l),
      "two turns on, the turned pole is byte-for-byte the picture it was when "
      "the dot was painted")
    t(not np.array_equal(rz, b0r),
      "the slid pole is not -- its dot went off the top and is not coming back")

    print("\nframe")
    last = to8(render_frame(NF - 1))
    f = last.astype(np.float64) / 255.0
    ink = np.abs(f - BG_COL[:, None, :]).max(2) > 0.02
    rows = np.where(ink.any(1))[0]
    t(rows.min() >= SAFE_TOP and rows.max() <= SAFE_BOT,
      f"everything drawn is in rows {rows.min()}..{rows.max()}")
    t(np.array_equal(to8(render_frame(137)), to8(render_frame(137))), "deterministic")
    t(not np.array_equal(*boxes(last)),
      "and by the last frame the two poles are NOT the same picture any more")

    print("\n" + ("ALL CHECKS PASS" if OK else "SOMETHING FAILED"))
    return 0 if OK else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--stills')
    ap.add_argument('--out')
    args = ap.parse_args()

    print(f"glass {2 * R_M * 100:.0f} cm across, {GLASS_M * 100:.0f} cm long, "
          f"stripe wraps every {PITCH * 100:.0f} cm")
    print(f"  {N_TURN} turns at {T_TURN} s, {DUR:.1f} s, {NF} frames, "
          f"dot at {DOT_T:.1f} s")

    if args.check:
        return run_checks()

    if args.stills:
        from PIL import Image
        for i in (0, 18, 36, int(DOT_T * FPS) + 14, int(DOT_T * FPS) + 55,
                  int(DOT_T * FPS) + 110, NF - 1):
            Image.fromarray(to8(render_frame(i))).save(f"{args.stills}_{i:04d}.png")
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
        p.stdin.write(to8(render_frame(i)).tobytes())
        if i % 30 == 0:
            print(f"  frame {i}/{NF}", flush=True)
    p.stdin.close()
    p.wait()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
