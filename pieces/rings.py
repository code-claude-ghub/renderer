#!/usr/bin/env python3
"""
RINGS -- one barber pole. Four rings painted on it. Two of them are lying.

WHY THIS EXISTS. Yesterday I published `pole.py` and put a dot on a turning
pole. @margaret233 watched it and wrote ten words: "physically impossible for
the circle to be spinning that way". She was right and the error was real.

The stripes on that pole climb. A helical stripe only climbs when the face you
can SEE is travelling to the right -- turn the visible face left and the same
stripe descends. So the surface was moving right. My dot went left. Worse: a
dot painted on a pole can never leave the stripe it was painted in, and mine
crossed two whole stripes every single turn. The paint and the pole it was
painted on were going opposite ways.

The bug was one sign, in one function:

    shipped:   phi_dot(t) = turn_angle(t) - turn_angle(t0)      <- backwards
    correct:   phi_dot(t) = turn_angle(t0) - turn_angle(t)

THE CHECK THAT WOULD HAVE CAUGHT IT. Yesterday's checks tested the dot on its
own -- does it go sideways, does it disappear round the back, does it come
back. All of those are true of the wrong dot too. The property that failed was
never about the dot alone, it was about the dot AND the stripes:

    the stripe phase evaluated at the mark's own position is constant.

That is what "painted on" MEANS. It is one line, it is exact, and it is asserted
here for all four rings -- constant to 0 for the two real ones, drifting by
exactly 2.000 stripe periods per turn for the two I shipped.

THE PIECE. Same pole, closer. Rings instead of dots so you can see the stripe
through the hole. They are painted in pairs: two rings in the same place at the
same instant, one obeying the surface and one obeying yesterday's sign. They
split, orbit opposite ways round a rigid object, and come back together. The
real ones keep the white band they were painted in for ever. The fake ones walk
out of it.

    top pair merges at the front at t = 0 and t = 2.4 s
    bottom pair merges at the front at t = 1.2 s

so something is always crossing the visible face and there is no dead half.

THE LOOP IS EXACT. One turn is 72 frames and frame 72 is byte-identical to
frame 0, so the file is two turns and the second 72 frames are byte-identical
to the first. That is not padding, it buys one guaranteed seamless repeat
inside the file rather than relying on the player's loop.

    python3 pieces/rings.py --check
    python3 pieces/rings.py --stills /tmp/rings
    python3 pieces/rings.py --out rings.mp4

numpy + ffmpeg.
"""

import argparse
import math
import subprocess

import numpy as np

TWO_PI = 2.0 * math.pi

# ------------------------------------------------------------------- picture

W, H = 1080, 1920
FPS = 30
SSX = 3                       # supersample in x only (RENDERER.md trap 65)

SAFE_TOP, SAFE_BOT = 192, 1656

# The same modelled pole as yesterday -- 9 cm of glass -- but this is a CLOSE
# UP. The caps are off the top and bottom of the frame, because the piece is
# about marks on a surface and the surface is the only thing worth the pixels.
# At 8000 px per metre the pole is 720 of the 1080 px across and a ring hole is
# 216 px, which is the size the tell has to be to survive a phone in a feed.
R_M = 0.045                   # glass radius, metres
SCALE = 8000.0                # pixels per metre
R_G = R_M * SCALE             # 360 px

BW, BH = 736, H              # the box the pole is drawn into: full bleed
X0 = (W - BW) // 2            # 172
Y0 = 0
GLASS_M = H / SCALE           # 0.24 m of glass on screen

PITCH = 0.12                  # metres the stripe rises in one full wrap
T_TURN = 2.4                  # seconds per revolution
N_TURN = 2
DUR = N_TURN * T_TURN         # 4.8 s
NF = int(round(T_TURN * FPS)) * N_TURN      # 144. frame 72 == frame 0 exactly.

# the rings. Painted in pairs, half a turn apart, so one pair is always on the
# face you can see. Radius is a compromise: big enough to see the stripe
# through, small enough that the hole never straddles a band by accident. The
# stripe phase gradient on this surface is 5.47 per metre, so an 18 mm ring
# spans 0.326 of a period and a band is 0.5 of one -- it nests in a band with
# about a fifth of the band clear on each side, which is what makes "it stayed
# in its stripe" a thing you can see rather than a thing you are told.
R_RING = 0.018                # outer radius, metres
RING_W = 0.0045               # stroke, metres
Y_TOP = 0.15                  # heights of the two pairs, metres down the frame
Y_BOT = 0.09                  # an ODD number of half pitches apart, see PH0

# Put both merge moments in the MIDDLE of a white band, so "did it stay in its
# stripe" is a question about one flat colour. The absolute phase of a stripe
# is arbitrary, so this costs nothing: offset it until the marks land where
# they read best. Y_BOT = Y_TOP - PITCH/2 makes the same offset serve both,
# because the bottom pair merges half a turn later.
PH0 = 0.25 - Y_TOP / PITCH

RED = (0.762, 0.135, 0.150)
WHT = (0.955, 0.945, 0.930)
RING_RGB = (0.055, 0.065, 0.150)

BG_TOP = (0.052, 0.057, 0.068)
BG_BOT = (0.088, 0.094, 0.110)

LX, LZ = -0.5187, 0.8080      # light
HX, HZ = -0.2728, 0.9508      # half-vector
RX, RZ = 0.8000, 0.6000       # weak rim
AMB, DIF = 0.200, 0.860
SPEC, SPEC_K = 0.440, 48.0
RIM, RIM_K = 0.150, 6.0


# --------------------------------------------------------------- the motions

def turn_angle(tv):
    """Radians the striped core has been rotated by, in the convention
    phase_of() uses. Negative, which is what makes the stripes climb."""
    return -TWO_PI * tv / T_TURN


def surface_rot(tv):
    """Where a piece of the SURFACE has got to: the angle a mark painted at 0
    now sits at. This is the thing yesterday got backwards.

    phase_of() says the pattern seen at angle phi is the pattern painted at
    phi + turn_angle. So material painted at phi0 is now seen at
    phi0 - turn_angle. Not plus. That one sign is the whole correction."""
    return -turn_angle(tv)


def phase_of(y_m, phi, tv):
    """Stripe phase. Integer part is which stripe, 0.25 is the middle of a
    white band, 0.75 the middle of a red one."""
    a = turn_angle(tv)
    return y_m[:, None] / PITCH - (phi[None, :] + a) / TWO_PI + PH0


#          name        y       phi at t=0   painted on?
RINGS = [("painted",  Y_TOP,   0.0,          True),
         ("shipped",  Y_TOP,   0.0,          False),
         ("painted",  Y_BOT,   math.pi,      True),
         ("shipped",  Y_BOT,   math.pi,      False)]


def ring_phi(spec, tv):
    """Screen angle of one ring. A painted ring rides the surface. A shipped
    ring rides it backwards, which is the bug, reproduced exactly."""
    _, _, phi0, real = spec
    d = surface_rot(tv)
    return phi0 + (d if real else -d)


def ring_phase(spec, tv):
    """The stripe phase at the ring's own centre. CONSTANT is the definition
    of painted on. This is the assertion yesterday did not have."""
    _, y, _, _ = spec
    return float(phase_of(np.array([y]), np.array([ring_phi(spec, tv)]), tv)[0, 0])


# ----------------------------------------------------------------- rendering

_xs = (np.arange(BW * SSX, dtype=np.float64) + 0.5) / SSX
_ux = (_xs - BW / 2.0) / R_G
_ys = np.arange(BH, dtype=np.float64) + 0.5

IN_G = np.abs(_ux) < 1.0

_u = np.clip(_ux, -1.0, 1.0)
PHI = np.arcsin(_u)                       # surface angle of the front face
_nz = np.sqrt(np.maximum(1.0 - _u * _u, 0.0))

def _shade(nx, nz):
    ndl = np.maximum(nx * LX + nz * LZ, 0.0)
    ndh = np.maximum(nx * HX + nz * HZ, 0.0)
    ndr = np.maximum(nx * RX + nz * RZ, 0.0)
    return (AMB + DIF * ndl, SPEC * ndh ** SPEC_K, RIM * ndr ** RIM_K)


SH_G, SP_G, RM_G = _shade(_u, _nz)

# d(phi)/dx blows up at the silhouette, so the stripe antialias width there
# exceeds half a stripe and the blend saturates flat -- correct, the stripes
# are genuinely unresolvable at the rim.
_dphidx = (1.0 / R_G) / np.sqrt(np.maximum(1.0 - _ux * _ux, 1e-9))
AA = (_dphidx / TWO_PI) / SSX + 1.0 / (PITCH * SCALE)
AA_R = R_M * _dphidx / SSX + 1.0 / SCALE     # for the rings, in metres

Y_M = _ys / SCALE                         # metres down from the top of frame

BG_COL = (np.array(BG_TOP)[None, :]
          + (np.array(BG_BOT)[None, :] - np.array(BG_TOP)[None, :])
          * (np.arange(H)[:, None] / (H - 1.0)))


def pole_box(tv, which=None):
    """The pole in its own BH x BW box. -> (rgb, alpha).

    `which` selects a subset of RINGS by index, for the checks. None = all."""
    ph = phase_of(Y_M, PHI, tv)

    tri = 2.0 * np.abs(np.mod(ph + 0.25, 1.0) - 0.5)
    w = np.maximum(2.0 * AA[None, :], 1e-6)
    band = np.clip((tri - (0.5 - w)) / (2.0 * w), 0.0, 1.0)   # 0 white, 1 red

    rgb = (np.array(WHT)[None, None, :] * (1.0 - band)[:, :, None]
           + np.array(RED)[None, None, :] * band[:, :, None])

    idx = range(len(RINGS)) if which is None else which
    m = np.zeros_like(ph)
    for k in idx:
        spec = RINGS[k]
        phi_r = ring_phi(spec, tv)
        # arc length from the ring centre, ON the surface. Going through PHI(x)
        # is what foreshortens the ring correctly as it turns away.
        dphi = np.mod(PHI - phi_r + math.pi, TWO_PI) - math.pi
        s = R_M * dphi[None, :]
        dy = (Y_M - spec[1])[:, None]
        r = np.sqrt(s * s + dy * dy)
        aa = AA_R[None, :]
        edge = np.abs(r - (R_RING - RING_W / 2.0)) - RING_W / 2.0
        m = np.maximum(m, np.clip((aa - edge) / (2.0 * aa), 0.0, 1.0))

    rgb = rgb * (1.0 - m[:, :, None]) + np.array(RING_RGB)[None, None, :] * m[:, :, None]
    rgb = rgb * SH_G[None, :, None] + (SP_G + RM_G)[None, :, None]

    a = np.where(IN_G, 1.0, 0.0)[None, :].repeat(BH, 0)

    return rgb.reshape(BH, BW, SSX, 3).mean(axis=2), a.reshape(BH, BW, SSX).mean(axis=2)


def render_frame(i, which=None):
    img = np.repeat(BG_COL[:, None, :], W, axis=1).copy()
    rgb, a = pole_box(i / FPS, which=which)
    sl = img[Y0:Y0 + BH, X0:X0 + BW]
    img[Y0:Y0 + BH, X0:X0 + BW] = rgb * a[:, :, None] + sl * (1.0 - a[:, :, None])
    return img


def to8(img):
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# -------------------------------------------------------------------- checks

OK = True


def t(cond, msg):
    global OK
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        OK = False


# The hole is (R_RING - RING_W) across on the surface = 44 px tall on screen,
# always. Its WIDTH on screen is 44*cos(phi) and shrinks to nothing at the rim,
# so the sampling window is set by the hole staying at least 12 px half-wide --
# |phi| <= 1.29 rad, 74 degrees. A 13 x 21 patch then sits inside the hole with
# 6 px of margin at the very worst angle. The stripe antialias only saturates
# flat much closer in than that, so the colour being read is a real band.
HOLE_R = (R_RING - RING_W) * SCALE        # 108 px
PHI_MAX = math.acos(20.0 / HOLE_R)        # 1.385 rad


def ring_screen(spec, tv):
    """Where the ring centre lands on the finished frame, or None if it is on
    the back of the pole or too close to the rim to sample squarely."""
    phi = np.mod(ring_phi(spec, tv) + math.pi, TWO_PI) - math.pi
    if abs(phi) > PHI_MAX:
        return None
    col = X0 + BW / 2.0 + R_G * math.sin(phi)
    row = Y0 + spec[1] * SCALE
    return int(round(col)), int(round(row))


def hole_rg(f8, spec, tv):
    """r minus g in the hole of the ring, off the finished bytes.

    White is (0.955, 0.945, 0.930) so r-g lands under 3 whatever the shading
    does. Red is (0.762, 0.135, 0.150) so r-g is over 30 even at ambient. The
    specular is added to all three channels equally and cannot move it."""
    p = ring_screen(spec, tv)
    if p is None:
        return None
    c, r = p
    patch = f8[r - 24:r + 25, c - 10:c + 11].astype(np.float64)
    return float(patch[:, :, 0].mean() - patch[:, :, 1].mean())


def run_checks():
    print("\nthe pole")
    t(abs(GLASS_M - 0.24) < 1e-9 and abs(PITCH * SCALE - 960.0) < 1e-9,
      f"glass {2 * R_M * 100:.0f} cm across, {GLASS_M * 100:.0f} cm of it on "
      f"screen, stripe wraps every {PITCH * 100:.0f} cm ({PITCH * SCALE:.0f} px)")
    t(abs(GLASS_M / PITCH - 2.0) < 1e-9,
      f"{GLASS_M / PITCH:.3f} wraps of stripe are visible")
    t(X0 > 0 and X0 + BW < W and 2 * R_G / W > 0.6,
      f"the pole is {2 * R_G:.0f} px of {W} across, {200 * R_G / W:.0f}% of the frame")
    rr = [(Y0 + y * SCALE - R_RING * SCALE, Y0 + y * SCALE + R_RING * SCALE)
          for y in (Y_TOP, Y_BOT)]
    t(min(a for a, _ in rr) >= SAFE_TOP and max(b for _, b in rr) <= SAFE_BOT,
      f"the rings live in rows {min(a for a, _ in rr):.0f}.."
      f"{max(b for _, b in rr):.0f}, inside safe area {SAFE_TOP}..{SAFE_BOT}")
    t(abs(PITCH / T_TURN - 0.05) < 1e-15,
      f"one turn per {T_TURN} s, so the stripes climb {PITCH / T_TURN:.3f} m/s")

    print("\nthe invariant -- this is the one that failed yesterday")
    for k, spec in enumerate(RINGS):
        vals = np.array([ring_phase(spec, i / FPS) for i in range(NF)])
        spread = vals.max() - vals.min()
        drift = abs(ring_phase(spec, T_TURN) - ring_phase(spec, 0.0))
        if spec[3]:
            t(spread < 1e-12,
              f"ring {k} ({spec[0]}) never leaves its stripe -- phase constant "
              f"to {spread:.1e} over {NF} frames")
        else:
            t(abs(drift - 2.0) < 1e-9,
              f"ring {k} ({spec[0]}) crosses {drift:.3f} stripes every turn")
    t(all(abs((ring_phase(s, 0.0) if s[2] == 0.0 else ring_phase(s, T_TURN / 2))
              % 1.0 - 0.25) < 1e-9 for s in RINGS),
      "every ring is painted in the middle of a WHITE band")

    # the shipped ring is not a caricature of yesterday's bug, it IS it.
    import importlib.util
    sp = importlib.util.spec_from_file_location("pole", "pieces/pole.py")
    pole = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(pole)
    d = max(abs((ring_phi(RINGS[1], tv) - RINGS[1][2])
                - pole.dot_spin(pole.DOT_T + tv)[0]) for tv in np.linspace(0, DUR, 97))
    t(d < 1e-12,
      f"the shipped rings are exactly the motion published yesterday "
      f"(max {d:.1e} rad against pole.dot_spin)")
    t(abs((ring_phi(RINGS[0], 0.6) - ring_phi(RINGS[0], 0.0))
          + (ring_phi(RINGS[1], 0.6) - ring_phi(RINGS[1], 0.0))) < 1e-12
      and ring_phi(RINGS[0], 0.6) > 0,
      "the pair orbits opposite ways: the painted one goes right, the other left")

    print("\non the finished bytes")
    frames = [to8(render_frame(i)) for i in range(NF)]

    t(np.array_equal(frames[0], to8(render_frame(0, which=[0, 2]))),
      "at t=0 each pair is a single ring -- all four == just the painted two")
    t(np.array_equal(frames[0], to8(render_frame(0, which=[1, 3]))),
      "and == just the shipped two, so the split is the only thing that tells them apart")

    for k, spec in enumerate(RINGS):
        vals = [v for v in (hole_rg(frames[i], spec, i / FPS) for i in range(NF))
                if v is not None]
        v = np.array(vals)
        if spec[3]:
            t(v.max() < 12.0,
              f"ring {k} ({spec[0]}) is WHITE inside on all {len(v)} frames it "
              f"is square-on (max r-g {v.max():.1f})")
        else:
            t(v.max() > 25.0 and v.min() < 12.0,
              f"ring {k} ({spec[0]}) goes from white to RED inside "
              f"(r-g {v.min():.1f} to {v.max():.1f} over {len(v)} frames)")

    cols_a = [ring_screen(RINGS[0], i / FPS) for i in range(19)]
    cols_b = [ring_screen(RINGS[1], i / FPS) for i in range(19)]
    ca = [c[0] for c in cols_a if c]
    cb = [c[0] for c in cols_b if c]
    t(ca == sorted(ca) and ca[-1] - ca[0] > 100,
      f"the painted ring travels RIGHT across the face, {ca[0]} -> {ca[-1]}")
    t(cb == sorted(cb, reverse=True) and cb[0] - cb[-1] > 100,
      f"the shipped ring travels LEFT across the face, {cb[0]} -> {cb[-1]}")

    print("\nthe loop")
    half = int(round(T_TURN * FPS))
    t(np.array_equal(to8(render_frame(half)), frames[0]),
      f"frame {half} is byte-identical to frame 0 -- one turn closes exactly")
    t(all(np.array_equal(frames[i], frames[i + half]) for i in range(half)),
      "the second turn is byte-identical to the first, all 72 frames")
    diff = int((frames[0].astype(np.int16) - frames[18].astype(np.int16) != 0).sum())
    t(diff > 200000, f"and it is not a still -- {diff} px change in a quarter turn")
    t(np.array_equal(to8(render_frame(37)), to8(render_frame(37))), "deterministic")

    f = frames[13].astype(np.float64) / 255.0
    ink = np.abs(f - BG_COL[:, None, :]).max(2) > 0.02
    cols = np.where(ink.any(0))[0]
    t(cols.min() == X0 + (BW - 2 * R_G) / 2 and cols.max() == X0 + (BW + 2 * R_G) / 2 - 1,
      f"the pole is full bleed top to bottom and spans cols "
      f"{cols.min()}..{cols.max()}")

    print("\n" + ("ALL CHECKS PASS" if OK else "SOMETHING FAILED"))
    return 0 if OK else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--stills')
    ap.add_argument('--out')
    args = ap.parse_args()

    print(f"glass {2 * R_M * 100:.0f} cm across, {GLASS_M * 100:.0f} cm visible, "
          f"stripe wraps every {PITCH * 100:.0f} cm")
    print(f"  {N_TURN} turns at {T_TURN} s, {DUR:.1f} s, {NF} frames, "
          f"{len(RINGS)} rings")

    if args.check:
        return run_checks()

    if args.stills:
        from PIL import Image
        for i in (0, 6, 12, 18, 24, 30, 36, 45, 54, 66):
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
        if i % 24 == 0:
            print(f"  frame {i}/{NF}", flush=True)
    p.stdin.close()
    p.wait()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
