#!/usr/bin/env python3
"""MOIRE — the overlap of two stripe patterns is a 16x magnifier.

Three strips on paper grey. Bottom: vertical stripes, pitch 30 px. Middle:
the same stripes laid OVER a fixed pitch-32 grating — the overlap beats
into broad dark bands with period 30*32/(32-30) = 480 px. Top: the bottom
stripes honestly re-drawn at 16x scale (pitch 480 px).

The claim is an exact identity. Move the fine grating by d and the beat
fringes move by d * q/(q-p) = 16 d — same period AND same motion as the
16x blow-up above. The moire pattern IS a magnified image of the grating,
made of nothing but overlap.

The piece: the stripes step one pixel at a time (12 ticks). One pixel is
the smallest motion this video can contain, and it is invisible in the
stripes — but the dark bands and the 16x strip leap 16 px in lockstep,
every tick. Then a smooth glide covers the remaining 18 px; after a total
displacement of exactly one pitch (30 px) every strip maps onto itself and
the last frame equals the first, byte-identical. The video loops on a
closed identity, not on an edit.

Family: the render proves a property about itself (POLE, TRAIN, WHEEL,
UNSTIR). All three strip phases are measured off the shipped h264 with an
exact-window DFT; fringe motion / stripe motion must come out 16.000.

Self-referential footnote: moire is trap 70 in this channel's renderer
notes — the artifact that ruins fine patterns at watch size. This piece
is made of it on purpose, at a pitch chosen to survive the downscale.
"""
import os
import subprocess
import sys
import time

import numpy as np

# ---------------------------------------------------------------- geometry
W, H = 1080, 1920
FPS = 30

P = 30.0                        # moving grating pitch, px
Q = 32.0                        # fixed grating pitch, px
M = Q / (Q - P)                 # magnification: exactly 16
L = P * Q / (Q - P)             # beat period: exactly 480 px
DUTY = 0.5                      # dark-bar fraction of each pitch
SS = 16                         # x-only supersampling (trap 65)

PAPER = 0.845                   # trap 69: warm grey, not full white
INK = 0.13
HAIR = (0.70, 0.16, 0.14)       # fixed reference hairline
HAIR_X = 539                    # hairline columns HAIR_X..HAIR_X+1
HAIR_Y0, HAIR_Y1 = 300, 1600

# strips: (y0, y1). all inside safe area 192..1632 (trap 3)
ZOOM = (340, 760)               # the honest 16x re-render of the grating
OVER = (830, 1250)              # moving grating OVER fixed grating: moire
RAW = (1320, 1560)              # the moving grating alone

# ---------------------------------------------------------------- timeline
F_HOLD0 = 18                    # open hold at shift 0
TICKS = 12                      # one-pixel steps
F_TICK = 12                     # 0.4 s per step
F_GLIDE = 60                    # smooth glide covering the remaining px
F_HOLD1 = 18                    # end hold at shift 30 == 0 (seamless loop)
N_FRAMES = F_HOLD0 + TICKS * F_TICK + F_GLIDE + F_HOLD1   # 240 -> 8.0 s
GLIDE_PX = P - TICKS            # 18 px

OUT = f"out/moire_{time.strftime('%H%M%S')}.mp4"


def shift(f):
    """Displacement of the fine grating at frame f, in px."""
    if f < F_HOLD0:
        return 0.0
    f -= F_HOLD0
    if f < TICKS * F_TICK:
        return float(f // F_TICK + 1)          # instant 1-px steps
    f -= TICKS * F_TICK
    if f < F_GLIDE:
        t = (f + 1) / F_GLIDE
        s = t * t * (3.0 - 2.0 * t)            # smoothstep
        return TICKS + GLIDE_PX * s
    return P                                    # == 0 (mod P): closure


# ---------------------------------------------------------------- render
# Exact per-pixel coverage. The light function (product of 0/1 gratings) is
# piecewise constant with ~150 breakpoints per row, so its integral is
# piecewise linear and pixel coverage is exact interval arithmetic — no
# supersampling, no quadrature noise. (First draft supersampled at SS=16
# and the ~0.03 px edge error became ~0.6 px of measured fringe-phase
# noise. Exactness is cheaper than the tolerance it saves.)


def exact_light_row(gratings):
    """Pixel-exact mean of the product of square gratings over each pixel.

    gratings: list of (pitch, offset); light where frac((x-a)/p) >= DUTY.
    """
    xs = [0.0, float(W)]
    for (p, a) in gratings:
        k0 = int(np.floor((0.0 - a) / p)) - 1
        k1 = int(np.ceil((W - a) / p)) + 1
        ks = np.arange(k0, k1 + 1, dtype=np.float64)
        xs.append(a + ks * p)                   # dark-bar starts
        xs.append(a + (ks + DUTY) * p)          # dark-bar ends
    xs = np.unique(np.clip(np.concatenate(
        [np.atleast_1d(np.asarray(v, np.float64)) for v in xs]), 0.0, W))
    mids = (xs[:-1] + xs[1:]) / 2.0
    light = np.ones(len(mids), bool)
    for (p, a) in gratings:
        light &= (((mids - a) / p) % 1.0) >= DUTY
    F = np.concatenate([[0.0], np.cumsum(light * np.diff(xs))])
    G = np.interp(np.arange(W + 1, dtype=np.float64), xs, F)
    return np.diff(G)


BZ = 120.0   # constant phase offset of the zoom strip, px: puts its dark
             # bar exactly over the moire dark band. Both patterns have
             # period 480 and identical velocity, so one constant aligns
             # them for all time (computed by matching the two bin phases
             # at shift 0; it comes out exactly a quarter beat).


def strip_rows(a):
    """The three strips' transmission rows (0=ink, 1=paper) at shift a."""
    a = a % P                                   # exact closure (trap 27/28)
    zoom = exact_light_row([(P * M, a * M + BZ)])   # the grating at 16x
    over = exact_light_row([(P, a), (Q, 0.0)])
    raw = exact_light_row([(P, a)])
    return zoom, over, raw


def lum(t):
    return INK + (PAPER - INK) * t


def frame_at(f):
    zoom, over, raw = strip_rows(shift(f))
    fr = np.full((H, W, 3), PAPER, np.float64)
    for (y0, y1), row in ((ZOOM, zoom), (OVER, over), (RAW, raw)):
        fr[y0:y1, :, :] = lum(row)[None, :, None]
    fr[HAIR_Y0:HAIR_Y1, HAIR_X:HAIR_X + 2, :] = HAIR
    return (np.clip(fr, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def render_frames():
    for f in range(N_FRAMES):
        yield frame_at(f)


# ---------------------------------------------------------------- measure
X0, X1 = 60, 1020    # 960 px window: exactly 2 beats, 32 P-pitches, 30 Q-pitches
                     # -> every fundamental has an integer period count and the
                     # single-bin DFT phase is leakage-free (no window needed)


def phase_pos(row, period):
    """Position (mod period) of a periodic row via its own DFT bin, px."""
    seg = row[X0:X1].astype(np.float64)
    n = X1 - X0
    x = np.arange(n)
    z = (seg * np.exp(-2j * np.pi * x / period)).sum()
    return (-np.angle(z)) * period / (2.0 * np.pi)


def wrap(d, period):
    """Signed displacement d folded into (-period/2, period/2]."""
    return (d + period / 2.0) % period - period / 2.0


def phases_of_rows(zoom, over, raw):
    return (phase_pos(zoom, L), phase_pos(over, L), phase_pos(raw, P))


def alpha(m):
    """Fourier coefficient of the light indicator (light where frac >= DUTY)."""
    if m == 0:
        return 1.0 - DUTY
    return (1.0 - np.exp(-2j * np.pi * m * DUTY)) / (-2j * np.pi * m)


def z_beat_pred(a):
    """The overlay's beat-frequency DFT bin, predicted in closed form.

    Components of tA(x-a)*tB(x) landing at spatial frequency 1/L satisfy
    m/P + n/Q = 1/L, i.e. 16m + 15n = 1: (m, n) = (1+15t, -1-16t). The
    t=0 term is the fringe and moves as e^{-2pi i a/P} — exactly 16a. The
    |t|>=2 odd-harmonic terms land in the SAME bin with a different phase
    velocity, so the measured fringe position wobbles ~0.2 px about 16a.
    Trap 62: predict the quantity the instrument actually reads.
    """
    z = 0.0 + 0.0j
    for t in range(-6, 7):
        m, n = 1 + 15 * t, -1 - 16 * t
        z += alpha(m) * alpha(n) * np.exp(-2j * np.pi * m * a / P)
    return z


def pred_beat_pos(a):
    return (-np.angle(z_beat_pred(a))) * L / (2.0 * np.pi)


def phases_of_frame(fr, y_off=0):
    """Measure the three strip phases from pixel data (uint8 HxWx3)."""
    rows = []
    for (y0, y1) in (ZOOM, OVER, RAW):
        band = fr[(y0 + y1) // 2 - y_off, :, :].astype(np.float64).mean(axis=1)
        rows.append(band)
    return (phase_pos(rows[0], L), phase_pos(rows[1], L), phase_pos(rows[2], P))


# ---------------------------------------------------------------- checks
def run_checks():
    ok = []

    def check(name, cond, detail=""):
        ok.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")

    # 1. the identity itself, exact
    check("magnification q/(q-p) = 16 exactly", M == 16.0, f"M={M}")
    check("beat period pq/(q-p) = 480 = 16*pitch exactly",
          L == 480.0 and L == M * P, f"L={L}")

    # 2. independent overlay implementation: dense midpoint sampling,
    #    different code path (the renderer is exact; this bounds it)
    a = 7.3
    ss2 = 512
    u = (np.arange(W * ss2) + 0.5) / ss2
    tA2 = (((u - (a % P)) / P) % 1.0) >= DUTY
    tB2 = ((u / Q) % 1.0) >= DUTY
    over2 = (tA2 & tB2).astype(np.float64).reshape(W, ss2).mean(axis=1)
    _, over1, _ = strip_rows(a)
    dmax = np.abs(over1 - over2).max()
    check("overlay row matches independent implementation", dmax < 4e-3,
          f"max diff {dmax:.2e}")

    # 3. phase instrument self-test on a case with a known answer (trap 42)
    z0, o0, r0 = phases_of_rows(*strip_rows(0.0))
    z1, o1, r1 = phases_of_rows(*strip_rows(1.0))
    dz, do, dr = wrap(z1 - z0, L), wrap(o1 - o0, L), wrap(r1 - r0, P)
    check("1 px of grating -> 16.000 px of zoom strip (model, exact)",
          abs(dz - 16.0) < 1e-6, f"dz={dz:.9f}")
    check("raw strip moved exactly 1.000 px", abs(dr - 1.0) < 1e-6,
          f"dr={dr:.9f}")
    check("1 px of grating -> ~16 px of fringe (0.2 px harmonic wobble)",
          abs(do - 16.0) < 0.35, f"do={do:.6f}")

    # 3b. the wobble itself has a closed form — assert the measurement
    #     matches the prediction, not a loosened bound (trap 62)
    worst_pred = 0.0
    for a1, a2 in [(0.0, 0.37), (3.0, 4.0), (12.5, 12.9), (20.0, 21.0)]:
        _, oo1, _ = phases_of_rows(*strip_rows(a1))
        _, oo2, _ = phases_of_rows(*strip_rows(a2))
        meas = wrap(oo2 - oo1, L)
        pred = wrap(pred_beat_pos(a2) - pred_beat_pos(a1), L)
        worst_pred = max(worst_pred, abs(meas - pred))
    check("fringe-bin motion matches closed-form prediction",
          worst_pred < 1e-3, f"worst |meas-pred|={worst_pred:.2e} px")

    # 4. the coupling over the whole timeline (trap 66): zoom strip motion
    #    is EXACTLY 16x the grating; the fringe bin tracks it within its
    #    harmonic wobble; and over the closed loop the totals are exact
    # each instrument is graded against the TRUE timeline shift with its
    # own error budget: the zoom bin is exact; the raw bin wobbles ~6e-4 px
    # (its 29th/31st harmonics alias through the 1-px box filter onto the
    # pitch-30 bin); the fringe bin wobbles ~0.2 px (check 3b's formula).
    # do NOT grade one instrument against another's noise — the first
    # draft compared dz to 16*dr and read the raw bin's 6e-4 px sixteenfold.
    worst_zoom = 0.0
    worst_raw = 0.0
    worst_fringe = 0.0
    worst_lock = 0.0
    tot_o = tot_r = 0.0
    prev = phases_of_rows(*strip_rows(shift(0)))
    prev_s = shift(0)
    for f in range(1, N_FRAMES):
        s = shift(f)
        cur = phases_of_rows(*strip_rows(s))
        if s != prev_s:
            ds = s - prev_s
            dz = wrap(cur[0] - prev[0], L)
            do = wrap(cur[1] - prev[1], L)
            dr = wrap(cur[2] - prev[2], P)
            worst_zoom = max(worst_zoom, abs(dz - M * ds))
            worst_raw = max(worst_raw, abs(dr - ds))
            worst_fringe = max(worst_fringe, abs(do - M * ds))
            worst_lock = max(worst_lock, abs(dz - do))
            tot_o += do
            tot_r += dr
        prev, prev_s = cur, s
    check("zoom motion = 16 x true shift, every moving frame, exact",
          worst_zoom < 1e-4, f"worst |dz-16*ds|={worst_zoom:.2e} px")
    check("raw bin reads the true shift within its aliasing budget",
          worst_raw < 2e-3, f"worst |dr-ds|={worst_raw:.2e} px")
    check("fringe motion = 16 x true shift within harmonic wobble",
          worst_fringe < 0.35, f"worst |do-16*ds|={worst_fringe:.3f} px")
    check("zoom strip and fringes in lockstep within wobble",
          worst_lock < 0.35, f"worst |dz-do|={worst_lock:.3f} px")

    # 4b. not just lockstep — ALIGNED: the zoom dark bar sits on the moire
    #     dark band at every shift (BZ calibrates the constant, this
    #     asserts it stays zero because the velocities are identical)
    worst_align = 0.0
    for aa in (0.0, 1.0, 4.5, 12.0, 17.3, 25.0):
        zz, oo, _ = phases_of_rows(*strip_rows(aa))
        worst_align = max(worst_align, abs(wrap(zz - oo, L)))
    check("zoom dark bar aligned with moire dark band at all shifts",
          worst_align < 0.5, f"worst offset {worst_align:.3f} px")
    check("closed loop totals exact: fringes 480.000, grating 30.000",
          abs(tot_o - L) < 1e-6 and abs(tot_r - P) < 1e-6,
          f"fringes {tot_o:.6f} px, grating {tot_r:.6f} px, "
          f"ratio {tot_o / tot_r:.6f}")

    # 5. discrete steps: exactly TICKS jumps, each exactly 1 px of grating
    jumps = []
    for f in range(1, N_FRAMES):
        s0, s1 = shift(f - 1), shift(f)
        if s1 != s0 and f < F_HOLD0 + TICKS * F_TICK:
            jumps.append(s1 - s0)
    check(f"exactly {TICKS} one-pixel ticks",
          len(jumps) == TICKS and all(j == 1.0 for j in jumps),
          f"{len(jumps)} jumps, sizes {sorted(set(jumps))}")

    # 6. closure: total displacement one pitch, last frame == first frame
    check("total displacement = one pitch exactly", shift(N_FRAMES - 1) == P,
          f"shift={shift(N_FRAMES - 1)}")
    same = np.array_equal(frame_at(0), frame_at(N_FRAMES - 1))
    check("last frame byte-identical to first (loop is an identity)", same)

    # 7. holds are truly static
    still = np.array_equal(frame_at(2), frame_at(F_HOLD0 - 1))
    check("open hold static", still)

    # 8. watch-size survival (trap 70 made numeric): pool to 360 px wide and
    #    require both the beat and the raw stripes to keep modulation
    _, over_r, raw_r = strip_rows(4.0)
    o360 = lum(over_r).reshape(360, 3).mean(axis=1)
    r360 = lum(raw_r).reshape(360, 3).mean(axis=1)
    k = np.ones(40) / 40.0                       # ~beat-scale lowpass (480/3=160)
    o_lp = np.convolve(o360, k, mode="valid")
    beat_depth = o_lp.max() - o_lp.min()
    stripe_depth = (r360.max() - r360.min())
    check("beat modulation survives 360 px downscale", beat_depth > 0.10,
          f"depth {beat_depth:.3f}")
    check("raw stripes survive 360 px downscale", stripe_depth > 0.25,
          f"depth {stripe_depth:.3f}")

    # 9. pixel sanity (trap 56): read a real frame
    fr = frame_at(40)
    g = fr.astype(np.float64).mean(axis=2) / 255.0
    lit = (g > 0.5).mean()
    check("frame neither blank nor solid", 0.30 < lit < 0.97, f"lit {lit:.2f}")
    hl = fr[900, HAIR_X, :]
    check("hairline present and red", hl[0] > 150 and hl[0] > hl[1] + 60,
          f"rgb {hl.tolist()}")
    ink_rows = np.where((g < 0.5).any(axis=1))[0]
    check("all ink inside safe area (trap 3)",
          ink_rows.min() >= 192 and ink_rows.max() <= 1632,
          f"rows {ink_rows.min()}..{ink_rows.max()}")

    # 10. timeline
    check("240 frames = 8.0 s", N_FRAMES == 240 and N_FRAMES / FPS == 8.0)

    print()
    if not all(ok):
        print(f"{ok.count(False)} CHECK(S) FAILED")
        sys.exit(1)
    print(f"ALL {len(ok)} CHECKS PASSED")


# ---------------------------------------------------------------- encode
def encode():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-crf", "18", "-preset", "slow",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stderr=subprocess.DEVNULL)
    for fr in render_frames():                   # stream, never a list (trap 34)
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    assert p.returncode == 0, "ffmpeg failed"
    print(f"encoded {OUT} ({os.path.getsize(OUT)} bytes)")


def decode_frame(n):
    """One full decoded frame as uint8 HxWx3 (small frame count, fine)."""
    cmd = ["ffmpeg", "-i", OUT, "-vf", f"select=eq(n\\,{n})",
           "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    assert len(raw) == W * H * 3, f"decode size {len(raw)}"
    return np.frombuffer(raw, np.uint8).reshape(H, W, 3)


def check_encode():
    """Measure the identity off the shipped bytes."""
    # a tick: frame just before and just after the 4th step
    fa = F_HOLD0 + 3 * F_TICK - 1                # shift 3
    fb = F_HOLD0 + 3 * F_TICK                    # shift 4
    pa = phases_of_frame(decode_frame(fa))
    pb = phases_of_frame(decode_frame(fb))
    dz = wrap(pb[0] - pa[0], L)
    do = wrap(pb[1] - pa[1], L)
    dr = wrap(pb[2] - pa[2], P)
    print("ENCODE CHECK — one tick, measured off the shipped h264:")
    print(f"    grating moved {dr:6.3f} px   (model: 1)")
    print(f"    fringes moved {do:6.3f} px   (model: 16)")
    print(f"    16x strip     {dz:6.3f} px   (model: 16)")
    assert abs(dr - 1.0) < 0.25, f"raw step {dr}"
    assert abs(do - 16.0) < 0.60, f"fringe step {do}"
    assert abs(dz - 16.0) < 0.60, f"zoom step {dz}"
    assert abs(dz - do) < 0.60, "lockstep broken on encode"
    # the glide: ratio over a long baseline
    ga, gb = F_HOLD0 + TICKS * F_TICK + 5, F_HOLD0 + TICKS * F_TICK + 55
    qa = phases_of_frame(decode_frame(ga))
    qb = phases_of_frame(decode_frame(gb))
    s_true = shift(gb) - shift(ga)
    do_g = wrap(qb[1] - qa[1], L)
    dr_g = wrap(qb[2] - qa[2], P)
    ratio = do_g / dr_g
    print(f"    glide: grating {dr_g:6.3f} px (true {s_true:.3f}), "
          f"fringes {do_g:6.3f} px, ratio {ratio:6.3f} (model: 16)")
    assert abs(ratio - 16.0) < 0.5, f"glide ratio {ratio}"
    # loop closure on the shipped bytes
    d_loop = np.abs(decode_frame(0).astype(int)
                    - decode_frame(N_FRAMES - 1).astype(int))
    print(f"    loop: first vs last decoded frame, mean |diff| "
          f"{d_loop.mean():.3f}, max {d_loop.max()}")
    assert d_loop.mean() < 1.0, "loop not closed on encode"
    # container
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", OUT],
        capture_output=True, text=True).stdout.strip()
    assert int(probe) == N_FRAMES, f"frames {probe} != {N_FRAMES}"
    print(f"    {probe} frames; ratio holds on the uploaded file")


def review_stills():
    """Trap 67: look at it at the size it will be watched."""
    for name, f in [("open", 4), ("tick", F_HOLD0 + 5 * F_TICK + 2),
                    ("glide", F_HOLD0 + TICKS * F_TICK + 30)]:
        fr = frame_at(f)
        p = OUT.replace(".mp4", f"_{name}.png")
        tmp = p + ".raw"
        with open(tmp, "wb") as fh:
            fh.write(fr.tobytes())
        subprocess.run(
            ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-i", tmp, "-vf", "scale=360:-1", p],
            capture_output=True)
        os.remove(tmp)
        print(f"still: {p}")


if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)
    run_checks()
    review_stills()
    if "--ship" in sys.argv:
        encode()
        check_encode()
