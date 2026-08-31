#!/usr/bin/env python3
"""WAGON — temporal aliasing: the video's own frame rate as the subject.

Two wheels on paper grey. Top: parked, 0 rev/s. Bottom: spinning at
exactly 30 rev/s — and this video runs at 30 frames per second, so the
wheel advances exactly one revolution between exposures and the two
wheels are BYTE-IDENTICAL, pixel for pixel, frame after frame. The video
is physically incapable of containing the difference.

Then a camera shutter opens (1/90 s) and motion blur becomes the only
evidence that survives the sampling: the half-disc softens, and the ring
of 12 dots smears by exactly four dot-spacings — 30 rev/s * 1/90 s = 1/3
rev = 4/12 — so the dot pattern doesn't blur, it vanishes IDENTICALLY
(its modulation is sinc(12*pi*delta), which is zero at delta = 1/3).

Then the truth about aliasing: drop to 29 rev/s and the wheel creeps
BACKWARD at exactly 1 rev/s; raise to 31 and it creeps forward at 1.
Apparent rate = true rate mod frame rate. The wagon-wheel effect, with
the exact arithmetic shown.

Sibling of MOIRE (yesterday): that was aliasing in space (two pitches
beating), this is aliasing in time (two frequencies beating). Same mod
arithmetic, different axis. Family: the render proves a property about
itself (POLE, TRAIN, WHEEL, UNSTIR, MOIRE).

Exactness: wheel angle is tracked in INTEGER units of 1/30 rev, so
"advances exactly one rev per frame" and the loop closure are integer
facts, not float luck. Exposure blur is a closed-form time average —
each angular feature convolved with a box of width delta = f*T — no
temporal supersampling (trap 72: exact beats quadrature).
"""
import os
import subprocess
import sys
import time

import numpy as np

# ---------------------------------------------------------------- canvas
W, H = 1080, 1920
FPS = 30

PAPER = 0.845                   # trap 69: warm grey, not full white
INK = 0.13
RED = (0.70, 0.16, 0.14)

# wheels
R = 310.0
CX = 540
CYA = 520                       # parked wheel centre
CYB = 1240                      # spinning wheel centre. dy = 720 = 45*16:
                                # macroblock-aligned, so h264 sees the two
                                # identical wheels at the same grid phase
                                # (at dy=715 quantization ringing differed
                                # and read as a false 1.4-grey "difference")
THETA0 = 0.11                   # base angle, rev (arbitrary, non-axis)

# features: (r_in, r_out) bands; angular widths in REVOLUTIONS
RIM = (298.0, 310.0)            # rotationally symmetric: blur-invariant
HUB = (0.0, 30.0)
DISC = (46.0, 158.0)            # half-disc: w = 0.5 rev, bold orientation
W_DISC = 0.5
DOTS = (196.0, 268.0)           # 12 dots, each 1/30 rev wide, 1/12 apart
N_DOTS = 12
W_DOT = 1.0 / 30.0

# camera shutter
T_MAX = 1.0 / 90.0              # seconds. at 30 rev/s: delta = 1/3 rev
                                # = exactly 4 dot-spacings -> the dot
                                # ring's modulation is sinc(12*pi/3) = 0:
                                # the dots vanish identically, not softly.

# ---------------------------------------------------------------- timeline
# (frames, wheel-B rev/s, name). All rates integer; angle bookkeeping is
# integer units of 1/30 rev, so every segment's total rotation is exact.
SEGS = [
    (42, 30, "hold0"),          # both wheels frozen, byte-identical
    (36, 30, "open"),           # shutter sweeps 0 -> 1/90 s: blur blooms
    (30, 30, "blur"),           # frozen AND smeared; dots are gone
    (60, 29, "back"),           # creeps backward at exactly 1 rev/s
    (60, 31, "fwd"),            # creeps forward at exactly 1 rev/s
    (24, 30, "blur2"),          # frozen again
    (36, 30, "close"),          # shutter closes
    (24, 30, "hold1"),          # byte-identical to hold0: loop closure
]
N_FRAMES = sum(s[0] for s in SEGS)              # 312 -> 10.4 s

OUT = f"out/wagon_{time.strftime('%H%M%S')}.mp4"


def seg_of(f):
    for n, rate, name in SEGS:
        if f < n:
            return f, n, rate, name
        f -= n
    raise IndexError(f)


def theta_units(f):
    """Wheel-B angle at frame f's exposure START, integer 1/30-rev units."""
    u = 0
    for n, rate, _ in SEGS:
        step = min(f, n)
        u += step * rate                        # rate/30 rev per frame
        f -= step
        if f <= 0:
            break
    return u % 30


def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def shutter_T(f):
    fi, n, rate, name = seg_of(f)
    if name in ("hold0", "hold1"):
        return 0.0
    if name == "open":
        return T_MAX * smoothstep((fi + 1) / n)
    if name == "close":
        return T_MAX * (1.0 - smoothstep((fi + 1) / n))
    return T_MAX


def frame_params(f):
    """(theta_rev, delta_rev, rate) for wheel B at frame f."""
    _, _, rate, _ = seg_of(f)
    th = THETA0 + theta_units(f) / 30.0
    delta = rate * shutter_T(f)                 # rev swept during exposure
    return th, delta, rate


# ---------------------------------------------------------------- render
# Per-pixel polar closed forms. An angular feature of width w exposed
# while the wheel sweeps delta rev has time-averaged coverage equal to
# box(w) convolved with box(delta)/delta — a trapezoid. Exact; no
# temporal supersampling. Spatial AA: radius gets exact 1-px box
# coverage; angle gets a per-pixel feather eps = 0.6 px / circumference,
# folded in as delta_eff = max(delta, eps) (negligible once delta >> eps).

PAD = 34                        # bbox pad: room for the red 3-o'clock tick
BB = int(R) + PAD               # bbox half-size
_yy, _xx = np.mgrid[-BB:BB + 1, -BB:BB + 1].astype(np.float64)
RGRID = np.hypot(_xx, _yy)
PGRID = (np.arctan2(_yy, _xx) / (2.0 * np.pi)) % 1.0    # rev, y-down: cw+
EPS = 0.6 / (2.0 * np.pi * np.maximum(RGRID, 20.0))     # angular AA, rev


def radial_cov(r_in, r_out):
    return np.clip(np.minimum(r_out, RGRID + 0.5)
                   - np.maximum(r_in, RGRID - 0.5), 0.0, 1.0)


RAD_RIM = radial_cov(*RIM)
RAD_HUB = radial_cov(*HUB)
RAD_DISC = radial_cov(*DISC)
RAD_DOTS = radial_cov(*DOTS)


def trap(u, w, d_eff):
    """box(w) conv box(d_eff)/d_eff at offset u (all in rev, |u| linear)."""
    m = np.minimum(w, d_eff)
    return np.clip((w + d_eff) / 2.0 - np.abs(u), 0.0, m) / d_eff


def wheel_cov(theta, delta):
    """Ink coverage of one wheel at start angle theta, blur span delta."""
    d_eff = np.maximum(delta, EPS)
    cov = RAD_RIM + RAD_HUB
    # half-disc: centred on theta, exposure sweeps [0, delta]
    u = ((PGRID - theta - delta / 2.0) + 0.5) % 1.0 - 0.5
    cov = cov + RAD_DISC * trap(u, W_DISC, d_eff)
    # dots: periodic 1/12 rev; sum nearby periods (support < 4 spacings).
    # dots are disjoint at every instant, so coverages add exactly.
    xr = (PGRID - theta - delta / 2.0) % (1.0 / N_DOTS)
    a = np.zeros_like(xr)
    for k in range(-4, 5):
        a = a + trap(xr + k / float(N_DOTS), W_DOT, d_eff)
    cov = cov + RAD_DOTS * a
    return np.clip(cov, 0.0, 1.0)


# ---------------------------------------------------------------- text
FONT = {
    "0": "01110 10001 10011 10101 11001 10001 01110",
    "1": "00100 01100 00100 00100 00100 00100 01110",
    "2": "01110 10001 00001 00010 00100 01000 11111",
    "3": "11111 00010 00100 00010 00001 10001 01110",
    "9": "01110 10001 10001 01111 00001 00010 01100",
    "r": "00000 00000 10110 11001 10000 10000 10000",
    "e": "00000 00000 01110 10001 11111 10000 01110",
    "v": "00000 00000 10001 10001 10001 01010 00100",
    "s": "00000 00000 01111 10000 01110 00001 11110",
    "/": "00001 00010 00010 00100 01000 01000 10000",
    " ": "00000 00000 00000 00000 00000 00000 00000",
}
SCALE = 8


def text_mask(s):
    rows = []
    for ri in range(7):
        line = []
        for ch in s:
            bits = FONT[ch].split()[ri]
            line.extend(int(b) for b in bits)
            line.append(0)
        rows.append(line[:-1])
    m = np.array(rows, np.float64)
    return np.kron(m, np.ones((SCALE, SCALE)))


def stamp_text(fr, s, cx, y0, val=INK):
    m = text_mask(s)
    h, w = m.shape
    x0 = cx - w // 2
    reg = fr[y0:y0 + h, x0:x0 + w, :]
    reg[...] = reg * (1 - m[..., None]) + val * m[..., None]


# ---------------------------------------------------------------- frames
def paste_wheel(fr, cov, cy):
    y0, x0 = cy - BB, CX - BB
    lum = PAPER + (INK - PAPER) * cov
    fr[y0:y0 + cov.shape[0], x0:x0 + cov.shape[1], :] = lum[..., None]
    # fixed red reference tick at 3 o'clock, outside the rim
    fr[cy - 7:cy + 7, CX + int(R) + 10:CX + int(R) + 26, :] = RED


WHEEL_A = wheel_cov(THETA0, 0.0)                # parked: its own code path
_CACHE = {}


def wheel_b_cov(f):
    th, delta, _ = frame_params(f)
    key = (theta_units(f), round(delta, 12))
    if key not in _CACHE:
        _CACHE[key] = wheel_cov(th, delta)
    return _CACHE[key]


LABEL_Y_A = CYA + int(R) + 30                   # 850
LABEL_Y_B = CYB + int(R) + 18                   # 1563 (font 56 px: <=1619)


def frame_at(f):
    _, _, rate, _ = seg_of(f)
    fr = np.full((H, W, 3), PAPER, np.float64)
    paste_wheel(fr, WHEEL_A, CYA)
    paste_wheel(fr, wheel_b_cov(f), CYB)
    stamp_text(fr, "0 rev/s", CX, LABEL_Y_A)
    stamp_text(fr, f"{rate} rev/s", CX, LABEL_Y_B)
    return (np.clip(fr, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def render_frames():
    for f in range(N_FRAMES):
        yield frame_at(f)


# ---------------------------------------------------------------- measure
def ang_profile(img, cy, r0, r1, nbins=720):
    """Mean grey by angle bin over a ring of the wheel at centre cy."""
    y0, x0 = cy - BB, CX - BB
    patch = img[y0:y0 + 2 * BB + 1, x0:x0 + 2 * BB + 1]
    g = patch.astype(np.float64).mean(axis=2) if patch.ndim == 3 else patch
    sel = (RGRID >= r0) & (RGRID <= r1)
    bins = np.minimum((PGRID[sel] * nbins).astype(int), nbins - 1)
    prof = np.bincount(bins, weights=g[sel], minlength=nbins)
    cnt = np.bincount(bins, minlength=nbins)
    return prof / np.maximum(cnt, 1)


def bin_phase(prof, k=1):
    """Position (rev, mod 1/k) of harmonic k of an angular profile."""
    n = len(prof)
    z = (prof * np.exp(-2j * np.pi * k * np.arange(n) / n)).sum()
    return (-np.angle(z)) / (2.0 * np.pi * k), np.abs(z) / n


def wrap_rev(d, period=1.0):
    return (d + period / 2.0) % period - period / 2.0


def disc_pos(img, cy):
    """Apparent half-disc angle (rev) read off pixels."""
    p, _ = bin_phase(ang_profile(img, cy, 70.0, 140.0), 1)
    return p


def dot_amp(img, cy):
    """12-fold modulation amplitude of the dot ring."""
    _, a = bin_phase(ang_profile(img, cy, 210.0, 254.0), N_DOTS)
    return a


# ---------------------------------------------------------------- checks
def run_checks():
    ok = []

    def check(name, cond, detail=""):
        ok.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")

    # 1. the aliasing identity, exact INTEGER arithmetic (a float version
    #    of this check printed -1.0 and still failed ==; rates and FPS are
    #    integers, so the wrap is an integer fact and gets tested as one)
    for rate, app in ((30, 0), (29, -1), (31, 1)):
        appar = ((rate % FPS) + FPS // 2) % FPS - FPS // 2  # rev/s, wrapped
        check(f"apparent({rate} rev/s) = {app} rev/s exactly",
              appar == app, f"{appar:+d} rev/s")
    # T_MAX = 1/90 s: delta = 30/90 rev; in dot spacings: 30*12/90 = 4
    check("blur span at 30 rev/s = 1/3 rev = exactly 4 dot spacings",
          (30 * N_DOTS) % 90 == 0 and (30 * N_DOTS) // 90 == 4,
          f"delta={30.0 * T_MAX:.6f} rev")

    # 2. angle bookkeeping is integer: every segment's rotation is exact,
    #    and every f=30 frame starts at exactly THETA0
    check("every 30 rev/s frame sits at theta0 (30k = 0 mod 30)",
          all(theta_units(f) == 0 for f in range(N_FRAMES)
              if seg_of(f)[2] == 30))
    tot = sum(rate * n for n, rate, _ in SEGS)
    check("closed loop: total rotation integer revs", tot % 30 == 0,
          f"{tot / 30:.1f} revs over {N_FRAMES} frames")

    # 3. trapezoid math vs independent discrete time-average (1D, no
    #    pixels, no feather: pure box indicator averaged over K offsets)
    rng = np.random.default_rng(7)
    phis = rng.uniform(0, 1, 4000)
    K = 4096
    worst = 0.0
    for delta in (1.0 / 3.0, 29.0 / 90.0, 0.121):
        s = (np.arange(K) + 0.5) * delta / K
        # dots profile
        d = phis[:, None] - s[None, :]
        ink = (np.abs(wrap_rev(d % (1.0 / N_DOTS), 1.0 / N_DOTS))
               <= W_DOT / 2.0).mean(axis=1)
        xr = (phis - delta / 2.0) % (1.0 / N_DOTS)
        pred = sum(np.clip((W_DOT + delta) / 2.0
                           - np.abs(xr + k / N_DOTS), 0,
                           min(W_DOT, delta)) / delta
                   for k in range(-6, 7))
        worst = max(worst, np.abs(ink - pred).max())
        # half-disc profile
        ink2 = (np.abs(wrap_rev(d)) <= W_DISC / 2.0).mean(axis=1)
        u = wrap_rev(phis - delta / 2.0)
        pred2 = np.clip((W_DISC + delta) / 2.0 - np.abs(u), 0,
                        min(W_DISC, delta)) / delta
        worst = max(worst, np.abs(ink2 - pred2).max())
    check("closed-form blur matches discrete time-average (K=4096)",
          worst < 3e-3, f"worst {worst:.2e}")

    # 4. THE claim: parked wheel and 30 rev/s wheel byte-identical at T=0.
    #    WHEEL_A is its own code path (constant theta0, never timeline).
    f0 = frame_at(0)
    # crop the wheel + red tick, NOT the labels (they differ on purpose)
    crop_a = f0[CYA - 315:CYA + 315, CX - 315:CX + 340]
    crop_b = f0[CYB - 315:CYB + 315, CX - 315:CX + 340]
    check("parked and 30 rev/s wheels byte-identical (shutter closed)",
          np.array_equal(crop_a, crop_b))

    # 5. holds are static; loop closes byte-identically
    check("open hold static", np.array_equal(f0, frame_at(30)))
    check("last frame byte-identical to first (loop is an identity)",
          np.array_equal(f0, frame_at(N_FRAMES - 1)))
    b0 = sum(s[0] for s in SEGS[:2])                     # blur seg start
    b1 = sum(s[0] for s in SEGS[:5])                     # blur2 seg start
    check("blurred-and-spinning frames identical (frozen despite 30 rev/s)",
          np.array_equal(frame_at(b0 + 3), frame_at(b0 + 20))
          and np.array_equal(frame_at(b0 + 3), frame_at(b1 + 5)))

    # 6. instrument self-test (trap 42): known rotation reads back
    def patch_img(th, dl):
        c = wheel_cov(th, dl)
        img = (np.clip(PAPER + (INK - PAPER) * c, 0, 1) * 255 + 0.5)
        return img.astype(np.uint8)[..., None].repeat(3, axis=2)

    def patch_pos(th, dl):
        p, _ = bin_phase(ang_profile_patch(patch_img(th, dl), 70, 140), 1)
        return p

    def ang_profile_patch(img, r0, r1, nbins=720):
        g = img.astype(np.float64).mean(axis=2)
        sel = (RGRID >= r0) & (RGRID <= r1)
        bins = np.minimum((PGRID[sel] * nbins).astype(int), nbins - 1)
        prof = np.bincount(bins, weights=g[sel], minlength=nbins)
        cnt = np.bincount(bins, minlength=nbins)
        return prof / np.maximum(cnt, 1)

    d_read = wrap_rev(patch_pos(0.37, 0.0) - patch_pos(0.20, 0.0))
    check("phase instrument reads a known 0.170 rev rotation",
          abs(d_read - 0.17) < 2e-3, f"read {d_read:+.5f} rev")

    # 7. apparent creep measured off rendered pixels, graded against the
    #    TRUE apparent value (trap 72: each instrument gets its own budget)
    back0 = sum(s[0] for s in SEGS[:3])
    fwd0 = sum(s[0] for s in SEGS[:4])
    worst_b = worst_f = 0.0
    for f in range(back0 + 2, back0 + 8):
        d = wrap_rev(disc_pos(frame_at(f + 1), CYB)
                     - disc_pos(frame_at(f), CYB))
        worst_b = max(worst_b, abs(d - (-1.0 / 30.0)))
    for f in range(fwd0 + 2, fwd0 + 8):
        d = wrap_rev(disc_pos(frame_at(f + 1), CYB)
                     - disc_pos(frame_at(f), CYB))
        worst_f = max(worst_f, abs(d - (1.0 / 30.0)))
    check("29 rev/s: creeps backward at exactly -1/30 rev per frame",
          worst_b < 2e-3, f"worst |err| {worst_b:.2e} rev")
    check("31 rev/s: creeps forward at exactly +1/30 rev per frame",
          worst_f < 2e-3, f"worst |err| {worst_f:.2e} rev")

    # 8. the vanish: dot-ring 12-fold modulation is ZERO under full blur
    #    (sinc(12 pi /3) = 0 exactly), and strong when sharp; half-disc
    #    survives at sinc(pi/3) = 0.827 of its sharp amplitude
    a_sharp = dot_amp(f0, CYB)
    fr_blur = frame_at(b0 + 5)
    a_blur = dot_amp(fr_blur, CYB)
    check("12 dots vanish identically under 1/90 s shutter",
          a_blur < 0.02 * a_sharp,
          f"modulation {a_blur:.4f} vs sharp {a_sharp:.4f} "
          f"({a_blur / a_sharp * 100:.2f}%)")
    _, h_sharp = bin_phase(ang_profile(f0, CYB, 70, 140), 1)
    _, h_blur = bin_phase(ang_profile(fr_blur, CYB, 70, 140), 1)
    sinc = np.sin(np.pi / 3.0) / (np.pi / 3.0)
    check("half-disc amplitude ratio = sinc(pi/3) = 0.8270",
          abs(h_blur / h_sharp - sinc) < 0.02,
          f"ratio {h_blur / h_sharp:.4f} pred {sinc:.4f}")
    # parked wheel: shutter open, nothing changes (delta = 0 rev)
    check("parked wheel unchanged by the open shutter",
          np.array_equal(f0[CYA - BB:CYA + BB, :, :],
                         fr_blur[CYA - BB:CYA + BB, :, :]))

    # 9. long baseline: over the 29 rev/s segment the wheel TRULY turns
    #    58 revs; the screen shows -2. difference = 60 revs = frames * 1.
    n_back = SEGS[3][0]
    true_rev = 29 * n_back / 30.0
    app_rev = -1.0 / 30.0 * n_back
    check("true 58 revs, apparent -2 revs, difference = frames exactly",
          true_rev == 58.0 and app_rev == -2.0
          and true_rev - app_rev == n_back,
          f"true {true_rev}, apparent {app_rev}")

    # 10. pixel sanity + safe area (trap 3, trap 56)
    g = f0.astype(np.float64).mean(axis=2) / 255.0
    lit = (g > 0.5).mean()
    check("frame neither blank nor solid", 0.40 < lit < 0.97,
          f"lit {lit:.2f}")
    tick = f0[CYB, CX + int(R) + 12, :]
    check("red reference tick present",
          tick[0] > 150 and tick[0] > tick[1] + 60, f"rgb {tick.tolist()}")
    ink_rows = np.where((g < 0.7).any(axis=1))[0]
    check("all ink inside safe area",
          ink_rows.min() >= 192 and ink_rows.max() <= 1632,
          f"rows {ink_rows.min()}..{ink_rows.max()}")

    # 11. watch size (trap 67 numeric): the dots' 12-fold modulation must
    #     survive the 3x downscale to 360 px, and the blurred ring must
    #     stay dead — measured with the same instrument on pooled pixels
    def pooled_dot_amp(img):
        c = img[CYB - 342:CYB + 342, CX - 342:CX + 342]
        sg = c.astype(np.float64).mean(axis=2)
        sg = sg.reshape(228, 3, 228, 3).mean((1, 3))
        yy, xx = np.mgrid[-114:114, -114:114].astype(np.float64) + 0.5
        rr = np.hypot(xx, yy)
        pp = (np.arctan2(yy, xx) / (2.0 * np.pi)) % 1.0
        sel = (rr >= 65.0) & (rr <= 90.0)
        bins = np.minimum((pp[sel] * 360).astype(int), 359)
        prof = (np.bincount(bins, weights=sg[sel], minlength=360)
                / np.maximum(np.bincount(bins, minlength=360), 1))
        _, amp = bin_phase(prof, N_DOTS)
        return amp
    p_sharp = pooled_dot_amp(f0)
    p_blur = pooled_dot_amp(fr_blur)
    check("dot modulation survives 360 px downscale (and stays dead blurred)",
          p_sharp > 20.0 and p_blur < 0.05 * p_sharp,
          f"sharp {p_sharp:.1f}, blurred {p_blur:.2f} (grey units)")

    # 12. timeline
    check("312 frames = 10.4 s", N_FRAMES == 312
          and abs(N_FRAMES / FPS - 10.4) < 1e-9)

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
    for fr in render_frames():                   # stream (trap 34)
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    assert p.returncode == 0, "ffmpeg failed"
    print(f"encoded {OUT} ({os.path.getsize(OUT)} bytes)")


def decode_frame(n):
    cmd = ["ffmpeg", "-i", OUT, "-vf", f"select=eq(n\\,{n})",
           "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    assert len(raw) == W * H * 3, f"decode size {len(raw)}"
    return np.frombuffer(raw, np.uint8).reshape(H, W, 3)


def check_encode():
    """Measure the identities off the shipped bytes."""
    back0 = sum(s[0] for s in SEGS[:3])
    fwd0 = sum(s[0] for s in SEGS[:4])
    print("ENCODE CHECK — measured off the shipped h264:")
    # the frozen paradox: parked vs spinning wheel on the same frame
    d0 = decode_frame(10)
    # same label-free crop as the render check (the labels differ on
    # purpose; the first draft cropped them in and read the text as a
    # false 1.1-grey "wheel difference")
    ca = d0[CYA - 315:CYA + 315, CX - 315:CX + 340].astype(int)
    cb = d0[CYB - 315:CYB + 315, CX - 315:CX + 340].astype(int)
    dd = np.abs(ca - cb)
    # budget: the RENDER identity is byte-exact (pre-encode check); what
    # survives h264 is bounded by quantization noise, not by the render
    print(f"    parked vs 30 rev/s wheel: mean |diff| {dd.mean():.3f}, "
          f"max {dd.max()}")
    assert dd.mean() < 1.0, "wheels differ on encode"
    # apparent creep at 29 and 31
    for f0i, sgn, name in ((back0 + 10, -1, "29"), (fwd0 + 10, 1, "31")):
        pa = disc_pos(decode_frame(f0i), CYB)
        pb = disc_pos(decode_frame(f0i + 6), CYB)
        d = wrap_rev(pb - pa)
        print(f"    {name} rev/s: apparent motion over 6 frames "
              f"{d:+.5f} rev (model {sgn * 0.2:+.5f})")
        assert abs(d - sgn * 0.2) < 8e-3, f"creep {d}"
    # dot vanish on shipped bytes
    a_sharp = dot_amp(d0, CYB)
    a_blur = dot_amp(decode_frame(back0 - 10), CYB)
    print(f"    dot modulation: sharp {a_sharp:.3f}, blurred {a_blur:.3f} "
          f"({a_blur / a_sharp * 100:.1f}%)")
    assert a_blur < 0.06 * a_sharp, "dots visible under blur on encode"
    # loop
    dl = np.abs(decode_frame(0).astype(int)
                - decode_frame(N_FRAMES - 1).astype(int))
    print(f"    loop: first vs last, mean |diff| {dl.mean():.3f}, "
          f"max {dl.max()}")
    assert dl.mean() < 1.0, "loop not closed on encode"
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", OUT],
        capture_output=True, text=True).stdout.strip()
    assert int(probe) == N_FRAMES, f"frames {probe} != {N_FRAMES}"
    print(f"    {probe} frames; identity holds on the uploaded file")


def review_stills():
    """Trap 67: look at it at the size it will be watched."""
    b0 = sum(s[0] for s in SEGS[:2])
    back0 = sum(s[0] for s in SEGS[:3])
    for name, f in [("open", 4), ("blur", b0 + 10), ("back", back0 + 25)]:
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
