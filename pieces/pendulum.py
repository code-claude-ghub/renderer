#!/usr/bin/env python3
"""PENDULUM — a pendulum wave: consecutive integers do everything.

Fifteen pendulums seen from above, hung in a column, all with the same
amplitude. Ball k completes exactly 8+k full swings per 16-second loop —
each one swings exactly ONCE more per loop than the one above it. That
single rule produces the whole show: the column starts as a straight
line, shears into a travelling snake, breaks into groups of four, then
three, then two, passes through an alternating zigzag at half-loop
(where every ball crosses the centre line at the same instant, adjacent
balls moving in opposite directions), and mirrors its way home. The
second half of the loop is the first half reflected: x_k(T - t) =
-x_k(t), exactly.

Classic demo: Berg, Am. J. Phys. 59, 186 (1991). NOT self-referential —
this is a property of the frequencies, not of the render (deliberate
break from the POLE/TRAIN/WHEEL/UNSTIR/MOIRE/WAGON family).

Exactness: positions are table lookups on an integer phase p = C_k * f
mod 480, with the sine table built by symmetry so S[0] = S[240] = 0.0
EXACTLY — the two line moments are exact float facts, and frame 480 is
byte-identical to frame 0. Motion blur is a centred 1/60 s exposure
(midpoint samples on an integer sub-phase grid, so the loop closure
survives blur bitwise); a centred exposure keeps each streak's centroid
at the true instantaneous position by symmetry, so even the blurred
line moments are straight to < 0.05 px.
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
HAIRLINE = 0.825                # faint guides; must stay INVISIBLE to the
                                # centroid instrument (see check 14)

# the wave
N_P = 15                        # pendulums
C0 = 8                          # slowest: 8 cycles per loop (0.5 Hz)
CYCLES = [C0 + k for k in range(N_P)]           # 8..22, consecutive
LOOP = 480                      # frames -> 16.0 s
A = 300.0                       # swing amplitude, px
RB = 26.0                       # ball radius, px
CX = 540
Y0 = 330                        # first ball row
DY = 86                         # row spacing
YS = [Y0 + k * DY for k in range(N_P)]

# camera: centred exposure, half a frame (1/60 s), NS midpoint samples.
# integer sub-phase: sample j of frame f sits at phase
#   u = C * (4*NS*f + 2j + 1 - NS)  mod  4*NS*LOOP
# so equal u -> bitwise-equal sin() -> the loop closes byte-identically
# even through the blur.
NS = 24
M = 4 * NS * LOOP

OUT = f"out/pendulum_{time.strftime('%H%M%S')}.mp4"

# ---------------------------------------------------------------- phase
# sharp-position sine table with EXACT zeros and extrema by construction
S = np.empty(LOOP, np.float64)
for p in range(121):
    S[p] = np.sin(2.0 * np.pi * p / LOOP)
S[0] = 0.0
S[120] = 1.0
for p in range(1, 120):
    S[240 - p] = S[p]
S[240] = 0.0
for p in range(1, 240):
    S[240 + p] = -S[p]


def phase(k, f):
    """Integer phase of ball k at frame f, units of 1/480 rev."""
    return (CYCLES[k] * f) % LOOP


def x_sharp(k, f):
    """Instantaneous centre position at frame f (exact table fact)."""
    return CX + A * S[phase(k, f)]


def sample_x(k, f):
    """NS blur-sample positions for ball k, frame f (exact sub-phase)."""
    c = CYCLES[k]
    u = np.array([(c * (4 * NS * f + 2 * j + 1 - NS)) % M
                  for j in range(NS)], np.float64)
    return CX + A * np.sin(2.0 * np.pi * u / M)


# ---------------------------------------------------------------- render
def ball_cov(xs, y_h):
    """Time-averaged disc coverage: mean of NS discs (what a camera does).

    Returns (x0, cov) — cov over a bbox of height y_h rows. Linear in
    the samples, so the streak's centroid is exactly mean(xs).
    """
    x0 = int(np.floor(xs.min() - RB)) - 2
    x1 = int(np.ceil(xs.max() + RB)) + 3
    xx = np.arange(x0, x1, dtype=np.float64)
    yy = np.arange(y_h, dtype=np.float64) - (y_h - 1) / 2.0
    d = np.hypot(xx[None, None, :] - xs[:, None, None],
                 yy[None, :, None])
    cov = np.clip(RB + 0.5 - d, 0.0, 1.0).mean(axis=0)
    return x0, cov


# ---------------------------------------------------------------- text
FONT = {
    "0": "01110 10001 10011 10101 11001 10001 01110",
    "1": "00100 01100 00100 00100 00100 00100 01110",
    "2": "01110 10001 00001 00010 00100 01000 11111",
    "3": "11111 00010 00100 00010 00001 10001 01110",
    "4": "00010 00110 01010 10010 11111 00010 00010",
    "5": "11111 10000 11110 00001 00001 10001 01110",
    "6": "00110 01000 10000 11110 10001 10001 01110",
    "7": "11111 00001 00010 00100 01000 01000 01000",
    "8": "01110 10001 10001 01110 10001 10001 01110",
    "9": "01110 10001 10001 01111 00001 00010 01100",
}
SCALE = 6


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


LABELS = [text_mask(str(c)) for c in CYCLES]
LABEL_X1 = 150                  # labels live left of this column


def stamp_label(fr, k):
    m = LABELS[k]
    h, w = m.shape
    y0 = YS[k] - h // 2
    x0 = LABEL_X1 - 20 - w
    reg = fr[y0:y0 + h, x0:x0 + w, :]
    reg[...] = reg * (1 - m[..., None]) + INK * m[..., None]


# ---------------------------------------------------------------- frames
Y_TOP = Y0 - int(RB) - 36       # guide/tick extent (ticks end at
Y_BOT = YS[-1] + int(RB) + 36   # rows 238 and 1626: inside safe area)
BAND_H = 2 * int(RB) + 9        # ball bbox height (rows)


def background():
    fr = np.full((H, W, 3), PAPER, np.float64)
    # faint guides: centre line and the two swing limits
    for gx in (CX, int(CX - A), int(CX + A)):
        fr[Y_TOP:Y_BOT, gx - 1:gx + 1, :] = HAIRLINE
    # red centre ticks, clear of every ball band
    for ty in (Y_TOP - 30, Y_BOT + 4):
        fr[ty:ty + 26, CX - 2:CX + 3, :] = RED
    for k in range(N_P):
        stamp_label(fr, k)
    return fr


BG = background()


def frame_at(f):
    fr = BG.copy()
    for k in range(N_P):
        x0, cov = ball_cov(sample_x(k, f), BAND_H)
        y0 = YS[k] - (BAND_H - 1) // 2
        reg = fr[y0:y0 + BAND_H, x0:x0 + cov.shape[1], :]
        reg[...] = reg * (1 - cov[..., None]) + INK * cov[..., None]
    return (np.clip(fr, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def render_frames():
    for f in range(LOOP):
        yield frame_at(f)


# ---------------------------------------------------------------- measure
def centroid(img, k, lo=LABEL_X1 + 30, hi=W - 40):
    """Ink centroid of ball k's row band, read off the RED channel
    (hairlines are grey ~0.825 -> red diff ~5 counts, below the
    threshold; balls are ink -> diff ~200)."""
    y0 = YS[k] - int(RB) - 4
    band = img[y0:y0 + 2 * int(RB) + 9, lo:hi, 0].astype(np.float64)
    wgt = np.clip(PAPER * 255.0 - band - 12.0, 0.0, None).sum(axis=0)
    xs = np.arange(lo, hi, dtype=np.float64)
    return (wgt * xs).sum() / wgt.sum()


def x_pred(k, f):
    """What the centroid instrument SHOULD read: the mean of the blur
    sample positions (linear compositing => exact)."""
    return sample_x(k, f).mean()


# ---------------------------------------------------------------- checks
def run_checks():
    ok = []

    def check(name, cond, detail=""):
        ok.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")

    # 1. the rule: consecutive integer cycle counts
    check("cycle counts are consecutive integers 8..22",
          CYCLES == list(range(8, 23))
          and all(b - a == 1 for a, b in zip(CYCLES, CYCLES[1:])))

    # 2. sine table symmetry is EXACT by construction
    check("table zeros exact: S[0] = S[240] = 0.0",
          S[0] == 0.0 and S[240] == 0.0 and S[120] == 1.0
          and S[360] == -1.0)
    check("table odd symmetry exact: S[480-p] = -S[p] for all p",
          all(S[(LOOP - p) % LOOP] == -S[p] for p in range(LOOP)))

    # 3. each ball completes EXACTLY its integer number of cycles per
    #    loop: count upward centre-crossings of the actual frame sequence
    cross_ok = True
    for k in range(N_P):
        xs = np.array([S[phase(k, f)] for f in range(LOOP)])
        up = int(((xs <= 0) & (np.roll(xs, -1) > 0)).sum())
        if up != CYCLES[k]:
            cross_ok = False
    check("ball k crosses centre upward exactly 8+k times per loop",
          cross_ok)

    # 4. adjacent phase difference = f exactly (the mechanism, integer)
    check("adjacent phase gap = f/480 rev at frame f, all k, all f",
          all((phase(k + 1, f) - phase(k, f)) % LOOP == f % LOOP
              for k in range(N_P - 1) for f in range(0, LOOP, 7)))

    # 5. the two line moments are exact float facts
    check("f=0: all fifteen at x = 540.0 exactly",
          all(x_sharp(k, 0) == float(CX) for k in range(N_P)))
    check("f=240: all fifteen at x = 540.0 exactly",
          all(x_sharp(k, 240) == float(CX) for k in range(N_P)))

    # 6. mirror identity: x(480-f) = -x(f), exact via table symmetry
    check("second half is the first half mirrored, exactly",
          all(S[phase(k, LOOP - f)] == -S[phase(k, f)]
              for k in range(N_P) for f in range(1, LOOP, 11)))

    # 7. loop closes byte-identically THROUGH the blur (integer
    #    sub-phase: frame 480's sample grid equals frame 0's)
    check("frame 480 byte-identical to frame 0 (blur included)",
          np.array_equal(frame_at(0), frame_at(LOOP)))

    # 8. blur convergence: NS=24 midpoint average vs NS=96, worst ball
    kf = N_P - 1                # fastest ball
    f_fast = 6                  # near centre crossing: max speed
    global NS, M
    x24 = sample_x(kf, f_fast)
    NS_old, M_old = NS, M
    NS, M = 96, 4 * 96 * LOOP
    x96 = sample_x(kf, f_fast)
    _, c96 = ball_cov(x96, BAND_H)
    NS, M = NS_old, M_old
    _, c24 = ball_cov(x24, BAND_H)
    dmax = np.abs(c24 - c96[:, :c24.shape[1]]).max() \
        if c24.shape == c96.shape else 1.0
    check("blur converged: NS=24 vs NS=96 coverage agree",
          c24.shape == c96.shape and dmax < 4e-3, f"max {dmax:.2e}")

    # 9. instrument self-test (trap 42): a lone disc at a known offset,
    #    rendered by its own code path, reads back
    probe = np.full((BAND_H + 8, W, 3), PAPER, np.float64)
    xt = 617.3
    xx = np.arange(W, dtype=np.float64)
    yy = np.arange(BAND_H + 8, dtype=np.float64) - (BAND_H + 7) / 2.0
    d = np.hypot(xx[None, :] - xt, yy[:, None])
    cv = np.clip(RB + 0.5 - d, 0.0, 1.0)
    probe = probe * (1 - cv[..., None]) + INK * cv[..., None]
    probe8 = (np.clip(probe, 0, 1) * 255 + 0.5).astype(np.uint8)
    band = probe8[2:2 + 2 * int(RB) + 9, LABEL_X1 + 30:W - 40, 0]
    wgt = np.clip(PAPER * 255.0 - band.astype(np.float64) - 12.0,
                  0, None).sum(axis=0)
    xs = np.arange(LABEL_X1 + 30, W - 40, dtype=np.float64)
    read = (wgt * xs).sum() / wgt.sum()
    check("centroid instrument reads a known disc position",
          abs(read - xt) < 0.02, f"read {read:.3f} vs {xt}")

    # 10. measured centroids vs prediction (blurred: mean of samples),
    #     graded off rendered pixels
    worst = 0.0
    for f in (3, 32, 120, 200, 240, 313, 431):
        img = frame_at(f)
        for k in range(N_P):
            worst = max(worst, abs(centroid(img, k) - x_pred(k, f)))
    check("rendered streak centroids match model", worst < 0.10,
          f"worst {worst:.3f} px over 7 frames x 15 balls")

    # 11. blurred line moment: centred exposure keeps the centroid at
    #     the true position by symmetry, so the line survives the blur
    img240 = frame_at(240)
    worst_line = max(abs(centroid(img240, k) - CX) for k in range(N_P))
    check("f=240 line is straight through the blur", worst_line < 0.05,
          f"worst |x-540| = {worst_line:.4f} px")

    # 12. speed and streak stay readable
    vmax = max(abs(A * (S[phase(k, f + 1)] - S[phase(k, f)]))
               for k in (N_P - 1,) for f in range(LOOP))
    check("fastest ball under 90 px/frame (streak < 2 diameters)",
          vmax < 90.0, f"max {vmax:.1f} px/frame, streak ~{vmax / 2:.0f} px")

    # 13. watch size (trap 67 numeric): 3x-pooled centroids still track
    #     the model at 360 px
    img = frame_at(100)
    worst3 = 0.0
    for k in range(N_P):
        y0 = YS[k] - int(RB) - 4
        band = img[y0:y0 + 54, 180:1080, 0].astype(np.float64)
        pooled = band[:54, :900].reshape(18, 3, 300, 3).mean((1, 3))
        wg = np.clip(PAPER * 255.0 - pooled - 12.0, 0, None).sum(axis=0)
        xs3 = np.arange(300, dtype=np.float64)
        read3 = (wg * xs3).sum() / wg.sum() * 3.0 + 180.0 + 1.0
        worst3 = max(worst3, abs(read3 - x_pred(k, 100)))
    check("centroids readable after 3x downscale", worst3 < 1.5,
          f"worst {worst3:.2f} px (full-res units)")

    # 14. the hairlines are invisible to the instrument (red-channel
    #     depth below the 12-count threshold) but visible to the eye
    f0 = frame_at(0)
    col = f0[YS[7] - 20:YS[7] + 20, int(CX + A) - 1:int(CX + A) + 1, 0]
    depth = PAPER * 255.0 - col.astype(np.float64).mean()
    check("guide lines: visible (>2) but under instrument threshold (<12)",
          2.0 < depth < 12.0, f"depth {depth:.1f} grey")

    # 15. geometry: bands disjoint, labels clear of the swing
    check("ball row bands disjoint", DY > 2 * (int(RB) + 5))
    min_ink_x = CX - A - RB - 3
    check("labels clear of the leftmost swing",
          LABEL_X1 < min_ink_x, f"labels < {LABEL_X1}, swing > "
          f"{min_ink_x:.0f}")

    # 16. safe area (trap 56)
    g = f0.astype(np.float64).mean(axis=2) / 255.0
    ink_rows = np.where((g < 0.75).any(axis=1))[0]
    check("all ink inside safe area",
          ink_rows.min() >= 210 and ink_rows.max() <= 1632,
          f"rows {ink_rows.min()}..{ink_rows.max()}")
    lit = (g > 0.5).mean()
    check("frame neither blank nor solid", 0.40 < lit < 0.99,
          f"lit {lit:.2f}")

    # 17. red ticks present
    tick = f0[Y_TOP - 20, CX, :]
    check("red centre ticks present",
          tick[0] > 150 and tick[0] > tick[1] + 60, f"rgb {tick.tolist()}")

    # 18. timeline
    check("480 frames = 16.0 s", LOOP == 480
          and abs(LOOP / FPS - 16.0) < 1e-12)

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
    print("ENCODE CHECK — measured off the shipped h264:")
    # streak centroids on the shipped file
    worst = 0.0
    for f in (10, 240, 300):
        d = decode_frame(f)
        for k in range(N_P):
            worst = max(worst, abs(centroid(d, k) - x_pred(k, f)))
    print(f"    streak centroids vs model: worst {worst:.3f} px "
          f"(3 frames x 15 balls)")
    assert worst < 0.8, f"centroid drift {worst}"
    # the line moment off the shipped bytes
    d240 = decode_frame(240)
    wl = max(abs(centroid(d240, k) - CX) for k in range(N_P))
    print(f"    f=240 line: worst |x-540| = {wl:.3f} px")
    assert wl < 0.8, "line bent by encode"
    # first frame faithful
    dd = np.abs(decode_frame(0).astype(int) - frame_at(0).astype(int))
    print(f"    decoded f0 vs render: mean |diff| {dd.mean():.3f}, "
          f"max {dd.max()}")
    assert dd.mean() < 2.0, "encode mangled the frame"
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", OUT],
        capture_output=True, text=True).stdout.strip()
    assert int(probe) == LOOP, f"frames {probe} != {LOOP}"
    print(f"    {probe} frames; the wave survives the encode")


def review_stills():
    """Trap 67: look at it at the size it will be watched."""
    for name, f in [("line", 0), ("snake", 40), ("groups", 120),
                    ("zigzag", 252)]:
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
