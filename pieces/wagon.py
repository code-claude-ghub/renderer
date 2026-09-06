#!/usr/bin/env python3
"""THE WAGON WHEEL — the camera lies at exactly 2.5 turns per second.

A 12-spoke wheel, rendered at 30 fps by a renderer that IS a 30 fps
camera, so the aliasing below is genuine and not simulated.  When the
wheel turns at exactly 5/2 rev/s, each frame advances it by exactly
    (5/2) / 30  =  1/12 rev  =  one spoke pitch,
so the eleven white spokes land exactly on each other's positions and
the drawn white layer is IDENTICAL from frame to frame — asserted with
np.array_equal, not approximately.  One spoke is painted red: it breaks
the symmetry, so it alone reports the truth, sweeping 30 degrees per
frame while the wheel it belongs to appears frozen.

Either side of the lock the wheel turns at 2.35 and 2.65 rev/s and the
screen shows -0.15 and +0.15 rev/s: the nearest alias.  Both the true
rate and the screen rate are printed on the frame, both derived.

The whole rotation ledger is kept in exact Fractions.  ZENO's trail
wiggle was 30 fps aliasing owned in a footnote; this piece is the
footnote promoted to subject.

Checks: model arithmetic in exact rationals; pixel-level frame identity
at lock (held out: the buffers, not the model); apparent-motion sign
measured by circular cross-correlation of the drawn spokes (the pixels
say backward while the ledger says forward); the shipped h264 re-checked
for the same facts after decode.
"""

import os
import subprocess
import sys
import time
from fractions import Fraction

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- model
W, H = 1080, 1920
FPS = 30
N_SP = 12                          # spokes
R_LOCK = Fraction(5, 2)            # rev/s at which frame step = 1/12 rev
R_LO = Fraction(47, 20)            # 2.35 -> screen shows -0.15
R_HI = Fraction(53, 20)            # 2.65 -> screen shows +0.15

# phases in frames: (end_frame, r_start, r_end) — r linear across phase
PHASES = [
    (12,  Fraction(0), Fraction(0)),       # still, labels settle
    (72,  Fraction(0), R_LO),              # ramp up
    (132, R_LO, R_LO),                     # hold: screen runs backward
    (162, R_LO, R_LOCK),                   # approach
    (258, R_LOCK, R_LOCK),                 # THE LOCK: 96 frames frozen
    (288, R_LOCK, R_HI),                   # leave
    (348, R_HI, R_HI),                     # hold: screen crawls forward
    (372, R_HI, R_HI),                     # end hold, still spinning
]
N = PHASES[-1][0]                  # 372 frames = 12.4 s


def rate_at(n):
    """True rev/s at frame n, exact."""
    prev = 0
    for (end, r0, r1) in PHASES:
        if n <= end:
            span = end - prev
            if span == 0 or r0 == r1:
                return r0
            return r0 + (r1 - r0) * Fraction(n - prev, span)
        prev = end
    return PHASES[-1][2]


def build_revs():
    """rev(n) as exact Fractions, trapezoid over each frame (exact for a
    piecewise-linear rate)."""
    revs = [Fraction(0)]
    for n in range(N):
        dr = (rate_at(n) + rate_at(n + 1)) / 2 / FPS
        revs.append(revs[-1] + dr)
    return revs


REVS = build_revs()


def alias_rate(r):
    """What a 30 fps screen shows for a 12-spoke wheel at r rev/s: the
    rate folded to the nearest alias, in (-FPS/2N, +FPS/2N] rev/s."""
    pitch_rate = Fraction(FPS, N_SP)           # 5/2 rev/s per whole pitch
    half = pitch_rate / 2
    return (r + half) % pitch_rate - half


# ---------------------------------------------------------------- layout
CX, CY = 540.0, 830.0
R_RIM_OUT, R_RIM_IN = 392.0, 362.0
R_HUB = 58.0
R_SPO_IN, R_SPO_OUT = 74.0, 352.0
LW_SPOKE = 14.0

BGC = (0.055, 0.060, 0.078)
C_BONE = (0.90, 0.91, 0.93)
C_RED = (0.90, 0.16, 0.12)
C_LBL = (0.55, 0.57, 0.62)
C_TRUE = (0.85, 0.86, 0.90)

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
OUT_DIR = "/home/maroon-beret/projects/active/youtube/youtube-channel/out"
STAMP = time.strftime("%H%M%S")
OUT_MP4 = f"{OUT_DIR}/wagon_{STAMP}.mp4"

Y_TOP_LBL = 220
Y_TRUE, Y_SCRN, Y_TURN = 1430, 1520, 1640
WHEEL_Y0, WHEEL_Y1 = 380, 1280     # rows owned by the wheel, no text here


# ---------------------------------------------------------------- prims
def comp_bbox(img, x0, y0, cov, color):
    h, w = cov.shape
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x0 + w, W), min(y0 + h, H)
    if x1c <= x0c or y1c <= y0c:
        return
    cv = cov[y0c - y0:y1c - y0, x0c - x0:x1c - x0]
    reg = img[y0c:y1c, x0c:x1c, :]
    col = np.asarray(color, np.float64)
    reg[...] = reg * (1 - cv[..., None]) + col[None, None, :] * cv[..., None]


def seg_cov(xa, ya, xb, yb, lw):
    pad = lw / 2 + 2
    x0 = int(np.floor(min(xa, xb) - pad))
    x1 = int(np.ceil(max(xa, xb) + pad)) + 1
    y0 = int(np.floor(min(ya, yb) - pad))
    y1 = int(np.ceil(max(ya, yb) + pad)) + 1
    xx = np.arange(x0, x1, dtype=np.float64)[None, :] - xa
    yy = np.arange(y0, y1, dtype=np.float64)[:, None] - ya
    dx, dy = xb - xa, yb - ya
    L2 = dx * dx + dy * dy
    if L2 == 0:
        d = np.hypot(xx, yy)
    else:
        t = np.clip((xx * dx + yy * dy) / L2, 0.0, 1.0)
        d = np.hypot(xx - t * dx, yy - t * dy)
    return x0, y0, np.clip(lw / 2 + 0.5 - d, 0.0, 1.0)


def disc_cov(cx_, cy_, r):
    x0, x1 = int(np.floor(cx_ - r)) - 2, int(np.ceil(cx_ + r)) + 3
    y0, y1 = int(np.floor(cy_ - r)) - 2, int(np.ceil(cy_ + r)) + 3
    xx = np.arange(x0, x1, dtype=np.float64)
    yy = np.arange(y0, y1, dtype=np.float64)
    d = np.hypot(xx[None, :] - cx_, yy[:, None] - cy_)
    return x0, y0, np.clip(r + 0.5 - d, 0.0, 1.0)


def ring_cov(cx_, cy_, r_in, r_out):
    x0, x1 = int(np.floor(cx_ - r_out)) - 2, int(np.ceil(cx_ + r_out)) + 3
    y0, y1 = int(np.floor(cy_ - r_out)) - 2, int(np.ceil(cy_ + r_out)) + 3
    xx = np.arange(x0, x1, dtype=np.float64)
    yy = np.arange(y0, y1, dtype=np.float64)
    d = np.hypot(xx[None, :] - cx_, yy[:, None] - cy_)
    return x0, y0, np.clip(np.minimum(r_out + 0.5 - d, d - r_in + 0.5),
                           0.0, 1.0)


def stamp_max(buf, x0, y0, cov):
    h, w = cov.shape
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x0 + w, W), min(y0 + h, H)
    if x1c <= x0c or y1c <= y0c:
        return
    cv = cov[y0c - y0:y1c - y0, x0c - x0:x1c - x0]
    reg = buf[y0c:y1c, x0c:x1c]
    np.maximum(reg, cv, out=reg)


_TEXT_CACHE = {}


def text_cov(s, px):
    key = (s, px)
    if key in _TEXT_CACHE:
        return _TEXT_CACHE[key]
    f = ImageFont.truetype(FONT, px * 4)
    im = Image.new("L", (px * (len(s) + 2) * 4, px * 8), 0)
    ImageDraw.Draw(im).text((8, 8), s, font=f, fill=255)
    a = np.asarray(im, np.float64) / 255.0
    ys, xs = np.where(a > 0)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h4, w4 = a.shape
    h4 -= h4 % 4
    w4 -= w4 % 4
    a = a[:h4, :w4].reshape(h4 // 4, 4, w4 // 4, 4).mean((1, 3))
    _TEXT_CACHE[key] = a
    return a


def put_text(img, s, px, cx_, cy_, color):
    cov = text_cov(s, px)
    h, w = cov.shape
    comp_bbox(img, int(cx_ - w / 2), int(cy_ - h / 2), cov, color)


# ---------------------------------------------------------------- static
BG = np.empty((H, W, 3), np.float64)
BG[..., 0], BG[..., 1], BG[..., 2] = BGC

RIM = np.zeros((H, W), np.float64)
stamp_max(RIM, *ring_cov(CX, CY, R_RIM_IN, R_RIM_OUT))
stamp_max(RIM, *disc_cov(CX, CY, R_HUB))

LBL_TOP = "12 spokes   rendered at 30 fps"
LBL_END = "it never stopped."


# ---------------------------------------------------------------- frame
def spoke_angles(n):
    """Exact fractional turn of each spoke at frame n, as Fractions in
    [0, 1).  Spoke 0 is the red one."""
    return [(REVS[n] + Fraction(j, N_SP)) % 1 for j in range(N_SP)]


def white_layer(n):
    """Rim + hub + ALL twelve spokes.  The red spoke is painted OVER its
    white self, because marking a spoke does not remove it — a first
    version drew only eleven here and check 11 failed: the missing slot
    travelled with the red spoke and the 'frozen' layer secretly carried
    a moving hole.  Drawn full, the layer is genuinely 12-fold symmetric
    and at lock it is bit-identical frame to frame."""
    buf = RIM.copy()
    for a in spoke_angles(n):
        th = 2 * np.pi * float(a)
        ca, sa = np.cos(th), np.sin(th)
        stamp_max(buf, *seg_cov(CX + R_SPO_IN * ca, CY + R_SPO_IN * sa,
                                CX + R_SPO_OUT * ca, CY + R_SPO_OUT * sa,
                                LW_SPOKE))
    return buf


def red_layer(n):
    buf = np.zeros((H, W), np.float64)
    a = spoke_angles(n)[0]
    th = 2 * np.pi * float(a)
    ca, sa = np.cos(th), np.sin(th)
    stamp_max(buf, *seg_cov(CX + R_SPO_IN * ca, CY + R_SPO_IN * sa,
                            CX + R_SPO_OUT * ca, CY + R_SPO_OUT * sa,
                            LW_SPOKE))
    return buf


def labels_for(n):
    r = rate_at(n)
    a = alias_rate(r)
    return (f"true   {float(r):5.2f} rev/s",
            f"screen {float(a):+5.2f} rev/s",
            f"turns  {float(REVS[n]):6.2f}")


def frame_at(n):
    img = BG.copy()
    comp_bbox(img, 0, 0, white_layer(n), C_BONE)
    comp_bbox(img, 0, 0, red_layer(n), C_RED)
    put_text(img, LBL_TOP, 34, W / 2, Y_TOP_LBL, C_LBL)
    lt, ls, lc = labels_for(n)
    put_text(img, lt, 40, W / 2, Y_TRUE, C_TRUE)
    put_text(img, ls, 40, W / 2, Y_SCRN,
             C_RED if abs(alias_rate(rate_at(n))) < rate_at(n) else C_LBL)
    put_text(img, lc, 40, W / 2, Y_TURN, C_TRUE)
    if n >= N - 20:
        put_text(img, LBL_END, 34, W / 2, 1750, C_LBL)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def render_frames():
    for n in range(N):
        yield frame_at(n)
        if n % 60 == 0:
            print(f"  {n}/{N}", flush=True)


# ---------------------------------------------------------------- checks
CHK = [0]


def ok(name, cond, detail=""):
    CHK[0] += 1
    tag = "ok" if cond else "FAIL"
    print(f"  [{CHK[0]:02d}] {tag}  {name}  {detail}", flush=True)
    if not cond:
        sys.exit(1)


def ang_profile(layer, bins=3600):
    """Angular occupancy of a layer sampled on an annulus through the
    spokes — reads the picture, not the model."""
    th = np.arange(bins) * (2 * np.pi / bins)
    prof = np.zeros(bins)
    for rr in (150.0, 220.0, 290.0):
        xs = np.clip((CX + rr * np.cos(th)).astype(int), 0, W - 1)
        ys = np.clip((CY + rr * np.sin(th)).astype(int), 0, H - 1)
        prof += layer[ys, xs]
    return prof


def circ_shift(pa, pb):
    """Shift (in rad) that best maps profile pa onto pb, by circular
    cross-correlation, sub-bin by parabolic refinement."""
    fa, fb = np.fft.rfft(pa), np.fft.rfft(pb)
    xc = np.fft.irfft(fb * np.conj(fa), len(pa))
    k = int(np.argmax(xc))
    y0, y1, y2 = xc[(k - 1) % len(pa)], xc[k], xc[(k + 1) % len(pa)]
    denom = (y0 - 2 * y1 + y2)
    off = 0.0 if denom == 0 else 0.5 * (y0 - y2) / denom
    sh = (k + off) % len(pa)
    if sh > len(pa) / 2:
        sh -= len(pa)
    return sh * 2 * np.pi / len(pa)


def run_checks():
    print("MODEL", flush=True)
    pitch = Fraction(1, N_SP)
    ok("lock step is one spoke pitch exactly",
       R_LOCK / FPS == pitch, f"(5/2)/30 = {R_LOCK / FPS} = 1/12")
    ok("lock rate is FPS/spokes exactly",
       R_LOCK == Fraction(FPS, N_SP), f"{R_LOCK} = 30/12")
    ok("backward hold aliases to -3/20 rev/s exactly",
       alias_rate(R_LO) == Fraction(-3, 20), f"{alias_rate(R_LO)}")
    ok("forward hold aliases to +3/20 rev/s exactly",
       alias_rate(R_HI) == Fraction(3, 20), f"{alias_rate(R_HI)}")
    ok("at lock the screen rate is exactly zero",
       alias_rate(R_LOCK) == 0, "")

    # the ledger: accumulated turns vs closed-form areas, both exact
    total = Fraction(0)
    prev = 0
    for (end, r0, r1) in PHASES:
        total += (r0 + r1) / 2 * Fraction(end - prev, FPS)
        prev = end
    ok("turn ledger closes exactly (trapezoid vs area)",
       REVS[N] == total, f"{float(REVS[N]):.4f} turns = {REVS[N]}")

    lockf = [n for n in range(N) if 162 <= n < 258]
    steps = {REVS[n + 1] - REVS[n] for n in lockf}
    ok("every lock frame advances exactly 1/12 rev",
       steps == {pitch}, f"{len(lockf)} frames, steps={steps}")

    sa, sb = set(spoke_angles(200)), set(spoke_angles(201))
    ok("spoke angle SET identical across lock frames (exact fractions)",
       sa == sb, "12 angles, same set, relabelled")
    ok("red spoke angle NOT identical across lock frames",
       spoke_angles(200)[0] != spoke_angles(201)[0], "")

    r_bound = [rate_at(e) for (e, _, _) in PHASES]
    cont = all(rate_at(PHASES[i][0]) == PHASES[i][1] or True
               for i in range(len(PHASES)))
    ok("rate is continuous at phase boundaries",
       all(abs(float(rate_at(e) - rate_at(e - 1))) < 0.1
           for (e, _, _) in PHASES[:-1]), "")

    print("FRAME", flush=True)
    # THE claim, held out: at lock the drawn white layer is IDENTICAL,
    # bit for bit, frame to frame.  Not close.  Identical.
    for na in (170, 200, 230, 256):
        wa, wb = white_layer(na), white_layer(na + 1)
        ok(f"white layer frames {na}/{na + 1}: np.array_equal",
           np.array_equal(wa, wb), "bit-identical")
    ra, rb = red_layer(200), red_layer(201)
    ok("red layer differs at lock",
       not np.array_equal(ra, rb) and float(np.abs(ra - rb).max()) > 0.5,
       f"max diff {float(np.abs(ra - rb).max()):.2f}")
    da = circ_shift(ang_profile(ra), ang_profile(rb))
    ok("red spoke advances one pitch per lock frame (from pixels)",
       abs(da - 2 * np.pi / N_SP) < 0.02,
       f"measured {da:.4f} rad vs 2pi/12 = {2 * np.pi / 12:.4f}")

    # the lie, measured off the pixels: during the backward hold the
    # white pattern moves NEGATIVE while the ledger moves positive.
    # A 12-fold pattern defines its own shift only modulo one pitch —
    # the correlator is subject to the same aliasing as the screen,
    # which is the entire point — so the raw peak is folded into
    # (-pitch/2, pitch/2] before it means anything.
    pitch_rad = 2 * np.pi / N_SP

    def fold(sh):
        return (sh + pitch_rad / 2) % pitch_rad - pitch_rad / 2

    wa, wb = white_layer(100), white_layer(101)
    sh = fold(circ_shift(ang_profile(wa), ang_profile(wb)))
    true_step = 2 * np.pi * float(REVS[101] - REVS[100])
    want = 2 * np.pi * float(alias_rate(R_LO)) / FPS
    ok("backward hold: pixels move backward",
       sh < -0.02, f"measured {sh:.4f} rad/frame")
    ok("backward hold: pixel shift equals the alias, not the truth",
       abs(sh - want) < 0.004 and abs(sh - true_step) > 0.4,
       f"pixels {sh:.4f}, alias {want:.4f}, truth {true_step:.4f}")
    wa, wb = white_layer(320), white_layer(321)
    sh2 = fold(circ_shift(ang_profile(wa), ang_profile(wb)))
    want2 = 2 * np.pi * float(alias_rate(R_HI)) / FPS
    ok("forward hold: pixels crawl forward at the alias rate",
       sh2 > 0.02 and abs(sh2 - want2) < 0.004,
       f"pixels {sh2:.4f}, alias {want2:.4f}")

    # the wheel itself, read back
    prof = ang_profile(white_layer(0) + red_layer(0))
    f12 = np.abs(np.fft.rfft(prof))[N_SP]
    base = np.abs(np.fft.rfft(prof))[1:N_SP].max()
    ok("12 spokes, counted by the 12th harmonic of the profile",
       f12 > 3 * base, f"|H12| {f12:.0f} vs strongest lower {base:.0f}")

    fr = frame_at(200)
    red = ((fr[..., 0] > 150) & (fr[..., 1] < 90)
           & (fr[..., 2] < 90))
    red[:WHEEL_Y0] = False          # the screen readout is red on purpose
    red[WHEEL_Y1:] = False          # during the lie; this check is about
    ys, xs = np.where(red)          # the wheel's own red, so scope to it
    dist = np.hypot(xs - CX, ys - CY)
    ok("red pixels exist and live inside the spoke annulus",
       len(ys) > 400 and dist.max() < R_SPO_OUT + 12
       and dist.min() > R_SPO_IN - 12,
       f"{len(ys)} px, r {dist.min():.0f}..{dist.max():.0f}")

    lt, ls, _ = labels_for(200)
    ok("lock readout says true 2.50, screen +0.00",
       lt == "true    2.50 rev/s" and ls == "screen +0.00 rev/s",
       f"'{lt}' / '{ls}'")
    lt, ls, _ = labels_for(100)
    ok("backward readout says true 2.35, screen -0.15",
       lt == "true    2.35 rev/s" and ls == "screen -0.15 rev/s",
       f"'{lt}' / '{ls}'")

    for n in (0, 60, 100, 200, 300, N - 1):
        fr = frame_at(n)
        ink = (np.abs(fr.astype(float) / 255.0 - np.array(BGC)).sum(2)
               > 0.10).mean()
        ok(f"ink fraction frame {n} sane", 0.02 < ink < 0.5,
           f"{ink:.3f}")

    print(f"ALL {CHK[0]} CHECKS PASSED", flush=True)


# ---------------------------------------------------------------- encode
def encode():
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "18",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT_MP4]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    for fr in render_frames():
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        print("ENCODE FAILED", flush=True)
        sys.exit(1)
    print(f"encoded {OUT_MP4} ({os.path.getsize(OUT_MP4)} bytes)",
          flush=True)


def decode_frame(n):
    r = subprocess.run(
        ["ffmpeg", "-i", OUT_MP4, "-vf", f"select=eq(n\\,{n})",
         "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True)
    return np.frombuffer(r.stdout, np.uint8).reshape(H, W, 3)


def check_encode():
    print("ENCODE CHECKS", flush=True)
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames",
                        "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_frames", "-of", "csv=p=0",
                        OUT_MP4], capture_output=True, text=True)
    ok("shipped file has all frames", r.stdout.strip() == str(N),
       f"{r.stdout.strip()}/{N}")
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                        "-show_entries", "stream=codec_name", "-of",
                        "csv=p=0", OUT_MP4], capture_output=True,
                       text=True)
    ok("no audio stream — this one is silent on purpose",
       r.stdout.strip() == "", "")

    # the lock survives h264: consecutive decoded frames differ only in
    # the red spoke and the counter digits.  Measure it, then bound it.
    da, db = decode_frame(200), decode_frame(201)
    wheel = np.zeros((H, W), bool)
    wheel[WHEEL_Y0:WHEEL_Y1, :] = True
    ra = red_layer(200) + red_layer(201)
    reddil = ra > 0.01
    for _ in range(3):
        reddil = (reddil | np.roll(reddil, 1, 0) | np.roll(reddil, -1, 0)
                  | np.roll(reddil, 1, 1) | np.roll(reddil, -1, 1))
    still = wheel & ~reddil
    d_still = np.abs(da.astype(int) - db.astype(int))[still].mean()
    d_red = np.abs(da.astype(int) - db.astype(int))[wheel & reddil].mean()
    ok("decoded lock: white wheel statistically still",
       d_still < 0.6, f"mean |diff| {d_still:.3f}/255 off-red")
    ok("decoded lock: red spoke plainly moves",
       d_red > 12.0, f"mean |diff| {d_red:.1f}/255 in red region")

    db2 = decode_frame(100)
    bright = (db2.astype(int).sum(2) > 330)
    ys, xs = np.where(bright[WHEEL_Y0:WHEEL_Y1])
    ok("decoded wheel present", len(ys) > 20000, f"{len(ys)} bright px")
    print("ENCODE CHECKS PASSED — DONE", flush=True)


def review_stills():
    for n in (6, 60, 100, 200, 256, 320):
        Image.fromarray(frame_at(n)).save(f"{OUT_DIR}/wagon_f{n:03d}.png")
    subprocess.run(
        ["ffmpeg", "-y", "-pattern_type", "glob",
         "-i", f"{OUT_DIR}/wagon_f*.png",
         "-filter_complex", "scale=270:-1,tile=3x2",
         f"{OUT_DIR}/wagon_sheet.png"],
        capture_output=True)
    print("sheet:", f"{OUT_DIR}/wagon_sheet.png", flush=True)


if __name__ == "__main__":
    run_checks()
    encode()
    check_encode()
    review_stills()
