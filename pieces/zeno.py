#!/usr/bin/env python3
"""ZENO'S BALL — infinitely many bounces, and they end on schedule.

The channel's first sonification piece: the audio is not a bed under
the physics, the audio IS the claim. A ball dropped from 1.225 m with
restitution e = 0.8 bounces infinitely many times, and the whole
infinity completes in EXACTLY 4.5 seconds:

    t0 = sqrt(2 h0 / g) = 1/2 s          (g = 9.8, h0 = 49/40 m)
    T  = t0 (1+e)/(1-e) = 9/2 s          (exact, rational)

Impact k (k = 0, 1, 2, ...) lands at   B_k = 9/2 - 4 * (4/5)^k
so the intervals are EXACTLY (4/5)^k seconds: 0.8, 0.64, 0.512, ...

The ball drifts right at constant vx, so the bounce marks on the
ground draw the geometric series spatially, converging on a red line
drawn at x_inf BEFORE the drop. Each bounce is a click whose
amplitude is proportional to the impact speed (declared choice).
Around bounce 14 the click interval crosses 1/20 s and the train
becomes a pitch; the first 48 intervals span at least one sample at
48 kHz, and the remaining infinitely many bounces complete inside a
single sample. After t = 4.5 s the track is silence BY CONSTRUCTION
(asserted: exact zeros in the wav, floor-ratio on the decoded aac).
No fade-out is applied anywhere — a fade would fake the ending.

Idealisation, declared in the description: instantaneous contact,
constant e, no air. A real ball's finite contact time swamps the late
intervals, so real bounces are finitely many; the infinity belongs to
the ideal ball.

  PRE  (n   0.. 35): ball held at 1.225 m; red line + "4.5 s" label.
  FALL (n  36..170): physics t = (n-36)/30; bounces, ticks, clicks.
  INF  (n 171..178): t = 4.5 exactly at n = 171; counter flips to
                     the infinity sign; the red line flashes white.
  ROLL (n 171..199): ball rolls right at vx, exits the frame.
  END  (n 200..249): the finished picture; closing caption.
"""
import math
import os
import subprocess
import sys
import time
import wave
from fractions import Fraction

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- model
W, H = 1080, 1920
FPS = 30
N = 250
N_DROP = 36                       # frame of release; phys t = (n-36)/30
N_INF = 171                       # 36 + 135 = 4.5 s of physics, exact

E = 0.8                           # restitution (exactly 4/5)
T0 = 0.5                          # first fall time (exactly 1/2)
G = 9.8
V0 = 4.9                          # g * t0, impact-0 speed
H0 = 1.225                        # 49/40 m
T_INF = 4.5                       # t0 (1+e)/(1-e), exact — see check 1

SCALE = 600.0                     # px per metre
YG = 1500                         # ground row (top of ground line)
R_BALL = 22.0
X0, VX = 120.0, 180.0             # x(t) = X0 + VX*t   (px, px/s)
X_INF = X0 + VX * T_INF           # 930.0 — the line, drawn before the drop

LW_TRAIL, LW_MARK, LW_GROUND = 5.0, 5.0, 4.0
TICK_W, TICK_H = 4.0, 20.0

BGC = (0.055, 0.060, 0.078)
C_BALL = (0.99, 0.70, 0.16)
C_TRAIL = (0.28, 0.33, 0.45)
C_GROUND = (0.42, 0.44, 0.50)
C_TICK = (0.93, 0.94, 0.96)
C_MARK = (0.88, 0.18, 0.14)
C_FLASH = (0.98, 0.98, 0.99)
C_LBL = (0.55, 0.57, 0.62)
C_CNT = (0.85, 0.86, 0.90)

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
OUT_DIR = "/home/maroon-beret/projects/active/youtube/youtube-channel/out"
STAMP = time.strftime("%H%M%S")
OUT_MP4 = f"{OUT_DIR}/zeno_{STAMP}.mp4"       # silent video
OUT_WAV = f"{OUT_DIR}/zeno_{STAMP}.wav"
OUT_FIN = f"{OUT_DIR}/zeno_{STAMP}_final.mp4"  # muxed, the shipped file

SR = 48000
CLICK_F = 620.0                   # damped-sine fundamental, Hz
CLICK_DECAY = 125.0               # 1/s  (~8 ms time constant)
CLICK_LEN = int(0.033 * SR)       # 33 ms tail
K_MAX = 56                        # amplitude 0.8^56 ~ 4e-6: below hearing
A0 = 0.55


def bounce_time(k):
    """Impact k lands at B_k = 4.5 - 4 * 0.8^k  (k = 0, 1, 2, ...)."""
    return T_INF - 4.0 * (E ** k)


def n_bounces(t):
    """How many impacts have happened by physics time t (t < T_INF)."""
    if t < T0:
        return 0
    # B_k <= t  <=>  0.8^k >= (4.5 - t)/4  <=>  k <= log_{0.8}((4.5-t)/4)
    k = math.log((T_INF - t) / 4.0) / math.log(E)
    n = int(math.floor(k + 1e-12)) + 1
    # guard the floor against float edges by checking the neighbours
    while n >= 1 and bounce_time(n - 1) > t:
        n -= 1
    while bounce_time(n) <= t:
        n += 1
    return n


def height(t):
    """Ball height in metres at physics time t. Exact piecewise parabolas."""
    if t < T0:
        return H0 - 0.5 * G * t * t
    if t >= T_INF:
        return 0.0
    n = n_bounces(t)              # impacts so far; current segment is n-1
    k = n - 1
    tau = t - bounce_time(k)
    u = V0 * (E ** (k + 1))       # rebound speed off impact k
    return max(0.0, u * tau - 0.5 * G * tau * tau)


def ball_xy(n):
    """Screen centre of the ball at frame n."""
    t = max(0.0, (n - N_DROP) / FPS)
    x = X0 + VX * t
    y = YG - R_BALL - SCALE * height(t)
    return x, y


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


def disc_cov(cx_, cy_, r):
    x0, x1 = int(np.floor(cx_ - r)) - 2, int(np.ceil(cx_ + r)) + 3
    y0, y1 = int(np.floor(cy_ - r)) - 2, int(np.ceil(cy_ + r)) + 3
    xx = np.arange(x0, x1, dtype=np.float64)
    yy = np.arange(y0, y1, dtype=np.float64)
    d = np.hypot(xx[None, :] - cx_, yy[:, None] - cy_)
    return x0, y0, np.clip(r + 0.5 - d, 0.0, 1.0)


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


def box_cov(x0f, y0f, x1f, y1f):
    x0, y0 = int(np.floor(x0f)) - 1, int(np.floor(y0f)) - 1
    x1, y1 = int(np.ceil(x1f)) + 1, int(np.ceil(y1f)) + 1
    xx = np.arange(x0, x1, dtype=np.float64)
    yy = np.arange(y0, y1, dtype=np.float64)
    cx_ = np.clip(np.minimum(xx - x0f + 0.5, x1f - xx + 0.5), 0, 1)
    cy_ = np.clip(np.minimum(yy - y0f + 0.5, y1f - yy + 0.5), 0, 1)
    return x0, y0, cy_[:, None] * cx_[None, :]


def stamp_max(buf, x0, y0, cov):
    h, w = cov.shape
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x0 + w, W), min(y0 + h, H)
    if x1c <= x0c or y1c <= y0c:
        return
    cv = cov[y0c - y0:y1c - y0, x0c - x0:x1c - x0]
    reg = buf[y0c:y1c, x0c:x1c]
    np.maximum(reg, cv, out=reg)


def text_cov(s, px):
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
    return a


def put_text(img, s, px, cx_, cy_, color):
    cov = text_cov(s, px)
    h, w = cov.shape
    comp_bbox(img, int(cx_ - w / 2), int(cy_ - h / 2), cov, color)


# ---------------------------------------------------------------- static
BG = np.empty((H, W, 3), np.float64)
BG[..., 0], BG[..., 1], BG[..., 2] = BGC

# ground line and the pre-announced convergence line, built once
GROUND = np.zeros((H, W), np.float64)
stamp_max(GROUND, *box_cov(60.0, YG, 1020.0, YG + LW_GROUND))
MARK = np.zeros((H, W), np.float64)
stamp_max(MARK, *box_cov(X_INF - LW_MARK / 2, 880.0,
                         X_INF + LW_MARK / 2, YG + 30.0))

LBL_TOP = "drop 1.225 m      e = 0.8"
LBL_MARK = "4.5 s"
LBL_END1 = "every bounce is over."
LBL_END2 = "all infinitely many of them."


# ---------------------------------------------------------------- state
def new_bufs():
    return (np.zeros((H, W), np.float64),      # trail
            np.zeros((H, W), np.float64))      # ticks


def append_trail(tr, n):
    if n <= N_DROP:
        return
    xa, ya = ball_xy(n - 1)
    xb, yb = ball_xy(n)
    stamp_max(tr, *seg_cov(xa, ya, xb, yb, LW_TRAIL))


def append_ticks(tk, n):
    """Stamp a tick for every impact that happened during frame n."""
    t_prev = max(0.0, (n - 1 - N_DROP) / FPS)
    t_now = max(0.0, (n - N_DROP) / FPS)
    if t_now <= 0.0:
        return
    k = 0
    while k <= K_MAX:
        bt = bounce_time(k)
        if t_prev < bt <= t_now:
            x = X0 + VX * bt
            stamp_max(tk, *box_cov(x - TICK_W / 2, YG + 8,
                                   x + TICK_W / 2, YG + 8 + TICK_H))
        k += 1


def counter_label(n):
    if n < N_DROP:
        return "bounces 0"
    t = (n - N_DROP) / FPS
    if t >= T_INF - 1e-9:
        return "bounces ∞"
    return f"bounces {n_bounces(t)}"


def composite(n, tr, tk):
    img = BG.copy()
    # trail under everything moving
    a = tr[..., None]
    img = img * (1 - a) + np.asarray(C_TRAIL)[None, None, :] * a
    comp_bbox(img, 0, 0, GROUND, C_GROUND)
    comp_bbox(img, 0, 0, tk, C_TICK)
    mark_c = C_FLASH if N_INF <= n < N_INF + 8 else C_MARK
    comp_bbox(img, 0, 0, MARK, mark_c)
    put_text(img, LBL_TOP, 30, W / 2, 262, C_LBL)
    put_text(img, LBL_MARK, 30, X_INF, 850, C_MARK)
    put_text(img, counter_label(n), 40, W / 2, 392, C_CNT)
    if n >= 205:
        put_text(img, LBL_END1, 34, W / 2, 580, C_LBL)
        put_text(img, LBL_END2, 34, W / 2, 646, C_LBL)
    x, y = ball_xy(n)
    if x < W + R_BALL + 4:
        comp_bbox(img, *disc_cov(x, y, R_BALL), C_BALL)
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def frame_at(n):
    tr, tk = new_bufs()
    for m in range(1, n + 1):
        append_trail(tr, m)
        append_ticks(tk, m)
    return composite(n, tr, tk)


def render_frames():
    tr, tk = new_bufs()
    for n in range(N):
        if n >= 1:
            append_trail(tr, n)
            append_ticks(tk, n)
        yield composite(n, tr, tk)


# ---------------------------------------------------------------- audio
def build_audio():
    """Sample-accurate click per bounce; amplitude ~ impact speed.
    NO fades anywhere: after the last click's tail the samples are
    exact zeros, because the silence is the piece's claim."""
    n_tot = int(round(N / FPS * SR))
    x = np.zeros(n_tot, np.float64)
    tau = np.arange(CLICK_LEN) / SR
    click = np.exp(-CLICK_DECAY * tau) * np.sin(2 * np.pi * CLICK_F * tau)
    starts = []
    for k in range(K_MAX + 1):
        t = N_DROP / FPS + bounce_time(k)
        s = int(round(t * SR))
        starts.append(s)
        seg = x[s:s + CLICK_LEN]
        seg += (A0 * (E ** k)) * click[:len(seg)]
    peak = np.abs(x).max()
    x *= 0.7 / peak
    data = (x * 32767.0).astype(np.int16)
    with wave.open(OUT_WAV, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())
    return starts


# ---------------------------------------------------------------- checks
CHECKS = {"pass": 0, "fail": 0}


def ok(name, cond, detail=""):
    tag = "ok  " if cond else "FAIL"
    CHECKS["pass" if cond else "fail"] += 1
    print(f"  {tag} {name}" + (f"  [{detail}]" if detail else ""), flush=True)


def mask_of(fr, lo, hi):
    r, g, b = fr[..., 0].astype(int), fr[..., 1].astype(int), \
        fr[..., 2].astype(int)
    return ((r >= lo[0]) & (r <= hi[0]) & (g >= lo[1]) & (g <= hi[1])
            & (b >= lo[2]) & (b <= hi[2]))


def ball_mask(fr):
    return mask_of(fr, (215, 140, 0), (255, 210, 90))


def tick_mask(fr):
    return mask_of(fr, (200, 200, 200), (255, 255, 255))


def mark_mask(fr):
    return mask_of(fr, (170, 0, 0), (255, 90, 80))


def run_checks():
    print("== model checks ==", flush=True)

    # 1. the total time is exact, in rationals — no float in the claim
    t0, e = Fraction(1, 2), Fraction(4, 5)
    T = t0 * (1 + e) / (1 - e)
    ok("T = t0(1+e)/(1-e) = 9/2 exactly", T == Fraction(9, 2), str(T))
    ok("B_k = 9/2 - 4 e^k closed form (rational)",
       all(t0 + sum(2 * t0 * e ** j for j in range(1, k + 1))
           == Fraction(9, 2) - 4 * e ** k for k in range(40)))

    # 2. float closed form vs direct partial sums
    err = max(abs(bounce_time(k)
                  - (T0 + sum(E ** j for j in range(1, k + 1))))
              for k in range(60))
    ok("float B_k vs partial sums", err < 1e-9, f"max err {err:.2e}")
    ok("intervals are exactly 0.8^k",
       all(abs((bounce_time(k + 1) - bounce_time(k)) - E ** (k + 1)) < 1e-9
           for k in range(50)))

    # 3. trajectory: y(B_k) == 0, apexes on h0 * e^(2k)
    ok("height zero at every impact",
       all(height(bounce_time(k) + 1e-12) < 1e-6 for k in range(20)))
    apex_err = 0.0
    for k in range(1, 12):
        tm = bounce_time(k - 1) + 0.5 * E ** k   # apex of segment k
        apex_err = max(apex_err, abs(height(tm) - H0 * E ** (2 * k)))
    ok("apex heights h0 e^{2k}", apex_err < 1e-9, f"max err {apex_err:.2e}")

    # 4. the count function vs brute force
    ok("n_bounces vs brute force",
       all(n_bounces(t) == sum(1 for k in range(200)
                               if bounce_time(k) <= t)
           for t in np.linspace(0.0, 4.4999, 977)))
    ok("counter's last finite value is 22",
       n_bounces((N_INF - 1 - N_DROP) / FPS) == 22,
       f"{n_bounces((N_INF - 1 - N_DROP) / FPS)}")

    # 5. geometry: the line is where the physics says, drawn from t=0
    ok("x_inf = X0 + VX*T = 930", abs(X_INF - 930.0) < 1e-9)
    ok("frame N_INF is physics t = 4.5 exactly",
       (N_INF - N_DROP) / FPS == 4.5)
    ok("ball apex clears safe area",
       YG - R_BALL - SCALE * H0 > 0.10 * H + 40,
       f"apex row {YG - R_BALL - SCALE * H0:.0f}")

    print("== render checks (buffer) ==", flush=True)
    f60 = frame_at(60)                      # phys 0.8: mid-flight
    bm = ball_mask(f60)
    ys, xs = np.where(bm)
    t = (60 - N_DROP) / FPS
    ex, ey = X0 + VX * t, YG - R_BALL - SCALE * height(t)
    ok("ball on its parabola (frame 60)",
       bm.sum() > 900 and abs(xs.mean() - ex) < 2 and abs(ys.mean() - ey) < 2,
       f"{bm.sum()} px, c=({xs.mean():.1f},{ys.mean():.1f}) "
       f"exp=({ex:.1f},{ey:.1f})")

    fin = frame_at(240)                     # the finished picture
    ok("ball has left the frame (frame 240)", ball_mask(fin).sum() == 0)

    # trap 80: probe model-given points, not screen bands.
    band = fin[YG + 8:YG + 8 + int(TICK_H), :]
    tickcols = tick_mask(band)
    hits = 0
    for k in range(10):
        x = int(round(X0 + VX * bounce_time(k)))
        if tickcols[:, x - 2:x + 3].any():
            hits += 1
    ok("ticks at the first 10 impact points", hits == 10, f"{hits}/10")
    miss = 0
    for k in range(6):                       # midpoints must be empty
        xm = int(round(X0 + VX * (bounce_time(k) + bounce_time(k + 1)) / 2))
        if not tickcols[:, xm - 1:xm + 2].any():
            miss += 1
    ok("no tick between the first 6 pairs", miss == 6, f"{miss}/6")

    # HELD OUT: read the ratio off the picture. Tick centroids from the
    # buffer, successive gaps, fit the common ratio — the render must
    # hand back e without being told it.
    cols = np.where(tickcols.any(0))[0]
    groups = np.split(cols, np.where(np.diff(cols) > 3)[0] + 1)
    cents = sorted(float(g.mean()) for g in groups if len(g) >= 2)
    cents = [c for c in cents if c < X_INF - 8]      # clear of the red line
    gaps = np.diff(cents[:9])
    ratios = gaps[1:] / gaps[:-1]
    ok("HELD OUT: gap ratio read from pixels == e",
       len(gaps) >= 6 and float(np.abs(ratios - E).max()) < 0.03,
       f"{len(gaps)} gaps, ratios {np.round(ratios, 3)}")

    mk = mark_mask(fin)
    ys2, xs2 = np.where(mk)
    ok("red line at x = 930 on the buffer",
       mk.sum() > 1500 and abs(xs2.mean() - X_INF) < 3.0,
       f"{mk.sum()} px at x={xs2.mean():.1f}")

    f171 = frame_at(N_INF)
    bx, by = ball_xy(N_INF)
    ok("ball centre ON the line at t = 4.5",
       abs(bx - X_INF) < 1e-9 and abs(by - (YG - R_BALL)) < 1e-9,
       f"({bx:.1f},{by:.1f})")
    ok("line flashes white at t = 4.5",
       mask_of(f171, (240, 240, 240), (255, 255, 255)).sum() > 800)

    # counter changes; end caption obeys the safe area; some ink overall
    c1, c2 = counter_label(80), counter_label(110)
    ok("counter advances", c1 != c2, f"{c1!r} -> {c2!r}")
    lit = (fin.astype(int).sum(2) > 3 * 40)
    frac = lit.mean()
    ok("ink fraction sane", 0.005 < frac < 0.30, f"{frac:.3f}")
    rows = np.where(lit.any(1))[0]
    f240 = frame_at(240)
    lit240 = (f240.astype(int).sum(2) > 3 * 40)
    trows = np.where(lit240.any(1))[0]
    ok("all ink inside safe area",
       rows.min() > 0.10 * H and trows.max() < 0.85 * H,
       f"rows {rows.min()}..{trows.max()}")

    print("== audio checks ==", flush=True)
    starts = build_audio()
    with wave.open(OUT_WAV, "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    x = data.astype(np.float64) / 32767.0

    # click k begins at the sample the physics dictates
    terr = 0.0
    for k in range(12):
        t_exp = N_DROP / FPS + bounce_time(k)
        s0 = int(round(t_exp * SR))
        w0 = np.abs(x[s0 - 960:s0 + 960])
        onset = s0 - 960 + int(np.argmax(w0 > 0.05 * w0.max()))
        terr = max(terr, abs(onset - s0) / SR)
    ok("first 12 clicks at B_k (sample-accurate)", terr < 0.002,
       f"max err {terr * 1000:.2f} ms")

    # distinct starting samples: the file resolves ~48 bounces
    uniq = len(set(starts))
    ok("~48 bounces get distinct samples", 45 <= uniq <= K_MAX + 1,
       f"{uniq} distinct of {K_MAX + 1} placed")

    # the silence is EXACT — no fade faked it
    s_end = int((N_DROP / FPS + T_INF + 0.06) * SR)
    ok("exact zeros after the last click's tail",
       np.abs(x[s_end:]).max() == 0.0,
       f"{len(x) - s_end} samples of true zero")
    ok("wav length matches the video",
       abs(len(x) / SR - N / FPS) < 0.001, f"{len(x) / SR:.3f} s")

    amp = np.abs(x).max()
    ok("peak level sane", 0.5 < amp <= 0.71, f"{amp:.3f}")

    print()
    if CHECKS["fail"]:
        print(f"{CHECKS['fail']} CHECK FAILURES")
        sys.exit(1)
    print(f"ALL {CHECKS['pass']} CHECKS PASSED", flush=True)


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
    # mux with NO fades — asciilib.add_audio applies a fade-out, which
    # would manufacture the exact silence this piece claims to prove.
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", OUT_MP4, "-i", OUT_WAV,
         "-map", "0:v", "-map", "1:a", "-c:v", "copy",
         "-c:a", "aac", "-b:a", "192k", "-shortest",
         "-movflags", "+faststart", OUT_FIN],
        check=True)
    print(f"encoded {OUT_FIN} ({os.path.getsize(OUT_FIN)} bytes)",
          flush=True)


def decode_frame(n):
    r = subprocess.run(
        ["ffmpeg", "-i", OUT_FIN, "-vf", f"select=eq(n\\,{n})",
         "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True)
    return np.frombuffer(r.stdout, np.uint8).reshape(H, W, 3)


def check_encode():
    print("== encode checks ==", flush=True)
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-count_frames", "-select_streams",
         "v", "-show_entries",
         "stream=nb_read_frames,width,height,r_frame_rate",
         "-of", "csv=p=0", OUT_FIN], capture_output=True, text=True)
    print("ffprobe:", r.stdout.strip(), flush=True)
    ok("250 frames in the file", f"{N}" in r.stdout)
    r2 = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "a",
         "-show_entries", "stream=codec_name,sample_rate",
         "-of", "csv=p=0", OUT_FIN], capture_output=True, text=True)
    ok("aac audio stream present", "aac" in r2.stdout, r2.stdout.strip())

    d240 = decode_frame(240)
    # ticks survive h264 at model-given points (traps 78/79: claim only
    # the wide early ticks on the decode; topology stays on the buffer)
    band = d240[YG + 6:YG + 32, :]
    tm = tick_mask(band)
    hits = sum(1 for k in range(8)
               if tm[:, int(round(X0 + VX * bounce_time(k))) - 3:
                     int(round(X0 + VX * bounce_time(k))) + 4].any())
    ok("first 8 ticks survive h264", hits == 8, f"{hits}/8")
    mk = mark_mask(d240)
    ys, xs = np.where(mk)
    ok("red line survives h264", mk.sum() > 1000 and
       abs(xs.mean() - X_INF) < 3.5, f"{mk.sum()} px x={xs.mean():.1f}")
    d60 = decode_frame(60)
    bm = ball_mask(d60)
    ys2, xs2 = np.where(bm)
    t = (60 - N_DROP) / FPS
    ok("ball on its parabola on the SHIPPED file",
       bm.sum() > 700 and abs(xs2.mean() - (X0 + VX * t)) < 3.0
       and abs(ys2.mean() - (YG - R_BALL - SCALE * height(t))) < 3.0,
       f"{bm.sum()} px c=({xs2.mean():.1f},{ys2.mean():.1f})")

    # audio round-trip: decode the aac, check onsets and the silence
    ra = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", OUT_FIN, "-f", "s16le",
         "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(SR), "-"],
        capture_output=True)
    y = np.frombuffer(ra.stdout, np.int16).astype(np.float64) / 32767.0
    terr = 0.0
    for k in range(8):
        t_exp = N_DROP / FPS + bounce_time(k)
        s0 = int(round(t_exp * SR))
        w0 = np.abs(y[s0 - 1440:s0 + 1440])
        onset = s0 - 1440 + int(np.argmax(w0 > 0.10 * w0.max()))
        terr = max(terr, abs(onset - s0) / SR)
    ok("first 8 clicks land on B_k after aac", terr < 0.005,
       f"max err {terr * 1000:.2f} ms")
    s_click = int((N_DROP / FPS + T0) * SR)
    rms_click = np.sqrt((y[s_click:s_click + SR // 10] ** 2).mean())
    s_quiet = int((N_DROP / FPS + T_INF + 0.25) * SR)
    rms_quiet = np.sqrt((y[s_quiet:s_quiet + SR] ** 2).mean())
    ok("silence after 4.5 s survives aac",
       rms_quiet < 0.01 * rms_click,
       f"quiet/click RMS = {rms_quiet / rms_click:.5f}")

    print()
    if CHECKS["fail"]:
        print(f"{CHECKS['fail']} FAILURES (incl. render)")
        sys.exit(1)
    print("ENCODE CHECKS PASSED — DONE", flush=True)


def review_stills():
    for n in (20, 60, 120, 165, 173, 240):
        Image.fromarray(frame_at(n)).save(f"{OUT_DIR}/zeno_f{n:03d}.png")
    subprocess.run(
        ["ffmpeg", "-y", "-pattern_type", "glob",
         "-i", f"{OUT_DIR}/zeno_f*.png",
         "-filter_complex", "scale=270:-1,tile=3x2",
         f"{OUT_DIR}/zeno_sheet.png"],
        capture_output=True)
    print("sheet:", f"{OUT_DIR}/zeno_sheet.png", flush=True)


if __name__ == "__main__":
    run_checks()
    encode()
    check_encode()
    review_stills()
