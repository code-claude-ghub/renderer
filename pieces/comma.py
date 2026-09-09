#!/usr/bin/env python3
"""THE COMMA — twelve perfect fifths do not come home.

Music/synthesis piece: the audio is the claim. Start at A = 220 Hz.
Multiply by exactly 3/2 twelve times, folding back into the octave
(divide by 2) whenever the ratio leaves [1, 2). Exact Fractions
throughout. After twelve fifths and seven folds the ratio is

    (3/2)^12 / 2^7  =  3^12 / 2^19  =  531441 / 524288
                    ~  1.0136433    ~  23.46 cents

— the Pythagorean comma (Euclid's proportion; known in China by
122 BC). The walk lands 1.36% sharp of its own starting note.

The payoff is COUNTABLE: play the start (220 Hz) against the finish
(220 * 531441/524288 = 223.0015 Hz) and the error beats at exactly
223.0015 - 220 = 3.0015 Hz — the comma, expressed as three throbs a
second. Beat frequency = difference of frequencies (sourced). Piano
tuners tune by counting exactly this.

All tones are pure sines, 220-436 Hz (nothing above 1 kHz anywhere).
Everything measurable is measured; the one thing the checks cannot
reach is the fusion of two tones into one wobbling note — that
happens in the listener (trap 68: printed, not claimed).

  START (n   0.. 23): A 220 Hz sounds. circle + start marker.
  WALK  (n  24..311): 12 steps x 24 frames. each step: new dot on
                      the log-frequency circle (octave = full turn),
                      chord from the last dot, exact ratio readout.
  REF   (n 312..341): 220 Hz again, for the ear's benefit.
  BEAT  (n 342..503): both notes, 5.4 s. envelope chart draws the
                      throbs as they happen; a tick per null.
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
N = 504                            # 16.8 s
SR = 48000
SPF = SR // FPS                    # 1600 samples per frame

F_BASE = 220.0                     # A3
COMMA = Fraction(531441, 524288)   # 3^12 / 2^19

# the exact walk: multiply by 3/2, fold into [1, 2)
RATIOS = []                        # r_1 .. r_12 (folded, exact)
FOLDS = 0
_r = Fraction(1)
for _ in range(12):
    _r *= Fraction(3, 2)
    if _r >= 2:
        _r /= 2
        FOLDS += 1
    RATIOS.append(_r)

FREQS = [F_BASE * float(r) for r in RATIOS]          # step pitches, Hz
F_END = F_BASE * float(COMMA)                        # 223.0015... Hz
F_BEAT = F_END - F_BASE                              # 3.0015... Hz
NOTE_NAMES = ["E", "B", "F#", "C#", "G#", "D#",
              "A#", "F", "C", "G", "D", "A+"]

# frame plan
N_START, N_WALK0, N_STEP = 0, 24, 24
N_REF = 312                        # 24 + 12*24
N_BEAT = 342
T_BEAT0 = N_BEAT / FPS             # 11.4 s

A_NOTE = 0.55                      # walk/ref note amplitude
A_PAIR = 0.36                      # each tone of the beat pair
GATE = 0.010                       # raised-cosine edge, s

# layout
CX, CY, R = 540, 790, 370
Y_HEAD = 290
Y_FRAC, Y_SUB = 1290, 1372
CH_L, CH_R, CH_TOP, CH_BOT = 120, 960, 1240, 1490
Y_CAP = 1552
SAFE_TOP, SAFE_BOT = int(0.10 * H), int(0.85 * H)

BGC = (14, 15, 20)
C_DOT = (252, 178, 41)
C_CHORD = (92, 108, 148)
C_CIRC = (70, 73, 82)
C_MARK = (240, 242, 246)
C_RED = (224, 46, 36)
C_CURVE = (235, 238, 242)
C_HEAD = (200, 203, 210)
C_CAP = (152, 156, 166)
C_FRAC = (222, 225, 232)
C_LBL = (140, 144, 153)

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
OUT_DIR = "/home/maroon-beret/projects/active/youtube/youtube-channel/out"
STAMP = time.strftime("%H%M%S")
OUT_MP4 = f"{OUT_DIR}/comma_{STAMP}.mp4"
OUT_WAV = f"{OUT_DIR}/comma_{STAMP}.wav"
OUT_FIN = f"{OUT_DIR}/comma_{STAMP}_final.mp4"

_fonts = {}


def font(sz):
    if sz not in _fonts:
        _fonts[sz] = ImageFont.truetype(FONT, sz)
    return _fonts[sz]


def theta(ratio_float):
    """Angle round the circle: one octave = one full turn, from 12
    o'clock, clockwise. log-frequency, which is what pitch is."""
    return 2 * math.pi * math.log2(ratio_float)


def dot_xy(ratio_float, rad=R):
    th = theta(ratio_float)
    return CX + rad * math.sin(th), CY - rad * math.cos(th)


# ---------------------------------------------------------------- audio
def gate_env(n_samp, attack=GATE, release=GATE):
    e = np.ones(n_samp)
    na, nr = int(attack * SR), int(release * SR)
    t = np.arange(na) / na
    e[:na] = 0.5 - 0.5 * np.cos(np.pi * t)
    t = np.arange(nr) / nr
    e[-nr:] = 0.5 + 0.5 * np.cos(np.pi * t)
    return e


def tone(freq, t0, dur, amp, x):
    s0 = int(round(t0 * SR))
    n = int(round(dur * SR))
    tt = np.arange(n) / SR
    x[s0:s0 + n] += amp * np.sin(2 * np.pi * freq * tt) * gate_env(n)


def build_audio():
    x = np.zeros(N * SPF, np.float64)
    tone(F_BASE, 0.02, 0.72, A_NOTE, x)                    # START
    for k in range(12):                                    # WALK
        tone(FREQS[k], (N_WALK0 + k * N_STEP) / FPS + 0.02,
             0.72, A_NOTE, x)
    tone(F_BASE, N_REF / FPS + 0.02, 0.92, A_NOTE, x)      # REF
    dur = (N - N_BEAT) / FPS - 0.06                        # BEAT
    tone(F_BASE, T_BEAT0 + 0.02, dur, A_PAIR, x)
    tone(F_END, T_BEAT0 + 0.02, dur, A_PAIR, x)
    data = (x * 32767.0).astype(np.int16)
    with wave.open(OUT_WAV, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())
    return x


def hop_env(x, s0, s1, hop=480):
    """RMS envelope in 10 ms hops over x[s0:s1]."""
    seg = x[s0:s1]
    n = len(seg) // hop
    e = np.sqrt((seg[:n * hop].reshape(n, hop) ** 2).mean(1))
    return e


def env_minima(e, hop=480):
    """Indices (hop units) of local minima that are genuine nulls."""
    lo = e < 0.30 * np.median(e[e > 1e-6]) if (e > 1e-6).any() else e < 0
    mins = []
    for i in range(1, len(e) - 1):
        if lo[i] and e[i] <= e[i - 1] and e[i] < e[i + 1]:
            if not mins or i - mins[-1] > 8:      # >80 ms apart
                mins.append(i)
    return mins


X_AUDIO = None                     # filled in main
ENV = None                         # beat-act envelope (10 ms hops)
ENV_MINS = None                    # null positions, hop units


def beat_env_from(x):
    s0 = int((T_BEAT0 + 0.10) * SR)
    s1 = int(((N - 2) / FPS) * SR)
    return hop_env(x, s0, s1), s0


# ---------------------------------------------------------------- frame
def act_of(n):
    if n < N_WALK0:
        return "START", 0
    if n < N_REF:
        return "WALK", (n - N_WALK0) // N_STEP + 1     # step 1..12
    if n < N_BEAT:
        return "REF", 12
    return "BEAT", 12


def head_of(n):
    act, s = act_of(n)
    return {"START": "the starting note",
            "WALK": f"fifth {s} of 12",
            "REF": "back to the start",
            "BEAT": "start + finish, together"}[act]


def cap_of(n):
    act, _ = act_of(n)
    return {"START": "twelve perfect fifths should come home in seven octaves",
            "WALK": "every step: exactly × 3/2, folded back into one octave",
            "REF": "the walk came home 1.36% sharp. here is 220 Hz again",
            "BEAT": "the comma beats 3.0× a second — count the throbs"}[act]


def frac_of(n):
    act, s = act_of(n)
    if act == "START":
        return "1/1", f"{F_BASE:.2f} Hz"
    r = RATIOS[s - 1]
    return (f"{r.numerator}/{r.denominator}",
            f"= {F_BASE * float(r):.2f} Hz")


def fit_text(d, xy, s, sz, col, maxw=1000):
    f = font(sz)
    while d.textlength(s, font=f) > maxw and sz > 24:
        sz -= 2
        f = font(sz)
    d.text(xy, s, font=f, fill=col, anchor="mm")


def frame_at(n):
    act, step = act_of(n)
    img = Image.new("RGB", (W, H), BGC)
    d = ImageDraw.Draw(img)

    # header / captions
    fit_text(d, (CX, Y_HEAD), head_of(n), 76, C_HEAD)
    fit_text(d, (CX, Y_CAP), cap_of(n), 46, C_CAP)

    # the circle, the start marker
    d.ellipse([CX - R, CY - R, CX + R, CY + R], outline=C_CIRC, width=3)
    mx0, my0 = dot_xy(1.0, R - 30)
    mx1, my1 = dot_xy(1.0, R + 30)
    d.line([mx0, my0, mx1, my1], fill=C_MARK, width=6)
    lx, ly = dot_xy(1.0, R + 64)
    fit_text(d, (lx, ly - 14), "A  220 Hz", 34, C_MARK)

    # chords + dots up to the current step
    prev = 1.0
    for k in range(step):
        fr = float(RATIOS[k])
        x0, y0 = dot_xy(prev)
        x1, y1 = dot_xy(fr)
        d.line([x0, y0, x1, y1], fill=C_CHORD, width=4)
        prev = fr
    for k in range(step):
        fr = float(RATIOS[k])
        x1, y1 = dot_xy(fr)
        rr = 16 if k == 11 else 12
        d.ellipse([x1 - rr, y1 - rr, x1 + rr, y1 + rr], fill=C_DOT)
        nx, ny = dot_xy(fr, R + 36)
        fit_text(d, (nx, ny), NOTE_NAMES[k], 30, C_LBL)

    # the gap: red arc from the start marker to the landing point
    if step == 12:
        deg = math.degrees(theta(float(COMMA)))
        d.arc([CX - R, CY - R, CX + R, CY + R],
              start=-90, end=-90 + deg, fill=C_RED, width=14)
        gx, gy = dot_xy(float(COMMA) ** 0.5, R - 96)
        fit_text(d, (gx, gy), "1.36% too high", 40, C_RED)

    if act == "BEAT":
        # envelope chart: the throbs drawn as they happen, from the
        # SAME array the speaker is playing (ENV, 10 ms hops)
        d.rectangle([CH_L, CH_TOP, CH_R, CH_BOT], outline=C_CIRC, width=2)
        t_now = n / FPS - (T_BEAT0 + 0.10)
        hops_now = max(0, min(len(ENV), int(t_now * 100)))
        if hops_now > 1:
            emax = ENV.max()
            pts = []
            for i in range(hops_now):
                px = CH_L + (CH_R - CH_L) * i / (len(ENV) - 1)
                py = CH_BOT - 8 - (CH_BOT - CH_TOP - 20) * ENV[i] / emax
                pts.append((px, py))
            d.line(pts, fill=C_CURVE, width=3)
            for m in ENV_MINS:
                if m < hops_now:
                    px = CH_L + (CH_R - CH_L) * m / (len(ENV) - 1)
                    d.line([px, CH_BOT - 26, px, CH_BOT - 4],
                           fill=C_DOT, width=5)
    else:
        f1, f2 = frac_of(n)
        fit_text(d, (CX, Y_FRAC), f1, 68, C_FRAC)
        fit_text(d, (CX, Y_SUB), f2, 44, C_CAP)

    return np.asarray(img, np.uint8)


def render_frames():
    for n in range(N):
        yield frame_at(n)


# ---------------------------------------------------------------- checks
CHECKS = {"pass": 0, "fail": 0}


def ok(name, cond, detail=""):
    tag = "ok  " if cond else "FAIL"
    CHECKS["pass" if cond else "fail"] += 1
    print(f"  {tag} {name}" + (f"  [{detail}]" if detail else ""), flush=True)


def mask_of(fr, lo, hi):
    r, g, b = (fr[..., i].astype(int) for i in range(3))
    return ((r >= lo[0]) & (r <= hi[0]) & (g >= lo[1]) & (g <= hi[1])
            & (b >= lo[2]) & (b <= hi[2]))


def dot_mask(fr):
    return mask_of(fr, (215, 140, 0), (255, 210, 90))


def red_mask(fr):
    return mask_of(fr, (180, 0, 0), (255, 90, 80))


def peak_freq(x, s0, s1):
    """Dominant frequency of x[s0:s1], parabolic sub-bin interp."""
    seg = x[s0:s1] * np.hanning(s1 - s0)
    sp = np.abs(np.fft.rfft(seg))
    k = int(np.argmax(sp[1:])) + 1
    if 1 <= k < len(sp) - 1:
        a, b, c = sp[k - 1], sp[k], sp[k + 1]
        k = k + 0.5 * (a - c) / (a - 2 * b + c)
    return k * SR / (s1 - s0)


def run_checks():
    global X_AUDIO, ENV, ENV_MINS
    print("== model checks (exact fractions) ==", flush=True)
    ok("walk lands on 531441/524288 exactly", RATIOS[-1] == COMMA,
       str(RATIOS[-1]))
    ok("the comma is 3^12 / 2^19",
       COMMA == Fraction(3 ** 12, 2 ** 19))
    ok("seven folds — seven octaves", FOLDS == 7, f"{FOLDS}")
    cents = 1200 * math.log2(float(COMMA))
    ok("comma = 23.46 cents", abs(cents - 23.46) < 0.01, f"{cents:.3f}")
    ok("all 12 folded pitches inside [220, 440)",
       all(220.0 <= f < 440.0 for f in FREQS),
       f"{min(FREQS):.1f}..{max(FREQS):.1f} Hz")
    ok("12 distinct pitch classes", len(set(RATIOS)) == 12)
    ok("beat = f_end - f_base = 3.0015 Hz",
       abs(F_BEAT - 3.001511) < 1e-4, f"{F_BEAT:.6f}")
    exp_throbs = F_BEAT * ((N - N_BEAT) / FPS - 0.06)
    ok("~16 throbs fit in the beat act", 15.5 < exp_throbs < 16.5,
       f"{exp_throbs:.2f}")
    et = 1200 * math.log2(1.5) - 700
    ok("ET flattens each fifth ~1.955 cents (desc figure)",
       abs(et - 1.955) < 0.001, f"{et:.4f}")

    print("== audio checks (master wav) ==", flush=True)
    X_AUDIO = build_audio()
    x = X_AUDIO
    ok("wav length matches the video",
       abs(len(x) / SR - N / FPS) < 0.001, f"{len(x) / SR:.3f} s")
    pk = np.abs(x).max()
    ok("peak level sane", 0.60 < pk <= 0.80, f"{pk:.3f}")

    ferr = 0.0
    for k in range(12):
        s0 = int(((N_WALK0 + k * N_STEP) / FPS + 0.12) * SR)
        f_meas = peak_freq(x, s0, s0 + 26400)
        ferr = max(ferr, abs(f_meas - FREQS[k]))
    ok("all 12 step pitches measured off the wav", ferr < 0.8,
       f"max err {ferr:.3f} Hz")

    s0 = int((T_BEAT0 + 0.30) * SR)
    seg = x[s0:s0 + 230400] * np.hanning(230400)      # 4.8 s window
    sp = np.abs(np.fft.rfft(seg))
    fbin = SR / 230400
    k1, k2 = int(round(220.0 / fbin)), int(round(F_END / fbin))
    third = np.max(np.delete(sp, range(min(k1, k2) - 8,
                                       max(k1, k2) + 9)))
    ok("beat act holds exactly two tones (220, 223.0015)",
       sp[k1] > 20 * third and sp[k2] > 20 * third,
       f"peaks/{third / max(sp[k1], sp[k2]):.4f} bg")
    ok("energy ceiling 1 kHz honoured",
       np.abs(np.fft.rfft(x))[int(1000 * len(x) / SR):].max()
       < 0.001 * np.abs(np.fft.rfft(x)).max())

    ENV, _ = beat_env_from(x)
    ENV_MINS = env_minima(ENV)
    gaps = np.diff(ENV_MINS) * 0.010                   # hops -> s
    ok("throb nulls counted on the wav: 15-16", 15 <= len(ENV_MINS) <= 16,
       f"{len(ENV_MINS)} nulls")
    ok("null spacing = 1/3.0015 s (the comma, timed)",
       len(gaps) > 0 and abs(gaps.mean() - 1 / F_BEAT) < 0.006,
       f"mean {gaps.mean():.4f} vs {1 / F_BEAT:.4f} s")

    # note 1 sounds 0.02..0.74 s of its 0.8 s slot; probe 0.76..0.79
    q = int((N_WALK0 / FPS - 0.04) * SR)
    ok("gates close between notes", np.abs(x[q:q + 800]).max() < 1e-3)

    print("== render checks (buffer) ==", flush=True)
    fin = frame_at(N - 1)
    dm = dot_mask(fin)
    derr, hit = 0.0, 0
    for k in range(12):
        ex, ey = dot_xy(float(RATIOS[k]))
        ys, xs = np.where(dm[int(ey) - 22:int(ey) + 22,
                             int(ex) - 22:int(ex) + 22])
        if len(xs) > 40:
            hit += 1
            derr = max(derr,
                       math.hypot(xs.mean() + int(ex) - 22 - ex,
                                  ys.mean() + int(ey) - 22 - ey))
    ok("12 dots at their log-frequency angles", hit == 12 and derr < 4.0,
       f"{hit}/12, max err {derr:.1f} px")

    rm = red_mask(fin)
    ys, xs = np.where(rm)
    angs = np.degrees(np.arctan2(xs - CX, -(ys - CY)))
    arc = angs[(np.hypot(xs - CX, ys - CY) > R - 20)
               & (np.hypot(xs - CX, ys - CY) < R + 20)]
    ok("red gap arc spans the comma (7.04 deg)",
       len(arc) > 50 and -1.0 < arc.min() < 1.5
       and 5.5 < arc.max() < 8.5,
       f"{arc.min():.1f}..{arc.max():.1f} deg, {len(arc)} px")
    f200 = frame_at(200)
    ok("no red arc before the walk ends", red_mask(f200).sum() == 0)

    mid_hits = 0
    prev = 1.0
    cm = mask_of(fin, (60, 75, 115), (125, 140, 185))
    for k in range(12):
        fr = float(RATIOS[k])
        x0, y0 = dot_xy(prev)
        x1, y1 = dot_xy(fr)
        mx, my = int((x0 + x1) / 2), int((y0 + y1) / 2)
        if cm[my - 5:my + 6, mx - 5:mx + 6].any():
            mid_hits += 1
        prev = fr
    ok("all 12 chords drawn (midpoint probes)", mid_hits == 12,
       f"{mid_hits}/12")

    # chart <-> audio coupling (trap 66): a tick per null, placed
    # where the envelope's own minima are
    tick = dot_mask(fin[CH_BOT - 28:CH_BOT - 2, :])
    cols = np.where(tick.any(0))[0]
    groups = np.split(cols, np.where(np.diff(cols) > 4)[0] + 1)
    cents_px = [float(g.mean()) for g in groups if len(g) >= 2]
    exp_px = [CH_L + (CH_R - CH_L) * m / (len(ENV) - 1) for m in ENV_MINS]
    perr = max(abs(a - b) for a, b in zip(cents_px, exp_px)) \
        if len(cents_px) == len(exp_px) else 999
    ok("one chart tick per audio null, in place",
       len(cents_px) == len(ENV_MINS) and perr < 4.0,
       f"{len(cents_px)} ticks, max err {perr:.1f} px")

    # five probes, five distinct headers (two walk steps differ too)
    heads = {head_of(n) for n in (5, 60, 200, 320, 400)}
    ok("header changes across acts", len(heads) == 5, str(len(heads)))
    lit = (fin.astype(int).sum(2) > 3 * 40)
    ok("ink fraction sane", 0.01 < lit.mean() < 0.30, f"{lit.mean():.3f}")
    rows = np.where(lit.any(1))[0]
    ok("all ink inside safe area",
       rows.min() > SAFE_TOP and rows.max() < SAFE_BOT,
       f"rows {rows.min()}..{rows.max()}")

    print()
    if CHECKS["fail"]:
        print(f"{CHECKS['fail']} CHECK FAILURES")
        sys.exit(1)
    print(f"ALL {CHECKS['pass']} CHECKS PASSED (so far)", flush=True)


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
         "v", "-show_entries", "stream=nb_read_frames",
         "-of", "csv=p=0", OUT_FIN], capture_output=True, text=True)
    ok("504 frames in the file", f"{N}" in r.stdout, r.stdout.strip())
    r2 = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "a",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0",
         OUT_FIN], capture_output=True, text=True)
    ok("aac audio stream present", "aac" in r2.stdout)

    ra = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", OUT_FIN, "-f", "s16le",
         "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(SR), "-"],
        capture_output=True)
    y = np.frombuffer(ra.stdout, np.int16).astype(np.float64) / 32767.0
    e, _ = beat_env_from(y)
    mins = env_minima(e)
    gaps = np.diff(mins) * 0.010
    ok("throb nulls survive aac: 15-16 at 1/3.0015 s",
       15 <= len(mins) <= 16 and abs(gaps.mean() - 1 / F_BEAT) < 0.010,
       f"{len(mins)} nulls, mean gap {gaps.mean():.4f} s")
    s0 = int((T_BEAT0 + 0.30) * SR)
    seg = y[s0:s0 + 230400] * np.hanning(230400)
    sp = np.abs(np.fft.rfft(seg))
    fbin = SR / 230400
    k1, k2 = int(round(220.0 / fbin)), int(round(F_END / fbin))
    ok("both tones present on the shipped file",
       sp[k1] > 0.5 * sp[k2] and sp[k2] > 0.5 * sp[k1]
       and min(sp[k1], sp[k2]) > 10 * np.median(sp))

    dfin = decode_frame(N - 2)
    ok("red gap arc survives h264", red_mask(dfin).sum() > 300,
       f"{red_mask(dfin).sum()} px")
    ok("dots survive h264", dot_mask(dfin).sum() > 2500,
       f"{dot_mask(dfin).sum()} px")
    tick = dot_mask(dfin[CH_BOT - 28:CH_BOT - 2, :])
    cols = np.where(tick.any(0))[0]
    groups = [g for g in np.split(cols, np.where(np.diff(cols) > 4)[0] + 1)
              if len(g) >= 2]
    ok("chart ticks countable on the shipped file",
       14 <= len(groups) <= 17, f"{len(groups)} ticks")

    print()
    if CHECKS["fail"]:
        print(f"{CHECKS['fail']} FAILURES (incl. render)")
        sys.exit(1)
    print("ENCODE CHECKS PASSED", flush=True)
    print("\nNOT VERIFIED HERE (trap 68): that the two tones fuse into "
          "ONE note that wobbles, and that you can count three throbs "
          "a second. Beats under ~10 Hz fuse in the listener, not in "
          "the file. The file carries two steady sines; the counting "
          "is yours.", flush=True)


def review_stills():
    for n in (10, 100, 220, 311, 330, 420, 502):
        Image.fromarray(frame_at(n)).save(f"{OUT_DIR}/comma_f{n:03d}.png")
    subprocess.run(
        ["ffmpeg", "-y", "-pattern_type", "glob",
         "-i", f"{OUT_DIR}/comma_f*.png",
         "-filter_complex", "scale=270:-1,tile=4x2",
         f"{OUT_DIR}/comma_sheet.png"], capture_output=True)
    # trap 67: look at it at watch size
    subprocess.run(
        ["ffmpeg", "-y", "-i", f"{OUT_DIR}/comma_f502.png",
         "-vf", "scale=360:-1", f"{OUT_DIR}/comma_gate360.png"],
        capture_output=True)
    print(f"sheet: {OUT_DIR}/comma_sheet.png", flush=True)


if __name__ == "__main__":
    run_checks()
    encode()
    check_encode()
    review_stills()
