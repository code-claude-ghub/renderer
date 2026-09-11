#!/usr/bin/env python3
"""THE BORDER — rhythm becomes pitch.

One fixed thump, repeated. 110 a minute (the CPR tempo) for 3 s, then the
rate rises exponentially to 110 a second (the guitar's A string) over 12 s,
then holds 3 s. Nothing about the thump changes except how often it comes.

Writing-and-sound-first: seven text cards carry the piece. The screen is a
diagram, not a simulation — a rate readout, a "one second" ruler whose ticks
re-space with the rate (no phase, so nothing aliases at 30 fps), and a pulse
dot that flashes with the beat ONLY up to 3 flashes/s (WCAG 2.3.1 general
flash threshold) and then holds steady.

Facts verified 2026-09-11:
  - lower limit of melodic pitch ~30 Hz for harmonic complexes with energy
    below 800 Hz: Pressnitzer, Patterson & Krumbholz, JASA 109(5):2074, 2001
  - A2 (guitar 5th string, standard tuning) = 110 Hz; A0 (piano bottom key)
    = 27.5 Hz — both from A4 = 440 Hz by exact octaves
  - WCAG 2.3.1: no more than three flashes in any one-second period
Audio: the thump is band-limited 0-850 Hz BY CONSTRUCTION (designed in the
frequency domain), so the standing <1 kHz accessibility promise holds
without a warning label; still FFT-verified on master AND decoded AAC.
"""
import os, subprocess, sys, time, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
FPS = 30
SR = 48000

R0 = 110.0 / 60.0        # 1.8333 Hz  (110 / min)
R1 = 110.0               # 110 Hz     (110 / s)
T_HOLD0 = 3.0
T_SWEEP = 11.0
T_HOLD1 = 4.0
DUR = T_HOLD0 + T_SWEEP + T_HOLD1          # 18.0 s
FRAMES = int(round(DUR * FPS))            # 540
N = int(round(DUR * SR))                  # 864000
FLASH_LIMIT = 3.0                          # flashes / s (WCAG 2.3.1)
LLMP = 30.0                                # Pressnitzer et al. 2001
T_OFF = 0.050                              # the whole grid starts 50 ms in

BG      = np.array([10, 12, 16], np.float32)
C_TXT   = np.array([236, 240, 246], np.float32)   # card text + ruler ticks
C_ACC   = np.array([252, 178, 41], np.float32)    # readout + accent lines only
C_PULSE = np.array([198, 52, 44], np.float32)     # the dot only
C_DIM   = np.array([120, 126, 138], np.float32)   # ruler caption + end caps only

STAMP = time.strftime("%H%M%S")
OUT_RAW = f"out/border_{STAMP}_video.mp4"
OUT_WAV = f"out/border_{STAMP}.wav"
OUT_MP4 = f"out/border_{STAMP}_final.mp4"
SHEET   = f"out/border_{STAMP}_sheet.png"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_M = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

FAILS = []
def ck(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if (detail and not ok) else ""))
    if not ok: FAILS.append(name)

# ------------------------------------------------------------------ rate ---
def rate(t):
    """pulses per second at time t (exponential sweep between the holds)."""
    t = np.asarray(t, np.float64) - T_OFF
    u = np.clip((t - T_HOLD0) / T_SWEEP, 0.0, 1.0)
    return R0 * (R1 / R0) ** u

def t_at_rate(r):
    return T_OFF + T_HOLD0 + T_SWEEP * np.log(r / R0) / np.log(R1 / R0)

T_FLASH_STOP = float(t_at_rate(FLASH_LIMIT))
T_LLMP = float(t_at_rate(LLMP))
print(f"rate hits {FLASH_LIMIT}/s at {T_FLASH_STOP:.2f} s; {LLMP}/s at {T_LLMP:.2f} s")

# --------------------------------------------------------------- writing ---
# (lines, t0, t1, accent_line_index)
CARDS = [
    (["this is 110 a minute.", "the tempo you would", "push a chest to."],      0.0,  3.0, 0),
    (["same thump.", "I will only change", "how often it comes."],              3.0,  5.5, None),
    (["the dot stopped flashing at 3.", "faster is unsafe for a screen.",
      "the ear keeps going."],                                                  5.5,  8.0, None),
    (["still separate thumps.", "listen for the moment",
      "they fuse into a note."],                                                8.0, 10.3, None),
    (["about 30 a second.", "below it you cannot tell", "which note it is.",
      "above it, you can."],                                                   10.3, 14.0, 0),
    (["110 a second.", "the A string on a guitar."],                          14.0, 16.0, 0),
    (["rhythm and pitch are one thing.", "the border is near 30.",
      "a piano's bottom key is 27.5 —", "just across it."],                   16.0, 18.0, None),
]
ck("cards tile 0..18 s contiguously",
   CARDS[0][1] == 0.0 and CARDS[-1][2] == DUR and
   all(CARDS[i][2] == CARDS[i+1][1] for i in range(len(CARDS)-1)))
ck("flash-stop happens before its card (card 2) opens",
   T_FLASH_STOP < CARDS[2][1], f"stop={T_FLASH_STOP:.2f} card2={CARDS[2][1]}")
ck("30/s falls inside card 4's span",
   CARDS[4][1] <= T_LLMP <= CARDS[4][2], f"t30={T_LLMP:.2f}")
ck("110/s hold begins with card 5 (±T_OFF)", abs(CARDS[5][1] - (T_HOLD0 + T_SWEEP)) <= T_OFF)
ck("A2 and A0 by exact octaves from 440",
   abs(440 / 4 - 110) < 1e-9 and abs(440 / 16 - 27.5) < 1e-9)

# ------------------------------------------------------------------ thump --
# designed in the frequency domain: sin^2 rise 0-80 Hz, flat 80-350, cos^2
# fall 350-850, ZERO above 850. zero-phase, then Hann-windowed.
L = 4096
fr = np.fft.rfftfreq(L, 1 / SR)
S = np.zeros_like(fr)
m = fr < 80;                  S[m] = np.sin(np.pi / 2 * fr[m] / 80) ** 2
m = (fr >= 80) & (fr <= 350); S[m] = 1.0
m = (fr > 350) & (fr < 850);  S[m] = np.cos(np.pi / 2 * (fr[m] - 350) / 500) ** 2
S[0] = 0.0
pulse = np.fft.irfft(S, L)
pulse = np.roll(pulse, L // 2)
pulse *= np.hanning(L)
pulse /= np.abs(pulse).max()
ps = np.abs(np.fft.rfft(pulse, 8 * L)) ** 2
pf = np.fft.rfftfreq(8 * L, 1 / SR)
ck("thump: energy above 1 kHz < 1e-8 (design)", ps[pf > 1000].sum() / ps.sum() < 1e-8,
   f"{ps[pf > 1000].sum() / ps.sum():.2e}")
PEAK = int(np.argmax(np.abs(pulse)))
core = np.where(np.abs(pulse) > 0.1)[0]
print(f"thump core (|x|>0.1): {(core[-1]-core[0]) / SR * 1e3:.2f} ms; peak at sample {PEAK} of {L}")

# ----------------------------------------------------------------- onsets --
# phase accumulator on the exact rate curve: onset whenever phase crosses an
# integer. The first onset is at t=0.
ts = np.arange(N) / SR
phase = np.cumsum(rate(ts + T_OFF)) / SR      # integrate on GRID time
OFF = int(0.050 * SR)                      # grid offset: 50 ms, so pulse 0's pre-ring fits
onsets = np.concatenate([[0], np.where(np.diff(np.floor(phase)) > 0)[0] + 1]) + OFF
onsets = onsets[onsets < N - OFF]
ONSET_T = onsets / SR
ONSET_T_GRID = (onsets - OFF) / SR         # time on the rate curve (t=0 is the first thump)
audio = np.zeros(N + 2 * L, np.float64)
for s0 in onsets:
    a0 = s0 - PEAK + L          # pulse PEAK lands on s0 (offset by L pad)
    audio[a0:a0 + L] += pulse
audio = audio[L:L + N]
audio *= 0.85 / np.abs(audio).max()
fade_n = int(0.020 * SR)
audio[-fade_n:] *= np.linspace(1, 0, fade_n)

# onset arithmetic checks
iv_hold0 = np.diff(onsets[ONSET_T_GRID < T_HOLD0 - 0.01])
iv_hold1 = np.diff(onsets[ONSET_T_GRID > T_HOLD0 + T_SWEEP + 0.05])
ck("hold 0: intervals == SR/1.8333 ±1 sample",
   abs(iv_hold0 - SR / R0).max() <= 1.0, f"{iv_hold0}")
ck("hold 1: intervals == SR/110 ±1 sample",
   abs(iv_hold1 - SR / R1).max() <= 1.0, f"min={iv_hold1.min()} max={iv_hold1.max()}")
n_pred = int(np.floor(phase[N - 2 * OFF - 1])) + 1
ck("onset count == floor(∫rate over the grid)+1", len(onsets) == n_pred, f"{len(onsets)} vs {n_pred}")
ck("first thump's pre-ring fits (OFF >= PEAK)", OFF >= PEAK)
print(f"onsets: {len(onsets)}  (hold0 {int((ONSET_T < T_HOLD0).sum())}, "
      f"hold1 {int((ONSET_T >= T_HOLD0 + T_SWEEP).sum())})")
ck("no clipping; ends silent", np.abs(audio).max() <= 0.86 and abs(audio[-1]) < 1e-4)
# the claim "same thump": every onset's local peak is the same height ±5%
pk_h = np.array([np.abs(audio[max(0, s - 20):s + 20]).max() for s in onsets])
ck("same thump: onset peak heights within 5% (slow section) / 25% (overlapping fast)",
   (pk_h[ONSET_T[:len(pk_h)] < 8].std() / pk_h[ONSET_T[:len(pk_h)] < 8].mean() < 0.05)
   and (pk_h.min() > 0.75 * pk_h.max()),
   f"min={pk_h.min():.3f} max={pk_h.max():.3f}")

spec = np.abs(np.fft.rfft(audio)) ** 2
freqs = np.fft.rfftfreq(N, 1 / SR)
hf = spec[freqs > 1000].sum() / spec.sum()
ck("master: energy above 1 kHz < 1e-6", hf < 1e-6, f"{hf:.2e}")
SEG_N = int(2.4 * SR); SEG_A = int((DUR - 2.5) * SR)
seg = audio[SEG_A:SEG_A + SEG_N] * np.hanning(SEG_N)
sp = np.abs(np.fft.rfft(seg)); ff = np.fft.rfftfreq(len(seg), 1 / SR)
pk = ff[np.argmax(sp)]
ck("final hold: spectral peak at 110 Hz ±0.5", abs(pk - 110) < 0.5, f"peak={pk:.2f}")

os.makedirs("out", exist_ok=True)
with wave.open(OUT_WAV, "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
    wf.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())

# ------------------------------------------------------------------ layout --
SAFE_TOP, SAFE_BOT = 192, 1632
DOT_CX, DOT_CY, DOT_R = 540, 380, 64
READ_Y = 520
RULER_X0, RULER_X1 = 60, 1020     # 960 px = one second
RULER_Y0, RULER_Y1 = 690, 750
CAP_Y = 768
CARD_TOP, CARD_H = 900, 600

def text_layer(lines, accent_idx, size=58, leading=76):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dr = ImageDraw.Draw(img)
    fnt = ImageFont.truetype(FONT_B, size)
    while max(dr.textbbox((0, 0), ln, font=fnt)[2] for ln in lines) > 980:
        size -= 2; fnt = ImageFont.truetype(FONT_B, size)
    y0 = CARD_TOP + (CARD_H - leading * len(lines)) // 2
    box = [W, H, 0, 0]
    for i, ln in enumerate(lines):
        bb = dr.textbbox((0, 0), ln, font=fnt)
        x = (W - (bb[2] - bb[0])) // 2 - bb[0]; y = y0 + i * leading
        col = tuple(int(c) for c in (C_ACC if i == accent_idx else C_TXT))
        dr.text((x, y), ln, font=fnt, fill=col + (255,), stroke_width=3, stroke_fill=(10, 12, 16, 255))
        b2 = dr.textbbox((x, y), ln, font=fnt, stroke_width=3)
        box = [min(box[0], b2[0]), min(box[1], b2[1]), max(box[2], b2[2]), max(box[3], b2[3])]
    a = np.asarray(img, np.float32)
    return a[:, :, :3], a[:, :, 3:4] / 255.0, box

CARD_LAYERS = [text_layer(c[0], c[3]) for c in CARDS]
for i, (_, _, box) in enumerate(CARD_LAYERS):
    ck(f"card {i} inside safe area", box[1] >= SAFE_TOP and box[3] <= SAFE_BOT, f"{box}")

def readout_text(t):
    if t < T_HOLD0 + T_OFF: return "110 / min"
    r = float(rate(t))
    return (f"{r:.1f} / s" if r < 10 else f"{r:.0f} / s")

_RO_CACHE = {}
def readout_layer(txt):
    if txt in _RO_CACHE: return _RO_CACHE[txt]
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dr = ImageDraw.Draw(img)
    fnt = ImageFont.truetype(FONT_M, 64)
    bb = dr.textbbox((0, 0), txt, font=fnt)
    x = (W - (bb[2] - bb[0])) // 2 - bb[0]
    dr.text((x, READ_Y), txt, font=fnt, fill=tuple(int(c) for c in C_ACC) + (255,),
            stroke_width=3, stroke_fill=(10, 12, 16, 255))
    box = dr.textbbox((x, READ_Y), txt, font=fnt, stroke_width=3)
    a = np.asarray(img, np.float32)
    _RO_CACHE[txt] = (a[:, :, :3], a[:, :, 3:4] / 255.0, box)
    return _RO_CACHE[txt]
RO_FENCE = [RULER_X0, READ_Y - 10, RULER_X1, READ_Y + 90]   # amber allowed here

# ruler caption + end caps (static)
cap_img = Image.new("RGBA", (W, H), (0, 0, 0, 0)); cd = ImageDraw.Draw(cap_img)
cap_fnt = ImageFont.truetype(FONT_M, 34)
cap_txt = "one second"
cb = cd.textbbox((0, 0), cap_txt, font=cap_fnt)
cd.text(((W - (cb[2] - cb[0])) // 2 - cb[0], CAP_Y), cap_txt, font=cap_fnt,
        fill=tuple(int(c) for c in C_DIM) + (255,))
ca = np.asarray(cap_img, np.float32)
base = np.empty((H, W, 3), np.float32); base[:] = BG
base = base * (1 - ca[:, :, 3:4] / 255) + ca[:, :, :3] * (ca[:, :, 3:4] / 255)
base[RULER_Y0 - 8:RULER_Y1 + 8, RULER_X0 - 4:RULER_X0 - 1] = C_DIM
base[RULER_Y0 - 8:RULER_Y1 + 8, RULER_X1 + 1:RULER_X1 + 4] = C_DIM

TICK_W = 3
def tick_xs(t):
    r = float(rate(t))
    k = np.arange(int(np.ceil(r)))
    xs = RULER_X0 + (RULER_X1 - RULER_X0) * k / r
    return xs[xs < RULER_X1 - TICK_W]

# dot glow field
BX0, BX1, BY0, BY1 = DOT_CX - 200, DOT_CX + 200, DOT_CY - 200, DOT_CY + 200
yy, xx = np.mgrid[BY0:BY1, BX0:BX1]
GLOW = np.exp(-((xx - DOT_CX) ** 2 + (yy - DOT_CY) ** 2) / DOT_R ** 2 * 1.6)[:, :, None].astype(np.float32)
ck("glow ink at readout row < 3 grey levels",
   float(np.exp(-((READ_Y - DOT_CY) / DOT_R) ** 2 * 1.6) * C_PULSE.max()) < 3.0)

FLASH_ONSETS = ONSET_T[ONSET_T < T_FLASH_STOP]     # only these flash
STEADY = 0.45
def dot_env(t):
    """flash decay after each qualifying onset; steady once the rate passes 3/s."""
    if t >= T_FLASH_STOP: return STEADY
    k = np.searchsorted(FLASH_ONSETS, t, side="right") - 1
    if k < 0: return STEADY
    return STEADY + (1 - STEADY) * float(np.exp(-(t - FLASH_ONSETS[k]) / 0.14))

win_max = max(int(((FLASH_ONSETS >= a) & (FLASH_ONSETS < a + 1.0)).sum())
              for a in np.arange(0, DUR, 0.05))
ck("no 1-s window holds more than 3 flashes (WCAG 2.3.1)", win_max <= 3, f"max={win_max}")

FADE_F = 6
def card_at(f):
    t = f / FPS
    for i, (_, t0, t1, _) in enumerate(CARDS):
        if t0 <= t < t1: return i
    return len(CARDS) - 1
def card_alpha(f, i):
    f0, f1 = CARDS[i][1] * FPS, CARDS[i][2] * FPS
    return max(0.0, min(1.0, (f - f0) / FADE_F, (f1 - f) / FADE_F))

def render_frame(f):
    t = f / FPS
    fr = base.copy()
    fr[BY0:BY1, BX0:BX1] += GLOW * dot_env(t) * C_PULSE
    for x in tick_xs(t):
        xi = int(round(x)); fr[RULER_Y0:RULER_Y1, xi:xi + TICK_W] = C_TXT
    rgb, al, _ = readout_layer(readout_text(t))
    fr = fr * (1 - al) + rgb * al
    i = card_at(f); rgb, al, _ = CARD_LAYERS[i]; a = al * card_alpha(f, i)
    fr = fr * (1 - a) + rgb * a
    return np.clip(fr, 0, 255)

# ------------------------------------------------------- render-side checks
def near(fr, col, tol): return np.abs(fr - col).sum(2) < tol
def count_ticks(fr):
    row = fr[(RULER_Y0 + RULER_Y1) // 2, RULER_X0:RULER_X1]
    on = near(row[None, :, :], C_TXT, 40)[0]
    return int((np.diff(on.astype(int)) == 1).sum() + (1 if on[0] else 0))

for i in range(len(CARDS)):
    fmid = int((CARDS[i][1] + CARDS[i][2]) / 2 * FPS)
    fr = render_frame(fmid); x0, y0, x1, y1 = CARD_LAYERS[i][2]
    n = int(near(fr[y0:y1 + 1, x0:x1 + 1], C_TXT, 60).sum())
    ck(f"card {i} text ink at f{fmid}", n > 500, f"n={n}")

okt, det = True, []
for t in (0.5, 4.0, 7.0, 9.5, T_LLMP, 13.0, 16.5):
    f = int(t * FPS)
    n = count_ticks(render_frame(f)); want = len(tick_xs(f / FPS))
    det.append(f"t{t:.1f}:{n}/{want}");  okt &= (n == want)
ck("ruler tick count matches rate at 7 sampled times", okt, " ".join(det))
ck("final ruler shows 110 ticks", count_ticks(render_frame(FRAMES - 1)) == 110,
   f"{count_ticks(render_frame(FRAMES - 1))}")

for i in (1, 4, 6):
    fmid = int((CARDS[i][1] + CARDS[i][2]) / 2 * FPS); fr = render_frame(fmid)
    amber = near(fr, C_ACC, 70)
    amber[RO_FENCE[1]:RO_FENCE[3], RO_FENCE[0]:RO_FENCE[2]] = False
    if CARDS[i][3] is not None:
        bx = CARD_LAYERS[i][2]; amber[bx[1]:bx[3] + 1, bx[0]:bx[2] + 1] = False
    ck(f"amber fenced (card {i})", int(amber.sum()) == 0, f"stray={int(amber.sum())}")

# glow EXCESS over the base, fenced above the readout row (the readout's
# digits change every frame and live inside the glow box — trap 55/58)
GY1 = READ_Y - 12
b = lambda f: float((render_frame(f)[BY0:GY1, BX0:BX1, 0] - base[BY0:GY1, BX0:BX1, 0]).sum())
okc, det = True, []
for k in (1, 2, 3, 5):
    f_on = int(round(FLASH_ONSETS[k] * FPS)) + 1; r = b(f_on) / b(f_on + 6)
    det.append(f"k{k}:{r:.2f}x"); okc &= r > 1.2
ck("dot flash coupled to slow onsets", okc, " ".join(det))
vals = [b(f) for f in range(int(T_FLASH_STOP * FPS) + 1, FRAMES, 37)]
ck("dot brightness constant after 3/s", max(vals) - min(vals) < 1e-3, f"spread={max(vals)-min(vals):.4f}")
ck("readout: '110 / min' at t=1, '110 / s' at t=17",
   readout_text(1.0) == "110 / min" and readout_text(17.0) == "110 / s")

tiles = [Image.fromarray(render_frame(int(t * FPS)).astype(np.uint8)).resize((360, 640), Image.LANCZOS)
         for t in (1.0, 9.5, 17.0)]
sheet = Image.new("RGB", (1080, 640))
for j, tl in enumerate(tiles): sheet.paste(tl, (j * 360, 0))
sheet.save(SHEET); print(f"[gate] sheet -> {SHEET}")

if FAILS: print("RENDER-SIDE FAILURES:", FAILS); sys.exit(1)
if "--check-only" in sys.argv: sys.exit(0)

# ------------------------------------------------------------------ encode --
enc = subprocess.Popen(
    ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
     "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", OUT_RAW],
    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for f in range(FRAMES):
    enc.stdin.write(render_frame(f).astype(np.uint8).tobytes())
    if f % 90 == 0: print(f"enc {f}/{FRAMES}", flush=True)
enc.stdin.close(); enc.wait(); assert enc.returncode == 0
subprocess.run(["ffmpeg", "-y", "-i", OUT_RAW, "-i", OUT_WAV, "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                "-movflags", "+faststart", OUT_MP4], check=True, capture_output=True)

# ----------------------------------------------------------- encode checks --
probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                        "stream=nb_read_frames,duration", "-of", "csv=p=0", OUT_MP4],
                       capture_output=True, text=True).stdout.strip().split(",")
dur_v, nfr = float(probe[0]), int(probe[1])
ck("encode: 540 frames, 18.0 s", nfr == FRAMES and abs(dur_v - DUR) < 0.15, f"{nfr} {dur_v}")

ry = (RULER_Y0 + RULER_Y1) // 2
dec = subprocess.run(["ffmpeg", "-v", "error", "-i", OUT_MP4, "-vf",
                      f"select=eq(n\\,{FRAMES-1}),crop={RULER_X1-RULER_X0}:2:{RULER_X0}:{ry}",
                      "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True)
prof = np.frombuffer(dec.stdout, np.uint8).reshape(2, RULER_X1 - RULER_X0).astype(float).mean(0)
mid = (prof.max() + prof.min()) / 2
on = prof > mid
n_dec = int((np.diff(on.astype(int)) == 1).sum() + (1 if on[0] else 0))
ck("decoded: 110 ticks survive on the final ruler", n_dec == 110, f"n={n_dec} contrast={prof.max()-prof.min():.0f}")

x0, y0, x1, y1 = CARD_LAYERS[4][2]; cw, chh = (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2
fm = int((CARDS[4][1] + CARDS[4][2]) / 2 * FPS)
dec = subprocess.run(["ffmpeg", "-v", "error", "-i", OUT_MP4, "-vf", f"select=eq(n\\,{fm}),crop={cw}:{chh}:{x0}:{y0}",
                      "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
dfr = np.frombuffer(dec.stdout, np.uint8).reshape(chh, cw, 3).astype(np.float32)
ck("decoded: card-4 text survives", int(near(dfr, C_TXT, 90).sum()) > 400)

DEC_WAV = f"out/border_{STAMP}_dec.wav"
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", OUT_MP4, "-ac", "1", "-ar", str(SR), DEC_WAV], check=True, capture_output=True)
with wave.open(DEC_WAV) as wf:
    da = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float64) / 32768
present = 0
slow = onsets[ONSET_T < T_HOLD0 + 2.0]
for s0 in slow:
    w1 = np.abs(da[s0:s0 + 240]).max(); w0 = np.abs(da[max(0, s0 - 2400):max(1, s0 - 400)]).max()
    present += (w1 > max(4 * w0, 0.02))
ck(f"decoded: all {len(slow)} slow thumps present at grid positions", present == len(slow), f"{present}/{len(slow)}")
seg = da[SEG_A:SEG_A + SEG_N] * np.hanning(SEG_N)
sp = np.abs(np.fft.rfft(seg)); ff = np.fft.rfftfreq(len(seg), 1 / SR); pk = ff[np.argmax(sp)]
ck("decoded: final hold peaks at 110 Hz ±0.5", abs(pk - 110) < 0.5, f"peak={pk:.2f}")
dspec = np.abs(np.fft.rfft(da)) ** 2; dfreqs = np.fft.rfftfreq(len(da), 1 / SR)
dhf = dspec[dfreqs > 1000].sum() / dspec.sum()
ck("decoded audio: energy above 1 kHz < 1e-4 (AAC)", dhf < 1e-4, f"{dhf:.2e} ({10*np.log10(dhf+1e-30):.1f} dB)")
sz = os.path.getsize(OUT_MP4)
ck("file size sane", 200_000 < sz < 30_000_000, f"{sz}")

print()
if FAILS: print("FAILURES:", FAILS); sys.exit(1)
print(f"ALL CHECKS PASSED — {OUT_MP4} ({sz} bytes, {nfr} frames, {dur_v:.2f} s)")
print(f"t(3/s)={T_FLASH_STOP:.2f}s  t(30/s)={T_LLMP:.2f}s  onsets={len(onsets)}")
print()
print("NOT VERIFIED, and the piece does not claim it: WHERE any given listener")
print("hears the fusion — the 30 Hz figure is a population result from a melody")
print("task (Pressnitzer et al. 2001), not a wall; individual listeners vary and")
print("the transition is a zone, not a line. Whether a phone speaker reproduces")
print("110 Hz at all (most lean on harmonics 330-770 Hz, which this thump has).")
print("How a real piano's bottom key is heard in context (harmonics help it).")
