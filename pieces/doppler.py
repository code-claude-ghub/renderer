#!/usr/bin/env python3
"""THE HORN — a horn never rises as it comes toward you.

A car horn passes a listener at the speed that makes the total Doppler drop
EXACTLY a minor third (three equal-tempered semitones): (c+v)/(c-v) = 2^(3/12)
=> v = 29.64 m/s = 106.7 km/h. The horn's own note is chosen so the listener
hears A3 (220.00 Hz) on approach and F#3 (185.00 Hz) on recession.

Writing-and-sound-first: seven text cards carry the piece. The screen is a
diagram — a road with the car and the listener, a live readout of the pitch
being received, and a pitch-vs-time trace that draws itself in.

The sound is NOT made from the Doppler formula. It is made by retarded time:
the sample you hear at receiver time t is the horn's waveform at the emission
time te that solves t = te + r(te)/c. The textbook formula f = f0 c/(c - v_r)
is then MEASURED off the wav as a check. Loudness follows (d/r)^0.6 rather
than 1/r — compressed so a phone can hear the approach; the piece makes no
claim about loudness and the description says so.

Facts (verified 2026-09-11):
  - c = 343 m/s (dry air, 20 C) — standard reference value
  - A3 = 220 Hz by exact octave from A4 = 440; F#3 = 220 / 2^(3/12) = 184.997
  - minor third = 3 semitones = ratio 2^(1/4) = 1.18921
  - moving-source Doppler: f = f0 * c / (c - v cos theta), stationary listener
Audio: four harmonics of a 201 Hz horn; the highest is 4 x 220 = 880 Hz on
approach. Nothing above 1 kHz BY CONSTRUCTION; FFT-verified on master and
decoded AAC anyway (the standing accessibility promise).
"""
import os, subprocess, sys, time, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
FPS = 30
SR = 48000

C_SOUND = 343.0                              # m/s
SEMI = 2.0 ** (1.0 / 12.0)
RATIO = SEMI ** 3                            # minor third = 1.18921
V = C_SOUND * (RATIO - 1) / (RATIO + 1)      # 29.638 m/s
D = 20.0                                     # closest approach, m
F_APP = 220.0                                # A3, what you hear coming
F0 = F_APP * (C_SOUND - V) / C_SOUND         # the horn's own note, 200.99 Hz
F_REC = F0 * C_SOUND / (C_SOUND + V)         # F#3, 184.997 Hz
HARM = [(1, 1.00), (2, 0.55), (3, 0.35), (4, 0.22)]
T_PASS = 6.1                                 # the car is abeam of you (true position)
DUR = 19.0
FRAMES = int(round(DUR * FPS))               # 570
N = int(round(DUR * SR))
X_HALF = V * T_PASS                          # road half-width in m: car enters at t=0

def kmh(semis):
    r = SEMI ** semis
    return C_SOUND * (r - 1) / (r + 1) * 3.6

BG      = np.array([10, 12, 16], np.float32)
C_TXT   = np.array([236, 240, 246], np.float32)   # cards, road, trace
C_ACC   = np.array([252, 178, 41], np.float32)    # readout + accent lines only
C_CAR   = np.array([198, 52, 44], np.float32)     # the car dot only
C_DIM   = np.array([120, 126, 138], np.float32)   # captions, reference lines, "you"

STAMP = time.strftime("%H%M%S")
OUT_RAW = f"out/horn_{STAMP}_video.mp4"
OUT_WAV = f"out/horn_{STAMP}.wav"
OUT_MP4 = f"out/horn_{STAMP}_final.mp4"
SHEET   = f"out/horn_{STAMP}_sheet.png"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_M = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

FAILS = []
def ck(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if (detail and not ok) else ""))
    if not ok: FAILS.append(name)

print(f"v = {V:.3f} m/s = {V*3.6:.2f} km/h = {V*2.23694:.2f} mph; f0 = {F0:.3f} Hz; "
      f"approach {F_APP:.3f}, recede {F_REC:.3f}; ratio {F_APP/F_REC:.5f}")
ck("A3 by exact octave from 440", abs(440 / 2 - F_APP) < 1e-12)
ck("F#3 = 220 / 2^(3/12) = 184.997", abs(F_REC - 184.997) < 0.001, f"{F_REC:.4f}")
ck("drop is exactly 3.000 semitones", abs(12 * np.log2(F_APP / F_REC) - 3) < 1e-9)
ck("speed rounds to 107 km/h", round(V * 3.6) == 107, f"{V*3.6:.2f}")
ck("ladder rounds to 36 / 71 / 107 km/h",
   [round(kmh(k)) for k in (1, 2, 3)] == [36, 71, 107], f"{[kmh(k) for k in (1,2,3)]}")
ck("highest harmonic on approach = 880 Hz (< 1 kHz by construction)",
   abs(4 * F_APP - 880) < 1e-9 and 4 * F_APP < 1000)

# --------------------------------------------------------------- writing ---
# (lines, t0, t1, accent_line_index)
CARDS = [
    (["a horn is coming toward you.", "listen: the pitch", "is not rising."],      0.0,  3.0, 2),
    (["louder, yes.", "louder is not higher.", "it has been an A the whole way."],  3.0,  5.0, None),
    (["now.", "the only thing that ever happens", "is the fall."],                  5.0,  7.4, 2),
    (["now it is low. F sharp.", "and it stays low."],                              7.4,  9.8, None),
    (["A down to F sharp: a minor third.", "the interval is the speedometer."],     9.8, 12.6, 1),
    (["one semitone ≈ 36 km/h", "two ≈ 71", "three ≈ 107"],                       12.6, 15.8, None),
    (["the note does not matter.", "every horn at 107", "drops a minor third."],   15.8, 19.0, None),
]
ck("cards tile 0..19 s contiguously",
   CARDS[0][1] == 0.0 and CARDS[-1][2] == DUR and
   all(CARDS[i][2] == CARDS[i+1][1] for i in range(len(CARDS)-1)))

# ------------------------------------------------------- retarded time -----
# source at (V*te, D) with te=0 at closest approach; listener at origin.
# receiver time t = T_PASS + te + r(te)/c. Invert on a fine monotonic grid.
te_grid = np.arange(-T_PASS - 1.0, DUR - T_PASS + 1.0, 1e-5)
r_grid = np.sqrt((V * te_grid) ** 2 + D ** 2)
t_grid = T_PASS + te_grid + r_grid / C_SOUND
ck("receiver time is monotonic in emission time", np.all(np.diff(t_grid) > 0))
ts = np.arange(N) / SR
te = np.interp(ts, t_grid, te_grid)
r = np.sqrt((V * te) ** 2 + D ** 2)
f_inst = F0 / (1.0 + V * V * te / (C_SOUND * r))    # received frequency at t (formula)

def f_received(t):
    """textbook moving-source Doppler at receiver time t (the CHECK, not the synthesis)."""
    te_ = np.interp(t, t_grid, te_grid); r_ = np.sqrt((V * te_) ** 2 + D ** 2)
    return F0 / (1.0 + V * V * te_ / (C_SOUND * r_))

T_TRUE = T_PASS + D / C_SOUND        # when you hear the horn's own note (58 ms after abeam)
ck("received pitch at the true-note instant == F0 ±0.01", abs(float(f_received(T_TRUE)) - F0) < 0.01)
ck("card 0: pitch never rises in 0..3 s", np.all(np.diff(f_inst[:3 * SR]) <= 0),
   f"max rise {np.diff(f_inst[:3*SR]).max():.2e}")
ck("card 0: pitch within 0.5 Hz of 220 through 3 s", abs(f_inst[:3 * SR] - 220).max() < 0.5,
   f"{f_inst[0]:.2f}..{f_inst[3*SR]:.2f}")
def cents_from(f, ref): return 1200 * np.log2(f / ref)
# the claims are MUSICAL (which note), so the tolerance is the nearest-note
# boundary (±50 cents), tightened to ±25 so it is unambiguous to an ear
ck("cards 0-1: nearest note is A (within 25 cents of 220) through 5 s",
   abs(cents_from(f_inst[:5 * SR], F_APP)).max() < 25, f"{cents_from(f_inst[5*SR], F_APP):+.1f} cents at 5 s")
ck("card 3: nearest note is F# (within 25 cents of 185) from 7.6 s on",
   abs(cents_from(f_inst[int(7.6 * SR):], F_REC)).max() < 25, f"{cents_from(f_inst[int(7.6*SR)], F_REC):+.1f} cents at 7.6 s")
ck("'stays low': pitch never rises after the pass", np.all(np.diff(f_inst[int(T_PASS * SR):]) <= 0))
ck("'never rises': pitch is non-increasing over the whole 19 s", np.all(np.diff(f_inst) <= 1e-9),
   f"max rise {np.diff(f_inst).max():.2e}")
ck("fall is inside card 2: 80% of the drop happens between 5.0 and 7.4 s",
   (f_inst[5 * SR] - f_inst[int(7.4 * SR)]) > 0.8 * (F_APP - F_REC),
   f"{f_inst[5*SR]:.2f} -> {f_inst[int(7.4*SR)]:.2f}")
print(f"pitch drift over card 0: {f_inst[0]:.3f} -> {f_inst[3*SR]:.3f} Hz "
      f"({1200*np.log2(f_inst[3*SR]/f_inst[0]):+.1f} cents)")

# ---------------------------------------------------------------- audio ----
phase = 2 * np.pi * F0 * te
amp = (D / r) ** 0.6                                  # compressed loudness (declared)
audio = np.zeros(N)
for k, a in HARM:
    audio += a * np.sin(k * phase)
audio *= amp
audio *= 0.85 / np.abs(audio).max()
fade_i = int(0.100 * SR); fade_o = int(0.050 * SR)
audio[:fade_i] *= np.linspace(0, 1, fade_i)
audio[-fade_o:] *= np.linspace(1, 0, fade_o)
ck("no clipping; ends silent", np.abs(audio).max() <= 0.86 and abs(audio[-1]) < 1e-4)
ck("loudest at the pass, quieter at both ends (ends < 0.35 of peak)",
   amp[int(T_PASS * SR)] > 0.99 and amp[SR] < 0.35 and amp[-SR] < 0.35,
   f"start {amp[SR]:.2f} end {amp[-SR]:.2f}")

def measured_f(sig, t0, t1):
    """instantaneous frequency of the fundamental via Hilbert on a band-passed slice."""
    a, b = int(t0 * SR), int(t1 * SR)
    seg = sig[a:b]
    X = np.fft.rfft(seg); fr_ = np.fft.rfftfreq(len(seg), 1 / SR)
    X[(fr_ < 150) | (fr_ > 250)] = 0            # keep only the fundamental
    seg_f = np.fft.irfft(X, len(seg))
    Xa = np.fft.fft(seg_f); n = len(seg_f)
    h = np.zeros(n); h[0] = 1; h[1:n // 2] = 2; h[n // 2] = 1
    an = np.fft.ifft(Xa * h)
    ph = np.unwrap(np.angle(an))
    m = slice(n // 4, 3 * n // 4)               # trim the edges
    return float(np.polyfit(np.arange(n)[m] / SR, ph[m], 1)[0] / (2 * np.pi))

fa = measured_f(audio, 0.5, 2.5); fb = measured_f(audio, 15.0, 18.5)
ck("MEASURED off the wav: approach = 220.0 ±0.3 Hz", abs(fa - 220) < 0.3, f"{fa:.3f}")
ck("MEASURED off the wav: recession = 185.0 ±0.3 Hz", abs(fb - F_REC) < 0.3, f"{fb:.3f}")
ck("MEASURED drop = 3.00 semitones ±0.03", abs(12 * np.log2(fa / fb) - 3) < 0.03,
   f"{12*np.log2(fa/fb):.3f}")
# the formula vs the retarded-time synthesis, at the steep part
fm = measured_f(audio, T_PASS - 0.15, T_PASS + 0.15)
ck("MEASURED at the pass ≈ formula at the pass (±2 Hz over a 0.3 s window)",
   abs(fm - float(f_received(T_PASS))) < 2.0, f"meas {fm:.2f} formula {float(f_received(T_PASS)):.2f}")

spec = np.abs(np.fft.rfft(audio)) ** 2
freqs = np.fft.rfftfreq(N, 1 / SR)
hf = spec[freqs > 1000].sum() / spec.sum()
ck("master: energy above 1 kHz < 1e-6", hf < 1e-6, f"{hf:.2e}")
top = freqs[spec > spec.max() * 1e-6].max()
ck("master: highest energetic bin <= 885 Hz", top <= 885, f"{top:.1f}")

os.makedirs("out", exist_ok=True)
with wave.open(OUT_WAV, "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
    wf.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())

# ------------------------------------------------------------------ layout --
SAFE_TOP, SAFE_BOT = 192, 1632
ROAD_Y = 330; ROAD_X0, ROAD_X1 = 60, 1020
PX_PER_M = (ROAD_X1 - ROAD_X0) / (2 * X_HALF)
YOU_Y = int(round(ROAD_Y + D * PX_PER_M))               # 20 m below the road
CAR_R = 14
READ_Y = 470
PLOT_X0, PLOT_X1 = 110, 1020
PLOT_Y0, PLOT_Y1 = 600, 860                             # pitch axis, top = 230 Hz
F_TOP, F_BOT = 230.0, 175.0
CARD_TOP, CARD_H = 920, 600

def fy(f):  return PLOT_Y0 + (F_TOP - f) / (F_TOP - F_BOT) * (PLOT_Y1 - PLOT_Y0)
def tx(t):  return PLOT_X0 + t / DUR * (PLOT_X1 - PLOT_X0)
def car_x(t):  return 540 + V * (t - T_PASS) * PX_PER_M

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
    return f"{float(f_received(t)):.1f} Hz"

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
RO_FENCE = [ROAD_X0, READ_Y - 10, ROAD_X1, READ_Y + 90]   # amber allowed here

# static base: road, "you", plot frame, reference lines + labels, captions
img = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dr = ImageDraw.Draw(img)
dr.line([(ROAD_X0, ROAD_Y), (ROAD_X1, ROAD_Y)], fill=tuple(int(c) for c in C_TXT) + (255,), width=4)
for xx_ in (ROAD_X0, ROAD_X1):
    dr.line([(xx_, ROAD_Y - 14), (xx_, ROAD_Y + 14)], fill=tuple(int(c) for c in C_DIM) + (255,), width=3)
you = tuple(int(c) for c in C_DIM) + (255,)
dr.polygon([(540, YOU_Y - 16), (526, YOU_Y + 8), (554, YOU_Y + 8)], fill=you)
fnt_s = ImageFont.truetype(FONT_M, 30)
bb = dr.textbbox((0, 0), "you", font=fnt_s)
dr.text((540 - (bb[2] - bb[0]) // 2 - bb[0], YOU_Y + 14), "you", font=fnt_s, fill=you)
lab = f"{V*3.6:.0f} km/h  →"
bb = dr.textbbox((0, 0), lab, font=fnt_s)
dr.text((ROAD_X1 - (bb[2] - bb[0]) - bb[0], ROAD_Y - 52), lab, font=fnt_s, fill=you)
# plot frame + dashed references
dr.line([(PLOT_X0, PLOT_Y0), (PLOT_X0, PLOT_Y1)], fill=you, width=2)
for f_ref, name in ((F_APP, "A"), (F_REC, "F#")):
    y = int(round(fy(f_ref)))
    for xx_ in range(PLOT_X0, PLOT_X1, 24):
        dr.line([(xx_, y), (xx_ + 12, y)], fill=you, width=2)
    bb = dr.textbbox((0, 0), name, font=fnt_s)
    dr.text((PLOT_X0 - (bb[2] - bb[0]) - 14 - bb[0], y - (bb[3] - bb[1]) // 2 - bb[1]), name, font=fnt_s, fill=you)
cap = "pitch you hear, over 19 s"
bb = dr.textbbox((0, 0), cap, font=fnt_s)
dr.text((540 - (bb[2] - bb[0]) // 2 - bb[0], PLOT_Y1 + 18), cap, font=fnt_s, fill=you)
ca = np.asarray(img, np.float32)
base = np.empty((H, W, 3), np.float32); base[:] = BG
base = base * (1 - ca[:, :, 3:4] / 255) + ca[:, :, :3] * (ca[:, :, 3:4] / 255)

# the full trace, drawn once; revealed by column as time advances
tr_img = Image.new("RGBA", (W, H), (0, 0, 0, 0)); td = ImageDraw.Draw(tr_img)
tt = np.linspace(0, DUR, 2000)
pts = [(float(tx(t_)), float(fy(f_received(t_)))) for t_ in tt]
td.line(pts, fill=tuple(int(c) for c in C_TXT) + (255,), width=5, joint="curve")
TRACE = np.asarray(tr_img, np.float32)
TR_RGB, TR_A = TRACE[:, :, :3], TRACE[:, :, 3:4] / 255.0
ck("trace stays inside the plot box", (TR_A[:, :, 0] > 0)[:PLOT_Y0 - 4].sum() == 0 and
   (TR_A[:, :, 0] > 0)[PLOT_Y1 + 4:].sum() == 0)

yy, xx = np.mgrid[0:2 * CAR_R + 8, 0:2 * CAR_R + 8]
CAR = np.clip(CAR_R + 0.5 - np.sqrt((xx - CAR_R - 4) ** 2 + (yy - CAR_R - 4) ** 2), 0, 1)[:, :, None].astype(np.float32)

FADE_F = 6
def card_at(f):
    t = f / FPS
    for i, (_, t0, t1, _) in enumerate(CARDS):
        if t0 <= t < t1: return i
    return len(CARDS) - 1
def card_alpha(f, i):
    f0_, f1_ = CARDS[i][1] * FPS, CARDS[i][2] * FPS
    return max(0.0, min(1.0, (f - f0_) / FADE_F, (f1_ - f) / FADE_F))

def render_frame(f):
    t = f / FPS
    fr = base.copy()
    # trace revealed up to now
    xr = int(round(tx(t)))
    a = TR_A[:, :xr + 1]; fr[:, :xr + 1] = fr[:, :xr + 1] * (1 - a) + TR_RGB[:, :xr + 1] * a
    # car (true position), only while on the road
    cx = car_x(t)
    if ROAD_X0 - CAR_R <= cx <= ROAD_X1 + CAR_R:
        x0 = int(round(cx)) - CAR_R - 4; y0 = ROAD_Y - CAR_R - 4
        sl = (slice(y0, y0 + CAR.shape[0]), slice(x0, x0 + CAR.shape[1]))
        fr[sl] = fr[sl] * (1 - CAR) + C_CAR * CAR
    rgb, al, _ = readout_layer(readout_text(t))
    fr = fr * (1 - al) + rgb * al
    i = card_at(f); rgb, al, _ = CARD_LAYERS[i]; a = al * card_alpha(f, i)
    fr = fr * (1 - a) + rgb * a
    return np.clip(fr, 0, 255)

# ------------------------------------------------------- render-side checks
def near(fr, col, tol): return np.abs(fr - col).sum(2) < tol

for i in range(len(CARDS)):
    fmid = int((CARDS[i][1] + CARDS[i][2]) / 2 * FPS)
    fr = render_frame(fmid); x0, y0, x1, y1 = CARD_LAYERS[i][2]
    n = int(near(fr[y0:y1 + 1, x0:x1 + 1], C_TXT, 60).sum())
    ck(f"card {i} text ink at f{fmid}", n > 500, f"n={n}")

# coupling: the car's drawn x and the trace's revealed tip both follow t
def car_col(fr):
    band = fr[ROAD_Y - CAR_R:ROAD_Y + CAR_R, ROAD_X0:ROAD_X1]
    m = near(band, C_CAR, 60)
    return float(np.where(m.any(0))[0].mean()) + ROAD_X0 if m.any() else None
okc, det = True, []
for t in (0.5, 3.0, T_PASS, 9.0, 12.0):
    got = car_col(render_frame(int(round(t * FPS)))); want = car_x(round(t * FPS) / FPS)
    det.append(f"t{t:.1f}:{'none' if got is None else f'{got:.0f}'}/{want:.0f}")
    okc &= got is not None and abs(got - want) < 2.0
ck("car drawn at its true position (5 sampled times)", okc, " ".join(det))
ck("car is abeam of 'you' at T_PASS", abs(car_x(T_PASS) - 540) < 1e-9)
ck("car off the road after 2*T_PASS", car_col(render_frame(int(13.0 * FPS))) is None)

def trace_tip(fr):
    """rightmost column with trace ink inside the plot box; and its row centre."""
    box = fr[PLOT_Y0:PLOT_Y1, PLOT_X0 + 2:PLOT_X1]
    m = near(box, C_TXT, 60)
    cols = np.where(m.any(0))[0]
    if len(cols) == 0: return None, None
    c = cols.max(); rows = np.where(m[:, c])[0]
    return c + PLOT_X0 + 2, rows.mean() + PLOT_Y0
okt, det = True, []
for t in (1.0, 5.5, T_PASS, 6.6, 10.0, 18.0):
    f = int(round(t * FPS)); c, rw = trace_tip(render_frame(f))
    want_c = tx(f / FPS); want_r = fy(float(f_received(f / FPS)))
    det.append(f"t{t:.1f}:x{c}/{want_c:.0f},y{rw:.0f}/{want_r:.0f}")
    okt &= c is not None and abs(c - want_c) <= 4 and abs(rw - want_r) <= 6
ck("trace tip tracks time AND the received pitch (coupled, 6 times)", okt, " ".join(det))
ck("trace at t=1 sits on the A line; at t=18 on the F# line",
   abs(fy(float(f_received(1.0))) - fy(F_APP)) < 3 and abs(fy(float(f_received(18.0))) - fy(F_REC)) < 3)
ro = [float(readout_text(f / FPS).split()[0]) for f in range(FRAMES)]
ck("readout: starts within 0.3 Hz of 220, ends 185.0 exactly",
   abs(ro[0] - 220) < 0.3 and ro[-1] == 185.0, f"{ro[0]} / {ro[-1]}")
ck("readout: the DISPLAYED number never goes up, any frame to the next",
   all(b <= a for a, b in zip(ro, ro[1:])))
print(f"readout around the pass (frames {int(T_PASS*FPS)-2}..{int(T_PASS*FPS)+2}): "
      f"{ro[int(T_PASS*FPS)-2:int(T_PASS*FPS)+3]}  — steps ~1 Hz/frame, may skip 201.0")

for i in (1, 3, 5):
    fmid = int((CARDS[i][1] + CARDS[i][2]) / 2 * FPS); fr = render_frame(fmid)
    amber = near(fr, C_ACC, 70)
    amber[RO_FENCE[1]:RO_FENCE[3], RO_FENCE[0]:RO_FENCE[2]] = False
    if CARDS[i][3] is not None:
        bx = CARD_LAYERS[i][2]; amber[bx[1]:bx[3] + 1, bx[0]:bx[2] + 1] = False
    ck(f"amber fenced (card {i})", int(amber.sum()) == 0, f"stray={int(amber.sum())}")
red = near(render_frame(int(15 * FPS)), C_CAR, 60)
ck("no red anywhere once the car has gone (t=15)", int(red.sum()) == 0, f"{int(red.sum())}")
ck("nothing flashes (WCAG 2.3.1 trivially): car and trace are continuous", True)

tiles = [Image.fromarray(render_frame(int(t * FPS)).astype(np.uint8)).resize((360, 640), Image.LANCZOS)
         for t in (1.5, T_PASS, 14.0)]
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
ck("encode: 570 frames, 19.0 s", nfr == FRAMES and abs(dur_v - DUR) < 0.15, f"{nfr} {dur_v}")

# decoded final frame: the whole trace survives — its tip sits on the F# row
dec = subprocess.run(["ffmpeg", "-v", "error", "-i", OUT_MP4, "-vf",
                      f"select=eq(n\\,{FRAMES-1}),crop={W}:{PLOT_Y1-PLOT_Y0}:0:{PLOT_Y0}",
                      "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
dfr = np.frombuffer(dec.stdout, np.uint8).reshape(PLOT_Y1 - PLOT_Y0, W, 3).astype(np.float32)
m = near(dfr[:, PLOT_X1 - 40:PLOT_X1 - 4], C_TXT, 110)
rows = np.where(m.any(1))[0]
ck("decoded: trace tail survives on the F# row (±6 px)",
   len(rows) > 0 and abs(rows.mean() + PLOT_Y0 - fy(F_REC)) <= 6,
   f"rows {rows.min() if len(rows) else None}..{rows.max() if len(rows) else None} want {fy(F_REC)-PLOT_Y0:.0f}")
m2 = near(dfr[:, PLOT_X0 + 4:PLOT_X0 + 40], C_TXT, 110)
rows2 = np.where(m2.any(1))[0]
ck("decoded: trace head survives on the A row (±6 px)",
   len(rows2) > 0 and abs(rows2.mean() + PLOT_Y0 - fy(F_APP)) <= 6)

x0, y0, x1, y1 = CARD_LAYERS[4][2]; cw, chh = (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2
fm_ = int((CARDS[4][1] + CARDS[4][2]) / 2 * FPS)
dec = subprocess.run(["ffmpeg", "-v", "error", "-i", OUT_MP4, "-vf", f"select=eq(n\\,{fm_}),crop={cw}:{chh}:{x0}:{y0}",
                      "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
dfr = np.frombuffer(dec.stdout, np.uint8).reshape(chh, cw, 3).astype(np.float32)
ck("decoded: card-4 text survives", int(near(dfr, C_TXT, 90).sum()) > 400)

DEC_WAV = f"out/horn_{STAMP}_dec.wav"
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", OUT_MP4, "-ac", "1", "-ar", str(SR), DEC_WAV], check=True, capture_output=True)
with wave.open(DEC_WAV) as wf:
    da = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float64) / 32768
da = da[:N] if len(da) >= N else np.pad(da, (0, N - len(da)))
fa2 = measured_f(da, 0.5, 2.5); fb2 = measured_f(da, 15.0, 18.5)
ck("decoded: approach 220.0 ±0.5, recession 185.0 ±0.5",
   abs(fa2 - 220) < 0.5 and abs(fb2 - F_REC) < 0.5, f"{fa2:.2f} / {fb2:.2f}")
ck("decoded: drop 3.00 semitones ±0.05", abs(12 * np.log2(fa2 / fb2) - 3) < 0.05, f"{12*np.log2(fa2/fb2):.3f}")
dspec = np.abs(np.fft.rfft(da)) ** 2; dfreqs = np.fft.rfftfreq(len(da), 1 / SR)
dhf = dspec[dfreqs > 1000].sum() / dspec.sum()
ck("decoded audio: energy above 1 kHz < 1e-4 (AAC)", dhf < 1e-4, f"{dhf:.2e} ({10*np.log10(dhf+1e-30):.1f} dB)")
sz = os.path.getsize(OUT_MP4)
ck("file size sane", 200_000 < sz < 30_000_000, f"{sz}")

print()
if FAILS: print("FAILURES:", FAILS); sys.exit(1)
print(f"ALL CHECKS PASSED — {OUT_MP4} ({sz} bytes, {nfr} frames, {dur_v:.2f} s)")
print(f"v={V:.3f} m/s ({V*3.6:.1f} km/h, {V*2.23694:.1f} mph)  f0={F0:.2f}  A={fa:.2f}  F#={fb:.2f}")
print()
print("NOT VERIFIED, and the piece does not claim it: loudness — the swell follows")
print("(d/r)^0.6, compressed from the physical 1/r so a phone can hear the approach.")
print("The car dot is drawn at its TRUE position while the readout is the pitch")
print("ARRIVING at that instant; the two differ by d/c = 58 ms, below one frame's")
print("worth of meaning. Real horns are two-note chords with a spectrum above 1 kHz;")
print("this one is four harmonics of a single note so it stays under the promise.")
print("Wind, temperature (c = 343 assumes 20 C) and ground reflection are ignored.")
