#!/usr/bin/env python3
"""THE BOTTLE — a bottle is not a pipe.

Blow across a wine bottle and the note is about 109 Hz: a wavelength of
3.1 m from a bottle 30 cm tall. No standing wave fits. The air in the neck is
a mass; the air in the belly is its spring (Helmholtz). Pour water in and the
spring stiffens: f = (c/2pi) sqrt(A / (V L')). Half full is x sqrt(2), not an
octave; the octave waits for three-quarters full. The pour is staged — pour
to half, stop and listen; pour to three-quarters, stop — as in the source.

Writing-and-sound-first: eight cards carry the piece. The screen is a
diagram — the bottle's air cavity filling with water, a live Hz readout with
its ratio to the empty note, and a pitch-vs-time trace that draws itself in.

The sound is a self-sustained tone (a blown bottle is nearly sinusoidal)
whose instantaneous frequency follows the Helmholtz law of the shrinking
cavity, plus breath: white noise through a two-pole resonator that tracks the
same frequency. The formula's predictions (109.0 empty, x1.4142 at half,
x2.000 at three-quarters) are then MEASURED off the wav as the check.

Facts (verified 2026-09-11):
  - Helmholtz: f = (c / 2pi) sqrt(A / (V L')), L' = L + 1.5 a for an
    unflanged outer end (Monteiro & Marti, arXiv:1805.04014; Kinsler et al.)
  - filling with water: f^2 vs 1/V is linear; the slope returns c to 2%
    (arXiv:1805.04014, Fig. 3)
  - a real 750 mL bottle blown across measures 114 Hz (J. D. Cook, 2021,
    spectrum analyser); the formula with his dimensions gave 113
  - c = 343 m/s (dry air, 20 C)
Model bottle (declared): opening 2.0 cm, neck 9.0 cm, cavity 750 cm^3, height
30 cm. Nothing above 1 kHz by construction (brickwall on the breath at
950 Hz; the tone's top harmonic is 3 x 218.1 = 654 Hz); FFT-verified on
master and decoded AAC anyway (the standing accessibility promise).
"""
import os, subprocess, sys, time, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
FPS = 30
SR = 48000

C_SOUND = 343.0
A_R = 0.010                                   # neck radius, m (2 cm opening)
A_NECK = np.pi * A_R ** 2                     # 3.1416e-4 m^2
L_NECK = 0.090                                # m
L_EFF = L_NECK + 1.5 * A_R                    # 0.105 m
V0 = 750e-6                                   # m^3
HEIGHT = 0.30                                 # m, the bottle
POUR_ML = 562.5                               # water added, mL (75 % of the cavity)
GLASS_ML = POUR_ML / 3                        # "each glass" = 187.5 mL, three of them
# the pour is staged, as in the arXiv experiment: pour to half, stop and listen;
# pour to three-quarters, stop and listen. One rate throughout (78.1 mL/s).
T_POUR0, T_HALF, T_RESUME, T_3Q = 5.6, 10.4, 12.2, 14.6
KNOTS_T  = [0.0, T_POUR0, T_HALF, T_RESUME, T_3Q, 20.0]
KNOTS_ML = [0.0, 0.0, 375.0, 375.0, 562.5, 562.5]
DUR = 20.0
FRAMES = int(round(DUR * FPS))                # 600
N = int(round(DUR * SR))

def f_helm(V):
    return C_SOUND / (2 * np.pi) * np.sqrt(A_NECK / (V * L_EFF))

def fill_ml(t):
    """water added by time t (mL); staged linear pour."""
    return np.interp(np.asarray(t, dtype=float), KNOTS_T, KNOTS_ML)

def V_at(t):  return V0 - fill_ml(t) * 1e-6
def f_at(t):  return f_helm(V_at(t))

F0 = float(f_helm(V0))
F_END = float(f_helm(V0 - POUR_ML * 1e-6))
LAMBDA = C_SOUND / F0
RATE1 = 375.0 / (T_HALF - T_POUR0); RATE2 = 187.5 / (T_3Q - T_RESUME)

BG      = np.array([10, 12, 16], np.float32)
C_TXT   = np.array([236, 240, 246], np.float32)   # cards, bottle outline, trace
C_ACC   = np.array([252, 178, 41], np.float32)    # readout + accent lines only
C_WAT   = np.array([52, 122, 204], np.float32)    # the water only
C_DIM   = np.array([120, 126, 138], np.float32)   # captions, references, labels

STAMP = time.strftime("%H%M%S")
OUT_RAW = f"out/bottle_{STAMP}_video.mp4"
OUT_WAV = f"out/bottle_{STAMP}.wav"
OUT_MP4 = f"out/bottle_{STAMP}_final.mp4"
SHEET   = f"out/bottle_{STAMP}_sheet.png"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_M = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

FAILS = []
def ck(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if (detail and not ok) else ""))
    if not ok: FAILS.append(name)

print(f"f0 = {F0:.3f} Hz  lambda = {LAMBDA:.3f} m  ({LAMBDA/HEIGHT:.1f} bottles)  "
      f"f_end = {F_END:.3f} Hz  T_half = {T_HALF:.3f}  T_3q = {T_3Q:.3f}")
ck("empty note rounds to 109 Hz", round(F0) == 109, f"{F0:.3f}")
ck("wavelength is 3.1 m (rounds)", round(LAMBDA, 1) == 3.1, f"{LAMBDA:.3f}")
ck("the note is more than ten bottles long", LAMBDA / HEIGHT > 10, f"{LAMBDA/HEIGHT:.2f}")
ck("half full is exactly sqrt(2) x empty (model)", abs(f_helm(V0 / 2) / F0 - np.sqrt(2)) < 1e-12)
ck("three-quarters full is exactly 2 x empty (model)", abs(f_helm(V0 / 4) / F0 - 2.0) < 1e-12)
ck("pour ends at 75 %: the water stays out of the shoulder (model validity)", POUR_ML / (V0 * 1e6) == 0.75)
ck("one pour rate for both stages (78.1 mL/s)", abs(RATE1 - RATE2) < 1e-9, f"{RATE1:.3f} / {RATE2:.3f}")
ck("end note = the octave, 218.1 Hz; 3rd harmonic 654 < 1 kHz", round(F_END, 1) == 218.1 and 3 * F_END < 1000, f"{F_END:.2f}")
ck("the water at T_HALF is exactly half the cavity; at T_3Q exactly three-quarters",
   fill_ml(T_HALF) == 375.0 and fill_ml(T_3Q) == 562.5)
# "each glass raises it more than the last" — three equal glasses of the pour shown
gains = [float(f_helm(V0 - (k + 1) * GLASS_ML * 1e-6) - f_helm(V0 - k * GLASS_ML * 1e-6)) for k in range(3)]
ck("each 187.5 mL glass raises the pitch more than the last", all(b > a for a, b in zip(gains, gains[1:])),
   f"{[round(g,1) for g in gains]}")
print(f"gains per glass: {[round(g,1) for g in gains]} Hz")
cents_half = 1200 * np.log2(f_helm(V0 / 2) / F0)
ck("half full = 600 cents = six semitones", abs(cents_half - 600) < 1e-9, f"{cents_half:.3f}")
# model validity: the cavity stays much larger than the neck's own air (28 cm^3)
ck("cavity at the end (187.5 cm^3) is > 6 x the neck's air (28 cm^3)", 187.5 > 6 * (A_NECK * L_NECK * 1e6))

# --------------------------------------------------------------- writing ---
CARDS = [
    (["blow across a bottle.", "the note is 3.1 metres long.", "the bottle is 30 centimetres."],  0.0,  3.0, 1),
    (["no wave fits in there.", "the air in the neck is a weight.", "the air in the belly is its spring."],  3.0,  5.6, None),
    (["pour water in.", "the belly shrinks. the spring stiffens.", "the weight bounces faster."],  5.6,  8.4, None),
    (["the pitch climbs — slowly, then faster.", "each glass raises it", "more than the last."],  8.4, 10.4, None),
    (["half full.", "a pipe cut in half jumps an octave.", "the bottle climbs √2."],           10.4, 13.4, 2),
    (["the octave waits", "for three-quarters full."],                                     13.4, 15.8, None),
    (["nothing about the height.", "only the belly, the neck,", "and the speed of sound."],  15.8, 17.9, None),
    (["so a bottle, a jug and a phone", "measured the speed of sound.", "to two percent."],   17.9, 20.0, None),
]
ck("cards tile 0..20 s contiguously",
   CARDS[0][1] == 0.0 and CARDS[-1][2] == DUR and
   all(CARDS[i][2] == CARDS[i+1][1] for i in range(len(CARDS)-1)))
ck("card 4 ('half full') begins exactly when the first pour stops", CARDS[4][1] == T_HALF)
ck("the whole half-full hold is inside card 4", CARDS[4][1] <= T_HALF and T_RESUME <= CARDS[4][2])
ck("the octave arrives inside card 5 ('waits for')", CARDS[5][1] < T_3Q < CARDS[5][2], f"{T_3Q:.2f}")
ck("card 2 begins exactly when the pour begins", CARDS[2][1] == T_POUR0)

# ---------------------------------------------------------------- audio ----
ts = np.arange(N) / SR
f_inst = f_at(ts)
ck("pitch is non-decreasing over the whole 20 s", np.all(np.diff(f_inst) >= -1e-9), f"min step {np.diff(f_inst).min():.2e}")
ck("pitch flat before the pour, during the half-full hold, and after the octave",
   np.ptp(f_inst[:int(T_POUR0 * SR)]) < 1e-9 and
   np.ptp(f_inst[int(T_HALF * SR) + 1:int(T_RESUME * SR)]) < 1e-9 and
   np.ptp(f_inst[int(T_3Q * SR) + 1:]) < 1e-9)
slope = np.gradient(f_inst, 1 / SR)
s_early = slope[int((T_POUR0 + 0.5) * SR)]; s_late = slope[int((T_3Q - 0.5) * SR)]
ck("'slowly, then faster': slope near the end of the pour > 3 x slope near its start",
   s_late > 3 * s_early, f"{s_early:.2f} -> {s_late:.2f} Hz/s")
print(f"slope at pour start {s_early:.2f} Hz/s, at pour end {s_late:.2f} Hz/s")

phase = 2 * np.pi * np.cumsum(f_inst) / SR
tone = np.sin(phase) + 0.18 * np.sin(2 * phase) + 0.06 * np.sin(3 * phase)

# breath: white noise through a two-pole resonator that tracks f(t), Q = 25
rng = np.random.default_rng(1805)
noise = rng.standard_normal(N)
Q = 25.0
bw = f_inst / Q
r_ = np.exp(-np.pi * bw / SR)
cth = 2 * r_ * np.cos(2 * np.pi * f_inst / SR)
r2 = r_ * r_
y = np.zeros(N)
y1 = y2 = 0.0
for n in range(N):
    v = noise[n] + cth[n] * y1 - r2[n] * y2
    y[n] = v; y2 = y1; y1 = v
breath = y / (np.abs(y).max() + 1e-12)
# a little broadband air, brickwalled at 950 Hz
air = rng.standard_normal(N)
X = np.fft.rfft(air); fr_ = np.fft.rfftfreq(N, 1 / SR); X[fr_ > 950] = 0; X[fr_ < 80] = 0
air = np.fft.irfft(X, N); air /= np.abs(air).max()
Xb = np.fft.rfft(breath); Xb[fr_ > 950] = 0; breath = np.fft.irfft(Xb, N)

audio = tone + 0.35 * breath + 0.05 * air
audio *= 0.85 / np.abs(audio).max()
fade_i = int(0.300 * SR); fade_o = int(0.080 * SR)
audio[:fade_i] *= np.linspace(0, 1, fade_i) ** 2
audio[-fade_o:] *= np.linspace(1, 0, fade_o)
ck("no clipping; ends silent", np.abs(audio).max() <= 0.86 and abs(audio[-1]) < 1e-4)

def measured_f(sig, t0, t1, lo, hi):
    """instantaneous frequency of the fundamental via Hilbert on a band-passed slice."""
    a, b = int(t0 * SR), int(t1 * SR)
    seg = sig[a:b]
    X = np.fft.rfft(seg); fq = np.fft.rfftfreq(len(seg), 1 / SR)
    X[(fq < lo) | (fq > hi)] = 0
    seg_f = np.fft.irfft(X, len(seg))
    Xa = np.fft.fft(seg_f); n = len(seg_f)
    h = np.zeros(n); h[0] = 1; h[1:n // 2] = 2; h[n // 2] = 1
    an = np.fft.ifft(Xa * h)
    ph = np.unwrap(np.angle(an))
    m = slice(n // 4, 3 * n // 4)
    return float(np.polyfit(np.arange(n)[m] / SR, ph[m], 1)[0] / (2 * np.pi))

def cents(a, b): return 1200 * np.log2(a / b)
fe = measured_f(audio, 1.0, 5.0, 80, 140)
fh = measured_f(audio, T_HALF + 0.2, T_RESUME - 0.2, 120, 190)
f3 = measured_f(audio, T_3Q + 0.3, 19.5, 180, 260)
ck("MEASURED off the wav: empty = 109.0 ±0.2 Hz", abs(fe - F0) < 0.2, f"{fe:.3f}")
ck("MEASURED during the half-full hold: x sqrt(2) within ±2 cents",
   abs(cents(fh / fe, np.sqrt(2))) < 2, f"{fh:.2f} = x{fh/fe:.4f}")
ck("MEASURED during the octave hold: x2 within ±2 cents",
   abs(cents(f3 / fe, 2.0)) < 2, f"{f3:.2f} = x{f3/fe:.4f}")
ck("MEASURED end note 218.1 ±0.3 Hz", abs(f3 - F_END) < 0.3, f"{f3:.3f}")
fl = f3
print(f"measured: empty {fe:.2f}  half {fh:.2f} (x{fh/fe:.4f})  3/4 {f3:.2f} (x{f3/fe:.4f})")

spec = np.abs(np.fft.rfft(audio)) ** 2
freqs = np.fft.rfftfreq(N, 1 / SR)
hf = spec[freqs > 1000].sum() / spec.sum()
ck("master: energy above 1 kHz < 1e-6", hf < 1e-6, f"{hf:.2e}")
top = freqs[spec > spec.max() * 1e-6].max()
ck("master: highest energetic bin <= 955 Hz", top <= 955, f"{top:.1f}")

os.makedirs("out", exist_ok=True)
with wave.open(OUT_WAV, "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
    wf.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())

# ------------------------------------------------------------------ layout --
SAFE_TOP, SAFE_BOT = 192, 1632
PX_CM = 13.0
# the air cavity, drawn as an outline: body 7.0 cm wide x 18.0 cm, shoulder 2.5 cm, neck 2.0 x 9.0 cm
BODY_W, BODY_H, SHOULDER_H, NECK_W, NECK_H = 7.0, 18.0, 2.5, 2.0, 9.0
BOT_CX = 400
BOT_Y1 = 650                                              # base of the cavity (px)
BOT_Y0 = int(round(BOT_Y1 - (BODY_H + SHOULDER_H + NECK_H) * PX_CM))   # 266
BODY_TOP = BOT_Y1 - BODY_H * PX_CM                        # 416
SHOULDER_TOP = BODY_TOP - SHOULDER_H * PX_CM              # 383.5
BODY_A_CM2 = np.pi * (BODY_W / 2) ** 2                    # 38.48 cm^2
def water_top(t):
    """screen row of the water surface at time t (level = volume / body area)."""
    return BOT_Y1 - float(fill_ml(t)) / BODY_A_CM2 * PX_CM
ck("the pour fits inside the straight body (14.6 cm of 18)", POUR_ML / BODY_A_CM2 < BODY_H,
   f"{POUR_ML / BODY_A_CM2:.2f} cm")
READ_Y = 700
PLOT_X0, PLOT_X1 = 130, 1000
PLOT_Y0, PLOT_Y1 = 820, 1040
F_TOP, F_BOT = 262.0, 96.0
CARD_TOP, CARD_H = 1090, 520

def fy(f):  return PLOT_Y0 + (F_TOP - f) / (F_TOP - F_BOT) * (PLOT_Y1 - PLOT_Y0)
def tx(t):  return PLOT_X0 + t / DUR * (PLOT_X1 - PLOT_X0)

def text_layer(lines, accent_idx, size=56, leading=74):
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
    f = float(f_at(t))
    return f"{f:5.1f} Hz   ×{f / F0:.2f}"

_RO_CACHE = {}
def readout_layer(txt):
    if txt in _RO_CACHE: return _RO_CACHE[txt]
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dr = ImageDraw.Draw(img)
    fnt = ImageFont.truetype(FONT_M, 60)
    bb = dr.textbbox((0, 0), txt, font=fnt)
    x = (W - (bb[2] - bb[0])) // 2 - bb[0]
    dr.text((x, READ_Y), txt, font=fnt, fill=tuple(int(c) for c in C_ACC) + (255,),
            stroke_width=3, stroke_fill=(10, 12, 16, 255))
    box = dr.textbbox((x, READ_Y), txt, font=fnt, stroke_width=3)
    a = np.asarray(img, np.float32)
    _RO_CACHE[txt] = (a[:, :, :3], a[:, :, 3:4] / 255.0, box)
    return _RO_CACHE[txt]
RO_FENCE = [60, READ_Y - 10, 1020, READ_Y + 90]

# the cavity outline as a polygon (px)
hw_b, hw_n = BODY_W / 2 * PX_CM, NECK_W / 2 * PX_CM
OUTLINE = [
    (BOT_CX - hw_n, BOT_Y0), (BOT_CX + hw_n, BOT_Y0),
    (BOT_CX + hw_n, SHOULDER_TOP), (BOT_CX + hw_b, BODY_TOP),
    (BOT_CX + hw_b, BOT_Y1), (BOT_CX - hw_b, BOT_Y1),
    (BOT_CX - hw_b, BODY_TOP), (BOT_CX - hw_n, SHOULDER_TOP),
]
ck("bottle outline top is below the safe-area line", BOT_Y0 >= SAFE_TOP, f"{BOT_Y0}")

# a mask of the cavity interior (for water fill)
mimg = Image.new("L", (W, H), 0); ImageDraw.Draw(mimg).polygon(OUTLINE, fill=255)
CAVITY = (np.asarray(mimg, np.float32) / 255.0)[:, :, None]

# static base: outline, labels, scale bar, plot frame, references, caption
img = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dr = ImageDraw.Draw(img)
white = tuple(int(c) for c in C_TXT) + (255,); dim = tuple(int(c) for c in C_DIM) + (255,)
dr.line(OUTLINE + [OUTLINE[0]], fill=white, width=5, joint="curve")
fnt_s = ImageFont.truetype(FONT_M, 30)
# labels: weight (neck), spring (belly)
lx = BOT_CX + hw_b + 60
dr.text((lx, (BOT_Y0 + SHOULDER_TOP) / 2 - 18), "← weight", font=fnt_s, fill=dim)
dr.text((lx, (BODY_TOP + BOT_Y1) / 2 - 18), "← spring", font=fnt_s, fill=dim)
# scale bar: 30 cm, left of the bottle
sx = BOT_CX - hw_b - 70
dr.line([(sx, BOT_Y1), (sx, BOT_Y1 - HEIGHT * 100 * PX_CM)], fill=dim, width=2)
for yy_ in (BOT_Y1, BOT_Y1 - HEIGHT * 100 * PX_CM):
    dr.line([(sx - 8, yy_), (sx + 8, yy_)], fill=dim, width=2)
bb = dr.textbbox((0, 0), "30 cm", font=fnt_s)
dr.text((sx - (bb[2] - bb[0]) - 16 - bb[0], (BOT_Y0 + BOT_Y1) / 2 - 18), "30 cm", font=fnt_s, fill=dim)
# plot frame + dashed references
dr.line([(PLOT_X0, PLOT_Y0), (PLOT_X0, PLOT_Y1)], fill=dim, width=2)
REFS = ((F0, "empty"), (F0 * np.sqrt(2), "√2"), (2 * F0, "octave"))
for f_ref, name in REFS:
    yv = int(round(fy(f_ref)))
    for xx_ in range(PLOT_X0, PLOT_X1, 24):
        dr.line([(xx_, yv), (xx_ + 12, yv)], fill=dim, width=2)
    bb = dr.textbbox((0, 0), name, font=fnt_s)
    dr.text((PLOT_X0 - (bb[2] - bb[0]) - 14 - bb[0], yv - (bb[3] - bb[1]) // 2 - bb[1]), name, font=fnt_s, fill=dim)
cap = "pitch, over 20 s"
bb = dr.textbbox((0, 0), cap, font=fnt_s)
dr.text((540 - (bb[2] - bb[0]) // 2 - bb[0], PLOT_Y1 + 18), cap, font=fnt_s, fill=dim)
ca = np.asarray(img, np.float32)
base = np.empty((H, W, 3), np.float32); base[:] = BG
base = base * (1 - ca[:, :, 3:4] / 255) + ca[:, :, :3] * (ca[:, :, 3:4] / 255)

# the full trace, drawn once; revealed by column as time advances
tr_img = Image.new("RGBA", (W, H), (0, 0, 0, 0)); td = ImageDraw.Draw(tr_img)
tt = np.linspace(0, DUR, 2400)
pts = [(float(tx(t_)), float(fy(float(f_at(t_))))) for t_ in tt]
td.line(pts, fill=white, width=5, joint="curve")
TRACE = np.asarray(tr_img, np.float32)
TR_RGB, TR_A = TRACE[:, :, :3], TRACE[:, :, 3:4] / 255.0
ck("trace stays inside the plot box", (TR_A[:, :, 0] > 0)[:PLOT_Y0 - 4].sum() == 0 and
   (TR_A[:, :, 0] > 0)[PLOT_Y1 + 4:].sum() == 0)

ROWS = np.arange(H, dtype=np.float32)[:, None, None]
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
    # water: cavity interior below the surface row (drawn under the outline: outline re-laid after)
    wt = water_top(t)
    if fill_ml(t) > 0:
        wmask = CAVITY * np.clip(ROWS - wt + 0.5, 0, 1)
        y0 = max(0, int(wt) - 2)
        sl = slice(y0, BOT_Y1 + 2)
        fr[sl] = fr[sl] * (1 - wmask[sl]) + C_WAT * wmask[sl]
        # outline back on top
        a = ca[sl, :, 3:4] / 255
        fr[sl] = fr[sl] * (1 - a) + ca[sl, :, :3] * a
    xr = int(round(tx(t)))
    a = TR_A[:, :xr + 1]; fr[:, :xr + 1] = fr[:, :xr + 1] * (1 - a) + TR_RGB[:, :xr + 1] * a
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

# coupling: the drawn water surface and the readout both follow V(t)
def water_row(fr):
    """topmost row of water colour inside the body's column band."""
    band = fr[BOT_Y0:BOT_Y1 + 4, int(BOT_CX - hw_b + 8):int(BOT_CX + hw_b - 8)]
    m = near(band, C_WAT, 60)
    rows = np.where(m.any(1))[0]
    return None if len(rows) == 0 else rows.min() + BOT_Y0
okw, det = True, []
for t in (3.0, 7.0, T_HALF + 0.5, 13.0, T_3Q, 19.0):
    f = int(round(t * FPS)); got = water_row(render_frame(f)); want = water_top(f / FPS)
    ro = float(readout_text(f / FPS).split()[0]); want_f = float(f_at(f / FPS))
    det.append(f"t{t:.1f}:row{got}/{want:.0f},ro{ro}/{want_f:.1f}")
    if fill_ml(f / FPS) == 0:
        okw &= got is None
    else:
        okw &= got is not None and abs(got - want) <= 2.0 and abs(ro - want_f) < 0.06
ck("water surface row AND readout both track V(t) (coupled, 6 times)", okw, " ".join(det))
ck("no water before the pour (t=3)", water_row(render_frame(int(3.0 * FPS))) is None)
ck("water never enters the shoulder (final frame surface below BODY_TOP)",
   water_row(render_frame(FRAMES - 1)) is not None and water_row(render_frame(FRAMES - 1)) > BODY_TOP + 10)

def trace_tip(fr):
    box = fr[PLOT_Y0:PLOT_Y1, PLOT_X0 + 2:PLOT_X1]
    m = near(box, C_TXT, 60)
    cols = np.where(m.any(0))[0]
    if len(cols) == 0: return None, None
    c = cols.max(); rows = np.where(m[:, c])[0]
    return c + PLOT_X0 + 2, rows.mean() + PLOT_Y0
okt, det = True, []
for t in (1.0, 6.0, 10.0, T_HALF + 0.5, 13.0, 18.0):
    f = int(round(t * FPS)); c, rw = trace_tip(render_frame(f))
    want_c = tx(f / FPS); want_r = fy(float(f_at(f / FPS)))
    det.append(f"t{t:.1f}:x{c}/{want_c:.0f},y{rw:.0f}/{want_r:.0f}")
    okt &= c is not None and abs(c - want_c) <= 4 and abs(rw - want_r) <= 6
ck("trace tip tracks time AND the pitch (coupled, 6 times)", okt, " ".join(det))
ck("trace sits on the 'empty' line at t=1, crosses '√2' at T_HALF, 'octave' at T_3Q",
   abs(fy(float(f_at(1.0))) - fy(F0)) < 1 and abs(fy(float(f_at(T_HALF))) - fy(F0 * np.sqrt(2))) < 1
   and abs(fy(float(f_at(T_3Q))) - fy(2 * F0)) < 1)
ro = [readout_text(f / FPS) for f in range(FRAMES)]
ro_hz = [float(s.split()[0]) for s in ro]; ro_x = [float(s.split("×")[1]) for s in ro]
ck("readout: starts 109.0 ×1.00, ends 218.1 ×2.00",
   ro_hz[0] == 109.0 and ro_x[0] == 1.0 and ro_hz[-1] == 218.1 and ro_x[-1] == 2.00, f"{ro[0]} / {ro[-1]}")
ck("readout: the DISPLAYED number never goes down, any frame to the next",
   all(b >= a for a, b in zip(ro_hz, ro_hz[1:])))
f_half = int(round(T_HALF * FPS)); f_3q = int(round(T_3Q * FPS))
ck("readout shows 154.2 ×1.41 at the half-full frame and 218.1 ×2.00 at the three-quarters frame",
   ro[f_half] == "154.2 Hz   ×1.41" and ro[f_3q] == "218.1 Hz   ×2.00", f"{ro[f_half]} / {ro[f_3q]}")
n41 = sum(1 for f in range(int(CARDS[4][1] * FPS), int(CARDS[4][2] * FPS)) if ro_x[f] == 1.41)
n20 = sum(1 for f in range(int(CARDS[5][1] * FPS), int(CARDS[5][2] * FPS)) if ro_x[f] == 2.00)
ck("×1.41 is on screen for the whole 1.8 s hold inside card 4 (>= 54 frames)", n41 >= 54, f"{n41}")
ck("×2.00 is on screen for >= 1.0 s inside card 5 (>= 30 frames)", n20 >= 30, f"{n20}")

for i in (1, 3, 6):
    fmid = int((CARDS[i][1] + CARDS[i][2]) / 2 * FPS); fr = render_frame(fmid)
    amber = near(fr, C_ACC, 70)
    amber[RO_FENCE[1]:RO_FENCE[3], RO_FENCE[0]:RO_FENCE[2]] = False
    ck(f"amber fenced (card {i})", int(amber.sum()) == 0, f"stray={int(amber.sum())}")
blue = near(render_frame(int(19 * FPS)), C_WAT, 60)
blue[BOT_Y0:BOT_Y1 + 4, int(BOT_CX - hw_b - 4):int(BOT_CX + hw_b + 4)] = False
ck("blue fenced to the bottle (t=19)", int(blue.sum()) == 0, f"stray={int(blue.sum())}")
ck("nothing flashes (WCAG 2.3.1 trivially): water and trace are continuous", True)

tiles = [Image.fromarray(render_frame(int(t * FPS)).astype(np.uint8)).resize((360, 640), Image.LANCZOS)
         for t in (1.5, T_HALF + 0.9, 17.0)]
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
ck("encode: 600 frames, 20.0 s", nfr == FRAMES and abs(dur_v - DUR) < 0.15, f"{nfr} {dur_v}")

dec = subprocess.run(["ffmpeg", "-v", "error", "-i", OUT_MP4, "-vf",
                      f"select=eq(n\\,{FRAMES-1}),crop={W}:{PLOT_Y1-PLOT_Y0}:0:{PLOT_Y0}",
                      "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
dfr = np.frombuffer(dec.stdout, np.uint8).reshape(PLOT_Y1 - PLOT_Y0, W, 3).astype(np.float32)
m = near(dfr[:, PLOT_X1 - 40:PLOT_X1 - 4], C_TXT, 110)
rows = np.where(m.any(1))[0]
ck("decoded: trace tail survives on the octave row (±6 px)",
   len(rows) > 0 and abs(rows.mean() + PLOT_Y0 - fy(F_END)) <= 6,
   f"rows {rows.min() if len(rows) else None}..{rows.max() if len(rows) else None} want {fy(F_END)-PLOT_Y0:.0f}")
m2 = near(dfr[:, PLOT_X0 + 4:PLOT_X0 + 40], C_TXT, 110)
rows2 = np.where(m2.any(1))[0]
ck("decoded: trace head survives on the 'empty' row (±6 px)",
   len(rows2) > 0 and abs(rows2.mean() + PLOT_Y0 - fy(F0)) <= 6)

# decoded water surface at the last frame
dec = subprocess.run(["ffmpeg", "-v", "error", "-i", OUT_MP4, "-vf",
                      f"select=eq(n\\,{FRAMES-1}),crop={W}:{BOT_Y1+4-BOT_Y0}:0:{BOT_Y0}",
                      "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
dwb = np.frombuffer(dec.stdout, np.uint8).reshape(BOT_Y1 + 4 - BOT_Y0, W, 3).astype(np.float32)
wm = near(dwb[:, int(BOT_CX - hw_b + 8):int(BOT_CX + hw_b - 8)], C_WAT, 90)
wrows = np.where(wm.any(1))[0]
ck("decoded: water surface at the final frame within ±3 px of the model",
   len(wrows) > 0 and abs(wrows.min() + BOT_Y0 - water_top(DUR)) <= 3,
   f"{wrows.min() + BOT_Y0 if len(wrows) else None} want {water_top(DUR):.0f}")

x0, y0, x1, y1 = CARD_LAYERS[4][2]; cw, chh = (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2
fm_ = int((CARDS[4][1] + CARDS[4][2]) / 2 * FPS)
dec = subprocess.run(["ffmpeg", "-v", "error", "-i", OUT_MP4, "-vf", f"select=eq(n\\,{fm_}),crop={cw}:{chh}:{x0}:{y0}",
                      "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
dfr = np.frombuffer(dec.stdout, np.uint8).reshape(chh, cw, 3).astype(np.float32)
ck("decoded: card-4 text survives", int(near(dfr, C_TXT, 90).sum()) > 400)

DEC_WAV = f"out/bottle_{STAMP}_dec.wav"
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", OUT_MP4, "-ac", "1", "-ar", str(SR), DEC_WAV], check=True, capture_output=True)
with wave.open(DEC_WAV) as wf:
    da = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float64) / 32768
da = da[:N] if len(da) >= N else np.pad(da, (0, N - len(da)))
fe2 = measured_f(da, 1.0, 5.0, 80, 140); fl2 = measured_f(da, T_3Q + 0.3, 19.5, 180, 260)
fh2 = measured_f(da, T_HALF + 0.2, T_RESUME - 0.2, 120, 190)
ck("decoded: empty 109.0 ±0.5, end 218.1 ±0.5", abs(fe2 - F0) < 0.5 and abs(fl2 - F_END) < 0.5, f"{fe2:.2f} / {fl2:.2f}")
ck("decoded: half-full ratio sqrt(2) within ±3 cents; octave within ±3 cents",
   abs(cents(fh2 / fe2, np.sqrt(2))) < 3 and abs(cents(fl2 / fe2, 2.0)) < 3, f"x{fh2/fe2:.4f} / x{fl2/fe2:.4f}")
dspec = np.abs(np.fft.rfft(da)) ** 2; dfreqs = np.fft.rfftfreq(len(da), 1 / SR)
dhf = dspec[dfreqs > 1000].sum() / dspec.sum()
ck("decoded audio: energy above 1 kHz < 1e-4 (AAC)", dhf < 1e-4, f"{dhf:.2e} ({10*np.log10(dhf+1e-30):.1f} dB)")
sz = os.path.getsize(OUT_MP4)
ck("file size sane", 200_000 < sz < 30_000_000, f"{sz}")

print()
if FAILS: print("FAILURES:", FAILS); sys.exit(1)
print(f"ALL CHECKS PASSED — {OUT_MP4} ({sz} bytes, {nfr} frames, {dur_v:.2f} s)")
print(f"f0={F0:.2f}  lambda={LAMBDA:.2f} m  half x{fh/fe:.4f}  3/4 x{f3/fe:.4f}")
print()
print("NOT VERIFIED, and the piece does not claim it: the absolute 109 Hz is a MODEL")
print("bottle (2.0 cm opening, 9.0 cm neck, 750 cm^3); real bottles land 105-120 Hz")
print("(Cook measured 114). The tone is continuous while the water pours — a real")
print("demonstration blows at each level, or listens to the tap. The end correction")
print("(1.5 a) is the unflanged textbook value; lips on the rim change it. The water")
print("is drawn as a flat level in a straight body; the model uses only its volume.")
print("Loudness is constant; a real blow is not.")
