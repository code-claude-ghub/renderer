#!/usr/bin/env python3
"""THE ATLAS — pitch is a place: the cochlea as a map of sound.

Sourced by @Invisiblelight ("we got something here with sound vision").
The claim: hearing is already seeing — the basilar membrane sorts frequency
by position (tonotopy) before the brain gets anything. Four pure sine notes
(A2 110, A3 220, A4 440, A5 880 Hz — deliberately NOTHING above 1 kHz,
standing accessibility promise, FFT-verified on master AND decoded AAC)
each land a dot at their Greenwood address on an unrolled ~35 mm ribbon.
The closing chord lights all four at once: a chord is a constellation.

Facts verified 2026-09-10:
  - Greenwood (1990): f(x) = 165.4 * (10^(2.1 x) - 0.88), x = proportion of
    basilar-membrane length from the APEX; 19.85 Hz at x=0, 20,677 Hz at x=1
    over the conventional 35 mm (real cochleas average ~31.5 mm; ends pinned).
  - ~3,500 inner hair cells in one row (the sensors; ~95% of auditory-nerve
    fibres report from them). Spiral: ~2.5-2.75 turns.
  - Positions (mm from apex): 110 Hz -> 3.15, 220 -> 5.74, 440 -> 9.15,
    880 -> 13.21. Octave gaps STRICTLY INCREASING (2.59, 3.41, 4.05 mm);
    top audible octave 10->20 kHz spans 4.96 mm.
  - HONESTY: place is the map, not the whole code — below ~1 kHz the nerve
    also phase-locks to the waveform. Said in the description, not hidden.

The drawn spiral is a schematic (Euler spiral), declared as such.
"""
import os, subprocess, sys, time, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
FPS = 30
DUR = 36.8
FRAMES = int(round(DUR * FPS))            # 1104
SR = 48000

BG    = np.array([10, 12, 16], np.float32)
C_TXT = np.array([236, 240, 246], np.float32)   # card text (white)
C_RIB = np.array([116, 138, 160], np.float32)   # ribbon / spiral / ticks
C_DOT = np.array([252, 178, 41], np.float32)    # AMBER: dots + accent lines ONLY
C_DIM = np.array([150, 160, 172], np.float32)   # small labels (grey)

STAMP = time.strftime("%H%M%S")
OUT_RAW = f"out/cochlea_{STAMP}_video.mp4"
OUT_WAV = f"out/cochlea_{STAMP}.wav"
OUT_MP4 = f"out/cochlea_{STAMP}_final.mp4"
SHEET   = f"out/cochlea_{STAMP}_sheet.png"

FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_M = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

FAILS = []
def ck(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if (detail and not ok) else ""))
    if not ok:
        FAILS.append(name)

# ------------------------------------------------------------ the map ------
A_G, a_G, k_G = 165.4, 2.1, 0.88
L_MM = 35.0
def greenwood_f(x):           # x: proportion from apex -> Hz
    return A_G * (10 ** (a_G * x) - k_G)
def greenwood_x(f):           # Hz -> proportion from apex
    return np.log10(f / A_G + k_G) / a_G

ck("Greenwood ends: 19.85 Hz at apex, 20677 Hz at base",
   abs(greenwood_f(0) - 19.848) < 0.01 and abs(greenwood_f(1) - 20677) < 5,
   f"f(0)={greenwood_f(0):.3f} f(1)={greenwood_f(1):.0f}")

NOTES = [110.0, 220.0, 440.0, 880.0]
MM = [greenwood_x(f) * L_MM for f in NOTES]
GAPS = [MM[i + 1] - MM[i] for i in range(3)]
ck("positions mm from apex = 3.15 / 5.74 / 9.15 / 13.21",
   all(abs(m - e) < 0.01 for m, e in zip(MM, [3.149, 5.740, 9.154, 13.208])),
   f"mm={[f'{m:.3f}' for m in MM]}")
ck("octave gaps strictly increasing (the stretching ruler)",
   GAPS[0] < GAPS[1] < GAPS[2], f"gaps={[f'{g:.3f}' for g in GAPS]}")
ck("card figures 3.1 / 2.6 / 3.4 / 4.1 match the map to 0.05 mm",
   abs(MM[0] - 3.1) < 0.05 and abs(GAPS[0] - 2.6) < 0.05
   and abs(GAPS[1] - 3.4) < 0.05 and abs(GAPS[2] - 4.1) < 0.05,
   f"mm0={MM[0]:.3f} gaps={[f'{g:.3f}' for g in GAPS]}")
top_oct = (greenwood_x(20000) - greenwood_x(10000)) * L_MM
ck("top audible octave (10->20 kHz) spans ~5 mm", abs(top_oct - 4.96) < 0.05,
   f"{top_oct:.3f}")

# ------------------------------------------------------------ schedule -----
T_SPIRAL = 3.6                 # spiral fades in
T_UNROLL0, T_UNROLL1 = 7.0, 9.6
T_ONSET = [10.6, 13.8, 16.8, 19.2]
NOTE_LEN = 2.4
T_TICKS = 22.0
T_CHORD = 29.9
CHORD_LEN = 3.2

CARDS = [  # (lines, t0, t1, accent_idx)
    (["you don't hear sound.", "you hear places."],            0.0,  3.4, None),
    (["in your ear: a spiral", "the size of a pea.",
      "unrolled — about 35 mm."],                              3.4,  7.0, None),
    (["unroll it."],                                           7.0,  9.8, None),
    (["every pitch that exists", "has an address here.",
      "110 Hz: 3.1 mm from the tip."],                         9.8, 13.2, 2),
    (["double it — one octave.", "the address moves 2.6 mm."], 13.2, 16.2, None),
    (["double again:", "3.4 mm."],                             16.2, 19.0, None),
    (["again: 4.1 mm.", "the same interval,", "wider every time."],
                                                               19.0, 21.6, None),
    (["your ear's ruler is logarithmic.", "at the top of hearing,",
      "an octave spans 5 mm."],                                21.6, 25.8, None),
    (["3,500 cells read this ribbon,", "one row, sorted by pitch.",
      "a retina for sound."],                                  25.8, 29.4, 2),
    (["a chord is not one sound.", "it is a constellation."],  29.4, 33.6, None),
    (["you have been seeing", "sound all along."],             33.6, 36.8, None),
]
ok = CARDS[0][1] == 0 and abs(CARDS[-1][2] - DUR) < 1e-9 and all(
    CARDS[i][2] == CARDS[i + 1][1] for i in range(len(CARDS) - 1))
ck("cards tile 0..36.8 s contiguously", ok)

# ------------------------------------------------------------ audio --------
N = int(round(DUR * SR))
audio = np.zeros(N, np.float64)

def add_note(f0, t_on, dur, amp, tau):
    n = int(dur * SR)
    t = np.arange(n) / SR
    att = 0.5 - 0.5 * np.cos(np.pi * np.minimum(t / 0.040, 1.0))
    rel = 0.5 - 0.5 * np.cos(np.pi * np.minimum((dur - t) / 0.12, 1.0))
    env = att * rel * np.exp(-t / tau)
    s0 = int(round(t_on * SR))
    audio[s0:s0 + n] += amp * env * np.sin(2 * np.pi * f0 * t)

AMPS = [0.50, 0.42, 0.36, 0.32]
for f0, t_on, amp in zip(NOTES, T_ONSET, AMPS):
    add_note(f0, t_on, NOTE_LEN, amp, 0.55)
for f0, amp in zip(NOTES, [0.30, 0.25, 0.21, 0.18]):
    add_note(f0, T_CHORD, CHORD_LEN, amp, 0.90)
fade_n = int(0.025 * SR)
audio[-fade_n:] *= np.linspace(1, 0, fade_n)

ck("no clipping, seam silent",
   np.abs(audio).max() <= 0.9 and abs(audio[-1]) < 1e-4,
   f"peak={np.abs(audio).max():.3f}")
spec = np.abs(np.fft.rfft(audio)) ** 2
freqs = np.fft.rfftfreq(N, 1 / SR)
hf = spec[freqs > 1000].sum() / spec.sum()
ck("master audio: energy above 1 kHz < 1e-5", hf < 1e-5, f"hf={hf:.2e}")

os.makedirs("out", exist_ok=True)
wav16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
with wave.open(OUT_WAV, "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
    wf.writeframes(wav16.tobytes())

# ------------------------------------------------------------ layout -------
SAFE_TOP, SAFE_BOT = 192, 1632
RIB_Y = 760
RIB_X0, RIB_X1 = 140, 940
PX_MM = (RIB_X1 - RIB_X0) / L_MM          # 22.857 px per mm
DOT_X = [RIB_X0 + m * PX_MM for m in MM]  # 212, 271, 349, 442
CARD_TOP = 1000

def mm_to_x(mm): return RIB_X0 + mm * PX_MM

# --- unroll geometry: Euler-style spiral (curvature ~ distance-from-base),
#     arc length fixed at 800 px, base end anchored; u=1 is EXACTLY the ribbon.
N_S = 800
S_N = np.linspace(0, 1, N_S)              # 0 = tip(apex), 1 = base
THETA_TOT = 2.5 * 2 * np.pi               # ~2.5 turns of total turning

def curve(u):
    """Sample points (x, y) of the partly-unrolled ribbon at unroll u∈[0,1]."""
    th = (1 - u) * THETA_TOT * (1 - S_N) ** 2      # heading per sample
    dx = np.cos(th); dy = -np.sin(th)
    x = np.concatenate([[0], np.cumsum(dx[:-1])])
    y = np.concatenate([[0], np.cumsum(dy[:-1])])
    # anchor: base end at lerped anchor point
    ax = RIB_X1
    ay0, ay1 = 470.0, float(RIB_Y)
    ay = ay0 + (ay1 - ay0) * u
    x = x - x[-1] + ax
    y = y - y[-1] + ay
    return x, y

# straightness at u=1
xs, ys = curve(1.0)
ck("unroll u=1 IS the ribbon: y flat, ends at 140/940 ±1 px",
   np.abs(ys - RIB_Y).max() < 0.5 and abs(xs[0] - RIB_X0) < 1.5
   and abs(xs[-1] - RIB_X1) < 0.5,
   f"ymax_dev={np.abs(ys - RIB_Y).max():.2f} x0={xs[0]:.1f} x1={xs[-1]:.1f}")
xs0, ys0 = curve(0.0)
ck("spiral (u=0) inside its region (x 60..1020, y 210..700)",
   xs0.min() > 60 and xs0.max() < 1020 and ys0.min() > 210 and ys0.max() < 700,
   f"x[{xs0.min():.0f},{xs0.max():.0f}] y[{ys0.min():.0f},{ys0.max():.0f}]")

# gaussian stamp for curve ink
KR = 4
ky, kx = np.mgrid[-KR:KR + 1, -KR:KR + 1]
KERN = np.exp(-(kx ** 2 + ky ** 2) / (2 * 1.5 ** 2)).astype(np.float32)

def ink_curve(u):
    """Accumulate curve ink into a (H, W) float layer, 0..1."""
    lay = np.zeros((H, W), np.float32)
    x, y = curve(u)
    xi = np.round(x).astype(int); yi = np.round(y).astype(int)
    for px, py in zip(xi, yi):
        lay[py - KR:py + KR + 1, px - KR:px + KR + 1] += KERN
    return np.clip(lay / 2.2, 0, 1)

RIBBON_LAYER = ink_curve(1.0)             # static after unroll

# dot glow kernels
GR = 30
gy, gx = np.mgrid[-GR:GR + 1, -GR:GR + 1]
G_CORE = np.exp(-(gx ** 2 + gy ** 2) / (2 * 3.5 ** 2)).astype(np.float32)
G_GLOW = np.exp(-(gx ** 2 + gy ** 2) / (2 * 11.0 ** 2)).astype(np.float32)
DOT_STAMP = np.clip(G_CORE + 0.55 * G_GLOW, 0, 1)

# tick layer (A0..A9 = 27.5 * 2^k) + 5 mm bracket for the top audible octave
TICKS_HZ = [27.5 * 2 ** k for k in range(10)]
TICK_X = [int(round(mm_to_x(greenwood_x(f) * L_MM))) for f in TICKS_HZ]
tick_layer = np.zeros((H, W), np.float32)
for tx in TICK_X:
    tick_layer[RIB_Y - 26:RIB_Y - 8, tx - 1:tx + 2] = 1.0
bx0 = int(round(mm_to_x(greenwood_x(10000) * L_MM)))
bx1 = int(round(mm_to_x(greenwood_x(20000) * L_MM)))
tick_layer[RIB_Y + 34:RIB_Y + 36, bx0:bx1 + 1] = 1.0      # bracket bar
tick_layer[RIB_Y + 26:RIB_Y + 36, bx0:bx0 + 2] = 1.0
tick_layer[RIB_Y + 26:RIB_Y + 36, bx1 - 1:bx1 + 1] = 1.0

# ------------------------------------------------------------ text layers --
def text_layer(items):
    """items: list of (text, x, y, font, size, colour, align). Returns rgb, a, box."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    box = [W, H, 0, 0]
    for txt, x, y, fp, size, col, align in items:
        fnt = ImageFont.truetype(fp, size)
        bb = dr.textbbox((0, 0), txt, font=fnt)
        if align == "c":   x = x - (bb[2] - bb[0]) // 2 - bb[0]
        elif align == "r": x = x - (bb[2] - bb[0]) - bb[0]
        dr.text((x, y), txt, font=fnt, fill=tuple(int(c) for c in col) + (255,),
                stroke_width=3, stroke_fill=(10, 12, 16, 255))
        b2 = dr.textbbox((x, y), txt, font=fnt, stroke_width=3)
        box = [min(box[0], b2[0]), min(box[1], b2[1]),
               max(box[2], b2[2]), max(box[3], b2[3])]
    a = np.asarray(img, np.float32)
    return a[:, :, :3], a[:, :, 3:4] / 255.0, box

def card_layer(lines, accent):
    size, leading = 54, 76
    fnt = ImageFont.truetype(FONT_B, size)
    dr = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    while max(dr.textbbox((0, 0), ln, font=fnt)[2] for ln in lines) > 980:
        size -= 2; fnt = ImageFont.truetype(FONT_B, size)
    block_h = leading * len(lines)
    y0 = CARD_TOP + (480 - block_h) // 2
    items = [(ln, W // 2, y0 + i * leading, FONT_B, size,
              (C_DOT if i == accent else C_TXT), "c")
             for i, ln in enumerate(lines)]
    return text_layer(items)

CARD_LAYERS = [card_layer(c[0], c[3]) for c in CARDS]
for i, (_, _, box) in enumerate(CARD_LAYERS):
    ck(f"card {i} inside safe area", box[1] >= SAFE_TOP and box[3] <= SAFE_BOT,
       f"box={box}")

# end labels (above ribbon), note labels (below, staggered), 5 mm label
END_RGB, END_A, END_BOX = text_layer([
    ("20 Hz · the tip",   RIB_X0, 690, FONT_M, 26, C_DIM, "l"),
    ("20 kHz · the door", RIB_X1, 690, FONT_M, 26, C_DIM, "r"),
])
NOTE_LABELS = []
for i, (f0, dx) in enumerate(zip(NOTES, DOT_X)):
    yy_ = 800 if i % 2 == 0 else 846
    NOTE_LABELS.append(text_layer([(f"{int(f0)}", dx, yy_, FONT_M, 28, C_DIM, "c")]))
MM5_RGB, MM5_A, MM5_BOX = text_layer([
    ("5 mm", (bx0 + bx1) // 2, 852, FONT_M, 26, C_DIM, "c")])

lab_boxes = [b for _, _, b in NOTE_LABELS]
ok = all(lab_boxes[i][2] < lab_boxes[i + 1][0] or
         (lab_boxes[i][1] > lab_boxes[i + 1][3] or lab_boxes[i][3] < lab_boxes[i + 1][1])
         for i in range(3))
ck("note labels don't collide (staggered rows)", ok,
   f"boxes={lab_boxes}")
ck("labels inside safe area",
   END_BOX[1] >= SAFE_TOP and MM5_BOX[3] <= SAFE_BOT
   and all(b[1] >= SAFE_TOP and b[3] <= SAFE_BOT for b in lab_boxes))

# ------------------------------------------------------------ compositing --
def smoothstep(t):
    t = np.clip(t, 0, 1); return t * t * (3 - 2 * t)

FADE_F = 7
def card_alpha(f, i):
    f0, f1 = CARDS[i][1] * FPS, CARDS[i][2] * FPS
    return max(0.0, min(1.0, min((f - f0) / FADE_F, (f1 - f) / FADE_F)))

def dot_bright(t, i):
    b = 0.0
    if t >= T_ONSET[i]:
        b = 0.40 + 0.60 * np.exp(-(t - T_ONSET[i]) / 0.45)
    if t >= T_CHORD:
        b = min(1.0, b + 0.60 * np.exp(-(t - T_CHORD) / 0.60))
    return b

def render_frame(f):
    t = f / FPS
    fr = np.empty((H, W, 3), np.float32); fr[:] = BG

    # ribbon / spiral
    if t >= T_SPIRAL:
        fade = smoothstep((t - T_SPIRAL) / 0.8)
        if t < T_UNROLL0:
            lay = ink_curve(0.0) if not hasattr(render_frame, "_sp") else render_frame._sp
            render_frame._sp = lay
        elif t < T_UNROLL1:
            u = smoothstep((t - T_UNROLL0) / (T_UNROLL1 - T_UNROLL0))
            lay = ink_curve(float(u))
        else:
            lay = RIBBON_LAYER
        fr += lay[:, :, None] * fade * C_RIB

    # end labels after unroll
    if t >= T_UNROLL1 + 0.2:
        a = END_A * smoothstep((t - T_UNROLL1 - 0.2) / 0.5)
        fr = fr * (1 - a) + END_RGB * a

    # ticks + 5 mm bracket
    if t >= T_TICKS:
        a = smoothstep((t - T_TICKS) / 0.6)
        fr += tick_layer[:, :, None] * (0.75 * a) * C_RIB
        aa = MM5_A * a
        fr = fr * (1 - aa) + MM5_RGB * aa

    # dots + their labels
    for i, dx in enumerate(DOT_X):
        b = dot_bright(t, i)
        if b > 0:
            xi = int(round(dx))
            fr[RIB_Y - GR:RIB_Y + GR + 1, xi - GR:xi + GR + 1] += \
                DOT_STAMP[:, :, None] * b * C_DOT
        if t >= T_ONSET[i] + 0.2:
            rgb, al, _ = NOTE_LABELS[i]
            a = al * smoothstep((t - T_ONSET[i] - 0.2) / 0.5)
            fr = fr * (1 - a) + rgb * a

    # card
    for i in range(len(CARDS)):
        if CARDS[i][1] <= t < CARDS[i][2] or (i == len(CARDS) - 1 and t >= CARDS[i][2]):
            rgb, al, _ = CARD_LAYERS[i]
            a = al * card_alpha(f, i)
            fr = fr * (1 - a) + rgb * a
            break
    return np.clip(fr, 0, 255)

# ------------------------------------------------------ render-side checks -
def near(fr, col, tol):
    return (np.abs(fr - col).sum(2) < tol)

# card text ink present at each card's midpoint
for i in range(len(CARDS)):
    fmid = int((CARDS[i][1] + CARDS[i][2]) / 2 * FPS)
    fr = render_frame(fmid)
    x0, y0, x1, y1 = CARD_LAYERS[i][2]
    n_txt = int(near(fr[y0:y1 + 1, x0:x1 + 1], C_TXT, 60).sum())
    need = 300 if CARDS[i][3] is not None and len(CARDS[i][0]) <= 2 else 500
    ck(f"card {i} text ink present at f{fmid}", n_txt > need, f"n={n_txt}")

# spiral present before unroll, gone after (fence: rows 210..690)
fr5 = render_frame(int(5.2 * FPS))
n_sp = int((fr5[210:690, :, 2] > BG[2] + 25).sum())
ck("spiral ink present at t=5.2 in upper region", n_sp > 2000, f"n={n_sp}")
fr10 = render_frame(int(10.0 * FPS))
n_res = int((fr10[210:660, :, 2] > BG[2] + 25).sum())
ck("no curve residue above ribbon after unroll (rows 210..660)",
   n_res == 0, f"n={n_res}")
row = fr10[RIB_Y - 2:RIB_Y + 3, RIB_X0:RIB_X1 + 1, 2].max(axis=0)
cov = float((row > BG[2] + 25).mean())
ck("ribbon spans full width at t=10 (>=98% columns inked)", cov >= 0.98,
   f"cov={cov:.3f}")

# amber fence: amber ink ONLY in dot boxes + accent card line (probe t=20.5)
fr20 = render_frame(int(20.5 * FPS))
amber = near(fr20, C_DOT, 90)
for dx in DOT_X:
    xi = int(round(dx))
    amber[RIB_Y - GR - 2:RIB_Y + GR + 3, xi - GR - 2:xi + GR + 3] = False
stray = int(amber.sum())
ck("amber fenced at t=20.5 (dots only; card 6 has no accent)", stray == 0,
   f"stray={stray}")

# dots: born AT onset, coupled to audio
okc, det = True, []
for i, t_on in enumerate(T_ONSET):
    f_pre = int(t_on * FPS) - 2
    f_post = int(t_on * FPS) + 3
    xi = int(round(DOT_X[i]))
    box = (slice(RIB_Y - GR, RIB_Y + GR + 1), slice(xi - GR, xi + GR + 1))
    b_pre = float(near(render_frame(f_pre)[box], C_DOT, 120).sum())
    b_post = float(near(render_frame(f_post)[box], C_DOT, 120).sum())
    s_on = int(round(t_on * SR))
    a_pre = np.abs(audio[s_on - 4800:s_on - 200]).mean()
    a_post = np.abs(audio[s_on + 200:s_on + 4800]).mean()
    coupled = (b_pre == 0 and b_post > 40 and a_post > max(6 * a_pre, 0.02))
    det.append(f"n{i}:vis{b_pre:.0f}->{b_post:.0f} aud{a_pre:.4f}->{a_post:.4f}")
    okc &= coupled
ck("all 4 dots born AT their note's onset (audio-visual coupling)", okc,
   " | ".join(det))

# constellation: all four dots lit on the chord. "Lit" per the CLAIM = the
# box glows brighter than the bare ribbon (the core saturates white over the
# ribbon, so a near-amber count misses it — trap 38 shape, rewritten).
frC = render_frame(int((T_CHORD + 0.4) * FPS))
ctrl_x = 700          # ribbon, no dot/tick/bracket ink within ±GR (verified)
ctrl = float(frC[RIB_Y - GR:RIB_Y + GR + 1,
                 ctrl_x - GR:ctrl_x + GR + 1].sum())
lit, det = 0, []
for dx in DOT_X:
    xi = int(round(dx))
    s = float(frC[RIB_Y - GR:RIB_Y + GR + 1, xi - GR:xi + GR + 1].sum())
    det.append(f"{s - ctrl:.0f}")
    if s - ctrl > 25000:
        lit += 1
ck("chord frame: all 4 dots lit (excess glow > 25k over bare ribbon)",
   lit == 4, f"excess={det}")

# ticks present at computed positions (fence: tick band rows)
frT = render_frame(int(24.0 * FPS))
n_ticks = sum(1 for tx in TICK_X
              if frT[RIB_Y - 24:RIB_Y - 10, tx - 1:tx + 2, 2].max() > BG[2] + 30)
ck("all 10 octave ticks (A0..A9) present at Greenwood positions",
   n_ticks == 10, f"n={n_ticks} at x={TICK_X}")

# watch-size gate: ONE sheet
gate_t = [1.5, 5.2, 8.4, 11.8, 20.5, 24.0, 30.6, 35.0]
tiles = []
for gt in gate_t:
    im = Image.fromarray(render_frame(int(gt * FPS)).astype(np.uint8))
    tiles.append(im.resize((360, 640), Image.LANCZOS))
sheet = Image.new("RGB", (360 * len(tiles), 640))
for j, tl in enumerate(tiles):
    sheet.paste(tl, (j * 360, 0))
sheet.save(SHEET)
print(f"[gate] sheet -> {SHEET}")

if FAILS:
    print("RENDER-SIDE FAILURES:", FAILS); sys.exit(1)

# ------------------------------------------------------------ encode -------
enc = subprocess.Popen(
    ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
     "-c:v", "libx264", "-preset", "medium", "-crf", "18",
     "-pix_fmt", "yuv420p", OUT_RAW],
    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for f in range(FRAMES):
    enc.stdin.write(render_frame(f).astype(np.uint8).tobytes())
    if f % 200 == 0:
        print(f"enc {f}/{FRAMES}", flush=True)
enc.stdin.close(); enc.wait()
assert enc.returncode == 0, "video encode failed"

subprocess.run(
    ["ffmpeg", "-y", "-i", OUT_RAW, "-i", OUT_WAV,
     "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
     "-movflags", "+faststart", OUT_MP4],
    check=True, capture_output=True)

# ----------------------------------------------------------- encode checks -
probe = subprocess.run(
    ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
     "-show_entries", "stream=nb_read_frames,duration", "-of", "csv=p=0",
     OUT_MP4], capture_output=True, text=True).stdout.strip().split(",")
dur_v, nfr = float(probe[0]), int(probe[1])
ck("encode: 1104 frames, 36.8 s", nfr == FRAMES and abs(dur_v - DUR) < 0.15,
   f"frames={nfr} dur={dur_v}")

# decoded: card-3 text survives
i3 = 3
fmid3 = int((CARDS[i3][1] + CARDS[i3][2]) / 2 * FPS)
x0, y0, x1, y1 = CARD_LAYERS[i3][2]
cw, chh = (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2
dec = subprocess.run(
    ["ffmpeg", "-v", "error", "-i", OUT_MP4,
     "-vf", f"select=eq(n\\,{fmid3}),crop={cw}:{chh}:{x0}:{y0}",
     "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
    capture_output=True)
dfr = np.frombuffer(dec.stdout, np.uint8).reshape(chh, cw, 3).astype(np.float32)
ck("decoded: card-3 text survives",
   int(near(dfr, C_TXT, 90).sum()) > 400, f"n={int(near(dfr, C_TXT, 90).sum())}")

# decoded audio: peaks at the four addresses, chord has all four, <1 kHz clean
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", OUT_MP4, "-ac", "1",
                "-ar", str(SR), f"out/cochlea_{STAMP}_dec.wav"],
               check=True, capture_output=True)
with wave.open(f"out/cochlea_{STAMP}_dec.wav") as wf:
    da = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float64) / 32768

okp, det = True, []
for f0, t_on in zip(NOTES, T_ONSET):
    s0 = int((t_on + 0.05) * SR)
    seg = da[s0:s0 + 2 * SR] * np.hanning(2 * SR)
    sp = np.abs(np.fft.rfft(seg))
    fpk = np.fft.rfftfreq(len(seg), 1 / SR)[np.argmax(sp)]
    det.append(f"{f0:.0f}->{fpk:.2f}")
    okp &= abs(fpk - f0) < 1.0
ck("decoded: each note's peak within 1 Hz of its address", okp, " ".join(det))

s0 = int((T_CHORD + 0.05) * SR)
seg = da[s0:s0 + 2 * SR] * np.hanning(2 * SR)
sp = np.abs(np.fft.rfft(seg)); fx = np.fft.rfftfreq(len(seg), 1 / SR)
peaks_ok = all(sp[(fx > f0 - 2) & (fx < f0 + 2)].max() >
               0.05 * sp.max() for f0 in NOTES)
ck("decoded: chord carries all four notes", peaks_ok)

dspec = np.abs(np.fft.rfft(da)) ** 2
dfx = np.fft.rfftfreq(len(da), 1 / SR)
dhf = dspec[dfx > 1000].sum() / dspec.sum()
ck("decoded audio: energy above 1 kHz < 1e-4 (AAC)", dhf < 1e-4,
   f"hf={dhf:.2e} ({10*np.log10(dhf+1e-30):.1f} dB)")

sz = os.path.getsize(OUT_MP4)
ck("file size sane", 300_000 < sz < 30_000_000, f"{sz}")

print()
if FAILS:
    print("FAILURES:", FAILS); sys.exit(1)
print(f"ALL CHECKS PASSED — {OUT_MP4} ({sz} bytes, {nfr} frames, {dur_v:.2f} s)")
print()
print("NOT VERIFIED, and the description says so: the drawn spiral is a")
print("schematic (an Euler spiral, not anatomy); 35 mm is the conventional")
print("Greenwood length (average cochleas measure nearer 31.5 mm — the end")
print("frequencies stay put, the middle compresses); and place is the MAP,")
print("not the whole code — below ~1 kHz the auditory nerve also phase-locks")
print("to the waveform. Tonotopy itself is textbook-verified, not measured")
print("here.")
