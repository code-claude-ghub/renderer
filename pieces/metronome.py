#!/usr/bin/env python3
"""THE METRONOME — a CPR tempo aid that is also a Short.

110 beats at 110 bpm = exactly 60.000 s, so the Short loops seamlessly ON
the beat grid. Writing-led: ten text cards carry the piece; the sound is
functional (a low ~85 Hz thump; deliberately NOTHING above 1 kHz — standing
accessibility promise, verified by FFT on master AND decoded audio).

Facts verified 2026-09-10 against AHA 2025 Guidelines (Circulation, Part 7:
Adult BLS) and cpr.heart.org facts page:
  - rate 100-120/min (unchanged in 2025), depth 5-6 cm, full recoil
  - immediate CPR doubles or triples survival; ~40% receive bystander CPR
  - >70% of out-of-hospital arrests happen at home
  - compression depth decays after 90-120 s; swap ~2 min
"""
import os, subprocess, sys, time, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
FPS = 30
BPM = 110
BEATS = 110
P = 60.0 / BPM                       # beat period, s
DUR = BEATS * P                      # exactly 60.0
FRAMES = int(round(DUR * FPS))       # 1800
SR = 48000

BG      = np.array([10, 12, 16], np.float32)
C_TXT   = np.array([236, 240, 246], np.float32)   # reserved: card text only
C_ACC   = np.array([252, 178, 41], np.float32)    # reserved: tempo figures only
C_PULSE = np.array([198, 52, 44], np.float32)     # reserved: the pulse only

STAMP = time.strftime("%H%M%S")
OUT_RAW = f"out/metronome_{STAMP}_video.mp4"
OUT_WAV = f"out/metronome_{STAMP}.wav"
OUT_MP4 = f"out/metronome_{STAMP}_final.mp4"
SHEET   = f"out/metronome_{STAMP}_sheet.png"

FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_M = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

# ---------------------------------------------------------------- checks ---
FAILS = []
def ck(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if (detail and not ok) else ""))
    if not ok:
        FAILS.append(name)

# ---------------------------------------------------------------- writing ---
# (card lines, start_beat, end_beat, accent_line_index_or_None)
CARDS = [
    (["someone's heart", "just stopped."],                       0,   8, None),
    (["CPR will not restart it.", "that was never its job."],    8,  18, None),
    (["its job is to be the heart —", "from outside. hands, pushing",
      "blood to a brain that cannot wait."],                    18,  30, None),
    (["the rate that works:", "100–120 pushes a minute.",
      "this sound is 110."],                                    30,  40, 2),
    (["push hard — 5 to 6 cm.", "let the chest rise all the way",
      "back up between pushes."],                               40,  50, None),
    (["started at once, CPR doubles", "or triples survival.",
      "only 4 people in 10 get it."],                           50,  62, None),
    (["over 70% of cardiac arrests", "happen at home.",
      "the chest under your hands will", "probably belong to someone you love."],
                                                                62,  76, None),
    (["your pushes weaken after", "90 seconds. everyone's do.",
      "swap if you can. never stop."],                          76,  88, None),
    (["call emergency first —", "put it on speaker.",
      "then push, to this beat."],                              88,  98, None),
    (["this video loops.", "so does the beat.",
      "the rest takes one course,", "one afternoon."],          98, 110, None),
]

# contiguity: spans tile 0..110 exactly
ok = CARDS[0][1] == 0 and CARDS[-1][2] == BEATS and all(
    CARDS[i][2] == CARDS[i + 1][1] for i in range(len(CARDS) - 1))
ck("cards tile the 110 beats contiguously", ok)
ck("loop arithmetic: BEATS*P == DUR, frames integer",
   abs(BEATS * P - 60.0) < 1e-9 and FRAMES == 1800)

# ---------------------------------------------------------------- audio ----
N = int(round(DUR * SR))            # 2,880,000
audio = np.zeros(N, np.float64)
THUMP_LEN = int(0.50 * SR)
t_th = np.arange(THUMP_LEN) / SR
# slight downward chirp 105->82 Hz over 60 ms, then steady — all far below 1 kHz
f_inst = 82.0 + 23.0 * np.exp(-t_th / 0.045)
phase = 2 * np.pi * np.cumsum(f_inst) / SR
att = 0.5 - 0.5 * np.cos(np.pi * np.minimum(t_th / 0.008, 1.0))   # 8 ms raised-cos
env = att * np.exp(-t_th / 0.18)
thump = (np.sin(phase) * env * 0.58)

onsets = [int(round(k * SR * P)) for k in range(BEATS)]
for s0 in onsets:
    seg = min(THUMP_LEN, N - s0)
    audio[s0:s0 + seg] += thump[:seg]
# kill the loop seam: fade the final 25 ms to zero (tail there is ~2% already)
fade_n = int(0.025 * SR)
audio[-fade_n:] *= np.linspace(1, 0, fade_n)

iv = np.diff(onsets)
ck("110 onsets, intervals exact to ±1 sample",
   len(onsets) == BEATS and abs(iv.max() - iv.min()) <= 1,
   f"ivmin={iv.min()} ivmax={iv.max()}")
ck("no clipping, seam silent",
   np.abs(audio).max() <= 0.9 and audio[0] == 0.0 and abs(audio[-1]) < 1e-4,
   f"peak={np.abs(audio).max():.3f} last={audio[-1]:.2e}")

spec = np.abs(np.fft.rfft(audio)) ** 2
freqs = np.fft.rfftfreq(N, 1 / SR)
hf = spec[freqs > 1000].sum() / spec.sum()
ck("master audio: energy above 1 kHz < 1e-5",
   hf < 1e-5, f"hf_frac={hf:.2e} ({10*np.log10(hf+1e-30):.1f} dB)")

wav16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
os.makedirs("out", exist_ok=True)
with wave.open(OUT_WAV, "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
    wf.writeframes(wav16.tobytes())

# ---------------------------------------------------------------- layout ---
PULSE_CX, PULSE_CY, PULSE_R = 540, 440, 130
LABEL_Y = 655
CARD_TOP = 900          # cards live 900..1500; safe area is 192..1632
SAFE_TOP, SAFE_BOT = 192, 1632

def text_layer(lines, accent_idx, size=56, leading=74):
    """Full-frame RGBA float32 layer; returns (rgb, alpha, box)."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    fnt = ImageFont.truetype(FONT_B, size)
    while max(dr.textbbox((0, 0), ln, font=fnt)[2] for ln in lines) > 980:
        size -= 2
        fnt = ImageFont.truetype(FONT_B, size)
    block_h = leading * len(lines)
    y0 = CARD_TOP + (600 - block_h) // 2
    box = [W, H, 0, 0]
    for i, ln in enumerate(lines):
        bb = dr.textbbox((0, 0), ln, font=fnt)
        x = (W - (bb[2] - bb[0])) // 2 - bb[0]
        y = y0 + i * leading
        col = tuple(int(c) for c in (C_ACC if i == accent_idx else C_TXT))
        dr.text((x, y), ln, font=fnt, fill=col + (255,),
                stroke_width=3, stroke_fill=(10, 12, 16, 255))
        bb2 = dr.textbbox((x, y), ln, font=fnt, stroke_width=3)
        box = [min(box[0], bb2[0]), min(box[1], bb2[1]),
               max(box[2], bb2[2]), max(box[3], bb2[3])]
    a = np.asarray(img, np.float32)
    return a[:, :, :3], a[:, :, 3:4] / 255.0, box

CARD_LAYERS = [text_layer(c[0], c[3]) for c in CARDS]

# persistent tempo label under the pulse — the only always-amber ink
lab_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
lab_dr = ImageDraw.Draw(lab_img)
lab_fnt = ImageFont.truetype(FONT_M, 46)
lab_txt = "110 / min"
lb = lab_dr.textbbox((0, 0), lab_txt, font=lab_fnt)
lab_x = (W - (lb[2] - lb[0])) // 2 - lb[0]
lab_dr.text((lab_x, LABEL_Y), lab_txt, font=lab_fnt,
            fill=tuple(int(c) for c in C_ACC) + (255,),
            stroke_width=3, stroke_fill=(10, 12, 16, 255))
LAB_BOX = lab_dr.textbbox((lab_x, LABEL_Y), lab_txt, font=lab_fnt, stroke_width=3)
la = np.asarray(lab_img, np.float32)
LAB_RGB, LAB_A = la[:, :, :3], la[:, :, 3:4] / 255.0

for i, (_, _, box) in enumerate(CARD_LAYERS):
    ck(f"card {i} inside safe area rows {SAFE_TOP}-{SAFE_BOT}",
       box[1] >= SAFE_TOP and box[3] <= SAFE_BOT, f"box={box}")
ck("label inside safe area & clear of cards",
   LAB_BOX[1] >= SAFE_TOP and LAB_BOX[3] < CARD_TOP, f"lab={LAB_BOX}")
# the defect is glow INK under the label, not box overlap (first check here
# asserted the bounding box and failed a fine frame — trap 6/38)
glow_at_label = float(np.exp(-((LABEL_Y - PULSE_CY) / PULSE_R) ** 2 * 1.6)
                      * C_PULSE.max())
ck("pulse glow at label row < 5 grey levels",
   glow_at_label < 5.0, f"amp={glow_at_label:.1f}")

# static base = BG + label (composited once)
base = np.empty((H, W, 3), np.float32)
base[:] = BG
base = base * (1 - LAB_A) + LAB_RGB * LAB_A

# pulse glow field (precomputed radial falloff in a box)
BX0, BX1 = PULSE_CX - 260, PULSE_CX + 260
BY0, BY1 = PULSE_CY - 260, PULSE_CY + 260
yy, xx = np.mgrid[BY0:BY1, BX0:BX1]
r2 = ((xx - PULSE_CX) ** 2 + (yy - PULSE_CY) ** 2).astype(np.float32)
GLOW = np.exp(-r2 / (PULSE_R ** 2) * 1.6)[:, :, None]     # soft disc

def beat_env(t):
    tb = t % P
    return float(np.exp(-tb / 0.16))

def card_at(f):
    b = (f / FPS) / P
    for i, (_, b0, b1, _) in enumerate(CARDS):
        if b0 <= b < b1:
            return i
    return len(CARDS) - 1

FADE_F = 7
def card_alpha(f, i):
    b0, b1 = CARDS[i][1], CARDS[i][2]
    f0, f1 = b0 * P * FPS, b1 * P * FPS
    a_in = min(1.0, (f - f0) / FADE_F)
    a_out = min(1.0, (f1 - f) / FADE_F)
    return max(0.0, min(a_in, a_out))

def render_frame(f):
    t = f / FPS
    fr = base.copy()
    e = beat_env(t)
    fr[BY0:BY1, BX0:BX1] += GLOW * (0.28 + 0.72 * e) * C_PULSE
    i = card_at(f)
    rgb, al, _ = CARD_LAYERS[i]
    a = al * card_alpha(f, i)
    fr = fr * (1 - a) + rgb * a
    return np.clip(fr, 0, 255)

# ------------------------------------------------------- render-side checks
def near(fr, col, tol):
    return (np.abs(fr - col).sum(2) < tol)

for i in range(len(CARDS)):
    fmid = int(((CARDS[i][1] + CARDS[i][2]) / 2) * P * FPS)
    fr = render_frame(fmid)
    x0, y0, x1, y1 = CARD_LAYERS[i][2]
    n_txt = int(near(fr[y0:y1 + 1, x0:x1 + 1], C_TXT, 60).sum())
    need = 800 if CARDS[i][3] is None else 500   # accent card has fewer white px
    ck(f"card {i} text ink present at f{fmid}", n_txt > need, f"n={n_txt}")

# amber fence: amber may exist only in label box + accent line's card box
for probe_card in (3, 5):
    fmid = int(((CARDS[probe_card][1] + CARDS[probe_card][2]) / 2) * P * FPS)
    fr = render_frame(fmid)
    amber = near(fr, C_ACC, 70)
    amber[LAB_BOX[1]:LAB_BOX[3] + 1, LAB_BOX[0]:LAB_BOX[2] + 1] = False
    if CARDS[probe_card][3] is not None:
        bx = CARD_LAYERS[probe_card][2]
        amber[bx[1]:bx[3] + 1, bx[0]:bx[2] + 1] = False
    stray = int(amber.sum())
    ck(f"amber fenced (card {probe_card} frame)", stray == 0, f"stray={stray}")

# pulse-beat coupling: brightness at onset >> mid-decay, for sampled beats
okc, det = True, []
for k in (0, 13, 37, 54, 81, 99, 109):
    f_on = int(round(k * P * FPS)) + 1
    f_off = f_on + 8
    if f_off >= FRAMES: f_off = FRAMES - 1
    b_on = float(render_frame(f_on)[BY0:BY1, BX0:BX1, 0].sum())
    b_off = float(render_frame(f_off)[BY0:BY1, BX0:BX1, 0].sum())
    det.append(f"k{k}:{b_on/b_off:.2f}x")
    if b_on < 1.25 * b_off: okc = False
ck("pulse coupled to beat grid (onset brighter than +8f)", okc, " ".join(det))

# watch-size gate: ONE sheet, three frames at 360 px
gate_frames = [int(((CARDS[i][1] + CARDS[i][2]) / 2) * P * FPS) for i in (2, 3, 6)]
tiles = []
for gf in gate_frames:
    im = Image.fromarray(render_frame(gf).astype(np.uint8))
    tiles.append(im.resize((360, 640), Image.LANCZOS))
sheet = Image.new("RGB", (360 * 3, 640))
for j, tl in enumerate(tiles):
    sheet.paste(tl, (j * 360, 0))
sheet.save(SHEET)
print(f"[gate] sheet -> {SHEET}")

if FAILS:
    print("RENDER-SIDE FAILURES:", FAILS); sys.exit(1)

# ---------------------------------------------------------------- encode ---
enc = subprocess.Popen(
    ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
     "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
     "-c:v", "libx264", "-preset", "medium", "-crf", "18",
     "-pix_fmt", "yuv420p", OUT_RAW],
    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for f in range(FRAMES):
    enc.stdin.write(render_frame(f).astype(np.uint8).tobytes())
    if f % 300 == 0:
        print(f"enc {f}/{FRAMES}", flush=True)
enc.stdin.close(); enc.wait()
assert enc.returncode == 0, "video encode failed"

subprocess.run(
    ["ffmpeg", "-y", "-i", OUT_RAW, "-i", OUT_WAV,
     "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
     "-movflags", "+faststart", OUT_MP4],
    check=True, capture_output=True)

# ---------------------------------------------------------- encode checks --
probe = subprocess.run(
    ["ffprobe", "-v", "error", "-select_streams", "v:0",
     "-count_frames", "-show_entries",
     "stream=nb_read_frames,duration", "-of", "csv=p=0", OUT_MP4],
    capture_output=True, text=True).stdout.strip().split(",")
dur_v, nfr = float(probe[0]), int(probe[1])
ck("encode: 1800 frames, 60.0 s", nfr == FRAMES and abs(dur_v - 60.0) < 0.15,
   f"frames={nfr} dur={dur_v}")

astream = subprocess.run(
    ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
     "stream=sample_rate,duration", "-of", "csv=p=0", OUT_MP4],
    capture_output=True, text=True).stdout.strip()
ck("encode: audio stream present @48k", astream.startswith("48000"), astream)

# decoded text: crop card-6 text region from the decoded file
i6 = 6
fmid6 = int(((CARDS[i6][1] + CARDS[i6][2]) / 2) * P * FPS)
x0, y0, x1, y1 = CARD_LAYERS[i6][2]
cw, chh = (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2
dec = subprocess.run(
    ["ffmpeg", "-v", "error", "-i", OUT_MP4,
     "-vf", f"select=eq(n\\,{fmid6}),crop={cw}:{chh}:{x0}:{y0}",
     "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
    capture_output=True)
dfr = np.frombuffer(dec.stdout, np.uint8)
dfr = dfr.reshape(chh, cw, 3).astype(np.float32)
n_dec = int(near(dfr, C_TXT, 90).sum())
ck("decoded: card-6 text survives", n_dec > 500, f"n={n_dec}")

# decoded audio: onsets + the 1 kHz promise on what viewers actually hear
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", OUT_MP4,
                "-ac", "1", "-ar", str(SR), f"out/metronome_{STAMP}_dec.wav"],
               check=True, capture_output=True)
with wave.open(f"out/metronome_{STAMP}_dec.wav") as wf:
    da = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float64) / 32768
present = 0
for s0 in onsets:
    w1 = np.abs(da[s0 + 200: s0 + 2400]).mean()          # just after onset
    w0 = np.abs(da[max(0, s0 - 2400): max(1, s0 - 200)]).mean()  # just before
    if w1 > max(2.5 * w0, 0.01):
        present += 1
ck("decoded: all 110 beats present at exact grid positions", present == BEATS,
   f"present={present}")
dspec = np.abs(np.fft.rfft(da)) ** 2
dfreqs = np.fft.rfftfreq(len(da), 1 / SR)
dhf = dspec[dfreqs > 1000].sum() / dspec.sum()
ck("decoded audio: energy above 1 kHz < 1e-4 (AAC)", dhf < 1e-4,
   f"hf={dhf:.2e} ({10*np.log10(dhf+1e-30):.1f} dB)")

sz = os.path.getsize(OUT_MP4)
ck("file size sane", 300_000 < sz < 30_000_000, f"{sz}")

print()
if FAILS:
    print("FAILURES:", FAILS); sys.exit(1)
print(f"ALL CHECKS PASSED — {OUT_MP4} ({sz} bytes, {nfr} frames, {dur_v:.2f} s)")
print()
print("NOT VERIFIED, and the piece does not claim it: whether a viewer's")
print("compressions reach 5-6 cm or fully recoil (that needs a training")
print("manikin); whether this tempo aid is used correctly under stress (that")
print("needs a real course); the thump's 85 Hz voicing is design, not")
print("guideline — AHA specifies the RATE, not the sound. The video is a")
print("metronome and a reason to take a course. It is not the course.")
