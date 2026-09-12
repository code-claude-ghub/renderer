#!/usr/bin/env python3
"""THE SINK — your sink does not know which hemisphere it is in.

The Coriolis effect on a draining basin is real and it is tiny: still water
at 42 deg N turns with the Earth about the vertical once every 35.5 hours
(Omega sin phi). A sink filled a minute ago is turning far faster than that
from the tap, so the tap decides. In 1962 Ascher Shapiro (MIT) filled a tank
6 ft across and 6 in deep, swirling it CLOCKWISE on purpose, covered it, and
waited 24 hours. Then he opened a 3/8-inch hole: 20 minutes to drain; the float
over the hole did not move for 12-15 minutes; then it turned counterclockwise,
reaching about one turn every 3-4 s by the end. Sydney, 1965: same method,
clockwise.

The arithmetic (held-out check): angular momentum. Water from the rim (3 ft)
squeezed to the 3/16-inch hole radius is (R/r)^2 = 36,864 x less area, so its
once-per-35.5-hours becomes once per 3.47 s. Shapiro's float: 3 to 4 s.
The estimate ignores friction and the float's own 1-inch span — it is the
right order, not a precision result, and the description says so.

Writing-first, silent, declared. Eight cards carry the piece. The screen is a
diagram: the tank top-down, N at the top, the float over the hole (enlarged),
an amber caption for the moment, and the float turning at the TRUE peak rate
in real time once it turns (the spin-up is compressed to 2 s).

Facts (verified 2026-09-12):
  - Shapiro, A. "Bath-Tub Vortex", Nature 196, 1080-1081 (1962)
  - Trefethen, Bilger, Fink, Luxton, Tanner, "The Bath-Tub Vortex in the
    Southern Hemisphere", Nature 207, 1084-1085 (1965): five runs, >= 18 h
    settling, clockwise
  - tank 6 ft x 6 in, 3/8-in hole, 20-ft hose, filled clockwise, covered,
    24 h, ~20 min drain, float still 12-15 min, then CCW, ~1 rev / 3-4 s
    (MIT Technology Review, "Verifying a Vortex", 2012-10-24)
  - Omega = 7.2921e-5 rad/s; Watertown MA 42.37 N; Sydney 33.87 S
"""
import os, subprocess, sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
FPS = 30
DUR = 20.0
FRAMES = int(round(DUR * FPS))

OMEGA = 7.2921159e-5
LAT_MIT, LAT_SYD = 42.37, -33.87
R_TANK = 3 * 0.3048                            # 0.9144 m (6 ft across)
R_HOLE = (3 / 16) * 0.0254                     # 4.7625 mm (3/8-in hole)
R_SINK, R_SINK_HOLE = 0.20, 0.02               # a declared model sink

def w_earth(lat): return OMEGA * np.sin(np.radians(lat))   # vertical spin of still water, rad/s
W_MIT = w_earth(LAT_MIT); W_SYD = w_earth(LAT_SYD)
AMP = (R_TANK / R_HOLE) ** 2                                 # 36,864
W_PEAK = W_MIT * AMP                                         # 1.812 rad/s
T_PEAK = 2 * np.pi / W_PEAK                                  # 3.47 s
T_EARTH_H = 2 * np.pi / W_MIT / 3600                         # 35.5 h
T_SYD = 2 * np.pi / abs(W_SYD * AMP)                         # 4.19 s (same tank, model)
AMP_SINK = (R_SINK / R_SINK_HOLE) ** 2                       # 100
T_SINK_MIN = 2 * np.pi / (W_MIT * AMP_SINK) / 60             # 21.3 min

FAILS = []
def ck(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if (detail and not ok) else ""))
    if not ok: FAILS.append(name)

print(f"Omega sin phi (42.37 N) = {W_MIT:.4e} rad/s -> {T_EARTH_H:.2f} h;  amp = {AMP:.0f};  "
      f"peak {W_PEAK:.3f} rad/s = one turn / {T_PEAK:.2f} s;  Sydney {T_SYD:.2f} s;  sink {T_SINK_MIN:.1f} min")
ck("still water at 42 N: one turn per 36 h (rounds)", round(T_EARTH_H) == 36, f"{T_EARTH_H:.2f}")
ck("rim-to-hole area ratio rounds to 37,000", round(AMP, -3) == 37000, f"{AMP:.0f}")
ck("peak: one turn per 3.5 s (rounds), inside Shapiro's 3-4 s", round(T_PEAK, 1) == 3.5 and 3 <= T_PEAK <= 4, f"{T_PEAK:.3f}")
ck("the card's own arithmetic 36 h / 37,000 also gives 3.5 s", round(36 * 3600 / 37000, 1) == 3.5)
ck("held-out: T_peak computed from Omega, latitude and geometry only", abs(T_PEAK - 2 * np.pi * R_HOLE ** 2 / (OMEGA * np.sin(np.radians(LAT_MIT)) * R_TANK ** 2)) < 1e-12)
ck("southern hemisphere: the vertical spin has the opposite sign (clockwise)", W_SYD < 0 < W_MIT)
ck("sink: 100 x, one turn per 21 min (rounds)", AMP_SINK == 100 and round(T_SINK_MIN) == 21, f"{T_SINK_MIN:.2f}")
ck("Earth's share at the sink drain is slower than one turn per minute (so any tap swirl wins)", T_SINK_MIN > 1)

# --------------------------------------------------------------- writing ---
T_PLUG, T_TURN, T_FULL = 9.0, 12.0, 14.0        # caption phases; float ramps 12.0 -> 14.0 s
CARDS = [
    (["your sink does not know", "which hemisphere it is in."],                                            0.0,  3.0, 1),
    (["at 42° north, still water turns", "with the Earth once per 36 hours.", "the tap left more swirl than that."], 3.0, 6.0, None),
    (["1962. Ascher Shapiro, MIT.", "a 6-foot tank, filled swirling", "CLOCKWISE on purpose.", "covered. 24 hours."], 6.0, T_PLUG, None),
    (["then a ⅜-inch hole opened.", "20 minutes to drain.", "for 12 of them the float", "over the hole did not move."],  T_PLUG, T_TURN, None),
    (["then it turned. counterclockwise.", "one turn every 3 to 4 seconds", "by the end."],               T_TURN, 14.5, 0),
    (["the arithmetic: 3 ft of rim into a", "⅜-inch hole is 37,000× less area.", "36 hours ÷ 37,000 = 3.5 seconds."], 14.5, 17.0, 2),
    (["Sydney, 1965. same method, five runs.", "clockwise."],                                              17.0, 18.5, 1),
    (["your sink squeezes 100×.", "the Earth's share at its drain:", "one turn per 21 minutes.", "the rest is the tap."], 18.5, DUR, None),
]
ck("cards tile 0..20 s contiguously",
   CARDS[0][1] == 0.0 and CARDS[-1][2] == DUR and all(CARDS[i][2] == CARDS[i+1][1] for i in range(len(CARDS)-1)))
ck("card 4 ('then it turned') begins exactly when the float starts", CARDS[4][1] == T_TURN)
ck("card 3 (plug pulled, float still) spans exactly the still-after-plug phase", CARDS[3][1] == T_PLUG and CARDS[3][2] == T_TURN)

# the float's angular velocity (rad/s, +ve = counterclockwise on screen) and angle
def omega_at(t):
    t = np.asarray(t, dtype=float)
    return np.where(t < T_TURN, 0.0, np.where(t < T_FULL, W_PEAK * (t - T_TURN) / (T_FULL - T_TURN), W_PEAK))
def theta_at(t):
    """integrated analytically: 0 | quadratic ramp | linear."""
    t = float(t)
    if t <= T_TURN: return 0.0
    ramp = W_PEAK * min(t - T_TURN, T_FULL - T_TURN) ** 2 / (2 * (T_FULL - T_TURN))
    lin = W_PEAK * max(0.0, t - T_FULL)
    return ramp + lin
def theta_syd(t):
    """the Sydney mini-float: same model peak rate, clockwise; visible from 17.0 s."""
    return -abs(W_SYD * AMP) * max(0.0, float(t) - CARDS[6][1])

ck("float still before 12.0 s; at the peak rate from 14.0 s", omega_at(11.99) == 0 and abs(float(omega_at(14.0)) - W_PEAK) < 1e-12)
turns_end = theta_at(DUR) / (2 * np.pi)
ck("the float completes >= 1.9 turns on screen (real-time peak rate for 6 s + the ramp)", turns_end >= 1.9, f"{turns_end:.2f}")
print(f"float turns on screen: {turns_end:.2f}; Sydney float turns: {abs(theta_syd(DUR)) / (2*np.pi):.2f}")
# WCAG 2.3.1: nothing flashes — the float rotates continuously (max rate 0.29 rev/s), captions change 4 times in 20 s
ck("WCAG 2.3.1: max float rate 0.29 rev/s; no element flashes", W_PEAK / (2 * np.pi) < 0.5)

# ------------------------------------------------------------------ layout --
BG    = np.array([10, 12, 16], np.float32)
C_TXT = np.array([236, 240, 246], np.float32)
C_ACC = np.array([252, 178, 41], np.float32)
C_WAT = np.array([28, 62, 110], np.float32)        # the water (top view)
C_DIM = np.array([120, 126, 138], np.float32)
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_M = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
SAFE_TOP, SAFE_BOT = 192, 1632
CX, CY, R_PX = 540, 585, 290                       # the tank, top view (6 ft = 580 px)
FLOAT_L, FLOAT_W = 84, 8                           # the float, enlarged (real: 1-inch slivers = 8 px)
SYD_CX, SYD_CY, SYD_L = 960, 585, 56
CAP_Y = 950                                        # amber caption row (below the tank + its label)
CARD_TOP, CARD_H = 1090, 520
STAMP = time.strftime("%H%M%S")
OUT_MP4 = f"out/sink_{STAMP}_final.mp4"
SHEET   = f"out/sink_{STAMP}_sheet.png"

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
    return a[:, :, :3], a[:, :, 3:4] / 255.0, box, size

CARD_LAYERS = [text_layer(c[0], c[3]) for c in CARDS]
for i, (_, _, box, sz) in enumerate(CARD_LAYERS):
    ck(f"card {i} inside safe area, size >= 44", box[1] >= SAFE_TOP and box[3] <= SAFE_BOT and sz >= 44, f"{box} size {sz}")

def caption_text(t):
    if t < 6.0:      return "still water, 42° N: one turn / 36 h"
    if t < T_PLUG:   return "filled clockwise ↻ · covered · 24 h"
    if t < T_TURN:   return "hole opened · minute 12 · float still"
    if t < T_FULL:   return "minute 15 · float begins to turn ↺"
    return "minute 19 · one turn / 3.5 s ↺"
_CAP = {}
def caption_layer(txt):
    if txt in _CAP: return _CAP[txt]
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dr = ImageDraw.Draw(img)
    fnt = ImageFont.truetype(FONT_M, 44)
    bb = dr.textbbox((0, 0), txt, font=fnt)
    x = (W - (bb[2] - bb[0])) // 2 - bb[0]
    dr.text((x, CAP_Y), txt, font=fnt, fill=tuple(int(c) for c in C_ACC) + (255,), stroke_width=3, stroke_fill=(10, 12, 16, 255))
    a = np.asarray(img, np.float32)
    _CAP[txt] = (a[:, :, :3], a[:, :, 3:4] / 255.0, dr.textbbox((x, CAP_Y), txt, font=fnt, stroke_width=3))
    return _CAP[txt]
CAP_FENCE = [40, CAP_Y - 12, 1040, CAP_Y + 70]
for f_ in (0, int(7 * FPS), int(10 * FPS), int(13 * FPS), int(19 * FPS)):
    _, _, bb = caption_layer(caption_text(f_ / FPS))
    ck(f"caption at f{f_} inside its fence and the safe area", bb[0] >= CAP_FENCE[0] and bb[2] <= CAP_FENCE[2]
       and bb[1] >= CAP_FENCE[1] and bb[3] <= CAP_FENCE[3] and bb[1] >= SAFE_TOP, f"{bb}")

# static base: water disc, rim, N mark, hole, labels
img = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dr = ImageDraw.Draw(img)
white = tuple(int(c) for c in C_TXT) + (255,); dim = tuple(int(c) for c in C_DIM) + (255,)
wat = tuple(int(c) for c in C_WAT) + (255,)
dr.ellipse([CX - R_PX, CY - R_PX, CX + R_PX, CY + R_PX], fill=wat, outline=white, width=5)
dr.ellipse([CX - 7, CY - 7, CX + 7, CY + 7], fill=dim)                           # the hole (enlarged)
fnt_s = ImageFont.truetype(FONT_M, 30); fnt_n = ImageFont.truetype(FONT_B, 40)
dr.line([(CX, CY - R_PX - 34), (CX, CY - R_PX - 8)], fill=dim, width=3)
bb = dr.textbbox((0, 0), "N", font=fnt_n); dr.text((CX - (bb[2] - bb[0]) // 2 - bb[0], CY - R_PX - 84), "N", font=fnt_n, fill=dim)
ck("the N mark sits below the safe-area line", CY - R_PX - 84 >= SAFE_TOP, f"{CY - R_PX - 84}")
lab = "6 ft tank, top view · hole and floats enlarged"
bb = dr.textbbox((0, 0), lab, font=fnt_s); dr.text((CX - (bb[2] - bb[0]) // 2 - bb[0], CY + R_PX + 22), lab, font=fnt_s, fill=dim)
sl = "Sydney"
bb = dr.textbbox((0, 0), sl, font=fnt_s)
SYD_LABEL = (SYD_CX - (bb[2] - bb[0]) // 2 - bb[0], SYD_CY + SYD_L + 16)
ca = np.asarray(img, np.float32)
base = np.empty((H, W, 3), np.float32); base[:] = BG
base = base * (1 - ca[:, :, 3:4] / 255) + ca[:, :, :3] * (ca[:, :, 3:4] / 255)

def cross_layer(cx, cy, L, th, width, colour, label=None):
    """two crossed slivers at angle th (radians, +ve = counterclockwise ON SCREEN: y grows downward)."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dr = ImageDraw.Draw(img)
    col = tuple(int(c) for c in colour) + (255,)
    for a in (th, th + np.pi / 2):
        dx, dy = L * np.cos(a), -L * np.sin(a)
        dr.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=col, width=width)
    if label: dr.text(SYD_LABEL, label, font=fnt_s, fill=dim)
    a = np.asarray(img, np.float32)
    return a[:, :, :3], a[:, :, 3:4] / 255.0

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
    rgb, al = cross_layer(CX, CY, FLOAT_L, theta_at(t), FLOAT_W, C_TXT)
    fr = fr * (1 - al) + rgb * al
    if t >= CARDS[6][1]:
        rgb, al = cross_layer(SYD_CX, SYD_CY, SYD_L, theta_syd(t), 6, C_TXT, label="Sydney")
        fr = fr * (1 - al) + rgb * al
    rgb, al, _ = caption_layer(caption_text(t))
    fr = fr * (1 - al) + rgb * al
    i = card_at(f); rgb, al, _, _ = CARD_LAYERS[i]; a = al * card_alpha(f, i)
    fr = fr * (1 - a) + rgb * a
    return np.clip(fr, 0, 255)

# ------------------------------------------------------- render-side checks
def near(fr, col, tol): return np.abs(fr - col).sum(2) < tol

for i in range(len(CARDS)):
    fmid = int((CARDS[i][1] + CARDS[i][2]) / 2 * FPS)
    fr = render_frame(fmid); x0, y0, x1, y1 = CARD_LAYERS[i][2]
    n = int(near(fr[y0:y1 + 1, x0:x1 + 1], C_TXT, 60).sum())
    ck(f"card {i} text ink at f{fmid}", n > 500, f"n={n}")

def measured_angle(fr, cx, cy, L):
    """orientation of a cross from PIXELS: white pixels in an annulus near the arm tips, folded mod 90 deg.
    Returns degrees in [0, 90), y-up convention (counterclockwise on screen = increasing)."""
    y, x = np.mgrid[cy - L - 6:cy + L + 7, cx - L - 6:cx + L + 7]
    box = fr[cy - L - 6:cy + L + 7, cx - L - 6:cx + L + 7]
    m = near(box, C_TXT, 60)
    rr = np.hypot(x - cx, y - cy)
    m &= (rr > L * 0.7) & (rr < L + 4)
    ang = np.degrees(np.arctan2(-(y[m] - cy), x[m] - cx)) % 90.0
    # circular mean on the 90-degree period
    z = np.exp(1j * np.radians(ang) * 4)
    return float(np.degrees(np.angle(z.mean()) / 4) % 90.0), int(m.sum())

def dang(a, b):
    """signed difference a - b on a 90-degree circle, in (-45, 45]."""
    d = (a - b + 45.0) % 90.0 - 45.0
    return d

okp, det = True, []
for t in (1.0, 8.0, 11.0, 13.0, 15.0, 16.5, 18.0, 19.5, 19.97):
    f = int(round(t * FPS)); fr = render_frame(f)
    got, n = measured_angle(fr, CX, CY, FLOAT_L); want = np.degrees(theta_at(f / FPS)) % 90.0
    det.append(f"t{t}:{got:.1f}/{want:.1f}(n{n})")
    okp &= n > 100 and abs(dang(got, want)) <= 2.0
ck("float angle measured from PIXELS matches theta(t) mod 90 within 2 deg (9 frames)", okp, " ".join(det))
ck("float bitwise still on pixels from 0 to 12.0 s (three frames identical in the float box)",
   all(np.array_equal(render_frame(0)[CY-100:CY+100, CX-100:CX+100], render_frame(f)[CY-100:CY+100, CX-100:CX+100])
       for f in (int(5 * FPS), int(T_TURN * FPS) - 1)))
# direction from PIXELS: successive frames at the peak rate advance counterclockwise by the frame step
step = np.degrees(W_PEAK / FPS)                                   # 3.46 deg / frame
okd, det = True, []
for f in (int(15 * FPS), int(17 * FPS), int(19 * FPS)):
    a0, _ = measured_angle(render_frame(f), CX, CY, FLOAT_L); a1, _ = measured_angle(render_frame(f + 5), CX, CY, FLOAT_L)
    d = dang(a1, a0); det.append(f"f{f}:{d:+.2f}")
    okd &= abs(d - 5 * step) < 1.5 and d > 0
ck(f"float turns COUNTERCLOCKWISE on screen at {step:.2f} deg/frame, measured from pixels over 5-frame steps (3 pairs)", okd, " ".join(det))
oks, det = True, []
step_s = np.degrees(abs(W_SYD * AMP) / FPS)
for f in (int(17.5 * FPS), int(19 * FPS)):
    a0, n0 = measured_angle(render_frame(f), SYD_CX, SYD_CY, SYD_L); a1, _ = measured_angle(render_frame(f + 5), SYD_CX, SYD_CY, SYD_L)
    d = dang(a1, a0); det.append(f"f{f}:{d:+.2f}(n{n0})")
    oks &= n0 > 40 and abs(d + 5 * step_s) < 1.5 and d < 0
ck(f"Sydney float turns CLOCKWISE on screen at {step_s:.2f} deg/frame, from pixels over 5-frame steps (2 pairs)", oks, " ".join(det))
ck("no Sydney float before 17.0 s (its box has no white ink at 16.9 s)",
   int(near(render_frame(int(16.9 * FPS))[SYD_CY-70:SYD_CY+70, SYD_CX-70:SYD_CX+70], C_TXT, 60).sum()) == 0)
ck("Sydney float clear of the tank rim and inside the frame", SYD_CX - SYD_L - 8 > CX + R_PX + 5 and SYD_CX + SYD_L + 8 < W)
ck("float strokes >= 4 px wide (decode survival)", FLOAT_W >= 4 and 6 >= 4)
for f in (int(2 * FPS), int(10 * FPS), int(19 * FPS)):
    fr = render_frame(f); amber = near(fr, C_ACC, 70)
    amber[CAP_FENCE[1]:CAP_FENCE[3], CAP_FENCE[0]:CAP_FENCE[2]] = False
    amber[CARD_TOP:CARD_TOP + CARD_H] = False
    ck(f"amber fenced to caption + card (f{f})", int(amber.sum()) == 0, f"stray={int(amber.sum())}")
fr = render_frame(int(10 * FPS)); blue = near(fr, C_WAT, 40)
yy, xx = np.mgrid[0:H, 0:W]; blue &= np.hypot(xx - CX, yy - CY) > R_PX + 3
ck("water colour fenced to the tank disc", int(blue.sum()) == 0, f"stray={int(blue.sum())}")
fr = render_frame(int(10 * FPS)); inside = near(fr, C_WAT, 40) & (np.hypot(xx - CX, yy - CY) < R_PX - 6)
ck("the tank disc IS water (>= 95 % of the interior is water colour)", inside.sum() > 0.95 * np.pi * (R_PX - 6) ** 2 - 4000, f"{inside.sum()}")

tiles = [Image.fromarray(render_frame(int(t * FPS)).astype(np.uint8)).resize((360, 640), Image.LANCZOS)
         for t in (1.5, 10.5, 18.0)]
sheet = Image.new("RGB", (1080, 640))
for j, tl in enumerate(tiles): sheet.paste(tl, (j * 360, 0))
os.makedirs("out", exist_ok=True); sheet.save(SHEET); print(f"[gate] sheet -> {SHEET}")

if FAILS: print("RENDER-SIDE FAILURES:", FAILS); sys.exit(1)
if "--check-only" in sys.argv: sys.exit(0)

# ------------------------------------------------------------------ encode --
enc = subprocess.Popen(
    ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
     "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT_MP4],
    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for f in range(FRAMES):
    enc.stdin.write(render_frame(f).astype(np.uint8).tobytes())
    if f % 90 == 0: print(f"enc {f}/{FRAMES}", flush=True)
enc.stdin.close(); enc.wait(); assert enc.returncode == 0

# ----------------------------------------------------------- encode checks --
probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                        "stream=nb_read_frames,duration", "-of", "csv=p=0", OUT_MP4],
                       capture_output=True, text=True).stdout.strip().split(",")
dur_v, nfr = float(probe[0]), int(probe[1])
ck("encode: 600 frames, 20.0 s", nfr == FRAMES and abs(dur_v - DUR) < 0.15, f"{nfr} {dur_v}")
astreams = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", OUT_MP4],
                          capture_output=True, text=True).stdout.strip()
ck("silent piece: no audio stream (declared in the description)", astreams == "")

def decoded(fn, x0, y0, w, h):
    w, h = w // 2 * 2, h // 2 * 2
    dec = subprocess.run(["ffmpeg", "-v", "error", "-i", OUT_MP4, "-vf", f"select=eq(n\\,{fn}),crop={w}:{h}:{x0}:{y0}",
                          "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    return np.frombuffer(dec.stdout, np.uint8).reshape(h, w, 3).astype(np.float32)
# decoded float orientation at two late frames
okdec, det = True, []
for f in (int(16 * FPS), FRAMES - 1):
    L = FLOAT_L; box = decoded(f, CX - L - 6, CY - L - 6, 2 * L + 13, 2 * L + 13)
    full = np.zeros((H, W, 3), np.float32); full[CY - L - 6:CY - L - 6 + box.shape[0], CX - L - 6:CX - L - 6 + box.shape[1]] = box
    got, n = measured_angle(full, CX, CY, L); want = np.degrees(theta_at(f / FPS)) % 90.0
    det.append(f"f{f}:{got:.1f}/{want:.1f}(n{n})"); okdec &= n > 80 and abs(dang(got, want)) <= 3.0
ck("decoded: float orientation survives h264 within 3 deg (2 frames)", okdec, " ".join(det))
x0, y0, x1, y1 = CARD_LAYERS[5][2]
fm_ = int((CARDS[5][1] + CARDS[5][2]) / 2 * FPS)
dfr = decoded(fm_, x0, y0, x1 - x0, y1 - y0)
ck("decoded: card-5 (the arithmetic) text survives", int(near(dfr, C_TXT, 90).sum()) > 400)
sz = os.path.getsize(OUT_MP4)
ck("file size sane", 100_000 < sz < 30_000_000, f"{sz}")

print()
if FAILS: print("FAILURES:", FAILS); sys.exit(1)
print(f"ALL CHECKS PASSED — {OUT_MP4} ({sz} bytes, {nfr} frames, {dur_v:.2f} s)")
print(f"T_earth={T_EARTH_H:.2f} h  amp={AMP:.0f}  T_peak={T_PEAK:.3f} s  T_syd={T_SYD:.2f} s  sink={T_SINK_MIN:.1f} min")
print()
print("NOT VERIFIED, and the piece does not claim it: the 3.5 s is an inviscid angular-")
print("momentum estimate at the HOLE radius; Shapiro's float was a 1-inch cross, and the")
print("tank floor's friction is ignored (it is why nothing moved for 12 minutes). The")
print("agreement with 3-4 s is the right order, not a precision result. The spin-up on")
print("screen is compressed to 2 s; the peak rate is real time. The Sydney float turns")
print("at the SAME-TANK model rate (4.2 s), not a rate from the 1965 paper. The sink")
print("numbers use a declared model sink (20 cm radius, 2 cm drain).")
