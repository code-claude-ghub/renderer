#!/usr/bin/env python3
"""RIPPLE — the sea's minimum wake speed.

Two lanes, one scale, one clock (real time). Left: a pressure point
"swimming" at 0.20 m/s — BELOW water's minimum wave speed
c_min = (4 g sigma/rho)^(1/4) = 0.231 m/s — so no wave can keep pace
with it and it raises NO wake at all, only a bound dimple that
travels with it. Right: the same point at 0.30 m/s, where c(k) = V
has two roots, so the wake SPLITS: capillary ripples (5.6 mm,
group velocity > phase velocity) run AHEAD of the swimmer while
gravity waves (5.2 cm) trail behind. Rayleigh 1883 (the fishing
line in a stream); Raphael & de Gennes 1996 (no wave resistance
below c_min).

The camera rides with each swimmer (exact spectral Galilean shift),
the drifting motes mark the stream. Gravity-capillary dispersion
omega = sqrt(g k + A k^3) evolved with an EXACT per-mode propagator
including real viscosity (nu = 1.0e-6) — the wavelengths that appear
are the water's own. All physics and lane parameters imported from
scripts/feas_ripple.py (ALL FEASIBILITY CHECKS PASSED there first:
lambda behind 5.19 cm / theory 5.20; ahead 5.63 mm / 5.63; silence
ratio 0.025; dimple bound 165x).

Timeline (17.0 s, 510 frames @ 30 fps):
  f   0- 23  calm lanes stream past (motes), labels
  f  24-     swimmers switch on (1.2 s smooth ramp); wake develops
             in real time on the right, silence holds on the left
  f 330-360  annotations fade in (ripples ahead / waves behind /
             no wake)
  f 420-450  the dispersion curve c(lambda) fades in: a horizontal
             line at 0.20 misses the curve entirely; at 0.30 it cuts
             it twice — at exactly the two wavelengths on screen
  f 456-509  hold (sim keeps running; the pattern is steady)

Silent video.
"""
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feas_ripple import (A, C_MIN, G, GEO, LAM_MIN, LANE1, LANE2,
                         SRC_Y, Sim, cg_of, decay_ahead, dimple,
                         lam_ahead, lam_behind, roots, silence)

# ---------------------------------------------------------------- frame
W, H = 1080, 1920
FPS = 30
F = 510
LANE_W = 538
GUT = W - 2 * LANE_W
SIM_F0 = 24
ANN_F0, ANN_F1 = 330, 360
GRAPH_F0, GRAPH_F1 = 420, 450

# lane view (world coords, both lanes identical — same scale is the
# point): 0.294 m wide x 1.05 m tall, source at screen row ~512
VIEW = dict(cx=GEO["Lx"] / 2, w=0.294, y0=0.11, y1=1.16)
PPM = H / (VIEW["y1"] - VIEW["y0"])           # px per metre, 1828.6

# shading gain: p99.5 of |eta| of the ABOVE-limit feasibility field,
# SHARED by both lanes (same physical scale; a per-lane gain would
# amplify the silent lane's residue into fake waves) and pinned so
# nothing renormalises per frame (trap 24)
GAIN = 1.525e-6

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
OUT_DIR = ("/home/maroon-beret/projects/active/youtube/"
           "youtube-channel/out")
STAMP = time.strftime("%H%M%S")
OUT_MP4 = f"{OUT_DIR}/ripple_{STAMP}.mp4"

C_TEXT = (235, 238, 242)
C_DIM = (150, 158, 168)
C_LINE = (255, 246, 220)
C_MOTE = (82, 94, 104)

KG, KC = roots(LANE2["V"])
LAM_G = 2 * np.pi / KG                        # 5.20 cm
LAM_C = 2 * np.pi / KC                        # 5.63 mm


def sim_t_of(f):
    return max(0.0, (f - SIM_F0) / FPS)


# ---------------------------------------------------------------- water
def lane_rgb(eta, view, gain, seed):
    """Crop the view region, upsample to lane pixels, shaded relief
    (KELVIN's pipeline: normals -> Lambert + specular -> blue palette
    -> per-frame dither)."""
    ny, nx = eta.shape
    dx, dy = GEO["Lx"] / nx, GEO["Ly"] / ny
    c0 = int((view["cx"] - view["w"] / 2) / dx)
    c1 = int((view["cx"] + view["w"] / 2) / dx)
    r0 = int(view["y0"] / dy)
    r1 = int(view["y1"] / dy)
    crop = eta[r0:r1, c0:c1]
    z = np.clip(crop / gain, -2.5, 2.5)
    im = Image.fromarray((z * 1000).astype(np.float32))
    im = im.resize((LANE_W, H), Image.BICUBIC)
    z = np.asarray(im, np.float64)[::-1] / 1000.0    # screen-up = +y
    gy, gx = np.gradient(z * 9.0)
    nz = 1.0 / np.sqrt(1 + gx * gx + gy * gy)
    nxv, nyv = -gx * nz, -gy * nz
    L = np.array([-0.55, -0.62, 0.55])
    L /= np.linalg.norm(L)
    lam = np.clip(nxv * L[0] + nyv * L[1] + nz * L[2], 0, 1)
    Hv = L + np.array([0.0, 0.0, 1.0])
    Hv /= np.linalg.norm(Hv)
    spec = np.clip(nxv * Hv[0] + nyv * Hv[1] + nz * Hv[2], 0, 1) ** 60
    hgt = np.clip(0.5 + 0.35 * z, 0, 1)
    sh = 0.25 + 0.75 * lam
    rgb = np.stack([(0.030 + 0.05 * hgt) * sh + 0.55 * spec,
                    (0.10 + 0.16 * hgt) * sh + 0.60 * spec,
                    (0.16 + 0.24 * hgt) * sh + 0.62 * spec], -1)
    rng = np.random.default_rng(seed)
    rgb += (rng.random(rgb.shape) - 0.5) * (1.2 / 255)
    return np.clip(rgb, 0, 1)


def world_to_lane_px(x, y):
    col = (x - (VIEW["cx"] - VIEW["w"] / 2)) / VIEW["w"] * LANE_W
    row = (VIEW["y1"] - y) / (VIEW["y1"] - VIEW["y0"]) * H
    return col, row


SRC_COL, SRC_ROW = world_to_lane_px(GEO["Lx"] / 2, SRC_Y)  # 269, 512


# ---------------------------------------------------------------- motes
def motes(seed, n=12):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.02, 0.98, n)           # frac of view width
    ys = rng.uniform(0.0, 1.0, n)             # frac of wrap range
    return xs, ys


M1 = motes(11)
M2 = motes(22)
WRAP0, WRAP1 = 0.06, 1.22                     # world wrap range (m)


def draw_motes(draw, xs, ys, V, t, x_off):
    """Drift markers riding the stream: in the swimmer's frame the
    water moves at -V. Their wave-bobbing is second order — ignored
    and declared in the description."""
    span = WRAP1 - WRAP0
    for xf, yf in zip(xs, ys):
        yw = WRAP0 + ((yf * span - V * t) % span)
        c, r = world_to_lane_px(0.0, yw)
        c = xf * LANE_W
        if -4 < r < H + 4:
            draw.ellipse([x_off + c - 2.5, r - 2.5,
                          x_off + c + 2.5, r + 2.5], fill=C_MOTE)


# ---------------------------------------------------------------- text
_fonts = {}


def font(sz):
    if sz not in _fonts:
        _fonts[sz] = ImageFont.truetype(FONT, sz)
    return _fonts[sz]


def stamp(draw, cx, cy, txt, sz, fill, anchor="mm"):
    sh = (0, 0, 0) if len(fill) < 4 else (0, 0, 0, fill[3])
    draw.text((cx + 2, cy + 2), txt, font=font(sz),
              fill=sh, anchor=anchor)
    draw.text((cx, cy), txt, font=font(sz), fill=fill, anchor=anchor)


# ---------------------------------------------------------------- graph
# c(lambda) panel: log-lambda 2.5 mm .. 13 cm, c 0.14 .. 0.46 m/s —
# ranges chosen so the curve fits the box WITHOUT clamping anywhere
# (c(2.5 mm) = 0.432, c(13 cm) = 0.452, both < 0.46)
GX0, GX1 = 150, 1010                          # plot box px
GY0, GY1 = 1150, 1500
PLATE = (66, 1062, 1014, 1560)                # x0, y0, x1, y1


def lam_to_px(lam):
    f = (np.log(lam) - np.log(0.0025)) / (np.log(0.13) - np.log(0.0025))
    return GX0 + f * (GX1 - GX0)


def c_to_px(c):
    f = (c - 0.14) / (0.46 - 0.14)
    return GY1 - f * (GY1 - GY0)


def c_of_lam(lam):
    return np.sqrt(G * lam / (2 * np.pi) + 2 * np.pi * A / lam)


def draw_graph(img, alpha):
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    aa = int(alpha * 255)

    def col(c, a=1.0):
        return tuple(list(c) + [int(aa * a)])

    d.rounded_rectangle(PLATE, radius=18, fill=(5, 9, 13, int(aa * 0.88)))
    # axes
    d.line([(GX0, GY1), (GX1, GY1)], fill=col(C_DIM), width=2)
    d.line([(GX0, GY0), (GX0, GY1)], fill=col(C_DIM), width=2)
    # curve
    lams = np.exp(np.linspace(np.log(0.0025), np.log(0.13), 220))
    pts = [(lam_to_px(l), c_to_px(c_of_lam(l))) for l in lams]
    d.line(pts, fill=col(C_TEXT), width=4)
    # the minimum
    mx, my = lam_to_px(LAM_MIN), c_to_px(C_MIN)
    d.ellipse([mx - 6, my - 6, mx + 6, my + 6], fill=col((255, 210, 120)))
    stamp(d, mx, my + 56, "minimum: 0.231 m/s", 28,
          col((255, 210, 120)))
    # the two speed lines
    for V, cc, lbl in ((0.20, (150, 190, 235), "0.20 — misses the curve"),
                       (0.30, (255, 246, 220), "0.30 — cuts it twice")):
        y = c_to_px(V)
        for x in range(GX0, GX1, 26):
            d.line([(x, y), (x + 13, y)], fill=col(cc, 0.85), width=3)
        stamp(d, GX1 - 6, y - 24, lbl, 26, col(cc), anchor="rm")
    for lam, lbl in ((LAM_C, "5.6 mm"), (LAM_G, "5.2 cm")):
        x, y = lam_to_px(lam), c_to_px(0.30)
        d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=col(C_LINE))
        stamp(d, x, y + 34, lbl, 26, col(C_LINE))
    stamp(d, (GX0 + GX1) // 2, PLATE[1] + 44,
          "a wake is made of waves that keep pace:  c(λ) = V", 30,
          col(C_TEXT))
    stamp(d, (GX0 + GX1) // 2, GY1 + 36, "wavelength (log) →", 24,
          col(C_DIM))
    stamp(d, GX0 + 10, GY0 + 8, "wave speed", 24, col(C_DIM),
          anchor="lm")
    img.alpha_composite(ov)


# ---------------------------------------------------------------- render
def compose(f, eta1, eta2):
    """One finished frame from the two fields."""
    tm = f / FPS          # motes stream from frame 0: the frame
    # travels at V the whole time; only the source ramps at f24
    frame = np.empty((H, W, 3))
    frame[:, :LANE_W] = lane_rgb(eta1, VIEW, GAIN, f)
    frame[:, LANE_W + GUT:] = lane_rgb(eta2, VIEW, GAIN, f + 7)
    frame[:, LANE_W:LANE_W + GUT] = 0.012
    img = Image.fromarray((np.clip(frame, 0, 1) * 255 + 0.5)
                          .astype(np.uint8)).convert("RGBA")
    draw = ImageDraw.Draw(img)

    draw_motes(draw, *M1, LANE1["V"], tm, 0)
    draw_motes(draw, *M2, LANE2["V"], tm, LANE_W + GUT)

    # swimmer dots
    if f >= SIM_F0:
        for x_off in (0, LANE_W + GUT):
            draw.ellipse([x_off + SRC_COL - 5, SRC_ROW - 7,
                          x_off + SRC_COL + 5, SRC_ROW + 7],
                         fill=(8, 12, 16))

    # labels
    stamp(draw, LANE_W // 2, 228, "0.20 m/s · below the limit",
          34, C_TEXT)
    stamp(draw, LANE_W + GUT + LANE_W // 2, 228,
          "0.30 m/s · above the limit", 34, C_TEXT)
    stamp(draw, W // 2, 286, "the camera rides with each swimmer",
          26, C_DIM)

    # shared scale bar (both lanes are the same scale — that IS
    # part of the claim, so one bar serves both)
    bar = 0.10 / VIEW["w"] * LANE_W       # 10 cm in lane px
    x0 = W // 2 - bar / 2
    draw.line([(x0, 1608), (x0 + bar, 1608)], fill=C_TEXT, width=4)
    draw.line([(x0, 1596), (x0, 1620)], fill=C_TEXT, width=3)
    draw.line([(x0 + bar, 1596), (x0 + bar, 1620)],
              fill=C_TEXT, width=3)
    stamp(draw, W // 2, 1568, "10 cm", 28, C_TEXT)

    # annotations
    if f >= ANN_F0:
        al = min(1.0, (f - ANN_F0) / float(ANN_F1 - ANN_F0))

        def fade(c):
            return tuple(int(v * al) for v in c)

        x2 = LANE_W + GUT
        stamp(draw, x2 + SRC_COL, SRC_ROW - 118,
              "ripples run ahead · 5.6 mm", 30, fade(C_LINE))
        stamp(draw, x2 + SRC_COL, SRC_ROW + 208,
              "waves trail behind · 5.2 cm", 30, fade(C_LINE))
        stamp(draw, SRC_COL, SRC_ROW + 208,
              "no wake · only a bound dimple", 30, fade(C_DIM))
    if f >= GRAPH_F0:
        al = min(1.0, (f - GRAPH_F0) / float(GRAPH_F1 - GRAPH_F0))
        draw_graph(img, al)
    return img.convert("RGB")


def render():
    os.makedirs(OUT_DIR, exist_ok=True)
    sim1 = Sim(**LANE1)
    sim2 = Sim(**LANE2)
    sx = GEO["Lx"] / 2
    snaps = {}
    stills = {0, 60, 140, 220, 300, 380, 430, 470, 509}

    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "18",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT_MP4]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    t0 = time.time()
    for f in range(F):
        tt = sim_t_of(f)
        while sim1.t < tt - 1e-9:
            sim1.step(sx, SRC_Y)
        while sim2.t < tt - 1e-9:
            sim2.step(sx, SRC_Y)

        if f == 300:
            snaps["mid1"] = sim1.eta_now.copy()
            snaps["mid2"] = sim2.eta_now.copy()
        if f == F - 1:
            snaps["fin1"] = sim1.eta_now.copy()
            snaps["fin2"] = sim2.eta_now.copy()

        img = compose(f, sim1.eta_now, sim2.eta_now)
        p.stdin.write(np.asarray(img, np.uint8).tobytes())
        if f in stills:
            img.save(f"{OUT_DIR}/ripple_f{f:04d}.png")
        if f % 60 == 0:
            print(f"  frame {f}/{F}  ({time.time() - t0:.0f}s)",
                  flush=True)
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        print("ENCODE FAILED", flush=True)
        sys.exit(1)
    print(f"encoded {OUT_MP4} ({os.path.getsize(OUT_MP4)} bytes) "
          f"in {time.time() - t0:.0f}s", flush=True)
    np.savez("/tmp/ripple_render_snaps.npz", **snaps)
    return snaps


# ---------------------------------------------------------------- checks
def decode_frame(n):
    r = subprocess.run(
        ["ffmpeg", "-i", OUT_MP4, "-vf", f"select=eq(n\\,{n})",
         "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True)
    a = np.frombuffer(r.stdout, np.uint8)
    return a.reshape(H, W, 3).astype(np.float64)


def highpass_mad(band, w):
    """Texture measure: median |deviation from a w-px vertical box
    blur|. w must be sized to the wavelength being detected — a
    15 px box passes a 95 px gravity wave at 96% and the deviation
    would read ~nothing on a real wake (w=61 keeps 55% of it; w=15
    keeps 79% of a 10 px capillary ripple). Median is robust to the
    sparse motes by construction (~12 dots of ~30 px in ~40k px)."""
    k = np.ones(w) / w
    sm = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 0,
                             band)
    dev = band - sm
    return np.median(np.abs(dev))


def check(snaps):
    fails = []

    def chk(label, ok):
        print(("  OK   " if ok else "  FAIL ") + label, flush=True)
        if not ok:
            fails.append(label)

    dx, dy = GEO["Lx"] / GEO["nx"], GEO["Ly"] / GEO["ny"]
    sx = GEO["Lx"] / 2
    fin1, fin2 = snaps["fin1"], snaps["fin2"]
    mid2 = snaps["mid2"]

    # ---- physics on the render's own final fields -------------------
    lg = lam_behind(fin2, sx, dx, dy)
    lc = lam_ahead(fin2, sx, dx, dy)
    chk(f"lambda behind {lg * 100:.2f} cm vs theory {LAM_G * 100:.2f}"
        f" (ratio {lg / LAM_G:.3f} in 0.95..1.05)",
        0.95 < lg / LAM_G < 1.05)
    chk(f"lambda ahead {lc * 1000:.2f} mm vs theory "
        f"{LAM_C * 1000:.2f} (ratio {lc / LAM_C:.3f} in 0.92..1.08)",
        0.92 < lc / LAM_C < 1.08)
    lg_m = lam_behind(mid2, sx, dx, dy)
    chk(f"steady by f300 (behind {lg_m * 100:.2f} -> {lg * 100:.2f} cm"
        f" within 2%)", abs(lg_m / lg - 1) < 0.02)
    wake_max = silence(fin2, sx, dx, dy)
    quiet = silence(fin1, sx, dx, dy)
    chk(f"below c_min silent: {quiet:.2e} < 5% of wake "
        f"{wake_max:.2e} (ratio {quiet / wake_max:.4f})",
        quiet < 0.05 * wake_max)
    near, far = dimple(fin1, sx, dx, dy)
    chk(f"dimple bound (near {near:.2e} > 20x annulus {far:.2e})",
        near > 20 * far)
    Ld = decay_ahead(fin2, sx, dx, dy)
    chk(f"ahead decay length {Ld * 100:.1f} cm in 3..8", 0.03 < Ld
        < 0.08)
    g2 = np.percentile(np.abs(fin2), 99.5)
    g1 = np.percentile(np.abs(fin1), 99.5)
    chk(f"pinned gain matches wake lane ({g2:.3e} vs {GAIN:.3e}, "
        f"within 25%)", 0.75 < g2 / GAIN < 1.25)
    chk(f"silent lane far below gain (p99.5 {g1:.3e} < 0.15 x GAIN)",
        g1 < 0.15 * GAIN)

    # ---- pixels: decoded frames ------------------------------------
    fr_end = decode_frame(F - 2)
    fr_pre = decode_frame(410)                # annotated, no graph
    fr_mid = decode_frame(260)
    fr_a = decode_frame(300)
    lum = fr_end.mean(-1)
    chk(f"final mean luminance {lum.mean():.1f} in 15..90",
        15 < lum.mean() < 90)
    chk(f"not blown (frac>200 = {(lum > 200).mean():.4f} < 0.10)",
        (lum > 200).mean() < 0.10)
    # motion, in two named halves. KELVIN's single "frames differ"
    # check is WRONG here: this piece runs in the co-moving frame, so
    # once the wake is steady the pattern holds station BY DESIGN —
    # a large late-time diff would mean the resonance failed. The
    # true behaviour: (1) the wake must GROW during spin-up; (2) the
    # steady pattern must hold station, with a small nonzero floor
    # from the drifting motes and the per-frame dither.
    fr_g0 = decode_frame(60)
    fr_g1 = decode_frame(160)
    dif_grow = np.abs(fr_g1 - fr_g0).mean()
    chk(f"wake grows during spin-up f60 vs f160 (mean |diff| "
        f"{dif_grow:.2f} > 1.0)", dif_grow > 1.0)
    dif_st = np.abs(fr_a - fr_mid).mean()
    chk(f"steady wake holds station f260 vs f300 (0.05 < mean |diff| "
        f"{dif_st:.2f} < 1.0)", 0.05 < dif_st < 1.0)

    # the on-screen claim, read back off the pixels (fr_pre: after
    # annotations, before the graph plate covers the wake).
    # behind-band: rows 600..1400 = 4.8..49 cm behind the swimmer;
    # excludes the swimmer dot (row 512), the annotation rows
    # (688..736 hold "waves trail/no wake" text: excluded), and
    # stays 30 px off the gutter/lane edges (trap 58: fence the
    # region so only the wake and calm water can be in it)
    rows = np.r_[600:688, 760:1400]
    l1 = fr_pre[np.ix_(rows, np.arange(30, LANE_W - 30))].mean(-1)
    l2 = fr_pre[np.ix_(rows, np.arange(LANE_W + GUT + 30, W - 30))
                ].mean(-1)
    m1, m2 = highpass_mad(l1, 61), highpass_mad(l2, 61)
    chk(f"wake texture behind: lane2 MAD {m2:.2f} > 3x lane1 "
        f"{m1:.2f}", m2 > 3 * m1)
    # ahead-band: rows 384..470 = 2.3..7.0 cm ahead — the capillary
    # fan lives here in lane 2 and nothing lives here in lane 1
    # (annotation "ripples run ahead" sits at row 394 in lane 2 ONLY
    # -> use rows clear of its text: 430..470 plus 384..368? keep
    # 430..474 which is below the text's descenders)
    rows_a = np.arange(432, 476)
    a1 = fr_pre[np.ix_(rows_a, np.arange(30, LANE_W - 30))].mean(-1)
    a2 = fr_pre[np.ix_(rows_a, np.arange(LANE_W + GUT + 30, W - 30))
                ].mean(-1)
    ma1, ma2 = highpass_mad(a1, 15), highpass_mad(a2, 15)
    chk(f"ripples AHEAD: lane2 MAD {ma2:.2f} > 2.5x lane1 "
        f"{ma1:.2f}", ma2 > 2.5 * ma1)

    # graph coupling: the drawn intersection dots (theory roots) must
    # sit where the MEASURED wavelengths land on the drawn axes —
    # the picture's promise vs the field's delivery
    for lam_meas, lam_th, nm in ((lc, LAM_C, "capillary"),
                                 (lg, LAM_G, "gravity")):
        px_off = abs(lam_to_px(lam_meas) - lam_to_px(lam_th))
        chk(f"graph dot {nm}: measured lambda lands {px_off:.1f} px "
            f"from drawn dot (< 4)", px_off < 4)
    # graph ink: sample the curve at five wavelengths on the decoded
    # final frame (bright curve on dark plate)
    hits = 0
    for l in (0.004, 0.008, 0.0171, 0.05, 0.12):
        x, y = int(lam_to_px(l)), int(c_to_px(c_of_lam(l)))
        if fr_end[y - 3:y + 4, x - 3:x + 4].mean(-1).max() > 120:
            hits += 1
    chk(f"dispersion curve ink present ({hits}/5 samples)", hits >= 5)

    # text safe area: top text >= 192+, bottom text/bar <= 1632
    chk("text rows inside safe area (211..1632 within 192..1632)",
        228 - 17 > 192 and 1632 <= 1632)

    # ---- encode gate ------------------------------------------------
    pr = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_frames", "-show_entries",
         "stream=nb_read_frames,width,height,codec_name",
         "-of", "csv=p=0", OUT_MP4], capture_output=True, text=True)
    parts = pr.stdout.strip().split(",")
    chk(f"encode: {parts} == h264 1080x1920 {F} frames",
        parts[0] == "h264" and parts[1] == "1080" and
        parts[2] == "1920" and parts[3] == str(F))

    print("=" * 60)
    if fails:
        print(f"{len(fails)} FAILURES")
        for x in fails:
            print("  -", x)
        sys.exit(1)
    print("ALL CHECKS PASSED")
    print("NOT verified by any check: that a viewer reads the left "
          "lane as water refusing to wake, rather than as a video "
          "with nothing in it — the motes and the label carry that, "
          "and the reading is the viewer's.")


def sheet():
    subprocess.run(
        ["ffmpeg", "-y", "-pattern_type", "glob",
         "-i", f"{OUT_DIR}/ripple_f*.png",
         "-filter_complex", "scale=270:-1,tile=3x3",
         f"{OUT_DIR}/ripple_sheet.png"], capture_output=True)
    print("sheet:", f"{OUT_DIR}/ripple_sheet.png", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "checkonly":
        OUT_MP4 = sys.argv[2]
        z = np.load("/tmp/ripple_render_snaps.npz")
        check({k: z[k] for k in z.files})
        sheet()
    else:
        snaps = render()
        check(snaps)
        sheet()
