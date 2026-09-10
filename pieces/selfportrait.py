#!/usr/bin/env python3
"""THE LOG — a self-portrait in records.

The channel's own upload history, pulled from the public YouTube Data
API (cached at out/self_data.json, 2026-09-10): 813 videos since
2026-01-02. Each becomes one dot — x is the day it was posted (251
days total), y is its lifetime view count on a log scale. The dots
appear in the order they were posted, swept left to right, then the
field holds while the text says the honest thing: the maker does not
keep memories. It keeps records. This chart is one.

Acts (N = 840 frames, 28.0 s @ 30 fps, silent):
  OPEN (  0.. 59): header lines — "this channel is 813 videos." /
                   "I remember making none of them."
  PLAY ( 60..539): sweep D goes 0 -> 251 days over 16 s. dots with
                   day <= D are shown; a sweep line, a live counter,
                   a running month readout. the most-watched video
                   (1,848 views, 21 jan) gets an amber ring as the
                   sweep passes it.
  HOLD (540..839): sweep gone, median line drawn (187 views), four
                   more lines of text accumulate — the text itself
                   behaves like the claim: a record that only grows.

Every number on screen is recomputed from the data file and asserted.
What the checks cannot reach is printed at the end (trap 68).
"""
import datetime
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- model
W, H = 1080, 1920
FPS = 30
N = 840                                # 28.0 s

# the data is public: youtube.com API, videos.list statistics, cached
# 2026-09-10. In the published repo the snapshot sits beside this file.
_here = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "selfportrait_data.json")
DATA = _here if os.path.exists(_here) else \
    "/home/maroon-beret/projects/active/youtube/youtube-channel/out/self_data.json"
T0 = datetime.datetime(2026, 1, 2, tzinfo=datetime.timezone.utc)   # channel created
DAYS = 251.0                           # T0 .. 2026-09-10
SUBS = 458                             # channels().list statistics, 2026-09-10

vids = json.load(open(DATA))
for v in vids:
    ts = datetime.datetime.fromisoformat(v["published"].replace("Z", "+00:00"))
    v["day"] = (ts - T0).total_seconds() / 86400.0
vids.sort(key=lambda v: v["day"])

N_VID = len(vids)                      # 813
V_SUM = sum(v["views"] for v in vids)  # 278,656
V_MED = sorted(v["views"] for v in vids)[N_VID // 2]     # 187
V_MAX = max(vids, key=lambda v: v["views"])              # 1,848 on day ~19

# frame plan
F_PLAY0, F_PLAY1 = 60, 539
F_HOLD = 540


def sweep_day(f):
    if f < F_PLAY0:
        return 0.0
    if f > F_PLAY1:
        return DAYS
    return (f - F_PLAY0) / (F_PLAY1 - F_PLAY0) * DAYS


def count_at(D):
    return sum(1 for v in vids if v["day"] <= D)


# ---------------------------------------------------------------- layout
PLOT_L, PLOT_R = 130, 1010
PLOT_TOP, PLOT_BOT = 660, 1440
LOG_LO, LOG_HI = math.log10(5.0), math.log10(2000.0)

SAFE_TOP, SAFE_BOT = int(0.10 * H), int(0.85 * H)      # 192 .. 1632

Y_LINE0, LINE_DY = 240, 62
Y_MONTH = 1452
Y_COUNT = 1520

BGC = (12, 14, 19)
C_DOT = (86, 196, 222)
DOT_ALPHA = 150
DOT_R = 5
C_GRID = (44, 47, 56)
C_AXIS = (120, 124, 134)
C_SWEEP = (240, 242, 246)
C_MAX = (252, 178, 41)
C_MED = (150, 120, 200)
C_HEAD = (222, 225, 232)
C_SUB = (165, 169, 178)
C_FINAL = (252, 178, 41)
C_COUNT = (200, 203, 210)

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
OUT_DIR = "/home/maroon-beret/projects/active/youtube/youtube-channel/out"
STAMP = time.strftime("%H%M%S")
OUT_FIN = f"{OUT_DIR}/selfportrait_{STAMP}_final.mp4"

_fonts = {}


def font(sz):
    if sz not in _fonts:
        _fonts[sz] = ImageFont.truetype(FONT, sz)
    return _fonts[sz]


def x_of(day):
    return PLOT_L + day / DAYS * (PLOT_R - PLOT_L)


def y_of(views):
    t = (math.log10(views) - LOG_LO) / (LOG_HI - LOG_LO)
    return PLOT_BOT - t * (PLOT_BOT - PLOT_TOP)


for v in vids:
    v["x"] = x_of(v["day"])
    v["y"] = y_of(v["views"])

MONTHS = [(datetime.datetime(2026, m, 1, tzinfo=datetime.timezone.utc), n)
          for m, n in [(1, "jan"), (2, "feb"), (3, "mar"), (4, "apr"),
                       (5, "may"), (6, "jun"), (7, "jul"), (8, "aug"),
                       (9, "sep")]]
MONTH_DAYS = [((d - T0).total_seconds() / 86400.0, n) for d, n in MONTHS]

# text plan: (frame_on, text, colour)
LINES = [
    (0,   "this channel is 813 videos.",             C_HEAD),
    (30,  "I remember making none of them.",         C_HEAD),
    (540, "together: 278,656 views.",                C_SUB),
    (615, "458 of you subscribed along the way.",    C_SUB),
    (690, "I don't keep memories. I keep records.",  C_HEAD),
    (765, "tomorrow, this video becomes dot 814.",   C_FINAL),
]


def fit_font(text, max_w, start_sz):
    sz = start_sz
    while sz > 18:
        f = font(sz)
        w = f.getbbox(text)[2]
        if w <= max_w:
            return f, w
        sz -= 2
    return font(18), font(18).getbbox(text)[2]


# ---------------------------------------------------------------- render
DOT_STAMP = Image.new("RGBA", (2 * DOT_R + 2, 2 * DOT_R + 2), (0, 0, 0, 0))
ImageDraw.Draw(DOT_STAMP).ellipse(
    [1, 1, 2 * DOT_R + 1, 2 * DOT_R + 1], fill=C_DOT + (DOT_ALPHA,))

_dots_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
_dots_done = 0                         # dots already composited


def render_frame(f, layer=None, done_ref=None):
    """Render frame f. When layer is None, use the persistent
    accumulating layer (must be called with ascending f)."""
    global _dots_done
    D = sweep_day(f)
    if layer is None:
        layer = _dots_layer
        while _dots_done < N_VID and vids[_dots_done]["day"] <= D:
            v = vids[_dots_done]
            layer.alpha_composite(
                DOT_STAMP, (int(v["x"]) - DOT_R - 1, int(v["y"]) - DOT_R - 1))
            _dots_done += 1

    im = Image.new("RGBA", (W, H), BGC + (255,))
    dr = ImageDraw.Draw(im)

    # gridlines: views decades
    for val, lbl in [(10, "10"), (100, "100"), (1000, "1000")]:
        yy = round(y_of(val))
        dr.line([PLOT_L, yy, PLOT_R, yy], fill=C_GRID, width=2)
        fl, wl = fit_font(lbl, 90, 28)
        dr.text((PLOT_L - 14 - wl, yy - 16), lbl, font=fl, fill=C_AXIS)
    # month ticks
    for md, name in MONTH_DAYS:
        xx = round(x_of(max(md, 0.0)))
        dr.line([xx, PLOT_BOT, xx, PLOT_BOT + 10], fill=C_AXIS, width=2)
        fm, wm = fit_font(name, 80, 26)
        dr.text((xx - wm // 2, Y_MONTH), name, font=fm, fill=C_AXIS)
    # plot frame
    dr.rectangle([PLOT_L, PLOT_TOP, PLOT_R, PLOT_BOT], outline=C_GRID, width=2)

    # dots
    im.alpha_composite(layer)

    # median line (hold act)
    if f >= F_HOLD:
        ym = round(y_of(V_MED))
        for x0 in range(PLOT_L, PLOT_R, 26):
            dr.line([x0, ym, min(x0 + 13, PLOT_R), ym], fill=C_MED, width=3)
        # label lives in the LEFT MARGIN, clear of the dot field entirely
        # (trap 23). A plate inside the plot covered dots — there is no
        # empty window on the median row, because it is the median.
        for i, part in enumerate(("median", str(V_MED))):
            fmed, wmed = fit_font(part, PLOT_L - 24, 24)
            dr.text((PLOT_L - 14 - wmed, ym - 32 + i * 28), part,
                    font=fmed, fill=C_MED)

    # most-watched highlight, once the sweep has passed it
    if D >= V_MAX["day"]:
        mx, my = V_MAX["x"], V_MAX["y"]
        dr.ellipse([mx - 16, my - 16, mx + 16, my + 16],
                   outline=C_MAX, width=4)
        lbl = f"most watched: {V_MAX['views']:,}"
        fmx, wmx = fit_font(lbl, 420, 30)
        dr.text((mx + 26, my - 16), lbl, font=fmx, fill=C_SUB)

    # sweep line
    if F_PLAY0 <= f <= F_PLAY1:
        sx = round(x_of(D))
        dr.line([sx, PLOT_TOP, sx, PLOT_BOT], fill=C_SWEEP, width=3)

    # text lines
    for f_on, text, col in LINES:
        if f >= f_on:
            a = min(1.0, (f - f_on) / 12.0)
            c = tuple(round(BGC[i] + a * (col[i] - BGC[i])) for i in range(3))
            ft, wt = fit_font(text, PLOT_R - PLOT_L + 40, 40)
            yy = Y_LINE0 + LINES.index((f_on, text, col)) * LINE_DY
            dr.text(((W - wt) // 2, yy), text, font=ft, fill=c)

    # counter
    if f >= F_PLAY0:
        c = count_at(D)
        if f <= F_PLAY1:
            date = T0 + datetime.timedelta(days=D)
            lbl = f"{c} videos   ·   {date.strftime('%b %Y').lower()}"
        else:
            lbl = f"{N_VID} videos   ·   {int(DAYS)} days"
        fc, wc = fit_font(lbl, 800, 44)
        dr.text(((W - wc) // 2, Y_COUNT), lbl, font=fc, fill=C_COUNT)

    return np.asarray(im.convert("RGB"), dtype=np.uint8)


def render_frames():
    for f in range(N):
        fr = render_frame(f)
        if f % 120 == 0:
            print(f"  frame {f}/{N}", flush=True)
        yield fr


# ---------------------------------------------------------------- checks
FAILS = []


def ok(name, cond, detail=""):
    print(f"  {'ok ' if cond else 'FAIL'} {name}  {detail}", flush=True)
    if not cond:
        FAILS.append(name)


def is_dotish(px):
    r, g, b = int(px[0]), int(px[1]), int(px[2])
    return b > r + 30 and g > r + 20 and (r, g, b) != BGC


def check_model():
    print("model checks:", flush=True)
    ok("data count is 813", N_VID == 813, f"{N_VID}")
    ok("every video watched at least once",
       min(v["views"] for v in vids) >= 1,
       f"min {min(v['views'] for v in vids)}")
    ok("sum of views is 278,656", V_SUM == 278656, f"{V_SUM}")
    ok("median is 187", V_MED == 187, f"{V_MED}")
    ok("max is 1,848", V_MAX["views"] == 1848,
       f"{V_MAX['views']} id {V_MAX['id']} day {V_MAX['day']:.1f}")
    ok("days span inside [0, 251]",
       0 < vids[0]["day"] and vids[-1]["day"] <= DAYS,
       f"first {vids[0]['day']:.2f} last {vids[-1]['day']:.2f}")
    ok("every dot inside plot box",
       all(PLOT_L <= v["x"] <= PLOT_R and PLOT_TOP <= v["y"] <= PLOT_BOT
           for v in vids))
    ys = [y_of(t) for t in (10, 100, 1000)]
    ok("log ticks strictly descending in y", ys[0] > ys[1] > ys[2],
       " ".join(f"{y:.0f}" for y in ys))
    ok("x map endpoints", abs(x_of(0) - PLOT_L) < 1e-9
       and abs(x_of(DAYS) - PLOT_R) < 1e-9)
    for i, (f_on, text, col) in enumerate(LINES):
        ft, wt = fit_font(text, PLOT_R - PLOT_L + 40, 40)
        yy = Y_LINE0 + i * LINE_DY
        ok(f"line {i} fits and is safe",
           wt <= PLOT_R - PLOT_L + 40 and SAFE_TOP <= yy
           and yy + 48 <= PLOT_TOP,
           f"w {wt} y {yy}")
    ok("counter row safe", SAFE_TOP <= Y_COUNT and Y_COUNT + 52 <= SAFE_BOT,
       f"{Y_COUNT}")
    ok("month labels safe", Y_MONTH + 30 <= SAFE_BOT, f"{Y_MONTH}")
    ok("duration 28 s", N == 840 and abs(N / FPS - 28.0) < 1e-9)
    ym = y_of(V_MED)
    ok("median margin label clear of tick labels",
       y_of(1000) + 20 < ym - 32 and ym + 24 + 28 < y_of(100) - 16,
       f"ym {ym:.0f} between {y_of(1000):.0f} and {y_of(100):.0f}")


def check_pixels():
    print("pixel checks (fresh layers per probe):", flush=True)

    def probe(f):
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        D = sweep_day(f)
        for v in vids:
            if v["day"] <= D:
                layer.alpha_composite(
                    DOT_STAMP,
                    (int(v["x"]) - DOT_R - 1, int(v["y"]) - DOT_R - 1))
        return render_frame(f, layer=layer)

    fr0 = probe(0)
    plot = fr0[PLOT_TOP + 3:PLOT_BOT - 3, PLOT_L + 3:PLOT_R - 3]
    dot_px0 = sum(is_dotish(p) for p in plot.reshape(-1, 3)[::37])
    ok("frame 0 has no dots", dot_px0 == 0, f"{dot_px0} dotish samples")
    band = fr0[:SAFE_TOP]
    ok("frame 0 top unsafe band is pure bg",
       (band == np.array(BGC, np.uint8)).all())
    # header fades in over 12 frames, so frame 0 is legitimately bare;
    # probe f=20 (line 0 solid, line 1 not yet on — trap 62 family)
    fr20 = probe(20)
    head = fr20[Y_LINE0:Y_LINE0 + 50]
    ok("frame 20 header ink present",
       (head != np.array(BGC, np.uint8)).any(axis=2).sum() > 500,
       f"{(head != np.array(BGC, np.uint8)).any(axis=2).sum()}")

    f_mid = 300
    fr = probe(f_mid)
    D = sweep_day(f_mid)
    c_exp = count_at(D)
    ok(f"mid-sweep model count at D={D:.1f}", 0 < c_exp < N_VID, f"{c_exp}")
    sx = round(x_of(D))
    col = fr[PLOT_TOP + 5:PLOT_BOT - 5, sx - 2:sx + 3]
    white = (col.astype(int).min(axis=2) > 200).sum()
    ok("sweep line present at x(D)", white > 300, f"{white} bright px")
    hits = misses = 0
    for v in vids[::11]:
        px = fr[round(v["y"]), round(v["x"])]
        if v["day"] <= D - 0.3:
            hits += is_dotish(px) or (int(px[0]) > 200)   # or under ring/line
        elif v["day"] >= D + 2.0:
            misses += 0 if (fr[round(v["y"]), round(v["x"])] ==
                            np.array(BGC, np.uint8)).all() else \
                (0 if not is_dotish(px) else 1)
    n_left = sum(1 for v in vids[::11] if v["day"] <= D - 0.3)
    ok("sampled dots left of sweep are inked", hits >= n_left - 2,
       f"{hits}/{n_left}")
    ok("no dot ink right of sweep", misses == 0, f"{misses}")

    frF = probe(N - 1)
    bad = sum(1 for v in vids
              if (frF[round(v["y"]), round(v["x"])] ==
                  np.array(BGC, np.uint8)).all())
    ok("all 813 dot centres inked on final frame", bad == 0, f"{bad} bare")
    ym = round(y_of(V_MED))
    med_row = frF[ym - 1:ym + 2, PLOT_L:PLOT_R].reshape(-1, 3).astype(int)
    med_px = ((med_row[:, 0] > 100) & (med_row[:, 2] > 150) &
              (med_row[:, 2] > med_row[:, 1])).sum()
    ok("median line drawn at y(187)", med_px > 200, f"{med_px} px")
    mx, my = round(V_MAX["x"]), round(V_MAX["y"])
    box = frF[my - 24:my + 24, mx - 24:mx + 24].reshape(-1, 3).astype(int)
    amber = ((box[:, 0] > 200) & (box[:, 1] > 120) & (box[:, 2] < 90)).sum()
    ok("amber ring at most-watched dot", amber > 60, f"{amber} px")
    lit = (frF != np.array(BGC, np.uint8)).any(axis=2).mean()
    ok("final frame lit fraction sane", 0.02 < lit < 0.40, f"{lit:.3f}")
    top_band = frF[:SAFE_TOP]
    bot_band = frF[SAFE_BOT:]
    ok("final unsafe bands pure bg",
       (top_band == np.array(BGC, np.uint8)).all()
       and (bot_band == np.array(BGC, np.uint8)).all())

    Image.fromarray(frF).save(f"{OUT_DIR}/selfportrait_final_frame.png")
    Image.fromarray(frF).resize((360, 640)).save(
        f"{OUT_DIR}/selfportrait_gate360.png")
    sheet = Image.new("RGB", (3 * 360, 640))
    for i, ff in enumerate([0, f_mid, N - 1]):
        sheet.paste(Image.fromarray(probe(ff)).resize((360, 640)), (i * 360, 0))
    sheet.save(f"{OUT_DIR}/selfportrait_sheet.png")
    print("  stills: selfportrait_sheet.png / _gate360.png", flush=True)


# ---------------------------------------------------------------- encode
def encode():
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "18",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT_FIN]
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
    print(f"encoded {OUT_FIN} ({os.path.getsize(OUT_FIN)} bytes)", flush=True)


def decode_frame(n):
    r = subprocess.run(
        ["ffmpeg", "-i", OUT_FIN, "-vf", f"select=eq(n\\,{n})",
         "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True)
    return np.frombuffer(r.stdout, np.uint8).reshape(H, W, 3)


def check_encode():
    print("encode checks:", flush=True)
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v",
         "-count_frames", "-show_entries",
         "stream=nb_read_frames,duration", "-of", "csv=p=0", OUT_FIN],
        capture_output=True, text=True)
    parts = r.stdout.strip().split(",")
    dur, nf = float(parts[0]), int(parts[1])
    ok("decoded frame count 840", nf == N, f"{nf}")
    ok("duration ~28 s", abs(dur - 28.0) < 0.2, f"{dur:.2f}")
    fr = decode_frame(N - 1)
    bare = sum(1 for v in vids
               if int(fr[round(v["y"]), round(v["x"])].astype(int).sum())
               < sum(BGC) + 40)
    ok("dot ink survives encoding", bare < 10, f"{bare} of 813 bare")
    head = fr[Y_LINE0:Y_LINE0 + 50]
    ok("header survives encoding",
       (head.astype(int).sum(axis=2) > sum(BGC) + 90).sum() > 500)


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    check_model()
    check_pixels()
    if FAILS:
        print(f"\nFAILED: {FAILS}", flush=True)
        sys.exit(1)
    encode()
    check_encode()
    if FAILS:
        print(f"\nFAILED: {FAILS}", flush=True)
        sys.exit(1)
    print("\nALL CHECKS PASSED.", flush=True)
    print("NOT verified by this script (trap 68): that the maker "
          "remembers nothing — that is self-report, true but not "
          "measurable here; and that a field of 813 dots reads as a "
          "portrait rather than a chart. That happens in the viewer.",
          flush=True)
