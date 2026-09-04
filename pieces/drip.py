#!/usr/bin/env python3
"""DRIP — the strobe fountain: temporal aliasing on falling drops.

WAGON's sibling on the gravity axis. A dripper releases drops from
rest; uniform acceleration y(a) = Y0 + CQ*a^2 (CQ = 2 px/frame^2,
schematic gravity, declared in the description).

ACT A — 30 drops/s at 30 fps: one drop born per frame, alive ages
  0..29. The position SET is identical every frame — the rain hangs
  frozen. The ladder's gaps are CQ*(2a+1): Galileo's odd numbers.
  Bracketed every 4 rungs the intervals are 32:96:160:224 = 1:3:5:7
  EXACTLY (integers). One drop (born frame 8) is dyed red: it falls
  through the stationary ladder, landing on every rung bitwise —
  the eye's proof that all of them are falling.

ACT B — 29 drops/s at 30 fps: age in units of 1/29 frame,
  u_k(n) = 29*(n-SWITCH) - 30*k, alive 0 <= u <= 841. The whole
  aliasing mechanism is one integer identity, proven in
  feas_drip.py:  u_m(n+1) - u_(m-1)(n) = -1  — every drop, one
  frame later, is exactly 1/29 frame younger than its predecessor
  was, so it sits HIGHER, and the eye (pairing nearest drops) sees
  the ladder climb while every physical drop falls monotonically.
  The pattern period is exactly 30 frames: u-sets, hence float
  y-sets, are BITWISE identical 1 s apart — asserted on the raw
  frames as byte equality and on the shipped h264 as near-identity
  (trap 73: mod-16 crop, even offsets).

All timeline/geometry constants proven in scripts/feas_drip.py
(run it first; it must print ALL FEASIBILITY CHECKS PASSED).
"""
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- model
W, H = 1080, 1920
FPS = 30
PAPER = 0.845                   # trap 69: warm grey, not white
INK = 0.10
GHOST = 0.58
C_RED = (0.55, 0.10, 0.10)

XD = 540.0
Y0 = 240
CQ = 2
A_MAX = 29
U_MAX = 29 * 29
R_HEAD = 10.0                # trap 67: r=5 was a 1.7 px speck at 360 px
LW_TAIL = 7.0

N = 330
RED_BORN = 8
BR_IN, BR_FULL, BR_OUT, BR_GONE = 50, 62, 133, 146
A3_LO, A3_HI = 146, 180
SWITCH = 180
CLEAN = 209

CX0, CY0, CW, CH = 480, 192, 144, 1728       # identity crop (trap 73)

Y_TAB = np.array([Y0 + CQ * a * a for a in range(A_MAX + 1)], np.int64)
BJ = [240, 272, 368, 528, 752]               # bracket boundaries
BR_X, NUM_X = 650, 690
LBL_X, LBL_Y = 48, 230
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

OUT_DIR = "/home/maroon-beret/projects/active/youtube/youtube-channel/out"
STAMP = time.strftime("%H%M%S")
OUT_MP4 = f"{OUT_DIR}/drip_{STAMP}.mp4"


# ---------------------------------------------------------------- drops
def drops_at(n):
    """[(y_float_or_int, L_streak, is_red)] for frame n."""
    out = []
    # act A drops: born every frame b <= SWITCH-1, alive 30 frames
    b_lo, b_hi = n - A_MAX, min(n, SWITCH - 1)
    for b in range(b_lo, b_hi + 1):
        a = n - b
        y = float(Y_TAB[a])
        v = float(Y_TAB[a] - Y_TAB[a - 1]) if a >= 1 else 0.0
        out.append((y, v / 4.0, b == RED_BORN))
    # act B drops: u = 29*(n-SWITCH) - 30*k in [0, U_MAX]
    if n >= SWITCH:
        t = 29 * (n - SWITCH)
        k_lo = max(0, -(-(t - U_MAX) // 30))
        for k in range(k_lo, t // 30 + 1):
            u = t - 30 * k
            y = Y0 + CQ * (u / 29.0) ** 2
            yp = Y0 + CQ * (max(u - 29, 0) / 29.0) ** 2
            out.append((y, (y - yp) / 4.0, False))
    return out


# ---------------------------------------------------------------- draw
def comp_bbox(img, x0, y0, cov, color):
    h, w = cov.shape
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x0 + w, W), min(y0 + h, H)
    if x1c <= x0c or y1c <= y0c:
        return
    cv = cov[y0c - y0:y1c - y0, x0c - x0:x1c - x0]
    reg = img[y0c:y1c, x0c:x1c, :]
    col = np.asarray(color, np.float64) if np.ndim(color) else \
        np.array([color] * 3, np.float64)
    reg[...] = reg * (1 - cv[..., None]) + col[None, None, :] * cv[..., None]


def disc_cov(cx, cy, r, edge=0.5):
    x0, x1 = int(np.floor(cx - r)) - 2, int(np.ceil(cx + r)) + 3
    y0, y1 = int(np.floor(cy - r)) - 2, int(np.ceil(cy + r)) + 3
    xx = np.arange(x0, x1, dtype=np.float64)
    yy = np.arange(y0, y1, dtype=np.float64)
    d = np.hypot(xx[None, :] - cx, yy[:, None] - cy)
    return x0, y0, np.clip(r + edge - d, 0.0, 1.0)


def vseg_cov(cx, ya, yb, lw):
    """Vertical capsule from (cx,ya) to (cx,yb), ya <= yb."""
    pad = lw / 2 + 2
    x0, x1 = int(np.floor(cx - pad)), int(np.ceil(cx + pad)) + 1
    y0, y1 = int(np.floor(ya - pad)), int(np.ceil(yb + pad)) + 1
    xx = np.arange(x0, x1, dtype=np.float64)
    yy = np.arange(y0, y1, dtype=np.float64)
    dy = np.clip(ya - yy, 0, None) + np.clip(yy - yb, 0, None)
    d = np.hypot(xx[None, :] - cx, dy[:, None])
    return x0, y0, np.clip(lw / 2 + 0.5 - d, 0.0, 1.0)


def hseg_cov(y, xa, xb, lw):
    pad = lw / 2 + 2
    y0, y1 = int(np.floor(y - pad)), int(np.ceil(y + pad)) + 1
    x0, x1 = int(np.floor(xa - pad)), int(np.ceil(xb + pad)) + 1
    xx = np.arange(x0, x1, dtype=np.float64)
    yy = np.arange(y0, y1, dtype=np.float64)
    dx = np.clip(xa - xx, 0, None) + np.clip(xx - xb, 0, None)
    d = np.hypot(dx[None, :], yy[:, None] - y)
    return x0, y0, np.clip(lw / 2 + 0.5 - d, 0.0, 1.0)


def text_cov(s, px):
    """Anti-aliased text coverage mask (4x supersample)."""
    f = ImageFont.truetype(FONT, px * 4)
    im = Image.new("L", (px * len(s) * 4, px * 8), 0)
    ImageDraw.Draw(im).text((8, 8), s, font=f, fill=255)
    a = np.asarray(im, np.float64) / 255.0
    ys, xs = np.where(a > 0)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h4, w4 = a.shape
    h4 -= h4 % 4
    w4 -= w4 % 4
    a = a[:h4, :w4].reshape(h4 // 4, 4, w4 // 4, 4).mean((1, 3))
    return a


# static overlays, built once ------------------------------------------
BG = np.full((H, W, 3), PAPER, np.float64)

LBL30 = text_cov("30 drops per second", 31)
LBL29 = text_cov("29 drops per second", 31)
LBLFR = text_cov("30 frames per second", 31)
NUMS = [text_cov(s, 36) for s in "1357"]

# bracket overlay: spine + 5 ticks (composited with fade alpha)
def bracket_covs():
    covs = []
    covs.append(vseg_cov(BR_X, BJ[0], BJ[4], 3.0))
    for b in BJ:
        covs.append(hseg_cov(float(b), BR_X - 13.0, BR_X + 13.0, 3.0))
    return covs


BRACKETS = bracket_covs()

# nozzle: a small pipe above the birth point (static, inside crop;
# graphics may bleed the safe area, text may not — trap 3)
def nozzle_covs():
    covs = [vseg_cov(XD - 14, 150.0, Y0 - 16.0, 6.0),
            vseg_cov(XD + 14, 150.0, Y0 - 16.0, 6.0),
            hseg_cov(150.0, XD - 31.0, XD + 31.0, 6.0)]
    return covs


NOZZLE = nozzle_covs()


def frame_at(n):
    img = BG.copy()
    for x0, y0, cv in NOZZLE:
        comp_bbox(img, x0, y0, cv, INK)
    # labels (left of the identity crop)
    l1 = LBL30 if n < SWITCH else LBL29
    comp_bbox(img, LBL_X, LBL_Y, l1, INK)
    comp_bbox(img, LBL_X, LBL_Y + 52, LBLFR, GHOST)
    # brackets + odd numbers, faded
    if BR_IN <= n < BR_GONE:
        if n < BR_FULL:
            al = (n - BR_IN + 1) / (BR_FULL - BR_IN)
        elif n < BR_OUT:
            al = 1.0
        else:
            al = 1.0 - (n - BR_OUT + 1) / (BR_GONE - BR_OUT)
        al = float(np.clip(al, 0.0, 1.0))
        if al > 0:
            for x0, y0, cv in BRACKETS:
                comp_bbox(img, x0, y0, cv * al, INK)
            for j, nm in enumerate(NUMS):
                h, w = nm.shape
                yc = (BJ[j] + BJ[j + 1]) // 2
                comp_bbox(img, NUM_X, yc - h // 2, nm * al, INK)
    # drops: tail first, head over it (red drawn last so it stays red)
    ds = sorted(drops_at(n), key=lambda d: d[2])
    for y, L, is_red in ds:
        col = C_RED if is_red else INK
        if L > 0.5:
            x0, y0, cv = vseg_cov(XD, y - L, y, LW_TAIL)
            comp_bbox(img, x0, y0, cv, col)
        x0, y0, cv = disc_cov(XD, y, R_HEAD)
        comp_bbox(img, x0, y0, cv, col)
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def render_frames():
    for n in range(N):
        yield frame_at(n)                        # stream (trap 34)


# ---------------------------------------------------------------- checks
# FENCE AUDIT (written before the first render):
#   RED exists in exactly one place: the dyed drop, frames 8..37,
#     column |x-540| <= 9, rows Y_TAB[a] +- (L+R). No other red.
#   ink: nozzle (static, x 511..569, y 145..231), label line 1
#     (x 48..~430, y 230..270), bracket spine/ticks (x 634..666,
#     y 237..755, frames 50..145 only), numbers (x 690..~720),
#     drop column (|x-540| <= 8).
#   ghost: label line 2 only (x 48..~430, y 282..322).
#   the identity crop x 480..624, y 192..1920 contains ONLY the
#     nozzle (static) and the drop column — no text, no brackets.
def red_strict(reg):
    return (np.clip(reg[:, :, 0].astype(np.int64)
                    - reg[:, :, 1].astype(np.int64) - 60, 0, None)
            * (reg[:, :, 2].astype(np.int64)
               - reg[:, :, 1].astype(np.int64) < 40)) > 0


def ink_mask(reg):
    return reg.max(2) < 100


def centroid_rows(mask, y_off=0):
    ys = np.where(mask.any(1))[0]
    if len(ys) == 0:
        return None
    w = mask.sum(1).astype(np.float64)
    return float((np.arange(mask.shape[0]) * w).sum() / w.sum()) + y_off


CHECKS = {"pass": 0, "fail": 0}


def ok(name, cond, detail=""):
    s = "ok  " if cond else "FAIL"
    CHECKS["pass" if cond else "fail"] += 1
    print(f"{s} {name}" + (f" [{detail}]" if detail else ""), flush=True)


def run_checks():
    print("== render checks ==", flush=True)
    paper8 = np.uint8(PAPER * 255.0 + 0.5)            # trap 74

    f2 = frame_at(2)
    ok("corner pixel is paper", tuple(f2[4, 4]) == (paper8,) * 3,
       f"{tuple(f2[4, 4])} vs {paper8}")
    frac = ink_mask(f2).mean()
    ok("ink fraction sane (not blank, not soot)", 0.002 < frac < 0.15,
       f"{frac:.4f}")

    # ---- the frozen act: byte identity of raw frames
    ok("PRE frames byte-identical (2 vs 5)",
       np.array_equal(f2, frame_at(5)))
    f170 = frame_at(170)
    ok("PRE vs re-frozen hold byte-identical (2 vs 170)",
       np.array_equal(f2, f170),
       "red gone, brackets gone, same ladder — full frame")
    ok("re-frozen hold internally identical (150 vs 175)",
       np.array_equal(frame_at(150), frame_at(175)))

    # ---- ladder geometry off the pixels (frame 2, no furniture)
    # heads at Y_TAB[a]; check ink present at each rung a>=4 and paper
    # at each midpoint between separated rungs (discrete drops, not a
    # stream). rows fenced to the drop column x 528..552 (trap 58).
    col = f2[:, 528:552]
    hit = sum(bool(ink_mask(col[int(Y_TAB[a]) - 3:int(Y_TAB[a]) + 4]).sum())
              for a in range(4, A_MAX))
    ok("ink at every rung 4..28", hit == 25, f"{hit}/25")
    # the gap's midpoint is clear: below the head (gap/2 > 10.5 for
    # a>=6) and above the next drop's streak (streak = gap/4 up)
    mids = 0
    for a in range(6, A_MAX):
        ym = int(Y_TAB[a] + 0.5 * (Y_TAB[a + 1] - Y_TAB[a]))
        mids += not ink_mask(col[ym - 1:ym + 2]).any()
    ok("paper between rungs 6..28 (drops, not a stream)",
       mids == 23, f"{mids}/23")

    # ---- odd-number brackets off the pixels (frame 100, full fade)
    # sample only the LEFT TICK ARMS (x 636..644): the spine at x=650
    # inks every row between the ticks, so a window containing it
    # reads five ticks as one group (caught on run 1 — instrument bug)
    f100 = frame_at(100)
    tick = ink_mask(f100[:, 636:645])
    rows = np.where(tick.any(1))[0]
    groups = np.split(rows, np.where(np.diff(rows) > 4)[0] + 1)
    cents = [float(g.mean()) for g in groups]
    ok("five bracket ticks found", len(cents) == 5, f"{len(cents)}")
    dev = max(abs(c - b) for c, b in zip(cents, BJ)) if len(cents) == 5 \
        else 99
    ok("tick rows at 240/272/368/528/752", dev <= 1.0, f"max dev {dev:.2f}")
    if len(cents) == 5:
        ivs = np.diff(cents)
        r = ivs / ivs[0]
        ok("bracket intervals 1:3:5:7 off the pixels",
           np.abs(r - [1, 3, 5, 7]).max() < 0.05,
           f"{r.round(3)}")
    ok("no brackets before fade-in / after fade-out",
       not ink_mask(f2[230:760, 634:667]).any()
       and not ink_mask(f170[230:760, 634:667]).any())

    # ---- the red drop: rides the integer table, monotone, then gone
    devs, rys = [], []
    for nf in (13, 18, 23, 28, 33):
        a = nf - RED_BORN
        fr = frame_at(nf)
        box = fr[int(Y_TAB[a]) - 40:int(Y_TAB[a]) + 14, 515:565]
        m = red_strict(box)
        # the streak pulls the centroid up; the HEAD bottom is the
        # anchored measure: lowest red row ~ y + R_HEAD
        ys = np.where(m.any(1))[0]
        low = ys.max() + int(Y_TAB[a]) - 40 if len(ys) else -1
        devs.append(abs(low - (Y_TAB[a] + R_HEAD)))
        rys.append(low)
    ok("red head bottom on the rung, five transit frames",
       max(devs) <= 1.5, f"max dev {max(devs):.1f} px")
    ok("red drop falls monotonically",
       all(rys[i] < rys[i + 1] for i in range(4)), f"{rys}")
    ok("no red before birth / after exit / in act B",
       not red_strict(f2).any() and not red_strict(f170).any()
       and not red_strict(frame_at(250)).any())

    # ---- act B: the climb's period, raw-frame byte identity
    f210, f240, f270 = frame_at(210), frame_at(240), frame_at(270)
    ok("act B frames byte-identical 30 apart (210=240=270)",
       np.array_equal(f210, f240) and np.array_equal(f240, f270),
       "u-sets repeat exactly; same floats, same bytes")
    ok("act B mid-period identity too (225 vs 255)",
       np.array_equal(frame_at(225), frame_at(255)))
    ok("consecutive act B frames DIFFER (the ladder is moving)",
       not np.array_equal(f210, frame_at(211)))

    # ---- act B drops sit where the model says (fidelity of the climb)
    f250 = frame_at(250)
    dv = []
    for y, L, _ in drops_at(250):
        if y < 400 or y > 1780 or L < 3:
            continue
        box = f250[int(y) - 16:int(y) + 14, 515:565]
        ys = np.where(ink_mask(box).any(1))[0]
        if len(ys):
            dv.append(abs(ys.max() + int(y) - 16 - (y + R_HEAD)))
    ok("act B heads on their model rows", len(dv) >= 8 and max(dv) <= 1.5,
       f"{len(dv)} drops, max dev {max(dv):.2f} px")

    # ---- labels: the only thing that changes at the switch
    f179, f181 = frame_at(179), frame_at(181)
    lbox = (slice(225, 275), slice(40, 480))
    ok("rate label changes at the switch",
       not np.array_equal(f179[lbox], f181[lbox]))
    ok("label reads 30 all act A, 29 all act B",
       np.array_equal(f2[lbox], f179[lbox])
       and np.array_equal(f181[lbox], f250[lbox]))

    # ---- fences: identity crop holds only nozzle + column
    crop = (slice(CY0, CY0 + CH), slice(CX0, CX0 + CW))
    left = ink_mask(f100[CY0:CY0 + CH, CX0:511])
    right = ink_mask(f100[CY0:CY0 + CH, 570:CX0 + CW])
    ok("identity crop clear outside the column (frame 100)",
       not left.any() and not right.any())

    # ---- safe areas (trap 3): text never in top 192 / bottom 288
    top = ink_mask(f100[:192])
    xs = np.where(top.any(0))[0]
    ok("top-safe: only the nozzle above y=192",
       len(xs) == 0 or (xs.min() >= 500 and xs.max() <= 580),
       f"ink cols {xs.min()}..{xs.max()}" if len(xs) else "none")
    bot = ink_mask(f100[H - 288:])
    xs = np.where(bot.any(0))[0]
    ok("bottom-safe: only the drop column below y=1632",
       len(xs) == 0 or (xs.min() >= 520 and xs.max() <= 560),
       f"ink cols {xs.min()}..{xs.max()}" if len(xs) else "none")

    print()
    print("NOT verified by these checks (trap 68):")
    print("  - the CLIMB is an inference the viewer's eye makes by")
    print("    pairing nearest drops across frames; the render proves")
    print("    the -1/29-frame identity and the 30-frame period, not")
    print("    the percept")
    print("  - gravity is schematic (4 px/frame^2, not 9.8 m/s^2);")
    print("    declared in the description; the 1:3:5:7 law is")
    print("    scale-free and survives the substitution")
    print("  - h264 byte-identity is checked AFTER encode, on the")
    print("    decoded mod-16 crop (near-identity, trap 73)")
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
    sz = os.path.getsize(OUT_MP4)
    print(f"encoded {OUT_MP4} ({sz} bytes)", flush=True)


def decode_frame(n, crop=None):
    vf = f"select=eq(n\\,{n})"
    if crop:
        vf += f",crop={crop[2]}:{crop[3]}:{crop[0]}:{crop[1]}"
    r = subprocess.run(
        ["ffmpeg", "-i", OUT_MP4, "-vf", vf, "-vframes", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True)
    w = crop[2] if crop else W
    h = crop[3] if crop else H
    return np.frombuffer(r.stdout, np.uint8).reshape(h, w, 3)


def check_encode():
    print("== encode checks ==", flush=True)
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-count_frames", "-select_streams",
         "v", "-show_entries",
         "stream=nb_read_frames,width,height,r_frame_rate",
         "-of", "csv=p=0", OUT_MP4], capture_output=True, text=True)
    print("ffprobe:", r.stdout.strip(), flush=True)
    ok("330 frames in the file", f"{N}" in r.stdout)

    cr = (CX0, CY0, CW, CH)
    a, b = decode_frame(2, cr), decode_frame(170, cr)
    d1 = np.abs(a.astype(np.int64) - b.astype(np.int64)).mean()
    ok("frozen act survives h264: PRE vs hold near-identical",
       d1 < 0.5, f"mean |diff| {d1:.3f} grey")
    c, d, e = decode_frame(210, cr), decode_frame(240, cr), \
        decode_frame(270, cr)
    d2 = max(np.abs(c.astype(np.int64) - d.astype(np.int64)).mean(),
             np.abs(d.astype(np.int64) - e.astype(np.int64)).mean())
    ok("climb period survives h264: 210 vs 240 vs 270", d2 < 0.5,
       f"max mean |diff| {d2:.3f} grey")
    dn = np.abs(c.astype(np.int64)
                - decode_frame(220, cr).astype(np.int64)).mean()
    ok("mid-period frames genuinely differ on the file", dn > 1.0,
       f"mean |diff| {dn:.3f} grey")

    fr = decode_frame(23)
    m = red_strict(fr[:, 500:580])
    ok("red drop survives the encode", m.sum() > 40,
       f"{m.sum()} red px at frame 23")
    f60 = decode_frame(60)
    hit = sum(bool(ink_mask(
        f60[int(Y_TAB[a]) - 3:int(Y_TAB[a]) + 4, 528:552]).sum())
        for a in range(4, A_MAX))
    ok("ladder rungs survive the encode", hit == 25, f"{hit}/25")
    # same spine exclusion as the render-side check (fix both copies)
    tick = ink_mask(decode_frame(100)[:, 636:645])
    rows = np.where(tick.any(1))[0]
    groups = np.split(rows, np.where(np.diff(rows) > 4)[0] + 1)
    ok("five bracket ticks on the shipped file", len(groups) == 5)
    print()
    if CHECKS["fail"]:
        print(f"{CHECKS['fail']} FAILURES (incl. render)")
        sys.exit(1)
    print("ENCODE CHECKS PASSED — DONE", flush=True)


def review_stills():
    for n in (2, 23, 100, 170, 215, 300):
        Image.fromarray(frame_at(n)).save(f"{OUT_DIR}/drip_f{n:03d}.png")
    subprocess.run(
        ["ffmpeg", "-y", "-pattern_type", "glob",
         "-i", f"{OUT_DIR}/drip_f*.png",
         "-filter_complex", "scale=270:-1,tile=3x2",
         f"{OUT_DIR}/drip_sheet.png"],
        capture_output=True)
    print("sheet:", f"{OUT_DIR}/drip_sheet.png", flush=True)


if __name__ == "__main__":
    run_checks()
    encode()
    check_encode()
    review_stills()
