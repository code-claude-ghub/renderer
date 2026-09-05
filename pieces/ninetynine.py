#!/usr/bin/env python3
"""NINETY-NINE — the word is a hidden equation.

99 dots are on screen for the whole piece — never more, never
fewer. Four languages' words for 99 appear in turn, and the dots
regroup into the arithmetic each word literally encodes:

  ENGLISH  ninety-nine                        9x10 + 9
  FRENCH   quatre-vingt-dix-neuf              4x20 + 10 + 9
  DANISH   nioghalvfems                       9 + (4 1/2)x20
           (halvfems < halvfemsindstyve = halvfemte "half fifth"
            = 4 1/2, x tyve "twenty" — the missing half-twenty is
            drawn as ten violet ghost slots)
  WELSH    pedwar ar bymtheg a phedwar ugain  (4+15) + 4x20

All word forms and etymologies verified live against Wiktionary /
Wikipedia this wake (quoted in the description); all layout, text
and palette claims proven in scripts/feas_ninetynine.py (65
checks) before this file was written.

The idea — wordplay pieces, "you're a language model, do words" —
is Cassius's, the operator. Credited in the description.

Acts (30 fps, N=420, 14.0 s, silent):
  INTRO  n 0..44     99 white dots in a 9x11 grid
  EN     n 45..116   regroup to 9 tens + 9
  FR     n 117..188  regroup to 4 twenties + a ten + 9
  DA     n 189..260  regroup to 9 + 4 twenties + HALF a twenty
  CY     n 261..332  regroup to 4 + a fifteen + 4 twenties
  FIN    n 333..419  back to the grid; the four equations
  (morph = first 18 frames of each act; FREEZE from 401)
"""
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
RD = 13.0
PITCH = 38.0
MORPH = 18
N = 420
FREEZE = 401
STARTS = [0, 45, 117, 189, 261, 333]
SEQ = ["INTRO", "EN", "FR", "DA", "CY", "FIN"]
SETTLE = {"INTRO": 30, "EN": 105, "FR": 177, "DA": 249, "CY": 321,
          "FIN": 410}
REG_X = (70.0, 1010.0)
REG_Y = (530.0, 1435.0)

BGC = (0.063, 0.067, 0.086)
INK = {
    "WHITE": (0.92, 0.925, 0.94),
    "CYAN": (0.227, 0.714, 0.902),
    "WARM": (0.922, 0.722, 0.20),
    "GREEN": (0.298, 0.804, 0.361),
    "RED": (0.89, 0.165, 0.118),
    "GHOST": (0.349, 0.310, 0.529),   # violet: feas fence reasoning
    "GREY": (0.588, 0.608, 0.647),
}
LW_GHOST = 4.5   # 2.5 shattered under 4:2:0 chroma (run-1 diag:
                 # 13-16 fragments of 2-23 px per ring on the decoded
                 # file) and was ~0.8 px at watch size (trap 67)

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
OUT_DIR = ("/home/maroon-beret/projects/active/youtube/"
           "youtube-channel/out")
STAMP = time.strftime("%H%M%S")
OUT_MP4 = f"{OUT_DIR}/ninetynine_{STAMP}.mp4"


# ---------------------------------------------------------------- layouts
def block(cx, cy, ncols, nrows, pitch=PITCH):
    xs = cx + (np.arange(ncols) - (ncols - 1) / 2.0) * pitch
    ys = cy + (np.arange(nrows) - (nrows - 1) / 2.0) * pitch
    return [(float(x), float(y)) for y in ys for x in xs]


def row(cy, xs):
    return [(float(x), float(cy)) for x in xs]


def lay_intro():
    return [(p, "WHITE") for p in block(540, 980, 9, 11, 64)]


def lay_en():
    pts = []
    for by in (660, 900, 1140):
        for bx in (300, 540, 780):
            pts += [(p, "WARM") for p in block(bx, by, 2, 5)]
    pts += [(p, "RED") for p in row(1330, 220 + 80 * np.arange(9))]
    return pts


def lay_fr():
    pts = []
    for by in (660, 960):
        for bx in (330, 750):
            pts += [(p, "CYAN") for p in block(bx, by, 4, 5)]
    pts += [(p, "WARM") for p in block(270, 1260, 2, 5)]
    pts += [(p, "RED") for p in row(1260, 430 + 60 * np.arange(9))]
    return pts


def lay_da():
    pts = [(p, "RED") for p in row(590, 220 + 80 * np.arange(9))]
    for by in (800, 1100):
        for bx in (330, 750):
            pts += [(p, "CYAN") for p in block(bx, by, 4, 5)]
    pts += [(p, "CYAN") for p in block(540, 1330, 4, 5)
            if p[0] < 540.0]
    return pts


DA_GHOSTS = [p for p in block(540, 1330, 4, 5) if p[0] > 540.0]


def lay_cy():
    pts = [(p, "RED") for p in row(580, 390 + 100 * np.arange(4))]
    pts += [(p, "GREEN") for p in block(540, 810, 3, 5)]
    for bx in (195, 425, 655, 885):
        pts += [(p, "CYAN") for p in block(bx, 1150, 4, 5)]
    return pts


def sorted_lay(lay):
    """(99,2) positions + (99,3) colours, sorted by (y, x) so the
    morph pairing is deterministic and roughly local."""
    s = sorted(lay, key=lambda pc: (pc[0][1], pc[0][0]))
    pos = np.asarray([p for p, _ in s], np.float64)
    col = np.asarray([INK[c] for _, c in s], np.float64)
    return pos, col


LAYS = [sorted_lay(f()) for f in
        (lay_intro, lay_en, lay_fr, lay_da, lay_cy, lay_intro)]
for _pos, _ in LAYS:
    assert _pos.shape == (99, 2)


# ---------------------------------------------------------------- prims
def comp_bbox(img, x0, y0, cov, color):
    h, w = cov.shape
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x0 + w, W), min(y0 + h, H)
    if x1c <= x0c or y1c <= y0c:
        return
    cv = cov[y0c - y0:y1c - y0, x0c - x0:x1c - x0]
    reg = img[y0c:y1c, x0c:x1c, :]
    col = np.asarray(color, np.float64)
    reg[...] = reg * (1 - cv[..., None]) + col[None, None, :] * cv[..., None]


def disc_cov(cx_, cy_, r):
    x0, x1 = int(np.floor(cx_ - r)) - 2, int(np.ceil(cx_ + r)) + 3
    y0, y1 = int(np.floor(cy_ - r)) - 2, int(np.ceil(cy_ + r)) + 3
    xx = np.arange(x0, x1, dtype=np.float64)
    yy = np.arange(y0, y1, dtype=np.float64)
    d = np.hypot(xx[None, :] - cx_, yy[:, None] - cy_)
    return x0, y0, np.clip(r + 0.5 - d, 0.0, 1.0)


def ring_cov(cx_, cy_, r, lw):
    pad = lw / 2 + 2
    x0, x1 = int(np.floor(cx_ - r - pad)), int(np.ceil(cx_ + r + pad)) + 1
    y0, y1 = int(np.floor(cy_ - r - pad)), int(np.ceil(cy_ + r + pad)) + 1
    xx = np.arange(x0, x1, dtype=np.float64)
    yy = np.arange(y0, y1, dtype=np.float64)
    d = np.hypot(xx[None, :] - cx_, yy[:, None] - cy_)
    return x0, y0, np.clip(lw / 2 + 0.5 - np.abs(d - r), 0.0, 1.0)


def text_cov(s, px):
    f = ImageFont.truetype(FONT, px * 4)
    im = Image.new("L", (px * (len(s) + 2) * 4, px * 8), 0)
    ImageDraw.Draw(im).text((px * 4, px * 2), s, font=f, fill=255)
    a = np.asarray(im, np.float64) / 255.0
    ys, xs = np.where(a > 0)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h4, w4 = a.shape
    h4 -= h4 % 4
    w4 -= w4 % 4
    a = a[:h4, :w4].reshape(h4 // 4, 4, w4 // 4, 4).mean((1, 3))
    return a


def text_cov_band(s, px):
    """x-cropped, y in a FIXED band so equation segments rendered
    separately share a baseline when stamped at one y0."""
    f = ImageFont.truetype(FONT, px * 4)
    im = Image.new("L", (px * (len(s) + 2) * 4, px * 8), 0)
    ImageDraw.Draw(im).text((px * 4, px * 2), s, font=f, fill=255)
    a = np.asarray(im, np.float64) / 255.0
    xs = np.where(a.any(0))[0]
    a = a[:, xs.min():xs.max() + 1]
    h4 = a.shape[0] - a.shape[0] % 4
    w4 = a.shape[1] - a.shape[1] % 4
    a = a[:h4, :w4].reshape(h4 // 4, 4, w4 // 4, 4).mean((1, 3))
    return a


def fit_word(s, px0=60, wmax=980):
    px = px0
    while px > 20:
        f = ImageFont.truetype(FONT, px * 4)
        im = Image.new("L", (px * (len(s) + 2) * 4, px * 8), 0)
        ImageDraw.Draw(im).text((px * 4, px * 2), s, font=f, fill=255)
        xs = np.where(np.asarray(im).any(0))[0]
        if (xs.max() - xs.min() + 1) / 4.0 <= wmax:
            break
        px -= 2
    return px


# ---------------------------------------------------------------- text
WORDS = {
    "INTRO": ("", "99 dots", "four languages will now count them"),
    "EN": ("ENGLISH", "ninety-nine", "nine tens and nine"),
    "FR": ("FRENCH", "quatre-vingt-dix-neuf",
           "four twenties, a ten, a nine"),
    "DA": ("DANISH", "nioghalvfems", "nine and half-fifth twenties"),
    "CY": ("WELSH", "pedwar ar bymtheg a phedwar ugain",
           "four on fifteen and four twenties"),
    "FIN": ("", "same 99 dots", "every word was already doing the math"),
}
EQ = {
    "EN": [("9×10", "WARM"), ("+", "GREY"), ("9", "RED"),
           ("= 99", "WHITE")],
    "FR": [("4×20", "CYAN"), ("+", "GREY"), ("10", "WARM"),
           ("+", "GREY"), ("9", "RED"), ("= 99", "WHITE")],
    "DA": [("9", "RED"), ("+", "GREY"), ("4½×20", "CYAN"),
           ("= 99", "WHITE")],
    "CY": [("4", "RED"), ("+", "GREY"), ("15", "GREEN"),
           ("+", "GREY"), ("4×20", "CYAN"), ("= 99", "WHITE")],
}
FIN_EQ = "9×10+9 · 4×20+10+9 · 9+4½×20 · 4+15+4×20"
FIN_ALL = "all of them are 99"

PX_NAME, PX_GLOSS, PX_EQ, PX_FINEQ, PX_FIN2 = 26, 30, 44, 34, 38
Y_NAME, Y_WORD, Y_GLOSS = 230, 278, 388
Y0_EQ, Y_FINEQ, Y_FIN2 = 1455, 1462, 1548
GAPX = 18

LBL = {}
for k, (nm, word, gloss) in WORDS.items():
    LBL[k] = {
        "name": text_cov(nm, PX_NAME) if nm else None,
        "word": text_cov(word, fit_word(word)),
        "gloss": text_cov(gloss, PX_GLOSS),
    }
EQSEG = {}
for k, segs in EQ.items():
    covs = [(text_cov_band(s, PX_EQ), INK[c]) for s, c in segs]
    tot = sum(c.shape[1] for c, _ in covs) + GAPX * (len(covs) - 1)
    x = (W - tot) // 2
    out = []
    for c, col in covs:
        out.append((x, c, col))
        x += c.shape[1] + GAPX
    EQSEG[k] = out
FINEQ_COV = text_cov(FIN_EQ, PX_FINEQ)
FIN2_COV = text_cov(FIN_ALL, PX_FIN2)
assert FINEQ_COV.shape[1] <= 980 and FIN2_COV.shape[1] <= 980

BG = np.empty((H, W, 3), np.float64)
BG[..., 0], BG[..., 1], BG[..., 2] = BGC


# ---------------------------------------------------------------- frames
def fade(n, n0, span=10):
    return float(np.clip((n - n0 + 1) / span, 0.0, 1.0))


def act_of(n):
    k = max(i for i, s in enumerate(STARTS) if n >= s)
    return k, n - STARTS[k]


def dots_at(n):
    """(99,2) positions and (99,3) colours at frame n."""
    k, ph = act_of(n)
    pos1, col1 = LAYS[k]
    if k == 0 or ph >= MORPH:
        return pos1, col1
    pos0, col0 = LAYS[k - 1]
    u = ph / MORPH
    e = u * u * (3.0 - 2.0 * u)
    return pos0 + (pos1 - pos0) * e, col0 + (col1 - col0) * e


def ghost_alpha(n):
    k, ph = act_of(n)
    if SEQ[k] == "DA":
        return fade(n, STARTS[k] + MORPH, 10)
    if SEQ[k] == "CY":
        return max(0.0, 1.0 - ph / 8.0)
    return 0.0


def draw_labels(img, k, alpha):
    """Stamp act k's labels at the given master alpha."""
    if alpha <= 0.0:
        return
    key = SEQ[k]
    L = LBL[key]
    if L["name"] is not None:
        cv = L["name"] * 0.85 * alpha
        comp_bbox(img, (W - L["name"].shape[1]) // 2, Y_NAME, cv,
                  INK["GREY"])
    cv = L["word"] * alpha
    comp_bbox(img, (W - L["word"].shape[1]) // 2, Y_WORD, cv,
              INK["WHITE"])
    cv = L["gloss"] * 0.9 * alpha
    comp_bbox(img, (W - L["gloss"].shape[1]) // 2, Y_GLOSS, cv,
              INK["GREY"])
    if key in EQSEG:
        for x0, c, col in EQSEG[key]:
            comp_bbox(img, x0, Y0_EQ, c * alpha, col)
    if key == "FIN":
        cv = FINEQ_COV * 0.9 * alpha
        comp_bbox(img, (W - FINEQ_COV.shape[1]) // 2, Y_FINEQ, cv,
                  INK["GREY"])
        cv = FIN2_COV * alpha
        comp_bbox(img, (W - FIN2_COV.shape[1]) // 2, Y_FIN2, cv,
                  INK["WHITE"])


def label_alphas(n):
    """[(act, alpha)] — the current act fades in, the previous
    fades out over the first 8 morph frames."""
    k, ph = act_of(n)
    out = []
    if k > 0 and ph < 8:
        out.append((k - 1, 1.0 - ph / 8.0))
    a_in = fade(n, STARTS[k] + (6 if k else 2), 8)
    out.append((k, a_in))
    return out


def frame_at(n):
    img = BG.copy()
    ga = ghost_alpha(n)
    if ga > 0.0:
        for gx, gy in DA_GHOSTS:
            x0, y0, cv = ring_cov(gx, gy, RD, LW_GHOST)
            comp_bbox(img, x0, y0, cv * ga, INK["GHOST"])
    pos, col = dots_at(n)
    for i in range(99):
        x0, y0, cv = disc_cov(pos[i, 0], pos[i, 1], RD)
        comp_bbox(img, x0, y0, cv, col[i])
    for k, a in label_alphas(n):
        draw_labels(img, k, a)
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def render_frames():
    for n in range(N):
        yield frame_at(n)


# ---------------------------------------------------------------- checks
CHECKS = {"pass": 0, "fail": 0}


def ok(name, cond, detail=""):
    s = "ok  " if cond else "FAIL"
    CHECKS["pass" if cond else "fail"] += 1
    print(f"{s} {name}" + (f" [{detail}]" if detail else ""), flush=True)


def m_white(fr):
    return fr.min(2) >= 200


def m_cyan(fr):
    r = fr[:, :, 0].astype(np.int64)
    g = fr[:, :, 1].astype(np.int64)
    b = fr[:, :, 2].astype(np.int64)
    return (b > 150) & (b - r > 60) & (g > 120)


def m_warm(fr):
    return (fr[:, :, 0] > 180) & (fr[:, :, 1] > 130) & (fr[:, :, 2] < 100)


def m_green(fr):
    r = fr[:, :, 0].astype(np.int64)
    g = fr[:, :, 1].astype(np.int64)
    b = fr[:, :, 2].astype(np.int64)
    return (g > 170) & (g - r > 60) & (g - b > 60)


def m_red(fr):
    r = fr[:, :, 0].astype(np.int64)
    g = fr[:, :, 1].astype(np.int64)
    return (r > 190) & (r - g > 60)


def m_ghost(fr):
    r = fr[:, :, 0].astype(np.int64)
    g = fr[:, :, 1].astype(np.int64)
    b = fr[:, :, 2].astype(np.int64)
    return (b >= 100) & (b <= 160) & (b - g >= 45) & (b - r >= 35)


MASKS = {"WHITE": m_white, "CYAN": m_cyan, "WARM": m_warm,
         "GREEN": m_green, "RED": m_red, "GHOST": m_ghost}
EXPECT = {
    "INTRO": {"WHITE": 99},
    "EN": {"WARM": 90, "RED": 9},
    "FR": {"CYAN": 80, "WARM": 10, "RED": 9},
    "DA": {"CYAN": 90, "RED": 9, "GHOST": 10},
    "CY": {"CYAN": 80, "GREEN": 15, "RED": 4},
    "FIN": {"WHITE": 99},
}
DOTBOX = (slice(510, 1450), slice(50, 1030))   # dot region + margin


def count_blobs(mask, min_px=30):
    """4-connected component count via BFS — no scipy dependency.
    min_px floors out decode strays (trap 78: <= ~30 px; a dot is
    ~531 px, a ghost ring ~200+)."""
    m = mask.copy()
    hh, ww = m.shape
    cnt = 0
    ys, xs = np.where(m)
    todo = list(zip(ys.tolist(), xs.tolist()))
    for sy, sx in todo:
        if not m[sy, sx]:
            continue
        stack = [(sy, sx)]
        m[sy, sx] = False
        size = 0
        while stack:
            y, x = stack.pop()
            size += 1
            if y > 0 and m[y - 1, x]:
                m[y - 1, x] = False
                stack.append((y - 1, x))
            if y < hh - 1 and m[y + 1, x]:
                m[y + 1, x] = False
                stack.append((y + 1, x))
            if x > 0 and m[y, x - 1]:
                m[y, x - 1] = False
                stack.append((y, x - 1))
            if x < ww - 1 and m[y, x + 1]:
                m[y, x + 1] = False
                stack.append((y, x + 1))
        if size >= min_px:
            cnt += 1
    return cnt


def blob_report(fr, act, min_px=30):
    """{fence: count} over the dot region for the act's fences."""
    reg = fr[DOTBOX[0], DOTBOX[1], :]
    return {f: count_blobs(MASKS[f](reg), min_px)
            for f in EXPECT[act]}


def run_checks():
    print("== render checks ==", flush=True)
    bg8 = tuple(np.uint8(np.clip(v, 0, 1) * 255.0 + 0.5) for v in BGC)
    frames = {a: frame_at(n) for a, n in SETTLE.items()}

    f30 = frames["INTRO"]
    ok("corner pixel is background", tuple(f30[4, 4]) == bg8,
       f"{tuple(f30[4, 4])} vs {bg8}")
    frac = (frames["DA"].max(2) > 40).mean()
    ok("lit fraction sane (not blank, not floodlit)",
       0.01 < frac < 0.30, f"{frac:.4f}")

    # THE CLAIM: exactly 99 dots, in the groups the word names,
    # counted from pixels in every settled act
    for act in SEQ:
        rep = blob_report(frames[act], act)
        want = EXPECT[act]
        ndots = sum(v for k, v in rep.items() if k != "GHOST")
        ok(f"{act}: blob counts {rep} == expected {want}",
           rep == want)
        if act != "DA" or True:
            ok(f"{act}: dot total from pixels == 99", ndots == 99,
               f"{ndots}")

    # colour exclusivity: fences that should be EMPTY per act
    for act, absent in (("EN", ("CYAN", "GREEN", "WHITE")),
                        ("FR", ("GREEN", "WHITE", "GHOST")),
                        ("DA", ("WARM", "GREEN", "WHITE")),
                        ("CY", ("WARM", "WHITE", "GHOST")),
                        ("FIN", ("CYAN", "WARM", "GREEN", "RED",
                                 "GHOST"))):
        reg = frames[act][DOTBOX[0], DOTBOX[1], :]
        leak = {f: int(MASKS[f](reg).sum()) for f in absent
                if MASKS[f](reg).sum() > 0}
        ok(f"{act}: absent fences empty in dot region", not leak,
           f"{leak}")

    # every blob sits ON a model dot centre (positions, not just count)
    for act in ("EN", "DA", "CY"):
        k = SEQ.index(act)
        pos, col = LAYS[k]
        fr = frames[act]
        worst = 0.0
        for i in range(99):
            x, y = int(pos[i, 0]), int(pos[i, 1])
            box = fr[y - 6:y + 7, x - 6:x + 7]
            lit = (box.max(2) > 100).mean()
            worst = max(worst, 1.0 - lit)
        ok(f"{act}: ink present at all 99 model centres",
           worst < 0.5, f"worst miss {worst:.2f}")

    # ghosts: DA only, at the model ghost slots
    fr = frames["DA"]
    on = all(m_ghost(fr[int(gy) - 16:int(gy) + 17,
                        int(gx) - 16:int(gx) + 17]).any()
             for gx, gy in DA_GHOSTS)
    ok("DA: ghost ink at all 10 ghost slots", on)
    ok("CY: ghosts gone",
       not m_ghost(frames["CY"][DOTBOX[0], DOTBOX[1], :]).any())
    ok("FR: no ghosts before Danish",
       not m_ghost(frames["FR"][DOTBOX[0], DOTBOX[1], :]).any())

    # equations present, colour-matched (trap 61), bottom band
    band = (slice(1440, 1632), slice(0, W))
    for act in ("EN", "FR", "DA", "CY"):
        fr = frames[act]
        need = {c for _, c in EQ[act]} - {"GREY"}
        miss = [c for c in need if not MASKS[c](fr[band]).any()]
        ok(f"{act}: equation segments present in their colours",
           not miss, f"missing {miss}")
    ok("FIN: closing lines present (grey eqs + white)",
       m_white(frames["FIN"][band]).sum() > 200)

    # words present in the top band
    top = (slice(200, 470), slice(0, W))
    for act in SEQ:
        ok(f"{act}: word present (white, top band)",
           m_white(frames[act][top]).sum() > 300)

    # motion + freeze
    ok("dots move between acts (EN settle != FR settle)",
       not np.array_equal(frames["EN"], frames["FR"]))
    ok("morph midframe differs from both ends",
       not np.array_equal(frame_at(126), frames["EN"])
       and not np.array_equal(frame_at(126), frames["FR"]))
    ok("freeze frames byte-identical (405 == 415)",
       np.array_equal(frame_at(405), frame_at(415)))

    # safe areas (trap 3)
    ok("top 192 rows pure background, all settled frames",
       all((frames[a][:192] == np.asarray(bg8, np.uint8)).all()
           for a in SEQ))
    ok("rows >= 1632 pure background, all settled frames",
       all((frames[a][1632:] == np.asarray(bg8, np.uint8)).all()
           for a in SEQ))

    print()
    print("NOT verified by these checks (trap 68):")
    print("  - the LINGUISTIC facts (word forms, etymologies) are")
    print("    verified against the cited sources, not by pixels;")
    print("    the Welsh 99 form is COMPOSED from sourced parts")
    print("    (19, 80, and the sourced pattern for 91)")
    print("  - glosses are literal morpheme readings; nothing here")
    print("    claims speakers COMPUTE these when talking")
    print("  - blob counts use a 30 px floor (trap 78); the floor")
    print("    is 18x below a dot, 6x above a measured stray")
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
    print(f"encoded {OUT_MP4} ({os.path.getsize(OUT_MP4)} bytes)",
          flush=True)


def decode_frame(n):
    r = subprocess.run(
        ["ffmpeg", "-i", OUT_MP4, "-vf", f"select=eq(n\\,{n})",
         "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True)
    return np.frombuffer(r.stdout, np.uint8).reshape(H, W, 3)


def check_encode():
    print("== encode checks ==", flush=True)
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-count_frames", "-select_streams",
         "v", "-show_entries",
         "stream=nb_read_frames,width,height,r_frame_rate",
         "-of", "csv=p=0", OUT_MP4], capture_output=True, text=True)
    print("ffprobe:", r.stdout.strip(), flush=True)
    ok(f"{N} frames in the file", f"{N}" in r.stdout)

    # the piece's whole claim, counted back off the SHIPPED file
    # (trap 78: 100 px floor on the decoded side). GHOST is graded
    # separately: 4:2:0 chroma slices a thin dim ring into
    # sub-chroma-block fragments (run-1 diag: 13-16 pieces of 2-23
    # px per ring, all below any honest floor), so blob TOPOLOGY is
    # proven on the render buffer, and the decode proves the ink
    # SURVIVES at exactly the ten proven slots and nowhere else.
    for act in SEQ:
        d = decode_frame(SETTLE[act])
        want = {k: v for k, v in EXPECT[act].items() if k != "GHOST"}
        rep = blob_report(d, act, min_px=100)
        rep_dots = {k: v for k, v in rep.items() if k != "GHOST"}
        ok(f"{act}: decoded blob counts {rep_dots} == {want}",
           rep_dots == want)
        if "GHOST" in EXPECT[act]:
            reg = d[DOTBOX[0], DOTBOX[1], :]
            gm = m_ghost(reg)
            per = []
            inslots = np.zeros_like(gm)
            for gx, gy in DA_GHOSTS:
                sy = slice(int(gy) - 18 - DOTBOX[0].start,
                           int(gy) + 19 - DOTBOX[0].start)
                sx = slice(int(gx) - 18 - DOTBOX[1].start,
                           int(gx) + 19 - DOTBOX[1].start)
                per.append(int(gm[sy, sx].sum()))
                inslots[sy, sx] = True
            phantom = int((gm & ~inslots).sum())
            ok(f"{act}: ghost ink survives at all 10 slots (>=60 px)",
               min(per) >= 60, f"per-slot {per}")
            ok(f"{act}: no phantom ghost ink outside the slots",
               phantom < 30, f"{phantom} px")
    a, b = decode_frame(405), decode_frame(415)
    d1 = np.abs(a.astype(np.int64) - b.astype(np.int64)).mean()
    ok("freeze survives h264 (405 vs 415 near-identical)", d1 < 0.5,
       f"mean |diff| {d1:.3f} grey")
    d2 = np.abs(decode_frame(50).astype(np.int64)
                - decode_frame(60).astype(np.int64)).mean()
    ok("morph frames genuinely differ (50 vs 60)", d2 > 0.5,
       f"mean |diff| {d2:.3f} grey")
    print()
    if CHECKS["fail"]:
        print(f"{CHECKS['fail']} FAILURES (incl. render)")
        sys.exit(1)
    print("ENCODE CHECKS PASSED — DONE", flush=True)


def review_stills():
    for a in SEQ:
        Image.fromarray(frame_at(SETTLE[a])).save(
            f"{OUT_DIR}/ninetynine_f{SETTLE[a]:03d}.png")
    subprocess.run(
        ["ffmpeg", "-y", "-pattern_type", "glob",
         "-i", f"{OUT_DIR}/ninetynine_f*.png",
         "-filter_complex", "scale=270:-1,tile=3x2",
         f"{OUT_DIR}/ninetynine_sheet.png"],
        capture_output=True)
    print("sheet:", f"{OUT_DIR}/ninetynine_sheet.png", flush=True)


if __name__ == "__main__":
    run_checks()
    encode()
    check_encode()
    review_stills()
