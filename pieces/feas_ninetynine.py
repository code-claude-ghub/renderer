#!/usr/bin/env python3
"""Feasibility for NINETY-NINE — the word is a hidden equation.

99 dots stay on screen for the whole piece. Four languages' words
for 99 appear in turn, and the dots regroup into the arithmetic the
word literally encodes:

  ENGLISH  ninety-nine                        9x10 + 9
  FRENCH   quatre-vingt-dix-neuf              4x20 + 10 + 9
  DANISH   nioghalvfems                       9 + (4 1/2)x20
  WELSH    pedwar ar bymtheg a phedwar ugain  (4+15) + 4x20

Sources (verified live this wake, quoted in the description):
  - Wiktionary "quatre-vingt-dix-neuf": 99, quatre+vingt+dix+neuf
  - Wiktionary "nioghalvfems": 99; "halvfems": clipping of
    halvfemsindstyve; "halvfemsindstyve": "From halvfemte ('four
    and a half', literally 'half fifth') + sinde ('times') + tyve
    ('twenty')" -> 4.5 x 20 = 90
  - Wiktionary "pedwar ar bymtheg": 19, "From pedwar ('four') + ar
    ('on') + pymtheg ('fifteen')"; Wikipedia "Welsh numerals":
    80 = pedwar ugain, 91 = un ar ddeg a phedwar ugain (pattern);
    the full 99 form is COMPOSED from those sourced parts (said
    plainly in the description)
  - Wiktionary "ninety": equivalent to nine + -ty; "-ty": "suffix
    indicating single-digit integer multiples of ten", OE -tig,
    P-Gmc *teguz "group of ten"
  - Wikipedia "Vigesimal": quatre-vingt-dix = "four-twenties-ten"

This script proves, before any rendering:
  - the title fits, every equation is exactly 99
  - every act's layout has exactly 99 dot centres, inside the dot
    region, non-overlapping (and Danish's 10 ghost slots besides)
  - every label fits the frame at its chosen size, in safe areas
  - the palette: each ink lands in exactly its own fence AS
    COMPOSITED (full alpha over BG — trap 77), and every AA blend
    of ink with BG lands in NO fence (trap 78's span floor covers
    the decoded side)
  - act frame ranges tile N exactly
"""
import math
import sys
from fractions import Fraction

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
FPS = 30
RD = 13.0                      # dot radius
PITCH = 38.0                   # dot pitch inside a block
MORPH = 18

# act frame plan
INTRO_N = 45
ACT_N = 72
FIN_LO = INTRO_N + 4 * ACT_N   # 333
N = 420
ACT_STARTS = {"EN": 45, "FR": 117, "DA": 189, "CY": 261, "FIN": FIN_LO}
SETTLE = {k: v + 60 for k, v in ACT_STARTS.items() if k != "FIN"}
SETTLE["INTRO"] = 30
SETTLE["FIN"] = 410
FREEZE = 401

# dot region (centres); text lives outside it
REG_X = (70.0, 1010.0)
REG_Y = (530.0, 1435.0)

TITLE = ("the word for 99 is a hidden equation. English: 9×10+9. "
         "French: 4×20+10+9. Danish: 9+4½×20.")

BGC = (0.063, 0.067, 0.086)
INK = {
    "WHITE": (0.92, 0.925, 0.94),
    "CYAN": (0.227, 0.714, 0.902),   # twenties
    "WARM": (0.922, 0.722, 0.20),    # tens
    "GREEN": (0.298, 0.804, 0.361),  # the fifteen
    "RED": (0.89, 0.165, 0.118),     # the singles
    # Danish missing half-slots: dim VIOLET, because every NEUTRAL
    # dim colour sits on the BG-fade path of white and grey ink; the
    # b-g channel gap is the fence no other ink's AA blend crosses
    # (cyan's fade peaks at b-g=36 inside the b band; audit below)
    "GHOST": (0.349, 0.310, 0.529),
    "GREY": (0.588, 0.608, 0.647),   # gloss text
}


def to8(c):
    return tuple(int(v * 255.0 + 0.5) for v in c)


# fences (uint8 space) — each names the ONE thing it measures
def fence_white(p):
    return min(p) >= 200


def fence_cyan(p):
    r, g, b = p
    return b > 150 and b - r > 60 and g > 120


def fence_warm(p):
    r, g, b = p
    return r > 180 and g > 130 and b < 100


def fence_green(p):
    r, g, b = p
    return g > 170 and g - r > 60 and g - b > 60


def fence_red(p):
    r, g, b = p
    return r > 190 and r - g > 60


def fence_ghost(p):
    r, g, b = p
    return 100 <= b <= 160 and b - g >= 45 and b - r >= 35


FENCES = {"WHITE": fence_white, "CYAN": fence_cyan, "WARM": fence_warm,
          "GREEN": fence_green, "RED": fence_red, "GHOST": fence_ghost}
FENCE_OF = {"WHITE": "WHITE", "CYAN": "CYAN", "WARM": "WARM",
            "GREEN": "GREEN", "RED": "RED", "GHOST": "GHOST",
            "GREY": None}   # gloss text: measured by no fence at all


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
    half = block(540, 1330, 4, 5)
    real = [p for p in half if p[0] < 540.0]
    pts += [(p, "CYAN") for p in real]
    return pts


def da_ghosts():
    half = block(540, 1330, 4, 5)
    return [p for p in half if p[0] > 540.0]


def lay_cy():
    pts = [(p, "RED") for p in row(580, 390 + 100 * np.arange(4))]
    pts += [(p, "GREEN") for p in block(540, 810, 3, 5)]
    for bx in (195, 425, 655, 885):
        pts += [(p, "CYAN") for p in block(bx, 1150, 4, 5)]
    return pts


LAYOUTS = {"INTRO": lay_intro(), "EN": lay_en(), "FR": lay_fr(),
           "DA": lay_da(), "CY": lay_cy(), "FIN": lay_intro()}

EXPECT = {                       # per-act blob counts by fence
    "INTRO": {"WHITE": 99},
    "EN": {"WARM": 90, "RED": 9},
    "FR": {"CYAN": 80, "WARM": 10, "RED": 9},
    "DA": {"CYAN": 90, "RED": 9, "GHOST": 10},
    "CY": {"CYAN": 80, "GREEN": 15, "RED": 4},
    "FIN": {"WHITE": 99},
}

# ---------------------------------------------------------------- text
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

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
    "EN": [("9×10", "WARM"), (" + ", "GREY"), ("9", "RED"),
           (" = 99", "WHITE")],
    "FR": [("4×20", "CYAN"), (" + ", "GREY"), ("10", "WARM"),
           (" + ", "GREY"), ("9", "RED"), (" = 99", "WHITE")],
    "DA": [("9", "RED"), (" + ", "GREY"), ("4½×20", "CYAN"),
           (" = 99", "WHITE")],
    "CY": [("4", "RED"), (" + ", "GREY"), ("15", "GREEN"),
           (" + ", "GREY"), ("4×20", "CYAN"), (" = 99", "WHITE")],
}
FIN_EQ = "9×10+9 · 4×20+10+9 · 9+4½×20 · 4+15+4×20"

PX_NAME, PX_GLOSS, PX_EQ, PX_FINEQ = 26, 30, 44, 34
Y_NAME, Y_WORD, Y_GLOSS = 230, 278, 388
Y_EQ, Y_FINEQ, Y_FIN2 = 1480, 1470, 1540


def text_wh(s, px):
    f = ImageFont.truetype(FONT, px * 4)
    im = Image.new("L", (max(px * len(s) * 4, 64), px * 8), 0)
    ImageDraw.Draw(im).text((8, 8), s, font=f, fill=255)
    a = np.asarray(im, np.float64)
    ys, xs = np.where(a > 0)
    return ((xs.max() - xs.min() + 1) / 4.0,
            (ys.max() - ys.min() + 1) / 4.0)


def fit_word(s, px0=60, wmax=980):
    px = px0
    while px > 20 and text_wh(s, px)[0] > wmax:
        px -= 2
    return px


# ---------------------------------------------------------------- checks
PASS = FAIL = 0


def ok(name, cond, detail=""):
    global PASS, FAIL
    s = "ok  " if cond else "FAIL"
    PASS, FAIL = PASS + cond, FAIL + (not cond)
    print(f"{s} {name}" + (f" [{detail}]" if detail else ""), flush=True)


def main():
    # -- title
    ok("title <= 100 chars", len(TITLE) <= 100, f"{len(TITLE)}")

    # -- the four equations, exactly (Fraction: no float in the claim)
    ok("English  9*10+9 == 99", 9 * 10 + 9 == 99)
    ok("French   4*20+10+9 == 99", 4 * 20 + 10 + 9 == 99)
    ok("Danish   9 + (9/2)*20 == 99",
       9 + Fraction(9, 2) * 20 == 99, "half-fifth = 4 1/2 = 9/2")
    ok("Danish   halvfemte*20 == 90 (the sourced etymology)",
       Fraction(9, 2) * 20 == 90)
    ok("Welsh    (4+15) + 4*20 == 99", (4 + 15) + 4 * 20 == 99)

    # -- layouts
    for name, lay in LAYOUTS.items():
        pts = [p for p, _ in lay]
        ok(f"{name}: exactly 99 dots", len(pts) == 99, f"{len(pts)}")
        inx = all(REG_X[0] <= x <= REG_X[1] for x, _ in pts)
        iny = all(REG_Y[0] <= y <= REG_Y[1] for _, y in pts)
        ok(f"{name}: all centres inside the dot region", inx and iny)
        a = np.asarray(pts)
        d = np.hypot(a[:, None, 0] - a[None, :, 0],
                     a[:, None, 1] - a[None, :, 1])
        np.fill_diagonal(d, 1e9)
        mind = float(d.min())
        ok(f"{name}: min centre distance >= 2r+8",
           mind >= 2 * RD + 8, f"{mind:.1f} px")

    gh = da_ghosts()
    ok("DA: exactly 10 ghost slots", len(gh) == 10, f"{len(gh)}")
    a = np.asarray([p for p, _ in LAYOUTS["DA"]])
    g = np.asarray(gh)
    dg = np.hypot(a[:, None, 0] - g[None, :, 0],
                  a[:, None, 1] - g[None, :, 1]).min()
    ok("DA: ghosts clear of real dots", float(dg) >= 2 * RD + 8,
       f"{float(dg):.1f} px")
    ok("DA: ghosts + dots == a 4x5 block x centred slots",
       len(gh) + sum(1 for p, c in LAYOUTS["DA"]
                     if c == "CYAN" and abs(p[1] - 1330) < 80) == 20)

    # -- expected blob counts sum to 99 per act
    for name, exp in EXPECT.items():
        tot = sum(v for k, v in exp.items() if k != "GHOST")
        ok(f"{name}: fence counts sum to 99", tot == 99, f"{tot}")

    # -- act tiling
    ok("acts tile N exactly",
       INTRO_N + 4 * ACT_N + (N - FIN_LO) == N
       and FIN_LO == 333 and N == 420, f"N={N} = {N/FPS:.1f}s")
    ok("settle frames sit after morph+fades, inside their acts",
       all(ACT_STARTS[k] + MORPH + 20 <= SETTLE[k]
           < ACT_STARTS[k] + ACT_N for k in ("EN", "FR", "DA", "CY"))
       and SETTLE["INTRO"] < INTRO_N and FREEZE < SETTLE["FIN"] < N)

    # -- text fits + safe areas
    for name, (nm, word, gloss) in WORDS.items():
        pw = fit_word(word)
        w, h = text_wh(word, pw)
        ok(f"{name}: word fits at {pw}px", w <= 980 and pw >= 40,
           f"'{word}' {w:.0f}px wide")
        if gloss:
            gw, _ = text_wh(gloss, PX_GLOSS)
            ok(f"{name}: gloss fits", gw <= 980, f"{gw:.0f}px")
    for name, segs in EQ.items():
        tw = sum(text_wh(s, PX_EQ)[0] for s, _ in segs)
        ok(f"{name}: equation fits at {PX_EQ}px", tw <= 980,
           f"{tw:.0f}px")
    fw, _ = text_wh(FIN_EQ, PX_FINEQ)
    ok("FIN: four-equation line fits", fw <= 980, f"{fw:.0f}px")
    ok("text bands inside safe area (192..1632)",
       Y_NAME >= 192 and Y_GLOSS + PX_GLOSS * 2 < 530
       and Y_EQ + PX_EQ * 2 < 1632 and Y_FIN2 + 40 * 2 < 1632)
    ok("dot region clear of text bands",
       REG_Y[0] - RD > Y_GLOSS + PX_GLOSS * 1.6
       and REG_Y[1] + RD < min(Y_EQ, Y_FINEQ))

    # -- palette: every ink in exactly its own fence, AS COMPOSITED
    #    (all inks are drawn at full alpha — trap 77)
    for iname, c in INK.items():
        p = to8(c)
        hits = [f for f, fn in FENCES.items() if fn(p)]
        want = [FENCE_OF[iname]] if FENCE_OF[iname] else []
        ok(f"ink {iname} {p} hits exactly its own fence",
           hits == want, f"hits {hits}")

    # -- AA blends of each ink with BG hit NO fence
    bg = to8(BGC)
    worst = []
    for iname, c in INK.items():
        p = to8(c)
        for t in (0.75, 0.5, 0.25, 0.12):
            q = tuple(int(t * a_ + (1 - t) * b_ + 0.5)
                      for a_, b_ in zip(p, bg))
            hits = [f for f, fn in FENCES.items() if fn(q)]
            # a blend may stay in its OWN fence (thick core), never
            # in another's
            bad = [h for h in hits if h != FENCE_OF[iname]]
            if bad:
                worst.append((iname, t, q, bad))
    ok("no ink-x-BG AA blend crosses ANOTHER fence", not worst,
       f"{worst[:3]}")

    # -- blob-size floor vs chroma strays (trap 78)
    dot_px = math.pi * RD * RD
    ok("dot area >> decode floor 100 px >> stray max ~30 px",
       dot_px > 300 and 100 < dot_px / 3, f"dot ~{dot_px:.0f} px^2")
    gap = PITCH - 2 * RD
    ok("inter-dot gap survives ~2px chroma bleed each side",
       gap >= 8, f"{gap:.0f} px")

    # -- ghost ring reaches full ink at its core (fence measurable)
    ok("ghost ring core coverage saturates (lw/2+0.5 >= 1)",
       2.5 / 2 + 0.5 >= 1.0)

    print()
    if FAIL:
        print(f"{FAIL} FEASIBILITY FAILURES")
        sys.exit(1)
    print(f"ALL {PASS} FEASIBILITY CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
