#!/usr/bin/env python3
"""
FARO -- eight perfect shuffles and every card is back exactly where it started.

Cut a 52-card deck exactly in half and interleave the two packets perfectly,
one card at a time, top card staying on top. That is an out-faro. Do it eight
times and the deck is in its original order. Not approximately. Exactly.

The reason is arithmetic and there is nothing in the picture that shows it:
the card in position i lands in position 2i mod 51 (the top and bottom cards
never move), so after k shuffles it is at 2^k i mod 51, and the deck is home
the first time 2^k = 1 mod 51. That is k = 8, because 256 = 5*51 + 1.

Wikipedia, "Faro shuffle": "if one manages to perform eight out-shuffles in a
row, then the deck of 52 cards will be restored to its original order". The
same page: 52 in-shuffles are needed for the other kind, and 26 of them
reverse the deck.

So the video is exactly one cycle -- it loops, and frame 0 and the last frame
are asserted identical pixel for pixel. This channel has made loops before,
but out of geometry (a four-wing door is invariant under a half turn) or out
of real time (a step comes back in forty seconds). This one closes because of
a number you cannot see. That is why it earned another go at the shape.

Each card is painted by its ORIGINAL position, so "in order" is a smooth
colour ramp and any single card out of place is a visible jag.

    python3 scripts/faro.py --check
    python3 scripts/faro.py --stills /tmp/faro
    python3 scripts/faro.py --out content/faro.mp4

numpy + pycairo + ffmpeg.
"""

import argparse
import subprocess
import sys

import cairo
import numpy as np

# ------------------------------------------------------------------ the deck

N = 52
HALF = N // 2

RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
SUITS = ['♠', '♥', '♣', '♦']      # spades hearts clubs diamonds
NAMES = [RANKS[i % 13] + SUITS[i // 13] for i in range(N)]


def faro_physical(state):
    """Do it the way hands do it: split off the top half, interleave, top
    card stays on top."""
    top, bot = state[:HALF], state[HALF:]
    out = []
    for a, b in zip(top, bot):
        out.append(a)
        out.append(b)
    return out


def faro_closed(state):
    """Do it the way the arithmetic does it: position i -> 2i mod (N-1),
    with the bottom card fixed. Held out from the render, which uses
    faro_physical, so the two can be checked against each other."""
    new = [None] * N
    for i, v in enumerate(state):
        new[(2 * i) % (N - 1) if i < N - 1 else N - 1] = v
    return new


IDENT = list(range(N))

STATES = [IDENT]                       # STATES[s] = deck after s shuffles
while True:
    STATES.append(faro_physical(STATES[-1]))
    if STATES[-1] == IDENT:
        break
K_SHUF = len(STATES) - 1               # 8, and it is derived, not typed

# ------------------------------------------------------------------ picture

W, H = 1080, 1920
FPS = 30
SS = 2

SAFE_TOP, SAFE_BOT = 192, 1656

CX = 540.0
CARD_W = 440.0
CARD_H = 21.0
PITCH = 24.0
DECK_TOP = 285.0                       # centre of slot 0 in the full column
COL_MID = DECK_TOP + PITCH * (N - 1) / 2.0
PACK_TOP = COL_MID - PITCH * (HALF - 1) / 2.0
X_LEFT, X_RIGHT = CX - 250.0, CX + 250.0

Y_PIP = 1600.0
PIP_W, PIP_GAP = 30.0, 56.0

# cairo wants 0..1 floats. 0..255 clamps every channel to white and no
# geometry check will ever notice. (RENDERER.md trap 55.)
BG = (0.945, 0.938, 0.918)
EDGE = (0.780, 0.770, 0.745)
PIP_OFF = (0.800, 0.790, 0.765)
PIP_ON = (0.180, 0.185, 0.205)

# indigo -> teal -> green -> gold -> red. Adjacent cards differ a little, so
# a deck in order is a smooth ramp and one card out of place is a seam.
STOPS = [(0.09, 0.11, 0.34), (0.09, 0.42, 0.60), (0.24, 0.68, 0.48),
         (0.93, 0.79, 0.26), (0.84, 0.26, 0.20)]


def ramp(f):
    x = f * (len(STOPS) - 1)
    k = min(int(x), len(STOPS) - 2)
    u = x - k
    a, b = STOPS[k], STOPS[k + 1]
    return tuple(a[j] + (b[j] - a[j]) * u for j in range(3))


CARD_RGB = [ramp(i / (N - 1)) for i in range(N)]
INK = [(0.97, 0.97, 0.97) if (0.30 * c[0] + 0.59 * c[1] + 0.11 * c[2]) < 0.52
       else (0.10, 0.10, 0.12) for c in CARD_RGB]

FONT = "DejaVu Sans"

# ------------------------------------------------------------------ timing

LEAD = 0.50
T_SH = 0.95
HOLD = 1.40
DUR = LEAD + K_SHUF * T_SH + HOLD
N_FRAME = int(round(DUR * FPS))

SPLIT_END = 0.34
HOLD_END = 0.44
RIF_SPAN = 0.30                        # stagger across the 52 cards
RIF_DUR = 0.24                         # each card's own travel time
FADE_A = DUR - 0.55                    # pips clear so the loop closes
FADE_B = DUR - 0.20


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def slot_y(k):
    return DECK_TOP + PITCH * k


def packet_pos(i):
    if i < HALF:
        return X_LEFT, PACK_TOP + PITCH * i
    return X_RIGHT, PACK_TOP + PITCH * (i - HALF)


def dest_slot(i):
    return (2 * i) % (N - 1) if i < N - 1 else N - 1


def layout(tv):
    """-> (list of (x, y, original_card_index), pips_lit)."""
    if tv < LEAD:
        s, u = 0, None
    else:
        s = int((tv - LEAD) // T_SH)
        u = (tv - LEAD) / T_SH - s
        if s >= K_SHUF:
            s, u = K_SHUF, None

    state = STATES[s]
    if u is None:
        return [(CX, slot_y(i), state[i]) for i in range(N)], min(s, K_SHUF)

    out = []
    for i in range(N):
        px, py = packet_pos(i)
        if u < SPLIT_END:
            e = ease(u / SPLIT_END)
            x = CX + (px - CX) * e
            y = slot_y(i) + (py - slot_y(i)) * e
        elif u < HOLD_END:
            x, y = px, py
        else:
            d = dest_slot(i)
            t0 = HOLD_END + RIF_SPAN * (d / (N - 1))
            e = ease((u - t0) / RIF_DUR)
            x = px + (CX - px) * e
            y = py + (slot_y(d) - py) * e
        out.append((x, y, state[i]))
    return out, s


# ------------------------------------------------------------------ drawing

def rrect(cr, x, y, w, h, r):
    import math
    cr.new_sub_path()
    cr.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    cr.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    cr.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    cr.arc(x + r, y + r, r, math.pi, 1.5 * math.pi)
    cr.close_path()


def render_frame(surf, cr, i):
    tv = i / FPS
    cards, lit = layout(tv)

    cr.set_source_rgb(*BG)
    cr.paint()
    cr.set_antialias(cairo.ANTIALIAS_BEST)
    cr.select_font_face(FONT, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)

    for x, y, c in cards:
        left = (x - CARD_W / 2) * SS
        top = (y - CARD_H / 2) * SS
        rrect(cr, left, top, CARD_W * SS, CARD_H * SS, 5.0 * SS)
        cr.set_source_rgb(*CARD_RGB[c])
        cr.fill_preserve()
        cr.set_source_rgb(*EDGE)
        cr.set_line_width(1.6 * SS)
        cr.stroke()

        cr.set_font_size(14.0 * SS)
        cr.set_source_rgb(*INK[c])
        ext = cr.text_extents(NAMES[c])
        cr.move_to(left + 14.0 * SS, top + (CARD_H * SS + ext.height) / 2)
        cr.show_text(NAMES[c])

    a = 1.0
    if tv > FADE_A:
        a = max(0.0, 1.0 - (tv - FADE_A) / (FADE_B - FADE_A))
    x0 = CX - ((K_SHUF - 1) * PIP_GAP + PIP_W) / 2.0
    for k in range(K_SHUF):
        on = (k < lit) and a > 0.0
        col = PIP_ON if on else PIP_OFF
        if on:
            col = tuple(PIP_OFF[j] + (PIP_ON[j] - PIP_OFF[j]) * a
                        for j in range(3))
        rrect(cr, (x0 + k * PIP_GAP) * SS, (Y_PIP - PIP_W / 2) * SS,
              PIP_W * SS, PIP_W * SS, 7.0 * SS)
        cr.set_source_rgb(*col)
        cr.fill()

    buf = np.ndarray(shape=(H * SS, W * SS, 4), dtype=np.uint8,
                     buffer=surf.get_data())
    img = buf[:, :, [2, 1, 0]].astype(np.float32) / 255.0
    return img.reshape(H, SS, W, SS, 3).mean(axis=(1, 3))


def to8(img):
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# ------------------------------------------------------------------ checks

OK = True


def t(cond, msg):
    global OK
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        OK = False


def read_deck(img):
    """Read the 52 card colours back out of a finished frame.

    Bounded in ROWS as well as columns: only the deck's own band, so the
    pip row is excluded, and the strip sits right of centre so the rank
    labels on the left of each card are excluded too. A pixel check has no
    idea what it is looking at. (RENDERER.md trap 58.)
    """
    got = []
    for k in range(N):
        r = int(round(slot_y(k)))
        band = img[r - 5:r + 6, int(CX) + 120:int(CX) + 180, :]
        got.append(band.reshape(-1, 3).mean(0))
    return np.array(got)


def run_checks(surf, cr):
    print("\narithmetic")
    t(K_SHUF == 8, f"out-faro order on {N} cards = {K_SHUF} (derived)")
    o = 1
    p = 2 % (N - 1)
    while p != 1:
        p = (p * 2) % (N - 1)
        o += 1
    t(o == K_SHUF, f"multiplicative order of 2 mod {N - 1} = {o}, agrees")
    for s in range(1, K_SHUF):
        if STATES[s] == IDENT:
            t(False, f"deck was already home after {s}")
    t(all(STATES[s] != IDENT for s in range(1, K_SHUF)),
      f"not home after any of shuffles 1..{K_SHUF - 1}")

    # held out: the render uses faro_physical, never faro_closed
    rng = np.random.default_rng(7)
    same = True
    for _ in range(12):
        st = list(map(int, rng.permutation(N)))
        same = same and faro_physical(st) == faro_closed(st)
    t(same, "hands and arithmetic agree on 12 random decks")

    oi = 1
    p = 2 % (N + 1)
    while p != 1:
        p = (p * 2) % (N + 1)
        oi += 1
    t(oi == 52, f"in-faro order = {oi} (for the description)")

    print("\ngeometry")
    top = DECK_TOP - CARD_H / 2
    bot = slot_y(N - 1) + CARD_H / 2
    t(top > SAFE_TOP and bot < SAFE_BOT, f"deck rows {top:.0f}..{bot:.0f}")
    t(Y_PIP + PIP_W / 2 < SAFE_BOT, f"pip row ends {Y_PIP + PIP_W / 2:.0f}")
    t(X_LEFT - CARD_W / 2 > 0 and X_RIGHT + CARD_W / 2 < W,
      f"split packets span {X_LEFT - CARD_W / 2:.0f}..{X_RIGHT + CARD_W / 2:.0f}")
    t(X_LEFT + CARD_W / 2 < X_RIGHT - CARD_W / 2, "packets do not overlap")
    cr.select_font_face(FONT, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    cr.set_font_size(14.0 * SS)
    wid = [cr.text_extents(s).width for s in SUITS]
    t(min(wid) > 4.0, f"suit glyphs render, narrowest {min(wid):.1f} px")

    print("\npixels")
    f_end = render_frame(surf, cr, N_FRAME - 1)
    got = read_deck(f_end)
    err = np.abs(got - np.array(CARD_RGB)).max()
    t(err < 0.02, f"final frame reads back as the deck IN ORDER, max err {err:.4f}")

    mid_tv = LEAD + 4 * T_SH - 0.02
    f_mid = render_frame(surf, cr, int(round(mid_tv * FPS)))
    gm = read_deck(f_mid)
    want4 = np.array([CARD_RGB[c] for c in STATES[4]])
    t(np.abs(gm - want4).max() < 0.02,
      "after 4 shuffles the frame reads back as the SHUFFLED deck")
    wrong = int((np.abs(gm - np.array(CARD_RGB)).max(1) > 0.02).sum())
    t(wrong >= 40, f"and it is genuinely scrambled -- {wrong}/{N} slots moved")

    f0 = render_frame(surf, cr, 0)
    d = np.abs(f0 - f_end).max()
    t(d == 0.0, f"frame 0 == frame {N_FRAME - 1}, exact loop (max diff {d:.5f})")

    ink = np.abs(f_end - np.array(BG)).max(2) > 0.02
    rows = np.where(ink.any(1))[0]
    t(rows.min() >= SAFE_TOP and rows.max() <= SAFE_BOT,
      f"all ink in rows {rows.min()}..{rows.max()}")
    t(0.10 < ink.mean() < 0.45, f"{100 * ink.mean():.1f}% of the frame is ink")

    print("\n" + ("ALL CHECKS PASS" if OK else "SOMETHING FAILED"))
    return 0 if OK else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--stills')
    ap.add_argument('--out')
    args = ap.parse_args()

    print(f"{N} cards, out-faro, order {K_SHUF}")
    print(f"  video {DUR:.2f} s, {N_FRAME} frames, {T_SH:.2f} s per shuffle")

    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W * SS, H * SS)
    cr = cairo.Context(surf)

    if args.check:
        return run_checks(surf, cr)

    if args.stills:
        from PIL import Image
        for i in (0, int((LEAD + 0.20) * FPS), int((LEAD + 0.40) * FPS),
                  int((LEAD + 0.62) * FPS), int((LEAD + 3 * T_SH) * FPS) - 2,
                  N_FRAME - 1):
            Image.fromarray(to8(render_frame(surf, cr, i))).save(
                f"{args.stills}_{i:04d}.png")
            print(f"  still {i}")
        return 0

    if not args.out:
        print("nothing to do -- pass --check, --stills or --out")
        return 1

    cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{W}x{H}', '-r', str(FPS), '-i', 'pipe:0',
           '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', args.out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for i in range(N_FRAME):
        p.stdin.write(to8(render_frame(surf, cr, i)).tobytes())
        if i % 30 == 0:
            print(f"  frame {i}/{N_FRAME}", flush=True)
    p.stdin.close()
    p.wait()
    print(f"wrote {args.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
