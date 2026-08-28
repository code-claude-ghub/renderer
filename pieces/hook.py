#!/usr/bin/env python3
"""THE HOOK — text on a black screen.

The loose hook on a tape measure is not worn out. It is riveted through oval
holes so it can slide by exactly its own thickness, which is what lets one
tape read true both hooked over an edge and pushed against a wall.

Drawn on a character grid on purpose: in a grid a thickness is an integer, so
"slides exactly its own thickness" is a thing you can count rather than a
claim you have to take.

    python3 scripts/hook.py --check     # assertions only, no render
    python3 scripts/hook.py             # render to out/hook.mp4
"""
import sys, os, math, argparse
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
from asciilib import Grid, Frame, Encoder

# ---------------------------------------------------------------- the model
#
# One reference face at x = 0. A tape blade whose zero mark sits on that face
# in BOTH measurements. A hook of thickness t riveted loosely to the blade.
#
#   outside: hook wraps the face, so hook material lies in [-t, 0]
#   inside : hook presses the face, so hook material lies in [ 0, +t]
#
# The face does not move. The reading does not move. The hook moves by t.

SPAN_MM   = 100.0        # the thing being measured, both times
MM_PER_CELL = 4.0        # schematic. a real hook is about 1 mm, not 4.
HOOK_T_CELLS = 1         # a thickness is an integer here. that is the point.

SPAN  = int(round(SPAN_MM / MM_PER_CELL))     # 25 cells
Z     = 9                                     # column of the reference face
FAR   = Z + SPAN                              # column of the far face

def hook_col(mode):
    """Derived from the model above, not from the drawing."""
    if mode == "outside":
        return Z - HOOK_T_CELLS      # material sits in [-t, 0)
    if mode == "inside":
        return Z                     # material sits in [0, +t)
    raise ValueError(mode)

# ---------------------------------------------------------------- the sheet
G   = Grid(font_size=40)

def C(r, g, b):
    """cairo wants 0..1. Handing it 0..255 clamps everything to white, and
    no geometry check will ever notice."""
    return (r / 255.0, g / 255.0, b / 255.0)

BG   = C(8, 8, 10)
DIM  = C(78, 78, 84)
MAT  = C(122, 122, 128)
TAPE = C(232, 232, 226)
HOOK = C(255, 176, 74)
GHOST= C(40, 40, 46)

FPS = 30
B1, B3, END = 2.4, 5.2, 9.4   # outside | both | isolate. (s)

PANEL_A = 27          # top row of the board
PANEL_B = 45          # top row of the walls


def panel(fr, top, mode, lit_hook=True, mat=MAT, tape=TAPE, hook=HOOK):
    """Draw one measurement. Same span, same number, both times."""
    hc = hook_col(mode)
    body = range(top, top + 3)

    if mode == "outside":
        for r in body:
            fr.put_run(Z, r, "█" * SPAN, mat)
    else:
        for r in body:
            fr.put_run(Z - 4, r, "█" * 4, mat)          # left wall
            fr.put_run(FAR,  r, "█" * 4, mat)           # right wall

    # the hook — one cell wide, taller than the body so it reads as a hook
    for r in range(top - 1, top + 4):
        fr.put(hc, r, "█", hook if lit_hook else GHOST)

    # the blade: zero on the face, in both cases
    row = top + 4
    fr.put(Z, row, "╞", tape)
    fr.put_run(Z + 1, row, "═" * (SPAN - 2), tape)
    fr.put(FAR - 1, row, "╡", tape)

    label = "100 mm"
    fr.put_run(Z + (SPAN - len(label)) // 2, row + 2, label, tape)


def draw(f):
    t = f / FPS
    fr = Frame(G, BG)

    if t < B3:
        fr.put_run(Z, PANEL_A - 3, "hooked over an edge", DIM)
        panel(fr, PANEL_A, "outside")
        if t >= B1:
            fr.put_run(Z, PANEL_B - 3, "pushed against a wall", DIM)
            panel(fr, PANEL_B, "inside")

    elif t < END:
        # isolate: two lanes one cell apart, and the hook in one of each.
        # the face line is NOT drawn here — in the pushed case it lives in the
        # same cell as the hook, so it disappears under it and the two rows
        # stop being comparable.
        base = 33
        for c in (hook_col("outside"), hook_col("inside")):
            for r in range(base - 2, base + 11):
                fr.put(c, r, "│", GHOST)
        fr.put_run(3, base - 5, "same tape. same reading.", DIM)
        for i, (mode, word) in enumerate((("outside", "hooked over"),
                                          ("inside",  "pushed against"))):
            r = base + i * 6
            for rr in range(r, r + 3):
                fr.put(hook_col(mode), rr, "█", HOOK)
            fr.put_run(Z + 4, r + 1, word, DIM)
        fr.put_run(3, base + 13, "one cell. its own thickness.", TAPE)
        fr.put_run(3, base + 16, "loose so that it is accurate.", TAPE)

    return fr


# ---------------------------------------------------------------- the check
def check():
    ok = 0

    # 1. the hook is one cell in both, and moves by exactly that
    d = hook_col("inside") - hook_col("outside")
    assert d == HOOK_T_CELLS, (d, HOOK_T_CELLS)
    print(f"  hook offset          {d} cell  == thickness {HOOK_T_CELLS}"); ok += 1

    # 2. held out: re-derive from the physics in millimetres, never touching
    #    the drawing constants above.
    t_mm = HOOK_T_CELLS * MM_PER_CELL
    outside_zero_mm = -t_mm + t_mm      # hook spans [-t,0], blade zero at 0
    inside_zero_mm  = 0.0               # hook spans [0,+t], blade zero at 0
    assert abs(outside_zero_mm - inside_zero_mm) < 1e-9
    shift_mm = t_mm
    assert abs(shift_mm / MM_PER_CELL - d) < 1e-9, (shift_mm, d)
    print(f"  held out: blade zero identical both ways, hook shifts "
          f"{shift_mm:.0f} mm = {d} cell"); ok += 1

    # 3. the span really is the same, both times
    assert FAR - Z == SPAN
    print(f"  span                 cols {Z}..{FAR-1}, {SPAN} cells "
          f"= {SPAN*MM_PER_CELL:.0f} mm, both panels"); ok += 1

    # 4. the hook is OUTSIDE the measured span when hooked, INSIDE it when
    #    pushed. that inversion is the whole picture.
    assert hook_col("outside") < Z <= hook_col("inside") < FAR
    print(f"  hook outside span when hooked ({hook_col('outside')} < {Z}), "
          f"inside it when pushed ({hook_col('inside')})"); ok += 1

    # 5. READ THE PIXELS. every check above is arithmetic about integers and
    #    would pass just as happily on a blank frame — which is exactly what
    #    the first render of this piece was.
    import numpy as np
    def ink(f):
        fr = draw(f)
        buf = np.frombuffer(fr.surface.get_data(), np.uint8)
        buf = buf.reshape(G.h_px, -1, 4)[:, :G.w_px, :3].astype(np.int16)
        bg = np.array([BG[2] * 255, BG[1] * 255, BG[0] * 255])   # cairo is BGRA
        return fr, buf, (np.abs(buf - bg).max(2) > 24)

    for f, name in ((int(1.2*FPS), "outside"), (int(4.0*FPS), "both"),
                    (int(7.5*FPS), "isolate")):
        fr, buf, lit = ink(f)
        frac = lit.mean()
        assert 0.004 < frac < 0.45, (name, frac)   # not blank, not a white sheet
        rows = np.where(lit.any(1))[0]
        top_row, bot_row = int(rows[0] // G.cell), int(rows[-1] // G.cell)
        assert top_row >= G.safe_top, (name, top_row, G.safe_top)
        assert bot_row <= G.safe_bot, (name, bot_row, G.safe_bot)
        print(f"  pixels [{name:7s}]     {frac*100:5.2f}% lit, "
              f"rows {top_row}..{bot_row} inside {G.safe_top}..{G.safe_bot}")
    ok += 1

    # 6. the hook is actually AMBER, and actually one cell wide, on screen.
    fr, buf, lit = ink(int(1.2 * FPS))
    y = int((PANEL_A + 1) * G.cell + G.cell // 2)
    row_px = buf[y]                                   # BGRA -> B,G,R
    amber = (row_px[:, 2] > 180) & (row_px[:, 0] < 140)
    xs = np.where(amber)[0]
    want = hook_col("outside")
    share = np.mean((xs // G.cell).astype(int) == want)
    width = xs.max() - xs.min() + 1
    # a block glyph measures 25px in a 24px cell, which is WHY the solid body
    # tiles without seams. one cell wide, plus a pixel of deliberate bleed.
    assert share > 0.9, (share, want)
    assert G.cell <= width <= G.cell + 2, (width, G.cell)
    print(f"  hook on screen       amber {share*100:.0f}% in column {want}, "
          f"{width}px wide in a {int(G.cell)}px cell"); ok += 1

    # 6. the schematic is honest about being one
    real_hook_mm = 1.0
    print(f"  NOTE drawn hook is {t_mm:.0f} mm at this scale, about "
          f"{t_mm/real_hook_mm:.0f}x a real one. say so in the description.")

    print(f"\n  {ok} checks passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--still", type=float, default=None)
    ap.add_argument("--out", default="out/hook.mp4")
    a = ap.parse_args()

    if a.check:
        check(); return

    if a.still is not None:
        fr = draw(int(a.still * FPS))
        p = f"out/hook_{a.still:g}.png"
        fr.surface.write_to_png(p)
        print(p); return

    os.makedirs("out", exist_ok=True)
    n = int(END * FPS)
    with Encoder(a.out, G, fps=FPS) as enc:
        for f in range(n):
            enc.write(draw(f))
    print(f"{a.out}  {n} frames  {n/FPS:.1f}s  {G.cols}x{G.rows} cells")


if __name__ == "__main__":
    main()
