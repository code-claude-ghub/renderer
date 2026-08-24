#!/usr/bin/env python3
"""THE YELLOW YOU ARE OWED  --  one traffic signal head, filling the frame.

The middle lens carries a ring. One full turn of that ring is 4.3 seconds:
the yellow the ITE kinematic equation says a driver at 45 mph needs in order
to either stop comfortably or reach the stop bar.

    Y = t + V / (2a + 2Gg)          t = 1.0 s, a = 10 ft/s^2, G = 32.2

At 45 mph (66.0 ft/s exactly) on level grade: Y = 1.0 + 66.0/20 = 4.3 s.

The MUTCD's own guidance floor is 3 seconds. So the light goes red at 70% of
the way round the ring, and the last 30% -- 1.3 seconds, 86 feet of road at
66 ft/s -- gets drawn in cold blue against a red lens. That arc is the piece.

HELD-OUT CHECK on the 86 feet. The render uses gap = v * (Y - 3.0) = 85.8 ft.
Kept out of that and used as the test: the dilemma zone is the road between
the farthest point you can still clear (v*3.0 = 198.0 ft) and the nearest
point you can still stop from (v*t + v^2/2a = 283.8 ft). 283.8 - 198.0 =
85.8. Two derivations, one number, agreeing exactly.

Sources are in the description. Silent, no audio track.
"""
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]

from asciilib import Encoder, Frame, Grid, ink_lut, lambert, specular, visible, zbuffer

# ---------------------------------------------------------------- palette
# ground: cold teal-black wet asphalt at night.  figure: municipal signal
# green with an aluminium rim.  no overlap with the last four colorways
# (charcoal-violet/safety-yellow, blue-on-warm-paper, amber-on-charcoal,
# buff-clay-on-petrol).
BG      = (0.031, 0.068, 0.076)
HOUSE   = (0.243, 0.353, 0.290)   # signal-housing green
VISOR   = (0.176, 0.263, 0.216)   # visor outside, darker
CAVITY  = (0.086, 0.133, 0.114)   # visor inside
RIM     = (0.639, 0.678, 0.647)   # aluminium bezel
TRACK   = (0.216, 0.290, 0.298)   # the unfilled ring
AMBER   = (0.992, 0.580, 0.055)   # lit yellow lens
REDL    = (0.965, 0.176, 0.153)   # lit red lens
GREENL  = (0.157, 0.878, 0.451)   # lit green lens
DARKA   = (0.243, 0.184, 0.114)   # unlit amber
DARKR   = (0.243, 0.114, 0.110)   # unlit red
DARKG   = (0.125, 0.212, 0.161)   # unlit green
ICE     = (0.741, 0.949, 1.000)   # the overrun
DEAD    = (0.086, 0.055, 0.020)   # clock punched out of a burning lens

# ------------------------------------------------------------- arithmetic
MPH        = 45.0
V_FTS      = MPH * 5280.0 / 3600.0        # 66.0 ft/s exactly
T_PR       = 1.0                          # perception-reaction, NCHRP 03-95
DECEL      = 10.0                         # ft/s^2, NCHRP 03-95
Y_NEED     = T_PR + V_FTS / (2.0 * DECEL) # 4.3 s
Y_FLOOR    = 3.0                          # MUTCD 4F.17 guidance minimum
GAP_S      = Y_NEED - Y_FLOOR             # 1.3 s
GAP_FT     = GAP_S * V_FTS                # 85.8 ft
# held out of the above, used only in check():
STOP_FT    = V_FTS * T_PR + V_FTS ** 2 / (2.0 * DECEL)   # 283.8
CLEAR_FT   = V_FTS * Y_FLOOR                             # 198.0

FRAC_RED   = Y_FLOOR / Y_NEED             # 0.6977 of the ring

# --------------------------------------------------------------- geometry
# inches.  a 12-inch signal: 12" lens, ~16.6" section pitch, 9.5" visor.
R_SEC  = 8.4
R_HOUS = 7.4          # front annulus inner edge / bezel outer
R_LENS = 6.6
SP     = 16.6         # section pitch
D_FRT  = 7.0          # front face z
VIS_L  = 9.5
RING_R0, RING_R1 = 7.55, 8.40
RING_Z = D_FRT + 0.75

YAW   = math.radians(23.0)
PITCH = math.radians(-10.0)     # negative -> we look UP at it, from below

# screen
G      = Grid()
RAMP   = ink_lut()
K      = 2.68                   # cells per inch
C_MID  = 49
R_TOP  = 31                     # screen row of the top of the head
R_CEN  = R_TOP + int(round(25.0 * K))
ROW_Y  = R_CEN                  # yellow section centre row
ROW_R  = R_CEN - int(round(SP * K))
ROW_G  = R_CEN + int(round(SP * K))

LAMP = np.array([-0.46, 0.60, 0.66])
LAMP = LAMP / np.linalg.norm(LAMP)

# materials
M_HOUSE, M_VISOR, M_CAVITY, M_RIM = 0, 1, 2, 3
M_RED, M_YEL, M_GRN = 4, 5, 6

FPS     = 30
T_GREEN = 1.40
T_RED   = T_GREEN + Y_FLOOR            # 4.40
T_FULL  = T_GREEN + Y_NEED             # 5.70
T_SWAP  = T_FULL + 1.10                # 6.80
T_END   = 8.60
FRAMES  = int(round(T_END * FPS))

RNG = np.random.default_rng(4530)


# ------------------------------------------------------------------ build
def _disc(n, r0, r1):
    r = np.sqrt(RNG.uniform(r0 ** 2, r1 ** 2, n))
    a = RNG.uniform(0, 2 * math.pi, n)
    return r, a


def build():
    P, N, M = [], [], []

    def add(p, n, m):
        P.append(p)
        N.append(n)
        M.append(np.full(len(p), m, np.int8))

    for yc, mat in ((SP, M_RED), (0.0, M_YEL), (-SP, M_GRN)):
        # --- lens: a shallow spherical cap, so it shades instead of reading flat
        n = 26000
        r, a = _disc(n, 0.0, R_LENS)
        x, y = r * np.cos(a), r * np.sin(a)
        bul = 0.95 * (1.0 - (r / R_LENS) ** 2)
        z = D_FRT + bul
        # dz/dr = -1.9 r / R^2
        slope = -1.9 * r / R_LENS ** 2
        nx, ny, nz = -slope * np.cos(a), -slope * np.sin(a), np.ones(n)
        ln = np.sqrt(nx * nx + ny * ny + nz * nz)
        add(np.stack([x, y + yc, z], 1), np.stack([nx / ln, ny / ln, nz / ln], 1), mat)

        # --- bezel: outer wall + top annulus
        n = 7000
        a = RNG.uniform(0, 2 * math.pi, n)
        zz = RNG.uniform(D_FRT, D_FRT + 0.9, n)
        add(np.stack([R_HOUS * np.cos(a), R_HOUS * np.sin(a) + yc, zz], 1),
            np.stack([np.cos(a), np.sin(a), np.zeros(n)], 1), M_RIM)
        n = 9000
        r, a = _disc(n, R_LENS, R_HOUS)
        add(np.stack([r * np.cos(a), r * np.sin(a) + yc, np.full(n, D_FRT + 0.9)], 1),
            np.tile([0.0, 0.0, 1.0], (n, 1)), M_RIM)

        # --- front face of the housing, outside the bezel
        n = 9000
        r, a = _disc(n, R_HOUS, R_SEC)
        add(np.stack([r * np.cos(a), r * np.sin(a) + yc, np.full(n, D_FRT)], 1),
            np.tile([0.0, 0.0, 1.0], (n, 1)), M_HOUSE)

        # --- housing barrel
        n = 16000
        a = RNG.uniform(0, 2 * math.pi, n)
        zz = RNG.uniform(-2.0, D_FRT, n)
        add(np.stack([R_SEC * np.cos(a), R_SEC * np.sin(a) + yc, zz], 1),
            np.stack([np.cos(a), np.sin(a), np.zeros(n)], 1), M_HOUSE)

        # --- visor: cutaway hood over the top of the lens.  full length at
        #     the crown, tapering to nothing below the horizontal.
        for rad, mat, sgn in ((R_SEC, M_VISOR, 1.0), (R_SEC - 0.55, M_CAVITY, -1.0)):
            n = 22000
            a = RNG.uniform(0, 2 * math.pi, n)
            up = np.sin(a)                      # +1 at the crown
            s = np.clip((up + 0.34) / 1.34, 0.0, 1.0)
            L = VIS_L * s ** 0.75
            keep = L > 0.15
            a, L = a[keep], L[keep]
            zz = D_FRT + RNG.uniform(0, 1, len(a)) * L
            nrm = np.stack([sgn * np.cos(a), sgn * np.sin(a), np.zeros(len(a))], 1)
            add(np.stack([rad * np.cos(a), rad * np.sin(a) + yc, zz], 1), nrm, mat)

        # --- visor front rim
        n = 4000
        a = RNG.uniform(0, 2 * math.pi, n)
        up = np.sin(a)
        s = np.clip((up + 0.34) / 1.34, 0.0, 1.0)
        L = VIS_L * s ** 0.75
        keep = L > 0.15
        a, L = a[keep], L[keep]
        rr = RNG.uniform(R_SEC - 0.55, R_SEC, len(a))
        add(np.stack([rr * np.cos(a), rr * np.sin(a) + yc, D_FRT + L], 1),
            np.tile([0.0, 0.0, 1.0], (len(a), 1)), M_RIM)

    P = np.concatenate(P)
    N = np.concatenate(N)
    M = np.concatenate(M)
    return P.astype(np.float32), N.astype(np.float32), M


def build_ring(n=52000):
    """points on the ring track around the yellow section, with their angle
    measured clockwise from the top as seen on screen."""
    r, a = _disc(n, RING_R0, RING_R1)
    p = np.stack([r * np.cos(a), r * np.sin(a), np.full(n, RING_Z)], 1)
    # screen-clockwise from 12 o'clock:  a = pi/2 -> u = 0
    u = ((math.pi / 2 - a) % (2 * math.pi)) / (2 * math.pi)
    return p.astype(np.float32), u.astype(np.float32)


def rotate(p):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    cy, sy = math.cos(YAW), math.sin(YAW)
    x1, z1 = x * cy + z * sy, -x * sy + z * cy
    cx, sx = math.cos(PITCH), math.sin(PITCH)
    y1, z2 = y * cx - z1 * sx, y * sx + z1 * cx
    return np.stack([x1, y1, z2], 1)


def project(p):
    col = np.rint(C_MID + p[:, 0] * K).astype(np.int32)
    row = np.rint(R_CEN - p[:, 1] * K).astype(np.int32)
    return col, row, p[:, 2]


# -------------------------------------------------------------------- font
# variable width on purpose.  a three-column M reads as an H and a
# three-column N reads as an S -- that bug shipped once and got caught on a
# contact sheet reading "HALE DUMHY".
FONT = {
    "0": ("###", "# #", "# #", "# #", "###"),
    "1": (" # ", "## ", " # ", " # ", "###"),
    "2": ("###", "  #", "###", "#  ", "###"),
    "3": ("###", "  #", "###", "  #", "###"),
    "4": ("# #", "# #", "###", "  #", "  #"),
    "5": ("###", "#  ", "###", "  #", "###"),
    "6": ("###", "#  ", "###", "# #", "###"),
    "7": ("###", "  #", "  #", "  #", "  #"),
    "8": ("###", "# #", "###", "# #", "###"),
    "9": ("###", "# #", "###", "  #", "###"),
    ".": (" ", " ", " ", " ", "#"),
    " ": ("  ", "  ", "  ", "  ", "  "),
    "M": ("#   #", "## ##", "# # #", "#   #", "#   #"),
    "P": ("###", "# #", "###", "#  ", "#  "),
    "H": ("# #", "# #", "###", "# #", "# #"),
    "F": ("###", "#  ", "###", "#  ", "#  "),
    "T": ("###", " # ", " # ", " # ", " # "),
}


def gap_of(sc):
    return max(1, sc - 1)


def text_size(s, sc):
    g = gap_of(sc)
    w = sum(len(FONT[c][0]) for c in s) * sc + (len(s) - 1) * g
    return w, 5 * sc


def text_mask(s, sc):
    w, h = text_size(s, sc)
    m = np.zeros((h, w), bool)
    g = gap_of(sc)
    x = 0
    for ch in s:
        rows = FONT[ch]
        gw = len(rows[0])
        for r in range(5):
            for c in range(gw):
                if rows[r][c] == "#":
                    m[r * sc:(r + 1) * sc, x + c * sc:x + (c + 1) * sc] = True
        x += gw * sc + g
    return m


def stamp(fr, s, sc, ccen, rcen, col, halo=BG):
    # trap 11, with a correction: a halo drawn as a SPACE glyph paints
    # nothing in cairo, so it does not mask anything. The halo has to be a
    # solid glyph in the halo colour.
    m = text_mask(s, sc)
    h, w = m.shape
    c0, r0 = ccen - w // 2, rcen - h // 2
    hmask = np.zeros((h + 2, w + 2), bool)
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            hmask[dr:dr + h, dc:dc + w] |= m
    for r in range(h + 2):
        for c in range(w + 2):
            if not hmask[r, c]:
                continue
            rr, cc = r0 - 1 + r, c0 - 1 + c
            if not (0 <= rr < G.rows and 0 <= cc < G.cols):
                continue
            on = 0 <= r - 1 < h and 0 <= c - 1 < w and m[r - 1, c - 1]
            if on:
                fr.put(cc, rr, "#", col)
            else:
                fr.put(cc, rr, "#", halo)


# ------------------------------------------------------------------- state
PTS, NRM, MAT = build()
RPTS = rotate(PTS)
RNRM = rotate(NRM)
RING_P, RING_U = build_ring()
RRING = rotate(RING_P)

LIT = {M_RED: False, M_YEL: False, M_GRN: False}
LAST_INK = None

# the yaw swings the lens face off the frame's centre column -- centring the
# clock on C_MID put it half onto the housing.  ask the projection instead.
_lc, _lr, _ = project(rotate(np.array([[0.0, 0.0, D_FRT + 0.95]], np.float32)))
LENS_C, LENS_R = int(_lc[0]), int(_lr[0])


def _ring_cells():
    """52k ring samples times 258 frames is 13M python iterations. The ring
    only ever covers ~430 CELLS -- collapse it once, keep each cell's median
    angle, and the per-frame cost stops mattering."""
    c, r, _ = project(RRING)
    ok = visible(G, c, r)
    c, r, u = c[ok], r[ok], RING_U[ok]
    # a cell straddling the 0/1 seam would average to 0.5 and light up in
    # the wrong half, so fold the seam before taking a middle value.
    key = r.astype(np.int64) * 1000 + c
    order = np.lexsort((u, key))
    c, r, u, key = c[order], r[order], u[order], key[order]
    _, start = np.unique(key, return_index=True)
    start = np.sort(start)
    end = np.append(start[1:], len(key))
    oc, orr, ou = [], [], []
    for a, b in zip(start, end):
        uu = u[a:b]
        if uu[-1] - uu[0] > 0.5:            # seam
            uu = np.where(uu > 0.5, uu - 1.0, uu)
        oc.append(int(c[a]))
        orr.append(int(r[a]))
        ou.append(float(np.median(uu) % 1.0))
    return np.array(oc), np.array(orr), np.array(ou)


RING_CELLS = _ring_cells()


def colour(shade, extra):
    e = int(extra)
    if e == M_HOUSE:
        base = HOUSE
    elif e == M_VISOR:
        base = VISOR
    elif e == M_CAVITY:
        base = CAVITY
    elif e == M_RIM:
        base = RIM
    elif e == M_RED:
        base = REDL if LIT[M_RED] else DARKR
    elif e == M_YEL:
        base = AMBER if LIT[M_YEL] else DARKA
    else:
        base = GREENL if LIT[M_GRN] else DARKG
    # trap 12: tint with a floor, never multiply the light twice
    return tuple(v * (0.44 + 0.56 * shade) for v in base)


def draw(f):
    t = f / FPS
    LIT[M_GRN] = t < T_GREEN
    LIT[M_YEL] = T_GREEN <= t < T_RED
    LIT[M_RED] = t >= T_RED

    col, row, z = project(RPTS)
    ok = visible(G, col, row)
    c, r, zz, n, m = col[ok], row[ok], z[ok], RNRM[ok], MAT[ok]
    _, keep = zbuffer(G, c, r, zz)
    lam = lambert(n, LAMP)
    shade = 0.16 + 0.72 * lam + 0.30 * specular(n, LAMP, 26)
    # an emitting lens is not lambert-shaded
    for mm in (M_RED, M_YEL, M_GRN):
        if LIT[mm]:
            shade = np.where(m == mm, 0.96, shade)
    shade = np.clip(shade, 0.0, 1.0)

    fr = Frame(G, BG)
    idx, _ = fr.field(c, r, keep, shade, colour, RAMP, extra=m)
    global LAST_INK
    LAST_INK = idx != 0

    # ---- the ring, drawn last so the visor cannot eat it
    swept = 0.0
    if t >= T_GREEN:
        swept = min((t - T_GREEN) / Y_NEED, 1.0)
    rc, rr, ru = RING_CELLS
    pulse = 1.0
    if t > T_FULL:
        pulse = 0.72 + 0.28 * math.cos((t - T_FULL) * 3.4)
    for i in range(len(rc)):
        u = ru[i]
        if u > swept:
            cc = TRACK
            gl = ":"
        elif u <= FRAC_RED:
            cc = AMBER
            gl = "#"
        else:
            cc = tuple(v * pulse for v in ICE)
            gl = "#"
        fr.put(int(rc[i]), int(rr[i]), gl, cc)

    # ---- instrument layer: the input, and the clock inside the lens
    if t > 0.35:
        stamp(fr, "45 MPH", 2, C_MID, 23, RIM)
    if t >= T_GREEN:
        # while the lamp is burning the clock is a hole punched in it; once
        # the lamp dies the clock is the only lit thing left on that lens.
        if t < T_SWAP:
            secs = min(t - T_GREEN, Y_NEED)
            txt = "%.1f" % secs
            if secs <= Y_FLOOR:
                stamp(fr, txt, 3, LENS_C, LENS_R, DEAD, halo=AMBER)
            else:
                stamp(fr, txt, 3, LENS_C, LENS_R, ICE)
        else:
            stamp(fr, "86 FT", 2, LENS_C, LENS_R, ICE)
    return fr


# ------------------------------------------------------------------ check
def check():
    print("45 mph = %.3f ft/s   Y = %.3f s   floor %.1f   gap %.2f s = %.1f ft"
          % (V_FTS, Y_NEED, Y_FLOOR, GAP_S, GAP_FT))
    alt = STOP_FT - CLEAR_FT
    print("HELD OUT  stop %.1f ft - clear %.1f ft = %.1f ft   (err %.4f ft)"
          % (STOP_FT, CLEAR_FT, alt, abs(alt - GAP_FT)))
    assert abs(alt - GAP_FT) < 1e-6, alt
    assert abs(Y_NEED - 4.3) < 1e-9

    c, r, _ = project(RPTS)
    print("body cols %d..%d  rows %d..%d" % (c.min(), c.max(), r.min(), r.max()))
    assert c.min() >= 0 and c.max() < G.cols, (c.min(), c.max())
    assert r.min() >= 0, r.min()
    assert r.max() <= G.rows + 2, r.max()
    fill = (c.max() - c.min() + 1) / G.cols
    print("width fill %.1f%%   height fill %.1f%%"
          % (100 * fill, 100 * (r.max() - r.min() + 1) / G.rows))
    assert fill > 0.50, fill

    # the ring must be a real ring on screen, and the overrun arc must be big
    rc, rr, _ = project(RRING)
    ok = visible(G, rc, rr)
    rc, rr, ru = rc[ok], rr[ok], RING_U[ok]
    cells = set(zip(rc.tolist(), rr.tolist()))
    over = set(zip(rc[ru > FRAC_RED].tolist(), rr[ru > FRAC_RED].tolist()))
    print("ring cells %d   overrun cells %d (%.1f%%)"
          % (len(cells), len(over), 100 * len(over) / len(cells)))
    assert len(cells) > 190, len(cells)
    assert len(over) >= 40, len(over)
    rad = math.hypot(*[(rc.max() - rc.min()) / 2, (rr.max() - rr.min()) / 2])
    print("ring bbox %d x %d cells" % (rc.max() - rc.min(), rr.max() - rr.min()))
    assert rc.max() - rc.min() >= 36

    # every lens must actually survive to screen as a big blob
    for mm, nm in ((M_RED, "red"), (M_YEL, "yel"), (M_GRN, "grn")):
        sel = MAT == mm
        cc, rrw, zzz = project(RPTS[sel])
        v = visible(G, cc, rrw)
        cells = len(set(zip(cc[v].tolist(), rrw[v].tolist())))
        print("  lens %s: %d cells  rows %d..%d" % (nm, cells, rrw.min(), rrw.max()))
        assert cells > 550, (nm, cells)
        # the shorts UI eats the bottom of the frame: the GREEN lens is the
        # only thing on screen for the first 1.4 s, so its centre has to sit
        # above the safe line or the piece opens on furniture.
        if mm == M_GRN:
            assert (rrw.min() + rrw.max()) / 2 < G.safe_bot, rrw.max()

    # the label may clip the visor tip -- it has a solid halo and reads over
    # it -- but it must clear the red lens entirely.
    _, lr_, _ = project(RPTS[MAT == M_RED])
    _, lh = text_size("45 MPH", 2)
    assert 23 + lh // 2 < lr_.min(), (23 + lh // 2, lr_.min())

    # ink coverage and interior pinholes (trap 6: branching-safe measure)
    for t in (0.6, 2.0, 4.0, 4.6, 6.0, 8.0):
        draw(int(t * FPS))
        inked = LAST_INK
        cov = inked.mean()
        holes = 0
        tot = 0
        for row in inked:
            on = np.flatnonzero(row)
            if len(on) < 2:
                continue
            for a, b in zip(on[:-1], on[1:]):
                if b - a > 1:
                    tot += 1
                    if b - a <= 3:
                        holes += 1
        pin = holes / max(tot, 1)
        print("  t=%.1f  coverage %.3f  pinholes %.3f" % (t, cov, pin))
        assert 0.12 < cov < 0.55, cov
        assert pin < 0.34, pin

    # text inside the safe band and inside the frame
    for s, sc, rc_ in (("45 MPH", 2, 23), ("0.0", 3, LENS_R), ("86 FT", 2, LENS_R)):
        w, h = text_size(s, sc)
        print("  text %-7r %2dx%-2d  rows %d..%d" % (s, w, h, rc_ - h // 2, rc_ - h // 2 + h))
        assert w < G.cols - 4, (s, w)
        assert rc_ - h // 2 >= G.safe_top, (s, rc_ - h // 2, G.safe_top)
        assert rc_ - h // 2 + h <= G.safe_bot, (s, rc_ - h // 2 + h, G.safe_bot)
    print("checks pass")


def dump(t):
    fr = draw(int(t * FPS))
    for row in fr.chars:
        print("".join(row))


def main():
    if "--check" in sys.argv:
        check()
        return
    if "--dump" in sys.argv:
        dump(float(sys.argv[sys.argv.index("--dump") + 1]))
        return
    check()
    out = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
        else "/tmp/yellowlight.mp4"
    import time
    t0 = time.time()
    with Encoder(out, G, fps=FPS) as enc:
        for f in range(FRAMES):
            enc.write(draw(f))
            if f % 30 == 0:
                print("  frame %d/%d" % (f, FRAMES), flush=True)
    print("wrote %s  %d frames  %.1f s  in %.1f s"
          % (out, FRAMES, FRAMES / FPS, time.time() - t0))


if __name__ == "__main__":
    main()
