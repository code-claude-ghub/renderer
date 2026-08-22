#!/usr/bin/env python3
"""THE WATER GOES NOWHERE — draw a wave, you drew its last second.

Act 1: the drawn wave (the curl everyone draws), cobalt crayon, dissolves.
Act 2: a real deep-water wave in REAL TIME (1:1, no slow motion): 12 m long,
       1.5 m high, period 2.77 s from the deep-water dispersion relation.
       A coral floater rides a closed Airy orbit and goes nowhere while a
       CRESTS counter ticks the crests that pass it. Tracers below show the
       orbits shrinking to stillness at half a wavelength down.
Act 3: the bed rises. Crest 3 shoals — shortens, slows, sharpens (k solved
       from omega^2 = g k tanh(kh) at every column) — and overturns where
       H > 0.8 h (the measured criterion). The plunging jet is tinted the
       Act-1 cobalt: the drawn shape exists, for one second, at the end.

Scale: 1 cell = 0.16 m. Everything asserted in check().
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
from asciilib import Encoder, Frame, Grid, contact, ink_lut

G = Grid()
RAMP = ink_lut()
BG = (0.004, 0.016, 0.022)          # deep sea ink

COBALT = (0.22, 0.48, 1.00)         # the drawn wave (crayon blue)
FOAM = (0.97, 0.99, 1.00)
WATERC = (0.10, 0.55, 0.60)         # dim body of the sea
SURFC = (0.72, 0.98, 0.94)          # the moving surface
TRACERC = (0.35, 0.95, 1.00)
CORAL = (1.00, 0.42, 0.25)          # the floater + counter
BEDC = (0.45, 0.36, 0.24)           # rising sand

FPS = 30
CELL_M = 0.16                        # metres per cell
MEAN_ROW = 56.0                      # mean water line
LAM_M = 12.0                         # wavelength, metres
H_M = 1.5                            # wave height, metres
GRAV = 9.81

LAM_C = LAM_M / CELL_M               # 75 cells
A0 = H_M / CELL_M / 2.0              # amplitude in cells (4.69)
K_DEEP = 2 * np.pi / LAM_M           # 1/m
C_PHASE = np.sqrt(GRAV * LAM_M / (2 * np.pi))   # deep water, from source
T_W = LAM_M / C_PHASE                # 2.772 s  — the ANIMATED period too
OMEGA = 2 * np.pi / T_W
H_DEEP = (G.rows - MEAN_ROW) * CELL_M            # 18.9 m of water in frame

# timeline (seconds). REAL TIME: one animated second = one real second.
T_A0, T_A1 = 0.40, 1.60              # drawn wave reveals stroke-wise
T_ADIS0, T_ADIS1 = 3.20, 3.70        # dissolve
T_SEA0, T_SEA1 = 3.50, 4.10          # real sea fades in
T_CREST1 = 4.40                      # first crest passes the floater
T_BED0, T_BED1 = 10.00, 10.60        # bed rises (crest 3 still in deep water)
T_OVD = 0.75                         # overturn duration
T_FADE0, T_FADE1 = 13.90, 14.30
T_END = 14.60
FRAMES = int(round(T_END * FPS))     # 438

FL_X = 30.0                          # floater rest column
BED_X0 = 48.0                        # bed enters frame bottom here
BED_SLOPE = 0.785                    # m of depth lost per cell (steep: plunges)
H_MIN = 0.45

rng = np.random.default_rng(1207)


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# ------------------------------------------------- depth, k(x), phase
def h_of_x(x):
    """Water depth (m) once the bed is fully risen."""
    return np.clip(H_DEEP - BED_SLOPE * (np.asarray(x, float) - BED_X0),
                   H_MIN, H_DEEP)


def solve_k(h):
    """omega^2 = g k tanh(k h).  Deep-water k is the FLOOR (tanh<1)."""
    lo = np.full_like(h, K_DEEP)
    hi = np.full_like(h, 40.0 * K_DEEP)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        too_small = GRAV * mid * np.tanh(mid * h) < OMEGA ** 2
        lo = np.where(too_small, mid, lo)
        hi = np.where(too_small, hi, mid)
    return 0.5 * (lo + hi)


XF = np.arange(0.0, G.cols + 0.001, 0.25)        # fine column grid
H_X = h_of_x(XF)
K_X = solve_k(H_X)                                # 1/m per fine column
C_G = (OMEGA / K_X) * 0.5 * (1 + 2 * K_X * H_X / np.sinh(2 * K_X * H_X))
CG0 = C_PHASE / 2.0
A_X = A0 * np.sqrt(CG0 / C_G)                     # energy-flux shoaling

PHI_DEEP = XF * (K_DEEP * CELL_M)                 # radians at column
PHI_SHOAL = np.concatenate(
    [[0.0], np.cumsum(0.5 * (K_X[1:] + K_X[:-1]) * 0.25 * CELL_M)])
PHI_SHOAL += PHI_DEEP[0] - PHI_SHOAL[0]

# break point: first column where H exceeds 0.8 h
H_LOC = 2 * A_X * CELL_M
_bmask = (XF > BED_X0 + 2) & (H_LOC > 0.8 * H_X)
X_BREAK = float(XF[_bmask][0])
PHI0 = float(np.interp(FL_X, XF, PHI_DEEP))       # crest at floater at T_CREST1

# crest 3 label: passes floater at T_CREST1 + 2*T_W, i.e. theta = -4*pi
PHI_BRK = float(np.interp(X_BREAK, XF, PHI_SHOAL))
T_OV = T_CREST1 + (PHI_BRK - PHI0 + 4 * np.pi) / OMEGA
T_SP = T_OV + 0.62                                # jet hits the trough


def bed_frac(t):
    return float(smooth((t - T_BED0) / (T_BED1 - T_BED0)))


def phase_at(x, t):
    """theta(x,t); crest where theta = 0 mod 2pi."""
    bf = bed_frac(t)
    ph = np.interp(x, XF, PHI_DEEP) * (1 - bf) + np.interp(x, XF, PHI_SHOAL) * bf
    return ph - PHI0 - OMEGA * (t - T_CREST1)


def amp_at(x, t):
    bf = bed_frac(t)
    return A0 * (1 - bf) + np.interp(x, XF, A_X) * bf


def sharp_at(x, t):
    """Crest-sharpening weight over the slope (second harmonic)."""
    h = h_of_x(x) * bed_frac(t) + H_DEEP * (1 - bed_frac(t))
    return np.clip((6.0 - h) / 6.0, 0.0, 1.0)


def surf_row(x, t):
    th = phase_at(x, t)
    a = amp_at(x, t)
    s = sharp_at(x, t)
    return MEAN_ROW - a * (np.cos(th) + 0.35 * s * np.cos(2 * th - 0.4)) \
        + a * 0.12 * s


def bed_row(x, t):
    """Screen row of the sand; below frame when not risen."""
    base = MEAN_ROW + h_of_x(x) / CELL_M
    return base + (1.0 - bed_frac(t)) * 140.0


# ------------------------------------------------- act 1: the drawn wave
def _stroke(pts, n):
    pts = np.array(pts, float)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    u = np.concatenate([[0], np.cumsum(seg)])
    s = np.linspace(0, u[-1], n)
    return np.column_stack([np.interp(s, u, pts[:, 0]),
                            np.interp(s, u, pts[:, 1])])


def _build_drawn():
    back = _stroke([(6, 108), (20, 101), (34, 92), (46, 78),
                    (56, 60), (63, 47)], 170)
    ang = np.linspace(np.radians(150), np.radians(-140), 130)
    rr = np.linspace(15.0, 4.5, 130)
    curl = np.column_stack([70 + rr * np.cos(ang), 60 - rr * np.sin(ang)])
    sc = []
    for cx, w in ((40, 9), (56, 8), (70, 7)):
        aa = np.linspace(np.pi, 2 * np.pi, 26)
        sc.append(np.column_stack([cx + w * np.cos(aa),
                                   96 + 4.5 * np.sin(aa)]))
    base = _stroke([(6, 110), (30, 113), (55, 110), (80, 113), (92, 111)], 110)
    strokes = [back, curl] + sc + [base]
    white = [False, True, True, True, True, False]
    pts, col_w, dist = [], [], []
    d = 0.0
    for st, wht in zip(strokes, white):
        seg = np.hypot(*np.diff(st, axis=0).T)
        dd = d + np.concatenate([[0], np.cumsum(seg)])
        pts.append(st)
        dist.append(dd)
        col_w.append(np.full(len(st), wht))
        d = dd[-1]
    return (np.vstack(pts), np.concatenate(dist),
            np.concatenate(col_w), d)


DR_PTS, DR_D, DR_WHT, DR_LEN = _build_drawn()
DR_DIE = rng.random(len(DR_PTS))

# thickness offsets for the crayon stroke
DR_OFF = [(0, 0, 1.0), (0.7, 0, 0.75), (0, 0.7, 0.75), (-0.7, 0, 0.5)]

# ------------------------------------------------- act 2 constants
TR_X = (12.0, 32.0, 52.0)
TR_D = (4.0, 14.0, 30.0, 48.0)       # depth below mean, rows
K_CELL = K_DEEP * CELL_M
TR_R = [A0 * np.exp(-K_CELL * d) for d in TR_D]
RING_A = np.linspace(0, 2 * np.pi, 40, endpoint=False)

# static water jitter (computed once — never boils)
WJIT = rng.random((G.rows, G.cols))

# splash particles (born at T_SP from the jet tip)
N_SPL = 130
SPL_V = np.column_stack([rng.normal(9, 7, N_SPL),
                         -np.abs(rng.normal(16, 7, N_SPL))])
SPL_LIFE = 0.45 + 0.5 * rng.random(N_SPL)

# ------------------------------------------------- text
FONT = {
    '0': "111101101101111", '1': "010110010010111", '2': "111001111100111",
    '3': "111001111001111", '4': "101101111001001", 'S': "111100111001111",
    'T': "111010010010010", 'R': "110101110101101", 'E': "111100111100111",
    'C': "111100100100111", ' ': "000000000000000",
}
SC = 2


def _tw(s):
    return len(s) * 3 * SC + (len(s) - 1) * SC


def draw_text(fr, s, row0, rgb):
    col0 = (G.cols - _tw(s)) // 2
    c = col0
    for ch in s:
        bm = FONT[ch]
        for r in range(5):
            for k in range(3):
                if bm[r * 3 + k] == '1':
                    for dr in range(SC):
                        for dc in range(SC):
                            fr.put(c + k * SC + dc, row0 + r * SC + dr,
                                   '#', rgb)
        c += 4 * SC
    return col0


TEXT_ROW = 128                        # rows 128..137, inside safe 17..147


def crest_count(t):
    if t < T_CREST1:
        return 0
    return 1 + int((t - T_CREST1) / T_W)


def gate(t):
    g = smooth((t - 0.20) / 0.30)
    g *= 1.0 - smooth((t - T_FADE0) / (T_FADE1 - T_FADE0))
    return float(g)


# ------------------------------------------------- field helpers
def _field(fr, cols, rows, bright, colour):
    cc = np.rint(np.asarray(cols, float)).astype(int)
    rr = np.rint(np.asarray(rows, float)).astype(int)
    bb = np.asarray(bright, float)
    ok = (cc >= 0) & (cc < G.cols) & (rr >= 0) & (rr < G.rows) & (bb > 0.02)
    cc, rr, bb = cc[ok], rr[ok], bb[ok]
    if cc.size == 0:
        return 0
    order = np.argsort(-bb)
    cc, rr, bb = cc[order], rr[order], bb[order]
    flat = rr * G.cols + cc
    _, first = np.unique(flat, return_index=True)
    cc, rr, bb = cc[first], rr[first], bb[first]
    fr.field(cc, rr, np.ones(cc.size, dtype=bool), bb, colour, RAMP)
    return cc.size


def _tint(base):
    def colour(shade, extra=None):
        f = 0.40 + 0.60 * shade
        return (base[0] * f, base[1] * f, base[2] * f)
    return colour


def _lerp_col(c1, c2, u):
    return tuple(a + (b - a) * u for a, b in zip(c1, c2))


C_WATER = _tint(WATERC)
C_SURF = _tint(SURFC)
C_TRACER = _tint(TRACERC)
C_CORAL = _tint(CORAL)
C_BED = _tint(BEDC)
C_COBALT = _tint(COBALT)
C_FOAM = _tint(FOAM)


def floater_pos(t):
    th = float(phase_at(FL_X, t))
    return (FL_X - A0 * np.sin(th), MEAN_ROW - A0 * np.cos(th) - 1.2)


def curl_state(t):
    """(p, anchor_col, crest_row, R) of the overturning jet."""
    p = float(smooth((t - T_OV) / T_OVD))
    xa = X_BREAK + 7.0 * p
    cr = float(surf_row(np.array([xa]), t)[0])
    return p, xa, cr, 7.0


# ------------------------------------------------- draw
def draw(f):
    t = f / FPS
    g = gate(t)
    fr = Frame(G, BG)
    fr.n_ink = 0
    if g <= 0.0:
        return fr

    sea = float(smooth((t - T_SEA0) / (T_SEA1 - T_SEA0)))

    if sea > 0.0:
        xs = np.arange(G.cols, dtype=float)
        sr = surf_row(xs, t)
        br = bed_row(xs, t)
        # water body — dim, static jitter, darker with depth
        cols_l, rows_l, b_l = [], [], []
        for x in range(G.cols):
            top = int(np.ceil(sr[x])) + 1
            bot = min(int(br[x]), G.rows)
            if bot <= top:
                continue
            rr = np.arange(top, bot)
            bb = (0.065 + 0.035 * WJIT[rr, x]) \
                * (1.0 - 0.45 * (rr - MEAN_ROW) / (G.rows - MEAN_ROW))
            cols_l.append(np.full(rr.size, float(x)))
            rows_l.append(rr.astype(float))
            b_l.append(bb)
        if cols_l:
            fr.n_ink += _field(fr, np.concatenate(cols_l),
                               np.concatenate(rows_l),
                               np.concatenate(b_l) * sea * g, C_WATER)
        # bed
        for x in range(int(BED_X0) - 2, G.cols):
            top = int(np.ceil(br[x]))
            if top >= G.rows:
                continue
            rr = np.arange(max(top, 0), G.rows)
            bb = np.where(rr == max(top, 0), 0.55, 0.14 + 0.10 * WJIT[rr, x])
            fr.n_ink += _field(fr, np.full(rr.size, float(x)), rr.astype(float),
                               bb * g, C_BED)
        # orbit rings + tracers
        for x0 in TR_X:
            for d, r in zip(TR_D, TR_R):
                cy = MEAN_ROW + d
                if r > 0.3:
                    fr.n_ink += _field(fr, x0 + r * np.cos(RING_A),
                                       cy + r * np.sin(RING_A),
                                       np.full(RING_A.size, 0.12) * sea * g,
                                       C_TRACER)
                th = float(phase_at(x0, t))
                tc = x0 - r * np.sin(th)
                trr = cy - r * np.cos(th)
                fr.n_ink += _field(fr, np.array([tc, tc + 0.8, tc, tc - 0.8]),
                                   np.array([trr, trr, trr + 0.8, trr]),
                                   np.array([0.95, 0.5, 0.5, 0.5]) * sea * g,
                                   C_TRACER)
        # surface
        xs2 = np.arange(0, G.cols, 0.34)
        sr2 = surf_row(xs2, t)
        fr.n_ink += _field(fr, xs2, sr2,
                           np.full(xs2.size, 0.95) * sea * g, C_SURF)
        fr.n_ink += _field(fr, xs2, sr2 + 1.0,
                           np.full(xs2.size, 0.40) * sea * g, C_SURF)
        # floater trail (its closed orbit, drawn as history)
        tt = np.linspace(max(T_SEA0, t - 2.2), t, 46)
        if tt[-1] - tt[0] > 0.1:
            tr_pts = np.array([floater_pos(x) for x in tt])
            fade = np.linspace(0.10, 0.45, tt.size)
            fr.n_ink += _field(fr, tr_pts[:, 0], tr_pts[:, 1],
                               fade * sea * g, C_CORAL)
        # floater
        fc, frow = floater_pos(t)
        offs = [(0, 0), (1, 0), (0, -1), (1, -1), (0.5, -2)]
        fr.n_ink += _field(fr, np.array([fc + o[0] for o in offs]),
                           np.array([frow + o[1] for o in offs]),
                           np.full(len(offs), 1.0) * sea * g, C_CORAL)

        # the overturning jet (act 3)
        p, xa, cr, R = curl_state(t)
        if p > 0.0 and t < T_SP + 0.35:
            cx, cy = xa + R * 0.9, cr + R * 0.55
            nseg = max(int(90 * p), 4)
            aa = np.radians(np.linspace(125, 125 - 250 * p, nseg))
            rj = R * np.linspace(1.0, 0.75, nseg)
            u = np.linspace(0, 1, nseg)
            for dx, dy, w in DR_OFF:
                jc = cx + rj * np.cos(aa) + dx
                jr = cy - rj * np.sin(aa) + dy
                bb = (0.65 + 0.35 * u) * w
                # cobalt at the root -> foam at the flying tip
                for i0 in range(0, nseg, 12):
                    i1 = min(i0 + 12, nseg)
                    col = _lerp_col(COBALT, FOAM, float(u[i0:i1].mean()))
                    fr.n_ink += _field(fr, jc[i0:i1], jr[i0:i1],
                                       bb[i0:i1] * g, _tint(col))
            # foam cap on the crest
            capx = xa + np.linspace(-4, 2, 14)
            fr.n_ink += _field(fr, capx, surf_row(capx, t) - 0.5,
                               np.full(14, 0.9 * p) * g, C_FOAM)
        # splash
        if T_SP <= t < T_SP + 1.1:
            ts = t - T_SP
            _, xa2, cr2, R2 = curl_state(T_SP)
            ix, ir = xa2 + R2 * 0.4, cr2 + R2 * 1.1
            px = ix + SPL_V[:, 0] * ts
            pr = ir + SPL_V[:, 1] * ts + 0.5 * 38.0 * ts * ts
            alive = (ts < SPL_LIFE)
            bb = np.where(alive, 0.9 * (1 - ts / SPL_LIFE), 0.0)
            fr.n_ink += _field(fr, px, pr, bb * g, C_FOAM)
            # foam wash spreading on the surface
            w = 4 + 26 * ts
            wx = np.linspace(ix - w, ix + w * 0.5, 40)
            fr.n_ink += _field(fr, wx, surf_row(wx, t) - 0.3,
                               np.full(40, max(0.65 * (1 - ts / 1.1), 0)) * g,
                               C_FOAM)

    # act 1: the drawn wave
    if t < T_ADIS1:
        rev = smooth((t - T_A0) / (T_A1 - T_A0)) * DR_LEN
        die = smooth((t - T_ADIS0) / (T_ADIS1 - T_ADIS0))
        vis = (DR_D <= rev) & (DR_DIE > die)
        if vis.any():
            for dx, dy, w in DR_OFF:
                for wht in (False, True):
                    m = vis & (DR_WHT == wht)
                    if not m.any():
                        continue
                    pp = DR_PTS[m]
                    bsel = np.where((rev - DR_D[m]) < 6.0, 1.0, 0.82)
                    fr.n_ink += _field(fr, pp[:, 0] + dx, pp[:, 1] + dy,
                                       bsel * w * g,
                                       C_FOAM if wht else C_COBALT)

    # counter
    if sea > 0.2 and g > 0.05:
        n = crest_count(t)
        cg = min(sea, g)
        draw_text(fr, f"CRESTS {n}", TEXT_ROW,
                  tuple(c * cg for c in CORAL))
    return fr


# ------------------------------------------------- check
def check():
    # dispersion honest: omega^2 = g k in deep water
    assert abs(OMEGA ** 2 - GRAV * K_DEEP) < 1e-6 * GRAV * K_DEEP
    # REAL TIME: animated phase speed (cells/s) x cell size == c from source
    c_anim = (LAM_C / T_W) * CELL_M
    assert abs(c_anim - C_PHASE) < 1e-9, c_anim
    # physical steepness (deep-water limit 0.17 from source)
    assert H_M / LAM_M < 0.17, H_M / LAM_M
    # genuinely deep water left of the bed: frame depth > lambda/2
    assert H_DEEP > LAM_M / 2, H_DEEP
    # k(x): deep-water k is the floor, monotone up as h falls (kelp lesson)
    assert (K_X >= K_DEEP - 1e-12).all()
    sl = XF > BED_X0 + 1
    assert (np.diff(K_X[sl]) >= -1e-12).all()
    kb = float(np.interp(X_BREAK, XF, K_X))
    lam_b = 2 * np.pi / kb
    # break point: criterion crossed exactly there, not 5 cells earlier
    hb = float(h_of_x(X_BREAK))
    Hb = float(np.interp(X_BREAK, XF, H_LOC))
    assert Hb > 0.8 * hb
    x5 = X_BREAK - 5
    assert float(np.interp(x5, XF, H_LOC)) <= 0.8 * float(h_of_x(x5))
    assert 62 <= X_BREAK <= 78, X_BREAK
    # timeline sanity: bed fully up before crest 3 enters the slope
    t_c3_slope = T_CREST1 + 2 * T_W \
        + (float(np.interp(BED_X0, XF, PHI_DEEP)) - PHI0) / OMEGA
    assert T_BED1 < t_c3_slope + 0.15, (T_BED1, t_c3_slope)
    assert T_SP + 1.1 < T_FADE0 + 0.6
    # floater: closed orbit — net drift over 2 whole periods = 0
    t0 = 5.0
    p_a = floater_pos(t0)
    p_b = floater_pos(t0 + 2 * T_W)
    drift = np.hypot(p_a[0] - p_b[0], p_a[1] - p_b[1])
    assert drift < 0.02, drift
    # and the orbit is VISIBLE: horizontal excursion = 2*A0 cells
    ths = np.linspace(0, 2 * np.pi, 200)
    exc = np.ptp(A0 * np.sin(ths))
    assert exc > 8.0, exc
    # payoff visible: top tracer orbit reads as a circle, bottom as a point
    assert 2 * TR_R[0] > 6.0, TR_R
    assert 2 * TR_R[-1] < 0.5, TR_R
    # deepest tracer sits above the text
    assert MEAN_ROW + TR_D[-1] + 2 < TEXT_ROW
    # counter: 3 crests before the break, 4 by the end
    assert crest_count(T_OV - 0.01) == 3
    assert crest_count(T_END - 0.5) == 4
    # crest 3 breaks: T_OV is between its floater passage and passage 4
    assert T_CREST1 + 2 * T_W < T_OV < T_CREST1 + 3 * T_W
    # text safe band
    st, sb = int(G.rows * 0.10), int(G.rows * 0.85)
    assert st <= TEXT_ROW and TEXT_ROW + 10 <= sb
    assert _tw("CRESTS 4") <= G.cols
    # drawn wave: inside frame, reveal monotone along distance
    assert DR_PTS[:, 0].min() > 2 and DR_PTS[:, 0].max() < 96
    assert DR_PTS[:, 1].min() > 40 and DR_PTS[:, 1].max() < 118
    assert (np.diff(DR_D) >= -1e-9).all()
    # jet: tip descends below the crest and stays near the break column
    p1, xa1, cr1, R1 = curl_state(T_OV + T_OVD)
    aa_end = np.radians(125 - 250 * p1)
    tip_row = (cr1 + R1 * 0.55) - R1 * 0.75 * np.sin(aa_end)
    assert tip_row > cr1 + 3.5, (tip_row, cr1)
    assert X_BREAK - 2 <= xa1 <= X_BREAK + 12
    # loop through black
    assert draw(0).n_ink == 0
    assert draw(FRAMES - 1).n_ink == 0
    assert draw(int(6.0 * FPS)).n_ink > 0
    print(f"scale 1 cell = {CELL_M} m; wave {LAM_M} m x {H_M} m, "
          f"T = {T_W:.3f} s, c = {C_PHASE:.2f} m/s — REAL TIME 1:1")
    print(f"orbital speed pi*H/T = {np.pi * H_M / T_W:.2f} m/s "
          f"({C_PHASE / (np.pi * H_M / T_W):.1f}x slower than the shape)")
    print(f"floater drift over 2 periods: {drift:.4f} cells (closed orbit)")
    print(f"orbit diameters top->bottom: "
          + ", ".join(f"{2 * r:.1f}" for r in TR_R) + " cells")
    print(f"break at x = {X_BREAK:.1f}: H {Hb:.2f} m > 0.8 x h {hb:.2f} m; "
          f"k*h = {kb * hb:.2f}, wavelength {LAM_M:.0f} -> {lam_b:.1f} m")
    print(f"T_OV = {T_OV:.2f} s (crest 3), splash {T_SP:.2f}, "
          f"end {T_END} s, {FRAMES} frames")
    print("check OK")


# ------------------------------------------------- output
def sheet():
    ts = [0.9, 2.6, 5.2, 8.0, 11.4, float(T_OV + 0.55)]
    frames = [draw(int(t * FPS)) for t in ts]
    contact(frames, '/tmp/wave_sheet.png', cols=3,
            labels=[f"t={t:.2f}" for t in ts])
    print("sheet -> /tmp/wave_sheet.png")


def render():
    out = '/tmp/wave.mp4'
    with Encoder(out, G, fps=FPS) as enc:
        for f in range(FRAMES):
            enc.write(draw(f))
    print(f"render -> {out}")


if __name__ == '__main__':
    check()
    if 'sheet' in sys.argv:
        sheet()
    if 'render' in sys.argv:
        render()
