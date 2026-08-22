#!/usr/bin/env python3
"""THE FLASH GOES UP — the lightning bolt you draw vs the one the sky makes.

Phase A: the bolt you draw (emoji-gold, two bends, revealed top-down: it
"strikes down"). It dissolves. Phase B: a stepped leader stutters DOWN from
the cloud in ~45 m steps, branching as it goes (that is why branches point
down), while an upward streamer rises from the ground to meet it. Phase C:
the return stroke — the only part anyone sees — floods UP the channel, four
times (dart leader down, stroke up), with a STROKES counter. The flicker of
real lightning is 3-4 strokes reusing one channel, 40-50 ms apart.

Facts verified at opened sources (Wikipedia: Lightning; NWS lightning
science pages) — see the description. Time is stretched UNEVENLY and the
description says by how much.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
from asciilib import Encoder, Frame, Grid, contact, ink_lut

G = Grid()
RAMP = ink_lut()
BG = (0.014, 0.009, 0.034)          # deep storm indigo

GOLD = (1.00, 0.78, 0.22)           # the drawn bolt (emoji-gold)
VIOLET = (0.58, 0.44, 0.95)         # leader / dart
WHITE = (0.97, 0.96, 1.00)          # return stroke
LILAC = (0.70, 0.58, 1.00)          # branches
GREEN = (0.55, 1.00, 0.30)          # STROKES counter
CLOUDC = (0.38, 0.34, 0.55)
GROUNDC = (0.32, 0.27, 0.48)

FPS = 30
T_A0, T_A1 = 0.35, 1.15             # drawn bolt reveals top-down
T_AH, T_ADIS = 2.35, 2.85           # hold, then dissolve
T_L0, T_LG = 3.00, 6.35             # stepped leader descends
T_STREAM0, T_STREAM1 = 6.05, 6.35   # upward streamer rises
STROKES = [6.45, 7.75, 8.90, 10.05]
T_UP = [0.50, 0.30, 0.28, 0.26]
PEAK = [1.00, 0.90, 0.85, 0.80]
T_DART = 0.14                        # dart leader sweep (strokes 2-4)
T_FADE0, T_FADE1 = 11.90, 12.25
T_END = 12.40
FRAMES = int(round(T_END * FPS))     # 372

CLOUD_BOT = 10
GROUND = 144                         # < safe_bot 147 (horizon rule)
STREAM_TOP = 136                     # streamer meets leader here
M_PER_ROW = 2000.0 / (GROUND - CLOUD_BOT)   # ~14.9 m/row (stated decision)

rng = np.random.default_rng(452)


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# ---------------------------------------------------------------- drawn bolt
# the emoji glyph: 7 vertices, exactly TWO bends (the notches at rows 62/74)
POLY = [(36.0, 24.0), (58.0, 24.0), (47.0, 62.0), (62.0, 62.0),
        (40.0, 120.0), (46.0, 74.0), (33.0, 74.0)]


def _poly_mask():
    cc, rr = np.meshgrid(np.arange(G.cols, dtype=float),
                         np.arange(G.rows, dtype=float))
    inside = np.zeros(cc.shape, dtype=bool)
    n = len(POLY)
    for i in range(n):
        x1, y1 = POLY[i]
        x2, y2 = POLY[(i + 1) % n]
        cond = ((y1 <= rr) != (y2 <= rr))
        with np.errstate(divide='ignore', invalid='ignore'):
            xs = x1 + (rr - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cond & (cc < xs)
    return inside


BOLT_MASK = _poly_mask()
_e = np.zeros_like(BOLT_MASK)
_e[1:, :] |= ~BOLT_MASK[:-1, :]
_e[:-1, :] |= ~BOLT_MASK[1:, :]
_e[:, 1:] |= ~BOLT_MASK[:, :-1]
_e[:, :-1] |= ~BOLT_MASK[:, 1:]
BOLT_EDGE = BOLT_MASK & _e
BR, BC = np.nonzero(BOLT_MASK)
BOLT_B = np.where(BOLT_EDGE[BR, BC], 1.0,
                  0.84 + rng.normal(0, 0.04, BR.size).clip(-0.10, 0.10))
BOLT_DIE = rng.random(BR.size)        # static dissolve order

# ---------------------------------------------------------------- channel
def _build_channel():
    """Main channel: cloud to ground, stuttering steps of ~3 rows (~45 m)."""
    nodes = [(49.0 + rng.normal(0, 1.0), float(CLOUD_BOT))]
    drift = rng.normal(0, 0.35)
    while nodes[-1][1] < GROUND:
        c, r = nodes[-1]
        dr = float(rng.integers(2, 5))          # 2..4 rows, median 3
        dc = drift * dr + rng.normal(0, 2.0)
        nc = c + dc
        if nc < 24 or nc > 74:                  # reflect off the margins
            drift = -drift
            nc = c - dc
        nodes.append((float(np.clip(nc, 22.0, 76.0)),
                      min(r + dr, float(GROUND))))
        if rng.random() < 0.18:
            drift = rng.normal(0, 0.40)
    return np.array(nodes)


NODES = _build_channel()

# node reveal times: leader part scaled onto [T_L0, T_LG] with pauses,
# streamer part (below STREAM_TOP) revealed upward from the ground
_raw = (1.0 + 3.0 * (rng.random(len(NODES)) < 0.25)) \
       * (1.0 + 0.3 * rng.random(len(NODES)))
_raw = np.cumsum(_raw)
lead = NODES[:, 1] <= STREAM_TOP
_lr = _raw[lead]
NODE_T = np.empty(len(NODES))
NODE_T[lead] = T_L0 + (_lr - _lr[0]) / (_lr[-1] - _lr[0]) * (T_LG - T_L0)
strm = ~lead
NODE_T[strm] = T_STREAM0 + (GROUND - NODES[strm, 1]) \
    / max(GROUND - STREAM_TOP, 1) * (T_STREAM1 - T_STREAM0)


def _sample_polyline(nodes, node_t, rng, step=0.35, jit=0.14):
    pts, tt, ss = [], [], []
    s = 0.0
    for i in range(len(nodes) - 1):
        p0, p1 = nodes[i], nodes[i + 1]
        seg = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        n = max(int(seg / step), 1)
        for k in range(n):
            u = k / n
            pts.append((p0[0] + u * (p1[0] - p0[0]) + rng.normal(0, jit),
                        p0[1] + u * (p1[1] - p0[1]) + rng.normal(0, jit)))
            tt.append(node_t[i] + u * (node_t[i + 1] - node_t[i]))
            ss.append(s + u * seg)
        s += seg
    return np.array(pts), np.array(tt), np.array(ss), s


CH_PTS, CH_T, CH_S, CH_LEN = _sample_polyline(NODES, NODE_T, rng)
CH_NOISE = rng.random(len(CH_PTS)) * 0.10

# branches: spawned at leader nodes, always heading DOWN (they were made by
# a descending leader — that is the whole reason branches point down)
def _build_branches():
    idxs = [i for i in range(2, len(NODES) - 4)
            if 16 < NODES[i, 1] < 118 and NODE_T[i] < T_LG - 0.15]
    rng.shuffle(idxs)
    picked, taken = [], []
    for i in idxs:
        if all(abs(NODES[i, 1] - NODES[j, 1]) > 7 for j in taken):
            picked.append(i)
            taken.append(i)
        if len(picked) >= 12:
            break
    out = []
    for i in picked:
        c, r = NODES[i]
        d = rng.choice([-1, 1]) * rng.uniform(0.8, 2.0)
        L = int(rng.uniform(6, 20))
        nds = [(c, r)]
        for k in range(L):
            c = float(np.clip(c + d + rng.normal(0, 0.55), 4.0, 93.0))
            r += 1.0
            nds.append((c, r))
            if rng.random() < 0.04:
                d *= 0.6
        nds = np.array(nds)
        dt = min(0.05, (T_LG + 0.05 - NODE_T[i]) / max(L, 1))
        nt = NODE_T[i] + np.arange(L + 1) * dt
        p, t, s, _ = _sample_polyline(nds, nt, rng)
        out.append(dict(attach_s=CH_S[np.argmin(np.abs(CH_T - NODE_T[i]))],
                        pts=p, t=t, noise=rng.random(len(p)) * 0.10))
    return out


BRANCHES = _build_branches()

# ---------------------------------------------------------------- cloud/ground
_cbot = (6 + 4 * rng.random(G.cols)).astype(int).clip(2, CLOUD_BOT)
_cl_c, _cl_r, _cl_b = [], [], []
for c in range(G.cols):
    for r in range(_cbot[c]):
        _cl_c.append(c)
        _cl_r.append(r)
        _cl_b.append(0.10 + 0.06 * rng.random())
CLOUD = (np.array(_cl_c), np.array(_cl_r), np.array(_cl_b))
_gr_c, _gr_r = np.meshgrid(np.arange(G.cols), np.arange(GROUND, G.rows))
_gr_b = np.where(_gr_r == GROUND, 0.14, 0.05) \
    * (1 + 0.3 * rng.random(_gr_r.shape))
GROUND_F = (_gr_c.ravel(), _gr_r.ravel(), _gr_b.ravel())

# ---------------------------------------------------------------- text
FONT = {
    '0': "111101101101111", '1': "010110010010111", '2': "111001111100111",
    '3': "111001111001111", '4': "101101111001001", 'S': "111100111001111",
    'T': "111010010010010", 'R': "110101110101101", 'O': "111101101101111",
    'K': "101101110101101", 'E': "111100111100111", ' ': "000000000000000",
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


TEXT_ROW = 130                        # rows 130..139, inside safe 17..147

# ---------------------------------------------------------------- envelopes
def stroke_env(s, t):
    """Brightness multiplier along the trunk (s from cloud top)."""
    b = np.zeros_like(s)
    if t >= STROKES[0]:
        b[:] = 0.11                   # channel stays faintly conductive
    for n, tn in enumerate(STROKES):
        if t < tn:
            break
        if t < tn + T_UP[n]:
            front = CH_LEN * (1.0 - (t - tn) / T_UP[n])
            b = np.maximum(b, np.where(s >= front, PEAK[n], b))
        else:
            b = np.maximum(b, PEAK[n] * np.exp(-(t - tn - T_UP[n]) / 0.16))
    for n in (1, 2, 3):               # dart leaders: down the trunk, fast
        td = STROKES[n] - 0.16
        if td <= t < STROKES[n]:
            front = CH_LEN * np.clip((t - td) / T_DART, 0, 1)
            b = np.maximum(b, np.where(s <= front, 0.50, b))
    return b


def stroke1_at(s_attach, t):
    """Scalar stroke-1 envelope at a branch attachment point."""
    tn = STROKES[0]
    if t < tn:
        return 0.0
    t_pass = tn + T_UP[0] * (1.0 - s_attach / CH_LEN)
    if t < t_pass:
        return 0.0
    return float(np.exp(-max(t - tn - T_UP[0], 0.0) / 0.09))


def flash(t):
    f = 0.0
    for n, tn in enumerate(STROKES):
        if tn <= t:
            if t < tn + T_UP[n] + 0.10:
                f = max(f, PEAK[n])
            else:
                f = max(f, PEAK[n] * np.exp(-(t - tn - T_UP[n] - 0.10) / 0.14))
    return f


def gate(t):
    g = smooth((t - 0.20) / 0.25)
    g *= 1.0 - smooth((t - T_FADE0) / (T_FADE1 - T_FADE0))
    return float(g)


# ---------------------------------------------------------------- field
def _field(fr, cols, rows, bright, colour):
    cc = np.rint(cols).astype(int)
    rr = np.rint(rows).astype(int)
    ok = (cc >= 0) & (cc < G.cols) & (rr >= 0) & (rr < G.rows) & (bright > 0.02)
    cc, rr, bb = cc[ok], rr[ok], bright[ok]
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


def chan_colour(shade, extra=None):
    u = np.clip((shade - 0.55) / 0.45, 0.0, 1.0)
    f = 0.42 + 0.58 * shade
    return ((VIOLET[0] + u * (WHITE[0] - VIOLET[0])) * f,
            (VIOLET[1] + u * (WHITE[1] - VIOLET[1])) * f,
            (VIOLET[2] + u * (WHITE[2] - VIOLET[2])) * f)


# ---------------------------------------------------------------- draw
def reveal_row(t):
    return 24.0 + 98.0 * smooth((t - T_A0) / (T_A1 - T_A0))


def draw(f):
    t = f / FPS
    fr = Frame(G, BG)
    g = gate(t)
    fr.n_ink = 0
    if g <= 0.0:
        return fr
    fl = flash(t)

    # cloud and ground, lit by the flash
    fr.n_ink += _field(fr, CLOUD[0], CLOUD[1],
                       CLOUD[2] * (1 + 1.8 * fl) * g, _tint(CLOUDC))
    fr.n_ink += _field(fr, GROUND_F[0], GROUND_F[1],
                       GROUND_F[2] * (1 + 2.5 * fl) * g, _tint(GROUNDC))

    # phase A: the drawn bolt
    if T_A0 <= t < T_ADIS + 0.1:
        d = smooth((t - T_AH) / (T_ADIS - T_AH))
        vis = (BR <= reveal_row(t)) & (BOLT_DIE > d)
        if vis.any():
            b = BOLT_B[vis] * (1.0 - 0.7 * d) * g
            fr.n_ink += _field(fr, BC[vis].astype(float),
                               BR[vis].astype(float), b, _tint(GOLD))

    # trunk: leader reveal + stroke envelopes
    if t >= T_L0:
        present = CH_T <= t
        if present.any():
            dt_tip = t - CH_T[present]
            leader_b = (0.26 + CH_NOISE[present]) \
                + 0.58 * np.exp(-dt_tip / 0.18)
            env = stroke_env(CH_S[present], t)
            b = np.clip(np.maximum(leader_b * (t < STROKES[0] + 0.02), env),
                        0, 1) * g
            fr.n_ink += _field(fr, CH_PTS[present, 0], CH_PTS[present, 1],
                               b, chan_colour)
            # glare width: only while a stroke is hot, the channel gets fat
            hot = env > 0.30
            if hot.any():
                hc = CH_PTS[present, 0][hot]
                hr = CH_PTS[present, 1][hot]
                he = env[hot]
                for off, wf in ((-1.5, 0.38), (-0.75, 0.78),
                                (0.75, 0.78), (1.5, 0.38)):
                    fr.n_ink += _field(fr, hc + off, hr,
                                       np.clip(he * wf, 0, 1) * g,
                                       chan_colour)
        for br in BRANCHES:
            p = br['t'] <= t
            if not p.any():
                continue
            dt_tip = t - br['t'][p]
            leader_b = (0.20 + br['noise'][p]) \
                + 0.46 * np.exp(-dt_tip / 0.18)
            e1 = 0.55 * stroke1_at(br['attach_s'], t)
            en = 0.12 * max(stroke_env(np.array([br['attach_s']]), t)[0]
                            - 0.11, 0.0) if t > STROKES[0] + T_UP[0] else 0.0
            b = np.clip(np.maximum(leader_b * (t < STROKES[0] + 0.02),
                                   max(e1, en)), 0, 1) * g
            fr.n_ink += _field(fr, br['pts'][p, 0], br['pts'][p, 1],
                               b, _tint(LILAC))

    # counter
    n_shown = sum(1 for tn in STROKES if tn <= t)
    if n_shown and t < T_FADE1:
        rgb = (GREEN[0] * g, GREEN[1] * g, GREEN[2] * g)
        draw_text(fr, f"STROKES {n_shown}", TEXT_ROW, rgb)
    return fr


# ---------------------------------------------------------------- check
def check():
    # the drawing: 7 vertices, two bends, one solid interval per row
    assert len(POLY) == 7
    for r in range(26, 119):
        cs = np.nonzero(BOLT_MASK[r])[0]
        if cs.size:
            assert cs.max() - cs.min() + 1 == cs.size, f"bolt hole row {r}"
    n_fill = int(BOLT_MASK.sum())
    assert n_fill > 900, n_fill
    assert BR.min() >= 22 and BR.max() <= 122
    assert BC.min() >= 3 and BC.max() <= 94
    # phase A reveals DOWNWARD (the drawn lie)
    assert reveal_row(0.6) < reveal_row(0.9) < reveal_row(1.14)

    # channel: descends, spans big, steps ~45 m
    rows = NODES[:, 1]
    assert (np.diff(rows) > 0).all()
    assert rows[0] <= 12 and rows[-1] == GROUND
    steps = np.diff(rows)[:-1]
    med = float(np.median(steps))
    assert 2 <= med <= 4, med
    step_m = med * M_PER_ROW
    assert 30 <= step_m <= 60, step_m
    span = rows[-1] - rows[0]
    assert span >= 120, span
    assert NODES[:, 0].min() >= 3 and NODES[:, 0].max() <= 94

    # branches: enough of them, and every one heads DOWN
    assert len(BRANCHES) >= 8, len(BRANCHES)
    for br in BRANCHES:
        assert br['pts'][-1, 1] > br['pts'][0, 1] + 4

    # return stroke moves UP, dart moves DOWN
    def front_row(tt):
        b = stroke_env(CH_S, tt)
        lit = b > 0.5
        return CH_PTS[lit, 1].min() if lit.any() else 1e9
    t1 = STROKES[0]
    assert front_row(t1 + 0.10) > front_row(t1 + 0.25) > front_row(t1 + 0.45)
    td = STROKES[1] - 0.16

    def dart_bot(tt):
        b = stroke_env(CH_S, tt)
        lit = b > 0.4
        return CH_PTS[lit, 1].max() if lit.any() else -1e9
    assert dart_bot(td + 0.04) < dart_bot(td + 0.12)

    # pop: blazing at stroke peak, faint between strokes (the flicker)
    def peak_cells(tt):
        b = stroke_env(CH_S, tt)
        return int((b >= 0.9).sum())
    n_peak = peak_cells(STROKES[0] + 0.48)
    assert n_peak > 300, n_peak
    b_between = float(stroke_env(CH_S, STROKES[1] - 0.30).max())
    assert b_between <= 0.35, b_between

    # counter text fits and sits in the safe band
    assert _tw("STROKES 4") <= G.cols
    assert 17 <= TEXT_ROW and TEXT_ROW + 5 * SC <= 147
    assert GROUND < 147

    # loop: first and last frames are pure background
    assert draw(0).n_ink == 0
    assert draw(FRAMES - 1).n_ink == 0
    assert draw(int(1.8 * FPS)).n_ink > 0

    # honesty numbers for the description
    chan_m = span * M_PER_ROW
    sweep_x = T_UP[0] / (chan_m / 1.0e8)
    gap_x = (STROKES[1] - STROKES[0]) / 0.045
    print(f"check OK — bolt fill {n_fill} cells, 2 bends; "
          f"channel {span:.0f} rows = {chan_m:.0f} m, median step "
          f"{step_m:.0f} m; {len(BRANCHES)} branches, all downward; "
          f"stroke peak {n_peak} cells >=0.9, between-stroke max "
          f"{b_between:.2f}; sweep stretch {sweep_x:,.0f}x, "
          f"inter-stroke stretch {gap_x:.0f}x; "
          f"{FRAMES} frames, {T_END:.1f}s")


# ---------------------------------------------------------------- outputs
def sheet():
    ts = [0.7, 1.9, 4.8, 6.30, 6.70, 7.88]
    frames = [draw(int(t * FPS)) for t in ts]
    contact(frames, '/tmp/bolt_sheet.png',
            cols=3, labels=[f"t={t}s" for t in ts])
    print("sheet -> /tmp/bolt_sheet.png")


def render():
    out = '/tmp/bolt.mp4'
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
