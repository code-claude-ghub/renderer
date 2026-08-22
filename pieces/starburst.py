#!/usr/bin/env python3
"""THE SPIKES ARE YOURS — Sirius recedes to a point; your eye adds the points.

MEASURED (opened sources, quoted in the description):
  - Sirius: angular diameter 5.936 +/- 0.016 mas = 0.005936 arcsec, distance
    8.61 +/- 0.03 ly, radius 1.7144 R_sun  (Wikipedia: Sirius)
  - naked-eye angular resolution ~1 arcminute = 60 arcsec (Wikipedia: Naked eye)
    -> Sirius arrives 10,108x smaller than the smallest shape the eye resolves
  - the spikes are entoptic: diffraction through eyelashes / eyelid edges plus
    fibers in the lens called suture lines; after a blink the lashes reseat and
    the spikes JUMP around (Wikipedia: Diffraction spike)
  - the suture pattern is unique per person — "everybody sees their own stars"
    (wonderdome.co.uk, pointy-stars-diffraction-spikes-explained)

DRAWN (declared in the description):
  - the shrink is log-compressed so it can be watched (true shrink 300,000x);
    the on-screen arcsec counter is the real number at every moment
  - spike geometry is schematic: 6 stable suture spikes + fine lash streaks
    that jump at each blink; a real pattern is yours alone
"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
import numpy as np
from asciilib import Grid, Frame, Encoder, ink_lut, contact

G = Grid()
RAMP = ink_lut()
BG = (0.005, 0.006, 0.011)                      # space black

WHITE = (0.97, 0.98, 1.00)                      # Sirius A1V blue-white
BLUE  = (0.60, 0.71, 1.00)
ICE   = (0.72, 0.82, 1.00)                      # spike tips
CRIM  = (0.96, 0.28, 0.30)                      # the eye's line
LIDEDGE = (0.14, 0.115, 0.15)                   # lid rim highlight

FPS = 30
T_HOLD   = 0.9                                  # sphere big, turning
T_REC    = 6.3                                  # recession ends: a point
T_POINT  = 7.8                                  # bare point, twinkling
T_BLOOM  = 9.6                                  # spikes grown (config A)
T_BLINK1 = 10.4                                 # -> config B
T_BLINK2 = 12.9                                 # -> config C
T_CLOSE  = 15.2                                 # lids close and stay
T_END    = 15.8
FRAMES   = int(round(T_END * FPS))              # 474
BLINK_D  = 0.5                                  # close .18 / shut .12 / open .20

THETA0, THETA1 = 1800.0, 0.005936               # arcsec: start (1.8 AU) -> real
EYE_LIM = 60.0                                  # arcsec, ~1 arcminute
CX, CY = G.cols // 2, 58                        # star centre cell
R0_PX, RMIN_PX = 46.0, 0.7                      # drawn radius, log-compressed
ROW_STAR, ROW_EYE = 118, 130                    # text rows (safe 17..147)

# ---------------------------------------------------------------- timeline
def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)

def recede_u(t):
    """0 at full size, 1 at point. Ease-in: the leaving accelerates."""
    if t <= T_HOLD:
        return 0.0
    if t >= T_REC:
        return 1.0
    q = (t - T_HOLD) / (T_REC - T_HOLD)
    return q ** 1.35

def theta_of(t):
    return THETA0 * (THETA1 / THETA0) ** recede_u(t)

def sphere_px(t):
    return R0_PX * (RMIN_PX / R0_PX) ** recede_u(t)

def fmt_theta(th):
    if th >= 10:
        return str(int(round(th)))
    if th >= 1:
        return f"{th:.1f}"
    return f"{th:.3f}"

def bloom_g(t):
    """Spike growth 0->1 across the bloom."""
    if t <= T_POINT:
        return 0.0
    return _smooth((t - T_POINT) / (T_BLOOM - T_POINT))

def blink_c(t):
    """Lid closure 0..1. Two blinks plus the final close."""
    for t0 in (T_BLINK1, T_BLINK2):
        dt = t - t0
        if 0 <= dt < BLINK_D:
            if dt < 0.18:
                return _smooth(dt / 0.18)
            if dt < 0.30:
                return 1.0
            return 1.0 - _smooth((dt - 0.30) / 0.20)
    if t >= T_CLOSE:
        return _smooth((t - T_CLOSE) / 0.25)
    return 0.0

def config_of(t):
    if t < T_BLINK1 + 0.24:
        return 0
    if t < T_BLINK2 + 0.24:
        return 1
    return 2

# ---------------------------------------------------------------- twinkle
_rng_tw = np.random.default_rng(5936)
_raw = _rng_tw.normal(0, 1, FRAMES + 8)
_k = np.exp(-0.5 * (np.arange(-6, 7) / 2.4) ** 2)
_sm = np.convolve(_raw, _k / _k.sum(), mode='same')[:FRAMES]
_base_tw = _sm / max(1e-9, np.abs(_sm).max())
# scintillation is weak while the disc is resolved and strong once it is a
# point — which is the true reason stars twinkle and planets barely do
_amp = np.array([0.06 + 0.16 * _smooth((f / FPS - T_HOLD) / (T_REC - T_HOLD))
                 for f in range(FRAMES)])
TWINKLE = 1.0 + _amp * _base_tw                   # bounded 0.78..1.22

# ---------------------------------------------------------------- the sphere
_rng_s = np.random.default_rng(5936)
_NU, _NP = 380, 240
_u = (np.arange(_NU)[:, None] + _rng_s.random((_NU, _NP))) / _NU   # cos(theta)
_p = (np.arange(_NP)[None, :] + _rng_s.random((_NU, _NP))) / _NP
_cu = 2 * _u.ravel() - 1
_su = np.sqrt(np.clip(1 - _cu ** 2, 0, 1))
_ph = 2 * np.pi * _p.ravel()
SPH = np.stack([_su * np.cos(_ph), _cu, _su * np.sin(_ph)], axis=1)
GRAN = np.clip(_rng_s.normal(0, 1, len(SPH)) * 0.10, -0.18, 0.18)  # static, rotates
PITCH = 0.20

def sphere_pts(t):
    yaw = 0.30 * t
    cy_, sy_ = np.cos(yaw), np.sin(yaw)
    x = SPH[:, 0] * cy_ + SPH[:, 2] * sy_
    z = -SPH[:, 0] * sy_ + SPH[:, 2] * cy_
    y = SPH[:, 1]
    cp, sp = np.cos(PITCH), np.sin(PITCH)
    y, z = y * cp - z * sp, y * sp + z * cp
    vis = z > 0.02
    return x[vis], y[vis], z[vis], GRAN[vis]

# ---------------------------------------------------------------- the burst
# 6 suture spikes: SAME in every config (the lens does not move).
_rng_b = np.random.default_rng(8610)
SUT_ANG = np.deg2rad(np.array([30, 90, 150, 210, 270, 330])
                     + _rng_b.uniform(-9, 9, 6))

def _max_len(ang):
    """Longest spike that keeps its endpoint inside col 3..94, row 6..111."""
    dx, dy = np.cos(ang), np.sin(ang)
    lim = 1e9
    if dx > 1e-9:
        lim = min(lim, (94 - CX) / dx)
    if dx < -1e-9:
        lim = min(lim, (3 - CX) / dx)
    if dy > 1e-9:
        lim = min(lim, (111 - CY) / dy)
    if dy < -1e-9:
        lim = min(lim, (6 - CY) / dy)
    return lim

SUT_LEN = np.array([min(_rng_b.uniform(40, 56), _max_len(a) - 1.5)
                    for a in SUT_ANG])
SUT_BRT = _rng_b.uniform(0.95, 1.0, 6)

def _spike_samples(ang, length, brt, width_sub, rng, fall_p):
    """Sample points along one spike: (col, row, r_along, base_bright)."""
    rs = np.arange(1.5, length, 0.33)
    out = []
    offsets = [0.0] if not width_sub else [0.0, 0.5, -0.5]
    for k, off in enumerate(offsets):
        w = 1.0 if k == 0 else 0.5
        jit = rng.normal(0, 0.14, len(rs))
        c = CX + np.cos(ang) * rs - np.sin(ang) * (off + jit)
        r = CY + np.sin(ang) * rs + np.cos(ang) * (off + jit)
        fall = (1.0 - rs / length) ** fall_p
        b = brt * w * fall
        out.append(np.stack([c, r, rs / length, b], axis=1))
    return np.concatenate(out)

def _build_config(seed):
    """Suture spikes (shared geometry) + this blink's lash streaks."""
    rng = np.random.default_rng(seed)
    parts, spike_id = [], []
    for i in range(6):
        s = _spike_samples(SUT_ANG[i], SUT_LEN[i], SUT_BRT[i], True, rng, 0.7)
        parts.append(s)
        spike_id.append(np.full(len(s), i))
    n_lash = rng.integers(21, 28)
    lash_ang = []
    for j in range(n_lash):
        base = 90.0 if rng.random() < 0.5 else 270.0
        a = np.deg2rad(base + rng.normal(0, 20))
        lash_ang.append(np.rad2deg(a) % 360)
        L = rng.uniform(13, 32)
        s = _spike_samples(a, L, rng.uniform(0.42, 0.68), False, rng, 1.0)
        parts.append(s)
        spike_id.append(np.full(len(s), 6 + j))
    # halo speckle
    n_h = 130
    hr = rng.uniform(3.5, 10.5, n_h)
    ha = rng.uniform(0, 2 * np.pi, n_h)
    halo = np.stack([CX + hr * np.cos(ha), CY + hr * np.sin(ha),
                     np.full(n_h, 0.12), rng.uniform(0.12, 0.32, n_h)], axis=1)
    parts.append(halo)
    spike_id.append(np.full(n_h, -1))
    pts = np.concatenate(parts)
    sid = np.concatenate(spike_id).astype(int)
    n_spk = 6 + n_lash
    flick_w = rng.uniform(4.0, 9.0, n_spk)
    flick_p = rng.uniform(0, 2 * np.pi, n_spk)
    return dict(pts=pts, sid=sid, n_spk=n_spk, n_lash=n_lash,
                flick_w=flick_w, flick_p=flick_p,
                lash_ang=np.array(sorted(lash_ang)))

CONFIGS = [_build_config(s) for s in (11, 22, 33)]

CORE = np.array(
    [[CX, CY, 0.0, 1.00],
     [CX + 1, CY, 0.0, 0.60], [CX - 1, CY, 0.0, 0.60],
     [CX, CY + 1, 0.0, 0.60], [CX, CY - 1, 0.0, 0.60],
     [CX + 1, CY + 1, 0.0, 0.30], [CX - 1, CY - 1, 0.0, 0.30],
     [CX + 1, CY - 1, 0.0, 0.30], [CX - 1, CY + 1, 0.0, 0.30],
     [CX + 2, CY, 0.0, 0.14], [CX - 2, CY, 0.0, 0.14],
     [CX, CY + 2, 0.0, 0.14], [CX, CY - 2, 0.0, 0.14]])

# ---------------------------------------------------------------- text
FONT = {
    '0': "111101101101111", '1': "010110010010111", '2': "111001111100111",
    '3': "111001111001111", '4': "101101111001001", '5': "111100111001111",
    '6': "111100111101111", '7': "111001001010010", '8': "111101111101111",
    '9': "111101111001111", 'A': "010101111101101", 'R': "110101110101101",
    'S': "111100111001111", 'T': "111010010010010", 'E': "111100111100111",
    'Y': "101101010010010", 'O': "111101101101111", 'U': "101101101101111",
    '.': "000000000000010", '"': "101101000000000", ' ': "000000000000000",
}
SC = 2

def _tw(s):
    return len(s) * 3 * SC + (len(s) - 1) * SC

def draw_text(fr, s, row0, rgb, hidden=None):
    col0 = (G.cols - _tw(s)) // 2
    c = col0
    for ch in s:
        bm = FONT[ch]
        for r in range(5):
            for k in range(3):
                if bm[r * 3 + k] == '1':
                    for dr in range(SC):
                        for dc in range(SC):
                            cc, rr = c + k * SC + dc, row0 + r * SC + dr
                            if hidden is not None and hidden(cc, rr):
                                continue
                            fr.put(cc, rr, '#', rgb)
        c += 4 * SC
    return col0

# ---------------------------------------------------------------- crossing
def _find_cross():
    ts = np.linspace(T_HOLD, T_REC, 4000)
    th = np.array([theta_of(t) for t in ts])
    i = int(np.argmax(th <= EYE_LIM))
    return float(ts[i])

T_CROSS = _find_cross()

# ---------------------------------------------------------------- draw
def _colour_star(shade, extra):
    b = np.clip(shade, 0, 1)
    base = tuple(w + (v - w) * (1 - b) * 0.55 for w, v in zip(WHITE, BLUE))
    f = 0.55 + 0.45 * b
    return (base[0] * f, base[1] * f, base[2] * f)

def _colour_spike(shade, extra):
    b = np.clip(shade, 0, 1)
    base = tuple(i + (w - i) * b for i, w in zip(ICE, WHITE))
    f = 0.55 + 0.45 * b
    return (base[0] * f, base[1] * f, base[2] * f)

def _field(fr, cols, rows, bright, colour, te=None, be=None):
    c = np.rint(cols).astype(int)
    r = np.rint(rows).astype(int)
    ok = (c >= 0) & (c < G.cols) & (r >= 1) & (r < G.rows - 1)
    if te is not None:
        cc = np.clip(c, 0, G.cols - 1)
        ok &= (r >= te[cc]) & (r <= be[cc])         # lids mask the world out
    c, r, b = c[ok], r[ok], np.clip(bright[ok], 0, 1)
    if len(c) == 0:
        return None
    flat = r * G.cols + c
    best = np.full(G.rows * G.cols, -1.0)
    np.maximum.at(best, flat, b)
    keep = b >= best[flat] - 1e-12
    # one sample per cell: the brightest
    seen = np.zeros(G.rows * G.cols, bool)
    order = np.argsort(-b)
    sel = np.zeros(len(c), bool)
    for i in order:
        f_ = flat[i]
        if not seen[f_]:
            seen[f_] = True
            sel[i] = True
    idx, _ = fr.field(c[sel], r[sel], np.ones(sel.sum(), bool),
                      b[sel], colour, RAMP)
    return idx

def lid_edges(c):
    """Per-column (top_edge, bottom_edge): covered is row < te or row > be.

    Glyphs leak (a dark '#' over a bright cell ghosts), so the lid is not
    painted OVER the scene — the scene is masked out and the lid stays black
    with a visible rim.
    """
    cols = np.arange(G.cols)
    bulge = 10.0 * ((cols - CX) / CX) ** 2
    te = (c * (CY + 10) - (1 - c) * 6 - bulge * (1 - c * 0.7)).astype(int)
    be = (G.rows - 2 - c * (G.rows - 2 - (CY + 12))
          + bulge * (1 - c * 0.7)).astype(int)
    if c >= 0.97:
        te = np.full(G.cols, G.rows)
        be = np.full(G.cols, -1)
    return te, be

def draw(f):
    t = f / FPS
    tw = TWINKLE[min(f, FRAMES - 1)]
    fr = Frame(G, BG)
    idx = None
    c_lid = blink_c(t)
    te, be = lid_edges(c_lid) if c_lid > 0 else (None, None)

    def hidden(cc, rr):
        if te is None:
            return False
        return rr < te[cc] or rr > be[cc]

    s_px = sphere_px(t)
    if c_lid >= 0.97:
        pass                                       # eyes shut: black frame
    elif t < T_REC and s_px >= 1.6:
        x, y, z, gn = sphere_pts(t)
        mu = np.clip(z, 0, 1)
        shade = (0.4 + 0.6 * mu) * (1 + gn) * tw          # limb darkening u=0.6
        idx = _field(fr, CX + x * s_px, CY + y * s_px,
                     np.clip(shade, 0.04, 1), _colour_star, te, be)
    else:
        g = bloom_g(t)
        cfg = CONFIGS[config_of(t)]
        pts, sid = cfg['pts'], cfg['sid']
        grown = pts[:, 2] <= max(g, 1e-9)
        if g <= 0:
            grown = sid == -2                              # nothing but core
        p = pts[grown]
        s_ = sid[grown]
        flick = np.ones(len(p))
        m = s_ >= 0
        flick[m] = 0.86 + 0.14 * np.sin(cfg['flick_w'][s_[m]] * t
                                        + cfg['flick_p'][s_[m]])
        halo_gate = _smooth(g * 3)
        hm = s_ == -1
        bright = p[:, 3] * flick * tw
        bright[hm] *= halo_gate
        core_b = CORE[:, 3] * tw
        allc = np.concatenate([p[:, 0], CORE[:, 0]])
        allr = np.concatenate([p[:, 1], CORE[:, 1]])
        allb = np.concatenate([bright, core_b])
        idx = _field(fr, allc, allr, allb, _colour_spike, te, be)

    # instruments
    th = theta_of(t)
    star_line = f'STAR {fmt_theta(th)}"'
    eye_line = f'YOUR EYE {int(EYE_LIM)}"'
    pulse = np.clip(1.0 - abs(t - T_CROSS) / 0.5, 0, 1)
    star_rgb = tuple(v * 0.82 for v in WHITE)
    eye_rgb = tuple(min(1.0, v * (1.0 + 0.8 * pulse)) for v in CRIM)
    if c_lid < 0.97:
        draw_text(fr, star_line, ROW_STAR, star_rgb, hidden)
        draw_text(fr, eye_line, ROW_EYE, eye_rgb, hidden)

    # the lid rims: the only visible part of the lid, sweeping over black
    if te is not None and c_lid < 0.97:
        for col in range(G.cols):
            if 1 <= te[col] < G.rows - 1:
                fr.put(col, te[col], '#', LIDEDGE)
            if 1 <= be[col] < G.rows - 1:
                fr.put(col, be[col], '#', LIDEDGE)
    fr.last_idx = idx
    return fr

# ---------------------------------------------------------------- checks
def check():
    ratio = EYE_LIM / THETA1
    assert ratio > 1e4, ratio                              # the title's claim
    ts = np.linspace(0, T_END, 2000)
    th = np.array([theta_of(t) for t in ts])
    assert np.all(np.diff(th) <= 1e-9), "theta must fall monotonically"
    assert abs(theta_of(T_REC) - THETA1) < 1e-9
    assert T_HOLD < T_CROSS < T_REC
    r_at_cross = sphere_px(T_CROSS)

    # suture geometry is shared; lash sets genuinely differ
    for cfg in CONFIGS:
        assert 21 <= cfg['n_lash'] <= 28
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = CONFIGS[i]['lash_ang'], CONFIGS[j]['lash_ang']
            d = np.abs(a[:, None] - b[None, :])
            d = np.minimum(d, 360 - d)
            mnn = d.min(axis=1).mean()
            assert mnn > 1.5, (i, j, mnn)

    # spike extents stay clear of frame edge and text
    for cfg in CONFIGS:
        p = cfg['pts']
        assert p[:, 0].min() >= 2 and p[:, 0].max() <= 95, \
            (p[:, 0].min(), p[:, 0].max())
        assert p[:, 1].min() >= 5 and p[:, 1].max() <= 112, \
            (p[:, 1].min(), p[:, 1].max())
    max_len = SUT_LEN.max()
    assert max_len >= 40, max_len                          # payoff must read big

    # text: fits the grid, sits in the safe band
    widest = max(('STAR 0.006"', f'STAR {int(THETA0)}"', 'YOUR EYE 60"'),
                 key=_tw)
    assert _tw(widest) <= G.cols - 2, (widest, _tw(widest))
    assert ROW_STAR >= int(G.rows * 0.10) + 1
    assert ROW_EYE + 5 * SC <= int(G.rows * 0.85), ROW_EYE + 5 * SC

    # blink profile actually closes, and the ending stays closed
    mids = [T_BLINK1 + 0.24, T_BLINK2 + 0.24]
    for m in mids:
        assert blink_c(m) == 1.0
    for t in np.linspace(T_CLOSE + 0.3, T_END - 1 / FPS, 40):
        assert blink_c(t) == 1.0
    assert blink_c(T_BLINK1 - 0.1) == 0.0 and blink_c(T_BLOOM) == 0.0

    # configs must differ ON SCREEN: cell sets of full-grown bursts
    def cells(cfg):
        p = cfg['pts']
        return set(zip(np.rint(p[:, 0]).astype(int),
                       np.rint(p[:, 1]).astype(int)))
    cA, cB, cC = (cells(c) for c in CONFIGS)
    dAB, dBC = len(cA ^ cB), len(cB ^ cC)
    assert dAB > 400 and dBC > 400, (dAB, dBC)
    # suture cells persist across blinks (the lens does not move)
    def sut_cells(cfg):
        p = cfg['pts'][(cfg['sid'] >= 0) & (cfg['sid'] < 6)]
        return set(zip(np.rint(p[:, 0]).astype(int),
                       np.rint(p[:, 1]).astype(int)))
    sA, sB = sut_cells(CONFIGS[0]), sut_cells(CONFIGS[1])
    overlap = len(sA & sB) / len(sA)
    assert overlap > 0.9, overlap

    # burst is big enough to be the ONE FORM
    n_burst = len(cA)
    assert n_burst > 800, n_burst

    # opening sphere: full coverage, convex pinhole check on the real frame
    fr0 = draw(0)
    idx0 = fr0.last_idx
    assert idx0 is not None
    filled = idx0 > 0
    assert filled.sum() > 5000, filled.sum()
    worst = 0
    for r in range(filled.shape[0]):
        cs = np.where(filled[r])[0]
        if len(cs) > 3:
            gaps = np.diff(cs) - 1
            worst = max(worst, int(gaps.max()))
    assert worst <= 1, worst

    # shut eyes draw NOTHING: no field, no text (glyphs leak; black is honest)
    fr_mid = draw(int((T_BLINK1 + 0.24) * FPS))
    assert fr_mid.last_idx is None
    fr_end = draw(FRAMES - 1)
    assert fr_end.last_idx is None
    assert blink_c((FRAMES - 1) / FPS) == 1.0
    # a half-closed lid masks part of the burst
    n_open = int((draw(int((T_BLINK1 - 0.3) * FPS)).last_idx > 0).sum())
    n_half = int((draw(int((T_BLINK1 + 0.15) * FPS)).last_idx > 0).sum())
    assert n_half < n_open * 0.65, (n_open, n_half)

    assert 0.75 <= TWINKLE.min() and TWINKLE.max() <= 1.25

    print(f"check OK — eye/star ratio {ratio:.0f}x; cross at t={T_CROSS:.2f}s "
          f"(drawn r={r_at_cross:.1f} cells); suture len max {max_len:.0f}, "
          f"burst cells {n_burst}; lash jump AB {dAB} / BC {dBC} cells, "
          f"suture overlap {overlap:.2f}; sphere fill {int(filled.sum())} "
          f"worst gap {worst}; {FRAMES} frames, {T_END}s")

# ---------------------------------------------------------------- outputs
def sheet():
    ts = [0.0, 3.2, 7.0, 9.5, 10.55, 11.5]
    frames = [draw(int(t * FPS)) for t in ts]
    contact(frames, '/tmp/starburst_sheet.png', cols=3,
            labels=[f"t={t}s" for t in ts])
    print("sheet -> /tmp/starburst_sheet.png")

def render():
    with Encoder('/tmp/starburst.mp4', G, fps=FPS) as enc:
        for f in range(FRAMES):
            enc.write(draw(f))
    print("render -> /tmp/starburst.mp4")

if __name__ == '__main__':
    check()
    if 'sheet' in sys.argv:
        sheet()
    if 'render' in sys.argv:
        render()
