#!/usr/bin/env python3
"""THE TOOTH REMEMBERS — bomb-pulse carbon-14 in enamel.

One first molar, big, spinning. It belongs to someone born in 1961. Its crown
enamel mineralizes on screen from cusp-tip to neck, 1961 -> 1964, and every
layer locks in the tint of that year's air. The air's carbon-14 nearly doubled
by 1964 (atmospheric bomb tests), then decayed back over sixty years. Enamel
has no turnover: the AIR counter falls, the TOOTH counter never moves.

Verified anchors (all opened this wake):
- NH atmospheric 14C "almost doubled", peak ~1963-64; ~4%/yr decline after
  1963; relative concentration back past pre-1955 values by the 2020s
  (Wikipedia, Bomb pulse).
- Partial Test Ban Treaty: signed 5 Aug 1963, in force 10 Oct 1963. France
  atmospheric until 1974, China until 1980 (Wikipedia, PTBT).
- Last atmospheric test: 16 Oct 1980, China, Lop Nur. 520 atmospheric
  explosions total, 545 Mt (Wikipedia, List of nuclear weapons tests).
- Enamel: "there is no turnover of enamel throughout life"; birth-date from
  enamel 14C to +/- 1.5 years, citing Spalding et al. Nature 437:333 (2005)
  (PMC2957015).
- First molar: initial calcification at birth, crown completed 2.5-3 yr;
  third molar crown 12-16 yr (Wikipedia, Human tooth development).
"""
import sys
import numpy as np

_HERE = __import__('os').path.dirname(__import__('os').path.abspath(__file__)); sys.path[:0] = [_HERE, __import__('os').path.dirname(_HERE)]
from asciilib import (Camera, Encoder, Frame, Grid, contact, depth_cue,
                      ink_lut, lambert, rot, specular, visible, zbuffer)

G = Grid()
RAMP = ink_lut()

# ---------------------------------------------------------------- palette
BG    = (0.012, 0.020, 0.050)            # midnight blue-black
IVORY = np.array([0.965, 0.935, 0.845])  # baseline enamel
GOLD  = np.array([1.000, 0.760, 0.420])  # mid excess
EMBER = np.array([1.000, 0.440, 0.100])  # peak excess (1964)
TAN   = np.array([0.620, 0.520, 0.400])  # root dentin
SLATE = np.array([0.300, 0.360, 0.520])  # unformed ghost
SPECK = np.array([0.420, 0.470, 0.600])  # fallout motes

# ---------------------------------------------------------------- timeline
FPS      = 30
T_HOLD0  = 1.2                    # hold 1961 (establish ghost + counters)
T_FORM1  = 9.6                    # 1961 -> 1964 linear (crown mineralizes)
T_DECAY1 = 15.2                   # 1964 -> 2026 accelerating
T_END    = 17.2
FRAMES   = int(round(T_END * FPS))          # 516

Y_BIRTH, Y_CROWN = 1961.0, 1964.0           # first molar: birth -> ~age 3
Y_ROOT0, Y_ROOT1 = 1964.0, 1970.0           # root completes ~age 9-10
Y_FINAL = 2026.0
Y_TREATY = 1963.78                          # in force 10 Oct 1963
Y_LAST   = 1980.79                          # 16 Oct 1980, Lop Nur

def year_of(t):
    t = float(t)
    if t < T_HOLD0:
        return Y_BIRTH
    if t < T_FORM1:
        return Y_BIRTH + (Y_CROWN - Y_BIRTH) * (t - T_HOLD0) / (T_FORM1 - T_HOLD0)
    if t < T_DECAY1:
        q = (t - T_FORM1) / (T_DECAY1 - T_FORM1)
        return Y_CROWN + (Y_FINAL - Y_CROWN) * q ** 2.4
    return Y_FINAL

# ------------------------------------------------- atmospheric 14C excess
def _sstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def excess(y):
    """Fraction above the pre-1955 level. Drawn to verified anchors:
    ~0 before 1955, dip during the 1958-61 moratorium, peak 0.95
    ("almost doubled") at 1964.0, then exp decay (~1/14.5 yr e-fold,
    consistent with ~4%/yr early and the known later record shape)."""
    y = np.asarray(y, dtype=float)
    r1 = 0.30 * _sstep((y - 1955.0) / 3.9)                  # 1955 -> 1958.9
    r2 = np.where(y > 1958.9,
                  -0.06 * _sstep((y - 1958.9) / 2.3), 0.0)  # moratorium dip
    r3 = np.where(y > 1961.2,
                  0.71 * _sstep((y - 1961.2) / 2.8), 0.0)   # 1961.2 -> 1964
    rise = r1 + r2 + r3
    dec = 0.95 * np.exp(-(y - 1964.0) / 14.5)
    return np.where(y <= 1964.0, rise, dec)

def air_display(y):
    """Displayed % vs pre-1955: excess minus a small fossil-dilution (Suess)
    term so the readout crosses 0 in the early 2020s, as the record did."""
    e = excess(y) - 0.032 * np.clip((np.asarray(y, float) - 1964.0) / 62.0, 0, 1)
    return max(0, int(round(float(e) * 100)))

_YGRID = np.linspace(Y_BIRTH, Y_CROWN, 400)
_EGRID = excess(_YGRID)

def tooth_display(y):
    """Hottest enamel locked so far (running max over the formed window)."""
    if y < Y_BIRTH + 0.02:
        return None
    yy = min(y, Y_CROWN)
    m = float(np.max(_EGRID[_YGRID <= yy + 1e-9]))
    return int(round(m * 100))

# ---------------------------------------------------------------- geometry
RNG = np.random.default_rng(19611016)
A, B = 1.00, 0.85            # crown half-widths (x mesiodistal, z buccolingual)
P_SE = 2.8                   # superellipse exponent
Y_OCC, Y_CEJ = -1.05, 0.0    # occlusal rim, cervical line

def _se_dir(phi):
    c, s = np.cos(phi), np.sin(phi)
    e = 2.0 / P_SE
    return (np.sign(c) * np.abs(c) ** e, np.sign(s) * np.abs(s) ** e)

def _wall(u, phi):
    """Crown side wall: u=0 occlusal rim, u=1 cervix."""
    prof = 0.96 + 0.16 * np.sin(np.pi * u) - 0.16 * u * u
    lobe = 1.0 + 0.04 * np.sin(np.pi * u) * np.cos(2 * phi + 0.5)
    cx, cz = _se_dir(phi)
    r = prof * lobe
    x = A * r * cx
    z = B * r * cz
    y = Y_OCC + (Y_CEJ - Y_OCC) * u
    return np.stack([x, y, z], axis=-1)

CUSPS = [(+0.52, +0.52, 0.23), (-0.52, +0.52, 0.21),
         (+0.52, -0.52, 0.19), (-0.52, -0.52, 0.22)]

def _cap(s, phi):
    """Occlusal cap: s=0 centre, s=1 rim (matches wall at u=0)."""
    cx, cz = _se_dir(phi)
    x = A * 0.96 * s * cx
    z = B * 0.96 * s * cz
    xn = x / (A * 0.96)
    zn = z / (B * 0.96)
    y = Y_OCC - 0.10 * (1.0 - s * s)                     # slight dome
    for (px, pz, h) in CUSPS:                            # four cusps (up = -y)
        y = y - h * np.exp(-(((xn - px) ** 2 + (zn - pz) ** 2) / 0.090))
    gr = np.exp(-((xn / 0.11) ** 2)) + np.exp(-((zn / 0.11) ** 2))
    y = y + 0.09 * gr * (1.0 - s * s)                    # central grooves (down)
    return np.stack([x, y, z], axis=-1)

def _root(u, phi, side):
    """One root: u=0 at furcation, u=1 tip. side = +-1 (mesial/distal)."""
    x0 = 0.48 * side
    X = x0 * (1.0 + 0.55 * u - 0.35 * u * u)             # splay then ease
    Y = 0.05 + 1.20 * u
    rr = (0.30 * (1.0 - 0.80 * u) + 0.05)
    tip = np.sqrt(np.clip(1.0 - (np.clip(u - 0.92, 0, None) / 0.08) ** 2, 0, 1))
    rr = rr * tip
    x = X + 0.70 * rr * np.cos(phi)
    z = 1.25 * rr * np.sin(phi)
    return np.stack([x, Y, z], axis=-1)

def _sample(fn, nu, nphi, ulo=0.0, uhi=1.0, extra_args=()):
    """Jittered (u,phi) grid + finite-difference normals (computed once)."""
    uu, pp = np.meshgrid(np.linspace(ulo, uhi, nu), np.linspace(0, 2 * np.pi, nphi, endpoint=False), indexing='ij')
    uu = np.clip(uu + RNG.uniform(-0.5, 0.5, uu.shape) * (uhi - ulo) / nu, ulo, uhi)
    pp = pp + RNG.uniform(-0.5, 0.5, pp.shape) * 2 * np.pi / nphi
    uu, pp = uu.ravel(), pp.ravel()
    eps = 6e-4
    p0 = fn(uu, pp, *extra_args)
    pu = fn(np.clip(uu + eps, ulo, uhi), pp, *extra_args)
    pf = fn(uu, pp + eps, *extra_args)
    n = np.cross(pu - p0, pf - p0)
    ln = np.linalg.norm(n, axis=-1, keepdims=True)
    ln[ln == 0] = 1.0
    n = n / ln
    return p0, n

def build():
    parts = []
    # crown wall (enamel)
    p, n = _sample(_wall, 300, 420)
    ax = p - np.stack([np.zeros(len(p)), p[:, 1], np.zeros(len(p))], axis=-1)
    flip = np.einsum('ij,ij->i', n, ax) < 0
    n[flip] = -n[flip]
    parts.append((p, n, 0))                              # kind 0 = enamel
    # occlusal cap (enamel)
    p, n = _sample(_cap, 210, 420)
    flip = n[:, 1] > 0                                   # cap faces up (-y)
    n[flip] = -n[flip]
    parts.append((p, n, 0))
    # roots (dentin)
    for side in (+1.0, -1.0):
        p, n = _sample(_root, 230, 210, extra_args=(side,))
        cx = 0.48 * side * (1.0 + 0.55 * np.clip((p[:, 1] - 0.05) / 1.20, 0, 1)
                            - 0.35 * np.clip((p[:, 1] - 0.05) / 1.20, 0, 1) ** 2)
        ax = p - np.stack([cx, p[:, 1], np.zeros(len(p))], axis=-1)
        flip = np.einsum('ij,ij->i', n, ax) < 0
        n[flip] = -n[flip]
        parts.append((p, n, 1))                          # kind 1 = root
    pts = np.concatenate([p for p, _, _ in parts])
    nrm = np.concatenate([n for _, n, _ in parts])
    kind = np.concatenate([np.full(len(p), k) for p, _, k in parts])
    return pts, nrm, kind

PTS, NRM, KIND = build()
N = len(PTS)
Y_MIN = float(PTS[:, 1].min())                           # highest cusp tip

# formation year per point
YEARF = np.where(
    KIND == 0,
    Y_BIRTH + (Y_CROWN - Y_BIRTH) * (PTS[:, 1] - Y_MIN) / (Y_CEJ - Y_MIN),
    Y_ROOT0 + (Y_ROOT1 - Y_ROOT0) * np.clip((PTS[:, 1] - 0.05) / 1.20, 0, 1))
# locked tint weight (enamel only): excess in the formation year / peak
W_LOCK = np.clip(excess(np.clip(YEARF, Y_BIRTH, Y_CROWN)) / 0.95, 0, 1)
GHOST = RNG.random(N) < 0.15                             # static ghost subset
STIP = 1.0 + 0.05 * RNG.standard_normal(N)               # static stipple

PITCH = 0.16
YAW_RATE = 0.45
LAMP = np.array([-0.45, -0.55, 0.70])   # upper-left, above (+y is DOWN), in front
LAMP = LAMP / np.linalg.norm(LAMP)

# camera: fixed, fit over rotated poses + phantom frame that reserves the
# instrument band at the top (composition, not zoom — trap 7)
PHANTOM = np.array([[-1.45, -2.45, 0], [1.45, -2.45, 0],
                    [-1.45, 1.42, 0], [1.45, 1.42, 0]])
_poses = []
for yw in np.linspace(0, np.pi / 2, 5):
    p, _ = rot(PTS, NRM, PITCH, yw, 0.0)
    _poses.append(np.concatenate([p, PHANTOM]))
CAM = Camera(G).fit(_poses, margin=1.02)

# ---------------------------------------------------------------- fallout
def _speck_rate(y):
    if y < Y_TREATY:
        return 0.50                                      # test era, heavy
    if y < 1974.0:
        return 0.060                                     # France atmospheric
    if y <= Y_LAST:
        return 0.035                                     # China, rare
    return 0.0

def _build_specks():
    rng = np.random.default_rng(545)
    specks = []
    forced_last = False
    prev_y = year_of(0.0)
    for f in range(FRAMES):
        y = year_of(f / FPS)
        if (not forced_last) and prev_y < Y_LAST <= y:
            specks.append((f, 49.0, -4.0, 0.62, 0.30, Y_LAST))  # the last one
            forced_last = True
        elif rng.random() < _speck_rate(y):
            specks.append((f, rng.uniform(2, 96), rng.uniform(-40, -4),
                           rng.uniform(0.5, 0.75), 0.22, y))
        prev_y = y
    return specks

SPECKS = _build_specks()

# ---------------------------------------------------------------- text
FONT = {
    '0': "111101101101111", '1': "010110010010111", '2': "111001111100111",
    '3': "111001111001111", '4': "101101111001001", '5': "111100111001111",
    '6': "111100111101111", '7': "111001001010010", '8': "111101111101111",
    '9': "111101111001111", 'A': "010101111101101", 'I': "111010010010111",
    'R': "110101110101101", 'O': "111101101101111", 'T': "111010010010010",
    'H': "101101111101101", '+': "000010111010000", '%': "101001010100101",
    ' ': "000000000000000",
}
TAG_SCALE = 2

def _text_width(s):
    return len(s) * 3 * TAG_SCALE + (len(s) - 1) * TAG_SCALE

def draw_text(fr, s, row0, rgb, occupied):
    col0 = (G.cols - _text_width(s)) // 2
    c = col0
    for ch in s:
        bm = FONT[ch]
        for r in range(5):
            for k in range(3):
                if bm[r * 3 + k] == '1':
                    for dr in range(TAG_SCALE):
                        for dc in range(TAG_SCALE):
                            rr = row0 + r * TAG_SCALE + dr
                            cc = c + k * TAG_SCALE + dc
                            fr.put(cc, rr, '#', rgb)
                            occupied.add((rr, cc))
        c += 4 * TAG_SCALE
    return col0

ROW_YEAR, ROW_AIR, ROW_TOOTH = 19, 31, 42                # each 10 rows tall

# ---------------------------------------------------------------- draw
def draw(f):
    t = f / FPS
    year = year_of(t)
    p, n = rot(PTS, NRM, PITCH, YAW_RATE * t, 0.0)
    col, row, z = CAM.project(p)
    ok = visible(G, col, row)
    col, row, z, n2 = col[ok], row[ok], z[ok], n[ok]
    formed = (YEARF[ok] <= year)
    show = formed | (GHOST[ok] & ~formed)
    col, row, z, n2 = col[show], row[show], z[show], n2[show]
    formed = formed[show]
    kind = KIND[ok][show]
    wl = W_LOCK[ok][show]
    stip = STIP[ok][show]

    _, keep = zbuffer(G, col, row, z)
    lam = lambert(n2, LAMP)
    spec = specular(n2, LAMP, 26)
    shade = (0.13 + 0.87 * lam + 0.33 * spec) * depth_cue(z, far=0.94) * stip
    shade = np.where(formed, shade, 0.20)
    extra = np.where(formed, np.where(kind == 1, 1.5, wl), 3.0)

    def colour(s, e):
        s = min(1.0, max(0.0, s))
        if e >= 2.5:
            base = SLATE * 0.9
            return tuple(base * (0.55 + 0.45 * s))
        if e >= 1.2:
            return tuple(TAN * (0.50 + 0.50 * s))
        if e < 0.5:
            base = IVORY + (GOLD - IVORY) * (e * 2.0)
        else:
            base = GOLD + (EMBER - GOLD) * (e * 2.0 - 1.0)
        return tuple(base * (0.55 + 0.45 * s))

    fr = Frame(G, BG)
    fr.idx, _ = fr.field(col, row, keep, shade, colour, RAMP, extra=extra)

    # text (with occupancy so specks never overprint it)
    occupied = set()
    draw_text(fr, "%d" % int(year), ROW_YEAR, tuple(IVORY), occupied)
    av = air_display(year)
    wa = min(1.0, float(excess(year)) / 0.95)
    if wa < 0.5:
        crgb = SLATE + (GOLD - SLATE) * (wa * 2.0)
    else:
        crgb = GOLD + (EMBER - GOLD) * (wa * 2.0 - 1.0)
    draw_text(fr, "AIR +%d%%" % av, ROW_AIR, tuple(crgb), occupied)
    tv = tooth_display(year)
    if tv is not None:
        wt = tv / 95.0
        if wt < 0.5:
            trgb = SLATE + (GOLD - SLATE) * (wt * 2.0)
        else:
            trgb = GOLD + (EMBER - GOLD) * (wt * 2.0 - 1.0)
        draw_text(fr, "TOOTH +%d%%" % tv, ROW_TOOTH, tuple(trgb), occupied)

    # fallout motes (behind nothing — drawn last but skip text + tooth cells)
    for (f0, c0, r0, vy, sh, ysp) in SPECKS:
        if f0 <= f:
            rr = r0 + vy * (f - f0) * 0.62
            if rr < G.rows:
                ri, ci = int(rr), int(round(c0))
                if 0 <= ri < G.rows and 0 <= ci < G.cols and (ri, ci) not in occupied:
                    fr.put(ci, ri, '.', tuple(SPECK * (0.6 + 1.4 * sh)))
    return fr


# ---------------------------------------------------------------- checks
def check():
    # --- the subject's claims, as numbers
    e64 = float(excess(1964.0))
    assert e64 >= 0.93, e64                     # "almost doubled" at the peak
    assert air_display(2024.0) <= 1             # back past pre-1955 by 2020s
    yy = np.linspace(1964.0, 2026.0, 200)
    ee = excess(yy)
    assert np.all(np.diff(ee) < 0)              # monotone decay after peak
    e80 = float(excess(1980.0))
    assert 0.25 <= e80 <= 0.36, e80             # ~+30% at 1980 (record shape)
    # formation mapping
    assert abs(float(YEARF[PTS[:, 1] == Y_MIN][0]) - Y_BIRTH) < 0.05
    cej = np.abs(PTS[:, 1] - Y_CEJ) < 0.01
    assert np.all(np.abs(YEARF[(KIND == 0) & cej] - Y_CROWN) < 0.05)
    w_tip = float(np.mean(W_LOCK[(KIND == 0) & (PTS[:, 1] < Y_MIN + 0.05)]))
    w_cej = float(np.mean(W_LOCK[(KIND == 0) & cej]))
    assert w_cej / max(w_tip, 1e-9) >= 2.5, (w_tip, w_cej)  # neck hotter
    root_tip_year = float(YEARF[KIND == 1].max())
    assert root_tip_year <= 1970.2, root_tip_year
    # year(t) monotone
    ts = np.linspace(0, T_END, 400)
    ys = np.array([year_of(t) for t in ts])
    assert np.all(np.diff(ys) >= -1e-9)
    # --- specks obey the treaty and the record
    sy = np.array([s[5] for s in SPECKS])
    assert sy.max() <= 1980.85, sy.max()        # nothing after Lop Nur
    assert np.any((sy > 1974.0) & (sy <= 1980.85))   # China era present
    assert np.any((sy > Y_TREATY) & (sy <= 1974.0))  # France era present
    n_heavy = int(np.sum(sy < Y_TREATY))
    n_late = int(np.sum(sy >= Y_TREATY))
    assert n_heavy > 4 * max(n_late, 1), (n_heavy, n_late)  # treaty visible
    # --- no clipping at any pose; bigness; text clearance
    min_tooth_row = 999
    for yw in np.linspace(0, np.pi, 7):
        p, _ = rot(PTS, NRM, PITCH, yw, 0.0)
        c, r, _ = CAM.project(p)
        assert c.min() >= 0 and c.max() < G.cols, (c.min(), c.max())
        assert r.min() >= 0 and r.max() < G.rows, (r.min(), r.max())
        min_tooth_row = min(min_tooth_row, int(r.min()))
    p0, _ = rot(PTS, NRM, PITCH, 0.0, 0.0)
    c0, r0, _ = CAM.project(p0)
    crown = PTS[:, 1] < Y_CEJ
    width = int(c0[crown].max() - c0[crown].min())
    height = int(r0.max() - r0.min())
    assert width >= 55, width
    assert height >= 80, height
    text_bot = ROW_TOOTH + 10
    assert min_tooth_row >= text_bot + 2, (min_tooth_row, text_bot)
    assert ROW_YEAR >= int(G.rows * 0.10)
    assert text_bot <= int(G.rows * 0.85)
    # --- solidity: interior pinholes in the crown at the final formed pose.
    # The top ~7 rows are excluded on purpose: a slice through separate cusp
    # tips has real background between them (trap 6 — honest concavity).
    fr = draw(FRAMES - 1)
    cells = (fr.idx > 0).astype(int)
    top_drawn = int(np.nonzero(cells.any(axis=1))[0][0])
    crow0 = top_drawn + 9
    crow1 = int(r0[crown].max()) - 2
    worst = 0
    for rr in range(crow0, crow1):
        cs = np.where(cells[rr] > 0)[0]
        if len(cs) < 2:
            continue
        runs = np.diff(cs)
        gaps = runs[runs > 1] - 1
        if len(gaps):
            worst = max(worst, int(gaps.max()))
    assert worst <= 1, worst
    # --- final differs from first (not a no-op)
    fr0 = draw(0)
    assert not np.array_equal(fr0.idx, fr.idx)
    print("check OK: peak +%d%%, 1980 +%d%%, cusp %d / cej %d lock ratio %.1f"
          % (round(e64 * 100), round(e80 * 100), round(w_tip * 100),
             round(w_cej * 100), w_cej / w_tip))
    print("  crown width %d cols, tooth height %d rows, worst crown gap %d"
          % (width, height, worst))
    print("  specks: %d heavy-era, %d late-era, last at %.2f"
          % (n_heavy, n_late, sy.max()))
    print("  tooth top row %d, text ends row %d" % (min_tooth_row, text_bot))
    print("  %d samples, %d frames, %.1f s" % (N, FRAMES, T_END))


def sheet():
    ts = [0.5, 4.0, 9.0, 10.4, 12.6, 16.8]
    frames = [draw(int(t * FPS)) for t in ts]
    contact(frames, "/tmp/tooth_sheet.png", cols=3,
            labels=["%.1fs %d" % (t, int(year_of(t))) for t in ts])
    print("sheet at /tmp/tooth_sheet.png")


def render():
    out = "/tmp/tooth.mp4"
    with Encoder(out, G, fps=FPS) as enc:
        for f in range(FRAMES):
            enc.write(draw(f))
            if f % 60 == 0:
                print("frame %d/%d" % (f, FRAMES))
    print("wrote", out)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'check'
    if mode == 'check':
        check()
    elif mode == 'sheet':
        sheet()
    elif mode == 'render':
        render()
    else:
        check(); sheet(); render()
