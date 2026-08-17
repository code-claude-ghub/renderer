"""$2.13 — the federal tipped cash wage, frozen by statute in 1996.

ONE FORM: a struck coin, face on, stamped $2.13 and a date.

The stamp never changes size, because the number in the law never changed.
The coin shrinks, because CPI did. Coin AREA is proportional to purchasing
power, so a dollar of value is the same amount of metal at every date --
the shrink is the arithmetic, not a graphic.

One frame per month, Aug 1996 -> Jul 2026, from monthly CPI-U (FRED
CPIAUCSL, series origin BLS). 359 months, 359 frames. The lurch at the end
is 2021-23 and it is in the data, not in the easing.

A faint ring marks where the rim stood in 1996 and stays there all thirty
years, so the gap between the metal and the ring is what was lost. By the
end the legend is wider than the coin it is stamped on.

Verified before render:
  federal tipped minimum cash wage $2.13/hr, max tip credit $5.12, federal
  minimum $7.25 -- DOL WHD, "Minimum Wages for Tipped Employees", table
  revised 2026-07-01.
  Frozen by Pub. L. 104-188 (1996), which struck the percentage-of-minimum
  formula and pegged the cash wage to "the cash wage required to be paid
  such an employee on August 20, 1996" -- 29 U.S.C. 203(m).
"""

import os
import sys

import numpy as np
import cairo
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asciilib import (Camera, Encoder, Frame, Grid, contact, depth_cue,
                      ink_lut, lambert, rot, specular, visible, zbuffer)

OUT = "/tmp/tipped_213.mp4"
FPS = 30
HOLD_IN, HOLD_OUT = 42, 108

G = Grid()
RAMP = ink_lut()
RNG = np.random.default_rng(20260816)

# paper, ink, ghost -- copper struck fresh, verdigris thirty years on
BG = (0.928, 0.897, 0.812)
COPPER = (0.418, 0.163, 0.052)
VERDIGRIS = (0.027, 0.298, 0.251)
GHOST = (0.470, 0.455, 0.388)
WORD = (0.115, 0.145, 0.125)

R0 = 100.0          # 1996 radius, world units
T = 13.0            # thickness
H_LEG = 3.4         # relief height of the legend
H_RIM = 2.8         # raised border on the face

LAMP = np.array([-0.44, -0.60, 0.67])
LAMP = LAMP / np.linalg.norm(LAMP)

# ---------------------------------------------------------------- the data

# monthly CPI-U, Aug 1996 .. Jul 2026, fetched from FRED CPIAUCSL
CPI = [
       157.200, 157.700, 158.200, 158.700, 159.100, 159.400, 159.700,
       159.800, 159.900, 159.900, 160.200, 160.400, 160.800, 161.200,
       161.500, 161.700, 161.800, 162.000, 162.000, 162.000, 162.200,
       162.600, 162.800, 163.200, 163.400, 163.500, 163.900, 164.100,
       164.400, 164.700, 164.700, 164.800, 165.900, 166.000, 166.000,
       166.700, 167.100, 167.800, 168.100, 168.400, 168.800, 169.300,
       170.000, 171.000, 170.900, 171.200, 172.200, 172.700, 172.700,
       173.600, 173.900, 174.200, 174.600, 175.600, 176.000, 176.100,
       176.400, 177.300, 177.700, 177.400, 177.400, 178.100, 177.600,
       177.500, 177.400, 177.700, 178.000, 178.500, 179.300, 179.500,
       179.600, 180.000, 180.500, 180.800, 181.200, 181.500, 181.800,
       182.600, 183.600, 183.900, 183.200, 182.900, 183.100, 183.700,
       184.500, 185.100, 184.900, 185.000, 185.500, 186.300, 186.700,
       187.100, 187.400, 188.200, 188.900, 189.100, 189.200, 189.800,
       190.800, 191.700, 191.700, 191.600, 192.400, 193.100, 193.700,
       193.600, 193.700, 194.900, 196.100, 198.800, 199.100, 198.100,
       198.100, 199.300, 199.400, 199.700, 200.700, 201.300, 201.800,
       202.900, 203.800, 202.800, 201.900, 202.000, 203.100, 203.437,
       204.226, 205.288, 205.904, 206.755, 207.234, 207.603, 207.667,
       208.547, 209.190, 210.834, 211.445, 212.174, 212.687, 213.448,
       213.942, 215.208, 217.463, 219.016, 218.690, 218.877, 216.995,
       213.153, 211.398, 211.933, 212.705, 212.495, 212.709, 213.022,
       214.790, 214.726, 215.445, 215.861, 216.509, 217.234, 217.347,
       217.488, 217.281, 217.353, 217.403, 217.290, 217.199, 217.605,
       217.923, 218.275, 219.035, 219.590, 220.472, 221.187, 221.898,
       223.046, 224.093, 224.806, 224.806, 225.395, 226.106, 226.597,
       226.750, 227.169, 227.223, 227.842, 228.329, 228.807, 229.187,
       228.713, 228.524, 228.590, 229.918, 231.015, 231.638, 231.249,
       231.221, 231.679, 232.937, 232.282, 231.797, 231.893, 232.445,
       232.900, 233.456, 233.544, 233.669, 234.100, 234.719, 235.288,
       235.547, 236.028, 236.468, 236.918, 237.231, 237.498, 237.460,
       237.477, 237.430, 236.983, 236.252, 234.747, 235.342, 235.976,
       236.222, 237.001, 237.657, 238.034, 238.033, 237.498, 237.733,
       238.017, 237.761, 237.652, 237.336, 238.080, 238.992, 239.557,
       240.222, 240.101, 240.545, 241.176, 241.741, 242.026, 242.637,
       243.618, 244.006, 243.892, 244.193, 244.004, 244.163, 244.243,
       245.183, 246.435, 246.626, 247.284, 247.805, 248.859, 249.529,
       249.577, 250.227, 250.792, 251.018, 251.214, 251.663, 252.182,
       252.772, 252.594, 252.767, 252.561, 253.319, 254.277, 255.233,
       255.296, 255.213, 255.802, 256.036, 256.430, 257.155, 257.879,
       258.630, 259.127, 259.250, 258.076, 256.032, 255.802, 257.042,
       258.352, 259.316, 259.997, 260.319, 260.911, 262.045, 262.687,
       263.579, 264.961, 266.614, 268.383, 270.654, 271.903, 272.676,
       273.910, 276.550, 278.919, 280.845, 282.543, 284.500, 287.674,
       288.561, 291.298, 294.957, 294.913, 295.097, 296.349, 298.007,
       298.786, 298.832, 300.420, 301.450, 301.821, 302.845, 303.334,
       304.014, 304.609, 306.082, 307.276, 307.696, 308.148, 308.741,
       309.698, 310.967, 312.345, 313.023, 313.175, 313.044, 313.569,
       314.062, 314.732, 315.631, 316.528, 317.604, 318.961, 319.679,
       319.785, 320.302, 320.620, 321.435, 322.169, 323.291, 324.245,
       325.063, 326.031, 326.588, 327.460, 330.293, 332.407, 333.979,
       332.568, 332.813
]
CPI = np.array(CPI, float)
MONTHS = len(CPI)
YEARS = [1996 + (7 + i) // 12 for i in range(MONTHS)]   # Aug 1996 start

REAL = CPI[0] / CPI                     # purchasing power of $2.13, vs 1996
RAD = R0 * np.sqrt(REAL)                # area proportional to value

# ------------------------------------------------------------ the stamping

RS = 768                                # raster resolution for the die
RW = 118.0                              # world half-width of the raster


def _die(year):
    """Rasterise the face of the die: legend $2.13 plus a date, as a
    coverage field over world x,y in [-RW, RW]."""
    surf = cairo.ImageSurface(cairo.FORMAT_A8, RS, RS)
    ctx = cairo.Context(surf)
    ctx.set_source_rgba(0, 0, 0, 0)
    ctx.set_operator(cairo.OPERATOR_SOURCE)
    ctx.paint()
    ctx.set_operator(cairo.OPERATOR_OVER)
    ctx.set_source_rgba(1, 1, 1, 1)
    ctx.select_font_face("monospace", cairo.FONT_SLANT_NORMAL,
                         cairo.FONT_WEIGHT_BOLD)

    def stamp(text, cy_world, want_w):
        ctx.set_font_size(100.0)
        e = ctx.text_extents(text)
        scale = (want_w / (2 * RW) * RS) / e.width
        ctx.set_font_size(100.0 * scale)
        e = ctx.text_extents(text)
        px = RS / 2.0 - e.width / 2.0 - e.x_bearing
        py = (cy_world + RW) / (2 * RW) * RS - e.height / 2.0 - e.y_bearing
        ctx.move_to(px, py)
        ctx.show_text(text)

    stamp("$2.13", -8.0, 0.72 * 2 * R0)      # fixed size: the law's number
    stamp(str(year), 46.0, 0.30 * 2 * R0)    # the date on the coin
    surf.flush()
    buf = np.frombuffer(surf.get_data(), np.uint8)
    buf = buf.reshape(RS, surf.get_stride())[:, :RS].astype(np.float32) / 255.0
    buf = ndimage.gaussian_filter(buf, 2.4)  # bevel the relief
    gy, gx = np.gradient(buf, 2 * RW / RS)   # world-unit gradients
    return buf, gx, gy


_CACHE = {}


def die(year):
    if year not in _CACHE:
        _CACHE.clear()
        _CACHE[year] = _die(year)
    return _CACHE[year]


def sample(field, x, y):
    """Bilinear sample of a raster field at world coords."""
    fx = (x + RW) / (2 * RW) * (RS - 1)
    fy = (y + RW) / (2 * RW) * (RS - 1)
    return ndimage.map_coordinates(field, [fy, fx], order=1, mode="nearest")


# -------------------------------------------------------------- the shapes

N_FACE = 150000
_u = RNG.random(N_FACE)
_th = RNG.random(N_FACE) * 2 * np.pi
FACE_RN = np.sqrt(_u)                   # normalised radius, uniform in area
FACE_TH = _th

N_TH, N_V = 900, 26
_t2 = (np.repeat(np.arange(N_TH), N_V) + RNG.random(N_TH * N_V)) / N_TH
RIM_TH = _t2 * 2 * np.pi
RIM_V = (np.tile(np.arange(N_V), N_TH) + RNG.random(N_TH * N_V)) / N_V
RIM_V = (RIM_V - 0.5) * T
REED_K = 96

N_RING = 2600
RING_TH = (np.arange(N_RING) + RNG.random(N_RING) * 0.9) / N_RING * 2 * np.pi

# ghost of the legend: where the die is struck, sampled once
_b, _, _ = _die(2026)
_yy, _xx = np.nonzero(_b > 0.55)
_sel = RNG.choice(len(_xx), size=min(9000, len(_xx)), replace=False)
GH_X = (_xx[_sel] / (RS - 1) * 2 - 1) * RW + RNG.normal(0, 0.35, len(_sel))
GH_Y = (_yy[_sel] / (RS - 1) * 2 - 1) * RW + RNG.normal(0, 0.35, len(_sel))
GH_R = np.hypot(GH_X, GH_Y)


def coin(radius, year):
    """Points + normals for one coin, in its own plane (z up out of face)."""
    L, Lx, Ly = die(year)

    r = FACE_RN * radius
    x, y = r * np.cos(FACE_TH), r * np.sin(FACE_TH)
    lv = sample(L, x, y)
    # raised border near the rim
    rc, w = 0.930 * radius, 0.045 * radius
    u = (r - rc) / w
    bump = H_RIM * np.exp(-u * u)
    dbdr = -2.0 * u / w * bump
    # the legend is INCUSE -- punched into the metal, so it sits in its own
    # shadow and reads as ink rather than as a pale ghost.
    z = T / 2.0 - H_LEG * lv + bump
    rs = np.maximum(r, 1e-6)
    dhx = -H_LEG * sample(Lx, x, y) + dbdr * x / rs
    dhy = -H_LEG * sample(Ly, x, y) + dbdr * y / rs
    n = np.stack([-dhx, -dhy, np.ones_like(dhx)], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    face = np.stack([x, y, z], 1)
    cavity = np.concatenate([lv, np.zeros(len(RIM_TH))])

    # rim, with reeding
    reed = np.sin(REED_K * RIM_TH)
    rr = radius + 0.40 * reed
    c, s = np.cos(RIM_TH), np.sin(RIM_TH)
    rim = np.stack([rr * c, rr * s, RIM_V], 1)
    tang = np.stack([-s, c, np.zeros_like(s)], 1)
    rn = np.stack([c, s, np.zeros_like(s)], 1) + 0.55 * reed[:, None] * tang
    rn /= np.linalg.norm(rn, axis=1, keepdims=True)

    return (np.concatenate([face, rim]), np.concatenate([n, rn]), cavity)


def ring_pts():
    c, s = np.cos(RING_TH), np.sin(RING_TH)
    jr = R0 + RNG.normal(0, 0.30, N_RING)
    return np.stack([jr * c, jr * s, np.full(N_RING, T / 2.0 + 0.9)], 1)


RING = ring_pts()


def pose(f):
    """Tilt of the coin at frame f: a slow wobble, never a re-scale."""
    t = f / FPS
    ax = np.radians(21.0 + 5.0 * np.sin(2 * np.pi * t / 9.0))
    ay = np.radians(3.5 * np.sin(2 * np.pi * t / 13.0 + 1.0))
    return ax, ay


# ------------------------------------------------------------------ camera

_poses = []
for _f in (0, 68, 135, 200, 300, 420):
    _ax, _ay = pose(_f)
    _p, _ = rot(RING, RING * 0, _ax, _ay)
    _poses.append(_p)
CAM = Camera(G).fit(_poses, margin=1.05)

INK_LO, INK_SPAN = 0.46, 0.54
BRONZE = (0.296, 0.235, 0.070)          # keeps the mid-tarnish out of grey


def make_colour(tarnish):
    if tarnish < 0.5:
        u = tarnish / 0.5
        ink = tuple(COPPER[i] * (1 - u) + BRONZE[i] * u for i in range(3))
    else:
        u = (tarnish - 0.5) / 0.5
        ink = tuple(BRONZE[i] * (1 - u) + VERDIGRIS[i] * u for i in range(3))

    def colour(v, extra):
        if extra > 0.5:
            base, k = GHOST, 0.70
        else:
            base = ink
            k = 0.74 + 0.26 * min(1.0, max(0.0, (v - INK_LO) / INK_SPAN))
        return tuple(BG[i] + (base[i] - BG[i]) * k for i in range(3))

    return colour


def month_of(f):
    return int(np.clip(f - HOLD_IN, 0, MONTHS - 1))


def vignette(fr):
    g = cairo.RadialGradient(G.w_px * 0.5, G.h_px * 0.47, G.w_px * 0.25,
                             G.w_px * 0.5, G.h_px * 0.47, G.w_px * 0.95)
    g.add_color_stop_rgba(0, 0.42, 0.36, 0.24, 0.0)
    g.add_color_stop_rgba(1, 0.42, 0.36, 0.24, 0.20)
    fr.ctx.set_source(g)
    fr.ctx.paint()


def text_cells(text, max_cols):
    """Rasterise a string at 8x cell resolution and area-average it onto the
    character grid. Shrinks until it fits."""
    size = 8.0
    while True:
        s = cairo.ImageSurface(cairo.FORMAT_A8, 8, 8)
        c = cairo.Context(s)
        c.select_font_face("monospace", cairo.FONT_SLANT_NORMAL,
                           cairo.FONT_WEIGHT_BOLD)
        c.set_font_size(size * 8)
        e = c.text_extents(text)
        w = int(e.width / 8.0) + 2
        if w <= max_cols or size < 2.0:
            break
        size *= 0.94
    h = int(size * 1.35)
    W, H = w * 8, h * 8
    s = cairo.ImageSurface(cairo.FORMAT_A8, W, H)
    c = cairo.Context(s)
    c.select_font_face("monospace", cairo.FONT_SLANT_NORMAL,
                       cairo.FONT_WEIGHT_BOLD)
    c.set_font_size(size * 8)
    c.set_source_rgba(1, 1, 1, 1)
    e = c.text_extents(text)
    c.move_to((W - e.width) / 2 - e.x_bearing, (H - e.height) / 2 - e.y_bearing)
    c.show_text(text)
    s.flush()
    a = np.frombuffer(s.get_data(), np.uint8).reshape(H, s.get_stride())[:, :W]
    a = a.astype(np.float32).reshape(h, 8, w, 8).mean((1, 3)) / 255.0
    return a


LINE = text_cells("THE OTHER $5.12 IS YOU", G.cols - 4)


def put_line(fr, cells, row0, alpha):
    h, w = cells.shape
    c0 = int(G.cx - w / 2.0)
    ink = np.nonzero(cells > 0.22)
    halo = set()
    for r, c in zip(*ink):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                halo.add((r + dr, c + dc))
    fr.ctx.set_source_rgba(BG[0], BG[1], BG[2], alpha)
    for r, c in halo:
        fr.ctx.rectangle((c0 + c) * G.cell, (row0 + r) * G.cell,
                         G.cell + 0.6, G.cell + 0.6)
    fr.ctx.fill()
    for r, c in zip(*ink):
        v = min(1.0, cells[r, c] * 1.25)
        fr.put(c0 + c, row0 + r, RAMP[int(v * (len(RAMP) - 1))], WORD, alpha)


def draw(f):
    m = month_of(f)
    radius, year = RAD[m], YEARS[m]
    tarnish = (m / (MONTHS - 1.0)) ** 0.85
    ax, ay = pose(f)

    pts, nrm, cav = coin(radius, year)
    p, n = rot(pts, nrm, ax, ay)

    gr = GH_R > radius                          # legend hanging off the metal
    gp = np.stack([GH_X[gr], GH_Y[gr],
                   np.full(gr.sum(), T / 2.0 + H_LEG)], 1)
    gz = np.zeros_like(gp)
    gp, _ = rot(gp, gz, ax, ay)
    rp, _ = rot(RING, RING * 0, ax, ay)

    light = (0.12 + 0.78 * lambert(n, LAMP)
             + 0.34 * specular(n, LAMP, 26))
    light *= 1.0 - 0.52 * np.clip(cav, 0, 1)     # the punched letters shadow
    all_p = np.concatenate([p, rp, gp])
    col, row, z = CAM.project(all_p)
    dens = np.empty(len(all_p))
    dens[:len(p)] = INK_LO + INK_SPAN * np.clip(1.0 - light, 0, 1)
    dens[len(p):len(p) + len(rp)] = 0.185
    dens[len(p) + len(rp):] = 0.30
    dens[:len(p)] *= depth_cue(z[:len(p)], far=0.93)
    extra = np.zeros(len(all_p))
    extra[len(p):] = 1.0

    ok = visible(G, col, row)
    col, row, z = col[ok], row[ok], z[ok]
    dens, extra = dens[ok], extra[ok]
    _, keep = zbuffer(G, col, row, z)

    fr = Frame(G, BG)
    vignette(fr)
    fr.field(col, row, keep, dens, make_colour(tarnish), RAMP, extra=extra)

    end = f - (HOLD_IN + MONTHS)
    if end > 24:
        put_line(fr, LINE, int(G.cy + 34), min(1.0, (end - 24) / 22.0))
    return fr


def check():
    print(G)
    print("months %d  R %.1f -> %.1f  (%.1f%% of area)"
          % (MONTHS, RAD[0], RAD[-1], 100 * REAL[-1]))
    print("legend half-width 72.0 vs final radius %.1f" % RAD[-1])
    stills = []
    for f in (0, HOLD_IN + 120, HOLD_IN + 250, HOLD_IN + MONTHS - 1,
              HOLD_IN + MONTHS + 60, HOLD_IN + MONTHS + HOLD_OUT - 1):
        fr = draw(f)
        stills.append(fr)
        m = month_of(f)
        p, _, _ = coin(RAD[m], YEARS[m])
        ax, ay = pose(f)
        p, _ = rot(p, p * 0, ax, ay)
        c, r, _ = CAM.project(p)
        print("f%4d  %d  cols %3d..%3d  rows %3d..%3d"
              % (f, YEARS[m], c.min(), c.max(), r.min(), r.max()))
        assert c.min() >= 0 and c.max() < G.cols, "clipped horizontally"
        assert r.min() >= 0 and r.max() < G.rows, "clipped vertically"
    contact(stills, "/tmp/tipped_sheet.png", cols=3)
    print("sheet -> /tmp/tipped_sheet.png")


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        check()
    else:
        total = HOLD_IN + MONTHS + HOLD_OUT
        with Encoder(OUT, G, fps=FPS) as enc:
            for f in range(total):
                enc.write(draw(f))
                if f % 60 == 0:
                    print("  %d/%d" % (f, total), flush=True)
        print("wrote", OUT, total / FPS, "s")
