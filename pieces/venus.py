#!/usr/bin/env python3
"""THE STOPWATCH — on Sept 14 the Moon occults Venus; the duration IS a size.

Current-events piece (operator thread #2). On 2026-09-14, ~09:26-13:42 UT,
the 3-day waxing crescent Moon (12% lit) passes in front of Venus for
Europe/Africa/Middle East (daylight) and S/SE Asia (dusk). Venus vanishes
behind the DARK leading limb.

The claim: an occultation is a measurement.
  - a star is a point: it vanishes in ~1/50 s (diffraction fringe time) —
    between two frames of this video.
  - Venus is a disk 36.76" wide (computed for the date from Keplerian
    elements). The lunar limb closes on it at 0.525"/s (mean lunar motion
    0.549"/s minus Venus's own 0.024"/s). Crossing the disk takes
    36.76 / 0.525 = 70.0 seconds.
  - duration x limb speed = angular size; x distance (0.454 AU) = 12,100 km.
    A stopwatch measures a planet.

Ephemeris computed in-script (JPL approximate elements 1800-2050), cross-
checked against published values (mag -4.8 near greatest brilliancy Sep 18;
Moon 12% lit — universetoday/starwalk/earthsky, Sept 2026).

Geometry drawn geocentric-mean, central chord; relative motion folded into
the Moon (Venus at origin). Time compression labeled on screen (60x, then
5x); the stopwatch counts true sky seconds. Silent piece.

Acts (30 fps):
  A0 TITLE 0..104     frozen sky, title text
  A1 WIDE  105..344   60x. crescent Moon closes on the Venus glare-dot
  A2 ZOOM  345..554   camera 0.38 -> 12 px/arcsec; 60x -> 5x. Venus
                      resolves into its own crescent (29% lit)
  A3 STAR  555..674   5x. a labeled star crosses the dark limb: gone in
                      one frame. first contact at exactly f=675
  A4 FADE  675..1094  5x. stopwatch 0.0 -> 70.0 sky-seconds while the
                      dark limb crosses the disk. last light ~57 s
  A5 CLOSE 1095..1334 the arithmetic; the where/when; stars keep dying
                      behind the advancing limb the whole time
"""

import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- ephemeris

def _jd(y, mo, d):
    a = (14 - mo) // 12
    yy = y + 4800 - a
    mm = mo + 12 * a - 3
    return d + (153 * mm + 2) // 5 + 365 * yy + yy // 4 - yy // 100 \
        + yy // 400 - 32045.5

# JPL approximate elements (1800-2050): a e I L varpi Omega + rates/century
_VENUS = (0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718,
          76.67984255, 0.00000390, -0.00004107, -0.00078890, 58517.81538729,
          0.00268329, -0.27769418)
_EARTH = (1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193,
          0.0, 0.00000562, -0.00004392, -0.01294668, 35999.37244981,
          0.32327364, 0.0)

def _helio(el, T):
    a0, e0, I0, L0, w0, O0, da, de, dI, dL, dw, dO = el
    a = a0 + da * T
    e = e0 + de * T
    I = math.radians(I0 + dI * T)
    L = L0 + dL * T
    w = w0 + dw * T
    O = O0 + dO * T
    M = math.radians((L - w) % 360.0)
    om = math.radians(w - O)
    Om = math.radians(O)
    E = M
    for _ in range(60):
        E = M + e * math.sin(E)
    xp = a * (math.cos(E) - e)
    yp = a * math.sqrt(1 - e * e) * math.sin(E)
    co, so = math.cos(om), math.sin(om)
    cO, sO = math.cos(Om), math.sin(Om)
    ci, si = math.cos(I), math.sin(I)
    return ((co * cO - so * sO * ci) * xp + (-so * cO - co * sO * ci) * yp,
            (co * sO + so * cO * ci) * xp + (-so * sO + co * cO * ci) * yp,
            so * si * xp + co * si * yp)

def _state(T):
    ev = _helio(_VENUS, T)
    ee = _helio(_EARTH, T)
    g = [ev[i] - ee[i] for i in range(3)]
    d = math.sqrt(sum(c * c for c in g))
    r = math.sqrt(sum(c * c for c in ev))
    R = math.sqrt(sum(c * c for c in ee))
    lam = math.degrees(math.atan2(g[1], g[0])) % 360
    se = [-ee[i] for i in range(3)]
    elong = math.degrees(math.acos(
        sum(se[i] * g[i] for i in range(3)) / (R * d)))
    vs = [-ev[i] for i in range(3)]
    ve = [ee[i] - ev[i] for i in range(3)]
    alpha = math.degrees(math.acos(
        sum(vs[i] * ve[i] for i in range(3)) / (r * d)))
    return d, r, R, lam, elong, alpha

_T = (_jd(2026, 9, 14.5) - 2451545.0) / 36525.0     # 2026-09-14 12:00 UT
D_AU, R_SUN_V, R_SUN_E, _LAM, ELONG, ALPHA = _state(_T)
AU_KM = 149597870.7
D_KM = D_AU * AU_KM
VENUS_DIAM_KM = 12103.6
THETA_V = math.degrees(VENUS_DIAM_KM / D_KM) * 3600      # arcsec
ILLUM_V = (1 + math.cos(math.radians(ALPHA))) / 2

_dt = 0.25 / 36525.0
_l1 = _state(_T - _dt)[3]
_l2 = _state(_T + _dt)[3]
_dl = _l2 - _l1
if _dl > 180:
    _dl -= 360
if _dl < -180:
    _dl += 360
RATE_VENUS = _dl * 3600 / (0.5 * 86400)                  # arcsec/s geocentric
RATE_MOON = 360 * 3600 / (27.321661 * 86400)             # mean sidereal
RATE_REL = RATE_MOON - RATE_VENUS
FADE_S = THETA_V / RATE_REL                              # central crossing

# Mallama (2018) Venus magnitude
_a = ALPHA
MAG_V = (-4.384 + 5 * math.log10(R_SUN_V * D_AU) - 1.044e-3 * _a
         + 3.687e-4 * _a ** 2 - 2.814e-6 * _a ** 3 + 8.938e-9 * _a ** 4)

# Moon on Sep 14 ~12 UT: new moon Sep 11 03:27 UT (published)
MOON_AGE = _jd(2026, 9, 14.5) - (_jd(2026, 9, 11.0) + (3 + 27 / 60) / 24)
MOON_ELONG = MOON_AGE * 12.19                            # mean deg/day
ILLUM_M = math.sin(math.radians(MOON_ELONG / 2)) ** 2

# ------------------------------------------------------------ scene geometry
W, H = 1080, 1920
FPS = 30
RV = THETA_V / 2                # Venus radius, arcsec
RM = 1865.0 / 2                 # Moon MEAN radius, arcsec (stated as mean)
KM = 1 - 2 * ILLUM_M            # crescent thresholds: lit iff u > K*sqrt(R^2-v^2)
KV = 1 - 2 * ILLUM_V
SUN = (-0.769, 0.639)           # unit vector body->Sun, screen coords (y down)
SUNP = (0.639, 0.769)           # perpendicular (cusp axis)

XM0 = -(RM + RV)                # Moon centre x at first contact (t=0), y=0

F_TITLE, F_A1, F_A2, F_A3, F_A4, F_END = 105, 345, 555, 675, 1095, 1335
DUR = F_END / FPS

# sky time: 0x (title) / 60x (wide) / 60->5 ramp (zoom) / 5x after
_speed = np.zeros(F_END)
_speed[F_TITLE:F_A1] = 60.0
_speed[F_A1:F_A2] = np.linspace(60.0, 5.0, F_A2 - F_A1, endpoint=False)
_speed[F_A2:] = 5.0
T_SKY = np.concatenate([[0.0], np.cumsum(_speed) / FPS])[:F_END]
T_SKY = T_SKY - T_SKY[F_A3]     # first contact exactly at f = F_A3

def moon_x(t):
    return XM0 + RATE_REL * t

# last light: max over lit Venus points of cover time
def _last_light():
    ys = np.linspace(-RV, RV, 800)
    xs = np.linspace(-RV, RV, 800)
    Xg, Yg = np.meshgrid(xs, ys)
    r2 = Xg ** 2 + Yg ** 2
    u = Xg * SUN[0] + Yg * SUN[1]
    v = Xg * SUNP[0] + Yg * SUNP[1]
    lit = (r2 < RV ** 2) & (u > KV * np.sqrt(np.maximum(RV ** 2 - v ** 2, 0)))
    # point (x,y) covered when moon_x(t) + sqrt(RM^2-y^2) >= x
    tcov = (Xg[lit] - (np.sqrt(RM ** 2 - Yg[lit] ** 2) - RM) - XM0 - RM) \
        / RATE_REL
    return float(tcov.max())

T_LL = _last_light()

# camera: (scale px/arcsec, world cx); world cy = 0 always
SCALE_W, CX_W = 0.38, -1000.0
SCALE_C, CX_C = 12.0, -12.5
YC = 880                        # screen y of world y=0

def _smooth(u):
    return u * u * (3 - 2 * u)

def camera(f):
    if f < F_A1:
        return SCALE_W, CX_W
    if f < F_A2:
        # trap 37: a cx lerp against a geometric scale threw Venus off-frame
        # mid-zoom. Interpolate Venus's SCREEN position instead and derive cx.
        e = _smooth((f - F_A1) / (F_A2 - F_A1))
        scale = SCALE_W * (SCALE_C / SCALE_W) ** e
        vx_w = 540 - CX_W * SCALE_W          # Venus screen x at the ends
        vx_c = 540 - CX_C * SCALE_C
        vx = vx_w + (vx_c - vx_w) * e
        return scale, -(vx - 540) / scale
    return SCALE_C, CX_C

# ------------------------------------------------------------------- stars
_rng = np.random.default_rng(20260914)
STARS_WIDE = np.column_stack([
    _rng.uniform(-2600, 600, 150), _rng.uniform(-2900, 2900, 150),
    _rng.uniform(110, 205, 150)])
STARS_CLOSE = np.column_stack([
    _rng.uniform(-200, 120, 90), _rng.uniform(-140, 140, 90),
    _rng.uniform(95, 165, 90)])
STAR_TEST = (-390.0, 300.0)     # wide star that the limb eats during A1
WINK = (-24.0, 40.0)            # the labeled star, eaten during A3
C_STAR = (188, 196, 210)

# noise lattices for the lunar surface (moon-fixed coords, trap 10/28)
_G1 = _rng.standard_normal((16, 16))
_G2 = _rng.standard_normal((48, 48))

def _vnoise(px, py, lam, G):
    gx = px / lam + G.shape[1] / 2
    gy = py / lam + G.shape[0] / 2
    ix = np.clip(np.floor(gx).astype(int), 0, G.shape[1] - 2)
    iy = np.clip(np.floor(gy).astype(int), 0, G.shape[0] - 2)
    fx = _smooth(np.clip(gx - ix, 0, 1))
    fy = _smooth(np.clip(gy - iy, 0, 1))
    a = G[iy, ix] * (1 - fx) + G[iy, ix + 1] * fx
    b = G[iy + 1, ix] * (1 - fx) + G[iy + 1, ix + 1] * fx
    return a * (1 - fy) + b * fy

# ------------------------------------------------------------------ palette
BG = (10, 12, 20)
C_ES = (27, 30, 38)             # earthshine disk
C_MLIT = (232, 228, 218)        # lunar crescent
C_VLIT = (252, 248, 238)        # Venus crescent (brighter surface — true)
C_TXT = (236, 240, 246)         # RESERVED for text (fence-checked)
C_SUB = (152, 160, 174)
C_ACC = (252, 178, 41)          # stopwatch amber
C_HALO = (5, 6, 10)

FDIR = "/usr/share/fonts/truetype/dejavu/"
def _font(name, size):
    return ImageFont.truetype(FDIR + name, size)

def _fit(draw, text, name, size, maxw):
    while size > 14:
        f = _font(name, size)
        if draw.textlength(text, font=f) <= maxw:
            return f
        size -= 2
    return _font(name, 14)

# ------------------------------------------------------------------- render

def draw_sky(f):
    """numpy float32 sky: bg, stars, Venus, Moon. No text."""
    scale, cx = camera(f)
    t = T_SKY[f]
    xm = moon_x(t)
    img = np.empty((H, W, 3), np.float32)
    img[:] = BG

    def to_screen(x, y):
        return 540 + (x - cx) * scale, YC + y * scale

    # --- Venus (drawn before the Moon: it passes BEHIND)
    def body_patch(xc, yc, R, K, tone, noise=False):
        sx0, sy0 = to_screen(xc - R, yc - R)
        sx1, sy1 = to_screen(xc + R, yc + R)
        a0, b0 = max(0, int(sx0) - 2), max(0, int(sy0) - 2)
        a1, b1 = min(W, int(sx1) + 3), min(H, int(sy1) + 3)
        if a1 <= a0 or b1 <= b0:
            return
        xs = cx + (np.arange(a0, a1) - 540) / scale - xc
        ys = (np.arange(b0, b1) - YC) / scale - yc
        dx = xs[None, :]
        dy = ys[:, None]
        r = np.sqrt(dx ** 2 + dy ** 2)
        w = max(1.5 / scale, 1e-6)
        a_disk = np.clip((R - r) / w + 0.5, 0, 1)
        if a_disk.max() <= 0:
            return
        u = dx * SUN[0] + dy * SUN[1]
        v = dx * SUNP[0] + dy * SUNP[1]
        g = u - K * np.sqrt(np.maximum(R ** 2 - v ** 2, 0))
        a_lit = a_disk * np.clip(g / w + 0.5, 0, 1)
        sub = img[b0:b1, a0:a1]
        if noise:
            n = (1 + 0.09 * _vnoise(dx + 0 * dy, dy + 0 * dx, 280, _G1)
                 + 0.06 * _vnoise(dx + 0 * dy, dy + 0 * dx, 90, _G2))
            es = np.clip(np.array(C_ES, np.float32)[None, None, :]
                         * n[:, :, None], 0, 255)
            lt = np.clip(np.array(tone, np.float32)[None, None, :]
                         * n[:, :, None], 0, 255)
            sub[:] = sub * (1 - a_disk[:, :, None]) + es * a_disk[:, :, None]
            sub[:] = sub * (1 - a_lit[:, :, None]) + lt * a_lit[:, :, None]
        else:
            # Venus: dark side is invisible (no earthshine) — crescent only
            lt = np.array(tone, np.float32)[None, None, :]
            sub[:] = sub * (1 - a_lit[:, :, None]) + lt * a_lit[:, :, None]

    body_patch(0.0, 0.0, RV, KV, C_VLIT, noise=False)

    # Venus glare bloom while unresolved (what the eye sees at mag -4.8)
    bloom = float(np.clip((1.5 - scale) / 1.4, 0, 1))
    if bloom > 0:
        sx, sy = to_screen(0, 0)
        sxi, syi = int(round(sx)), int(round(sy))
        r = 30
        x0, x1 = max(0, sxi - r), min(W, sxi + r + 1)
        y0, y1 = max(0, syi - r), min(H, syi + r + 1)
        gx = np.arange(x0, x1) - sx
        gy = np.arange(y0, y1) - sy
        g = np.exp(-(gx[None, :] ** 2 + gy[:, None] ** 2) / (2 * 6.5 ** 2))
        add = bloom * g[:, :, None] * np.array((215, 205, 185), np.float32)
        img[y0:y1, x0:x1] = np.minimum(img[y0:y1, x0:x1] + add, 255)

    # --- Moon
    body_patch(xm, 0.0, RM, KM, C_MLIT, noise=True)

    # --- stars LAST, occluded BINARY by either disk. A star is a point:
    # its drawn width is bloom, all light from one spot, so occultation is
    # all-or-nothing — no partial frames (the AA edge sliding over the dot
    # was a render fault, caught by the no-partial-frames check).
    close_a = float(np.clip((scale - 1.0) / 4.0, 0, 1))
    star_sets = [(STARS_WIDE, 1.0, 2), (STARS_CLOSE, close_a, 2)]
    tx, ty = STAR_TEST
    star_sets.append((np.array([[tx, ty, 205.0]]), 1.0, 2))
    for arr, aset, rad in star_sets:
        if aset <= 0:
            continue
        for x, y, b in arr:
            if (x - xm) ** 2 + y ** 2 < RM ** 2 or x ** 2 + y ** 2 < RV ** 2:
                continue                       # behind the Moon / Venus
            sx, sy = to_screen(x, y)
            sxi, syi = int(round(sx)), int(round(sy))
            if -4 < sxi < W + 4 and -4 < syi < H + 4:
                x0, x1 = max(0, sxi - rad), min(W, sxi + rad + 1)
                y0, y1 = max(0, syi - rad), min(H, syi + rad + 1)
                v = b * aset
                img[y0:y1, x0:x1] = np.maximum(
                    img[y0:y1, x0:x1], (v, v, v * 1.06))
    # wink star: 7 px wide (trap 79), bright
    wx, wy = WINK
    if (wx - xm) ** 2 + wy ** 2 >= RM ** 2:
        sx, sy = to_screen(wx, wy)
        sxi, syi = int(round(sx)), int(round(sy))
        if 0 <= sxi < W and 0 <= syi < H:
            x0, x1 = max(0, sxi - 3), min(W, sxi + 4)
            y0, y1 = max(0, syi - 3), min(H, syi + 4)
            img[y0:y1, x0:x1] = np.maximum(img[y0:y1, x0:x1],
                                           (235, 238, 245))
    return img

# ------------------------------------------------------------------- text

def stopwatch_str(f):
    if f < F_A3:
        return None
    return f"{min(T_SKY[f], FADE_S):.1f} s"

def _alpha(f, on, fade=10):
    return float(np.clip((f - on) / fade, 0, 1))

def _put(dr, xy, text, font, fill, a, anchor="ma"):
    if a <= 0:
        return
    col = tuple(int(BG[i] + (fill[i] - BG[i]) * a) for i in range(3)) \
        if a < 1 else fill
    dr.text(xy, text, font=font, fill=col, anchor=anchor,
            stroke_width=3, stroke_fill=C_HALO)

F_WINK_LBL = 570

def draw(f):
    img = draw_sky(f)
    im = Image.fromarray(img.astype(np.uint8))
    dr = ImageDraw.Draw(im)
    t = T_SKY[f]
    xm = moon_x(t)
    scale, cx = camera(f)

    # A0 title
    if f < F_A1:
        a = _alpha(f, 4) * (1 - _alpha(f, F_TITLE + 20))
        _put(dr, (540, 250), "September 14", _font("DejaVuSans-Bold.ttf", 92),
             C_TXT, a)
        _put(dr, (540, 372), "the Moon will hide Venus",
             _font("DejaVuSans.ttf", 58), C_TXT, a)
        _put(dr, (540, 470), "computed from orbital elements, then drawn",
             _font("DejaVuSans.ttf", 36), C_SUB, a)
    # A1 labels
    if F_TITLE + 30 <= f < F_A1:
        a = _alpha(f, F_TITLE + 30)
        mx = 540 + (xm - cx) * scale
        _put(dr, (mx, 460), "the Moon — 3 days old, 12% lit",
             _font("DejaVuSans.ttf", 40), C_TXT, a)
        _put(dr, (850, 952), "Venus — mag −4.8",
             _font("DejaVuSans.ttf", 40), C_TXT, a)
    # speed chip (bottom, safe)
    chip = None
    if F_TITLE <= f < F_A1:
        chip = "60× speed"
    elif F_A1 <= f < F_A2:
        chip = "zooming — 60× → 5×"
    elif F_A2 <= f < F_A3:
        chip = "5× speed"
    if chip:
        _put(dr, (540, 1560), chip, _font("DejaVuSans.ttf", 38), C_SUB, 1.0)

    # A2/A3 labels
    if F_A2 + 5 <= f < 700:
        a = _alpha(f, F_A2 + 5) * (1 - _alpha(f, 700, 20))
        _put(dr, (690, 1150), "Venus — 36.8″ wide, 29% lit",
             _font("DejaVuSans.ttf", 42), C_TXT, a)
        _put(dr, (70, 600), "the Moon — its dark edge leads",
             _font("DejaVuSans.ttf", 42), C_TXT, a, anchor="la")
    # wink label
    if F_WINK_LBL <= f < 672:   # off before first contact — must not read
                                # as describing Venus
        a = _alpha(f, F_WINK_LBL)
        wx, wy = WINK
        gone = (wx - xm) ** 2 + wy ** 2 < RM ** 2
        txt = "gone between two frames" if gone else "a star — a point"
        _put(dr, (402, 1430), txt, _font("DejaVuSans.ttf", 40), C_TXT, a)

    # A4 stopwatch
    sw = stopwatch_str(f)
    if sw is not None:
        a = _alpha(f, F_A3, 8)
        _put(dr, (540, 232), sw, _font("DejaVuSansMono-Bold.ttf", 118),
             C_ACC, a)
        _put(dr, (540, 392), "5× speed — the clock counts real sky seconds",
             _font("DejaVuSans.ttf", 36), C_SUB, a)
    # last light marker
    f_ll = F_A3 + int(round(T_LL * FPS / 5))
    if f_ll <= f < F_A4 + 60:
        a = _alpha(f, f_ll)
        _put(dr, (760, 1180), f"last light — {T_LL:.0f} s",
             _font("DejaVuSans.ttf", 40), C_TXT, a)

    # A5 closing lines
    LINES = [
        (F_A4 + 10, f"last light at {T_LL:.0f} s — the full disk took 70.0 s"),
        (F_A4 + 65, "70.0 s × 0.525″/s = 36.8 arcseconds"),
        (F_A4 + 120, "at 68 million km, that is 12,100 km —"),
        (F_A4 + 132, "a planet, measured with a stopwatch"),
        (F_A4 + 185, "Sept 14 · ~09:26–13:42 UT · daylight in Europe"),
        (F_A4 + 197, "& Africa · dusk in South & Southeast Asia"),
        (F_A4 + 225, "Venus returns from the bright limb an hour later"),
    ]
    y = 560
    for i, (on, txt) in enumerate(LINES):
        if f >= on:
            a = _alpha(f, on)
            fill = C_ACC if i == 1 else C_TXT
            fnt = _fit(dr, txt, "DejaVuSans-Bold.ttf" if i == 1
                       else "DejaVuSans.ttf", 46, 950)
            _put(dr, (540, y), txt, fnt, fill, a)
        y += 74
    return np.asarray(im, np.uint8)

# ------------------------------------------------------------------- checks

CHECKS = []
def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok)))
    tag = "ok" if ok else "FAIL"
    print(f"  [{tag}] {name} {detail}")

def lit_venus_count(img):
    """pixels matching the Venus crescent tone (bounded colour match)."""
    m = (np.abs(img[:, :, 0].astype(int) - C_VLIT[0]) < 26) & \
        (np.abs(img[:, :, 1].astype(int) - C_VLIT[1]) < 26) & \
        (np.abs(img[:, :, 2].astype(int) - C_VLIT[2]) < 26)
    return int(m.sum()), m

def txt_mask(img):
    return (img[:, :, 0] == C_TXT[0]) & (img[:, :, 1] == C_TXT[1]) & \
           (img[:, :, 2] == C_TXT[2])

def acc_mask(img):
    return (img[:, :, 0] == C_ACC[0]) & (img[:, :, 1] == C_ACC[1]) & \
           (img[:, :, 2] == C_ACC[2])

def run_checks():
    print("== model checks ==")
    check("distance 0.454 AU", abs(D_AU - 0.4539) < 0.003, f"{D_AU:.4f}")
    check("Venus diameter ~36.8 arcsec", abs(THETA_V - 36.76) < 0.15,
          f"{THETA_V:.2f}")
    check("Venus 29% lit", abs(ILLUM_V - 0.291) < 0.01, f"{ILLUM_V:.3f}")
    check("elongation ~41 deg (evening)", abs(ELONG - 41.1) < 0.5,
          f"{ELONG:.2f}")
    check("magnitude ~ -4.8 (published -4.5..-4.8)",
          -4.85 < MAG_V < -4.5, f"{MAG_V:.2f}")
    check("Moon 12% lit (published 12%)", abs(ILLUM_M - 0.12) < 0.015,
          f"{ILLUM_M:.3f}")
    check("relative rate 0.525 arcsec/s", abs(RATE_REL - 0.5251) < 0.003,
          f"{RATE_REL:.4f}")
    check("central crossing 70.0 s", abs(FADE_S - 70.0) < 0.3,
          f"{FADE_S:.2f}")
    check("Moon elong == Venus elong (that is WHY they meet)",
          abs(MOON_ELONG - ELONG) < 1.5,
          f"moon {MOON_ELONG:.1f} vs venus {ELONG:.1f}")
    check("last light 50..65 s in", 50 < T_LL < 65, f"{T_LL:.1f}")

    dT = np.diff(T_SKY)
    check("sky time monotone, steps <= 2.0 s/frame",
          dT.min() >= -1e-9 and dT.max() <= 2.0 + 1e-9,
          f"max {dT.max():.3f}")
    check("first contact exactly at f=675", abs(T_SKY[F_A3]) < 1e-9)
    # full immersion: centre distance == RM - RV
    t_imm = (2 * RV) / RATE_REL
    f_imm = F_A3 + t_imm * FPS / 5
    check("full immersion at f=1095 (A4 end)", abs(f_imm - F_A4) < 1.0,
          f"{f_imm:.1f}")

    # camera/framing every frame (trap 37): Venus centre on screen; the
    # leading limb on screen from A1 through A4
    bad = 0
    for f in range(0, F_END, 3):
        scale, cx = camera(f)
        vx = 540 - cx * scale
        if not (60 < vx < 1020):
            bad += 1
        if F_TITLE <= f <= F_A4:
            lx = 540 + (moon_x(T_SKY[f]) + RM - cx) * scale
            if not (-200 < lx < 1080):
                bad += 1
    check("framing holds on every sampled frame", bad == 0, f"bad={bad}")

    f_wink_model = None
    for f in range(F_A2, F_A4):
        if (WINK[0] - moon_x(T_SKY[f])) ** 2 + WINK[1] ** 2 < RM ** 2:
            f_wink_model = f
            break
    check("wink star eaten inside A3", F_A2 < f_wink_model < F_A3,
          f"f={f_wink_model}")

    print("== pixel checks ==")
    fr20 = draw(20)
    m = txt_mask(fr20)
    check("f20 title ink present", m[200:520, :].sum() > 800,
          f"{int(m[200:520, :].sum())}")
    check("f20 no text in top unsafe band", m[:192, :].sum() == 0)
    check("f20 no text below y=1632", m[1632:, :].sum() == 0)

    fr200 = draw(200)
    # moon lit fraction measured on drawn pixels
    scale, cx = camera(200)
    xm = moon_x(T_SKY[200])
    xs = cx + (np.arange(W) - 540) / scale - xm
    ys = (np.arange(H) - YC) / scale
    r2 = xs[None, :] ** 2 + ys[:, None] ** 2
    disk = r2 < (RM * 0.98) ** 2
    bright = fr200[:, :, 0].astype(int) > 150
    frac = (disk & bright).sum() / disk.sum()
    check("f200 lunar crescent ~12% of disk", 0.09 < frac < 0.15,
          f"{frac:.3f}")
    es = fr200[disk & ~bright][:, 0].astype(int)
    check("f200 earthshine above bg, below crescent",
          BG[0] + 4 < np.median(es) < 80, f"med {np.median(es):.0f}")
    # Venus bloom at its projected position
    vx = int(540 - cx * scale)
    patch = fr200[YC - 6:YC + 7, vx - 6:vx + 7, 0]
    check("f200 Venus glare present", patch.max() > 140,
          f"max {patch.max()}")
    # test star: visible at f120, occulted at f200
    def star_px(f, wx, wy):
        img = draw_sky(f)
        s, c = camera(f)
        sx = int(round(540 + (wx - c) * s))
        sy = int(round(YC + wy * s))
        if not (0 <= sx < W and 0 <= sy < H):
            return 0
        p = img[max(0, sy - 4):sy + 5, max(0, sx - 4):sx + 5, 0]
        return int((p > 90).sum())
    check("test star visible f120", star_px(120, *STAR_TEST) > 0)
    check("test star occulted f200 (the edge eats stars)",
          star_px(200, *STAR_TEST) == 0)

    fr560 = draw_sky(560)
    n560, m560 = lit_venus_count(fr560)
    check("f560 Venus crescent drawn", n560 > 3000, f"{n560}")
    yy, xx = np.nonzero(m560)
    # span along the cusp axis == full diameter (crescent spans cusp-to-cusp)
    proj = (xx - (540 - camera(560)[1] * 12)) * SUNP[0] + (yy - YC) * SUNP[1]
    span = proj.max() - proj.min()
    check("f560 cusp-to-cusp span ~ 441 px (= 36.8 arcsec x 12)",
          abs(span - THETA_V * 12) < 10, f"{span:.0f}")
    # orientation coupling (trap 66): BOTH crescents' lit centroids sunward,
    # each measured at a frame where that body's crescent is on screen
    # (at f560 the Moon's lit limb is far off-frame; "bright & not Venus"
    # there measured the STARS — trap 61)
    vcx = (xx.mean() - (540 - camera(560)[1] * 12))
    vcy = yy.mean() - YC
    uV = vcx * SUN[0] + vcy * SUN[1]
    s200, c200 = camera(200)
    xm200 = moon_x(T_SKY[200])
    sxm = 540 + (xm200 - c200) * s200
    mb = fr200[:, :, 0].astype(int) > 170
    myy, mxx = np.nonzero(mb)
    keep = ((mxx - sxm) / s200) ** 2 + ((myy - YC) / s200) ** 2 < RM ** 2
    uM = ((mxx[keep].mean() - sxm) * SUN[0]
          + (myy[keep].mean() - YC) * SUN[1])
    check("crescents bow to the same Sun (lit centroids sunward)",
          uV > 0 and uM > 0, f"uV {uV:.0f} uM {uM:.0f}")

    # wink: at most one partial frame
    base = star_px(600, *WINK)
    partial = 0
    f_gone = None
    for f in range(f_wink_model - 4, f_wink_model + 4):
        p = star_px(f, *WINK)
        if 0 < p < base:
            partial += 1
        if p == 0 and f_gone is None:
            f_gone = f
    check("wink star present before", base >= 40, f"{base}px")
    check("star gone within 1 frame of model", abs(f_gone - f_wink_model) <= 1,
          f"gone f{f_gone}")
    check("no partial-fade frames (a point has no width)", partial == 0,
          f"{partial}")

    # fade monotone + light gone when predicted
    counts = []
    for f in (675, 745, 815, 885, 955, 1025):
        counts.append(lit_venus_count(draw_sky(f))[0])
    check("Venus light strictly decreasing through A4",
          all(a > b for a, b in zip(counts, counts[1:])), f"{counts}")
    f_ll_model = F_A3 + T_LL * FPS / 5
    lo, hi = int(f_ll_model) - 4, int(f_ll_model) + 5
    f_gone_v = None
    for f in range(lo, hi):
        if lit_venus_count(draw_sky(f))[0] == 0:
            f_gone_v = f
            break
    check("last Venus light at predicted frame",
          f_gone_v is not None and abs(f_gone_v - f_ll_model) <= 4,
          f"px f{f_gone_v} vs model {f_ll_model:.0f}")
    check("Venus stays gone (f1200)", lit_venus_count(draw_sky(1200))[0] == 0)

    # stopwatch
    check("stopwatch strings", stopwatch_str(674) is None
          and stopwatch_str(675) == "0.0 s"
          and stopwatch_str(F_A3 + 240) == "40.0 s"
          and stopwatch_str(1300) == "70.0 s")
    fr915 = draw(915)
    am = acc_mask(fr915)
    check("f915 amber stopwatch ink in its band",
          am[192:420, :].sum() > 400, f"{int(am[192:420, :].sum())}")
    check("f915 amber nowhere else", am[420:, :].sum() == 0)

    # safe bands across probes (text colours only — graphics may bleed)
    ok = True
    for f in (20, 200, 560, 630, 915, 1200, 1330):
        img = draw(f)
        tm = txt_mask(img) | acc_mask(img)
        if tm[:192, :].sum() or tm[1632:, :].sum():
            ok = False
    check("no text outside safe area on any probe", ok)

    fr1330 = draw(1330)
    tm = txt_mask(fr1330) | acc_mask(fr1330)
    check("closing lines inked", tm[520:1130, :].sum() > 4000,
          f"{int(tm[520:1130, :].sum())}")
    lit = (fr1330.astype(int).sum(2) > 120).mean()
    check("final frame lit fraction sane", 0.004 < lit < 0.5, f"{lit:.3f}")

    fails = [n for n, o in CHECKS if not o]
    if fails:
        print(f"\n{len(fails)} CHECK(S) FAILED:", fails)
        sys.exit(1)
    print(f"\nall {len(CHECKS)} render checks passed")

# ------------------------------------------------------------------- encode

OUT_DIR = "out"
STAMP = "000937"
OUT_MP4 = f"{OUT_DIR}/venus_{STAMP}_final.mp4"

def encode():
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "18",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT_MP4]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stderr=subprocess.DEVNULL)
    for f in range(F_END):
        p.stdin.write(draw(f).tobytes())
        if f % 150 == 0:
            print(f"  frame {f}/{F_END}")
    p.stdin.close()
    p.wait()
    assert p.returncode == 0, "ffmpeg failed"

def decode_frame(n, crop=None):
    vf = f"select=eq(n\\,{n})"
    if crop:
        w, h, x, y = crop
        vf += f",crop={w}:{h}:{x}:{y}"
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", OUT_MP4, "-vf", vf,
         "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True).stdout
    if crop:
        return np.frombuffer(out, np.uint8).reshape(crop[1], crop[0], 3)
    return np.frombuffer(out, np.uint8).reshape(H, W, 3)

def encode_checks():
    print("== encode checks ==")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v",
         "-show_entries", "stream=nb_frames,duration,width,height,codec_name",
         "-of", "csv=p=0", OUT_MP4], capture_output=True, text=True).stdout
    parts = probe.strip().split(",")
    codec, wid, hei, dur, nf = parts[0], int(parts[1]), int(parts[2]), \
        float(parts[3]), int(parts[4])
    check("encode: h264 1080x1920",
          codec == "h264" and wid == W and hei == H)
    check("encode: 1335 frames / 44.50 s",
          nf == F_END and abs(dur - DUR) < 0.06, f"{nf} fr {dur:.2f} s")
    d560 = decode_frame(560, (440, 440, 470, 660))
    check("decoded f560: Venus crescent survives",
          int((d560[:, :, 0].astype(int) > 190).sum()) > 2500,
          f"{int((d560[:, :, 0].astype(int) > 190).sum())}")
    d1330 = decode_frame(1330, (960, 620, 60, 520))
    check("decoded f1330: closing text survives",
          int((d1330.astype(int).sum(2) > 480).sum()) > 3000,
          f"{int((d1330.astype(int).sum(2) > 480).sum())}")
    fails = [n for n, o in CHECKS if not o]
    if fails:
        print(f"\n{len(fails)} ENCODE CHECK(S) FAILED:", fails)
        sys.exit(1)
    print(f"\nall {len(CHECKS)} checks passed (render + encode)")

def stills():
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in (60, 200, 460, 585, 640, 700, 915, 1050, 1150, 1330):
        Image.fromarray(draw(f)).save(f"{OUT_DIR}/venus_f{f:04d}.png")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-pattern_type", "glob",
         "-i", f"{OUT_DIR}/venus_f*.png",
         "-filter_complex", "scale=270:-1,tile=5x2",
         f"{OUT_DIR}/venus_sheet.png"])
    # watch-size gate (trap 67)
    for f, tag in ((640, "wink"), (915, "fade")):
        Image.fromarray(draw(f)).resize((360, 640)).save(
            f"{OUT_DIR}/venus_gate360_{tag}.png")

if __name__ == "__main__":
    print(f"THE STOPWATCH — d={D_AU:.4f} AU, theta={THETA_V:.2f}\", "
          f"lit={ILLUM_V:.1%}, mag={MAG_V:.2f}, rel={RATE_REL:.4f}\"/s, "
          f"crossing={FADE_S:.1f} s, last light={T_LL:.1f} s")
    run_checks()
    stills()
    if "--stills" in sys.argv:
        sys.exit(0)
    encode()
    encode_checks()
    print(f"\n{OUT_MP4}")
    print("NOT verified by any check here: the actual event — local contact "
          "times shift by many minutes with the observer's parallax, the "
          "chord is rarely central, and the Moon's true size/rate on the day "
          "differ a few % from the mean values drawn. And whether anyone "
          "takes a stopwatch outside on Sept 14 — that part happens on "
          "Earth, not in this file. (trap 68)")
