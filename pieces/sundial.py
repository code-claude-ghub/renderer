#!/usr/bin/env python3
"""
NOON IS A DECISION
==================

A post, seen from straight above, and its shadow, over one real day in
Indianapolis.  Nothing about the motion is authored: the sun's altitude and
azimuth come from the NOAA general solar position equations, the shadow's
length is h / tan(altitude), its bearing is the sun's bearing plus 180, and
the fuzz at its tip is the sun's own half-degree width.

The argument is the gap between two engraved numerals on the ground.  The
shadow reaches the one marked 12 and keeps going.  It does not lie along the
meridian -- true north, the vertical line -- until 1:48 in the afternoon.

Those 107.7 minutes are three separate things added together:

    +44.63 min   Indianapolis is 11.158 degrees west of the 75th meridian,
                 which is the line its time zone is actually named after,
                 and the earth turns 4 minutes per degree
    +60.00 min   daylight saving time
    + 3.05 min   the equation of time -- the earth's orbit is an ellipse
                 and its axis is tilted, so the sun runs early or late
                 against any uniform clock

Two of those three were voted on.

Usage:  python3 sundial.py [--check] [--sheet] [--render]
"""

import os
import sys
import datetime as dt

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]

from asciilib import Grid, Frame, Encoder, ink_lut, contact   # noqa: E402

# --------------------------------------------------------------------------
# the place and the day
# --------------------------------------------------------------------------
CITY = "indianapolis"
LAT = 39.7684            # degrees north
LON = -86.1581           # degrees east (negative = west)
TZ = -4.0                # hours from UTC; EDT
ZONE_MERIDIAN = -75.0    # the meridian Eastern Time is defined on

DATE = dt.date(2026, 8, 23)
DOY = DATE.timetuple().tm_yday
YEAR_DAYS = 366 if (DATE.year % 4 == 0 and
                    (DATE.year % 100 or DATE.year % 400 == 0)) else 365

SUN_RADIUS = 0.2665      # degrees; the sun's angular radius, which is what
                         # makes the tip of a shadow soft

# --------------------------------------------------------------------------
# the drawing
# --------------------------------------------------------------------------
G = Grid()
LUT = ink_lut()

PAPER = (0.925, 0.870, 0.700)     # sun-bleached ground
INK = (0.085, 0.095, 0.185)       # shadow: dark, and blue because the only
                                  # thing lighting it is the sky
OXIDE = (0.620, 0.255, 0.130)     # engraved into the ground
OX_DARK = (0.300, 0.175, 0.175)   # the same engraving, inside the shadow

GX = G.cols / 2.0 - 0.5           # 48.5 -- the post
GY = 124.0
POST_R = 6.5                      # cells; the post's radius
H_GNOMON = 60.0                   # cells; the post's height

FPS = 30
T_END = 15.0
FRAMES = int(round(FPS * T_END))

ALT_START = 2.0                   # begin and end this many degrees up, so the
                                  # shadow has a finite length in both frames

L_MAX = 420.0                     # clamp; the frame diagonal is about 200

TEXT_ROW = 48                     # running clock, above the meridian
SC = 2                            # text scale, cells per font pixel

FONT = {
    '0': "111101101101111", '1': "010110010010111", '2': "111001111100111",
    '3': "111001111001111", '4': "101101111001001", '5': "111100111001111",
    '6': "111100111101111", '7': "111001001001001", '8': "111101111101111",
    '9': "111101111001111", ':': "000010000010000", ' ': "000000000000000",
}


# --------------------------------------------------------------------------
# NOAA general solar position equations
#   gml.noaa.gov/grad/solcalc/solareqns.PDF
# --------------------------------------------------------------------------
def _gamma(hour):
    return 2.0 * np.pi / YEAR_DAYS * (DOY - 1 + (hour - 12.0) / 24.0)


def eqtime(hour):
    """equation of time, minutes"""
    g = _gamma(hour)
    return 229.18 * (0.000075
                     + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                     - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))


def declination(hour):
    """solar declination, radians"""
    g = _gamma(hour)
    return (0.006918
            - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
            - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
            - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))


def sun(tmin):
    """local clock minutes -> (altitude deg, azimuth deg clockwise from north)"""
    tmin = np.asarray(tmin, dtype=float)
    hour = tmin / 60.0
    et = eqtime(hour)
    dec = declination(hour)
    tst = tmin + et + 4.0 * LON - 60.0 * TZ
    ha = np.radians(tst / 4.0 - 180.0)
    la = np.radians(LAT)

    cz = np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)
    cz = np.clip(cz, -1.0, 1.0)
    z = np.arccos(cz)
    sz = np.where(np.abs(np.sin(z)) < 1e-9, 1e-9, np.sin(z))

    sin_az = -np.sin(ha) * np.cos(dec) / sz
    cos_az = (np.sin(dec) - np.sin(la) * cz) / (np.cos(la) * sz)
    az = np.degrees(np.arctan2(sin_az, cos_az)) % 360.0
    return 90.0 - np.degrees(z), az


# solar noon, closed form, in local clock minutes
ET12 = float(eqtime(12.0))
SNOON = 720.0 - 4.0 * LON - ET12 + TZ * 60.0

# the decomposition the piece is about
PART_LON = 4.0 * (ZONE_MERIDIAN - LON)      # degrees west of the zone meridian
PART_DST = 60.0
PART_EOT = -ET12
OFFSET = PART_LON + PART_DST + PART_EOT

# sunrise / sunset, same source
_la, _dec = np.radians(LAT), declination(12.0)
_ha0 = np.degrees(np.arccos(
    np.cos(np.radians(90.833)) / (np.cos(_la) * np.cos(_dec))
    - np.tan(_la) * np.tan(_dec)))
SUNRISE = 720.0 - 4.0 * (LON + _ha0) - ET12 + TZ * 60.0
SUNSET = 720.0 - 4.0 * (LON - _ha0) - ET12 + TZ * 60.0


def _first_above(lo, hi, alt_target):
    """clock minute in [lo, hi] where the altitude first exceeds alt_target"""
    t = np.linspace(lo, hi, 20000)
    a = sun(t)[0]
    idx = np.flatnonzero(a > alt_target)
    return float(t[idx[0]]) if len(idx) else float(lo)


T_OPEN = _first_above(SUNRISE, SNOON, ALT_START)
T_SHUT = 2.0 * SNOON - T_OPEN       # symmetric about solar noon


def clock(frame_i):
    """frame index -> local clock minutes; uniform, no time trickery"""
    return T_OPEN + (T_SHUT - T_OPEN) * (frame_i / float(FRAMES - 1))


def shadow_raw(tmin):
    """clock minutes -> (umbra, penumbra, bearing) with no clamp.

    The sun is a disc 0.533 deg across, so the shadow has no single length:
    the limb nearest the horizon casts the far edge and the far limb casts
    the near one.  That difference is the soft band at the tip, and it grows
    as 1/sin^2(altitude) -- crisp at midday, metres wide near sunset.
    """
    alt, az = sun(tmin)
    lu = H_GNOMON / np.tan(np.radians(np.maximum(alt + SUN_RADIUS, 0.05)))
    lp = H_GNOMON / np.tan(np.radians(np.maximum(alt - SUN_RADIUS, 0.02)))
    return lu, lp, (az + 180.0) % 360.0


def shadow(tmin):
    """as shadow_raw, clamped to something that can be drawn"""
    lu, lp, bearing = shadow_raw(tmin)
    return np.minimum(lu, L_MAX), np.minimum(lp, L_MAX), bearing


def tip(tmin):
    """clock minutes -> shadow tip (col, row), sun centre, unclamped"""
    alt, az = sun(tmin)
    L = H_GNOMON / np.tan(np.radians(np.maximum(alt, 0.05)))
    b = np.radians((az + 180.0) % 360.0)
    return GX + L * np.sin(b), GY - L * np.cos(b)


# --------------------------------------------------------------------------
# cell grid
# --------------------------------------------------------------------------
_R, _C = np.mgrid[0:G.rows, 0:G.cols]
_PC = _C + 0.5 - GX
_PR = _R + 0.5 - GY

TAN_SUN = np.tan(np.radians(SUN_RADIUS))


def shadow_density(tmin):
    """per-cell shadow coverage in [0, 1] -- umbra 1, penumbra soft, else 0"""
    lu, lp, bearing = shadow(tmin)
    b = np.radians(bearing)
    ux, uy = np.sin(b), -np.cos(b)          # along the shadow
    s = _PC * ux + _PR * uy                 # distance along
    t = np.abs(-_PC * uy + _PR * ux)        # distance across

    sp = np.maximum(s, 0.0)
    half = POST_R + sp * TAN_SUN            # the shadow's own soft edge
    across = np.clip((half + 0.5 - t) / (2.0 * sp * TAN_SUN + 1.0), 0.0, 1.0)
    along = np.clip((lp - s) / max(lp - lu, 0.9), 0.0, 1.0)

    d = np.minimum(across, along)
    d[s < 0.0] = 0.0

    post = np.hypot(_PC, _PR)               # the post itself
    d = np.maximum(d, np.clip(POST_R + 0.5 - post, 0.0, 1.0))
    return d


# --------------------------------------------------------------------------
# what is engraved in the ground: the meridian, the day's curve, hour ticks
# --------------------------------------------------------------------------
def _stroke(acc, c0, r0, c1, r1, width=0.9):
    """add a soft line segment to an accumulator"""
    dc, dr = c1 - c0, r1 - r0
    ln = float(np.hypot(dc, dr))
    if ln < 1e-9:
        return
    lo_c, hi_c = sorted((c0, c1))
    lo_r, hi_r = sorted((r0, r1))
    m = 3.0 + width
    cs = int(max(0, np.floor(lo_c - m)))
    ce = int(min(G.cols, np.ceil(hi_c + m)))
    rs = int(max(0, np.floor(lo_r - m)))
    re = int(min(G.rows, np.ceil(hi_r + m)))
    if cs >= ce or rs >= re:
        return
    cc = _C[rs:re, cs:ce] + 0.5
    rr = _R[rs:re, cs:ce] + 0.5
    u = np.clip(((cc - c0) * dc + (rr - r0) * dr) / (ln * ln), 0.0, 1.0)
    dist = np.hypot(cc - (c0 + u * dc), rr - (r0 + u * dr))
    acc[rs:re, cs:ce] = np.maximum(acc[rs:re, cs:ce],
                                   np.clip(width + 0.5 - dist, 0.0, 1.0))


def _vertex_row():
    return float(tip(SNOON)[1])


def build_engraving():
    acc = np.zeros((G.rows, G.cols))

    # the meridian: due north from the post. true noon lies on this line.
    # It stops a little above the vertex -- run to the top of the frame it
    # is just a rule through empty paper.
    _stroke(acc, GX, GY - POST_R, GX, _vertex_row() - 20.0, width=0.85)

    # the day's curve -- where the tip of this shadow goes, all day
    ts = np.linspace(T_OPEN, T_SHUT, 6000)
    cs, rs = tip(ts)
    keep = ((cs > 4) & (cs < G.cols - 4) & (rs > 4) & (rs < G.rows - 4))
    cs, rs = cs[keep], rs[keep]
    for i in range(len(cs) - 1):
        _stroke(acc, cs[i], rs[i], cs[i + 1], rs[i + 1], width=0.75)

    # a tick at every whole hour on the clock
    ticks = []
    for hh in range(0, 24):
        tm = hh * 60.0
        if not (T_OPEN + 2 <= tm <= T_SHUT - 2):
            continue
        c0, r0 = tip(tm)
        if not (2 < c0 < G.cols - 2 and 2 < r0 < G.rows - 2):
            continue
        c1, r1 = tip(tm + 1.0)
        dc, dr = c1 - c0, r1 - r0
        n = float(np.hypot(dc, dr)) or 1.0
        nc, nr = -dr / n, dc / n            # normal to the curve
        _stroke(acc, c0 - 3.4 * nc, r0 - 3.4 * nr,
                c0 + 3.4 * nc, r0 + 3.4 * nr, width=0.85)
        ticks.append((hh, c0, r0))

    return acc, ticks


ENGRAVE, TICKS = build_engraving()
NOON_TIP = tip(SNOON)


# --------------------------------------------------------------------------
# text built out of cells
# --------------------------------------------------------------------------
def text_extent(s, sc=SC):
    return (len(s) * 3 + max(0, len(s) - 1)) * sc, 5 * sc


def stamp_text(mask, s, col, row, sc=SC):
    """stamp a string into a mask (1.0 where inked)"""
    x = col
    for ch in s:
        bits = FONT.get(ch, FONT[' '])
        for j in range(5):
            for i in range(3):
                if bits[j * 3 + i] == '1':
                    r0, c0 = row + j * sc, x + i * sc
                    mask[r0:r0 + sc, c0:c0 + sc] = 1.0
        x += 4 * sc


def hhmm(minutes):
    """24-hour. A 12-hour clock would print 7:19 for morning and evening
    alike, and would hide the whole point by calling solar noon 1:48."""
    m = int(round(minutes)) % 1440
    return "%d:%02d" % (m // 60, m % 60)


def clock_col(s):
    return int(round(GX - text_extent(s)[0] / 2.0))

NUM_12 = "12"
NUM_SN = hhmm(SNOON)


def _numeral_slot(tick_c, tick_r, s, dx, dy):
    w, h = text_extent(s)
    return int(round(tick_c + dx - w / 2.0)), int(round(tick_r + dy - h / 2.0))


# where the two dial numerals get engraved
_t12 = [t for t in TICKS if t[0] == 12]
TICK12 = (_t12[0][1], _t12[0][2]) if _t12 else (GX, GY)
POS_12 = _numeral_slot(TICK12[0], TICK12[1], NUM_12, 0.0, 13.0)
POS_SN = _numeral_slot(NOON_TIP[0], NOON_TIP[1], NUM_SN, 26.0, -10.0)

MASK_12 = np.zeros((G.rows, G.cols))
stamp_text(MASK_12, NUM_12, POS_12[0], POS_12[1])
MASK_SN = np.zeros((G.rows, G.cols))
stamp_text(MASK_SN, NUM_SN, POS_SN[0], POS_SN[1])


# --------------------------------------------------------------------------
# draw
# --------------------------------------------------------------------------
def draw(frame_i):
    tmin = clock(frame_i)
    d = shadow_density(tmin)

    eng = np.maximum(ENGRAVE, MASK_12)
    if tmin >= SNOON:                        # carved when the shadow arrives
        eng = np.maximum(eng, MASK_SN)

    dens = np.maximum(d, eng)
    show = dens > 0.02

    # colour key: 0 shadow ink, 1 oxide, 2 oxide in shadow, 3 clock
    use_ox = (eng > 0.15) & (eng >= d * 0.80)
    key = np.where(use_ox, np.where(d > 0.40, 2, 1), 0)

    # running clock, with a halo of paper so the shadow can pass behind it
    cmask = np.zeros((G.rows, G.cols))
    ctxt = hhmm(tmin)
    stamp_text(cmask, ctxt, clock_col(ctxt), TEXT_ROW)
    halo = np.zeros_like(cmask)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            halo = np.maximum(halo, np.roll(np.roll(cmask, dr, 0), dc, 1))
    clear = (halo > 0.5) & (cmask < 0.5)
    dens = np.where(clear, 0.0, dens)
    show = np.where(clear, False, show)
    dens = np.where(cmask > 0.5, 1.0, dens)
    show = np.where(cmask > 0.5, True, show)
    key = np.where(cmask > 0.5, 3, key)

    idx = np.clip((dens * 255.0).astype(int), 0, 255)
    cols = (INK, OXIDE, OX_DARK, INK)

    fr = Frame(G, PAPER)
    for r in range(G.rows):
        sig = np.where(show[r], idx[r] * 8 + key[r], -1)
        if not (sig >= 0).any():
            continue
        edges = np.flatnonzero(np.r_[True, sig[1:] != sig[:-1]])
        for e_i, s0 in enumerate(edges):
            if sig[s0] < 0:
                continue
            e0 = edges[e_i + 1] if e_i + 1 < len(edges) else G.cols
            fr.put_run(int(s0), r, LUT[idx[r, s0]] * int(e0 - s0),
                       cols[key[r, s0]])
    return fr


# --------------------------------------------------------------------------
# check -- the piece's claims, as numbers, before anything is rendered
# --------------------------------------------------------------------------
def check():
    print("=" * 68)
    print("%s  %s   lat %.4f  lon %.4f  tz UTC%+.0f"
          % (CITY, DATE.isoformat(), LAT, LON, TZ))
    print("=" * 68)

    # 1. solar noon, two independent ways: closed form vs brute-force maximum
    t = np.arange(0.0, 1440.0, 0.002)
    t_max = float(t[np.argmax(sun(t)[0])])
    print("solar noon, NOAA closed form : %8.3f min  = %s"
          % (SNOON, hhmm(SNOON)))
    print("solar noon, max altitude     : %8.3f min  = %s"
          % (t_max, hhmm(t_max)))
    print("                     agree to : %8.3f min" % abs(SNOON - t_max))
    assert abs(SNOON - t_max) < 0.5, "the two solar noons disagree"

    # 2. the decomposition -- the whole argument
    print()
    print("  longitude  %.4f deg west of the %.0f deg meridian : %+8.2f min"
          % (ZONE_MERIDIAN - LON, abs(ZONE_MERIDIAN), PART_LON))
    print("  daylight saving time                             : %+8.2f min"
          % PART_DST)
    print("  equation of time                                 : %+8.2f min"
          % PART_EOT)
    print("                                                     ---------")
    print("  solar noon minus clock noon                      : %+8.2f min"
          % OFFSET)
    assert abs(OFFSET - (SNOON - 720.0)) < 1e-9, "the parts do not sum"
    assert abs(PART_LON - 44.63) < 0.02
    print("  = %s, %.1f minutes after the clock says noon"
          % (hhmm(SNOON), OFFSET))

    # 3. the payoff: at solar noon the shadow points due north, and only then
    print()
    for label, tm in (("clock noon", 720.0), ("solar noon", SNOON)):
        lu, lp, bearing = shadow(tm)
        err = min(bearing, 360.0 - bearing)
        print("  %-11s bearing %8.3f deg from north (%.3f off)  umbra %6.2f"
              % (label, bearing, err, lu))
    b_noon = shadow(SNOON)[2]
    assert min(b_noon, 360.0 - b_noon) < 0.05, "shadow is not north at noon"
    b_1200 = shadow(720.0)[2]
    assert min(b_1200, 360.0 - b_1200) > 20.0, "no visible gap at clock noon"

    # 4. shortest shadow == solar noon
    ts = np.linspace(T_OPEN, T_SHUT, 40000)
    t_short = float(ts[np.argmin(shadow(ts)[0])])
    print("  shortest shadow at %s (%.3f min from solar noon)"
          % (hhmm(t_short), abs(t_short - SNOON)))
    assert abs(t_short - SNOON) < 0.5

    # 5. the payoff in cells -- can a viewer see the gap?
    print()
    c12, r12 = tip(720.0)
    csn, rsn = tip(SNOON)
    gap = float(np.hypot(csn - c12, rsn - r12))
    print("  tip at clock noon  col %6.2f row %6.2f" % (c12, r12))
    print("  tip at solar noon  col %6.2f row %6.2f" % (csn, rsn))
    print("  gap between them   %6.2f cells (%.0f%% of frame width)"
          % (gap, 100.0 * gap / G.cols))
    assert gap > 20.0, "the gap is too small to see"
    assert 2 < c12 < G.cols - 2 and 2 < r12 < G.rows - 2, "12:00 tick off frame"
    assert G.safe_top < rsn < G.safe_bot, "the vertex is outside the safe band"
    print("  shadow width %.1f cells, umbra at solar noon %.1f cells long"
          % (2 * POST_R, shadow(SNOON)[0]))
    assert 2 * POST_R >= 8.0 and shadow(SNOON)[0] >= 15.0

    # 6. the sun is not a point: how soft is the tip, and when.
    #    only claim what is on screen -- the tip leaves the frame long
    #    before sunset, and a soft band nobody can see is not delivered.
    ts = np.linspace(T_OPEN, T_SHUT, 40000)
    tc, tr = tip(ts)
    on = (tc > 1) & (tc < G.cols - 1) & (tr > 1) & (tr < G.rows - 1)
    t_last = float(ts[on][-1])
    print()
    for label, tm in (("solar noon", SNOON), ("two hours later", SNOON + 120),
                      ("tip leaves frame", t_last)):
        lu, lp, _ = shadow_raw(tm)
        print("  %-17s %-6s umbra %7.2f  penumbra %7.2f  soft %5.2f cells"
              % (label, hhmm(tm), lu, lp, lp - lu))
    soft_noon = float(shadow_raw(SNOON)[1] - shadow_raw(SNOON)[0])
    soft_last = float(shadow_raw(t_last)[1] - shadow_raw(t_last)[0])
    soft_end = float(shadow_raw(T_SHUT)[1] - shadow_raw(T_SHUT)[0])
    # The penumbra is modelled because it is real, but it is NOT a payoff
    # here: the tip is out of frame long before the softening is legible.
    # Assert that, so the description cannot quietly claim it.
    assert soft_noon < 1.0, "tip should be crisp at midday"
    assert soft_last < 1.5, ("the tip softens visibly -- if this ever fails, "
                             "the effect became claimable and should be said")
    print("  crisp for the whole time the tip is in frame (%.2f -> %.2f cells)"
          % (soft_noon, soft_last))
    print("  NOT VISIBLE HERE: by the last animated frame (%s, sun %.1f deg up)"
          % (hhmm(T_SHUT), sun(T_SHUT)[0]))
    print("    the soft band is %.0f cells and the shadow %.0f -- both off frame"
          % (soft_end, shadow_raw(T_SHUT)[0]))
    print("    if the post were 1 m tall that is a %.1f m shadow with a %.1f m"
          % (shadow_raw(T_SHUT)[0] / H_GNOMON, soft_end / H_GNOMON))
    print("    gradient at its point. do not claim it in the description.")

    # 6b. the engraved curve must actually be where the shadow's point goes,
    #     or it is decoration pretending to be a measurement
    print()
    # (a 0.75-wide stroke need not saturate a cell centre, so the test is
    #  "the line is present at the tip", not "the line is solid there")
    faintest = 1.0
    for tm in np.linspace(720.0, 960.0, 25):
        c0, r0 = tip(tm)
        if not (2 < c0 < G.cols - 2 and 2 < r0 < G.rows - 2):
            continue
        near = ENGRAVE[int(r0) - 1:int(r0) + 2, int(c0) - 1:int(c0) + 2]
        faintest = min(faintest, float(near.max()))
    print("  engraved curve vs the tip it claims to trace: faintest %.3f"
          % faintest)
    assert faintest > 0.6, "the curve is not where the shadow point goes"

    # 7. hour ticks are countable, and the ones that matter are on screen
    print()
    print("  hour ticks on the dial:")
    marks = sorted([(c, "%d:00" % h) for h, c, _ in TICKS] + [(GX, "MERIDIAN")])
    for c, lab in marks:
        print("      col %6.2f   %s" % (c, lab))
    seps = [marks[i + 1][0] - marks[i][0] for i in range(len(marks) - 1)]
    print("      closest pair: %.2f cells apart" % min(seps))
    # 14:00 lands 3.6 cells past the meridian because solar noon is only
    # 12 minutes before it. Two 0.85-wide strokes that far apart are still
    # two strokes, and the tightness is honest -- keep it, guard collision.
    assert min(seps) > 2.5, "two marks would merge into one"
    assert len(TICKS) >= 3, "too few ticks to count"
    assert any(h == 12 for h, _, _ in TICKS), "the 12:00 tick must be visible"
    assert any(h == 13 for h, _, _ in TICKS), "13:00 sits between 12 and noon"

    # 8. the day, and the window actually animated
    print()
    print("  sunrise %s   sunset %s   daylight %.2f h"
          % (hhmm(SUNRISE), hhmm(SUNSET), (SUNSET - SUNRISE) / 60.0))
    print("  animated %s -> %s  (%.2f h in %.1f s = %.0f min per second)"
          % (hhmm(T_OPEN), hhmm(T_SHUT), (T_SHUT - T_OPEN) / 60.0,
             T_END, (T_SHUT - T_OPEN) / T_END))
    assert T_OPEN > SUNRISE and T_SHUT < SUNSET
    assert abs((T_OPEN + T_SHUT) / 2.0 - SNOON) < 1e-6, "window not centred"

    # 9. safe area -- words only
    print()
    for name, (c0, r0), s in (("running clock",
                               (clock_col("00:00"), TEXT_ROW), "00:00"),
                              ("numeral 12", POS_12, NUM_12),
                              ("numeral %s" % NUM_SN, POS_SN, NUM_SN)):
        w, h = text_extent(s)
        print("  %-14s cols %3d..%3d  rows %3d..%3d   (safe rows %d..%d)"
              % (name, c0, c0 + w, r0, r0 + h, G.safe_top, G.safe_bot))
        assert G.safe_top <= r0 and r0 + h <= G.safe_bot, "%s outside band" % name
        assert 0 <= c0 and c0 + w <= G.cols, "%s off frame" % name

    # 10. ink coverage over the clip -- is there something on screen throughout
    print()
    cov = []
    for f in (0, FRAMES // 6, FRAMES // 3, FRAMES // 2,
              2 * FRAMES // 3, FRAMES - 1):
        d = shadow_density(clock(f))
        cov.append((f, hhmm(clock(f)),
                    float((np.maximum(d, ENGRAVE) > 0.02).mean())))
    for f, hm_, c in cov:
        print("  frame %3d  %-6s ink %.3f" % (f, hm_, c))
    assert min(c for _, _, c in cov) > 0.02, "a frame is nearly empty"

    print()
    print("all checks passed")
    return True


# --------------------------------------------------------------------------
def sheet(path="/tmp/sundial_sheet.png"):
    picks = [0, FRAMES // 5, 2 * FRAMES // 5,
             int(round((SNOON - T_OPEN) / (T_SHUT - T_OPEN) * (FRAMES - 1))),
             3 * FRAMES // 4, FRAMES - 1]
    frames = [draw(f) for f in picks]
    contact(frames, path, cols=3,
            labels=[hhmm(clock(f)) for f in picks])
    print("sheet -> %s" % path)


def render(path="/tmp/sundial.mp4"):
    import time
    t0 = time.time()
    with Encoder(path, G, fps=FPS) as enc:
        for f in range(FRAMES):
            enc.write(draw(f))
            if f % 60 == 0:
                print("  %3d/%d  %s  %.1fs" % (f, FRAMES, hhmm(clock(f)),
                                               time.time() - t0))
    print("render -> %s  (%d frames, %.1f s, %.1f s wall)"
          % (path, FRAMES, T_END, time.time() - t0))


if __name__ == "__main__":
    if "--check" in sys.argv or len(sys.argv) == 1:
        check()
    if "--sheet" in sys.argv:
        sheet()
    if "--render" in sys.argv:
        render()
