"""stars_left.py -- the whole sky above one observer, emptying.

Every mark inside the disc is ONE REAL STAR from the HYG catalogue, at its
real right ascension and declination, with its real visual magnitude. There
is no painted Milky Way and no invented band: whatever structure shows up is
the actual distribution of naked-eye stars. (The catalogue is complete to
about magnitude 6.5 -- 8,920 stars, which matches the accepted whole-sky
count -- and past magnitude 9.5 its plane-to-pole ratio drops below 1, which
is a selection effect, not the galaxy. So nothing fainter than 8.0 is used.)

The projection is Lambert azimuthal equal-area centred on the zenith, so the
disc is the entire visible hemisphere and equal areas of sky get equal areas
of screen. Zenith at the centre, horizon at the rim, north up, east left --
the sky as you would see it lying on your back.

The number above the disc is the count of stars above the horizon whose
extincted magnitude beats the limiting magnitude of the sky at their
altitude. It is computed per frame from the catalogue. It is not a graphic.

Two things drive it:

  * atmospheric extinction, 0.25 mag per airmass, so stars near the rim are
    dimmed by the air they are seen through -- true on the darkest night
    there has ever been, and the reason the disc empties from the edge in.
  * skyglow, which rises through the piece. Phase B is a traverse: a dark
    field (naked-eye limiting magnitude 6.5) to an inner-city sky (3.0).
    Phase D is time: Kyba et al. 2023 (Science 379:265) measured sky
    brightness rising 9.6% a year from 51,351 Globe at Night observations,
    2011-2022, and gave the worked example of a site with 250 visible stars
    falling to 100 in 18 years. The same proportional loss is applied here
    and the limiting magnitude is solved for, per frame, to hit it.

The counter does not stop when the video does.
"""
import math
import os
import sys
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asciilib import Encoder, Frame, Grid, ink_lut  # noqa: E402

OUT = "/tmp/stars_left.mp4"
FPS = 30

HYG_URL = ("https://raw.githubusercontent.com/astronexus/HYG-Database/"
           "main/hyg/CURRENT/hygdata_v41.csv")
HYG_CSV = "/tmp/hyg.csv"
HYG_NPZ = "/tmp/hyg_stars.npz"

LAT = 40.0            # degrees north
K_EXT = 0.25          # mag per airmass, clear V band
MAG_FLOOR = 8.0       # nothing fainter is ever visible; also where HYG stops
                      # being trustworthy

NELM_DARK = 6.5       # a dark rural sky. Conservative on purpose: the
                      # catalogue is only trustworthy this deep, so the
                      # dark-end count is an UNDERcount, never a boast.
NELM_CITY = 4.0       # Bortle class 9, inner city, published as "<= 4.0"
KYBA_RATIO = 100.0 / 250.0     # the paper's own worked example
KYBA_YEARS = 18.0

# --- the frame -------------------------------------------------------------
G = Grid()
RAMP = ink_lut()

BG = (0.031, 0.036, 0.075)        # indigo-black, the ground everything sits on
STAR = (0.902, 0.945, 0.985)      # pale blue-white
SODIUM = (0.870, 0.445, 0.130)    # the colour of a street doing this to a sky

DISC_R = 48.0
DISC_C = G.cols / 2.0 - 0.5
DISC_ROW = 104.0

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# --- timing ----------------------------------------------------------------
A_HOLD = 70           # the sky as it is
B_RISE = 280          # the traverse: dark field -> inner city
C_HOLD = 55           # the city, held
D_YEARS = 165         # and then it keeps going
FRAMES = A_HOLD + B_RISE + C_HOLD + D_YEARS

LST0 = 18.5           # hours -- puts the summer band across the dome
LST_DRIFT = 6.0       # hours of sky over the whole piece


# --- the catalogue ---------------------------------------------------------
def load_stars():
    """Real positions and magnitudes. Downloaded, never typed."""
    if os.path.exists(HYG_NPZ):
        d = np.load(HYG_NPZ)
        return d["ra"], d["dec"], d["mag"]
    if not os.path.exists(HYG_CSV):
        sys.stderr.write("fetching HYG catalogue...\n")
        urllib.request.urlretrieve(HYG_URL, HYG_CSV)
    import csv
    ra, dec, mag = [], [], []
    with open(HYG_CSV) as f:
        for r in csv.DictReader(f):
            if r["proper"] == "Sol":
                continue
            try:
                m = float(r["mag"])
                if m > MAG_FLOOR:
                    continue
                ra.append(float(r["rarad"]))
                dec.append(float(r["decrad"]))
                mag.append(m)
            except (ValueError, KeyError):
                continue
    ra = np.array(ra)
    dec = np.array(dec)
    mag = np.array(mag)
    np.savez(HYG_NPZ, ra=ra, dec=dec, mag=mag)
    return ra, dec, mag


RA, DEC, MAG = load_stars()
PHI = math.radians(LAT)
SIN_PHI, COS_PHI = math.sin(PHI), math.cos(PHI)


def sky(lst_hours):
    """RA/Dec -> altitude, airmass, and equal-area disc coordinates."""
    ha = np.radians(lst_hours * 15.0) - RA
    sin_alt = (np.sin(DEC) * SIN_PHI + np.cos(DEC) * COS_PHI * np.cos(ha))
    sin_alt = np.clip(sin_alt, -1.0, 1.0)
    alt = np.arcsin(sin_alt)
    up = alt > 0.0

    cos_alt = np.cos(alt)
    cos_az = np.clip((np.sin(DEC) - sin_alt * SIN_PHI)
                     / np.maximum(cos_alt * COS_PHI, 1e-9), -1.0, 1.0)
    az = np.arccos(cos_az)
    az = np.where(np.sin(ha) > 0.0, 2.0 * math.pi - az, az)

    # Lambert azimuthal equal-area from the zenith, normalised so the
    # horizon lands exactly on the rim.
    zen = math.pi / 2.0 - alt
    r = np.sin(zen / 2.0) / math.sin(math.pi / 4.0)
    x = -r * np.sin(az)          # east to the left: you are looking up
    y = r * np.cos(az)           # north up

    airmass = 1.0 / np.maximum(sin_alt, 1.0 / 12.0)
    return alt, up, airmass, x, y


def limiting(nelm_z, airmass, glow):
    """The faintest magnitude the sky permits, per star.

    Extinction always costs you the rim. Skyglow costs you more of it as it
    rises, because the light is scattered by the same long path.
    """
    return nelm_z - K_EXT * (airmass - 1.0) * (1.0 + 0.6 * glow)


def count_visible(nelm_z, up, airmass, glow):
    lim = limiting(nelm_z, airmass, glow)
    return int(np.count_nonzero(up & (MAG <= lim)))


def nelm_for_count(target, up, airmass, glow, lo=0.0, hi=7.0):
    """Solve for the sky brightness that leaves exactly this many stars."""
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        if count_visible(mid, up, airmass, glow) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --- words made out of cells ----------------------------------------------
def text_cells(s, max_cols, rows_target):
    """Rasterise at 8x and area-average down, with a fit loop."""
    sc = 8
    rt = rows_target
    while rt >= 4:
        f = ImageFont.truetype(FONT, int(rt * sc / 0.72))
        bb = f.getbbox(s)
        w, h = max(1, bb[2] - bb[0]), max(1, bb[3] - bb[1])
        cw, ch = int(math.ceil(w / sc)), int(math.ceil(h / sc))
        if cw <= max_cols:
            img = Image.new("L", (cw * sc, ch * sc), 0)
            ImageDraw.Draw(img).text((-bb[0], -bb[1]), s, font=f, fill=255)
            small = img.resize((cw, ch), Image.BOX)
            return np.asarray(small, np.float64) / 255.0
        rt -= 1
    return np.zeros((1, 1))


def stamp(fr, mask, row0, col0, rgb):
    """Text with a one-cell background halo, so it survives any ground."""
    ch, cw = mask.shape
    lit = mask > 0.06
    if not lit.any():
        return
    pad = np.zeros((ch + 2, cw + 2), bool)
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            pad[dr:dr + ch, dc:dc + cw] |= lit
    fr.ctx.set_source_rgb(*BG)
    for r, c in zip(*np.nonzero(pad)):
        fr.ctx.rectangle((col0 + c - 1) * G.cell, (row0 + r - 1) * G.cell,
                         G.cell, G.cell)
    fr.ctx.fill()
    for r, c in zip(*np.nonzero(lit)):
        v = min(1.0, mask[r, c] * 1.25)
        fr.put(col0 + c, row0 + r,
               RAMP[int(v * (len(RAMP) - 1))], rgb)


def centred(fr, s, row, rows_target, rgb):
    m = text_cells(s, G.cols - 4, rows_target)
    stamp(fr, m, row, int(round((G.cols - m.shape[1]) / 2.0)), rgb)


# --- the disc --------------------------------------------------------------
RR, CC = np.mgrid[0:G.rows, 0:G.cols]
DR = (RR - DISC_ROW) / DISC_R
DC = (CC - DISC_C) / DISC_R
DIST = np.sqrt(DR * DR + DC * DC)
IN_DISC = DIST <= 1.0
# radius on an equal-area disc maps straight back to altitude
DISC_ALT = np.where(IN_DISC,
                    math.pi / 2.0 - 2.0 * np.arcsin(
                        np.clip(DIST, 0, 1) * math.sin(math.pi / 4.0)),
                    0.0)
DISC_X = 1.0 / np.maximum(np.sin(np.maximum(DISC_ALT, 1e-3)), 1.0 / 12.0)

# The light does not come from everywhere. It comes from a town, and it piles
# up as a dome over the direction the town is in.
DISC_AZ = np.arctan2(-DC, -DR)
CITY_AZ = math.radians(202.0)
CITY_LOBE = np.clip(np.cos(DISC_AZ - CITY_AZ), 0.0, 1.0) ** 1.6

# A smooth radial field quantised to ten glyphs comes out as contour rings.
# Stipple it, once, off a fixed seed.
_RNG = np.random.default_rng(20260816)
DITHER = _RNG.uniform(-1.0, 1.0, (G.rows, G.cols))


def schedule(f):
    """-> (nelm_z, glow, year or None)"""
    if f < A_HOLD:
        return NELM_DARK, 0.0, None
    if f < A_HOLD + B_RISE:
        t = (f - A_HOLD) / float(B_RISE)
        t = t * t * (3.0 - 2.0 * t)                 # smoothstep
        nelm = NELM_DARK + (NELM_CITY - NELM_DARK) * t
        return nelm, min(1.0, t * 1.02), None
    if f < A_HOLD + B_RISE + C_HOLD:
        return NELM_CITY, 1.0, 2026
    t = (f - A_HOLD - B_RISE - C_HOLD) / float(D_YEARS)
    # 18 years is exactly as far as Kyba et al. run their own example.
    # Not one year further.
    return None, 1.0 + 0.24 * t, 2026 + t * KYBA_YEARS


def draw(f):
    lst = LST0 + LST_DRIFT * f / float(FRAMES)
    alt, up, airmass, x, y = sky(lst)
    nelm_z, glow, year = schedule(f)

    if nelm_z is None:                              # the future: hit the count
        yrs = year - 2026.0
        target = max(1.0, CITY_COUNT * (KYBA_RATIO ** (yrs / KYBA_YEARS)))
        nelm_z = nelm_for_count(target, up, airmass, glow)

    lim = limiting(nelm_z, airmass, glow)
    seen = up & (MAG <= lim)
    n_seen = int(np.count_nonzero(seen))

    col = np.rint(DISC_C + x * DISC_R).astype(np.int64)
    row = np.rint(DISC_ROW - y * DISC_R).astype(np.int64)   # north is UP
    ok = (seen & (col >= 0) & (col < G.cols) & (row >= 1)
          & (row < G.rows - 1))

    # brightness: how far above the limit the star is, accumulated per cell,
    # so a crowded patch of sky is genuinely brighter than an empty one
    s = np.clip((lim - MAG) / 3.0, 0.0, 1.0)
    starcell = np.zeros(G.rows * G.cols)
    np.add.at(starcell, row[ok] * G.cols + col[ok], s[ok])
    starcell = np.clip(starcell, 0.0, 1.0).reshape(G.rows, G.cols)

    # the wash: skyglow, thicker at the rim where the air is longest, and
    # thicker still on the side the town is on
    wash = np.zeros((G.rows, G.cols))
    w = np.clip(0.34 + 0.66 * (DISC_X - 1.0) / 2.4, 0.0, 1.0)
    w = glow * 0.70 * w * (0.66 + 0.52 * CITY_LOBE)
    w = w * (1.0 + 0.20 * DITHER) + 0.028 * DITHER * np.clip(w * 6.0, 0, 1)
    wash[IN_DISC] = np.clip(w, 0.0, 1.0)[IN_DISC]

    total = np.clip(starcell + wash, 0.0, 1.0)
    frac = starcell / np.maximum(starcell + wash, 1e-6)

    live = total > 0.012
    rr, cc = np.nonzero(live)
    flat = rr * G.cols + cc
    shade = total[rr, cc]
    extra = frac[rr, cc]

    fr = Frame(G, BG)
    keep = np.ones(len(flat), bool)

    def colour(sh, fx):
        base = tuple(SODIUM[i] + (STAR[i] - SODIUM[i]) * fx for i in range(3))
        k = 0.55 + 0.45 * sh
        return (base[0] * k, base[1] * k, base[2] * k)

    fr.field(cc, rr, keep, shade, colour, RAMP, extra=extra)

    tint = min(1.0, max(0.0, glow))
    num = tuple(STAR[i] + (SODIUM[i] - STAR[i]) * tint * 0.55 for i in range(3))
    centred(fr, "{:,}".format(n_seen), 18, 22, num)
    if year is not None:
        centred(fr, str(int(year)), 43, 9,
                tuple(0.62 * c for c in SODIUM))
    return fr, n_seen, nelm_z


# --- the number the future is measured down from --------------------------
_alt, _up, _am, _x, _y = sky(LST0 + LST_DRIFT * (A_HOLD + B_RISE) / FRAMES)
CITY_COUNT = count_visible(NELM_CITY, _up, _am, 1.0)


def check():
    print(G)
    print("catalogue: %d stars to mag %.1f" % (len(MAG), MAG_FLOOR))
    alt, up, am, x, y = sky(LST0)
    # Polaris must sit due north at altitude == latitude
    p = int(np.argmax(DEC))
    print("polaris  alt %.2f deg (lat %.1f)  x %.3f y %.3f"
          % (math.degrees(alt[p]), LAT, x[p], y[p]))
    assert abs(math.degrees(alt[p]) - LAT) < 1.5
    assert abs(x[p]) < 0.02 and y[p] > 0.55
    assert np.all(np.sqrt(x[up] ** 2 + y[up] ** 2) <= 1.0 + 1e-9)
    for nelm in (NELM_DARK, 6.0, 5.0, 4.0, NELM_CITY):
        print("  nelm_z %.1f -> %5d stars above the horizon"
              % (nelm, count_visible(nelm, up, am,
                                     0.0 if nelm > 6.0 else 1.0)))
    print("city count used for the future: %d" % CITY_COUNT)
    for yrs in (0, 9, 18, 30):
        t = CITY_COUNT * (KYBA_RATIO ** (yrs / KYBA_YEARS))
        print("  %d -> %.0f stars" % (2026 + yrs, t))


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
        sys.exit(0)
    if "--stills" in sys.argv:
        from asciilib import contact
        picks = [0, 120, 220, 300, 380, 405, 480, 560]
        fs, labs = [], []
        for p in picks:
            fr, n, nel = draw(p)
            fs.append(fr)
            labs.append("f%d n=%d nelm=%.2f" % (p, n, nel))
            print(labs[-1])
        contact(fs, "/tmp/stars_sheet.png", cols=4, labels=labs)
        print("/tmp/stars_sheet.png")
        sys.exit(0)
    check()
    with Encoder(OUT, G, fps=FPS) as enc:
        for f in range(FRAMES):
            fr, n, nel = draw(f)
            enc.write(fr)
            if f % 60 == 0:
                print("  f%3d  n=%5d  nelm %.2f" % (f, n, nel), flush=True)
    print("wrote", OUT, FRAMES / float(FPS), "s")
