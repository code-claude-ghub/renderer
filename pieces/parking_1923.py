#!/usr/bin/env python3
"""
COLUMBUS, OHIO -- the parking one restaurant is required to build.

1923  Columbus writes the first off-street parking minimum in the country.
1954  a 2,500 sq ft restaurant must provide 9 spaces.
2022  the same restaurant must provide 34.
2024  on the new mixed-use corridors, none.

One site seen from almost overhead: a 50x50 ft building, the lot the code
demands in front of it, one person at the door. Nothing else is drawn, so
the lot's area is the whole argument. The count is PAINTED ON THE ASPHALT
rather than captioned on top of the picture -- it turns with the site.

    python3 parking_1923.py check      one still + numbers, no encode
    python3 parking_1923.py            full render
"""

import math
import os
import sys

import cairo
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asciilib import (Camera, Encoder, Frame, Grid, depth_cue, ink_lut,
                      lambert, specular, visible, zbuffer)

OUT = os.path.expanduser("~/projects/active/youtube/youtube-channel/"
                         "renders/parking_1923.mp4")

# ---------------------------------------------------------------- geometry
# World feet.  x = width.  y = depth, SMALLER y is farther (up the frame).
# z = height, toward the camera.  A pitch about x tilts the plan view.

BLD_W, BLD_D, BLD_H = 50.0, 50.0, 14.0     # 50 x 50 = exactly 2,500 sq ft
SETBACK = 12.0
LOT_X = 40.5                               # 81 ft wide = 9 stalls of 9 ft
LOT_Y0 = BLD_D + SETBACK                   # 62
STALL_D, AISLE_D, STALL_W = 18.0, 24.0, 9.0

# rows of stalls: (y_near_edge_of_row_start, y_end, n_stalls)
ROWS = [(62.0, 80.0, 9),      # row A
        (104.0, 122.0, 9),    # row B      aisle 80..104 between
        (122.0, 140.0, 8),    # row C      backs onto B
        (164.0, 182.0, 8)]    # row D      aisle 140..164 between
LOT_Y1 = 182.0                             # 81 x 120 ft = 9,720 sq ft
TOTAL_STALLS = sum(r[2] for r in ROWS)     # 34

# the painted number lives in the first aisle, present in every lot state
NUM_Y0, NUM_Y1 = 83.0, 101.0
NUM_HALF_X = 17.0

PITCH = 0.42                               # 24 deg off vertical
SPIN = 0.09
LAMP = (-0.62, 0.36, 0.70)

# ---------------------------------------------------------------- palette
# fresh colourway: chrome yellow + bone + cyan on near-black plum.
BG = (0.043, 0.038, 0.058)
ASPH = (0.62, 0.56, 0.75)
PAINT = (1.00, 0.82, 0.16)
BONE = (0.92, 0.94, 0.98)
CYAN = (0.24, 0.98, 0.92)

M_ASPH, M_PAINT, M_ROOF, M_WALL, M_PEOPLE, M_EDGE, M_GHOST = range(7)
GAIN = {M_ASPH: 0.72, M_PAINT: 1.50, M_ROOF: 1.02, M_WALL: 0.86,
        M_PEOPLE: 1.55, M_EDGE: 1.00, M_GHOST: 0.85}
TINT = {M_ASPH: ASPH, M_PAINT: PAINT, M_ROOF: BONE, M_WALL: BONE,
        M_PEOPLE: CYAN, M_EDGE: BONE, M_GHOST: PAINT}

# A 4 in stripe is a fifth of a cell at this scale, and a thing below one
# cell cannot be drawn -- the first render dashed every stall line into
# unreadable specks. Paint is drawn about 1 ft wide so it reads as a line.
STRIPE_W = 0.55


GAIN_ARR = np.array([GAIN[i] for i in range(len(GAIN))])

FPS, DUR = 30, 22.0
FRAMES = int(FPS * DUR)

G = Grid()
RAMP = ink_lut()


# ------------------------------------------------------------ text as cells
def text_cells(text, h_cells, ss=8, thresh=0.38, face="sans-serif"):
    """Rasterise a string into a boolean cell mask h_cells tall.

    One-cell-tall glyphs are unreadable on a phone -- roughly 4 px of a
    real screen.  Words have to be built OUT of cells, so they are drawn
    at ss x resolution and area-averaged down onto the character grid.
    """
    px = h_cells * ss
    probe = cairo.Context(cairo.ImageSurface(cairo.FORMAT_A8, 8, 8))
    probe.select_font_face(face, cairo.FONT_SLANT_NORMAL,
                           cairo.FONT_WEIGHT_BOLD)
    probe.set_font_size(px)
    e = probe.text_extents(text)
    if e.height <= 0:
        raise ValueError("empty text: %r" % text)
    size = px * px / e.height                      # make CAP height == px
    probe.set_font_size(size)
    e = probe.text_extents(text)

    W = int(math.ceil(e.width)) + 2 * ss
    H = int(math.ceil(e.height)) + 2 * ss
    surf = cairo.ImageSurface(cairo.FORMAT_A8, W, H)
    c = cairo.Context(surf)
    c.select_font_face(face, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    c.set_font_size(size)
    c.move_to(ss - e.x_bearing, ss - e.y_bearing)
    c.set_source_rgba(1, 1, 1, 1)
    c.show_text(text)
    surf.flush()

    stride = surf.get_stride()
    buf = np.frombuffer(surf.get_data(), np.uint8)
    buf = buf.reshape(H, stride)[:, :W].astype(np.float32) / 255.0
    hc, wc = H // ss, W // ss
    buf = buf[:hc * ss, :wc * ss].reshape(hc, ss, wc, ss).mean((1, 3))
    return buf > thresh


def stamp(fr, mask, col0, row0, rgb, alpha=1.0, halo=True):
    """Words over a lit render need a knockout or they lose.

    The site fills the frame at every moment of this piece, so there is no
    empty corner to put a label in. A one-cell halo of background painted
    behind the glyphs keeps them readable over the roof without costing a
    rectangle of picture the way a filled plate would.
    """
    if halo:
        h = np.zeros(mask.shape, bool)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                h |= np.roll(np.roll(mask, dr, 0), dc, 1)
        rr, cc = np.nonzero(h & ~mask)
        for r, c in zip(rr, cc):
            fr.put(col0 + int(c), row0 + int(r), "#", BG, alpha)
    rr, cc = np.nonzero(mask)
    for r, c in zip(rr, cc):
        fr.put(col0 + int(c), row0 + int(r), "#", rgb, alpha)


# ------------------------------------------------------------------ world
def value_noise(x, y, seed=7):
    """Smooth, cheap, and computed ONCE -- texture must not swim."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(x)
    for k, (fx, fy, a) in enumerate([(0.19, 0.23, 1.0), (0.47, 0.41, 0.5),
                                     (0.91, 1.07, 0.26)]):
        px, py = rng.uniform(0, 7, 2)
        out += a * np.sin(x * fx + px) * np.sin(y * fy + py)
    return out / 1.76


def build_ground():
    """The lot: asphalt samples plus a static paint mask for the stripes."""
    step = 0.25                                   # 4 samples per foot
    xs = np.arange(-LOT_X, LOT_X + step, step)
    ys = np.arange(LOT_Y0, LOT_Y1 + step, step)
    X, Y = np.meshgrid(xs, ys)
    rng = np.random.default_rng(11)
    X = X + rng.uniform(-0.36, 0.36, X.shape) * step
    Y = Y + rng.uniform(-0.36, 0.36, Y.shape) * step

    n = value_noise(X, Y)
    Z = 0.16 * n
    # normal from the noise gradient, so the asphalt has grain
    gx = 0.16 * np.gradient(n, axis=1) / step
    gy = 0.16 * np.gradient(n, axis=0) / step
    N = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)

    stripe = np.zeros(X.shape, bool)
    for (y0, y1, k) in ROWS:
        band = (Y >= y0) & (Y <= y1)
        half = k * STALL_W / 2.0
        for i in range(k + 1):
            stripe |= band & (np.abs(X - (-half + i * STALL_W)) < STRIPE_W)
    edge = np.abs(np.abs(X) - LOT_X) < STRIPE_W
    stripe |= edge
    stripe |= np.abs(Y - LOT_Y0) < STRIPE_W
    # the outline of the whole required lot, kept for after it drains away
    ghost = edge | (np.abs(Y - LOT_Y0) < STRIPE_W) | (np.abs(Y - LOT_Y1)
                                                      < STRIPE_W)

    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], -1)
    return P, N.reshape(-1, 3), stripe.ravel(), ghost.ravel()


def paint_number(P, text):
    """Boolean over ground points: the required count, painted big."""
    if not text:
        return np.zeros(len(P), bool)
    h_ft = NUM_Y1 - NUM_Y0
    res = 6                                        # mask samples per foot
    mask = text_cells(text, int(h_ft * res), ss=4, thresh=0.42)
    mh, mw = mask.shape
    w_ft = mw / float(res)
    if w_ft > 2 * NUM_HALF_X:                      # never wider than the lot
        w_ft = 2 * NUM_HALF_X
    x0, y0 = -w_ft / 2.0, NUM_Y0
    cx = ((P[:, 0] - x0) / w_ft * mw).astype(np.int64)
    cy = ((P[:, 1] - y0) / h_ft * mh).astype(np.int64)
    ok = (cx >= 0) & (cx < mw) & (cy >= 0) & (cy < mh)
    out = np.zeros(len(P), bool)
    out[ok] = mask[cy[ok], cx[ok]]
    return out


def slab(x0, x1, y0, y1, z0, z1, normal, step=0.25):
    ax = np.arange(x0, x1 + step, step) if x1 > x0 else np.array([x0])
    ay = np.arange(y0, y1 + step, step) if y1 > y0 else np.array([y0])
    az = np.arange(z0, z1 + step, step) if z1 > z0 else np.array([z0])
    X, Y, Z = np.meshgrid(ax, ay, az, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], -1)
    # A REGULAR lattice, spun against the character grid and rounded, beats
    # itself into a moire dot-screen: the building rendered as gauze rather
    # than a solid. Jitter once, off a fixed seed, and it fills.
    rng = np.random.default_rng(int(abs(x0 * 31 + y0 * 7 + z0 * 3)) + 5)
    j = rng.uniform(-0.38, 0.38, P.shape) * step
    j[:, [i for i, (a, b) in enumerate(((x0, x1), (y0, y1), (z0, z1)))
          if b <= a]] = 0.0
    P = P + j
    N = np.tile(np.asarray(normal, float), (len(P), 1))
    return P, N


SLOPE = 0.45                       # ridge stands 11.25 ft over the eaves


def roofline(x):
    return BLD_H + SLOPE * (BLD_W / 2.0 - np.abs(x))


def build_building():
    """A gable, not a slab.

    The first build was a flat roof, and a flat plane under a near-vertical
    lamp shades to ONE glyph across its whole face -- 40,000 samples of
    identical grey that read as window screen. A crown of 3 ft moved the
    shading by less than a single ramp step, which is the tell: if a form
    needs texture to be visible, the form is wrong. A ridge splits the roof
    into a lit face and a dark one, four ramp steps apart, and the building
    reads as a solid object from the first frame.
    """
    hw = BLD_W / 2.0
    nx = SLOPE / math.sqrt(1.0 + SLOPE * SLOPE)
    nz = 1.0 / math.sqrt(1.0 + SLOPE * SLOPE)

    parts = [slab(-hw, 0.0, 0.0, BLD_D, 0.0, 0.0, (-nx, 0, nz)),
             slab(0.0, hw, 0.0, BLD_D, 0.0, 0.0, (nx, 0, nz)),
             # nudged 0.3 ft proud of the roof edge: coplanar with it, the
             # gable end and the roof z-fought and half the peak vanished
             slab(-hw, hw, BLD_D + 0.3, BLD_D + 0.3, 0.0, roofline(0.0),
                  (0, 1, 0)),
             slab(-hw, -hw, 0.0, BLD_D, 0.0, BLD_H, (-1, 0, 0)),
             slab(hw, hw, 0.0, BLD_D, 0.0, BLD_H, (1, 0, 0))]
    P = np.concatenate([p for p, _ in parts])
    N = np.concatenate([n for _, n in parts])
    mat = np.full(len(P), M_WALL)
    nroof = len(parts[0][0]) + len(parts[1][0])
    P[:nroof, 2] = roofline(P[:nroof, 0])
    mat[:nroof] = M_ROOF
    rim = ((np.abs(P[:nroof, 0]) < 1.3)                        # ridge cap
           | (np.abs(np.abs(P[:nroof, 0]) - hw) < 1.2)          # eaves
           | (P[:nroof, 1] < 1.2) | (P[:nroof, 1] > BLD_D - 1.2))
    mat[:nroof][rim] = M_EDGE

    # the gable end is a rectangle capped by a triangle: drop what is sky
    face = np.zeros(len(P), bool)
    face[nroof:nroof + len(parts[2][0])] = True
    sky = face & (P[:, 2] > roofline(P[:, 0]))
    keep = ~sky

    # the doorway, and one person standing in it
    door = (face & (P[:, 0] > 5.6) & (P[:, 0] < 10.4) & (P[:, 2] < 8.0))
    mat[door] = M_PEOPLE
    P, N, mat = P[keep], N[keep], mat[keep]

    pp, pn = slab(7.0, 9.2, 52.4, 53.6, 0.0, 5.6, (0, 1, 0), step=0.22)
    P = np.concatenate([P, pp])
    N = np.concatenate([N, pn])
    mat = np.concatenate([mat, np.full(len(pp), M_PEOPLE)])
    return P, N, mat


GP, GN, GSTRIPE, GGHOST = build_ground()
BP, BN, BMAT = build_building()
NUMBERS = {n: paint_number(GP, str(n)) for n in (9, 18, 26, 34)}


# ------------------------------------------------------------------ motion
def smooth(a, b, t):
    t = min(max(t, 0.0), 1.0)
    return a + (b - a) * (t * t * (3 - 2 * t))


def front_y(t):
    """The near edge of the lot the code demands. Drives area and count."""
    if t < 3.0:
        return 104.0
    if t < 10.0:
        return smooth(104.0, LOT_Y1, (t - 3.0) / 7.0)
    if t < 13.5:
        return LOT_Y1
    if t < 16.0:
        return smooth(LOT_Y1, LOT_Y0, (t - 13.5) / 2.5)
    return LOT_Y0


def frame_y(t):
    """Constant. A zoom that pulls back as the lot grows makes the building
    shrink, and then the viewer is watching a camera move instead of an
    area. Hold the whole site still and let the asphalt do the arguing."""
    return LOT_Y1


def revealed(front):
    return sum(k for (y0, y1, k) in ROWS if y1 <= front + 1e-6)


def state(t):
    if t < 3.0:
        return "1954", "COLUMBUS, OHIO REQUIRES"
    if t < 13.5:
        return "2022", "COLUMBUS, OHIO REQUIRES"
    return "2024", "NEW CORRIDORS: NONE"


def _rot_z(p, n, a):
    if not a:
        return p, n
    c, s = math.cos(a), math.sin(a)
    r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return p @ r.T, n @ r.T


def _rot_x(p, n, a):
    c, s = math.cos(a), math.sin(a)
    r = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    return p @ r.T, n @ r.T


def place(p, n, t):
    """The exact transform draw() uses. fit() must see this, not the model.

    First cut fed fit() the raw world points of every pose, unclipped and
    unrotated: at the closing zoom of 2.94 that put ground 400 ft behind the
    camera's own subject inside the bounding box, and the scale collapsed to
    a third of what it should be. A camera must be fitted on what is drawn.
    """
    fy = frame_y(t)
    p = (p - np.array([0.0, fy / 2.0, 0.0])) * (LOT_Y1 / fy)
    p, n = _rot_z(np.ascontiguousarray(p), np.ascontiguousarray(n),
                  SPIN * math.sin(2 * math.pi * t / DUR))
    return _rot_x(p, n, PITCH)


def pose(t):
    keep = GP[:, 1] <= front_y(t)
    p = np.concatenate([GP[keep], BP])
    n = np.concatenate([GN[keep], BN])
    return place(p, n, t)[0]


CAM = Camera(G).fit([pose(t) for t in (0.0, 5.5, 11.0, 16.0, 21.0)],
                    margin=1.0)
CAM.scale *= 1.30          # graphics may bleed the safe band; words may not


def _recentre():
    """fit() centres on the SAFE band, which is 5 rows above the middle of
    the frame. For a figure allowed to bleed, that quietly clips the far
    edge -- the building's roofline sat at row -1. Centre on the frame."""
    r = CAM.project(pose(11.0))[1]
    CAM.off[1] += ((r.min() + r.max()) / 2.0 - G.rows / 2.0) / CAM.scale


_recentre()


def colour(s, e):
    m = int(round(e))
    base = TINT.get(m, ASPH)
    k = 0.30 + 0.80 * s
    return (min(1.0, base[0] * k), min(1.0, base[1] * k),
            min(1.0, base[2] * k))


def draw(f):
    t = f / float(FPS)
    front = front_y(t)
    paved = GP[:, 1] <= front
    if front <= LOT_Y0 + 0.6:                       # nothing is required
        paved = np.zeros(len(GP), bool)

    # after the requirement is repealed, the lot it used to demand stays as
    # an outline. the absence needs a shape or it is just black frame.
    gfade = min(1.0, max(0.0, (t - 13.5) / 2.0))
    keep_g = paved | (GGHOST if gfade > 0 else False)

    n_show = revealed(front)
    paint = GSTRIPE.copy()
    if n_show:
        paint |= NUMBERS[n_show]
    paint |= np.abs(GP[:, 1] - front) < STRIPE_W        # the moving near edge
    gmat = np.where(paint, M_PAINT, M_ASPH)
    gmat = np.where(paved, gmat, M_GHOST)

    # Paint and asphalt were coplanar, so the z-buffer chose between them
    # per cell at random and every stall line rendered as dashes. Lift the
    # paint clear of the surface it is painted on.
    GPZ = GP.copy()
    GPZ[paint | GGHOST, 2] += 0.5

    p = np.concatenate([GPZ[keep_g], BP])
    n = np.concatenate([GN[keep_g], BN])
    mat = np.concatenate([gmat[keep_g], BMAT])
    ghost_fade = np.concatenate([
        np.where(paved[keep_g], 1.0, gfade), np.ones(len(BP))])

    p, n = place(p, n, t)

    col, row, z = CAM.project(p)
    ok = visible(G, col, row)
    col, row, z, n, mat = col[ok], row[ok], z[ok], n[ok], mat[ok]
    ghost_fade = ghost_fade[ok]
    _, keep = zbuffer(G, col, row, z)

    lit = (0.26 + 0.62 * lambert(n, LAMP)
           + 0.22 * specular(n, LAMP, 26)) * depth_cue(z, 1.0, 0.84)
    gain = np.take(GAIN_ARR, mat) * ghost_fade
    shade = np.clip(lit * gain, 0.0, 1.0)
    # The ridge and eaves are a DRAWN LINE, not a lit surface. Left as a
    # gain they inherited their own face's shading, so the outline glowed
    # on the sunlit half of the roof and vanished on the other -- the
    # building read as half a building.
    shade[mat == M_EDGE] = 0.94

    fr = Frame(G, BG)
    fr.field(col, row, keep, shade, colour, RAMP, extra=mat.astype(float))
    overlay(fr, t)
    return fr


# ------------------------------------------------------------------ overlay
YEAR_H, SUB_H, CTA_H = 13, 5, 7
CTA = ["WHAT DOES YOUR", "TOWN REQUIRE?"]
_CACHE = {}


def cells(text, h):
    """Cell mask for a string, shrunk until it fits the frame width.

    A hand-picked height is a render waiting to break: one longer phrase in
    a later state and the words run off the side, or the assert fires after
    the geometry is already right.  Fit it, and floor it at 4 cells so a
    line can never shrink its way into being unreadable instead.
    """
    key = (text, h)
    if key not in _CACHE:
        for hh in range(h, 3, -1):
            m = text_cells(text, hh)
            if m.shape[1] + 8 <= G.cols:
                _CACHE[key] = m
                break
        else:
            raise AssertionError("%r cannot fit at >= 4 cells" % text)
    return _CACHE[key]


def overlay(fr, t):
    year, sub = state(t)
    top = G.safe_top + 2
    a = 1.0
    for edge in (3.0, 13.5):                       # brief flicker on a change
        if 0.0 <= t - edge < 0.30:
            a = 0.25
    stamp(fr, cells(year, YEAR_H), 4, top, BONE, a)
    stamp(fr, cells(sub, SUB_H), 4, top + YEAR_H + 3, BONE, a * 0.85)

    if t > 17.0:
        f = min(1.0, (t - 17.0) / 1.2)
        r = G.safe_bot - 2 * CTA_H - 8
        for i, line in enumerate(CTA):
            m = cells(line, CTA_H)
            stamp(fr, m, (G.cols - m.shape[1]) // 2, r + i * (CTA_H + 3),
                  PAINT, f)


# -------------------------------------------------------------------- check
def check():
    print(G)
    print("ground pts %d   building pts %d   stalls %d"
          % (len(GP), len(BP), TOTAL_STALLS))
    assert TOTAL_STALLS == 34, TOTAL_STALLS
    lot_area = (2 * LOT_X) * (LOT_Y1 - LOT_Y0)
    bld_area = BLD_W * BLD_D
    print("lot %.0f sq ft  building %.0f sq ft  ratio %.2f  per stall %.0f"
          % (lot_area, bld_area, lot_area / bld_area,
             lot_area / TOTAL_STALLS))
    assert abs(bld_area - 2500) < 1e-6
    for t in (0.0, 3.0, 6.5, 11.0, 16.0, 21.0):
        c, r, _ = CAM.project(pose(t))
        print("t=%5.1f front=%6.1f zoom=%4.2f shown=%2d %-5s "
              "cols %4d..%-4d rows %4d..%-4d"
              % (t, front_y(t), LOT_Y1 / frame_y(t), revealed(front_y(t)),
                 state(t)[0], c.min(), c.max(), r.min(), r.max()))
    assert revealed(104.0) == 9 and revealed(LOT_Y1) == 34
    assert revealed(LOT_Y0) == 0 and revealed(front_y(21.0)) == 0

    for txt, h in [(state(0)[0], YEAR_H), (state(0)[1], SUB_H),
                   (state(20)[1], SUB_H)] + [(c, CTA_H) for c in CTA]:
        m = cells(txt, h)
        print("  text %-28r %2d x %3d cells" % (txt, m.shape[0], m.shape[1]))
        assert m.shape[1] + 4 <= G.cols, (txt, m.shape)
    bot = G.safe_bot - 2 * CTA_H - 8 + (CTA_H + 3) + CTA_H
    assert bot <= G.safe_bot, bot
    assert G.safe_top + 2 + YEAR_H + 3 + SUB_H <= G.safe_bot

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for t in (0.0, 6.5, 11.0, 16.0, 21.0):
        fr = draw(int(t * FPS))
        fr.surface.write_to_png(OUT.replace(".mp4", "_t%04.1f.png" % t))
    print("stills written next to", OUT)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with Encoder(OUT, G, fps=FPS) as enc:
        for f in range(FRAMES):
            enc.write(draw(f))
            if f % 60 == 0:
                print("  frame %d/%d" % (f, FRAMES), flush=True)
    print("wrote", OUT)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check()
    else:
        main()
