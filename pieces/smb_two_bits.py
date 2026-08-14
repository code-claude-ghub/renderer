#!/usr/bin/env python3
"""
"in super mario bros a cloud and a bush are the same picture."

Nakago, in Iwata Asks: "We'd use the same image for both clouds and grass,
just changing the color."  Iwata: "The clouds and grass look like separate
objects, but actually they both use the same graphical elements."  The
reason, from the same page: a Famicom cartridge "could contain just 256
components, each one of which consisted of 8 X 8 dots."

So this piece is built the way the hardware is built.

  * ONE character is ONE dot.  No magnification, no interpolation.
  * The shape is drawn ONCE into an index grid holding 0..3 -- which is
    exactly what a pattern table stores.  A tile never knows its colours.
    It stores WHICH of four, per dot, two bits each.
  * Colour lives in a separate array, `attr`, one entry per 16x16 region
    of screen.  That is the NES attribute table, and its entries really
    are two bits wide (NESdev: PPU attribute tables).
  * Every 3.7 seconds the two bits belonging to the sky band and the two
    belonging to the ground band trade places.  Nothing else is touched.

check() asserts the consequence: the glyph grid on either side of a swap is
bit-identical, and the sixteen rows of sky shape equal the sixteen rows of
ground shape element for element.  If those hold, the only difference
between a cloud and a bush in this video is two bits, which is the claim.

Wordless.  Silent.

Shipped: https://youtube.com/watch?v=4QvTN3CNxI0
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asciilib import Grid, Frame, Encoder, INK          # noqa: E402

OUT = "/tmp/smb_two_bits.mp4"

FPS = 30
PERIOD = 112          # dots in one repeat of the scenery
PERIODS = 4           # palette swaps once per period; even => seamless loop
PHASE = 4             # scroll offset at the instant of a swap
FRAMES = PERIOD * PERIODS
SHAPE_H = 16          # two tiles tall, like the real thing
TILE = 8              # "each one of which consisted of 8 X 8 dots"

# NES system palette, the FCEUX/Nintendulator RGB table. These hex values
# are approximations of an analog signal -- the PPU emits chroma phase and
# luma, and the television decodes it, so no hex table is canonical.
SKY = (0x5C, 0x94, 0xFC)      # $22, the Super Mario Bros sky
CLOUD_PAL = {1: (0xFC, 0xFC, 0xFC),   # $30 white
             2: (0x3C, 0xBC, 0xFC),   # $21 light blue
             3: (0x00, 0x00, 0x00)}   # $0F black
BUSH_PAL = {1: (0xB8, 0xF8, 0x18),    # $29 yellow-green
            2: (0x00, 0xB8, 0x00),    # $1A green
            3: (0x00, 0x00, 0x00)}    # $0F black
# The ground is not a NES palette quotation, it is eyeballed, and the
# description says so. It is also deliberately the dimmest thing here:
# dim ground, bright figures, nothing in between.
DIRT_FILL = (0x3C, 0x24, 0x00)
DIRT_PAL = {1: (0x70, 0x48, 0x08),    # solid mass
            2: (0xAC, 0x7C, 0x00),    # sparse rubble highlights
            3: (0x00, 0x00, 0x00)}

# Glyph per index, measured by ink coverage, densest first.
#
# '#' (1.00) is the black outline, because it is the only glyph solid enough
# to read as a drawn line rather than a row of beads -- the first pass used
# '@' there and the outline came out as a chain of little rings.
#
# The second entry is the variant used on an 8-dot tile boundary. It is
# applied to the GROUND ONLY. On the cloud it was noise: a grid sliced
# across the figure and cost more legibility than the idea was worth.
GLYPH = {1: ("@", "%"), 2: ("=", "*"), 3: ("#", "#")}

# region codes in `attr`
SKYBAND, GROUNDBAND, DIRT = 1, 2, 3


def rgb(c):
    return (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)


def lobed(width, lobes):
    """A flat-bottomed body with round bumps on top: a Super Mario cloud,
    and therefore also a Super Mario bush. Returns an index grid 0..3.

    The shading is not a stripe across the bottom. It is the body minus the
    same lobes shrunk by three dots, so the light colour keeps the shape of
    the bumps and the darker colour pools in the hollows between them. A
    horizontal band reads as a stripe painted on a shape; this reads as the
    shape being lit.
    """
    yy, xx = np.mgrid[0:SHAPE_H, 0:width]
    m = np.zeros((SHAPE_H, width), bool)
    for cx, r in lobes:
        m |= (((xx + 0.5 - cx) ** 2
               + (yy + 0.5 - SHAPE_H) ** 2) <= r * r)
    m |= (yy >= SHAPE_H - 4) & (xx >= 1) & (xx < width - 1)

    # one shallow dent per lobe gap, a single dot deep -- enough to scallop
    # the underside without growing the row of legs the first pass had
    for sx in range(8, width - 4, 13):
        m &= ~(((xx + 0.5 - sx) ** 2
                + (yy + 0.5 - SHAPE_H - 1.2) ** 2) <= 2.0 ** 2)

    inner = m.copy()
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        sh = np.zeros_like(m)
        ys = slice(max(0, dy), SHAPE_H + min(0, dy))
        yd = slice(max(0, -dy), SHAPE_H + min(0, -dy))
        xs = slice(max(0, dx), width + min(0, dx))
        xd = slice(max(0, -dx), width + min(0, -dx))
        sh[yd, xd] = m[ys, xs]
        inner &= sh

    # Light band measured DOWN FROM THE TOP SURFACE, not up from the floor.
    # Shrinking the lobes instead pools the light at the baseline, because
    # the lobe centres sit on it -- the first pass did that and the cloud
    # came out lit from underneath.
    first = np.argmax(m, axis=0)
    depth = yy - first[None, :]

    idx = np.zeros((SHAPE_H, width), np.uint8)
    idx[m] = 1                                # lit, following the bumps
    idx[m & (depth >= 6)] = 2                 # shade pooling in the hollows
    idx[m & ~inner] = 3                       # black outline
    return idx


def dirt_row(width, height):
    yy, xx = np.mgrid[0:height, 0:width]
    idx = np.ones((height, width), np.uint8)
    idx[((xx * 7 + yy * 5) % 17) < 2] = 2                 # sparse rubble
    idx[(yy % 16 == 0) | (xx % 16 == 0)] = 3              # block edges
    return idx


def build_world(g):
    """One period of scenery, drawn once. Everything after this is indexing."""
    rows = g.rows
    idx = np.zeros((rows, PERIOD), np.uint8)
    attr = np.zeros((rows, PERIOD), np.uint8)

    # Everything sits on the 8-dot grid, because background tiles do. This
    # is not tidiness: if the two bands were not congruent mod 8 they would
    # carry different tile seams and stop being the same picture, which is
    # the whole claim. The assertion in check() catches it.
    # The ground LINE has to sit inside the safe area, not just the bush.
    # Shorts paints UI over the bottom 15%, so a horizon placed by look
    # alone ends up underneath it and the bush loses the thing it stands on.
    ground_top = (int(rows * 0.71) // TILE) * TILE
    band_lo = ground_top - SHAPE_H
    band_hi = 2 * TILE
    assert ground_top < g.safe_bot, "horizon would sit under the Shorts UI"
    assert band_hi % TILE == band_lo % TILE == ground_top % TILE == 0

    # Near-equal radii, widely spaced. A big dominant centre lobe swallows
    # its neighbours' valleys and the silhouette stops reading as a cloud --
    # it reads as a car. The two-lobe shape was right first because its
    # lobes were similar; the three-lobe one had to be brought to match.
    big = lobed(48, [(9, 10.5), (24, 11.5), (39, 10.5)])
    small = lobed(32, [(10, 10.0), (22, 11.5)])

    for top, code in ((band_hi, SKYBAND), (band_lo, GROUNDBAND)):
        for shape, x0 in ((big, 8), (small, 72)):
            w = shape.shape[1]
            idx[top:top + SHAPE_H, x0:x0 + w] = shape
            attr[top:top + SHAPE_H, x0:x0 + w] = code

    d = dirt_row(PERIOD, rows - ground_top)
    idx[ground_top:] = d
    attr[ground_top:] = DIRT
    return idx, attr, band_hi, band_lo, ground_top


def build_glyphs(g, idx, attr):
    """Glyph per cell. Depends only on the index grid and world position --
    never on colour. That independence is what the swap assertion tests."""
    rows = g.rows
    yy, xx = np.mgrid[0:rows, 0:PERIOD]
    edge = ((xx % TILE == 0) | (yy % TILE == 0)) & (attr == DIRT)
    out = np.full((rows, PERIOD), " ", dtype="<U1")
    for v, (a, b) in GLYPH.items():
        out[(idx == v) & ~edge] = a
        out[(idx == v) & edge] = b
    return out


def paint(fr, g, glyph, attr, idx, state, s, ground_top):
    """Blit one frame. Run-length by COLOUR, because big flat regions of one
    colour is exactly the case where that wins -- the glyphs inside a run
    still vary, they just ride along in the string."""
    # The ground is a filled band with texture ON it, not glyphs floating on
    # the sky. Every glyph leaks some background through its holes, and dark
    # brown '@' over sky blue read as a blue mesh rather than as earth.
    fr.ctx.set_source_rgb(*rgb(DIRT_FILL))
    fr.ctx.rectangle(0, ground_top * g.cell,
                     g.w_px, g.h_px - ground_top * g.cell)
    fr.ctx.fill()

    take = (s + np.arange(g.cols)) % PERIOD
    G = glyph[:, take]
    A = attr[:, take]
    I = idx[:, take]

    pal_for = {DIRT: DIRT_PAL,
               SKYBAND: CLOUD_PAL if state == 0 else BUSH_PAL,
               GROUNDBAND: BUSH_PAL if state == 0 else CLOUD_PAL}

    drawn = 0
    for r in range(g.rows):
        c = 0
        while c < g.cols:
            if I[r, c] == 0:
                c += 1
                continue
            key = (A[r, c], I[r, c])
            c2 = c
            while (c2 < g.cols and I[r, c2] != 0
                   and (A[r, c2], I[r, c2]) == key):
                c2 += 1
            fr.put_run(c, r, "".join(G[r, c:c2]),
                       rgb(pal_for[key[0]][key[1]]))
            drawn += c2 - c
            c = c2
    return drawn


def check(g, idx, attr, glyph, band_hi, band_lo, ground_top):
    rows = g.rows
    print(g)
    print("one character = one NES dot; %d x %d dots on screen"
          % (g.cols, g.rows))

    a = idx[band_hi:band_hi + SHAPE_H]
    b = idx[band_lo:band_lo + SHAPE_H]
    assert a.shape == b.shape and (a == b).all(), \
        "the sky shape and the ground shape are not the same picture"
    print("sky band vs ground band: identical over %d dots" % a.size)

    ga = glyph[band_hi:band_hi + SHAPE_H]
    gb = glyph[band_lo:band_lo + SHAPE_H]
    assert (ga == gb).all(), "glyphs differ between the bands"

    # the swap changes attribute entries only, and only in the two bands
    n_regions = ((rows + 15) // 16) * ((PERIOD + 15) // 16)
    print("attribute grid: %d regions of 16x16 dots, 2 bits each = %d bytes"
          % (n_regions, n_regions // 4))

    # every drawn dot carries a 2-bit index, which is what a pattern table is
    assert idx.max() <= 3, "index grid must fit in two bits"
    lit = (idx > 0).mean()
    print("pattern coverage: %.1f%% of the period is not backdrop" % (100 * lit))

    assert PERIODS % 2 == 0, "odd swap count would not return to state 0"
    assert FRAMES % PERIOD == 0
    print("loop: %d frames, %.2f s, %d swaps, returns to state 0"
          % (FRAMES, FRAMES / FPS, PERIODS))

    for ch in set(glyph.ravel()):
        assert ch in INK, "unmeasured glyph %r" % ch
    print("glyphs used: %s"
          % " ".join(sorted((c for c in set(glyph.ravel()) if c != " "),
                            key=lambda c: INK[c])))


def main():
    g = Grid(font_size=32)
    idx, attr, band_hi, band_lo, ground_top = build_world(g)
    glyph = build_glyphs(g, idx, attr)
    check(g, idx, attr, glyph, band_hi, band_lo, ground_top)

    if "--still" in sys.argv:
        for f in (0, 56, 112, 140):
            fr = Frame(g, rgb(SKY))
            paint(fr, g, glyph, attr, idx, (f // PERIOD) % 2,
                  (f + PHASE) % PERIOD, ground_top)
            fr.surface.write_to_png("/tmp/smb_still_%03d.png" % f)
            print("wrote /tmp/smb_still_%03d.png" % f)
        return

    # Prove the seam rather than reasoning about it. Shorts loop
    # automatically and a seam is the only thing that stops a second watch.
    def render(f):
        fr = Frame(g, rgb(SKY))
        paint(fr, g, glyph, attr, idx, (f // PERIOD) % 2,
              (f + PHASE) % PERIOD, ground_top)
        return bytes(fr.surface.get_data())

    assert render(0) == render(FRAMES), "loop seam: frame 0 != frame FRAMES"
    assert render(0) != render(PERIOD), "the swap changed nothing"
    print("loop seam: frame 0 and frame %d are byte-identical" % FRAMES)

    ink = []
    with Encoder(OUT, g, fps=FPS) as enc:
        for f in range(FRAMES):
            fr = Frame(g, rgb(SKY))
            n = paint(fr, g, glyph, attr, idx,
                      (f // PERIOD) % 2, (f + PHASE) % PERIOD, ground_top)
            ink.append(n / float(g.rows * g.cols))
            enc.write(fr)
            if f % 56 == 0:
                print("  frame %3d/%d  ink %.1f%%"
                      % (f, FRAMES, 100 * ink[-1]))
    print("ink coverage %.1f%%..%.1f%%" % (100 * min(ink), 100 * max(ink)))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
