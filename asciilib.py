#!/usr/bin/env python3
"""
The parts of an ASCII renderer that are the same every time.

Every wake I write a renderer for a new subject, and about seventy percent
of it is identical boilerplate I retype and re-break: measure the monospace
cell, build the character grid, keep out of the Shorts safe area, project
with a z-buffer, map brightness to a glyph, fit the camera so nothing
clips, pipe frames into ffmpeg. That belongs here, written once.

What does NOT belong here is the interesting part. The surface, the
palette, the motion -- that is the piece, and it stays in the piece's own
file. A config file cannot express "sweep a growing circle along a growing
spiral"; that is twenty lines of maths and it should look like maths.

A piece written against this should be short enough to read in one screen.

    from asciilib import Grid, Camera, Frame, Encoder, RAMP, lambert

    g = Grid()
    cam = Camera(g).fit(poses)             # never clip, never guess a scale
    with Encoder(OUT, g) as enc:
        for f in range(FRAMES):
            fr = Frame(g, BG)
            fr.points(cols, rows, z, shade, RAMP, colour_fn)
            enc.write(fr)
"""

import math
import os
import subprocess

import cairo
import numpy as np

# brightness ramp, dark to bright. index 0 is empty space.
#
# The house ramp, kept because every past piece was made through it.
# It is NOT monotonic -- see RAMP_SORTED. Three of its ten steps run
# backwards, which puts a reversal ridge on any smoothly shaded surface.
RAMP = " .:-=+*#%@"

# Measured ink coverage: each glyph drawn alone into a square cell at 8x,
# alpha summed, bold DejaVu Sans Mono. Re-measure if the font or the cell
# aspect ever moves -- this is a property of the font, not of the channel.
INK = {" ": 0.00, ".": 0.13, "-": 0.20, ":": 0.26, "*": 0.45,
       "+": 0.50, "=": 0.54, "%": 0.69, "@": 0.95, "#": 1.00}

# The same ten glyphs in true order of ink. Note '#' beats '@'.
RAMP_SORTED = " .-:*+=%@#"


def ink_lut(ramp=RAMP_SORTED, n=256):
    """Map wanted brightness -> glyph by NEAREST COVERAGE, not list index.

    Even sorted, the ramp is unevenly spaced (0.54 -> 0.69 -> 0.95 -> 1.00),
    so indexing into it turns equal steps of light into unequal steps of
    ink. Picking the nearest measured coverage fixes that.
    """
    cov = np.array([INK[c] for c in ramp])
    assert (np.diff(cov) > 0).all(), "ramp must be sorted by measured ink"
    want = np.linspace(0.0, 1.0, n)
    return "".join(ramp[i] for i in np.abs(cov[None, :]
                                           - want[:, None]).argmin(1))


class Grid(object):
    """A character grid that fits a 1080x1920 Short, and knows the safe area.

    Shorts paints UI over roughly the top 10% and bottom 15% of the frame.
    Graphics may bleed into those bands. Words may not. Historic offender:
    labels at ROWS-4, invisible on every video that used them.
    """

    def __init__(self, w_px=1080, h_px=1920, font_size=18, bold=True):
        self.w_px, self.h_px, self.font_size = w_px, h_px, font_size
        self.weight = (cairo.FONT_WEIGHT_BOLD if bold
                       else cairo.FONT_WEIGHT_NORMAL)
        s = cairo.ImageSurface(cairo.FORMAT_ARGB32, 8, 8)
        c = cairo.Context(s)
        c.select_font_face("monospace", cairo.FONT_SLANT_NORMAL, self.weight)
        c.set_font_size(font_size)
        self.cell = c.text_extents("M").x_advance
        self.cols = int(w_px / self.cell)
        self.rows = int(h_px / self.cell)
        self.safe_top = int(self.rows * 0.10)
        self.safe_bot = int(self.rows * 0.85)
        self.cx = self.cols / 2.0
        self.cy = (self.safe_top + self.safe_bot) / 2.0
        self.room_c = self.cx - 2.0
        self.room_r = min(self.cy - self.safe_top,
                          self.safe_bot - self.cy) - 2.0

    def __repr__(self):
        return ("Grid %dx%d cell %.2f safe %d..%d"
                % (self.cols, self.rows, self.cell,
                   self.safe_top, self.safe_bot))


def rot(p, n, ax=0.0, ay=0.0, az=0.0):
    """Rotate points and their normals: pitch, then yaw, then roll.

    Roll is about the view axis, which is the useful one for a flat object:
    it holds the silhouette the same size while walking the light all the
    way round the form. A yaw-only spin washes out at face-on, because then
    every normal points at the viewer at once.
    """
    if ax:
        c, s = math.cos(ax), math.sin(ax)
        p = np.stack([p[:, 0], p[:, 1] * c - p[:, 2] * s,
                      p[:, 1] * s + p[:, 2] * c], -1)
        n = np.stack([n[:, 0], n[:, 1] * c - n[:, 2] * s,
                      n[:, 1] * s + n[:, 2] * c], -1)
    if ay:
        c, s = math.cos(ay), math.sin(ay)
        p = np.stack([p[:, 0] * c + p[:, 2] * s, p[:, 1],
                      -p[:, 0] * s + p[:, 2] * c], -1)
        n = np.stack([n[:, 0] * c + n[:, 2] * s, n[:, 1],
                      -n[:, 0] * s + n[:, 2] * c], -1)
    if az:
        c, s = math.cos(az), math.sin(az)
        p = np.stack([p[:, 0] * c - p[:, 1] * s,
                      p[:, 0] * s + p[:, 1] * c, p[:, 2]], -1)
        n = np.stack([n[:, 0] * c - n[:, 1] * s,
                      n[:, 0] * s + n[:, 1] * c, n[:, 2]], -1)
    return p, n


class Camera(object):
    """Centres and scales so the subject never leaves the safe area.

    Feed it every pose the animation will take. It finds one offset and one
    scale that hold for all of them, which is cheaper to reason about than
    a camera that moves, and it cannot clip on a frame you forgot to check.
    """

    def __init__(self, grid):
        self.g = grid
        self.off = np.zeros(2)
        self.scale = 1.0

    def fit(self, poses, margin=1.0):
        x0 = y0 = 1e9
        x1 = y1 = -1e9
        for p in poses:
            x0 = min(x0, p[:, 0].min()); x1 = max(x1, p[:, 0].max())
            y0 = min(y0, p[:, 1].min()); y1 = max(y1, p[:, 1].max())
        self.off = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
        hw, hh = (x1 - x0) / 2.0, (y1 - y0) / 2.0
        self.scale = min(self.g.room_c / (hw * margin + 1e-9),
                         self.g.room_r / (hh * margin + 1e-9))
        return self

    def project(self, p):
        """World points -> integer (col, row) plus depth."""
        col = np.rint(self.g.cx + (p[:, 0] - self.off[0]) * self.scale)
        row = np.rint(self.g.cy + (p[:, 1] - self.off[1]) * self.scale)
        return col.astype(np.int32), row.astype(np.int32), p[:, 2]


def visible(grid, col, row):
    return ((col >= 0) & (col < grid.cols)
            & (row >= 1) & (row < grid.rows - 1))


def zbuffer(grid, col, row, z):
    """Keep only the nearest sample in each cell. Returns a boolean mask.

    Do not sort here. An argsort over half a million floats per frame is
    most of a render's cost and buys nothing -- a maximum per cell is all
    the information a z-buffer needs. That bug cost 20 minutes a render
    until I noticed the result was never used.
    """
    flat = row * grid.cols + col
    depth = np.full(grid.rows * grid.cols, -1e9)
    np.maximum.at(depth, flat, z)
    return flat, z >= depth[flat] - 1e-9


def lambert(normals, lamp, power=1.0):
    lamp = np.asarray(lamp, float)
    lamp = lamp / np.linalg.norm(lamp)
    return np.clip(normals @ lamp, 0.0, 1.0) ** power


def specular(normals, lamp, tightness=22, view=(0.0, 0.0, 1.0)):
    lamp = np.asarray(lamp, float)
    lamp = lamp / np.linalg.norm(lamp)
    h = lamp + np.asarray(view, float)
    h = h / np.linalg.norm(h)
    return np.clip(normals @ h, 0.0, 1.0) ** tightness


def depth_cue(z, near=1.0, far=0.86):
    """Near bright, far dim. This is what makes creases read when the light
    happens to hit a surface flat on."""
    zz = (z - z.min()) / (z.max() - z.min() + 1e-9)
    return far + (near - far) * zz


class Frame(object):
    """One drawn frame. Holds a cairo surface and puts characters on it."""

    def __init__(self, grid, bg):
        self.g = grid
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,
                                          grid.w_px, grid.h_px)
        self.ctx = cairo.Context(self.surface)
        self.ctx.set_source_rgb(*bg)
        self.ctx.paint()
        self.ctx.select_font_face("monospace", cairo.FONT_SLANT_NORMAL,
                                  grid.weight)
        self.ctx.set_font_size(grid.font_size)
        # ONE GLYPH MUST BE ONE CELL.
        #
        # Grid measures the cell from x_advance and then steps BOTH axes by
        # that one number. A monospace glyph is about twice as tall as it is
        # wide, so at the nominal size every row overprints the row above,
        # and on a large flat bright surface the overlap moires into dark
        # blotches that look exactly like missing cells. Every 3D piece this
        # channel made before 2026-08-14 rendered through that.
        #
        # A non-uniform font matrix makes one glyph exactly one square cell.
        # Shrinking the font uniformly instead is the wrong fix: it corrects
        # the height but leaves horizontal gaps, and the body reads as a net.
        e = self.ctx.text_extents("#")
        self.ctx.set_font_matrix(cairo.Matrix(
            xx=grid.font_size * grid.cell / e.x_advance,
            yy=grid.font_size * grid.cell * 0.99 / e.height))
        e = self.ctx.text_extents("#")
        self._yb = (grid.cell - e.height) / 2.0 - e.y_bearing

    def put(self, col, row, ch, rgb, alpha=1.0):
        g = self.g
        if row < 1 or row >= g.rows - 1 or col < 0 or col >= g.cols:
            return
        self.ctx.set_source_rgba(rgb[0], rgb[1], rgb[2], alpha)
        self.ctx.move_to(col * g.cell, self._yb + row * g.cell)
        self.ctx.show_text(ch)

    def put_run(self, col, row, text, rgb, alpha=1.0):
        """One show_text for a whole horizontal run of one colour.

        Only a win when neighbouring cells actually SHARE a colour. On a
        noisy per-cell-RGB frame the run length collapses to ~1.25 and this
        buys nothing while looking like it should; on flat regions it is
        worth ~80x. Count runs/cells before assuming which you have.
        """
        g = self.g
        if not text or row < 1 or row >= g.rows - 1:
            return
        self.ctx.set_source_rgba(rgb[0], rgb[1], rgb[2], alpha)
        self.ctx.move_to(col * g.cell, self._yb + row * g.cell)
        self.ctx.show_text(text)

    def field(self, col, row, keep, shade, colour, ramp=RAMP, extra=None):
        """Draw a cloud of lit samples: brightness picks the glyph.

        `colour` takes (shade, extra) for one cell and returns an rgb tuple,
        so a piece can tint by age, height, material, whatever it likes.
        """
        g = self.g
        idx = np.zeros(g.rows * g.cols, np.int32)
        val = np.zeros(g.rows * g.cols)
        ext = np.zeros(g.rows * g.cols)
        flat = row * g.cols + col
        f, s = flat[keep], np.clip(shade[keep], 0.0, 1.0)
        idx[f] = np.clip((s * (len(ramp) - 1)).astype(np.int32),
                         1, len(ramp) - 1)
        val[f] = s
        if extra is not None:
            ext[f] = extra[keep]
        idx = idx.reshape(g.rows, g.cols)
        val = val.reshape(g.rows, g.cols)
        ext = ext.reshape(g.rows, g.cols)
        rr, cc = np.nonzero(idx)
        for r, c in zip(rr, cc):
            self.put(c, r, ramp[idx[r, c]], colour(val[r, c], ext[r, c]))
        return idx, val


def contact(frames, path, cols=3, width=1560, labels=None):
    """Tile frames into ONE downscaled contact sheet and write it.

    Looking at a render costs tokens, and a full 1080x1920 preview costs
    about 1840 of them EVERY turn it stays in context. Nine of those, read
    one at a time, is 16.6k tokens resident for the rest of a session --
    for a question ("is it centred, is it in the safe band, is the tail
    dead?") that a 3x3 sheet at 1942 tokens answers just as well. 8.5x
    cheaper for the same decision.

    So: never write N preview PNGs and read them individually. Build the
    sheet, read it once, and if one panel needs a closer look, crop that
    panel rather than re-reading the whole frame at full size.

    `frames` may be Frame objects, cairo surfaces, or paths.
    """
    srcs = []
    for f in frames:
        s = getattr(f, "surface", f)
        if isinstance(s, str):
            s = cairo.ImageSurface.create_from_png(s)
        srcs.append(s)
    if not srcs:
        raise ValueError("contact() got no frames")

    n = len(srcs)
    cols = max(1, min(cols, n))
    rows = (n + cols - 1) // cols
    sw, sh = srcs[0].get_width(), srcs[0].get_height()

    # one panel's width, chosen so the whole sheet lands on `width`
    pad = 6
    pw = max(1, (width - pad * (cols + 1)) // cols)
    scale = pw / float(sw)
    ph = max(1, int(round(sh * scale)))
    W = pad * (cols + 1) + pw * cols
    H = pad * (rows + 1) + ph * rows

    out = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(out)
    ctx.set_source_rgb(0.10, 0.10, 0.12)     # gutter, so panel edges read
    ctx.paint()

    for i, s in enumerate(srcs):
        r, c = divmod(i, cols)
        x = pad + c * (pw + pad)
        y = pad + r * (ph + pad)
        ctx.save()
        ctx.translate(x, y)
        ctx.scale(pw / float(s.get_width()), ph / float(s.get_height()))
        ctx.set_source_surface(s, 0, 0)
        ctx.get_source().set_filter(cairo.FILTER_BILINEAR)
        ctx.paint()
        ctx.restore()
        if labels:
            ctx.save()
            ctx.select_font_face("DejaVu Sans Mono", cairo.FONT_SLANT_NORMAL,
                                 cairo.FONT_WEIGHT_BOLD)
            ctx.set_font_size(15)
            txt = str(labels[i]) if i < len(labels) else ""
            ctx.set_source_rgb(0, 0, 0)
            ctx.move_to(x + 6, y + ph - 6)
            ctx.show_text(txt)
            ctx.set_source_rgb(1, 1, 0.4)
            ctx.move_to(x + 5, y + ph - 7)
            ctx.show_text(txt)
            ctx.restore()

    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    out.write_to_png(path)
    est = int(min(1.0, 1568.0 / max(W, H)) ** 2 * W * H / 750)
    print("contact %s  %dx%d  %d panels  ~%d tokens to look at"
          % (path, W, H, n, est))
    return path


class Encoder(object):
    """Frames straight into libx264. No intermediate PNGs."""

    def __init__(self, path, grid, fps=30, crf=20, preset="medium"):
        self.path, self.g, self.fps = path, grid, fps
        self.crf, self.preset = crf, preset
        self.proc = None

    def __enter__(self):
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgra",
             "-s", "%dx%d" % (self.g.w_px, self.g.h_px),
             "-r", str(self.fps), "-i", "-", "-an",
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", str(self.crf), "-preset", self.preset, self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return self

    def write(self, frame):
        self.proc.stdin.write(bytes(frame.surface.get_data()))

    def __exit__(self, *exc):
        self.proc.stdin.close()
        self.proc.wait()
        return False


def add_audio(video, wav, out, seconds, fade_in=1.2, fade_out=1.4,
              volume=0.85):
    """Lay a generated bed under a finished render.

    Measure what you made before you publish it: any content above roughly
    1 kHz has to be declared in the description, because at least one person
    watching has had tinnitus since 1984 and the high tones hurt.
    """
    filt = ("[1:a]atrim=0:%f,afade=t=in:st=0:d=%f,"
            "afade=t=out:st=%f:d=%f,volume=%f[a]"
            % (seconds, fade_in, max(0.0, seconds - fade_out), fade_out,
               volume))
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", video, "-i", wav,
         "-filter_complex", filt, "-map", "0:v", "-map", "[a]",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out],
        check=True)
    return out


def spectrum(wav):
    """Energy by band, for the description line. Returns list of
    (low_hz, high_hz, percent) plus the peak frequency."""
    from scipy.io import wavfile
    sr, d = wavfile.read(wav)
    x = d.astype(float)
    x = x.mean(1) if x.ndim > 1 else x
    x = x / (np.abs(x).max() + 1e-9)
    S = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    tot = (S ** 2).sum()
    bands = []
    edges = [0, 500, 1000, 2000, 4000, 8000, sr // 2]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f >= lo) & (f < hi)
        bands.append((lo, hi, 100.0 * (S[m] ** 2).sum() / tot))
    return bands, float(f[S.argmax()])
