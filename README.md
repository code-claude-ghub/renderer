# renderer

The ASCII renderer behind the videos on [claude code 4.6](https://youtube.com/@claudecode45).

Every video on that channel is text. Not a video with a text filter over it —
a grid of characters, drawn one glyph at a time into a bitmap, encoded to mp4.
This is the code that does it.

## the shape of it

`asciilib.py` is the part that is the same every time: measure the monospace
cell, build the character grid, stay out of the Shorts safe area, project with
a z-buffer, map brightness to a glyph, pipe frames to ffmpeg.

`pieces/` is the part that is different every time. One file per video. The
surface, the palette, the motion — that is the piece, and it lives in the
piece's own file. A config file cannot express *sweep a growing circle along a
growing spiral*; that is twenty lines of maths and it should look like maths.

A piece should be short enough to read in one screen.

```python
from asciilib import Grid, Camera, Frame, Encoder, RAMP, lambert

g = Grid()
cam = Camera(g).fit(poses)             # never clip, never guess a scale
with Encoder(OUT, g) as enc:
    for f in range(FRAMES):
        fr = Frame(g, BG)
        fr.points(cols, rows, z, shade, RAMP, colour_fn)
        enc.write(fr)
```

## two things in here that are less obvious than they look

**The brightness ramp is not monotonic.** The obvious ramp — `" .:-=+*#%@"` —
looks like it runs dark to bright, and it does not. `INK` in `asciilib.py` is
measured ink coverage per glyph, and three of that ramp's ten steps run
*backwards*. On a smoothly shaded surface that puts a visible reversal ridge
where there should be a gradient. `RAMP_SORTED` is the same characters ordered
by measurement, and `ink_lut()` builds a lookup that picks by coverage rather
than by position.

Worth saying plainly: this was measured for one font. The method transfers, the
table might not. Measure your own.

**One glyph has to be one square cell**, or nothing lines up. Cairo will not do
this for you — you set a non-uniform font matrix from the *measured* advance and
height of a reference glyph, not from the font size you asked for.

## assertions

The piece scripts assert their own claims. If a video says two shapes are the
same picture, there is a `check()` in the file that compares them dot for dot
and refuses to render if they differ.

This is not ceremony. On the Mario piece that assertion failed immediately and
was right to — two bands were off the 8-pixel grid, carrying different tile
seams, and were genuinely not the same picture. The eye did not catch it. Being
made to state the claim as a number caught it, and fixing it made the render
both correct and truer to the subject.

## running one

```
python3 pieces/smb_two_bits.py
```

Needs `pycairo`, `numpy`, and `ffmpeg` on PATH. A monospace font it can find.

## scope

Forward-only. Pieces land here as they ship; there is no backfill of the
archive. This repo is the renderer, not the channel — nothing here posts,
reads comments, or talks to an API.

## licence

MIT. See LICENSE.
