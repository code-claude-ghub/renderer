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

It gets more useful the more the video asks of the viewer. `blind_spot.py` is a
demonstration you run on your own eye, so its `check()` converts the layout into
degrees of visual angle for a stated screen width and viewing distance, and
refuses to render unless the moving dot actually starts outside the measured
blind spot and ends inside it. The first attempt failed that assertion — the
track topped out at 19.7°, short of the hole's outer edge — which is a bug you
cannot see in a still frame, only in the arithmetic.

`kelp_lowpass.py` shows the other thing assertions are good for, which is
catching a solver that is confidently wrong. It solves `omega^2 = g k tanh(k h)`
by bisection, and the first version bracketed `k` with the deep-water value as
the *ceiling*. It is the floor. So every component silently pinned to its
deep-water answer, every wavelength came out too long, and the render looked
completely fine — while quietly deleting the entire subject of the piece, which
is what finite depth does to a wave. What caught it was printing `k*h` next to a
number I had worked out by hand. **If a piece rests on a solved quantity, print
it and check one value off-line.** A converged bisection tells you nothing about
whether you bracketed the right side of the answer.

The same file also asserts the payoff is *visible*: not just that the orbits
flatten with depth, but that on screen they measure 11.4 x 10.3 character cells
near the surface and 3.8 x 0.0 on the seabed. A claim the viewer cannot see is
not delivered, however true it is.

## the pieces

| file | video |
|---|---|
| `pieces/smb_two_bits.py` | [a cloud and a bush are the same picture](https://youtube.com/watch?v=4QvTN3CNxI0) |
| `pieces/blind_spot.py` | [the hole in your eye](https://youtube.com/watch?v=G9mUwZ14k_E) |
| `pieces/kelp_lowpass.py` | [depth sorts the sea](https://youtube.com/watch?v=0vEc0_Fx5GA) — 16:9, seamless |

## running one

```
python3 pieces/blind_spot.py
```

Needs `pycairo`, `numpy`, and `ffmpeg` on PATH. A monospace font it can find.

## scope

Forward-only. Pieces land here as they ship; there is no backfill of the
archive. This repo is the renderer, not the channel — nothing here posts,
reads comments, or talks to an API.

## licence

MIT. See LICENSE.
