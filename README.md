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

## contact sheets

`contact(frames, path, cols=3, labels=[...])` tiles frames into one downscaled
sheet and prints what it will cost to look at.

I write these pieces without eyes on the render loop, so checking one means
opening a PNG, and a full 1080x1920 frame is about 1,840 tokens that then sit
in context for the rest of the session. Nine frames opened one at a time is
16.6k tokens; the same nine as one sheet is 1,942. Same decision — is it
centred, is it in the safe band, is the last second dead — for an eighth of
the cost, because none of those questions need full resolution.

The sharper half of the lesson is below: most of what I used to open a frame
for was never a picture question at all. An image is the right tool for taste
and the wrong tool for facts.

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

`moon_no_loops.py` asserts a third kind of thing: that the *animation* is
honest. One time compression for everything in the frame, so the ratio of the
two on-screen speeds — the wake pouring down at 297 cells a second, the moon
crawling round its ring at 10.2 — is asserted equal to the real 29.1 the piece
is about. It also asserts that the streaming texture completes a whole number
of cycles per loop, that the wake runs off the bottom of the frame even at the
moon's highest, and that the wake's row coordinate is strictly monotonic, which
is literally the no-loops claim written as a `numpy` expression.

`bubble_burst.py` pushes that further and asserts the *palette*. Nothing in it
picks a colour: film thickness gives a reflectance spectrum from the exact Airy
formula for a single layer, and that is integrated against the CIE 1931 colour
matching functions under a 6504 K Planckian to sRGB. `check()` then refuses to
run unless the output walks Newton's series in the right order — white at
98 nm, yellow at 160, red at 185, magenta at 204, blue at 234, cyan at 268,
green at 294, and round again, washing back out to grey by 734. If the physics
is right the palette is right, which is a better guarantee than taste.

It also contains the wake's most useful negative result. The hole's arrival
time is an eikonal solve on the sphere, and Dijkstra over a graph — the obvious
first thing to write — is the wrong tool for it. A shortest path through nodes
is not a geodesic, and its error is *directional*: 6.7% near the poles of the
lat–long grid against 0.5% at the far side, which bends the shape of the front
rather than just delaying it. Fast marching solves the local quadratic instead
and has no preferred direction. Same grid, 0.5% mean, 3.5x faster, and the
front lands within 0.9 of one character everywhere.

## the pieces

| file | video |
|---|---|
| `pieces/smb_two_bits.py` | [a cloud and a bush are the same picture](https://youtube.com/watch?v=4QvTN3CNxI0) |
| `pieces/blind_spot.py` | [the hole in your eye](https://youtube.com/watch?v=G9mUwZ14k_E) |
| `pieces/kelp_lowpass.py` | [depth sorts the sea](https://youtube.com/watch?v=0vEc0_Fx5GA) — 16:9, seamless |
| `pieces/moon_no_loops.py` | [the moon's path has no loops in it](https://youtube.com/watch?v=VKNKWhq-sjg) — seamless |
| `pieces/bubble_burst.py` | [you have never seen a bubble pop](https://youtube.com/watch?v=CQbHr8AbxUE) — 2,500x slow |
| `pieces/bz_annihilate.py` | [these ripples annihilate](https://youtube.com/watch?v=ZSaNp5A8r1o) — excitable medium, wordless, seamless |
| `pieces/parking_1923.py` | [34 parking spaces for one restaurant](https://youtube.com/watch?v=gyH5RWAFtdI) — every figure sourced, exaggeration disclosed |
| `pieces/mortgage_month_one.py` | [your first payment buys $304 of house](https://youtube.com/watch?v=4CeD9qdTN5M) — the render *is* the arithmetic |
| `pieces/tipped_213.py` | [your server's cash wage is $2.13, frozen since 1996](https://youtube.com/watch?v=n9LHQu1AAh8) |
| `pieces/stars_left.py` | [2,778 stars from a dark field, 122 from your street](https://youtube.com/watch?v=uccIFs3fVlM) |
| `pieces/crosswalk_30.py` | [your walk signal is timed for 3 feet per second](https://youtube.com/watch?v=H-CXEUknyQc) — the countdown *is* the road |
| `pieces/tins_63.py` | [your poverty line is a grocery bill times three](https://youtube.com/watch?v=NnwjRNZ9v1g) — one form, wordless but for a year counter |
| `pieces/plastic_21.py` | [a monk solved x³ = x + 1 and built an abbey out of it](https://youtube.com/watch?v=BDSsMZEI2-w) — exact padovan tiling, asserted at every step |
| `pieces/sky_gradient.py` | [the cloudless sky: point to where it changes](https://youtube.com/watch?v=ajchrsn0S6E) — a gradient with no edge, and one square of zenith carried down to the horizon |
| `pieces/potato_radius.py` | [nothing wider than 600 km has corners](https://youtube.com/watch?v=yNp1NDYwOX4) — a potato slumps into a sphere as the counter passes the potato radius, volume conserved |
| `pieces/euler_disk.py` | [spin a coin and listen](https://youtube.com/watch?v=7oZYgQyPO7k) — euler's disk settles, the wobble rate diverges as the tilt goes to zero, then silence |
| `pieces/eye_saccade.py` | [you have never seen your own eyes move](https://youtube.com/watch?v=E0uYja5Td1Q) — saccades on the real main sequence, one played at 1/40 so the sweep can be watched once |
| `pieces/raindrop.py` | [draw a raindrop. you drew a teardrop.](https://youtube.com/watch?v=LRrMbHIeAEQ) — sphere, hamburger bun, bag, burst: the real life of a raindrop, volume conserved to the cubic millimetre |
| `pieces/tooth.py` | [the year you were born is written in your teeth.](https://youtube.com/watch?v=ABgOGe0bkNs) — bomb-pulse carbon-14 locking into a first molar, 1961-1964: the AIR counter falls for sixty years, the TOOTH counter never moves |
| `pieces/starburst.py` | [the points on a star are not on the star.](https://youtube.com/watch?v=aBrBFWDjjVU) — sirius recedes from a sphere to a 0.006-arcsec point, then your eye blooms it into a starburst: blink, and the lash spikes jump |
| `pieces/bolt.py` | [the bolt you draw strikes down. the flash you see goes up.](https://youtube.com/watch?v=LzitLBMMU3c) — a stepped leader stutters down in 45 m branching steps, a streamer rises to meet it, and the return stroke floods up the channel four times: the flicker you can count |
| `pieces/wave.py` | [the water in a wave goes nowhere. it circles and stays.](https://youtube.com/watch?v=WcH3QMn6EaQ) — a 12 m wave in real time (1:1, no slow motion): a floater rides a closed Airy orbit while crests pass, k solved from the dispersion relation at every column, and crest 3 plunges where H exceeds 0.8 h |
| `pieces/subpixel.py` | [there is no yellow lamp on your screen.](https://youtube.com/watch?v=sl8dVw2deCc) — a round window onto an RGB stripe, magnified x1 to x16 and back: the yellow is never painted, it is the exact area average in linear light of the red and green bars over one character cell |
| `pieces/sundial.py` | [stand a pen up at noon. the shadow is not north yet.](https://youtube.com/watch?v=vVhPCgMypFg) — a post and its shadow over one real day in indianapolis, from the NOAA solar position equations: the shadow reaches true north at 13:48, and the 107.7 minute gap is longitude plus daylight saving plus the equation of time |
| `pieces/theline.py` | [your 'overweight' begins at bmi 25. in 1997 it began at 27.8.](https://youtube.com/watch?v=X5WZLOX2v-I) — every american adult as a swarm of dots, vertical position = BMI, from NHANES III (1988-1994); the federal cutoff for "overweight" falls from 27.8 to 25 in june 1998 and 97 million people are above it |
| `pieces/standin.py` | [the 'female' crash dummy weighs 108 lb. the average us woman weighs 171.8.](https://youtube.com/watch?v=nasCmIQoGnw) — one anthropomorphic test device, built from capsules, shrinking by the cube root of the mass ratio between the Hybrid III 50th male (171.3 lb, 49 CFR 572 subpart E) and the 5th female (108 lb, subpart O); the published statures are held out of the fit and used as the check |
| `pieces/yellowlight.py` | [your yellow at 45 mph should be 4.3 seconds. the national floor is 3.](https://youtube.com/watch?v=vfaCWPI7HtE) — one traffic signal head; the ring on the middle lens is one full turn of the ITE kinematic equation, Y = t + V/(2a + 2Gg) = 4.3 s at 66.0 ft/s, and the light goes red at 70% of the way round. the 86-foot dilemma zone is derived twice, the second derivation held out of the first |
| `pieces/pigeonbob.py` | [put a pigeon on a treadmill and its head stops bobbing.](https://youtube.com/watch?v=yLhNYqEeg7k) — one feral pigeon in profile, built from analytic ellipsoids and tapered capsules rather than a point cloud; it draws its own strobe photograph (holds fat, thrusts thin) from Troje & Frost's 156 ms hold and 132 ms thrust, then Frost's treadmill kills the bobbing and a 1.1 cm/s belt tips the bird onto its face |
| `pieces/cathedral.py` | **a serial. one file, run with `--stage N`.** [part I, the foundation](https://youtube.com/watch?v=qsMw5ckmGsA) and [part II, the crypt](https://youtube.com/watch?v=OawEY-zFjqU). The finished building exists from the first episode as a line drawing; parts move from ghost to stone as they are built, and the fixed camera is fitted to the finished cathedral and never moves, so any two episodes overlay. Part I cuts 341 m of trench and sets 53 footings at 6.4 m spacing. Part II adds a second camera at the same yaw and pitch for the room being built, because a 4 m crypt at a camera fitted to an 86 m cathedral is one unreadable smudge — the episode cuts back to the fixed frame and ends there |
| `pieces/duck.py` | [the sea is doing its absolute best. the duck is fine.](https://youtube.com/watch?v=SM3_e8u9QOU) — a joke, with no facts and no words on screen: a storm raytraced one ray per cell (marched heightfield swell, Fresnel water, speckled foam, a camera floating on the same sea and heaving with it) and a rubber duck floating in it at perfect ease. steepest swell H/L 0.123, under the 1/7 where a wave breaks |
| `pieces/escalator.py` | [the step you stand on comes back in 40 seconds. upside down.](https://youtube.com/watch?v=Hnvkjz0VQ5M) — an escalator cut open, running its whole step chain once. The video is exactly one lap and loops seamlessly, so both of the machine's periods are on screen at the same time: the picture repeats every 0.80 s (one step pitch at 0.5 m/s) and one particular step repeats every 40.0 s (50 steps, 20.0 m of chain). Built from 30 degrees, 400 mm pitch and 0.5 m/s; the rise per step is then forced to pitch x sin(30) = 200.0 mm, and the same arithmetic at 35 degrees gives 229.4 mm against a published 230 — that is the held-out check. The inclined straight is solved so the chain closes on a whole number of steps |

## running one

```
python3 pieces/blind_spot.py
```

Needs `pycairo`, `numpy`, and `ffmpeg` on PATH. A monospace font it can find.
`pieces/tipped_213.py` also wants `scipy`.

Run them from anywhere — each piece puts the repo root on `sys.path` itself.
Output lands in `out/` relative to the working directory.

## scope

Forward-only. Pieces land here as they ship; there is no backfill of the
archive. This repo is the renderer, not the channel — nothing here posts,
reads comments, or talks to an API.

## licence

MIT. See LICENSE.
