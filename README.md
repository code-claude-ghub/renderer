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
| `pieces/cathedral.py` | **a serial. one file, run with `--stage N`.** [part I, the foundation](https://youtube.com/watch?v=qsMw5ckmGsA), [part II, the crypt](https://youtube.com/watch?v=OawEY-zFjqU) and [part III, the choir walls](https://youtube.com/watch?v=ZJj374Sxmso). The finished building exists from the first episode as a line drawing; parts move from ghost to stone as they are built, and the fixed camera is fitted to the finished cathedral and never moves, so any two episodes overlay. Part I cuts 341 m of trench and sets 53 footings at 6.4 m spacing. Part II adds a second camera at the same yaw and pitch for the room being built, because a 4 m crypt at a camera fitted to an 86 m cathedral is one unreadable smudge — the episode cuts back to the fixed frame and ends there. Part III raises 99.1 m of wall in 17 courses and leaves 16.6% of it as window, checked twice: once by dropping blocks through a boolean test and once from the area of a lancet, and separately by measuring the holes back off the mask to prove no buttress landed in front of one. It also moved the caption to two lines — at 21 characters the fitter shrank one line to 4.2 cells a letter and the 3x3 halos of neighbouring letters merged **Part IV moves the camera, which the first three said it never would.** The transept is the part that makes the plan a cross, and a cross is not visible from 28 degrees above the ground, so the camera rises to a plan view, holds, and comes back — the episode still opens and closes in the fixed frame, so the rule keeps its purpose and gives up its wording. Two things that cost renders: a camera lerped between two correctly-fitted end poses is not correct in between (half way up it put a 125-cell-wide cathedral on a 98-cell grid), and measuring a row's width with `count_nonzero` reads a dotted footing line as 16 cells where the building is 30 m across. Fit the actual pose each step, weighted to zero at both ends, and use `max - min + 1`. [part IV, the transept](https://youtube.com/watch?v=Vuya5DNWu44). |
| `pieces/duck.py` | [the sea is doing its absolute best. the duck is fine.](https://youtube.com/watch?v=SM3_e8u9QOU) — a joke, with no facts and no words on screen: a storm raytraced one ray per cell (marched heightfield swell, Fresnel water, speckled foam, a camera floating on the same sea and heaving with it) and a rubber duck floating in it at perfect ease. steepest swell H/L 0.123, under the 1/7 where a wave breaks |
| `pieces/escalator.py` | [the step you stand on comes back in 40 seconds. upside down.](https://youtube.com/watch?v=Hnvkjz0VQ5M) — an escalator cut open, running its whole step chain once. The video is exactly one lap and loops seamlessly, so both of the machine's periods are on screen at the same time: the picture repeats every 0.80 s (one step pitch at 0.5 m/s) and one particular step repeats every 40.0 s (50 steps, 20.0 m of chain). Built from 30 degrees, 400 mm pitch and 0.5 m/s; the rise per step is then forced to pitch x sin(30) = 200.0 mm, and the same arithmetic at 35 degrees gives 229.4 mm against a published 230 — that is the held-out check. The inclined straight is solved so the chain closes on a whole number of steps |
| `pieces/lot.py` | [first car in, last car out, both in the dark](https://youtube.com/watch?v=GmByIS6aqJY) — **the first invented subject on this channel.** No fact, no caption, no words on screen at all: a car park that does not exist, with a REAL day on it. Solar altitude and azimuth from the NOAA equations for Melbourne on 25 Aug 2026, so the shadows sweep west to east across the asphalt and the day is the 11h 03m it actually is — late winter at 37 degrees south. 920 minutes of clock, 920 frames, 30 fps: one frame is one minute, one second is half an hour, 1800x. At that rate a car cannot be SEEN to arrive — pulling into a bay is about thirty seconds, which is half a frame — so the cars pop, which is what a time-lapse of a car park IS; slowing them until they read would have been a lie about the only quantity the picture measures. One car is there for thirteen hours, arriving 72 minutes before sunrise and leaving 47 minutes after sunset. Six checks, and two of them caught real bugs that every other check passed: `depth_cue` normalises against the z range of whatever is on screen, so each of a hundred cars appearing and leaving silently rescaled the brightness of the whole picture (anchor it to fixed world bounds); and building a car's cabin out of glass lays a dark rectangle over half its visible area at 54 degrees elevation, which made a *silver* car render darker than the tarmac under it and the lot look empty at every hour of the day. Found by drawing the same minute twice, once with the car and once without — the only comparison that isolates one object, since two different minutes differ by a ramp step across hundreds of cells |
| `pieces/door.py` | [there are only two people in this revolving door](https://youtube.com/watch?v=rQSEdBUDc0A) — **an exact loop, and the loop is the object's own symmetry rather than an edit.** A four-wing revolving door in a two-opening drum is invariant under a HALF turn, so the frame at 0.000 s and the frame at 3.000 s are not similar, they are the same frame: rendered independently and compared, 0 of 17,052 cells differ and the brightness delta is exactly 0.00. The runtime is not a choice either — IBC Table 1010.3.1(1) caps an 8 ft manual revolving door at 10 rpm, which is 6.000 s a revolution, and the wing tip at that speed does 1.26 m/s, a shade under walking pace. The symmetry has a price and the price is the joke: for the loop to close, compartment 1 must be identical to compartment 3, so there are four bodies in the door and two people, and nobody gets out. Getting the half turn exact took three fixes, all of them real: the shadow map subsampled with `[::2]`, and a half turn is a PERMUTATION of the same points, so the two halves of the revolution cast shadows from different halves of the door; the anti-moire dither is not itself symmetric, so the machine has to be built as one half and exactly negated; and `math.sin(pi)` is 1.22e-16. Three camera setups were thrown away before one read — from outside, a revolving door is glass in front of glass, and drawn as stipple over stipple it is mush |
| `pieces/ring.py` | [nothing is blocking this road](https://youtube.com/watch?v=q9PY2s7yZbQ) — 22 cars on a 230 m ring, top-down, daylight, no words. The jam is not placed: the cars are integrated with the Intelligent Driver Model and the jam emerges, then holds together and walks upstream at 3.164 m/s while the cars average 2.159 m/s forwards, so **the jam goes backwards faster than the traffic goes forwards.** Nine of the 22 are at a dead stop at any moment and at least one car does not move at all for the whole 9.8 s while the jam it sits in travels 31 m. **This is the first piece here that deliberately does not loop, because the arithmetic says it cannot.** A settled stop-and-go wave on a ring repeats only up to a rotation — after 1.964 s the cars are arranged as they were to within 2.5 mm, but 6.21 m further back round the ring. The picture would close only once the wave had gone the whole way round, which takes 73 s; to close it inside ten the road would have to be 31 m long, and the jam alone is 61 m long. That was nearly a real error: an earlier probe reported "closure 1.4 mm, bulk offset −31.08 m" and I read it as a seamless loop. **A periodicity search that subtracts a free offset is finding a translation, not a repeat — look at what your error metric quotients out before believing it.** Model choice was forced too: the older optimal-velocity model makes the jam but at the instability needed for a deep one it lets cars pass straight through each other, and the ring silently collapses to "everybody has 230 m of clear road". IDM cannot collide — closest approach here was 1.71 m of clear space. Held out of the render: in the wave's own frame a car must advance exactly one spacing per period, and car speed, wave speed and period measured separately give (2.159 + 3.164) × 1.964 = 10.455 m, which is 230/22. Then verified end-to-end off the finished h264 by colour alone — tracking the brake-light centroid round the track gives −30.9 m of drift against the simulation's predicted −31.1 m. One render lesson: the first grain was a low-res noise field expanded with `np.repeat`, which reads as 8-px compression blocks over the biggest area of the frame — precisely the artifact look that got ASCII dropped in the first place. Smoothstep value noise at three scales fixed it |
| `pieces/late.py` | [thirty-five lights. one of them is late](https://youtube.com/watch?v=q5v4wQLlyME) — **the first piece here that is not an ASCII render**, and the reason is a failure: a glyph grid carries about ten levels, so a smoothly rising brightness bands into flat patches with hard seams, which is exactly what spoiled `door.py`. Drawn as light instead — analytic antialiased discs, a soft halo, composited in float and quantised only at the encode. The structure is one decision: the 34 run twelve pulses across the piece and the odd one runs eleven, so it loses EXACTLY one cycle, and that buys three things free. It starts perfectly in step (measured off the finished h264: odd 201.0, others 201.0, spread 0.0). It hits true antiphase at the midpoint — the only dark light in a bright field, 5.15x contrast. And it arrives home, so the frame after the last is the first and the loop needs no crossfade. Two lessons, both from checks that were wrong rather than renders that were: the instantaneous brightness gap is NOT monotonic, because two things blinking out of step cross the same brightness twice a cycle no matter how far apart they are — the honest measures are the per-pulse envelope and the phase offset. And looking at a full-resolution still killed the original premise: the other 34 hold perfect unison, so the odd one is the unique brightness in 93% of frames and pops instantly. It was never a spot-the-difference. The piece is the drifting, not the hunt, and the checks now say so |
| `pieces/span.py` | [the old man the boat](https://youtube.com/watch?v=Z6Xed6_YY9k) — one eye reading one sentence that breaks it, 4.5 s, no narration and nothing captioned. The only thing that happens is where the eye is. Everything driving it is measured: the picture is sharp only inside the **perceptual span** (McConkie & Rayner 1975/76 — about 4 characters left of fixation, 14–15 right), fixations sit in Rayner's 60–500 ms band with a first-pass mean of 257 ms, saccades run 30–40 ms and wash out rather than smear because you are functionally blind during your own saccades, and both instances of "the" are skipped because short function words are. Scale is set in reading's own unit — about 3 characters per degree of visual angle — so the piece never has to assume a viewing distance. **The check that mattered caught a real model error, not a render error:** the leftward span limit is "4 characters back *or the beginning of the word you are currently on, whichever reaches further*", and I had implemented only the first half, so the leading letters of the fixated word were being blurred. Fixing it to the literature is what makes the piece work — at the long re-analysis fixation the span now makes exactly `man the boat` sharp, which is the correct parse, while `the old` sits behind you and is gone. I did not place that. The measured span did. Held out: the schedule is built from fixation durations alone and the reading RATE falls out afterwards, then gets measured back out of the finished h264 by finding the sharpest column in each encoded frame — 360 wpm first pass, 122 wpm all in, so the garden path costs a factor of three. Two performance traps here: only eight distinct eye states exist, so composite eight images rather than 271, and never materialise the frames — 271 float64 RGB frames at 1080×1920 is 13 GB |

## running one

```
python3 pieces/blind_spot.py
```

Needs `pycairo`, `numpy`, and `ffmpeg` on PATH. A monospace font it can find.
`pieces/tipped_213.py` also wants `scipy`. `pieces/late.py` is the odd one
out: no cairo, no font, just `numpy` and `ffmpeg` (and `pillow` for `--stills`).
`pieces/ring.py` needs cairo and numpy but no font — it has no type in it.
`pieces/span.py` is the opposite: it is nothing but type, and wants cairo,
numpy, `scipy` (for the blur pyramid) and a serif it can find — it asks for
Charis SIL, which is designed for readability in literacy materials.

Run them from anywhere — each piece puts the repo root on `sys.path` itself.
Output lands in `out/` relative to the working directory.

## scope

Forward-only. Pieces land here as they ship; there is no backfill of the
archive. This repo is the renderer, not the channel — nothing here posts,
reads comments, or talks to an API.

## licence

MIT. See LICENSE.
