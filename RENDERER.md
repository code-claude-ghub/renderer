# `asciilib` — a toolbox

**Demoted 2026-08-24.** This file used to read as the definition of the work.
It is a set of tools and a list of mistakes that have each cost a render. It
does not say what to make, and the shapes it happens to make easy are not
therefore the shapes to make.

**Read this instead of opening a past renderer.** Reading an old piece's file
to learn the calling idiom costs ~300 lines of geometry specific to a tablet
or a nail and teaches you nothing this page doesn't. Everything below is the
whole interface plus the traps that have actually cost renders.

`asciilib.py` holds what is the same every time. The surface, the palette and
the motion are the piece and belong in the piece's own file.

---

## The shape of a piece

```python
from asciilib import (Camera, Encoder, Frame, Grid, depth_cue, ink_lut,
                      lambert, specular, visible, zbuffer)

G = Grid()                                  # 98 x 174 cells at 1080x1920
RAMP = ink_lut()                            # 256 steps, smooth
BG = (r, g, b)

PTS, NRM = build()                          # your geometry, once
CAM = Camera(G).fit([pose_a, pose_b], margin=1.02)   # EVERY pose

def draw(f):
    col, row, z = CAM.project(pts)
    ok = visible(G, col, row);  col, row, z, n = col[ok], row[ok], z[ok], n[ok]
    _, keep = zbuffer(G, col, row, z)
    shade = (0.1 + 0.8 * lambert(n, lamp) + 0.4 * specular(n, lamp, 30)) \
            * depth_cue(z)
    fr = Frame(G, BG)
    fr.field(col, row, keep, shade, colour, RAMP, extra=None)
    return fr

with Encoder(OUT, G, fps=30) as enc:
    for f in range(FRAMES):
        enc.write(draw(f))
```

`colour(shade, extra) -> (r, g, b)` is called per cell, so a piece can tint by
height, age, material, whatever it likes.

## The rest of the interface

| call | does |
|---|---|
| `Grid(w_px, h_px, font_size, bold)` | grid + `.cols .rows .cell .cx .cy .safe_top .safe_bot .room_c .room_r` |
| `rot(p, n, ax, ay, az)` | pitch, then yaw, then roll — rotates points AND normals |
| `Camera(g).fit(poses, margin)` | one offset + scale that holds for all poses |
| `.project(p)` | world → `(col, row, z)`, ints |
| `visible(g, col, row)` | in-frame mask |
| `zbuffer(g, col, row, z)` | `(flat, keep)` — nearest sample per cell |
| `lambert(n, lamp, power)` / `specular(n, lamp, tightness)` | shading |
| `depth_cue(z, near, far)` | near bright, far dim — makes creases read |
| `Frame.put / put_run / field` | one glyph / a text run / a lit cloud |
| `Encoder(path, g, fps, crf, preset)` | context manager, frames straight to x264 |
| `add_audio(video, wav, out, seconds, ...)` | lay a bed under a finished render |
| `spectrum(wav)` | energy by band + peak, for the description line |

**Ramps.** `RAMP = " .:-=+*#%@"` is the house ramp and is **not monotonic** —
three of its ten steps run backwards, which puts a reversal ridge on any
smoothly shaded surface. `RAMP_SORTED` is the same glyphs in true ink order.
`ink_lut()` maps brightness to glyph by *nearest measured coverage* and is
what you want for smooth metal, skin, cloth, anything continuous.

---

## Traps that have actually cost renders

1. **Negative y is screen-UP.** Gravity is a *positive* y deflection. The nail
   piece bent upward on its first render.
2. **Fit the camera over every pose the animation will take**, or it clips on
   the one frame you didn't check.
3. **Safe area: words may not sit in the top 10% or bottom 15%.** Shorts paints
   UI there. Graphics may bleed; text may not. Historic offender: labels at
   `rows-4`, invisible on every video that used them.
4. **One glyph must be one cell.** `Frame` fixes this with a non-uniform font
   matrix. Do *not* "fix" overlap by shrinking the font uniformly — that
   corrects the height and leaves horizontal gaps, and the body reads as a net.
   Every 3D piece before 2026-08-14 rendered through that bug.
5. **Don't argsort in the z-buffer.** A per-cell maximum is all it needs; the
   sort was most of a render's cost and bought nothing.
6. **The coverage check is shape-dependent.** The asteroid rule — per row, min
   col to max col, every gap is a hole — is only valid for a *convex-ish*
   silhouette. On a bent, thin or branching form one row legitimately crosses
   two parts of the object with real background between, and the honest gap
   scores as a fault. It failed at 5% on a perfectly good frame. For those
   shapes measure **interior pinholes** instead: gaps of 1–3 cells between
   filled cells in a row. **If a check fails and raising the sample count does
   not move it, suspect the check.**

7. **`fit()` must see exactly what `draw()` draws** — clipped and rotated. Fed
   the raw model at every pose, its bounding box contained ground no frame
   ever showed and the scale collapsed to a third. And if one fitted scale
   cannot serve your widest pose *and* your closest, **the zoom is usually
   the thing to delete**: a fixed camera keeps the subject one size, which is
   what makes a size comparison an argument rather than a camera move.
8. **Coplanar surfaces z-fight per cell.** Paint sampled on the same plane as
   the asphalt under it made the z-buffer choose at random, and every line
   rendered as dashes. Lift a marking clear of the surface it marks. Same fix
   for a wall meeting a roof edge: nudge one 0.3 ft proud.
9. **A flat plane cannot be shaded.** Under a near-vertical lamp it returns
   one glyph across its whole face and reads as gauze. **If a form needs
   texture to become visible, the form is wrong** — a 3 ft crown moved the
   shading less than one ramp step; a ridge split it four steps and the
   building read instantly. Related: an outline that must survive on *both*
   the lit and unlit face is a **drawn line with a fixed shade**, not a
   material gain, or it glows on one half and vanishes on the other.
10. **Jitter every sampled surface, not just the one that showed the bug.** A
    regular lattice spun against the character grid and rounded beats itself
    into a moire dot screen. Once, off a fixed seed.
11. **On-screen words must be built OUT of cells.** A one-cell glyph is ~4 px
    on a phone. Rasterise the string at 8x, area-average onto the cell grid,
    stamp it. Give it a **fit loop** that shrinks until it fits `G.cols`, or a
    longer phrase in a later state runs off the frame. Over a full frame,
    add a one-cell halo of background behind the glyphs — cheaper than a
    filled plate and it survives any background. **The halo must be a SOLID
    glyph in the halo colour.** `put(c, r, " ", BG)` calls `show_text(" ")`,
    which paints nothing, so a halo written that way has been a silent no-op
    in every piece that used it (found 2026-08-23, the traffic signal). Once
    it actually masks, it also becomes an instrument: a halo in the *lens*
    colour turns a number into a hole punched in a burning lamp.
    And **name the mask something other than `halo`** if `halo` is also the
    colour parameter — the shadowing surfaces as
    "only length-1 arrays can be converted to Python scalars" out of
    `set_source_rgba`, which points at the colour and not at the collision.

19. **Yaw moves the face off the centre column.** A lens sitting 7.9 in proud
    of a body yawed 23° projects eight cells to the right of `C_MID`, so text
    centred on the frame's middle lands half on the housing. **Ask the
    projection where the thing is** — project the one point you care about
    and centre on that — rather than assuming the object's face is centred
    just because the object is.

20. **Collapse an overlay to CELLS before animating it.** A 52k-sample ring
    redrawn per frame is 13M Python iterations across a short piece; the ring
    only ever covers ~430 cells. Dedupe once, keep a representative value per
    cell, and fold the 0/1 seam before taking a median or the cell that
    straddles it averages to the middle and lights up in the wrong half.

12. **Do not darken twice.** `field()` already encodes brightness as ink, so a
    `colour()` that returns `base * shade` darkens the same light twice and
    every hue collapses toward the background — burnt orange rendered as
    dried blood. Tint with a floor (`base * (0.46 + 0.54*shade)`) and let the
    glyph carry the light.
13. **`depth_cue` fades the bottom of a tall object.** Under an elevated
    camera, low world-z maps to *far* camera-z, so the base of a tall form
    dims into sparse speckle and reads as fringe rather than as an edge. On
    anything much taller than it is deep, use `far` ~0.94, not the 0.86
    default.
14. **A calendar you cannot count is texture.** Bands three cells apart read
    as wood grain, not as units — space them so they can be counted (one mark
    per five, not per one). And size a groove as a *fraction of its own band*,
    never a fixed width, or the bands that shrink below a cell get eaten and
    the edge frays.
15. **On a light ground, invert the ink AND compress it.** Ink on paper means
    the darkest place gets the densest glyph, so pass `1-light` as the field
    value. But then the lit face of a large object returns almost no ink and
    the subject dissolves into the paper. Give it a floor:
    `dens = 0.46 + 0.54*(1-light)` inks the whole form and leaves the paper
    clean. Tint the same way — blend from BG toward the ink colour by density,
    never multiply.
16. **Raised type does not read on a light ground. Punch it in.** A legend
    modelled as relief catches the lamp and comes out paler than the metal
    around it — a ghost. Make it *incuse*: negative displacement, plus a
    cheap cavity term (`light *= 1 - 0.5*mask`) so the sunken letters sit in
    their own shadow. Same geometry, opposite sign, and the number goes from
    unreadable to the hardest thing on screen.
17. **A two-stop colour lerp passes through grey.** Copper→verdigris across
    thirty years desaturated to mud at the midpoint, which is exactly where
    half the frames live. Put a waypoint on the path (bronze-olive at t=0.5)
    and interpolate in two segments.
18. **A smooth analytic field contours into rings.** Trap 10 is a regular
    *lattice* beating against the cell grid. This is the opposite cause and
    the same look: a perfectly smooth radial gradient, quantised to ten glyph
    steps, draws topographic contour lines across the form — the skyglow disc
    came out as a bullseye. Stipple it, once, off a fixed seed; `w * (1 +
    0.20*noise)` was enough. And if the field is *physically* asymmetric, say
    so in the geometry — real skyglow domes over the town it comes from
    rather than ringing the horizon evenly, and building that in hid the rest
    of the banding for free. **An honest asymmetry is cheaper than a
    dithering trick and it is also true.**

21. **A 2D piece does not need a point cloud.** Analytic ellipsoids and
    tapered capsules evaluated straight on the 98x174 cell grid give real
    normals (`n = (qx/a^2, qy/b^2, z/c^2)` for an ellipsoid; `(e/r, z/r)` for
    a swept sphere) and a whole articulated animal costs about 8 ms a frame.
    Z-order by `max(z + bias)` per cell. Keep both the rotated and the
    UNROTATED query coordinates: when a body pitches over its feet, the legs
    must be drawn in the unrotated frame from a hip that has moved, or the
    feet come off the floor.

22. **A parameter that is only meaningful in one act must be GATED to that
    act.** A "how far is the head out past the toes" term was computed every
    frame; during free walking the head oscillates about its rest position by
    design, so the term went positive twice a second and tipped the bird 18
    degrees every step. It passed `check()` — the bounding boxes were fine
    and the pose was merely wrong. It was caught by tiling six frames out of
    the FINISHED ENCODE, which is the only reason that habit exists. Then it
    got an assertion: max tip before the crawl == 0.

23. **A trace lane at the height of the thing being traced gets covered by
    it.** The eye's mark pile is directly under the eye, so the head hid the
    single most important mark in the piece. Put the lane clear of the
    subject's silhouette at every pose and run a plumb line down to the
    point, which also says whose line it is.

24. **`depth_cue` renormalises per frame, so anything appearing or leaving
    rescales the WHOLE picture.** It divides by the z range of whatever it
    is handed. In a scene where objects come and go — fifty cars arriving
    and fifty leaving — every single one of those hundred events silently
    changed the brightness of every other cell in the frame. Anchor the cue
    to fixed world bounds computed once over all geometry. Caught by
    diffing the two frames either side of one car leaving and finding 89%
    of the change was nowhere near the car.

25. **Do not compare two frames at different times to isolate one object.**
    One ramp step is a large pixel change, so a 4% shift in ambient flips a
    few hundred cells across a glyph boundary and swamps the thing you were
    measuring. Draw the SAME frame twice, once with the object and once
    without. Same light, one variable.

26. **From above, a car is mostly roof.** Building a vehicle's cabin out of
    glass put a dark rectangle over more than half its visible area at 54°
    elevation, and a *silver* car rendered darker than the asphalt under it
    — car-to-surround luminance ran 0.78–1.03 across the entire day, so the
    lot looked empty at every hour and every check still passed. The
    general form: **a material assigned to the largest visible face decides
    what the object reads as**, regardless of what the object is made of.
    And glossy things need a specular term matte things do not get, or
    painted metal returns the same ink as tarmac.

27. **If a piece claims two frames are IDENTICAL, audit every place array
    ORDER leaks into the picture.** A half turn of a symmetric object is a
    permutation of the same point set, so anything order-dependent silently
    produces a different result from the same geometry. Two did. The shadow
    map subsampled its casters with `[::2]`, which takes every second point
    *in array order* — so the two halves of the revolution cast their shadows
    from different halves of the door. And `field()` writes cells in array
    order, so when two samples land in one cell at exactly equal z the winner
    is whichever sits earlier in the array. Fix the second with a canonical
    tie-break (`lexsort` on flat, z, shade, material — pick the winner by
    VALUE, never by position). The symptom was a "max brightness delta" of
    0.44, which is not a rounding error and should have said so immediately.

28. **A random dither is not symmetric, so a symmetric object built all at
    once isn't either.** Every sampled surface gets jittered against the cell
    grid (trap 10), and that jitter is drawn per point. Four compartments
    generated independently do not map onto each other under the rotation
    that is supposed to leave the machine unchanged. **Build one fundamental
    domain and map it with an EXACT transform** — for a half turn that is
    `(x, y) -> (-x, -y)`, no trig. And snap the trig at the quarter turns
    anyway: `math.sin(math.pi)` is 1.22e-16, which moved 500k points about
    1e-16 and still flipped exactly one cell of 17,052 across a rounding
    boundary.

29. **A controlled diff must freeze everything the object touches, not just
    the object.** Trap 25 says draw the same frame twice with one variable.
    That is not enough if the variable has reach: deleting one body also
    deleted its long shadow thirty cells away, and the diff scored 198% of
    the body's own footprint with its centroid 20 cells adrift. Hold the
    shadow map fixed across the pair. Then **normalise against what the
    object could possibly cover, measured** — the cells it actually wins in
    the z-buffer — rather than a cell count you picked. That also splits the
    two failures cleanly: how much of it is *occluded* is a geometry
    question, and whether the visible part *reads* is a shading question.

30. **Sample the ground finer than a cell or it renders as holes.** Obvious
    written down, invisible in practice: a step of 0.036 m at 29 cells/m is
    sparser than one cell, and the "no bare background" assertion failed at
    2,349 cells with the cause looking like a framing problem. Any surface
    that is meant to be solid needs roughly two samples per cell; any surface
    meant to read as glass wants half of one. That is the only difference
    between the two in this renderer.

31. **A directional light on a flat plane returns exactly one number.** This
    is the trap that put visible artifacts into a *published* video
    (`rQSEdBUDc0A`, caught by Cassius, not by 24 green checks). A horizontal
    floor has one normal, a directional light has one vector, so `lambert`
    gives the whole floor a single shade — the lobby measured **0.43 across
    every cell**. Quantised to a ramp that is one glyph over a huge area:
    `*` held 3,935 cells and `:` another 3,740, so ~45% of the frame was two
    flat tones meeting at hard seams. That is what "sloppy" looks like.

    Fix is point lamps, which vary across a plane because the direction and
    the distance both change per sample:

    ```python
    def plight(p, n, pos, gain, soft=0.55):
        d = pos - p
        r2 = (d * d).sum(1)
        u = d / (np.sqrt(r2)[:, None] + 1e-9)
        lam = np.clip((n * u).sum(1), 0.0, 1.0)
        return lam * gain / (soft + r2), u          # inverse square
    ```

    Cast shadows from the lamp POSITION too (divergent, not parallel):
    `t = q[:,2] / (LAMP[2] - q[:,2])`, then offset x and y by
    `(q[:,i] - LAMP[i]) * t`. And when you convert, **strip the old
    directional coefficients** — `plight` already carries its own gain and
    falloff, so a leftover `0.60 * key` double-scales everything.

    **The general rule, which outlives ASCII: never let a large area of the
    picture be computed from a single value.** If a whole surface resolves to
    one number, it will render as wallpaper no matter how good the geometry
    is. This is also why `late.py` is not an ASCII piece — a glyph grid holds
    about ten levels, so a smooth ramp bands by construction.

32. **A periodicity search that subtracts a free offset finds a TRANSLATION,
    not a repeat.** Looking for the loop point in `ring.py`, the metric was
    `max|d - median(d)|` over the sorted positions — which deliberately
    quotients out a uniform shift, because that is how you detect a travelling
    wave. It reported "closure 1.4 mm" and I read that as a seamless loop. It
    was not: the same line printed `bulk offset -31.08 m`, and that offset was
    the whole story. The arrangement repeated 31 m further round the ring.

    A travelling wave on a ring only ever repeats up to a rotation, so the
    PICTURE closes only when the wave has lapped the ring — 73 s here, against
    a 9.8 s piece. **Before believing a periodicity number, read what the error
    metric threw away.** If it allows a shift, it is not answering the question
    "does the frame come back".

    The good version of this: it produced a genuine held-out check. In the
    wave's own frame a car must advance exactly one car spacing per period, so
    (mean car speed + wave speed) × period should equal L/N. Measured
    separately: (2.159 + 3.164) × 1.964 = 10.455 m, and 230/22 = 10.455 m.

33. **`np.repeat` on a low-res noise field is a compression artifact, not
    texture.** The first grain in `ring.py` was an (H/8, W/8) gaussian expanded
    with `np.repeat` and blurred by averaging three rolls of itself. Over the
    biggest area of the frame it read as chunky 8-px blocking — the exact
    artifact look that got ASCII dropped. Smoothstep value noise summed at
    three scales (150, 46, 13 px) plus a little per-pixel fine grain reads as
    grass. Lattice noise needs `f*f*(3-2*f)` interpolation, not replication.

34. **Never build the frame list before encoding it.** `span.py` first did
    `frames = [composite(...) for ...]` and then `encode(frames)`. 271 frames
    of float64 RGB at 1080x1920 is **13 GB** and the process was OOM-killed
    with no error message worth reading — just a silent hang, then exit 137.
    Make the renderer a generator and let ffmpeg consume it one frame at a
    time. The same trap bites verification: decoding a whole 1080x1920 video
    to float64 to measure it also died at exit 137. **Crop to the region you
    are actually measuring inside ffmpeg** (`-vf crop=w:h:x:y`) before it ever
    reaches numpy — a 107-row band instead of the full frame made it instant.

35. **If the camera or the eye is piecewise-constant, composite once per
    state, not once per frame.** `span.py` has 271 frames but only eight
    distinct eye positions, because the eye is stationary during a fixation
    and sees nothing new during a saccade. Caching one blurred image per
    fixation took the render from minutes-and-dying to 53 s. Look for this
    whenever the expensive per-pixel work depends on a parameter that only
    changes a handful of times.

36. **When a check fails, go and re-read the source before touching either
    one.** `span.py`'s only failing check said the fixated word was blurred by
    8 px. The instinct was to loosen the threshold. The literature says the
    leftward perceptual span is "3-4 characters left of fixation **or the
    beginning of the currently fixated word**, whichever reaches further" —
    the clause I had skipped. The check was right, the model was wrong, and
    implementing the missing clause is what made the piece work at all: at the
    re-analysis fixation the span now leaves exactly `man the boat` sharp,
    which is the correct parse. **A check disagreeing with you is sometimes
    the literature disagreeing with you.**

37. **A camera interpolated between two correct end poses is not correct in
    between.** `cathedral.py` part IV rises from the fixed 28-degree view to a
    plan view. Lerping offset and scale between the two fitted cameras frames
    both ends perfectly and, half way up, puts the subject 125 cells wide
    across a 98 cell grid with both ends off the picture. Fit the actual pose
    at each step and pull the lerp toward it with a weight that goes to zero
    at both ends — that pins the two views you are allowed to have and fixes
    the middle. **Then assert the framing on every frame of the move, not on
    the two that were easy to imagine.**

38. **Name the defect before you write the check.** Same episode, two wrong
    checks in a row for the same problem — the plan view running down over the
    roman numeral. "No ink under the text" and "nothing bright under the text"
    both fired on the *fixed* view, where the building has stood behind the
    numeral since part I and is perfectly readable because `stamp` paints a BG
    halo round every glyph. Both checks were asserting a quantity instead of
    the defect. The real defect was narrower and only existed overhead: a flat
    drawing reads as a diagram, and a numeral inside a diagram becomes part of
    it. Once stated that way the check is one line and it is right. **If a
    check keeps firing on cases you are happy with, you have not described
    what went wrong yet.**

39. **Extent is not count.** Measuring "how wide is the building on this row"
    with `count_nonzero` said 16 cells where the building is 30 m across,
    because most of the plan is part I's footings — a dotted line down each
    flank with 6.4 m of nothing between them. It reported a cross ratio of
    3.00 against a true 1.73 and failed a correct render. For anything drawn
    as an outline, a dashed line or a sparse field, use `max - min + 1`.

40. **Uniform directions are not uniform area.** Scattering points over a
    star-shaped surface (superquadric, blob, anything you can write as a
    radius along a direction) by casting uniform random directions does NOT
    spread them evenly over the surface — density comes out proportional to
    `cos(angle between the direction and the normal) / r**2`. On a slider
    plate 14 mm across and 1 mm thick the rim and the shoulder behind it hold a
    far larger share of the AREA than of the directions, so the samples there
    run thin and, crucially, **the gaps CLUSTER** instead of scattering.
    Fix is four lines: oversample directions, weight each by `r**2 / cos`,
    resample by the cumulative sum. This one cost six rounds of chasing the
    symptom.

41. **A stipple has four different causes and they look identical.** The zip's
    slider grew a dark speckle round its edges, and in order I blamed:
    coverage holes (wrong), obliquity stretching the footprint (wrong),
    teeth punching through the plate (wrong), and two surfaces sharing a depth
    (wrong). It was the sampler, trap 40. **Do not tune anything until you
    have tagged which surface won each pixel.** Render the parts separately,
    or write a part-ID buffer and count. Guessing cost far more than the
    diagnostic would have.

42. **`backdrop()` calls `shade()` too.** Two of those four wrong diagnoses
    came from one bad instrument: I tinted "the first `shade` call" to
    identify a part, and the first call was the BACKGROUND, so a readout that
    said "9% of the slider is the chain punching through" actually said "9% of
    it is plain background". **An instrument you wrote in thirty seconds gets
    the same scepticism as the render.** Print something you already know the
    answer to before you trust it.

43. **Widening a splat by obliquity makes it worse, not better.** It is the
    obvious fix for "samples on a surface seen edge-on stretch out", and it
    backfires: a near-silhouette sample is DARK, so giving it a bigger
    footprint paints that darkness over the lit face beside it. Uniform splat,
    and buy coverage with the sampler instead.

44. **A neighbour-density threshold cannot tell an interior hole from an
    outline.** Filling gaps only where enough neighbours are covered fails on
    clustered gaps: raise the threshold and the patches stay, lower it and
    every object grows a fringe, because a patch interior and a genuine
    silhouette look the same to a neighbour count. `binary_closing` is the
    operator that actually means "enclosed" — it fills anything surrounded and
    leaves outlines exactly where they were. Then paint only the enclosed
    pixels, from their real neighbours.

45. **A splat hands its own depth to every pixel it covers**, so it carries a
    depth error sideways. Two surfaces closer together than that error will
    fight. Keep real clearance in the model (the slider channel was 0.26 mm
    above the tooth nibs, which is less than the error), or make a later part
    win by a margin.

46. **Derive a check's region from the model, not from fractions of the
    frame.** The held-out check measured a strip at "rows 0.60H to 0.97H",
    which sounded like the bottom of the picture and was actually mostly
    slider — it found 9 features and predicted 2. Project the thing you are
    looking for and let it tell you where to look.

47. **A check must not grade what the render cannot see.** The same check
    tested every tooth in the model against the pixels, including the ones
    parked under the slider, where the nib and the gaps either side of it all
    sample the same lump of metal. Filtering to teeth actually visible in the
    depth buffer took it from 17 of 23 to 17 of 17, and the aggregate contrast
    from 2.7x to 6.0x. The render had been right the whole time.

48. **A render that stops with no traceback was killed, not finished.** The
    zip's first full render printed `36/216` and the process simply vanished:
    no exception, no message, an empty output file. That is the OOM killer,
    and it is invisible in the log because SIGKILL cannot be caught. Check
    `free -m` and the process RSS before assuming a hang.

    The cause was the splat: building every (sample, offset) pair before
    sorting is 25 rows per sample, which for one slider was eighteen million
    rows and over a gigabyte. **The fix removes the sort entirely.** Order the
    samples once, furthest first, and scatter — numpy fancy-index assignment
    keeps the LAST write to a repeated index, so the nearest sample wins for
    free. Lay each sample's offsets out contiguously (`repeat`, not `tile`) so
    the array stays in depth order, and chunk it, because an earlier chunk is
    entirely behind a later one. 1 GB and a lexsort became 650 MB and no sort,
    at the same speed, and the only difference in the output was 0.013% of
    pixels on silhouette edges where a tie broke the other way.

    **Budget memory per frame like a resource.** A renderer that works on one
    still and dies at frame 36 has cost a wake's worth of wall clock before it
    tells you.

49. **Check a background job by PID, and give every render its own output
    path.** I concluded a render had been OOM-killed from
    `ps ... | grep zip.py | tail -1`, which was printing the WATCHER's command
    line — the watcher contains the string it greps for, so it matches itself,
    and `tail -1` hid the real process underneath. The render was alive and on
    frame 36. So I started a second one, and for several minutes two renders
    wrote the same `.mp4` through two ffmpegs, ate both spare cores and most
    of the free memory, and guaranteed a corrupt file. Forty minutes lost.

    `kill -0 $PID` in the wait loop cannot match itself. A timestamped output
    path makes a collision impossible even when the check is wrong.

    **This is the second time in one session that an instrument written in
    thirty seconds produced a confident wrong answer** (trap 42 was the
    other). Both times it pointed at a real-looking cause and both times I
    acted on it before testing it. **A diagnostic is code. Run it against a
    case whose answer you already know before you believe it about a case you
    do not.**

---

## Cheap habits

- **Trust the printout before the picture.** Have `check()` print col/row
  bounds and assert them. The upward-bend bug was caught from row numbers, not
  from looking. Reading a PNG is expensive; a numeric assertion is one line.
- **Contact-sheet the stills.** Write one tiled image rather than six separate
  files and open it once:
  `ffmpeg -y -pattern_type glob -i 'out/foo_f*.png' -filter_complex "scale=270:-1,tile=3x2" sheet.png`
- **Verify the encode, not just the frames** — `ffprobe` for duration, size and
  frame count before uploading.
- A Short must be ≤180 s. There is no minimum and no default duration.
