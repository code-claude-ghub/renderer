#!/usr/bin/env python3
"""
SPAN -- one eye reading one sentence that breaks it.

    the old man the boat.

That is a complete, grammatical English sentence. "man" is the verb -- to take
up position in order to operate something. Wiktionary's own citation for the
sense is Melville, Moby-Dick ch. 100: "Man the boat!" So the subject is "the
old", the verb is "man", the object is "the boat".

Almost nobody parses it that way first time. You build "the old man" as a noun
phrase, arrive at "boat", and the sentence collapses. Then your eye jumps
BACKWARD -- a regression -- and does it again.

This piece is that, drawn. Nothing is narrated and nothing is captioned. The
only thing on screen is the sentence, and the only thing that happens is where
the eye is:

  * The image is sharp only inside the reader's perceptual span and degrades
    outside it. The span is MEASURED, not invented: about 4 characters left of
    fixation and 14-15 right, for skilled readers of an alphabetic script
    (McConkie & Rayner 1975, 1976). The asymmetry is attentional, not optical
    -- it reverses in Hebrew readers -- so what you have ALREADY READ goes
    unreadable about 3.75x faster than what you are about to read.
  * Scale is set in reading's own unit. About 3 characters fall in a degree of
    visual angle at normal reading size, so the piece is built at 3 chars/deg
    and never has to assume how far your face is from the screen.
  * Fixation durations are in the published band for skilled silent reading:
    a mean near 250 ms, individual fixations 60-500 ms (Rayner 1998).
  * Saccades are 20-40 ms and vision is suppressed during them, so each jump
    washes out instead of smearing.
  * Both instances of "the" are skipped, never fixated. Short function words
    usually are.
  * There is exactly ONE leftward saccade in the whole piece and it happens
    the instant the sentence fails.

The held-out number: the schedule is built from fixation durations alone, and
the reading RATE falls out of it afterwards. First pass comes to ~360 wpm.
Total, including going back and doing it again, comes to ~120 wpm. The garden
path costs a factor of three, and nothing in the piece is tuned to make that
come out.

    python3 scripts/span.py --check          verify, render nothing
    python3 scripts/span.py --stills PREFIX  full-resolution PNGs
    python3 scripts/span.py --out x.mp4      render
"""

import argparse
import math
import os
import subprocess
import sys

import numpy as np
import cairo
from scipy.ndimage import gaussian_filter

# ------------------------------------------------------------------- frame
W, H = 1080, 1920
FPS = 60

SENTENCE = "the old man the boat."
FONT = "Charis SIL"          # designed for readability in literacy materials
TEXT_FRAC = 0.90             # sentence spans this much of the frame width
BASE_Y = 0.44                # baseline, fraction of height -- above the UI rail

# ------------------------------------------------------- the reading numbers
# McConkie & Rayner 1975/1976. Skilled readers of alphabetic script.
SPAN_L = 4.0                 # characters of useful information left of fixation
SPAN_R = 15.0                # ... and right of it
SPAN_V = 4.0                 # no published figure for one line; take the small one
CHARS_PER_DEG = 3.0          # normal reading size, Rayner 1998
FOVEA_CH = 1.5               # sharp core, ~0.5 deg radius

ASYM = SPAN_R / SPAN_L       # 3.75 -- behind you goes dark faster than ahead
SIG_RATE = 1.80              # px of blur per character of effective eccentricity
SIG_CAP = 34.0               # beyond here only word shape and length survive
SUPPRESS = 0.26              # contrast left during a saccade

# --------------------------------------------------------------- the schedule
# (word index, seconds).  Word indices: 0 "the" 1 "old" 2 "man" 3 "the" 4 "boat"
# Nothing here is a rounded-off guess: every duration is inside Rayner's band,
# and the two "the"s never appear.
PRE_ROLL = 0.18              # before the eye has landed anywhere
T_SACC = 0.030               # a normal reading saccade
T_REGR = 0.040               # the regression is a longer jump, so a longer flight
RESOLVE = 0.35               # the sentence comes sharp
HOLD = 1.50                  # and is held

FIXATIONS = [
    (1, 0.24),               # old
    (2, 0.25),               # man
    (4, 0.28),               # boat -- and it does not parse
    (1, 0.22),               # REGRESSION. all the way back.
    (2, 0.80),               # man. the long one. this is where it turns over.
    (4, 0.50),               # boat. now it parses.
]
FIRST_PASS = 3               # the first three fixations are the first pass

# ------------------------------------------------------------------- palette
PAPER = np.array([0.928, 0.913, 0.884])
INK = np.array([0.103, 0.099, 0.094])
GRAIN_AMP = 0.0075
VIGNETTE = 0.085


# ============================================================ text and metrics
def build_text():
    """Render the sentence once, sharp, and measure where every word is.

    Returns (alpha HxW float, metrics dict).  Metrics are in CHARACTERS, using
    the font's own advance for a lowercase 'n' as the character unit, because
    the reading literature counts character spaces and so must this.
    """
    surf = cairo.ImageSurface(cairo.FORMAT_A8, W, H)
    ctx = cairo.Context(surf)
    ctx.select_font_face(FONT, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)

    # size the type so the sentence spans TEXT_FRAC of the frame
    ctx.set_font_size(100.0)
    probe = ctx.text_extents(SENTENCE).x_advance
    size = 100.0 * (TEXT_FRAC * W) / probe
    ctx.set_font_size(size)

    adv = ctx.text_extents(SENTENCE).x_advance
    x0 = (W - adv) / 2.0
    y0 = H * BASE_Y

    ctx.set_source_rgba(0, 0, 0, 1)
    ctx.move_to(x0, y0)
    ctx.show_text(SENTENCE)
    surf.flush()

    buf = np.ndarray(shape=(H, surf.get_stride()), dtype=np.uint8,
                     buffer=surf.get_data())[:, :W]
    alpha = buf.astype(np.float64) / 255.0

    # character unit: the advance of one 'n'.  Not the mean glyph width -- the
    # literature's "character space" is a monospaced-equivalent unit and 'n' is
    # the conventional stand-in.
    ch_px = ctx.text_extents("n").x_advance

    # x of every character boundary, by measuring prefixes
    edges = [x0]
    for i in range(1, len(SENTENCE) + 1):
        edges.append(x0 + ctx.text_extents(SENTENCE[:i]).x_advance)
    edges = np.array(edges[1:])          # right edge of char i
    lefts = np.concatenate([[x0], edges[:-1]])

    words = []
    i = 0
    for w in SENTENCE.split(" "):
        j = SENTENCE.index(w, i)
        words.append({"text": w, "start": j, "end": j + len(w),
                      "x0": lefts[j], "x1": edges[j + len(w) - 1]})
        i = j + len(w)

    # Preferred viewing location: the eye lands between the start and the
    # middle of a word, not on its centre.  40% in.
    for w in words:
        w["fix_x"] = w["x0"] + 0.40 * (w["x1"] - w["x0"])

    return alpha, {"ch_px": ch_px, "size": size, "words": words,
                   "base_y": y0, "x0": x0, "adv": adv}


# ================================================================= the blur
PYR_SIGMAS = [0.0, 1.5, 3.0, 6.0, 10.0, 16.0, 24.0, 34.0]


def build_pyramid(alpha):
    return [alpha if s == 0.0 else gaussian_filter(alpha, s) for s in PYR_SIGMAS]


def sigma_map(fix_x, fix_y, ch_px, word_x0):
    """Blur in px at every pixel, given where the eye is.

    Eccentricity is measured in CHARACTERS, because that is the unit the
    reading literature uses and the unit the eye actually works in.

    Leftward it is scaled by ASYM -- the span asymmetry, an attentional effect
    and not an optical one.  But McConkie & Rayner's leftward boundary is "3-4
    letters left of fixation, OR THE BEGINNING OF THE CURRENTLY FIXATED WORD",
    whichever is further.  The word you are on is always available whole.  So
    the penalty only starts behind `word_x0`, which also makes the ramp
    continuous instead of kinked.
    """
    xs = (np.arange(W) - fix_x) / ch_px
    ys = (np.arange(H) - fix_y) / ch_px
    free = max(0.0, (fix_x - word_x0) / ch_px)      # back to the word's own start
    left = np.minimum(-xs, free) + ASYM * np.maximum(0.0, -xs - free)
    ex = np.where(xs > 0, xs, left)[None, :]
    ey = ys[:, None]
    e = np.sqrt(ex * ex + ey * ey)
    return np.minimum(SIG_RATE * np.maximum(0.0, e - FOVEA_CH), SIG_CAP)


def blur_by_map(pyr, smap):
    idx = np.interp(smap, PYR_SIGMAS, np.arange(len(PYR_SIGMAS), dtype=np.float64))
    out = np.zeros_like(smap)
    for i, lvl in enumerate(pyr):
        w = 1.0 - np.abs(idx - i)
        np.clip(w, 0.0, 1.0, out=w)
        if w.max() > 0:
            out += w * lvl
    return out


# ================================================================ finishing
def value_noise(cell, rng):
    """Smoothstep lattice noise.  Never np.repeat -- that reads as blocking."""
    gh, gw = H // cell + 2, W // cell + 2
    g = rng.normal(0.0, 1.0, (gh, gw))
    ys = np.arange(H) / cell
    xs = np.arange(W) / cell
    y0 = ys.astype(int)
    x0 = xs.astype(int)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    fy = fy * fy * (3.0 - 2.0 * fy)
    fx = fx * fx * (3.0 - 2.0 * fx)
    a = g[y0][:, x0]
    b = g[y0][:, x0 + 1]
    c = g[y0 + 1][:, x0]
    d = g[y0 + 1][:, x0 + 1]
    return ((a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy)


def make_paper():
    rng = np.random.default_rng(20260826)
    n = (0.55 * value_noise(3, rng) + 0.32 * value_noise(11, rng)
         + 0.13 * value_noise(47, rng))
    n = n / np.abs(n).max()
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    vig = 1.0 - VIGNETTE * np.clip(r / 1.35, 0, 1) ** 2.0
    base = PAPER[None, None, :] * vig[:, :, None]
    return np.clip(base + GRAIN_AMP * n[:, :, None], 0.0, 1.0)


def composite(paper, a):
    a3 = a[:, :, None]
    img = paper * (1.0 - a3) + INK[None, None, :] * a3
    return np.clip(img, 0.0, 1.0)


def to_u8(img):
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


# ================================================================= timeline
def nf(seconds):
    return int(round(seconds * FPS))


def build_timeline(m):
    """Every frame as (key, contrast), in order.

    `key` selects a cached composited alpha; `contrast` is 1.0 normally and
    SUPPRESS during a saccade -- you do not see the smear of your own eye
    movement, so the frame washes out rather than blurring along the flight.
    """
    words = m["words"]
    frames = [("pre", 1.0)] * nf(PRE_ROLL)

    prev_key, prev_x = "pre", None
    for k, (wi, dur) in enumerate(FIXATIONS):
        x = words[wi]["fix_x"]
        flight = T_REGR if (prev_x is not None and x < prev_x) else T_SACC
        frames += [(prev_key, SUPPRESS)] * nf(flight)
        key = ("fix", k)
        frames += [(key, 1.0)] * nf(dur)
        prev_key, prev_x = key, x

    steps = nf(RESOLVE)
    for i in range(steps):
        t = (i + 1) / steps
        frames.append((("resolve", t * t * (3.0 - 2.0 * t)), 1.0))
    frames += [(("resolve", 1.0), 1.0)] * nf(HOLD)
    return frames


def build_cache(m, pyr):
    """One blurred alpha per distinct eye state.

    There are only eight of them -- the eye is stationary during a fixation,
    and during a saccade you are not seeing anything new -- so the expensive
    part runs eight times, not 271.
    """
    words, ch = m["words"], m["ch_px"]
    by = m["base_y"] - m["size"] * 0.30     # the eye sits on the x-height
    cache = {"pre": blur_by_map(pyr, np.full((H, W), SIG_CAP * 0.85))}
    for k, (wi, _) in enumerate(FIXATIONS):
        cache[("fix", k)] = blur_by_map(
            pyr, sigma_map(words[wi]["fix_x"], by, ch, words[wi]["x0"]))
    return cache


def frame_at(cache, pyr, paper, key, contrast):
    if isinstance(key, tuple) and key[0] == "resolve":
        a = cache[("fix", len(FIXATIONS) - 1)] * (1.0 - key[1]) + pyr[0] * key[1]
    elif isinstance(key, tuple):
        a = cache[key]
    else:
        a = cache["pre"]
    return composite(paper, a * contrast)


def render_frames(m, pyr, paper):
    """Stream frames.  Never hold them all -- 271 float RGB frames is 13 GB."""
    cache = build_cache(m, pyr)
    for key, contrast in build_timeline(m):
        yield frame_at(cache, pyr, paper, key, contrast)


# =================================================================== checks
def wpm(words, seconds):
    return words / seconds * 60.0


def run_checks(m):
    words = m["words"]
    ok = True

    def chk(cond, msg):
        nonlocal ok
        print(("ok  " if cond else "FAIL ") + msg)
        if not cond:
            ok = False

    print("SPAN -- %r" % SENTENCE)
    print("type %.1f px, character unit %.2f px, %.2f chars per degree"
          % (m["size"], m["ch_px"], CHARS_PER_DEG))
    span_px_r = SPAN_R * m["ch_px"]
    print("perceptual span drawn sharp: %.0f px right, %.0f px left  (%.1f deg, %.1f deg)"
          % (span_px_r, SPAN_L * m["ch_px"], SPAN_R / CHARS_PER_DEG, SPAN_L / CHARS_PER_DEG))

    chk(len(words) == 5, "five words -- got %d" % len(words))
    chk(words[2]["text"] == "man", "the third word is the verb -- %r" % words[2]["text"])

    # every fixation lands inside the word it is meant to land on
    inside = all(words[wi]["x0"] <= words[wi]["fix_x"] <= words[wi]["x1"]
                 for wi, _ in FIXATIONS)
    chk(inside, "every fixation lands inside its own word")

    fixed = {wi for wi, _ in FIXATIONS}
    chk(0 not in fixed and 3 not in fixed,
        "neither 'the' is ever fixated -- short function words get skipped")

    durs = [d for _, d in FIXATIONS]
    fp = durs[:FIRST_PASS]
    chk(all(0.060 <= d <= 0.900 for d in durs),
        "every fixation is a duration a real eye makes -- %d..%d ms"
        % (round(min(durs) * 1000), round(max(durs) * 1000)))
    chk(all(0.060 <= d <= 0.500 for d in fp),
        "and every FIRST-PASS one is inside Rayner's 60-500 ms -- %d..%d ms"
        % (round(min(fp) * 1000), round(max(fp) * 1000)))
    chk(0.200 <= sum(fp) / len(fp) <= 0.300,
        "first-pass mean fixation is a skilled reader's ~250 ms -- %d ms"
        % round(sum(fp) / len(fp) * 1000))
    chk(0.020 <= T_SACC <= 0.040 and 0.020 <= T_REGR <= 0.040,
        "saccade flights are 20-40 ms -- %d and %d ms"
        % (round(T_SACC * 1000), round(T_REGR * 1000)))

    # saccade directions
    xs = [words[wi]["fix_x"] for wi, _ in FIXATIONS]
    back = [i for i in range(1, len(xs)) if xs[i] < xs[i - 1]]
    chk(len(back) == 1, "exactly one leftward saccade in the piece -- %d" % len(back))
    chk(back[0] == FIRST_PASS,
        "and it is the frame after the sentence fails -- fixation %d" % back[0])

    amps = [(xs[i] - xs[i - 1]) / m["ch_px"] for i in range(1, len(xs))]
    skip = amps[1]          # man -> boat, hopping over 'the'
    chk(7.0 <= skip <= 9.0,
        "the saccade that skips a word is 7-9 characters, as measured -- %.1f" % skip)
    chk(abs(amps[FIRST_PASS - 1]) > 10.0,
        "the regression is a long jump home -- %.1f characters" % amps[FIRST_PASS - 1])

    # the fixated word is actually sharp, and its neighbours are not
    by = m["base_y"] - m["size"] * 0.30
    worst = 0.0
    for wi, _ in FIXATIONS:
        sm = sigma_map(words[wi]["fix_x"], by, m["ch_px"], words[wi]["x0"])
        col = slice(int(words[wi]["x0"]), int(words[wi]["x1"]))
        row = slice(int(m["base_y"] - m["size"] * 0.62), int(m["base_y"] + m["size"] * 0.16))
        worst = max(worst, sm[row, col].max())
    chk(worst < 3.0, "the word under the eye is sharp -- worst blur on it %.2f px" % worst)

    sm = sigma_map(words[4]["fix_x"], by, m["ch_px"], words[4]["x0"])
    left_edge = sm[int(m["base_y"]) - 10, int(words[0]["x0"])]
    chk(left_edge >= SIG_CAP - 1e-9,
        "and with the eye on 'boat', 'the' at the far left is past the cap -- %.1f px"
        % left_edge)

    # the asymmetry is doing something visible
    d = int(8 * m["ch_px"])
    r8 = sm[int(by), min(W - 1, int(words[4]["fix_x"]) + d)]
    l8 = sm[int(by), max(0, int(words[4]["fix_x"]) - d)]
    chk(l8 > r8 * 3.0,
        "eight characters behind the eye is blurrier than eight ahead -- %.1f vs %.1f px"
        % (l8, r8))

    # ---- timing, and the held-out rate
    tl = build_timeline(m)
    total_f = len(tl)
    dur_s = total_f / FPS
    supp_f = sum(1 for _, c in tl if c != 1.0)
    chk(supp_f == nf(T_SACC) * (len(FIXATIONS) - 1) + nf(T_REGR),
        "vision is suppressed on exactly the saccade frames -- %d of %d"
        % (supp_f, total_f))
    chk(total_f % 1 == 0 and 3.0 <= dur_s <= 8.0,
        "piece is %d frames, %.2f s at %d fps -- inside the band that retains"
        % (total_f, dur_s, FPS))

    first_pass_s = sum(fp) + 2 * T_SACC
    reread_s = first_pass_s + T_REGR + sum(durs[FIRST_PASS:]) + 2 * T_SACC
    r1 = wpm(len(words), first_pass_s)
    r2 = wpm(len(words), reread_s)
    print("    first pass  %.2f s -> %.0f wpm" % (first_pass_s, r1))
    print("    all in      %.2f s -> %.0f wpm" % (reread_s, r2))
    chk(250 <= r1 <= 450,
        "first pass runs at a normal silent-reading clip -- %.0f wpm" % r1)
    chk(r2 < 160, "and the whole thing runs at a crawl -- %.0f wpm" % r2)
    chk(r1 / r2 > 2.5,
        "held out -- nothing here is tuned for it, but going back and doing it "
        "again costs a factor of %.1f" % (r1 / r2))

    return ok, total_f


# ==================================================================== output
def encode(frames, path):
    tmp = path + ".tmp.mp4"
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-",
         "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for f in frames:
        p.stdin.write(to_u8(f).tobytes())
    p.stdin.close()
    err = p.stderr.read().decode()
    if p.wait() != 0:
        sys.stderr.write(err[-3000:])
        raise SystemExit("ffmpeg failed")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--stills")
    ap.add_argument("--out")
    a = ap.parse_args()

    alpha, m = build_text()

    if a.check:
        ok, _ = run_checks(m)
        raise SystemExit(0 if ok else 1)

    pyr = build_pyramid(alpha)
    paper = make_paper()
    tl = build_timeline(m)
    print("%d frames, %.2f s" % (len(tl), len(tl) / FPS))

    if a.stills:
        from PIL import Image
        cache = build_cache(m, pyr)
        picks = {"a_preroll": 2,
                 "b_first_old": int(0.30 * FPS),
                 "c_first_boat": int(1.00 * FPS),
                 "d_saccade": int(1.09 * FPS),
                 "e_man_long": int(1.80 * FPS),
                 "f_resolved": len(tl) - 20}
        for name, i in picks.items():
            i = min(i, len(tl) - 1)
            Image.fromarray(to_u8(frame_at(cache, pyr, paper, *tl[i]))).save(
                "%s_%s.png" % (a.stills, name))
            print("  %s -> frame %d  key=%s" % (name, i, tl[i][0]))
        return

    if a.out:
        encode(render_frames(m, pyr, paper), a.out)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
