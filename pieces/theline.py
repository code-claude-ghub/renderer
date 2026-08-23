#!/usr/bin/env python3
"""THE LINE — where the United States put the word "overweight", and when.

One form: the body mass index of every American adult, drawn as a swarm.
Vertical position is BMI and nothing else. Horizontal position is noise --
a dot's left-right place carries no information, it is only there so the
mass has a body.

Then a rule comes down across it.

Until 1998 the federal cutoff for "overweight" was BMI 27.8 for men and
27.3 for women, set at the 1985 NIH Consensus Development Conference. The
NHLBI's own guidelines say what those numbers were: "the 85th percentile of
body mass index for men and women aged 20 through 29 years in NHANES II",
with "no particular relation to a specific increase in disease risk." In
June 1998 that was replaced by a single line at 25 for everybody, to match
the WHO. The same report calls the new cutpoints "somewhat arbitrary".

The swarm is NHANES III (1988-1994), the survey the panel was reading.
97 million American adults sat above the new line. The distribution is a
three-parameter shifted lognormal solved to hit three published prevalences
exactly; the survey's published MEDIAN is held out and used as a test.

Numbers, all from NHANES III, adults aged 20 and over:
    BMI >= 25   59.4% of men, 50.7% of women  -> 54.9% overall, 97 million
    BMI >= 30   19.5% of men, 25.0% of women
    BMI <  19    1.6% of men,  5.7% of women
    median BMI  25.5                          <- held out, predicted 25.61

Silent. No audio track.
"""

import os
import sys
import math

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]

from asciilib import Encoder, Frame, Grid, contact, ink_lut  # noqa: E402

# ----------------------------------------------------------------- palette
# pine ground, bone population, vermilion for the flagged half, chrome rule.
BG    = (0.043, 0.086, 0.078)
BONE  = (0.902, 0.878, 0.796)
FLAG  = (0.949, 0.353, 0.239)
RULE  = (0.996, 0.851, 0.310)

G = Grid()
LUT = ink_lut()
RC = G.rows * G.cols


def lerp(a, b, u):
    return tuple(a[i] + (b[i] - a[i]) * u for i in range(3))


# ------------------------------------------------------- the survey itself
# NHLBI Clinical Guidelines background chapter (NHANES III, adults 20+),
# sex-specific crude prevalence; Kuczmarski 1997 for the BMI<19 figure.
FRAC_M, FRAC_W = 0.485, 0.515          # adult sex split

P_GE25 = FRAC_M * 0.594 + FRAC_W * 0.507      # 0.5492  (published 54.9%)
P_GE30 = FRAC_M * 0.195 + FRAC_W * 0.250
P_LT19 = FRAC_M * 0.016 + FRAC_W * 0.057

MEDIAN_PUBLISHED = 25.5                # Kuczmarski 1997 -- HELD OUT
COUNT_GE25 = 97.1e6                    # "97 million Americans" -- NHLBI
ADULTS = COUNT_GE25 / P_GE25           # 176.8 million adults 20+
PER_DOT = 15000.0
N_DOTS = int(round(ADULTS / PER_DOT))  # 11,790

OLD_LINE = 27.8                        # men; 27.3 for women (see description)
NEW_LINE = 25.0


def _ncdf(z):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(z, float)
                                               / math.sqrt(2.0)))


def _fit_lognormal():
    """Solve a 3-parameter shifted lognormal onto three published points.

    BMI = m0 + exp(mu + sigma Z).  Damped Newton on a numerical jacobian --
    no scipy, so a clean checkout of the public repo can run this.
    """
    anchors = ((19.0, P_LT19), (25.0, 1.0 - P_GE25), (30.0, 1.0 - P_GE30))

    def resid(v):
        m0, mu, s = v
        out = []
        for x, c in anchors:
            out.append(float(_ncdf((math.log(x - m0) - mu) / s)) - c)
        return np.array(out)

    v = np.array([10.0, math.log(15.0), 0.35])
    for _ in range(200):
        r = resid(v)
        if np.abs(r).max() < 1e-13:
            break
        J = np.empty((3, 3))
        for j in range(3):
            h = 1e-6 * max(1.0, abs(v[j]))
            vp = v.copy()
            vp[j] += h
            J[:, j] = (resid(vp) - r) / h
        v = v - 0.9 * np.linalg.solve(J, r)
    return v, resid(v)


(M0, MU, SIG), FIT_RESID = _fit_lognormal()


def cdf(x):
    x = np.asarray(x, float)
    return np.where(x <= M0, 0.0,
                    _ncdf((np.log(np.maximum(x - M0, 1e-12)) - MU) / SIG))


def pdf(x):
    x = np.asarray(x, float)
    z = (np.log(np.maximum(x - M0, 1e-12)) - MU) / SIG
    return np.where(x <= M0, 0.0,
                    np.exp(-0.5 * z * z) / (math.sqrt(2 * math.pi) * SIG
                                            * np.maximum(x - M0, 1e-12)))


MODE = M0 + math.exp(MU - SIG * SIG)
PDF_MAX = float(pdf(MODE))
MEDIAN_FIT = M0 + math.exp(MU)

# ------------------------------------------------------------- the framing
R_MED = 84.0            # screen row of the survey median
K_ROW = 4.4             # rows per BMI unit
CX = G.cols / 2.0 - 0.5
W_MAX = 34.0            # half-width of the swarm at its widest, in cells

BMI_TOP = 65.0          # sampled up to here; the tail bleeds off the frame
BMI_BOT = M0            # the fitted distribution has no mass below this


def ROW(bmi):
    return R_MED - (np.asarray(bmi, float) - MEDIAN_PUBLISHED) * K_ROW


def BMI_OF_ROW(row):
    return MEDIAN_PUBLISHED + (R_MED - row) / K_ROW


def half_width(bmi):
    return W_MAX * pdf(bmi) / PDF_MAX


# --------------------------------------------------------------- the swarm
def build_swarm():
    """Uniform areal density inside the violin, so dots per BMI slice ~ pdf.

    Rejection sampling, fixed seed. Sampling BMI from the distribution AND
    setting the width from the density would count the density twice.
    """
    rng = np.random.default_rng(70425)
    bs, xs = [], []
    have = 0
    while have < N_DOTS:
        n = max(4 * (N_DOTS - have), 20000)
        b = rng.uniform(BMI_BOT, BMI_TOP, n)
        x = rng.uniform(-W_MAX, W_MAX, n)
        ok = np.abs(x) <= half_width(b)
        bs.append(b[ok])
        xs.append(x[ok])
        have += int(ok.sum())
    b = np.concatenate(bs)[:N_DOTS]
    x = np.concatenate(xs)[:N_DOTS]

    # Life. Horizontal sway carries no meaning and is free. The vertical
    # shimmer is deliberately tiny (+-0.35 rows = 0.08 BMI) and its phases
    # are random, so the marginal the piece asserts does not move; check()
    # measures the flagged count at six frames and refuses if it does.
    hamp = rng.uniform(0.7, 2.6, N_DOTS)
    hph = rng.uniform(0.0, 2 * math.pi, N_DOTS)
    hk = rng.integers(1, 4, N_DOTS).astype(float)
    vamp = rng.uniform(0.02, 0.09, N_DOTS)
    vph = rng.uniform(0.0, 2 * math.pi, N_DOTS)
    vk = rng.integers(1, 4, N_DOTS).astype(float)
    return b, x, hamp, hph, hk, vamp, vph, vk


B0, X0, HAMP, HPH, HK, VAMP, VPH, VK = build_swarm()

# ------------------------------------------------------------------ timing
FPS = 30
T_END = 11.5
FRAMES = int(round(T_END * FPS))

T_FALL = 1.70           # the rule comes down onto the 1985 cutoff
T_HOLD = 5.00           # ...and sits there
T_DROP = 5.30           # 1998: it drops to 25 and stays
T_1998 = (4.85, 7.40)
T_COUNT = 7.60

ENTRY_BMI = 44.0        # on frame, and already moving, at frame zero


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def line_bmi(t):
    if t <= 0.0:
        return ENTRY_BMI
    if t < T_FALL:
        u = t / T_FALL
        u = 1.0 - (1.0 - u) ** 3          # ease out: fast in, settles
        return ENTRY_BMI + (OLD_LINE - ENTRY_BMI) * u
    if t < T_HOLD:
        return OLD_LINE
    if t < T_DROP:
        return OLD_LINE + (NEW_LINE - OLD_LINE) * smooth(
            (t - T_HOLD) / (T_DROP - T_HOLD))
    return NEW_LINE


# ------------------------------------------------------------------- type
FONT = {
    "0": ("###", "# #", "# #", "# #", "###"),
    "1": (" # ", "## ", " # ", " # ", "###"),
    "2": ("###", "  #", "###", "#  ", "###"),
    "3": ("###", "  #", "###", "  #", "###"),
    "4": ("# #", "# #", "###", "  #", "  #"),
    "5": ("###", "#  ", "###", "  #", "###"),
    "6": ("###", "#  ", "###", "# #", "###"),
    "7": ("###", "  #", "  #", "  #", "  #"),
    "8": ("###", "# #", "###", "# #", "###"),
    "9": ("###", "# #", "###", "  #", "###"),
    ".": ("   ", "   ", "   ", "   ", " # "),
    ",": ("   ", "   ", "   ", " # ", "#  "),
}


def text_size(s, sc):
    return len(s) * 3 * sc + (len(s) - 1) * sc, 5 * sc


def text_cells(s, col0, row0, sc):
    """Boolean mask of the cells a string inks. Words are built OUT of cells:
    a one-cell glyph is about four pixels on a phone."""
    m = np.zeros((G.rows, G.cols), bool)
    c = col0
    for ch in s:
        pat = FONT[ch]
        for gr in range(5):
            for gc in range(3):
                if pat[gr][gc] != "#":
                    continue
                r0 = row0 + gr * sc
                x0 = c + gc * sc
                r1, x1 = min(r0 + sc, G.rows), min(x0 + sc, G.cols)
                if r0 < G.rows and x0 < G.cols and r1 > 0 and x1 > 0:
                    m[max(r0, 0):r1, max(x0, 0):x1] = True
        c += 4 * sc
    return m


def halo(mask, pad=1):
    out = mask.copy()
    for _ in range(pad):
        g = out.copy()
        g[1:, :] |= out[:-1, :]
        g[:-1, :] |= out[1:, :]
        g[:, 1:] |= out[:, :-1]
        g[:, :-1] |= out[:, 1:]
        out = g
    return out


READOUT_SC = 2
READOUT_COL = 61
BIG_SC = 3
COUNT_SC = 2
COUNT_STR = "97,000,000"
YEAR_STR = "1998"


def layout(t):
    """Every string on screen this frame: (text, col, row, scale, rgb)."""
    out = []
    lr = float(ROW(line_bmi(t)))

    s = "%.1f" % line_bmi(t)
    w, h = text_size(s, READOUT_SC)
    r0 = int(round(lr - h / 2.0))
    if r0 >= G.safe_top and r0 + h - 1 <= G.safe_bot:
        out.append((s, READOUT_COL, r0, READOUT_SC, RULE))

    if T_1998[0] <= t < T_1998[1]:
        w, h = text_size(YEAR_STR, BIG_SC)
        out.append((YEAR_STR, int(round(CX - w / 2.0)), 22, BIG_SC, RULE))

    if t >= T_COUNT:
        w, h = text_size(COUNT_STR, COUNT_SC)
        out.append((COUNT_STR, int(round(CX - w / 2.0)), 25, COUNT_SC, FLAG))
    return out


# ------------------------------------------------------------------ render
PEAK = 4.0              # dots in a cell that counts as fully inked


def cells(t):
    """Per-cell dot count, flagged-dot count, and the flat index."""
    ph = 2 * math.pi * t / T_END
    b = B0 + VAMP * np.sin(HK * ph + VPH)
    x = X0 + HAMP * np.sin(VK * ph + HPH)
    r = np.rint(ROW(b)).astype(np.int32)
    c = np.rint(x + CX).astype(np.int32)
    ok = (r >= 0) & (r < G.rows) & (c >= 0) & (c < G.cols)
    flat = (r[ok] * G.cols + c[ok])
    red = b[ok] > line_bmi(t)
    cnt = np.bincount(flat, minlength=RC)
    rcnt = np.bincount(flat[red], minlength=RC)
    return cnt.reshape(G.rows, G.cols), rcnt.reshape(G.rows, G.cols), b


def draw(f):
    t = f / float(FPS)
    cnt, rcnt, _ = cells(t)

    strings = layout(t)
    tmask = np.zeros((G.rows, G.cols), bool)
    for s, c0, r0, sc, _rgb in strings:
        tmask |= text_cells(s, c0, r0, sc)
    clear = halo(tmask, 2)
    # The rule is cut by the string's whole BOUNDING BOX, not by the glyph
    # halo -- cutting to the halo leaves rule stubs standing in the counters
    # and in the gaps between digits, which reads as damage rather than as a
    # window. The swarm still uses the tighter halo, so it fills the box.
    window = np.zeros((G.rows, G.cols), bool)
    for s, c0, r0, sc, _rgb in strings:
        w, h = text_size(s, sc)
        window[max(r0 - 2, 0):r0 + h + 2, max(c0 - 2, 0):c0 + w + 2] = True

    dens = np.clip(cnt / PEAK, 0.0, 1.0)
    shade = np.where(cnt > 0, 0.32 + 0.68 * dens, 0.0)
    idx = np.clip((shade * 255).astype(np.int32), 0, 255)
    key = (rcnt * 2 > cnt).astype(np.int32)
    show = (cnt > 0) & ~clear

    fr = Frame(G, BG)

    # swarm, run-length blitted: only two colour keys, so runs are long.
    inks = (BONE, FLAG)
    for r in range(G.rows):
        sig = np.where(show[r], idx[r] * 4 + key[r], -1)
        cuts = np.flatnonzero(np.r_[True, sig[1:] != sig[:-1]])
        for a, bnd in zip(cuts, np.r_[cuts[1:], G.cols]):
            if sig[a] < 0:
                continue
            fr.put_run(int(a), r, LUT[idx[r, a]] * int(bnd - a),
                       inks[key[r, a]])

    # the rule. A single row of any glyph is a row of dashes; '#' is the only
    # glyph at full ink, and two rows of it read as a solid bar.
    lb = line_bmi(t)
    lr = float(ROW(lb))
    speed = abs(lb - line_bmi(max(t - 1.0 / FPS, 0.0))) * K_ROW * FPS

    def rule_row(rr, ch, rgb):
        """Draw across the frame, but cut out around the readout so the
        number sits in a window in the rule instead of behind it."""
        if not (0 <= rr < G.rows):
            return
        gap = window[rr]
        cuts = np.flatnonzero(np.r_[True, gap[1:] != gap[:-1]])
        for a, bnd in zip(cuts, np.r_[cuts[1:], G.cols]):
            if not gap[a]:
                fr.put_run(int(a), rr, ch * int(bnd - a), rgb)

    if speed > 6.0:
        for j, u in enumerate((0.45, 0.65, 0.82)):
            rule_row(int(round(lr)) - 2 - j, "=-."[j], lerp(RULE, BG, u))
    for rr in (int(math.floor(lr)), int(math.floor(lr)) + 1):
        rule_row(rr, "#", RULE)

    for s, c0, r0, sc, rgb in strings:
        m = text_cells(s, c0, r0, sc)
        rr, cc = np.nonzero(m)
        for r in range(G.rows):
            row = m[r]
            if not row.any():
                continue
            cuts = np.flatnonzero(np.r_[True, row[1:] != row[:-1]])
            for a, bnd in zip(cuts, np.r_[cuts[1:], G.cols]):
                if row[a]:
                    fr.put_run(int(a), r, "#" * int(bnd - a), rgb)
    return fr


# ------------------------------------------------------------------- check
def check():
    print(G)
    print("fit  m0=%.4f mu=%.4f sigma=%.4f  max|resid|=%.2e"
          % (M0, MU, SIG, np.abs(FIT_RESID).max()))
    assert np.abs(FIT_RESID).max() < 1e-10, "distribution fit did not close"

    # HELD OUT: the survey's published median was not used to fit anything.
    print("median  predicted %.3f   published %.1f   miss %.3f"
          % (MEDIAN_FIT, MEDIAN_PUBLISHED, MEDIAN_FIT - MEDIAN_PUBLISHED))
    assert abs(MEDIAN_FIT - MEDIAN_PUBLISHED) < 0.15, "median test failed"

    print("P(BMI>=25) = %.4f   published 0.549" % P_GE25)
    print("adults 20+ = %.1f million   dots %d at %.0f each"
          % (ADULTS / 1e6, N_DOTS, PER_DOT))

    # THE CLAIM, AS A NUMBER: the swarm above the new line is 97 million.
    for t in (0.0, 2.0, 4.9, 6.0, 8.0, 11.4):
        _, rcnt, b = cells(t)
        above = int((b > NEW_LINE).sum()) * PER_DOT
        flagged = int((b > line_bmi(t)).sum()) * PER_DOT
        print("  t=%5.2f  line %5.2f   above 25: %5.1f M   flagged %5.1f M"
              % (t, line_bmi(t), above / 1e6, flagged / 1e6))
        assert abs(above - COUNT_GE25) / COUNT_GE25 < 0.015, \
            "the 97 million moved when the swarm breathed"

    # the two lines, on screen, in cells
    r_old, r_new = float(ROW(OLD_LINE)), float(ROW(NEW_LINE))
    print("rule rows: 27.8 -> %.1f   25.0 -> %.1f   drop %.1f cells"
          % (r_old, r_new, r_new - r_old))
    assert r_new - r_old > 10.0, "the drop is too small to see"
    assert G.safe_top < r_old < G.safe_bot
    assert G.safe_top < r_new < G.safe_bot

    # the line lands BELOW the middle of the country -- that is the piece.
    r_med = float(ROW(MEDIAN_PUBLISHED))
    print("median row %.1f  vs new rule row %.1f  (rule is %.1f cells lower)"
          % (r_med, r_new, r_new - r_med))
    assert r_new > r_med, "the new line must sit below the median"

    # every string inside the safe band and on the frame
    worst = 0
    for f in range(FRAMES):
        for s, c0, r0, sc, _ in layout(f / float(FPS)):
            w, h = text_size(s, sc)
            assert r0 >= G.safe_top, (s, r0)
            assert r0 + h - 1 <= G.safe_bot, (s, r0 + h - 1)
            assert c0 >= 0 and c0 + w <= G.cols, (s, c0, c0 + w)
            worst = max(worst, c0 + w)
    print("widest string ends at col %d of %d" % (worst, G.cols))

    # density: a mass that reads as a mass, with air around it
    for t in (0.0, 3.0, 6.0, 9.0, 11.4):
        cnt, _, _ = cells(t)
        ink = float((cnt > 0).mean())
        core = cnt[cnt > 0]
        print("  t=%5.2f  ink %.3f   median dots/cell %.1f   max %d"
              % (t, ink, np.median(core), core.max()))
        assert 0.14 < ink < 0.42, "ink coverage out of band"

    # the mass must be solid where it is thickest, not a net
    cnt, _, _ = cells(3.0)
    band = cnt[int(ROW(26.5)):int(ROW(23.5)), 20:78]
    holes = float((band == 0).mean())
    print("core holes %.3f  (rows %d..%d)"
          % (holes, int(ROW(26.5)), int(ROW(23.5))))
    assert holes < 0.12, "the core reads as a net, raise the dot count"

    print("frames %d  %.2f s at %d fps" % (FRAMES, T_END, FPS))


if __name__ == "__main__":
    out = os.path.join(_HERE, "..", "out", "theline.mp4")
    if "--check" in sys.argv:
        check()
    elif "--sheet" in sys.argv:
        check()
        picks = [0, 12, 45, 90, 150, 158, 168, 230, 300]
        contact([draw(f) for f in picks],
                os.path.join(_HERE, "..", "out", "theline_sheet.png"),
                cols=3, labels=["%.2fs" % (f / FPS) for f in picks])
        print("sheet written")
    else:
        check()
        import time
        t0 = time.time()
        with Encoder(out, G, fps=FPS) as enc:
            for f in range(FRAMES):
                enc.write(draw(f))
                if f % 60 == 0:
                    print("  %d/%d" % (f, FRAMES), flush=True)
        print("wrote %s in %.1f s" % (out, time.time() - t0))
