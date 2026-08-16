#!/usr/bin/env python3
"""
"two ripples in a pond pass through each other. these ones wipe each other
out."

An excitable medium -- the Belousov-Zhabotinsky reaction in a thin layer --
with two pacemakers firing on the same clock. Rings go out from both. Along
the line halfway between them every pair of fronts meets and both stop.
Nothing crosses.

The claim is the whole piece, so check() asserts it four ways:

  * the rest state is STABLE (both eigenvalues of the 2x2 Jacobian have
    negative real part). Nothing here oscillates on its own. Every patch of
    liquid sits still until a neighbour pokes it.
  * a poke DOES produce a pulse, so the medium is excitable rather than dead.
  * a second poke inside the recovery window produces NO pulse. That
    refractory wake is the entire reason two fronts cannot cross, and the
    pacemaker period is asserted to be longer than it.
  * in 1-D, two pulses launched at each other annihilate and leave nothing.
    The SAME collision under the linear wave equation is run as a control and
    both pulses come out the far side with their amplitude intact. Two rules,
    one collision, opposite answers.

What is on screen is `v`, the oxidised catalyst, because in a real dish that
is the only thing you can see. The species that actually carries the front,
HBrO2, is colourless. The colour is the wake, not the wave.

Model: the two-variable Oregonator in the Tyson-Fife scaling,

    du/dt = [ u(1-u) - f v (u-q)/(u+q) ] / eps  +  D lap(u)
    dv/dt = u - v

Dimensionless. Diffusion on the activator only; the catalyst is effectively
immobile on these timescales.

NOTHING HERE IS EYEBALLED. The renderer calibrates itself:

  * `eps` is chosen from a scan, on the one criterion that decides whether the
    picture is legible -- the ratio of wavelength to front thickness, which is
    a property of the kinetics alone and does not move when you rescale space.
  * `D` is then solved for so the front lands at a stated number of simulation
    cells, i.e. so it is RESOLVED rather than hoped for.
  * the pacemaker period is set from the measured recovery time, not picked.
  * the timestep is set from the diffusion stability limit for that D.

Colours follow the ruthenium-catalysed variant: reduced Ru(II) is red-orange,
oxidised Ru(III) green. That is also the light-sensitive one, which is why
these waves can be steered with a projector.

Wordless. Silent. Seamless: one pacemaker period is rendered and repeated, and
the residual drift across that period is measured and asserted small.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asciilib import Grid, Frame, Encoder, ink_lut            # noqa: E402

OUT = "/tmp/bz_annihilate.mp4"

FPS = 30
SUP = 2                     # simulation cells per character cell
FONT = 18

# --- kinetics -------------------------------------------------------------
Q = 0.002
F = 3.0
EPS = 0.05                  # only value in the scan that sustains a pulse
D = 1.0                     # solved for by calibrate()
DT = 0.002                  # set from the diffusion limit by calibrate()
STIM = 1.0                  # the pacemaker's own stimulus: as hard as it gets
FIRED = 0.05                # a fresh rise in the catalyst this big = a pulse

# --- staging --------------------------------------------------------------
FRONT_CELLS = 4.0           # simulation cells across the activator front
RINGS_ACROSS = 3.5          # wavelengths across the width of the frame
PERIOD_SAFETY = 1.35        # pacemaker period as a multiple of recovery time
FRAMES_PER_PERIOD = 60      # one period = 2.0 s of video
PERIODS = 6                 # -> 12.0 s
BURN_PERIODS = 9
PACE_R = 5.0
BLACK_POINT = 0.20          # fraction of peak clipped to the ground

# --- palette: ruthenium BZ. reduced Ru(II) red-orange, oxidised Ru(III)
# green. The ground is that red taken very dark, which is a rendering choice
# and the description says so.
PAPER = (0.052, 0.022, 0.014)
EMBER = (0.63, 0.21, 0.06)
LIME = (0.78, 0.99, 0.29)
NCOL = 24

LUT = ink_lut()


# --------------------------------------------------------------------------
# kinetics
# --------------------------------------------------------------------------

def steady():
    """The rest state. u* solves u(1-u) = f u (u-q)/(u+q); v* = u*."""
    b = 1.0 - Q - F
    u = 0.5 * (b + np.sqrt(b * b + 4.0 * Q * (1.0 + F)))
    return u, u


def react(u, v):
    du = (u - u * u - F * v * (u - Q) / (u + Q)) / EPS
    dv = u - v
    return du, dv


def lap2(a):
    p = np.pad(a, 1, mode="edge")           # no-flux walls, like a dish
    return (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
            - 4.0 * a)


def lap1(a):
    p = np.pad(a, 1, mode="edge")
    return p[:-2] + p[2:] - 2.0 * a


def nolap(a):
    return 0.0


def advance(u, v, t, laplace, dt=None, diff=None):
    dt = DT if dt is None else dt
    dd = D if diff is None else diff
    n = max(1, int(np.ceil(t / dt)))
    h = t / n
    for _ in range(n):
        du, dv = react(u, v)
        u = u + h * (du + dd * laplace(u))
        v = v + h * dv
        np.clip(u, 0.0, 1.5, out=u)
        np.clip(v, 0.0, 1.5, out=v)
    return u, v


# --------------------------------------------------------------------------
# measurements. all at D = 1; the PDE is exactly invariant under
# x -> x / sqrt(D), so speed scales as sqrt(D) and lengths as sqrt(D).
# --------------------------------------------------------------------------

def jacobian():
    us, vs = steady()
    h = 1e-7
    J = np.zeros((2, 2))
    f0 = react(np.array([us]), np.array([vs]))
    for j, p in enumerate(((h, 0.0), (0.0, h))):
        f1 = react(np.array([us + p[0]]), np.array([vs + p[1]]))
        J[0, j] = (f1[0][0] - f0[0][0]) / h
        J[1, j] = (f1[1][0] - f0[1][0]) / h
    return np.linalg.eigvals(J)


def fires(gap=None, tmax=30.0):
    """Poke a well-stirred patch. If `gap` is given, poke twice, `gap` time
    units apart, and report only the SECOND response.

    The discriminator is the CATALYST, not the activator. In this model the
    excited plateau of u is not near 1 -- it is pinned by v at about 0.74 --
    so "u went above 0.9" is not a test of anything. What a pulse actually is,
    and what the eye actually sees, is v being driven up and then decaying. So
    the measurement is the fresh RISE in v after the poke, above whatever v
    was at the moment of poking. At rest that rise is large. Inside the
    recovery window it is nothing, because v is already up there.
    """
    us, vs = steady()
    u = np.array([us])
    v = np.array([vs])
    u[0] = STIM
    if gap is not None:
        t = 0.0
        while t < gap:
            u, v = advance(u, v, 0.02, nolap, diff=0.0)
            t += 0.02
        u[0] = max(float(u[0]), STIM)
    base = float(v[0])
    peak_v = base
    peak_u = float(u[0])
    t = 0.0
    while t < tmax:
        u, v = advance(u, v, 0.02, nolap, diff=0.0)
        peak_v = max(peak_v, float(v[0]))
        peak_u = max(peak_u, float(u[0]))
        t += 0.02
        if t > 1.0 and float(v[0]) < 0.5 * peak_v and float(u[0]) < 0.02:
            break
    return peak_v - base, peak_u


def recovery_time():
    """Smallest gap at which a second full-strength poke still produces a
    pulse. That is the refractory window, and it is the whole reason two
    fronts cannot cross."""
    if fires(gap=30.0)[0] < FIRED:
        return float("nan")
    lo, hi = 0.05, 30.0
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        if fires(gap=mid)[0] > FIRED:
            hi = mid
        else:
            lo = mid
    return hi


def front_1d(n=900):
    """One pulse down a line at D = 1. Returns (speed, activator front
    thickness in cells, visible band thickness in cells)."""
    us, vs = steady()
    u = np.full(n, us)
    v = np.full(n, vs)
    u[:6] = 1.0
    tt, xx = [], []
    t = 0.0
    fw = bw = 0.0
    for _ in range(20000):
        u, v = advance(u, v, 0.05, lap1, diff=1.0)
        t += 0.05
        hot = np.nonzero(u > 0.3)[0]
        if len(hot) == 0:
            continue
        x = float(hot.max())
        tt.append(t)
        xx.append(x)
        if 0.45 * n < x < 0.55 * n:
            fw = float(len(hot))
            band = np.nonzero(v > 0.2 * max(v.max(), 1e-9))[0]
            bw = float(band.max() - band.min() + 1) if len(band) else 0.0
        if x > n - 40:
            break
    tt = np.array(tt)
    xx = np.array(xx)
    keep = (xx > 0.25 * n) & (xx < 0.85 * n)
    if keep.sum() < 5:
        return float("nan"), float("nan"), float("nan")
    c = float(np.polyfit(tt[keep], xx[keep], 1)[0])
    return c, fw, bw


def collide_excitable(n=500):
    """Two pulses at each other in 1-D. Returns what is left afterwards."""
    us, vs = steady()
    u = np.full(n, us)
    v = np.full(n, vs)
    u[:6] = 1.0
    u[-6:] = 1.0
    met = False
    peak_at_meeting = 0.0
    for _ in range(40000):
        u, v = advance(u, v, 0.05, lap1, diff=1.0)
        hot = np.nonzero(u > 0.3)[0]
        if not met:
            if len(hot) and (hot.max() - hot.min()) < 0.08 * n:
                met = True
                peak_at_meeting = float(u.max())
        else:
            if float(u.max()) < 0.05:
                break
    return peak_at_meeting, float(np.abs(u - us).max())


def collide_linear(n=500, sig=6.0):
    """The same collision under the linear wave equation, as a control.

    Courant number 1 exactly, so the discrete scheme IS the shift operator and
    the control carries no numerical dispersion of its own.
    """
    x = np.arange(n, dtype=float)
    a, b = 60.0, n - 61.0

    def two(sa, sb):
        return (np.exp(-0.5 * ((x - sa) / sig) ** 2)
                + np.exp(-0.5 * ((x - sb) / sig) ** 2))

    y = two(a, b)
    yp = two(a - 1.0, b + 1.0)          # one step ago -> directions
    before = float(y.max())
    for _ in range(int(n - 2 * a)):
        yn = np.roll(y, 1) + np.roll(y, -1) - yp
        yp, y = y, yn
    return before, float(y[:n // 2].max()), float(y[n // 2:].max())


def scan(verbose=True):
    """Which eps actually carries a wave? Reported because the answer is
    narrow and it is the reason EPS is not a free knob."""
    global EPS
    keep = EPS
    for eps in (0.02, 0.035, 0.05, 0.08, 0.12, 0.2):
        EPS = eps
        ev = jacobian()
        rise, pu = fires()
        c, fw, bw = front_1d(500)
        print("  eps %.3f  stable %-5s  poke: v +%.3f, u peaks %.3f  "
              "c %s  front %s  band %s"
              % (eps, bool((ev.real < 0).all()), rise, pu,
                 "  n/a" if c != c else "%5.2f" % c,
                 " n/a" if fw != fw else "%4.1f" % fw,
                 " n/a" if bw != bw else "%4.1f" % bw))
    EPS = keep


def calibrate(g=None, verbose=True):
    """Everything downstream of EPS is solved for, not chosen."""
    global D, DT
    ev = jacobian()
    rise, pu = fires()
    trec = recovery_time()
    c1, fw, bw = front_1d()
    if fw != fw or c1 != c1:
        raise SystemExit("no propagating pulse at eps=%.3f" % EPS)

    # D sets the length scale. The PDE is exactly invariant under
    # x -> x/sqrt(D), so this is a choice of units, made so the front is
    # RESOLVED rather than hoped for.
    D = (FRONT_CELLS / fw) ** 2
    DT = min(0.002, 0.15 / D)
    c = c1 * np.sqrt(D)
    band = bw * np.sqrt(D)

    width = (g.cols if g is not None else 100) * SUP
    T = max(PERIOD_SAFETY * trec, (width / RINGS_ACROSS) / c)
    lam = c * T

    best = dict(eps=EPS, trec=trec, c=c, fw=FRONT_CELLS, bw=band,
                rise=rise, pu=pu, ev=ev)
    if verbose:
        print("eps %.3f   D %.3f   dt %.5f" % (EPS, D, DT))
        print("front %.1f cells   visible band %.1f cells (%.1f characters)"
              % (FRONT_CELLS, band, band / SUP))
        print("speed %.2f cells/time   recovery %.2f   period %.2f"
              % (c, trec, T))
        print("wavelength %.1f cells (%.1f characters), %.1f across the frame,"
              " band is %.0f%% of it"
              % (lam, lam / SUP, width / lam, 100 * band / lam))
    return best, c, T, lam


# --------------------------------------------------------------------------
# the picture
# --------------------------------------------------------------------------

def smoke(T, n=140):
    """Does one imposed spot actually launch a ring in 2-D? Curvature costs a
    front more than it costs a plane front, so a stimulus that propagates fine
    on a line can sit there and die on a disc. Cheap, and it caught exactly
    that."""
    us, vs = steady()
    u = np.full((n, n), us)
    v = np.full((n, n), vs)
    yy, xx = np.mgrid[0:n, 0:n]
    d = ((yy - n // 2) ** 2 + (xx - n // 2) ** 2) <= PACE_R ** 2
    u[d] = 1.0
    v[d] = 0.0
    for _ in range(30):
        u, v = advance(u, v, T / 30.0, lap2)
    hot = v > vs + 0.05
    rad = 0.0
    if hot.any():
        ry, rx = np.nonzero(hot)
        rad = float(np.hypot(ry - n // 2, rx - n // 2).max())
    print("  2-D nucleation: peak v %.3f, excited %.1f%%, ring radius %.0f "
          "cells after one period" % (v.max(), 100 * hot.mean(), rad))
    return rad


def simulate(g, t_period, verbose=True):
    R, C = g.rows * SUP, g.cols * SUP
    us, vs = steady()
    u = np.full((R, C), us)
    v = np.full((R, C), vs)

    yy, xx = np.mgrid[0:R, 0:C]
    # Placed on exactly opposite integer cells, so the whole field is exactly
    # symmetric under a half turn about the centre. That is asserted later --
    # it is free, and it catches an indexing or laplacian error at once.
    r1, c1 = int(round(0.30 * R)), int(round(0.27 * C))
    centres = [(r1, c1), (R - 1 - r1, C - 1 - c1)]
    disks = [((yy - cy) ** 2 + (xx - cx) ** 2) <= PACE_R ** 2
             for cy, cx in centres]

    dt_frame = t_period / FRAMES_PER_PERIOD
    k0 = BURN_PERIODS * FRAMES_PER_PERIOD
    k1 = (BURN_PERIODS + 1) * FRAMES_PER_PERIOD
    frames, prev, drift = [], None, None

    for k in range(k1):
        if k % FRAMES_PER_PERIOD == 0:
            # A leading centre, imposed. u := 1 alone is below the critical
            # nucleus at this radius in 2-D -- curvature costs the front more
            # than it costs a plane front -- so the spot is also handed back
            # its recovery, which is what a genuinely rested speck of dust
            # would have.
            for d in disks:
                u[d] = 1.0
                v[d] = 0.0
        u, v = advance(u, v, dt_frame, lap2)
        if k == k0 - 1:
            prev = v.copy()
        if k >= k0:
            frames.append(v.reshape(g.rows, SUP, g.cols, SUP).mean((1, 3)))
        if verbose and k % 60 == 0:
            print("  sim frame %4d/%d   excited %.1f%%   peak v %.3f"
                  % (k, k1, 100.0 * (v > vs + 0.05).mean(), float(v.max())))
    drift = float(np.abs(v - prev).max())
    return np.array(frames), drift


def shades(frames):
    """Catalyst -> brightness.

    The rest state is NOT zero -- v* is 0.004 and the pulse peaks near 0.16 --
    so dividing by the peak leaves the whole resting dish sitting above the
    ink threshold, and the first render came out 100% covered. Subtract the
    rest state, then clip the bottom of the decaying tail to the ground, which
    is the one frankly aesthetic number here and is stated in the description.
    """
    _, vs = steady()
    hi = float(np.percentile(frames, 99.5))
    v0 = vs + BLACK_POINT * (hi - vs)
    s = np.clip((frames - v0) / max(hi - v0, 1e-9), 0.0, 1.0) ** 0.72
    print("black point %.4f, white point %.4f (rest is %.4f)" % (v0, hi, vs))
    return s


def colours():
    out = []
    for k in range(NCOL):
        w = (k / float(NCOL - 1)) ** 0.85
        out.append(tuple(EMBER[i] + (LIME[i] - EMBER[i]) * w
                         for i in range(3)))
    return out


def paint(fr, g, s, cols):
    idx = np.clip((s * 255.0).astype(np.int32), 0, 255)
    glyph = np.array(list(LUT))[idx]
    qi = np.clip((s * (NCOL - 1) + 0.5).astype(np.int32), 0, NCOL - 1)
    drawn = runs = 0
    for r in range(g.rows):
        gr, qr = glyph[r], qi[r]
        c = 0
        while c < g.cols:
            if gr[c] == " ":
                c += 1
                continue
            k = qr[c]
            c2 = c
            while c2 < g.cols and gr[c2] != " " and qr[c2] == k:
                c2 += 1
            fr.put_run(c, r, "".join(gr[c:c2]), cols[k])
            drawn += c2 - c
            runs += 1
            c = c2
    return drawn, runs


def check(best, c, T, lam, drift, s, g):
    us, _ = steady()
    print("rest state u* = v* = %.5f" % us)

    ev = jacobian()
    print("jacobian eigenvalues %s"
          % np.array2string(ev, precision=3, suppress_small=True))
    assert (ev.real < 0).all(), "rest state unstable -- medium self-oscillates"

    rise, pu = fires()
    print("one poke (u := %.1f): catalyst rises %.3f, activator peaks %.3f"
          % (STIM, rise, pu))
    assert rise > FIRED, "poke produced no pulse -- medium is not excitable"

    trec = best["trec"]
    half, hu = fires(gap=0.5 * trec)
    print("a second, identical poke at %.2f -- half the recovery time -- "
          "raises the catalyst by %.4f. nothing happens." % (0.5 * trec, half))
    assert half < FIRED, "no refractory window -- fronts could cross"
    print("recovery time %.2f, pacemaker period %.2f" % (trec, T))
    assert T > trec, "pacemaker fires into its own refractory wake"

    meet, resid = collide_excitable()
    print("excitable collision: peak u where they meet %.3f, "
          "largest thing left on the line afterwards %.5f" % (meet, resid))
    assert resid < 0.05, "something survived the collision"

    b, left, right = collide_linear()
    print("linear control: both pulses emerge at %.1f%% and %.1f%% of "
          "amplitude" % (100 * left / b, 100 * right / b))
    assert left > 0.9 * b and right > 0.9 * b, "control failed to superpose"

    rings = (g.cols * SUP) / lam
    print("%.1f wavelengths across the frame" % rings)
    assert 2.0 < rings < 9.0, "ring spacing is unreadable at this size"

    print("loop drift over one period: %.5f" % drift)
    assert drift < 0.03, "field still changing -- burn in longer"

    # The two leading centres sit on exactly opposite cells, so the field owes
    # us a half turn. NOT to machine precision, and it was wrong of me to
    # assert that: a cell and its mirror sum their four neighbours in mirrored
    # ORDER, float addition is commutative but not associative, and an
    # excitable medium amplifies the last bit. Measured 7e-4 of full scale
    # over 22,000 steps. The check still does its job -- a genuine indexing
    # error is O(1), not O(1e-3).
    sym = float(np.abs(s - s[:, ::-1, ::-1]).max())
    print("half-turn symmetry: worst cell differs by %.2e of full scale" % sym)
    assert sym < 0.02, "the two halves are not mirror images -- indexing bug"

    print("brightest cell %.3f, dimmest %.3f" % (s.max(), s.min()))
    assert s.max() > 0.9, "nothing in the frame is bright -- no waves formed"

    ink = float((s > 0.02).mean())
    print("ink coverage %.1f%%" % (100 * ink))
    assert 0.05 < ink < 0.6, "coverage outside the readable band"


def main():
    g = Grid(font_size=FONT)
    print(g)
    if "--probe" in sys.argv:
        scan()
        _, _, T, _ = calibrate(g)
        smoke(T)
        return

    best, c, T, lam = calibrate(g)
    rad = smoke(T)
    assert rad > 0.6 * c * T, "the imposed spot is not launching a ring"

    cache = "/tmp/bz_frames_%dx%d.npz" % (g.rows, g.cols)
    if os.path.exists(cache) and "--fresh" not in sys.argv:
        z = np.load(cache)
        frames, drift = z["frames"], float(z["drift"])
        print("reusing simulation from %s" % cache)
    else:
        frames, drift = simulate(g, T)
        np.savez_compressed(cache, frames=frames, drift=drift)

    s = shades(frames)
    check(best, c, T, lam, drift, s, g)
    cols = colours()

    if "--still" in sys.argv:
        for i in (0, 15, 30, 45):
            fr = Frame(g, PAPER)
            paint(fr, g, s[i], cols)
            fr.surface.write_to_png("/tmp/bz_still_%02d.png" % i)
            print("wrote /tmp/bz_still_%02d.png" % i)
        return

    with Encoder(OUT, g, fps=FPS) as enc:
        for p in range(PERIODS):
            for i in range(FRAMES_PER_PERIOD):
                fr = Frame(g, PAPER)
                n, runs = paint(fr, g, s[i], cols)
                enc.write(fr)
            print("  period %d/%d   ink %.1f%%   runs/cells %.2f"
                  % (p + 1, PERIODS, 100.0 * n / (g.rows * g.cols),
                     runs / float(g.rows * g.cols)))
    print("wrote %s  %.1f s"
          % (OUT, PERIODS * FRAMES_PER_PERIOD / float(FPS)))


if __name__ == "__main__":
    main()
