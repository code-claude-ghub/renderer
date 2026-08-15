#!/usr/bin/env python3
"""
"you have never seen a bubble pop."

Two facts, and the second one is why the first is true.

1. A soap bubble drains downwards.  The top ends up thinner than the light
   that is trying to reflect off it, so the two reflections -- one from the
   front of the film, one from the back -- cancel, and the crown of the
   bubble goes BLACK.  Not clear.  Black.  Newton saw it in 1704 and the
   thing is still called a Newton black film.  It is about six nanometres
   thick, which is two soap molecules back to back with almost no water
   left between them.

2. That is where it breaks, because that is where it is weakest, and the
   speed a hole opens in a liquid film is Taylor--Culick:

       v = sqrt( 2 sigma / (rho h) )

   Thinner film, faster hole.  At six nanometres and a soap-solution
   surface tension of 0.030 N/m that is exactly 100 metres a second.  The
   rim slows as it runs down into thicker film, and a five-centimetre
   bubble is completely gone in under five milliseconds.

   One frame of 30 fps video is 33 milliseconds.  The whole burst fits
   inside a single frame seven times over.  You have seen a bubble,
   and then you have seen no bubble.  You have never seen the hole move.

So this render is that event at 2,500x slow motion: 12.5 seconds of video
is five thousandths of a second of the world -- one second of whole bubble
and then exactly the burst, ending on the frame the last of the film goes.

EVERYTHING IN THE FRAME IS COMPUTED, NOT PICKED.

* The colours are not a palette.  For each cell the film thickness gives a
  reflectance spectrum from the exact Airy formula for a single layer,
  Fresnel s and p averaged at the true local angle of incidence.  That
  spectrum is multiplied by a 6504 K Planckian illuminant and integrated
  against the CIE 1931 colour matching functions (Wyman, Sloan & Shirley
  2013 multi-lobe analytic fit), then converted to sRGB.  check() asserts
  the result walks Newton's series in the right order: black, grey, white,
  straw, magenta, blue, green, and round again, washing out at high order.
  If the physics is right the palette is right, and I do not get a vote.

* The hole is not an expanding circle.  Its arrival time is the solution of
  the eikonal equation on the sphere with speed v(h(theta)), computed by
  Dijkstra over a 200x400 lat-long graph with exact great-circle edge
  lengths.  It nucleates off-centre in the black cap, races up over the
  crown where the film is thinnest, and crawls down into the thick coloured
  bands at a ninth of that speed.  check() verifies the solver against the
  closed form (T = R*angle/v) on a constant-speed sphere.

* Nothing falls.  Over the whole 5 ms window gravity moves anything in
  the frame by 0.12 mm -- under a third of one character cell -- against
  droplet flights of a quarter of a metre.  The bands do not drift either:
  draining to a black film takes tens of seconds, so at this timescale the
  thickness map is frozen, and the swirls in it are frozen with it.  Both
  of those are asserted, not assumed.

WHAT IS NOT TO SCALE, stated plainly:
  - The retracting rim is really about 40 micrometres of collected film,
    which is a twelfth of one character cell.  It is drawn 1.6 cells wide.
  - The droplets are really about 80 micrometres across and there are of
    order ten thousand of them.  A few hundred are drawn, each about six
    times too big.
  - You are seeing the near surface of the film only.  A real bubble also
    shows you the far side through it and the two sets of bands cross.

Everything else -- geometry, timing, speeds, colour -- is at true scale
under one single time compression.

Wordless.  Silent.

Shipped: https://youtube.com/watch?v=CQbHr8AbxUE
"""

import heapq
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asciilib import Grid, Frame, Encoder, ink_lut      # noqa: E402

OUT = "/tmp/bubble_burst.mp4"

FPS = 30
DUR = 12.5
FRAMES = int(round(FPS * DUR))
WINDOW = 5.0e-3                 # seconds of real time in the whole video
SLOWDOWN = DUR / WINDOW         # 2500x, exactly
# LEAD is not a free parameter. The video ends on the frame the last of the
# visible film goes, so the run-up is whatever is left of the window after
# the burst has taken its share -- see Scene.lead and check().

# --- the bubble, in SI ---
R_BUB = 0.025                   # 5 cm across, an ordinary wand bubble
SIGMA = 0.030                   # N/m, soap solution well above the CMC
RHO = 1000.0                    # kg/m3
N_FILM = 1.34                   # soapy water

D_BLACK = 6.0e-9                # Newton black film: a surfactant bilayer
D_COMMON = 30.0e-9              # common black film, just below the colours
D_FOOT = 900.0e-9                # thickest film, at the foot of the bubble
TH_BLACK = math.radians(50.0)   # the black cap reaches this far down
PROFILE_Q = 1.15                # how fast thickness grows below the cap

NUC_TH = math.radians(18.0)     # hole nucleates here, inside the black cap
NUC_PH = math.radians(110.0)    # front of the sphere, left of centre

DRAIN_TIME = 30.0               # order-of-magnitude: seconds to go black

# --- frame ---
GRID_FONT = 16
BUB_CELLS = 104.0               # bubble diameter in character cells
BUB_ROW = 0.43                  # sits high; the spray needs the room below
SS = 3                          # subsamples per cell per axis
RIM_CELLS = 1.7                 # drawn rim width at the equator
RIM_MAX = 4.2                   # ... and the cap on it near the foot
N_DROPS = 1600                  # drawn droplets (see docstring)
EXPOSE = 0.45                   # display gamma on reflectance
PEDESTAL = 0.12                 # below this the film is drawn as nothing

BG = (0.013, 0.011, 0.026)
RIM_COL = (1.000, 0.985, 0.955)
DROP_COL = (0.880, 0.900, 0.960)

NTH, NPH = 200, 400             # eikonal grid
LAM = np.arange(390.0, 731.0, 4.0)

LUT = ink_lut()

# 16-neighbourhood: the 8-neighbour version leaves a visible octagon on a
# front this large. Knight moves cut the anisotropy to well under a percent.
# (the 16-neighbourhood the Dijkstra version needed is gone with it)


# --------------------------------------------------------------- colour ---
def _lobe(x, mu, s1, s2):
    return np.exp(-0.5 * ((x - mu) / np.where(x < mu, s1, s2)) ** 2)


def cmf(lam):
    """CIE 1931 2-deg observer, multi-lobe analytic fit."""
    x = (1.056 * _lobe(lam, 599.8, 37.9, 31.0)
         + 0.362 * _lobe(lam, 442.0, 16.0, 26.7)
         - 0.065 * _lobe(lam, 501.1, 20.4, 26.2))
    y = (0.821 * _lobe(lam, 568.8, 46.9, 40.5)
         + 0.286 * _lobe(lam, 530.9, 16.3, 31.1))
    z = (1.217 * _lobe(lam, 437.0, 11.8, 36.0)
         + 0.681 * _lobe(lam, 459.0, 26.0, 13.8))
    return x, y, z


def planck(lam_nm, T):
    l = lam_nm * 1e-9
    return 1.0 / l ** 5 / (np.exp(1.4388e-2 / (l * T)) - 1.0)


XB, YB, ZB = cmf(LAM)
ILLUM = planck(LAM, 6504.0)
ILLUM /= ILLUM.max()
WY = (YB * ILLUM).sum()
XYZ2RGB = np.array([[3.2406, -1.5372, -0.4986],
                    [-0.9689, 1.8758, 0.0415],
                    [0.0557, -0.2040, 1.0570]])


def film_reflectance(d, cos_i):
    """Airy formula for one layer in air, s and p averaged.

    d and cos_i are arrays of shape (n,); returns (n, len(LAM)).
    The pi phase jump at the front surface is already in the algebra:
    r_back = -r_front, which is what makes a vanishingly thin film DARK
    rather than bright, and is the whole reason a bubble goes black.
    """
    sin_t = np.sqrt(np.clip(1.0 - cos_i ** 2, 0.0, 1.0)) / N_FILM
    cos_t = np.sqrt(np.clip(1.0 - sin_t ** 2, 0.0, 1.0))
    rs = (cos_i - N_FILM * cos_t) / (cos_i + N_FILM * cos_t)
    rp = (N_FILM * cos_i - cos_t) / (N_FILM * cos_i + cos_t)
    delta = (4.0 * math.pi * N_FILM * d[:, None] * cos_t[:, None]
             / (LAM[None, :] * 1e-9))
    cd = np.cos(delta)
    out = 0.0
    for r in (rs, rp):
        r2 = (r ** 2)[:, None]
        out = out + 0.5 * (2.0 * r2 * (1.0 - cd)
                           / (1.0 + r2 * r2 - 2.0 * r2 * cd))
    return out


def spectrum_to_srgb(refl):
    """(n, nlam) reflectance -> linear XYZ -> sRGB, plus luminance Y."""
    x = (XB * ILLUM * refl).sum(1) / WY
    y = (YB * ILLUM * refl).sum(1) / WY
    z = (ZB * ILLUM * refl).sum(1) / WY
    lin = np.stack([x, y, z], 1) @ XYZ2RGB.T
    lin = np.clip(lin, 0.0, None)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin,
                    1.055 * np.clip(lin, 0, None) ** (1 / 2.4) - 0.055)
    return np.clip(srgb, 0.0, 1.0), y


# ------------------------------------------------------------ thickness ---
def thickness(theta):
    """Film thickness against polar angle from the crown.

    A Newton black film is not a gradient. It is an equilibrium bilayer at
    a fixed thickness, which is why real bubbles show a hard-edged black
    cap rather than a fade. Below it the film thickens toward the foot.
    """
    u = np.clip((theta - TH_BLACK) / (math.pi - TH_BLACK), 0.0, 1.0)
    below = D_COMMON + (D_FOOT - D_COMMON) * u ** PROFILE_Q
    return np.where(theta <= TH_BLACK, D_BLACK, below)


def swirl(theta, phi):
    """Marginal regeneration: real drainage is patchy, not laminar, so the
    bands on a real bubble wander instead of running dead level. A small
    frozen perturbation of the polar angle, and frozen is correct -- see
    the drainage assertion in check()."""
    return (0.030 * np.sin(3.0 * phi + 5.2 * theta)
            + 0.018 * np.sin(5.0 * phi - 2.7 * theta + 1.1))


def speed(d):
    return np.sqrt(2.0 * SIGMA / (RHO * d))


# ------------------------------------------------------------- eikonal ---
def arrival_times(uniform=None):
    """Fast marching on the sphere: the eikonal |grad T| = 1/v.

    Dijkstra was the first thing I wrote here and it is the wrong tool. A
    shortest path through graph nodes is not a geodesic, and its error is
    DIRECTIONAL -- it came out at 6.7% near the poles against 0.5% at the
    far side, which does not shift the timing so much as bend the shape of
    the front. Fast marching solves the local quadratic instead and has no
    preferred direction: 0.2% mean on the same grid, see check().

    The theta/phi grid is orthogonal on a sphere, with spacings R*dtheta
    and R*sin(theta)*dphi, so the standard two-direction update applies
    unchanged. The poles need no special edge: the first row sits at
    dtheta/2, and its whole ring is shorter than one cell across.

    `uniform` forces a constant thickness, to test against the closed form.
    """
    th = (np.arange(NTH) + 0.5) * math.pi / NTH
    ph = (np.arange(NPH) + 0.5) * 2.0 * math.pi / NPH
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    d = (thickness(TH + swirl(TH, PH)) if uniform is None
         else np.full(TH.shape, uniform))
    slow = (1.0 / speed(d)).ravel()

    dth = math.pi / NTH
    dph = 2.0 * math.pi / NPH
    a = R_BUB * dth                                   # theta spacing
    b = (R_BUB * np.sin(th) * dph)                    # phi spacing, per row

    T = np.full(NTH * NPH, np.inf)
    known = np.zeros(NTH * NPH, bool)
    i0 = min(NTH - 1, int(NUC_TH / math.pi * NTH))
    j0 = int(NUC_PH / (2 * math.pi) * NPH) % NPH
    start = i0 * NPH + j0
    T[start] = 0.0
    heap = [(0.0, start)]

    def solve(i, j):
        k = i * NPH + j
        t1 = min(T[(i - 1) * NPH + j] if i > 0 else np.inf,
                 T[(i + 1) * NPH + j] if i < NTH - 1 else np.inf)
        t2 = min(T[i * NPH + (j - 1) % NPH], T[i * NPH + (j + 1) % NPH])
        f = slow[k]
        bb = b[i]
        if t1 == np.inf and t2 == np.inf:
            return np.inf
        if t1 == np.inf:
            return t2 + bb * f
        if t2 == np.inf:
            return t1 + a * f
        ia2, ib2 = 1.0 / (a * a), 1.0 / (bb * bb)
        A = ia2 + ib2
        B = -2.0 * (t1 * ia2 + t2 * ib2)
        C = t1 * t1 * ia2 + t2 * t2 * ib2 - f * f
        disc = B * B - 4.0 * A * C
        if disc >= 0.0:
            t = (-B + math.sqrt(disc)) / (2.0 * A)
            if t >= max(t1, t2):
                return t
        return min(t1 + a * f, t2 + bb * f)

    while heap:
        t, k = heapq.heappop(heap)
        if known[k]:
            continue
        known[k] = True
        i, j = divmod(k, NPH)
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ii, jj = i + di, (j + dj) % NPH
            if not 0 <= ii < NTH:
                continue
            m = ii * NPH + jj
            if known[m]:
                continue
            nt = solve(ii, jj)
            if nt < T[m]:
                T[m] = nt
                heapq.heappush(heap, (nt, m))
    return T.reshape(NTH, NPH)


def sample_grid(field, theta, phi):
    """Bilinear lookup into a [NTH, NPH] field, wrapping in phi."""
    fi = np.clip(theta / math.pi * NTH - 0.5, 0.0, NTH - 1.0001)
    fj = (phi / (2 * math.pi) * NPH - 0.5) % NPH
    i0 = fi.astype(np.int64)
    j0 = fj.astype(np.int64)
    a, b = fi - i0, fj - j0
    i1 = np.minimum(i0 + 1, NTH - 1)
    j1 = (j0 + 1) % NPH
    return ((1 - a) * ((1 - b) * field[i0, j0] + b * field[i0, j1])
            + a * ((1 - b) * field[i1, j0] + b * field[i1, j1]))


# ---------------------------------------------------------------- scene ---
class Scene(object):
    def __init__(self, grid):
        self.g = grid
        self.scale = BUB_CELLS / (2.0 * R_BUB)      # cells per metre
        self.cx = grid.cols / 2.0
        self.cy = grid.rows * BUB_ROW
        self.T = arrival_times()

        # sub-sample lattice over the whole frame
        n = SS * SS
        cols = np.arange(grid.cols)
        rows = np.arange(grid.rows)
        off = (np.arange(SS) + 0.5) / SS
        cc = (cols[None, :, None] + off[None, None, :]).reshape(1, -1)
        rr = (rows[:, None, None] + off[None, None, :]).reshape(-1, 1)
        C = np.repeat(cc, grid.rows * SS, 0)
        Rw = np.repeat(rr, grid.cols * SS, 1)
        x = (C - self.cx) / self.scale
        y = (self.cy - Rw) / self.scale
        r2 = x * x + y * y
        inside = r2 < R_BUB * R_BUB
        z = np.sqrt(np.clip(R_BUB * R_BUB - r2, 0.0, None))

        theta = np.arccos(np.clip(y / R_BUB, -1.0, 1.0))
        phi = np.arctan2(np.where(inside, z, 1.0), x) % (2 * math.pi)
        cos_i = np.clip(np.sqrt(np.clip(r2, 0, None)) * 0.0 + z / R_BUB,
                        0.03, 1.0)          # |n . view| on the near surface

        th_eff = np.clip(theta + swirl(theta, phi), 0.0, math.pi)
        d = thickness(th_eff)
        flat = inside.ravel()
        rgb = np.zeros(inside.shape + (3,))
        lum = np.zeros(inside.shape)
        refl = film_reflectance(d.ravel()[flat], cos_i.ravel()[flat])
        c, y_ = spectrum_to_srgb(refl)
        rgb.reshape(-1, 3)[flat] = c
        lum.ravel()[flat] = y_

        # Exposure is set by the brightest band AT NORMAL INCIDENCE, not by
        # the brightest cell. The brightest cell is on the silhouette, where
        # the film is edge-on and Fresnel sends reflectance toward 1 -- eight
        # times the face-on peak. Normalising by that would have crushed the
        # entire bubble to a tenth of its brightness to make room for a
        # four-cell rim. So the rim clips, which is what a real one does.
        norm = spectrum_to_srgb(film_reflectance(
            np.array([550e-9 / (4 * N_FILM)]), np.array([1.0])))[1][0]
        self.peak_lum, self.max_lum = norm, lum.max()
        b = np.clip((np.clip(lum / norm, 0.0, 1.0)) ** EXPOSE - PEDESTAL,
                    0.0, None)
        b = b / (1.0 - PEDESTAL)
        self.sub_bright = b
        self.sub_rgb = rgb
        self.sub_inside = inside
        self.sub_T = np.where(inside, sample_grid(self.T, th_eff, phi), np.inf)
        self.sub_v = speed(d)

        # The rim is not a line of fixed width. It is every gram of film the
        # front has already eaten, gathered into a ring, so it fattens all the
        # way down and piles up as the ring shrinks toward the foot. Drawn
        # width follows sqrt of the real cross-section, which is the right
        # RELATIVE profile even though the absolute width is a lie (see
        # check()). Capped, because at the very bottom the true one diverges.
        self.rim_w = self._rim_widths()
        self.sub_w = np.interp(theta, np.linspace(0, math.pi, len(self.rim_w)),
                               self.rim_w) / self.scale

        # fold sub-samples down to cells
        def fold(a):
            s = a.reshape(grid.rows, SS, grid.cols, SS)
            return s.mean((1, 3))

        self.cover = fold(inside.astype(float))
        self.rgb = np.stack([np.divide(fold(rgb[..., k] * inside),
                                       np.maximum(self.cover, 1e-9))
                             for k in range(3)], -1)
        self.n_sub = n

        # normalise hue to full chroma; glyph density carries brightness
        mx = self.rgb.max(-1, keepdims=True)
        self.hue_rgb = self.rgb / np.maximum(mx, 1e-6)

        self.t_end = float(self.sub_T[np.isfinite(self.sub_T)].max())
        self.t_full = float(self.T.max())
        self.lead = WINDOW - self.t_end
        self.drops = self._seed_drops()

    @staticmethod
    def rim_area():
        """Cross-sectional area of the collected rim against polar angle.

        All the film between the crown and theta is in the ring, and the ring
        is 2*pi*R*sin(theta) long, so A = R * integral(h sin) / sin(theta).
        """
        n = 2000
        th = np.linspace(0.0, math.pi, n)
        cum = np.concatenate([[0.0], np.cumsum(
            0.5 * (thickness(th[1:]) * np.sin(th[1:])
                   + thickness(th[:-1]) * np.sin(th[:-1])) * (th[1] - th[0]))])
        a = R_BUB * cum / np.maximum(np.sin(th), 1e-3)
        return th, a

    def _rim_widths(self):
        th, a = self.rim_area()
        ref = np.interp(math.pi / 2.0, th, a)
        w = RIM_CELLS * np.sqrt(np.maximum(a, 0.0) / ref)
        return np.clip(w, 0.9, RIM_MAX)

    def _seed_drops(self):
        """Shed droplets from the rim, uniformly over swept area.

        Velocity is the local front direction (the surface gradient of the
        arrival-time field) at the local Taylor-Culick speed. No gravity:
        over the whole window it is worth a tenth of a micrometre.
        """
        rng = np.random.default_rng(20260815)
        gi, gj = np.gradient(self.T)
        th = (np.arange(NTH) + 0.5) * math.pi / NTH
        dth = math.pi / NTH
        dph = 2.0 * math.pi / NPH
        # front direction in (theta, phi); normalise on the sphere metric
        gt = gi / (R_BUB * dth)
        gp = gj / (R_BUB * np.sin(th)[:, None] * dph)
        nrm = np.hypot(gt, gp) + 1e-12
        gt, gp = gt / nrm, gp / nrm

        # sample nodes weighted by area so the spray is even over the sphere
        w = np.repeat(np.sin(th), NPH)
        w = w / w.sum()
        idx = rng.choice(NTH * NPH, size=N_DROPS, p=w, replace=False)
        i, j = np.divmod(idx, NPH)
        # jitter off the solver lattice: unjittered births drew the lat-long
        # grid into the spray as a set of straight radial streaks
        THs = np.clip(th[i] + (rng.random(N_DROPS) - 0.5) * dth, 1e-3,
                      math.pi - 1e-3)
        PHs = (j + 0.5 + (rng.random(N_DROPS) - 0.5)) * dph
        st, ct = np.sin(THs), np.cos(THs)
        sp, cp = np.sin(PHs), np.cos(PHs)
        pos = np.stack([R_BUB * st * cp, R_BUB * ct, R_BUB * st * sp], 1)
        e_th = np.stack([ct * cp, -st, ct * sp], 1)
        e_ph = np.stack([-sp, np.zeros_like(sp), cp], 1)
        d = thickness(THs + swirl(THs, PHs))
        v = speed(d)[:, None]
        vel = v * (gt[i, j][:, None] * e_th + gp[i, j][:, None] * e_ph)
        return pos, vel, self.T[i, j]

    def paint(self, fr, t):
        g = self.g

        def fold(a):
            return a.reshape(g.rows, SS, g.cols, SS).mean((1, 3))

        # A sub-sample outside the sphere is not "film that has survived".
        # Leaving it in left a ghost arc of the original silhouette hanging in
        # the frame after the film under it was gone.
        alive = self.sub_inside & (self.sub_T > t)

        behind = (t - self.sub_T) * self.sub_v
        rim = (behind >= 0.0) & (behind < self.sub_w) & self.sub_inside
        rimcov = fold(rim.astype(float))

        # area-weighted, so a cell the silhouette only clips is drawn dim
        bright = fold(self.sub_bright * alive)
        idx = np.clip((bright * 255).astype(np.int32), 0, 255)
        rr, cc = np.nonzero(idx)
        hue = self.hue_rgb
        for r, c in zip(rr, cc):
            ch = LUT[idx[r, c]]
            if ch == " ":
                continue
            b = bright[r, c]
            k = 0.45 + 0.55 * b
            col = hue[r, c]
            fr.put(c, r, ch, (col[0] * k, col[1] * k, col[2] * k))

        rr, cc = np.nonzero(rimcov > 0.12)
        for r, c in zip(rr, cc):
            a = min(1.0, rimcov[r, c] * 1.6)
            fr.put(c, r, "#" if a > 0.55 else "%", RIM_COL, a)

        pos, vel, t0 = self.drops
        live = t0 < t
        if live.any():
            p = pos[live] + vel[live] * (t - t0[live])[:, None]
            keep = (p[:, 2] > 0.0) | (np.linalg.norm(p, axis=1) > R_BUB)
            p = p[keep]
            col = np.rint(self.cx + p[:, 0] * self.scale).astype(int)
            row = np.rint(self.cy - p[:, 1] * self.scale).astype(int)
            for c, r in zip(col, row):
                fr.put(int(c), int(r), ".", DROP_COL, 0.95)


# ---------------------------------------------------------------- check ---
def check(scene):
    g = scene.g
    print(repr(g))

    # 1. the observer
    assert abs(LAM[YB.argmax()] - 555.0) <= 4.0, LAM[YB.argmax()]
    wx = (XB * ILLUM).sum() / WY
    wz = (ZB * ILLUM).sum() / WY
    s = wx + 1.0 + wz
    print("illuminant chromaticity %.4f, %.4f  (D65 = 0.3127, 0.3290)"
          % (wx / s, 1.0 / s))
    assert abs(wx / s - 0.3127) < 0.010 and abs(1.0 / s - 0.3290) < 0.010

    # 2. the film, against the closed form
    r = (N_FILM - 1.0) / (N_FILM + 1.0)
    want = 4 * r * r / (1 + r * r) ** 2
    got = film_reflectance(np.array([550e-9 / (4 * N_FILM)]),
                           np.array([1.0]))[0]
    j = np.abs(LAM - 550.0).argmin()
    print("peak reflectance %.4f, closed form %.4f" % (got[j], want))
    assert abs(got[j] - want) < 1e-3

    # 3. Newton's series, in order
    def look(d_nm):
        c, y = spectrum_to_srgb(film_reflectance(np.array([d_nm * 1e-9]),
                                                 np.array([1.0])))
        c = c[0]
        mx, mn = c.max(), c.min()
        sat = 0.0 if mx <= 0 else (mx - mn) / mx
        hue = math.degrees(math.atan2(math.sqrt(3) * (c[1] - c[2]),
                                      2 * c[0] - c[1] - c[2])) % 360.0
        return y[0], sat, hue

    peak = look(550.0 / (4 * N_FILM))[0]
    ybk = look(D_BLACK * 1e9)[0]
    print("black film %.1f nm reflects %.3f%% of the brightest band"
          % (D_BLACK * 1e9, 100 * ybk / peak))
    assert ybk / peak < 0.01, ybk / peak
    print("newton's series, in order:")
    series = [(98, "white", None), (160, "yellow", (30, 60)),
              (185, "red", (335, 360)), (204, "magenta", (265, 300)),
              (234, "blue", (195, 235)), (268, "cyan", (175, 200)),
              (294, "green", (100, 140)), (324, "yellow", (45, 70)),
              (394, "magenta", (290, 320)), (454, "cyan", (170, 195)),
              (494, "green", (110, 145)), (734, "washed out", None)]
    for d_nm, name, band in series:
        y, sat, hue = look(d_nm)
        if band is None:
            print("  %4d nm  %-11s sat %.3f  (%4.0f%% of peak)"
                  % (d_nm, name, sat, 100 * y / peak))
            assert sat < 0.08, (d_nm, name, sat)
        else:
            print("  %4d nm  %-11s hue %3.0f deg  sat %.2f  (%4.0f%% of peak)"
                  % (d_nm, name, hue, sat, 100 * y / peak))
            assert band[0] <= hue <= band[1], (d_nm, name, hue)

    # 4. Taylor-Culick
    v0 = speed(np.array([D_BLACK]))[0]
    print("taylor-culick at %.0f nm: %.1f m/s (%.0f km/h)"
          % (D_BLACK * 1e9, v0, v0 * 3.6))
    assert abs(v0 - 100.0) < 0.5
    vf = speed(np.array([D_FOOT]))[0]
    print("  at the foot, %.0f nm: %.1f m/s -- the rim slows by %.1fx"
          % (D_FOOT * 1e9, vf, v0 / vf))
    assert v0 / vf > 5.0

    # 5. eikonal solver against the closed form on a constant-speed sphere
    Tc = arrival_times(uniform=D_BLACK)
    th = (np.arange(NTH) + 0.5) * math.pi / NTH
    ph = (np.arange(NPH) + 0.5) * 2 * math.pi / NPH
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    dot = (np.sin(TH) * np.cos(PH) * math.sin(NUC_TH) * math.cos(NUC_PH)
           + np.cos(TH) * math.cos(NUC_TH)
           + np.sin(TH) * np.sin(PH) * math.sin(NUC_TH) * math.sin(NUC_PH))
    ang = np.arccos(np.clip(dot, -1, 1))
    exact = R_BUB * ang / v0
    cell_arc = R_BUB * math.pi / NTH
    off = np.abs(Tc - exact) * v0 / cell_arc        # front error, in grid cells
    far = ang > math.radians(30.0)
    rel = (Tc[far] - exact[far]) / exact[far]
    print("eikonal vs closed form on a constant-speed sphere:")
    print("  timing beyond 30 deg of arc: mean %+.3f%%, worst %.2f%%"
          % (100 * rel.mean(), 100 * np.abs(rel).max()))
    print("  front POSITION is out by at most %.2f solver cells = %.2f "
          "character cells everywhere"
          % (off.max(), off.max() * cell_arc * scene.scale))
    assert abs(rel.mean()) < 0.01 and off.max() < 1.5

    # 6. the burst, and how it compares to one frame of video
    t_burst = scene.t_full
    frame = 1.0 / FPS
    print("burst: whole sphere gone in %.3f ms; the front of it in %.3f ms"
          % (1e3 * t_burst, 1e3 * scene.t_end))
    print("  one 30 fps frame is %.1f ms -- the burst fits inside it %.1f times"
          % (1e3 * frame, frame / t_burst))
    assert 2.0e-3 < t_burst < 5.0e-3, t_burst
    assert frame / t_burst > 5.0

    # 7. one time base, and it is exact
    print("%.1f s of video = %.2f ms of the world = %.0fx slow"
          % (DUR, 1e3 * WINDOW, SLOWDOWN))
    print("  effective shutter %.0f frames a second; the fastest phone "
          "slow-motion is 960" % (FPS * SLOWDOWN))
    print("  %.2f s of whole bubble, then exactly the burst: the video ends "
          "on the frame the last of the visible film goes"
          % (scene.lead * SLOWDOWN))
    assert 0.15e-3 < scene.lead < 0.7e-3, scene.lead
    assert abs(SLOWDOWN - 2500.0) < 1e-6

    # 8. nothing falls, and nothing drains
    fall = 0.5 * 9.81 * WINDOW ** 2
    flight = speed(np.array([D_COMMON]))[0] * WINDOW
    print("gravity over the whole window: %.3f mm = %.2f of a cell, against "
          "a %.2f m droplet flight" % (1e3 * fall, fall * scene.scale, flight))
    assert fall * scene.scale < 0.5 and fall / flight < 1e-3
    print("drainage over the window: %.4f%% of the way to a black film"
          % (100 * WINDOW / DRAIN_TIME))
    assert WINDOW / DRAIN_TIME < 1e-3

    # 9. the rim and the droplets, and by how much they are drawn wrong
    swept = 4 * math.pi * R_BUB ** 2
    th = (np.arange(NTH) + 0.5) * math.pi / NTH
    dmean = float((thickness(th) * np.sin(th)).sum()
                  / np.sin(th).sum())
    vol = swept * dmean
    half = R_BUB * (thickness(th[th <= math.pi / 2])
                    * np.sin(th[th <= math.pi / 2])).sum() * (math.pi / NTH)
    a_rim = half / 1.0                      # cross-section at the equator
    d_rim = 2.0 * math.sqrt(a_rim / math.pi)
    d_drop = 3.78 * (d_rim / 2.0)           # Rayleigh-Plateau, 9.02 a
    n_drop = vol / (math.pi / 6.0 * d_drop ** 3)
    cell_m = 1.0 / scene.scale
    print("film volume %.2f uL, mean thickness %.0f nm" % (1e9 * vol, 1e9 * dmean))
    print("rim at the equator: %.0f um = %.2f of a cell; drawn %.1f cells (%.0fx)"
          % (1e6 * d_rim, d_rim / cell_m, RIM_CELLS, RIM_CELLS / (d_rim / cell_m)))
    print("droplets: %.0f um across (%.2fx the rim), about %.0f of them; %d drawn"
          % (1e6 * d_drop, d_drop / d_rim, n_drop, N_DROPS))
    assert abs(d_drop / d_rim - 1.89) < 0.02
    assert 20e-6 < d_drop < 200e-6
    assert n_drop > N_DROPS

    # 10. the frame
    top = scene.cy - BUB_CELLS / 2.0
    bot = scene.cy + BUB_CELLS / 2.0
    print("bubble spans rows %.0f..%.0f in a %d-row frame, safe band %d..%d"
          % (top, bot, g.rows, g.safe_top, g.safe_bot))
    assert top > g.safe_top and bot < g.safe_bot
    print("one cell is %.2f mm of bubble; the film is %.0f nm, so the thing "
          "being drawn is %.0fx thinner than one character"
          % (1e3 * cell_m, 1e9 * dmean, cell_m / dmean))

    # 11. the bands actually land where they should
    for lab, ang in [("crown", 10), ("cap edge", 55), ("equator", 90),
                     ("foot", 170)]:
        d = float(thickness(np.array([math.radians(ang)]))[0])
        y, sat, hue = look(d * 1e9)
        print("  %-9s theta %3d deg  %6.0f nm  Y %.4f  sat %.2f  hue %3.0f"
              % (lab, ang, 1e9 * d, y, sat, hue))
    print("exposure: face-on peak Y %.4f; brightest cell on the silhouette "
          "is %.1fx that and clips" % (scene.peak_lum, scene.max_lum / scene.peak_lum))
    print("check ok")


def main():
    g = Grid(font_size=GRID_FONT)
    scene = Scene(g)
    check(scene)
    if "--check" in sys.argv:
        return
    if "--probe" in sys.argv:
        for frac in (0.0, 0.10, 0.25, 0.45, 0.65, 0.85, 1.0):
            f = int(frac * (FRAMES - 1))
            t = -scene.lead + WINDOW * f / (FRAMES - 1.0)
            fr = Frame(g, BG)
            scene.paint(fr, t)
            fr.surface.write_to_png("/tmp/bub_%02d.png" % int(frac * 100))
            print("  probe %.0f%%  t = %+.3f ms" % (100 * frac, 1e3 * t))
        return
    with Encoder(OUT, g, fps=FPS) as enc:
        for f in range(FRAMES):
            t = -scene.lead + WINDOW * f / (FRAMES - 1.0)
            fr = Frame(g, BG)
            scene.paint(fr, t)
            enc.write(fr)
            if f % 60 == 0:
                print("  frame %d/%d  t = %+.3f ms" % (f, FRAMES, 1e3 * t))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
