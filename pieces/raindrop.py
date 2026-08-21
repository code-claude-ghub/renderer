#!/usr/bin/env python3
"""THE LIFE OF A RAINDROP — sphere, bun, bag, burst.

One drop at fixed camera scale, growing 1.0 -> 6.0 mm by eating smaller
drops, flattening into the hamburger bun (ram pressure vs surface tension),
then the base blows out into a bag with a ring of water and it bursts into
a size-sorted shower. The camera settles on a 1.0 mm survivor: the loop.

The teardrop shape appears nowhere, because it has never fallen.

Facts anchored to opened sources (USGS water science school; Wikipedia
Drop (liquid) + Rain; Villermaux & Bossa 2009, Nat. Phys. 5, 697):
  - spherical below ~2 mm diameter
  - flattened bun bottom at 2-5 mm
  - bag instability past ~6 mm; record measured drop 8.8 mm
  - terminal velocity ~2 m/s at 0.5 mm, ~9 m/s at 5 mm
Relative vertical drift uses the Atlas et al. fit v = 9.65 - 10.3 e^(-0.6 d)
(m/s, d in mm), asserted in check() against both source anchors.

The clock is compressed (real growth takes minutes; a 3 mm drop rings at
tens of Hz). The bag morphology is staged after published photographs, not
solved. The description says which is which.
"""
import numpy as np
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
from asciilib import (Camera, Encoder, Frame, Grid, contact, depth_cue,
                      ink_lut, lambert, specular, visible, zbuffer)

G = Grid()
RAMP = ink_lut()

BG   = (0.055, 0.038, 0.028)          # warm near-black brown
DEEP = np.array([0.03, 0.17, 0.13])   # pine shadow
MIDC = np.array([0.33, 0.85, 0.58])   # celadon
WHT  = np.array([0.96, 1.00, 0.92])   # warm white
BGA  = np.array(BG)

FPS    = 30
T_END  = 18.3
FRAMES = int(round(T_END * FPS))      # 549
LAMP   = np.array([-0.45, -0.60, 0.66]); LAMP /= np.linalg.norm(LAMP)

# ---------------------------------------------------------------- timeline
D0, DMAX = 1.0, 6.0                  # mm, tracked diameter
T_GROW0, T_GROW1 = 1.2, 13.2         # continuous growth window
T_DIMPLE0, T_DIMPLE1 = 11.0, 13.8    # base depression develops
T_BAG0, T_BURST = 13.8, 15.0         # bag inflation ("almost explosively")
T_SETTLE1 = 15.9                     # camera settled on survivor
T_CLEAR = 17.4                       # all non-survivors must be gone
MERGES = [(3.2, 0.9), (5.4, 1.0), (7.8, 1.2), (10.4, 1.3)]  # (t, feeder mm)
ABSORB = 0.35                        # s, merge blend window
APPROACH = 1.6                       # s, feeder approach time
W_HZ = 1.15                          # displayed wobble rate (slowed; real is tens of Hz)

def vterm(d_mm):
    """Atlas et al. terminal velocity fit, m/s."""
    return 9.65 - 10.3 * np.exp(-0.6 * np.asarray(d_mm, float))

def sstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)

SUM_F3 = sum(f ** 3 for _, f in MERGES)
C_CONT = DMAX ** 3 - D0 ** 3 - SUM_F3   # volume gathered from sub-cell drops

def d_of(t):
    """Tracked diameter (mm). Volume-conserving: merges add exactly f^3."""
    if t >= T_BURST:
        return D0
    d3 = D0 ** 3 + C_CONT * sstep((t - T_GROW0) / (T_GROW1 - T_GROW0))
    for tm, f in MERGES:
        d3 += f ** 3 * sstep((t - tm) / ABSORB)
    return d3 ** (1.0 / 3.0)

def flatten_of(t):
    d = d_of(t)
    f = 0.55 * np.clip((d - 2.0) / 4.0, 0, 1)
    f += 0.45 * sstep((t - 12.0) / 1.8) * (1 if t < T_BURST else 0)
    return min(f, 1.0)

def dimple_of(t):
    if t >= T_BURST: return 0.0
    return 0.5 * sstep((t - T_DIMPLE0) / (T_DIMPLE1 - T_DIMPLE0))

def bag_of(t):
    if t < T_BAG0 or t >= T_BURST: return 0.0
    x = (t - T_BAG0) / (T_BURST - T_BAG0)
    return 2.0 * x ** 3                # ease-in: "almost explosively"

def wobble_of(t):
    """Amplitude envelope x global oscillator. Exactly 0 at t=0 and T_END."""
    if t < T_BURST:
        A = (0.02 + 0.05 * np.clip((d_of(t) - 2.0) / 4.0, 0, 1)) \
            * sstep((t - 2.0) / 1.2)
        for tm, _ in MERGES:
            if t > tm:
                A += 0.045 * np.exp(-(t - tm) / 0.8)
        A *= np.clip((14.2 - t) / 1.0, 0, 1)          # still for the bag
    else:
        A = 0.05 * np.exp(-(t - 15.3) / 0.6) if t > 15.3 else 0.0
        A *= np.clip((T_CLEAR - t) / 0.5, 0, 1)       # exactly 0 by T_CLEAR
    return min(A, 0.09) * np.sin(2 * np.pi * W_HZ * t)

# ---------------------------------------------------------------- geometry
RNG = np.random.default_rng(7)
NU, NP = 520, 520
uu = np.linspace(0.02, np.pi - 0.02, NU)
pp = np.linspace(0, 2 * np.pi, NP, endpoint=False)
UG, PG = np.meshgrid(uu, pp, indexing='ij')
UG = (UG + (RNG.random(UG.shape) - 0.5) * (uu[1] - uu[0]) * 0.9).ravel()
PG = (PG + (RNG.random(PG.shape) - 0.5) * (pp[1] - pp[0]) * 0.9).ravel()
STIP = 1.0 + 0.06 * (RNG.random(UG.size) - 0.5) * 2   # fixed-seed stipple

U_RIM = 1.05

def drop_surface(u, phi, t):
    """Deformed sphere, mm units. u=0 is the BOTTOM pole (+y, screen-down)."""
    R = d_of(t) / 2.0
    f, dim, B = flatten_of(t), dimple_of(t), bag_of(t)
    w = wobble_of(t)
    hx = (1 + w) * (1 + 0.18 * f)
    vy = 1.0 / (hx * hx)
    su, cu = np.sin(u), np.cos(u)
    x = su * np.cos(phi) * hx
    z = su * np.sin(phi) * hx
    y = cu * vy
    # bottom flattening + dimple (bottom = +y)
    y = y - 0.50 * f * np.exp(-(u / 0.80) ** 2) \
          - 0.45 * dim * np.exp(-(u / 0.45) ** 2)
    rim = 1 + 0.22 * f * su * su
    x, z = x * rim, z * rim
    # bag: membrane balloons upward off the rim ring — rounded top,
    # convex walls (a bag, not a cone)
    if B > 0:
        uw = np.clip(u / U_RIM, 0, 1)
        h = np.where(u < U_RIM, np.cos(uw * np.pi / 2) ** 0.6, 0.0)
        y = y - 1.55 * B * h
        gm = 1 + 0.32 * min(B, 1.0) * np.sin(uw * np.pi) * (u < U_RIM)
        g = 1 + 0.22 * min(B, 1.0) * np.exp(-((u - U_RIM) / 0.28) ** 2)
        x, z = x * g * gm, z * g * gm
    return np.stack([x * R, y * R, z * R], 1)

def drop_recenter(t):
    """Area-weighted mean y of the bag-free shape (dome rises, body stays)."""
    us = np.linspace(0.03, np.pi - 0.03, 60)
    R = d_of(t) / 2.0
    f, dim = flatten_of(t), dimple_of(t)
    hx = (1 + wobble_of(t)) * (1 + 0.18 * f)
    y = np.cos(us) / (hx * hx) - 0.50 * f * np.exp(-(us / 0.80) ** 2) \
        - 0.45 * dim * np.exp(-(us / 0.45) ** 2)
    wgt = np.sin(us)
    return float((y * wgt).sum() / wgt.sum() * R)

def sample_drop(t):
    """Point cloud + FD normals for the main drop at time t."""
    eps = 8e-4
    p0 = drop_surface(UG, PG, t)
    pu = drop_surface(UG + eps, PG, t)
    pf = drop_surface(UG, PG + eps, t)
    n = np.cross(pu - p0, pf - p0)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    # outward: for a star-shaped body, n . p >= 0
    flip = (n * p0).sum(1) < 0
    n[flip] *= -1
    p0[:, 1] -= drop_recenter(t)
    return p0, n, STIP

# small spheres (feeders, fragments) — coarse jittered param grid
nu3, np3 = 14, 14
u3 = np.linspace(0.10, np.pi - 0.10, nu3)
p3 = np.linspace(0, 2 * np.pi, np3, endpoint=False)
U3, P3 = np.meshgrid(u3, p3, indexing='ij')
U3 = (U3 + (RNG.random(U3.shape) - 0.5) * (u3[1] - u3[0]) * 0.9).ravel()
P3 = (P3 + (RNG.random(P3.shape) - 0.5) * (p3[1] - p3[0]) * 0.9).ravel()
SPH_T = np.stack([np.sin(U3) * np.cos(P3), np.cos(U3),
                  np.sin(U3) * np.sin(P3)], 1)
STIP3 = 1.0 + 0.06 * (RNG.random(U3.size) - 0.5) * 2

nu2, np2 = 64, 64
u2 = np.linspace(0.03, np.pi - 0.03, nu2)
p2 = np.linspace(0, 2 * np.pi, np2, endpoint=False)
U2, P2 = np.meshgrid(u2, p2, indexing='ij')
U2 = (U2 + (RNG.random(U2.shape) - 0.5) * (u2[1] - u2[0]) * 0.9).ravel()
P2 = (P2 + (RNG.random(P2.shape) - 0.5) * (p2[1] - p2[0]) * 0.9).ravel()
SPH = np.stack([np.sin(U2) * np.cos(P2), np.cos(U2), np.sin(U2) * np.sin(P2)], 1)
STIP2 = 1.0 + 0.06 * (RNG.random(U2.size) - 0.5) * 2

def sphere_at(center, d_mm, squash=1.0):
    pts = SPH.copy()
    pts[:, 1] *= squash
    n = SPH.copy()
    pts = pts * (d_mm / 2.0) + center
    return pts, n, STIP2

# ---------------------------------------------------------------- feeders
def feeder_state(t):
    """[(center xyz, diameter)] for feeders currently in flight."""
    out = []
    for (tm, f), xoff in zip(MERGES, (-0.9, 0.8, -1.1, 1.0)):
        t0, t1 = tm - APPROACH, tm + ABSORB * 0.6
        if not (t0 <= t <= t1):
            continue
        # contact point: bottom of the drop at merge time
        Rm = d_of(tm) / 2.0
        ybot = Rm * 0.85 - drop_recenter(tm)
        start = np.array([xoff, 8.0, 0.0])
        end = np.array([xoff * 0.12, ybot + f * 0.25, 0.0])
        s = sstep((t - t0) / APPROACH)
        c = start + (end - start) * s
        shrink = 1.0 - sstep((t - tm) / (ABSORB * 0.6)) if t > tm else 1.0
        if shrink <= 0.02:
            continue
        out.append((c, f * shrink))
    return out

# ---------------------------------------------------------------- fragments
# rim ring -> a necklace of BIG chunks (they fall away fast, down-frame);
# bag film -> fine mist (falls slower than the survivor, rises up-frame).
N_RIM, N_MIST = 12, 300
ang = np.linspace(0, 2 * np.pi, N_RIM, endpoint=False) + 0.26
RIM_D = 2.0 + 0.7 * RNG.random(N_RIM)            # 2.0-2.7 mm chunks
SURV = 0                                          # survivor index
RIM_D[SURV] = D0                                  # the loop drop, exactly 1.0
MIST_A = RNG.random(N_MIST) * 2 * np.pi
MIST_H = RNG.random(N_MIST)                       # 0=near rim, 1=dome top
# conserve volume exactly: the mist carries the remainder of 6.0^3
mist_share = DMAX ** 3 - (RIM_D ** 3).sum()
assert mist_share > 0, "rim chunks already exceed parent volume"
mw = 0.45 + 0.55 * RNG.random(N_MIST)
MIST_D = mw * (mist_share / (mw ** 3).sum()) ** (1.0 / 3.0)
assert MIST_D.max() <= 0.85, MIST_D.max()        # mist must be mist
V_RAD_RIM = 2.6 + 1.2 * RNG.random(N_RIM)        # burst radial kick, mm/s eq
V_RAD_MIST = 5.0 + 3.5 * RNG.random(N_MIST)
SPEED = 5.2                                       # mm of drift per (m/s)/s
RB = DMAX / 2.0
RIM_R = RB * 1.22                                 # rim ring radius at burst
S_SEAT = np.array([np.cos(ang[SURV]) * RIM_R, 0.35 * RB,
                   np.sin(ang[SURV]) * RIM_R])

def survivor_world(t):
    """Survivor's world position: rim seat + decaying radial kick."""
    tb = t - T_BURST
    tau = 0.55
    kick = tau * (1 - np.exp(-tb / tau))
    rad = np.array([np.cos(ang[SURV]), 0.12, np.sin(ang[SURV])])
    return S_SEAT + rad * V_RAD_RIM[SURV] * kick

def view_shift(t):
    """Camera pans from the parent's centre onto the survivor."""
    settle = sstep((t - T_BURST) / (T_SETTLE1 - T_BURST))
    return survivor_world(t) * settle

def burst_centres(t):
    """Non-survivor centres/diameters after the burst, VIEW coords.
    Returns (rim_c (11,3), rim_d (11,), mist_c (300,3), mist_d (300,))."""
    tb = t - T_BURST
    shift = view_shift(t)
    tau = 0.55                                    # radial kick decays (drag)
    kick = tau * (1 - np.exp(-tb / tau)) + 0.18 * tb   # + slow residual drift
    keep = np.arange(N_RIM) != SURV
    a = ang[keep]
    base = np.stack([np.cos(a) * RIM_R, np.full(a.shape, 0.35 * RB),
                     np.sin(a) * RIM_R], 1)
    rad = np.stack([np.cos(a), np.full(a.shape, 0.12), np.sin(a)], 1)
    dv = vterm(RIM_D[keep]) - vterm(D0)           # heavier falls faster: +y
    rim_c = base + rad * (V_RAD_RIM[keep] * kick)[:, None]
    rim_c[:, 1] += dv * SPEED * tb
    rim_c -= shift
    xz_r = RIM_R * (1 - 0.72 * MIST_H)            # dome narrows toward its top
    y0 = -RB * (0.5 + 2.6 * MIST_H)               # spawn along the burst bag
    mbase = np.stack([np.cos(MIST_A) * xz_r, y0, np.sin(MIST_A) * xz_r], 1)
    mrad = np.stack([np.cos(MIST_A), np.full(MIST_A.shape, -0.45),
                     np.sin(MIST_A)], 1)
    mdv = vterm(MIST_D) - vterm(D0)               # lighter falls slower: up
    mist_c = mbase + mrad * (V_RAD_MIST * kick)[:, None]
    mist_c[:, 1] += mdv * SPEED * tb
    mist_c -= shift
    return rim_c, RIM_D[keep], mist_c, MIST_D

def burst_state(t):
    """Flat [(centre, d)] view of burst_centres, for checks."""
    rc, rd, mc, md = burst_centres(t)
    return [(rc[i], rd[i]) for i in range(len(rd))] + \
           [(mc[j], md[j]) for j in range(len(md))]

# ---------------------------------------------------------------- camera
def pose_pts(t):
    # fit on the main drop only: feeders ENTER from off-frame by design,
    # and burst fragments EXIT through the edges by design
    return sample_drop(t)[0]

FIT_TS = [0.0, 6.0, 10.0, 13.0, 13.79, 14.55, T_BURST - 1e-3]
CAM = Camera(G).fit([pose_pts(t) for t in FIT_TS], margin=1.06)

# ---------------------------------------------------------------- counter
FONT = {
    '0': ("111","101","101","101","111"), '1': ("010","110","010","010","111"),
    '2': ("111","001","111","100","111"), '3': ("111","001","011","001","111"),
    '4': ("101","101","111","001","001"), '5': ("111","100","111","001","111"),
    '6': ("111","100","111","101","111"), '7': ("111","001","010","010","010"),
    '8': ("111","101","111","101","111"), '9': ("111","101","111","001","111"),
    '.': ("0","0","0","0","1"),
    'm': ("00000","11110","10101","10101","10101"),
    ' ': ("00","00","00","00","00"),
}
TAG_SCALE = 2
TAG_ROW = 24

def draw_counter(fr, d):
    text = f"{d:.1f} mm"
    widths = [len(FONT[ch][0]) * TAG_SCALE + TAG_SCALE for ch in text]
    total = sum(widths) - TAG_SCALE
    c0 = (G.cols - total) // 2
    # halo plate: one clean band behind the glyphs, survives any background
    for r in range(TAG_ROW - 1, TAG_ROW + 5 * TAG_SCALE + 1):
        for c in range(c0 - 2, c0 + total + 2):
            fr.put(c, r, ' ', BG)
    col = c0
    ink = tuple((BGA + (WHT - BGA) * 0.92))
    for ch in text:
        rows = FONT[ch]
        w = len(rows[0])
        for rr in range(5):
            for cc in range(w):
                for sy in range(TAG_SCALE):
                    for sx in range(TAG_SCALE):
                        r = TAG_ROW + rr * TAG_SCALE + sy
                        c = col + cc * TAG_SCALE + sx
                        if rows[rr][cc] == '1':
                            fr.put(c, r, '@', ink)
                        else:
                            fr.put(c, r, ' ', BG)
        col += w * TAG_SCALE + TAG_SCALE
    return TAG_ROW, TAG_ROW + 5 * TAG_SCALE

# ---------------------------------------------------------------- shading
def shade_pts(p, n):
    lam = lambert(n, LAMP)
    s = 0.14 + 0.86 * lam
    s = s + 0.50 * specular(n, LAMP, 14) + 0.90 * specular(n, LAMP, 90)
    s = s + 0.18 * (1 - np.abs(n[:, 2])) ** 3
    return s

def colour(s, extra):
    s = float(min(max(s, 0.0), 1.0))
    if s < 0.55:
        c = DEEP + (MIDC - DEEP) * (s / 0.55)
    else:
        c = MIDC + (WHT - MIDC) * ((s - 0.55) / 0.45)
    c = BGA + (c - BGA) * (0.38 + 0.62 * s)
    return tuple(np.clip(c, 0, 1))

def gather(t):
    """All visible bodies -> one point/normal/stipple set."""
    P, N, S = [], [], []
    if t < T_BURST:
        p, n, st = sample_drop(t)
        P.append(p); N.append(n); S.append(st)
        for c, d in feeder_state(t):
            p, n, st = sphere_at(c, d)
            P.append(p); N.append(n); S.append(st)
    else:
        # the survivor IS the main drop now: same fine grid, so the loop
        # closes bit-identically on frame 0
        p, n, st = sample_drop(t)
        p = p + (survivor_world(t) - view_shift(t))
        P.append(p); N.append(n); S.append(st)
        rc, rd, mc, md = burst_centres(t)
        for i in range(len(rd)):
            p, n, st = sphere_at(rc[i], rd[i])
            P.append(p); N.append(n); S.append(st)
        # mist, vectorised: 300 tiny spheres in one shot
        mp = (SPH_T[None, :, :] * (md / 2)[:, None, None]
              + mc[:, None, :]).reshape(-1, 3)
        P.append(mp)
        N.append(np.tile(SPH_T, (len(md), 1)))
        S.append(np.tile(STIP3, len(md)))
    return np.concatenate(P, 0), np.concatenate(N, 0), np.concatenate(S, 0)

def draw(i):
    t = i / FPS
    p, n, st = gather(t)
    col, row, z = CAM.project(p)
    ok = visible(G, col, row)
    col, row, z = col[ok], row[ok], z[ok]
    n2, st2 = n[ok], st[ok]
    _, keep = zbuffer(G, col, row, z)
    s = shade_pts(p[ok], n2) * st2 * depth_cue(z, far=0.94)
    fr = Frame(G, BG)
    fr.field(col, row, keep, np.clip(s, 0, 1), colour, RAMP, extra=None)
    d_show = d_of(t) if t < T_BURST else D0
    draw_counter(fr, d_show)
    return fr

# ---------------------------------------------------------------- check
def frame_key(i):
    t = i / FPS
    p, n, st = gather(t)
    col, row, z = CAM.project(p)
    ok = visible(G, col, row)
    return col[ok], row[ok], np.round(
        shade_pts(p[ok], n[ok]) * st[ok] * depth_cue(z[ok], far=0.94), 6)

def check():
    # 1. terminal-velocity fit against opened-source anchors
    assert abs(vterm(0.5) - 2.0) < 0.4, vterm(0.5)   # wikipedia rain: 2 m/s
    assert abs(vterm(5.0) - 9.0) < 0.5, vterm(5.0)   # wikipedia rain: 9 m/s
    # 2. volume conservation: growth curve and burst products
    for tm, f in MERGES:
        before, after = d_of(tm - 1e-4) ** 3, d_of(tm + ABSORB + 1e-4) ** 3
        cont = C_CONT * (sstep((tm + ABSORB + 1e-4 - T_GROW0) / (T_GROW1 - T_GROW0))
                         - sstep((tm - 1e-4 - T_GROW0) / (T_GROW1 - T_GROW0)))
        assert abs(after - before - cont - f ** 3) < 1e-9
    frag_v = (RIM_D ** 3).sum() + (MIST_D ** 3).sum()
    assert abs(frag_v - DMAX ** 3) < 1e-9, frag_v
    assert abs(d_of(T_BAG0) - DMAX) < 1e-9
    # 3. loop seam: state and rendered samples identical at 0 and T_END
    assert wobble_of(0.0) == 0.0 and wobble_of(T_END) == 0.0
    assert np.allclose(survivor_world(T_END) - view_shift(T_END), 0)
    assert abs(d_of(T_END) - D0) < 1e-12
    ca, ra, sa = frame_key(0)
    cb, rb, sb = frame_key(FRAMES)
    assert ca.shape == cb.shape and (ca == cb).all() and (ra == rb).all() \
        and np.allclose(sa, sb), "loop seam"
    # ...and the loop is not a no-op
    cm, _, _ = frame_key(FRAMES // 2)
    assert cm.shape != ca.shape or not (cm == ca).all()
    # 4. every non-survivor is out of frame at T_CLEAR
    for c, d in burst_state(T_CLEAR):
        pcol, prow, _ = CAM.project(sphere_at(c, d)[0])
        assert not visible(G, pcol, prow).any(), (c, d)
    # 5. size sorting after the burst: lighter up, heavier down
    for c, d in burst_state(T_BURST + 1.2):
        dv = vterm(d) - vterm(D0)
        assert (c[1] > 0) == (dv > 0) or abs(dv) < 0.15, (d, dv, c[1])
    # 6. bun frame: big, solid, unclipped
    p, n, st = gather(13.0)
    col, row, z = CAM.project(p)
    ok = visible(G, col, row)
    assert ok.all(), f"clipped {(~ok).sum()} samples at bun"
    width = col.max() - col.min()
    assert width >= 70, width
    grid = np.zeros((G.rows, G.cols), bool)
    grid[row, col] = True
    worst = 0
    for r in range(G.rows):
        cs = np.where(grid[r])[0]
        if len(cs) > 1:
            worst = max(worst, int(np.diff(cs).max()) - 1)
    assert worst <= 1, f"interior gap {worst}"
    # bag frame unclipped
    p, _, _ = gather(14.9)
    col, row, _ = CAM.project(p)
    assert visible(G, col, row).all(), "clipped at bag"
    # 7. counter inside the safe band
    r0, r1 = TAG_ROW, TAG_ROW + 5 * TAG_SCALE
    assert G.rows * 0.10 <= r0 and r1 <= G.rows * 0.85, (r0, r1)
    # 8. d(t) monotone through growth
    ds = [d_of(x) for x in np.linspace(0, T_BAG0, 400)]
    assert all(b - a > -1e-9 for a, b in zip(ds, ds[1:]))
    print(f"check OK: {FRAMES} frames, {T_END}s")
    print(f"  vterm: 0.5mm={vterm(0.5):.2f}  1mm={vterm(1.0):.2f}  "
          f"5mm={vterm(5.0):.2f}  6mm={vterm(6.0):.2f} m/s")
    print(f"  bun width {width} cols, worst interior gap {worst}")
    print(f"  fragments: rim {N_RIM} ({RIM_D.min():.2f}-{RIM_D.max():.2f}mm) "
          f"+ mist {N_MIST} ({MIST_D.min():.2f}-{MIST_D.max():.2f}mm), "
          f"volume {frag_v:.3f} = {DMAX**3:.0f}")

# ---------------------------------------------------------------- main
if __name__ == '__main__':
    check()
    if '--sheet' in sys.argv:
        ts = [0.0, 4.0, 8.5, 12.6, 13.79, 14.55, 14.97, 15.6, 17.0]
        frames = [draw(int(t * FPS)) for t in ts]
        contact(frames, '/tmp/raindrop_sheet.png', cols=3,
                labels=[f"t={t}s d={d_of(min(t, T_BURST-1e-3)):.1f}" for t in ts])
        print("sheet -> /tmp/raindrop_sheet.png")
        sys.exit(0)
    out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/raindrop.mp4'
    with Encoder(out, G, fps=FPS) as enc:
        for i in range(FRAMES):
            enc.write(draw(i))
            if i % 60 == 0:
                print(f"  frame {i}/{FRAMES}")
    print(f"done -> {out}")
