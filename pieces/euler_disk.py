#!/usr/bin/env python3
"""
THE DEATH OF A SPUN COIN — Euler's disk.

A disk settles on a table. As the tilt angle falls, the rolling constraint
Omega^2 * sin(alpha) = 4g/a forces the wobble rate UP — with no ceiling.
The whirr rises until the disk snaps flat, and then there is nothing.

The render obeys the constraint relation exactly (the undisputed physics).
The time-law alpha ~ (t0 - t)^(2/3) is one of the published rolling-friction
exponents — the literature disputes the exponent because it disputes the
dissipation mechanism (Moffatt 2000: air film; Caps et al. 2004: slipping
friction). The dispute lives in the description, not the frame.

The face marking rotates slowly at Omega*(1 - cos(alpha)) — the famous
near-stationary image on the toy — and freezes as the wobble diverges.

Counter: the disk's actual wobble rate in Hz, integrated per frame.
It vanishes when the disk stops. Silence, drawn.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# Works from scripts/ (asciilib alongside) and from
# the public repo, where pieces live in pieces/. Insert both.
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
from asciilib import (Camera, Encoder, Frame, Grid, contact, depth_cue,
                      ink_lut, lambert, specular, visible, zbuffer)

G = Grid()
RAMP = ink_lut()

BG = (0.013, 0.075, 0.090)              # deep ink-teal
BRONZE = np.array([0.55, 0.36, 0.16])   # unlit metal
AMBER = np.array([0.88, 0.62, 0.28])    # waypoint (trap 17: no grey middle)
GOLD = np.array([1.00, 0.93, 0.66])     # lit metal
CNT_RGB = (0.95, 0.92, 0.80)

FPS = 30
T_SNAP = 16.0                            # the whirr ends here
T_END = 17.8                             # then stillness
FRAMES = int(T_END * FPS)                # 534

ALPHA0 = np.deg2rad(50.0)                # starting tilt
ALPHA_MIN = np.deg2rad(1.0)              # below this: snap flat
F0 = 0.75                                # starting wobble rate, Hz
OMEGA0 = 2 * np.pi * F0
FOURG_A = OMEGA0**2 * np.sin(ALPHA0)     # the constraint constant 4g/a

H = 0.055                                # half thickness (radius = 1)
DOME = 0.10                              # face relief so the face can shade

LAMP = np.array([-0.45, -0.55, 0.70])
LAMP /= np.linalg.norm(LAMP)
ELEV = 0.42                              # camera elevation, rad


# ---------------------------------------------------------------- kinematics
def alpha_of(t):
    """Tilt angle. Rolling-friction settling law, snap below ALPHA_MIN."""
    if t >= T_SNAP:
        return 0.0
    a = ALPHA0 * (1.0 - t / T_SNAP) ** (2.0 / 3.0)
    if a < ALPHA_MIN:
        # the last shudder: ride the law down to zero over its final sliver
        return a
    return a


def precompute():
    """Per-frame alpha, precession phase, face-spin phase, frequency."""
    dt = 1.0 / FPS
    al = np.zeros(FRAMES)
    ph = np.zeros(FRAMES)
    ps = np.zeros(FRAMES)
    fq = np.zeros(FRAMES)
    phi = psi = 0.0
    for i in range(FRAMES):
        t = i * dt
        a = alpha_of(t)
        al[i] = a
        if a > 1e-6:
            om = np.sqrt(FOURG_A / np.sin(max(a, ALPHA_MIN * 0.35)))
            fq[i] = om / (2 * np.pi)
            phi += om * dt
            psi += om * (1.0 - np.cos(a)) * dt
        ph[i], ps[i] = phi, psi
    return al, ph, ps, fq


AL, PH, PS, FQ = precompute()


# ------------------------------------------------------------------ geometry
def build():
    rng = np.random.default_rng(7)
    n_face, n_rim = 53000, 14000

    def face(sign):
        r = np.sqrt(rng.uniform(0, 1, n_face))
        th = rng.uniform(0, 2 * np.pi, n_face)
        u, v = r * np.cos(th), r * np.sin(th)
        w = sign * (H + DOME * (1 - r**2))
        p = np.stack([u, v, w], axis=1)
        n = np.stack([2 * DOME * u, 2 * DOME * v,
                      np.full(n_face, float(sign))], axis=1)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        return p, n, r, th

    pt, nt, rt, tht = face(+1)
    pb, nb, rb, thb = face(-1)

    th = rng.uniform(0, 2 * np.pi, n_rim)
    w = rng.uniform(-H, H, n_rim)
    pr = np.stack([np.cos(th), np.sin(th), w], axis=1)
    nr = np.stack([np.cos(th), np.sin(th), np.zeros(n_rim)], axis=1)

    pts = np.concatenate([pt, pb, pr])
    nrm = np.concatenate([nt, nb, nr])
    pts += rng.normal(0, 0.004, pts.shape)          # trap 10: jitter the lattice

    # material gain: engraved ring dim, sector mark bright, rim slightly dim
    gain = np.ones(len(pts))
    r_all = np.concatenate([rt, rb, np.ones(n_rim)])
    th_all = np.concatenate([tht, thb, th])
    ring = (r_all > 0.75) & (r_all < 0.84)
    gain[ring] = 0.68
    is_face = np.arange(len(pts)) < 2 * n_face
    wedge = is_face & (r_all > 0.28) & (r_all < 0.70) \
        & (np.mod(th_all, 2 * np.pi) < 0.55)
    gain[wedge] = 1.22
    gain[2 * n_face:] = 0.92                         # rim

    stip = 1.0 + 0.08 * rng.uniform(-1, 1, len(pts))  # trap 18, fixed seed
    return pts, nrm, gain * stip


PTS, NRM, GAIN = build()


def body(i):
    """World pose at frame i -> view-space points and normals."""
    a, phi, psi = AL[i], PH[i], PS[i]
    ca, sa = np.cos(a), np.sin(a)
    cf, sf = np.cos(phi), np.sin(phi)

    # face spin about the disk axis (the slow image rotation)
    cp, sp = np.cos(psi), np.sin(psi)
    u = PTS[:, 0] * cp - PTS[:, 1] * sp
    v = PTS[:, 0] * sp + PTS[:, 1] * cp
    w = PTS[:, 2]
    nu = NRM[:, 0] * cp - NRM[:, 1] * sp
    nv = NRM[:, 0] * sp + NRM[:, 1] * cp
    nw = NRM[:, 2]

    # disk frame -> world (z up). n_d = axis, e1/e2 span the face plane.
    e1 = np.array([ca * cf, ca * sf, -sa])
    e2 = np.array([-sf, cf, 0.0])
    nd = np.array([sa * cf, sa * sf, ca])
    zc = max(sa, H + DOME)                           # center rides down
    p = (u[:, None] * e1 + v[:, None] * e2 + w[:, None] * nd)
    p[:, 2] += zc
    n = (nu[:, None] * e1 + nv[:, None] * e2 + nw[:, None] * nd)

    # world -> view: camera toward +y, elevated ELEV, orthographic
    ce, se = np.cos(ELEV), np.sin(ELEV)
    vx = p[:, 0]
    vy = p[:, 1] * se - p[:, 2] * ce                 # screen-down
    vz = p[:, 1] * ce + p[:, 2] * se                 # toward viewer
    nx = n[:, 0]
    ny = n[:, 1] * se - n[:, 2] * ce
    nz = n[:, 1] * ce + n[:, 2] * se
    pv = np.stack([vx, vy, vz], axis=1)
    nv_ = np.stack([nx, ny, nz], axis=1)
    return pv, nv_


# fit over the whole flight: tall tilted start, flat end, and the between
CAM = Camera(G).fit([body(i)[0] for i in
                     [0, 90, 200, 320, 420, 470, 478, FRAMES - 1]],
                    margin=1.05)


# ------------------------------------------------------------------- counter
FONT = {
    '0': ["111", "101", "101", "101", "111"],
    '1': ["010", "110", "010", "010", "111"],
    '2': ["111", "001", "111", "100", "111"],
    '3': ["111", "001", "111", "001", "111"],
    '4': ["101", "101", "111", "001", "001"],
    '5': ["111", "100", "111", "001", "111"],
    '6': ["111", "100", "111", "101", "111"],
    '7': ["111", "001", "001", "010", "010"],
    '8': ["111", "101", "111", "101", "111"],
    '9': ["111", "101", "111", "001", "111"],
    '.': ["000", "000", "000", "000", "010"],
    'H': ["101", "101", "111", "101", "101"],
    'Z': ["111", "001", "010", "100", "111"],
    ' ': ["000", "000", "000", "000", "000"],
}
CNT_ROW = 21
CNT_SCALE = 2


def stamp_counter(fr, text):
    wch = 3 * CNT_SCALE + CNT_SCALE
    total = len(text) * wch - CNT_SCALE
    c0 = (G.cols - total) // 2
    for k, ch in enumerate(text):
        glyph = FONT[ch]
        for gr in range(5):
            for gc in range(3):
                if glyph[gr][gc] == '1':
                    for dr in range(CNT_SCALE):
                        for dc in range(CNT_SCALE):
                            fr.put(c0 + k * wch + gc * CNT_SCALE + dc,
                                   CNT_ROW + gr * CNT_SCALE + dr,
                                   '@', CNT_RGB)


def colour(s, _):
    s = min(1.0, s)
    if s < 0.5:
        c = BRONZE + (AMBER - BRONZE) * (s * 2.0)
    else:
        c = AMBER + (GOLD - AMBER) * ((s - 0.5) * 2.0)
    return (c[0], c[1], c[2])


# -------------------------------------------------------------------- render
def draw(i):
    pv, nv_ = body(i)
    keep_face = nv_[:, 2] > -0.05
    pv, nv_ = pv[keep_face], nv_[keep_face]
    g = GAIN[keep_face]

    col, row, z = CAM.project(pv)
    ok = visible(G, col, row)
    col, row, z, nv_, g = col[ok], row[ok], z[ok], nv_[ok], g[ok]
    _, keep = zbuffer(G, col, row, z)

    shade = (0.26 + 0.88 * lambert(nv_, LAMP)
             + 0.40 * specular(nv_, LAMP, 18)) * depth_cue(z, far=0.96) * g

    fr = Frame(G, BG)
    fr.field(col, row, keep, shade, colour, RAMP)
    t = i / FPS
    if t < T_SNAP and FQ[i] > 0:
        stamp_counter(fr, f"{FQ[i]:.1f} HZ")
    return fr


# --------------------------------------------------------------------- check
def check():
    dt = 1.0 / FPS
    print("t      alpha°   f(Hz)   zc")
    for t in np.arange(0, T_END, 2.0):
        i = min(int(t * FPS), FRAMES - 1)
        print(f"{t:5.1f}  {np.degrees(AL[i]):6.2f}  {FQ[i]:6.2f}"
              f"  {max(np.sin(AL[i]), H + DOME):5.3f}")
    i_last = int(T_SNAP * FPS) - 1
    print(f"\nwobble rate: {FQ[0]:.2f} -> {FQ[i_last]:.2f} Hz"
          f"  (x{FQ[i_last]/FQ[0]:.1f})")
    assert FQ[i_last] / FQ[0] > 5.0, "the rise IS the piece"
    live = FQ[:i_last + 1]
    assert np.all(np.diff(live) > -1e-9), "frequency must only rise"
    assert 30.0 / FQ.max() >= 5.0, f"aliasing: {30/FQ.max():.1f} frames/cycle"

    # counter inside safe band
    assert CNT_ROW >= int(G.rows * 0.10) + 1
    assert CNT_ROW + 5 * CNT_SCALE < int(G.rows * 0.85)

    # frame bounds + solidity on extreme poses
    for i in [0, 240, 470, FRAMES - 1]:
        pv, nv_ = body(i)
        m = nv_[:, 2] > -0.05
        col, row, z = CAM.project(pv[m])
        ok = visible(G, col, row)
        assert ok.all() or (col.min() >= 0 and col.max() < G.cols
                            and row.min() >= 0 and row.max() < G.rows), \
            f"clipped at frame {i}"
        # interior pinholes (convex silhouette)
        filled = np.zeros((G.rows, G.cols), bool)
        filled[row[ok], col[ok]] = True
        worst = 0
        for r in range(G.rows):
            cs = np.where(filled[r])[0]
            if len(cs) > 2:
                worst = max(worst, int(np.max(np.diff(cs))) - 1)
        print(f"frame {i:3d}: cols {col.min()}..{col.max()}"
              f" rows {row.min()}..{row.max()} worst gap {worst}")
        assert worst <= 1, f"holes in the body at frame {i}"
    print("check OK")


def stills():
    os.makedirs("out", exist_ok=True)
    idx = [0, 120, 300, 420, 475, FRAMES - 1]
    sheet = contact([draw(i) for i in idx], "out/euler_sheet.png", cols=3,
                    labels=[f"f{i} t={i/FPS:.1f}s" for i in idx])
    print("sheet:", sheet)


def render():
    out = "out/euler_disk.mp4"
    os.makedirs("out", exist_ok=True)
    with Encoder(out, G, fps=FPS) as enc:
        for i in range(FRAMES):
            enc.write(draw(i))
            if i % 60 == 0:
                print(f"  frame {i}/{FRAMES}")
    print("wrote", out)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "check":
        check()
    elif mode == "stills":
        check()
        stills()
    else:
        check()
        render()
