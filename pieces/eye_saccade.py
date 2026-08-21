#!/usr/bin/env python3
"""YOU HAVE NEVER SEEN YOUR OWN EYES MOVE — saccades, from the mirror's side.

One giant eyeball floating in the dark, doing what eyes actually do:
fixate, dart, fixate. The darts are saccades and every one of them uses the
real main-sequence kinematics (duration = 21 + 2.2*A ms, Bahill et al. 1975)
— which at 30 fps means each one is over in one to three frames, exactly as
invisible as in life. Mid-piece, ONE saccade (28.6 deg) is played 40x slower
so the ballistic sweep can be seen at all: the movement you have made about
three times a second your whole life and never once watched yourself make.

Physics vs decision:
  physics  — saccade durations from the main sequence; fixation lengths and
             microdrift in the measured range; pupil hippus; a saccade is
             ballistic (no mid-course correction), so one smooth profile.
  decision — 40x on the featured saccade; the eyeball has no lids (the piece
             is about the movement, not the blink); palette; the stare.

Colorway: bone sclera + ice-blue iris on dark plum. (Teal/gold spent.)
"""
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
from asciilib import (Camera, Encoder, Frame, Grid, contact, depth_cue,
                      ink_lut, lambert, specular, visible, zbuffer)

G = Grid()
RAMP = ink_lut()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- palette
BG = (0.045, 0.018, 0.055)                       # dark plum
BONE = np.array([1.00, 0.96, 0.90])              # sclera
IRIS_DEEP = np.array([0.08, 0.20, 0.40])
IRIS_MID = np.array([0.30, 0.58, 0.78])          # trap 17 waypoint
IRIS_ICE = np.array([0.82, 0.97, 1.00])
PUPIL_HOT = np.array([0.92, 0.96, 1.00])         # only the catchlight gets here
TAG_RGB = (0.85, 0.93, 1.00)

FPS = 30
LAMP = np.array([-0.5, -0.55, 0.66])
LAMP = LAMP / np.linalg.norm(LAMP)

IRIS_A = math.radians(29.0)      # iris angular radius on the globe (cornea ~11.7mm/24mm)
PUP_A0 = math.radians(9.2)       # mean pupil angular radius
HIPPUS = 0.12                    # pupil wanders +-12% (physiological hippus)
HIPPUS_CYCLES = 3                # integer cycles over the loop -> seam-free

SLOW = 40.0                      # featured saccade slow-motion factor


def main_seq(A_deg):
    """Saccade duration in seconds from the main sequence: 21 + 2.2*A ms."""
    return (21.0 + 2.2 * A_deg) / 1000.0


# ---------------------------------------------------------------- schedule
# Entries: (target_deg(ax, ay), fixation_seconds_after_arrival).
# ax > 0 looks up-screen? no: rot() pitch maps +z toward -y for ax>0, and
# -y is screen-up, so ax>0 = looking up. ay>0 = looking to viewer's right.
SCAN_1 = [
    ((0.0, 0.0), 1.00),          # the stare (loop start)
    ((10.0, -7.0), 0.34),
    ((-12.0, 6.0), 0.28),
    ((3.0, 14.0), 0.42),
    ((0.0, 0.0), 0.50),          # back to you
    ((-9.0, -13.0), 0.30),
    ((14.0, 2.0), 0.26),
    ((-3.0, 8.0), 0.38),
    ((7.0, -14.0), 0.30),
    ((0.0, 0.0), 0.55),          # back to you
    ((-13.0, -3.0), 0.30),
    ((5.0, 5.0), 0.34),
]
FEAT_FROM = (-7.0, -13.0)
FEAT_TO = (5.0, 13.0)
FEAT_HOLD_IN = 0.65
FEAT_HOLD_OUT = 0.75
SCAN_2 = [
    ((-11.0, 4.0), 0.30),
    ((8.0, -8.0), 0.30),
    ((0.0, 0.0), 0.46),
    ((13.0, 9.0), 0.28),
    ((-5.0, -13.0), 0.32),
    ((0.0, 0.0), 1.40),          # final stare (loop end)
]


def build_schedule():
    """Piecewise gaze(t): list of (t0, t1, g0, g1, kind). kind: fix|sac|feat."""
    segs = []
    t = 0.0
    cur = np.array(SCAN_1[0][0])
    segs.append([t, t + SCAN_1[0][1], cur.copy(), cur.copy(), "fix"])
    t += SCAN_1[0][1]
    for tgt, fix in SCAN_1[1:]:
        tgt = np.array(tgt, float)
        A = float(np.linalg.norm(tgt - cur))
        d = main_seq(A)
        segs.append([t, t + d, cur.copy(), tgt.copy(), "sac"])
        t += d
        segs.append([t, t + fix, tgt.copy(), tgt.copy(), "fix"])
        t += fix
        cur = tgt
    # travel to the featured start
    tgt = np.array(FEAT_FROM, float)
    A = float(np.linalg.norm(tgt - cur))
    d = main_seq(A)
    segs.append([t, t + d, cur.copy(), tgt.copy(), "sac"]); t += d
    segs.append([t, t + FEAT_HOLD_IN, tgt.copy(), tgt.copy(), "fix"])
    t += FEAT_HOLD_IN
    cur = tgt
    # the featured saccade, 40x slower
    tgt = np.array(FEAT_TO, float)
    A = float(np.linalg.norm(tgt - cur))
    d = main_seq(A) * SLOW
    segs.append([t, t + d, cur.copy(), tgt.copy(), "feat"]); t += d
    segs.append([t, t + FEAT_HOLD_OUT, tgt.copy(), tgt.copy(), "fix"])
    t += FEAT_HOLD_OUT
    cur = tgt
    for tgt, fix in SCAN_2:
        tgt = np.array(tgt, float)
        A = float(np.linalg.norm(tgt - cur))
        d = main_seq(A)
        segs.append([t, t + d, cur.copy(), tgt.copy(), "sac"]); t += d
        segs.append([t, t + fix, tgt.copy(), tgt.copy(), "fix"]); t += fix
        cur = tgt
    return segs, t


SEGS, T_END = build_schedule()
FRAMES = int(round(T_END * FPS))
FEAT_SEG = next(s for s in SEGS if s[4] == "feat")
FEAT_AMP = float(np.linalg.norm(FEAT_SEG[3] - FEAT_SEG[2]))


def smoothstep(u):
    return u * u * (3.0 - 2.0 * u)


def gaze_at(t):
    """Gaze (ax, ay) in RADIANS at time t, microdrift included."""
    t = min(max(t, 0.0), T_END)
    g = None
    for t0, t1, g0, g1, kind in SEGS:
        if t <= t1 or (t0, t1, kind) == (SEGS[-1][0], SEGS[-1][1], SEGS[-1][4]):
            if kind == "fix":
                g = g0.copy()
            else:
                u = 0.0 if t1 == t0 else np.clip((t - t0) / (t1 - t0), 0, 1)
                g = g0 + (g1 - g0) * smoothstep(u)
            break
    # fixational microdrift, eased to zero at both ends for the loop seam
    amp = 0.28 * min(1.0, t / 0.8) * min(1.0, (T_END - t) / 0.8)
    dax = amp * (math.sin(2 * math.pi * 0.43 * t + 1.1)
                 + 0.5 * math.sin(2 * math.pi * 1.07 * t + 4.0))
    day = amp * (math.sin(2 * math.pi * 0.37 * t + 2.6)
                 + 0.5 * math.sin(2 * math.pi * 0.91 * t + 0.7))
    return math.radians(g[0] + dax), math.radians(g[1] + day)


def pup_a(t):
    ph = 2 * math.pi * HIPPUS_CYCLES * t / T_END
    return PUP_A0 * (1.0 + HIPPUS * math.sin(ph))


# ---------------------------------------------------------------- geometry
def build():
    rng = np.random.default_rng(11)
    n_pts = 200_000
    p = rng.normal(size=(n_pts, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    keep = p[:, 2] > -0.35                     # never rotates far enough to show
    p = p[keep]
    # jitter the sample positions slightly off the shell (trap 10)
    p = p * (1.0 + rng.normal(scale=0.0025, size=(len(p), 1)))
    nrm = p / np.linalg.norm(p, axis=1, keepdims=True)

    alpha = np.arccos(np.clip(nrm[:, 2], -1, 1))   # angle from local gaze axis
    phi = np.arctan2(nrm[:, 1], nrm[:, 0])
    rel = alpha / IRIS_A

    gain = np.ones(len(p))
    iris = alpha < IRIS_A
    # radial fibres + collarette, all in the eye's own frame so they travel
    fib = np.sin(phi * 24 + 2.1 * np.sin(phi * 7 + 1.3) + rel * 3.0)
    gain[iris] = (1.0 + 0.20 * fib[iris]
                  + 0.28 * np.exp(-((rel[iris] - 0.42) / 0.10) ** 2))
    # limbal ring: darken the outer rim of the iris
    lim = np.clip((rel - 0.80) / 0.20, 0, 1)
    gain[iris] *= (1.0 - 0.68 * smoothstep(lim[iris]))
    # fixed-seed stipple everywhere (trap 18)
    gain *= 1.0 + 0.06 * rng.normal(size=len(p))
    return p, nrm, alpha, gain, iris


PTS, NRM, ALPHA, GAIN, IRIS_M = build()


def rot_xy(p, ax, ay):
    """Pitch then yaw (matches asciilib.rot order), matrix form."""
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return p @ (Ry @ Rx).T


CAM = Camera(G).fit([PTS], margin=1.06)


# ---------------------------------------------------------------- 3x5 tag
FONT = {
    '0': ["111", "101", "101", "101", "111"],
    '4': ["101", "101", "111", "001", "001"],
    'X': ["101", "101", "010", "101", "101"],
}
TAG_ROW = 22
TAG_SCALE = 2


def stamp_tag(fr, text="40X", alpha=1.0):
    w = len(text) * 4 * TAG_SCALE - TAG_SCALE
    c0 = (G.cols - w) // 2
    rgb = tuple(a * alpha + b * (1 - alpha) for a, b in zip(TAG_RGB, BG))
    for k, ch in enumerate(text):
        pat = FONT[ch]
        for r in range(5):
            for c in range(3):
                if pat[r][c] == '1':
                    for dr in range(TAG_SCALE):
                        for dc in range(TAG_SCALE):
                            fr.put(c0 + (k * 4 + c) * TAG_SCALE + dc,
                                   TAG_ROW + r * TAG_SCALE + dr, '@', rgb)


# ---------------------------------------------------------------- draw
def colour(s, m):
    if m > 1.5:                                   # pupil: void, unless catchlight
        w = np.clip(s, 0, 1) ** 2
        return tuple(BG[i] * (1 - w) + PUPIL_HOT[i] * w for i in range(3))
    if m > 0.5:                                   # iris: two-segment ice ramp
        if s < 0.5:
            c = IRIS_DEEP + (IRIS_MID - IRIS_DEEP) * (s * 2)
        else:
            c = IRIS_MID + (IRIS_ICE - IRIS_MID) * ((s - 0.5) * 2)
        return tuple(c)
    # sclera: bone, floored so the glyph carries the light (trap 12)
    w = 0.25 + 0.75 * np.clip(s, 0, 1)
    return tuple(BG[i] * (1 - w) + BONE[i] * w for i in range(3))


def pose_arrays(t):
    ax, ay = gaze_at(t)
    p = rot_xy(PTS, ax, ay)
    n = rot_xy(NRM, ax, ay)
    ok = n[:, 2] > 0.03                            # front-face cull
    p, n = p[ok], n[ok]
    a, g_ = ALPHA[ok], GAIN[ok]
    ir = IRIS_M[ok]
    pu = a < pup_a(t)

    lam = lambert(n, LAMP)
    sp_b = specular(n, LAMP, 10)
    sp_t = specular(n, LAMP, 90)
    shade = (0.13 + 0.87 * lam) * g_
    shade[pu] *= 0.05                              # the hole
    shade += 0.28 * sp_b * (~pu)                   # wet sheen, not in the hole
    shade += 1.00 * sp_t                           # catchlight rides over all
    mat = np.zeros(len(p))
    mat[ir] = 1.0
    mat[pu] = 2.0
    return p, shade, mat


def sub_times(i):
    """Sub-frame sample times: >1 only when the gaze moves fast (real darts)."""
    t0, t1 = i / FPS, (i + 1) / FPS
    a0 = np.degrees(gaze_at(t0))
    a1 = np.degrees(gaze_at(min(t1, T_END)))
    if np.hypot(a1[0] - a0[0], a1[1] - a0[1]) > 1.0:
        return [t0 + (t1 - t0) * k / 6.0 for k in range(7)]
    return [t0]


def draw(i):
    ts = sub_times(i)
    ps, ss, ms = [], [], []
    for t in ts:
        p, s, m = pose_arrays(t)
        ps.append(p); ss.append(s); ms.append(m)
    p = np.concatenate(ps); s = np.concatenate(ss); m = np.concatenate(ms)
    col, row, z = CAM.project(p)
    ok = visible(G, col, row)
    col, row, z, s, m = col[ok], row[ok], z[ok], s[ok], m[ok]
    s = s * depth_cue(z, far=0.90)
    _, keep = zbuffer(G, col, row, z)
    fr = Frame(G, BG)
    fr.field(col, row, keep, s, colour, RAMP, extra=m)
    t = i / FPS
    f0, f1 = FEAT_SEG[0], FEAT_SEG[1]
    if f0 - 0.2 <= t <= f1 + 0.3:
        a = min(1.0, (t - (f0 - 0.2)) / 0.25, ((f1 + 0.3) - t) / 0.25)
        stamp_tag(fr, "40X", alpha=max(0.0, a))
    return fr


# ---------------------------------------------------------------- checks
def check():
    print(f"T_END {T_END:.3f}s  FRAMES {FRAMES}  featured amp {FEAT_AMP:.1f} deg")
    # 1. every ordinary saccade obeys the main sequence and is over in <3 frames
    for t0, t1, g0, g1, kind in SEGS:
        if kind == "sac":
            A = float(np.linalg.norm(g1 - g0))
            assert abs((t1 - t0) - main_seq(A)) < 1e-9
            assert (t1 - t0) <= 0.095, f"slow ordinary saccade {t1-t0:.3f}s"
    # 2. the featured one is exactly the main sequence times SLOW
    d_feat = FEAT_SEG[1] - FEAT_SEG[0]
    assert abs(d_feat - main_seq(FEAT_AMP) * SLOW) < 1e-9
    print(f"featured saccade: real {main_seq(FEAT_AMP)*1000:.0f} ms "
          f"-> {d_feat:.2f} s at {SLOW:.0f}x")
    # 3. gaze bounds stay modest (silhouette never clips, cull cap never shows)
    gz = np.array([gaze_at(t) for t in np.linspace(0, T_END, 400)])
    assert np.degrees(np.abs(gz)).max() < 27.0
    # 4. loop seam: inputs identical at t=0 and t=T_END
    assert gaze_at(0.0) == gaze_at(T_END)
    assert abs(pup_a(0.0) - pup_a(T_END)) < 1e-12
    # 5. tag inside the safe band
    assert TAG_ROW >= int(G.rows * 0.10) and TAG_ROW + 10 <= int(G.rows * 0.85)
    # 6. geometry: the ball is BIG, in frame, and solid
    for fi in [0, FRAMES // 3, int((FEAT_SEG[0] + 0.5) * FPS), FRAMES - 1]:
        p, s, m = pose_arrays(fi / FPS)
        col, row, z = CAM.project(p)
        ok = visible(G, col, row)
        assert ok.all(), f"frame {fi}: {np.sum(~ok)} samples clipped"
        c, r = col[ok], row[ok]
        dia = c.max() - c.min()
        worst = 0
        for rr in range(r.min(), r.max() + 1, 6):
            cc = np.unique(c[r == rr])
            if len(cc) > 2:
                worst = max(worst, int(np.diff(cc).max()) - 1)
        print(f"frame {fi:4d}: cols {c.min()}..{c.max()} rows {r.min()}.."
              f"{r.max()} dia {dia} worst-gap {worst}")
        assert dia >= 84, "eye not big enough"
        assert worst <= 1, "holes in the ball"
    # 7. print the gaze-speed table around the featured saccade
    print(" t      ax      ay     deg/s")
    for t in np.arange(FEAT_SEG[0] - 0.3, FEAT_SEG[1] + 0.3, 0.35):
        a0 = np.degrees(gaze_at(t)); a1 = np.degrees(gaze_at(t + 0.01))
        v = np.hypot(a1[0] - a0[0], a1[1] - a0[1]) / 0.01
        print(f"{t:5.2f}  {a0[0]:6.2f}  {a0[1]:6.2f}  {v:8.1f}")
    print("check: all good")


def stills():
    idx = [0, int(2.2 * FPS), int(4.9 * FPS),
           int((FEAT_SEG[0] + d_mid()) * FPS),
           int((FEAT_SEG[1] + 1.2) * FPS), FRAMES - 1]
    labels = ["stare t0", "scan", "scan 2", "featured 40x", "after", "last"]
    contact([draw(i) for i in idx], os.path.join(OUT, "eye_sheet.png"),
            cols=3, labels=labels)
    print("sheet: out/eye_sheet.png")


def d_mid():
    return (FEAT_SEG[1] - FEAT_SEG[0]) / 2


def render():
    path = os.path.join(OUT, "eye_saccade.mp4")
    with Encoder(path, G, fps=FPS) as enc:
        for i in range(FRAMES):
            enc.write(draw(i))
            if i % 60 == 0:
                print(f"  frame {i}/{FRAMES}")
    print("wrote", path)


if __name__ == "__main__":
    check()
    if "--stills" in sys.argv:
        stills()
    elif "--render" in sys.argv:
        render()
