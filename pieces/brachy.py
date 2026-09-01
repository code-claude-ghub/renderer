#!/usr/bin/env python3
"""BRACHY — the shortest path loses the race.

Three beads slide without friction from the same start to the same
finish (1.55 m across, 1.55 m down), released together from rest:
  - the straight chord: the SHORTEST path (2.19 m). It finishes LAST.
  - a circular arc, vertical at the start (2.43 m, the longest).
  - the cycloid (2.28 m) — the brachistochrone, the answer to Johann
    Bernoulli's June 1696 challenge in Acta Eruditorum.
Times: cycloid 0.726 s, circle 0.737 s, straight 0.795 s. The straight
track loses by 69 ms because speed comes from DEPTH (v = sqrt(2gy)):
a track that dives early buys speed when it matters most, and the
cycloid is the exact optimum — every perturbed neighbour tested here
is slower (measured, see checks).

Beads are drawn three sizes so all three stay visible when they pile
up concentric at the finish; without friction, size does not change
the time. Slow-motion is an 8x replay (a 240 fps camera with a 1/480 s
shutter); ghosts strobe every 0.1 s of TRUE time on all three tracks,
so the spacing of each ghost ladder is a speed ruler — on the straight
track the ladder MUST run 1:3:5:7 (Galileo's odd-number rule), and the
check reads that ratio off the pixels.

Modelled honestly, and said in the checks: frictionless sliding beads
(a ball that ROLLS without slipping is slower by exactly sqrt(7/5) on
every track — the order cannot change), dead stop at the finish peg,
no air resistance.
"""
import os
import subprocess
import sys
import time

import numpy as np

# ---------------------------------------------------------------- canvas
W, H = 1080, 1920
FPS = 30

PAPER = 0.845                   # trap 69: warm grey, not full white
INK_TRACK = 0.38
GHOST = 0.60                    # thin rings need more contrast than
                                # plank's 0.74 board outlines (trap 67)
C_DARK = (0.10, 0.10, 0.10)     # straight bead (biggest)
C_BLUE = (0.12, 0.16, 0.52)     # circle bead
C_RED = (0.55, 0.10, 0.10)      # cycloid bead (smallest, wins)

# ---------------------------------------------------------------- physics
G_ACC = 9.81
D_RUN = 1.55                    # horizontal run, m
H_DROP = 1.55                   # vertical drop, m  (y positive DOWN)

# cycloid through A=(0,0), B=(D,H): x=a(p-sin p), y=a(1-cos p)


def _solve_phiB():
    # p = 0 is itself a (trivial) root and 1-cos(p), p-sin(p) both
    # underflow to exactly 0.0 there, so the bracket must START clear
    # of it: f(0.5) is solidly positive, f(2pi-) solidly negative.
    f = lambda p: (1 - np.cos(p)) * D_RUN - (p - np.sin(p)) * H_DROP
    lo, hi = 0.5, 2 * np.pi - 1e-9
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


PHI_B = _solve_phiB()
A_CYC = D_RUN / (PHI_B - np.sin(PHI_B))
OMEGA = np.sqrt(G_ACC / A_CYC)          # phi(t) = OMEGA * t, exactly
T_CYC = PHI_B / OMEGA

R_CIRC = (D_RUN ** 2 + H_DROP ** 2) / (2 * D_RUN)   # vertical tangent at A

L_STR = float(np.hypot(D_RUN, H_DROP))
SIN_AL = H_DROP / L_STR
T_STR = float(np.sqrt(2 * L_STR / (G_ACC * SIN_AL)))
L_CYC = 4 * A_CYC * (1 - np.cos(PHI_B / 2))
L_CIRC = R_CIRC * np.pi / 2

# circle bead: RK4 on psi (x=r(1-cos psi), y=r sin psi), from rest
DT = 2e-5


def integrate_circle(dt):
    psi, v, t = 0.0, 0.0, 0.0
    ts, ps = [0.0], [0.0]
    while psi < np.pi / 2:
        def acc(p):
            return G_ACC * np.cos(p)        # tangential: g * dy/ds
        k1v = acc(psi);                 k1p = v / R_CIRC
        k2v = acc(psi + dt / 2 * k1p);  k2p = (v + dt / 2 * k1v) / R_CIRC
        k3v = acc(psi + dt / 2 * k2p);  k3p = (v + dt / 2 * k2v) / R_CIRC
        k4v = acc(psi + dt * k3p);      k4p = (v + dt * k3v) / R_CIRC
        psi += dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        v += dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        t += dt
        ts.append(t)
        ps.append(min(psi, np.pi / 2))
    return np.array(ts), np.array(ps), t


TC_ARR, PSI_ARR, T_CIRC = integrate_circle(DT)

TIMES = {"cyc": T_CYC, "circ": T_CIRC, "str": T_STR}


def pos_cyc(t):
    p = min(OMEGA * max(t, 0.0), PHI_B)
    return A_CYC * (p - np.sin(p)), A_CYC * (1 - np.cos(p))


def pos_circ(t):
    p = np.pi / 2 if t >= T_CIRC else float(np.interp(t, TC_ARR, PSI_ARR))
    return R_CIRC * (1 - np.cos(p)), R_CIRC * np.sin(p)


def pos_str(t):
    s = min(0.5 * G_ACC * SIN_AL * max(t, 0.0) ** 2, L_STR)
    return s / L_STR * D_RUN, s / L_STR * H_DROP


BEADS = [                        # draw order: big first, small on top
    ("str", pos_str, 0.034, C_DARK),
    ("circ", pos_circ, 0.027, C_BLUE),
    ("cyc", pos_cyc, 0.021, C_RED),
]

# ---------------------------------------------------------------- layout
SCALE = 560.0                   # px per metre
AX, AY = 106, 320               # start point A, px

BX_PX = AX + D_RUN * SCALE      # finish B
BY_PX = AY + H_DROP * SCALE


def to_px(wx, wy):
    return AX + wx * SCALE, AY + wy * SCALE


OUT = f"out/brachy_{time.strftime('%H%M%S')}.mp4"

# ---------------------------------------------------------------- timeline
# spec: (true time, shutter seconds or None, ghosts-up-to or None)
SH_RT = 1.0 / 60.0              # real-time shutter
SH_SM = 1.0 / 480.0             # 240 fps camera, 180-degree shutter
# blur samples per shutter: spacing must stay ~1 px. peak speed is
# 5.51 m/s -> a 1/60 s streak is 51 px (needs 48 samples); the slow-mo
# 1/480 s streak is 6.4 px (16 is plenty).
NS_RT = 48
NS_SM = 16


def ns_for(shutter):
    return NS_RT if shutter >= SH_RT - 1e-12 else NS_SM
GHOST_STEP = 0.1
GHOST_LAST = 0.7
SM = 8                          # slow-motion factor
N_SM = 192                      # slow-mo frames (t to 191/240 > T_STR)


def build_timeline():
    spec = []
    for _ in range(36):                          # A0 hold at the start
        spec.append((0.0, None, None))
    for k in range(1, 25):                       # A1 real time, blurred
        spec.append((k / 30.0, SH_RT, None))
    for _ in range(24):                          # A2 hold, results up
        spec.append((0.85, None, None))
    for j in range(N_SM):                        # A3 slow motion 8x
        t = j / (FPS * SM)
        spec.append((t, SH_SM, t))
    for _ in range(30):                          # A4 freeze with ladders
        t = (N_SM - 1) / (FPS * SM)
        spec.append((t, None, t))
    for k in range(9):                           # A5 rewind, blurred
        spec.append((0.72 - k * 0.08, SH_RT, None))
    return spec                                  # last t=0.08, wraps to 0


TIMELINE = build_timeline()
N_FRAMES = len(TIMELINE)
IDX_A1 = 36
IDX_A2 = 60
IDX_A3 = 84
IDX_A4 = IDX_A3 + N_SM
IDX_MID = IDX_A3 + 96            # t = 0.400 s, mid race
IDX_END = IDX_A4 + 15

# ---------------------------------------------------------------- drawing


def comp_bbox(img, x0, y0, cov, color):
    h, w = cov.shape
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x0 + w, W), min(y0 + h, H)
    if x1c <= x0c or y1c <= y0c:
        return
    cv = cov[y0c - y0:y1c - y0, x0c - x0:x1c - x0]
    reg = img[y0c:y1c, x0c:x1c, :]
    col = np.asarray(color, np.float64) if np.ndim(color) else \
        np.array([color] * 3, np.float64)
    reg[...] = reg * (1 - cv[..., None]) + col[None, None, :] * cv[..., None]


def disc_cov(cx, cy, r, edge=0.5):
    x0, x1 = int(np.floor(cx - r)) - 2, int(np.ceil(cx + r)) + 3
    y0, y1 = int(np.floor(cy - r)) - 2, int(np.ceil(cy + r)) + 3
    xx = np.arange(x0, x1, dtype=np.float64)
    yy = np.arange(y0, y1, dtype=np.float64)
    d = np.hypot(xx[None, :] - cx, yy[:, None] - cy)
    return x0, y0, np.clip(r + edge - d, 0.0, 1.0)


def ring_cov(cx, cy, r, lw):
    x0, x1 = int(np.floor(cx - r - lw)) - 2, int(np.ceil(cx + r + lw)) + 3
    y0, y1 = int(np.floor(cy - r - lw)) - 2, int(np.ceil(cy + r + lw)) + 3
    xx = np.arange(x0, x1, dtype=np.float64)
    yy = np.arange(y0, y1, dtype=np.float64)
    d = np.abs(np.hypot(xx[None, :] - cx, yy[:, None] - cy) - r)
    return x0, y0, np.clip(lw / 2 + 0.5 - d, 0.0, 1.0)


def polyline_cov(pts_px, lw=3.0):
    """Coverage of a polyline (dense points, sub-pixel step)."""
    xs = np.array([p[0] for p in pts_px])
    ys = np.array([p[1] for p in pts_px])
    x0 = int(np.floor(xs.min() - lw)) - 2
    x1 = int(np.ceil(xs.max() + lw)) + 3
    y0 = int(np.floor(ys.min() - lw)) - 2
    y1 = int(np.ceil(ys.max() + lw)) + 3
    cov = np.zeros((y1 - y0, x1 - x0), np.float64)
    # dense stamps: distance field via per-point splat of a small disc
    # (points are 0.5 px apart, so max over stamps ~ tube of radius lw/2)
    xx = np.arange(-int(lw) - 2, int(lw) + 3, dtype=np.float64)
    for px, py in zip(xs, ys):
        cxi, cyi = int(round(px)), int(round(py))
        gx = cxi + xx - px
        gy = cyi + xx - py
        d = np.hypot(gx[None, :], gy[:, None])
        st = np.clip(lw / 2 + 0.5 - d, 0.0, 1.0)
        ox, oy = cxi - int(lw) - 2 - x0, cyi - int(lw) - 2 - y0
        reg = cov[oy:oy + st.shape[0], ox:ox + st.shape[1]]
        np.maximum(reg, st, out=reg)
    return x0, y0, cov


# ---------------------------------------------------------------- text
FONT = {
    "0": "01110 10001 10011 10101 11001 10001 01110",
    "1": "00100 01100 00100 00100 00100 00100 01110",
    "2": "01110 10001 00001 00010 00100 01000 11111",
    "3": "11111 00010 00100 00010 00001 10001 01110",
    "4": "00010 00110 01010 10010 11111 00010 00010",
    "5": "11111 10000 11110 00001 00001 10001 01110",
    "6": "00110 01000 10000 11110 10001 10001 01110",
    "7": "11111 00001 00010 00100 01000 01000 01000",
    "8": "01110 10001 10001 01110 10001 10001 01110",
    "9": "01110 10001 10001 01111 00001 00010 01100",
    ".": "00000 00000 00000 00000 00000 01100 01100",
    "s": "00000 00000 01111 10000 01110 00001 11110",
    "m": "00000 00000 11010 10101 10101 10101 10101",
    " ": "00000 00000 00000 00000 00000 00000 00000",
}
FSCALE = 7


def text_mask(sstr):
    rows = []
    for ri in range(7):
        line = []
        for ch in sstr:
            bits = FONT[ch].split()[ri]
            line.extend(int(b) for b in bits)
            line.append(0)
        rows.append(line[:-1])
    m = np.array(rows, np.float64)
    return np.kron(m, np.ones((FSCALE, FSCALE)))


# label strings are BUILT from the physics, then asserted in checks
LEN_STRS = {"str": f"{L_STR:.2f}m", "cyc": f"{L_CYC:.2f}m",
            "circ": f"{L_CIRC:.2f}m"}
TIME_STRS = {k: f"{v:.3f}s" for k, v in TIMES.items()}
COLORS = {"str": C_DARK, "circ": C_BLUE, "cyc": C_RED}

TX, TY0, TDY = 122, 1282, 74     # label block, bottom-left (clear of all ink)


def stamp_text(img, sstr, color, line):
    m = text_mask(sstr)
    h, w = m.shape
    y0 = TY0 + line * TDY
    reg = img[y0:y0 + h, TX:TX + w, :]
    reg[...] = reg * (1 - m[..., None]) + \
        np.array(color)[None, None, :] * m[..., None]


# ---------------------------------------------------------------- statics


def track_points(posf, T_end):
    ts = np.linspace(0, T_end, 8000)
    pts = [to_px(*posf(t)) for t in ts]
    # resample to ~0.5 px steps by arc length
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    u = np.arange(0, s[-1], 0.5)
    return list(zip(np.interp(u, s, xs), np.interp(u, s, ys)))


def background():
    fr = np.full((H, W, 3), PAPER, np.float64)
    for name, posf, _, _ in BEADS:
        x0, y0, cv = polyline_cov(track_points(posf, TIMES[name]), lw=3.0)
        comp_bbox(fr, x0, y0, cv * 1.0, INK_TRACK)
    # small finish peg: a short tick just past B along each track's exit
    fr[int(BY_PX) + 22:int(BY_PX) + 44,
       int(BX_PX) - 2:int(BX_PX) + 3, :] = INK_TRACK
    return fr


BG = background()


def scene(t, ghosts=None):
    img = BG.copy()
    if ghosts is not None:
        tg = GHOST_STEP
        while tg <= min(ghosts, GHOST_LAST) + 1e-9:
            for name, posf, r, _ in BEADS:
                gx, gy = to_px(*posf(tg))
                x0, y0, cv = ring_cov(gx, gy, r * SCALE, 1.8)
                comp_bbox(img, x0, y0, cv, GHOST)
            tg += GHOST_STEP
    for name, posf, r, col in BEADS:
        bx, by = to_px(*posf(t))
        x0, y0, cv = disc_cov(bx, by, r * SCALE)
        comp_bbox(img, x0, y0, cv, col)
    return img


def labels_for(t):
    """Label lines visible at nominal time t (uniform rule, no act gate).
    t == 0: the three path lengths, shortest first.
    t > 0: arrival times, in arrival order, each once its bead is home."""
    if t == 0.0:
        return [(COLORS[k], LEN_STRS[k])
                for k in sorted(LEN_STRS, key=lambda k: {"str": L_STR,
                                "cyc": L_CYC, "circ": L_CIRC}[k])]
    done = sorted([k for k in TIMES if t >= TIMES[k] - 1e-12],
                  key=lambda k: TIMES[k])
    return [(COLORS[k], TIME_STRS[k]) for k in done]


def frame_at(i):
    t, shutter, ghosts = TIMELINE[i]
    if shutter is None:
        img = scene(t, ghosts)
    else:
        ns = ns_for(shutter)
        acc = np.zeros((H, W, 3), np.float64)
        for j in range(ns):
            off = (2 * j + 1 - ns) / (2.0 * ns) * shutter
            acc += scene(max(t + off, 0.0), ghosts)
        img = acc / ns
    for line, (col, sstr) in enumerate(labels_for(t)):
        stamp_text(img, sstr, col, line)
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def render_frames():
    for i in range(N_FRAMES):
        yield frame_at(i)


# ---------------------------------------------------------------- measure
def mask_centroid(img, mask_fn, x_lo, x_hi, y_lo, y_hi):
    reg = img[y_lo:y_hi, x_lo:x_hi, :].astype(np.float64)
    w = mask_fn(reg)
    xs = np.arange(x_lo, x_hi, dtype=np.float64)
    ys = np.arange(y_lo, y_hi, dtype=np.float64)
    tot = w.sum()
    if tot <= 0:
        return None
    return ((w.sum(0) * xs).sum() / tot, (w.sum(1) * ys).sum() / tot)


def m_red(reg):
    return np.clip(reg[:, :, 0] - reg[:, :, 1] - 60.0, 0.0, None)


def m_blue(reg):
    return np.clip(reg[:, :, 2] - reg[:, :, 1] - 60.0, 0.0, None)


def m_dark(reg):
    return np.clip(60.0 - reg[:, :, 1], 0.0, None) * \
        (np.abs(reg[:, :, 0] - reg[:, :, 1]) < 25.0) * \
        (np.abs(reg[:, :, 2] - reg[:, :, 1]) < 25.0)


def m_ghost(reg):
    g = reg.mean(axis=2)
    return np.clip(14.0 - np.abs(g - GHOST * 255.0), 0.0, None) * \
        (np.abs(reg[:, :, 0] - reg[:, :, 1]) < 12.0)


MASKS = {"cyc": m_red, "circ": m_blue, "str": m_dark}


def bead_centroid(img, name, t, box=55):
    """Centroid of one bead by its unique colour, bounded to a box round
    the MODEL position (trap 58/64) and to rows above the label block."""
    posf = dict((n, f) for n, f, _, _ in BEADS)[name]
    mx, my = to_px(*posf(t))
    x_lo, x_hi = int(mx) - box, int(mx) + box
    y_lo, y_hi = max(int(my) - box, 0), min(int(my) + box, 1250)
    return mask_centroid(img, MASKS[name], x_lo, x_hi, y_lo, y_hi)


def shutter_mean_px(name, t, shutter):
    posf = dict((n, f) for n, f, _, _ in BEADS)[name]
    ns = ns_for(shutter)
    xs, ys = [], []
    for j in range(ns):
        off = (2 * j + 1 - ns) / (2.0 * ns) * shutter
        px, py = to_px(*posf(max(t + off, 0.0)))
        xs.append(px)
        ys.append(py)
    return float(np.mean(xs)), float(np.mean(ys))


# ---------------------------------------------------------------- checks
def travel_time_polyline(pts):
    dx = np.diff(pts[:, 0]); dy = np.diff(pts[:, 1])
    ds = np.hypot(dx, dy)
    ymid = (pts[1:, 1] + pts[:-1, 1]) / 2
    v = np.sqrt(2 * G_ACC * np.clip(ymid, 1e-12, None))
    return float((ds / v).sum())


def run_checks():
    ok = []

    def check(name, cond, detail=""):
        ok.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")

    # -- the physics facts
    ex = A_CYC * (PHI_B - np.sin(PHI_B)) - D_RUN
    ey = A_CYC * (1 - np.cos(PHI_B)) - H_DROP
    check("cycloid hits B exactly", abs(ex) < 1e-9 and abs(ey) < 1e-9,
          f"endpoint error ({ex:.1e}, {ey:.1e}) m")

    ph = np.linspace(0, PHI_B, 40001)
    cycpts = np.stack([A_CYC * (ph - np.sin(ph)),
                       A_CYC * (1 - np.cos(ph))], 1)
    T_quad = travel_time_polyline(cycpts)
    check("cycloid time: closed form vs quadrature",
          abs(T_quad - T_CYC) < 5e-5,
          f"closed {T_CYC:.6f} s, quadrature {T_quad:.6f} s")

    _, _, t_half = integrate_circle(DT / 2)
    check("circle time Richardson-stable", abs(T_CIRC - t_half) < 1e-4,
          f"dt {T_CIRC:.6f} s, dt/2 {t_half:.6f} s")
    # circle quadrature: y ~ r*psi near the vertical start gives the
    # integrand a 1/sqrt(psi) singularity, so uniform-psi sampling
    # undershoots by ~1 ms. psi = (pi/2) u^2 clusters samples there
    # and regularizes it (ds ~ u du, sqrt(y) ~ u).
    psq = (np.pi / 2) * np.linspace(0, 1, 40001) ** 2
    circpts = np.stack([R_CIRC * (1 - np.cos(psq)), R_CIRC * np.sin(psq)], 1)
    Tq_circ = travel_time_polyline(circpts)
    check("circle time: RK4 vs quadrature", abs(Tq_circ - T_CIRC) < 5e-4,
          f"RK4 {T_CIRC:.6f} s, quadrature {Tq_circ:.6f} s")

    # energy along the circle RK4 trajectory: v^2 = 2 g y everywhere
    v_arr = np.gradient(R_CIRC * PSI_ARR, TC_ARR)
    y_arr = R_CIRC * np.sin(PSI_ARR)
    e_err = np.abs(v_arr[5:-5] ** 2 - 2 * G_ACC * y_arr[5:-5]).max()
    check("energy conserved on the circle (v^2 = 2gy)", e_err < 2e-3,
          f"max |v^2 - 2gy| = {e_err:.1e} (gradient-limited)")

    # the order and the margins
    check("cycloid < circle < straight",
          T_CYC < T_CIRC < T_STR,
          f"{T_CYC * 1000:.1f} < {T_CIRC * 1000:.1f} < "
          f"{T_STR * 1000:.1f} ms")
    check("straight loses by ~69 ms",
          abs((T_STR - T_CYC) * 1000 - 69.3) < 1.0,
          f"margin {(T_STR - T_CYC) * 1000:.1f} ms")
    check("straight is the SHORTEST path (the title's claim)",
          L_STR < L_CYC < L_CIRC,
          f"{L_STR:.4f} < {L_CYC:.4f} < {L_CIRC:.4f} m")

    # variational: every perturbed neighbour of the cycloid is slower
    tx = np.gradient(cycpts[:, 0]); ty = np.gradient(cycpts[:, 1])
    nrm = np.hypot(tx, ty)
    nx, ny = -ty / nrm, tx / nrm
    worst_dt = np.inf
    for amp in (0.01, -0.01, 0.02, -0.02):
        for mode in (1, 2, 3):
            bump = amp * np.sin(mode * np.pi * ph / PHI_B)
            pert = cycpts + np.stack([nx * bump, ny * bump], 1)
            dtp = travel_time_polyline(pert) - T_quad
            worst_dt = min(worst_dt, dtp)
    check("cycloid beats every perturbed neighbour (12 tested)",
          worst_dt > 5e-4,
          f"closest challenger +{worst_dt * 1000:.2f} ms "
          f"(quadrature err {abs(T_quad - T_CYC) * 1000:.3f} ms)")

    # all three arrive at the same speed (energy, not time)
    v_end = [np.sqrt(2 * G_ACC * H_DROP)] * 3
    v_circ_end = v_arr[-3]
    check("exit speeds all sqrt(2gh) (energy is path-blind)",
          abs(v_circ_end - v_end[0]) < 0.01,
          f"{v_end[0]:.4f} m/s; circle RK4 gives {v_circ_end:.4f}")

    # Galileo's odd-number rule on the straight track (the ghost ladder)
    sg = [0.5 * G_ACC * SIN_AL * (k * GHOST_STEP) ** 2 for k in range(8)]
    gaps = np.diff(sg)
    ratios = gaps / gaps[0]
    check("straight-track ladder spacing runs 1:3:5:7 (model)",
          np.allclose(ratios, [1, 3, 5, 7, 9, 11, 13]),
          f"ratios {np.round(ratios, 6).tolist()}")

    # -- layout: every track point inside frame and safe area
    all_pts = []
    for name, posf, _, _ in BEADS:
        all_pts += track_points(posf, TIMES[name])
    xs = np.array([p[0] for p in all_pts])
    ys = np.array([p[1] for p in all_pts])
    check("tracks inside the frame with margin",
          xs.min() > 40 and xs.max() < 1040 and
          ys.min() > 200 and ys.max() < 1250,
          f"x {xs.min():.0f}..{xs.max():.0f}, y {ys.min():.0f}..{ys.max():.0f}")

    # -- instruments (trap 42: self-test on known discs first)
    probe = np.full((200, 400, 3), PAPER, np.float64)
    for (cx, cy, col) in ((77.3, 61.6, C_RED), (201.2, 101.8, C_BLUE),
                          (322.9, 141.3, C_DARK)):
        x0, y0, cv = disc_cov(cx, cy, 14.0)
        comp_bbox(probe, x0, y0, cv, col)
    p8 = (np.clip(probe, 0, 1) * 255 + 0.5).astype(np.uint8)
    got_r = mask_centroid(p8, m_red, 0, 400, 0, 200)
    got_b = mask_centroid(p8, m_blue, 0, 400, 0, 200)
    got_d = mask_centroid(p8, m_dark, 0, 400, 0, 200)
    check("three bead instruments read known discs",
          all(g is not None for g in (got_r, got_b, got_d)) and
          abs(got_r[0] - 77.3) < 0.05 and abs(got_r[1] - 61.6) < 0.05 and
          abs(got_b[0] - 201.2) < 0.05 and abs(got_b[1] - 101.8) < 0.05 and
          abs(got_d[0] - 322.9) < 0.05 and abs(got_d[1] - 141.3) < 0.05,
          f"red ({got_r[0]:.2f},{got_r[1]:.2f}) "
          f"blue ({got_b[0]:.2f},{got_b[1]:.2f}) "
          f"dark ({got_d[0]:.2f},{got_d[1]:.2f})")

    # -- pixels vs model
    f0 = frame_at(0)
    worst0 = 0.0
    for name, _, _, _ in BEADS:
        c = bead_centroid(f0, name, 0.0)
        worst0 = max(worst0, abs(c[0] - AX), abs(c[1] - AY))
    check("f0: three beads concentric at the start", worst0 < 0.5,
          f"worst offset from A = {worst0:.3f} px")

    # mid-race, slow-mo (blurred): centroid = shutter-mean position
    t_mid = TIMELINE[IDX_MID][0]
    img_mid = frame_at(IDX_MID)
    worst_m = 0.0
    for name, _, _, _ in BEADS:
        c = bead_centroid(img_mid, name, t_mid)
        mx, my = shutter_mean_px(name, t_mid, SH_SM)
        worst_m = max(worst_m, np.hypot(c[0] - mx, c[1] - my))
    check("mid-race beads at shutter-mean model positions",
          worst_m < 0.5, f"worst {worst_m:.3f} px at t={t_mid:.3f} s")

    # beads' INK is disjoint at the measured frame. masks are colour-
    # unique so boxes may overlap freely — the defect this guards is
    # OCCLUSION (a bead drawn over another shifts the hidden bead's
    # centroid). worst ink reach: r_i + r_j + slow-mo streak ~ 46 px.
    pos_mid = [to_px(*f(t_mid)) for _, f, _, _ in BEADS]
    dmin = min(np.hypot(pos_mid[i][0] - pos_mid[j][0],
                        pos_mid[i][1] - pos_mid[j][1])
               for i in range(3) for j in range(i + 1, 3))
    check("bead inks disjoint mid-race (occlusion-free centroids)",
          dmin > 60.0, f"min pairwise distance {dmin:.0f} px")

    # the finish: all three concentric at B (the bullseye)
    f_end = frame_at(IDX_END)
    worst_e = 0.0
    for name, _, _, _ in BEADS:
        c = bead_centroid(f_end, name, 1.0)
        worst_e = max(worst_e, abs(c[0] - BX_PX), abs(c[1] - BY_PX))
    check("finish: three beads concentric at B", worst_e < 0.5,
          f"worst offset from B = {worst_e:.3f} px")

    # ghost ladder on the straight track, measured off the pixels:
    # consecutive gap ratios must be Galileo's odd numbers. k=1 is
    # excluded: at t=0.1 all three tracks' ghosts are still ~17 px
    # apart, so a box around it would cross-capture (trap 58) — from
    # k=2 the nearest foreign ghost ink is > 45 px outside every box.
    img_end = f_end
    gpos = []
    for k in range(2, 8):
        tg = k * GHOST_STEP
        mx, my = to_px(*pos_str(tg))
        c = mask_centroid(img_end, m_ghost, int(mx) - 26, int(mx) + 26,
                          int(my) - 26, int(my) + 26)
        gpos.append(c)
    dists = [np.hypot(gpos[i + 1][0] - gpos[i][0],
                      gpos[i + 1][1] - gpos[i][1]) for i in range(5)]
    meas = np.array(dists) / dists[0]
    check("ghost ladder measures 5:7:9:11:13 off the pixels",
          np.allclose(meas, [1, 7 / 5, 9 / 5, 11 / 5, 13 / 5],
                      atol=0.03),
          f"gap ratios x5: {np.round(meas * 5, 2).tolist()}")

    # -- the labels
    check("length labels are the exact strings from the model",
          LEN_STRS == {"str": "2.19m", "cyc": "2.28m", "circ": "2.43m"},
          str(LEN_STRS))
    check("time labels are the exact strings from the model",
          TIME_STRS == {"cyc": "0.726s", "circ": "0.737s",
                        "str": "0.795s"},
          str(TIME_STRS))
    # arrival list order on the freeze frame: red, blue, dark
    ll = labels_for(TIMELINE[IDX_END][0])
    check("freeze shows three times in arrival order",
          [c for c, _ in ll] == [C_RED, C_BLUE, C_DARK] and
          [s for _, s in ll] == ["0.726s", "0.737s", "0.795s"],
          " / ".join(s for _, s in ll))
    # label block region holds ONLY paper before any text (trap 58)
    reg = BG[TY0 - 8:TY0 + 3 * TDY + 8, TX - 8:TX + 300, :]
    check("label block sits on clean paper",
          np.abs(reg - PAPER).max() < 1e-9)
    # the three time labels are present in their colours on the freeze
    r1 = f_end[TY0:TY0 + 49, TX:TX + 260, :].astype(int)
    r2 = f_end[TY0 + TDY:TY0 + TDY + 49, TX:TX + 260, :].astype(int)
    r3 = f_end[TY0 + 2 * TDY:TY0 + 2 * TDY + 49, TX:TX + 260, :].astype(int)
    check("freeze labels: red then blue then dark ink present",
          (r1[:, :, 0] - r1[:, :, 1] > 60).sum() > 200 and
          (r2[:, :, 2] - r2[:, :, 1] > 60).sum() > 200 and
          ((r3[:, :, 1] < 60) & (np.abs(r3[:, :, 0] - r3[:, :, 1]) < 25)
           ).sum() > 200)

    # -- frame hygiene
    g = f0.astype(np.float64).mean(axis=2) / 255.0
    ink_rows = np.where((g < 0.78).any(axis=1))[0]
    check("all ink inside the safe area (trap 56)",
          ink_rows.min() >= 192 and ink_rows.max() < 1632,
          f"rows {ink_rows.min()}..{ink_rows.max()}")
    lit = (g > 0.5).mean()
    check("frame neither blank nor solid", 0.55 < lit < 0.995,
          f"lit {lit:.3f}")

    # watch size (trap 67, numeric)
    check("smallest bead >= 7 px at 360-wide watch size",
          2 * 0.021 * SCALE / 3.0 >= 7.0,
          f"{2 * 0.021 * SCALE / 3.0:.1f} px")
    check("label glyphs >= 16 px tall at watch size",
          7 * FSCALE / 3.0 >= 16.0, f"{7 * FSCALE / 3.0:.1f} px")

    # timeline
    check("A0 frames identical (loop-stable hold)",
          np.array_equal(f0, frame_at(35)))
    check(f"{N_FRAMES} frames = {N_FRAMES / FPS:.2f} s, a Short",
          N_FRAMES == 315 and N_FRAMES / FPS <= 180.0)
    check("rewind cadence meets the loop: last t = one step above 0",
          abs(TIMELINE[-1][0] - 0.08) < 1e-12,
          f"last t {TIMELINE[-1][0]:.3f} s, wraps to 0.000")

    # blur convergence: NS_RT=48 vs 96 on the fastest real-time frame
    global NS_RT
    i_fast = IDX_A1 + 20
    a48 = frame_at(i_fast).astype(np.int64)
    NS_old, NS_RT = NS_RT, 96
    a96 = frame_at(i_fast).astype(np.int64)
    NS_RT = NS_old
    dmax = np.abs(a48 - a96).max()
    check("blur converged: NS_RT=48 vs 96", dmax <= 3,
          f"max byte diff {dmax}")

    print()
    print("NOT verified by any check above, stated per trap 68:")
    print("  - beads are frictionless SLIDERS. a ball rolling without")
    print("    slipping is slower by exactly sqrt(7/5) on every track,")
    print("    so the finishing order cannot change")
    print("  - the stop at the finish peg is modelled dead (no bounce)")
    print("  - air resistance ignored")
    print("  - bead sizes differ ONLY for visibility; without friction")
    print("    size does not enter the equation of motion")
    print()
    if not all(ok):
        print(f"{ok.count(False)} CHECK(S) FAILED")
        sys.exit(1)
    print(f"ALL {len(ok)} CHECKS PASSED")


# ---------------------------------------------------------------- encode
def encode():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-crf", "18", "-preset", "slow",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stderr=subprocess.DEVNULL)
    for fr in render_frames():                   # stream (trap 34)
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    assert p.returncode == 0, "ffmpeg failed"
    print(f"encoded {OUT} ({os.path.getsize(OUT)} bytes)")


def decode_frame(n):
    cmd = ["ffmpeg", "-i", OUT, "-vf", f"select=eq(n\\,{n})",
           "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    assert len(raw) == W * H * 3, f"decode size {len(raw)}"
    return np.frombuffer(raw, np.uint8).reshape(H, W, 3)


def check_encode():
    print("ENCODE CHECK — measured off the shipped h264:")
    worst = 0.0
    d_mid = decode_frame(IDX_MID)
    t_mid = TIMELINE[IDX_MID][0]
    for name, _, _, _ in BEADS:
        c = bead_centroid(d_mid, name, t_mid)
        mx, my = shutter_mean_px(name, t_mid, SH_SM)
        worst = max(worst, np.hypot(c[0] - mx, c[1] - my))
    print(f"    mid-race bead centroids vs model: worst {worst:.3f} px")
    assert worst < 0.8, f"centroid drift {worst}"
    d_end = decode_frame(IDX_END)
    worst_e = 0.0
    for name, _, _, _ in BEADS:
        c = bead_centroid(d_end, name, 1.0)
        worst_e = max(worst_e, abs(c[0] - BX_PX), abs(c[1] - BY_PX))
    print(f"    finish bullseye concentric off the file: "
          f"worst {worst_e:.3f} px")
    assert worst_e < 0.8
    dd = np.abs(decode_frame(0).astype(int) - frame_at(0).astype(int))
    print(f"    decoded f0 vs render: mean |diff| {dd.mean():.3f}, "
          f"max {dd.max()}")
    assert dd.mean() < 2.0, "encode mangled the frame"
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams",
         "v", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
         OUT], capture_output=True, text=True).stdout.strip()
    assert int(probe) == N_FRAMES, f"frames {probe} != {N_FRAMES}"
    print(f"    {probe} frames; the race survives the encode")


def review_stills():
    """Trap 67: look at it at the size it will be watched."""
    for name, i in [("start", 0), ("realtime", IDX_A1 + 14),
                    ("midrace", IDX_MID), ("finish", IDX_END)]:
        fr = frame_at(i)
        p = OUT.replace(".mp4", f"_{name}.png")
        tmp = p + ".raw"
        with open(tmp, "wb") as fh:
            fh.write(fr.tobytes())
        subprocess.run(
            ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-i", tmp, "-vf", "scale=360:-1", p],
            capture_output=True)
        os.remove(tmp)
        print(f"still: {p}")


if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)
    print(f"cycloid {T_CYC:.6f} s | circle {T_CIRC:.6f} s | "
          f"straight {T_STR:.6f} s  (margin {1000 * (T_STR - T_CYC):.1f} ms)")
    print(f"lengths: straight {L_STR:.4f} < cycloid {L_CYC:.4f} < "
          f"circle {L_CIRC:.4f} m")
    run_checks()
    review_stills()
    if "--ship" in sys.argv:
        encode()
        check_encode()
