#!/usr/bin/env python3
"""WHEEL — the bottom of a rolling wheel is standing still.

A wheel rolls in from the left with an honest 1/60 shutter. At t=2.4 s a
photograph fires: one 1/8-second exposure, rendered by averaging 96 true
positions of the wheel, held on screen with a slow push-in.

The claim is an exact identity: rolling without slipping puts the
instantaneous centre of rotation at the contact point, so every material
point's speed is omega * (its distance from the contact point) — zero at
the bottom, v at the hub, 2v at the top. The photograph proves it about
itself: twelve sidewall dots come out as streaks, and each streak's
length, measured off the shipped h264, must equal omega * d * T_exposure.
The bottom dot barely smears; the top dot is a ~94 px streak.

Born in the comments: after TRAIN, @rorucopexperements pointed out that a
car wheel, unlike a train wheel, never moves backward — nothing on it
hangs below the road. Right. This is what a car wheel does instead.
"""
import os
import subprocess
import sys
import time

import numpy as np

# ---------------------------------------------------------------- geometry
W, H = 1080, 1920
FPS = 30
Y_ROAD = 1400.0                 # road surface row (the contact line)
R_TIRE = 400.0                  # tire outer radius = rolling radius, px
R_TIRE_IN = 280.0               # tire inner radius (meets the rim)
R_RIM = 280.0                   # rim outer
R_RIM_IN = 258.0                # rim inner
# NOTE: no tread ticks — the wheel TRANSLATES ~50 px during the photo
# exposure, so any bright feature radially adjacent to the dot ring smears
# horizontally into the streak boxes and no radial gap can prevent it.
R_DOTS = 352.0                  # sidewall dot ring radius
R_DOT = 13.0                    # dot radius
N_DOTS = 12
R_SPOKE_IN = 70.0
R_SPOKE_OUT = 262.0
N_SPOKES = 8
W_SPOKE = 26.0
HUB_Y = Y_ROAD - R_TIRE         # 1000

V = 400.0                       # ground speed px/s
OMEGA = V / R_TIRE              # 1.0 rad/s exactly

T_FREEZE = 2.4                  # the photograph fires here
X_FREEZE = 540.0                # hub x at the photograph (frame centre)
HUB_X0 = X_FREEZE - V * T_FREEZE            # -420: fully off-frame left
PSI0 = -OMEGA * T_FREEZE        # dot 0 / tick 0 exactly at the bottom at freeze

N_A = 72                        # rolling frames (2.4 s)
N_H = 84                        # photograph hold (2.8 s)
N_FRAMES = N_A + N_H            # 156 -> 5.2 s

T_SHUT_A = 1.0 / 60.0           # rolling-footage shutter
SUB_A = 6
T_PHOTO = 0.125                 # the photograph's exposure
SUB_P = 96

ZOOM_MAX = 1.05                 # push-in over the hold
BORDER_IN = 26                  # photo border inset, px
BORDER_TH = 4

# ---------------------------------------------------------------- palette (0..1 floats, trap 55)
BG      = np.array([0.030, 0.035, 0.052])
C_ROAD  = np.array([0.095, 0.096, 0.100])
C_EDGE  = np.array([0.310, 0.315, 0.330])   # bright road top edge
C_STONE = np.array([0.190, 0.188, 0.185])
C_TIRE  = np.array([0.160, 0.160, 0.168])
C_RIMC  = np.array([0.720, 0.725, 0.740])
C_SPOKE = np.array([0.600, 0.605, 0.620])
C_HUB   = np.array([0.340, 0.345, 0.360])
C_CAP   = np.array([0.750, 0.755, 0.770])
C_AXLE  = np.array([0.100, 0.102, 0.110])
C_DOT   = np.array([0.960, 0.945, 0.900])   # the twelve marks
C_BORD  = np.array([0.920, 0.920, 0.920])
SHADOW_A = 0.45

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out",
                   time.strftime("wheel_%H%M%S.mp4"))
OUT = os.path.abspath(OUT)


# ---------------------------------------------------------------- kinematics
def hub_x(t):
    return HUB_X0 + V * t


def theta(t):
    return OMEGA * t


def point(t, r, psi):
    """Material point at radius r, phase psi (psi+theta=0 is the bottom)."""
    a = theta(t) + psi
    return hub_x(t) - r * np.sin(a), HUB_Y + r * np.cos(a)


def point_vel(t, r, psi):
    a = theta(t) + psi
    return V - r * OMEGA * np.cos(a), -r * OMEGA * np.sin(a)


def contact(t):
    return hub_x(t), Y_ROAD


PHOTO_T = (T_FREEZE - T_PHOTO / 2.0
           + (np.arange(SUB_P) + 0.5) / SUB_P * T_PHOTO)


def dot_model():
    """Per dot: sample positions over the exposure, mean distance from the
    instantaneous contact point, path length, pixel extents."""
    out = []
    cx = hub_x(PHOTO_T)
    for i in range(N_DOTS):
        psi = PSI0 + i * 2.0 * np.pi / N_DOTS
        xs, ys = point(PHOTO_T, R_DOTS, psi)
        d = np.hypot(xs - cx, ys - Y_ROAD)
        seg = np.hypot(np.diff(xs), np.diff(ys)).sum()
        out.append(dict(
            i=i, psi=psi, xs=xs, ys=ys, d_mean=float(d.mean()),
            path=float(seg),
            ext_x=float(xs.max() - xs.min() + 2 * R_DOT),
            ext_y=float(ys.max() - ys.min() + 2 * R_DOT),
            box=(xs.min() - R_DOT - 5, xs.max() + R_DOT + 5,
                 ys.min() - R_DOT - 5, ys.max() + R_DOT + 5)))
    return out


DOTS = dot_model()

LUM_TIRE = float(C_TIRE.mean())
LUM_DOT = float(C_DOT.mean())
THR = LUM_TIRE + 0.055          # streak-pixel threshold (luminance)


def dilate(m, r):
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


def predict_thresholded_extent(dm):
    """What the pixels will actually measure: the averaged disc coverage
    tapers at the streak tips, so the thresholded extent is shorter than the
    geometric one. Independent 1-pixel-grid computation of
    lum = LUM_TIRE + (LUM_DOT - LUM_TIRE) * mean_t(coverage_t).
    Also returns the predicted bright MASK: the measurement region must be
    the streak's own shape, not a rectangle — a rectangle's corners dip into
    the smeared rim's annulus (the wheel translates ~50 px during the
    exposure, so every bright ring is wider than it is drawn)."""
    x0, x1, y0, y1 = [int(v) for v in dm["box"]]
    gy, gx = np.mgrid[y0:y1, x0:x1].astype(float)
    acc = np.zeros_like(gx)
    for xs, ys in zip(dm["xs"], dm["ys"]):
        acc += np.clip(R_DOT + 0.6 - np.hypot(gx - xs, gy - ys),
                       0.0, 1.2) / 1.2
    lum = LUM_TIRE + (LUM_DOT - LUM_TIRE) * acc / len(dm["xs"])
    m = lum > THR
    cols = np.where(m.any(axis=0))[0]
    rows = np.where(m.any(axis=1))[0]
    ext = ((float(cols.max() - cols.min() + 1),
            float(rows.max() - rows.min() + 1))
           if len(cols) else (0.0, 0.0))
    return ext[0], ext[1], dilate(m, 3)


for _dm in DOTS:
    _dm["pext_x"], _dm["pext_y"], _dm["pmask"] = \
        predict_thresholded_extent(_dm)


# ---------------------------------------------------------------- rasterisers
def over(img, alpha, col):
    img *= (1.0 - alpha)[..., None]
    img += col * alpha[..., None]


def aa(d):
    """Signed 'inside-ness' -> AA coverage over ~1.2 px."""
    return np.clip(d + 0.6, 0.0, 1.2) / 1.2


def capsule_local(win, xx, yy, x0, y0, x1, y1, width, col):
    """AA capsule drawn into a window, with its own tight bbox."""
    r = width / 2.0
    lo_x = int(max(0, np.floor(min(x0, x1) - r - 2 - xx[0, 0])))
    hi_x = int(min(win.shape[1], np.ceil(max(x0, x1) + r + 2 - xx[0, 0])))
    lo_y = int(max(0, np.floor(min(y0, y1) - r - 2 - yy[0, 0])))
    hi_y = int(min(win.shape[0], np.ceil(max(y0, y1) + r + 2 - yy[0, 0])))
    if lo_x >= hi_x or lo_y >= hi_y:
        return
    sx = xx[lo_y:hi_y, lo_x:hi_x]
    sy = yy[lo_y:hi_y, lo_x:hi_x]
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        d = np.hypot(sx - x0, sy - y0)
    else:
        u = np.clip(((sx - x0) * dx + (sy - y0) * dy) / L2, 0.0, 1.0)
        d = np.hypot(sx - (x0 + u * dx), sy - (y0 + u * dy))
    s = aa(r - d)
    sub = win[lo_y:hi_y, lo_x:hi_x]
    sub *= (1.0 - s)[..., None]
    sub += col * s[..., None]


# ---------------------------------------------------------------- layers
def static_layer():
    img = np.empty((H, W, 3))
    img[:] = BG
    yy = np.arange(H, dtype=float)[:, None]
    road = aa(yy - Y_ROAD) * np.ones((1, W))
    over(img, road, C_ROAD)
    edge = np.clip(1.6 - np.abs(yy - (Y_ROAD + 1.6)), 0, 1) * np.ones((1, W))
    over(img, edge, C_EDGE)
    # stones: the ground is sharp in every photograph (static, seeded)
    rng = np.random.default_rng(7)
    gy, gx = np.mgrid[int(Y_ROAD) + 8:int(Y_ROAD) + 190, 0:W].astype(float)
    for _ in range(240):
        sx = rng.uniform(0, W)
        sy = rng.uniform(Y_ROAD + 10, Y_ROAD + 185)
        sr = rng.uniform(1.0, 2.8)
        b = rng.uniform(0.55, 1.0)
        m = aa(sr - np.hypot(gx - sx, gy - sy))
        sub = img[int(Y_ROAD) + 8:int(Y_ROAD) + 190]
        sub *= (1.0 - m * b * 0.5)[..., None]
        sub += C_STONE * (m * b * 0.5)[..., None]
    return img


def draw_wheel_at(win, xx, yy, t):
    """One instantaneous wheel, composited over the window."""
    hx = hub_x(t)
    th = theta(t)
    # contact shadow
    sd = np.hypot((xx - hx) / 150.0, (yy - (Y_ROAD + 9.0)) / 13.0)
    win *= (1.0 - np.clip(1.0 - sd, 0, 1) * SHADOW_A)[..., None]
    d = np.hypot(xx - hx, yy - HUB_Y)
    # tire
    over(win, aa(np.minimum(d - R_TIRE_IN, R_TIRE - d)), C_TIRE)
    # rim
    over(win, aa(np.minimum(d - R_RIM_IN, R_RIM - d)), C_RIMC)
    # spokes
    for k in range(N_SPOKES):
        a = th + k * 2.0 * np.pi / N_SPOKES
        s, c = np.sin(a), np.cos(a)
        capsule_local(win, xx, yy,
                      hx - R_SPOKE_IN * s, HUB_Y + R_SPOKE_IN * c,
                      hx - R_SPOKE_OUT * s, HUB_Y + R_SPOKE_OUT * c,
                      W_SPOKE, C_SPOKE)
    # hub
    over(win, aa(76.0 - d), C_HUB)
    over(win, aa(22.0 - d), C_CAP)
    over(win, aa(7.0 - d), C_AXLE)
    # the twelve sidewall dots
    for i in range(N_DOTS):
        a = th + PSI0 + i * 2.0 * np.pi / N_DOTS
        capsule_local(win, xx, yy,
                      hx - R_DOTS * np.sin(a), HUB_Y + R_DOTS * np.cos(a),
                      hx - R_DOTS * np.sin(a), HUB_Y + R_DOTS * np.cos(a),
                      2 * R_DOT, C_DOT)


STATIC = static_layer()


def exposure(t0, t_shut, sub):
    """Full frame = STATIC with the wheel time-averaged over the shutter."""
    ts = t0 + (np.arange(sub) + 0.5) / sub * t_shut
    x_lo = hub_x(ts[0]) - R_TIRE - 40
    x_hi = hub_x(ts[-1]) + R_TIRE + 40
    lo_x = int(max(0, np.floor(x_lo)))
    hi_x = int(min(W, np.ceil(x_hi)))
    lo_y = int(HUB_Y - R_TIRE - 24)
    hi_y = int(min(H, Y_ROAD + 40))
    img = STATIC.copy()
    if lo_x >= hi_x:
        return img
    yy, xx = np.mgrid[lo_y:hi_y, lo_x:hi_x].astype(float)
    acc = np.zeros((hi_y - lo_y, hi_x - lo_x, 3))
    base = STATIC[lo_y:hi_y, lo_x:hi_x]
    for t in ts:
        win = base.copy()
        draw_wheel_at(win, xx, yy, t)
        acc += win
    img[lo_y:hi_y, lo_x:hi_x] = acc / sub
    return img


def zoom_about(img, z, cx, cy):
    """Separable bilinear zoom about (cx, cy). z=1 returns img exactly."""
    if z == 1.0:
        return img.copy()
    xs = cx + (np.arange(W) - cx) / z
    ys = cy + (np.arange(H) - cy) / z
    x0 = np.clip(np.floor(xs).astype(int), 0, W - 2)
    y0 = np.clip(np.floor(ys).astype(int), 0, H - 2)
    fx = np.clip(xs - x0, 0.0, 1.0)
    fy = np.clip(ys - y0, 0.0, 1.0)
    a = img[np.ix_(y0, x0)]
    b = img[np.ix_(y0, x0 + 1)]
    c = img[np.ix_(y0 + 1, x0)]
    d = img[np.ix_(y0 + 1, x0 + 1)]
    top = a * (1 - fx)[None, :, None] + b * fx[None, :, None]
    bot = c * (1 - fx)[None, :, None] + d * fx[None, :, None]
    return top * (1 - fy)[:, None, None] + bot * fy[:, None, None]


def add_border(img, alpha):
    i, t_ = BORDER_IN, BORDER_TH
    m = np.zeros((H, W))
    m[i:i + t_, i:W - i] = 1.0
    m[H - i - t_:H - i, i:W - i] = 1.0
    m[i:H - i, i:i + t_] = 1.0
    m[i:H - i, W - i - t_:W - i] = 1.0
    over(img, m * alpha, C_BORD)


PHOTO = None


def get_photo():
    global PHOTO
    if PHOTO is None:
        PHOTO = exposure(T_FREEZE - T_PHOTO / 2.0, T_PHOTO, SUB_P)
    return PHOTO


def render_frames():
    photo = get_photo()
    for f in range(N_FRAMES):
        if f < N_A:
            img = exposure(f / FPS, T_SHUT_A, SUB_A)
        else:
            i = f - N_A
            z = 1.0 + (ZOOM_MAX - 1.0) * i / (N_H - 1)
            img = zoom_about(photo, z, X_FREEZE, HUB_Y)
            add_border(img, min(1.0, (i + 1) / 5.0))
        yield (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def frame_at(f):
    for i, fr in enumerate(render_frames()):
        if i == f:
            return fr
    return None


# ---------------------------------------------------------------- streak measurement
def measure_streaks(fr, off_x=0, off_y=0):
    """Extent of each dot's streak in a frame (uint8), bounded to the box the
    model predicts (traps 58/64: a colour is only unique inside the thing you
    meant). Returns list of (ext_x, ext_y, n_px)."""
    lum = fr.astype(float).mean(axis=2) / 255.0
    res = []
    for dm in DOTS:
        bx0, bx1, by0, by1 = [int(v) for v in dm["box"]]
        x0 = bx0 - off_x; x1 = bx1 - off_x
        y0 = by0 - off_y; y1 = by1 - off_y
        assert 0 <= x0 and x1 <= fr.shape[1] and 0 <= y0 \
            and y1 <= fr.shape[0], "streak box outside measured crop"
        sub = (lum[y0:y1, x0:x1] > THR) & dm["pmask"]
        if not sub.any():
            res.append((0.0, 0.0, 0))
            continue
        cols = np.where(sub.any(axis=0))[0]
        rows = np.where(sub.any(axis=1))[0]
        res.append((float(cols.max() - cols.min() + 1),
                    float(rows.max() - rows.min() + 1),
                    int(sub.sum())))
    return res


# ---------------------------------------------------------------- checks
def run_checks():
    ok = []

    def chk(name, cond, detail=""):
        ok.append(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name}  {detail}")

    # 1. no-slip: hub advance == R * rotation, identically
    ts = np.linspace(0, N_A / FPS, 500)
    slip = np.abs((hub_x(ts) - HUB_X0) - R_TIRE * theta(ts))
    chk("no-slip", slip.max() < 1e-9, f"max {slip.max():.2e} px")

    # 2. the contact point is at rest
    vx, vy = point_vel(T_FREEZE, R_TIRE, PSI0)
    chk("contact point at rest", np.hypot(vx, vy) < 1e-9,
        f"|v|={np.hypot(vx, vy):.2e}")

    # 3. THE IDENTITY: |v(p)| == omega * dist(p, contact) for random points
    rng = np.random.default_rng(3)
    rr = rng.uniform(0, R_TIRE, 20000)
    pp = rng.uniform(0, 2 * np.pi, 20000)
    px, py = point(T_FREEZE, rr, pp)
    pvx, pvy = point_vel(T_FREEZE, rr, pp)
    cx, cy = contact(T_FREEZE)
    err = np.abs(np.hypot(pvx, pvy) - OMEGA * np.hypot(px - cx, py - cy))
    chk("|v| = omega * d(contact)", err.max() < 1e-9,
        f"max err {err.max():.2e} over 20000 points")

    # 4. top of wheel moves at exactly 2v
    vx, vy = point_vel(T_FREEZE, R_TIRE, PSI0 + np.pi)
    chk("top speed = 2v", abs(vx - 2 * V) < 1e-9 and abs(vy) < 1e-9,
        f"vx={vx:.1f} = {vx/V:.3f} v")

    # 5. streak path length == omega * integral(d dt) — the identity,
    # integrated over the exposure (dense sampling; the rendered PHOTO_T
    # midpoints span only (n-1)/n of the window, so integrate separately)
    td = np.linspace(T_FREEZE - T_PHOTO / 2, T_FREEZE + T_PHOTO / 2, 4001)
    cxd = hub_x(td)
    worst = 0.0
    for i in range(N_DOTS):
        psi = PSI0 + i * 2.0 * np.pi / N_DOTS
        xs, ys = point(td, R_DOTS, psi)
        path = np.hypot(np.diff(xs), np.diff(ys)).sum()
        dd = np.hypot(xs - cxd, ys - Y_ROAD)
        want = OMEGA * np.trapezoid(dd, td)
        worst = max(worst, abs(path / want - 1.0))
    chk("streak length = omega * integral d dt", worst < 1e-5,
        f"max rel err {worst:.2e} over {N_DOTS} dots")

    # 6. the punchline numbers
    d_sort = sorted(DOTS, key=lambda m: m["d_mean"])
    lo, hi = d_sort[0], d_sort[-1]
    chk("bottom dot barely smears", lo["path"] < 9.0,
        f"{lo['path']:.1f} px of travel (d={lo['d_mean']:.0f})")
    chk("top dot streaks", hi["path"] > 85.0,
        f"{hi['path']:.1f} px of travel (d={hi['d_mean']:.0f})")

    # 7. dot boxes are disjoint (the measurement cannot cross-count)
    disj = True
    for a in range(N_DOTS):
        for b in range(a + 1, N_DOTS):
            ax0, ax1, ay0, ay1 = DOTS[a]["box"]
            bx0, bx1, by0, by1 = DOTS[b]["box"]
            if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                disj = False
    chk("streak boxes disjoint", disj)

    # 8. framing: the whole photo (streaks included) is inside the frame
    all_x = np.concatenate([dm["xs"] for dm in DOTS])
    all_y = np.concatenate([dm["ys"] for dm in DOTS])
    chk("photo fully in frame",
        all_x.min() - R_DOT > 10 and all_x.max() + R_DOT < W - 10
        and all_y.min() - R_DOT > 10 and all_y.max() + R_DOT < H - 10,
        f"x {all_x.min():.0f}..{all_x.max():.0f}")

    # 9. duration
    chk("duration", 4.5 <= N_FRAMES / FPS <= 6.5, f"{N_FRAMES/FPS:.2f} s")

    # 10. wheel starts fully off-frame
    chk("starts off-frame", HUB_X0 + R_TIRE < -10, f"right edge {HUB_X0+R_TIRE:.0f}")

    # ---- pixel checks
    photo8 = (np.clip(get_photo(), 0, 1) * 255.0 + 0.5).astype(np.uint8)

    lit = np.count_nonzero(photo8.max(axis=2) > 40) / (W * H)
    chk("lit fraction (photo)", 0.10 < lit < 0.85, f"{lit:.3f}")

    meas = measure_streaks(photo8)
    worst_x = max(abs(m[0] - dm["pext_x"]) for m, dm in zip(meas, DOTS))
    worst_y = max(abs(m[1] - dm["pext_y"]) for m, dm in zip(meas, DOTS))
    chk("streak extents match model (render)", worst_x < 3.0 and worst_y < 3.0,
        f"max dx={worst_x:.1f} dy={worst_y:.1f} px")

    # sharpness asymmetry: the crisp dot is bright, the long streak is dim,
    # and by the same amount the ink is spread
    lum = photo8.astype(float).mean(axis=2) / 255.0
    def peak(dm):
        x0, x1, y0, y1 = [int(v) for v in dm["box"]]
        return lum[max(0, y0):y1, max(0, x0):x1].max()
    chk("bottom dot crisp, top streak dim",
        peak(lo) > 0.80 and peak(hi) < 0.60,
        f"peaks {peak(lo):.2f} vs {peak(hi):.2f}")

    # bright pixels on the sidewall ring live ONLY inside the streak boxes
    # (ring bounds derived from the model, trap 46: the ink's own radial span)
    yy, xx = np.mgrid[0:H, 0:W]
    rad_all = np.hypot(np.concatenate([dm["xs"] for dm in DOTS]) - X_FREEZE,
                       np.concatenate([dm["ys"] for dm in DOTS]) - HUB_Y)
    r_in = rad_all.min() - R_DOT - 2
    r_out = rad_all.max() + R_DOT + 2
    rr_ = np.hypot(xx - X_FREEZE, yy - HUB_Y)
    ring = (rr_ > r_in) & (rr_ < r_out)
    bright = (lum > THR) & ring
    inbox = np.zeros((H, W), bool)
    for dm in DOTS:
        x0, x1, y0, y1 = [int(v) for v in dm["box"]]
        inbox[max(0, y0):y1, max(0, x0):x1] = True
    stray = np.count_nonzero(bright & ~inbox)
    chk("no bright ink outside streak boxes (sidewall ring)", stray == 0,
        f"{stray} px stray")

    # first hold frame is the photograph exactly (z=1), outside the border
    f_hold = frame_at(N_A)
    crop = (slice(60, 1460), slice(60, 1020))
    chk("hold frame 0 == photo (measured region)",
        np.array_equal(f_hold[crop], photo8[crop]))

    # rolling frame: wheel pixels only inside its exposure bbox
    f_mid = frame_at(45)
    t_mid = 45 / FPS
    diff = np.abs(f_mid.astype(int)
                  - (np.clip(STATIC, 0, 1) * 255 + 0.5).astype(int)).max(axis=2)
    cols_d = np.where((diff > 8).any(axis=0))[0]
    x_lo = hub_x(t_mid) - R_TIRE - 42
    x_hi = hub_x(t_mid + T_SHUT_A) + R_TIRE + 42
    chk("rolling wheel where the model says",
        len(cols_d) > 0 and cols_d.min() >= x_lo - 2 and cols_d.max() <= x_hi + 2,
        f"cols {cols_d.min() if len(cols_d) else '-'}..{cols_d.max() if len(cols_d) else '-'}"
        f" vs model {x_lo:.0f}..{x_hi:.0f}")

    print("\n  streak table (model): d from contact -> path px over "
          f"{T_PHOTO*1000:.0f} ms")
    for dm in sorted(DOTS, key=lambda m: m["d_mean"]):
        print(f"    d={dm['d_mean']:6.1f}  path={dm['path']:5.1f}  "
              f"ext_x={dm['ext_x']:5.1f}")
    if not all(ok):
        print(f"\n{ok.count(False)} CHECK(S) FAILED")
        sys.exit(1)
    print(f"\nALL {len(ok)} CHECKS PASSED")


# ---------------------------------------------------------------- encode
def encode():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-crf", "18", "-preset", "slow",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stderr=subprocess.DEVNULL)
    for fr in render_frames():
        p.stdin.write(fr.tobytes())
    p.stdin.close()
    p.wait()
    assert p.returncode == 0, "ffmpeg failed"
    print(f"encoded {OUT} ({os.path.getsize(OUT)} bytes)")


def check_encode():
    """Measure the claim off the shipped bytes: decode the photograph frame,
    measure all twelve streaks against omega*d*T (crop inside ffmpeg, trap 34)."""
    cx, cy, cw, ch = 60, 520, 960, 960
    cmd = ["ffmpeg", "-i", OUT,
           "-vf", f"select=eq(n\\,{N_A}),crop={cw}:{ch}:{cx}:{cy}",
           "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    assert len(raw) == cw * ch * 3, f"decode size {len(raw)}"
    fr = np.frombuffer(raw, np.uint8).reshape(ch, cw, 3)
    meas = measure_streaks(fr, off_x=cx, off_y=cy)
    print("ENCODE CHECK — streaks measured off the shipped h264:")
    worst = 0.0
    for m, dm in zip(meas, DOTS):
        want = dm["pext_x"]
        err = abs(m[0] - want)
        worst = max(worst, err)
        print(f"    dot {dm['i']:2d}  d={dm['d_mean']:6.1f}  "
              f"measured ext_x={m[0]:5.1f}  model={want:5.1f}  "
              f"err={err:4.1f}")
    assert worst < 5.0, f"worst streak error {worst:.1f} px"
    # monotone: streak extent orders with distance from contact
    order = np.argsort([dm["d_mean"] for dm in DOTS])
    ext_sorted = np.array([meas[i][0] for i in order])
    mono_ok = np.all(np.diff(ext_sorted) > -6.0)
    assert mono_ok, f"streaks not ordered by d: {ext_sorted}"
    # frame count
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", OUT],
        capture_output=True, text=True).stdout.strip()
    assert int(probe) == N_FRAMES, f"frames {probe} != {N_FRAMES}"
    print(f"    worst error {worst:.1f} px over {N_DOTS} streaks; "
          f"{probe} frames; extents ordered by distance from contact")


def review_stills():
    """Trap 67: look at it at the size it will be watched."""
    import shutil
    photo8 = (np.clip(get_photo(), 0, 1) * 255.0 + 0.5).astype(np.uint8)
    mid = frame_at(58)
    for name, fr in [("photo", photo8), ("roll", mid)]:
        p = OUT.replace(".mp4", f"_{name}.png")
        tmp = p + ".raw"
        with open(tmp, "wb") as fh:
            fh.write(fr.tobytes())
        subprocess.run(
            ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-i", tmp, "-vf", "scale=360:-1", p],
            capture_output=True)
        os.remove(tmp)
        print(f"review still: {p}")


if __name__ == "__main__":
    print(f"WHEEL — {N_FRAMES} frames @ {FPS} fps = {N_FRAMES/FPS:.2f} s")
    run_checks()
    if "--stills" in sys.argv:
        review_stills()
    elif "--check" not in sys.argv:
        encode()
        check_encode()
