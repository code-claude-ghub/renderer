#!/usr/bin/env python3
"""TRAIN — part of every moving train is always going backwards.

A flanged wheel rolls right along a rail. A marked point on the flange
traces a prolate trochoid, drawn persistently: cyan while its ground
velocity points forward, red while it points backward.

The claim is an exact identity, not an approximation: a point fixed to a
wheel rolling without slipping moves backward precisely while it is BELOW
the level of the contact point (the railhead top). Proof in one line:
  v_x = v(1 - (r/R) cos th) < 0  <=>  cos th > R/r  <=>  y_point > y_railhead
So the colour rule (red iff below the railhead) and the velocity rule
(red iff moving backward) are the same rule, and the finished video can be
checked for it: no red pixel above the railhead, no cyan pixel below.

Real numbers (description): a standard wheel R=460 mm with a 30 mm flange
spends arccos(460/490)/pi = 11.2% of every revolution moving backward.
This render exaggerates the flange (r/R = 1.46) so the dip is visible at
phone size; the identity holds at any ratio.

Diagram licence: the flange really passes BESIDE the rail; a side view
flattens that, so here it is drawn in front.
"""
import os
import subprocess
import sys
import time

import numpy as np

# ---------------------------------------------------------------- geometry
W, H = 1080, 1920
FPS = 30
Y_RAIL = 1150.0            # railhead top row (the threshold of the claim)
R_TREAD = 130.0            # rolling radius, px
R_FLANGE = 200.0           # flange outer radius, px
R_MARK = 190.0             # marked point radius (inside the flange rim)
RAIL_H = 16.0              # railhead bar height
HUB_Y = Y_RAIL - R_TREAD

T_REV = 2.1                # seconds per revolution
V = 2.0 * np.pi * R_TREAD / T_REV          # ground speed, px/s (~389)
OMEGA = V / R_TREAD                        # rad/s

HUB_X0 = -(R_FLANGE + 55.0)                # start fully off-frame left
LOOP1_X = 130.0                            # hub x at first dip bottom
T_LOOP1 = (LOOP1_X - HUB_X0) / V
TH0 = -OMEGA * T_LOOP1                     # so theta=0 exactly at the dip

N_A = 123                                  # crossing frames (4.1 s)
N_B = 54                                   # hold on the finished trail
N_FRAMES = N_A + N_B                       # 177 -> 5.9 s
SUB = 8                                    # trail substeps per frame

THC = np.arccos(R_TREAD / R_MARK)          # backward half-angle
CIRC = 2.0 * np.pi * R_TREAD               # loop spacing on the rail

# ---------------------------------------------------------------- palette
BG       = np.array([0.030, 0.035, 0.050])
C_RAIL   = np.array([0.400, 0.420, 0.460])
C_RAILTOP= np.array([0.640, 0.660, 0.700])
C_SLEEP  = np.array([0.130, 0.140, 0.160])
C_TREAD  = np.array([0.300, 0.330, 0.380])
C_FLANGE = np.array([0.215, 0.235, 0.285])
C_HOLE   = np.array([0.070, 0.078, 0.100])
C_HUB    = np.array([0.500, 0.520, 0.560])
C_PIN    = np.array([0.720, 0.730, 0.760])
C_FWD    = np.array([0.230, 0.720, 0.930])   # cyan: moving forward
C_BACK   = np.array([1.000, 0.210, 0.110])   # red: moving backward
C_MARK   = np.array([1.000, 0.970, 0.900])   # the marked point
C_HUBTR  = np.array([0.400, 0.420, 0.450])   # hub trail (straight line)
W_TRAIL = 6.0
W_HUBTR = 3.0
R_DOT = 9.0

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out",
                   time.strftime("train_%H%M%S.mp4"))
OUT = os.path.abspath(OUT)


# ---------------------------------------------------------------- kinematics
def hub_x(t):
    return HUB_X0 + V * t


def theta(t):
    return TH0 + OMEGA * t


def point(t, r):
    """Position of a material point at radius r (th=0 is the bottom)."""
    th = theta(t)
    return hub_x(t) - r * np.sin(th), HUB_Y + r * np.cos(th)


def point_vel(t, r):
    th = theta(t)
    return V - r * OMEGA * np.cos(th), -r * OMEGA * np.sin(th)


# ---------------------------------------------------------------- trail
def build_trail():
    """Substep samples of the mark over phase A: (t, x, y, vx, back)."""
    n = N_A * SUB + 1
    t = np.arange(n) / (FPS * SUB)
    x, y = point(t, R_MARK)
    vx, _ = point_vel(t, R_MARK)
    back = vx < 0.0
    return t, x, y, vx, back


TR_T, TR_X, TR_Y, TR_VX, TR_BACK = build_trail()


# ---------------------------------------------------------------- rasterisers
def stamp_capsule(rgb, a, x0, y0, x1, y1, width, col):
    """AA capsule (line segment with round caps), premultiplied-over."""
    r = width / 2.0
    lo_x = int(max(0, np.floor(min(x0, x1) - r - 2)))
    hi_x = int(min(W, np.ceil(max(x0, x1) + r + 2)))
    lo_y = int(max(0, np.floor(min(y0, y1) - r - 2)))
    hi_y = int(min(H, np.ceil(max(y0, y1) + r + 2)))
    if lo_x >= hi_x or lo_y >= hi_y:
        return
    yy, xx = np.mgrid[lo_y:hi_y, lo_x:hi_x]
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        d = np.hypot(xx - x0, yy - y0)
    else:
        u = np.clip(((xx - x0) * dx + (yy - y0) * dy) / L2, 0.0, 1.0)
        d = np.hypot(xx - (x0 + u * dx), yy - (y0 + u * dy))
    s = np.clip(r + 0.6 - d, 0.0, 1.2) / 1.2
    if s.max() <= 0:
        return
    win_rgb = rgb[lo_y:hi_y, lo_x:hi_x]
    win_a = a[lo_y:hi_y, lo_x:hi_x]
    win_rgb *= (1.0 - s)[..., None]
    win_rgb += col * s[..., None]
    win_a *= (1.0 - s)
    win_a += s


def disc(xx, yy, cx, cy, r):
    return np.clip(r + 0.6 - np.hypot(xx - cx, yy - cy), 0.0, 1.2) / 1.2


def over(img, alpha, col):
    img *= (1.0 - alpha)[..., None]
    img += col * alpha[..., None]


# ---------------------------------------------------------------- layers
def static_layer():
    img = np.empty((H, W, 3))
    img[:] = BG
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    # sleepers
    for sx in range(-60, W + 260, 240):
        m = ((xx >= sx) & (xx < sx + 46)
             & (yy >= Y_RAIL + RAIL_H + 6) & (yy < Y_RAIL + 96))
        img[m] = C_SLEEP
    # railhead bar + bright top edge (the threshold line of the claim)
    bar = np.clip(np.minimum(yy - Y_RAIL, Y_RAIL + RAIL_H - yy) + 0.5,
                  0, 1)
    over(img, bar, C_RAIL)
    top = np.clip(1.5 - np.abs(yy - (Y_RAIL + 1.5)), 0, 1)
    over(img, top, C_RAILTOP)
    return img


STATIC = static_layer()


def draw_wheel(img, t):
    hx = hub_x(t)
    th = theta(t)
    if hx + R_FLANGE < -4 or hx - R_FLANGE > W + 4:
        return
    lo_x = int(max(0, np.floor(hx - R_FLANGE - 4)))
    hi_x = int(min(W, np.ceil(hx + R_FLANGE + 4)))
    lo_y = int(max(0, np.floor(HUB_Y - R_FLANGE - 4)))
    hi_y = int(min(H, np.ceil(HUB_Y + R_FLANGE + 4)))
    if lo_x >= hi_x or lo_y >= hi_y:
        return
    yy, xx = np.mgrid[lo_y:hi_y, lo_x:hi_x].astype(float)
    win = img[lo_y:hi_y, lo_x:hi_x]
    over(win, disc(xx, yy, hx, HUB_Y, R_FLANGE), C_FLANGE)
    over(win, disc(xx, yy, hx, HUB_Y, R_TREAD), C_TREAD)
    # five lightening holes, smeared along their motion (half-frame shutter)
    dth = OMEGA / (2.0 * FPS)
    for k in range(5):
        ang = th + k * 2.0 * np.pi / 5.0
        hx0 = hx - 72.0 * np.sin(ang)
        hy0 = HUB_Y + 72.0 * np.cos(ang)
        hx1 = hub_x(t + 0.5 / FPS) - 72.0 * np.sin(ang + dth)
        hy1 = HUB_Y + 72.0 * np.cos(ang + dth)
        L2 = (hx1 - hx0) ** 2 + (hy1 - hy0) ** 2
        u = (np.clip(((xx - hx0) * (hx1 - hx0) + (yy - hy0) * (hy1 - hy0))
                     / max(L2, 1e-12), 0.0, 1.0))
        d = np.hypot(xx - (hx0 + u * (hx1 - hx0)),
                     yy - (hy0 + u * (hy1 - hy0)))
        over(win, np.clip(17.0 + 0.6 - d, 0.0, 1.2) / 1.2, C_HOLE)
    over(win, disc(xx, yy, hx, HUB_Y, 26.0), C_HUB)
    over(win, disc(xx, yy, hx, HUB_Y, 6.0), C_PIN)


def draw_mark(img, t):
    x0, y0 = point(t, R_MARK)
    x1, y1 = point(t + 0.5 / FPS, R_MARK)   # half-frame shutter smear
    if max(x0, x1) < -20 or min(x0, x1) > W + 20:
        return
    a = np.zeros((H, W))
    rgb = np.zeros((H, W, 3))
    stamp_capsule(rgb, a, x0, y0, x1, y1, 2 * R_DOT, C_MARK)
    m = a > 1e-4
    img[m] = img[m] * (1.0 - a[m])[:, None] + rgb[m]


# ---------------------------------------------------------------- render
def render_frames():
    """Yield uint8 frames. Trail accumulates monotonically (phase A only)."""
    tr_rgb = np.zeros((H, W, 3))
    tr_a = np.zeros((H, W))
    hold = None
    for f in range(N_FRAMES):
        if f < N_A:
            t = f / FPS
            # add this frame's trail substeps
            i0, i1 = f * SUB, (f + 1) * SUB
            for i in range(i0, min(i1, len(TR_X) - 1)):
                col = C_BACK if TR_BACK[i] else C_FWD
                stamp_capsule(tr_rgb, tr_a, TR_X[i], TR_Y[i],
                              TR_X[i + 1], TR_Y[i + 1], W_TRAIL, col)
            # hub trail: straight line from start to current hub x
            img = STATIC.copy()
            hbx = min(hub_x(t), float(W + 10))
            if hbx > 0:
                a2 = np.zeros((H, W))
                r2 = np.zeros((H, W, 3))
                stamp_capsule(r2, a2, max(HUB_X0, -10.0), HUB_Y, hbx, HUB_Y,
                              W_HUBTR, C_HUBTR)
                over_m = a2 > 1e-4
                img[over_m] = (img[over_m] * (1.0 - a2[over_m])[:, None]
                               + r2[over_m])
            draw_wheel(img, t)
            # trail rides OVER the wheel: the path is an annotation, and the
            # red ink is laid down live at the dip instead of being hidden
            # behind the flange until the wheel clears it
            img *= (1.0 - tr_a)[..., None]
            img += tr_rgb
            draw_mark(img, t)
            out = (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)
            if f == N_A - 1:
                # freeze the finished picture for the hold
                hold = out.copy()
            yield out
        else:
            yield hold


def frame_at(f):
    """Recompute one frame deterministically (for checks)."""
    gen = render_frames()
    fr = None
    for i, fr in enumerate(gen):
        if i == f:
            return fr
    return fr


# ---------------------------------------------------------------- checks
def run_checks():
    ok = []

    def chk(name, cond, detail=""):
        ok.append(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name}  {detail}")

    # 1. the tread contact point is instantaneously at rest
    vx, vy = point_vel(T_LOOP1, R_TREAD)
    chk("contact point at rest", abs(vx) < 1e-9 and abs(vy) < 1e-9,
        f"|v|={np.hypot(vx, vy):.2e}")

    # 2. the mark at the dip bottom moves backward at v(1 - r/R)
    vx, _ = point_vel(T_LOOP1, R_MARK)
    want = V * (1.0 - R_MARK / R_TREAD)
    chk("dip-bottom backward speed", abs(vx - want) < 1e-9,
        f"vx={vx:.2f} px/s = {vx/V:+.3f} v")

    # 3. THE IDENTITY: moving backward <=> below the railhead (exact)
    band = 0.05  # px tolerance at the crossing itself
    back = TR_VX < -1e-9
    below = TR_Y > Y_RAIL + band
    above = TR_Y < Y_RAIL - band
    bad = np.count_nonzero((back & above) | (~back & below))
    chk("v_x<0 <=> below railhead", bad == 0,
        f"violations={bad}/{len(TR_X)}")

    # 4. backward fraction of a revolution = arccos(R/r)/pi
    # window runs dip-bottom to dip-bottom so it is not clipped by t=0
    per = (TR_T >= T_LOOP1) & (TR_T < T_LOOP1 + 2.0 * np.pi / OMEGA)
    frac = np.count_nonzero(back & per) / max(np.count_nonzero(per), 1)
    want = THC / np.pi
    chk("backward fraction", abs(frac - want) < 2.0 / (SUB * FPS * T_REV),
        f"measured {frac:.4f} vs arccos(R/r)/pi = {want:.4f}")

    # 5. loop width from the traced polyline vs closed form
    seg = np.where(np.abs(TR_T - T_LOOP1) < THC / OMEGA + 0.02)[0]
    x_ext = TR_X[seg]
    meas = x_ext.max() - x_ext.min()
    want_w = 2.0 * (R_MARK * np.sin(THC) - R_TREAD * THC)
    chk("loop width", abs(meas - want_w) < 1.5,
        f"measured {meas:.1f} px vs 2(r sin - R th) = {want_w:.1f} px")

    # 6. dip depth = r - R exactly
    chk("dip depth", abs((TR_Y.max() - Y_RAIL) - (R_MARK - R_TREAD)) < 0.05,
        f"{TR_Y.max()-Y_RAIL:.2f} px vs r-R = {R_MARK-R_TREAD:.1f}")

    # 7. no-slip: hub advance equals arc rolled, identically
    ts = np.linspace(0, N_A / FPS, 500)
    slip = np.abs((hub_x(ts) - hub_x(0)) - R_TREAD * (theta(ts) - theta(0)))
    chk("no-slip", slip.max() < 1e-9, f"max {slip.max():.2e} px")

    # 8. two dips, both in frame, spacing = circumference
    t2 = T_LOOP1 + 2.0 * np.pi / OMEGA
    x1, x2 = hub_x(T_LOOP1), hub_x(t2)
    chk("two dips in frame",
        60 < x1 < W - 60 and 60 < x2 < W - 60 and t2 < N_A / FPS,
        f"x = {x1:.0f}, {x2:.0f}")
    chk("dip spacing = 2piR", abs((x2 - x1) - CIRC) < 1e-6,
        f"{x2-x1:.2f} vs {CIRC:.2f}")

    # 9. trail is dense enough to draw
    gaps = np.hypot(np.diff(TR_X), np.diff(TR_Y))
    chk("trail substep gaps", gaps.max() < 5.0, f"max {gaps.max():.1f} px")

    # 10. duration
    chk("duration", 5.5 <= N_FRAMES / FPS <= 6.5, f"{N_FRAMES/FPS:.2f} s")

    # 11. wheel fully exits before the hold
    chk("wheel exits", hub_x((N_A - 1) / FPS) - R_FLANGE > W + 2,
        f"hub at {hub_x((N_A-1)/FPS):.0f}")

    # ---- pixel checks on rendered frames
    def red(fr):
        return (fr[..., 0] > 170) & (fr[..., 1] < 90) & (fr[..., 2] < 90)

    def cyan(fr):
        return (fr[..., 2] > 140) & (fr[..., 1] > 120) & (fr[..., 0] < 110)

    f_final = frame_at(N_A - 1)
    f_mid = frame_at(60)
    f_early = frame_at(20)

    lit = np.count_nonzero(f_mid.max(axis=2) > 40) / (W * H)
    chk("lit fraction (mid)", 0.03 < lit < 0.55, f"{lit:.3f}")

    m = 10  # AA margin around the railhead line
    r_fin = red(f_final)
    chk("red exists (final)", np.count_nonzero(r_fin) > 800,
        f"{np.count_nonzero(r_fin)} px")
    chk("no red above railhead", np.count_nonzero(r_fin[:int(Y_RAIL - m)]) == 0,
        f"{np.count_nonzero(r_fin[:int(Y_RAIL-m)])} px")
    c_fin = cyan(f_final)
    chk("no cyan below railhead",
        np.count_nonzero(c_fin[int(Y_RAIL + m):]) == 0,
        f"{np.count_nonzero(c_fin[int(Y_RAIL+m):])} px")
    chk("no red before first dip",
        np.count_nonzero(red(f_early)) == 0,
        f"frame 20, {np.count_nonzero(red(f_early))} px")

    # hub trail is a horizontal line (measure it off the pixels, final frame)
    ht = (np.abs(f_final[..., 0].astype(int) - int(C_HUBTR[0] * 255)) < 24) \
        & (np.abs(f_final[..., 1].astype(int) - int(C_HUBTR[1] * 255)) < 24) \
        & (np.abs(f_final[..., 2].astype(int) - int(C_HUBTR[2] * 255)) < 24) \
        & (np.arange(H)[:, None] > HUB_Y - 40) \
        & (np.arange(H)[:, None] < HUB_Y + 40)
    rows = np.where(ht.any(axis=1))[0]
    chk("hub trail horizontal", len(rows) > 0
        and rows.min() >= HUB_Y - W_HUBTR and rows.max() <= HUB_Y + W_HUBTR,
        f"rows {rows.min() if len(rows) else '-'}..{rows.max() if len(rows) else '-'} (hub {HUB_Y:.0f})")

    # the mark renders where the model says (mid-crossing frame)
    mk = (f_mid[..., 0] > 230) & (f_mid[..., 1] > 220) & (f_mid[..., 2] > 200)
    mx, my = point(60 / FPS, R_MARK)
    mx1, my1 = point(60 / FPS + 0.5 / FPS, R_MARK)
    if np.any(mk):
        cy_, cx_ = np.argwhere(mk).mean(axis=0)
    else:
        cy_, cx_ = -1, -1
    chk("mark at predicted spot", np.any(mk)
        and abs(cx_ - (mx + mx1) / 2) < 6 and abs(cy_ - (my + my1) / 2) < 6,
        f"px ({cx_:.0f},{cy_:.0f}) vs model ({(mx+mx1)/2:.0f},{(my+my1)/2:.0f})")

    # hold frames byte-identical
    chk("hold frames identical",
        np.array_equal(frame_at(N_A + 5), frame_at(N_FRAMES - 1)))

    print(f"\n  real-wheel number for the description: R=460mm, flange 30mm"
          f" -> {np.arccos(460/490)/np.pi*100:.1f}% of every revolution")
    print(f"  this render: r/R = {R_MARK/R_TREAD:.3f}"
          f" -> {THC/np.pi*100:.1f}% of every revolution")
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
    """Read the claim back off the shipped bytes (crop inside ffmpeg, trap 34)."""
    def band(y0, h):
        cmd = ["ffmpeg", "-i", OUT, "-vf", f"crop={W}:{h}:0:{y0}",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        raw = subprocess.run(cmd, capture_output=True).stdout
        n = len(raw) // (W * h * 3)
        return np.frombuffer(raw, np.uint8).reshape(n, h, W, 3)

    below = band(int(Y_RAIL) + 8, 70)
    above = band(int(Y_RAIL) - 380, 370)
    n = below.shape[0]
    assert n == N_FRAMES, f"frame count {n} != {N_FRAMES}"
    fin_b = below[-1]
    fin_a = above[-1]

    def redm(fr):
        return (fr[..., 0] > 160) & (fr[..., 1] < 100) & (fr[..., 2] < 100)

    rb = redm(fin_b)
    ra = redm(fin_a)
    n_below = int(np.count_nonzero(rb))
    n_above = int(np.count_nonzero(ra))
    assert n_below > 600, f"red below railhead: {n_below}"
    assert n_above == 0, f"red above railhead: {n_above}"

    # two red clusters, one circumference apart — measured off the h264
    cols = np.where(rb.any(axis=0))[0]
    splits = np.where(np.diff(cols) > 60)[0]
    assert len(splits) == 1, f"red clusters: {len(splits)+1}"
    c1 = cols[:splits[0] + 1].mean()
    c2 = cols[splits[0] + 1:].mean()
    spacing = c2 - c1
    assert abs(spacing - CIRC) < 4.0, f"spacing {spacing:.1f} vs {CIRC:.1f}"
    print(f"ENCODE CHECK: red only below railhead ({n_below} px below, "
          f"{n_above} above); dip spacing off the h264 = {spacing:.1f} px "
          f"vs 2piR = {CIRC:.1f} px")


if __name__ == "__main__":
    print(f"TRAIN — {N_FRAMES} frames @ {FPS} fps = {N_FRAMES/FPS:.2f} s")
    run_checks()
    if "--check" not in sys.argv:
        encode()
        check_encode()
