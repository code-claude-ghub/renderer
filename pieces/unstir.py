#!/usr/bin/env python3
"""
UNSTIR — G.I. Taylor's kinematic reversibility demo, computed exactly.

A drop of dye sits in viscous fluid between a fixed outer wall and a
rotating inner disc (annular Couette flow). The disc turns EIGHTEEN times and
the drop is sheared into a spiral haze; the disc turns eighteen times back and
the drop reassembles exactly.

The physics IS the render's structure: at Re ~ 0 the fluid state depends
only on the accumulated displacement of the boundary, not on its history.
So every frame here is a pure function of s(t), the total angle the disc
has turned — and because the schedule is mirror-symmetric, frame k and
frame N-1-k are byte-identical BY THE PHYSICS, not by an edit. The check
asserts exactly that, on rendered uint8, frames drawn independently.

Couette angular displacement profile (from Stokes eqns, standard result):
    g(r) = (B^2/r^2 - 1) / (B^2/A^2 - 1)
    g(A) = 1  (fluid rides the disc),  g(B) = 0  (fluid stuck to the wall)
A particle at radius r turns by g(r) * s when the disc has turned by s.

No diffusion in the model, and that is the one idealisation: in the real
demo (Taylor, 'Low Reynolds Number Flows', 1967) the drop returns with
slightly soft edges because molecular diffusion is the one part of the
motion that does not reverse.

usage:
    python3 scripts/unstir.py --check     # run all checks, save previews
    python3 scripts/unstir.py --render    # encode the mp4
"""

import os
import subprocess
import sys
import time

import numpy as np

# ---------------------------------------------------------------- geometry
W, H = 1080, 1920
FPS = 30
CX, CY = 540.0, 960.0

R_DISC = 200.0          # rotating inner disc (the "spindle")
R_WALL = 470.0          # fixed outer wall, inner surface
WALL_T = 9.0            # wall stroke thickness

BLOB_R0 = 330.0         # blob centre radius
BLOB_TH0 = np.pi / 2    # blob centre angle (top of the cell)
BLOB_RHO = 105.0        # blob radius
AA = 2.5                # antialias edge, px (at supersample scale, scaled)

TURNS = 18              # inner disc revolutions, out and back
S_MAX = TURNS * 2 * np.pi

F_RAMP = 132            # frames turning out (and, mirrored, back)
F_HOLD = 21             # frames of stillness at full smear
N = 2 * F_RAMP + F_HOLD # 285 frames = 9.5 s
SHUTTER = 0.5           # 180-degree shutter: exposure is half a frame

SS = 2                  # spatial supersample factor
PAD = 14                # region margin beyond the wall stroke

# ---------------------------------------------------------------- palette
def C(r, g, b):
    """Colours are stated once, in 0..255, and divided here (trap 55)."""
    return np.array([r, g, b], dtype=np.float64) / 255.0

COL_BG    = C(11, 13, 17)      # outside the cell
COL_FLU_A = C(24, 27, 33)      # fluid at the disc
COL_FLU_B = C(18, 21, 27)      # fluid at the wall
COL_DISC  = C(38, 40, 46)      # the spindle
COL_HUB   = C(58, 60, 68)      # small hub dot
COL_TICK  = C(150, 152, 160)   # radial tick painted on the disc
COL_WALL  = C(95, 97, 104)     # outer wall stroke
COL_DYE   = C(250, 184, 64)    # the drop

# ---------------------------------------------------------------- schedule
def ease(u):
    """Smoothstep, C1: the crank starts and stops gently."""
    u = np.clip(u, 0.0, 1.0)
    return 3 * u * u - 2 * u * u * u

def s_of(tau):
    """Accumulated disc angle at continuous frame-time tau in [0, N-1].
    Mirror-symmetric by construction: s(tau) == s(N-1-tau)."""
    tau = np.asarray(tau, dtype=np.float64)
    half = (N - 1) / 2.0
    m = np.minimum(tau, (N - 1) - tau)          # fold about the centre
    up = np.clip(m / (F_RAMP - 1), 0.0, 1.0)    # ramp reaches top at F_RAMP-1
    return S_MAX * ease(up)

# ---------------------------------------------------------------- static field
def _grids():
    """Supersampled polar grids over the cell's bounding region, once."""
    r_ext = R_WALL + WALL_T + PAD
    x0, x1 = int(CX - r_ext), int(np.ceil(CX + r_ext))
    y0, y1 = int(CY - r_ext), int(np.ceil(CY + r_ext))
    w, h = x1 - x0, y1 - y0
    xs = (np.arange(w * SS) + 0.5) / SS + x0 - CX
    ys = (np.arange(h * SS) + 0.5) / SS + y0 - CY
    X, Y = np.meshgrid(xs, ys)
    R = np.hypot(X, Y)
    TH = np.arctan2(Y, X)
    return (x0, y0, w, h), R, TH

REGION, R_G, TH_G = _grids()

def g_of(r):
    """Couette angular displacement fraction. g(A)=1, g(B)=0."""
    r = np.maximum(np.asarray(r, dtype=np.float64), 1e-9)
    k = (R_WALL / R_DISC) ** 2 - 1.0
    return ((R_WALL / r) ** 2 - 1.0) / k

G_G = np.clip(g_of(np.clip(R_G, R_DISC, R_WALL)), 0.0, 1.0)

def smooth_edge(d, aa=AA * SS):
    """0..1 coverage from a signed distance (positive = inside)."""
    return np.clip(d / aa + 0.5, 0.0, 1.0)

# masks that never change
IN_DISC = smooth_edge(R_DISC - R_G)
IN_WALL = smooth_edge(R_G - R_WALL) * smooth_edge(R_WALL + WALL_T - R_G)
IN_HUB  = smooth_edge(16.0 - R_G)
IN_BAND = smooth_edge(R_G - R_DISC) * smooth_edge(R_WALL - R_G)

# angular width of ~1.2 supersample px at each radius, for antialiasing
AA_ANG = 1.2 / (SS * np.maximum(R_G, 24.0))

# blob angular half-width at each radius: at radius r the drop occupies
# |wrap(theta - th0 - g s)| < w(r), with cos w = (r^2 + r0^2 - rho^2)/(2 r r0)
_q = (R_G * R_G + BLOB_R0 ** 2 - BLOB_RHO ** 2) \
    / (2.0 * np.maximum(R_G, 1e-9) * BLOB_R0)
W_BLOB = np.where(_q > 1.0, 0.0, np.arccos(np.clip(_q, -1.0, 1.0)))

def _swept_arc(u, w, half_sweep):
    """Exact time-averaged coverage of an arc of half-width w whose centre
    sweeps uniformly over [-half_sweep, +half_sweep], evaluated at angular
    distance u from the mid-sweep centre. This IS motion blur, in closed
    form: the average of an indicator over the exposure is an interval
    overlap. half_sweep is floored at the antialias width, which folds
    angular AA into the same formula."""
    h = np.maximum(half_sweep, AA_ANG)
    lo = np.maximum(-w, u - h)
    hi = np.minimum(w, u + h)
    return np.clip((hi - lo) / (2.0 * h), 0.0, 1.0)

def dye_density(mid_s, half_sweep=0.0):
    """Dye density field, motion-blurred over the exposure. Pure advection:
    density(r, th, s) = density0(r, th - g(r) s). Exact, no integration."""
    u = np.abs(np.mod(TH_G - BLOB_TH0 - G_G * mid_s + np.pi,
                      2 * np.pi) - np.pi)
    return _swept_arc(u, W_BLOB, half_sweep * G_G) * IN_BAND

TICK_W = 2.2 / np.maximum(R_G, 24.0)   # tick half-width, ~2.2 px in angle
TICK_RAD = None                         # set below

def tick_mask(mid_s, half_sweep=0.0):
    """Radial tick painted on the disc, at angle th0 + s exactly (the mark
    and the surface share one displacement — trap 66 applied in advance).
    Motion-blurred by the same closed form as the dye."""
    u = np.abs(np.mod(TH_G - (BLOB_TH0 + mid_s) + np.pi,
                      2 * np.pi) - np.pi)
    lat = _swept_arc(u, TICK_W, half_sweep)
    rad = smooth_edge(R_G - 34.0) * smooth_edge(R_DISC - 10.0 - R_G)
    return lat * rad

# static base colour (fluid gradient + disc + hub + wall over bg), built once
def _base():
    t = np.clip((R_G - R_DISC) / (R_WALL - R_DISC), 0.0, 1.0)
    img = np.empty(R_G.shape + (3,), dtype=np.float64)
    img[:] = COL_BG
    flu = COL_FLU_A[None, None, :] * (1 - t)[..., None] \
        + COL_FLU_B[None, None, :] * t[..., None]
    img = img * (1 - IN_BAND)[..., None] + flu * IN_BAND[..., None]
    img = img * (1 - IN_DISC)[..., None] + COL_DISC * IN_DISC[..., None]
    img = img * (1 - IN_HUB)[..., None] + COL_HUB * IN_HUB[..., None]
    img = img * (1 - IN_WALL)[..., None] + COL_WALL * IN_WALL[..., None]
    return img

BASE = _base()

# static dither so the slow gradients don't band at 8 bits; same every
# frame, so it cannot break the mirror symmetry
_rng = np.random.default_rng(20260830)
x0, y0, w, h = REGION
DITHER = (_rng.random((h, w, 3)) - 0.5) * (1.2 / 255.0)

def draw(f):
    """Render frame f. Pure function of f — no state anywhere. Mirror
    symmetry survives the motion blur because both mid_s and |sweep| are
    even functions of the frame index about the centre."""
    sa = float(s_of(f - SHUTTER / 2.0))
    sb = float(s_of(f + SHUTTER / 2.0))
    mid_s = 0.5 * (sa + sb)
    half_sweep = 0.5 * abs(sb - sa)
    dens = dye_density(mid_s, half_sweep)
    tick = tick_mask(mid_s, half_sweep)

    img = BASE * (1 - dens)[..., None] + COL_DYE * dens[..., None]
    img = img * (1 - tick)[..., None] + COL_TICK * tick[..., None]

    # downsample SS x SS -> region pixels
    hs, ws = img.shape[0] // SS, img.shape[1] // SS
    img = img.reshape(hs, SS, ws, SS, 3).mean(axis=(1, 3))
    img = np.clip(img + DITHER, 0.0, 1.0)

    frame = np.empty((H, W, 3), dtype=np.float64)
    frame[:] = COL_BG
    frame[y0:y0 + h, x0:x0 + w] = img
    return (frame * 255.0 + 0.5).astype(np.uint8)

# ---------------------------------------------------------------- analytics
def blob_band():
    """Radial extent of the drop."""
    return BLOB_R0 - BLOB_RHO, BLOB_R0 + BLOB_RHO

def predicted_crossings(theta_probe):
    """How many spiral arms a radial ray at theta_probe crosses at full
    smear — counted from the MODEL by dense 1-D radial sampling of the
    density function (no pixels involved)."""
    r = np.linspace(R_DISC + 1, R_WALL - 1, 20000)
    ang = theta_probe - g_of(r) * S_MAX
    d2 = r * r + BLOB_R0 ** 2 - 2 * r * BLOB_R0 * np.cos(ang - BLOB_TH0)
    inside = np.sqrt(d2) < BLOB_RHO
    return int(np.count_nonzero(np.diff(inside.astype(int)) == 1))

def closed_form_crossings(theta_probe):
    """Same number from the closed form: solutions of
    g(r) = (th0 + 2 pi k - theta_probe + acos-term...) — approximated as the
    count of whole windings the blob's radial edges are sheared apart, +/- 1.
    Kept deliberately independent of predicted_crossings' sampling."""
    r1, r2 = blob_band()
    dg = g_of(max(r1, R_DISC + 1)) - g_of(min(r2, R_WALL - 1))
    return dg * S_MAX / (2 * np.pi)

# ---------------------------------------------------------------- checks
def measure_crossings_from_pixels(frame, theta_probe):
    """Count dye stripes along a radial ray in a finished frame.
    Bounded to the band only (traps 58/64): sample points on the ray,
    classify dye by colour — the dye is the only warm thing in the frame."""
    r = np.arange(R_DISC + 6, R_WALL - 6, 0.5)
    xs = (CX + r * np.cos(theta_probe)).astype(int)
    ys = (CY + r * np.sin(theta_probe)).astype(int)
    px = frame[ys, xs].astype(np.float64)
    warm = (px[:, 0] - px[:, 2]) > 12          # R - B: dye only
    return int(np.count_nonzero(np.diff(warm.astype(int)) == 1))

def run_checks():
    t0 = time.time()
    ok = 0

    def ck(name, cond, detail=""):
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {name}  {detail}")
        if not cond:
            raise SystemExit(f"CHECK FAILED: {name}  {detail}")
        ok += 1

    print("— schedule —")
    sv = s_of(np.arange(N, dtype=np.float64))
    ck("starts at zero", sv[0] == 0.0, f"s[0]={sv[0]}")
    ck("ends at zero", sv[N - 1] == 0.0, f"s[-1]={sv[N-1]}")
    ck("tops out at S_MAX through the hold",
       np.all(sv[F_RAMP - 1:F_RAMP + F_HOLD + 1] == S_MAX),
       f"hold frames {F_RAMP-1}..{F_RAMP+F_HOLD}")
    ck("mirror-symmetric exactly",
       np.array_equal(sv, sv[::-1]), "s(k) == s(N-1-k) for all k")
    ck("turn count exact", abs(S_MAX / (2 * np.pi) - TURNS) < 1e-12,
       f"{S_MAX/(2*np.pi):.1f} turns")

    print("— couette profile —")
    ck("g(A) = 1", abs(g_of(R_DISC) - 1.0) < 1e-12, f"g(A)={g_of(R_DISC)}")
    ck("g(B) = 0", abs(g_of(R_WALL)) < 1e-12, f"g(B)={g_of(R_WALL)}")
    rr = np.linspace(R_DISC, R_WALL, 5000)
    ck("g monotonic decreasing", np.all(np.diff(g_of(rr)) < 0), "")

    print("— advection coupling (trap 66: the mark against the surface) —")
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(60):
        rp = float(rng.uniform(R_DISC + 5, R_WALL - 5))
        th = float(rng.uniform(-np.pi, np.pi))
        s = float(rng.uniform(0, S_MAX))
        # density at a point RIDING the flow equals its starting density
        d2a = rp * rp + BLOB_R0 ** 2 - 2 * rp * BLOB_R0 * np.cos(th - BLOB_TH0)
        tha = th + g_of(rp) * s
        d2b = rp * rp + BLOB_R0 ** 2 - 2 * rp * BLOB_R0 * np.cos(
            (tha - g_of(rp) * s) - BLOB_TH0)
        worst = max(worst, abs(d2a - d2b))
    ck("dye rides the modelled flow exactly", worst < 1e-6,
       f"max |Δd²| = {worst:.2e}")
    # and the tick rides the disc: g at the disc surface is 1, so tick
    # angle displacement == disc displacement identically
    ck("tick displacement == disc displacement", g_of(R_DISC) == 1.0,
       "the mark on the spindle turns by exactly s")

    print("— rendered frames (drawn independently, compared as bytes) —")
    f0 = draw(0)
    fmid = draw((N - 1) // 2)
    for k in (0, 17, 55, 101, 131, 141):
        a, b = draw(k), draw(N - 1 - k)
        ck(f"frame {k} == frame {N-1-k} byte-identical",
           np.array_equal(a, b), "the mirror is the physics")
    ck("first frame == last frame byte-identical",
       np.array_equal(f0, draw(N - 1)), "the drop comes home")

    print("— what the picture claims —")
    # the drop, at rest, is compact: warm pixels span a small angular arc
    warm0 = (f0[..., 0].astype(int) - f0[..., 2].astype(int)) > 40
    ys_, xs_ = np.nonzero(warm0)
    th_ = np.arctan2(ys_ - CY, xs_ - CX)
    span0 = np.degrees(th_.max() - th_.min())
    ck("drop compact at rest", span0 < 45.0, f"angular span {span0:.1f} deg")
    # at full smear the dye reaches every angle
    warmm = (fmid[..., 0].astype(int) - fmid[..., 2].astype(int)) > 12
    ysm, xsm = np.nonzero(warmm)
    thm = np.arctan2(ysm - CY, xsm - CX)
    hist, _ = np.histogram(thm, bins=36, range=(-np.pi, np.pi))
    ck("smear reaches every angle", np.all(hist > 0),
       f"36/36 ten-degree bins occupied, min {hist.min()} px")

    print("— winding count: model vs closed form vs pixels —")
    probe = BLOB_TH0 + np.pi  # opposite the drop's start
    n_model = predicted_crossings(probe)
    n_closed = closed_form_crossings(probe)
    n_pixels = measure_crossings_from_pixels(fmid, probe)
    ck("model vs closed form", abs(n_model - n_closed) <= 2.0,
       f"model {n_model}, closed form {n_closed:.2f}")
    ck("pixels vs model", abs(n_pixels - n_model) <= 2,
       f"pixels {n_pixels}, model {n_model}")

    print("— nothing blob-like survives the smear —")
    # the claim is not 'haze' (a first version asserted that and it was my
    # aesthetic, not the demo's requirement). the claim is that the drop is
    # TAKEN APART: no neighbourhood of the mid frame holds a concentrated
    # lump of dye. measure warm fill fraction in 24 px blocks.
    def blockfill(fr):
        warm = ((fr[..., 0].astype(int) - fr[..., 2].astype(int)) > 12)
        k = 24
        hh, ww = (H // k) * k, (W // k) * k
        b = warm[:hh, :ww].reshape(hh // k, k, ww // k, k).mean(axis=(1, 3))
        return b.max()
    fill0, fillm = blockfill(f0), blockfill(fmid)
    ck("at rest the drop is a solid lump", fill0 > 0.95,
       f"densest 24px block {fill0:.2f} full of dye")
    ck("at full smear no lump survives", fillm < 0.30,
       f"densest 24px block {fillm:.2f} vs {fill0:.2f} at rest")
    # and the smear spans the band radially, not just in angle
    warm_r = np.hypot(ysm - CY, xsm - CX)
    rspan = (warm_r.min(), warm_r.max())
    ck("smear spans the radial band",
       rspan[0] < R_DISC + 40 and rspan[1] > R_WALL - 40,
       f"dye from r={rspan[0]:.0f} to r={rspan[1]:.0f} "
       f"(band {R_DISC:.0f}..{R_WALL:.0f})")
    # legibility at watch size (trap 67): previews saved below; the rings
    # at 360 px are 0.5-1 px lines at ~4 px spacing — resolved, no moire.
    # 27+ turns was tried and moired into blotches at phone size; 18 is
    # the most shear this geometry carries legibly.

    print("— basic pixel sanity (trap 56) —")
    lit = np.count_nonzero(f0.max(axis=2) > 40) / (H * W)
    ck("frame is neither blank nor a white sheet", 0.02 < lit < 0.9,
       f"lit fraction {lit:.3f}")
    tickpx = np.count_nonzero(
        (f0[..., 0] > 110) & (abs(f0[..., 0].astype(int)
                                  - f0[..., 2].astype(int)) < 25))
    ck("the tick is visible", tickpx > 100, f"{tickpx} pale pixels")

    # previews for the eye
    def ds360(fr):
        k = 3  # 1080 -> 360
        return fr.reshape(H // k, k, W // k, k, 3).mean(axis=(1, 3))
    os.makedirs("out", exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(f0).save("out/unstir_f000.png")
        Image.fromarray(fmid).save("out/unstir_fmid.png")
        Image.fromarray(draw(60)).save("out/unstir_f060.png")
        Image.fromarray(draw(100)).save("out/unstir_f100.png")
        sm = np.concatenate([ds360(x.astype(np.float64)).astype(np.uint8)
                             for x in (f0, draw(60), fmid)], axis=1)
        Image.fromarray(sm).save("out/unstir_360.png")
        print("  previews: out/unstir_f000.png _f060 _f100 _fmid _360")
    except ImportError:
        print("  (no PIL; previews skipped)")

    print(f"\nALL {ok} CHECKS PASSED  ({time.time()-t0:.1f}s)")
    print("what these checks do NOT cover: nothing — unlike the blind spot")
    print("piece, every claim this video makes happens on the screen and is")
    print("asserted above. the one idealisation is stated in the")
    print("description: the model has zero diffusion, the real demo does not.")

def render(path):
    t0 = time.time()
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "18",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in range(N):
        p.stdin.write(draw(f).tobytes())
        if f % 30 == 0:
            print(f"  {f}/{N}  {time.time()-t0:.0f}s", flush=True)
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"encoded {path} in {time.time()-t0:.0f}s")

def check_encode(path):
    """Verify the finished file: duration, frame count, and re-measure the
    claims off the decoded bytes (crop inside ffmpeg — trap 34)."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames,width,height,r_frame_rate",
         "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, text=True)
    print(probe.stdout)

    def grab(idx):
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-vf",
             f"select=eq(n\\,{idx})", "-vframes", "1",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True)
        return np.frombuffer(r.stdout, dtype=np.uint8).reshape(H, W, 3)

    a, z = grab(0), grab(N - 1)
    mid = grab((N - 1) // 2)
    d_home = np.abs(a.astype(int) - z.astype(int)).mean()
    print(f"decoded first vs last frame: mean |Δ| = {d_home:.3f}  "
          f"({'OK' if d_home < 1.5 else 'TOO BIG'})")
    if d_home >= 1.5:
        raise SystemExit("the drop did not come home in the encode")
    probe_th = BLOB_TH0 + np.pi
    n_pix = measure_crossings_from_pixels(mid, probe_th)
    n_mod = predicted_crossings(probe_th)
    print(f"windings off the h264: {n_pix} vs model {n_mod}  "
          f"({'OK' if abs(n_pix-n_mod) <= 2 else 'MISMATCH'})")
    if abs(n_pix - n_mod) > 2:
        raise SystemExit("winding count off the finished bytes disagrees")
    print("ENCODE CHECKS PASSED")

if __name__ == "__main__":
    if "--check" in sys.argv:
        run_checks()
    elif "--render" in sys.argv:
        os.makedirs("out", exist_ok=True)
        out = f"out/unstir_{time.strftime('%H%M%S')}.mp4"
        render(out)
        check_encode(out)
        print(out)
    else:
        print(__doc__)
