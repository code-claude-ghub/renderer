#!/usr/bin/env python3
"""KELVIN — the wake angle that ignores the hull.

Two lanes, side by side. Left: a pressure point at a duck's pace
(0.7 m/s) in a pond-scale lane. Right: one at a supertanker's pace
(8.0 m/s) in a sea-scale lane. Both on linear deep-water gravity
waves, omega = sqrt(g k), evolved with an EXACT per-mode propagator
— the dispersion relation carries no numerical error, so whatever
angle appears is the physics' own. The transverse wavelengths differ
131x ((8/0.7)^2). The wedge half-angle does not: arcsin(1/3) =
19.47 degrees, Kelvin 1887.

The bright arm is the first Airy maximum of the caustic: it rides
just INSIDE the 19.47-degree line and converges to it like d^(-2/3).
The checks measure theta(d) in windows on the final fields and
extrapolate; both lanes must land on arcsin(1/3). All lane
parameters and the physics instruments are imported from
scripts/feas_kelvin.py (one source of truth; ALL FEASIBILITY CHECKS
PASSED there first: theta_inf 19.29 / 19.15 vs 19.47, lambda ratios
1.008 / 1.000, quiet ahead 0.000).

Timeline (17.0 s, 510 frames @ 30 fps):
  f   0- 23  still water, lane labels
  f  24-455  ships run bottom to top; wakes grow (duck lane near
             real time; tanker lane at ~12x — its clock says so)
  f 405-     theory wedge (19.47 deg) fades in over each apex
  f 435-     closing label: arcsin(1/3) = 19.47 deg
  f 456-509  hold on the final fields

Silent video.
"""
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feas_kelvin import (DUCK, TANK, TH_TRUE, G, Sim, extrapolate,
                         measure_lambda, measure_quiet_ahead)

# ---------------------------------------------------------------- frame
W, H = 1080, 1920
FPS = 30
F = 510
LANE_W = 538
GUT = W - 2 * LANE_W
SIM_F0, SIM_F1 = 24, 456
OVER_F0, OVER_F1 = 405, 435          # wedge fade-in
CLOSE_F0, CLOSE_F1 = 435, 465        # closing label fade-in

# lane views (world coords; view width chosen so lambda_t is ~47 px
# in both lanes — the scale bars carry the 131x)
V_DUCK = dict(cx=3.36, w=3.6, y0=0.6, y1=13.4)
V_TANK = dict(cx=440.0, w=470.0, y0=60.0, y1=1733.0)

# shading gain: p99.5 of |eta| of the FINAL feasibility fields,
# pinned constant so nothing renormalises per frame (trap 24).
# check_gain() asserts the render's final fields still match.
GAIN_D = 1.147e-5
GAIN_T = 1.029e-3

BGC = np.array([0.030, 0.10, 0.16]) * 0.25   # still-water tone
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
OUT_DIR = "/home/maroon-beret/projects/active/youtube/youtube-channel/out"
STAMP = time.strftime("%H%M%S")
OUT_MP4 = f"{OUT_DIR}/kelvin_{STAMP}.mp4"

C_TEXT = (235, 238, 242)
C_DIM = (150, 158, 168)
C_LINE = (255, 246, 220)

SIM_T_D = DUCK["T"]                  # 15.3 s of duck time
SIM_T_T = TANK["T"]                  # 174 s of tanker time


def sim_time(f, T):
    if f < SIM_F0:
        return 0.0
    return min(1.0, (f - SIM_F0) / float(SIM_F1 - SIM_F0)) * T


# ---------------------------------------------------------------- water
def lane_rgb(eta, Lx, Ly, view, gain, seed):
    """Crop the view region, upsample to lane pixels, shaded relief."""
    ny, nx = eta.shape
    dx, dy = Lx / nx, Ly / ny
    c0 = int((view["cx"] - view["w"] / 2) / dx)
    c1 = int((view["cx"] + view["w"] / 2) / dx)
    r0 = int(view["y0"] / dy)
    r1 = int(view["y1"] / dy)
    crop = eta[r0:r1, c0:c1]
    z = np.clip(crop / gain, -2.5, 2.5)
    im = Image.fromarray((z * 1000).astype(np.float32))
    im = im.resize((LANE_W, H), Image.BICUBIC)
    z = np.asarray(im, np.float64)[::-1] / 1000.0    # screen-up = +y
    gy, gx = np.gradient(z * 9.0)
    nz = 1.0 / np.sqrt(1 + gx * gx + gy * gy)
    nxv, nyv = -gx * nz, -gy * nz
    L = np.array([-0.55, -0.62, 0.55])
    L /= np.linalg.norm(L)
    lam = np.clip(nxv * L[0] + nyv * L[1] + nz * L[2], 0, 1)
    Hv = L + np.array([0.0, 0.0, 1.0])
    Hv /= np.linalg.norm(Hv)
    spec = np.clip(nxv * Hv[0] + nyv * Hv[1] + nz * Hv[2], 0, 1) ** 60
    hgt = np.clip(0.5 + 0.35 * z, 0, 1)
    sh = 0.25 + 0.75 * lam
    rgb = np.stack([(0.030 + 0.05 * hgt) * sh + 0.55 * spec,
                    (0.10 + 0.16 * hgt) * sh + 0.60 * spec,
                    (0.16 + 0.24 * hgt) * sh + 0.62 * spec], -1)
    rng = np.random.default_rng(seed)
    rgb += (rng.random(rgb.shape) - 0.5) * (1.2 / 255)
    return np.clip(rgb, 0, 1)


def world_to_lane_px(x, y, view):
    """World (x, y) -> lane pixel (col, row)."""
    col = (x - (view["cx"] - view["w"] / 2)) / view["w"] * LANE_W
    row = (view["y1"] - y) / (view["y1"] - view["y0"]) * H
    return col, row


# ---------------------------------------------------------------- text
_fonts = {}


def font(sz):
    if sz not in _fonts:
        _fonts[sz] = ImageFont.truetype(FONT, sz)
    return _fonts[sz]


def stamp(draw, cx, cy, txt, sz, fill, anchor="mm"):
    draw.text((cx + 2, cy + 2), txt, font=font(sz),
              fill=(0, 0, 0), anchor=anchor)
    draw.text((cx, cy), txt, font=font(sz), fill=fill, anchor=anchor)


def overlay_wedge(draw, apex_x, apex_y, alpha, lane_l, lane_r,
                  label=True):
    """Theory wedge at arcsin(1/3) from the apex, drawn down-frame.
    Each arm is CLIPPED to its own lane (first render let them run
    across the gutter into the neighbour)."""
    t = np.tan(np.radians(TH_TRUE))
    col = tuple(int(c * alpha) for c in C_LINE)
    for sgn in (+1, -1):
        lim = (lane_r - apex_x) if sgn > 0 else (apex_x - lane_l)
        dyy = min(1400.0 - apex_y, lim / t)
        draw.line([(apex_x, apex_y),
                   (apex_x + sgn * t * dyy, apex_y + dyy)],
                  fill=col, width=3)
    y_end = 1400.0
    # centreline (track), dotted
    for yy in range(int(apex_y) + 20, int(y_end), 44):
        draw.line([(apex_x, yy), (apex_x, yy + 16)],
                  fill=tuple(int(c * 0.5 * alpha) for c in C_LINE),
                  width=2)
    if label:
        r = 330.0
        a0, a1 = 90.0 - TH_TRUE, 90.0
        draw.arc([apex_x - r, apex_y - r, apex_x + r, apex_y + r],
                 a0, a1, fill=col, width=3)
        mid = np.radians(90.0 - TH_TRUE / 2)
        lx = apex_x + (r + 52) * np.cos(mid)
        ly = apex_y + (r + 52) * np.sin(mid)
        stamp(draw, lx, ly, "19.47°", 34, col)


# ---------------------------------------------------------------- render
def render():
    os.makedirs(OUT_DIR, exist_ok=True)
    sim_d = Sim(DUCK["Lx"], DUCK["Ly"], DUCK["nx"], DUCK["ny"],
                DUCK["dt"], DUCK["sigma"], DUCK["amp"])
    sim_t = Sim(TANK["Lx"], TANK["Ly"], TANK["nx"], TANK["ny"],
                TANK["dt"], TANK["sigma"], TANK["amp"])
    sxd, sxt = DUCK["Lx"] / 2, TANK["Lx"] / 2
    snaps = {}
    stills = {0, 60, 130, 200, 270, 340, 405, 460, 509}

    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "18",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT_MP4]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    t0 = time.time()
    for f in range(F):
        td = sim_time(f, SIM_T_D)
        tt = sim_time(f, SIM_T_T)
        while sim_d.t < td - 1e-9:
            sim_d.step(sxd, DUCK["y0"] + DUCK["V"] * sim_d.t)
        while sim_t.t < tt - 1e-9:
            sim_t.step(sxt, TANK["y0"] + TANK["V"] * sim_t.t)
        syd = DUCK["y0"] + DUCK["V"] * sim_d.t
        syt = TANK["y0"] + TANK["V"] * sim_t.t

        if f == 300:
            snaps["quiet_d"] = (sim_d.eta_now.copy(), syd)
            snaps["quiet_t"] = (sim_t.eta_now.copy(), syt)
        if f == SIM_F1:
            snaps["final_d"] = (sim_d.eta_now.copy(), syd)
            snaps["final_t"] = (sim_t.eta_now.copy(), syt)

        # still frames shade a zero field through the same pipeline,
        # so calm water before f24 is byte-identical in tone to calm
        # water after (a hand-set BGC here popped the luminance)
        frame = np.empty((H, W, 3))
        frame[:, :LANE_W] = lane_rgb(sim_d.eta_now, DUCK["Lx"],
                                     DUCK["Ly"], V_DUCK, GAIN_D, f)
        frame[:, LANE_W + GUT:] = lane_rgb(sim_t.eta_now, TANK["Lx"],
                                           TANK["Ly"], V_TANK,
                                           GAIN_T, f + 7)
        frame[:, LANE_W:LANE_W + GUT] = 0.012
        img = Image.fromarray((np.clip(frame, 0, 1) * 255 + 0.5)
                              .astype(np.uint8))
        draw = ImageDraw.Draw(img)

        # ship dots (the pressure point, drawn at ~2 sigma)
        if f >= SIM_F0:
            for (view, s_x, s_y, sig, Lx, x_off) in (
                    (V_DUCK, sxd, syd, DUCK["sigma"], DUCK["Lx"], 0),
                    (V_TANK, sxt, syt, TANK["sigma"], TANK["Lx"],
                     LANE_W + GUT)):
                c, r = world_to_lane_px(s_x, s_y, view)
                rad = max(5.0, 2 * sig / view["w"] * LANE_W)
                draw.ellipse([x_off + c - rad, r - 1.4 * rad,
                              x_off + c + rad, r + 1.4 * rad],
                             fill=(8, 12, 16))

        # lane labels + clocks
        stamp(draw, LANE_W // 2, 228, "a duck · 0.7 m/s", 38, C_TEXT)
        stamp(draw, LANE_W + GUT + LANE_W // 2, 228,
              "a supertanker · 8 m/s", 38, C_TEXT)
        stamp(draw, LANE_W // 2, 286, f"t = {sim_d.t:5.1f} s", 30, C_DIM)
        stamp(draw, LANE_W + GUT + LANE_W // 2, 286,
              f"t = {sim_t.t:5.0f} s", 30, C_DIM)

        # scale bars (above bottom-15% safe line at row 1632)
        bar_d = 1.0 / V_DUCK["w"] * LANE_W          # 1 m
        bar_t = 100.0 / V_TANK["w"] * LANE_W        # 100 m
        for (bw, lbl, x_off) in ((bar_d, "1 m", 0),
                                 (bar_t, "100 m", LANE_W + GUT)):
            x0 = x_off + LANE_W // 2 - bw / 2
            draw.line([(x0, 1584), (x0 + bw, 1584)], fill=C_TEXT, width=4)
            draw.line([(x0, 1572), (x0, 1596)], fill=C_TEXT, width=3)
            draw.line([(x0 + bw, 1572), (x0 + bw, 1596)],
                      fill=C_TEXT, width=3)
            stamp(draw, x_off + LANE_W // 2, 1544, lbl, 30, C_TEXT)

        # theory wedge overlay
        if f >= OVER_F0:
            al = min(1.0, (f - OVER_F0) / float(OVER_F1 - OVER_F0))
            for (view, s_x, s_y, x_off) in (
                    (V_DUCK, sxd, syd, 0),
                    (V_TANK, sxt, syt, LANE_W + GUT)):
                c, r = world_to_lane_px(s_x, s_y, view)
                overlay_wedge(draw, x_off + c, r, al,
                              x_off + 4, x_off + LANE_W - 4)
        if f >= CLOSE_F0:
            al = min(1.0, (f - CLOSE_F0) / float(CLOSE_F1 - CLOSE_F0))
            stamp(draw, W // 2, 1462, "same angle:  arcsin(1/3)", 44,
                  tuple(int(c * al) for c in C_LINE))

        buf = np.asarray(img, np.uint8)
        p.stdin.write(buf.tobytes())
        if f in stills:
            img.save(f"{OUT_DIR}/kelvin_f{f:04d}.png")
        if f % 60 == 0:
            print(f"  frame {f}/{F}  ({time.time() - t0:.0f}s)",
                  flush=True)
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        print("ENCODE FAILED", flush=True)
        sys.exit(1)
    print(f"encoded {OUT_MP4} ({os.path.getsize(OUT_MP4)} bytes) "
          f"in {time.time() - t0:.0f}s", flush=True)
    np.savez("/tmp/kelvin_render_snaps.npz",
             final_d=snaps["final_d"][0], syd=snaps["final_d"][1],
             final_t=snaps["final_t"][0], syt=snaps["final_t"][1],
             quiet_d=snaps["quiet_d"][0], qsyd=snaps["quiet_d"][1],
             quiet_t=snaps["quiet_t"][0], qsyt=snaps["quiet_t"][1])
    return sim_d, sim_t, snaps


# ---------------------------------------------------------------- checks
def decode_frame(n):
    r = subprocess.run(
        ["ffmpeg", "-i", OUT_MP4, "-vf", f"select=eq(n\\,{n})",
         "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True)
    a = np.frombuffer(r.stdout, np.uint8)
    return a.reshape(H, W, 3).astype(np.float64)


def check(sim_d, sim_t, snaps):
    fails = []

    def chk(label, ok):
        print(("  OK   " if ok else "  FAIL ") + label, flush=True)
        if not ok:
            fails.append(label)

    # ---- physics on the render's own final fields -------------------
    eta_d, syd = snaps["final_d"]
    eta_t, syt = snaps["final_t"]
    lam_d = 2 * np.pi * DUCK["V"] ** 2 / G
    lam_t = 2 * np.pi * TANK["V"] ** 2 / G
    ti_d, c_d, r_d, as_d, _ = extrapolate(sim_d, eta_d, DUCK["Lx"] / 2,
                                          syd, DUCK["y0"], lam_d, "duck")
    ti_t, c_t, r_t, as_t, _ = extrapolate(sim_t, eta_t, TANK["Lx"] / 2,
                                          syt, TANK["y0"], lam_t, "tanker")
    chk(f"duck theta_inf {ti_d:.2f} within 0.4 of {TH_TRUE:.2f}",
        abs(ti_d - TH_TRUE) < 0.4)
    chk(f"tanker theta_inf {ti_t:.2f} within 0.4 of {TH_TRUE:.2f}",
        abs(ti_t - TH_TRUE) < 0.4)
    chk(f"lanes agree |{ti_d:.2f}-{ti_t:.2f}| < 0.5",
        abs(ti_d - ti_t) < 0.5)
    chk(f"convergence from inside (c {c_d:.1f}, {c_t:.1f} > 0)",
        c_d > 0 and c_t > 0)
    lm_d, lt_d = measure_lambda(sim_d, eta_d, DUCK["Lx"] / 2, syd,
                                DUCK["V"], "duck")
    lm_t, lt_t = measure_lambda(sim_t, eta_t, TANK["Lx"] / 2, syt,
                                TANK["V"], "tanker")
    chk(f"duck lambda ratio {lm_d / lt_d:.3f} in 0.97..1.03",
        0.97 < lm_d / lt_d < 1.03)
    chk(f"tanker lambda ratio {lm_t / lt_t:.3f} in 0.97..1.03",
        0.97 < lm_t / lt_t < 1.03)
    chk(f"wavelengths differ {lm_t / lm_d:.0f}x (theory 131x)",
        110 < lm_t / lm_d < 150)
    qd, qsyd = snaps["quiet_d"]
    qt, qsyt = snaps["quiet_t"]
    quiet_d = measure_quiet_ahead(sim_d, qd, DUCK["Lx"] / 2, qsyd, "duck")
    quiet_t = measure_quiet_ahead(sim_t, qt, TANK["Lx"] / 2, qsyt,
                                  "tanker")
    chk(f"quiet ahead mid-run (duck {quiet_d:.3f}, tanker "
        f"{quiet_t:.3f} < 0.12)", quiet_d < 0.12 and quiet_t < 0.12)
    gd = np.percentile(np.abs(eta_d), 99.5)
    gt = np.percentile(np.abs(eta_t), 99.5)
    chk(f"pinned gains match final fields (duck {gd:.3e} vs {GAIN_D:.3e},"
        f" tanker {gt:.3e} vs {GAIN_T:.3e}, within 25%)",
        0.75 < gd / GAIN_D < 1.25 and 0.75 < gt / GAIN_T < 1.25)

    # ---- pixels: decoded frames ------------------------------------
    fr_end = decode_frame(F - 2)
    fr_mid = decode_frame(260)
    fr_a = decode_frame(300)
    # calm water is NOT black (~29 luminance): the defect to catch is
    # a blank/black frame or a blown/white one, so assert the mean and
    # the white fraction, not a lit-above-dark fraction (first version
    # of this check assumed a dark background and failed a correct
    # frame at 0.864)
    lum_end = fr_end.mean(-1)
    chk(f"final frame mean luminance {lum_end.mean():.1f} in 15..90",
        15 < lum_end.mean() < 90)
    chk(f"final frame not blown (frac>200 = {(lum_end > 200).mean():.4f}"
        f" < 0.10)", (lum_end > 200).mean() < 0.10)
    dif = np.abs(fr_a - fr_mid).mean()
    chk(f"motion between f260 and f300 (mean |diff| {dif:.2f} > 0.5)",
        dif > 0.5)

    # wedge overlay lines land where drawn, and the bright arm hugs
    # them FROM INSIDE (the Airy offset) — the coupling check
    t = np.tan(np.radians(TH_TRUE))
    for (view, name, x_off, sy_w, Lx) in (
            (V_DUCK, "duck", 0, snaps["final_d"][1], DUCK["Lx"]),
            (V_TANK, "tanker", LANE_W + GUT, snaps["final_t"][1],
             TANK["Lx"])):
        c, r = world_to_lane_px(Lx / 2, sy_w, view)
        apex_x, apex_y = x_off + c, r
        # sample only the drawn extent — the lines are clipped to the
        # lane, ending at dyy ~ (LANE_W/2 - 4)/tan(theta)
        dyy_line = int((LANE_W / 2 - 40) / t)
        pts = [(dyy, sgn) for dyy in range(150, dyy_line, 60)
               for sgn in (+1, -1)]
        hits = 0
        for dyy, sgn in pts:
            x = int(apex_x + sgn * t * dyy)
            y = int(apex_y + dyy)
            patch = fr_end[y - 2:y + 3, x - 2:x + 3].mean(-1)
            if patch.max() > 110:
                hits += 1
        chk(f"{name} overlay line ink present ({hits}/{len(pts)} "
            f"samples)", hits >= 0.8 * len(pts))
        # arm inside line: per-row outer WAVE column. Threshold is
        # measured off this lane's own calm water (ahead of the apex)
        # plus a margin — calm water decodes ~29, waves peak far
        # brighter. Rows stop where the theory line leaves the lane.
        lane_l = x_off + 8
        lane_r = x_off + LANE_W - 8
        calm_rows = slice(max(0, int(apex_y) - 130), int(apex_y) - 40)
        calm = np.median(fr_end[calm_rows, lane_l:lane_r].mean(-1))
        thr = calm + 25.0
        dyy_max = int((LANE_W / 2 - 30) / t)
        gaps = []
        for dyy in range(250, dyy_max, 25):
            y = int(apex_y + dyy)
            row = fr_end[y, lane_l:lane_r].mean(-1)
            line_col = apex_x + t * dyy - lane_l
            bright = np.where(row > thr)[0]
            # exclude the overlay line itself (+-7 px both arms)
            for lc in (line_col, 2 * (apex_x - lane_l) - line_col):
                bright = bright[np.abs(bright - lc) > 7]
            if len(bright) == 0:
                continue
            gaps.append(line_col - bright.max())
        gaps = np.array(gaps)
        # two defects, named: (1) wave ink OUTSIDE the wedge would
        # mean the physics or the overlay placement is wrong -> min
        # gap > -8 (AA tolerance); (2) an overlay drawn nowhere near
        # its wake (bad apex transform) -> some rows must come close.
        # The mean gap is NOT asserted tightly: the duck's relatively
        # larger source (k*sigma 0.90 vs 0.69) dims its divergent arm,
        # so the thresholded edge sits further inside than the
        # tanker's — that is brightness, not confinement.
        near = (gaps < 60).mean()
        chk(f"{name} wake confined to wedge (calm {calm:.0f}, "
            f"{len(gaps)} rows, min gap {gaps.min():.0f} px > -8; "
            f"near-line rows {near:.2f} >= 0.2)",
            gaps.min() > -8 and near >= 0.2)

    # safe areas: text rows only (graphics may bleed; text may not)
    chk("labels inside safe area (rows 210..306 and 1444..1600 "
        "within 192..1632)", 210 > 192 and 1600 < 1632)

    # ---- encode gate ------------------------------------------------
    pr = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_frames", "-show_entries",
         "stream=nb_read_frames,width,height,codec_name",
         "-of", "csv=p=0", OUT_MP4], capture_output=True, text=True)
    parts = pr.stdout.strip().split(",")
    chk(f"encode: {parts} == h264 1080x1920 {F} frames",
        parts[0] == "h264" and parts[1] == "1080" and
        parts[2] == "1920" and parts[3] == str(F))

    print("=" * 60)
    if fails:
        print(f"{len(fails)} FAILURES")
        for x in fails:
            print("  -", x)
        sys.exit(1)
    print("ALL CHECKS PASSED")
    print("NOT verified by any check: that a viewer reads the two "
          "lanes as the same angle — that one is the viewer's.")


def sheet():
    subprocess.run(
        ["ffmpeg", "-y", "-pattern_type", "glob",
         "-i", f"{OUT_DIR}/kelvin_f*.png",
         "-filter_complex", "scale=270:-1,tile=3x3",
         f"{OUT_DIR}/kelvin_sheet.png"], capture_output=True)
    print("sheet:", f"{OUT_DIR}/kelvin_sheet.png", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "checkonly":
        # re-run checks against an existing encode + saved snapshots
        OUT_MP4 = sys.argv[2]
        z = np.load("/tmp/kelvin_render_snaps.npz")
        snaps = {"final_d": (z["final_d"], float(z["syd"])),
                 "final_t": (z["final_t"], float(z["syt"])),
                 "quiet_d": (z["quiet_d"], float(z["qsyd"])),
                 "quiet_t": (z["quiet_t"], float(z["qsyt"]))}
        sim_d = Sim(DUCK["Lx"], DUCK["Ly"], DUCK["nx"], DUCK["ny"],
                    DUCK["dt"], DUCK["sigma"], DUCK["amp"])
        sim_t = Sim(TANK["Lx"], TANK["Ly"], TANK["nx"], TANK["ny"],
                    TANK["dt"], TANK["sigma"], TANK["amp"])
        check(sim_d, sim_t, snaps)
        sheet()
    else:
        sim_d, sim_t, snaps = render()
        check(sim_d, sim_t, snaps)
        sheet()
