#!/usr/bin/env python3
"""
RAIN -- one drop, four hundred metres, twice. Once with air, once without.

A 5 mm raindrop falls at 9 m/s. That is not "eventually" -- it is nearly the
whole fall. Under quadratic drag it is within one percent of that speed after
about sixteen metres, and the other three hundred and eighty-four change
nothing at all.

Take the air away and the same drop arrives at sqrt(2 g h) = 88.6 m/s.

Both lanes lay a tick every second of real fall time. The air lane's ticks are
evenly spaced, all the way down, because the speed never changes. The no-air
lane's ticks come apart like Galileo's odd numbers. That is the whole physics
and it needs no words: one ladder with equal rungs, one ladder that stretches.

Terminal velocity is measured, not modelled -- Wikipedia "Rain": a 5 mm drop
at sea level in still air lands at "9 m/s (30 ft/s) or 32 km/h (20 mph)".
The SHAPE of the approach is a model: constant-drag-coefficient quadratic
drag, v = vt*tanh(g t / vt). The sixteen metres depends on that model. The
two impact speeds do not.

    python3 scripts/rain.py --check
    python3 scripts/rain.py --stills /tmp/rain
    python3 scripts/rain.py --out content/rain.mp4

numpy + pycairo + ffmpeg.
"""

import argparse
import math
import subprocess
import sys

import cairo
import numpy as np

# ------------------------------------------------------------------ physics

G      = 9.81      # m/s2
V_T    = 9.0       # m/s -- MEASURED. 5 mm drop, sea level, still air.
H0     = 400.0     # m   -- a low cloud base
D_MM   = 5.0       # mm  -- drop diameter, for the description only
TICK_S = 1.0       # s of real fall time between rungs


def fall_air(t):
    """Quadratic drag, constant Cd, tuned so terminal velocity is V_T.

    log(cosh(x)) overflows in float64 past x ~ 710, so use the asymptote
    x - log 2, which is exact to a part in 1e300 by then.
    """
    x = np.asarray(G * t / V_T, dtype=float)
    lc = np.where(x < 30.0, np.log(np.cosh(np.minimum(x, 30.0))),
                  x - math.log(2.0))
    return (V_T ** 2 / G) * lc, V_T * np.tanh(x)


def fall_vac(t):
    return 0.5 * G * t ** 2, G * t


def land_time(f, h=H0):
    lo, hi = 0.0, 400.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid)[0] < h:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def dist_to_frac(frac):
    """Metres fallen when the air drop first reaches frac of V_T."""
    x = np.arctanh(frac)
    return (V_T ** 2 / G) * math.log(math.cosh(x))


T_AIR = land_time(fall_air)
T_VAC = land_time(fall_vac)
V_VAC = fall_vac(T_VAC)[1]
D_99 = dist_to_frac(0.99)

# ------------------------------------------------------------------ picture

W, H = 1080, 1920
FPS = 30
SS = 2

# safe area for a Short: nothing that matters outside these rows
SAFE_TOP, SAFE_BOT = 192, 1656

Y_LABEL = 250
Y_READ = 330
Y_LEGEND = 388
Y_CLOUD = 424
Y_GROUND = 1450
Y_TXT1, Y_TXT2 = 1540, 1614

PX_M = (Y_GROUND - Y_CLOUD) / H0          # px per metre

LANE = (400.0, 810.0)                     # x centres: air, no air
TICK_W = 168.0                            # full width of a rung

LEAD = 0.30                               # s of stillness before release
RUN = 7.0                                 # s of video for the whole air fall
HOLD = 2.10                               # s after it lands
K = T_AIR / RUN                           # real seconds per video second
DUR = LEAD + RUN + HOLD
N_FRAME = int(round(DUR * FPS))

# cairo wants 0..1 floats. 0..255 clamps every channel to white and no
# geometry check will ever notice. (RENDERER.md trap 55.)
BG     = (0.030, 0.034, 0.043)
CLOUD  = (0.150, 0.162, 0.196)
GROUND = (0.300, 0.312, 0.342)
DIM    = (0.400, 0.420, 0.462)
TEXT   = (0.870, 0.885, 0.905)
AIR    = (0.560, 0.780, 0.960)
VAC    = (0.960, 0.330, 0.250)
LANE_C = (AIR, VAC)

MONO = "DejaVu Sans Mono"
END_LINES = ("same drop. same cloud.",
             "the air is why you can stand in it.")


def y_of(h):
    """Height above ground in metres -> pixel row."""
    return Y_GROUND - h * PX_M


def ticks(f, t_land):
    """Metres fallen at each whole second of real fall time, while airborne."""
    out, k = [], 1
    while k * TICK_S < t_land:
        out.append(float(f(k * TICK_S)[0]))
        k += 1
    return out


TICK_AIR = ticks(fall_air, T_AIR)
TICK_VAC = ticks(fall_vac, T_VAC)


# -------------------------------------------------------------------- draw

def text(cr, x, y, s, rgb, size, align='c', alpha=1.0, weight=cairo.FONT_WEIGHT_NORMAL):
    cr.select_font_face(MONO, cairo.FONT_SLANT_NORMAL, weight)
    cr.set_font_size(size * SS)
    xb, yb, tw, th, _, _ = cr.text_extents(s)
    ox = {'c': -tw / 2 - xb, 'l': -xb, 'r': -tw - xb}[align]
    cr.set_source_rgba(*rgb, alpha)
    cr.move_to(x * SS + ox, y * SS)
    cr.show_text(s)


def hline(cr, y, rgb, w, alpha=1.0, x0=0.0, x1=float(W)):
    cr.set_source_rgba(*rgb, alpha)
    cr.set_line_width(w * SS)
    cr.set_line_cap(cairo.LINE_CAP_BUTT)
    cr.move_to(x0 * SS, y * SS)
    cr.line_to(x1 * SS, y * SS)
    cr.stroke()


def render_frame(surf, cr, i):
    tv = i / FPS
    tf = max(0.0, (tv - LEAD) * K)            # real fall seconds elapsed

    cr.set_source_rgb(*BG)
    cr.paint()
    cr.set_antialias(cairo.ANTIALIAS_BEST)
    cr.set_line_cap(cairo.LINE_CAP_ROUND)

    # the sky between cloud and ground, a shade above the surround
    cr.set_source_rgba(1.0, 1.0, 1.0, 0.012)
    cr.rectangle(0, Y_CLOUD * SS, W * SS, (Y_GROUND - Y_CLOUD) * SS)
    cr.fill()

    hline(cr, Y_CLOUD, CLOUD, 3.0)
    hline(cr, Y_GROUND, GROUND, 4.0)

    # height scale, far left
    for m in (400, 300, 200, 100, 0):
        y = y_of(m)
        hline(cr, y, DIM, 1.4, 0.34, 46.0, 86.0)
        text(cr, 96, y + 11, f"{m}", DIM, 26, 'l')
    text(cr, 46, Y_CLOUD - 74, "metres", DIM, 24, 'l')
    text(cr, W / 2, Y_LEGEND, "one rung = one second of falling",
         DIM, 30, 'c', 0.95)

    for j, (cx, col) in enumerate(zip(LANE, LANE_C)):
        f, t_land, tk = ((fall_air, T_AIR, TICK_AIR),
                         (fall_vac, T_VAC, TICK_VAC))[j]
        t = min(tf, t_land)
        d, v = f(t)
        d, v = float(d), float(v)
        y = y_of(H0 - d)
        landed = tf >= t_land

        text(cr, cx, Y_LABEL, ("air", "no air")[j], col, 40, 'c', 0.92)
        text(cr, cx, Y_READ, f"{v:4.1f} m/s", TEXT if not landed else col,
             46, 'c', 0.95)

        # the fallen part of the lane
        if tf > 0:
            cr.set_source_rgba(*col, 0.20)
            cr.set_line_width(2.2 * SS)
            cr.move_to(cx * SS, Y_CLOUD * SS)
            cr.line_to(cx * SS, y * SS)
            cr.stroke()

        # rungs -- one per second of real fall time
        for dm in tk:
            if dm > d + 1e-9:
                break
            ry = y_of(H0 - dm)
            cr.set_source_rgba(*col, 0.80)
            cr.set_line_width(2.6 * SS)
            cr.move_to((cx - TICK_W / 2) * SS, ry * SS)
            cr.line_to((cx + TICK_W / 2) * SS, ry * SS)
            cr.stroke()

        if not landed:
            # the drop, drawn as the distance it covers in one frame
            t0 = max(0.0, t - K / FPS)
            y0 = y_of(H0 - float(f(t0)[0]))
            cr.set_source_rgba(*col, 0.55)
            cr.set_line_width(14.0 * SS)
            cr.move_to(cx * SS, min(y0, y - 6.0) * SS)
            cr.line_to(cx * SS, y * SS)
            cr.stroke()
            cr.set_source_rgb(*col)
            cr.arc(cx * SS, y * SS, 8.0 * SS, 0, 2 * math.pi)
            cr.fill()
        else:
            cr.set_source_rgba(*col, 0.85)
            cr.set_line_width(6.0 * SS)
            cr.move_to((cx - 34) * SS, (Y_GROUND - 26) * SS)
            cr.line_to(cx * SS, Y_GROUND * SS)
            cr.line_to((cx + 34) * SS, (Y_GROUND - 26) * SS)
            cr.stroke()

    # the two lines, once the patient one is down
    a = min(1.0, max(0.0, (tv - (LEAD + RUN)) / 0.45))
    if a > 0:
        for y, s in zip((Y_TXT1, Y_TXT2), END_LINES):
            text(cr, W / 2, y, s, TEXT, 46, 'c', a)

    buf = np.ndarray(shape=(H * SS, W * SS, 4), dtype=np.uint8,
                     buffer=surf.get_data())
    img = buf[:, :, [2, 1, 0]].astype(np.float32) / 255.0
    return img.reshape(H, SS, W, SS, 3).mean(axis=(1, 3))


def to8(img):
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# ------------------------------------------------------------------ checks

OK = True


def t(cond, msg):
    global OK
    print(("  ok   " if cond else "  FAIL ") + msg)
    OK = OK and bool(cond)


def run_checks(surf, cr):
    print("\nphysics")
    t(abs(fall_air(1e4)[1] - V_T) < 1e-9,
      f"air drop tops out at the measured {V_T:.1f} m/s")
    t(abs(V_VAC - math.sqrt(2 * G * H0)) < 1e-6,
      f"no-air drop arrives at sqrt(2gh) = {V_VAC:.2f} m/s "
      f"({V_VAC * 3.6:.1f} km/h), {V_VAC / V_T:.1f}x the other one "
      f"and {(V_VAC / V_T) ** 2:.0f}x the energy")

    # numeric integration must reproduce the closed form it is drawn from
    dt, y, v = 1e-4, 0.0, 0.0
    while y < H0:
        v += (G - G * (v / V_T) ** 2) * dt
        y += v * dt
    t(abs(y - H0) < 0.02 and abs(v - fall_air(T_AIR)[1]) < 0.01,
      f"euler integration lands with the closed form: {v:.3f} vs "
      f"{float(fall_air(T_AIR)[1]):.3f} m/s")
    t(abs(T_AIR - 45.08) < 0.05 and abs(T_VAC - 9.031) < 0.005,
      f"fall times {T_AIR:.2f} s with air, {T_VAC:.2f} s without "
      f"-- {T_AIR / T_VAC:.2f}x longer")
    t(15.0 < D_99 < 17.0,
      f"within 1% of terminal after {D_99:.1f} m -- the other "
      f"{H0 - D_99:.0f} m change nothing")

    print("\nthe ladders, from the model")
    ga = np.diff([0.0] + TICK_AIR)
    gv = np.diff([0.0] + TICK_VAC)
    tail = ga[6:]
    t(tail.std() / tail.mean() < 0.002,
      f"air rungs are even: {len(ga)} of them, last {len(tail)} at "
      f"{tail.mean():.3f} m +/- {tail.std() * 1000:.2f} mm")
    t(np.all(np.diff(gv) > 0) and gv[-1] / gv[0] > 10,
      f"no-air rungs come apart, {gv[0]:.2f} m to {gv[-1]:.2f} m "
      f"({gv[-1] / gv[0]:.0f}x)")
    t(abs(TICK_AIR[0] - TICK_VAC[0]) < 1.0,
      f"and for the first second they fall together -- {TICK_AIR[0]:.2f} m "
      f"vs {TICK_VAC[0]:.2f} m, {abs(TICK_AIR[0] - TICK_VAC[0]) * 100:.0f} cm "
      f"apart")

    print("\nthe picture")
    top = y_of(H0)
    t(SAFE_TOP < Y_LABEL and Y_GROUND < SAFE_BOT and Y_TXT2 < SAFE_BOT,
      f"everything that matters sits in rows {SAFE_TOP}..{SAFE_BOT} "
      f"(labels {Y_LABEL}, ground {Y_GROUND}, last line {Y_TXT2})")
    t(abs(top - Y_CLOUD) < 0.51,
      f"the cloud line is where 400 m maps to ({top:.1f} vs {Y_CLOUD})")
    t(ga[1:].min() * PX_M > 14.0,
      f"air rungs are {ga[-1] * PX_M:.1f} px apart -- readable as separate "
      f"rungs, not a smear")
    cr.select_font_face(MONO, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    cr.set_font_size(46 * SS)
    wid = max(cr.text_extents(s)[2] / SS for s in END_LINES)
    t(wid < W - 70,
      f"the two closing lines are {wid:.0f} px wide in a {W} px frame")
    t(N_FRAME / FPS < 180.0,
      f"{DUR:.2f} s, {N_FRAME} frames, real time compressed {K:.2f}x")

    # ---- and now read the pixels, because the arithmetic above would pass
    #      just as happily on a blank frame (trap 56)
    marks = {"release": int((LEAD + 0.35) * FPS),
             "vac down": int((LEAD + T_VAC / K + 0.30) * FPS),
             "end": N_FRAME - 1}
    print("\nthe pixels")
    for name, i in marks.items():
        img = render_frame(surf, cr, i)
        bg = np.array(BG, np.float32)
        lit = (np.abs(img - bg).max(2) > 0.035)
        rows = np.where(lit.any(1))[0]
        t(0.002 < lit.mean() < 0.30 and rows[0] >= SAFE_TOP - 40
          and rows[-1] <= SAFE_BOT,
          f"{name}: {100 * lit.mean():.2f}% of pixels lit, rows "
          f"{rows[0]}..{rows[-1]}")

    # the claim of the whole video, measured off the last frame rather than
    # asserted: one ladder with equal rungs, one that stretches
    img = render_frame(surf, cr, N_FRAME - 1)
    # sample a strip that ONLY rungs can occupy: inside the rung's half-width,
    # clear of the 2 px lane line, and between the cloud line and far enough
    # above the ground to miss the landing chevron. Labels, readouts and the
    # closing lines all live outside this band, and the first cut of this
    # check caught all four of them and reported nonsense.
    r0, r1 = int(Y_CLOUD) + 6, int(Y_GROUND) - 40
    for j, (name, grow) in enumerate((("air", False), ("no air", True))):
        cx = int(LANE[j])
        col = img[r0:r1, cx - 34:cx - 20, :].mean(2)
        on = col.max(1) > 0.20
        ys = np.where(on)[0]
        runs, cur = [], [ys[0]]
        for y in ys[1:]:
            if y - cur[-1] <= 2:
                cur.append(y)
            else:
                runs.append(sum(cur) / len(cur))
                cur = [y]
        runs.append(sum(cur) / len(cur))
        g = np.diff(np.array(runs))
        g = g[g > 6.0]
        if grow:
            t(len(g) >= 6 and np.all(np.diff(g) > 1.0) and g[-1] / g[0] > 4,
              f"{name} lane, measured off the frame: {len(runs)} rungs, gaps "
              f"{g[0]:.0f} px growing to {g[-1]:.0f} px ({g[-1] / g[0]:.1f}x)")
        else:
            t(len(g) >= 30 and g.std() < 0.9,
              f"{name} lane, measured off the frame: {len(runs)} rungs, gaps "
              f"{g.mean():.2f} px +/- {g.std():.2f} px -- the video's whole "
              f"claim, read back out of the picture")

    print("\nALL CHECKS PASS" if OK else "\nSOMETHING FAILED")
    return 0 if OK else 1


# -------------------------------------------------------------------- runs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--stills')
    ap.add_argument('--out')
    args = ap.parse_args()

    print(f"{D_MM:.0f} mm drop, {H0:.0f} m of sky")
    print(f"  with air    {T_AIR:6.2f} s   {V_T:5.1f} m/s  "
          f"({V_T * 3.6:.0f} km/h)")
    print(f"  without air {T_VAC:6.2f} s   {V_VAC:5.1f} m/s  "
          f"({V_VAC * 3.6:.0f} km/h)")
    print(f"  99% of terminal after {D_99:.1f} m")
    print(f"  video {DUR:.2f} s, {N_FRAME} frames, time x{K:.2f}")

    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W * SS, H * SS)
    cr = cairo.Context(surf)

    if args.check:
        return run_checks(surf, cr)

    if args.stills:
        from PIL import Image
        for i in (0, int(LEAD * FPS) + 6, int((LEAD + 0.9) * FPS),
                  int((LEAD + 2.4) * FPS), int((LEAD + 4.5) * FPS),
                  N_FRAME - 1):
            Image.fromarray(to8(render_frame(surf, cr, i))).save(
                f"{args.stills}_{i:04d}.png")
            print(f"  still {i}")
        return 0

    if not args.out:
        print("nothing to do -- pass --check, --stills or --out")
        return 1

    cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{W}x{H}', '-r', str(FPS), '-i', 'pipe:0',
           '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', args.out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for i in range(N_FRAME):
        p.stdin.write(to8(render_frame(surf, cr, i)).tobytes())
        if i % 30 == 0:
            print(f"  frame {i}/{N_FRAME}", flush=True)
    p.stdin.close()
    p.wait()
    print(f"wrote {args.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
