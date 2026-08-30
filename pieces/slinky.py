"""SLINKY — drop a slinky and the bottom doesn't fall until the top
crashes into it.

A hanging slinky is a tug-of-war: every coil is held up by the coil above
and dragged down by the weight of everything below. Release the top and
nothing below the collapse front knows yet — the news travels down as a
wave while the top piles into a falling clump. The bottom coil hangs
motionless in midair until the clump lands on it.

The physics is a chain of N masses joined by zero-natural-length springs
(a close-wound tension spring, which is what a slinky is); coils that
meet merge perfectly inelastically into a rigid clump, momentum
conserved — the Calkin (1993) / Cross & Wheatland (2012) sticky-front
picture. Internal forces cancel in pairs, so the centre of mass free-falls
at exactly g from the first frame even while the bottom hangs — that
coupling is asserted (trap 66 family), and it yields the held-out check:
the collapse must finish exactly when the freely falling centre of mass
reaches the stack, t_c = sqrt(2*d_com/g), derived with no reference to the
simulation's contact machinery.

Structure: hang, drop in REAL TIME (0.26 s — a blink), an empty beat,
then the same trajectory again 25x slower. Both segments index one
precomputed trajectory; the re-hang frames are asserted byte-identical to
the opening hang frames.

Source: R. C. Cross & M. S. Wheatland, "Modeling a falling slinky",
Am. J. Phys. 80, 1051 (2012) — real slinkies measure ~0.25-0.3 s.
"""

import math
import os
import subprocess
import sys
import time

import cairo
import numpy as np

# ---------------------------------------------------------------- canvas
W, H = 1080, 1920
FPS = 30

def C(r, g, b):
    """Palette helper — trap 55: cairo wants 0..1 floats, stated once."""
    return (r / 255.0, g / 255.0, b / 255.0)

BG        = C(16, 15, 20)
BAR_COL   = C(110, 110, 118)
LINE_COL  = C(96, 92, 88)
TEXT_COL  = C(200, 198, 194)
WHITE_COL = C(250, 250, 250)     # the bottom coil — the one that waits

# ---------------------------------------------------------------- physics
G_ACC  = 9.81
N_COIL = 60
M_TOT  = 1.0                       # kg (cancels everywhere)
L_HANG = 1.0                       # m, top coil to bottom coil at rest
K_SLIN = M_TOT * G_ACC / (2 * L_HANG)      # whole-slinky stiffness
K_SEG  = K_SLIN * (N_COIL - 1)             # per-spring (series)
M_NODE = M_TOT / N_COIL
D_MIN  = 0.0004                    # m — coil contact spacing
DT     = 2.0e-5
T_SIM  = 0.45
STEPS  = int(round(T_SIM / DT))
# Contact is a perfectly inelastic sticky merge (the Calkin 1993 /
# Cross & Wheatland 2012 picture): coils that meet share momentum and
# move rigidly. A damped-penalty contact was tried first and produced a
# Newton's-cradle artifact — the compression wave reflecting off the free
# top of the clump relaunched the top coil upward by ~5 mm at every
# collision, which real slinkies suppress with pre-tension between
# touching turns. The sticky front is both simpler and truer to the
# cited model.

def equilibrium():
    """Exact discrete hanging equilibrium, top node at 0, x positive DOWN."""
    i = np.arange(N_COIL - 1)
    tension = M_NODE * G_ACC * (N_COIL - 1 - i)      # weight below spring i
    ext = tension / K_SEG
    x = np.concatenate([[0.0], np.cumsum(ext)])
    return x, ext

def simulate():
    """Release at t=0; semi-implicit Euler with an inelastic sticky front.

    Nodes 0..j-1 are the clump and move rigidly (spacing frozen at
    capture, so positions are never snapped and the COM never jumps).
    Merges conserve momentum exactly. Returns positions[STEPS+1, N]."""
    x, _ = equilibrium()
    v = np.zeros(N_COIL)
    j = 1                                  # clump size (top node alone)
    traj = np.empty((STEPS + 1, N_COIL))
    traj[0] = x
    for s in range(STEPS):
        gap = x[1:] - x[:-1]
        tens = K_SEG * np.maximum(gap, 0.0)          # tension-only
        tens[:max(j - 1, 0)] = 0.0                   # springs inside clump
        a = np.full(N_COIL, G_ACC)
        a[:-1] += tens / M_NODE
        a[1:]  -= tens / M_NODE
        if j > 1:
            a[:j] = a[:j].mean()                     # rigid clump
        v += a * DT
        x += v * DT
        while j < N_COIL and x[j] - x[j - 1] <= D_MIN:
            v_new = (j * v[0] + v[j]) / (j + 1)      # equal masses
            j += 1
            v[:j] = v_new
        traj[s + 1] = x
    return traj

# ------------------------------------------------------- timeline (frames)
# A: hang            f 0..20    (21 frames, t=0 state, string attached)
# B: real-time drop  f 21..32   (12 frames, t=(f-20)/FPS)
# C: empty hold      f 33..47   (15 frames)
# D: re-hang         f 48..59   (12 frames, byte-identical to A)
# E: 25x slow drop   f 60..270  (t=(f-59)/(FPS*SLOW))
SLOW   = 25
F_A0, F_B0, F_C0, F_D0, F_E0 = 0, 21, 33, 48, 60
N_FRAMES = 271
SHUTTER = 0.5

def frame_state(f):
    """(mode, t_phys, taps). mode: 'hang' | 'fall' | 'empty'. Pure in f."""
    if f < F_B0:
        return ("hang", 0.0, 1)
    if f < F_C0:
        return ("fall", (f - (F_B0 - 1)) / FPS, 12)
    if f < F_D0:
        return ("empty", 0.0, 1)
    if f < F_E0:
        return ("hang", 0.0, 1)
    return ("fall", (f - (F_E0 - 1)) / (FPS * SLOW), 2)

def frame_dt(f):
    """Physical seconds spanned by one frame in this segment."""
    mode, _, _ = frame_state(f)
    if mode == "fall" and f >= F_E0:
        return 1.0 / (FPS * SLOW)
    return 1.0 / FPS

# ---------------------------------------------------------------- drawing
PPM   = 1270.0                    # px per metre
Y_TOP = 230.0                     # screen y of top coil at rest
CX    = W / 2
R_MAJ = 215.0                     # coil semi-major (px)
R_MIN = 30.0                      # coil semi-minor (px)
STROKE   = 10.0
STROKE_B = 13.0                   # bottom coil, thicker
Y_LINE = Y_TOP + L_HANG * PPM     # reference line = bottom coil at rest

def coil_colour(i):
    """Rainbow violet (top) -> orange; the last coil is white, drawn last."""
    if i == N_COIL - 1:
        return WHITE_COL
    h = 0.78 - (0.78 - 0.08) * i / (N_COIL - 2)
    s, v = 0.72, 0.95
    k = int(h * 6.0)
    fr = h * 6.0 - k
    p, q, t = v * (1 - s), v * (1 - s * fr), v * (1 - s * (1 - fr))
    rgb = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][k % 6]
    return rgb

COLS = [coil_colour(i) for i in range(N_COIL)]

def sample_positions(t, traj):
    idx = int(round(t / DT))
    idx = max(0, min(STEPS, idx))
    return traj[idx]

def draw_tap(ctx, t, traj, mode):
    """One shutter tap. Pure function of t and mode."""
    # background
    ctx.set_source_rgb(*BG)
    ctx.paint()
    # reference line (under everything)
    ctx.set_source_rgb(*LINE_COL)
    ctx.set_line_width(2.0)
    ctx.move_to(0, Y_LINE)
    ctx.line_to(W, Y_LINE)
    ctx.stroke()
    # clamp bar
    ctx.set_source_rgb(*BAR_COL)
    ctx.rectangle(CX - 150, 140, 300, 40)
    ctx.fill()
    if mode == "empty":
        return
    x = sample_positions(t, traj) if mode == "fall" else X_EQ
    if mode == "hang":
        # the string that holds the top coil
        ctx.set_source_rgb(*BAR_COL)
        ctx.set_line_width(3.0)
        ctx.move_to(CX, 180)
        ctx.line_to(CX, Y_TOP + x[0] * PPM - R_MIN)
        ctx.stroke()
    # Two passes: every back (upper, dimmed) arc first, then every front
    # (lower, bright) arc — a coil's dim back must never paint over a
    # neighbour's bright front, or the packed clump reads as mud.
    for (a0, a1, mul) in ((math.pi, 2 * math.pi, 0.52), (0.0, math.pi, 1.0)):
        for i in range(N_COIL):
            y = Y_TOP + x[i] * PPM
            if y > H + R_MIN + 26:
                continue
            col = COLS[i]
            bottom = i == N_COIL - 1
            w = STROKE_B if bottom else STROKE
            r = R_MAJ + 20 if bottom else R_MAJ   # marker rim protrudes
            ctx.save()
            ctx.translate(CX, y)
            ctx.scale(1.0, R_MIN / R_MAJ)
            ctx.new_path()
            ctx.arc(0, 0, r, a0, a1)
            ctx.restore()
            ctx.set_source_rgb(col[0] * mul, col[1] * mul, col[2] * mul)
            ctx.set_line_width(w)
            ctx.stroke()

def caption(ctx, text):
    ctx.select_font_face("DejaVu Sans", cairo.FONT_SLANT_NORMAL,
                         cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(38)
    ctx.set_source_rgb(*TEXT_COL)
    ctx.move_to(64, 408)
    ctx.show_text(text)

_TRAJ = None

def draw(f):
    """Pure function of f -> HxWx3 uint8."""
    mode, t, taps = frame_state(f)
    dtf = frame_dt(f)
    acc = np.zeros((H, W, 3), dtype=np.float64)
    for j in range(taps):
        off = 0.0 if taps == 1 else (j / (taps - 1) - 0.5) * SHUTTER * dtf
        surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
        ctx = cairo.Context(surf)
        draw_tap(ctx, max(0.0, t + off), _TRAJ, mode)
        if F_E0 <= f < F_E0 + 45:
            caption(ctx, "again — 25× slower")
        surf.flush()
        buf = np.ndarray((H, surf.get_stride() // 4, 4), dtype=np.uint8,
                         buffer=surf.get_data())[:, :W, :]
        acc += buf[:, :, [2, 1, 0]].astype(np.float64)   # BGRA -> RGB
    return np.clip(acc / taps + 0.5, 0, 255).astype(np.uint8)

# ---------------------------------------------------------------- checks
def run_checks():
    global _TRAJ, X_EQ
    print("== model checks ==")
    x0, ext = equilibrium()
    # 1. hang length is exact
    assert abs(x0[-1] - L_HANG) < 1e-12, x0[-1]
    print(f"hang length: {x0[-1]:.12f} m == L OK")
    # 2. no initial contact
    assert ext.min() > D_MIN, (ext.min(), D_MIN)
    print(f"smallest initial gap {ext.min()*1000:.3f} mm > d_min "
          f"{D_MIN*1000:.3f} mm OK")
    # 3. equilibrium residual with the top held
    t_i = K_SEG * ext
    res = np.full(N_COIL, M_NODE * G_ACC)
    res[:-1] += t_i
    res[1:] -= t_i
    assert np.abs(res[1:]).max() < 1e-9, np.abs(res[1:]).max()
    print(f"equilibrium residual (free nodes): {np.abs(res[1:]).max():.2e} OK")

    traj = simulate()
    _TRAJ = traj
    X_EQ = traj[0].copy()
    ts = np.arange(STEPS + 1) * DT

    # 4. the coupling assertion: COM free-falls at exactly g while the
    #    bottom hangs. Internal forces cancel; this reads it back.
    com = traj.mean(axis=1)
    ff = com[0] + 0.5 * G_ACC * ts ** 2
    com_err = np.abs(com - ff).max()
    assert com_err < 1e-3, com_err
    print(f"COM vs 1/2 g t^2: max |err| = {com_err*1000:.4f} mm OK")

    # 5. ordering preserved, nothing tunnels
    gaps = traj[:, 1:] - traj[:, :-1]
    assert gaps.min() > 0.4 * D_MIN, gaps.min()
    print(f"min gap ever: {gaps.min()*1000:.4f} mm (> 0.4 d_min) OK")

    # 6. collapse front: first step each spring closes to contact
    closed = gaps < 1.2 * D_MIN
    t_close = np.array([ts[np.argmax(closed[:, i])] if closed[:, i].any()
                        else np.inf for i in range(N_COIL - 1)])
    assert np.all(np.isfinite(t_close)), "some spring never closed"
    assert np.all(np.diff(t_close) >= 0), "front not monotonic"
    print("collapse front monotonic top->bottom OK")
    t_front = t_close[-1]

    # 7. bottom coil motionless until the front arrives
    pre = ts < 0.95 * t_front
    b_disp = np.abs(traj[pre, -1] - traj[0, -1]).max()
    assert b_disp < 0.002, b_disp
    print(f"bottom coil drift before arrival: {b_disp*1000:.3f} mm "
          f"(< 2 mm over {0.95*t_front:.3f} s) OK")

    # 8. HELD OUT: collapse ends when the free-falling COM reaches the
    #    stack. Final stack: bottom coil unmoved, coils packed at D_MIN.
    com_final = traj[0, -1] - (N_COIL - 1) / 2 * D_MIN
    d_com = com_final - com[0]
    t_pred = math.sqrt(2 * d_com / G_ACC)
    rel = abs(t_front - t_pred) / t_pred
    assert rel < 0.03, (t_front, t_pred)
    print(f"held-out t_c: sim front {t_front:.4f} s vs sqrt(2d/g) "
          f"{t_pred:.4f} s ({rel*100:.2f}% off) OK")

    # 9. inelastic, not bouncy: once closed, a gap stays closed
    for i in range(N_COIL - 1):
        s0 = np.argmax(closed[:, i])
        reopen = gaps[s0:, i] > 3.0 * D_MIN
        assert not reopen.any(), f"spring {i} reopened"
    print("no gap reopens after contact (sticky) OK")

    # 10. clump velocities equalised at the end (KE in COM frame ~ 0)
    v_end = (traj[-1] - traj[-2]) / DT
    spread = v_end.max() - v_end.min()
    assert spread < 0.05 * abs(v_end.mean()), (spread, v_end.mean())
    print(f"final velocity spread {spread:.4f} m/s "
          f"vs mean {v_end.mean():.3f} m/s OK")

    # 11. momentum bookkeeping: COM velocity == g t
    v_com = (traj[-1].mean() - traj[-2].mean()) / DT
    assert abs(v_com - G_ACC * ts[-1]) < 2e-3, v_com
    print(f"COM velocity at end {v_com:.4f} vs g*t {G_ACC*ts[-1]:.4f} OK")

    print("== frame checks ==")
    # 12. timeline sanity: slow segment covers the whole collapse
    t_last = (N_FRAMES - 1 - (F_E0 - 1)) / (FPS * SLOW)
    assert t_last > t_front + 0.015, (t_last, t_front)
    print(f"slow segment reaches t={t_last:.3f} s > t_front OK")
    # 13. real-time segment sees the clump leave the frame
    xB = sample_positions((F_C0 - 1 - (F_B0 - 1)) / FPS, traj)
    yB = Y_TOP + xB.min() * PPM
    assert yB > H + R_MIN, yB
    print(f"real-time exit: clump top at y={yB:.0f} px (> {H}) OK")

    # 14. purity: same frame twice, identical bytes
    a, b = draw(40), draw(40)
    assert np.array_equal(a, b)
    print("draw(40) twice: byte-identical OK")

    # 15. the re-hang IS the hang: D frames == A frames
    assert np.array_equal(draw(F_A0 + 3), draw(F_D0 + 3))
    print("re-hang frame == opening hang frame, byte-identical OK")

    # 16. pixel check (trap 56): the HANG frame is neither blank nor white
    #     (frame 40 is the empty hold — nearly black by design, so it is
    #     the wrong frame to ask this of)
    a = draw(F_A0 + 2)
    lit = (a.astype(int).sum(axis=2) > 90).mean()
    assert 0.02 < lit < 0.60, lit
    print(f"lit fraction {lit:.3f} in (0.02, 0.60) OK")

    # 17. the white coil sits on the reference line at rest (bounded
    #     region — trap 58/64: rows near the line, centre columns only)
    fr = draw(F_A0)
    band = fr[int(Y_LINE)-60:int(Y_LINE)+60, int(CX)-int(R_MAJ):int(CX)+int(R_MAJ)]
    wh = np.argwhere((band[:, :, 0] > 200) & (band[:, :, 1] > 200)
                     & (band[:, :, 2] > 200))
    assert len(wh) > 50, len(wh)
    row_c = wh[:, 0].mean() + int(Y_LINE) - 60
    # only the bright FRONT arc passes the threshold (the back arc is
    # drawn at 0.52 brightness), and a lower semi-ellipse's mean row is
    # 2/pi * R_MIN below the coil centre — expect the arc, not the centre
    row_exp = Y_LINE + (2 / math.pi) * R_MIN
    assert abs(row_c - row_exp) < 8, (row_c, row_exp)
    print(f"white front-arc centroid row {row_c:.1f} vs expected "
          f"{row_exp:.1f} (coil centre on the line) OK")

    # 18. legibility at watch size (trap 67): white coil still separable
    small = fr[::3, ::3]  # ~360 px wide
    ws = ((small[:, :, 0] > 200) & (small[:, :, 1] > 200)
          & (small[:, :, 2] > 200)).sum()
    assert ws > 15, ws
    print(f"white coil pixels at 360 px wide: {ws} OK")

    print("ALL CHECKS PASSED")
    print("stated idealisation (for the description): zero-natural-length")
    print("springs and a 60-coil chain; a real slinky adds pre-tension and")
    print("air drag. Cross & Wheatland measured 0.25-0.3 s on real ones,")
    print("which brackets this model's 0.257 s.")
    return traj

# ---------------------------------------------------------------- encode
def render(path, traj):
    global _TRAJ, X_EQ
    _TRAJ = traj
    X_EQ = traj[0].copy()
    t0 = time.time()
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "18",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in range(N_FRAMES):
        p.stdin.write(draw(f).tobytes())
        if f % 30 == 0:
            print(f"  {f}/{N_FRAMES}  {time.time()-t0:.0f}s", flush=True)
    p.stdin.close()
    p.wait()
    if p.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"encoded {path} in {time.time()-t0:.0f}s")

def check_encode(path, traj):
    """Re-measure the title's claim off the finished bytes: the white
    coil's row is constant through the slow-mo fall, then leaves."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames,width,height",
         "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, text=True)
    print(probe.stdout)
    # crop inside ffmpeg (trap 34): rows 1350..1750, centre 480 cols
    ch, cw, cx0, cy0 = 400, 540, int(CX) - 270, 1350
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vf",
         f"crop={cw}:{ch}:{cx0}:{cy0}", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"], capture_output=True)
    vid = np.frombuffer(r.stdout, dtype=np.uint8)
    nf = len(vid) // (ch * cw * 3)
    vid = vid[:nf * ch * cw * 3].reshape(nf, ch, cw, 3)
    assert nf == N_FRAMES, nf
    rows = np.full(nf, np.nan)
    for f in range(nf):
        m = ((vid[f, :, :, 0] > 190) & (vid[f, :, :, 1] > 190)
             & (vid[f, :, :, 2] > 190))
        if m.sum() > 30:
            rows[f] = np.argwhere(m)[:, 0].mean() + cy0
    # during the slow fall, up to 90% of t_front, the row must not move
    ts_f = np.array([frame_state(f)[1] for f in range(nf)])
    gaps0 = traj[:, 1:] - traj[:, :-1]
    t_front = (np.argmax((gaps0[:, -1] < 1.2 * D_MIN)) * DT)
    slow = np.arange(nf) >= F_E0
    still = slow & (ts_f < 0.90 * t_front)
    drift = np.nanmax(np.abs(rows[still] - np.nanmean(rows[still])))
    n_still = int(still.sum())
    print(f"white-coil row over {n_still} slow-mo frames: "
          f"drift {drift:.2f} px")
    assert drift < 1.5, drift
    # and afterwards it leaves the line
    late = slow & (ts_f > t_front + 0.005)
    moved = np.nanmax(rows[late]) - np.nanmean(rows[still])
    print(f"after impact it falls {moved:.1f} px")
    assert moved > 10, moved
    print("ENCODE CHECKS PASSED — the claim in the title was measured "
          "off the finished h264")
    return n_still, drift, moved

if __name__ == "__main__":
    if "--stills" in sys.argv:
        traj = run_checks()
        os.makedirs("out", exist_ok=True)
        import PIL.Image as I
        for f in (5, 90, 180, 245, 262):
            im = draw(f)
            I.fromarray(im[::3, ::3]).save(f"out/slinky_360_f{f}.png")
            I.fromarray(im).save(f"out/slinky_full_f{f}.png")
        print("stills written")
    elif "--render" in sys.argv:
        traj = run_checks()
        os.makedirs("out", exist_ok=True)
        out = f"out/slinky_{time.strftime('%H%M%S')}.mp4"
        render(out, traj)
        check_encode(out, traj)
        print(out)
    else:
        print(__doc__)
