#!/usr/bin/env python3
"""
DODGE -- two people meet on a footpath and both step the same way.

Nobody in this does anything wrong. Both walkers run one identical rule:

    look at where the other person is, see the two gaps their body leaves
    between itself and the two edges of the path, and walk to the middle of
    the bigger gap.

That is a sensible rule. It is also the whole problem, because THE BIGGER GAP
IS A FACT ABOUT THE PATH, NOT ABOUT WHO IS LOOKING. If both people are on the
same side of the middle, both of them measure the same two gaps, both pick the
same bigger one, and both walk to the same place. Then they are on the other
side, the bigger gap is now behind them, and they both come back. The dance is
not two people being clumsy. It is an attractor.

The second thing the model says, and it is the more interesting one: THE DANCE
EXISTS BECAUSE DECIDING TAKES TIME. `T_DEC` is how long a walker commits to a
choice before re-deciding. Set it to zero and they do not dance -- they chatter
against the centre line and vibrate. It is the commitment, not the politeness,
that turns a wobble into a stand-off.

HONESTY. This rule is made up. It is not a finding about pedestrians and it is
not taken from any paper. It is the simplest rule I could write that produces
the thing everyone has done at least once, and the only claims made here are
claims about the rule, verified against the simulation and then read back off
the finished frames. No number in this file describes a real person.

The render: straight down from above, low sun. From directly overhead two
people are two shapes. The shadows are what tell you they are people, and the
shadows are what walk -- the legs scissor, the arms swing, and when the two
of them stop dead the shadows stop with them.

Not a loop. The last frame is a stand-off and it stays one.

numpy + pycairo + ffmpeg.
"""

import argparse
import math
import subprocess
import sys

import cairo
import numpy as np

# ------------------------------------------------------------------ the world

W, H = 1080, 1920
FPS = 30
SS = 2                        # supersample factor

SCALE = 360.0                 # px per metre -> frame is 3.0 m x 5.33 m
PATH_H = 1.0                  # half-width of the pavement, metres
KERB = 0.20                   # edging strip either side

SAFE_TOP, SAFE_BOT = 192, 1656


def px(x, y):
    """world metres -> pixels. +x is right, +y is up the screen (north)."""
    return (W / 2.0 + x * SCALE, H / 2.0 - y * SCALE)


# ------------------------------------------------------------------ the rule

V_WALK = 1.15                 # m/s forward when unobstructed
V_LAT = 1.30                  # m/s sideways ceiling
T_DEC = 0.50                  # s of commitment between decisions -- THE knob
SEE = 3.2                     # m: close enough to start avoiding
EASE = 1.55                   # m: close enough to start slowing
STOP = 0.75                   # m: face to face, nobody moving
STOP_EPS = 0.12               # m of slack, or the exponential never lands

# How much daylight across the path do two people need to walk past each
# other? DERIVED FROM THE DRAWING, not chosen: a body in skeleton() is a
# shoulder bar of radius SH_R centred SHOULDER either side of the spine, so
# below 2*(SHOULDER + SH_R) the two rendered bodies literally overlap on
# screen. This is the most fragile number in the file and the result leans on
# it hard -- at 0.52 m nobody who starts on opposite sides ever deadlocks, at
# 0.70 m every single start deadlocks. --check prints the sensitivity rather
# than hiding it.
SHOULDER = 0.20
SH_R = 0.085
CLEAR = 2.0 * (SHOULDER + SH_R)

STRIDE = 1.45                 # m of travel per full walk cycle
START_Y = 2.55                # both already in frame at t = 0
START_X = 0.35                # BOTH slightly right of the middle -- same side


def bigger_gap_centre(x_other):
    """The one rule. Where does the other body leave the most room?

    Their body at x_other splits the path into a gap on each side. Return the
    middle of whichever is wider. Note what is NOT in this function: who is
    asking. That is the entire piece.
    """
    gap_left = x_other + PATH_H          # from -PATH_H up to them
    gap_right = PATH_H - x_other         # from them out to +PATH_H
    if gap_left >= gap_right:
        return (-PATH_H + x_other) / 2.0
    return (x_other + PATH_H) / 2.0


class Walker:
    def __init__(self, x, y, heading):
        self.x = x
        self.y = y
        self.dir = heading               # +1 walks north, -1 walks south
        self.phase = 0.0
        self.target = None
        self.hx, self.hy = 0.0, float(heading)
        self.moved = 0.0


def simulate(xa, xb, n_frames, sub=3):
    """Run the pair and return one record per frame.

    Nothing here is keyframed. The only inputs are the two start positions.
    """
    dt = 1.0 / (FPS * sub)
    a = Walker(xa, -START_Y, +1)
    b = Walker(xb, +START_Y, -1)
    stopped = False
    t = 0.0
    next_dec = 0.0
    decisions = []                       # (t, target_a, target_b)
    frames = []

    for f in range(n_frames):
        for _ in range(sub):
            sep = math.hypot(a.x - b.x, a.y - b.y)
            # Being NEAR somebody is not being blocked by them. Two people
            # walking past each other pass within a stride of one another and
            # nothing happens, because the clearance that matters is the one
            # ACROSS the path. Scoring the stop on straight-line separation
            # made a clean shoulder-to-shoulder pass read as a stand-off and
            # reported every one of 1,681 starts as a deadlock.
            dxg = abs(a.x - b.x)
            dyg = abs(a.y - b.y)
            blocked = dxg < CLEAR
            if not stopped and blocked and dyg <= STOP + STOP_EPS:
                stopped = True
            facing = (b.y - a.y) > 0.10          # still in front of each other
            engaged = facing and (sep < SEE) and not stopped
            if not facing:
                a.target = b.target = None
            if engaged and t >= next_dec:
                a.target = bigger_gap_centre(b.x)
                b.target = bigger_gap_centre(a.x)
                decisions.append((t, a.target, b.target))
                next_dec = t + T_DEC

            if stopped:
                scale = 0.0
            elif blocked:
                scale = max(0.0, min(1.0, (dyg - STOP) / EASE))
            else:
                scale = 1.0
            v = V_WALK * scale
            for w in (a, b):
                dy = w.dir * v * dt
                dx = 0.0
                if w.target is not None and not stopped:
                    want = w.target - w.x
                    lim = V_LAT * dt
                    dx = max(-lim, min(lim, want))
                w.x += dx
                w.y += dy
                d = math.hypot(dx, dy)
                w.moved += d
                w.phase += 2.0 * math.pi * d / STRIDE
                # Which way is a walker facing? NOT the way they are moving.
                # Near the stop the forward speed has decayed to nothing and
                # the only motion left is sideways, so a heading taken from
                # raw velocity spins them to face across the path at exactly
                # the moment the picture needs them face to face. Face the
                # way you intend to go, and lean into the dodge.
                fx, fy = dx * 0.5, w.dir * V_WALK * dt
                fn = math.hypot(fx, fy)
                w.hx, w.hy = fx / fn, fy / fn
            t += dt
        frames.append({
            'a': (a.x, a.y, a.phase, a.hx, a.hy),
            'b': (b.x, b.y, b.phase, b.hx, b.hy),
            'sep': math.hypot(a.x - b.x, a.y - b.y),
            'stopped': stopped,
        })
    return frames, decisions


DUR = 5.00
N_FRAME = int(round(DUR * FPS))
FRAMES, DECISIONS = simulate(START_X, START_X, N_FRAME)


# ------------------------------------------------------------------ the body

def skeleton(phase):
    """Bones in body-local metres: (p0, p1, radius, kind).

    Local axes: u across the body, f forward, z up. Nothing here is measured
    off a real person -- these are the proportions that read from above.
    """
    s = math.sin(phase)
    f_a, f_b = 0.32 * s, -0.32 * s              # feet, out of phase
    h_a, h_b = -0.19 * s, 0.19 * s              # hands, opposite the feet
    li_a, li_b = 0.09 * max(0.0, s), 0.09 * max(0.0, -s)
    bob = 0.018 * math.cos(2.0 * phase)

    hip_a = (-0.09, 0.0, 0.95 + bob)
    hip_b = (0.09, 0.0, 0.95 + bob)
    kne_a = (-0.095, f_a * 0.45, 0.52 + li_a * 0.5)
    kne_b = (0.095, f_b * 0.45, 0.52 + li_b * 0.5)
    ft_a = (-0.10, f_a, 0.055 + li_a)
    ft_b = (0.10, f_b, 0.055 + li_b)
    pel = (0.0, 0.0, 0.98 + bob)
    chest = (0.0, 0.0, 1.40 + bob)
    sh_a = (-SHOULDER, 0.0, 1.38 + bob)
    sh_b = (SHOULDER, 0.0, 1.38 + bob)
    el_a = (-0.235, h_a * 0.5, 1.14 + bob)
    el_b = (0.235, h_b * 0.5, 1.14 + bob)
    hd_a = (-0.232, h_a, 0.92 + bob)
    hd_b = (0.232, h_b, 0.92 + bob)
    head = (0.0, 0.03, 1.60 + bob)

    return [
        (hip_a, kne_a, 0.075, 'leg'), (kne_a, ft_a, 0.062, 'leg'),
        (hip_b, kne_b, 0.075, 'leg'), (kne_b, ft_b, 0.062, 'leg'),
        (sh_a, el_a, 0.055, 'arm'), (el_a, hd_a, 0.048, 'arm'),
        (sh_b, el_b, 0.055, 'arm'), (el_b, hd_b, 0.048, 'arm'),
        (pel, chest, 0.150, 'torso'), (sh_a, sh_b, SH_R, 'torso'),
        (head, head, 0.105, 'head'),
    ]


def to_world(p, wx, wy, hx, hy):
    """local (u, f, z) -> world (x, y, z), given position and heading."""
    u, f, z = p
    rx, ry = hy, -hx                     # the walker's right in world axes
    return (wx + u * rx + f * hx, wy + u * ry + f * hy, z)


# low sun. shadow of a point at height z lands this far from its base, in
# metres per metre of height: 2.0x, pointing down-left on screen.
SUN = (-0.84, -1.12)


# cairo takes colour as 0..1 FLOATS. Handing it 0..255 clamps every channel to
# white and no geometry assertion will ever notice. (RENDERER.md trap 55.)
SLAB = (0.792, 0.773, 0.735)
JOINT = (0.741, 0.718, 0.678)
EDGING = (0.845, 0.827, 0.788)
ROAD = (0.298, 0.298, 0.318)
GRASS = (0.435, 0.482, 0.353)
SHADOW = (0.235, 0.243, 0.318)

SKIN = {
    'a': {'leg': (0.180, 0.200, 0.271), 'arm': (0.196, 0.310, 0.478),
          'torso': (0.220, 0.345, 0.525), 'head': (0.325, 0.255, 0.196)},
    'b': {'leg': (0.302, 0.278, 0.259), 'arm': (0.596, 0.255, 0.200),
          'torso': (0.663, 0.286, 0.224), 'head': (0.596, 0.494, 0.337)},
}


def slab_tone(i, j):
    h = (i * 73856093) ^ (j * 19349663)
    h = (h ^ (h >> 13)) & 0xFFFFFFFF
    return ((h % 1000) / 1000.0 - 0.5) * 0.034


def draw_ground(cr):
    cr.set_source_rgb(*ROAD)
    cr.paint()

    # grass on the right of the path, road on the left
    x0, _ = px(PATH_H + KERB, 0)
    cr.set_source_rgb(*GRASS)
    cr.rectangle(x0, 0, W - x0, H)
    cr.fill()

    # the two edging strips
    for sgn in (-1, 1):
        a, _ = px(sgn * PATH_H, 0)
        b, _ = px(sgn * (PATH_H + KERB), 0)
        cr.set_source_rgb(*EDGING)
        cr.rectangle(min(a, b), 0, abs(b - a), H)
        cr.fill()

    # paving slabs: 4 across the 3.6 m path, 0.6 m along
    cr.set_source_rgb(*SLAB)
    xl, _ = px(-PATH_H, 0)
    xr, _ = px(PATH_H, 0)
    cr.rectangle(xl, 0, xr - xl, H)
    cr.fill()

    n_col = int(round(2 * PATH_H / 0.5))
    j0 = int(math.floor(-H / 2.0 / SCALE / 0.5)) - 1
    j1 = int(math.ceil(H / 2.0 / SCALE / 0.5)) + 1
    for i in range(n_col):
        ax = -PATH_H + i * 0.5
        for j in range(j0, j1):
            ay = j * 0.5
            p0 = px(ax, ay + 0.5)
            p1 = px(ax + 0.5, ay)
            t = slab_tone(i, j)
            cr.set_source_rgb(*[min(1.0, max(0.0, c + t)) for c in SLAB])
            cr.rectangle(p0[0], p0[1], p1[0] - p0[0], p1[1] - p0[1])
            cr.fill()

    cr.set_source_rgb(*JOINT)
    cr.set_line_width(0.011 * SCALE)
    for i in range(n_col + 1):
        x = -PATH_H + i * 0.5
        cr.move_to(*px(x, 4.0))
        cr.line_to(*px(x, -4.0))
        cr.stroke()
    for j in range(j0, j1 + 1):
        y = j * 0.5
        cr.move_to(*px(-PATH_H, y))
        cr.line_to(*px(PATH_H, y))
        cr.stroke()

    # the sun is up and to the right, so the ground is warmer that way
    g = cairo.LinearGradient(0, H, W, 0)
    g.add_color_stop_rgba(0.0, 1, 0.94, 0.82, 0.00)
    g.add_color_stop_rgba(1.0, 1, 0.94, 0.82, 0.11)
    cr.set_source(g)
    cr.rectangle(0, 0, W, H)
    cr.fill()


def stroke_bone(cr, a, b, radius):
    cr.set_line_width(2.0 * radius * SCALE)
    cr.set_line_cap(cairo.LINE_CAP_ROUND)
    if abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9:
        cr.arc(a[0], a[1], radius * SCALE, 0, 2 * math.pi)
        cr.fill()
        return
    cr.move_to(a[0], a[1])
    cr.line_to(b[0], b[1])
    cr.stroke()


def draw_shadows(cr, people):
    """Both shadows painted as ONE group, then composited once.

    Drawn as a group on purpose: overlapping translucent capsules would show
    a seam everywhere two bones cross, and a walking body is nothing but
    bones crossing.
    """
    cr.push_group()
    for (wx, wy, phase, hx, hy), _ in people:
        for p0, p1, r, _kind in skeleton(phase):
            a = to_world(p0, wx, wy, hx, hy)
            b = to_world(p1, wx, wy, hx, hy)
            pa = px(a[0] + SUN[0] * a[2], a[1] + SUN[1] * a[2])
            pb = px(b[0] + SUN[0] * b[2], b[1] + SUN[1] * b[2])
            # the higher the caster the softer its shadow, so widen with z
            zm = 0.5 * (a[2] + b[2])
            cr.set_source_rgb(*SHADOW)
            stroke_bone(cr, pa, pb, r * (1.0 + 0.42 * zm / 1.7))
    cr.pop_group_to_source()
    cr.paint_with_alpha(0.40)


def draw_person(cr, state, who):
    wx, wy, phase, hx, hy = state
    pal = SKIN[who]

    # seat them on the ground: a small contact darkening under the feet
    cr.save()
    cr.translate(*px(wx, wy))
    cr.scale(1.0, 1.0)
    g = cairo.RadialGradient(0, 0, 0, 0, 0, 0.42 * SCALE)
    g.add_color_stop_rgba(0.0, 0.10, 0.10, 0.14, 0.34)
    g.add_color_stop_rgba(1.0, 0.10, 0.10, 0.14, 0.0)
    cr.set_source(g)
    cr.arc(0, 0, 0.42 * SCALE, 0, 2 * math.pi)
    cr.fill()
    cr.restore()

    bones = skeleton(phase)
    # straight down from above: what is higher is drawn later
    order = sorted(range(len(bones)),
                   key=lambda i: 0.5 * (bones[i][0][2] + bones[i][1][2]))
    for i in order:
        p0, p1, r, kind = bones[i]
        a = to_world(p0, wx, wy, hx, hy)
        b = to_world(p1, wx, wy, hx, hy)
        cr.set_source_rgb(*pal[kind])
        stroke_bone(cr, px(a[0], a[1]), px(b[0], b[1]), r)


def render_frame(surf, cr, i):
    fr = FRAMES[i]
    cr.save()
    cr.scale(SS, SS)
    draw_ground(cr)
    people = [(fr['a'], 'a'), (fr['b'], 'b')]
    draw_shadows(cr, people)
    for state, who in people:
        draw_person(cr, state, who)
    cr.restore()
    surf.flush()

    buf = np.ndarray(shape=(H * SS, W * SS, 4), dtype=np.uint8,
                     buffer=surf.get_data())
    rgb = buf[:, :, [2, 1, 0]].astype(np.float32) / 255.0
    return rgb.reshape(H, SS, W, SS, 3).mean(axis=(1, 3))


def to8(img):
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


# ------------------------------------------------------------------- checking

def sweep(clear, n=41):
    """Where does this rule fail? Run the pair from a grid of starts.

    Split by whether the two of them started on the same side of the middle,
    because that is the thing the rule is blind to.
    """
    global CLEAR
    keep, CLEAR = CLEAR, clear
    xs = np.linspace(-0.9, 0.9, n)
    same = [0, 0]
    opp = [0, 0]
    for xa in xs:
        for xb in xs:
            fr, _ = simulate(float(xa), float(xb), int(5.5 * FPS))
            stuck = bool(fr[-1]['stopped'])
            box = same if (xa > 0) == (xb > 0) else opp
            box[0] += stuck
            box[1] += 1
    CLEAR = keep
    return same, opp


def run_checks(surf, cr):
    OK = True

    def t(cond, msg):
        nonlocal OK
        OK = OK and bool(cond)
        print(("  ok   " if cond else "  FAIL ") + msg)

    xa = np.array([f['a'][0] for f in FRAMES])
    xb = np.array([f['b'][0] for f in FRAMES])
    ya = np.array([f['a'][1] for f in FRAMES])
    yb = np.array([f['b'][1] for f in FRAMES])
    sep = np.array([f['sep'] for f in FRAMES])
    stop = np.array([f['stopped'] for f in FRAMES])
    pha = np.array([f['a'][2] for f in FRAMES])

    print("\n-- the rule --")
    t(np.max(np.abs(xa - xb)) == 0.0,
      f"the two walkers hold the same lateral position exactly, "
      f"max gap {np.max(np.abs(xa - xb)):.1e} m")
    same = sum(1 for _, ta, tb in DECISIONS if ta == tb)
    t(len(DECISIONS) >= 4 and same == len(DECISIONS),
      f"every decision picks the same gap: {same}/{len(DECISIONS)}")
    signs = [1 if ta > 0 else -1 for _, ta, _ in DECISIONS]
    flips = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
    t(flips >= 3, f"the choice alternates sides {flips} times -- a dance, "
                  f"not a drift")
    swing = float(np.max(xa) - np.min(xa))
    t(swing > 0.45, f"lateral swing {swing:.2f} m, visible at this scale")

    print("\n-- it is not cheating --")
    t(np.max(np.abs(xa)) + 0.25 < PATH_H,
      f"nobody steps off the pavement: max |x| {np.max(np.abs(xa)):.2f} m "
      f"+ shoulder, path half-width {PATH_H}")
    t(np.min(sep) > 0.70,
      f"nobody walks through anybody: closest approach {np.min(sep):.2f} m")
    t(abs(float(ya[0] + yb[0])) < 1e-9 and abs(float(ya[-1] + yb[-1])) < 1e-9,
      "they start and end symmetric about the middle of the frame")
    first_stop = int(np.argmax(stop)) if stop.any() else -1
    held = (N_FRAME - first_stop) / FPS
    t(first_stop > 0 and held >= 2.0,
      f"they stop dead at {first_stop / FPS:.2f} s and the stand-off holds "
      f"{held:.2f} s")
    t(float(pha[-1] - pha[first_stop]) == 0.0,
      "the walk cycle stops when they do -- the shadows stop stepping")
    t(float(pha[first_stop]) > 6.0,
      f"they took real steps first: {pha[first_stop] / (2 * math.pi):.1f} "
      f"walk cycles")

    print("\n-- the control: is the rule actually bad? --")
    fr2, dec2 = simulate(+0.35, -0.35, N_FRAME)
    t(not fr2[-1]['stopped'],
      "started on OPPOSITE sides of the middle, the same rule passes cleanly")
    t(abs(fr2[-1]['a'][1] - fr2[-1]['b'][1]) > 3.0,
      f"and they walk on past each other, ending "
      f"{abs(fr2[-1]['a'][1] - fr2[-1]['b'][1]):.2f} m up and down the path")
    fr3, _ = simulate(START_X, START_X, N_FRAME)
    t([f['sep'] for f in fr3] == list(sep), "the simulation is deterministic")

    print("\n-- and how bad, over every start --")
    same, opp = sweep(CLEAR)
    tot_d, tot_n = same[0] + opp[0], same[1] + opp[1]
    print(f"       {tot_n} starting pairs across the path, "
          f"{tot_d} of them deadlock ({100.0 * tot_d / tot_n:.1f}%)")
    print(f"       same side of the middle: {100.0 * same[0] / same[1]:.1f}%   "
          f"opposite sides: {100.0 * opp[0] / opp[1]:.1f}%")
    t(same[0] / same[1] > 3.0 * max(opp[0] / opp[1], 1e-9),
      "starting on the same side of the middle is far worse than starting on "
      "opposite sides")
    print("       and how much that leans on CLEAR, the one soft number:")
    for c in (0.52, CLEAR, 0.70):
        sm, op = sweep(c, 21)
        d_ = sm[0] + op[0]
        n_ = sm[1] + op[1]
        print(f"         {c:.2f} m -> {100.0 * d_ / n_:5.1f}% deadlock "
              f"(same side {100.0 * sm[0] / sm[1]:5.1f}%, "
              f"opposite {100.0 * op[0] / op[1]:5.1f}%)")

    print("\n-- read it back off the finished picture --")
    img = render_frame(surf, cr, N_FRAME - 1)
    fa, fb = FRAMES[-1]['a'], FRAMES[-1]['b']

    def column_of(wy, colour):
        """Find a walker's lateral centre in the frame, by colour.

        Bounded in ROWS to a 90 px band on that walker only, so this cannot
        pick up: the OTHER walker (180 px away up the frame), either shadow
        (both run down-left out of the band, and are grey-blue not navy or
        rust), the edging strips, the road or the grass. A pixel check has no
        idea what it is looking at. (RENDERER.md trap 58.)
        """
        r = int(round(px(0, wy)[1]))
        band = img[max(0, r - 45):r + 45, :, :]
        d = np.sqrt(((band - np.array(colour)) ** 2).sum(axis=2))
        m = d < 0.16
        counts = m.sum(axis=0).astype(float)
        tot = counts.sum()
        if tot == 0:
            return 0, -1.0
        return int(tot), float((counts * np.arange(counts.size)).sum() / tot)

    na, ca = column_of(fa[1], SKIN['a']['torso'])
    nb, cb = column_of(fb[1], SKIN['b']['torso'])
    t(na > 400 and nb > 400,
      f"both walkers are actually on screen: {na} and {nb} matched pixels")
    t(abs(ca - cb) < 18,
      f"THE PIECE, measured off the frame: they end in the same lane, "
      f"{abs(ca - cb):.1f} px apart across a {2 * PATH_H * SCALE:.0f} px path")
    want_a = px(fa[0], 0)[0]
    t(abs(ca - want_a) < 20,
      f"and where the simulation says: {ca:.0f} px against {want_a:.0f}")

    # shadows: darker than the pavement, and down-LEFT of the bodies
    lo = img.sum(axis=2) < (sum(SLAB) - 0.42)
    rows, cols = np.nonzero(lo)
    body_px = px(fa[0], 0.5 * (fa[1] + fb[1]))
    t(cols.mean() < body_px[0] - 60 and rows.mean() > body_px[1] + 60,
      f"the shadows fall down-left of the bodies, as a low sun behind and to "
      f"the right makes them: centroid ({cols.mean():.0f}, {rows.mean():.0f}) "
      f"vs bodies ({body_px[0]:.0f}, {body_px[1]:.0f})")
    t(lo.mean() > 0.02, f"and there is a real amount of them: "
                        f"{100 * lo.mean():.1f}% of the frame")

    top = min(px(fa[0], fa[1])[1], px(fb[0], fb[1])[1]) - 0.35 * SCALE
    bot = max(px(fa[0], fa[1])[1], px(fb[0], fb[1])[1]) + 0.35 * SCALE
    t(top > SAFE_TOP and bot < SAFE_BOT,
      f"both bodies inside the safe band: rows {top:.0f}..{bot:.0f} "
      f"inside {SAFE_TOP}..{SAFE_BOT}")

    f0 = render_frame(surf, cr, 0)
    t(float(np.abs(f0 - img).max()) > 0.2,
      "the first and last frames are not the same picture -- this one is "
      "deliberately not a loop")

    print("\n" + ("ALL CHECKS PASS" if OK else "SOMETHING FAILED"))
    return 0 if OK else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--stills')
    ap.add_argument('--out')
    args = ap.parse_args()

    print(f"two walkers, one rule, {DUR:.2f} s / {N_FRAME} frames")
    print(f"  {len(DECISIONS)} decisions at {T_DEC:.2f} s of commitment each")

    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W * SS, H * SS)
    cr = cairo.Context(surf)

    if args.check:
        return run_checks(surf, cr)

    if args.stills:
        from PIL import Image
        for i in (0, 20, 32, 44, 56, 68, 80, N_FRAME - 1):
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
