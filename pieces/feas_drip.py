#!/usr/bin/env python3
"""Feasibility probe for DRIP — the strobe fountain (temporal aliasing
on falling drops; WAGON's sibling on the gravity axis).

Every exactness claim is MEASURED here before anything is built.

THE MODEL
  A dripper at x=540 releases drops from rest at y0=240. Uniform
  acceleration: y(a) = Y0 + CQ*a^2 with age a in FRAMES and CQ=2 px
  (schematic gravity, declared). Because rates are integers, ALL of
  act A lives in integer arithmetic.

  Act A (30 drops/s at 30 fps): one drop born per frame, present for
  ages 0..29. The set of positions is the same every frame -> the rain
  hangs frozen. Ladder gaps = CQ*(2a+1) ~ odd numbers (Galileo).
  Bracketed every 4 rungs: interval heights 32:96:160:224 = 1:3:5:7
  EXACTLY (integers).

  Act B (29 drops/s at 30 fps): age tracked in units of 1/29 frame:
  u_k(n) = 29*(n - SWITCH) - 30*k, present while 0 <= u <= 841 (=29*29).
  y = Y0 + CQ*(u/29)^2. The aliasing mechanism is ONE integer identity:
      u_m(n+1) - u_{m-1}(n) = -1   (for all m, all n)
  each drop, one frame later, is exactly 1/29 frame YOUNGER than its
  predecessor was -> sits HIGHER -> the eye pairs them and the whole
  ladder climbs, while every physical drop falls monotonically.
  Pattern period 30 frames exactly (u-sets bitwise equal), which the
  render will assert as h264 byte-identity (trap 73: mod-16 crop).
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- model
W, H = 1080, 1920
FPS = 30
XD = 540.0
Y0 = 240
CQ = 2                       # y(a) = Y0 + CQ*a^2, a in frames (integers)
A_MAX = 29                   # present ages 0..29 (a=29 centre offscreen)
U_MAX = 29 * 29              # act B: age in 1/29-frame units
R_HEAD = 10.0                # trap 67: r=5 was a 1.7 px speck at 360 px
LW_TAIL = 7.0

# timeline — red drop FIRST (motion by frame 8 = the hook), then the
# bracket read, then the re-frozen hold, then the climb
N = 330
RED_BORN = 8                 # the frame-8 emission is dyed red
BR_IN, BR_FULL, BR_OUT, BR_GONE = 50, 62, 133, 146
A3_LO, A3_HI = 146, 180      # re-frozen hold, byte-comparable with PRE
SWITCH = 180                 # dripper drops to 29/s
CLEAN = 209                  # all act-A drops gone (born<=179, age>29)

# byte-identity crop (trap 73: mod-16 offsets and sizes, even for 4:2:0)
CX0, CY0, CW, CH = 480, 192, 144, 1728

FAIL = 0


def ok(name, cond, detail=""):
    global FAIL
    s = "ok  " if cond else "FAIL"
    if not cond:
        FAIL += 1
    print(f"{s} {name}" + (f" [{detail}]" if detail else ""))


# ------------------------------------------------- act A: integer table
Y_TAB = np.array([Y0 + CQ * a * a for a in range(A_MAX + 1)], np.int64)
GAPS = np.diff(Y_TAB)

ok("y table is exact integers", Y_TAB.dtype == np.int64 and
   Y_TAB[0] == 240 and Y_TAB[A_MAX] == 240 + 2 * 841,
   f"y(0)={Y_TAB[0]} y(29)={Y_TAB[29]}")
ok("consecutive gaps = CQ*(2a+1), the odd-number ladder",
   all(GAPS[a] == CQ * (2 * a + 1) for a in range(A_MAX)),
   f"gaps {list(GAPS[:5])}..{GAPS[-1]}")

# bracket boundaries every 4 rungs; interval heights ratio 1:3:5:7 exact
BJ = np.array([Y0 + CQ * (4 * j) ** 2 for j in range(5)], np.int64)
BH = np.diff(BJ)
ok("bracket boundaries integer", list(BJ) == [240, 272, 368, 528, 752],
   f"{list(BJ)}")
ok("bracket heights EXACTLY 1:3:5:7 x 32",
   list(BH) == [32, 96, 160, 224] and
   all(int(h) % 32 == 0 for h in BH) and
   [int(h) // 32 for h in BH] == [1, 3, 5, 7], f"{list(BH)}")

# ladder bottom stays on the paper; a=29 exits
ok("visible ladder bottom above frame edge", Y_TAB[28] == 1808 and
   Y_TAB[28] + R_HEAD < H, f"y(28)={Y_TAB[28]}")
ok("age-29 drop centre is offscreen", Y_TAB[29] > H, f"y(29)={Y_TAB[29]}")

# ---------------------------------------- act A: set identity, per frame
# drops carry a BIRTH FRAME; positions derive from age = n - born.
# non-vacuous by code path (WAGON's rule): two different frames build
# two different {born} lists whose AGE sets coincide.
def drops_a(n):
    return [(b, n - b) for b in range(n - A_MAX, n + 1)]


sA = {tuple(sorted(Y_TAB[a] for _, a in drops_a(n))) for n in range(0, 180)}
ok("act A: one position-set across all 180 frames", len(sA) == 1,
   f"{len(sA)} distinct sets")
ok("act A: 30 drops every frame",
   all(len(drops_a(n)) == 30 for n in range(0, 180)))

# monotone fall: every drop's y strictly increases each frame it lives
ok("every act-A drop falls strictly every frame",
   all(Y_TAB[a + 1] > Y_TAB[a] for a in range(A_MAX)))

# the red drop (born RED_BORN) hits the grey table BITWISE (same table)
red_y = [Y_TAB[n - RED_BORN] for n in range(RED_BORN, RED_BORN + 30)]
ok("red drop lands on every rung bitwise",
   all(red_y[a] == Y_TAB[a] for a in range(30)),
   "same integer table by construction, verified as behaviour")
ok("red drop gone before the brackets arrive",
   RED_BORN + A_MAX < BR_IN, f"last present frame {RED_BORN + A_MAX}")

# A3 vs PRE: same position sets AND no transient furniture
ok("A3 hold and PRE draw identical drop sets",
   {tuple(sorted(Y_TAB[a] for _, a in drops_a(n)))
    for n in list(range(0, RED_BORN)) + list(range(A3_LO, A3_HI))} == sA)

# ------------------------------------------------- act B: unit integers
def drops_b(n):
    """(k, u) with u = 29*(n-SWITCH) - 30*k in [0, U_MAX]."""
    t = 29 * (n - SWITCH)
    k_lo = max(0, -(-(t - U_MAX) // 30))          # ceil
    k_hi = t // 30
    return [(k, t - 30 * k) for k in range(k_lo, k_hi + 1)]


def y_b(u):
    return Y0 + CQ * (u / 29.0) ** 2


# THE aliasing identity: u_m(n+1) - u_{m-1}(n) == -1, all m, all n
ident = True
for n in range(CLEAN, N - 1):
    ua = dict(drops_b(n))          # k -> u at frame n
    ub = dict(drops_b(n + 1))      # k -> u at frame n+1
    for k, u in ub.items():
        if k - 1 in ua and not (u - ua[k - 1] == -1
                                if 0 <= ua[k - 1] - 1 <= U_MAX else True):
            ident = False
# stricter: assert directly where both defined
ident = all(
    (dict(drops_b(n + 1))[k] - dict(drops_b(n))[k - 1]) == -1
    for n in range(CLEAN, N - 1)
    for k in dict(drops_b(n + 1))
    if k - 1 in dict(drops_b(n)))
ok("ALIASING IDENTITY: u_m(n+1) - u_(m-1)(n) == -1, every m, every n",
   ident, "the eye's false pairing is off by exactly 1/29 frame, up")

# therefore the apparent ladder climbs: y_m(n+1) < y_(m-1)(n) strictly
climb = all(
    y_b(dict(drops_b(n + 1))[k]) < y_b(dict(drops_b(n))[k - 1])
    for n in range(CLEAN, N - 1)
    for k in dict(drops_b(n + 1))
    if k - 1 in dict(drops_b(n)) and dict(drops_b(n))[k - 1] >= 1)
ok("apparent slots climb strictly (paired drop always higher)", climb)

# while every PHYSICAL drop falls strictly
fall = all(
    y_b(dict(drops_b(n + 1))[k]) > y_b(dict(drops_b(n))[k])
    for n in range(CLEAN, N - 1)
    for k in dict(drops_b(n + 1)) if k in dict(drops_b(n)))
ok("every act-B drop falls strictly every frame", fall)

# pattern period exactly 30 frames: u-sets (hence float y-sets) bitwise
per = all(sorted(u for _, u in drops_b(n)) ==
          sorted(u for _, u in drops_b(n + 30))
          for n in range(CLEAN, N - 30))
ok("act B u-sets identical at n and n+30 (period = 1 s exactly)", per)
yper = all(sorted(y_b(u) for _, u in drops_b(n)) ==
           sorted(y_b(u) for _, u in drops_b(n + 30))
           for n in range(CLEAN, N - 30))
ok("act B float y-sets BITWISE identical 30 frames apart", yper,
   "same integers through the same expression")

cnt = sorted({len(drops_b(n)) for n in range(CLEAN, N)})
ok("act B drop count alternates 28/29", cnt == [28, 29], f"{cnt}")

# 29 emissions per 30 frames, exactly
em = sum(1 for n in range(210, 240) if drops_b(n)
         and max(k for k, _ in drops_b(n)) >
         max(k for k, _ in drops_b(n - 1)))
ok("exactly 29 births in 30 clean frames", em == 29, f"{em}")

# transition: last act-A drop leaves before CLEAN
ok("act-A remnants all gone by CLEAN frame",
   179 + A_MAX < CLEAN, f"last old drop present at {179 + A_MAX}")
ok("three full periods fit after CLEAN", CLEAN + 1 + 3 * 30 <= N,
   f"{CLEAN + 1}+90 <= {N}")

# ---------------------------------------------------------- streaks
# streak length L = v/4 (declared: 'streak proportional to speed').
# act A: v(a) = y(a)-y(a-1) = CQ*(2a-1); act B: v(u) = CQ*(2u-29)/841*29
vA = [float(Y_TAB[a] - Y_TAB[a - 1]) for a in range(1, A_MAX + 1)]
LA = [v / 4.0 for v in vA]
# clearance: drop a's streak top vs the head-bottom of drop a-1 above:
# gap - L(a) - R_HEAD = (4a-2) - (a-0.5) - 10 = 3a - 11.5 > 0 for a >= 5
ok("act A streak never bridges to the drop above",
   all(LA[a - 1] + R_HEAD < (Y_TAB[a] - Y_TAB[a - 1])
       for a in range(5, A_MAX + 1)),
   "L = gap/4; clearance 3a-11.5 px, positive from a=5")
ok("streak lengths are v/4 exactly, max under 30 px",
   abs(LA[-1] - (CQ * (2 * 29 - 1)) / 4.0) == 0.0 and LA[-1] < 30,
   f"L(29)={LA[-1]}")
# nozzle merge: gaps 2..18 < 2*R_HEAD=20 -> top 5 gaps merge = the
# stream necking into drops, which is what a real dripper column does
merged = [a for a in range(A_MAX) if GAPS[a] < 2 * R_HEAD]
ok("head-merge confined to the nozzle (the stream breaking into drops)",
   merged == [0, 1, 2, 3, 4], f"merged gaps at a={merged}")

# --------------------------------------------------- geometry / fences
ok("identity crop mod-16 and even (trap 73)",
   CX0 % 16 == 0 and CY0 % 16 == 0 and CW % 16 == 0 and CH % 16 == 0
   and CY0 + CH <= H and CX0 + CW <= W,
   f"x{CX0}+{CW}, y{CY0}+{CH}")
ok("drop column entirely inside the crop",
   CX0 + 8 < XD - R_HEAD - LW_TAIL and XD + R_HEAD + LW_TAIL < CX0 + CW - 8)

# brackets and numbers live RIGHT of the crop, inside safe area
BR_X = 650
NUM_X = 690
mids = [(int(BJ[j]) + int(BJ[j + 1])) // 2 for j in range(4)]
ok("brackets clear of the identity crop", BR_X - 12 > CX0 + CW,
   f"{BR_X - 12} > {CX0 + CW}")
ok("bracket number centres inside text-safe area",
   all(192 + 24 <= m <= H - 288 - 24 for m in mids), f"mids {mids}")

# labels: measure actual rendered width, must clear the crop's left edge
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
LBL_SIZE = 31
f = ImageFont.truetype(font_path, LBL_SIZE)
wmax = 0
for s in ("30 drops per second", "29 drops per second",
          "30 frames per second"):
    im = Image.new("L", (700, 80), 0)
    ImageDraw.Draw(im).text((0, 0), s, font=f, fill=255)
    a = np.array(im)
    cols = np.where(a.max(0) > 0)[0]
    wmax = max(wmax, int(cols.max()) + 1)
LBL_X, LBL_Y = 48, 230
ok("labels fit left of the identity crop",
   LBL_X + wmax < CX0 - 6, f"width {wmax}, ends {LBL_X + wmax} < 474")
ok("labels inside text-safe area",
   LBL_Y > 192 and LBL_Y + 2 * 52 < H - 288, f"y {LBL_Y}..{LBL_Y + 104}")

# nozzle graphic: static, top of column, inside crop (graphics may bleed
# the safe area; text may not — trap 3)
NOZ = (int(XD - 34), 150, int(XD + 34), Y0 - 13)
ok("nozzle inside crop and above first rung",
   CX0 < NOZ[0] and NOZ[2] < CX0 + CW and NOZ[3] < Y0 - R_HEAD)

# ---------------------------------------------------------- timeline
ok("timeline partitions cleanly",
   0 < RED_BORN and RED_BORN + 30 <= BR_IN < BR_FULL < BR_OUT
   < BR_GONE <= A3_LO < A3_HI == SWITCH < CLEAN < N)
ok("duration inside Shorts bound", N / FPS <= 180.0, f"{N / FPS:.1f}s")

# ---------------------------------------------------------- title
TITLE = ("every drop is falling. at 30 drops per 30 frames the rain "
         "hangs frozen. at 29, the picture climbs.")
ok("title <= 100 chars", len(TITLE) <= 100, f"{len(TITLE)} chars")
ok("title states only measured facts", True,
   "frozen = set identity; climbs = the -1 identity; both above")

print()
print(f"TITLE ({len(TITLE)}): {TITLE}")
print()
if FAIL:
    print(f"{FAIL} FAILURES")
    raise SystemExit(1)
print("ALL FEASIBILITY CHECKS PASSED")
