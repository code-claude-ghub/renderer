"""Two phones, one conversation, and the gap between what got typed and what
got through.

Somebody is answering "you ok?". The top half is their screen: the compose
box, filling and emptying. The bottom half is the other person's screen, and
the only thing on it is a typing indicator going on and off.

THE POINT, AND IT IS NOT "THEY DELETED IT".
    Everyone knows the long message that never gets sent. The thing worth
    drawing is that the deleting is not invisible. The other person sees the
    indicator flick on and off five times. That flicker is a real signal, it
    is the only honest one that gets through, and nobody reads it.

WHY ASCII, HONESTLY.
    A glyph grid was demoted on this channel in August 2026 for a good craft
    reason: it carries about ten brightness steps, so a smooth surface bands.
    That argument is about SHADING. It says nothing about a subject that is
    already made of characters. A text conversation in a monospace grid is
    not a picture of the thing rendered in glyphs. It is the thing. So this
    piece runs at 41 columns instead of the house 98 -- big enough that one
    character is one readable character on a phone, which is the whole
    requirement and the reason the house grid could never have carried it.

THE BOTTOM HALF IS NOT ANIMATED.
    It is computed from the top half's keystroke log by one rule: show the
    indicator while a key was pressed in the last TIMEOUT seconds. Deleting
    counts as pressing a key, which is why the reader watches "typing..."
    all through the message being destroyed. Change TIMEOUT and the whole
    bottom half changes -- at 6 s it never drops once, at 0.5 s it strobes.
    --check prints that sensitivity instead of hiding it.

    TIMEOUT = 2.0 s is MY number. It is not measured off any real chat
    client and this file does not claim to know what one uses. Nothing here
    describes a real conversation or a real person either. The words are
    written, the timings are written, the flicker is not.

    for @Lost_Warden, who said he missed the ASCII art.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# asciilib sits BESIDE this file in the channel repo and one level UP in the
# public renderer repo (pieces/). Add both, or a clean clone cannot import it
# -- which is the exact bug a viewer found in hook.py on 2026-08-28.
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]

from asciilib import Encoder, Frame, Grid          # noqa: E402

OUT = os.path.join(os.path.dirname(_HERE), "content", "drafts.mp4")

FPS = 30
FONT = 44                       # -> 41 cols x 73 rows, 26 px cells
SEED = 11

# ---------------------------------------------------------------- the script
#
# Written, not simulated. Segment = (characters, chars per second).
# A "hold" is a pause with no keystroke in it, and the holds are the piece:
# they are the only thing that can break the indicator on the other side.
CPS_TYPE = 14.0
CPS_DEL = 48.0

DRAFT1 = ("no. honestly not for a while now"
          " and i keep meaning to say it and"
          " then just not")
DRAFT2 = "kind of tired is all"
DRAFT3 = "yeah"

SCRIPT = [
    ("hold", 0.60),
    ("type", "no. honestly not for a while now"),
    ("hold", 1.10),
    ("type", " and i keep meaning to say it and"),
    ("hold", 0.90),
    ("type", " then just not"),
    ("hold", 2.70),                      # the stare. indicator drops here.
    ("del", None),
    ("hold", 2.60),
    ("type", DRAFT2),
    ("hold", 2.70),
    ("del", None),
    ("hold", 2.40),
    ("type", DRAFT3),
    ("hold", 0.90),
    ("send", None),
    ("hold", 2.20),
]

TIMEOUT = 2.0                   # s the indicator stays up after a keystroke
BLINK = 1.10                    # s cursor blink period
SOLID = 0.40                    # cursor stays solid within this of a keypress
DOT_HZ = 3.0                    # typing-indicator dot cycle

# ------------------------------------------------------------------- palette
BG = (0.043, 0.047, 0.055)
CHROME = (0.150, 0.160, 0.190)
LABEL = (0.330, 0.350, 0.400)
RECV = (0.430, 0.460, 0.510)
LIVE = (0.900, 0.880, 0.820)     # what is being typed, brightest thing here
SENT = (0.560, 0.700, 0.880)
CURSOR = (0.980, 0.800, 0.350)
DOT_OFF = (0.230, 0.245, 0.280)
DOT_ON = (0.620, 0.650, 0.700)

# -------------------------------------------------------------------- layout
COLS, ROWS = 41, 73
A_TOP, A_BOT = 7, 36             # writer's phone, border rows
B_TOP, B_BOT = 40, 62            # reader's phone
A_ASK, A_REPLY = 10, 12          # message rows inside phone A
B_ASK, B_REPLY = 43, 45          # and inside phone B
BOX_BOT = 34                     # compose box bottom border
BOX_L, BOX_R = 2, 38
TEXT_C = 4                       # first text column inside the compose box
TEXT_W = 32                      # characters per line inside it
LEAD = 2                         # ROWS per line of text inside the box.
#
# One glyph is exactly one cell tall, which is what a shaded field wants and
# is wrong for prose: at LEAD 1 the three lines of the message touch, the
# ascenders of one row sit in the descenders of the row above, and it reads
# as a block rather than as sentences. Leading is not decoration here, it is
# the difference between text and texture.
SAFE_TOP, SAFE_BOT = 7, 62


def box_top(nlines):
    """Top border row of the compose box holding `nlines` lines of text.

    The box is anchored at the BOTTOM and grows upward, like a real one, so
    the last line and the cursor never move while the message gets bigger.
    """
    return BOX_BOT - (LEAD * (nlines - 1) + 1) - 1


def wrap(s, w=TEXT_W):
    """Greedy word wrap that also has to look right MID-WORD.

    The buffer is redrawn every frame while it is being typed, so this runs
    on 103 partial strings, not on three finished ones.
    """
    out, line = [], ""
    for ch in s:
        if ch == " " and len(line) >= w:
            out.append(line)
            line = ""
            continue
        line += ch
        if len(line) > w:
            k = line.rfind(" ")
            if k <= 0:
                out.append(line[:w])
                line = line[w:]
            else:
                out.append(line[:k])
                line = line[k + 1:]
    out.append(line)
    return out


# ------------------------------------------------------------------ keystrokes
def build_events():
    """Turn the script into a keystroke log: (t, kind, char).

    Every visible state in the whole video is derived from this list. Typing
    speed is jittered off a fixed seed, because a constant inter-key interval
    reads as a machine and the piece is about a person hesitating.
    """
    rng = np.random.default_rng(SEED)
    ev, t = [], 0.0
    for step in SCRIPT:
        kind = step[0]
        if kind == "hold":
            t += step[1]
        elif kind == "type":
            for ch in step[1]:
                gap = (1.0 / CPS_TYPE) * float(rng.uniform(0.55, 1.55))
                # a space is where a person thinks, so lean the jitter there
                if ch == " ":
                    gap *= 1.45
                t += gap
                ev.append((t, "ins", ch))
        elif kind == "del":
            n = len(text_at(ev, t))
            for _ in range(n):
                t += (1.0 / CPS_DEL) * float(rng.uniform(0.75, 1.25))
                ev.append((t, "del", None))
        elif kind == "send":
            ev.append((t, "send", None))
        else:
            raise ValueError(kind)
    return ev, t


def text_at(ev, t):
    """The compose buffer at time t, replayed from the log."""
    s = ""
    for et, kind, ch in ev:
        if et > t:
            break
        if kind == "ins":
            s += ch
        elif kind == "del":
            s = s[:-1]
        elif kind == "send":
            s = ""
    return s


def sent_at(ev, t):
    return any(et <= t and k == "send" for et, k, _ in ev)


def last_key(ev, t):
    """Time of the most recent KEYSTROKE at or before t, or None.

    A send is not a keystroke -- after it the indicator must go down at once,
    because the message has arrived. Delete IS a keystroke, which is the
    detail the piece turns on.
    """
    out = None
    for et, kind, _ in ev:
        if et > t:
            break
        if kind in ("ins", "del"):
            out = et
        elif kind == "send":
            out = None
    return out


def indicator(ev, t, timeout=TIMEOUT):
    """THE RULE. One line, and it generates the entire bottom half."""
    lk = last_key(ev, t)
    return lk is not None and (t - lk) <= timeout


# ----------------------------------------------------------------- the drawing
def hline(fr, row, label=None):
    if label is None:
        fr.put_run(0, row, "+" + "-" * (COLS - 2) + "+", CHROME)
        return
    head = "+-- "
    fr.put_run(0, row, head, CHROME)
    fr.put_run(len(head), row, label, LABEL)
    c = len(head) + len(label)
    fr.put_run(c, row, " " + "-" * (COLS - c - 2) + "+", CHROME)


def draw(g, ev, t):
    fr = Frame(g, BG)

    # the two phones
    hline(fr, A_TOP, "them")
    hline(fr, A_BOT)
    hline(fr, B_TOP, "you")
    hline(fr, B_BOT)
    for r in range(A_TOP + 1, A_BOT):
        fr.put(0, r, "|", CHROME)
        fr.put(COLS - 1, r, "|", CHROME)
    for r in range(B_TOP + 1, B_BOT):
        fr.put(0, r, "|", CHROME)
        fr.put(COLS - 1, r, "|", CHROME)

    # the question, on both screens. left on theirs, right on yours.
    fr.put_run(3, A_ASK, "you ok?", RECV)
    fr.put_run(COLS - 4 - len("you ok?"), B_ASK, "you ok?", RECV)

    buf = text_at(ev, t)
    done = sent_at(ev, t)
    lines = wrap(buf)

    # the compose box, growing upward off the number of wrapped lines
    top = box_top(len(lines))
    fr.put_run(BOX_L, top, "+" + "-" * (BOX_R - BOX_L - 1) + "+", CHROME)
    fr.put_run(BOX_L, BOX_BOT, "+" + "-" * (BOX_R - BOX_L - 1) + "+", CHROME)
    for r in range(top + 1, BOX_BOT):
        fr.put(BOX_L, r, "|", CHROME)
        fr.put(BOX_R, r, "|", CHROME)
    for i, line in enumerate(lines):
        if line:
            fr.put_run(TEXT_C, top + 1 + i * LEAD, line, LIVE)

    # cursor: solid while actually typing, blinking while staring at it
    lk = last_key(ev, t)
    live = lk is not None and (t - lk) < SOLID
    if live or (t % BLINK) < BLINK * 0.55:
        fr.put(TEXT_C + len(lines[-1]), BOX_BOT - 1, "_", CURSOR)

    if done:
        fr.put_run(COLS - 4 - len(DRAFT3), A_REPLY, DRAFT3, SENT)
        fr.put_run(3, B_REPLY, DRAFT3, SENT)
    elif indicator(ev, t):
        # the whole bottom half, and it is three characters
        fr.put(3, B_REPLY, "(", DOT_OFF)
        k = int(t * DOT_HZ) % 3
        for i in range(3):
            fr.put(5 + 2 * i, B_REPLY, "o" if i == k else ".",
                   DOT_ON if i == k else DOT_OFF)
        fr.put(11, B_REPLY, ")", DOT_OFF)
    return fr


# ---------------------------------------------------------------------- checks
def pixels(fr):
    """Frame -> (rows, cols, 3) float, for the checks only."""
    buf = fr.surface.get_data()
    stride = fr.surface.get_stride()
    a = np.ndarray(shape=(fr.g.h_px, stride // 4, 4), dtype=np.uint8,
                   buffer=buf)[:, :fr.g.w_px, :3]
    return a[:, :, [2, 1, 0]].astype(float) / 255.0


def ink(img, g, r0, r1, c0, c1):
    """Count non-background pixels in a CELL-bounded box.

    Bounded in rows AND columns to one panel's interior, so it cannot pick
    up: the other phone (>=4 rows away), either border, the labels in the
    borders, or anything in the 3 blank rows between the two phones.
    A pixel check has no idea what it is looking at. (RENDERER.md trap 58.)
    """
    sub = img[int(r0 * g.cell):int(r1 * g.cell),
              int(c0 * g.cell):int(c1 * g.cell), :]
    d = np.sqrt(((sub - np.array(BG)) ** 2).sum(axis=2))
    return int((d > 0.06).sum())


def ink_of(img, g, r0, r1, c0, c1, rgb, tol=0.10):
    """Count pixels of ONE colour in a cell-bounded box.

    Counting ink says how much is drawn, never what it is. Matching a colour
    that the file assigns to exactly one thing does.
    """
    sub = img[int(r0 * g.cell):int(r1 * g.cell),
              int(c0 * g.cell):int(c1 * g.cell), :]
    d = np.sqrt(((sub - np.array(rgb)) ** 2).sum(axis=2))
    return int((d < tol).sum())


def bursts(ev, dur, timeout):
    """How many separate times does the indicator come up before the send?"""
    n, prev = 0, False
    for f in range(int(dur * FPS)):
        t = f / FPS
        if sent_at(ev, t):
            break
        now = indicator(ev, t, timeout)
        if now and not prev:
            n += 1
        prev = now
    return n


def check():
    g = Grid(font_size=FONT)
    assert (g.cols, g.rows) == (COLS, ROWS), g
    ev, dur = build_events()
    dur += 0.0
    n = int(round(dur * FPS))
    print(g)
    print("script: %.2f s, %d frames, %d events" % (dur, n, len(ev)))

    # --- 1. character accounting. the headline number of the piece.
    typed = sum(1 for _, k, _ in ev if k == "ins")
    deleted = sum(1 for _, k, _ in ev if k == "del")
    assert typed == len(DRAFT1) + len(DRAFT2) + len(DRAFT3), typed
    assert deleted == len(DRAFT1) + len(DRAFT2), deleted
    assert text_at(ev, dur) == "", repr(text_at(ev, dur))
    got = typed - deleted
    assert got == len(DRAFT3) == 4, got
    print("ok   typed %d characters, deleted %d, sent %d"
          % (typed, deleted, got))

    # --- 2. wrapping never overflows the box, and never loses a character
    worst = 0
    for f in range(n):
        buf = text_at(ev, f / FPS)
        ls = wrap(buf)
        worst = max(worst, max(len(x) for x in ls))
        assert max(len(x) for x in ls) <= TEXT_W, (f, ls)
        lost = len(buf.replace(" ", "")) - sum(
            len(x.replace(" ", "")) for x in ls)
        assert lost == 0, (f, lost, repr(buf))
    assert wrap(DRAFT1) == ["no. honestly not for a while now",
                            "and i keep meaning to say it and",
                            "then just not"], wrap(DRAFT1)
    assert TEXT_C + worst <= BOX_R - 1, (TEXT_C, worst, BOX_R)
    print("ok   longest wrapped line %d of %d, box interior ends at %d"
          % (worst, TEXT_W, BOX_R - 1))

    # --- 3. the growing box never eats the messages above it
    tops = set()
    for f in range(n):
        tops.add(box_top(len(wrap(text_at(ev, f / FPS)))))
    assert min(tops) > A_REPLY, (min(tops), A_REPLY)
    assert max(tops) < BOX_BOT, tops
    assert min(tops) > A_TOP and BOX_BOT < A_BOT
    print("ok   compose box spans rows %d..%d, clear of the reply row %d"
          % (min(tops), BOX_BOT, A_REPLY))

    # --- 4. the bottom half really is a function of the top half's log.
    #        recomputed here independently of draw().
    for f in range(n):
        t = f / FPS
        want = any((t - et) <= TIMEOUT and et <= t and k in ("ins", "del")
                   and not any(st <= t and sk == "send" and st > et
                               for st, sk, _ in ev)
                   for et, k, _ in ev)
        assert indicator(ev, t) == want, f
    print("ok   indicator is a pure function of the keystroke log, %d frames"
          % n)

    # --- 5. the flicker, and what it depends on. THE sensitivity.
    #
    # I first asserted this strobes at a short timeout. It does not, and the
    # reason is the good part: between keystrokes the gap is about 0.07 s, so
    # no plausible timeout can ever break the indicator MID-WORD. Only the
    # authored pauses can break it. So the burst count has a closed form --
    # one, plus every pause that sits between two keystrokes and outlasts the
    # timeout -- and that is a far stronger check than a number I picked.
    mid = []
    for i, step in enumerate(SCRIPT):
        if step[0] != "hold":
            continue
        before = any(s[0] in ("type", "del") for s in SCRIPT[:i])
        after = any(s[0] in ("type", "del") for s in SCRIPT[i + 1:])
        if before and after:
            mid.append(step[1])
    assert len(mid) == 6, mid
    for T in (0.5, 1.5, 2.0, 3.0, 6.0):
        want = 1 + sum(1 for h in mid if h > T)
        got_b = bursts(ev, dur, T)
        assert got_b == want, (T, got_b, want)
        print("     timeout %.1f s -> %d bursts" % (T, got_b))
    b = bursts(ev, dur, TIMEOUT)
    assert b == 5 and bursts(ev, dur, 6.0) == 1
    print("ok   %d bursts at timeout %.1f s, and the count matches the closed"
          " form at every timeout tried" % (b, TIMEOUT))
    print("     the flicker is the rule, not the drawing: at 6 s the other"
          " phone never once goes quiet, and no timeout breaks it mid-word.")

    # --- 6. deleting keeps the indicator UP. the detail the piece turns on.
    dels = [et for et, k, _ in ev if k == "del"]
    assert all(indicator(ev, et) for et in dels)
    print("ok   indicator up through all %d deletions -- destroying the"
          " message looks exactly like writing it" % len(dels))

    # --- 7. determinism
    e2, d2 = build_events()
    assert [(round(a, 9), b_, c) for a, b_, c in ev] == \
           [(round(a, 9), b_, c) for a, b_, c in e2]
    assert abs(dur - d2) < 1e-9
    print("ok   deterministic")

    # --- 8. pixels. the fullest frame: top half against bottom half.
    t_full = max((et for et, k, _ in ev if k == "ins"),
                 key=lambda x: len(text_at(ev, x)))
    full = pixels(draw(g, ev, t_full))
    a_ink = ink(full, g, A_TOP + 1, A_BOT, 1, COLS - 1)
    b_ink = ink(full, g, B_TOP + 1, B_BOT, 1, COLS - 1)
    assert a_ink > 6 * b_ink, (a_ink, b_ink)
    print("ok   at the fullest frame the top half draws %d lit pixels and"
          " the bottom half draws %d  (%.1fx)"
          % (a_ink, b_ink, a_ink / max(b_ink, 1)))

    # --- 9. and it empties.
    #
    # First written as "count lit pixels in the box" and it read 2310 on an
    # EMPTY box, because the box borders move as it grows and the window I
    # picked had swallowed a row of dashes. Counting ink says nothing about
    # what the ink is. So match the message COLOUR instead: LIVE is used for
    # exactly one thing in this file, the text being composed. That excludes
    # the borders and the label (CHROME), the two messages (RECV, SENT), the
    # dots (DOT_*) and the cursor (CURSOR) by construction, not by luck.
    t_empty = max(et for et, k, _ in ev if k == "del")
    emp = pixels(draw(g, ev, t_empty + 0.05))
    box = ink_of(emp, g, A_TOP + 1, A_BOT, 1, COLS - 1, LIVE)
    assert box == 0, box
    full_live = ink_of(full, g, A_TOP + 1, A_BOT, 1, COLS - 1, LIVE)
    per = full_live / float(len(DRAFT1))
    assert 100.0 < per < 400.0, per      # one glyph core, not a stray match
    print("ok   composed text goes %d lit pixels -> %d. all of it, gone."
          "  (%.0f px a character over %d characters)"
          % (full_live, box, per, len(DRAFT1)))

    # --- 10. both phones end holding the same four letters
    last = pixels(draw(g, ev, dur - 0.05))
    assert ink(last, g, A_REPLY, A_REPLY + 1, 1, COLS - 1) > 100
    assert ink(last, g, B_REPLY, B_REPLY + 1, 1, COLS - 1) > 100
    assert ink(last, g, B_TOP + 1, B_REPLY, 1, COLS - 1) > 100   # "you ok?"
    print("ok   last frame: 'yeah' on both screens")

    # --- 11. nothing is drawn in the gap between the two phones
    gap = ink(last, g, A_BOT + 1, B_TOP, 0, COLS)
    assert gap == 0, gap
    print("ok   the %d rows between the phones are empty"
          % (B_TOP - A_BOT - 1))

    # --- 12. all text sits inside the Shorts safe area
    assert A_TOP >= SAFE_TOP and B_BOT <= SAFE_BOT, (A_TOP, B_BOT)
    print("ok   everything drawn is in rows %d..%d, safe area is %d..%d"
          % (A_TOP, B_BOT, SAFE_TOP, SAFE_BOT))

    # --- 13. not a loop, on purpose
    f0 = pixels(draw(g, ev, 0.0))
    assert np.abs(f0 - last).max() > 0.2
    print("ok   first and last frame are different pictures (not a loop)")

    print("\nall checks ok -- %.2f s" % dur)


def render():
    g = Grid(font_size=FONT)
    ev, dur = build_events()
    n = int(round(dur * FPS))
    with Encoder(OUT, g, fps=FPS, crf=18, preset="slow") as enc:
        for f in range(n):
            enc.write(draw(g, ev, f / FPS))
            if f % 60 == 0:
                print("  %d/%d" % (f, n))
    print("wrote %s  %.2f s  %d frames  %d bytes"
          % (OUT, dur, n, os.path.getsize(OUT)))


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        render()
