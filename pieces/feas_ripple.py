#!/usr/bin/env python3
"""Feasibility for RIPPLE — the sea's minimum wake speed.

Claims proven here before any rendering:
  1. Gravity-CAPILLARY dispersion omega = sqrt(g k + (sigma/rho) k^3)
     has a minimum phase speed c_min = (4 g sigma/rho)^(1/4)
     = 0.2312 m/s at lambda_min = 2 pi sqrt(sigma/(rho g)) = 1.71 cm
     (verified: Britannica; Rayleigh 1883; Raphael & de Gennes 1996).
  2. A steady pressure point moving at V = 0.20 m/s (< c_min) raises
     NO wake: c(k) = V has no real root, so no stationary mode exists.
     Only a bound, non-radiating dimple travels with the source.
  3. At V = 0.30 m/s (> c_min), c(k) = V has TWO roots:
     k_g = 120.8 (lambda 5.20 cm, gravity branch, cg < c: trails
     BEHIND) and k_c = 1115.5 (lambda 5.63 mm, capillary branch,
     cg > c: runs AHEAD of the swimmer). Both wavelengths are
     measured off the simulated field on the centreline.
  4. The ahead-ripples decay over L = (cg - V) / (2 nu k_c^2)
     ~ 4.9 cm (real viscosity nu = 1.0e-6, folded into the exact
     per-mode propagator) — which is why a pond shows a compact
     ripple fan in front of a swimming beetle, not an infinite one.

Implementation: KELVIN's single complex normal variable
  u = eta_hat + i (K/omega) phi_hat, exact rotation e^{-i omega dt},
plus two new EXACT per-mode factors folded into the same rotation:
  viscous decay e^{-2 nu K^2 dt}, and a Galilean shift e^{+i ky V dt}
that puts the frame on the swimmer (the source then sits still and
the wake's stationarity is the resonance omega = ky V accumulating
coherently). Sponge is a real-space damp near the frame edges =
an absorbing boundary travelling with the swimmer.
"""
import time

import numpy as np

G = 9.81
A = 7.28e-5           # sigma/rho: 0.0728 N/m / 1000 kg/m^3 (clean water)
NU = 1.0e-6           # kinematic viscosity, water 20 C
C_MIN = (4 * G * A) ** 0.25          # 0.2312 m/s
LAM_MIN = 2 * np.pi * np.sqrt(A / G)  # 1.71 cm


def roots(V):
    """The two wavenumbers with c(k) = V, from A k^2 - V^2 k + g = 0.
    Returns (k_gravity, k_capillary) or None below c_min."""
    disc = V ** 4 - 4 * A * G
    if disc < 0:
        return None
    s = np.sqrt(disc)
    return (V * V - s) / (2 * A), (V * V + s) / (2 * A)


def om_of(K):
    return np.sqrt(G * K + A * K ** 3)


def cg_of(k):
    return (G + 3 * A * k * k) / (2 * om_of(k))


class Sim:
    """Co-moving frame: source fixed at (sx, sy); water streams past
    at -V. rot folds free evolution, viscosity and the frame shift."""

    def __init__(self, V, Lx, Ly, nx, ny, dt, sigma, amp,
                 sponge_x=0.08, sponge_y=0.09, ramp=1.2):
        self.V, self.Lx, self.Ly = V, Lx, Ly
        self.nx, self.ny, self.dt = nx, ny, dt
        kx = 2 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
        ky = 2 * np.pi * np.fft.fftfreq(ny, d=Ly / ny)
        KX, KY = np.meshgrid(kx, ky, indexing="xy")
        K = np.sqrt(KX * KX + KY * KY)
        K[0, 0] = 1e-12
        om = om_of(K)
        self.rot = np.exp((-1j * om + 1j * KY * V - 2 * NU * K * K) * dt)
        self.fenv = (-1j) * (K / om) * dt * amp * np.exp(
            -0.5 * (K * sigma) ** 2)
        self.ramp = ramp
        bx, by = sponge_x * Lx, sponge_y * Ly
        x = (np.arange(nx) + 0.5) * (Lx / nx)
        y = (np.arange(ny) + 0.5) * (Ly / ny)
        X, Y = np.meshgrid(x, y, indexing="xy")
        rx = np.clip((bx - np.minimum(X, Lx - X)) / bx, 0, 1)
        ry = np.clip((by - np.minimum(Y, Ly - Y)) / by, 0, 1)
        r = np.maximum(rx, ry)
        self.damp = np.exp(-6.0 * r * r * dt)
        self.sponge_x, self.sponge_y = bx, by
        self.u = np.zeros((ny, nx), complex)
        self.t = 0.0
        self.eta_now = np.zeros((ny, nx))

    def step(self, sx, sy):
        # the source is static in the co-moving frame: cache the
        # forcing field (pure refactor; regrouping shifts floats by
        # ~1e-16, far under every tolerance)
        if getattr(self, "_fkey", None) != (sx, sy):
            phx = np.exp(-1j * 2 * np.pi * np.fft.fftfreq(
                self.nx, d=self.Lx / self.nx) * sx)
            phy = np.exp(-1j * 2 * np.pi * np.fft.fftfreq(
                self.ny, d=self.Ly / self.ny) * sy)
            self._force = self.fenv * (phy[:, None] * phx[None, :])
            self._fkey = (sx, sy)
        a = 0.5 * (1 - np.cos(np.pi * min(self.t / self.ramp, 1.0)))
        u = self.rot * self.u + a * self._force
        w = np.fft.ifft2(u)
        self.eta_now = w.real.copy()
        w *= self.damp
        self.u = np.fft.fft2(w)
        self.t += self.dt


# lane geometry (shared): 0.512 x 1.28 m at 0.889 mm/px grid
GEO = dict(Lx=0.512, Ly=1.28, nx=576, ny=1440, dt=1.0 / 210,
           sigma=0.0008, amp=0.003)
SRC_Y = 0.88          # source position in the co-moving frame
T_END = 16.2
LANE1 = dict(V=0.20, **GEO)          # below c_min
LANE2 = dict(V=0.30, **GEO)          # above c_min


def run_lane(name, V, T=T_END, **kw):
    sim = Sim(V, **kw)
    sx = kw["Lx"] / 2
    nsteps = int(round(T / kw["dt"]))
    t0 = time.time()
    mid = None
    for i in range(nsteps):
        sim.step(sx, SRC_Y)
        if mid is None and sim.t >= 12.5:
            mid = sim.eta_now.copy()
    wall = time.time() - t0
    print(f"[{name}] {nsteps} steps in {wall:.1f}s "
          f"({1e3 * wall / nsteps:.1f} ms/step)")
    return sim, sim.eta_now, mid, sx


def lam_behind(eta, sx, dx, dy, lo=0.10, hi=0.60):
    """Centreline zero-crossing wavelength in [SRC_Y-hi, SRC_Y-lo]."""
    i = int(sx / dx)
    j0, j1 = int((SRC_Y - hi) / dy), int((SRC_Y - lo) / dy)
    col = eta[j0:j1, i]
    zc = np.where(np.diff(np.sign(col)) != 0)[0]
    if len(zc) < 4:
        return np.nan
    return 2 * np.mean(np.diff(zc)) * dy


def lam_ahead(eta, sx, dx, dy, lo=0.015, hi=0.115):
    """Centreline spectral peak wavelength in [SRC_Y+lo, SRC_Y+hi]
    (the ripples decay ~ e^{-d/5cm}, so zero-crossings weight the
    noisy tail; a windowed FFT peak with parabolic refine does not)."""
    i = int(sx / dx)
    j0, j1 = int((SRC_Y + lo) / dy), int((SRC_Y + hi) / dy)
    col = eta[j0:j1, i] * np.hanning(j1 - j0)
    n = 8 * len(col)
    sp = np.abs(np.fft.rfft(col, n)) ** 2
    sp[: int(n * dy / 0.03)] = 0          # ignore lambda > 3 cm
    j = int(np.argmax(sp))
    y1, y2, y3 = sp[j - 1], sp[j], sp[j + 1]
    off = 0.5 * (y1 - y3) / (y1 - 2 * y2 + y3)
    freq = (j + off) / (n * dy)
    return 1.0 / freq


def silence(eta, sx, dx, dy, r_excl=0.08):
    """Max |eta| outside r_excl of the source, inside the sponge-free
    region — for the below-limit lane this is the whole claim."""
    ny, nx = eta.shape
    x = (np.arange(nx) + 0.5) * dx - sx
    y = (np.arange(ny) + 0.5) * dy - SRC_Y
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.sqrt(X * X + Y * Y)
    m = (R > r_excl)
    m &= (X + sx > 0.06) & (X + sx < 0.512 - 0.06)
    m &= (Y + SRC_Y > 0.14) & (Y + SRC_Y < 1.28 - 0.14)
    return np.abs(eta[m]).max()


def dimple(eta, sx, dx, dy):
    """Bound near-field: RMS within 3 cm vs annulus 10..20 cm."""
    ny, nx = eta.shape
    x = (np.arange(nx) + 0.5) * dx - sx
    y = (np.arange(ny) + 0.5) * dy - SRC_Y
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.sqrt(X * X + Y * Y)
    near = np.sqrt((eta[R < 0.03] ** 2).mean())
    far = np.sqrt((eta[(R > 0.10) & (R < 0.20)] ** 2).mean())
    return near, far


def decay_ahead(eta, sx, dx, dy, lo=0.02, hi=0.12):
    """e-folding length of the ripple envelope ahead (centreline,
    |eta| smoothed over one ripple wavelength, log-linear fit)."""
    i = int(sx / dx)
    j0, j1 = int((SRC_Y + lo) / dy), int((SRC_Y + hi) / dy)
    a = np.abs(eta[j0:j1, i])
    w = max(3, int(0.00563 / dy))
    env = np.convolve(a, np.ones(w) / w, "same")[w:-w]
    d = (np.arange(len(env)) + w + j0) * dy - SRC_Y
    good = env > env.max() * 1e-3
    p = np.polyfit(d[good], np.log(env[good]), 1)
    return -1.0 / p[0]


def main():
    print("=" * 60)
    print(f"RIPPLE feasibility — c_min = {C_MIN:.4f} m/s at "
          f"lambda = {LAM_MIN * 100:.2f} cm")
    fails = []

    def chk(label, ok):
        print(("  OK   " if ok else "  FAIL ") + label)
        if not ok:
            fails.append(label)

    # ---- arithmetic: the two roots and the no-root regime ----------
    r1 = roots(LANE1["V"])
    r2 = roots(LANE2["V"])
    chk(f"V={LANE1['V']}: c(k)=V has NO real root (below c_min)",
        r1 is None)
    kg, kc = r2
    lam_g_th = 2 * np.pi / kg
    lam_c_th = 2 * np.pi / kc
    print(f"  V={LANE2['V']}: k_g={kg:.1f} (lambda {lam_g_th * 100:.2f}"
          f" cm), k_c={kc:.1f} (lambda {lam_c_th * 1000:.2f} mm)")
    chk("gravity branch trails (cg < c at k_g)",
        cg_of(kg) < LANE2["V"])
    chk("capillary branch leads (cg > c at k_c)",
        cg_of(kc) > LANE2["V"])
    L_pred = (cg_of(kc) - LANE2["V"]) / (2 * NU * kc * kc)
    print(f"  predicted ahead decay length {L_pred * 100:.1f} cm")

    # ---- lane 2: the double wake -----------------------------------
    sim2, eta2, mid2, sx = run_lane("above", **LANE2)
    dx, dy = sim2.Lx / sim2.nx, sim2.Ly / sim2.ny
    lg = lam_behind(eta2, sx, dx, dy)
    lc = lam_ahead(eta2, sx, dx, dy)
    chk(f"lambda behind {lg * 100:.2f} cm vs theory "
        f"{lam_g_th * 100:.2f} (ratio {lg / lam_g_th:.3f} in "
        f"0.95..1.05)", 0.95 < lg / lam_g_th < 1.05)
    chk(f"lambda ahead {lc * 1000:.2f} mm vs theory "
        f"{lam_c_th * 1000:.2f} (ratio {lc / lam_c_th:.3f} in "
        f"0.92..1.08)", 0.92 < lc / lam_c_th < 1.08)
    lg_mid = lam_behind(mid2, sx, dx, dy)
    lc_mid = lam_ahead(mid2, sx, dx, dy)
    chk(f"steady state: behind {lg_mid * 100:.2f}->{lg * 100:.2f} cm, "
        f"ahead {lc_mid * 1000:.2f}->{lc * 1000:.2f} mm agree within "
        f"2%", abs(lg_mid / lg - 1) < 0.02 and abs(lc_mid / lc - 1)
        < 0.02)
    Ld = decay_ahead(eta2, sx, dx, dy)
    chk(f"ahead decay length {Ld * 100:.1f} cm vs predicted "
        f"{L_pred * 100:.1f} (in 3..8 cm)", 0.03 < Ld < 0.08)
    wake_max = silence(eta2, sx, dx, dy)
    print(f"  [above] wake max |eta| outside 8 cm: {wake_max:.3e}")

    # ---- lane 1: the silence ---------------------------------------
    sim1, eta1, mid1, _ = run_lane("below", **LANE1)
    quiet = silence(eta1, sx, dx, dy)
    chk(f"below c_min: far field {quiet:.2e} < 5% of above-lane wake "
        f"{wake_max:.2e} (ratio {quiet / wake_max:.4f})",
        quiet < 0.05 * wake_max)
    near, far = dimple(eta1, sx, dx, dy)
    chk(f"the dimple is bound: near RMS {near:.2e} > 20x annulus RMS "
        f"{far:.2e}", near > 20 * far)

    print("=" * 60)
    if fails:
        print(f"{len(fails)} FAILURES")
        for f in fails:
            print("  -", f)
    else:
        print("ALL FEASIBILITY CHECKS PASSED")
    np.savez("/tmp/ripple_feas.npz", eta1=eta1, eta2=eta2)
    print("fields saved to /tmp/ripple_feas.npz")


if __name__ == "__main__":
    main()
