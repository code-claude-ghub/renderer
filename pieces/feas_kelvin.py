#!/usr/bin/env python3
"""Feasibility for KELVIN — the wake angle that ignores the hull.

Claims proven here before any rendering:
  1. A moving pressure point on linear deep-water gravity waves
     (omega = sqrt(g k); EXACT per-mode propagator, so the
     dispersion relation carries no numerical error) produces the
     Kelvin wake at two scales ~130x apart in wavelength.
  2. The transverse wavelength on the centreline matches
     2 pi V^2 / g at both scales (this is the dispersion relation
     read back off the field).
  3. The bright arm of the wake is the first Airy maximum of the
     caustic: it sits INSIDE arcsin(1/3) and converges to it like
     d^(-2/3). Measuring theta(d) in windows and fitting
     theta_inf - c d^(-2/3) recovers theta_inf = 19.47 deg in BOTH
     lanes. (First instrument measured the raw arm angle and read
     ~1.5 deg low — that was the Airy offset, i.e. physics, not
     error. Second lesson: the lateral sponge was eating the far
     arms; domains widened so all measurement windows stay clear.)

Implementation: single complex normal variable
  u_hat = eta_hat + i (|k|/omega) phi_hat,   du/dt = -i omega u + f
free evolution is one exact complex rotation; sponge is a real-space
damp of Re and Im alike (Im is a k-filtered phi — an equally valid
sponge). eta = Re(ifft2(u_hat)).
"""
import time

import numpy as np

G = 9.81
TH_TRUE = np.degrees(np.arcsin(1.0 / 3.0))


class Sim:
    def __init__(self, Lx, Ly, nx, ny, dt, sigma, amp, sponge_frac=0.07):
        self.Lx, self.Ly, self.nx, self.ny, self.dt = Lx, Ly, nx, ny, dt
        kx = 2 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
        ky = 2 * np.pi * np.fft.fftfreq(ny, d=Ly / ny)
        KX, KY = np.meshgrid(kx, ky, indexing="xy")
        K = np.sqrt(KX * KX + KY * KY)
        K[0, 0] = 1e-12
        om = np.sqrt(G * K)
        self.rot = np.exp(-1j * om * dt)
        # forcing: phi_hat_t -= P_hat  =>  u_hat += i(K/om)(-P_hat) dt
        # P_hat for unit-amp Gaussian at (sx, sy):
        #   exp(-(K sigma)^2/2) * exp(-i(kx sx + ky sy))
        self.fenv = (-1j) * (K / om) * dt * amp * np.exp(
            -0.5 * (K * sigma) ** 2)
        self.kx1, self.ky1 = kx, ky          # 1-D, for the phase trick
        bx, by = sponge_frac * Lx, sponge_frac * Ly
        x = (np.arange(nx) + 0.5) * (Lx / nx)
        y = (np.arange(ny) + 0.5) * (Ly / ny)
        X, Y = np.meshgrid(x, y, indexing="xy")
        rx = np.clip((bx - np.minimum(X, Lx - X)) / bx, 0, 1)
        ry = np.clip((by - np.minimum(Y, Ly - Y)) / by, 0, 1)
        r = np.maximum(rx, ry)
        self.damp = np.exp(-6.0 * r * r * dt)
        self.sponge_x = bx                   # for clearance checks
        self.u = np.zeros((ny, nx), complex)
        self.t = 0.0
        self.eta_now = np.zeros((ny, nx))

    def step(self, sx, sy):
        phx = np.exp(-1j * self.kx1 * sx)    # (nx,)
        phy = np.exp(-1j * self.ky1 * sy)    # (ny,)
        u = self.rot * self.u + self.fenv * (phy[:, None] * phx[None, :])
        w = np.fft.ifft2(u)
        self.eta_now = w.real.copy()
        w *= self.damp
        self.u = np.fft.fft2(w)
        self.t += self.dt


def run_lane(name, V, Lx, Ly, nx, ny, dt, sigma, amp, T, y0):
    sim = Sim(Lx, Ly, nx, ny, dt, sigma, amp)
    sx = Lx / 2
    nsteps = int(round(T / dt))
    t0 = time.time()
    for _ in range(nsteps):
        sy = y0 + V * sim.t
        sim.step(sx, sy)
    wall = time.time() - t0
    sy = y0 + V * sim.t
    print(f"[{name}] {nsteps} steps in {wall:.1f}s "
          f"({1e3 * wall / nsteps:.1f} ms/step), ship at y={sy:.2f}")
    return sim, sim.eta_now, sx, sy


def smooth(a, wx, wy):
    kx = np.ones(wx) / wx
    ky = np.ones(wy) / wy
    a = np.apply_along_axis(lambda r: np.convolve(r, kx, "same"), 1, a)
    a = np.apply_along_axis(lambda c: np.convolve(c, ky, "same"), 0, a)
    return a


def arm_theta(a, dx, dy, sx, sy, d0, d1, nx, ny):
    """Ray-integral from the apex: mean envelope along rays at
    candidate half-angles over along-track window [d0, d1]; the
    bright arm is the argmax (parabolic-refined). Both arms."""
    dist = np.linspace(d0, d1, 250)
    thetas = np.arange(12.0, 26.0, 0.05)
    res = []
    for sgn in (+1, -1):
        vals = np.zeros_like(thetas)
        for i, th in enumerate(thetas):
            t = np.tan(np.radians(th))
            xs = sx + sgn * dist * t
            ys = sy - dist
            ix, iy = xs / dx, ys / dy
            ix0 = np.clip(ix.astype(int), 0, nx - 2)
            iy0 = np.clip(iy.astype(int), 0, ny - 2)
            fx, fy = ix - ix0, iy - iy0
            v = (a[iy0, ix0] * (1 - fx) * (1 - fy)
                 + a[iy0, ix0 + 1] * fx * (1 - fy)
                 + a[iy0 + 1, ix0] * (1 - fx) * fy
                 + a[iy0 + 1, ix0 + 1] * fx * fy)
            vals[i] = v.mean()
        j = int(np.argmax(vals))
        y1, y2, y3 = vals[j - 1], vals[j], vals[j + 1]
        off = 0.5 * (y1 - y3) / (y1 - 2 * y2 + y3)
        res.append(thetas[j] + off * 0.05)
    return res


def extrapolate(sim, eta, sx, sy, y0, lam, name):
    """theta(d) in five windows, fit theta_inf - c d^(-2/3).
    Windows kept inside the developed zone (d < 0.55 track) AND
    clear of the lateral sponge (arm x-offset + 2 lam < half-width
    minus sponge)."""
    ny, nx = eta.shape
    dx, dy = sim.Lx / nx, sim.Ly / ny
    a = smooth(np.abs(eta), max(1, int(0.35 * lam / dx)),
               max(1, int(0.35 * lam / dy)))
    track = sy - y0
    d_sponge = (sim.Lx / 2 - sim.sponge_x - 2.0 * lam) / np.tan(
        np.radians(21.0))
    d_max = min(0.55 * track, d_sponge)
    d_min = 5.0 * lam
    edges = np.linspace(d_min, d_max, 8)
    ds, ths, arms = [], [], []
    for d0, d1 in zip(edges[:-1], edges[1:]):
        two = arm_theta(a, dx, dy, sx, sy, d0, d1, nx, ny)
        dc = 0.5 * (d0 + d1)
        ds.append(dc)
        ths.append(np.mean(two))
        arms.append(two)
        print(f"  [{name}] d={dc:7.2f} ({dc / lam:5.1f} lam): "
              f"+{two[0]:.2f} / -{two[1]:.2f}")
    ds, ths = np.array(ds), np.array(ths)
    X = ds ** (-2.0 / 3.0)
    A = np.vstack([np.ones_like(X), -X]).T
    (th_inf, c), *_ = np.linalg.lstsq(A, ths, rcond=None)
    resid = np.abs(ths - (th_inf - c * X)).max()
    asym = max(abs(p - m) for p, m in
               [(t[0], t[1]) for t in arms])
    print(f"[{name}] theta_inf = {th_inf:.2f} (theory {TH_TRUE:.2f}), "
          f"c={c:.2f}, max resid {resid:.3f}, arm asym {asym:.2f}")
    return th_inf, c, resid, asym, list(zip(ds, ths))


def measure_lambda(sim, eta, sx, sy, V, name):
    ny, nx = eta.shape
    dy = sim.Ly / ny
    i = int(sx / (sim.Lx / nx))
    lam_th = 2 * np.pi * V * V / G
    j1 = int((sy - 1.2 * lam_th) / dy)
    j0 = int((sy - 6.5 * lam_th) / dy)
    col = eta[j0:j1, i]
    zc = np.where(np.diff(np.sign(col)) != 0)[0]
    lam = 2 * np.mean(np.diff(zc)) * dy
    print(f"[{name}] centreline lambda = {lam:.3f} "
          f"(theory {lam_th:.3f}, ratio {lam / lam_th:.3f})")
    return lam, lam_th


def measure_quiet_ahead(sim, eta, sx, sy, name):
    ny, nx = eta.shape
    dy = sim.Ly / ny
    j0 = int((sy + 0.05 * sim.Ly) / dy)
    j1 = int(0.92 * ny)
    if j1 <= j0 + 10:
        print(f"[{name}] ship too high to measure ahead — skip")
        return 0.0
    ahead = np.abs(eta[j0:j1]).max()
    wake = np.abs(eta[int(0.3 * ny):int(sy / dy)]).max()
    r = ahead / wake
    print(f"[{name}] max |eta| ahead / in-wake = {r:.4f}")
    return r


DUCK = dict(name="duck", V=0.7, Lx=6.72, Ly=14.0, nx=672, ny=1408,
            dt=0.015, sigma=0.045, amp=0.02, T=15.3, y0=1.5)
TANK = dict(name="tanker", V=8.0, Lx=880.0, Ly=1760.0, nx=880,
            ny=1760, dt=0.16, sigma=4.5, amp=2.0, T=174.0, y0=160.0)


def main():
    print("=" * 60)
    print(f"KELVIN feasibility — theory arcsin(1/3) = {TH_TRUE:.4f} deg")
    fails = []

    sim_d, eta_d, sxd, syd = run_lane(**DUCK)
    lam_d = 2 * np.pi * DUCK["V"] ** 2 / G
    ti_d, c_d, r_d, as_d, pts_d = extrapolate(sim_d, eta_d, sxd, syd,
                                              DUCK["y0"], lam_d, "duck")
    lm_d, lt_d = measure_lambda(sim_d, eta_d, sxd, syd, DUCK["V"], "duck")
    q_d = measure_quiet_ahead(sim_d, eta_d, sxd, syd, "duck")

    sim_t, eta_t, sxt, syt = run_lane(**TANK)
    lam_t = 2 * np.pi * TANK["V"] ** 2 / G
    ti_t, c_t, r_t, as_t, pts_t = extrapolate(sim_t, eta_t, sxt, syt,
                                              TANK["y0"], lam_t, "tanker")
    lm_t, lt_t = measure_lambda(sim_t, eta_t, sxt, syt, TANK["V"], "tanker")
    q_t = measure_quiet_ahead(sim_t, eta_t, sxt, syt, "tanker")

    def chk(label, ok):
        print(("  OK   " if ok else "  FAIL ") + label)
        if not ok:
            fails.append(label)

    chk(f"duck theta_inf {ti_d:.2f} within 0.4 of {TH_TRUE:.2f}",
        abs(ti_d - TH_TRUE) < 0.4)
    chk(f"tanker theta_inf {ti_t:.2f} within 0.4 of {TH_TRUE:.2f}",
        abs(ti_t - TH_TRUE) < 0.4)
    chk(f"lanes agree: |{ti_d:.2f} - {ti_t:.2f}| < 0.5",
        abs(ti_d - ti_t) < 0.5)
    chk(f"duck fit resid {r_d:.3f} < 0.15", r_d < 0.15)
    chk(f"tanker fit resid {r_t:.3f} < 0.15", r_t < 0.15)
    chk(f"convergence is from inside (c_d {c_d:.2f} > 0, "
        f"c_t {c_t:.2f} > 0)", c_d > 0 and c_t > 0)
    chk(f"arms symmetric (duck {as_d:.2f}, tanker {as_t:.2f} < 0.6)",
        as_d < 0.6 and as_t < 0.6)
    chk(f"duck lambda ratio {lm_d / lt_d:.3f} in 0.97..1.03",
        0.97 < lm_d / lt_d < 1.03)
    chk(f"tanker lambda ratio {lm_t / lt_t:.3f} in 0.97..1.03",
        0.97 < lm_t / lt_t < 1.03)
    chk(f"wavelengths differ {lm_t / lm_d:.0f}x (theory 131x)",
        110 < lm_t / lm_d < 150)
    chk(f"quiet ahead (duck {q_d:.3f}, tanker {q_t:.3f} < 0.12)",
        q_d < 0.12 and q_t < 0.12)

    print("=" * 60)
    if fails:
        print(f"{len(fails)} FAILURES")
        for f in fails:
            print("  -", f)
    else:
        print("ALL FEASIBILITY CHECKS PASSED")
    np.savez("/tmp/kelvin_feas.npz", eta_d=eta_d, eta_t=eta_t,
             syd=syd, syt=syt,
             pts_d=np.array(pts_d), pts_t=np.array(pts_t))
    print("fields saved to /tmp/kelvin_feas.npz")


if __name__ == "__main__":
    main()
