"""The core reason the UKF exists, isolated from any filtering loop: take one
noisy polar measurement (range, bearing) and convert it to Cartesian (x, y).
That conversion -- x = r*cos(theta), y = r*sin(theta) -- is nonlinear, so a
Gaussian in (r, theta) doesn't map to a Gaussian in (x, y). Two ways to get a
mean/covariance estimate in (x, y) anyway:

  - EKF-style: linearize at the measurement (first-order Taylor expansion,
    via the Jacobian), then propagate the covariance through that linear
    approximation. One point, one derivative.
  - UKF-style: the unscented transform -- take 2n+1 sigma points, pass each
    through the *exact* nonlinear x=r*cos(theta), y=r*sin(theta), then take
    their weighted mean/covariance.

We compare both against Monte Carlo ground truth (transform a huge sample
of (r, theta) draws and look at the empirical mean/covariance), as the
bearing uncertainty grows -- i.e. as the nonlinearity gets stressed harder
relative to how tightly it's being evaluated.

Run:
    source .venv/bin/activate
    python polar_transform_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

rng = np.random.default_rng(0)

r0, theta0 = 10.0, 0.3     # true-ish polar measurement
sigma_r = 0.5
n_mc = 200_000


def monte_carlo_truth(sigma_theta):
    r_s = rng.normal(r0, sigma_r, n_mc)
    th_s = rng.normal(theta0, sigma_theta, n_mc)
    xy = np.stack([r_s * np.cos(th_s), r_s * np.sin(th_s)])
    return xy.mean(axis=1), np.cov(xy)


def ekf_style(sigma_theta):
    Sigma = np.diag([sigma_r ** 2, sigma_theta ** 2])
    mean = np.array([r0 * np.cos(theta0), r0 * np.sin(theta0)])
    # Jacobian of (x, y) wrt (r, theta) at the measurement.
    J = np.array([[np.cos(theta0), -r0 * np.sin(theta0)],
                  [np.sin(theta0),  r0 * np.cos(theta0)]])
    cov = J @ Sigma @ J.T
    return mean, cov


def ukf_style(sigma_theta, alpha=0.1, beta=2.0, kappa=0.0):
    n = 2
    Sigma = np.diag([sigma_r ** 2, sigma_theta ** 2])
    lam = alpha ** 2 * (n + kappa) - n
    S = np.linalg.cholesky((n + lam) * Sigma)
    z0 = np.array([r0, theta0])
    pts = np.empty((2 * n + 1, n))
    pts[0] = z0
    for i in range(n):
        pts[1 + i] = z0 + S[:, i]
        pts[1 + n + i] = z0 - S[:, i]

    Wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
    Wm[0] = lam / (n + lam)
    Wc = Wm.copy()
    Wc[0] += 1 - alpha ** 2 + beta

    pts_xy = np.stack([pts[:, 0] * np.cos(pts[:, 1]), pts[:, 0] * np.sin(pts[:, 1])], axis=1)
    mean = Wm @ pts_xy
    diff = pts_xy - mean
    cov = diff.T @ (Wc[:, None] * diff)
    return mean, cov


# ---------------------------------------------------------------------------
# 1. Table: as bearing uncertainty grows, how far off is each method's mean
#    and covariance from the Monte Carlo ground truth?
# ---------------------------------------------------------------------------
bearing_stds_deg = [2, 10, 20, 30, 45]

print(f"{'bearing std':>12s} {'mean err (EKF)':>15s} {'mean err (UKF)':>15s} "
      f"{'cov err (EKF)':>14s} {'cov err (UKF)':>14s}")
for deg in bearing_stds_deg:
    sigma_theta = np.radians(deg)
    mc_mean, mc_cov = monte_carlo_truth(sigma_theta)
    ekf_mean, ekf_cov = ekf_style(sigma_theta)
    ukf_mean, ukf_cov = ukf_style(sigma_theta)
    me_ekf = np.linalg.norm(ekf_mean - mc_mean)
    me_ukf = np.linalg.norm(ukf_mean - mc_mean)
    ce_ekf = np.linalg.norm(ekf_cov - mc_cov)
    ce_ukf = np.linalg.norm(ukf_cov - mc_cov)
    print(f"{deg:>10d}° {me_ekf:15.4f} {me_ukf:15.4f} {ce_ekf:14.4f} {ce_ukf:14.4f}")

# ---------------------------------------------------------------------------
# 2. Plot: at one illustrative bearing uncertainty, show the Monte Carlo
#    cloud (the true, curved "banana" distribution that a nonlinear function
#    produces from a Gaussian input) with the EKF's and UKF's 2-sigma
#    covariance ellipses overlaid.
# ---------------------------------------------------------------------------
def cov_ellipse(ax, mean, cov, n_std, **kwargs):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(vals)
    ax.add_patch(Ellipse(mean, width, height, angle=angle, fill=False, **kwargs))


fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

for ax, deg in zip(axes, [20, 40]):
    sigma_theta = np.radians(deg)
    r_s = rng.normal(r0, sigma_r, 3000)
    th_s = rng.normal(theta0, sigma_theta, 3000)
    xs, ys = r_s * np.cos(th_s), r_s * np.sin(th_s)
    ax.scatter(xs, ys, s=2, alpha=0.25, color="tab:gray", label="Monte Carlo samples")

    mc_mean, mc_cov = monte_carlo_truth(sigma_theta)
    ekf_mean, ekf_cov = ekf_style(sigma_theta)
    ukf_mean, ukf_cov = ukf_style(sigma_theta)

    cov_ellipse(ax, mc_mean, mc_cov, 2, edgecolor="black", linewidth=1.8, label="Monte Carlo (truth)")
    cov_ellipse(ax, ekf_mean, ekf_cov, 2, edgecolor="tab:orange", linewidth=1.8, label="EKF-style (linearized)")
    cov_ellipse(ax, ukf_mean, ukf_cov, 2, edgecolor="tab:blue", linewidth=1.8, label="UKF-style (sigma points)")
    ax.plot(*mc_mean, marker="x", color="black")
    ax.plot(*ekf_mean, marker="x", color="tab:orange")
    ax.plot(*ukf_mean, marker="x", color="tab:blue")

    ax.set_title(f"bearing std = {deg}°")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=7.5)

fig.tight_layout()
fig.savefig("polar_transform_demo.png", dpi=150)
print("\nSaved plot to polar_transform_demo.png")
