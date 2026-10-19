"""Radar flyby demo: track a target moving in 2D from noisy range/bearing
measurements, where EKF and UKF are compared against ground truth.

State x = [px, vx, py, vy] (position and velocity in each axis). The target
moves under a constant-velocity model -- *linear*, same as part 2's F. But
the radar, sitting at the origin, measures range and bearing:

    r     = sqrt(px^2 + py^2)
    theta = atan2(py, px)

which is a nonlinear function of the state. That's the whole story of why
this demo exists: you cannot write this measurement as H @ x for any fixed
matrix H, so part 2's plain linear filter can't be applied at all. The EKF
linearizes h at the current estimate (via its Jacobian) every step and
otherwise runs the linear filter; the UKF instead propagates sigma points
through the *exact* h and reconstructs a Gaussian from the result.

Run:
    source .venv/bin/activate
    python radar_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt

from kalman_ekf import ExtendedKalmanFilter
from kalman_ukf import UnscentedKalmanFilter

rng = np.random.default_rng(seed=3)

# ---------------------------------------------------------------------------
# 1. Simulate a target flying past a radar at the origin, and noisy
#    range/bearing measurements of it.
# ---------------------------------------------------------------------------
dt = 1.0
n_steps = 120

# True trajectory: starts at (-60, 8), moves at (+1, 0) m/s -- a close,
# near-tangential flyby (closest range ~8) where the bearing changes fastest
# and the range/bearing -> xy nonlinearity is most severe.
true_state = np.zeros((n_steps, 4))   # [px, vx, py, vy]
true_state[0] = [-60.0, 1.0, 8.0, 0.0]

vel_noise_std = 0.01   # small random accelerations (process noise)
for k in range(1, n_steps):
    px, vx, py, vy = true_state[k - 1]
    vx += rng.normal(0, vel_noise_std)
    vy += rng.normal(0, vel_noise_std)
    true_state[k] = [px + vx * dt, vx, py + vy * dt, vy]

range_noise_std = 1.0          # meters
bearing_noise_std = 0.03        # radians (~1.7 degrees)

true_range = np.hypot(true_state[:, 0], true_state[:, 2])
true_bearing = np.arctan2(true_state[:, 2], true_state[:, 0])
measurements = np.stack([
    true_range + rng.normal(0, range_noise_std, n_steps),
    true_bearing + rng.normal(0, bearing_noise_std, n_steps),
], axis=1)

# ---------------------------------------------------------------------------
# 2. Model: CV motion (linear) + range/bearing measurement (nonlinear).
# ---------------------------------------------------------------------------
F = np.array([[1, dt, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, dt],
              [0, 0, 0, 1]])


def fx(x):
    return F @ x


def Fjac(x):
    return F


def hx(x):
    px, _, py, _ = x
    return np.array([np.hypot(px, py), np.arctan2(py, px)])


def Hjac(x):
    px, _, py, _ = x
    r2 = px ** 2 + py ** 2
    r = np.sqrt(r2)
    return np.array([
        [px / r, 0, py / r, 0],
        [-py / r2, 0, px / r2, 0],
    ])


def residual_z(z, zhat):
    y = z - zhat
    y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi   # wrap bearing to [-pi, pi]
    return y


q = 0.01   # process noise power (random acceleration), same role as part 2's q
cv_block = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
Q = q * np.block([
    [cv_block, np.zeros((2, 2))],
    [np.zeros((2, 2)), cv_block],
])
R = np.diag([range_noise_std ** 2, bearing_noise_std ** 2])

# Deliberately poor initial guess (as if we only have the first measurement,
# converted naively, and no idea of velocity) with large initial uncertainty
# -- this is also where linearizing early vs. sampling matters most.
r0, theta0 = measurements[0]
x0 = np.array([r0 * np.cos(theta0), 0.0, r0 * np.sin(theta0), 0.0])
P0 = np.diag([25.0, 4.0, 25.0, 4.0])

ekf = ExtendedKalmanFilter(x=x0.copy(), P=P0.copy(), fx=fx, Fjac=Fjac, hx=hx, Hjac=Hjac,
                            Q=Q, R=R, residual_z=residual_z)
ukf = UnscentedKalmanFilter(x=x0.copy(), P=P0.copy(), fx=fx, hx=hx, Q=Q, R=R,
                             residual_z=residual_z)

ekf_est = np.zeros((n_steps, 4))
ukf_est = np.zeros((n_steps, 4))
ekf_est[0] = x0
ukf_est[0] = x0

for k in range(1, n_steps):
    ekf_est[k] = ekf.step(measurements[k])
    ukf_est[k] = ukf.step(measurements[k])

# ---------------------------------------------------------------------------
# 3. Compare against ground truth.
# ---------------------------------------------------------------------------
def pos_rmse(est, truth):
    err = est[:, [0, 2]] - truth[:, [0, 2]]
    return float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))


closest_k = int(np.argmin(true_range))
window = slice(max(0, closest_k - 15), closest_k + 15)

print(f"closest approach at step {closest_k}, true range {true_range[closest_k]:.2f} m\n")
print(f"{'filter':6s} {'pos RMSE (all steps)':>22s} {'pos RMSE (near closest approach)':>34s}")
for name, est in [("EKF", ekf_est), ("UKF", ukf_est)]:
    full_rmse = pos_rmse(est, true_state)
    near_rmse = pos_rmse(est[window], true_state[window])
    print(f"{name:6s} {full_rmse:22.3f} {near_rmse:34.3f}")

# ---------------------------------------------------------------------------
# 4. Plot: top-down trajectory, and position error over time.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
ax.plot(true_state[:, 0], true_state[:, 2], color="black", linewidth=1.5, label="true path")
ax.plot(ekf_est[:, 0], ekf_est[:, 2], color="tab:orange", linewidth=1.2, label="EKF estimate")
ax.plot(ukf_est[:, 0], ukf_est[:, 2], color="tab:blue", linewidth=1.2, label="UKF estimate")
ax.plot(0, 0, marker="^", color="black", markersize=10, label="radar")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Top-down view: target flyby past radar")
ax.legend(loc="upper center", fontsize=8)
ax.set_aspect("equal")

ax = axes[1]
t = np.arange(n_steps)
ekf_err = np.hypot(ekf_est[:, 0] - true_state[:, 0], ekf_est[:, 2] - true_state[:, 2])
ukf_err = np.hypot(ukf_est[:, 0] - true_state[:, 0], ukf_est[:, 2] - true_state[:, 2])
ax.plot(t, ekf_err, color="tab:orange", linewidth=1.2, label="EKF position error")
ax.plot(t, ukf_err, color="tab:blue", linewidth=1.2, label="UKF position error")
ax.axvline(closest_k, color="gray", linewidth=0.8, linestyle="--", label="closest approach")
ax.set_xlabel("time step (s)")
ax.set_ylabel("position error (m)")
ax.set_title("Position error over time")
ax.legend(loc="upper right", fontsize=8)

fig.tight_layout()
fig.savefig("radar_demo.png", dpi=150)
print("\nSaved plot to radar_demo.png")
