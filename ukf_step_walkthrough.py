"""One concrete UKF predict+update cycle, step 0 -> step 1, with the actual
numbers from radar_demo.py's setup, printed at every stage.

Run:
    source .venv/bin/activate
    python ukf_step_walkthrough.py
"""

import numpy as np

np.set_printoptions(precision=3, suppress=True)

rng = np.random.default_rng(seed=3)

dt = 1.0
n_steps = 120
true_state = np.zeros((n_steps, 4))
true_state[0] = [-60.0, 1.0, 8.0, 0.0]
vel_noise_std = 0.01
for k in range(1, n_steps):
    px, vx, py, vy = true_state[k - 1]
    vx += rng.normal(0, vel_noise_std)
    vy += rng.normal(0, vel_noise_std)
    true_state[k] = [px + vx * dt, vx, py + vy * dt, vy]

range_noise_std = 1.0
bearing_noise_std = 0.03
true_range = np.hypot(true_state[:, 0], true_state[:, 2])
true_bearing = np.arctan2(true_state[:, 2], true_state[:, 0])
measurements = np.stack([
    true_range + rng.normal(0, range_noise_std, n_steps),
    true_bearing + rng.normal(0, bearing_noise_std, n_steps),
], axis=1)

F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
fx = lambda x: F @ x
hx = lambda x: np.array([np.hypot(x[0], x[2]), np.arctan2(x[2], x[0])])

q = 0.01
cv_block = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
Q = q * np.block([[cv_block, np.zeros((2, 2))], [np.zeros((2, 2)), cv_block]])
R = np.diag([range_noise_std ** 2, bearing_noise_std ** 2])

r0, theta0 = measurements[0]
x = np.array([r0 * np.cos(theta0), 0.0, r0 * np.sin(theta0), 0.0])
P = np.diag([25.0, 4.0, 25.0, 4.0])

print("=" * 70)
print("STARTING BELIEF (x0, P0) -- from naively converting the first measurement")
print("=" * 70)
print(f"x0 = {x}   # [px, vx, py, vy]")
print(f"P0 diagonal = {np.diag(P)}   # huge uncertainty, esp. on velocity")

# --- sigma point machinery (same as kalman_ukf.py) ---
n = 4
alpha, beta, kappa = 0.1, 2.0, 0.0
lam = alpha ** 2 * (n + kappa) - n
Wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
Wm[0] = lam / (n + lam)
Wc = Wm.copy()
Wc[0] += 1 - alpha ** 2 + beta

print(f"\nlambda = {lam:.4f}   Wm[0] = {Wm[0]:.4f}   Wc[0] = {Wc[0]:.4f}   Wm[1:] = Wc[1:] = {Wm[1]:.4f} (each)")


def sigma_points(x, P):
    S = np.linalg.cholesky((n + lam) * P)
    pts = np.empty((2 * n + 1, n))
    pts[0] = x
    for i in range(n):
        pts[1 + i] = x + S[:, i]
        pts[1 + n + i] = x - S[:, i]
    return pts


print("\n" + "=" * 70)
print("PREDICT (no measurement used yet -- just the motion model)")
print("=" * 70)
pts = sigma_points(x, P)
print(f"9 sigma points around x0 (showing all, [px, vx, py, vy]):")
for i, p in enumerate(pts):
    print(f"  X[{i}] = {p}")

pts_f = np.array([fx(p) for p in pts])
x_pred = Wm @ pts_f
diff = pts_f - x_pred
P_pred = diff.T @ (Wc[:, None] * diff) + Q
print(f"\nEach pushed through fx (coast forward by dt={dt}) -> recombine:")
print(f"x_pred = {x_pred}")
print(f"P_pred diagonal = {np.diag(P_pred)}")

print("\n" + "=" * 70)
print("UPDATE (bring in the real measurement z)")
print("=" * 70)
z = measurements[1]
print(f"actual measurement z (range, bearing) = {z}")
print(f"(true state at step 1 was {true_state[1]}, true range/bearing "
      f"{true_range[1]:.3f}, {true_bearing[1]:.3f} -- z is that plus sensor noise)")

pts2 = sigma_points(x_pred, P_pred)
pts_h = np.array([hx(p) for p in pts2])
zhat = Wm @ pts_h
dz = pts_h - zhat
dx = pts2 - x_pred
Pzz = dz.T @ (Wc[:, None] * dz) + R
Pxz = dx.T @ (Wc[:, None] * dz)
K = Pxz @ np.linalg.inv(Pzz)

print(f"\npredicted measurement zhat (range, bearing) = {zhat}")
print(f"Pzz (predicted measurement covariance) =\n{Pzz}")
print(f"\nKalman gain K =\n{K}")

y = z - zhat
y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi
print(f"\ninnovation y = z - zhat = {y}   # the 'surprise'")

x_new = x_pred + K @ y
P_new = P_pred - K @ Pzz @ K.T

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)
print(f"x_pred (before seeing z) = {x_pred}")
print(f"x_new  (after update)    = {x_new}")
print(f"true_state[1]            = {true_state[1]}")
print(f"\nP_pred diagonal = {np.diag(P_pred)}")
print(f"P_new  diagonal = {np.diag(P_new)}   # shrunk -- more confident after the update")
