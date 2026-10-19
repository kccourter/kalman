"""Grid-search the constant-velocity model's process-noise scalar q against
RMSE, same idea as tune.py but for the 2-state filter. Only possible here
because we have ground truth (simulation) -- see tutorial_2_constant_velocity.md.
"""

import numpy as np

from kalman_nd import KalmanFilter
from thermostat_demo import true_temp, measurements, true_r, n_steps

dt = 1.0
F = np.array([[1.0, dt], [0.0, 1.0]])
H = np.array([[1.0, 0.0]])
R = np.array([[true_r]])

q_candidates = np.logspace(-6, 0, 25)
best = None

print(f"{'q':>12s} {'RMSE':>8s}")
for q in q_candidates:
    Q = q * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
    kf = KalmanFilter(x=np.array([measurements[0], 0.0]), P=np.diag([1.0, 1.0]), F=F, H=H, Q=Q, R=R)
    estimates = np.zeros(n_steps)
    estimates[0] = kf.x[0]
    for k in range(1, n_steps):
        estimates[k] = kf.step(measurements[k])[0]
    rmse = float(np.sqrt(np.mean((estimates - true_temp) ** 2)))
    print(f"{q:12.6f} {rmse:8.4f}")
    if best is None or rmse < best[1]:
        best = (q, rmse)

print(f"\nBest q = {best[0]:.6f}  (RMSE = {best[1]:.4f})")
