"""Grid-search Q against RMSE for the thermostat demo.

This only works because we have ground truth (it's a simulation). In the
real world you don't have true_temp to compare against -- see the tutorial
for how to tune using only the measurements (innovation statistics).
"""

import numpy as np

from kalman_1d import KalmanFilter1D
from thermostat_demo import true_temp, measurements, true_r

q_candidates = np.logspace(-5, 1, 25)
best = None

print(f"{'Q':>12s} {'RMSE':>8s}")
for q in q_candidates:
    kf = KalmanFilter1D(x=measurements[0], p=1.0, q=q, r=true_r)
    estimates = np.zeros_like(true_temp)
    estimates[0] = kf.x
    for k in range(1, len(measurements)):
        estimates[k] = kf.step(measurements[k])
    rmse = float(np.sqrt(np.mean((estimates - true_temp) ** 2)))
    print(f"{q:12.6f} {rmse:8.4f}")
    if best is None or rmse < best[1]:
        best = (q, rmse)

print(f"\nBest Q = {best[0]:.6f}  (RMSE = {best[1]:.4f})")
