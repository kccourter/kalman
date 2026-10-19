"""The same true radar-flyby trajectory from radar_demo.py, plotted in both
coordinate systems side by side: polar (range, bearing) -- what the radar
actually measures -- and Cartesian (x, y) -- what we're trying to track. A
straight line (constant velocity) in Cartesian is a curve in polar; that
curve is the nonlinearity the EKF/UKF exist to handle.

Run:
    source .venv/bin/activate
    python trajectory_coords_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=3)

dt = 1.0
n_steps = 120
true_state = np.zeros((n_steps, 4))   # [px, vx, py, vy]
true_state[0] = [-60.0, 1.0, 8.0, 0.0]

vel_noise_std = 0.01
for k in range(1, n_steps):
    px, vx, py, vy = true_state[k - 1]
    vx += rng.normal(0, vel_noise_std)
    vy += rng.normal(0, vel_noise_std)
    true_state[k] = [px + vx * dt, vx, py + vy * dt, vy]

true_range = np.hypot(true_state[:, 0], true_state[:, 2])
true_bearing = np.arctan2(true_state[:, 2], true_state[:, 0])

fig = plt.figure(figsize=(12, 5.5))

ax_polar = fig.add_subplot(1, 2, 1, projection="polar")
ax_polar.plot(true_bearing, true_range, color="black", linewidth=1.5)
ax_polar.scatter(true_bearing[0], true_range[0], color="tab:green", zorder=5, label="start")
ax_polar.scatter(true_bearing[-1], true_range[-1], color="tab:red", marker="s", zorder=5, label="end")
ax_polar.set_title("Polar: (range, bearing)\n-- what the radar measures")
ax_polar.legend(loc="upper right", fontsize=8)

ax_cart = fig.add_subplot(1, 2, 2)
ax_cart.plot(true_state[:, 0], true_state[:, 2], color="black", linewidth=1.5)
ax_cart.plot(0, 0, marker="^", color="black", markersize=10, label="radar")
ax_cart.scatter(true_state[0, 0], true_state[0, 2], color="tab:green", zorder=5, label="start")
ax_cart.scatter(true_state[-1, 0], true_state[-1, 2], color="tab:red", marker="s", zorder=5, label="end")
ax_cart.set_xlabel("x (m)")
ax_cart.set_ylabel("y (m)")
ax_cart.set_title("Cartesian: (x, y)\n-- what we want to track")
ax_cart.set_aspect("equal")
ax_cart.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)

fig.tight_layout()
fig.savefig("trajectory_coords_demo.png", dpi=150)
print("Saved plot to trajectory_coords_demo.png")
