"""Constant-velocity (position + rate-of-change) Kalman filter vs. the plain
random-walk filter, on the same simulated thermostat data.

State vector x = [temp, rate_of_change]. The state equation says:

    temp_k  = temp_{k-1} + rate_{k-1} * dt
    rate_k  = rate_{k-1}

i.e. "assume the current trend continues," instead of the random walk's
"assume nothing changes." This lets the filter anticipate ramps (like the
heater turning on) instead of only reacting to them after the fact.

Run:
    source .venv/bin/activate
    python thermostat_cv_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt

from kalman_1d import KalmanFilter1D
from kalman_nd import KalmanFilter
from thermostat_demo import true_temp, measurements, true_r, n_steps

dt = 1.0

# --- constant-velocity model matrices -------------------------------------
F = np.array([[1.0, dt],
              [0.0, 1.0]])
H = np.array([[1.0, 0.0]])
R = np.array([[true_r]])

# Discrete white-noise-acceleration process noise: q is the (tunable) power
# of random acceleration -- how much the *rate* is allowed to randomly
# change per step. This single scalar induces correlated noise on both
# position and velocity in a physically sensible way.
# (q=0.0001 found via tune_cv.py's grid search against ground-truth RMSE)
q = 0.0001
Q = q * np.array([[dt**4 / 4, dt**3 / 2],
                   [dt**3 / 2, dt**2]])

cv = KalmanFilter(
    x=np.array([measurements[0], 0.0]),
    P=np.diag([1.0, 1.0]),
    F=F, H=H, Q=Q, R=R,
)

# --- baseline: the well-tuned random-walk (position-only) filter from
#     thermostat_demo.py, for comparison -----------------------------------
rw = KalmanFilter1D(x=measurements[0], p=1.0, q=0.01, r=true_r)

cv_estimates = np.zeros(n_steps)
cv_velocity = np.zeros(n_steps)
rw_estimates = np.zeros(n_steps)
cv_estimates[0] = cv.x[0]
rw_estimates[0] = rw.x

for k in range(1, n_steps):
    x = cv.step(measurements[k])
    cv_estimates[k] = x[0]
    cv_velocity[k] = x[1]
    rw_estimates[k] = rw.step(measurements[k])

cv_rmse = float(np.sqrt(np.mean((cv_estimates - true_temp) ** 2)))
rw_rmse = float(np.sqrt(np.mean((rw_estimates - true_temp) ** 2)))
raw_rmse = float(np.sqrt(np.mean((measurements - true_temp) ** 2)))

print(f"raw sensor RMSE:               {raw_rmse:.3f}")
print(f"random-walk filter RMSE:       {rw_rmse:.3f}")
print(f"constant-velocity filter RMSE: {cv_rmse:.3f}")

# Lag check: how long after the heater starts (step 80) does each filter's
# estimate first cross 21.0C, vs. the true temp?
true_cross = next(k for k in range(80, n_steps) if true_temp[k] >= 21.0)
rw_cross = next((k for k in range(80, n_steps) if rw_estimates[k] >= 21.0), None)
cv_cross = next((k for k in range(80, n_steps) if cv_estimates[k] >= 21.0), None)
print(f"\ntrue temp crosses 21.0C at step {true_cross}")
print(f"random-walk filter crosses it at step {rw_cross}  (lag: {rw_cross - true_cross})")
print(f"constant-velocity filter crosses it at step {cv_cross}  (lag: {cv_cross - true_cross})")

# --- plot -------------------------------------------------------------------
t = np.arange(n_steps)
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                          gridspec_kw={"height_ratios": [2, 1]})

ax = axes[0]
ax.plot(t, true_temp, label="true temp", color="black", linewidth=1.5)
ax.plot(t, measurements, label="raw sensor", color="tab:gray", alpha=0.35, linewidth=0.8)
ax.plot(t, rw_estimates, label=f"random-walk filter (RMSE={rw_rmse:.3f})", color="tab:red", linewidth=1.3)
ax.plot(t, cv_estimates, label=f"constant-velocity filter (RMSE={cv_rmse:.3f})", color="tab:blue", linewidth=1.3)
ax.axvspan(80, 120, color="orange", alpha=0.08)
ax.axvspan(220, 250, color="purple", alpha=0.08)
ax.set_ylabel("°C")
ax.set_title("Random-walk vs. constant-velocity Kalman filter")
ax.legend(loc="upper left", fontsize=8)

ax = axes[1]
ax.plot(t, cv_velocity, color="tab:blue", linewidth=1.2)
ax.axhline(0, color="black", linewidth=0.6)
ax.axvspan(80, 120, color="orange", alpha=0.08, label="heater on")
ax.axvspan(220, 250, color="purple", alpha=0.08, label="window open")
ax.set_ylabel("°C / min")
ax.set_xlabel("time step (minutes)")
ax.set_title("constant-velocity filter's estimated rate of change (hidden state)")
ax.legend(loc="upper left", fontsize=8)

fig.tight_layout()
fig.savefig("thermostat_cv_demo.png", dpi=150)
print("\nSaved plot to thermostat_cv_demo.png")
