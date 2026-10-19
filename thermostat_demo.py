"""Thermostat demo: track a room's true temperature from a noisy sensor
using a 1D Kalman filter, and show how Q (process noise) and R (measurement
noise) change the filter's behavior.

Run:
    source .venv/bin/activate
    python thermostat_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt

from kalman_1d import KalmanFilter1D

rng = np.random.default_rng(seed=7)

# ---------------------------------------------------------------------------
# 1. Simulate the "true" room temperature and a noisy sensor reading of it.
# ---------------------------------------------------------------------------
n_steps = 300
dt_minutes = 1

true_temp = np.zeros(n_steps)
true_temp[0] = 20.0

# Physical process noise: real room temp wobbles a little minute to minute
# even with nothing happening (drafts, sunlight, etc).
process_noise_std = 0.05

for k in range(1, n_steps):
    true_temp[k] = true_temp[k - 1] + rng.normal(0, process_noise_std)
    # Heater kicks on around step 80 and ramps the room up over ~40 minutes.
    if 80 <= k < 120:
        true_temp[k] += 0.075
    # Someone opens a window around step 220, room cools over ~30 minutes.
    if 220 <= k < 250:
        true_temp[k] -= 0.06

# Sensor noise: the thermostat's reading is noisier than the real physics.
sensor_noise_std = 0.6
measurements = true_temp + rng.normal(0, sensor_noise_std, size=n_steps)

# ---------------------------------------------------------------------------
# 2. Run the filter with three different tunings, report numbers, and plot.
#    (guarded so other scripts can `import thermostat_demo` just for the
#    simulated true_temp / measurements / true_r arrays above)
# ---------------------------------------------------------------------------
true_r = sensor_noise_std ** 2          # ~0.36, matches the sensor
true_q = process_noise_std ** 2         # ~0.0025, matches the slow drift

if __name__ == "__main__":
    # The heater/window events move the room faster than the baseline drift,
    # so a Q tuned a bit above the raw physical noise tracks those events better.
    #tuned_q = 0.01
    tuned_q = 0.017783

    settings = {
        "well-tuned (Q=0.017783, R=0.36)": dict(q=tuned_q, r=true_r),
        "too smooth (Q=0.0001, R=0.36)": dict(q=0.0001, r=true_r),
        "too jittery (Q=1.0, R=0.36)": dict(q=1.0, r=true_r),
    }

    results = {}
    for name, params in settings.items():
        kf = KalmanFilter1D(x=measurements[0], p=1.0, q=params["q"], r=params["r"])
        estimates = np.zeros(n_steps)
        gains = np.zeros(n_steps)
        estimates[0] = kf.x
        for k in range(1, n_steps):
            kf.predict()
            gain = kf.p / (kf.p + kf.r)
            kf.update(measurements[k])
            estimates[k] = kf.x
            gains[k] = gain
        rmse = float(np.sqrt(np.mean((estimates - true_temp) ** 2)))
        results[name] = dict(estimates=estimates, gains=gains, rmse=rmse, **params)

    print(f"{'tuning':35s} {'Q':>10s} {'R':>8s} {'steady-state K':>15s} {'RMSE':>8s}")
    for name, r in results.items():
        steady_gain = r["gains"][-1]
        print(f"{name:35s} {r['q']:10.5f} {r['r']:8.3f} {steady_gain:15.4f} {r['rmse']:8.3f}")

    raw_rmse = float(np.sqrt(np.mean((measurements - true_temp) ** 2)))
    print(f"\n(raw, unfiltered sensor RMSE vs. true temp: {raw_rmse:.3f})")

    fig, axes = plt.subplots(len(settings), 1, figsize=(10, 9), sharex=True)
    t = np.arange(n_steps)
    for ax, (name, r) in zip(axes, results.items()):
        ax.plot(t, true_temp, label="true temp", color="black", linewidth=1.5)
        ax.plot(t, measurements, label="raw sensor", color="tab:gray", alpha=0.4, linewidth=0.8)
        ax.plot(t, r["estimates"], label="Kalman estimate", color="tab:red", linewidth=1.5)
        ax.set_title(f"{name}  (RMSE={r['rmse']:.3f})")
        ax.set_ylabel("°C")
        ax.legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("time step (minutes)")
    fig.tight_layout()
    fig.savefig("thermostat_demo.png", dpi=150)
    print("\nSaved plot to thermostat_demo.png")
