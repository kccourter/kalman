"""A minimal scalar (1D) Kalman filter.

State model (random walk):
    x_k = x_{k-1} + w_k        w_k ~ N(0, Q)   process noise
    z_k = x_k + v_k             v_k ~ N(0, R)   measurement noise

This is the simplest possible case: the thing we're estimating (true room
temperature) is modeled as a value that drifts slowly over time, and we get
noisy sensor readings of it.
"""

from dataclasses import dataclass


@dataclass
class KalmanFilter1D:
    x: float      # current state estimate
    p: float      # current estimate variance (uncertainty)
    q: float      # process noise variance (how much we expect x to drift per step)
    r: float      # measurement noise variance (how noisy the sensor is)

    def predict(self) -> None:
        # Random-walk model: best guess for next state is the current state.
        # Uncertainty grows because the process can drift.
        self.p = self.p + self.q

    def update(self, z: float) -> None:
        # Kalman gain: how much we trust the new measurement vs. our prediction.
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (z - self.x)
        self.p = (1 - k) * self.p

    def step(self, z: float) -> float:
        """Run one predict+update cycle given a new measurement; return the estimate."""
        self.predict()
        self.update(z)
        return self.x
