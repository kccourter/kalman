"""A general linear Kalman filter for state vectors of any dimension.

Model:
    x_k = F x_{k-1} + w_k        w_k ~ N(0, Q)
    z_k = H x_k + v_k             v_k ~ N(0, R)

This is the matrix generalization of kalman_1d.KalmanFilter1D: the same five
predict/update lines, just with matrices instead of scalars. x is a state
vector (n,), P its covariance (n, n). H maps state space to measurement
space (m, n); with m=1 (a single sensor reading per step) H is a (1, n) row
vector and R is a (1, 1) matrix.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanFilter:
    x: np.ndarray   # state estimate, shape (n,)
    P: np.ndarray   # state covariance, shape (n, n)
    F: np.ndarray   # state transition matrix, shape (n, n)
    H: np.ndarray   # measurement matrix, shape (m, n)
    Q: np.ndarray   # process noise covariance, shape (n, n)
    R: np.ndarray   # measurement noise covariance, shape (m, m)

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z) -> None:
        z = np.atleast_1d(z)
        y = z - self.H @ self.x                   # innovation
        S = self.H @ self.P @ self.H.T + self.R     # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)    # Kalman gain
        self.x = self.x + K @ y
        n = self.x.shape[0]
        self.P = (np.eye(n) - K @ self.H) @ self.P

    def step(self, z) -> np.ndarray:
        """Run one predict+update cycle given a new measurement; return the estimate."""
        self.predict()
        self.update(z)
        return self.x.copy()
