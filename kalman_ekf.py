"""A general Extended Kalman Filter for nonlinear state transition and/or
measurement functions, linearized at each step via their Jacobians.

Model:
    x_k = f(x_{k-1}) + w_k        w_k ~ N(0, Q)
    z_k = h(x_k) + v_k             v_k ~ N(0, R)

Same predict/update skeleton as kalman_nd.KalmanFilter, but F and H are no
longer fixed matrices -- they're the Jacobians of f and h, re-evaluated at
the current estimate every step ("linearize, then run the linear filter").
Compare with kalman_ukf.py, which instead propagates sample points through
the true nonlinear f/h and never needs a derivative at all.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class ExtendedKalmanFilter:
    x: np.ndarray                                # state estimate, shape (n,)
    P: np.ndarray                                 # state covariance, shape (n, n)
    fx: Callable[[np.ndarray], np.ndarray]        # nonlinear state transition
    Fjac: Callable[[np.ndarray], np.ndarray]      # Jacobian of fx at x, (n, n)
    hx: Callable[[np.ndarray], np.ndarray]        # nonlinear measurement function
    Hjac: Callable[[np.ndarray], np.ndarray]      # Jacobian of hx at x, (m, n)
    Q: np.ndarray
    R: np.ndarray
    residual_z: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda z, zhat: z - zhat

    def predict(self) -> None:
        F = self.Fjac(self.x)
        self.x = self.fx(self.x)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z) -> None:
        z = np.atleast_1d(z)
        H = self.Hjac(self.x)
        zhat = self.hx(self.x)
        y = self.residual_z(z, zhat)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        n = self.x.shape[0]
        self.P = (np.eye(n) - K @ H) @ self.P

    def step(self, z) -> np.ndarray:
        """Run one predict+update cycle given a new measurement; return the estimate."""
        self.predict()
        self.update(z)
        return self.x.copy()
