"""A general Unscented Kalman Filter for nonlinear state transition and/or
measurement functions.

Model:
    x_k = f(x_{k-1}) + w_k        w_k ~ N(0, Q)
    z_k = h(x_k) + v_k             v_k ~ N(0, R)

Unlike kalman_nd.KalmanFilter, f and h don't have to be linear -- there's no
F or H matrix. Instead of linearizing them (that's the Extended Kalman
Filter, kalman_ekf.py), the UKF propagates a small, deterministic set of
sample points -- "sigma points" -- through the true nonlinear f and h, then
reconstructs a Gaussian from the transformed points' weighted mean and
covariance. This "unscented transform" captures the mean and covariance of
f(x)/h(x) to higher order than a linearization does, without ever computing
a derivative, and it's exact whenever f/h actually are linear (it reduces
to exactly the linear Kalman filter in that case).
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class UnscentedKalmanFilter:
    x: np.ndarray                               # state estimate, shape (n,)
    P: np.ndarray                                # state covariance, shape (n, n)
    fx: Callable[[np.ndarray], np.ndarray]       # nonlinear state transition
    hx: Callable[[np.ndarray], np.ndarray]       # nonlinear measurement function
    Q: np.ndarray                                # process noise covariance, (n, n)
    R: np.ndarray                                # measurement noise covariance, (m, m)
    alpha: float = 0.1     # sigma point spread
    beta: float = 2.0      # prior knowledge of the distribution's shape (2 is optimal for Gaussians)
    kappa: float = 0.0     # secondary scaling parameter
    residual_z: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda z, zhat: z - zhat
    # ^ override for measurements with wraparound (e.g. an angle), where plain
    #   subtraction gives the wrong sign/magnitude of the innovation.

    def __post_init__(self) -> None:
        n = self.x.shape[0]
        self.n = n
        self.lambda_ = self.alpha ** 2 * (n + self.kappa) - n

        # weights for recombining sigma points into a mean (Wm) and a
        # covariance (Wc); they differ only in the center weight.
        self.Wm = np.full(2 * n + 1, 1.0 / (2 * (n + self.lambda_)))
        self.Wm[0] = self.lambda_ / (n + self.lambda_)
        self.Wc = self.Wm.copy()
        self.Wc[0] += 1 - self.alpha ** 2 + self.beta

    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """2n+1 points: the mean, plus n pairs straddling it along each
        eigenvector of the (scaled) covariance -- a cheap stand-in for "the
        full distribution" that's exact to 2nd order for any function."""
        n = self.n
        S = np.linalg.cholesky((n + self.lambda_) * P)
        pts = np.empty((2 * n + 1, n))
        pts[0] = x
        for i in range(n):
            pts[1 + i] = x + S[:, i]
            pts[1 + n + i] = x - S[:, i]
        return pts

    def predict(self) -> None:
        pts_f = np.array([self.fx(p) for p in self._sigma_points(self.x, self.P)])
        self.x = self.Wm @ pts_f
        diff = pts_f - self.x
        self.P = diff.T @ (self.Wc[:, None] * diff) + self.Q

    def update(self, z) -> None:
        z = np.atleast_1d(z)
        pts = self._sigma_points(self.x, self.P)
        pts_h = np.array([self.hx(p) for p in pts])
        zhat = self.Wm @ pts_h

        dz = pts_h - zhat
        dx = pts - self.x

        Pzz = dz.T @ (self.Wc[:, None] * dz) + self.R
        Pxz = dx.T @ (self.Wc[:, None] * dz)

        K = Pxz @ np.linalg.inv(Pzz)
        y = self.residual_z(z, zhat)
        self.x = self.x + K @ y
        self.P = self.P - K @ Pzz @ K.T

    def step(self, z) -> np.ndarray:
        """Run one predict+update cycle given a new measurement; return the estimate."""
        self.predict()
        self.update(z)
        return self.x.copy()
