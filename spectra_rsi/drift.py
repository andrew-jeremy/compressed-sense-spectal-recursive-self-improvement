"""Dictionary transport and drift monitoring (Sec. 4.3).

- Orthogonal Procrustes alignment of successive dictionaries.
- Principal angles quantify subspace motion.
- ResidualDriftMonitor: anytime-valid confidence sequence on predictive
  residuals; a boundary crossing triggers re-probing / dense fallback.
"""
import numpy as np


def procrustes_transport(Psi_new, Psi_old):
    """Q = argmin_{Q^T Q = I} ||Psi_new - Psi_old Q||_F  (rotation of old
    coordinates into the new basis). Returns (Q, alignment_residual)."""
    k = min(Psi_new.shape[1], Psi_old.shape[1])
    A = Psi_old[:, :k].T @ Psi_new[:, :k]
    U, _, Vt = np.linalg.svd(A)
    Q = U @ Vt
    resid = np.linalg.norm(Psi_new[:, :k] - Psi_old[:, :k] @ Q, "fro") / np.sqrt(k)
    return Q, resid


def principal_angles(Psi_a, Psi_b):
    """Principal angles (radians) between column spaces."""
    Qa, _ = np.linalg.qr(Psi_a)
    Qb, _ = np.linalg.qr(Psi_b)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.arccos(np.clip(s, -1, 1))


class ResidualDriftMonitor:
    """Sub-Gaussian mixture confidence sequence on the mean squared
    normalized residual. Anytime-valid: crossing is a drift alarm at level
    alpha regardless of when you look (Howard et al. style mixture bound)."""

    def __init__(self, alpha=0.05, scale=1.0):
        self.alpha = alpha
        self.scale = scale       # residuals assumed sub-Gaussian(scale)
        self.t = 0
        self.sum = 0.0

    def update(self, residuals):
        r = np.atleast_1d(residuals)
        self.t += r.size
        self.sum += float(np.sum(r))
        return self.is_alarmed()

    def radius(self):
        if self.t == 0:
            return np.inf
        t, s, a = self.t, self.scale, self.alpha
        # normal-mixture boundary: valid at all t simultaneously
        rho = 1.0
        return s * np.sqrt(2 * (t + rho) / t ** 2
                           * np.log(np.sqrt((t + rho) / rho) / a))

    def mean(self):
        return self.sum / max(self.t, 1)

    def is_alarmed(self):
        """Alarm when the anytime lower confidence bound on the mean
        residual exceeds 0 (systematic prediction error)."""
        return self.t > 0 and (self.mean() - self.radius()) > 0.0
