"""Step 3 -- Stochastic Lanczos quadrature from mBCG output (paper eq. 5, 6, 8).

For a probe z with Lanczos run on A started at z/||z||:
    z^T log(A) z  ~=  ||z||^2 * e1^T log(T) e1
and with E[z z^T] = I, E[z^T log(A) z] = tr log A = log|A|.

e1^T log(T) e1 is computed from the eigendecomposition T = V diag(lam) V^T
(App. B, eq. S7):  sum_k V[0,k]^2 * log(lam_k).

Preconditioned case (eq. 8): probes z ~ N(0, P), the Lanczos start vector
is P^{-1/2} z, whose squared norm is z^T P^{-1} z (= rz0 returned by mbcg).
This estimates log|P^{-1/2} A P^{-1/2}|; add log|P| to get log|A|.
"""
import torch


def lanczos_quadrature(T, weights):
    """Return the per-probe estimates w_i * e1^T log(T_i) e1, shape (t,)."""
    evals, evecs = torch.linalg.eigh(T)
    evals = evals.clamp_min(torch.finfo(T.dtype).tiny)
    return weights * (evecs[:, 0, :] ** 2 * torch.log(evals)).sum(-1)


def slq_logdet(T, weights):
    """Average of the per-probe quadratures: estimate of log-determinant."""
    return lanczos_quadrature(T, weights).mean()
