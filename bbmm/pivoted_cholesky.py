"""Step 4 -- Pivoted Cholesky preconditioner (paper sec. 4.1, Appendix C).

Pivoted Cholesky builds a rank-k approximation K ~= L L^T by greedily
picking the pivot with the largest remaining Schur-complement diagonal.
It needs only diag(K) and k rows of K: O(k * rho(K) + n k^2) time.

The preconditioner is  P = L L^T + sigma^2 I, and BBMM needs three things
from it (sec. 4.1, "Requirements of BBMM preconditioners"):
  * solves P^{-1} R         -- Woodbury identity, O(n k^2)
  * log|P|                  -- matrix determinant lemma, O(n k^2)
  * samples z ~ N(0, P)     -- reparameterisation L e1 + sigma e2, O(n k)
"""
import math

import torch


def pivoted_cholesky(diag_fn, row_fn, n, rank, tol=1e-10, dtype=torch.float64, device=None):
    """Rank-`rank` pivoted Cholesky. Returns L of shape (n, k) with k <= rank.

    diag_fn(): returns diag(K), shape (n,)
    row_fn(i): returns row i of K, shape (n,)
    """
    d = diag_fn().clone()
    perm = torch.arange(n, device=device)
    L = torch.zeros(rank, n, dtype=dtype, device=device)

    k = 0
    for m in range(rank):
        i = m + torch.argmax(d[perm[m:]]).item()
        perm[[m, i]] = perm[[i, m]]
        pivot = perm[m]
        pivot_val = d[pivot]
        if pivot_val <= tol:
            break
        L[m, pivot] = torch.sqrt(pivot_val)
        others = perm[m + 1:]
        row = row_fn(pivot.item())
        L[m, others] = (row[others] - L[:m, pivot] @ L[:m, others]) / L[m, pivot]
        d[others] = d[others] - L[m, others] ** 2
        k = m + 1
        if d[others].sum() <= tol:
            break
    return L[:k].T.contiguous()


class PivotedCholeskyPreconditioner:
    """P = L L^T + sigma^2 I."""

    def __init__(self, L, noise):
        self.L = L
        self.noise = noise
        n, k = L.shape
        self.n, self.k = n, k
        # inner = sigma^2 I_k + L^T L   (k x k)
        inner = noise * torch.eye(k, dtype=L.dtype, device=L.device) + L.T @ L
        self.inner_chol = torch.linalg.cholesky(inner)

    def solve(self, R):
        """P^{-1} R via Woodbury: (R - L (sigma^2 I + L^T L)^{-1} L^T R) / sigma^2."""
        if self.k == 0:
            return R / self.noise
        tmp = torch.cholesky_solve(self.L.T @ R, self.inner_chol)
        return (R - self.L @ tmp) / self.noise

    def logdet(self):
        """log|L L^T + s I_n| = (n - k) log s + log|s I_k + L^T L|."""
        inner_logdet = 2.0 * torch.log(torch.diagonal(self.inner_chol)).sum()
        return (self.n - self.k) * math.log(self.noise) + inner_logdet

    def sample(self, t, generator=None):
        """t samples z ~ N(0, P), returned as (n, t)."""
        L = self.L
        e1 = torch.randn(self.k, t, dtype=L.dtype, device=L.device, generator=generator)
        e2 = torch.randn(self.n, t, dtype=L.dtype, device=L.device, generator=generator)
        return L @ e1 + math.sqrt(self.noise) * e2
