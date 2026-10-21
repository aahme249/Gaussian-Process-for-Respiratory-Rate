"""Step 6 -- BBMM inference: marginal log-likelihood, its gradient, predictions.

The three expensive terms of GP training (eq. 2) all come from ONE mBCG call
on  [y, z_1, ..., z_t]  with probes z_i ~ N(0, P)   (eq. 3, 7):

  * K^{-1} y                 -> first column u_0
  * log|K|                   -> SLQ on T_1..T_t, plus log|P|          (eq. 6, 8)
  * Tr(K^{-1} dK/dtheta)     -> (1/t) sum_i (K^{-1} z_i)^T dK/dtheta (P^{-1} z_i)   (eq. 9)

Gradient trick: the derivative of
    S(theta) = (1/t) sum_i <K^{-1} z_i, K(theta) P^{-1} z_i>  -  <u_0, K(theta) u_0>
with the left/right vectors held fixed is exactly
    Tr(K^{-1} dK/dtheta) - y^T K^{-1} dK/dtheta K^{-1} y  =  d/dtheta [log|K| + y^T K^{-1} y].
So one extra *blackbox* matmul K(theta) [P^{-1}Z, u_0] plus autograd gives every
hyperparameter gradient -- no derivative code per model.
"""
import math

import torch

from .mbcg import mbcg
from .pivoted_cholesky import PivotedCholeskyPreconditioner, pivoted_cholesky
from .slq import slq_logdet


class BBMMSettings:
    def __init__(self, num_probes=10, precond_rank=5, max_iter=100, max_tridiag_iter=30, tol=1e-4):
        self.num_probes = num_probes
        self.precond_rank = precond_rank
        self.max_iter = max_iter
        self.max_tridiag_iter = max_tridiag_iter
        self.tol = tol


def build_preconditioner(op, rank):
    noise = float(op.noise.detach())
    with torch.no_grad():
        L = pivoted_cholesky(lambda: op.kernel_diag().detach(), lambda i: op.kernel_row(i).detach(),
                             op.n, rank, dtype=op.dtype, device=op.device)
    return PivotedCholeskyPreconditioner(L, noise)


def bbmm_marginal_log_likelihood(op, y, settings=None, generator=None, return_info=False):
    """log p(y | X, theta) = -1/2 y^T K^{-1} y - 1/2 log|K| - n/2 log 2pi, computed with BBMM.

    `op` is any KernelOperator; gradients flow to whatever parameters op.matmul uses.
    """
    s = settings or BBMMSettings()
    n = op.n
    precond = build_preconditioner(op, s.precond_rank)

    # ---- one mBCG call on [y, z_1..z_t] (eq. 3, 7) -- no autograd here
    Z = precond.sample(s.num_probes, generator=generator)
    with torch.no_grad():
        B = torch.cat([y.unsqueeze(-1), Z], dim=1)
        Usol, T, info = mbcg(op.matmul, B, precond=precond.solve, max_iter=s.max_iter,
                             tol=s.tol, max_tridiag_iter=s.max_tridiag_iter)
        u0, Kinv_Z = Usol[:, 0], Usol[:, 1:]

        inv_quad = y @ u0                                            # y^T K^{-1} y
        logdet = slq_logdet(T[1:], info["rz0"][1:]) + precond.logdet()  # eq. 8 + log|P|
        Pinv_Z = precond.solve(Z)

    # ---- surrogate whose gradient is d/dtheta [y^T K^{-1} y + log|K|] (eq. 2, 9)
    t = s.num_probes
    right = torch.cat([Pinv_Z / t, u0.unsqueeze(-1)], dim=1)
    K_right = op.matmul(right)                                       # single blackbox MMM
    surrogate = (Kinv_Z * K_right[:, :t]).sum() - (u0 * K_right[:, t]).sum()

    value = -0.5 * (inv_quad + logdet) - 0.5 * n * math.log(2 * math.pi)
    mll = value + (-0.5) * (surrogate - surrogate.detach())
    if return_info:
        return mll, {"inv_quad": inv_quad, "logdet": logdet, "num_iter": info["num_iter"]}
    return mll


def bbmm_predict(op, cross_cov, test_diag, y, settings=None):
    """Predictive mean/variance (eq. 1) with a single batched mBCG call on [y, K_X*].

    cross_cov: K_X* of shape (n, n_test); test_diag: k(x*, x*) of shape (n_test,)
    """
    s = settings or BBMMSettings()
    precond = build_preconditioner(op, s.precond_rank)
    with torch.no_grad():
        B = torch.cat([y.unsqueeze(-1), cross_cov], dim=1)
        Usol, _, _ = mbcg(op.matmul, B, precond=precond.solve, max_iter=s.max_iter, tol=s.tol,
                          max_tridiag_iter=1)
        mean = cross_cov.T @ Usol[:, 0]
        var = test_diag - (cross_cov * Usol[:, 1:]).sum(0)
    return mean, var.clamp_min(0)
