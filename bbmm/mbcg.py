"""Step 2 -- Modified Batched Conjugate Gradients, mBCG (paper Algorithm 2).

Two changes to Algorithm 1:
  (1) solve t right-hand sides at once: every vector op becomes a matrix op,
      and the scalars alpha, beta become length-t vectors. A is touched only
      through ONE matrix-matrix multiply per iteration.
  (2) record alpha/beta to build the Lanczos tridiagonal matrix T_i for every
      right-hand side (Observation 3 / eq. S5, Saad sec. 6.7.3):

        T[0,0]     = 1/alpha_1
        T[j,j]     = 1/alpha_{j+1} + beta_j / alpha_j
        T[j-1,j]   = sqrt(beta_j) / alpha_j

With a preconditioner P, T_i is the Lanczos matrix of P^{-1/2} A P^{-1/2}
started from P^{-1/2} b_i (used by step 3 / eq. 8).

Note: the printed Algorithm 2 has sign typos (R0 = A U0 - B, D = Z - beta D);
the standard CG signs are used here.
"""
import torch


def mbcg(mmm_A, B, precond=lambda R: R, max_iter=1000, tol=1e-6, max_tridiag_iter=None):
    """Batched preconditioned CG.

    mmm_A:   callable M (n x t) -> A M
    B:       (n, t) right-hand sides
    precond: callable R (n x t) -> P^{-1} R
    tol:     relative residual tolerance per column
    max_tridiag_iter: number of Lanczos steps kept in T (defaults to max_iter)

    Returns
      U:     (n, t) approximate A^{-1} B
      T:     (t, p, p) Lanczos tridiagonal matrices, one per column of B.
             Columns that converge early are padded with an identity block,
             which leaves e1^T f(T) e1 unchanged (block-diagonal structure).
      info:  dict(num_iter=..., rz0=r0^T P^{-1} r0 per column)
    """
    n, t = B.shape
    p = max_iter if max_tridiag_iter is None else min(max_iter, max_tridiag_iter)
    dtype, device = B.dtype, B.device

    U = torch.zeros_like(B)
    R = B - mmm_A(U)                                   # current errors
    Z = precond(R)                                     # preconditioned errors
    D = Z.clone()                                      # search directions
    rz = (R * Z).sum(0)                                # (R o Z)^T 1
    rz0 = rz.clone()
    b_norm = B.norm(dim=0).clamp_min(torch.finfo(dtype).tiny)
    active = R.norm(dim=0) / b_norm >= tol

    diag = torch.ones(t, p, dtype=dtype, device=device)
    offdiag = torch.zeros(t, max(p - 1, 0), dtype=dtype, device=device)
    prev_alpha = torch.ones(t, dtype=dtype, device=device)
    prev_beta = torch.zeros(t, dtype=dtype, device=device)

    num_iter = 0
    for j in range(max_iter):
        if not active.any():
            break
        num_iter = j + 1
        V = mmm_A(D)                                   # the one MMM per iteration
        dv = (D * V).sum(0)                            # (D o V)^T 1
        alpha = torch.where(active, rz / torch.where(active, dv, 1.0), 0.0)

        U = U + alpha * D                              # U + D diag(alpha)
        R = R - alpha * V
        Z = precond(R)
        rz_new = (R * Z).sum(0)
        beta = torch.where(active, rz_new / torch.where(active, rz, 1.0), 0.0)

        # Lanczos coefficients (red lines of Algorithm 2)
        if j < p:
            safe_alpha = torch.where(active, alpha, 1.0)
            d_j = 1.0 / safe_alpha + prev_beta / prev_alpha   # prev_beta = 0 at j = 0
            diag[:, j] = torch.where(active, d_j, 1.0)
            if j > 0:
                off = torch.sqrt(prev_beta.clamp_min(0)) / prev_alpha
                offdiag[:, j - 1] = torch.where(active, off, 0.0)
        prev_alpha = torch.where(active, alpha, 1.0)
        prev_beta = beta

        D = Z + beta * D
        rz = rz_new
        active = active & (R.norm(dim=0) / b_norm >= tol)

    p_used = max(min(num_iter, p), 1)
    T = torch.diag_embed(diag[:, :p_used])
    if p_used > 1:
        T = T + torch.diag_embed(offdiag[:, : p_used - 1], 1) + torch.diag_embed(offdiag[:, : p_used - 1], -1)
    return U, T, {"num_iter": num_iter, "rz0": rz0}
