"""Step 1 -- Standard preconditioned conjugate gradients (paper Algorithm 1).

CG solves A u = b while touching A *only* through a matrix-vector product.
This is the building block that mBCG (step 2) batches and extends.
"""
import torch


def identity_precond(r):
    return r


def pcg(mvm_A, b, precond=identity_precond, max_iter=1000, tol=1e-10):
    """Solve A u = b. Returns (u, num_iterations).

    mvm_A:   callable v -> A v
    b:       (n,) right-hand side
    precond: callable r -> P^{-1} r
    """
    u = torch.zeros_like(b)
    r = b - mvm_A(u)                  # residual
    z = precond(r)                    # preconditioned residual
    d = z.clone()                     # search direction
    rz = r @ z
    b_norm = b.norm()

    for j in range(1, max_iter + 1):
        v = mvm_A(d)
        alpha = rz / (d @ v)
        u = u + alpha * d
        r = r - alpha * v
        if r.norm() / b_norm < tol:
            return u, j
        z = precond(r)
        rz_new = r @ z
        beta = rz_new / rz
        d = z + beta * d
        rz = rz_new
    return u, max_iter
