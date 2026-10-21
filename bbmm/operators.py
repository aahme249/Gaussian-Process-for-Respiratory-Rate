"""Step 5 -- Blackbox kernel operators (paper sec. 5, "Programmability").

BBMM only ever needs:
  * matmul(M)  -> K_hat M         (differentiable w.r.t. hyperparameters theta;
                                   autograd then gives dK_hat/dtheta M for free)
  * diag(), row(i) of K (no noise) -- only for the pivoted Cholesky preconditioner
  * noise sigma^2

Any model that implements these gets exact training/prediction with no new
inference code. Three examples from the paper:
  * ExactKernelOperator    K_hat = K_XX + s I                  O(n^2 t) matmul, O(n) memory
  * SoRKernelOperator      K_hat = K_XU K_UU^{-1} K_UX + s I   O(n m t + m^3)   (SGPR)
  * LinearKernelOperator   K_hat = X X^T + s I                 O(n d t)         (Bayesian lin. reg.)
"""
import math

import torch
from torch import nn


class RBFKernel(nn.Module):
    """k(x, x') = s * exp(-||x - x'||^2 / (2 l^2)), parameters kept in log-space."""

    def __init__(self, lengthscale=1.0, outputscale=1.0):
        super().__init__()
        self.log_lengthscale = nn.Parameter(torch.tensor(math.log(lengthscale)))
        self.log_outputscale = nn.Parameter(torch.tensor(math.log(outputscale)))

    @property
    def lengthscale(self):
        return self.log_lengthscale.exp()

    @property
    def outputscale(self):
        return self.log_outputscale.exp()

    def forward(self, X1, X2):
        sq = torch.cdist(X1 / self.lengthscale, X2 / self.lengthscale).pow(2)
        return self.outputscale * torch.exp(-0.5 * sq)

    def diag(self, X):
        return self.outputscale.expand(X.shape[0])


class KernelOperator:
    """Interface. Subclasses implement kernel_matmul / kernel_diag / kernel_row."""

    def __init__(self, noise):
        self.noise = noise  # tensor (may require grad)

    def kernel_matmul(self, M):
        raise NotImplementedError

    def kernel_diag(self):
        raise NotImplementedError

    def kernel_row(self, i):
        raise NotImplementedError

    def matmul(self, M):
        return self.kernel_matmul(M) + self.noise * M

    def dense(self):
        """Materialise K_hat -- only for testing / the Cholesky baseline."""
        n = self.n
        eye = torch.eye(n, dtype=self.dtype, device=self.device)
        return self.matmul(eye)


class ExactKernelOperator(KernelOperator):
    """Exact GP kernel. K is never stored: rows are generated in chunks."""

    def __init__(self, kernel, X, noise, chunk_size=1024):
        super().__init__(noise)
        self.kernel, self.X, self.chunk_size = kernel, X, chunk_size
        self.n, self.dtype, self.device = X.shape[0], X.dtype, X.device

    def kernel_matmul(self, M):
        out = [self.kernel(self.X[i:i + self.chunk_size], self.X) @ M
               for i in range(0, self.n, self.chunk_size)]
        return torch.cat(out, 0)

    def kernel_diag(self):
        return self.kernel.diag(self.X)

    def kernel_row(self, i):
        return self.kernel(self.X[i:i + 1], self.X)[0]


class SoRKernelOperator(KernelOperator):
    """Subset-of-regressors / SGPR kernel K_XU K_UU^{-1} K_UX with inducing points U."""

    def __init__(self, kernel, X, U, noise, jitter=1e-6):
        super().__init__(noise)
        self.n, self.dtype, self.device = X.shape[0], X.dtype, X.device
        self.K_XU = kernel(X, U)
        K_UU = kernel(U, U) + jitter * torch.eye(U.shape[0], dtype=X.dtype, device=X.device)
        self.chol_UU = torch.linalg.cholesky(K_UU)

    def kernel_matmul(self, M):
        # K_XU (K_UU^{-1} (K_UX M)) -- group the products, never form n x n
        return self.K_XU @ torch.cholesky_solve(self.K_XU.T @ M, self.chol_UU)

    def kernel_diag(self):
        W = torch.linalg.solve_triangular(self.chol_UU, self.K_XU.T, upper=False)  # m x n
        return (W ** 2).sum(0)

    def kernel_row(self, i):
        return self.K_XU @ torch.cholesky_solve(self.K_XU[i:i + 1].T, self.chol_UU)[:, 0]


class LinearKernelOperator(KernelOperator):
    """Bayesian linear regression as a GP: K = s X X^T."""

    def __init__(self, X, log_scale, noise):
        super().__init__(noise)
        self.X, self.log_scale = X, log_scale
        self.n, self.dtype, self.device = X.shape[0], X.dtype, X.device

    def kernel_matmul(self, M):
        return self.log_scale.exp() * (self.X @ (self.X.T @ M))

    def kernel_diag(self):
        return self.log_scale.exp() * (self.X ** 2).sum(1)

    def kernel_row(self, i):
        return self.log_scale.exp() * (self.X @ self.X[i])
