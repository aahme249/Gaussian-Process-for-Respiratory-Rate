"""Walk through BBMM (Gardner et al. 2018) one step at a time, checking each
step against dense / Cholesky ground truth.

    .venv/bin/python step_by_step.py
"""
import math
import time

import torch

from bbmm import (BBMMSettings, ExactKernelOperator, LinearKernelOperator, PivotedCholeskyPreconditioner,
                  RBFKernel, SoRKernelOperator, bbmm_marginal_log_likelihood, bbmm_predict, mbcg, pcg,
                  pivoted_cholesky, slq_logdet)

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


def header(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def make_data(n=500, d=1):
    X = torch.rand(n, d) * 6
    y = torch.sin(2 * X).sum(-1) + 0.3 * torch.cos(5 * X).sum(-1) + 0.1 * torch.randn(n)
    return X, y


def exact_mll(K_hat, y):
    """Cholesky baseline: the O(n^3) way BBMM replaces."""
    L = torch.linalg.cholesky(K_hat)
    alpha = torch.cholesky_solve(y.unsqueeze(-1), L)[:, 0]
    logdet = 2 * torch.log(torch.diagonal(L)).sum()
    return -0.5 * (y @ alpha) - 0.5 * logdet - 0.5 * len(y) * math.log(2 * math.pi)


X, y = make_data()
n = X.shape[0]
kernel = RBFKernel(lengthscale=0.5, outputscale=1.0)
noise = torch.tensor(0.01)
op = ExactKernelOperator(kernel, X, noise)
with torch.no_grad():
    K_hat = op.dense()

# ---------------------------------------------------------------------------
header("Step 1: standard (preconditioned) CG -- Algorithm 1")
with torch.no_grad():
    u_cg, iters = pcg(lambda v: op.matmul(v.unsqueeze(-1))[:, 0], y, tol=1e-10)
    u_exact = torch.linalg.solve(K_hat, y)
print(f"CG iterations: {iters},  ||u_cg - K^-1 y|| / ||K^-1 y|| = "
      f"{(u_cg - u_exact).norm() / u_exact.norm():.2e}")

# ---------------------------------------------------------------------------
header("Step 2: mBCG -- batched solves + Lanczos tridiagonals (Algorithm 2)")
t = 8
Z = torch.randn(n, t)
B = torch.cat([y.unsqueeze(-1), Z], 1)
with torch.no_grad():
    U, T, info = mbcg(op.matmul, B, max_iter=n, tol=1e-10)
    U_exact = torch.linalg.solve(K_hat, B)
print(f"solved {B.shape[1]} RHS in {info['num_iter']} iterations (one MMM each)")
print(f"max relative solve error: {((U - U_exact).norm(dim=0) / U_exact.norm(dim=0)).max():.2e}")

# Check T against the definition z^T log(K) z ~= ||z||^2 e1^T log(T) e1
evals, evecs = torch.linalg.eigh(K_hat)
logK = evecs @ torch.diag(evals.log()) @ evecs.T
z = Z[:, 0]
true_quad = z @ logK @ z
evT, vecT = torch.linalg.eigh(T[1])
lanczos_quad = (z @ z) * (vecT[0] ** 2 * evT.log()).sum()
print(f"z^T log(K) z = {true_quad:.6f}   vs   ||z||^2 e1^T log(T) e1 = {lanczos_quad:.6f}")

# ---------------------------------------------------------------------------
header("Step 3: log-determinant via stochastic Lanczos quadrature (eq. 5-6)")
true_logdet = torch.linalg.slogdet(K_hat)[1]
for t in (10, 50, 200):
    Z = torch.randn(n, t)
    with torch.no_grad():
        _, T, info = mbcg(op.matmul, Z, max_iter=100, tol=1e-8)
    est = slq_logdet(T, info["rz0"])
    print(f"t={t:4d} probes: SLQ logdet = {est:10.3f}   exact = {true_logdet:10.3f}   "
          f"rel err = {abs(est - true_logdet) / abs(true_logdet):.2e}")

# ---------------------------------------------------------------------------
header("Step 4: pivoted Cholesky preconditioner (sec. 4.1, App. C)")
with torch.no_grad():
    K = K_hat - noise * torch.eye(n)
    print(" rank k | ||K - LL^T||_F / ||K||_F | CG iters (tol 1e-8) | log|P| err vs Woodbury check")
    for k in (0, 2, 5, 10, 20):
        L = pivoted_cholesky(op.kernel_diag, op.kernel_row, n, k) if k else torch.zeros(n, 0)
        P = PivotedCholeskyPreconditioner(L, float(noise))
        _, iters = pcg(lambda v: op.matmul(v.unsqueeze(-1))[:, 0], y,
                       precond=lambda r: P.solve(r.unsqueeze(-1))[:, 0], tol=1e-8)
        P_dense = L @ L.T + noise * torch.eye(n)
        ld_err = abs(P.logdet() - torch.linalg.slogdet(P_dense)[1])
        print(f" {k:6d} | {(K - L @ L.T).norm() / K.norm():24.2e} | {iters:19d} | {ld_err:.1e}")

    # Sampling check: Cov of z ~ N(0, P) should be P
    L = pivoted_cholesky(op.kernel_diag, op.kernel_row, n, 5)
    P = PivotedCholeskyPreconditioner(L, float(noise))
    S = P.sample(20000)
    P_dense = L @ L.T + noise * torch.eye(n)
    print(f"sample covariance rel err (20k samples): "
          f"{(S @ S.T / 20000 - P_dense).norm() / P_dense.norm():.2e}")

# ---------------------------------------------------------------------------
header("Step 5+6: BBMM marginal log-likelihood and gradients vs. Cholesky")
settings = BBMMSettings(num_probes=50, precond_rank=10, max_iter=200, max_tridiag_iter=50, tol=1e-6)


def grads(mll):
    kernel.zero_grad()
    if noise_param.grad is not None:
        noise_param.grad = None
    mll.backward()
    return torch.stack([kernel.log_lengthscale.grad, kernel.log_outputscale.grad, noise_param.grad])


noise_param = torch.tensor(math.log(0.01), requires_grad=True)
mll_b, info = bbmm_marginal_log_likelihood(ExactKernelOperator(kernel, X, noise_param.exp()), y,
                                           settings, return_info=True)
g_b = grads(mll_b)
mll_c = exact_mll(ExactKernelOperator(kernel, X, noise_param.exp()).dense(), y)
g_c = grads(mll_c)
print(f"MLL   BBMM = {mll_b.item():.4f}   Cholesky = {mll_c.item():.4f}   (mBCG iters: {info['num_iter']})")
print(f"grad  BBMM = {g_b.numpy().round(4)}")
print(f"grad  Chol = {g_c.numpy().round(4)}")
print("      (order: log_lengthscale, log_outputscale, log_noise)")

# The SAME inference code on different blackbox operators (sec. 5):
with torch.no_grad():
    Uind = torch.linspace(0, 6, 30).unsqueeze(-1)
    sor = SoRKernelOperator(kernel, X, Uind, noise)
    lin = LinearKernelOperator(torch.cat([X, X ** 2], 1), torch.tensor(0.0), torch.tensor(0.5))
    for name, o in (("SGPR / SoR", sor), ("Bayesian linear regression", lin)):
        m_b = bbmm_marginal_log_likelihood(o, y, settings)
        m_c = exact_mll(o.dense(), y)
        print(f"{name:28s} MLL BBMM = {m_b.item():10.4f}   Cholesky = {m_c.item():10.4f}")

# ---------------------------------------------------------------------------
header("Training an exact GP with BBMM (Adam on -MLL), then predicting (eq. 1)")
X, y = make_data(n=2000)
Xtest = torch.linspace(0, 6, 200).unsqueeze(-1)
ytest = torch.sin(2 * Xtest).sum(-1) + 0.3 * torch.cos(5 * Xtest).sum(-1)

kernel = RBFKernel(lengthscale=1.0, outputscale=1.0)
log_noise = torch.nn.Parameter(torch.tensor(math.log(0.1)))
opt = torch.optim.Adam(list(kernel.parameters()) + [log_noise], lr=0.1)
train_settings = BBMMSettings(num_probes=10, precond_rank=10, max_iter=50, max_tridiag_iter=20, tol=1e-3)

start = time.time()
for it in range(60):
    opt.zero_grad()
    op = ExactKernelOperator(kernel, X, log_noise.exp())
    loss = -bbmm_marginal_log_likelihood(op, y, train_settings) / len(y)
    loss.backward()
    opt.step()
    if it % 10 == 0 or it == 59:
        print(f"iter {it:3d}  loss {loss.item():.4f}  lengthscale {kernel.lengthscale.item():.3f}  "
              f"outputscale {kernel.outputscale.item():.3f}  noise {log_noise.exp().item():.4f}")
print(f"training time: {time.time() - start:.1f}s")

with torch.no_grad():
    op = ExactKernelOperator(kernel, X, log_noise.exp())
    mean, var = bbmm_predict(op, kernel(X, Xtest), kernel.diag(Xtest), y,
                             BBMMSettings(precond_rank=10, max_iter=500, tol=1e-8))
    K_hat = op.dense()
    Ks = kernel(X, Xtest)
    mean_c = Ks.T @ torch.linalg.solve(K_hat, y)
    var_c = kernel.diag(Xtest) - (Ks * torch.linalg.solve(K_hat, Ks)).sum(0)
print(f"test RMSE (BBMM)              : {(mean - ytest).pow(2).mean().sqrt():.4f}")
print(f"max |mean_BBMM - mean_Chol|   : {(mean - mean_c).abs().max():.2e}")
print(f"max |var_BBMM  - var_Chol|    : {(var - var_c.clamp_min(0)).abs().max():.2e}")
