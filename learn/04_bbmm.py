"""
04 - BBMM (Blackbox Matrix-Matrix inference): CG + Lanczos in ONE loop

THE GOAL
--------
GP training needs, for the kernel matrix K (n x n):
    (a) K^-1 y      -> the data-fit term  y^T K^-1 y
    (b) log|K|      -> the complexity term
File 01 got both from Cholesky: O(n^3), and needs K stored.

THE BBMM TRICK (picture it as one machine with many input slots)
-----------------------------------------------------------------
1. Stack the vectors you want to solve against into one matrix:
        B = [ y | z1 | z2 | ... | zt ]       z_i are random +-1 "probe" vectors
2. Run CG on ALL columns at the same time. Each iteration does a single
   K @ (n x (t+1) matrix): a matrix-matrix product, which is fast on a GPU.
   -> column 0 of the answer is K^-1 y                                   (a)
3. Bonus: the numbers alpha, beta that CG computes for column i are
   the same numbers Lanczos would produce starting from z_i. So we get
   a tridiagonal T_i for every probe for FREE, with no extra K multiplies.
4. Log-determinant (stochastic Lanczos quadrature):
        log|K| = trace(log K) ~= average over i of  z_i^T log(K) z_i
                               ~= average over i of |z_i|^2 * e1^T log(T_i) e1
   -> (b)

K is a black box: the algorithm only ever calls  K_matmul(M).

Run:  .venv/bin/python learn/04_bbmm.py
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# A small GP problem
# ---------------------------------------------------------------------------
n = 400
X = np.sort(rng.uniform(0, 10, n))
y = np.sin(X) + 0.1 * rng.standard_normal(n)
lengthscale, noise = 1.0, 0.1

K = np.exp(-0.5 * (X[:, None] - X[None, :]) ** 2 / lengthscale ** 2) + noise * np.eye(n)
# (Here we build K so we can check the answer. In real BBMM you could compute K @ M
#  in chunks or with structure, without ever holding all of K.)

matmul_calls = 0


def K_matmul(M):
    """The ONLY access the algorithm has to K."""
    global matmul_calls
    matmul_calls += 1
    return K @ M


# ---------------------------------------------------------------------------
# Step 1+2: batched CG on all columns at once  (mBCG, Algorithm 2 in the paper)
# Same lines as file 02, but x, r, d are now MATRICES (one column per system),
# and alpha, beta are VECTORS (one number per column).
# ---------------------------------------------------------------------------
def mbcg(K_matmul, B, max_iter, tol=1e-8):
    X = np.zeros_like(B)
    R = B - K_matmul(X)
    D = R.copy()
    alphas, betas = [], []

    for k in range(max_iter):
        KD = K_matmul(D)                                   # ONE matmul for all columns
        rr = np.sum(R * R, axis=0)                         # r^T r, per column
        alpha = rr / np.sum(D * KD, axis=0)                # per column
        X = X + alpha * D
        R = R - alpha * KD
        beta = np.sum(R * R, axis=0) / rr                  # per column
        D = R + beta * D
        alphas.append(alpha)
        betas.append(beta)
        if np.max(np.linalg.norm(R, axis=0) / np.linalg.norm(B, axis=0)) < tol:
            break
    return X, np.array(alphas), np.array(betas)


# ---------------------------------------------------------------------------
# Step 3: turn CG's alphas and betas into the Lanczos tridiagonal T (for one column)
#     T[j, j]   = 1/alpha_j + beta_{j-1}/alpha_{j-1}      (the second term is 0 for j = 0)
#     T[j, j+1] = sqrt(beta_j) / alpha_j
# ---------------------------------------------------------------------------
def cg_to_lanczos(alpha, beta):
    k = len(alpha)
    T = np.zeros((k, k))
    for j in range(k):
        T[j, j] = 1 / alpha[j] + (beta[j - 1] / alpha[j - 1] if j > 0 else 0)
        if j < k - 1:
            T[j, j + 1] = T[j + 1, j] = np.sqrt(beta[j]) / alpha[j]
    return T


# ---------------------------------------------------------------------------
# Run it
# ---------------------------------------------------------------------------
t = 30                                              # number of random probes
Z = rng.choice([-1.0, 1.0], size=(n, t))            # Rademacher probes: E[z z^T] = I
B = np.column_stack([y, Z])                         # [ y | z1 ... zt ]

U, alphas, betas = mbcg(K_matmul, B, max_iter=100)
print(f"mBCG: {len(alphas)} iterations, {matmul_calls} calls to K_matmul "
      f"(each multiplies K by a {n} x {t + 1} matrix)")

# (a) the solve
Kinv_y = U[:, 0]
# (b) the log-determinant, from each probe's free tridiagonal
estimates = []
for i in range(1, t + 1):
    T = cg_to_lanczos(alphas[:, i], betas[:, i])
    lam, V = np.linalg.eigh(T)
    estimates.append(n * np.sum(V[0, :] ** 2 * np.log(lam)))  # |z|^2 = n for +-1 probes
logdet_bbmm = np.mean(estimates)

mll_bbmm = -0.5 * y @ Kinv_y - 0.5 * logdet_bbmm - 0.5 * n * np.log(2 * np.pi)

# ---------------------------------------------------------------------------
# Compare with Cholesky (file 01)
# ---------------------------------------------------------------------------
L = np.linalg.cholesky(K)
Kinv_y_chol = np.linalg.solve(L.T, np.linalg.solve(L, y))
logdet_chol = 2 * np.sum(np.log(np.diag(L)))
mll_chol = -0.5 * y @ Kinv_y_chol - 0.5 * logdet_chol - 0.5 * n * np.log(2 * np.pi)

print("\n                      BBMM          Cholesky")
print(f"y^T K^-1 y      {y @ Kinv_y:12.4f}   {y @ Kinv_y_chol:12.4f}")
print(f"log|K|          {logdet_bbmm:12.4f}   {logdet_chol:12.4f}")
print(f"log p(y)        {mll_bbmm:12.4f}   {mll_chol:12.4f}")

print("\nlog|K| estimate as more probes are averaged (it's random, so it wiggles then settles):")
for m in [1, 3, 10, 30]:
    print(f"  {m:2d} probes: {np.mean(estimates[:m]):10.4f}   (truth {logdet_chol:.4f})")
print("  (the solve is exact; log|K| is a random estimate, error shrinks like 1/sqrt(probes).\n"
      "   That is fine for training: the gradient only needs to be right on average.)")

# Sanity check: the tridiagonal from CG really is the Lanczos T of file 03
T0 = cg_to_lanczos(alphas[:5, 1], betas[:5, 1])
q = Z[:, 0] / np.linalg.norm(Z[:, 0])
print(f"\nfirst diagonal entry of T from CG: {T0[0, 0]:.6f}"
      f"   from Lanczos (q^T K q): {q @ K @ q:.6f}")
