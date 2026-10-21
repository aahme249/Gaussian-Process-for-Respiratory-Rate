"""
03 - LANCZOS TRIDIAGONALIZATION:  squeeze a big matrix into a small tridiagonal one

PICTURE IN YOUR HEAD
--------------------
Start with one vector q1. Keep multiplying by K:   q1, K q1, K^2 q1, ...
These vectors span the "Krylov space", the part of the world K can reach from q1.
Lanczos makes them orthonormal (Q = [q1 q2 ... qk]) and records how K mixes
them. Because K is symmetric, each new vector only touches the previous two, so
the record is a TRIDIAGONAL matrix:

             [ a1 b1             ]
             [ b1 a2 b2          ]
    T = Q^T K Q = [    b2 a3 b3       ]      (k x k, with k much smaller than n)
             [       b3 a4 ...    ]

T is a tiny "summary" of K as seen from q1:
  * the eigenvalues of T approximate the eigenvalues of K (the extreme ones first)
  * so for any function f:   q1^T f(K) q1  ~=  e1^T f(T) e1
    With f = log, this gives the log-determinant (see file 04).

Again, we only ever compute K @ vector.

Run:  .venv/bin/python learn/03_lanczos.py
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)


def lanczos(matmul, q1, k, verbose=True):
    n = len(q1)
    Q = np.zeros((n, k))
    alphas, betas = [], []

    q = q1 / np.linalg.norm(q1)
    q_prev = np.zeros(n)
    beta = 0.0
    for j in range(k):
        Q[:, j] = q
        w = matmul(q)                      # <-- the ONLY use of K
        alpha = q @ w                      # how much of K q points back along q  -> diagonal
        w = w - alpha * q - beta * q_prev  # remove the parts along the last two q's
        w = w - Q[:, :j + 1] @ (Q[:, :j + 1].T @ w)   # (re-orthogonalize: fixes rounding error)
        alphas.append(alpha)
        beta = np.linalg.norm(w)           # what's left over                 -> off-diagonal
        if j < k - 1:
            betas.append(beta)
            q_prev, q = q, w / beta        # the leftover, normalized, is the next q
        if verbose:
            print(f"step {j + 1}: alpha={alpha:.4f}  beta={beta:.4f}")

    T = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
    return Q, T


# A random 6x6 symmetric positive-definite matrix with known eigenvalues
eigs = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 20.0])
U, _ = np.linalg.qr(rng.standard_normal((6, 6)))
K = U @ np.diag(eigs) @ U.T
z = rng.standard_normal(6)

print("=== Full Lanczos (k = n = 6) ===")
Q, T = lanczos(lambda v: K @ v, z, k=6)
print("\nT (tridiagonal) =\n", T)
print("\nQ^T K Q =\n", Q.T @ K @ Q, "\n(same as T)")
print("\neig(K) =", eigs)
print("eig(T) =", np.linalg.eigvalsh(T), "(same: T is K in a different basis)")

# ---------------------------------------------------------------------------
# The useful part: stop EARLY. With k < n, T is a compressed sketch of K.
# ---------------------------------------------------------------------------
print("\n=== Stop early: how eig(T) approaches eig(K) ===")
for k in range(1, 7):
    _, Tk = lanczos(lambda v: K @ v, z, k=k, verbose=False)
    print(f"k={k}: eig(T) = {np.linalg.eigvalsh(Tk)}")
print("(the biggest eigenvalue, 20, shows up first)")

# ---------------------------------------------------------------------------
# Quadrature:  z^T log(K) z  ~=  |z|^2 * e1^T log(T) e1
# log(T) = V diag(log(lam)) V^T  where T = V diag(lam) V^T  (T is tiny, so this is cheap)
# ---------------------------------------------------------------------------
print("\n=== z^T log(K) z  from the tiny T ===")
exact = z @ (U @ np.diag(np.log(eigs)) @ U.T) @ z
print(f"exact          : {exact:.6f}")
for k in [1, 2, 3, 4, 6]:
    _, Tk = lanczos(lambda v: K @ v, z, k=k, verbose=False)
    lam, V = np.linalg.eigh(Tk)
    approx = (z @ z) * np.sum(V[0, :] ** 2 * np.log(lam))
    print(f"Lanczos k={k}    : {approx:.6f}")
