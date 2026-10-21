"""
01 - CHOLESKY DECOMPOSITION  (the "old way" BBMM replaces)

PICTURE IN YOUR HEAD
--------------------
A symmetric positive-definite matrix K is like a "squared" thing.
Cholesky finds its "square root":   K = L @ L.T
where L is LOWER-TRIANGULAR (zeros above the diagonal):

        [ a 0 0 ]
    L = [ b c 0 ]
        [ d e f ]

Why a GP cares:
  * to solve  K x = y  you solve two triangular systems (easy: top-to-bottom, then bottom-to-top)
  * log|K| = 2 * sum(log(diag(L)))      (determinant of a triangle = product of its diagonal)

The catch: building L costs about n^3 / 3 operations. With n = 100,000 data
points that is ~3 * 10^14 operations, and you have to store the whole n x n matrix.

Run:  .venv/bin/python learn/01_cholesky.py
"""
import numpy as np

np.set_printoptions(precision=3, suppress=True)

K = np.array([[4.0, 2.0, 2.0],
              [2.0, 5.0, 3.0],
              [2.0, 3.0, 6.0]])
y = np.array([1.0, 2.0, 3.0])
n = len(K)

print("K =\n", K)

# ---------------------------------------------------------------------------
# Build L one entry at a time. Each entry is chosen so (L @ L.T) matches K.
# ---------------------------------------------------------------------------
L = np.zeros_like(K)
for i in range(n):              # row
    for j in range(i + 1):      # column (only up to the diagonal)
        already = sum(L[i, k] * L[j, k] for k in range(j))   # what earlier columns already give us
        if i == j:
            L[i, j] = np.sqrt(K[i, i] - already)              # diagonal: square root of what's left
        else:
            L[i, j] = (K[i, j] - already) / L[j, j]           # below diagonal: divide by the diagonal
        print(f"L[{i},{j}] = {L[i, j]:.3f}")

print("\nL =\n", L)
print("L @ L.T =\n", L @ L.T, "\n(same as K)")
print("numpy agrees:", np.allclose(L, np.linalg.cholesky(K)))


# ---------------------------------------------------------------------------
# Use L to solve K x = y.  K x = L (L.T x) = y
#   step 1: solve L w = y    (forward substitution, top row first)
#   step 2: solve L.T x = w  (back substitution, bottom row first)
# ---------------------------------------------------------------------------
w = np.zeros(n)
for i in range(n):
    w[i] = (y[i] - L[i, :i] @ w[:i]) / L[i, i]

x = np.zeros(n)
for i in reversed(range(n)):
    x[i] = (w[i] - L[i + 1:, i] @ x[i + 1:]) / L[i, i]

print("\nsolve K x = y  ->  x =", x)
print("check K @ x    =", K @ x, " (should be y =", y, ")")

# ---------------------------------------------------------------------------
# Log-determinant from the diagonal of L
# ---------------------------------------------------------------------------
logdet = 2 * np.sum(np.log(np.diag(L)))
print(f"\nlog|K| from L = {logdet:.6f}")
print(f"log|K| numpy  = {np.linalg.slogdet(K)[1]:.6f}")

# ---------------------------------------------------------------------------
# The two numbers a GP needs for its log marginal likelihood:
#   log p(y) = -1/2 y^T K^-1 y  -  1/2 log|K|  -  n/2 log(2 pi)
#   y^T K^-1 y  ->  needs a SOLVE
#   log|K|      ->  needs a LOG-DETERMINANT
# The next three files show how to get both WITHOUT Cholesky.
# ---------------------------------------------------------------------------
mll = -0.5 * y @ x - 0.5 * logdet - 0.5 * n * np.log(2 * np.pi)
print(f"\nGP log marginal likelihood = {mll:.6f}")
