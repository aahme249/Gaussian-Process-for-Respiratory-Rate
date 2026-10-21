"""
02 - CONJUGATE GRADIENT (CG):  solve K x = y using ONLY  K @ vector

PICTURE IN YOUR HEAD
--------------------
Solving K x = y is the same as finding the bottom of a bowl:

    f(x) = 1/2 x^T K x - y^T x          (its lowest point is exactly x = K^-1 y)

  * Plain gradient descent rolls downhill and zig-zags across the valley.
  * CG is smarter: each new step direction is "K-conjugate" to all earlier ones,
    so it never undoes progress it already made. In exact math it reaches the
    bottom of an n-dimensional bowl in at most n steps, and usually gets close
    much sooner.

The only thing CG ever does with K is multiply it by a vector.
It never looks inside K, never factorizes it. That is what makes BBMM possible.

Run:  .venv/bin/python learn/02_conjugate_gradient.py
      (saves learn/02_cg_path.png, a picture of the path on the bowl)
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)


def conjugate_gradient(matmul, y, max_iter, verbose=True):
    x = np.zeros_like(y)    # start guess: 0
    r = y - matmul(x)       # residual = how wrong we are = also the downhill direction
    d = r.copy()            # first search direction = straight downhill
    path = [x.copy()]

    for k in range(max_iter):
        Kd = matmul(d)                     # <-- the ONLY use of K
        alpha = (r @ r) / (d @ Kd)         # how far to walk along d (exact line search)
        x = x + alpha * d                  # take the step
        r_new = r - alpha * Kd             # update residual cheaply (no extra matmul)
        beta = (r_new @ r_new) / (r @ r)   # how much of the old direction to keep
        d = r_new + beta * d               # new direction: downhill, bent to be conjugate
        r = r_new
        path.append(x.copy())
        if verbose:
            print(f"iter {k + 1}: alpha={alpha:.4f}  beta={beta:.4f}  "
                  f"x={x}  |residual|={np.linalg.norm(r):.2e}")
        if np.linalg.norm(r) < 1e-10:
            break
    return x, np.array(path)


# ---------------------------------------------------------------------------
# Example 1: the same 3x3 matrix as the Cholesky demo
# ---------------------------------------------------------------------------
K = np.array([[4.0, 2.0, 2.0],
              [2.0, 5.0, 3.0],
              [2.0, 3.0, 6.0]])
y = np.array([1.0, 2.0, 3.0])

print("=== CG on 3x3: should finish in <= 3 iterations ===")
x, _ = conjugate_gradient(lambda v: K @ v, y, max_iter=10)
print("CG answer     :", x)
print("exact answer  :", np.linalg.solve(K, y))


# ---------------------------------------------------------------------------
# Example 2: the matrix is NEVER stored. We only have a function for K @ v.
# Here K = tridiagonal(-1, 3, -1), size 1000. Storing it would take 1,000,000
# numbers, while this function needs none.
# ---------------------------------------------------------------------------
print("\n=== CG with a 'black box' matrix (n=1000, never stored) ===")


def K_times(v):
    out = 3 * v
    out[1:] -= v[:-1]
    out[:-1] -= v[1:]
    return out


y_big = np.ones(1000)
x_big, path = conjugate_gradient(K_times, y_big, max_iter=1000, verbose=False)
print(f"converged in {len(path) - 1} iterations (n = 1000)")
print(f"|K x - y| = {np.linalg.norm(K_times(x_big) - y_big):.2e}")


# ---------------------------------------------------------------------------
# Example 3: SEE it. A 2D bowl: CG vs gradient descent.
# ---------------------------------------------------------------------------
K2 = np.array([[3.0, 2.0],
               [2.0, 6.0]])
y2 = np.array([2.0, -8.0])
_, cg_path = conjugate_gradient(lambda v: K2 @ v, y2, max_iter=2, verbose=False)

gd = [np.zeros(2)]
for _ in range(30):                       # gradient descent with exact line search
    x0 = gd[-1]
    r = y2 - K2 @ x0
    gd.append(x0 + (r @ r) / (r @ K2 @ r) * r)
gd = np.array(gd)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, b = np.meshgrid(np.linspace(-2, 5, 200), np.linspace(-4, 2, 200))
    pts = np.stack([a.ravel(), b.ravel()], 1)
    f = 0.5 * np.einsum("ni,ij,nj->n", pts, K2, pts) - pts @ y2
    plt.figure(figsize=(6, 5))
    plt.contour(a, b, f.reshape(a.shape), levels=25, cmap="Greys")
    plt.plot(gd[:, 0], gd[:, 1], "o-", color="tab:orange", label=f"gradient descent ({len(gd) - 1} steps)")
    plt.plot(cg_path[:, 0], cg_path[:, 1], "o-", color="tab:blue", lw=2, label="conjugate gradient (2 steps)")
    plt.plot(*np.linalg.solve(K2, y2), "r*", ms=15, label="answer K^-1 y")
    plt.legend()
    plt.title("Solving K x = y = finding the bottom of a bowl")
    plt.savefig("learn/02_cg_path.png", dpi=120, bbox_inches="tight")
    print("\nsaved picture: learn/02_cg_path.png")
except ImportError:
    print("\n(install matplotlib to get the picture)")
