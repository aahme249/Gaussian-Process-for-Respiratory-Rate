# Learn BBMM in 4 small files

Plain numpy, tiny matrices, every step printed. Read the docstring at the top of each file, then run it.

```bash
.venv/bin/python learn/01_cholesky.py
.venv/bin/python learn/02_conjugate_gradient.py   # also saves learn/02_cg_path.png
.venv/bin/python learn/03_lanczos.py
.venv/bin/python learn/04_bbmm.py
```

| File | Idea in one line | Mental picture |
|---|---|---|
| `01_cholesky.py` | `K = L Lᵀ` gives both `K⁻¹y` and `log\|K\|`, but costs O(n³) | the "square root" of a matrix, a triangle |
| `02_conjugate_gradient.py` | solve `Kx = y` using only `K @ v` | rolling to the bottom of a bowl, never undoing earlier steps |
| `03_lanczos.py` | compress K into a small tridiagonal `T` that "acts like" K | taking a low-res photo of K from one viewpoint `z` |
| `04_bbmm.py` | run CG on `[y, z₁…z_t]` together: solves come out, and the CG numbers give a free `T` per probe → `log\|K\|` | one machine, many input slots, two outputs |

The key fact linking 02 and 03: the `alpha`/`beta` numbers CG computes are exactly
the entries of the Lanczos `T` (`cg_to_lanczos` in file 04). So one CG run gives both.

After these, the full PyTorch version (with preconditioning, gradients and predictions) is in `bbmm/`
and `step_by_step.py`.
