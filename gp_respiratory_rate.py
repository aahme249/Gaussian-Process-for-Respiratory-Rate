"""
Respiratory rate from a single-lead ECG with Gaussian processes, after
Pimentel, Clifton, Clifton & Tarassenko, "Probabilistic Estimation of Respiratory Rate
using Gaussian Processes", IEEE EMBC 2013 — reproduced on BIDMC with GPyTorch.

Per 60 s window (new window every 10 s, as in the paper):
  1. detect R peaks in the ECG
  2. build two beat-by-beat series that breathing modulates:
       RPA = R-peak amplitude,  RSA = R-R interval
  3. z-normalise each and fit a GP with a periodic kernel
       k(r) = s^2 exp(-2 sin^2(pi r / P) / l^2)      (GPyTorch's PeriodicKernel, l = 1 fixed)
  4. the period P is the breath duration -> RR = 60 / P.
     Paper: MAP + grid approximation of p(P | data).  Here: a grid of P values
     (6-40 breaths/min, the paper's prior range), amplitude, noise and mean fitted
     for every P at once (batched GP), and p(P | data) ∝ marginal likelihood.
  5. keep whichever of RPA / RSA gives the lower posterior variance.

Baseline: spectral peak of each 4 Hz-resampled series (stronger peak wins).
Reference: manual breath annotations (annotator 1) inside the window.

Run:  .venv/bin/python gp_respiratory_rate.py [patient ...]
Data: data/bidmc_NN_Signals.csv and data/bidmc_NN_Breaths.csv from
      https://physionet.org/content/bidmc/1.0.0/bidmc_csv/
"""
import sys
import numpy as np
import scipy.signal as ss
import torch
import gpytorch
import matplotlib.pyplot as plt

from respiratory_rate import style, INK, MUTED, BLUE, BLUE_BAND, ORANGE

WIN, STEP = 60.0, 10.0
RR_GRID = np.arange(6.0, 40.01, 0.25)            # breaths/min
P_GRID = torch.tensor(60.0 / RR_GRID, dtype=torch.float64)
GREEN = "#1baf7a"


# ---------------------------------------------------------------- data
def load(patient):
    sig = np.loadtxt(f"data/bidmc_{patient:02d}_Signals.csv", delimiter=",", skiprows=1)
    fs = (len(sig) - 1) / (sig[-1, 0] - sig[0, 0])  # time column is rounded after 100 s
    br = np.loadtxt(f"data/bidmc_{patient:02d}_Breaths.csv", delimiter=",", skiprows=1)
    return dict(ecg=sig[:, 5], imp=sig[:, 1], fs=fs, breaths=br[:, 0] / fs)


def r_peaks(ecg, fs):
    x = ss.filtfilt(*ss.butter(3, [0.5 / (fs / 2), 30 / (fs / 2)], btype="band"), ecg)
    q = ss.filtfilt(*ss.butter(3, [5 / (fs / 2), 20 / (fs / 2)], btype="band"), ecg) ** 2
    pk, _ = ss.find_peaks(q, distance=int(0.33 * fs), height=np.percentile(q, 90) * 0.3)
    sgn = np.sign(np.median(x[pk]))
    pk = np.array([max(p - 5, 0) + np.argmax(sgn * x[max(p - 5, 0):p + 6]) for p in pk])
    return pk, sgn * x


def beat_series(pk, x, fs, t0):
    """RPA and RSA series inside [t0, t0+WIN): (times, z-scored values) each."""
    t = pk / fs
    m = (t >= t0) & (t < t0 + WIN)
    tr, amp = t[m], x[pk[m]]
    rr = np.diff(t)[m[1:]]
    trr = t[1:][m[1:]]
    ok = (rr > 0.3) & (rr < 2.0)                      # drop missed / extra beats
    z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    return {"RPA": (tr, z(amp)), "RSA": (trr[ok], z(rr[ok]))}


def reference_rr(breaths, t0):
    b = breaths[(breaths >= t0) & (breaths < t0 + WIN)]
    return 60 / np.mean(np.diff(b)) if len(b) >= 3 else np.nan


# ---------------------------------------------------------------- GP over a grid of periods
class PeriodicGP(gpytorch.models.ExactGP):
    def __init__(self, x, y, lik):
        super().__init__(x, y, lik)
        G = torch.Size([len(P_GRID)])
        self.mean = gpytorch.means.ConstantMean(batch_shape=G)
        k = gpytorch.kernels.PeriodicKernel(batch_shape=G)
        k.period_length = P_GRID.view(-1, 1, 1)
        k.raw_period_length.requires_grad_(False)      # P is fixed per batch member
        k.lengthscale = 1.0                             # paper's value l = 1, held fixed: a free
        k.raw_lengthscale.requires_grad_(False)         # l lets short periods fit noise
        self.cov = gpytorch.kernels.ScaleKernel(k, batch_shape=G)
        self.cov.outputscale = 1.0                      # sigma0 = 1

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean(x), self.cov(x))


def fit_series(t, y, iters=80):
    """Returns the posterior over RR on the grid, its mean/sd, and the fitted model."""
    G = len(P_GRID)
    x = torch.tensor(t - t[0], dtype=torch.float64).view(1, -1, 1).expand(G, -1, 1)
    yy = torch.tensor(y, dtype=torch.float64).expand(G, -1)
    lik = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([G]))
    lik.noise = 0.1
    model = PeriodicGP(x, yy, lik).double()
    lik.double()
    model.train(); lik.train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.05)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    for _ in range(iters):
        opt.zero_grad()
        loss = -mll(model(x), yy).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        logml = (mll(model(x), yy) * len(t)).numpy()   # log p(y | P, fitted rest)
    w = np.exp(logml - logml.max())
    w /= w.sum()
    mu = np.sum(w * RR_GRID)
    sd = np.sqrt(np.sum(w * (RR_GRID - mu) ** 2))
    model.eval(); lik.eval()
    return dict(post=w, mu=mu, sd=sd, model=model, lik=lik, t0=t[0])


def spectral_rr(t, y):
    g = np.arange(t[0], t[-1], 0.25)
    v = ss.detrend(np.interp(g, t, y)) * np.hanning(len(g))
    f = np.fft.rfftfreq(4096, 0.25)
    p = np.abs(np.fft.rfft(v, 4096)) ** 2
    m = (f >= 0.1) & (f <= 40 / 60)
    k = np.argmax(p[m])
    return 60 * f[m][k], p[m][k] / p[m].sum()


# ---------------------------------------------------------------- run one patient
def run(patient):
    d = load(patient)
    pk, x = r_peaks(d["ecg"], d["fs"])
    dur = len(d["ecg"]) / d["fs"]
    rows = []
    for t0 in np.arange(0, dur - WIN + 1e-9, STEP):
        ref = reference_rr(d["breaths"], t0)
        series = beat_series(pk, x, d["fs"], t0)
        fits = {k: fit_series(*v) for k, v in series.items()}
        best = min(fits, key=lambda k: fits[k]["sd"])
        spec = {k: spectral_rr(*v) for k, v in series.items()}
        sbest = max(spec, key=lambda k: spec[k][1])
        rows.append(dict(t0=t0, ref=ref, series=series, fits=fits, best=best,
                         gp=fits[best]["mu"], gp_sd=fits[best]["sd"], spec=spec[sbest][0]))
        print(f"  window {t0:5.0f}-{t0 + WIN:3.0f} s  ref {ref:5.1f}  GP {fits[best]['mu']:5.1f} "
              f"± {fits[best]['sd']:4.1f} ({best})  spectral {spec[sbest][0]:5.1f}")
    return d, pk, x, rows


# ---------------------------------------------------------------- figures
def gp_curve(fit, tt):
    """Posterior mean/band of the GP at its most likely period, on times tt."""
    i = int(np.argmax(fit["post"]))
    G = len(P_GRID)
    xs = torch.tensor(tt - fit["t0"], dtype=torch.float64).view(1, -1, 1).expand(G, -1, 1)
    with torch.no_grad():
        pred = fit["lik"](fit["model"](xs))
        lo, hi = pred.confidence_region()
    return pred.mean[i].numpy(), lo[i].numpy(), hi[i].numpy(), RR_GRID[i]


def figure_window(patient, d, pk, x, row, path):
    """Paper Fig. 1 for one window, plus the posterior over RR from each series."""
    fs, t0 = d["fs"], row["t0"]
    a, b = int(t0 * fs), int((t0 + WIN) * fs)
    t = np.arange(a, b) / fs
    br = d["breaths"][(d["breaths"] >= t0) & (d["breaths"] < t0 + WIN)]
    fig, axes = plt.subplots(5, 1, figsize=(12, 13), gridspec_kw=dict(height_ratios=[1, 1, 1.2, 1.2, 1.4]))

    ax = axes[0]; style(ax)
    ax.plot(t, x[a:b], color=INK, lw=0.7)
    m = (pk >= a) & (pk < b)
    ax.plot(pk[m] / fs, x[pk[m]], "o", ms=3, color=ORANGE, label="detected R peaks")
    ax.set_title("(a) ECG, lead II", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper right"); ax.set_yticks([])

    ax = axes[1]; style(ax)
    ax.plot(t, d["imp"][a:b], color=INK, lw=1)
    for v in br:
        ax.axvline(v, color=ORANGE, lw=0.8, alpha=0.7)
    ax.set_title(f"(b) Impedance respiration (reference only). Orange = annotated breaths "
                 f"→ {row['ref']:.1f} breaths/min", loc="left", fontsize=10, color=INK)
    ax.set_yticks([])

    for ax, key, name in [(axes[2], "RSA", "R-R interval (RSA)"), (axes[3], "RPA", "R-peak amplitude (RPA)")]:
        style(ax)
        ts, ys = row["series"][key]
        tt = np.linspace(ts[0], ts[-1], 600)
        mu, lo, hi, rr = gp_curve(row["fits"][key], tt)
        f = row["fits"][key]
        ax.fill_between(tt, lo, hi, color=BLUE_BAND, lw=0, label="GP 95% band")
        ax.plot(tt, mu, color=BLUE, lw=1.6, label=f"GP fit, most likely period {60 / rr:.2f} s")
        ax.plot(ts, ys, "o", ms=3, color=INK, label="one point per heartbeat")
        ax.set_title(f"({'c' if key == 'RSA' else 'd'}) {name}, z-scored  →  GP: "
                     f"{f['mu']:.1f} ± {f['sd']:.1f} breaths/min", loc="left", fontsize=10, color=INK)
        ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=3)
    axes[3].set_xlabel("time (s)", color=MUTED)

    ax = axes[4]; style(ax)
    for key, c in [("RSA", BLUE), ("RPA", GREEN)]:
        f = row["fits"][key]
        ax.plot(RR_GRID, f["post"], color=c, lw=2,
                label=f"{key}: {f['mu']:.1f} ± {f['sd']:.1f}" + ("  ← chosen (lower variance)" if key == row["best"] else ""))
    ax.axvline(row["ref"], color=INK, lw=1.5, ls="--", label=f"reference {row['ref']:.1f}")
    ax.set_xlabel("respiratory rate (breaths/min)", color=MUTED)
    ax.set_ylabel("posterior probability", color=MUTED)
    ax.set_title("(e) Posterior over respiratory rate from each series: a narrow peak = a confident estimate",
                 loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"Patient {patient:02d}, window {t0:.0f}-{t0 + WIN:.0f} s", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_track(patient, rows, path):
    tc = np.array([r["t0"] + WIN / 2 for r in rows])
    ref = np.array([r["ref"] for r in rows])
    gp = np.array([r["gp"] for r in rows])
    sd = np.array([r["gp_sd"] for r in rows])
    sp = np.array([r["spec"] for r in rows])
    fig, ax = plt.subplots(figsize=(12, 4.6))
    style(ax)
    ax.plot(tc, ref, color=INK, lw=2, label="reference (annotated breaths)")
    ax.errorbar(tc, gp, yerr=2 * sd, fmt="o", ms=5, color=BLUE, ecolor=BLUE, elinewidth=1, capsize=3,
                label="GP mean ± 2 SD")
    ax.plot(tc, sp, "s", ms=5, color=ORANGE, label="spectral baseline")
    ok = np.isfinite(ref)
    ax.set_title(f"Patient {patient:02d}: respiratory rate per 60 s window. MAE: GP "
                 f"{np.mean(np.abs(gp[ok] - ref[ok])):.2f}, spectral {np.mean(np.abs(sp[ok] - ref[ok])):.2f} breaths/min",
                 loc="left", fontsize=11, color=INK)
    ax.set_xlabel("window centre (s)", color=MUTED)
    ax.set_ylabel("breaths/min", color=MUTED)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    torch.manual_seed(0)
    torch.set_num_threads(1)   # tiny matrices: threading costs more than it saves
    patients = [int(a) for a in sys.argv[1:]] or [1]
    all_rows = []
    for p in patients:
        print(f"patient {p:02d}")
        d, pk, x, rows = run(p)
        all_rows += rows
        figure_track(p, rows, f"gp_rr_track_{p:02d}.png")
        mid = rows[len(rows) // 2]
        figure_window(p, d, pk, x, mid, f"gp_rr_window_{p:02d}.png")
    ref = np.array([r["ref"] for r in all_rows])
    ok = np.isfinite(ref)
    gp = np.array([r["gp"] for r in all_rows])[ok]
    sd = np.array([r["gp_sd"] for r in all_rows])[ok]
    sp = np.array([r["spec"] for r in all_rows])[ok]
    ref = ref[ok]
    print(f"\n{ok.sum()} windows   MAE  GP {np.mean(np.abs(gp - ref)):.2f}   "
          f"spectral {np.mean(np.abs(sp - ref)):.2f} breaths/min")
    print(f"reference inside GP mean ± 2 SD: {100 * np.mean(np.abs(gp - ref) <= 2 * sd):.0f}% of windows")
    for lo, hi in [(0, 2), (2, 5), (5, np.inf)]:
        m = (sd >= lo) & (sd < hi)
        if m.any():
            print(f"  GP SD {lo}-{hi} breaths/min: {m.sum():2d} windows, MAE {np.mean(np.abs(gp[m] - ref[m])):.2f}")
    print(f"chosen series: RSA {sum(r['best'] == 'RSA' for r in all_rows)}, "
          f"RPA {sum(r['best'] == 'RPA' for r in all_rows)}")


if __name__ == "__main__":
    main()
