"""
ECG waveform estimation with a GPyTorch time-series GP (BIDMC dataset).

Take 1004 consecutive ECG samples (125 Hz, ~8 s) from one patient. The GP sees the
first 80% and must predict the last 20% (about 1.6 s, 2 heartbeats) it never saw.

Kernel = (Periodic x RBF) + RBF
  - Periodic: the heartbeat repeats every ~1/HR seconds, so the shape learned on old
    beats is copied forward to new beats.
  - x RBF (long lengthscale): beats are only *locally* periodic: shape and timing
    drift slowly, so far-away beats count less.
  - + RBF: slow baseline wander (breathing, electrode movement).

Run:  .venv/bin/python ecg_waveform_gp.py [patient_number]
Data: data/bidmc_NN_Signals.csv from https://physionet.org/content/bidmc/1.0.0/bidmc_csv/
"""
import sys
import numpy as np
import scipy.signal as ss
import torch
import gpytorch
import matplotlib.pyplot as plt

from respiratory_rate import style, INK, MUTED, BLUE, BLUE_BAND, ORANGE

N = 1004
TRAIN_FRAC = 0.8
START_S = 60.0      # skip the first minute of the recording
SEED = 0


class ECGGP(gpytorch.models.ExactGP):
    def __init__(self, x, y, lik, period):
        super().__init__(x, y, lik)
        self.mean = gpytorch.means.ConstantMean()
        periodic = gpytorch.kernels.PeriodicKernel()
        periodic.period_length = period
        periodic.lengthscale = 0.3
        # keep "nearby beats look alike" over at least ~1.5 beats, and keep the
        # baseline slow, otherwise the GP explains each beat on its own and can't extrapolate
        decay = gpytorch.kernels.RBFKernel(lengthscale_constraint=gpytorch.constraints.GreaterThan(1.0))
        decay.lengthscale = 3.0
        trend = gpytorch.kernels.RBFKernel(lengthscale_constraint=gpytorch.constraints.GreaterThan(1.0))
        trend.lengthscale = 2.0
        self.beats = gpytorch.kernels.ScaleKernel(periodic * decay)
        self.trend = gpytorch.kernels.ScaleKernel(trend)
        self.trend.outputscale = 0.1

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean(x), self.beats(x) + self.trend(x))


def heartbeat_period(ecg, fs):
    """Median R-R interval (s), used to start the periodic kernel near the right period."""
    q = ss.filtfilt(*ss.butter(3, [5 / (fs / 2), 20 / (fs / 2)], btype="band"), ecg) ** 2
    peaks, _ = ss.find_peaks(q, distance=int(0.33 * fs), height=np.percentile(q, 90) * 0.3)
    return float(np.median(np.diff(peaks)) / fs)


def main():
    torch.manual_seed(SEED)
    patient = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    # bidmc_NN_Signals.csv columns: Time [s], RESP, PLETH, V, AVR, II  (125 Hz, normalised units)
    sig = np.loadtxt(f"data/bidmc_{patient:02d}_Signals.csv", delimiter=",", skiprows=1)
    fs = (len(sig) - 1) / (sig[-1, 0] - sig[0, 0])  # time column is rounded after 100 s
    a = int(START_S * fs)
    ecg = sig[a:a + N, 5]  # lead II
    t = np.arange(N) / fs
    n_tr = int(TRAIN_FRAC * N)

    period0 = heartbeat_period(ecg[:n_tr], fs)
    print(f"patient {patient:02d}: {N} ECG samples ({N / fs:.2f} s at {fs:.0f} Hz), "
          f"train {n_tr} / test {N - n_tr}; starting period {period0:.3f} s ({60 / period0:.0f} bpm)")

    y_mu, y_sd = ecg[:n_tr].mean(), ecg[:n_tr].std()
    x = torch.tensor(t, dtype=torch.float64)
    y = torch.tensor((ecg - y_mu) / y_sd, dtype=torch.float64)
    xtr, ytr, xte = x[:n_tr], y[:n_tr], x[n_tr:]

    lik = gpytorch.likelihoods.GaussianLikelihood()
    lik.noise = 1e-2
    model = ECGGP(xtr, ytr, lik, period0)
    model.train(); lik.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    for i in range(300):
        opt.zero_grad()
        loss = -mll(model(xtr), ytr)
        loss.backward()
        opt.step()
        if (i + 1) % 50 == 0:
            p = model.beats.base_kernel.kernels[0].period_length.item()
            print(f"  iter {i + 1:3d}  -mll/n {loss.item():7.3f}  period {p:.3f} s  "
                  f"noise sd {lik.noise.sqrt().item() * y_sd:.4f}  a.u.")
    model.eval(); lik.eval()

    with torch.no_grad():
        f_all = model(x)
        mu = f_all.mean.numpy() * y_sd + y_mu
        lo, hi = [c.numpy() * y_sd + y_mu for c in lik(f_all).confidence_region()]
        draws = f_all.sample(torch.Size([3])).numpy() * y_sd + y_mu

    te = slice(n_tr, N)
    err = mu[te] - ecg[te]
    rmse = np.sqrt(np.mean(err ** 2))
    corr = np.corrcoef(mu[te], ecg[te])[0, 1]
    cover = np.mean((ecg[te] >= lo[te]) & (ecg[te] <= hi[te]))
    p = model.beats.base_kernel.kernels[0].period_length.item()
    print(f"\ntest (unseen last {(N - n_tr) / fs:.2f} s):  RMSE {rmse:.4f}  a.u.  "
          f"(signal sd {ecg[te].std():.4f})  correlation {corr:.3f}  95% band covers {100 * cover:.0f}%")
    print(f"learned period {p:.3f} s = {60 / p:.1f} beats/min")

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw=dict(height_ratios=[1.3, 1]))
    ax = axes[0]
    style(ax)
    ax.axvspan(t[n_tr], t[-1], color="#f3f2ee", lw=0)
    ax.fill_between(t, lo, hi, color=BLUE_BAND, lw=0, label="GP 95% band")
    ax.plot(t[:n_tr], ecg[:n_tr], color=INK, lw=1.2, label="ECG the GP trained on")
    ax.plot(t[te], ecg[te], color=ORANGE, lw=1.6, label="true ECG (hidden from GP)")
    ax.plot(t, mu, color=BLUE, lw=1.4, label="GP estimate")
    ax.axvline(t[n_tr], color=MUTED, lw=1, ls="--")
    ax.text(t[n_tr], ax.get_ylim()[1], "  test: GP predicts →", va="top", fontsize=9, color=INK)
    ax.set_ylabel("ECG (a.u.)", color=MUTED)
    ax.set_title(f"Patient {patient:02d}: {N} ECG samples. Test RMSE {rmse:.3f}  a.u., "
                 f"correlation {corr:.2f}, band covers {100 * cover:.0f}% of true samples",
                 loc="left", fontsize=11, color=INK)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="lower left")

    ax = axes[1]
    style(ax)
    z = slice(n_tr - int(1.5 * fs), N)
    ax.fill_between(t[z], lo[z], hi[z], color=BLUE_BAND, lw=0)
    for k, d in enumerate(draws):
        ax.plot(t[z], d[z], color=BLUE, lw=0.8, alpha=0.5, label="GP posterior samples" if k == 0 else None)
    ax.plot(t[z], ecg[z], color=ORANGE, lw=1.6, label="true ECG")
    ax.axvline(t[n_tr], color=MUTED, lw=1, ls="--")
    ax.set_xlabel("time (s)", color=MUTED)
    ax.set_ylabel("ECG (a.u.)", color=MUTED)
    ax.set_title("Zoom on the test part: 3 waveforms sampled from the GP posterior",
                 loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig("ecg_waveform_test.png", dpi=130)
    print("saved ecg_waveform_test.png")

    explain(model, lik, xtr, ytr, x, t, n_tr, y_sd, y_mu, p)


def explain(model, lik, xtr, ytr, x, t, n_tr, y_sd, y_mu, period):
    """How the GP thinks: what the kernel says, what it believes before data, and its two parts."""
    model.double(); lik.double()
    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    with torch.no_grad():
        # A: correlation between two points as a function of how far apart they are
        lag = torch.linspace(0, 4, 800, dtype=torch.float64)
        zero = torch.zeros(1, dtype=torch.float64)
        kb = model.beats(lag, zero).to_dense().ravel().numpy()
        kt = model.trend(lag, zero).to_dense().ravel().numpy()
        ax = axes[0]
        style(ax)
        ax.plot(lag, kb / kb[0], color=BLUE, lw=2, label="heartbeat part: periodic × slow decay")
        ax.plot(lag, kt / kt[0], color=ORANGE, lw=2, label="baseline part: slow RBF")
        for k in range(1, int(4 / period) + 1):
            ax.axvline(k * period, color=MUTED, lw=0.8, ls=":")
        ax.set_xlabel("time apart, τ (s)   (dotted lines = 1, 2, 3 ... heartbeat periods)", color=MUTED)
        ax.set_ylabel("correlation", color=MUTED)
        ax.set_title("A. The kernel: how much does the ECG at time t tell the GP about time t+τ?  "
                     "High again every heartbeat, fading over a few beats.",
                     loc="left", fontsize=10, color=INK)
        ax.legend(fontsize=8, frameon=False)

        # B: functions the GP considers plausible before seeing any data
        ax = axes[1]
        style(ax)
        xs = x[: int(4 * 125)]
        prior = gpytorch.distributions.MultivariateNormal(
            torch.zeros(len(xs), dtype=torch.float64),
            (model.beats(xs) + model.trend(xs)).add_jitter(1e-6))
        for k, d in enumerate(prior.sample(torch.Size([3])).numpy()):
            ax.plot(xs.numpy(), d * y_sd, lw=1.2, color=[BLUE, ORANGE, "#1baf7a"][k], label=f"prior sample {k + 1}")
        ax.set_ylabel("ECG (a.u.)", color=MUTED)
        ax.set_title("B. Before data: random curves from the learned kernel. They already repeat every "
                     f"{period:.2f} s; the data then pins down *which* repeating shape.",
                     loc="left", fontsize=10, color=INK)
        ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper right")

        # C: posterior mean = heartbeat part + baseline part
        K = (model.beats(xtr) + model.trend(xtr)).to_dense() \
            + lik.noise * torch.eye(len(xtr), dtype=torch.float64)
        alpha = torch.linalg.solve(K, ytr - model.mean.constant)
        beats = (model.beats(x, xtr).to_dense() @ alpha).numpy() * y_sd
        trend = (model.trend(x, xtr).to_dense() @ alpha).numpy() * y_sd
        ax = axes[2]
        style(ax)
        ax.axvspan(t[n_tr], t[-1], color="#f3f2ee", lw=0)
        ax.plot(t, beats, color=BLUE, lw=1.2, label="heartbeat part")
        ax.plot(t, trend, color=ORANGE, lw=2, label="baseline part")
        ax.axvline(t[n_tr], color=MUTED, lw=1, ls="--")
        ax.set_xlabel("time (s)", color=MUTED)
        ax.set_ylabel("contribution (a.u.)", color=MUTED)
        ax.set_title("C. The prediction is a sum of the two parts. In the test region (shaded) the heartbeat "
                     "part is copied forward from recent beats.", loc="left", fontsize=10, color=INK)
        ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig("ecg_gp_explained.png", dpi=130)
    print("saved ecg_gp_explained.png")


if __name__ == "__main__":
    main()
