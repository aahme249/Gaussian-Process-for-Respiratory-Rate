"""
Respiratory rate (RR) estimation from ECG + PPG on the BIDMC dataset with GPyTorch.

Breathing leaves three fingerprints on every heartbeat signal:
  - baseline wander   (RIIV: the PPG/ECG baseline moves up and down with each breath)
  - amplitude changes (RIAV: pulse / R-wave height swells and shrinks)
  - frequency changes (RIFV: the heart speeds up on inhale, slows on exhale = RSA)

For each 32 s window we extract those respiratory signals from the PPG and ECG,
read the dominant breathing frequency off each one (spectral peak in 6-42 breaths/min),
and give the estimates plus how "clean" each peak is to an exact GP. The GP learns
which modulations to trust and returns an RR with an uncertainty band.

The reference RR comes from the manual breath annotations (on the impedance signal,
which is never used as an input). Train/test is split by patient, so test patients
are never seen in training.

Run:  .venv/bin/python respiratory_rate.py
Data: data/bidmc_data.mat from https://physionet.org/content/bidmc/1.0.0/
"""
import numpy as np
import scipy.io
import scipy.signal as ss
import torch
import gpytorch
import matplotlib.pyplot as plt

WIN = 32.0          # window length (s)
STEP = 8.0          # window stride (s)
N_SAMPLES = 1004    # windows used in total
FS_RESP = 4.0       # resampling rate for respiratory signals (Hz)
BAND = (0.1, 0.7)   # plausible breathing band: 6-42 breaths/min
SEED = 0

MODS = ["PPG baseline", "PPG amplitude", "PPG frequency", "ECG amplitude", "ECG frequency"]


# ---------------------------------------------------------------- data
def load_bidmc(path="data/bidmc_data.mat"):
    raw = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)["data"]
    subjects = []
    for s in raw:
        ann = np.asarray(s.ref.breaths.ann1, dtype=float).ravel() - 1  # MATLAB 1-based
        subjects.append(dict(
            ppg=np.asarray(s.ppg.v, float).ravel(),
            ecg=np.asarray(s.ekg.v, float).ravel(),
            imp=np.asarray(s.ref.resp_sig.imp.v, float).ravel(),
            fs=float(s.ppg.fs),
            breaths=ann,
        ))
    return subjects


# ---------------------------------------------------------------- respiratory signals
def bandpass(x, fs, lo, hi, order=3):
    b, a = ss.butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return ss.filtfilt(b, a, x)


def beat_series(sig, fs, peaks, troughs=None):
    """Per-beat baseline, amplitude and interval, each resampled evenly at FS_RESP."""
    t = peaks / fs
    if troughs is None:
        base = sig[peaks]
        amp = sig[peaks]
    else:
        base = sig[troughs]
        amp = sig[peaks] - sig[troughs]
    ibi = np.diff(t)
    grid = np.arange(t[1], t[-1], 1 / FS_RESP)
    out = [np.interp(grid, t, base), np.interp(grid, t, amp), np.interp(grid, t[1:], ibi)]
    return [ss.detrend(o) for o in out]


def spectral_peak(x):
    """Dominant frequency in the breathing band (breaths/min) and peak sharpness (0-1)."""
    x = bandpass(x, FS_RESP, *BAND, order=2) * np.hanning(len(x))
    n = 2048
    f = np.fft.rfftfreq(n, 1 / FS_RESP)
    p = np.abs(np.fft.rfft(x, n)) ** 2
    m = (f >= BAND[0]) & (f <= BAND[1])
    f, p = f[m], p[m]
    k = np.argmax(p)
    near = np.abs(f - f[k]) < 0.03
    return 60 * f[k], p[near].sum() / (p.sum() + 1e-12)


def ppg_signals(ppg, fs):
    x = bandpass(ppg, fs, 0.05, 8)
    peaks, _ = ss.find_peaks(x, distance=int(0.33 * fs), prominence=np.std(x) * 0.3)
    troughs = np.array([p0 + np.argmin(x[p0:p1]) for p0, p1 in zip(peaks[:-1], peaks[1:])])
    peaks = peaks[1:]
    return beat_series(x, fs, peaks, troughs)


def ecg_signals(ecg, fs):
    x = bandpass(ecg, fs, 0.5, 30)
    q = bandpass(ecg, fs, 5, 20) ** 2
    peaks, _ = ss.find_peaks(q, distance=int(0.33 * fs), height=np.percentile(q, 90) * 0.3)
    # snap to the local extremum of the ECG (works for inverted leads too)
    sgn = np.sign(np.median(x[peaks]))
    peaks = np.array([max(p - 5, 0) + np.argmax(sgn * x[max(p - 5, 0):p + 6]) for p in peaks])
    _, amp, ibi = beat_series(sgn * x, fs, peaks)
    return [amp, ibi]


def window_features(sub, t0):
    fs = sub["fs"]
    a, b = int(t0 * fs), int((t0 + WIN) * fs)
    sigs = ppg_signals(sub["ppg"][a:b], fs) + ecg_signals(sub["ecg"][a:b], fs)
    est, qual = zip(*(spectral_peak(s) for s in sigs))
    return np.array(est), np.array(qual)


def reference_rr(sub, t0):
    """Breaths/min from the manual annotations inside the window."""
    br = sub["breaths"] / sub["fs"]
    br = br[(br >= t0) & (br < t0 + WIN)]
    if len(br) < 3:
        return np.nan
    return 60 / np.mean(np.diff(br))


def build_dataset(subjects):
    rows = []
    for sid, sub in enumerate(subjects):
        dur = len(sub["ppg"]) / sub["fs"]
        for t0 in np.arange(0, dur - WIN + 1e-9, STEP):
            ref = reference_rr(sub, t0)
            if not np.isfinite(ref):
                continue
            try:
                est, qual = window_features(sub, t0)
            except (ValueError, IndexError):  # too few detected beats
                continue
            rows.append((sid, t0, ref, est, qual))
    rng = np.random.default_rng(SEED)
    keep = np.sort(rng.choice(len(rows), size=min(N_SAMPLES, len(rows)), replace=False))
    rows = [rows[i] for i in keep]
    sid = np.array([r[0] for r in rows])
    t0 = np.array([r[1] for r in rows])
    y = np.array([r[2] for r in rows])
    est = np.stack([r[3] for r in rows])
    qual = np.stack([r[4] for r in rows])
    return sid, t0, y, est, qual


# ---------------------------------------------------------------- GP
class RRGP(gpytorch.models.ExactGP):
    def __init__(self, x, y, lik):
        super().__init__(x, y, lik)
        self.mean = gpytorch.means.ConstantMean()
        self.cov = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=x.shape[-1]))

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean(x), self.cov(x))


def fit_gp(x, y, iters=300):
    lik = gpytorch.likelihoods.GaussianLikelihood()
    model = RRGP(x, y, lik)
    model.train(); lik.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    for i in range(iters):
        opt.zero_grad()
        loss = -mll(model(x), y)
        loss.backward()
        opt.step()
        if (i + 1) % 100 == 0:
            print(f"  iter {i + 1:3d}  -mll/n = {loss.item():.3f}  noise sd = {lik.noise.sqrt().item():.2f} bpm")
    model.eval(); lik.eval()
    return model, lik


# ---------------------------------------------------------------- plots
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
BLUE, BLUE_BAND, ORANGE = "#2a78d6", "#b7d3f6", "#eb6834"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot_example(sub, t0, path):
    """What the model sees: one window, raw signals and the extracted breathing signals."""
    fs = sub["fs"]
    a, b = int(t0 * fs), int((t0 + WIN) * fs)
    t = np.arange(b - a) / fs
    sigs = ppg_signals(sub["ppg"][a:b], fs) + ecg_signals(sub["ecg"][a:b], fs)
    br = sub["breaths"] / fs
    br = br[(br >= t0) & (br < t0 + WIN)] - t0

    fig, axes = plt.subplots(8, 1, figsize=(10, 11), sharex=True)
    panels = [("PPG (raw)", t, sub["ppg"][a:b]), ("ECG (raw)", t, sub["ecg"][a:b]),
              ("Impedance (reference only)", t, sub["imp"][a:b])]
    for name, s in zip(MODS, sigs):
        tg = np.arange(len(s)) / FS_RESP + (WIN - len(s) / FS_RESP) / 2
        panels.append((f"{name}  →  {spectral_peak(s)[0]:.1f} bpm", tg, s))
    for ax, (name, tt, s) in zip(axes, panels):
        style(ax)
        ax.plot(tt, s, color=INK if "raw" in name or "Imp" in name else BLUE, lw=1.2)
        for x in br:
            ax.axvline(x, color=ORANGE, lw=0.8, alpha=0.6)
        ax.set_yticks([])
        ax.set_title(name, loc="left", fontsize=9, color=INK)
    axes[-1].set_xlabel("time in window (s)", color=MUTED)
    fig.suptitle(f"One {WIN:.0f} s window: orange lines = annotated breaths "
                 f"(reference {reference_rr(sub, t0):.1f} bpm)", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_results(sid, t0, y, mu, lo, hi, base, test_subjects, path):
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1.4])

    # small multiples: every test patient over time
    for k, s in enumerate(test_subjects[:12]):
        ax = fig.add_subplot(gs[k // 4, k % 4])
        style(ax)
        m = sid == s
        o = np.argsort(t0[m])
        tt = t0[m][o] + WIN / 2
        ax.fill_between(tt, lo[m][o], hi[m][o], color=BLUE_BAND, lw=0, label="GP 95% band")
        ax.plot(tt, mu[m][o], color=BLUE, lw=2, marker="o", ms=3, label="GP estimate")
        ax.plot(tt, y[m][o], color=INK, lw=2, label="reference")
        err = np.mean(np.abs(mu[m] - y[m]))
        ax.set_title(f"patient {s + 1:02d}   MAE {err:.1f} bpm", loc="left", fontsize=9, color=INK)
        if k % 4 == 0:
            ax.set_ylabel("breaths/min", fontsize=8, color=MUTED)
        if k == 0:
            ax.legend(fontsize=7, frameon=False, loc="upper left")
    for k in range(len(test_subjects[:12]), 12):
        fig.add_subplot(gs[k // 4, k % 4]).axis("off")

    # predicted vs reference
    ax = fig.add_subplot(gs[3, 0:2])
    style(ax)
    lim = [min(y.min(), mu.min()) - 2, max(y.max(), mu.max()) + 2]
    ax.plot(lim, lim, color=MUTED, lw=1, ls="--")
    ax.errorbar(y, mu, yerr=[mu - lo, hi - mu], fmt="o", ms=4, color=BLUE,
                ecolor=BLUE_BAND, elinewidth=1, mec="white", mew=0.5)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("reference RR (breaths/min)", color=MUTED)
    ax.set_ylabel("GP estimate (breaths/min)", color=MUTED)
    ax.set_title("Every test window: estimate vs reference (bars = 95% band)",
                 loc="left", fontsize=10, color=INK)

    # Bland-Altman
    ax = fig.add_subplot(gs[3, 2:4])
    style(ax)
    avg, diff = (mu + y) / 2, mu - y
    bias, sd = diff.mean(), diff.std()
    ax.scatter(avg, diff, s=16, color=BLUE, edgecolor="white", lw=0.5)
    for v, lab in [(bias, f"bias {bias:+.2f}"), (bias + 1.96 * sd, f"+1.96 SD {bias + 1.96 * sd:+.1f}"),
                   (bias - 1.96 * sd, f"−1.96 SD {bias - 1.96 * sd:+.1f}")]:
        ax.axhline(v, color=MUTED, lw=1, ls="-" if v == bias else "--")
        ax.text(ax.get_xlim()[1] if False else avg.max(), v, " " + lab, va="bottom",
                ha="right", fontsize=8, color=INK)
    ax.set_xlabel("mean of estimate and reference (breaths/min)", color=MUTED)
    ax.set_ylabel("estimate − reference", color=MUTED)
    ax.set_title("Bland–Altman: agreement on test windows", loc="left", fontsize=10, color=INK)

    mae = np.mean(np.abs(mu - y))
    cover = np.mean((y >= lo) & (y <= hi))
    fig.suptitle(f"Respiratory rate on {len(test_subjects)} unseen test patients ({len(y)} windows):  "
                 f"GP MAE {mae:.2f} bpm  |  median-of-signals MAE {np.mean(np.abs(base - y)):.2f} bpm  |  "
                 f"95% band covers {100 * cover:.0f}% of truths", fontsize=11, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    torch.manual_seed(SEED)
    subjects = load_bidmc()
    print(f"loaded {len(subjects)} patients")

    sid, t0, y, est, qual = build_dataset(subjects)
    print(f"{len(y)} windows of {WIN:.0f} s, reference RR {y.min():.1f}-{y.max():.1f} bpm")

    # split by patient: 80% train, 20% test
    rng = np.random.default_rng(SEED)
    pats = rng.permutation(np.unique(sid))
    test_subjects = np.sort(pats[: round(0.2 * len(pats))])
    te = np.isin(sid, test_subjects)
    print(f"train: {(~te).sum()} windows / {len(pats) - len(test_subjects)} patients   "
          f"test: {te.sum()} windows / {len(test_subjects)} patients")

    X = np.hstack([est, qual])
    mu_x, sd_x = X[~te].mean(0), X[~te].std(0) + 1e-9
    Xs = (X - mu_x) / sd_x
    y_mean = y[~te].mean()
    xtr = torch.tensor(Xs[~te], dtype=torch.float64)
    ytr = torch.tensor(y[~te] - y_mean, dtype=torch.float64)
    xte = torch.tensor(Xs[te], dtype=torch.float64)

    print("training GP ...")
    model, lik = fit_gp(xtr, ytr)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = lik(model(xte))
        mu = pred.mean.numpy() + y_mean
        lo, hi = [c.numpy() + y_mean for c in pred.confidence_region()]

    yt = y[te]
    print("\ntest MAE (breaths/min):")
    for j, name in enumerate(MODS):
        print(f"  {name:14s} alone : {np.mean(np.abs(est[te, j] - yt)):5.2f}")
    base = np.median(est[te], axis=1)
    print(f"  median of the 5    : {np.mean(np.abs(base - yt)):5.2f}")
    print(f"  GP                 : {np.mean(np.abs(mu - yt)):5.2f}   "
          f"(RMSE {np.sqrt(np.mean((mu - yt) ** 2)):.2f}, "
          f"95% band covers {100 * np.mean((yt >= lo) & (yt <= hi)):.0f}%)")
    ls = model.cov.base_kernel.lengthscale.detach().numpy().ravel()
    names = [f"{m} RR" for m in MODS] + [f"{m} sharpness" for m in MODS]
    print("\nfeatures the GP relies on most (short lengthscale = more important):")
    for i in np.argsort(ls)[:5]:
        print(f"  {names[i]:26s} lengthscale {ls[i]:.2f}")

    s0 = test_subjects[0]
    plot_example(subjects[s0], t0[sid == s0][len(t0[sid == s0]) // 2], "rr_example_window.png")
    plot_results(sid[te], t0[te], yt, mu, lo, hi, base, test_subjects, "rr_test_results.png")
    print("\nsaved rr_example_window.png and rr_test_results.png")


if __name__ == "__main__":
    main()
