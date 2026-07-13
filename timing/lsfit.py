
import ROOT
import sys
import os
import csv

USE_CUDA = os.getenv("USE_CUDA", "0") == "1"

if USE_CUDA:
    import cupy as xp
    from cupyx.scipy import ndimage
else:
    import numpy as xp
    from scipy import ndimage

from ..registry import register_routine


def load_segments_txt(filename):
    """
    Format expected:

    xc x0 x1 a b c d
    """

    segs = []

    with open(filename, "r") as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"):
                continue

            vals = line.split()

            segs.append({
                "xc": float(vals[0]),
                "x0": float(vals[1]),
                "x1": float(vals[2]),
                "a":  float(vals[3]),
                "b":  float(vals[4]),
                "c":  float(vals[5]),
                "d":  float(vals[6]),
            })

    return segs


class PiecewiseCubicSplineNP:

    def __init__(self, segments, Ts):
        self.x0 = xp.array([s["x0"] for s in segments])
        self.x1 = xp.array([s["x1"] for s in segments])
        self.xc = xp.array([s["xc"] for s in segments])

        self.a = xp.array([s["a"] for s in segments])
        self.b = xp.array([s["b"] for s in segments])
        self.c = xp.array([s["c"] for s in segments])
        self.d = xp.array([s["d"] for s in segments])
        self.Ts = Ts

    def _select(self, x):


        x = xp.asarray(x)  # (EC, N)

        # broadcast-free search of interval index
        idx = xp.searchsorted(self.x1, x)

        idx = xp.clip(idx, 0, self.x0.shape[0] - 1)

        # safety correction (handles edge cases)
        idx_left = xp.maximum(idx - 1, 0)

        use_left = x < self.x0[idx]

        idx = xp.where(use_left, idx_left, idx)

        xc = self.xc[idx]

        dx = x - xc

        mask = xp.abs(dx) <= (self.Ts / 2.0)

        return idx, dx, mask

    def eval_and_derivative(self, x):

        idx, dx, mask = self._select(x)

        P = xp.where(mask, ((self.d[idx] * dx + self.c[idx]) * dx + self.b[idx]) * dx + self.a[idx], 0.0)
        dP = xp.where(mask, self.b[idx] + dx * (2*self.c[idx] + 3*self.d[idx]*dx), 0.0)

        return P, dP


def fit_pulse_iterative(waveforms, pulse, t, t_data_peak, t_template_peak, n_iter=4):

    if USE_CUDA: mempool = xp.get_default_memory_pool()

    if USE_CUDA: print("start iterative:, ", int(mempool.used_bytes()/(1024**2)), "MB")

    EC, N = waveforms.shape

    print("EC,N in lsfit: ", EC, N)

    # initial alignment from DATA only
    dt = xp.full((EC), t_data_peak - t_template_peak)   # (EC)
    A  = xp.ones((EC))


    if USE_CUDA: print("before for:, ", int(mempool.used_bytes()/(1024**2)), "MB")

    for _ in range(n_iter):

        #print("iter: ", _)

        # -------------------------
        # build shifted time grid
        # -------------------------
        # (EC, N)
        t_shift = xp.empty((EC, N), dtype=t.dtype)
        xp.subtract(t[None, :], dt[:, None], out=t_shift)

        #print("time: ", t_shift[0][0])
        if USE_CUDA: print("after time alignment, before eval: ", int(mempool.used_bytes()/(1024**2)), "MB")

        # -------------------------
        # evaluate pulse
        # -------------------------
        P, dP  = pulse.eval_and_derivative(t_shift)        # (EC, N)

        # -------------------------
        # projections (sum over samples)
        # -------------------------
        tmp = xp.empty_like(P)

        xp.multiply(waveforms, P, out=tmp)
        Ap = xp.sum(tmp, axis=1)

        xp.multiply(waveforms, dP, out=tmp)
        Ad = xp.sum(tmp, axis=1)

        xp.multiply(P, P, out=tmp)
        PP = xp.sum(tmp, axis=1)

        xp.multiply(P, dP, out=tmp)
        PdP = xp.sum(tmp, axis=1)

        xp.multiply(dP, dP, out=tmp)
        dPdP = xp.sum(tmp, axis=1)
        # -------------------------
        # solve 2x2 system
        # -------------------------

        #print("denom[0] preclip", (PP * dPdP - PdP * PdP)[0])
        denom = xp.clip(PP * dPdP - PdP * PdP, 1e-12, None)

        A_new = (Ap * dPdP - Ad * PdP) / denom
        Ccorr = (Ad * PP - Ap * PdP) / denom

        # -------------------------
        # update
        # -------------------------
        dt += xp.clip(xp.nan_to_num(-Ccorr / A_new, nan=0.0), -5, 5) ## NOT TO BE HARDCODED; SAVE MEEEEEEEEE!
        A = A_new

        print("denom[0], A_new[0], Ccorr[0]: ", denom[0], A_new[0], Ccorr[0])

    '''
    t_shift = t[None, :] - dt[:, None]
    P_fit, _ = pulse.eval_and_derivative(t_shift)          # (EC, N)
    model = A[:, None] * P_fit

    '''
    with open("fit_debug.csv", "a", newline="") as f:
        writer = csv.writer(f)

        for ch in range(250):
            for i in range(N):
                writer.writerow([
                    ch,                  # channel/event
                    i,                   # sample index
                    A[ch],               # fitted amplitude
                    dt[ch],              # fitted shift
                    waveforms[ch,i], # data
                    model[ch,i]      # fitted template
                ])
    '''
<<<<<<< Updated upstream
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
    return A, dt


@register_routine("lsfit")
def lsfit(signal_window, valid, max_idx, values_max, **kwargs):

    if USE_CUDA: mempool = xp.get_default_memory_pool()

    if USE_CUDA: print("lsfit start:, ", int(mempool.used_bytes()/(1024**2)), "MB")

    globals().update(kwargs)
    Ts = 1/sampling_rate

    idx_valid = xp.where(valid)

    if USE_CUDA: print("after idx valid: ", int(mempool.used_bytes()/(1024**2)), "MB")

    signal_window_valid = signal_window[idx_valid]

    if USE_CUDA: print("after signal window extraction: ", int(mempool.used_bytes()/(1024**2)), "MB")

    max_idx_valid = max_idx[idx_valid]

    if USE_CUDA: print("after max idx valid: ", int(mempool.used_bytes()/(1024**2)), "MB")

    segs = load_segments_txt(spline_file)
    pulse = PiecewiseCubicSplineNP(segs, Ts)

    t_grid = xp.arange(signal_window_valid.shape[1]) * Ts

    if USE_CUDA: print("before eval: ", int(mempool.used_bytes()/(1024**2)), "MB")
    pulse_values, _ = pulse.eval_and_derivative(t_grid)
    if USE_CUDA: print("after eval: ", int(mempool.used_bytes()/(1024**2)), "MB")

    t_pulse_peak = t_grid[xp.argmax(pulse_values)]


    if USE_CUDA: print("lsfit before iterative, after grid:, ", int(mempool.used_bytes()/(1024**2)), "MB")

    amp_valid, dt_valid = fit_pulse_iterative(
        signal_window_valid,
        pulse,
        t_grid,
        signal_samples_pre_peak/sampling_rate,  # data peak (scalar)
        t_pulse_peak     # model peak (scalar)
    )

    fit_time_valid = dt_valid + signal_samples_pre_peak*Ts + max_idx_valid*Ts

    fit_t = xp.zeros(valid.shape, dtype=xp.float32)

    fit_t[idx_valid] = fit_time_valid

    amp_fit = xp.zeros(valid.shape, dtype=xp.float32)

    amp_fit[idx_valid] = amp_valid

    return {"amp": amp_fit, "time": fit_t}
