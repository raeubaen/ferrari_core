import os

USE_CUDA = os.getenv("USE_CUDA", "0") == "1"

if USE_CUDA:
    import cupy as xp
else:
    import numpy as xp

from .pseudo_t import pseudo_t
from ..registry import register_routine

@register_routine("fixed_thr")
def fixed_thr(signal_window, valid, max_idx, values_max, **kwargs):
  globals().update(kwargs)

  pseudo_t_array = xp.zeros(signal_window.shape[:2])

  if xp.sum(valid) > 0:

    rise_valid = signal_window[valid, signal_samples_pre_peak - rise_samples_pre_peak:signal_samples_pre_peak + rise_samples_post_peak]
    thresholds = xp.ones_like(values_max)*timing_thr

    idx_valid = xp.where(valid)
    thr_valid  = thresholds[idx_valid]        # (N_valid,)

    pseudo_t_array = pseudo_t(rise_valid, valid, thr_valid, sampling_rate, interpolation_factor, max_idx, rise_interp_left_samples, rise_interp_right_samples, rise_samples_pre_peak, thr_tol=timing_thr_tol)

  return {"time": pseudo_t_array}

