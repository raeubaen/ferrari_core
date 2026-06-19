import os

USE_CUDA = os.getenv("USE_CUDA", "0") == "1"

if USE_CUDA:
    import cupy as xp
    from cupyx.scipy import ndimage
else:
    import numpy as xp
    from scipy import ndimage

from ..registry import register_routine
from .pseudo_t import pseudo_t

def build_peak_interp(
    rise_valid,
    rise_samples_pre_peak,
    rise_interp_left_samples,
    rise_interp_right_samples,
    interpolation_factor,
):
    """
    Returns:
        peak_interp
        peak_value
    """

    n_wf, _ = rise_valid.shape


    offsets = xp.arange(
        -rise_interp_left_samples,
        rise_interp_right_samples + 1,
        dtype=xp.int32,
    )

    # center is peak reference (corrected)
    idx_peak = rise_samples_pre_peak + offsets[None, :]
    idx_peak = xp.clip(idx_peak, 0, rise_valid.shape[1] - 1)

    peak_segment = xp.take_along_axis(
        rise_valid,
        idx_peak,
        axis=1,
    )

    #n_traces, n_samples = peak_segment.shape ##MC

    ## Degenerate case: nothing to interpolate ##MC
    #if n_samples == 0 or n_traces == 0: ##MC
    #    # Option A: return an empty array with the expected shape ##MC
    #    return xp.zeros((n_traces, 0), dtype=peak_segment.dtype) ##MC


    peak_interp = ndimage.zoom(
        peak_segment,
        [1, interpolation_factor],
        order=5, prefilter=False
    )

    peak_value = xp.max(peak_interp, axis=1)

    return peak_value


@register_routine("cf")
def cf(signal_window, valid, max_idx, values_max, **kwargs):
  globals().update(kwargs)

  pseudo_t_array = xp.zeros(signal_window.shape[:2])

  if xp.sum(valid) > 0:
    rise_valid = signal_window[valid, signal_samples_pre_peak - rise_samples_pre_peak:signal_samples_pre_peak + rise_samples_post_peak]

    thresholds = build_peak_interp(rise_valid, rise_samples_pre_peak, rise_interp_left_samples, rise_interp_right_samples, interpolation_factor) * cf
    pseudo_t_array = pseudo_t(rise_valid, valid, thresholds, sampling_rate, interpolation_factor, max_idx, rise_interp_left_samples, rise_interp_right_samples, rise_samples_pre_peak)

  return {"time": pseudo_t_array}
