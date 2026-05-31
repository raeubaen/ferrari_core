import os

USE_CUDA = os.getenv("USE_CUDA", "0") == "1"

if USE_CUDA:
    import cupy as xp
    from cupyx.scipy import ndimage
else:
    import numpy as xp
    from scipy import ndimage

def get_rise_interp(rise_valid, valid, thr_valid, interpolation_factor, argmax_idx, rise_interp_left_samples, rise_interp_right_samples, thr_tol=None):

    if thr_tol is not None:
      result = ( (xp.diff(rise_valid)>0)*rise_valid[:, :-1] > thr_valid[:, None] ) * (xp.abs(rise_valid - thr_valid[:, None]) < thr_tol)[:, :-1]
    else:
      result = ( (xp.diff(rise_valid)>0)*rise_valid[:, :-1] > thr_valid[:, None] )

    prelim_pseudo_t = xp.argmax(result, axis=1)

    offsets = xp.arange(
        -rise_interp_left_samples,
         rise_interp_right_samples + 1
    )

    idx = prelim_pseudo_t[:, None] + offsets[None, :]
    idx = xp.clip(idx, 0, rise_valid.shape[1] - 1)

    rise_segment = xp.take_along_axis(
        rise_valid,
        idx.astype(xp.int32),
        axis=1
    )

    rise_interp = ndimage.zoom(
        rise_segment,
        [1, interpolation_factor],
        order=3,
    )

    return prelim_pseudo_t, rise_interp  # remove dummy axis

def pseudo_t(rise_valid, valid, thr_valid, sampling_rate, interpolation_factor, argmax_idx, rise_interp_left_samples, rise_interp_right_samples, rise_samples_pre_peak, thr_tol=None):

    idx_valid = xp.where(valid)

    prelim_pseudo_t, rise_interp = get_rise_interp(rise_valid, valid, thr_valid, interpolation_factor, argmax_idx, rise_interp_left_samples, rise_interp_right_samples, thr_tol=thr_tol)

    pseudo_t_valid = xp.argmax(
        rise_interp > thr_valid[:, None],
        axis=1
    ).astype(xp.float32)

    pseudo_t_valid += xp.random.uniform(
        low=-0.5,
        high=0.5,
        size=pseudo_t_valid.shape
    )

    pseudo_t_valid = (
        pseudo_t_valid / interpolation_factor
        + prelim_pseudo_t
        - rise_interp_left_samples
        + argmax_idx[idx_valid]   # IMPORTANT: match indexing
        - rise_samples_pre_peak
    )

    pseudo_t_valid /= sampling_rate


    pseudo_t = xp.zeros(valid.shape, dtype=xp.float32)

    pseudo_t[idx_valid] = pseudo_t_valid

    return pseudo_t

