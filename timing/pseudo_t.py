import os

USE_CUDA = os.getenv("USE_CUDA", "0") == "1"

if USE_CUDA:
    import cupy as xp
    from cupyx.scipy import ndimage
else:
    import numpy as xp
    from scipy import ndimage



def cubic_spline_interp(y, factor):
    # y: (n_traces, n_samples)

    n_traces, n = y.shape

    x_new = xp.linspace(
        0,
        n - 1,
        n * factor,
        dtype=xp.float32
    )

    # segmento dove cade ogni nuovo punto
    i = xp.floor(x_new).astype(xp.int32)
    i = xp.clip(i, 0, n - 2)

    t = x_new - i

    # punti del segmento
    y0 = y[:, i]
    y1 = y[:, i + 1]

    # pendenze (Hermite)
    m = xp.empty_like(y)

    m[:, 1:-1] = 0.5 * (y[:, 2:] - y[:, :-2])
    m[:, 0] = y[:, 1] - y[:, 0]
    m[:, -1] = y[:, -1] - y[:, -2]

    m0 = m[:, i]
    m1 = m[:, i + 1]

    # coefficienti Hermite cubici
    t2 = t * t
    t3 = t2 * t

    h00 =  2*t3 - 3*t2 + 1
    h10 =      t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 =      t3 -   t2

    return (
        h00[None, :] * y0 +
        h10[None, :] * m0 +
        h01[None, :] * y1 +
        h11[None, :] * m1
    )

def get_rise_interp(rise_valid, valid, thr_valid, interpolation_factor, argmax_idx, rise_interp_left_samples, rise_interp_right_samples, thr_tol=None):

    #print(rise_valid.shape)
    #print(thr_valid.shape)
    #n_traces, n_samples = rise_valid.shape ##MC
    #if n_traces == 0 or n_samples == 0: ##MC
    #    # Decide what you want to return in this case:
    #    #   - prelim_pseudo_t: empty
    #    #   - rise_interp: empty
    #    prelim_pseudo_t = xp.empty((0,), dtype=xp.int32) ##MC
    #    rise_interp = xp.empty((0, 0), dtype=rise_valid.dtype) ##MC
    #    return prelim_pseudo_t, rise_interp ##MC

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


    #print("rise_segment: ", rise_segment[0, :])
    #print("rise segment dshape: ", rise_segment.shape)

    rise_interp = cubic_spline_interp(
        rise_segment,
        interpolation_factor
    )

    #print(rise_interp[0, :])

    #print("rise interp dshape: ", rise_interp.shape)

    # porta su CPU se sei su CUDA
    '''
    rise_seg = xp.asnumpy(rise_segment[0, :])
    rise_int = xp.asnumpy(rise_interp[0, :])

    # tempi
    t_seg = np.arange(len(rise_seg))
    t_interp = np.linspace(
        0,
        len(rise_seg) - 1,
        len(rise_int)
    )

    # CSV rise_segment
    np.savetxt(
        "rise_segment.csv",
        np.column_stack((t_seg, rise_seg)),
        delimiter=",",
        header="time,sample",
        comments=""
    )

    # CSV rise_interp
    np.savetxt(
        "rise_interp.csv",
        np.column_stack((t_interp, rise_int)),
        delimiter=",",
        header="time,sample",
        comments=""
    )
    '''

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

    corr_factor = rise_interp_left_samples+rise_interp_right_samples
    corr_factor /= (rise_interp_left_samples+rise_interp_right_samples + 1)

    pseudo_t_valid = (
        pseudo_t_valid / interpolation_factor * corr_factor
        + prelim_pseudo_t
        - rise_interp_left_samples
        + argmax_idx[idx_valid]   # IMPORTANT: match indexing
        - rise_samples_pre_peak
    )

    pseudo_t_valid /= sampling_rate


    pseudo_t = xp.zeros(valid.shape, dtype=xp.float32)

    #print("timing shape") ##MC
    #print(f"pseudo_t.shape: {pseudo_t.shape}") ##MC
    #print(f"pseudo_t_valid: {pseudo_t_valid.shape}") ##MC
    #idx_valid = xp.asarray(idx_valid)                           ##MC

    ## Degenerate case: nothing to assign                        ##MC
    #if pseudo_t_valid.size == 0 or idx_valid.sum() == 0:        ##MC
    #    # Nothing to write into pseudo_t for this batch         ##MC
    #    return pseudo_t                                         ##MC

    pseudo_t[idx_valid] = pseudo_t_valid

    #print("pseudo_t_valid", pseudo_t_valid)
    return pseudo_t

