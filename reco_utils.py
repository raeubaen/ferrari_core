import os

USE_CUDA = os.getenv("USE_CUDA", "0") == "1"

if USE_CUDA:
    import cupy as xp
else:
    import numpy as xp

def split(waveforms, threshold=None, pre=5, post=10, baseline_samples=10, signal_baseline_gap=5, peak_pos_from_highest_ch=False):

    # Assume waveforms is shape (E, C, S)
    E, C, S = waveforms.shape

    if peak_pos_from_highest_ch:
      peak_val_per_ch = xp.max(waveforms, axis=2)      # (E, C)
      best_ch = xp.argmax(peak_val_per_ch, axis=1)     # (E,)

      if threshold is not None:
          peak_pos_per_ch = xp.argmax(waveforms > threshold, axis=2)
      else:
          peak_pos_per_ch = xp.argmax(waveforms, axis=2)

      argmax_ref = peak_pos_per_ch[xp.arange(E), best_ch]

      argmax_idx = xp.broadcast_to(
          argmax_ref[:, None],
          (E, C)
      )


    else:
      if threshold is not None:
        argmax_idx = xp.argmax(waveforms > threshold, axis=2)  # shape (E, C)
      else:
        argmax_idx = xp.argmax(waveforms, axis=2)  # shape (E, C)

    # Step 2: Build offsets
    window_offsets = xp.arange(-int(pre), int(post)).reshape(1, 1, -1)         # shape (1,1,pre+post)
    baseline_offsets = xp.arange(-int(pre)-int(baseline_samples)-int(signal_baseline_gap), -int(pre)-int(signal_baseline_gap)).reshape(1, 1, -1)      # shape (1,1,bs_samples)

    # Expand argmax index for broadcasting
    argmax_exp = argmax_idx[:, :, xp.newaxis]  # shape (E, C, None)

    # Add offsets and wrap with modulo S to stay in bounds
    window_indices   = (argmax_exp + window_offsets) % S        # shape (E, C, pre+post)
    baseline_indices = (argmax_exp + baseline_offsets) % S      # shape (E, C, bs_samples)

    # Build broadcasted event/channel indices
    event_idx = xp.arange(E)[:, None, None]
    chan_idx  = xp.arange(C)[None, :, None]

    baseline_waveforms = waveforms[event_idx, chan_idx, baseline_indices]    # (E, C, bs_samples)

    # Step 3: Compute baseline mean
    baseline = xp.mean(baseline_waveforms, axis=2)	 # shape (E, C)
    baseline_std = xp.std(baseline_waveforms, axis=2)    # shape (E, C)
    baseline_integral = xp.sum(baseline_waveforms, axis=2)  # shape (E, C)

    return argmax_idx, baseline, baseline_std, baseline_integral, (event_idx, chan_idx, window_indices)


def find_central_region(charge_mean, ix, iy, central_region_width, fixed_central_region=None, min_outer_over_seed_ratio=None):
    fake_mask = xp.full(ix.shape, True)
    mask_central_region = xp.full(ix.shape, True)


    while True:
      if fixed_central_region:
          seed_ch = xp.argmax(xp.logical_and(ix==fixed_central_region[0],iy==fixed_central_region[1]))
          print(f"Seed channel: {seed_ch}", flush=True)
      else:
        charge_mean[~fake_mask] = 0
        seed_ch = xp.argmax(charge_mean)

      if int(central_region_width) % 2 == 0: raise ValueError("Central region width MUST be odd")

      all_minus_seed = central_region_width**2 - 1
      cut = (central_region_width+1)/2.

      ix_seed, iy_seed = ix[seed_ch], iy[seed_ch]
      mask_central_region = xp.logical_and(xp.abs(ix - ix_seed) < cut, xp.abs(iy - iy_seed) < cut)
      mask_central_region[seed_ch] = False
      seed_central_region_ratio = xp.sum(charge_mean[mask_central_region]) * all_minus_seed / xp.sum(mask_central_region) / charge_mean[seed_ch]
      mask_central_region[seed_ch] = True
      if fixed_central_region: return mask_central_region, seed_ch
      if seed_central_region_ratio < min_outer_over_seed_ratio:
        fake_mask[seed_ch] = False
        continue
      else:
        break
    print(f"Seed channel: {seed_ch}", flush=True)
    return mask_central_region, seed_ch

