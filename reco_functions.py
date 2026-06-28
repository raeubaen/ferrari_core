import time
import os
import numpy as np

USE_CUDA = os.getenv("USE_CUDA", "0") == "1"

print(USE_CUDA)

if USE_CUDA:
    import cupy as xp
else:
    import numpy as xp

from .import reco_utils
from .timing import *
from .registry import get_routine

def generic_reco(waves, detector_name, gain_is_high=False, gain_list=None, **kwargs):

  globals().update(kwargs)

  if USE_CUDA:

    mempool = xp.get_default_memory_pool()

    print("Before transfers, using in GPU:, ", int(mempool.used_bytes()/(1024**2)), "MB")

    print("starting to transfer to VRAM")
    waves = xp.asarray(waves).astype(xp.float32)

    print("transferred - starting processing")

    print("Before processing, using in GPU:, ", int(mempool.used_bytes()/(1024**2)), "MB")

  t0 = time.time()

  if pre_process_routine is not None:
    waves = get_routine(pre_process_routine)(waves, **kwargs)

  max_idx, baselines, baselines_std, baseline_integral, signal_window_3d_indices = reco_utils.split(waves, signal_baseline_gap=signal_baseline_gap, pre=signal_samples_pre_peak, post=signal_samples_post_peak, baseline_samples=baseline_samples, threshold=raw_threshold_before_peak_finding, peak_pos_from_highest_ch=peak_pos_from_highest_ch, peak_accept_window_ns_from_highest_ch=peak_accept_window_ns_from_highest_ch, sampling_rate=sampling_rate)

  print(f"baselines evaluation took: {time.time() - t0}")
  t0 = time.time()


  event_idx = xp.arange(waves.shape[0])[:, None]        # shape (E, 1)
  chan_idx  = xp.arange(waves.shape[1])[None, :]        # shape (1, C)

  values_mean = xp.mean(waves, axis=2) # mean of all values
  values_std = xp.std(waves, axis=2)   # std of all values

  signal_window = waves[tuple(signal_window_3d_indices)]

  if baseline_subtract:
    print("subtracting bline")
    signal_window = signal_window - baselines[:, :, None]

  signal_window = xp.nan_to_num(signal_window, nan=0.0)

  if gain_list is not None:
      gains = xp.asarray(gain_list[None, :, None])
      gain_is_high_window = xp.asarray(gain_is_high)[tuple(signal_window_3d_indices)]
      signal_window *= xp.where(gain_is_high_window, gains, 1)

  if intercalib_dict is not None:
    for key in intercalib_dict:
      signal_window *= xp.asarray(intercalib_dict[key][None, :, None])

  values_max = xp.max(signal_window, axis=2)

  mask_under_thr = values_max < charge_zerosup_peak_threshold

  signal_window[mask_under_thr, :] = 0

  charge = xp.zeros_like(values_max)
  charge[~mask_under_thr] = xp.sum(signal_window[~mask_under_thr, :], axis=-1)

  #charge[~mask_under_thr] = xp.clip(charge[~mask_under_thr], 0, None)

  if charge_unit_conversion:
     charge = charge * charge_unit_slope

  charge = xp.nan_to_num(charge, nan=0.0)

  ich = xp.repeat(xp.arange(0, waves.shape[1])[xp.newaxis, :], charge.shape[0], axis=0)

  return_dict = {}
  mask_selected_events = xp.ones((charge.shape[0],), dtype=bool)
  det = detector_name

  print(f"baseline subtraction, charge integration things took: {time.time() - t0}")
  t0 = time.time()

  if geo_dict is not None:
    ix, iy = (xp.asarray(geo_dict[key]) for key in coords_2d_list[:2])
    if coord_z is not None: iz = geo_dict[coord_z]
    else: iz = None

    if not do_central_region: mask_central_region = xp.full(ix.shape, True)
    else:
      charge_mean = xp.mean(charge, axis=0)
      seed_ch = -999

      mask_central_region, seed_ch = reco_utils.find_central_region(charge_mean, ix, iy, central_region_width, fixed_central_region=fixed_central_region, min_outer_over_seed_ratio=min_outer_over_seed_ratio)
      print(f"find central_region took: {time.time() - t0}")
      t0 = time.time()

      peak_seed = values_max[:, seed_ch]

      charge_seed = charge[:, seed_ch]
      charge_sum_central_region = xp.sum(charge[:, mask_central_region], axis=1)

      charge_sum_central_region = xp.clip(charge_sum_central_region, 1e-8, None)
      mask_low_charge_seed = charge_seed > seed_charge_threshold

      # amplitude_map of the central_region matrix
      charge_fraction_central_region = xp.zeros(charge.shape)
      charge_fraction_central_region[:, mask_central_region] = charge[:, mask_central_region] / charge_sum_central_region[:, xp.newaxis]

      print(f"seed/central_region/fractions took: {time.time() - t0}")
      t0 = time.time()

      iy_within_central_region = iy - iy[seed_ch]
      ix_within_central_region = ix - ix[seed_ch]

      ix_within_central_region = xp.repeat(ix_within_central_region[xp.newaxis, :], charge.shape[0], axis=0)
      iy_within_central_region = xp.repeat(iy_within_central_region[xp.newaxis, :], charge.shape[0], axis=0)

      t0 = time.time()

      seed_ch_app = seed_ch
      seed_ch = xp.repeat(xp.ones(1,)*seed_ch, charge_sum_central_region.shape[0], axis=0)

      highest_ch = xp.argmax(charge * mask_central_region, axis=1)
      highest_charge = xp.take_along_axis(charge, highest_ch[:, None], axis=1).squeeze()
      highest_peak = xp.take_along_axis(values_max, highest_ch[:, None], axis=1).squeeze()

      print(f"highest ch took: {time.time() - t0}")
      t0 = time.time()

      print(f"tau took: {time.time() - t0}")
      t0 = time.time()

      kxk = f"{central_region_width}x{central_region_width}"
      return_dict.update({
        f"{det}_charge_sum_{kxk}": charge_sum_central_region, f"{det}_charge_seed": charge_seed, f"{det}_peak_seed": peak_seed, f"{det}_seed_over_{kxk}": charge_fraction_central_region[:, seed_ch_app],
        f"{det}_highest_charge_over_{kxk}": highest_charge/charge_sum_central_region,
        f"{det}_{coords_2d_list[1]}_within_{kxk}": iy_within_central_region, f"{det}_{coords_2d_list[0]}_within_{kxk}": ix_within_central_region,
        f"{det}_charge_divided_{kxk}": charge_fraction_central_region, f"{det}_seed_ch": seed_ch,
        f"{det}_highest_ch": highest_ch, f"{det}_highest_charge": highest_charge, f"{det}_highest_peak": highest_peak,
      })

      #mask_selected_events = mask_low_charge_seed

    if do_centroid:
      if not do_central_region:
        charge_sum_central_region = xp.sum(charge[:, mask_central_region], axis=1)
        charge_fraction_central_region = xp.zeros(charge.shape)
        charge_fraction_central_region[:, mask_central_region] = charge[:, mask_central_region] / charge_sum_central_region[:, xp.newaxis]

      if w0_log_centroid is not None:
        w = xp.maximum(0.0,w0_log_centroid + xp.log(xp.clip(charge_fraction_central_region, 1e-8, None)))
        w /= (xp.sum(w, axis=1, keepdims=True))
      else: w = xp.clip(charge_fraction_central_region, 1e-8, None)

      ix_centroid = w[:, mask_central_region] @ ix[mask_central_region]
      iy_centroid = w[:, mask_central_region] @ iy[mask_central_region]

      return_dict.update({f"{det}_{coords_2d_list[0]}_centroid": ix_centroid, f"{det}_{coords_2d_list[1]}_centroid": iy_centroid})


      print(f"centrois took: {time.time() - t0}")

    ix = xp.repeat(ix[xp.newaxis, :], charge.shape[0], axis=0)
    iy = xp.repeat(iy[xp.newaxis, :], charge.shape[0], axis=0)
    if iz is not None: iz = xp.repeat(iz[xp.newaxis, :], charge.shape[0], axis=0)

  if do_tau:
      if do_central_region: tau_mask = mask_central_region
      else: tau_mask = xp.full((signal_window.shape[1],), True)

      descent = signal_window[:, tau_mask, signal_samples_pre_peak+1:signal_samples_pre_peak+tau_descent_samples+1]
      log_w = xp.log(xp.clip(descent, 1, None))
      log_slopes = xp.diff(log_w, axis=2) / descent.shape[2]
      tau = xp.zeros(charge.shape)
      tau[:, tau_mask] = -1.0 / (1e-12 + xp.median(log_slopes, axis=2) * sampling_rate)
      return_dict.update({f"{det}_tau": tau})

  if do_timing:
    if do_central_region: timing_mask = mask_central_region
    else: timing_mask = xp.full((signal_window.shape[1],), True)

    valid = ~mask_under_thr & timing_mask[None, :]

    for timing_method in timing_methods:
      timing_function = get_routine(timing_method)

      timing_function_result = timing_function(signal_window, valid, max_idx, values_max, **kwargs)

      for key in timing_function_result:
        return_dict.update({f"{det}_{timing_method}_{key}": timing_function_result[key]})

  per_ch_info = {
    f"{det}_peak_pos": max_idx, f"{det}_peak_time": max_idx/sampling_rate,
    f"{det}_charge": charge, f"{det}_peak": values_max, f"{det}_baseline_mean": baselines,
    f"{det}_baseline_std": baselines_std, f"{det}_baseline_integral": baseline_integral/baseline_samples*signal_window.shape[2],
  }

  if save_mean_rms_all_samples:
    per_ch_info.update({f"{det}_samples_mean": values_mean, f"{det}_samples_std": values_std})
  if geo_dict is not None:
    per_ch_info.update({f"{det}_{coords_2d_list[0]}": ix, f"{det}_{coords_2d_list[1]}": iy})
    if iz is not None: per_ch_info.update({f"{det}_{coord_z}": iz})
  if id is not None:
    for var in id:
      per_ch_info.update({f"{det}_{var}": xp.repeat(id[var][xp.newaxis, :], waves.shape[0], axis=0)})

  if do_central_region and save_only_central_region_info:
    for key in per_ch_info:
      return_dict[key] = xp.zeros(per_ch_info[key].shape)
      return_dict[key][:, mask_central_region] = per_ch_info[key][:, mask_central_region]
  else:
    return_dict.update(per_ch_info)

  if USE_CUDA:

    for br in return_dict:
       return_dict[br] = xp.asnumpy(return_dict[br])
    mask_selected_events = xp.asnumpy(mask_selected_events)

  return mask_selected_events, return_dict



#not implemented
def generic_reco_chunk(args):
    try:
        waves, det, kwargs = args
        return generic_reco(waves, det, **kwargs)
    except Exception:
        print(traceback.format_exc(), file=sys.stderr, flush=True)


#not implemented
def generic_reco_parallel(waves, detector_name, n_cpus=2, **kwargs):
    E = waves.shape[0]
    chunk_size = (E + n_cpus - 1) // n_cpus  # ceil division
    chunks = [(waves[i*chunk_size:(i+1)*chunk_size], detector_name, kwargs)
              for i in range(n_cpus)]

    print("opening pool")
    results = [generic_reco_chunk(chunk) for chunk in chunks]

    # Combine results
    masks_list, dicts_list = zip(*results)
    combined_mask = xp.concatenate(masks_list, axis=0)

    combined_dict = {}
    for key in dicts_list[0].keys():
        combined_dict[key] = xp.concatenate([d[key] for d in dicts_list], axis=0)

    return combined_mask, combined_dict
