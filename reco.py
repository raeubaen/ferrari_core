import os, json, uproot, argparse, sys, time, ROOT, copy
import numpy as np
import traceback

import pandas as pd
import importlib
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor as tpe

from . import reco_functions
from . import plot_functions_in_memory as plot_functions
from . import reco_utils

from .registry import get_routine
from .default_generic_reco_conf import default_generic_reco_conf

import cProfile


def main(arguments):

    BASE_DIR = Path(__file__).resolve().parent

    print("Entered reco.py")
    # start time
    time_start = time.time()

    # input parameters
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-i",  f"--input", type=str, required=True, help="input ROOT file with unpacked tree")
    parser.add_argument("-r",  f"--run", type=str, required=True, help="run number")
    parser.add_argument("-s",  f"--fragment", type=str, required=True, help="fragment number")
    parser.add_argument("-ro", f"--reco-output-dir", type=str, required=True, help="directory for reco output")
    parser.add_argument("-j", f"--detectors-conf-json", type=str, required=False, help="detectors reco configuration", default="confs/detectors_conf.json")
    parser.add_argument("-ct", f"--compression-type", type=str, required=False, help="mcp reco configuration", default="lz4")
    parser.add_argument("-po", f"--plot-output-folder", type=str, required=False, help="output folder for plots", default=None)
    parser.add_argument("-opt", f"--option", type=str, required=True, help="electrons/pions/laser")
    parser.add_argument("-n", f"--n-cpus", type=int, required=False, help="#cpus to use (if going parallel)", default=4)
    parser.add_argument("-dp",  f"--do-plots", type=int, required=False, help="do plots?", default=1)

    args = parser.parse_args(arguments)

    # read detectors configuration
    json_dict = json.load(open(args.detectors_conf_json, "r"))
    detectors_dict = json_dict["detectors"]
    mode = json_dict["global"]["fragment_type"][args.option]

    for p in json_dict["global"]["plugins"]:
      importlib.import_module(p)

    opt = mode["option"]
    if opt is not None:
        for detector in opt:
            for conf in opt[detector]["reco_conf"]:
                detectors_dict[detector]["generic_reco"]["reco_conf"][conf] = opt[detector]["reco_conf"][conf]
    print(f"args + conf took {-time_start + time.time():.1f} s")
    plot_list_file = mode["plot_list"]

    # open input file
    time_open = time.time()
    file = uproot.open(args.input)
    tree = file[mode["tree_name"]]
    n_events = tree.num_entries
    print(f"Tree contains: {n_events} events")
    if n_events == 0:
      print("Tree contains 0 events -> exiting")
      sys.exit(0)

    print(f"open file took {-time_open + time.time():.1f} s")

    # reconstruction
    time_reco = time.time()
    reco_dict = {}

    ready_waves = {}

    for detector in detectors_dict:

        try:
          time_reco_det = time.time()
          if detector not in mode["detector_list"]: continue
          print(f"reco {detector} ongoing")
          dd = detectors_dict[detector]

          reco_dict[detector] = {}

          if dd["generic_reco"] is not None:
              gen_reco_dict = dd["generic_reco"]
              geo_dict, chid_dict, gain_list, intercalib_list, gain_is_high = None, None, None, None, False

              if gen_reco_dict["ch_map"] == None: active_ch_list = slice(None)
              elif isinstance(gen_reco_dict["ch_map"], str):
                map_df = pd.read_csv(gen_reco_dict["ch_map"], comment='#')
                active_row_list = (map_df["type"] == detector).tolist()
                active_ch_list = (map_df["branch_ch"][map_df["type"] == detector]).tolist()
                chid_dict = {var: map_df[var].to_numpy()[active_row_list] for var in gen_reco_dict["chid_vars_list"]}
                if gen_reco_dict["geo_needed"] is not None:
                  geo_dict = {coord: map_df[coord].to_numpy()[active_row_list] for coord in gen_reco_dict["geo_needed"]}
                if gen_reco_dict["apply_gain_ratios"] is not None:
                  gain_list = map_df[gen_reco_dict["apply_gain_ratios"]].to_numpy()[active_row_list]
                if gen_reco_dict["apply_intercalib"]:
                  intercalib_list = map_df[gen_reco_dict["apply_intercalib"]].to_numpy()[active_row_list]
              elif isinstance(gen_reco_dict["ch_map"], list):
                active_ch_list = gen_reco_dict["ch_map"]

              if gen_reco_dict["waves_branch"] not in ready_waves.keys():
                time_readarrays = time.time()

                decompr_exec = tpe(max_workers=6)
                interpret_exec = tpe(max_workers=6)

                waves = tree[gen_reco_dict["waves_branch"]].array(library="np", decompression_executor=decompr_exec, interpretation_executor=interpret_exec)
                print(f"{detector} read into arrays took: {time.time() - time_readarrays}")

                print(f"{detector} waves shape: {waves.shape}")
                time_reshape = time.time()
                if len(waves.shape) == 4: waves = waves.reshape(waves.shape[0], waves.shape[1]*waves.shape[2], waves.shape[3]) #(n_board, n_ch) format

                ready_waves[gen_reco_dict["waves_branch"]] = waves

              else:
                waves = ready_waves[gen_reco_dict["waves_branch"]]

              waves = waves[:, active_ch_list, :]

              if gen_reco_dict["decode_and_select_gains"] is not None:
                  waves, gain_is_high = get_routine(gen_reco_dict["decode_and_select_gains"])(waves.astype(np.uint16))
              if gen_reco_dict["remove_last_n_samples"] != 0: waves = waves[:, :, : -gen_reco_dict["remove_last_n_samples"]]
              if gen_reco_dict["to_be_inverted"]: waves = 4096 - waves #must be inverted if the signal are with negative rising slope

              reco_conf = copy.deepcopy(default_generic_reco_conf)
              reco_conf.update(gen_reco_dict["reco_conf"])
              reco_dict[detector]["mask"], reco_dict[detector]["arrays"] = reco_functions.generic_reco(
                waves.astype(np.float16), detector, gain_is_high=gain_is_high, gain_list=gain_list, id=chid_dict, geo_dict=geo_dict, intercalib_list=intercalib_list, **reco_conf #n_cpus=args.n_cpus: not implemented
              )

              if reco_conf["post_process_routine"] is not None: reco_dict[detector]["mask"], reco_dict[detector]["arrays"] = get_routine(reco_conf["post_process_routine"])(reco_dict[detector]["mask"], reco_dict[detector]["arrays"], **reco_conf)
          else:
              reco_dict[detector]["mask"], reco_dict[detector]["arrays"] = get_routine(dd["custom_reco"])(tree, detector, dd)

          print(""f"{detector} reco took {-time_reco_det + time.time():.1f} s")
        except Exception:
          print(traceback.format_exc())
          reco_dict.pop(detector, None)
    print(f"reco took: {-time_reco + time.time():.1f} s")

    # time run controller

    # add event number
    n_events = np.arange(reco_dict[list(reco_dict.keys())[0]]["mask"].shape[0])
    reco_dict["event_info"] = {"mask": np.ones((n_events.shape[0],), dtype=bool), "arrays": {"n_event": n_events}}
    for b in mode["global_branches"]: reco_dict["event_info"]["arrays"].update({b: tree[b].array(library="np")})

    # merging
    time_merge = time.time()
    mask_global, arrays = np.logical_and.reduce([reco_dict[detector]["mask"] for detector in reco_dict]), {}
    for detector in reco_dict: arrays.update(reco_dict[detector]["arrays"])
    for branch in arrays: arrays[branch] = arrays[branch][mask_global, ...]
    print([br for br in arrays])
    print(f"merging took {-time_merge + time.time():.1f} s")

    if args.do_plots:
      # plotting
      time_plot = time.time()
      plotconf_df = pd.read_csv(plot_list_file, sep=",", comment='#', quotechar='"', engine='python')
      plotconf_df = plotconf_df.fillna("")

      ROOT.gROOT.LoadMacro(f"{BASE_DIR}/root_logon.C")
      os.system(f"mkdir {args.plot_output_folder}")
      php_files = ["index", "view"]
      for php_f in php_files:
        os.system(f"/bin/cp {BASE_DIR}/php/{php_f}.php {args.plot_output_folder}/{php_f}.php")
      os.system(f"/bin/cp {plot_list_file} {args.plot_output_folder}")

      f = {"canvases": ROOT.TFile(f"{args.plot_output_folder}/canvases.root", "recreate"), "histos": ROOT.TFile(f"{args.plot_output_folder}/histos.root", "recreate")}

      subfolders_list = []
      plotconf_df.apply(lambda row: plot_functions.plot(row, arrays, f"{args.plot_output_folder}/", subfolders_list, f, php_files=php_files), axis=1)

      for key in f: f[key].Close()
      print(f"plotting took {-time_plot + time.time():.1f} s")

    # writing
    time_write = time.time()


    branch_types = {}

    for k, v in arrays.items():
        if len(v.shape) == 1:
            branch_types[k] = v.dtype
        else:
            branch_types[k] = np.dtype((v.dtype, v.shape[1:]))

    compression_map = {"zlib": uproot.compression.ZLIB(level=1), "lz4": uproot.compression.LZ4(level=1), "none": None}
    outfile_name = f"{args.reco_output_dir}/{args.run}_{args.fragment}_reco.root"
    print("Saving in: ", outfile_name)
    with uproot.recreate(outfile_name, compression=compression_map[args.compression_type]) as f:
        tree = f.mktree("tree", branch_types)
        tree.extend(arrays)
    print(f"writing reco output took {-time_write + time.time():.1f} s")

    print(f"total reco.py took {-time_start + time.time():.1f} s")
    print("----------------- unpacking, reco and plotting single fragment done -----------------")

if __name__ == '__main__':
    main(sys.argv[1:])
