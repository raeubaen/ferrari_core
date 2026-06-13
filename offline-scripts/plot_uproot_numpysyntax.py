import sys

sys.path.append('./')

sys.path.append('../')

import os, json, uproot, argparse, sys, time, ROOT
import pandas as pd
import numpy as np
from pathlib import Path

from .. import plot_functions_in_memory as plot_functions


def main(arguments):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-i", f"--input-file", type=str, required=True, help="input file with reco tree")
    parser.add_argument("-po", f"--plot-output-folder", type=str, required=True, help="output folder for plots")
    parser.add_argument("-j", f"--detectors-conf-json", type=str, required=False, help="detectors reco configuration", default="confs/detectors_conf.json")
    parser.add_argument("-opt", f"--option", type=str, required=True, help="electrons/pions/laser")

    args = parser.parse_args(arguments)

    json_dict = json.load(open(args.detectors_conf_json, "r"))
    mode = json_dict["global"]["spill_type"][args.option]

    plot_list_file = mode["plot_list"]

    #read reco file
    file = uproot.open(args.input_file)
    arrays = file["tree"].arrays(library="np")

    plotconf_df = pd.read_csv(plot_list_file, sep=",", comment='#', quotechar='"', engine='python')
    plotconf_df = plotconf_df.fillna("")

    BASE_DIR = Path(__file__).resolve().parent

    ROOT.gROOT.LoadMacro(f"{BASE_DIR}/../root_logon.C")
    os.system(f"mkdir {args.plot_output_folder}")
    php_files = ["index", "view"]
    for php_f in php_files:
      os.system(f"/bin/cp {os.getenv('PHP_FILES_DIR')}/{php_f}.php {args.plot_output_folder}/{php_f}.php")
    os.system(f"/bin/cp {plot_list_file} {args.plot_output_folder}")

    f = {"canvases": ROOT.TFile(f"{args.plot_output_folder}/canvases.root", "recreate"), "histos": ROOT.TFile(f"{args.plot_output_folder}/histos.root", "recreate")}

    subfolders_list = []
    plotconf_df.apply(lambda row: plot_functions.plot(row, arrays, f"{args.plot_output_folder}/", subfolders_list, f, php_files=php_files), axis=1)

if __name__ == "__main__":
    main(sys.argv[1:])
