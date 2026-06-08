import sys

sys.path.append('./')

sys.path.append('../')

import os, json, uproot, argparse, sys, time, ROOT
import pandas as pd
import numpy as np

from .. import plot_functions_in_memory as plot_functions


def main(arguments):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-i", f"--input-file", type=str, required=True, help="input file with reco tree")
    parser.add_argument("-p",  f"--plot-list", type=str, required=True, help="csv file with plot list")
    parser.add_argument("-po", f"--plot-output-folder", type=str, required=True, help="output folder for plots")
    args = parser.parse_args(arguments)

    #read reco file
    file = uproot.open(args.input_file)
    arrays = file["tree"].arrays(library="np")

    plotconf_df = pd.read_csv(args.plot_list, sep=",", comment='#', quotechar='"', engine='python')
    plotconf_df = plotconf_df.fillna("")

    ROOT.gROOT.LoadMacro("root_logon.C")
    os.system(f"mkdir -p {args.plot_output_folder}")
    php_files = ["index", "view"]
    for php_f in php_files:
      os.system(f"/bin/cp php/{php_f}.php {args.plot_output_folder}/{php_f}.php")

    f = {"canvases": ROOT.TFile(f"{args.plot_output_folder}/canvases.root", "recreate"), "histos": ROOT.TFile(f"{args.plot_output_folder}/histos.root", "recreate")}

    subfolders_list = []
    plotconf_df.apply(lambda row: plot_functions.plot(row, arrays, f"{args.plot_output_folder}/", subfolders_list, f, php_files=php_files), axis=1)

if __name__ == "__main__":
    main(sys.argv[1:])
