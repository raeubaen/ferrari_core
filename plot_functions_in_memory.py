import json
import time, re, os, ROOT, sys
import numpy as np
import traceback
import importlib
import glob

from .registry import get_routine

plots_plugins = os.getenv("PLOTS_PLUGINS_FOLDER")

if plots_plugins:
    # Ensure path is valid and absolute
    plots_plugins = os.path.abspath(plots_plugins)

    if os.path.isdir(plots_plugins):
        # Add folder to Python path so imports work
        sys.path.insert(0, plots_plugins)

        for filename in os.listdir(plots_plugins):
            if filename.endswith(".py") and not filename.startswith("_"):
                module_name = filename[:-3]  # strip ".py"
                importlib.import_module(module_name)

#not implemented!!
def plot_chunk(args):
    """
    Wrapper to handle chunking for multiprocessing.
    """
    plotconf_df, arrays, plot_output_folder, kwargs = args

    plotconf_df.apply(lambda row: plot(row, arrays, plot_output_folder, **kwargs), axis=1)



def eval_formula(formula, data_dict):
    """
    Evaluate a formula with ${var} syntax and optional @ for broadcasting axes.

    - ${var} -> uproot_dict["var"]
    - ${var}@ -> uproot_dict["var"][:, np.newaxis]
    - ${var}@@ -> uproot_dict["var"][:, np.newaxis][:, np.newaxis] etc.
    """

    def replace_var(m):
        varname = m.group(1)
        at_symbols = m.group(2) or ""
        arr = f'uproot_dict["{varname}"]'
        if at_symbols:
            arr += ''.join(['[:, np.newaxis]' for _ in at_symbols])
        return arr

    # Match ${var} optionally followed by one or more @ symbols
    pattern = re.compile(r'\$\{\s*(\w+)\s*\}(\@*)')
    expr = pattern.sub(replace_var, formula)

    #print("Expression: ", expr)
    # Safe eval environment
    safe_globals = {"uproot_dict": data_dict, "np": np, "__builtins__": {}}
    return eval(expr, safe_globals)


def plot(row, uproot_dict, outputfolder, subfolders_list, f=None, just_draw=False, php_files=None):

  t0 = time.time()
  try:
    os.makedirs(f"{outputfolder}/{row.folder}/", exist_ok=True)

    if row.folder not in subfolders_list:
      if glob.glob(f"{outputfolder}/{row.folder}/*.csv"): os.system(f"rm {outputfolder}/{row.folder}/*.csv")
      if php_files is not None:
        for php_f in php_files:
          os.system(f"/bin/cp {outputfolder}/{php_f}.php {outputfolder}/{row.folder}/{php_f}.php")
          print(f"Done: /bin/cp {outputfolder}/{php_f}.php {outputfolder}/{row.folder}/{php_f}.php")

      subfolders_list.append(row.folder)

  except Exception:
    print(traceback.format_exc(), file=sys.stderr, flush=True)

  #print(f"copying files etc. took: {time.time() - t0}s")

  ROOT.gErrorIgnoreLevel = ROOT.kError

  #print(f"outputfolder: {outputfolder}", file=sys.stderr, flush=True)


  try:
    name = row['name']

    print(name, file=sys.stdout, flush=True)
    print("Drawing: ", name, file=sys.stderr, flush=True)

    ROOT.gROOT.SetBatch(ROOT.kTRUE)

    c = ROOT.TCanvas(f"{name}_canvas")

    with open(f"{outputfolder}/{row.folder}/plots.csv", "a") as plot_csv_for_php_file:
      plot_csv_for_php_file.write(f"{c.GetName()}\n")

    c.cd()

    if just_draw:
      pass
    else:
      if str(row.cuts).strip() == "":
        first_key = next(iter(uproot_dict.keys()))
        #print("no cuts in the .csv, using first key: ", first_key)
        weight = np.ones((uproot_dict[first_key].shape[0],), dtype=bool)
        #print("resulting weight (ones) with shape: ", weight.shape)
      else:
        #print("weighting: ")
        weight = eval_formula(row.cuts, uproot_dict)

      #print("evaluating x: ")
      x = eval_formula(row.x, uproot_dict)
      nevents = x.shape[0]
      x = np.nan_to_num(x.ravel(), nan=-9999999)

    if str(row.y).strip() == "0" and str(row.z).strip() == "0":
        if just_draw:
          h = f["histos"].Get(f"{name}")
        else:
          h = ROOT.TH1F(name, name, int(row.binsnx), float(row.binsminx), float(row.binsmaxx))

          time_fill = time.time()
          h.FillN(len(x), x.astype(np.float64), weight.astype(np.float64))
          #print(f"fillN 1D took {time.time() - time_fill}", file=sys.stderr, flush=True)

        t0 = time.time()
        h.Draw("HIST")
        #print(f"drawing plots took: {time.time() - t0}s")
        h.SetFillColorAlpha(ROOT.kBlue, 0.2)
        h.SetLineColor(eval(f"ROOT.{row.color}"))
        binw = (float(row.binsmaxx) - float(row.binsminx)) / int(row.binsnx)
        #h.GetXaxis().SetRangeUser(h.GetMean() - 3*h.GetRMS(), h.GetMean() + 3*h.GetRMS()) #iterative...
        #h.GetXaxis().SetRangeUser(h.GetMean() - 3*h.GetRMS(), h.GetMean() + 3*h.GetRMS())
        #h.GetXaxis().SetRangeUser(h.GetMean() - 5*h.GetRMS(), h.GetMean() + 5*h.GetRMS())
        h.GetYaxis().SetTitle(f"entries / {float(f'{binw:.1g}'):g} {row.ylabel}")

        c.Update()
        max_bin = h.GetMaximumBin()
        max_position = h.GetBinCenter(max_bin)
        max_value = h.GetBinContent(max_bin)
        bin1 = h.FindFirstBinAbove(max_value/2)
        bin2 = h.FindLastBinAbove(max_value/2)
        fwhm = h.GetBinCenter(bin2) - h.GetBinCenter(bin1)

        pave = ROOT.TPaveText(0.65, 0.7, 0.85, 0.88, "NDC")
        pave.SetFillColor(0)  # Transparent background
        pave.SetTextFont(42)
        pave.SetTextSize(0.03)
        pave.SetBorderSize(0)

        # add three lines
        pave.AddText(f"Events in hist. = {h.Integral()}")
        pave.AddText(f"FWHM/2.35 = {fwhm/2.35:.3f}")
        pave.AddText(f"Peak at x = {max_position:.3f}")
        if max_position > 1000: pave.AddText(f"Ratio = {fwhm/max_position/2.35:.3f}")

        pave.Draw()


    elif str(row.y).strip() != "0" and str(row.z).strip() == "0":
        if just_draw:
          h = f["histos"].Get(f"{name}")
        else:
          #print("evaluating y: ")
          y = eval_formula(row.y, uproot_dict).ravel()
          h = ROOT.TH2F(name, name,
                      int(row.binsnx), float(row.binsminx), float(row.binsmaxx),
                      int(row.binsny), float(row.binsminy), float(row.binsmaxy))
          #print("x.shape: ", x.shape, flush=True)
          #print("y.shape: ", y.shape, flush=True)
          time_fill = time.time()
          h.FillN(len(x), x.astype(np.float64), y.astype(np.float64), weight.astype(np.float64))
          #print(f"fillN 2D took {time.time() - time_fill}", file=sys.stderr, flush=True)

        t0 = time.time()
        h.Draw("ZCOL")
        #print(f"drawing plots took: {time.time() - t0}s")

        h.GetYaxis().SetTitle(row.ylabel)

    else:
        ROOT.gStyle.SetPalette(ROOT.kLightTemperature)
        if just_draw:
          h = f["histos"].Get(f"{name}")
        else:
          #print("evaluating y: ")
          y_notflat = eval_formula(row.y, uproot_dict)
          n_ch = y_notflat.shape[1]
          y = np.nan_to_num(y_notflat.ravel(), nan=-99999)
          #print("evaluating z: ")
          z = np.nan_to_num(eval_formula(row.z, uproot_dict).ravel(), nan=-999999)
          h = ROOT.TH2D(name, name,
                            int(row.binsnx), float(row.binsminx), float(row.binsmaxx),
                            int(row.binsny), float(row.binsminy), float(row.binsmaxy))

          time_fill = time.time()
          h.FillN(len(x),
                x.astype(np.float64),
                y.astype(np.float64),
                z.astype(np.float64)*n_ch)
          #print(f"fillN 2D took {time.time() - time_fill}", file=sys.stderr, flush=True)

        t0 = time.time()
        h.Draw("ZCOL")
        #print(f"drawing plots took: {time.time() - t0}s")
        # 5x5 grid fot TTs

        h.SetContour(int(row.contours))
        h.GetZaxis().SetTitle(row.zlabel)
        h.GetYaxis().SetTitle(row.ylabel)
        h.GetXaxis().SetNdivisions(505)
        h.GetYaxis().SetNdivisions(505)
        c.SetRightMargin(0.18)

    h.GetXaxis().SetTitle(row.xlabel)

    if row.normalize_with_entries != "":
        if int(row.normalize_with_entries) == 1:
          h.Scale(1/h.GetEntries())

    if row.add_commands.strip() != "":
      exec(row.add_commands)
      c.Update()

    t0 = time.time()
    f["canvases"].cd()
    c.Write()
    if not just_draw:
      if row.normalize_with_entries != "":
        if int(row.normalize_with_entries) == 1:
          h.Scale(h.GetEntries())
      f["histos"].cd()
      h.Write()
    #print(f"writing plots took: {time.time() - t0}s")

  except Exception:
    print(traceback.format_exc(), file=sys.stderr, flush=True)
