#! /usr/env/bin python3

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    import cartopy.crs as ccrs
except ModuleNotFoundError:
    pass


FILE_AND_MODEL_NAMES = {
    "Verification": "clstm1m_cyl_4x57_v0",
    "ConvLSTM Cyl (16M)": "clstm16m_cyl_4x228_v2",
    "U-Net Cyl (128M)": "unet128m_cyl_128-256-512-1024-2014_v2",
    "SwinTransformer Cyl (2M)": "swint2m_cyl_d88_l2x4_h2x4_v0",
    "FNO (64M)": "fno2d64m_cyl_d307_v0",
    "ConvLSTM HPX (16M)": "clstm16m_hpx8_4x228_v1",
    "U-Net HPX (16M)": "unet16m_hpx8_92-184-368-736_v0",
    "SwinTransformer HPX (16M)": "swint16m_hpx8_d120_l3x4_h3x4_v2",
    "TFNO (128M)": "tfno2d128m_cyl_d477_v0",
    r"FourCastNet $1x1$ (4M)": "fcnet4m_emb272_nopos_l6_v1",
    "SFNO (128M)": "sfno2d128m_cyl_d686_equi_nonorm_nopos_v0",
    "Pangu-Weather (32M)": "pangu32m_d216_h6-12-12-6_v1",
    r"FourCastNet $2x4$ (8M)": "fcnet8m_emb384_nopos_p2x4_l6_v2",
    r"FourCastNet $4x4$ (64M)": "fcnet64m_emb940_nopos_p4x4_l8_v0",
    "MeshGraphNet (32M)": "mgn32m_l8_d470_v0",
    "GraphCast (16M)": "gcast16m_p4_b1_d565_v2",
}


def multi_x_over_params_plot():

    # Set fontsizes for plots (10 is default)
    small_size = 10
    medium_size = 12
    bigger_size = 12
    plt.rc('font', size=small_size)          # controls default text sizes
    plt.rc('axes', titlesize=medium_size)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium_size+1)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=small_size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=medium_size)    # legend fontsize
    plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title

    #bbox_to_anchor = (0.5, -0.40)
    #bbox_to_anchor = (0.5, -0.45)
    #bbox_to_anchor = (0.4, -0.53)
    bbox_to_anchor = (0.5, -0.70)

    #fig_title = "rmse_over_params_long_rollout"
    #n = 1
    #fig, axs = plt.subplots(1, n, figsize=(n*6.1, 3))
    #ax = rmse_over_params_plot_long_rollout_physical_soundness(analysis="11-12", ax=axs, leftmost=True)
    ##ax.legend(bbox_to_anchor=(-0.017, -0.47, 1, 0.2), loc="lower left", ncol=6, handlelength=3.6)
    #ax.legend(loc='lower center', bbox_to_anchor=bbox_to_anchor, ncol=4)
    ##ax.set_ylim([800, 10000])
    #os.makedirs("plots/x_over_params/", exist_ok=True)
    #plt.savefig(f"plots/x_over_params/{fig_title}.pdf", bbox_inches="tight")
    #plt.savefig(f"plots/x_over_params/{fig_title}.png", bbox_inches="tight")
    ##plt.tight_layout()
    #plt.show()
    #plt.close()
    #exit()

    fig_title = "rmse_over_params_winds"
    n = 3
    #fig, axs = plt.subplots(1, n, figsize=(n*6.1, 3))
    fig, axs = plt.subplots(1, n, figsize=(n*4.8, 2.2))
    #__ = rmse_over_params_plot_long_rollout_physical_soundness(analysis="11-12", ax=axs[0], leftmost=True, metric="rmse")
    __ = rmse_over_params_plot_long_rollout_physical_soundness(analysis="01-12_trade-winds", ax=axs[0], leftmost=True)
    ax = rmse_over_params_plot_long_rollout_physical_soundness(analysis="01-12_south-westerlies", ax=axs[1])
    __ = rmse_over_params_plot_long_rollout_physical_soundness(analysis="01-12_global", ax=axs[2])
    #ax.legend(bbox_to_anchor=(-0.017, -0.47, 1, 0.2), loc="lower left", ncol=6, handlelength=3.6)
    #ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.85), ncol=5)
    ax.legend(loc='lower center', bbox_to_anchor=(0.43, -0.85), ncol=5, handlelength=2.3)
    os.makedirs("plots/x_over_params/", exist_ok=True)
    plt.savefig(f"plots/x_over_params/{fig_title}.pdf", bbox_inches="tight")
    plt.savefig(f"plots/x_over_params/{fig_title}.png", bbox_inches="tight")
    plt.show()
    plt.close()
    exit()

    #fig_title = "rmse_over_params_365d"
    #n = 3
    #fig, axs = plt.subplots(1, n, figsize=(n*6.1, 3))
    #__ = metric_over_params_plot(lead_time="14D", ax=axs[0], leftmost=True, metric="rmse")
    #ax = metric_over_params_plot(lead_time="172D", ax=axs[1], metric="rmse")
    #__ = metric_over_params_plot(lead_time="365D", ax=axs[2], metric="rmse")
    ##ax.legend(bbox_to_anchor=(-0.017, -0.47, 1, 0.2), loc="lower left", ncol=6, handlelength=3.6)
    #ax.legend(loc='lower center', bbox_to_anchor=bbox_to_anchor, ncol=4)
    #os.makedirs("plots/x_over_params/", exist_ok=True)
    #plt.savefig(f"plots/x_over_params/{fig_title}.pdf", bbox_inches="tight")
    #plt.savefig(f"plots/x_over_params/{fig_title}.png", bbox_inches="tight")
    #plt.show()
    #plt.close()
    #exit()
    
    # RMSE plot
    fig_title = "rmse_over_params"
    n = 3
    #fig, axs = plt.subplots(1, n, figsize=(n*6.1, 3))
    fig, axs = plt.subplots(1, n, figsize=(n*4.8, 2.2))
    __ = metric_over_params_plot(lead_time="3D", ax=axs[0], leftmost=True, metric="rmse")
    ax = metric_over_params_plot(lead_time="5D", ax=axs[1], metric="rmse")
    __ = metric_over_params_plot(lead_time="7D", ax=axs[2], metric="rmse")
    ax.legend(loc='lower center', bbox_to_anchor=bbox_to_anchor, ncol=4)
    #ax.legend(loc='lower center', bbox_to_anchor=bbox_to_anchor, ncol=4, handlelength=3.5)
    #ax.legend(bbox_to_anchor=(0.5, -0.85), loc="lower center", ncol=4)
    #plt.subplots_adjust(wspace=0.3)
    os.makedirs("plots/x_over_params/", exist_ok=True)
    plt.savefig(f"plots/x_over_params/{fig_title}.pdf", bbox_inches="tight")
    plt.savefig(f"plots/x_over_params/{fig_title}.png", bbox_inches="tight")
    plt.show()
    plt.close()

    # ACC plot
    fig_title = "acc_over_params"
    n = 3
    fig, axs = plt.subplots(1, n, figsize=(n*4.8, 2.2))
    __ = metric_over_params_plot(lead_time="3D", ax=axs[0], leftmost=True, metric="acc")
    ax = metric_over_params_plot(lead_time="5D", ax=axs[1], metric="acc")
    __ = metric_over_params_plot(lead_time="7D", ax=axs[2], metric="acc")
    #ax.legend(bbox_to_anchor=bbox_to_anchor, loc="lower center", ncol=7)
    plt.savefig(f"plots/x_over_params/{fig_title}.pdf", bbox_inches="tight")
    plt.savefig(f"plots/x_over_params/{fig_title}.png", bbox_inches="tight")
    plt.show()
    plt.close()

    # RMSE@d5, memory, runtime plot
    fig_title = "rmse5_mem_sec_over_params"
    n = 3
    fig, axs = plt.subplots(1, n, figsize=(n*4.8, 2.2))
    __ = metric_over_params_plot(lead_time="5D", ax=axs[0], leftmost=True, metric="rmse")
    ax = memory_over_params_plot(ax=axs[1])
    __ = runtime_over_params_plot(ax=axs[2])
    #ax.legend(bbox_to_anchor=(-0.017, -0.47, 1, 0.2), loc="lower left", ncol=6)
    #ax.legend(bbox_to_anchor=(0.4, -0.47, 1, 0.2), loc="lower left", ncol=6)
    #ax.legend(bbox_to_anchor=(0.5, -0.40, 1, 0.2), loc="lower left", ncol=6)
    #ax.legend(bbox_to_anchor=bbox_to_anchor, loc="lower center", ncol=7)
    ax.legend(bbox_to_anchor=(0.5, -0.7), loc="lower center", ncol=5)
    plt.savefig(f"plots/x_over_params/{fig_title}.pdf", bbox_inches="tight")
    plt.savefig(f"plots/x_over_params/{fig_title}.png", bbox_inches="tight")
    plt.show()
    plt.close()


def metric_over_params_plot(lead_time, ax, leftmost: bool = False, metric: str = "rmse"):

    print("\n####################################")
    print(f"### {metric.upper()} results for {lead_time} lead time ###")
    print("####################################\n")

    vname = "z500"
    params = np.array(["50k", "500k", "1M", "2M", "4M", "8M", "16M", "32M", "64M", "128M"])
    model_names_dict = {
        
        #"ConvLSTM": {"ls": "solid", "c": "yellowgreen", "clipped": None, "alpha": 1.0, "fname": "clstm*cyl_4x*v0"},
        #"ConvLSTM1": {"ls": "solid", "c": "blue", "clipped": None, "alpha": 0.8, "fname": "clstm*cyl*v1"},
        #"ConvLSTM2": {"ls": "solid", "c": "red", "clipped": None, "alpha": 0.6, "fname": "clstm*cyl*v2"},
        #"ConvLSTM HPX8": {"ls": "dashed", "c": "yellowgreen", "clipped": None, "alpha": 1.0, "fname": "clstm*hpx8_4*v0"},
        #"ConvLSTM HPX8 1": {"ls": "dashed", "c": "red", "clipped": None, "alpha": 1.0, "fname": "clstm*hpx8_4*v1"},
        #"ConvLSTM HPX8 2": {"ls": "dashed", "c": "blue", "clipped": None, "alpha": 1.0, "fname": "clstm*hpx8_4*v2"},
        
        #"U-Net": {"ls": "solid", "c": "darkgreen", "clipped": None, "alpha": 1.0, "fname": "unet*cyl*v0"},
        #"U-Net1": {"ls": "solid", "c": "blue", "clipped": None, "alpha": 0.8, "fname": "unet*cyl*v1"},
        #"U-Net2": {"ls": "solid", "c": "red", "clipped": None, "alpha": 0.6, "fname": "unet*cyl*v2"},
        #"U-Net HPX8": {"ls": "dashed", "c": "darkgreen", "clipped": None, "alpha": 1.0, "fname": "unet*hpx8*v0"},
        #"U-Net HPX8 1": {"ls": "dashed", "c": "blue", "clipped": None, "alpha": 0.8, "fname": "unet*hpx8*v1"},
        #"U-Net HPX8 2": {"ls": "dashed", "c": "red", "clipped": None, "alpha": 0.6, "fname": "unet*hpx8*v2"},
        #"U-Net HPX16": {"ls": "dotted", "c": "darkgreen", "clipped": None, "alpha": 0.7, "fname": "unet*hpx16*v0"},

        #"SwinTransformer": {"ls": "solid", "c": "darkorange", "clipped": None, "alpha": 1.0, "fname": "swint*cyl*v0"},
        #"SwinTransformer1": {"ls": "solid", "c": "green", "clipped": None, "alpha": 1.0, "fname": "swint*cyl*v1"},
        #"SwinTransformer2": {"ls": "solid", "c": "blue", "clipped": None, "alpha": 1.0, "fname": "swint*cyl*v2"},
        #"SwinTransformer HPX8": {"ls": "dashed", "c": "darkorange", "clipped": None, "alpha": 1.0, "ls": "dashed", "fname": "swint*hpx8*v0"},
        #"SwinTransformer HPX8 1": {"ls": "dashed", "c": "green", "clipped": None, "alpha": 1.0, "fname": "swint*hpx8*v1"},
        #"SwinTransformer HPX8 2": {"ls": "dashed", "c": "blue", "clipped": None, "alpha": 1.0, "fname": "swint*hpx8*v2"},
        
        #"Pangu-Weather": {"ls": "solid", "c": "deepskyblue", "clipped": None, "alpha": 1.0, "fname": "pangu*v0"},
        #"Pangu-Weather2": {"ls": "solid", "c": "green", "clipped": None, "alpha": 1.0, "fname": "pangu*v1"},
        #"Pangu-Weather3": {"ls": "solid", "c": "red", "clipped": None, "alpha": 1.0, "fname": "pangu*v2"},

        #"FNO2D": {"ls": "solid", "c": "lightcoral", "clipped": None, "alpha": 1.0, "fname": "fno2d*v0"},
        #"FNO2D1": {"ls": "solid", "c": "green", "clipped": None, "alpha": 1.0, "fname": "fno2d*v1"},
        #"FNO2D2": {"ls": "solid", "c": "red", "clipped": None, "alpha": 1.0, "fname": "fno2d*v2"},
        
        #"TFNO2D": {"ls": "solid", "c": "darkturquoise", "clipped": None, "alpha": 1.0, "fname": "tfno2d*v0"},
        #"TFNO2D1": {"ls": "solid", "c": "green", "clipped": None, "alpha": 1.0, "fname": "tfno2d*v1"},
        #"TFNO2D2": {"ls": "solid", "c": "red", "clipped": None, "alpha": 1.0, "fname": "tfno2d*v2"},
        
        "FourCastNet (AFNO) p1x1": {"ls": "solid", "c": "firebrick", "clipped": None, "alpha": 1.0, "fname": "fcnet*nopos_l*v0"},
        #"FourCastNet (AFNO) p1x1 1": {"ls": "solid", "c": "green", "clipped": None, "alpha": 1.0, "fname": "fcnet*nopos_l*v1"},
        #"FourCastNet (AFNO) p1x1 2": {"ls": "solid", "c": "blue", "clipped": None, "alpha": 1.0, "fname": "fcnet*nopos_l*v2"},

        "FourCastNet (AFNO) p1x2": {"ls": "solid", "c": "green", "clipped": None, "alpha": 1.0, "fname": "fcnet*nopos_p1x2*v0"},

        "FourCastNet (AFNO) p2x2": {"ls": "solid", "c": "pink", "clipped": None, "alpha": 1.0, "fname": "fcnet*nopos_p2x2*v0"},
        
        "FourCastNet (AFNO) p2x4": {"ls": "solid", "c": "goldenrod", "clipped": None, "alpha": 1.0, "fname": "fcnet*nopos_p2x4*v0"},
        #"FourCastNet (AFNO) p2x4 1": {"ls": "solid", "c": "green", "clipped": None, "alpha": 1.0, "fname": "fcnet*nopos_p2x4*v1"},
        #"FourCastNet (AFNO) p2x4 2": {"ls": "solid", "c": "blue", "clipped": None, "alpha": 1.0, "fname": "fcnet*nopos_p2x4*v2"},

        "FourCastNet (AFNO) p4x2": {"ls": "solid", "c": "cyan", "clipped": None, "alpha": 1.0, "fname": "fcnet*nopos_p4x2*v0"},
        
        "FourCastNet (AFNO) p4x4": {"ls": "solid", "c": "orangered", "clipped": None, "alpha": 1.0, "fname": "fcnet*nopos_p4x4*v0"},
        #"FourCastNet (AFNO) p4x4 1": {"ls": "solid", "c": "green", "clipped": None, "alpha": 1.0, "fname": "fcnet*nopos_p4x4*v1"},
        #"FourCastNet (AFNO) p4x4 2": {"ls": "solid", "c": "blue", "clipped": None, "alpha": 1.0, "fname": "fcnet*nopos_p4x4*v2"},

        "FourCastNet (AFNO) p4x8": {"ls": "solid", "c": "turquoise", "clipped": None, "alpha": 1.0, "fname": "fcnet*nopos_p4x8*v0"},

        #"FourCastNet (FNO2D) p1x1": {"ls": "solid", "c": "limegreen", "clipped": None, "alpha": 1.0, "fname": "fcv0net*nopos*v0"},

        #"FourCastNet (SFNO) p1x1": {"ls": "solid", "c": "plum", "clipped": None, "alpha": 1.0, "fname": "fcv2net*v0"},
        
        #"SFNO2D": {"ls": "solid", "c": "steelblue", "clipped": None, "alpha": 1.0, "fname": "sfno2d*equi_nonorm_nopos_v0"},
        #"SFNO2D1": {"ls": "solid", "c": "green", "clipped": None, "alpha": 1.0, "fname": "sfno2d*equi_nonorm_nopos_v1"},
        #"SFNO2D2": {"ls": "solid", "c": "red", "clipped": None, "alpha": 1.0, "fname": "sfno2d*equi_nonorm_nopos_v2"},
        
        #"SFNO2D": {"ls": "solid", "c": "steelblue", "clipped": None, "alpha": 1.0, "fname": "sfno2d*lre3e_equi_nonorm_nopos_v0"},
        
        #"MeshGraphNet":{"ls": "solid", "c": "blueviolet", "clipped": None, "alpha": 1.0, "fname": "mgn*v0"},
        #"MeshGraphNet1":{"ls": "solid", "c": "green", "clipped": None, "alpha": 1.0, "fname": "mgn*v1"},
        #"MeshGraphNet2":{"ls": "solid", "c": "red", "clipped": None, "alpha": 1.0, "fname": "mgn*v2"},
        
        #"GraphCast": {"ls": "solid", "c": "dodgerblue", "clipped": None, "alpha": 1.0, "fname": "gcast*p4_d*v0"},
        #"GraphCast1": {"ls": "solid", "c": "green", "clipped": None, "alpha": 1.0, "fname": "gcast*p4*v1"},
        #"GraphCast2": {"ls": "solid", "c": "red", "clipped": None, "alpha": 1.0, "fname": "gcast*p4*v2"},

        #"GraphCast": {"ls": "solid", "c": "darkblue", "clipped": None, "alpha": 1.0, "fname": "gcast*p4_b*v0"},
        #"GraphCast B2": {"ls": "solid", "c": "green", "clipped": None, "alpha": 1.0, "fname": "gcast*p4_b*v1"},
        #"GraphCast B3": {"ls": "solid", "c": "red", "clipped": None, "alpha": 1.0, "fname": "gcast*p4_b*v2"},


        "Persistence": {"ls": "solid", "c": "dimgray", "clipped": None, "fname": "persistence"},
        "Climatology": {"ls": "solid", "c": "darkgrey", "clipped": None, "alpha": 0.6, "fname": "climatology"},
    }

    p_low = np.array([s.lower() for s in params])
    for m_idx, model_name in enumerate(model_names_dict):
        if "Climatology" in model_name and metric == "acc": continue  # ACC is relative to climatology
        hasnan = np.zeros(len(params))
        scores = np.zeros(len(params))*np.nan
        stds = np.zeros(len(params))*np.nan
        fnames = np.sort(glob.glob(os.path.join("outputs", model_names_dict[model_name]["fname"])))
        for fname in fnames:
            if "persistence" in fname or "climatology" in fname:
                metric_file_path = os.path.join(fname, "evaluation", f"{metric}s.nc")
                scores[:] = xr.open_dataset(metric_file_path)[vname].sel(time=lead_time).values
            else:
                if "unet" in fname and "cyl" in fname and fname.count("-") != 4: continue
                #"""
                # Load scores of metric (rmse or acc) and compute average over v0, v1, v2 if they exist
                fnames_vx = [fname[:-1] + n for n in ["0", "1", "2"]]
                scores_vx = []
                for fname_vx in fnames_vx:
                    metric_file_path = os.path.join(fname_vx, "evaluation", f"{metric}s.nc")
                    if not os.path.exists(metric_file_path): continue
                    score = xr.open_dataset(metric_file_path)[vname].sel(time=lead_time).values
                    scores_vx.append(score)
                scores_vx = np.array(scores_vx)
                # Get parameter count of the model and insert metric score at the according position in the scores list
                nparams = re.search(r"\d{1,3}[k,m]", fname).group()
                idx = np.where(p_low == nparams)[0][0]
                if len(scores_vx) > 0 and scores_vx.max() > 5000: continue  # Ignore outliers
                scores[idx], stds[idx] = np.mean(scores_vx), np.std(scores_vx)
                if np.isnan(scores_vx).any() or len(scores_vx) != 3: hasnan[idx] = 1
                print(fnames_vx, scores_vx)
                #print(scores_vx)
                """ 
                # Load scores of metric (rmse or acc) and compute average over v0, v1, v2 if they exist
                metric_file_path = os.path.join(fname, "evaluation", f"{metric}s.nc")
                if not os.path.exists(metric_file_path): continue
                score = xr.open_dataset(metric_file_path)[vname].sel(time=lead_time).values
                #if "hpx" in fname: score -= 330.9664115969075
                # Get parameter count of the model and insert metric score at the according position in the scores list
                nparams = re.search(r"\d{1,3}[k,m]", fname).group()
                idx = np.where(p_low == nparams)[0][0]
                scores[idx] = score
                #"""

        print(model_name, fname)
        scores, stds = np.round(scores, 2), np.round(stds, 2)
        metric_table_string = " & ".join([f"${scores[s_idx]}\pm{stds[s_idx]}$" for s_idx in range(len(scores))])
        print(" &", metric_table_string, "\\\\\n")

        entry = model_names_dict[model_name]
        alpha = entry["alpha"] if "alpha" in entry.keys() else 1.0
        ax.plot(params, scores, label=model_name, ls=entry["ls"], c=entry["c"], marker="o", lw=2.5, markersize=5, alpha=alpha)
        ax.fill_between(params, scores-stds, scores+stds, facecolor=entry["c"], alpha=0.2*alpha)

        # Mark entries that exceed the threshold (do not consist of 3 measurements)
        broken_idcs = np.logical_and(np.invert(np.isnan(scores)), hasnan == 1.0)
        ax.scatter(params[broken_idcs], scores[broken_idcs], c=entry["c"], marker="d", s=50, zorder=2)

        ax.set_xticklabels(params)
        ax.set_xlabel("#parameters")
        if leftmost: ax.set_ylabel(metric.upper())
        if metric == "rmse": ax.set_yscale("log")
        elif metric == "acc": ax.set_ylim([0.1, 1.0])
        ax.grid(visible=True, which='minor', color='silver')
        ax.grid(visible=True, which='major', color='grey')
        ax.set_title(f"{lead_time[:-1]} days lead-time")
    
    return ax


def rmse_over_params_plot_long_rollout_physical_soundness(analysis, ax, leftmost: bool = False):
    # Function argument "analysis" can be any of
    #   o "11-12" (for long rollout evaluation)
    #   o "01-12_global" (for global physical soundness evaluation)
    #   o "01-12_south-westerlies" (for physical soundness evaluation on South Westerlies)
    #   o "01-12_trade-winds" (for physical soundness evaluation on Trade Winds)

    print("\n####################################")
    print(f"### RMSE results for {analysis} ###")
    print("####################################\n")

    if analysis == "11-12": title = ""
    elif analysis == "01-12_global": title = "Global"
    elif analysis == "01-12_south-westerlies": title = "South Westerlies"
    elif analysis == "01-12_trade-winds": title = "Trade Winds"
    vname = "u10" if "01-12_" in analysis else "z500"
    thrsh = 5000 if vname == "z500" else 100
    params = np.array(["50k", "500k", "1M", "2M", "4M", "8M", "16M", "32M", "64M", "128M"])
    model_names_dict = {
        "Persistence": {"ls": "solid", "c": "dimgray", "clipped": None, "fname": "persistence"},
        "Climatology": {"ls": "solid", "c": "darkgrey", "clipped": None, "alpha": 0.6, "fname": "climatology"},
        
        "ConvLSTM": {"ls": "solid", "c": "yellowgreen", "clipped": None, "alpha": 1.0, "fname": "clstm*cyl_4x*v0"},
        #"ConvLSTM1": {"ls": "solid", "c": "blue", "clipped": None, "alpha": 0.8, "fname": "clstm*cyl_4x*v1"},
        #"ConvLSTM2": {"ls": "solid", "c": "red", "clipped": None, "alpha": 0.6, "fname": "clstm*cyl_4x*v2"},
        "ConvLSTM HPX8": {"ls": "dashed", "c": "yellowgreen", "clipped": None, "alpha": 1.0, "fname": "clstm*hpx8_4*v0"},
        #"ConvLSTM HPX8 1": {"ls": "dashed", "c": "red", "clipped": None, "alpha": 1.0, "fname": "clstm*hpx8_4*v1"},
        #"ConvLSTM HPX8 2": {"ls": "dashed", "c": "blue", "clipped": None, "alpha": 1.0, "fname": "clstm*hpx8_4*v2"},
        
        "U-Net": {"ls": "solid", "c": "darkgreen", "clipped": None, "fname": "unet*cyl*v0"},
        #"U-Net1": {"ls": "solid", "c": "blue", "clipped": None, "fname": "unet*cyl*v1"},
        #"U-Net2": {"ls": "solid", "c": "red", "clipped": None, "fname": "unet*cyl*v2"},
        "U-Net HPX8": {"ls": "dashed", "c": "darkgreen", "clipped": None, "alpha": 1.0, "fname": "unet*hpx8*v0"},
        #"U-Net HPX8 1": {"ls": "dashed", "c": "blue", "clipped": None, "alpha": 0.8, "fname": "unet*hpx8*v1"},
        #"U-Net HPX8 2": {"ls": "dashed", "c": "red", "clipped": None, "alpha": 0.6, "fname": "unet*hpx8*v2"},
        #"U-Net HPX16": {"ls": "dotted", "c": "darkgreen", "clipped": None, "alpha": 0.7, "fname": "unet*hpx16*v0"},

        "SwinTransformer": {"ls": "solid", "c": "darkorange", "clipped": None, "alpha": 1.0, "fname": "swint*cyl*v0"},
        #"SwinTransformer1": {"ls": "solid", "c": "green", "clipped": None, "alpha": 1.0, "fname": "swint*cyl*v1"},
        #"SwinTransformer2": {"ls": "solid", "c": "blue", "clipped": None, "alpha": 1.0, "fname": "swint*cyl*v2"},
        "SwinTransformer HPX8": {"ls": "dashed", "c": "darkorange", "clipped": None, "alpha": 1.0, "ls": "dashed", "fname": "swint*hpx8*v0"},
        #"SwinTransformer HPX8 1": {"ls": "dashed", "c": "green", "clipped": None, "alpha": 1.0, "fname": "swint*hpx8*v1"},
        #"SwinTransformer HPX8 2": {"ls": "dashed", "c": "blue", "clipped": None, "alpha": 1.0, "fname": "swint*hpx8*v2"},
        
        "Pangu-Weather": {"ls": "solid", "c": "deepskyblue", "clipped": None, "alpha": 1.0, "fname": "pangu*v0"},
        #"Pangu-Weather2": {"ls": "solid", "c": "green", "clipped": None, "alpha": 1.0, "fname": "pangu*v1"},
        #"Pangu-Weather3": {"ls": "solid", "c": "red", "clipped": None, "alpha": 1.0, "fname": "pangu*v2"},

        "FNO2D": {"ls": "solid", "c": "lightcoral", "clipped": None, "alpha": 1.0, "fname": "fno2d*v0"},
        #"FNO2D1": {"ls": "solid", "c": "green", "clipped": None, "alpha": 1.0, "fname": "fno2d*v1"},
        #"FNO2D2": {"ls": "solid", "c": "red", "clipped": None, "alpha": 1.0, "fname": "fno2d*v2"},
        
        "TFNO2D": {"ls": "solid", "c": "darkturquoise", "clipped": None, "alpha": 1.0, "fname": "tfno2d*v0"},
        #"TFNO2D1": {"ls": "solid", "c": "green", "clipped": None, "alpha": 1.0, "fname": "tfno2d*v1"},
        #"TFNO2D2": {"ls": "solid", "c": "red", "clipped": None, "alpha": 1.0, "fname": "tfno2d*v2"},
        
        "FourCastNet (AFNO) p1x1": {"ls": "solid", "c": "firebrick", "clipped": None, "alpha": 1.0, "fname": "fcnet*nopos_l*v0"},
        #"FourCastNet (AFNO) p1x1 1": {"ls": "solid", "c": "green", "clipped": None, "alpha": 1.0, "fname": "fcnet*nopos_l*v1"},
        #"FourCastNet (AFNO) p1x1 2": {"ls": "solid", "c": "blue", "clipped": None, "alpha": 1.0, "fname": "fcnet*nopos_l*v2"},

        #"FourCastNet (AFNO) p1x2": {"ls": "solid", "c": "green", "clipped": None, "alpha": 1.0, "fname": "fcnet*nopos_p1x2*v0"},

        #"FourCastNet (AFNO) p2x2": {"ls": "solid", "c": "pink", "clipped": None, "alpha": 1.0, "fname": "fcnet*nopos_p2x2*v0"},
        
        #"FourCastNet (AFNO) p2x4": {"ls": "solid", "c": "goldenrod", "clipped": None, "alpha": 1.0, "fname": "fcnet*nopos_p2x4*v0"},
        #"FourCastNet (AFNO) p2x4 1": {"ls": "solid", "c": "green", "clipped": None, "alpha": 1.0, "fname": "fcnet*nopos_p2x4*v1"},
        #"FourCastNet (AFNO) p2x4 2": {"ls": "solid", "c": "blue", "clipped": None, "alpha": 1.0, "fname": "fcnet*nopos_p2x4*v2"},

        #"FourCastNet (AFNO) p4x2": {"ls": "solid", "c": "cyan", "clipped": None, "alpha": 1.0, "fname": "fcnet*nopos_p4x2*v0"},
        
        #"FourCastNet (AFNO) p4x4": {"ls": "solid", "c": "orangered", "clipped": None, "alpha": 1.0, "fname": "fcnet*nopos_p4x4*v0"},
        #"FourCastNet (AFNO) p4x4 1": {"ls": "solid", "c": "green", "clipped": None, "alpha": 1.0, "fname": "fcnet*nopos_p4x4*v1"},
        #"FourCastNet (AFNO) p4x4 2": {"ls": "solid", "c": "blue", "clipped": None, "alpha": 1.0, "fname": "fcnet*nopos_p4x4*v2"},

        #"FourCastNet (AFNO) p4x8": {"ls": "solid", "c": "turquoise", "clipped": None, "alpha": 1.0, "fname": "fcnet*nopos_p4x8*v0"},

        #"FourCastNet (FNO2D) p1x1": {"ls": "solid", "c": "limegreen", "clipped": None, "alpha": 1.0, "fname": "fcv0net*nopos*v0"},

        #"FourCastNet (SFNO) p1x1": {"ls": "solid", "c": "plum", "clipped": None, "alpha": 1.0, "fname": "fcv2net*v0"},
        
        #"SFNO2D": {"ls": "solid", "c": "steelblue", "clipped": None, "alpha": 1.0, "fname": "sfno2d*equi_nonorm_nopos_v0"},
        #"SFNO2D1": {"ls": "solid", "c": "green", "clipped": None, "alpha": 1.0, "fname": "sfno2d*equi_nonorm_nopos_v1"},
        #"SFNO2D2": {"ls": "solid", "c": "red", "clipped": None, "alpha": 1.0, "fname": "sfno2d*equi_nonorm_nopos_v2"},
        
        "SFNO2D": {"ls": "solid", "c": "steelblue", "clipped": None, "alpha": 1.0, "fname": "sfno2d*lre3e_equi_nonorm_nopos_v0"},
        
        "MeshGraphNet":{"ls": "solid", "c": "blueviolet", "clipped": None, "alpha": 1.0, "fname": "mgn*v0"},
        #"MeshGraphNet1":{"ls": "solid", "c": "green", "clipped": None, "alpha": 1.0, "fname": "mgn*v1"},
        #"MeshGraphNet2":{"ls": "solid", "c": "red", "clipped": None, "alpha": 1.0, "fname": "mgn*v2"},
        
        #"GraphCast": {"ls": "solid", "c": "dodgerblue", "clipped": None, "alpha": 1.0, "fname": "gcast*p4_d*v0"},
        #"GraphCast1": {"ls": "solid", "c": "green", "clipped": None, "alpha": 1.0, "fname": "gcast*p4*v1"},
        #"GraphCast2": {"ls": "solid", "c": "red", "clipped": None, "alpha": 1.0, "fname": "gcast*p4*v2"},

        "GraphCast": {"ls": "solid", "c": "darkblue", "clipped": None, "alpha": 1.0, "fname": "gcast*p4_b*v0"},
        #"GraphCast B2": {"ls": "solid", "c": "green", "clipped": None, "alpha": 1.0, "fname": "gcast*p4_b*v1"},
        #"GraphCast B3": {"ls": "solid", "c": "red", "clipped": None, "alpha": 1.0, "fname": "gcast*p4_b*v2"},
    }

    p_low = np.array([s.lower() for s in params])
    for m_idx, model_name in enumerate(model_names_dict):
        hasnan = np.zeros(len(params))
        scores = np.zeros(len(params))*np.nan
        stds = np.zeros(len(params))*np.nan
        fnames = np.sort(glob.glob(os.path.join("outputs", model_names_dict[model_name]["fname"])))
        for fname in fnames:
            if "persistence" in fname or "climatology" in fname:
                metric_file_path = os.path.join(fname, "evaluation", f"rmse_months_{analysis}.nc")
                scores[:] = xr.open_dataset(metric_file_path)[vname].values
            else:
                if "unet" in fname and "cyl" in fname and fname.count("-") != 4: continue
                #"""
                # Load scores of metric (rmse or acc) and compute average over v0, v1, v2 if they exist
                fnames_vx = [fname[:-1] + n for n in ["0", "1", "2"]]
                scores_vx = []
                for fname_vx in fnames_vx:
                    metric_file_path = os.path.join(fname_vx, "evaluation", f"rmse_months_{analysis}.nc")
                    if not os.path.exists(metric_file_path): continue
                    score = xr.open_dataset(metric_file_path)[vname].values
                    if score > thrsh: continue  # Ignore outliers
                    scores_vx.append(score)
                    #print(fname_vx, score)
                scores_vx = np.array(scores_vx)
                # Get parameter count of the model and insert metric score at the according position in the scores list
                nparams = re.search(r"\d{1,3}[k,m]", fname).group()
                idx = np.where(p_low == nparams)[0][0]
                #if len(scores_vx) > 0 and scores_vx.max() > thrsh: continue  # Ignore outliers
                scores[idx], stds[idx] = np.nanmean(scores_vx), np.nanstd(scores_vx)
                if np.isnan(scores_vx).any() or len(scores_vx) != 3: hasnan[idx] = 1
                """ 
                # Load scores of metric (rmse or acc) and compute average over v0, v1, v2 if they exist
                metric_file_path = os.path.join(fname, "evaluation", f"rmse_months_{analysis}.nc")
                if not os.path.exists(metric_file_path): continue
                score = xr.open_dataset(metric_file_path)[vname].values
                # Get parameter count of the model and insert metric score at the according position in the scores list
                nparams = re.search(r"\d{1,3}[k,m]", fname).group()
                idx = np.where(p_low == nparams)[0][0]
                scores[idx] = score
                #"""

        print(model_name, fname)
        scores, stds = np.round(scores, 2), np.round(stds, 2)
        metric_table_string = " & ".join([f"${scores[s_idx]}\pm{stds[s_idx]}$" for s_idx in range(len(scores))])
        print(" &", metric_table_string, "\\\\\n")

        entry = model_names_dict[model_name]
        alpha = entry["alpha"] if "alpha" in entry.keys() else 1.0
        ax.plot(params, scores, label=model_name, ls=entry["ls"], c=entry["c"], marker="o", lw=2.5, markersize=5, alpha=alpha)
        ax.fill_between(params, scores-0, scores+stds, facecolor=entry["c"], alpha=0.2*alpha)

        # Mark entries that exceed the threshold (do not consist of 3 measurements)
        broken_idcs = np.logical_and(np.invert(np.isnan(scores)), hasnan == 1.0)
        ax.scatter(params[broken_idcs], scores[broken_idcs], c=entry["c"], marker="d", s=50, zorder=2)

        ax.set_xticklabels(params)
        ax.set_xlabel("#parameters")
        if leftmost: ax.set_ylabel("RMSE")
        ax.set_yscale("log")
        ax.grid(visible=True, which='minor', color='silver')
        ax.grid(visible=True, which='major', color='grey')
        ax.set_title(f"{title}")
    
    return ax


def memory_over_params_plot(ax):

    params = np.array(["50k", "500k", "1M", "2M", "4M", "8M", "16M", "32M", "64M", "128M"])
    model_names_dict = {
        "ConvLSTM": {"c": "yellowgreen", "data": np.array([370, 420, 460, 516, 614, 770, 1060, 1576, 2490, np.nan])},
        #"ConvLSTM HPX": {"c": "yellowgreen", "data": np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])},
        "U-Net": {"c": "darkgreen", "data": np.array([350, 360, 366, 388, 426, 518, 672, 1038, 1584, 2902])},
        #"U-Net HPX": {"c": "darkgreen", "data": np.array([328, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])},
        "SwinTransformer": {"c": "darkorange", "data": np.array([1324, 2062, 2100, 2246, 2374, 2328, 2376, 2772, np.nan, np.nan])},
        #"SwinTransformer HPX": {"c": "darkorange", "data": np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])},
        "Pangu-Weather": {"c": "deepskyblue", "data": np.array([np.nan, 672, 832, 1240, 1726, 2098, 2516, 3412, 4466, np.nan])},
        #"FNO2D": {"c": "lightcoral", "data": np.array([410, 420, 434, 466, 490, 604, 712, 1046, 1702, 2868])},
        #"TFNO2D": {"c": "darkturquoise", "data": np.array([412, 432, 450, 474, 544, 642, 862, 1274, 2418, 4236])},
        r"FourCastNet $p=1x1$": {"c": "firebrick", "data": np.array([438, 570, 748, 920, 1314, 1784, 2660, 3642, 5156, 7988])},
        #r"FourCastNet $p=2\times 4$": {"c": "goldenrod", "data": np.array([376, 398, 426, 458, 552, 644, 870, 1434, 1928, 3506])},
        #r"FourCastNet $p=4x4$": {"c": "orangered", "data": np.array([372, 386, 404, 430, 508, 616, 836, 1306, 1814, 3190])},
        "SFNO": {"c": "steelblue", "data": np.array([378, 408, 426, 484, 524, 618, 820, 1156, 1780, 3212])},
        "MeshGraphNet": {"c": "blueviolet", "data": np.array([572, 1066, 1346, 1798, 2326, 3124, 4388, 6100, np.nan, np.nan])},
        "GraphCast": {"c": "darkblue", "data": np.array([494, 802, 998, 1270, 1674, 2238, 3138, 4328, np.nan, np.nan])},
    }
    
    # model.name=test training.batch_size=1 validation.batch_size=1 data.train_start_date=2014-01-01 data.val_start_date=2016-01-01

    for m_idx, model_name in enumerate(model_names_dict):
        entry = model_names_dict[model_name]
        ls = "--" if "HPX" in model_name else "-"
        ax.plot(params, entry["data"], label=model_name, ls=ls, c=entry["c"], marker="o", lw=2.5, markersize=5)

    ax.set_xticklabels(params)
    ax.set_xlabel("#parameters")
    ax.set_ylabel("Memory [MB]")
    #ax.set_xlim([-0.25, len(params)-0.75])
    #ax.set_ylim([5e-3, 0.7])
    ax.set_yscale("log")

    #ax.grid()
    ax.grid(visible=True, which='minor', color='silver')
    ax.grid(visible=True, which='major', color='grey')
    #ax.set_title("Experiment 1")
    #plt.legend(ncol=3, loc="lower right")

    return ax


def runtime_over_params_plot(ax):

    params = np.array(["50k", "500k", "1M", "2M", "4M", "8M", "16M", "32M", "64M", "128M"])
    model_names_dict = {
        "ConvLSTM": {"c": "yellowgreen", "data": np.array([14.39, 14.84, 14.94, 15.17, 15.25, 19.73, 28.45, 56.40, 117.45, np.nan])},
        #"ConvLSTM HPX": {"c": "yellowgreen", "data": np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])},
        "U-Net": {"c": "darkgreen", "data": np.array([20.54, 21.95, 20.94, 22.30, 21.30, 21.70, 23.02, 24.16, 25.87, 31.61])},
        #"U-Net HPX": {"c": "darkgreen", "data": np.array([222.11, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])},
        "SwinTransformer": {"c": "darkorange", "data": np.array([30.68, 57.02, 58.79, 60.17, 62.10, 69.35, 81.83, 83.61, np.nan, np.nan])},
        #"SwinTransformer HPX": {"c": "darkorange", "data": np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])},
        "Pangu-Weather": {"c": "deepskyblue", "data": np.array([np.nan, 77.36, 78.86, 79.11, 78.52, 77.07, 79.10, 81.54, 95.52, np.nan])},
        #"FNO2D": {"c": "lightcoral", "data": np.array([13.38, 13.28, 13.48, 13.78, 13.67, 13.86, 15.67, 20.43, 40.77, 83.66])},
        #"TFNO2D": {"c": "darkturquoise", "data": np.array([22.89, 22.58, 22.56, 23.07, 22.68, 22.91, 24.01, 26.28, 32.24, 48.16])},
        r"FourCastNet $p=1x1$": {"c": "firebrick", "data": np.array([19.30, 19.20, 34.64, 35.55, 52.39, 52.28, 68.68, 82.17, 127.32, 211.93])},
        #r"FourCastNet $p=2x4$": {"c": "goldenrod", "data": np.array([19.36, 19.63, 33.39, 34.15, 47.76, 50.96, 64.26, 66.03, 68.44, 69.92])},
        #r"FourCastNet $p=4x4$": {"c": "orangered", "data": np.array([19.45, 19.37, 35.25, 35.74, 50.71, 50.79, 68.22, 68.88, 70.71, 74.45])},
        "SFNO": {"c": "steelblue", "data": np.array([20.58, 19.53, 21.23, 20.72, 20.24, 21.48, 20.84, 23.27, 44.03, 79.80])},
        "MeshGraphNet": {"c": "blueviolet", "data": np.array([19.22, 19.61, 21.12, 27.75, 40.90, 60.89, 104.04, 175.79, np.nan, np.nan])},
        "GraphCast": {"c": "darkblue", "data": np.array([25.10, 25.95, 26.41, 27.15, 27.40, 37.01, 53.74, 87.32, np.nan, np.nan])},
    }

    # model.name=test training.batch_size=1 validation.batch_size=1 data.train_start_date=2014-01-01 data.val_start_date=2016-01-01

    for m_idx, model_name in enumerate(model_names_dict):
        entry = model_names_dict[model_name]
        ls = "--" if "HPX" in model_name else "-"
        ax.plot(params, entry["data"], label=model_name, ls=ls, c=entry["c"], marker="o", lw=2.5, markersize=5)

    ax.set_xticklabels(params)
    ax.set_xlabel("#parameters")
    ax.set_ylabel("Seconds per epoch")
    #ax.set_xlim([-0.25, len(params)-0.75])
    #ax.set_ylim([5e-3, 0.7])
    ax.set_yscale("log")
    #ax.yaxis.set_tick_params(which='minor', right="off")

    #ax.grid()
    ax.grid(visible=True, which='minor', color='silver')
    ax.grid(visible=True, which='major', color='grey')
    #ax.set_title("Experiment 1")
    #plt.legend()#loc="upper right")
    
    return ax


def end_conditions_plot(days: int = 96):

    sample = 0
    vname = "z500"
    scale = 98.1

    #file_and_model_names = {
    #    "Verification": "unet128m_cyl_128-256-512-1024-2014_v2",
    #    "ConvLSTM Cyl (32M)": "clstm32m_cyl_4x323_v2",
    #    #"U-Net Cyl (128M)": "unet128m_cyl_128-256-512-1024-2014_v2",
    #    #"SwinTransformer Cyl (16M)": "swint16m_cyl_d60_l4x4_h4x4_v0",
    #    #"FNO (64M)": "fno2d64m_cyl_d307_v0",
    #    "ConvLSTM HPX (16M)": "clstm16m_hpx8_4x228_v2",
    #    #"U-Net HPX (32M)": "unet32m_hpx8_130-260-520-1040_v0",
    #    #"SwinTransformer HPX (4M)": "swint4m_hpx8_d124_l2x4_h2x4_v1",
    #    #"TFNO (128M)": "tfno2d128m_cyl_d477_v2",
    #    #r"FourCastNet $p=1x1$ (16M)": "fcnet16m_emb472_nopos_l8_v0",
    #    "SFNO (64M)": "sfno2d64m_cyl_d485_equi_nonorm_nopos_v2",
    #    #"Pangu-Weather (32M)": "pangu32m_d216_h6-12-12-6_v2",
    #    #r"FourCastNet $p=2x4$ (16M)": "fcnet16m_emb472_nopos_p2x4_l8_v0",
    #    #r"FourCastNet $p=4x4$ (16M)": "fcnet16m_emb472_nopos_p4x4_l8_v2",
    #    #"MeshGraphNet (500k)": "mgn500k_l4_d116_v2",
    #    #"GraphCast (2M)": "gcast2m_p4_b1_d199_v2",
    #}
    fig, axs = plt.subplots(4, 4, figsize=(10, 6), sharex=True, sharey=True,
                            subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
    #fig, axs = plt.subplots(1, 4, figsize=(12, 2), sharex=True, sharey=True,
    #                        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
    extent = [-180, 180, -90, 90]

    for m_idx, model_name in enumerate(FILE_AND_MODEL_NAMES):
        # Load data
        base_path = os.path.join("outputs", FILE_AND_MODEL_NAMES[model_name], "evaluation")

        opts = xr.open_dataset(
            os.path.join(base_path, "outputs.nc")
        )[vname].isel(sample=sample).sel(time=pd.Timedelta(f"{days}D")).values/scale
        tgts = xr.open_dataset(
            os.path.join("outputs", "clstm1m_cyl_4x57_v0", "evaluation", "targets.nc")
        )[vname].isel(sample=sample).sel(time=pd.Timedelta(f"{days}D")).values/scale


        #tgts = xr.open_dataset(
        #    os.path.join(base_path, "targets.nc")
        #)[vname].isel(sample=sample).sel(time=pd.Timedelta(f"{days}D")).values/scale

        row = m_idx // 4
        col = m_idx % 4

        vmin, vmax = tgts.min(), tgts.max()
        ax = axs[row, col]
        #ax = axs[col]
        ax.coastlines()
        im = ax.imshow(tgts if m_idx == 0 else opts, origin="lower", vmin=vmin, vmax=vmax, extent=extent)
        ax.set_title(model_name)

    # Colorbar for single ax
    #div_pred = make_axes_locatable(axs[4, -1])
    #cax_pred = div_pred.append_axes("right", size="5%", pad=0.05)
    #fig.colorbar(im_pred, cax=cax_pred, orientation='vertical')

    # Colorbar for all subplots
    #'''
    fig.subplots_adjust(right=0.95)
    cbax = fig.add_axes([0.93, 0.025, 0.016, 0.917])
    fig.colorbar(im, cax=cbax, label="Z$_{500}$ [dam]")
    fig.subplots_adjust(left=0.01, bottom=0.02, right=0.92, top=0.95, wspace=0.05, hspace=0.25)
    '''
    fig.subplots_adjust(right=0.95)
    cbax = fig.add_axes([0.93, 0.16, 0.015, 0.66])
    fig.colorbar(im, cax=cbax, label="Z$_{500}$ [dam]")
    fig.subplots_adjust(left=0.01, bottom=0.01, right=0.92, top=0.97, wspace=0.05, hspace=0.01)
    #'''

    plt.savefig(os.path.join("plots", "end_conditions.pdf"))
    plt.savefig(os.path.join("plots", "end_conditions.png"))
    plt.close()


def long_rollout_plot(suffix=""):

    def contour_plot(ax, model_name, file_name, set_xlabel=False, set_ylabel=False, suffix=suffix):
        sample = 0
        scale = 98.1
        fname = "outputs" + suffix + ".nc"        

        # Load DataArray
        base_path = os.path.join("outputs", file_name, "evaluation")
        #da = xr.open_dataset(
        #    os.path.join(base_path, "targets.nc" if model_name=="Verification" else "outputs.nc")
        #).z500.isel(sample=sample)
        path = os.path.join("outputs/clstm1m_cyl_4x57_v0/evaluation/targets.nc" if model_name=="Verification" else os.path.join(base_path, fname))
        da = xr.open_dataset(path).z500.isel(sample=sample)
        #da = xr.open_dataset(
        #    os.path.join(base_path, "targets.nc" if model_name=="Verification" else "outputs.nc")
        #).u10.isel(sample=sample)
        #da = xr.open_mfdataset(
        #    "data/netcdf/weatherbench/10m_u_component_of_wind/10m*"
        #).u10.isel(time=slice(-600, -1))
        days = [pd.Timedelta(t).days for t in da.time.values]
        #days = [1 for t in da.time.values]
        #days = range(len(days))


        # claculated seasonal cycle using xarray rolling. The "construct" operation stacks the rolling windows as a new
        # dimension and allows for calculating the mean accross the window while ignoring NaNs.
        # https://github.com/pydata/xarray/issues/4278
        zonal_da = da.mean(dim='lon').squeeze().transpose().rolling(dim=dict(time=12), center=True).construct('new').mean('new',skipna=True)

        # Plot
        im = ax.contourf(days,da.lat,zonal_da/scale, cmap="Spectral_r", levels=np.arange(490,591,10), extend='both')
        


        #im = ax.contourf(days,da.lat,zonal_da, cmap="Spectral_r", extend='both')
        
        '''
        # Add 560 hPa ref line from ground truth
        #da_gt = xr.open_dataset(os.path.join(base_path, "targets.nc")).z500.isel(sample=sample)
        da_gt = xr.open_dataset(os.path.join("outputs", "clstm1m_cyl_4x57_v0", "evaluation", "targets.nc")).z500.isel(sample=sample)
        ref_fcst = da_gt.mean(dim='lon').squeeze().transpose().rolling(dim=dict(time=20), center=True).construct('new').mean('new',skipna=True)
        ax.contour(days, da_gt.lat, ref_fcst/scale,levels=[560],colors="black",linestyles="-",linewidths=1.5)
        
        # Plot 560 hPa ref line from forecast
        #if model_name != "Verification":
        #    ref_fcst = da.mean(dim='lon').squeeze().transpose().rolling(dim=dict(time=20), center=True).construct('new').mean('new',skipna=True)
        #    ax.contour(days, da.lat, ref_fcst/scale,levels=[560],colors="white",linestyles="--",linewidths=1.5)
        '''
            
        # Labeling
        if set_xlabel: ax.set_xlabel('Lead time [days]')
        if set_ylabel: ax.set_ylabel('Latitude')
        ax.set_title(model_name)

        return im

    #"""
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(10, 6), sharex=True, sharey=True)

    im = contour_plot(ax=axs[0, 0], model_name="Verification", file_name="clstm1m_cyl_4x57_v0", set_ylabel=True)
    contour_plot(ax=axs[0, 1], model_name="ConvLSTM Cyl (16M)", file_name="clstm16m_cyl_4x228_v2")
    contour_plot(ax=axs[0, 2], model_name="U-Net Cyl (128M)", file_name="unet128m_cyl_128-256-512-1024-2014_v2")
    contour_plot(ax=axs[0, 3], model_name="SwinTransformer Cyl (2M)", file_name="swint2m_cyl_d88_l2x4_h2x4_v0")
    contour_plot(ax=axs[1, 0], model_name="FNO (64M)", file_name="fno2d64m_cyl_d307_v0", set_ylabel=True)
    contour_plot(ax=axs[1, 1], model_name="ConvLSTM HPX (16M)", file_name="clstm16m_hpx8_4x228_v1")
    contour_plot(ax=axs[1, 2], model_name="U-Net HPX (16M)", file_name="unet16m_hpx8_92-184-368-736_v0")
    contour_plot(ax=axs[1, 3], model_name="SwinTransformer HPX (16M)", file_name="swint16m_hpx8_d120_l3x4_h3x4_v2")
    contour_plot(ax=axs[2, 0], model_name="TFNO (128M)", file_name="tfno2d128m_cyl_d477_v0", set_ylabel=True)
    contour_plot(ax=axs[2, 1], model_name=r"FourCastNet $1x1$ (4M)", file_name="fcnet4m_emb272_nopos_l6_v1")
    contour_plot(ax=axs[2, 2], model_name="SFNO (128M)", file_name="sfno2d128m_cyl_d686_equi_nonorm_nopos_v0")
    contour_plot(ax=axs[2, 3], model_name="Pangu-Weather (32M)", file_name="pangu32m_d216_h6-12-12-6_v1")
    contour_plot(ax=axs[3, 0], model_name=r"FourCastNet $2x4$ (8M)", file_name="fcnet8m_emb384_nopos_p2x4_l6_v2", set_ylabel=True, set_xlabel=True)
    contour_plot(ax=axs[3, 1], model_name=r"FourCastNet $4x4$ (64M)", file_name="fcnet64m_emb940_nopos_p4x4_l8_v0", set_xlabel=True)
    contour_plot(ax=axs[3, 2], model_name="MeshGraphNet (32M)", file_name="mgn32m_l8_d470_v0", set_xlabel=True)
    contour_plot(ax=axs[3, 3], model_name="GraphCast (16M)", file_name="gcast16m_p4_b1_d565_v2", set_xlabel=True)

    fig.subplots_adjust(right=.9)
    cbax = fig.add_axes([0.93, 0.08, 0.015, 0.87])
    fig.colorbar(im, cax=cbax, label="Z$_{500}$ [dam]")

    fig.subplots_adjust(left=0.07, bottom=0.08, right=0.9, top=0.95, wspace=0.15, hspace=0.3)
    """

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(10, 1.8), sharex=True, sharey=True)

    im = contour_plot(ax=axs[0], model_name="Verification", file_name="clstm1m_cyl_4x57_v0", set_ylabel=True, set_xlabel=True)
    contour_plot(ax=axs[1], model_name="ConvLSTM Cyl (16M)", file_name="clstm16m_cyl_4x228_v2", set_xlabel=True)
    contour_plot(ax=axs[2], model_name="ConvLSTM HPX (16M)", file_name="clstm16m_hpx8_4x228_v1", set_xlabel=True)
    contour_plot(ax=axs[3], model_name="SFNO (128M)", file_name="sfno2d128m_cyl_d686_equi_nonorm_nopos_v0", set_xlabel=True)

    #fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7, 3), sharex=True, sharey=True)
    #im = contour_plot(ax=axs[0], model_name="SFNO", file_name="clstm1m_cyl_4x57_v0", set_ylabel=True, set_xlabel=True)
    #im = contour_plot(ax=axs[0], model_name="era5 loaded", file_name="clstm16m_cyl_4x228_v2", set_xlabel=True, suffix="_long")
    #im = contour_plot(ax=axs[1], model_name="unet", file_name="unet128m_cyl_128-256-512-1024-2014_v2", set_xlabel=True, suffix="_long")
    #im = contour_plot(ax=axs[1], model_name="sfno", file_name="sfno2d128m_cyl_d686_equi_nonorm_nopos_v0", set_xlabel=True, suffix="_long")
    
    fig.subplots_adjust(right=.9)
    cbax = fig.add_axes([0.92, 0.25, 0.015, 0.6])
    fig.colorbar(im, cax=cbax, label="Z$_{500}$ [dam]")

    fig.subplots_adjust(left=0.06, bottom=0.25, right=0.9, top=0.85, wspace=0.05, hspace=0.02)
    #"""
    
    fig.savefig(os.path.join("plots", "long_rollout.pdf"))
    #fig.savefig(os.path.join("plots", "long_rollout_u10.pdf"))
    fig.savefig(os.path.join("plots", "long_rollout.png"))
    plt.close()


def multi_long_rollout_mean_plots():

    # Optionally, calculate statistics of ground truth to determine outliers (3*sigma-rule)
    #ds_gt = xr.open_dataset(os.path.join("outputs", "clstm1m_cyl_4x57_v0", "evaluation", "targets.nc"))
    #mu_z500, std_z500 = ds_gt.z500.mean().values, ds_gt.z500.std().values
    #mu_z500, std_z500 = ds_gt.z500.sel(lat=slice(-55, -45)).mean().values, ds_gt.sel(lat=slice(-55, -45)).z500.std().values
    mu_z500, std_z500 = 5.421e+04, 3.356e+03
    mu_u10, std_u10 = 6.771, 5.841

    #fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9, 3), sharex=True)
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 5), sharex=True, gridspec_kw={'height_ratios': [3, 2]})
    ax = long_rollout_mean_plot(axs=axs[:, 0], vname="z500", mu=mu_z500, ylim_mean=[53000, 55500], ylim_std=[0, 110], yshade=[mu_z500-0.2*std_z500, mu_z500+0.2*std_z500], title="Geopotential", ylabel=r"$\Phi_{500}\ [\operatorname{m}^2\operatorname{s}^{-2}]$")
    #long_rollout_std_plot(ax=axs[1, 0], vname="z500", ylim=[3000, 4000], std_gt=std_z500, ylabel=r"Geopotential $\Phi_{500}\ [\operatorname{m}^2\operatorname{s}^{-2}]$")
    long_rollout_mean_plot(axs=axs[:, 1], vname="u10", mu=mu_u10, ylim_mean=[3, 12], ylim_std=[0, 2], yshade=[mu_u10-0.4*std_u10, mu_u10+0.4*std_u10], title="South Westerlies", ylabel=r"$\operatorname{U}_{10m}\ [m\operatorname{s}^{-1}]$")
    #long_rollout_std_plot(ax=axs[1, 1], vname="u10", ylim=[3, 7], std_gt=std_u10, ylabel=r"South Westerlies $\operatorname{U}_{10}\ [m\operatorname{s}^{-1}]$")
    #fig.tight_layout()
    bbox_to_anchor = (1.0, -1.25)
    ax.legend(loc='lower center', bbox_to_anchor=bbox_to_anchor, ncol=4)
    #fig.subplots_adjust(left=0.12, bottom=0.4, right=0.98, top=0.95, wspace=0.2, hspace=0.25)
    fig.subplots_adjust(left=0.12, bottom=0.3, right=0.98, top=0.95, wspace=0.2, hspace=0.12)
    #plt.tight_layout()
    fig.savefig(os.path.join("plots", "long_rollout_mean.pdf"))


def long_rollout_mean_plot(
    axs,
    vname: str = "z500",
    mu: float = 5.421e+04,
    ylim_mean: list = [5e4, 7e4],
    ylim_std: list = [0, 100],
    yshade: list = [53500, 54800],
    title: str = "Geopotential",
    ylabel: str = r"$\Phi_{500}\ [\operatorname{m}^2\operatorname{s}^{-2}]$"
    ):

    ax_mean, ax_std = axs

    f = 12  # Muliplication factor to generate monthly statistics. Set to 1 for annual statistics (faster)

    import sys
    sys.path.append("")
    import scripts.evaluate as evaluate

    print(f"Creating long rollout mean plot for {vname}")

    for m_idx, model_path in enumerate(evaluate.MODEL_NAME_PLOT_ARGS):
        model_dict = evaluate.MODEL_NAME_PLOT_ARGS[model_path]
        model_name = model_dict["label"]
        if model_name in ["Persistence", "Climatology"]: continue  # Ignore baselines for this evaluation
        print(f"\tGenerating plot for {model_name}")

        #if model_name != "Pangu-Weather (32M)": continue
        #if "MeshGraphNet" not in model_name: continue
        #if "GraphCast" not in model_name: continue

        # Load data
        base_path = os.path.join("outputs", model_path, "evaluation")
        da = xr.open_dataset(os.path.join(base_path, "outputs_long.nc"))[vname].isel(time=slice(0, -1, 365//f))

        if vname == "u10": da = da.sel(lat=slice(-55, -45))  # Consider South Westerlies only
        mean = da.mean(dim=("sample", "lat", "lon"))
        #std = mean.rolling(dim=dict(time=len(da.time)//50), center=True).construct("new").std("new",skipna=True)
        std = mean.rolling(dim=dict(time=47), center=False).construct("new").std("new",skipna=True)
        #std = mean.rolling(dim=dict(time=93), center=True).construct("new").std("new",skipna=True)

        # Filter mean data by setting entries to nan once the forecast surpasses a threshold interval
        srp_thrsh = np.logical_or(mean > ylim_mean[1], mean < ylim_mean[0])
        if srp_thrsh.any():
            idcs = np.where(srp_thrsh)[0]
            first_idx = idcs[1] if len(idcs) > 1 else idcs[0]
            mean[first_idx:] = std[first_idx:] = np.nan
            print(f"\t\tSurpassed threshold at {da.time.isel(time=idcs[0]).values/(1e9*60*60*24*f)}")

        # Plot data
        x_range = np.arange(start=1, stop=len(mean)+1, step=1)/(4*f)
        ax_mean.plot(x_range, mean, c=model_dict["c"], ls=model_dict["ls"], label=model_dict["label"])
        ax_std.plot(x_range[:48], std[:48], c=model_dict["c"], ls=model_dict["ls"], alpha=0.4)
        ax_std.plot(x_range[48:], std[48:], c=model_dict["c"], ls=model_dict["ls"], label=model_dict["label"])

        #if m_idx > 2: break

    ax_mean.plot(x_range, np.ones_like(x_range)*mu, c="k", ls=":", lw=2.0, label=r"$\mu(\operatorname{Verification})$")
    ax_std.plot(x_range, -np.ones_like(x_range), c="k", ls=":", lw=2.0, label=r"$\mu(\operatorname{Verification})$")
    
    if vname == "z500": ax_mean.set_yscale('log'); ax_mean.set_yticks(np.arange(start=5e4, stop=7e4, step=5e3))
    ax_std.set_xscale("log")
    ax_mean.set_xscale("log")
    ax_mean.set_title(title)
    ax_mean.grid(visible=True, which='minor', color='silver')
    ax_mean.grid(visible=True, which='major', color='grey')
    ax_mean.set_ylim(ylim_mean)
    ax_mean.set_xlim([0.1, 50])
    #ax_mean.set_xlabel("Lead time [years]")
    ax_mean.set_ylabel(ylabel)
    #ax_mean.set_xticklabels([0, 0.1, 1, 10])
    ax_mean.fill_between(x_range, ylim_mean[0], yshade[0], color="lightgray")
    ax_mean.fill_between(x_range, yshade[1], ylim_mean[1], color="lightgray")
    ax_std.set_ylim(ylim_std)
    ax_std.set_xticklabels([0, 0.1, 1, 10])
    ax_std.set_ylabel(r"$\sigma$")
    ax_std.set_xlabel("Lead time [years]")
    ax_std.grid(visible=True, which='minor', color='silver')
    ax_std.grid(visible=True, which='major', color='grey')

    return ax_std


def long_rollout_std_plot(
    ax,
    vname: str = "z500",
    ylim: list = [5e4, 7e4],
    std_gt: float = 3.356e+03,
    ylabel: str = r"Geopotential $\Phi_{500}\ [\operatorname{m}^2\operatorname{s}^{-2}]$"
    ):

    f = 12  # Muliplication factor to generate monthly statistics. Set to 1 for annual statistics (faster)

    import sys
    sys.path.append("")
    import scripts.evaluate as evaluate

    print(f"Creating long rollout mean plot for {vname}")

    for m_idx, model_path in enumerate(evaluate.MODEL_NAME_PLOT_ARGS):
        model_dict = evaluate.MODEL_NAME_PLOT_ARGS[model_path]
        model_name = model_dict["label"]
        if model_name in ["Persistence", "Climatology"]: continue  # Ignore baselines for this evaluation
        print(f"\tGenerating plot for {model_name}")
        
        # Load data
        base_path = os.path.join("outputs", model_path, "evaluation")
        ds = xr.open_dataset(os.path.join(base_path, "outputs_long.nc"))[vname].isel(time=slice(0, -1, 365//f))
        if vname == "u10": ds = ds.sel(lat=slice(-55, -45))  # Consider South Westerlies only
        #data = ds.mean(dim=("sample", "lat", "lon")).values

        # Filter data by setting entries to nan once the forecast surpasses a threshold interval
        #srp_thrsh = np.logical_or(data > ylim[1], data < ylim[0])
        #if srp_thrsh.any():
        #    idcs = np.where(srp_thrsh)[0]
        #    first_idx = idcs[1] if len(idcs) > 1 else idcs[0]
        #    data[first_idx:] = np.nan
        #    print(f"\t\tSurpassed threshold at {ds.time.isel(idcs[0]).values}")

        # Plot data
        x_range = np.arange(start=1, stop=len(data)+1, step=1)/(4*f)
        ax.plot(x_range, data, c=model_dict["c"], ls=model_dict["ls"], label=model_dict["label"])

    ax.plot(x_range, np.ones_like(x_range)*std_gt, color="k", lw=2.0, ls=":")
    
    if vname == "z500": ax.set_yscale('log'); ax.set_yticks(np.arange(start=5e4, stop=7e4, step=5e3))
    ax.set_xscale("log")
    ax.grid(visible=True, which='minor', color='silver')
    ax.grid(visible=True, which='major', color='grey')
    ax.set_ylim(ylim)
    ax.set_xlim([0.1, 50])
    ax.set_xlabel("Lead time [years]")
    ax.set_ylabel(ylabel)
    ax.set_xticklabels([0, 0.1, 1, 10])
    #ax.fill_between(x_range, ylim[0], yshade[0], color="lightgray")
    #ax.fill_between(x_range, yshade[1], ylim[1], color="lightgray")

    return ax


def kinetic_energy_plot():

    vname = "u10"
    rolling_window_size = 4*4
    sample1, sample2 = 0, 51

    def prepare_data(path, s1, s2, vname):
        da = xr.open_dataset(path)[vname]
        data_s0 = da.isel(sample=s1).mean(dim=('lon')).squeeze().transpose().rolling(dim=dict(time=12), center=True).construct('new').mean('new',skipna=True)
        data_s1 = da.isel(sample=s2).mean(dim=('lon')).squeeze().transpose().rolling(dim=dict(time=12), center=True).construct('new').mean('new',skipna=True)
        data_all = da.mean(dim=("sample", "lon")).squeeze().transpose().rolling(dim=dict(time=12), center=True).construct('new').mean('new',skipna=True)
        return data_s0, data_s1, data_all

    #climatology = xr.open_dataset(os.path.join("outputs", "climatology", "evaluation", "outputs.nc"))[vname].mean(dim=("sample", "lat", "lon")).rolling(dim=dict(time=rolling_window_size), center=True).construct('new').mean('new',skipna=True)
    verif_s1, verif_s2, verif_all = prepare_data(path=os.path.join("outputs", "clstm1m_cyl_4x57_v0", "evaluation", "targets.nc"), s1=sample1, s2=sample2, vname=vname)
    model1_s1, model1_s2, model1_all = prepare_data(path=os.path.join("outputs", "clstm16m_cyl_4x228_v2", "evaluation", "outputs.nc"), s1=sample1, s2=sample2, vname=vname)
    #model2_s1, model2_s2, model2_all = prepare_data(path=os.path.join("outputs", "pangu32m_d216_h6-12-12-6_v2", "evaluation", "outputs.nc"), s1=sample1, s2=sample2, vname=vname)
    model2_s1, model2_s2, model2_all = prepare_data(path=os.path.join("outputs", "sfno2d128m_cyl_d686_equi_nonorm_nopos_v0", "evaluation", "outputs.nc"), s1=sample1, s2=sample2, vname=vname)

    #model1_s1, model1_s2, model1_all = prepare_data(path=os.path.join("outputs", "persistence", "evaluation", "outputs.nc"), s1=sample1, s2=sample2, vname=vname)
    #model2_s1, model2_s2, model2_all = prepare_data(path=os.path.join("outputs", "climatology", "evaluation", "outputs.nc"), s1=sample1, s2=sample2, vname=vname)

    # Plotting
    fig, axs = plt.subplots(3, 3, figsize=(10, 4), sharex=True, sharey=True)
    days = [pd.Timedelta(t).days for t in verif_s1.time.values]

    __ = axs[0, 0].contourf(days, verif_s1.lat, verif_s1, cmap="Spectral_r", extend='both')
    __ = axs[0, 1].contourf(days, verif_s2.lat, verif_s2, cmap="Spectral_r", extend='both')
    im = axs[0, 2].contourf(days, verif_all.lat, verif_all, cmap="Spectral_r", extend='both')

    __ = axs[1, 0].contourf(days, model1_s1.lat, model1_s1, cmap="Spectral_r", extend='both')
    __ = axs[1, 1].contourf(days, model1_s2.lat, model1_s2, cmap="Spectral_r", extend='both')
    __ = axs[1, 2].contourf(days, model1_all.lat, model1_all, cmap="Spectral_r", extend='both')

    __ = axs[2, 0].contourf(days, model2_s1.lat, model2_s1, cmap="Spectral_r", extend='both')
    __ = axs[2, 1].contourf(days, model2_s2.lat, model2_s2, cmap="Spectral_r", extend='both')
    __ = axs[2, 2].contourf(days, model2_all.lat, model2_all, cmap="Spectral_r", extend='both')


    axs[0, 0].set_title("Initialization in January")
    axs[0, 1].set_title("Initialization in June")
    axs[0, 2].set_title("Mean over 104 forecasts")

    axs[0, 0].text(10, 55, "Verification")
    axs[1, 0].text(10, 55, "ConvLSTM (16M)")
    axs[2, 0].text(10, 55, "SFNO (128M)")

    axs[0, 0].set_ylabel("Latitude")
    axs[1, 0].set_ylabel("Latitude")
    axs[2, 0].set_ylabel("Latitude")
    axs[2, 0].set_xlabel("Lead time [days]")
    axs[2, 1].set_xlabel("Lead time [days]")
    axs[2, 2].set_xlabel("Lead time [days]")

    fig.subplots_adjust(right=.9)
    cbax = fig.add_axes([0.92, 0.11, 0.015, 0.83])
    fig.colorbar(im, cax=cbax, label="U$_{10}$ [m/s]")

    fig.subplots_adjust(left=0.06, bottom=0.11, right=0.9, top=0.94, wspace=0.05, hspace=0.15)
    fig.savefig(os.path.join("plots", f"zonal_{vname}_clstm_sfno.pdf"))
        

if __name__ == "__main__":

    kinetic_energy_plot()
    #long_rollout_plot()
    #multi_long_rollout_mean_plots()
    #multi_x_over_params_plot()
    #rmse_over_params_plot()
    #memory_over_params_plot()
    #runtime_over_params_plot()
    #lag_resolution_plot()
    #end_conditions_plot(days=365)

    print("Done.")
