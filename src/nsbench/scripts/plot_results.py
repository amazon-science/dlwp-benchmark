#! /usr/env/bin python3

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable


def multi_x_over_params_plot():
    
    # Experiment 1 & 2
    #fig_title = "r1e3_1k_r1e4_1k"
    #n = 2
    #fig, axs = plt.subplots(1, n, figsize=(n*6.1, 3))
    #ax = rmse_over_params_plot(experiment=1, ax=axs[0], leftmost=True)
    #__ = rmse_over_params_plot(experiment=2, ax=axs[1])
    #ax.legend(bbox_to_anchor=(-0.017, -0.47, 1, 0.2), loc="lower left", ncol=5, handlelength=3.6)
    #plt.savefig(f"plots/rmse_over_params/{fig_title}.pdf", bbox_inches="tight")
    #plt.savefig(f"plots/rmse_over_params/{fig_title}.png", bbox_inches="tight")
    #plt.show()
    #plt.close()

    # Experiment 2 & 3
    fig_title = "r1e4_1k_r1e4_10k"
    n = 2
    fig, axs = plt.subplots(1, n, figsize=(n*6.1, 3))
    ax = rmse_over_params_plot(experiment=2, ax=axs[0], leftmost=True)
    __ = rmse_over_params_plot(experiment=3, ax=axs[1])
    ax.legend(bbox_to_anchor=(-0.017, -0.47, 1, 0.2), loc="lower left", ncol=5, handlelength=4.53)
    plt.savefig(f"plots/rmse_over_params/{fig_title}.pdf", bbox_inches="tight")
    plt.savefig(f"plots/rmse_over_params/{fig_title}.png", bbox_inches="tight")
    plt.show()
    plt.close()

    # RMSE & Mem & Runtime from Experiment 1
    #fig_title = "rmse_mem_secs_over_params_r1e4_1k"
    #n = 3
    #fig, axs = plt.subplots(1, n, figsize=(4*n, 3))
    #ax = rmse_over_params_plot(experiment=1, ax=axs[0], leftmost=True)
    #__ = memory_over_params_plot(ax=axs[1])
    #__ = runtime_over_params_plot(ax=axs[2])
    ##ax.legend(bbox_to_anchor=(-0.03, 1.08, 1, 0.2), loc="upper left", ncol=5, handlelength=3.4)
    #ax.legend(bbox_to_anchor=(-0.03, -0.47, 1, 0.2), loc="lower left", ncol=5, handlelength=3.4)
    #fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.35, hspace=None)
    #plt.savefig(f"plots/{fig_title}.pdf", bbox_inches="tight")
    #plt.savefig(f"plots/{fig_title}.png", bbox_inches="tight")
    #plt.show()
    #plt.close()


def rmse_over_params_plot(experiment, ax, leftmost: bool = False):

    params = np.array(["5k", "50k", "500k", "1M", "2M", "4M", "8M", "16M", "32M"])

    if experiment == 1:
        # Navier-Stokes [64x64], R=1e3, N=1k
        # Persistence, ConvLSTM, U-Net, FNO3D, TFNO3D, TFNO2D, SwinTransformer, FourCastNet, MS MeshGraphNet
        model_names_dict = {
            "Persistence": {"ls": "solid", "c": "dimgray", "clipped": None},
            "ConvLSTM": {"ls": (0, (1, 0.5)), "c": "lightgreen", "clipped": [2, 3, 4, 5]},
            "U-Net": {"ls": "dotted", "c": "darkgreen", "clipped": None},
            "FNO3D L1-8": {"ls": (0, (3, 1)), "c": "deepskyblue", "clipped": None},
            "TFNO3D L1-16": {"ls": (0, (5, 2)), "c": "steelblue", "clipped": None},
            "TFNO3D L4": {"ls": (5, (10, 3)), "c": "dodgerblue", "clipped": None},
            "TFNO2D L4": {"ls": "dashdot", "c": "darkturquoise", "clipped": None},
            "SwinTransformer": {"ls": (0, (3, 1, 1, 1, 1, 1)), "c": "darkorange", "clipped": None},
            "FourCastNet": {"ls": (0, (3, 2, 1, 2, 1, 2)), "c": "firebrick", "clipped": [4, 5, 6]},
            "MS MeshGraphNet":{"ls": (0, (5, 2, 5, 2, 1, 2)), "c": "blueviolet", "clipped": [0, 1, 2]},
        }
        rmses = np.zeros(shape=(len(model_names_dict), len(params)))*np.nan
        rmses[0, :] = 0.5993
        rmses[1, :6] = 0.1278, 0.0319, 0.0102, 0.009, 0.2329, 0.4443
        rmses[2, :7] = 0.5993, 0.0269, 0.0157, 0.0145, 0.0131, 0.0126, 0.0126
        rmses[3, :8] = 0.365, 0.2159, 0.1125, 0.1035, 0.105, 0.0383, 0.0144, 0.0095
        rmses[4, 3:] = 0.0873, 0.0889, 0.0221, 0.0083, 0.0066, 0.0069
        rmses[5, 1:7] = 0.0998, 0.0173, 0.0127, 0.0107, 0.0091, 0.0083
        rmses[6, :8] = 0.0632, 0.0139, 0.0055, 0.0046, 0.0043, 0.0054, 0.0041, 0.0046
        rmses[7, :5] = 0.1637, 0.0603, 0.0107, 0.0084, 0.007
        rmses[8, :7] = 0.1558, 0.0404, 0.0201, 0.0154, 0.0164, 0.0153, 0.0149
        rmses[9, :3] = 0.2559, 0.0976, 0.5209
    elif experiment == 2:
        # Navier-Stokes [64x64], R=1e4, N=1k
        # Persistence, U-Net, FNO3D, TFNO3D_1-8_layers, TFNO3D_4_layers, TFNO2D, SwinTransformer
        model_names_dict = {
            "Persistence": {"ls": "solid", "c": "dimgray", "clipped": None},
            "U-Net": {"ls": "dotted", "c": "darkgreen", "clipped": None},
            "FNO3D L4": {"ls": (0, (3, 1)), "c": "deepskyblue", "clipped": None},
            "TFNO3D L1-16": {"ls": (0, (5, 2)), "c": "steelblue", "clipped": None},
            "TFNO3D L4": {"ls": (5, (10, 3)), "c": "dodgerblue", "clipped": []},
            "TFNO2D L4": {"ls": "dashdot", "c": "darkturquoise", "clipped": None},
            "SwinTransformer": {"ls": (0, (3, 1, 1, 1, 1, 1)), "c": "darkorange", "clipped": None},
        }
        rmses = np.zeros(shape=(len(model_names_dict), len(params)))*np.nan
        rmses[0, :] = 1.2022
        rmses[1, 1:7] = 0.3874, 0.3217, 0.3117, 0.3239, 0.3269, 0.3085
        rmses[2, 6] = 0.3964
        rmses[3, 4:8] = 0.5407, 0.3811, 0.3105, 0.3219
        rmses[4, 1:7] = 0.5038, 0.3444, 0.3261, 0.3224, 0.3155, 0.3105
        rmses[5, :7] = 0.4955, 0.3091, 0.2322, 0.2203, 0.2236, 0.2349, 0.2358
        rmses[6, :5] = 0.6266, 0.4799, 0.2678, 0.2552, 0.2518
    elif experiment == 3:
        # Navier-Stokes [64x64], R=1e4, N=10k
        # Persistence, U-Net, FNO3D, TFNO3D_1-16_layers, TFNO3D_4_layers, TFNO2D, SwinTransformer
        model_names_dict = {
            "Persistence": {"ls": "solid", "c": "dimgray", "clipped": None},
            "U-Net": {"ls": "dotted", "c": "darkgreen", "clipped": None},
            "FNO3D L4": {"ls": (0, (3, 1)), "c": "deepskyblue", "clipped": None},
            "TFNO3D L1-16": {"ls": (0, (5, 2)), "c": "steelblue", "clipped": None},
            "TFNO3D L4": {"ls": (5, (10, 3)), "c": "dodgerblue", "clipped": []},
            "TFNO2D L4": {"ls": "dashdot", "c": "darkturquoise", "clipped": None},
            "SwinTransformer": {"ls": (0, (3, 1, 1, 1, 1, 1)), "c": "darkorange", "clipped": None},
        }
        rmses = np.zeros(shape=(len(model_names_dict), len(params)))*np.nan
        rmses[0, :] = 1.2022
        rmses[1, 1:7] = 0.3837, 0.3681, 0.2497, 0.3162, 0.235, 0.2383
        rmses[2, 6] = 0.2015
        rmses[3, 4:9] = 0.5146, 0.2805, 0.1814, 0.1570, 0.1709
        rmses[4, 1:8] = 0.4799, 0.2754, 0.2438, 0.2197, 0.2028, 0.1814, 0.1740
        rmses[5, :8] = 0.4846, 0.2897, 0.1778, 0.1585, 0.1449, 0.1322, 0.1248, 0.1210
        rmses[6, :5] = 0.6187, 0.4698, 0.2374, 0.2078, 0.191
    else:
        print(f"Experiment parameter is {experiment} but should be in [1, 2, 3].")
        exit()

    for m_idx, model_name in enumerate(model_names_dict):
        entry = model_names_dict[model_name]
        ax.plot(params, rmses[m_idx], label=model_name, ls=entry["ls"], c=entry["c"], marker="o", lw=2.5, markersize=5)

        if entry["clipped"]:
            clip_idcs = np.array(entry["clipped"])
            ax.scatter(params[clip_idcs], rmses[m_idx, clip_idcs], c=entry["c"], marker="v", s=100, zorder=2)

        #ax.scatter(5.5, 0.0084, color="blue", label="FNO3D from paper")
        #ax.scatter(1.7, 0.0128, color="brown", label="FNO2D from paper")

        ax.set_xticklabels(params)
        ax.set_xlabel("#parameters")
        if leftmost: ax.set_ylabel("RMSE")
        #ax.set_xlim([-0.25, len(params)-0.75])
        #ax.set_ylim([3.5e-3, 0.7])
        #ax.set_ylim([0.21, 0.7])
        ax.set_yscale("log")
        ax.grid(which="both")
        ax.set_title(f"Experiment {experiment}")

        if experiment == 3:
        ax.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], minor=True)
        ax.set_yticklabels([r"$2\times10^{-1}$", r"$3\times10^{-1}$", r"$4\times10^{-1}$", None, r"$6\times10^{-1}$",
                            None, None, None], minor=True)
    
    return ax


def memory_over_params_plot(ax):

    params = np.array(["5k", "50k", "500k", "1M", "2M", "4M", "8M", "16M", "32M"])
    model_names_dict = {
        "Persistence": {"ls": "solid", "c": "dimgray", "clipped": None},
        "ConvLSTM": {"ls": (0, (1, 0.5)), "c": "lightgreen", "clipped": [2, 3, 4, 5]},
        "U-Net": {"ls": "dotted", "c": "darkgreen", "clipped": None},
        "FNO3D L1-16": {"ls": (0, (3, 1)), "c": "deepskyblue", "clipped": None},
        "TFNO3D L1-32": {"ls": (0, (5, 2)), "c": "steelblue", "clipped": None},
        "TFNO2D L4": {"ls": "dashdot", "c": "darkturquoise", "clipped": None},
        "SwinTransformer": {"ls": (0, (3, 1, 1, 1, 1, 1)), "c": "darkorange", "clipped": None},
        "FourCastNet": {"ls": (0, (3, 2, 1, 2, 1, 2)), "c": "firebrick", "clipped": [4, 5, 6]},
        "MS MeshGraphNet":{"ls": (0, (5, 2, 5, 2, 1, 2)), "c": "blueviolet", "clipped": [0, 1, 2]},
    }

    memories = np.zeros(shape=(len(model_names_dict), len(params)))*np.nan

    # Persistence, ConvLSTM, ..., MS MeshGraphNet
    memories[0, :] = 0.0
    memories[1, :6] = 5594, 6846, 10490, 12778, 15718, 20434
    memories[2, :7] = 5176, 5264, 5454, 5648, 5790, 6204, 6668
    memories[3, :8] = 5586, 13144, 13144, 13146, 13148, 13332, 13908, 14936
    memories[4, 3:] = 13146, 13148, 13546, 14044, 15172, 17466
    memories[5, :8] = 7918, 8046, 8346, 8652, 9058, 9702, 10646, 13220#, 14528
    memories[6, :5] = 6072, 11336, 19146, 20720, 18660
    memories[7, :7] = 5178, 5336, 6452, 7210, 8066, 9344, 10660
    memories[8, :3] = 6422, 10402, 20448

    for m_idx, model_name in enumerate(model_names_dict):
        entry = model_names_dict[model_name]
        ax.plot(params, memories[m_idx], label=model_name, ls=entry["ls"], c=entry["c"], marker="o", lw=2.5, markersize=5)

        if entry["clipped"]:
            clip_idcs = np.array(entry["clipped"])
            ax.scatter(params[clip_idcs], memories[m_idx, clip_idcs], c=entry["c"], marker="v", s=100, zorder=2)

    ax.set_xticklabels(params)
    ax.set_xlabel("#parameters")
    ax.set_ylabel("Memory")
    #ax.set_xlim([-0.25, len(params)-0.75])
    #ax.set_ylim([5e-3, 0.7])
    #ax.set_yscale("log")

    ax.grid()
    ax.set_title("Experiment 1")
    #plt.legend(ncol=3, loc="lower right")

    return ax


def runtime_over_params_plot(ax):

    params = np.array(["5k", "50k", "500k", "1M", "2M", "4M", "8M", "16M", "32M"])
    model_names_dict = {
        "Persistence": {"ls": "solid", "c": "dimgray", "clipped": None},
        "ConvLSTM": {"ls": (0, (1, 0.5)), "c": "lightgreen", "clipped": [2, 3, 4, 5]},
        "U-Net": {"ls": "dotted", "c": "darkgreen", "clipped": None},
        "FNO3D L1-16": {"ls": (0, (3, 1)), "c": "deepskyblue", "clipped": None},
        "TFNO3D L1-32": {"ls": (0, (5, 2)), "c": "steelblue", "clipped": None},
        "TFNO2D": {"ls": "dashdot", "c": "darkturquoise", "clipped": None},
        "SwinTransformer": {"ls": (0, (3, 1, 1, 1, 1, 1)), "c": "darkorange", "clipped": None},
        "FourCastNet": {"ls": (0, (3, 2, 1, 2, 1, 2)), "c": "firebrick", "clipped": [4, 5, 6]},
        "MS MeshGraphNet":{"ls": (0, (5, 2, 5, 2, 1, 2)), "c": "blueviolet", "clipped": [0, 1, 2]},
    }

    runtime = np.zeros(shape=(len(model_names_dict), len(params)))*np.nan

    # Persistence, ConvLSTM, ..., MS MeshGraphNet
    runtime[0, :] = 0.0
    runtime[1, :6] = 83.07, 85.35, 112.80, 171.00, 268.72, 398.94
    runtime[2, :7] = 127.58, 131.25, 135.79, 135.81, 137.70, 140.45, 141.82
    runtime[3, :8] = 4.74, 13.24, 13.26, 13.41, 13.30, 24.17, 44.48, 86.37#, 170.03
    runtime[4, 3:] = 13.82, 14.36, 24.22, 45.06, 87.93, 173.79
    runtime[5, :8] = 60.13, 61.24, 62.83, 62.33, 62.71, 70.49, 88.57, 128.81#, 205.11
    runtime[6, :5] = 37.34, 132.17, 244.57, 251.57, 277.23
    runtime[7, :7] = 54.37, 55.62, 192.42, 191.04, 192.48, 196.50, 195.71
    runtime[8, :3] = 364.07, 398.49, 476.07

    for m_idx, model_name in enumerate(model_names_dict):
        entry = model_names_dict[model_name]
        ax.plot(params, runtime[m_idx], label=model_name, ls=entry["ls"], c=entry["c"], marker="o", lw=2.5, markersize=5)

        if entry["clipped"]:
            clip_idcs = np.array(entry["clipped"])
            ax.scatter(params[clip_idcs], runtime[m_idx, clip_idcs], c=entry["c"], marker="v", s=100, zorder=2)

    ax.set_xticklabels(params)
    ax.set_xlabel("#parameters")
    ax.set_ylabel("Seconds per epoch")
    #ax.set_xlim([-0.25, len(params)-0.75])
    #ax.set_ylim([5e-3, 0.7])
    #ax.set_yscale("log")

    ax.grid()
    ax.set_title("Experiment 1")
    #plt.legend()#loc="upper right")
    
    return ax


def lag_resolution_plot():

    lags = [1, 2, 4, 8]
    downscale_factors = [1, 2, 4, 8, 16, 32, 64]

    rmses = np.zeros(shape=(len(downscale_factors), len(lags)))*np.nan

    # TFNO3D
    #rmses[0, :] = 0.0141, 0.0141, 0.0139, 0.0143  # 1
    #rmses[1, :] = 0.0140, 0.0140, 0.0138, 0.0142  # 2
    #rmses[2, :] = 0.0140, 0.0141, 0.0140, 0.0144  # 4
    #rmses[3, :] = 0.0157, 0.0158, 0.0158, 0.0161  # 8
    #rmses[4, :] = 0.1262, 0.1046, 0.0210, 0.0218  # 16
    #rmses[5, :] = 0.1105, 0.0706, 0.0714, 0.0569  # 32
    #rmses[6, :] = 0.0001, 0.0009, 0.0006, 0.0004  # 64

    # TFNO2D
    rmses[0, :] = np.nan, 0.1409, 0.2210, 0.0122  # 0
    rmses[1, :] = np.nan, 0.6053, 0.0238, 0.0055  # 1
    rmses[2, :] = np.nan, 0.1481, 0.2425, np.nan  # 2
    rmses[3, :] = np.nan, 0.1204, np.nan, 0.0046  # 4
    rmses[4, :] = np.nan, 0.5421, 0.0382, 0.0062  # 8
    rmses[5, :] = 0.3540, 0.3305, np.nan, 0.0660  # 16
    #rmses[6, :] = 0.0000, 0.0000, 0.0000, 0.0000  # 32

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for m_idx, downscale_factor in enumerate(downscale_factors):
        shape_string = rf"[{64//downscale_factor}$\times${64//downscale_factor}]"
        ax.plot(range(len(lags)), rmses[m_idx], marker="o", label=rf"Downscale factor: {downscale_factor} $\rightarrow$ {shape_string}")

    ax.set_xticks(range(len(lags)))
    ax.set_xticklabels(lags)
    ax.set_xlabel("History frames (lags)")
    ax.set_ylabel("RMSE [log scale]")
    #ax.set_xlim([0.75, 4.25])
    #ax.set_ylim([5e-3, 1.1])
    ax.set_yscale("log")

    plt.grid()
    plt.legend(ncol=1)
    plt.show()


def navier_stokes_initial_vs_end_conditions():

    sample = 50

    """
    # R=1e3, N=1k
    file_and_model_names = {
        "ConvLSTM (1M)": "clstm64_4x57_sl50_tf10_clip_noise0",
        "U-Net (4M)": "unet64_23-46-92-184-368_c10_sl50_tf10_noise0",
        "FNO3D (16M)": "fno3d64_d32_12-12-12_l8_sl50_tf10_cl40_noise0",
        "TFNO3D (16M)": "tfno3d64_d45_12-12-12_l8_sl50_tf10_cl40_noise0",
        "TFNO2D (8M)": "tfno2d64_d108_12-12_l4_sl50_tf10_cl40_noise0",
        "SwinTransformer (2M)": "swint64_d88_l2x4_h2x4_sl50_tf10_noise0",
        "FourCastNet (8M)": "fcnet64_ps4x4_emb468_l4_c10_sl50_tf10_clip_noise0",
        "MS MeshGraphNet (50k)": "gcast64_p4_hop2_l2_dp34_sl50_tf10_clip_noise0"
    }
    fig, axs = plt.subplots(5, 4, figsize=(10, 12), sharex=True, sharey=True)
    vmin, vmax = -2.0, 2.0
    dmin, dmax = -0.1, 0.1

    for m_idx, model_name in enumerate(file_and_model_names):
        # Load data
        ds = xr.open_dataset(os.path.join("outputs", file_and_model_names[model_name], "forecast.nc"))
        initial_condition = ds.inputs[sample, 0, 0]
        end_condition = ds.targets[sample, -1, 0]
        prediction = ds.outputs[sample, -1, 0]
        diff = prediction - end_condition

        # Plot ground truth
        if m_idx == 0:
            axs[0, 0].imshow(ds.inputs[sample, 0, 0], origin="lower", vmin=vmin, vmax=vmax)
            axs[0, 1].imshow(ds.inputs[sample, 17, 0], origin="lower", vmin=vmin, vmax=vmax)
            axs[0, 2].imshow(ds.inputs[sample, 33, 0], origin="lower", vmin=vmin, vmax=vmax)
            axs[0, 3].imshow(ds.inputs[sample, -1, 0], origin="lower", vmin=vmin, vmax=vmax)
            axs[0, 0].set_title(r"Initial condition, $t=0$")
            axs[0, 1].set_title(r"$t=17$")
            axs[0, 2].set_title(r"$t=33$")
            axs[0, 3].set_title(r"End condition, $t=49$")
            axs[0, 0].set_ylabel(r"Ground truth $y_t$")
            axs[0, 0].set_xticks([])
            axs[0, 0].set_yticks([])
            axs[0, 1].set_xticks([])
            axs[0, 1].set_yticks([])
            axs[0, 2].set_xticks([])
            axs[0, 2].set_yticks([])
            axs[0, 3].set_xticks([])
            axs[0, 3].set_yticks([])

        row = 1 if m_idx < 4 else 3
        col = m_idx % 4

        # Plot difference
        im_diff = axs[row, col].imshow(diff, cmap="bwr", vmin=dmin, vmax=dmax, origin="lower")
        axs[row, col].set_yticks([])
        axs[row, col].set_yticks([])
        axs[row, col].set_title(model_name)

        # Plot prediction
        im_pred = axs[row+1, col].imshow(prediction, vmin=vmin, vmax=vmax, origin="lower")
        axs[row+1, col].set_xticks([])
        axs[row+1, col].set_yticks([])

        dmin = min(dmin, diff.min())
        dmax = max(dmax, diff.max())

    axs[1, 0].set_ylabel(r"Difference $(\hat{y}_{t}-y_{t}), t=49$")
    axs[3, 0].set_ylabel(r"Difference $(\hat{y}_{t}-y_{t}), t=49$")
    axs[2, 0].set_ylabel(r"Prediction $\hat{y}_{t=49}$")
    axs[4, 0].set_ylabel(r"Prediction $\hat{y}_{t=49}$")

    div_diff = make_axes_locatable(axs[3, -1])
    div_pred = make_axes_locatable(axs[4, -1])
    cax_diff = div_diff.append_axes("right", size="5%", pad=0.05)
    cax_pred = div_pred.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(im_diff, cax=cax_diff, orientation='vertical')
    fig.colorbar(im_pred, cax=cax_pred, orientation='vertical')

    fig.tight_layout()
    plt.savefig("plots/ic_vs_ec/r1e3_n1k_5x4.pdf")
    plt.savefig("plots/ic_vs_ec/r1e3_n1k_5x4.png")
    plt.show()
    """
    # R=1e4, N=1k
    plot_title = "r1e4_n1k"
    file_and_model_names = {
        "U-Net (8M)": "unet64_33-66-132-264-528_c10_sl30_tf10_noise0_r1e4",
        "TFNO3D (8M)": "tfno3d64_d32_12-12-12_l4_sl30_tf10_cl40_noise0_r1e4",
        "TFNO2D (1M)": "tfno2d64_d38_12-12_l4_sl30_tf10_cl40_noise0_r1e4",
        "SwinTransformer (2M)": "swint64_d88_l2x4_h2x4_sl30_tf10_noise0_r1e4",
    }
    
    # R=1e4, N=10k
    plot_title = "r1e4_n10k"
    file_and_model_names = {
        "U-Net (4M)": "unet64_23-46-92-184-368_c10_sl30_tf10_noise0_r1e4_n10k",
        "TFNO3D (16M)": "tfno3d64_d32_12-12-12_l16_sl30_tf10_cl40_noise0_r1e4_n10k",
        "TFNO2D (16M)": "tfno2d64_d154_12-12_l4_sl30_tf10_cl40_noise0_r1e4_n10k",
        "SwinTransformer (2M)": "swint64_d88_l2x4_h2x4_sl30_tf10_noise0_r1e4_n10k",
    }
    
    fig, axs = plt.subplots(2, 5, figsize=(11, 4), sharex=True, sharey=True)
    vmin, vmax = -3.8, 3.8
    dmin, dmax = -0.8, 0.8

    for m_idx, model_name in enumerate(file_and_model_names):
        # Load data
        ds = xr.open_dataset(os.path.join("outputs", file_and_model_names[model_name], "forecast.nc"))
        initial_condition = ds.inputs[sample, 0, 0]
        end_condition = ds.targets[sample, -1, 0]
        prediction = ds.outputs[sample, -1, 0]
        diff = prediction - end_condition

        # Plot
        if m_idx == 0:
            axs[0, 0].imshow(initial_condition, origin="lower")
            axs[1, 0].imshow(end_condition, origin="lower", vmin=vmin, vmax=vmax)
            axs[0, 0].set_title("Initial condition")
            axs[1, 0].set_title("Ground truth")
            #axs[0, 1].set_ylabel(r"Difference $(\hat{y}_{t}-y_{t}), t=49$")
            #axs[1, 1].set_ylabel(r"Prediction $\hat{y}_{t=29}$")
            axs[0, 0].set_xticks([])
            axs[0, 0].set_yticks([])
            axs[1, 0].set_xticks([])
            axs[1, 0].set_yticks([])
        axs[0, m_idx+1].set_yticks([])
        axs[0, m_idx+1].set_yticks([])
        im_top = axs[0, m_idx+1].imshow(diff, cmap="bwr", vmin=dmin, vmax=dmax, origin="lower")
        im_bot = axs[1, m_idx+1].imshow(prediction, vmin=vmin, vmax=vmax, origin="lower")
        axs[1, m_idx+1].set_title(model_name)

        dmin = min(dmin, diff.min())
        dmax = max(dmax, diff.max())

        axs[0, m_idx+1].set_xticks([])
        axs[0, m_idx+1].set_yticks([])

    div_top = make_axes_locatable(axs[0, -1])
    div_bot = make_axes_locatable(axs[1, -1])
    cax_top = div_top.append_axes("right", size="5%", pad=0.05)
    cax_bot = div_bot.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(im_top, cax=cax_top, orientation='vertical')
    fig.colorbar(im_bot, cax=cax_bot, orientation='vertical')
    
    fig.tight_layout()
    plt.savefig(f"plots/ic_vs_ec/{plot_title}.pdf")
    plt.savefig(f"plots/ic_vs_ec/{plot_title}.png")
    plt.show()
    #"""
        


if __name__ == "__main__":

    multi_x_over_params_plot()
    #rmse_over_params_plot()
    #memory_over_params_plot()
    #runtime_over_params_plot()
    #lag_resolution_plot()
    #navier_stokes_initial_vs_end_conditions()

    print("Done.")