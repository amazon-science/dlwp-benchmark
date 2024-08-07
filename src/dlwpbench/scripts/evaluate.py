#! bin/env/python3

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import gc  # Garbage collector
import sys
import time
import shutil
import argparse
import threading
import subprocess
import multiprocessing
from tqdm import tqdm

import hydra
import numpy as np
import torch as th
import pandas as pd
import xarray as xr
from omegaconf import DictConfig
from dask.diagnostics import ProgressBar

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append("")
from data.datasets import *
from data.processing.healpix_mapping import HEALPixRemap
from data.processing.istarmap import istarmap
from models import *


MODEL_NAME_PLOT_ARGS = {
    "persistence": {"c": "dimgray", "ls": "solid", "label": "Persistence"},
    "climatology": {"c": "darkgrey", "ls": "solid", "label": "Climatology"},
    "clstm16m_cyl_4x228_v2": {"c": "yellowgreen", "ls": "solid", "label": "ConvLSTM (16M)"},
    "clstm16m_hpx8_4x228_v1": {"c": "yellowgreen", "ls": "dashed", "label": "ConvLSTM HPX (16M)"},
    "unet128m_cyl_128-256-512-1024-2014_v2": {"c": "darkgreen", "ls": "solid", "label": "U-Net (128M)"},
    "unet16m_hpx8_92-184-368-736_v0": {"c": "darkgreen", "ls": "dashed", "label": "U-Net HPX (16M)"},
    "swint2m_cyl_d88_l2x4_h2x4_v0": {"c": "darkorange", "ls": "solid", "label": "SwinTransformer (2M)"},
    "swint16m_hpx8_d120_l3x4_h3x4_v2": {"c": "darkorange", "ls": "dashed", "label": "SwinTransformer HPX (16M)"},
    "fno2d64m_cyl_d307_v0": {"c": "lightcoral", "ls": "solid", "label": "FNO2D (64M)"},
    "tfno2d128m_cyl_d477_v0": {"c": "darkturquoise", "ls": "solid", "label": "TFNO2D (128M)"},
    "fcnet4m_emb272_nopos_l6_v1": {"c": "firebrick", "ls": "solid", "label": r"FourCastNet $p=1x1$ (4M)"},
    "fcnet8m_emb384_nopos_p2x4_l6_v2": {"c": "goldenrod", "ls": "solid", "label": r"FourCastNet $p=2x4$ (8M)"},
    "fcnet64m_emb940_nopos_p4x4_l8_v0": {"c": "orangered", "ls": "solid", "label": r"FourCastNet $p=4x4$ (64M)"},
    "sfno2d128m_cyl_d686_equi_nonorm_nopos_v0": {"c": "steelblue", "ls": "solid", "label": "SFNO (128M)"},
    "pangu32m_d216_h6-12-12-6_v1": {"c": "deepskyblue", "ls": "solid", "label": "Pangu-Weather (32M)"},
    "mgn32m_l8_d470_v0": {"c": "blueviolet", "ls": "solid", "label": "MeshGraphNet (32M)"},
    "gcast16m_p4_b1_d565_v2": {"c": "darkblue", "ls": "solid", "label": "GraphCast (16M)"},
}


def make_biweekly_inits(
    start: str = "2017-01-01",
    end: str = "2018-12-31",
    sequence_length: int = 57,
    timedelta: int = 6
):
    times1 = pd.date_range(start=start,
                           end=pd.Timestamp(end) - pd.Timedelta(hours=sequence_length*timedelta),
                           freq='7D')
    times2 = pd.date_range(start=pd.Timestamp(start) + pd.Timedelta(days=3),
                           end=pd.Timestamp(end) - pd.Timedelta(hours=sequence_length*timedelta),
                           freq='7D')
    return times1.append(times2).sort_values().to_numpy()


def remap(cfg, data, name=None):
    latitudes = 32
    longitudes = 64
    hpx_remapper = HEALPixRemap(
        latitudes=latitudes,
        longitudes=longitudes,
        nside=data.shape[-1],
        verbose=cfg.verbose
    )
    if len(data.shape) == 5:
        # Inits
        B, C, _, _, _ = data.shape
        arguments = []
        for b_idx in range(B):
            for c_idx in range(C):
                arguments.append([data[b_idx][c_idx]])
        data = mp_hpx2ll(remapper=hpx_remapper, arguments=arguments, name=name)
        data = np.reshape(data, (B, C, latitudes, longitudes))  # [(b c) lat lon] -> [b c lat lon]
    else:
        # Outputs and targets
        B, T, C, _, _, _ = data.shape
        arguments = []
        for b_idx in range(B):
            for t_idx in range(T):
                for c_idx in range(C):
                    arguments.append([data[b_idx][t_idx][c_idx]])
        data = mp_hpx2ll(remapper=hpx_remapper, arguments=arguments, name=name)
        data = np.reshape(data, (B, T, C, latitudes, longitudes))  # [(b t c) lat lon] -> [b t c lat lon]
    return data


def mp_hpx2ll(remapper, arguments, name=None):
    # Run the remapping in parallel
    poolsize = 5
    with multiprocessing.Pool(poolsize) as pool:
        data = np.array(list(tqdm(pool.istarmap(remapper.hpx2ll, arguments), total=len(arguments), desc=name)))
        pool.terminate()
        pool.join()
    return data


def evaluate_model(cfg: DictConfig, file_path: str, dataset: WeatherBenchDataset = None, complevel: int = 7) -> None:
    """
    Evaluates a single model for a given configuration.

    :param cfg: The hydra configuration for the model
    :param file_path: The destination path for the datasets
    :param dataloader: The PyTorch dataloader if it exists already
    :param complevel: The level of compression when writing datasets to disk (higher is stronger compression).
    :return: A list of model inputs, outputs, and targets, each of shape [B, T, D, H, W]
    """

    if cfg.verbose: print("\nInitializing model")

    if cfg.seed:
        np.random.seed(cfg.seed)
        th.manual_seed(cfg.seed)
    device = th.device(cfg.device)

    #
    # Set up model
    model = eval(cfg.model.type)(**cfg.model).to(device=device)
    if cfg.verbose:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\tModel {cfg.model.name} has {trainable_params} trainable parameters")

    # Load checkpoint from file to continue training or initialize training scalars
    checkpoint_path = os.path.join("outputs", cfg.model.name, "checkpoints", f"{cfg.model.name}_best.ckpt")
    if cfg.verbose: print(f"\tRestoring model from {checkpoint_path}")
    checkpoint = th.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    #
    # Initializing dataloader for testing
    init_dates = make_biweekly_inits(
            start=cfg.data.test_start_date,
            end=cfg.data.test_stop_date,
            sequence_length=cfg.testing.sequence_length,
            timedelta=cfg.data.timedelta
    )
    #cfg.data.test_stop_date = "2017-10-16"
    #dataset = #dataset_dict["hpx" if "hpx" in file_path else "cyl"]
    if dataset == None:
        if cfg.verbose: print("\nInitializing dataset")
        dataset = hydra.utils.instantiate(
            cfg.data,
            start_date=cfg.data.test_start_date,
            stop_date=cfg.data.test_stop_date,
            sequence_length=cfg.testing.sequence_length,
            init_dates=init_dates
        )
        #if "hpx" in file_path: dataset_dict["hpx"] = dataset
        #else: dataset_dict["cyl"] = dataset

    dataloader = th.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.testing.batch_size,
        shuffle=False,
        num_workers=0
    )
    print()

    # Evaluate (without gradients): iterate over all test samples
    with th.no_grad():
        inits = list()
        outputs = list()
        targets = list()
        for constants, prescribed, prognostic, target in tqdm(dataloader, desc="Generating forecasts"):
            # Load data and generate predictions
            constants = constants.to(device=device) if not constants.isnan().any() else None
            prescribed = prescribed.to(device=device) if not prescribed.isnan().any() else None
            prognostic = prognostic.to(device=device)

            target = target.to(device=device)
            output = model(
                constants=constants if not constants == None else None,
                prescribed=prescribed if not prescribed == None else None,
                prognostic=prognostic
            )
            inits.append(prognostic[:, 0].cpu())
            outputs.append(output.cpu())
            targets.append(target.cpu())
        inits = th.cat(inits).numpy()
        outputs = th.cat(outputs).numpy()
        targets = th.cat(targets).numpy()
    
    # Undo normalization per variable and level
    if cfg.data.normalize:
        v_idx = 0
        for p in cfg.data.prognostic_variable_names_and_levels:
            if len(cfg.data.prognostic_variable_names_and_levels[p]) > 0:
                for l in cfg.data.prognostic_variable_names_and_levels[p]:
                    mean, std = dataset.stats[p]["level"][l]["mean"], dataset.stats[p]["level"][l]["std"]
                    inits[:, v_idx] = inits[:, v_idx]*std + mean
                    targets[:, :, v_idx] = targets[:, :, v_idx]*std + mean
                    outputs[:, :, v_idx] = outputs[:, :, v_idx]*std + mean
                    v_idx += 1
            else:
                mean, std = dataset.stats[p]["mean"], dataset.stats[p]["std"]
                inits[:, v_idx] = inits[:, v_idx]*std + mean
                targets[:, :, v_idx] = targets[:, :, v_idx]*std + mean
                outputs[:, :, v_idx] = outputs[:, :, v_idx]*std + mean
                v_idx += 1
    
    # If data in HEALPix format, project to LatLon
    if len(inits.shape) == 5:
        if cfg.verbose: print("\nMapping initial conditions, outputs, and targets from HEALPix to LatLon")
        inits = remap(cfg=cfg, data=inits, name="Initial conditions")
        outputs = remap(cfg=cfg, data=outputs, name="Outputs")
        targets = remap(cfg=cfg, data=targets, name="Targets")
        print()

    build_dataset(
        cfg=cfg,
        inits=inits,
        outputs=outputs,
        targets=targets,
        statistics=dataset.stats,
        file_path=file_path,
        init_dates=init_dates,
        complevel=complevel
    )

    return dataset


def build_dataset(
    cfg: DictConfig,
    inits: np.array,
    outputs: np.array,
    targets: np.array,
    statistics: dict,
    init_dates: np.array,
    file_path: str,
    complevel: int = 7
):
    """
    Creates a netCDF dataset for initializations, outputs, and targets and writes them to file.
    
    :param cfg: The hydra configuration of the model
    :param inits: The first frame of the prognostic inputs to the model
    :param outputs: The outputs of the model (predictions)
    :param targets: The ground truth and target for prediction
    :param statistics: Dictionary containing mean and standard deviations per variable and level
    :param init_dates: The dates where the forecasts are initialized
    :param file_path: The path to the directory where the datasets are written to
    """

    # Determine data dimensions and set resolution in degree
    B, T, D, H, W = outputs.shape
    deg = 5.625

    dt = f"{cfg.data.timedelta}h"
    timedeltas = pd.timedelta_range(start=dt, periods=T, freq=dt)

    # Set up netCDF dataset
    coords = {}
    coords["sample"] = init_dates
    coords["time"] = timedeltas
    coords["lat"] = np.array(np.arange(start=(-90+(5.625/2)), stop=90, step=deg), dtype=np.float32)
    coords["lon"] = np.array(np.arange(start=0, stop=360, step=deg), dtype=np.float32)
    chunkdict = {coord: len(coords[coord]) for coord in coords}
    chunkdict["sample"] = 1

    # Prepare dictionaries for initializations, outputs, and targets to create according Datasets below
    v_idx = 0
    inits_dict = dict()
    outputs_dict = dict()
    targets_dict = dict()
    compress_dict = dict()
    for p in cfg.data.prognostic_variable_names_and_levels:
        if len(cfg.data.prognostic_variable_names_and_levels[p]) > 0:
            for l in cfg.data.prognostic_variable_names_and_levels[p]:
                vname = f"{p}{l}"
                attrs = statistics[p]["level"][l]
                #if vname != "z500": v_idx+=1; continue
                inits_dict[vname] = xr.DataArray(data=inits[:, v_idx], dims=["sample", "lat", "lon"], attrs=attrs)
                outputs_dict[vname] = xr.DataArray(data=outputs[:, :, v_idx], dims=["sample", "time", "lat", "lon"], attrs=attrs)
                targets_dict[vname] = xr.DataArray(data=targets[:, :, v_idx], dims=["sample", "time", "lat", "lon"], attrs=attrs)
                #compress_dict[vname] = {"scale_factor": 0.1, "zlib": True, "complevel": complevel}
                v_idx += 1
        else:
            vname = p
            #if vname not in ["t2m", "u10"]: v_idx+=1; continue
            attrs = statistics[p]
            inits_dict[vname] = xr.DataArray(data=inits[:, v_idx], dims=["sample", "lat", "lon"], attrs=attrs)
            outputs_dict[vname] = xr.DataArray(data=outputs[:, :, v_idx], dims=["sample", "time", "lat", "lon"], attrs=attrs)
            targets_dict[vname] = xr.DataArray(data=targets[:, :, v_idx], dims=["sample", "time", "lat", "lon"], attrs=attrs)
            #compress_dict[vname] = {"scale_factor": 0.1, "zlib": True, "complevel": complevel}
            v_idx += 1

    # Create datasets and write them to file
    def write_to_file(ds: xr.Dataset, dst_path_name: str, compress_dict: dict):
        if os.path.exists(dst_path_name): os.remove(dst_path_name)  # Delete file if it exists
        print(f"\tWriting to {dst_path_name}")
        if "outputs" in dst_path_name:  # Display progress bar when writing the targets.nc to file
            write_job = ds.to_netcdf(dst_path_name, compute=False, encoding=compress_dict)
            with ProgressBar(): write_job.compute()
        else:
            ds.to_netcdf(dst_path_name, encoding=compress_dict)  # Silently write inits.nc and targets.nc

    print("\nWriting datasets to file. This may take a while.")# Optionally, reduce compression level via the -z flag")
    t1 = threading.Thread(target=write_to_file, args=(
        xr.Dataset(coords=coords, data_vars=inits_dict).chunk(chunkdict),
        os.path.join(file_path, "inits.nc"),
        compress_dict,
    ))
    t2 = threading.Thread(target=write_to_file, args=(
        xr.Dataset(coords=coords, data_vars=outputs_dict).chunk(chunkdict),
        os.path.join(file_path, "outputs.nc"),
        compress_dict,
    ))
    t3 = threading.Thread(target=write_to_file, args=(
        xr.Dataset(coords=coords, data_vars=targets_dict).chunk(chunkdict),
        os.path.join(file_path, "targets.nc"),
        compress_dict,
    ))
    t1.start(); t2.start(); t3.start()
    t1.join(); t2.join(); t3.join()

    print("\tDatasets successfully written to file\n")


def generate_mp4(
    cfg: DictConfig,
    ds_outputs: xr.Dataset,
    ds_targets: xr.Dataset,
    file_path: str,
    normalize: bool = False
):
    """
    Generates mp4 video visualizing model output, target, and the difference between those.

    :param cfg: The hydra configuration of the model
    :param ds: An xarray dataset containing model inputs, outputs, and targets
    """

    sample = 0
    file_path = os.path.join(file_path, "videos")
    os.makedirs(os.path.join(file_path, "frames"), exist_ok=True)

    for vname in tqdm(list(ds_outputs.keys()), desc="Generating frames and a video of the model forecasts"):
        outputs, targets = ds_outputs[vname].isel(sample=sample), ds_targets[vname].isel(sample=sample)
        if normalize:
            outputs = (outputs-outputs.attrs["mean"])/outputs.attrs["std"]
            targets = (targets-targets.attrs["mean"])/targets.attrs["std"]
        outputs, targets = outputs.values, targets.values

        # Visualize results
        diff = outputs - targets
        diffmax = max(abs(np.min(diff[cfg.model.context_size:])),
                    abs(np.max(diff[cfg.model.context_size:])))
        vmin, vmax = np.min(targets), np.max(targets)
        for t in range(outputs.shape[0]):
            fig, ax = plt.subplots(1, 3, figsize=(12, 3), sharex=True, sharey=True)
            
            ax[0].imshow(outputs[t], origin="lower", vmin=vmin, vmax=vmax, extent=[-180, 180, -90, 90])
            ax[0].set_title(r"Prediction ($\hat{y}$)")
            ax[0].set_xlabel("Longitude")
            ax[0].set_ylabel("Latitude")

            im1 = ax[1].imshow(targets[t], origin="lower", vmin=vmin, vmax=vmax, extent=[-180, 180, -90, 90])
            ax[1].set_title(r"Ground truth ($y$)")
            ax[1].set_xlabel("Longitude")
            divider1 = make_axes_locatable(ax[1])
            cax1 = divider1.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im1, cax=cax1, orientation='vertical')

            im2 = ax[2].imshow(diff[t], origin="lower", vmin=-diffmax, vmax=diffmax, cmap="bwr",
                               extent=[-180, 180, -90, 90])
            ax[2].set_title(r"Difference ($\hat{y}-y$)")
            ax[2].set_xlabel("Longitude")
            divider2 = make_axes_locatable(ax[2])
            cax2 = divider2.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im2, cax=cax2, orientation='vertical')

            hour = str(pd.to_timedelta(ds_outputs.isel(time=t).time.values))
            init_date = str(pd.Timestamp(ds_outputs.isel(sample=sample).sample.values).date())
            fig.suptitle(f"{vname}, time step = {t+1}/{outputs.shape[0]}, "
                         f"init date = {init_date}, lead time = {hour} hours")
            fig.tight_layout()
            fig.savefig(os.path.join(file_path, "frames", f"state_{str(t).zfill(4)}.png"))
            plt.close()

        # Generate a video from the just generated frames with ffmpeg
        subprocess.run(["ffmpeg",  #"/usr/bin/ffmpeg",
                        "-f", "image2",
                        "-hide_banner",
                        "-loglevel", "error",
                        "-r", "15",
                        "-pattern_type", "glob",
                        "-i", f"{os.path.join(file_path, 'frames', '*.png')}",
                        "-vcodec", "libx264",
                        "-crf", "22",
                        "-pix_fmt", "yuv420p",
                        "-y",
                        f"{os.path.join(file_path, f'{vname}.mp4')}"])
    
    # Cleaning up
    shutil.rmtree(os.path.join(file_path, "frames"))


def plot_acc_over_time(
    cfg: DictConfig,
    performance_dict: dict,
    plot_title: str = "Model comparison"
):
    """
    Plot anomaly correlation coefficient of all models (averaged over samples, dimensions, height, width) over time.
    """

    file_path = "./plots"
    os.makedirs(file_path, exist_ok=True)
    dt = cfg.data.timedelta

    vnames = list(performance_dict[list(performance_dict.keys())[0]]["outputs"].keys())
    for vname in vnames:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        for m_idx, model_name in enumerate(performance_dict):
            if model_name == "climatology": continue
            acc_path = os.path.join("outputs", model_name, "evaluation", "accs.nc")
            if not os.path.exists(acc_path): continue
            acc = xr.open_dataset(acc_path)[vname]
            x_range = np.arange(start=dt, stop=len(acc)*dt + 1, step=dt) / 24
            if model_name in list(MODEL_NAME_PLOT_ARGS.keys()): kwargs = MODEL_NAME_PLOT_ARGS[model_name]
            else: kwargs = {"label": model_name}
            ax.plot(x_range, acc, **kwargs)

        if not "x_range" in locals(): continue
        ax.grid()
        ax.set_ylabel("ACC")
        ax.set_xlabel("Lead time [days]")
        ax.set_xlim([x_range[0], x_range[-1]])
        ax.set_ylim([0.1, 1.0])
        ax.legend(ncol=2, fontsize=9)
        #fig.suptitle(plot_title)
        fig.tight_layout()
        fig.savefig(os.path.join(file_path, f"acc_plot_{vname}.pdf"))
        plt.close()


def plot_rmse_over_time(
    cfg: DictConfig,
    performance_dict: dict,
    plot_title: str = "Model comparison"
):
    """
    Plot the root mean squared error of all models (averaged over samples, dimensions, height, width) over time.
    """

    file_path = "./plots"
    os.makedirs(file_path, exist_ok=True)
    dt = cfg.data.timedelta

    vnames = list(performance_dict[list(performance_dict.keys())[0]]["outputs"].keys())
    for vname in vnames:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        rmse_max = -np.infty

        for m_idx, model_name in enumerate(performance_dict):
            rmse = xr.open_dataset(os.path.join("outputs", model_name, "evaluation", "rmses.nc"))
            rmse = xr.open_dataset(os.path.join("outputs", model_name, "evaluation", "rmses.nc"))[vname]
            x_range = np.arange(start=dt, stop=len(rmse)*dt + 1, step=dt) / 24
            if model_name in list(MODEL_NAME_PLOT_ARGS.keys()): kwargs = MODEL_NAME_PLOT_ARGS[model_name]
            else: kwargs = {"label": model_name}
            ax.plot(x_range, rmse, **kwargs)
            rmse_max = max(rmse_max, rmse.max())

        ax.grid()
        ax.set_ylabel("RMSE")
        ax.set_xlabel("Lead time [days]")
        ax.set_xlim([x_range[0], x_range[-1]])
        #ax.set_xlim([x_range[0], x_range[55]])
        #ax.set_ylim([200, 2000])
        #ax.set_ylim([50, 1200])
        ax.legend(ncol=2, fontsize=9)
        #fig.suptitle(plot_title)
        fig.tight_layout()
        fig.savefig(os.path.join(file_path, f"rmse_plot_{vname}.pdf"))
        plt.close()


def compute_metrics(
    cfg: DictConfig,
    ds_outputs: xr.Dataset,
    ds_targets: xr.Dataset,
    file_path: str,
    overide: bool = False
) -> None:
    """
    Compute RMSE and Frobenius Norm (accumulated error) and print them to console.

    :param cfg: The configuration of the model
    :param ds_outputs: The dataset containing the model outputs (predictions)
    :param ds_targets: The dataset containing the targets (ground truth)
    :param file_path: The destination path where to write results
    """

    print("\nChecking whether to compute metrics for", cfg.model.name, "model")

    T = ds_outputs.sizes["time"]  # Number of time steps

    # Compute latitude-area weighting factors cos(lat_j) / (1/N_lat * sum(lat_j))
    # Equation (2) in https://arxiv.org/abs/2002.00469
    lats_rad = np.deg2rad(ds_outputs.lat.values)
    avg_lats = np.mean(np.cos(lats_rad))
    lat_weights = np.expand_dims(np.cos(lats_rad)/avg_lats, axis=(0, 1, 3))
    mean_over = ["sample", "lat", "lon"]

    #
    # Compute root mean squared error per variable and write to file
    file_path_ = os.path.join(file_path, "rmses.nc")
    if not os.path.exists(file_path_) or overide:
        print("\tComputing RMSE...")
        mean_over = ["sample", "lat", "lon"]
        diff_out_tar = ds_outputs - ds_targets
        rmses = np.sqrt((lat_weights*diff_out_tar**2).mean(dim=mean_over))
        rmses.to_netcdf(os.path.join(file_path, "rmses.nc"))
    
    #
    # Compute anomaly correlation coefficient per variable and write to file
    # Equation (A1) in https://arxiv.org/abs/2002.00469
    path_to_climatology = os.path.join("outputs", "climatology", "evaluation", "outputs.nc")
    file_path_ = os.path.join(file_path, "accs.nc")
    if os.path.exists(path_to_climatology) and (not os.path.exists(file_path_) or overide):
        print("\tComputing ACC...")
        ds_climatology = xr.open_dataset(path_to_climatology)
        diff_out_clim = ds_outputs - ds_climatology
        diff_tar_clim = ds_targets - ds_climatology
        nom = (lat_weights*diff_out_clim*diff_tar_clim).mean(dim=mean_over)
        denom = np.sqrt(
            (lat_weights*diff_out_clim**2).mean(dim=mean_over) * (lat_weights*diff_tar_clim**2).mean(dim=mean_over)
        )
        accs = nom/denom
        accs.to_netcdf(os.path.join(file_path, "accs.nc"))

    #
    # Compute annually averaged RMSE for U10

    file_path_ = os.path.join(file_path, "rmse_months_01-12_global.nc")
    if not os.path.exists(file_path_) or overide:
        # Global RMSE
        print("\tComputing RMSE for physical soundness of global winds...")
        avg_tar = ds_targets.mean(dim=("time", "lon"))
        avg_out = ds_outputs.mean(dim=("time", "lon"))
        rmse_global = np.sqrt(((avg_out-avg_tar)**2).mean())
        rmse_global.to_netcdf(file_path_)

        # Trade Winds RMSE (near north and south of equator)
        file_path_ = os.path.join(file_path, "rmse_months_01-12_trade-winds.nc")
        print("\tComputing RMSE for physical soundness of Trade Winds...")
        avg_tar_ = (xr.merge([avg_tar.sel(lat=slice(-20, -10)), avg_tar.sel(lat=slice(10, 20))]))
        avg_out_ = (xr.merge([avg_out.sel(lat=slice(-20, -10)), avg_out.sel(lat=slice(10, 20))]))
        rmse_trade_winds = np.sqrt(((avg_out_-avg_tar_)**2).mean())
        rmse_trade_winds.to_netcdf(file_path_)

        # South Westerlies RMSE (in southern extratropics)
        file_path_ = os.path.join(file_path, "rmse_months_01-12_south-westerlies.nc")
        print("\tComputing RMSE for physical soundness of South Westerlies...")
        avg_tar_ = avg_tar.sel(lat=slice(-55, -45))
        avg_out_ = avg_out.sel(lat=slice(-55, -45))
        rmse_south_westerlies = np.sqrt(((avg_out_-avg_tar_)**2).mean())
        rmse_south_westerlies.to_netcdf(file_path_)

        # Clear memory
        del avg_tar, avg_tar_, avg_out, avg_out_

    #
    # Compute average RMSE over lead times of 11 and 12 months for Z500
    file_path_ = os.path.join(file_path, "rmse_months_11-12.nc")
    if not os.path.exists(file_path_) or overide:
        print("\tComputing RMSE in months 11 and 12 of one-year rollout...")
        avg_tar = ds_targets.sel(time=slice(pd.Timedelta(334, "d"), pd.Timedelta(365, "d"))).mean(dim=("time"))
        avg_out = ds_outputs.sel(time=slice(pd.Timedelta(334, "d"), pd.Timedelta(365, "d"))).mean(dim=("time"))
        rmse = np.sqrt(((avg_out-avg_tar)**2).mean())
        rmse.to_netcdf(file_path_)
        del avg_tar, avg_out


def run_evaluations(
    configuration_dir_list: str,
    device: str,
    overide: bool = False,
    batch_size: int = None,
    sequence_length: int = None,
    plot_title: str = "Model comparison",
    normalize_video: bool = False,
    complevel: int = 7
):
    """
    Evaluates a model with the given configuration.

    :param configuration_dir_list: A list of hydra configuration directories to the models for evaluation
    :param device: The device where the evaluations are performed
    """

    performance_dict = {}
    #manager = multiprocessing.Manager()
    #dataset_dict = manager.dict(cyl=None, hpx=None)
    dataset_hpx = None
    dataset_cyl = None

    # Iterate over all configuration directories and perform evaluations
    for configuration_dir in configuration_dir_list:
        
        # If default configuration path has been overridden, append .hydra since then a custom path to a specific model
        # has been provided by the user
        if configuration_dir != "configs": configuration_dir = os.path.join(configuration_dir, ".hydra")

        # Initialize the hydra configurations for this forecast
        with hydra.initialize(version_base=None, config_path=os.path.join("..", configuration_dir)):
            cfg = hydra.compose(config_name="config")
            cfg.device = device
            if batch_size: cfg.testing.batch_size = batch_size
            if sequence_length: cfg.testing.sequence_length = sequence_length

        # Generate forecasts if they do not exist and load them
        output_fname = "outputs.nc"
        file_path = os.path.join("outputs", str(cfg.model.name), "evaluation")
        if os.path.exists(os.path.join(file_path, output_fname)):
            ds = xr.open_dataset(os.path.join(file_path, output_fname))
        if not os.path.exists(os.path.join(file_path, output_fname)) or overide:
            os.makedirs(file_path, exist_ok=True)
            dataset = dataset_hpx if "hpx" in file_path else dataset_cyl
            #process = multiprocessing.Process(target=evaluate_model, args=(cfg, file_path, dataset_dict, complevel, ))
            #process.start()  # Run in separate process to release memory upon process termination
            #process.join()
            dataset = evaluate_model(cfg=cfg, file_path=file_path, dataset=dataset, complevel=complevel)
            if "hpx" in file_path: dataset_hpx = dataset
            else: dataset_cyl = dataset
        ds_inits = xr.open_dataset(os.path.join(file_path, "inits.nc"))
        ds_outputs = xr.open_dataset(os.path.join(file_path, "outputs.nc")).isel(time=slice(0, 1460))
        ds_targets = xr.open_dataset(os.path.join(file_path, "targets.nc"))

        # Compute forecast error metrics if they don't yet exist and write results to file
        if not os.path.exists(os.path.join(file_path, "rmses.nc")) or overide:
            #process = multiprocessing.Process(
            #    target=compute_metrics,
            #    args=(cfg, ds_outputs, ds_targets, file_path, overide, )
            #)
            #process.start()  # Run in separate process to release memory upon process termination
            #process.join()
            compute_metrics(
                cfg=cfg, ds_outputs=ds_outputs, ds_targets=ds_targets, file_path=file_path, overide=overide
            )

        # Add the current model's datasets to the performance dict for cross model evaluation later
        performance_dict[cfg.model.name] = dict(inits=ds_inits, outputs=ds_outputs, targets=ds_targets)

        # Generate video showcasing model forecast
        if not os.path.exists(os.path.join(file_path, "videos")) or overide:
            generate_mp4(
                cfg=cfg,
                ds_outputs=ds_outputs,
                ds_targets=ds_targets,
                file_path=file_path,
                normalize=normalize_video
            )

        # Clear RAM by deleting the datasets used here and calling the garbage collector subsequently
        del ds_inits, ds_outputs, ds_targets
        gc.collect()

    #if overide: plot_rmse_over_time(cfg-cfg, performance_dict=performance_dict)
    plot_rmse_over_time(cfg=cfg, performance_dict=performance_dict, plot_title=plot_title)
    plot_acc_over_time(cfg=cfg, performance_dict=performance_dict, plot_title=plot_title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model with a given configuration. Particular properties of the configuration can be "
                    "overwritten, as listed by the -h flag.")
    parser.add_argument("-c", "--configuration-dir-list", nargs="*", default=["configs"],
                        help="List of directories where the configuration files of all models to be evaluated lie.")
    parser.add_argument("-d", "--device", type=str, default="cpu",
                        help="The device to run the evaluation. Any of ['cpu' (default), 'cuda:0', 'mpg'].")
    parser.add_argument("-o", "--overide", action="store_true",
                        help="Overide model forecasts and evaluation files if they exist already.")
    parser.add_argument("-b", "--batch-size", type=int, default=None,
                        help="Batch size used for evaluation. Defaults to None to take entire test set in one batch.")
    parser.add_argument("-s", "--sequence-length", type=int, default=None,
                        help="Sequence length for the evaluation. Use 57 to generate 14-days forecasts.")
    parser.add_argument("-pt", "--plot-title", type=str, default="Model comparison",
                        help="The title for the RMSE plot.")
    parser.add_argument("-nv", "--normalize-video", action="store_false",
                        help="Whether to normalize values for the .mp4 visualization. Default true.")
    parser.add_argument("-z", "--complevel", type=int, default=7,
                        help="Compression level when writing netcdf datasets to file.")

    run_args = parser.parse_args()
    run_evaluations(configuration_dir_list=run_args.configuration_dir_list,
                    device=run_args.device,
                    overide=run_args.overide,
                    batch_size=run_args.batch_size,
                    sequence_length=run_args.sequence_length,
                    plot_title=run_args.plot_title,
                    normalize_video=run_args.normalize_video,
                    complevel=run_args.complevel)
    
    print("Done.")
