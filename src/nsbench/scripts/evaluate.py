#! bin/env/python3

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import argparse
import subprocess

import hydra
import numpy as np
import torch as th
import xarray as xr
from omegaconf import DictConfig

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append("")
from data.datasets import *
from models import *
import utils.utils as utils


def evaluate_model(cfg: DictConfig) -> [np.array, np.array, np.array]:
    """
    Evaluates a singel model for a given configuration.

    :param cfg: The hydra configuration for the model
    :return: A list of model inputs, outputs, and targets, each of shape [B, T, D, H, W]
    """
    if cfg.verbose: print("\nInitializing datasets and model")

    if cfg.seed:
        np.random.seed(cfg.seed)
        th.manual_seed(cfg.seed)
    device = th.device(cfg.device)

    # Initializing dataloaders for testing
    dataset = eval(cfg.data.type)(
        data_path=os.path.join(cfg.data.path, cfg.data.test_set_name),
        sequence_length=cfg.testing.sequence_length,
        downscale_factor=cfg.data.downscale_factor
        )

    dataloader = th.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.testing.batch_size,
        shuffle=False,
        num_workers=0
        )

    # Set up model
    model = eval(cfg.model.type)(**cfg.model).to(device=device)
    if cfg.verbose:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model {cfg.model.name} has {trainable_params} trainable parameters")

    # Load checkpoint from file to continue training or initialize training scalars
    checkpoint_path = os.path.join("outputs", cfg.model.name, "checkpoints", f"{cfg.model.name}_best.ckpt")
    if cfg.verbose: print(f"Restoring model from {checkpoint_path}")
    checkpoint = th.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate (without gradients): iterate over all test samples
    with th.no_grad():
        if cfg.verbose: print("Generating prediction...")
        inputs = list()
        outputs = list()
        targets = list()
        for x, y in dataloader:
            x = x.to(device=device)
            y = y.to(device=device)
            #if cfg.data.normalize: x = (x - data_mean) / data_std
            y_hat = model(x=x, teacher_forcing_steps=cfg.testing.teacher_forcing_steps)
            #if cfg.data.normalize: y_hat = (y_hat*data_std) + data_mean
            inputs.append(x)
            outputs.append(y_hat)
            targets.append(y)
        inputs = th.cat(inputs).cpu().numpy()
        outputs = th.cat(outputs).cpu().numpy()
        targets = th.cat(targets).cpu().numpy()

    return inputs, outputs, targets


def write_to_file(
    cfg: DictConfig,
    inputs: np.array,
    outputs: np.array,
    targets: np.array,
):
    """
    Creates a netCDF dataset containing inputs, outputs, and targets and writes it to file.
    
    :param cfg: The hydra configuration of the model
    :param inputs: The inputs to the model
    :param outputs: The outputs of the model (predictions)
    :param targets: The ground truth and target for prediction
    """

    # Determine data dimensions
    D_in = inputs.shape[2]
    B, T, D_out, H, W = outputs.shape

    # Set up netCDF dataset
    coords = {}
    coords["sample"] = np.array(range(B), dtype=np.int32)
    coords["time"] = np.array(range(T), dtype=np.int32)
    coords["dim_in"] = np.array(range(D_in), dtype=np.int32)
    coords["dim_out"] = np.array(range(D_in), dtype=np.int32)
    coords["height"] = np.array(range(H), dtype=np.int32)
    coords["width"] = np.array(range(W), dtype=np.int32)
    chunkdict = {coord: len(coords[coord]) for coord in coords}
    chunkdict["sample"] = 1

    data_vars = {
        "inputs": (["sample", "time", "dim_in", "height", "width"], inputs),
        "outputs": (["sample", "time", "dim_out", "height", "width"], outputs),
        "targets": (["sample", "time", "dim_out", "height", "width"], targets),
    }

    ds = xr.Dataset(
        coords=coords,
        data_vars=data_vars,
    ).chunk(chunkdict)

    # Write dataset to file
    file_name = os.path.join("outputs", str(cfg.model.name), "forecast.nc")
    ds.to_netcdf(file_name)


def generate_mp4(cfg: DictConfig, ds: xr.Dataset):
    """
    Generates mp4 video visualizing model output, target, and the difference between those.

    :param cfg: The hydra configuration of the model
    :param ds: An xarray dataset containing model inputs, outputs, and targets
    """

    if cfg.verbose: print("Generating frames and a video of the model forecast...")

    ds = ds.isel(height=slice(0, 64), width=slice(0, 64))

    os.makedirs(os.path.join("outputs", str(cfg.model.name), "frames"), exist_ok=True)
    outputs, targets = ds.outputs.values, ds.targets.values

    # Visualize results
    diff = outputs - targets
    diffmax = max(abs(np.min(diff[0, cfg.testing.teacher_forcing_steps:, 0])),
                  abs(np.max(diff[0, cfg.testing.teacher_forcing_steps:, 0])))
    vmin, vmax = np.min(targets[0, :, 0]), np.max(targets[0, :, 0])
    for t in range(outputs.shape[1]):
        if cfg.verbose:
            mode = "teacher forcing" if t < cfg.testing.teacher_forcing_steps else "closed loop"
            #print(f"{t}/{cfg.testing.sequence_length}", mode)
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        
        ax[0].imshow(outputs[0, t, 0], origin="lower", vmin=vmin, vmax=vmax)
        ax[0].set_title(r"Prediction ($\hat{y}$)")

        im1 = ax[1].imshow(targets[0, t, 0], origin="lower", vmin=vmin, vmax=vmax)
        ax[1].set_title(r"Ground truth ($y$)")
        divider1 = make_axes_locatable(ax[1])
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im1, cax=cax1, orientation='vertical')
        #cax1.yaxis.set_ticks_position('left')

        im2 = ax[2].imshow(diff[0, t, 0], origin="lower", vmin=-diffmax, vmax=diffmax, cmap="bwr")
        ax[2].set_title(r"Difference ($\hat{y}-y$)")
        divider2 = make_axes_locatable(ax[2])
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im2, cax=cax2, orientation='vertical')

        fig.suptitle(f"Time step = {t+1}/{outputs.shape[1]} ({mode})")
        fig.tight_layout()
        fig.savefig(f"outputs/{cfg.model.name}/frames/state_{str(t).zfill(4)}.png")
        plt.close()

    # Generate a video from the just generated frames with ffmpeg
    subprocess.run(["ffmpeg",
                    "-f", "image2",
                    "-hide_banner",
                    "-loglevel", "error",
                    "-r", "15",
                    "-pattern_type", "glob",
                    "-i", f"{os.path.join('outputs', str(cfg.model.name), 'frames', '*.png')}",
                    "-vcodec", "libx264",
                    "-crf", "22",
                    "-pix_fmt", "yuv420p",
                    "-y",
                    f"{os.path.join('outputs', str(cfg.model.name), 'video.mp4')}"])


def plot_rmse_over_time(
    cfg: DictConfig,
    performance_dict: dict,
    legend_labels: list = None,
    plot_title: str = "Model comparison"
):
    """
    Plot the root mean squared error of all models (averaged over samples, dimensions, height, width) over time.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    rmse_max = -np.infty
    for m_idx, model_name in enumerate(performance_dict):
        ds = performance_dict[model_name]

        # Compute and plot rmse
        rmse = np.sqrt(((ds.outputs-ds.targets)**2).mean(dim=["sample", "dim_out", "height", "width"]))
        if legend_labels: model_name = legend_labels[m_idx]
        ax.plot(range(1, len(rmse)+1), rmse, label=model_name)

        rmse_max = max(rmse_max, rmse.max())

    ax.plot([cfg.testing.teacher_forcing_steps, cfg.testing.teacher_forcing_steps], [0, rmse_max], ls="--",
            color="grey", label="End of teacher forcing")
    ax.grid()
    ax.set_ylabel("RMSE")
    ax.set_xlabel("Time step")
    ax.set_xlim([1, len(rmse)])
    ax.set_yscale("log") 
    ax.legend()
    fig.suptitle(plot_title)
    fig.tight_layout()
    fig.savefig("rmse_plot.pdf")
    plt.close()


def compute_metrics(cfg: DictConfig, ds: xr.Dataset) -> None:
    """
    Compute RMSE and Frobenius Norm (accumulated error) and print them to console.

    :param cfg: The configuration of the model
    :param ds: The dataset containing the model outputs (predictions) and targets
    """

    print("Model name:", cfg.model.name)

    T = ds.dims["time"]  # Number of time steps
    tf = cfg.testing.teacher_forcing_steps
    #tf = 10
    diff = ds.outputs-ds.targets

    # Root mean squared error overall, in teacher forcing, and in closed loop
    rmse = str(np.round(np.sqrt(((diff)**2).mean()).values, 4)).ljust(6)
    rmse_tf = str(np.round(np.sqrt(((diff).sel(time=slice(0, tf))**2).mean()).values, 4)).ljust(6)
    rmse_cl = str(np.round(np.sqrt(((diff).sel(time=slice(tf, T))**2).mean()).values, 4)).ljust(6)
    print("RMSE:", rmse, "\tRMSE TF:", rmse_tf, "\tRMSE CL:", rmse_cl)
    
    mean_over = ["sample", "dim_out", "height", "width"]
    frob = str(np.round(np.sqrt(((diff)**2)).mean(dim=mean_over).sum().values, 4)).ljust(6)
    frob_tf = str(np.round(np.sqrt(((diff)**2)).sel(time=slice(0, tf)).mean(dim=mean_over).sum().values, 4)).ljust(6)
    frob_cl = str(np.round(np.sqrt(((diff)**2)).sel(time=slice(tf, T)).mean(dim=mean_over).sum().values, 4)).ljust(6)
    print("Frob:", frob, "\tFrob TF:", frob_tf, "\tFrob CL:", frob_cl)


def run_evaluations(
    configuration_dir_list: str,
    device: str,
    overide: bool = False,
    teacher_forcing_steps: int = 10,
    test_set: str = None,
    batch_size: int = None,
    legend_labels: list = None,
    plot_title: str = "Model comparison"
):
    """
    Evaluates a model with the given configuration.

    :param configuration_dir_list: A list of hydra configuration directories to the models for evaluation
    :param device: The device where the evaluations are performed
    """

    performance_dict = {}

    # Iterate over all configuration directories and perform evaluations
    for configuration_dir in configuration_dir_list:
        
        # If default configuration path has been overridden, append .hydra since then a custom path to a specific model
        # has been provided by the user
        if configuration_dir != "configs": configuration_dir = os.path.join(configuration_dir, ".hydra")

        # Initialize the hydra configurations for this forecast
        with hydra.initialize(version_base=None, config_path=os.path.join("..", configuration_dir)):
            cfg = hydra.compose(config_name="config")
            cfg.device = device
            cfg.testing.teacher_forcing_steps = teacher_forcing_steps
            if test_set: cfg.data.test_set_name = test_set
            if batch_size: cfg.testing.batch_size = batch_size

        # Try to load forecast if it exists, otherwise generate it
        file_name = os.path.join("outputs", str(cfg.model.name), "forecast.nc")
        if not os.path.exists(file_name) or overide:
            inputs, outputs, targets = evaluate_model(cfg=cfg)
            write_to_file(cfg=cfg, inputs=inputs, outputs=outputs, targets=targets)
        ds = xr.open_dataset(file_name)

        # Compute model forecasting error metrics
        compute_metrics(cfg=cfg, ds=ds)

        # Add the current model's dataset to the performance dict
        performance_dict[cfg.model.name] = ds

        # Generate video showcasing model forecast
        if not os.path.exists(os.path.join(os.path.dirname(file_name), "video.mp4")) or overide: 
            generate_mp4(cfg=cfg, ds=ds)

    #if overide: plot_rmse_over_time(cfg-cfg, performance_dict=performance_dict)
    plot_rmse_over_time(cfg=cfg, performance_dict=performance_dict, legend_labels=legend_labels, plot_title=plot_title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model with a given configuration. Particular properties of the configuration can be "
                    "overwritten, as listed by the -h flag.")
    parser.add_argument("-c", "--configuration-dir-list", nargs='*', default=["configs"],
                        help="List of directories where the configuration files of all models to be evaluated lies.")
    parser.add_argument("-d", "--device", type=str, default="cpu",
                        help="The device to run the evaluation. Any of ['cpu' (default), 'cuda', 'mpg'].")
    parser.add_argument("-tf", "--teacher-forcing-steps", type=int, default=10,
                        help="The number of teacher forcing steps during inference.")
    parser.add_argument("-o", "--overide", action="store_true",
                        help="Overide model forecasts and evaluation files if they exist already.")
    parser.add_argument("-ts", "--test-set", type=str, default=None,
                        help="The name of the test set.")
    parser.add_argument("-b", "--batch-size", type=int, default=None,
                        help="Batch size used for evaluation. Defaults to None to take entire test set in one batch.")
    parser.add_argument("-l", "--legend-labels", nargs='*', default=None,
                        help="List of names for the models that are put into the legend.")
    parser.add_argument("-pt", "--plot-title", type=str, default="Model comparison",
                        help="The title for the RMSE plot.")

    run_args = parser.parse_args()
    run_evaluations(configuration_dir_list=run_args.configuration_dir_list,
                    device=run_args.device,
                    overide=run_args.overide,
                    teacher_forcing_steps=run_args.teacher_forcing_steps,
                    test_set=run_args.test_set,
                    batch_size=run_args.batch_size,
                    legend_labels=run_args.legend_labels,
                    plot_title=run_args.plot_title)
    
    print("Done.")
