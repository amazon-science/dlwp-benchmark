#! /usr/bin/env python3

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import math
import einops
import shutil
import argparse
import subprocess
import numpy as np
import torch as th
import xarray as xr
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt

from random_fields import GaussianRF


#w0: initial vorticity
#f: forcing term
#visc: viscosity (1/Re)
#T: final time
#delta_t: internal time-step for solve (descrease if blow-up)
#record_steps: number of in-time snapshots to record
def navier_stokes_2d(w0, f, visc, T, delta_t=1e-4, record_steps=1):

    #Grid size - must be power of 2
    N = w0.size()[-1]

    #Maximum frequency
    k_max = math.floor(N/2.0)

    #Number of steps to final time
    steps = math.ceil(T/delta_t)

    #Initial vorticity to Fourier space
    w_h = th.rfft(w0, 2, normalized=False, onesided=False)

    #Forcing to Fourier space
    f_h = th.rfft(f, 2, normalized=False, onesided=False)

    #If same forcing for the whole batch
    if len(f_h.size()) < len(w_h.size()):
        f_h = th.unsqueeze(f_h, 0)

    #Record solution every this number of steps
    record_time = math.floor(steps/record_steps)

    #Wavenumbers in y-direction
    k_y = th.cat((th.arange(start=0, end=k_max, step=1, device=w0.device), th.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N,1)
    #Wavenumbers in x-direction
    k_x = k_y.transpose(0,1)
    #Negative Laplacian in Fourier space
    lap = 4*(math.pi**2)*(k_x**2 + k_y**2)
    lap[0,0] = 1.0
    #Dealiasing mask
    dealias = th.unsqueeze(th.logical_and(th.abs(k_y) <= (2.0/3.0)*k_max, th.abs(k_x) <= (2.0/3.0)*k_max).float(), 0)

    #Saving solution and time
    sol = th.zeros(*w0.size(), record_steps, device=w0.device)
    sol_t = th.zeros(record_steps, device=w0.device)

    #Record counter
    c = 0
    #Physical time
    t = 0.0
    for j in range(steps):

        #if j%1000 == 0: print(j, steps)

        #Stream function in Fourier space: solve Poisson equation
        psi_h = w_h.clone()
        psi_h[...,0] = psi_h[...,0]/lap
        psi_h[...,1] = psi_h[...,1]/lap

        #Velocity field in x-direction = psi_y
        q = psi_h.clone()
        temp = q[...,0].clone()
        q[...,0] = -2*math.pi*k_y*q[...,1]
        q[...,1] = 2*math.pi*k_y*temp
        q = th.irfft(q, 2, normalized=False, onesided=False, signal_sizes=(N,N))

        #Velocity field in y-direction = -psi_x
        v = psi_h.clone()
        temp = v[...,0].clone()
        v[...,0] = 2*math.pi*k_x*v[...,1]
        v[...,1] = -2*math.pi*k_x*temp
        v = th.irfft(v, 2, normalized=False, onesided=False, signal_sizes=(N,N))

        #Partial x of vorticity
        w_x = w_h.clone()
        temp = w_x[...,0].clone()
        w_x[...,0] = -2*math.pi*k_x*w_x[...,1]
        w_x[...,1] = 2*math.pi*k_x*temp
        w_x = th.irfft(w_x, 2, normalized=False, onesided=False, signal_sizes=(N,N))

        #Partial y of vorticity
        w_y = w_h.clone()
        temp = w_y[...,0].clone()
        w_y[...,0] = -2*math.pi*k_y*w_y[...,1]
        w_y[...,1] = 2*math.pi*k_y*temp
        w_y = th.irfft(w_y, 2, normalized=False, onesided=False, signal_sizes=(N,N))

        #Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        F_h = th.rfft(q*w_x + v*w_y, 2, normalized=False, onesided=False)

        #Dealias
        F_h[...,0] = dealias* F_h[...,0]
        F_h[...,1] = dealias* F_h[...,1]

        #Cranck-Nicholson update
        w_h[...,0] = (-delta_t*F_h[...,0] + delta_t*f_h[...,0] + (1.0 - 0.5*delta_t*visc*lap)*w_h[...,0])/(1.0 + 0.5*delta_t*visc*lap)
        w_h[...,1] = (-delta_t*F_h[...,1] + delta_t*f_h[...,1] + (1.0 - 0.5*delta_t*visc*lap)*w_h[...,1])/(1.0 + 0.5*delta_t*visc*lap)

        #Update real time (used only for recording)
        t += delta_t

        if (j+1) % record_time == 0:
            #Solution in physical space
            w = th.irfft(w_h, 2, normalized=False, onesided=False, signal_sizes=(N,N))

            #Record solution and time
            sol[...,c] = w
            sol_t[c] = t

            c += 1

    return sol, sol_t


def create_mp4(data: np.array):
    # Build temporary directory to store single frames
    os.makedirs("frames", exist_ok=True)

    # Iterate over time dimension of data and create individual frames
    mean, std = data.mean(), data.std()
    vmin, vmax = mean - 5*std, mean + 5*std
    for t in tqdm(range(data.shape[0]), desc="Creating video.mp4 from frames"):
        fig, ax = plt.subplots()
        im = ax.imshow(data[t], origin="lower", vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(mappable=im, ax=ax)
        cbar.set_label(r"Vorticity $\omega$")
        fig.tight_layout()
        fig.savefig(os.path.join("frames", f"f{str(t).zfill(5)}.png"))
        plt.close()
    
    # Convert frames into video and delete frames directory
    subprocess.run(["ffmpeg",
                    "-f", "image2",
                    "-hide_banner",
                    "-loglevel", "error",
                    "-r", "20",
                    "-pattern_type", "glob",
                    "-i", f"{os.path.join('frames', 'f*.png')}",
                    "-vcodec", "libx264",
                    "-crf", "22",
                    "-pix_fmt", "yuv420p",
                    "-y",
                    f"{os.path.join(f'video.mp4')}"])
    shutil.rmtree("frames")


def generate_data(
    resolution: int = 64,
    n_samples: int = 1000,
    batch_size: int = 50,
    max_simulation_time: int = 50,
    delta_t: float = 1e-3,
    record_steps: int = None,
    viscosity: float = 1e-3,
    alpha: float = 2.5,
    tau: float = 7.0,
    forcing_multiplicator: float = 2.0,
    device: str = "cpu",
    dst_path: str = os.path.join("data", "netcdf", "navier-stokes"),
    animate: bool = False
):

    device = th.device(device)
    s = resolution
    N = n_samples  # Number of solutions/samples to generate
    T = max_simulation_time
    f_mul = forcing_multiplicator  # 2.0 for 64x64, 0.2 for 256x256
    if record_steps is None: record_steps = max_simulation_time   # Number of snapshots from solution
    batch_size = min(N, batch_size)

    # Set up 2d GRF with covariance parameters
    GRF = GaussianRF(2, s, alpha=alpha, tau=tau, device=device)  # a=2.5 for 64x64, a=0.5 for 256x256

    # Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
    t = th.linspace(0, 1, s+1, device=device)
    t = t[0:-1]

    X,Y = th.meshgrid(t, t)
    f = 0.1*(th.sin(f_mul*math.pi*(X+Y))+th.cos(f_mul*math.pi*(X+Y)))  # 64x64

    # Inputs
    a = th.zeros(N, s, s)
    # Solutions
    u = th.zeros(N, record_steps, 1, s, s)

    # Solve equations in batches (order of magnitude speed-up)
    c = 0
    for j_idx, j in enumerate(tqdm(range(N//batch_size), desc="Generating data in batches")):

        # Sample random feilds
        w0 = GRF.sample(batch_size)

        # Solve NS
        sol, sol_t = navier_stokes_2d(w0=w0, f=f, visc=viscosity, T=T, delta_t=delta_t, record_steps=record_steps)

        a[c:(c+batch_size),...] = w0
        u[c:(c+batch_size),...] = einops.rearrange(sol, "n h w t -> n t 1 h w")

        c += batch_size

    # Optionally create video of first sample
    if animate and j_idx == 0: create_mp4(data=u[0, :, 0])

    #
    # Set up netCDF dataset
    print(f"Building netCDF dataset")
    coords = {}
    coords["sample"] = np.array(range(N), dtype=np.int32)
    coords["time"] = np.array(range(record_steps), dtype=np.int32)
    coords["dim"] = np.array(range(1), dtype=np.int32)
    coords["height"] = np.array(range(s), dtype=np.int32)
    coords["width"] = np.array(range(s), dtype=np.int32)
    chunkdict = {coord: len(coords[coord]) for coord in coords}
    chunkdict["sample"] = 1
    
    data_vars = {
        "a": (["sample", "height", "width"], a),
        "u": (list(coords.keys()), u),
        "t": (["time"], sol_t.cpu().numpy())
    }
    
    attributes = {
        "info": "Incompressible Navier-Stokes data",
        "a": "Initial condition",
        "u": "Solution",
        "t": "Time step in simulation",
        "viscosity": viscosity,
        "delta_t": "%.e" % delta_t,
        "simulation T": T,
        "recorded steps": record_steps
    }

    ds = xr.Dataset(
        coords=coords,
        data_vars=data_vars,
        attrs=attributes
    ).chunk(chunkdict)

    dst_path = os.path.join(dst_path, f"ns_r{'%.e' % int(1/viscosity)}_n{N}_t{T}_s{s}.nc")
    print(f"Data successfully built. Writing file to {dst_path}. This can take a while...")
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    ds.to_netcdf(dst_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Incompressible Navier-Stokes data generation script.")
    parser.add_argument("-r", "--resolution", type=int, default=64,
                        help="Spatial resolution of the square simulation field in pixels.")
    parser.add_argument("-n", "--n_samples", type=int, default=1000,
                        help="Number of samples to generate.")
    parser.add_argument("-b", "--batch-size", type=int, default=50,
                        help="Batch size used for data generation.")
    parser.add_argument("-t", "--max-simulation-time", type=int, default=50,
                        help="Temporal resolution of the simulation.")
    parser.add_argument("--delta-t", type=float, default=1e-3,
                        help="Temporal time stepping resolution for the simulation.")
    parser.add_argument("-s", "--record-steps", type=int, default=None,
                        help="Number of snapshots written out from simulation.")
    parser.add_argument("-v", "--viscosity", type=float, default=1e-3,
                        help="Viscosity of the fluid. Small numbers lead to high turbulence.")
    parser.add_argument("-a", "--alpha", type=float, default=2.0,
                        help="Alpha for initial condition.")
    parser.add_argument("--tau", type=float, default=7.0,
                        help="Tau for initial condition.")
    parser.add_argument("--forcing-multiplicator", type=float, default=2.0,
                        help="Multiplicator for the forcing field.")
    parser.add_argument("-d", "--device", type=str, default="cpu",
                        help="The device to run the evaluation. Any of ['cpu' (default), 'cuda', 'mpg'].")
    parser.add_argument("-p", "--dst-path", type=str, default=os.path.join("data", "netcdf", "navier-stokes"),
                        help="The directory where the data will be written to.")
    parser.add_argument("--animate", action="store_true",
                        help="Whether to create a video.mp4 file that animates the first sample.")

    run_args = parser.parse_args()
    generate_data(
        resolution=run_args.resolution,
        n_samples=run_args.n_samples,
        batch_size=run_args.batch_size,
        max_simulation_time=run_args.max_simulation_time,
        delta_t=run_args.delta_t,
        record_steps=run_args.record_steps,
        viscosity=run_args.viscosity,
        alpha=run_args.alpha,
        tau=run_args.tau,
        forcing_multiplicator=run_args.forcing_multiplicator,
        device=run_args.device,
        dst_path=run_args.dst_path,
        animate=run_args.animate

    )
    
    print("Done.")
