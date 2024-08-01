#! /usr/bin/env python3

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Parts of the code in this file have been adapted from dlwp-hpx repo Copyright (c) Matthias Karlbauer

import os
import glob
import numpy as np
import xarray as xr


if __name__ == "__main__":

    src_path = os.path.join("data", "netcdf", "weatherbench")
    dir_paths = glob.glob(os.path.join(src_path, "*"))

    for dir_path in dir_paths:
        dir_name = os.path.basename(dir_path)
        os.makedirs(dir_name.replace("netcdf", "zarr"), exist_ok=True)

        nc_file_paths = np.sort(glob.glob(os.path.join(dir_path, "*")))
        for nc_file_path in nc_file_paths:
            zarr_file_path = nc_file_path.replace("netcdf", "zarr").replace(".nc", ".zarr")
            if os.path.exists(zarr_file_path): continue
            xr.open_dataset(nc_file_path).to_zarr(zarr_file_path).close()
