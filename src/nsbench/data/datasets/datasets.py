#! /usr/bin/env python3

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch as th
import xarray as xr


class NavierStokesDataset(th.utils.data.Dataset):
    def __init__(
            self,
            data_path: str,
            sequence_length: int = 15,
            noise: float = 0.0,
            normalize: bool = False,
            downscale_factor: int = None
        ):
        """
        Constructor of a pytorch dataset module.
        """

        self.sequence_length = sequence_length
        self.noise = noise
        self.normalize = normalize
        self.downscale_factor = downscale_factor

        self.ds = xr.open_dataset(data_path)
        self.mean = self.ds.u.mean().values
        self.std = self.ds.u.std().values

        # Downscale dataset if desired
        if downscale_factor: self.ds = self.ds.coarsen(height=downscale_factor, width=downscale_factor).mean()
        
    def __len__(self):
        return self.ds.sizes["sample"]

    def __getitem__(self, item):
        r = np.random.randint(0, self.ds.sizes["time"]-self.sequence_length+1)
        x = np.float32(self.ds.u.isel(sample=item, time=slice(r, r+self.sequence_length-1)))
        x = x + np.float32(np.random.randn(*x.shape)*self.noise)
        y = np.float32(self.ds.u.isel(sample=item, time=slice(1+r, r+self.sequence_length)))
        return x, y
