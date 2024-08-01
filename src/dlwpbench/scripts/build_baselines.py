#! /usr/bin/env python3

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import glob
import tqdm
import numpy as np
import pandas as pd
import xarray as xr


def write_to_file(ds_inits: xr.Dataset, ds_outputs: xr.Dataset, ds_targets: xr.Dataset, dst_path: str):
	print("Writing to file...")
	os.makedirs(dst_path, exist_ok=True)
	ds_inits.to_netcdf(os.path.join(dst_path, "inits.nc"))
	ds_outputs.to_netcdf(os.path.join(dst_path, "outputs.nc"))
	ds_targets.to_netcdf(os.path.join(dst_path, "targets.nc"))


def persistence_forecast(ds_inits: xr.Dataset, ds_outputs: xr.Dataset, ds_targets: xr.Dataset):
	print("Creating persistence forecast...")
	# Compute persistence per variable
	for vname in tqdm.tqdm(list(ds_inits.keys()), desc="Overwriting ds_outputs with persistence"):
		_, ds_inits_broadcast = xr.broadcast(ds_outputs[vname], ds_inits[vname])
		ds_outputs[vname].values = ds_inits_broadcast.values

	# Write dataset to file
	dst_path = os.path.join("outputs", "persistence", "evaluation")
	write_to_file(ds_inits=ds_inits, ds_outputs=ds_outputs, ds_targets=ds_targets, dst_path=dst_path)
	

def climatology_forecast(ds_inits: xr.Dataset, ds_outputs: xr.Dataset, ds_targets: xr.Dataset):
	print("Creating climatology forecast...")
	# Specs according to climatological standard normal from 1981 through 2010
	# https://en.wikipedia.org/wiki/Climatological_normal
	start_date = "1981-01-01"
	stop_date = "2010-12-31"
	data_src_path = os.path.join("data", "zarr", "weatherbench")
	zarr_file_paths = glob.glob(os.path.join(data_src_path, "**", "*.zarr"), recursive=True)

	# Lazy load all data to base climatology calculation on
	print("Lazy loading all data")
	ds_climatology = xr.open_mfdataset(
		zarr_file_paths,
		engine="zarr"
	).chunk(dict(time=24, lat=32, lon=64)).sel(time=slice(start_date, stop_date))

	# Calculate climatological standard normal per variable over specified period
	for vname in list(ds_inits.keys()):
		
		# Select data array variable from climatology dataset
		if vname in list(ds_climatology.keys()):
			da_climatology = ds_climatology[vname]
		else:
			v, l = re.match(r"([a-z]+)([0-9]+)", vname, re.I).groups()
			da_climatology = ds_climatology[v].sel(level=int(l))

		print(f"Computing climatology for {vname} and loading it to memory")
		da_climatology = da_climatology.groupby(da_climatology.time.dt.month).mean().load()
		for s_idx, s in enumerate(tqdm.tqdm(ds_outputs.sample, desc="Overwriting ds_outputs with climatology")):
			for t_idx, t in enumerate(ds_outputs.time):
				ds_out_sel = ds_outputs[vname][s_idx, t_idx]
				month = pd.Timestamp((ds_out_sel.sample + ds_out_sel.time).values).month
				ds_outputs[vname][s_idx, t_idx] = da_climatology.sel(month=month)
	
	dst_path = os.path.join("outputs", "climatology", "evaluation")
	write_to_file(ds_inits=ds_inits, ds_outputs=ds_outputs, ds_targets=ds_targets, dst_path=dst_path)


if __name__ == "__main__":
	# Specs
	src_model_name = "clstm1m_cyl_4x57_v0"
	src_path = os.path.join("outputs", src_model_name, "evaluation")
	
	# Load data	
	ds_inits = xr.open_dataset(os.path.join(src_path, "inits.nc"))
	ds_outputs = xr.open_dataset(os.path.join(src_path, "outputs.nc"))
	ds_targets = xr.open_dataset(os.path.join(src_path, "targets.nc"))

	#persistence_forecast(ds_inits=ds_inits, ds_outputs=ds_outputs, ds_targets=ds_targets)
	climatology_forecast(ds_inits=ds_inits, ds_outputs=ds_outputs, ds_targets=ds_targets)
