#! /usr/bin/env python3

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import xarray as xr


if __name__ == "__main__":

	# Specs
	resolution = 64
	tf_steps = 10  # Number of teacher forcing steps
	#src_model_name = "clstm64_16-16_sl50_tf10"
	src_model_name = "swint64_d8_l1x1_h1x1_sl30_tf10_noise0_r1e4"
	
	# Load data	
	ds = xr.open_dataset(os.path.join("outputs", src_model_name, "forecast.nc"))
	inputs = np.copy(ds.inputs.values)
	outputs = np.copy(ds.outputs.values)

	# Overwrite outputs after tf_steps with last observation
	outputs[:, :tf_steps] = inputs[:, :tf_steps]
	outputs[:, tf_steps:] = inputs[:, tf_steps-1:tf_steps]
	ds.outputs.values = outputs

	# Write dataset to file
	dst_path = os.path.join("outputs", f"persistence_{resolution}x{resolution}_{tf_steps}_r1e4")
	os.makedirs(dst_path, exist_ok=True)
	ds.to_netcdf(os.path.join(dst_path, "forecast.nc"))
