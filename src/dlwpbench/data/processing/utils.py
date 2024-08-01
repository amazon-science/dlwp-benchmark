#! /usr/bin/env python3

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Parts of the code in this file have been adapted from dlwp-hpx repo Copyright (c) Matthias Karlbauer


def to_chunked_dataset(ds, chunking):
    """
    Create a chunked copy of a Dataset with proper encoding for netCDF export.

    :param ds: xarray.Dataset
    :param chunking: dict: chunking dictionary as passed to xarray.Dataset.chunk()
    :return: xarray.Dataset: chunked copy of ds with proper encoding
    """
    chunk_dict = dict(ds.dims)
    chunk_dict.update(chunking)
    ds_new = ds.chunk(chunk_dict)
    for var in ds_new.data_vars:
        ds_new[var].encoding['contiguous'] = False
        ds_new[var].encoding['original_shape'] = ds_new[var].shape
        try:
            ds_new[var].encoding['chunksizes'] = tuple([c[0] for c in ds_new[var].chunks])
        except TypeError:
            pass  # Constants have variables that cannot be iterated; these are skipped here
    return ds_new