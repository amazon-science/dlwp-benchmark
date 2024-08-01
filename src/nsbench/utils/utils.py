#! env/bin/python3

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch as th
import datetime as dt


def write_checkpoint(
		model,
		optimizer,
		scheduler,
		epoch: int,
		iteration: int,
		best_val_error: float,
		dst_path: str
	):
	"""
	Writes a checkpoint including model, optimizer, and scheduler state dictionaries along with current epoch,
	iteration, and best validation error to file.
	
	:param model: The network model
	:param optimizer: The pytorch optimizer
	:param scheduler: The pytorch learning rate scheduler
	:param epoch: Current training epoch
	:param iteration: Current training iteration
	:param best_val_error: The best validation error of the current training
	:param dst_path: Path where the checkpoint is written to
	"""
	os.makedirs(os.path.dirname(dst_path), exist_ok=True)
	th.save(obj={"model_state_dict": model.state_dict(),
				 "optimizer_state_dict": optimizer.state_dict(),
				 "scheduler_state_dict": scheduler.state_dict(),
				 "epoch": epoch + 1,
				 "iteration": iteration,
				 "best_val_error": best_val_error},
			f=dst_path)


def int_to_datetime(date: int):
	"""
	Converts an integer date represantion from YYYYMMDD into an according datetime object

	:param date: The date in integer representation
	"""
	date = str(date)
	year = int(date[:4])
	month = int(date[4:6])
	day = int(date[6:8])
	return dt.datetime(year=year, month=month, day=day)
