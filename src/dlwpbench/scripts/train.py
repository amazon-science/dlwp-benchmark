#! /usr/bin/env python3

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import time
import threading

import hydra
import numpy as np
import torch as th
import torch.utils.tensorboard as tb

sys.path.append("")
from data.datasets import *
from models import *
import utils.utils as utils


@hydra.main(config_path='../configs/', config_name='config', version_base=None)
def run_training(cfg):
    """
    Trains a model with the given configuration, printing progress to console and tensorboard and writing checkpoints
    to file.

    :param cfg: The hydra-configuration for the training
    """

    if cfg.seed:
        np.random.seed(cfg.seed)
        th.manual_seed(cfg.seed)
    device = th.device(cfg.device)

    if cfg.verbose: print("\nInitializing model")

    # Set up model
    model = eval(cfg.model.type)(**cfg.model).to(device=device)
    if cfg.verbose:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\tModel {cfg.model.name} has {trainable_params} trainable parameters")

    #print(model)
    #exit()

    # Initialize training modules
    criterion = th.nn.MSELoss()
    optimizer = th.optim.Adam(params=model.parameters(), lr=cfg.training.learning_rate)
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=cfg.training.epochs)

    # Load checkpoint from file to continue training or initialize training scalars
    checkpoint_path = os.path.join("outputs", cfg.model.name, "checkpoints", f"{cfg.model.name}_last.ckpt")
    if cfg.training.continue_training:
        if cfg.verbose: print(f"\tRestoring model from {checkpoint_path}")
        checkpoint = th.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch = checkpoint["epoch"]
        iteration = checkpoint["iteration"]
        best_val_error = checkpoint["best_val_error"]
    else:
        epoch = 0
        iteration = 0
        best_val_error = np.infty

    # Write the model configurations to the model save path
    os.makedirs(os.path.join("outputs", cfg.model.name), exist_ok=True)

    # Initialize tensorbaord to track scalars
    writer = tb.SummaryWriter(log_dir=os.path.join("outputs", cfg.model.name, "tensorboard"))

    if cfg.verbose: print("\nInitializing datasets")

    # Initializing dataloaders for training and validation
    train_dataset = hydra.utils.instantiate(
        cfg.data,
        start_date=cfg.data.train_start_date,
        stop_date=cfg.data.train_stop_date,
        sequence_length=cfg.training.sequence_length
    )
    val_dataset = hydra.utils.instantiate(
        cfg.data,
        start_date=cfg.data.val_start_date,
        stop_date=cfg.data.val_stop_date,
        sequence_length=cfg.validation.sequence_length
    )
    train_dataloader = th.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers
    )
    val_dataloader = th.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=cfg.validation.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers
    )

    # Perform training by iterating over all epochs
    if cfg.verbose: print("\nStart training. Inspect progress via 'tensorboard --logdir outputs'")
    for epoch in range(epoch, cfg.training.epochs):

        # Track epoch and learning rate in tensorboard
        writer.add_scalar(tag="Epoch", scalar_value=epoch, global_step=iteration)
        writer.add_scalar(tag="Learning Rate", scalar_value=optimizer.state_dict()["param_groups"][0]["lr"],
                          global_step=iteration)

        start_time = time.time()

        # Train: iterate over all training samples
        outputs = list()
        targets = list()
        for constants, prescribed, prognostic, target in train_dataloader:
            # Prepare inputs and targets
            split_size = max(1, prognostic.shape[0]//cfg.training.gradient_accumulation_steps)
            constants = constants.to(device=device).split(split_size) if not constants.isnan().any() else None
            prescribed = prescribed.to(device=device).split(split_size) if not prescribed.isnan().any() else None
            prognostic = prognostic.to(device=device).split(split_size)
            target = target.to(device=device).split(split_size)

            # Perform optimization step and record outputs
            optimizer.zero_grad()
            for accum_idx in range(len(prognostic)):
                output = model(
                    constants=constants[accum_idx] if not constants == None else None,
                    prescribed=prescribed[accum_idx] if not prescribed == None else None,
                    prognostic=prognostic[accum_idx]
                )
                train_loss = criterion(output, target[accum_idx])
                train_loss.backward()
                if cfg.training.clip_gradients:
                    curr_lr = optimizer.param_groups[-1]["lr"] if scheduler is None else scheduler.get_last_lr()[0]
                    th.nn.utils.clip_grad_norm_(model.parameters(), curr_lr)
                outputs.append(output.detach().cpu())
                targets.append(target[accum_idx].detach().cpu())
            optimizer.step()
            writer.add_scalar(tag="MSE/training", scalar_value=train_loss, global_step=iteration)
            iteration += 1
        with th.no_grad(): epoch_train_loss = criterion(th.cat(outputs), th.cat(targets)).numpy()

        # Validate (without gradients)
        with th.no_grad():
            outputs = list()
            targets = list()
            for constants, prescribed, prognostic, target in val_dataloader:
                split_size = max(1, prognostic.shape[0]//cfg.validation.gradient_accumulation_steps)
                constants = constants.to(device=device).split(split_size) if not constants.isnan().any() else None
                prescribed = prescribed.to(device=device).split(split_size) if not prescribed.isnan().any() else None
                prognostic = prognostic.to(device=device).split(split_size)
                target = target.to(device=device).split(split_size)
                for accum_idx in range(len(prognostic)):
                    output = model(
                        constants=constants[accum_idx] if not constants == None else None,
                        prescribed=prescribed[accum_idx] if not prescribed == None else None,
                        prognostic=prognostic[accum_idx]
                    )
                    outputs.append(output.cpu())
                    targets.append(target[accum_idx].cpu())
            epoch_val_loss = criterion(th.cat(outputs), th.cat(targets)).numpy()
        writer.add_scalar(tag="MSE/validation", scalar_value=epoch_val_loss, global_step=iteration)

        # Write model checkpoint to file, using a separate thread
        if cfg.training.save_model:
            if epoch_val_loss > best_val_error or epoch == cfg.training.epochs - 1:
                dst_path = checkpoint_path
            else:
                best_val_error = epoch_val_loss
                dst_path = f"{checkpoint_path.replace('last', 'best')}"
            thread = threading.Thread(
                target=utils.write_checkpoint,
                args=(model, optimizer, scheduler, epoch, iteration, best_val_error, dst_path, ))
            thread.start()

        # Print training progress to console
        if cfg.verbose:
            epoch_time = round(time.time() - start_time, 2)
            print(f"Epoch {str(epoch).zfill(3)}/{str(cfg.training.epochs)}\t"
                  f"{epoch_time}s\t"
                  f"MSE train: {'%.2E' % epoch_train_loss}\t"
                  f"MSE val: {'%.2E' % epoch_val_loss}")

        # Update learning rate
        scheduler.step()

    # Wrap up
    try: thread.join(); writer.flush(); writer.close()
    except NameError: pass


if __name__ == "__main__":
    run_training()
    print("Done.")
