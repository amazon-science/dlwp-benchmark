# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Parts of the code in this file have been adapted from ConvLSTM_pytorch repo Copyright (c) Andrea Palazzi

import numpy as np
import torch as th
import torch.nn as nn
import einops
from utils import CylinderPad
from utils import HEALPixLayer


class ConvLSTMCell(nn.Module):
    """
    A ConvLSTM implementation using Conv1d instead of Conv2d operations.
    """

    def __init__(
        self,
        batch_size: int,
        input_size: int,
        hidden_size: int,
        height: int,
        width: int,
        device: th.device,
        bias=True,
        mesh: str = "equirectangular"
    ):
        super(ConvLSTMCell, self).__init__()

        # Parameters
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.height = height
        self.width = width
        self.bias = bias
        self.device = device

        # Hidden (h) and cell (c) states
        self.h = th.zeros(size=(batch_size, hidden_size, height, width), device=device)
        self.c = th.zeros(size=(batch_size, hidden_size, height, width), device=device)

        # Convolution weights
        if mesh == "equirectangular":
            self.conv = nn.Sequential(
                CylinderPad(padding=1),
                nn.Conv2d(
                    in_channels=input_size + hidden_size,
                    out_channels=hidden_size*4,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    bias=bias
                )
            )
        elif mesh == "healpix":
            self.conv = HEALPixLayer(
                layer=nn.Conv2d,
                in_channels=input_size + hidden_size,
                out_channels=hidden_size*4,
                kernel_size=3,
                padding=1
            )

    def reset_states(self, batch_size):
        if self.batch_size == batch_size:
            self.h = th.zeros_like(self.h)
            self.c = th.zeros_like(self.c)
        else:
            self.batch_size = batch_size
            self.h = th.zeros(size=(batch_size, self.hidden_size, self.height, self.width), device=self.device)
            self.c = th.zeros(size=(batch_size, self.hidden_size, self.height, self.width), device=self.device)

    def reset_parameters(self):
        # Uniform distribution initialization of lstm weights with respect to
        # the number of lstm cells in the layer
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, h_prev=None, c_prev=None):
        """
        ...
        """

        # Set the previous hidden and cell states if not provided
        h_prev = self.h*1 if h_prev is None else h_prev
        c_prev = self.c*1 if c_prev is None else c_prev

        # Perform input and recurrent convolutions
        conv_res = self.conv(th.cat((x, h_prev), dim=1))

        # Split result into input and gate activations
        netin, igate, fgate, ogate = th.split(conv_res, self.hidden_size, dim=1)

        # Compute input and gate activations
        act_input = th.tanh(netin)
        act_igate = th.sigmoid(igate)
        act_fgate = th.sigmoid(fgate)
        act_ogate = th.sigmoid(ogate)

        # Compute the new cell and hidden states
        c_curr = act_fgate * c_prev + act_igate * act_input
        h_curr = act_ogate * th.tanh(c_curr)

        # Update the hidden and cells states
        self.h = h_curr
        self.c = c_curr

        return h_curr, c_curr


class ConvLSTM(nn.Module):
    """
    A ConvLSTM implementation using Conv1d instead of Conv2d operations.
    """

    def __init__(
        self,
        batch_size: int = 16,
        constant_channels: int = 4,
        prescribed_channels: int = 0,
        prognostic_channels: int = 1,
        hidden_sizes: list = [16, 16],
        height: int = 32,
        width: int = 64,
        device: th.device = th.device("cpu"),
        bias: bool = True,
        context_size: int = 1,
        mesh: str = "equirectangular",
        **kwargs
    ):
        super(ConvLSTM, self).__init__()

        # Parameters
        self.batch_size = batch_size
        self.hidden_sizes = hidden_sizes
        self.height = height
        self.width = width
        self.bias = bias
        self.context_size = context_size
        self.mesh = mesh

        in_size = constant_channels + (prescribed_channels+prognostic_channels)

        if mesh == "equirectangular":
            self.encoder = th.nn.Sequential(
                CylinderPad(padding=1),
                th.nn.Conv2d(in_channels=in_size, out_channels=hidden_sizes[0], kernel_size=3, padding=0),
                th.nn.Tanh(),
                CylinderPad(padding=1),
                th.nn.Conv2d(in_channels=hidden_sizes[0], out_channels=hidden_sizes[0], kernel_size=3, padding=0),
                th.nn.Tanh(),
                CylinderPad(padding=1),
                th.nn.Conv2d(in_channels=hidden_sizes[0], out_channels=hidden_sizes[0], kernel_size=3, padding=0),
            )
        elif mesh == "healpix":
            self.encoder = th.nn.Sequential(
                HEALPixLayer(layer=th.nn.Conv2d, in_channels=in_size, out_channels=hidden_sizes[0], kernel_size=3, padding=1),
                th.nn.Tanh(),
                HEALPixLayer(layer=th.nn.Conv2d, in_channels=hidden_sizes[0], out_channels=hidden_sizes[0], kernel_size=3, padding=1),
                th.nn.Tanh(),
                HEALPixLayer(layer=th.nn.Conv2d, in_channels=hidden_sizes[0], out_channels=hidden_sizes[0], kernel_size=3, padding=1)
            )

        clstm = []
        for layer, hidden_size in enumerate(hidden_sizes):
            clstm.append(ConvLSTMCell(
                batch_size=batch_size,
                input_size=hidden_size,
                hidden_size=hidden_size,
                height=height,
                width=width,
                device=device,
                bias=bias,
                mesh=mesh
                ))
        self.clstm = th.nn.Sequential(*clstm)

        if mesh == "equirectangular":
            self.decoder = th.nn.Sequential(
                CylinderPad(padding=1),
                th.nn.Conv2d(in_channels=hidden_sizes[-1], out_channels=prognostic_channels, kernel_size=3, padding=0),
                )
        elif mesh == "healpix":
            self.decoder = HEALPixLayer(
                layer=th.nn.Conv2d,
                in_channels=hidden_sizes[-1],
                out_channels=prognostic_channels,
                kernel_size=3,
                padding=1
            )

    def _prepare_inputs(
        self,
        constants: th.Tensor = None,
        prescribed: th.Tensor = None,
        prognostic: th.Tensor = None
    ) -> th.Tensor:
        """
        return: Tensor of shape [B, C, H, W] containing constants, prescribed, and prognostic/output variables
        """
        tensors = []
        if constants is not None: tensors.append(constants[:, 0])
        if prescribed is not None: tensors.append(prescribed)
        if prognostic is not None: tensors.append(prognostic)
        return th.cat(tensors, dim=1)

    def forward(
        self,
        constants: th.Tensor = None,
        prescribed: th.Tensor = None,
        prognostic: th.Tensor = None
    ) -> th.Tensor:
        """
        ...
        """
        # constants: [B, 1, C, H, W]
        # prescribed: [B, T, C, H, W]
        # prognostic: [B, T, C, H, W]
        if self.mesh == "healpix":
            B, _, _, F, _, _ = prognostic.shape
        else:
            F = 1

        # Set hidden and cell states of LSTM to zero
        self.reset(batch_size=prognostic.shape[0]*F)
        outs = []

        # Iterate over sequence
        for t in range(prognostic.shape[1]):
            # Prepare input either for teacher forcing or closed loop
            # When t < self.context_size: Teacher forcing -> Override model output with ground truth
            # else: Closed loop -> Feed model output from previous time step into model
            prognostic_t = prognostic[:, t] if t < self.context_size else outs[-1].to(device=prognostic.device)
            x_t = self._prepare_inputs(
                constants=constants,
                prescribed=prescribed[:, t] if prescribed is not None else None,
                prognostic=prognostic_t
            )
            # Forward data through model
            x_t = self.encoder(x_t)
            for clstm_cell in self.clstm:
                x_t, _ = clstm_cell(x_t)
            out = self.decoder(x_t)
            if self.mesh == "healpix": out = einops.rearrange(out, "(b f) c h w -> b c f h w", b=B, f=F)
            x_t = prognostic_t + out
            outs.append(x_t.cpu())

        return th.stack(outs[self.context_size:], dim=1)

    def reset(self, batch_size):
        for clstm_cell in self.clstm:
            clstm_cell.reset_states(batch_size=batch_size)


class ConvLSTMHPX(ConvLSTM):
    """
    A ConvLSTM implementation for operation on the HEALPix mesh.
    """

    def __init__(
        self,
        batch_size: int = 16,
        constant_channels: int = 4,
        prescribed_channels: int = 0,
        prognostic_channels: int = 1,
        hidden_sizes: list = [16, 16],
        height: int = 32,
        width: int = 64,
        device: th.device = th.device("cpu"),
        bias: bool = True,
        context_size: int = 1,
        mesh: str = "healpix",
        **kwargs
    ):
        super(ConvLSTMHPX, self).__init__(
            batch_size=batch_size,
            constant_channels=constant_channels,
            prescribed_channels=prescribed_channels,
            prognostic_channels=prognostic_channels,
            hidden_sizes=hidden_sizes,
            height=height,
            width=width,
            device=device,
            bias=bias,
            context_size=context_size,
            mesh=mesh,
        )

    def _prepare_inputs(
        self,
        constants: th.Tensor = None,
        prescribed: th.Tensor = None,
        prognostic: th.Tensor = None
    ) -> th.Tensor:
        """
        return: Tensor of shape [(F*B), C, H, W] containing constants, prescribed, and prognostic/output variables
        """
        tensors = []
        if constants is not None: tensors.append(einops.rearrange(constants[:, 0], "b c f h w -> (b f) c h w"))
        if prescribed is not None: tensors.append(einops.rearrange(prescribed, "b c f h w -> (b f) c h w"))
        if prognostic is not None: tensors.append(einops.rearrange(prognostic, "b c f h w -> (b f) c h w"))
        return th.cat(tensors, dim=1)
