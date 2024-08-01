# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Parts of the code in this file have been adapted from neuraloperator repo Copyright (c) Zongyi Li

import torch as th
import torch.nn as nn
from neuralop.models import FNO, TFNO


class FNOModule(nn.Module):
    """
    A Fourier Neural Operator implementation from the neuralop package:
    https://github.com/neuraloperator/neuraloperator/tree/main
    """

    def __init__(self, n_modes, in_channels, hidden_channels, lifting_channels, projection_channels, out_channels, n_layers, bias=True, **kwargs):
        super(FNOModule, self).__init__()

        self.fno = FNO(
            n_modes=n_modes,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            out_channels=out_channels,
            n_layers=n_layers,
            )

    def forward(self, x: th.Tensor, teacher_forcing_steps: int = 50) -> th.Tensor:
        """
        ...
        """
        outs = []

        # Iterate over sequence
        for t in range(x.shape[1]):
            x_t = x[:, t] if t < teacher_forcing_steps else x_t
            x_t = self.fno(x_t)
            outs.append(x_t)

        return th.stack(outs, dim=1)


class FNOContextModule(nn.Module):
    """
    A Fourier Neural Operator implementation from the neuralop package:
    https://github.com/neuraloperator/neuraloperator/tree/main
    """

    def __init__(self, n_modes, in_channels, hidden_channels, lifting_channels, projection_channels, out_channels,
                 n_layers, max_n_modes=None, bias=True, **kwargs):
        super(FNOContextModule, self).__init__()

        self.context_size = n_modes[0]

        self.fno = FNO(
            n_modes=n_modes,
            max_n_modes=max_n_modes,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            out_channels=out_channels,
            n_layers=n_layers,
            )

    def forward(
        self,
        x: th.Tensor,
        teacher_forcing_steps: int = 15
    ) -> th.Tensor:
        # x: [B, T, D, H, W]
        outs = []

        for t in range(x.shape[1]):

            # Prepare input, depending on teacher forcing or closed loop
            if t < teacher_forcing_steps:  # Teacher forcing: Feed observations into the model
                x_t = x[:, max(0, t-(self.context_size-1)):t+1]
            else:  # Closed loop: Feed output from previous time step as input into the model
                if self.context_size == 0:
                    x_t = out
                else:
                    ts = max(0, (teacher_forcing_steps - t - 1) + self.context_size)
                    x_obs = x[:, teacher_forcing_steps-ts:teacher_forcing_steps]
                    x_out = th.stack(outs[-(self.context_size-ts):], dim=1)
                    x_t = th.cat([x_obs, x_out], axis=1)
            x_t = th.transpose(x_t, dim0=1, dim1=2)

            # Forward input through model if context size has been reached
            if t < self.context_size - 1:
                # Not forwarding through model because context size not yet reached in output. Instead, consider the
                # most recent input as output.
                out = x_t[:, :, -1]
            else:
                out = self.fno(x_t)[:, :, -1]

            outs.append(out)
        
        return th.stack(outs, dim=1)


class FNO3DModule(nn.Module):
    """
    A Fourier Neural Operator implementation from the neuralop package:
    https://github.com/neuraloperator/neuraloperator/tree/main
    """

    def __init__(self, n_modes, in_channels, hidden_channels, lifting_channels, projection_channels, out_channels,
                 n_layers, bias=True, **kwargs):
        super(FNO3DModule, self).__init__()

        self.n_layers = n_layers

        self.fno = FNO(
            n_modes=n_modes,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            out_channels=out_channels,
            n_layers=n_layers,
        )
        
        t, h, w = n_modes
        self.output_shape = [(t, h, w) for _ in range(self.n_layers)]

    def forward(
        self,
        x: th.Tensor,
        teacher_forcing_steps: int = 15
    ) -> th.Tensor:
        # x: [B, T, D, H, W]

        # Determine output time steps required to map from t_in to t_out, such that t_in + t_out = t
        _, t, _, h, w = x.shape
        if self.output_shape[0][0] != t-teacher_forcing_steps: self.output_shape = [(t-teacher_forcing_steps, h, w) for _ in range(self.n_layers)]

        outs = self.fno(
            x=x[:, :10].transpose(dim0=1, dim1=2),
            output_shape=self.output_shape
        ).transpose(dim0=1, dim1=2)
        
        return th.cat([x[:, :teacher_forcing_steps], outs], dim=1)
        #return outs


class TFNO3DModule(nn.Module):
    """
    A Fourier Neural Operator implementation from the neuralop package:
    https://github.com/neuraloperator/neuraloperator/tree/main
    """

    def __init__(self, n_modes, in_channels, hidden_channels, lifting_channels, projection_channels, out_channels,
                 n_layers, bias=True, rank=1.0, **kwargs):
        super(TFNO3DModule, self).__init__()

        self.n_layers = n_layers

        self.tfno = TFNO(
            n_modes=n_modes,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            out_channels=out_channels,
            n_layers=n_layers,
            rank=rank
        )
        
        t, h, w = n_modes
        self.output_shape = [(t, h, w) for _ in range(self.n_layers)]

    def forward(
        self,
        x: th.Tensor,
        teacher_forcing_steps: int = 15
    ) -> th.Tensor:
        # x: [B, T, D, H, W]

        # Determine output time steps required to map from t_in to t_out, such that t_in + t_out = t
        _, t, _, h, w = x.shape
        if self.output_shape[0][0] != t-teacher_forcing_steps: self.output_shape = [(t-teacher_forcing_steps, h, w) for _ in range(self.n_layers)]

        outs = self.tfno(
            x=x[:, :10].transpose(dim0=1, dim1=2),
            output_shape=self.output_shape
        ).transpose(dim0=1, dim1=2)
        
        return th.cat([x[:, :teacher_forcing_steps], outs], dim=1)


class TFNO2DModule(nn.Module):
    """
    A Fourier Neural Operator implementation from the neuralop package:
    https://github.com/neuraloperator/neuraloperator/tree/main
    """

    def __init__(self, n_modes, in_channels, hidden_channels, lifting_channels, projection_channels, out_channels,
                 n_layers, max_n_modes=None, rank=1.0, bias=True, context_size=10, **kwargs):
        super(TFNO2DModule, self).__init__()

        self.context_size = context_size

        self.fno = FNO(
            n_modes=n_modes,
            max_n_modes=max_n_modes,
            in_channels=in_channels*context_size,
            hidden_channels=hidden_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            out_channels=out_channels,
            n_layers=n_layers,
            rank=rank
            )

    def forward(
        self,
        x: th.Tensor,
        teacher_forcing_steps: int = 10
    ) -> th.Tensor:
        # x: [B, T, D, H, W]
        outs = []

        for t in range(x.shape[1]):

            # Prepare input, depending on teacher forcing or closed loop
            if t < teacher_forcing_steps:  # Teacher forcing: Feed observations into the model
                x_t = x[:, max(0, t-(self.context_size-1)):t+1]
            else:  # Closed loop: Feed output from previous time step as input into the model
                if self.context_size == 0:
                    x_t = out
                else:
                    ts = max(0, (teacher_forcing_steps - t - 1) + self.context_size)
                    x_obs = x[:, teacher_forcing_steps-ts:teacher_forcing_steps]
                    x_out = th.stack(outs[-(self.context_size-ts):], dim=1)
                    x_t = th.cat([x_obs, x_out], axis=1)

            # Forward input through model if context size has been reached
            if t < self.context_size - 1:
                # Not forwarding through model because context size not yet reached in output. Instead, consider the
                # most recent input as output.
                out = x_t[:, -1]
            else:
                x_t = x_t.flatten(start_dim=1, end_dim=2)
                out = self.fno(x_t)

            outs.append(out)
        
        return th.stack(outs, dim=1)
