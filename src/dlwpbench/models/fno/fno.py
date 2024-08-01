# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Parts of the code in this file have been adapted from neuraloperator repo Copyright (c) Zongyi Li

import torch as th
import torch.nn as nn
from neuralop.models import FNO, TFNO
from torch_harmonics.examples.sfno import SphericalFourierNeuralOperatorNet as SFNO
import einops


class FNO2DModule(nn.Module):
    """
    A Fourier Neural Operator implementation from the neuralop package:
    https://github.com/neuraloperator/neuraloperator/tree/main
    """

    def __init__(
        self,
        n_modes: list = [12, 12],
        constant_channels: int = 4,
        prescribed_channels: int = 1,
        prognostic_channels: int = 8,
        hidden_channels: int = 32,
        lifting_channels: int = 256,
        projection_channels: int = 256,
        n_layers: int = 4,
        max_n_modes: int = None,
        bias: bool = True,
        context_size: int = 10,
        **kwargs
    ):
        super(FNO2DModule, self).__init__()

        self.context_size = context_size
        in_channels = constant_channels + (prescribed_channels+prognostic_channels)*context_size

        self.fno = FNO(
            n_modes=n_modes,
            max_n_modes=max_n_modes,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            out_channels=prognostic_channels,
            n_layers=n_layers
            )

    def _prepare_inputs(
        self,
        constants: th.Tensor = None,
        prescribed: th.Tensor = None,
        prognostic: th.Tensor = None
    ) -> th.Tensor:
        """
        return: Tensor of shape [B, (T*C), H, W] containing constants, prescribed, and prognostic/output variables
        """
        tensors = []
        if constants is not None: tensors.append(constants[:, 0])
        if prescribed is not None: tensors.append(einops.rearrange(prescribed, "b t c h w -> b (t c) h w"))
        if prognostic is not None: tensors.append(einops.rearrange(prognostic, "b t c h w -> b (t c) h w"))
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
        outs = []

        for t in range(self.context_size, prognostic.shape[1]):
            
            t_start = max(0, t-(self.context_size))
            if t == self.context_size:
                # Initial condition
                prognostic_t = prognostic[:, t_start:t]
                x_t = self._prepare_inputs(
                    constants=constants,
                    prescribed=prescribed[:, t_start:t] if prescribed is not None else None,
                    prognostic=prognostic_t
                )
            else:
                # In case of context_size > 1, blend prognostic input with outputs from previous time steps
                prognostic_t = th.cat(
                    tensors=[prognostic[:, t_start:self.context_size],        # Prognostic input before context_size
                             th.stack(outs, dim=1)[:, -self.context_size:]].to(device=prognostic.device),  # Outputs since context_size
                    dim=1
                )
                x_t = self._prepare_inputs(
                    constants=constants,
                    prescribed=prescribed[:, t-self.context_size:t] if prescribed is not None else None,
                    prognostic=prognostic_t
                )

            # Forward input through model
            out = prognostic_t[:, -1] + self.fno(x_t)
            outs.append(out.cpu())
        
        return th.stack(outs, dim=1)


class TFNO2DModule(FNO2DModule):
    """
    A Fourier Neural Operator implementation from the neuralop package:
    https://github.com/neuraloperator/neuraloperator/tree/main
    """

    def __init__(
        self,
        n_modes: list = [12, 12],
        constant_channels: int = 4,
        prescribed_channels: int = 1,
        prognostic_channels: int = 8,
        hidden_channels: int = 32,
        lifting_channels: int = 256,
        projection_channels: int = 256,
        n_layers: int = 4,
        max_n_modes: int = None,
        rank: float = 1.0,
        bias: bool = True,
        context_size: int = 10,
        **kwargs
    ):
        super(TFNO2DModule, self).__init__()

        self.context_size = context_size
        in_channels = constant_channels + (prescribed_channels+prognostic_channels)*context_size

        self.fno = TFNO(
            n_modes=n_modes,
            max_n_modes=max_n_modes,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            out_channels=prognostic_channels,
            n_layers=n_layers,
            rank=rank
            )


class SFNO2DModule(nn.Module):
    """
    A Spherical Fourier Neural Operator implementation from the torch harmonics package:
    https://github.com/NVIDIA/torch-harmonics/tree/main    
    """

    def __init__(
        self,
        constant_channels: int = 4,
        prescribed_channels: int = 1,
        prognostic_channels: int = 8,
        spectral_transform='sht',
        grid="legendre-gauss",
        num_layers=4,
        scale_factor=3,
        embed_dim=256,
        operator_type='driscoll-healy',
        context_size: int = 1,
        height: int = 32,
        width: int = 64,
        hard_thresholding_fraction: float = 1.0,
        factorization: str = None,
        rank: float = 1.0,
        big_skip: bool = False,
        pos_embed: bool = False,
        use_mlp: bool = False,
        normalization_layer: str = None,
        **kwargs
    ):
        super(SFNO2DModule, self).__init__()

        self.context_size = context_size
        in_channels = constant_channels + (prescribed_channels+prognostic_channels)*context_size
        
        self.sfno = SFNO(
            in_chans=in_channels,
            out_chans=prognostic_channels,
            spectral_transform=spectral_transform,
            img_size=(height, width),
            grid=grid,
            num_layers=num_layers,
            scale_factor=scale_factor,
            embed_dim=embed_dim,
            operator_type=operator_type,
            hard_thresholding_fraction=hard_thresholding_fraction,
            factorization=factorization,
            rank=rank,
            big_skip=big_skip,
            pos_embed=pos_embed,
            use_mlp=use_mlp,
            normalization_layer=normalization_layer
        )

    def _prepare_inputs(
        self,
        constants: th.Tensor = None,
        prescribed: th.Tensor = None,
        prognostic: th.Tensor = None
    ) -> th.Tensor:
        """
        return: Tensor of shape [B, (T*C), H, W] containing constants, prescribed, and prognostic/output variables
        """
        tensors = []
        if constants is not None: tensors.append(constants[:, 0])
        if prescribed is not None: tensors.append(einops.rearrange(prescribed, "b t c h w -> b (t c) h w"))
        if prognostic is not None: tensors.append(einops.rearrange(prognostic, "b t c h w -> b (t c) h w"))
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
        outs = []

        for t in range(self.context_size, prognostic.shape[1]):
            
            t_start = max(0, t-(self.context_size))
            if t == self.context_size:
                # Initial condition
                prognostic_t = prognostic[:, t_start:t]
                x_t = self._prepare_inputs(
                    constants=constants,
                    prescribed=prescribed[:, t_start:t] if prescribed is not None else None,
                    prognostic=prognostic_t
                )
            else:
                # In case of context_size > 1, blend prognostic input with outputs from previous time steps
                prognostic_t = th.cat(
                    tensors=[prognostic[:, t_start:self.context_size],        # Prognostic input before context_size
                             th.stack(outs, dim=1)[:, -self.context_size:]].to(device=prognostic.device),  # Outputs since context_size
                    dim=1
                )
                x_t = self._prepare_inputs(
                    constants=constants,
                    prescribed=prescribed[:, t-self.context_size:t] if prescribed is not None else None,
                    prognostic=prognostic_t
                )

            # Forward input through model
            out = prognostic_t[:, -1] + self.sfno(x_t)
            outs.append(out.cpu())
        
        return th.stack(outs, dim=1)
