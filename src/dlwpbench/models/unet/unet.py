# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch as th
import einops
from utils import CylinderPad
from utils import HEALPixLayer


class UNet(th.nn.Module):
    """
    A UNet implementation.
    """

    def __init__(
        self, 
        constant_channels: int = 4,
        prescribed_channels: int = 0,
        prognostic_channels: int = 1,
        hidden_channels: list = [8, 16, 32],
        n_convolutions: int = 2,
        activation: th.nn.Module = th.nn.ReLU(),
        context_size: int = 1,
        mesh: str = "equirectangular",
        **kwargs
    ):
        super(UNet, self).__init__()
        if isinstance(activation, str): activation = eval(activation)

        self.context_size = context_size
        self.mesh = mesh
        in_channels = constant_channels + (prescribed_channels+prognostic_channels)*context_size

        self.encoder = UNetEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            n_convolutions=n_convolutions,
            activation=activation,
            mesh=mesh
        )
        self.decoder = UNetDecoder(
            hidden_channels=hidden_channels,
            out_channels=prognostic_channels,
            n_convolutions=n_convolutions,
            activation=activation,
            mesh=mesh
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
        # Shapes of inputs, where (F) is the optional face dimension when using HEALPix data
        # constants: [B, 1, C, (F), H, W]
        # prescribed: [B, T, C, (F), H, W]
        # prognostic: [B, T, C, (F), H, W]
        if self.mesh == "healpix": B, _, _, F, _, _ = prognostic.shape
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
                             th.stack(outs, dim=1)[:, -self.context_size:]],  # Outputs since context_size
                    dim=1
                )
                x_t = self._prepare_inputs(
                    constants=constants,
                    prescribed=prescribed[:, t-self.context_size:t] if prescribed is not None else None,
                    prognostic=prognostic_t
                )

            # Forward input through model
            enc = self.encoder(x_t)
            out = self.decoder(x=enc[-1], skips=enc[::-1])
            if self.mesh == "healpix": out = einops.rearrange(out, "(b f) tc h w -> b tc f h w", b=B, f=F)
            out = prognostic_t[:, -1] + out
            outs.append(out)
        
        return th.stack(outs, dim=1)


class UNetHPX(UNet):

    def __init__(
        self,
        constant_channels: int = 4,
        prescribed_channels: int = 0,
        prognostic_channels: int = 1,
        hidden_channels: list = [8, 16, 32],
        n_convolutions: int = 2,
        activation: th.nn.Module = th.nn.ReLU(),
        context_size: int = 1,
        mesh: str = "healpix",
        **kwargs
    ):
        super(UNetHPX, self).__init__(
            constant_channels=constant_channels,
            prescribed_channels=prescribed_channels,
            prognostic_channels=prognostic_channels,
            hidden_channels=hidden_channels,
            n_convolutions=n_convolutions,
            activation=activation,
            context_size=context_size,
            mesh=mesh,
            kwargs=kwargs
        )
    
    def _prepare_inputs(
        self,
        constants: th.Tensor = None,
        prescribed: th.Tensor = None,
        prognostic: th.Tensor = None
    ) -> th.Tensor:
        """
        return: Tensor of shape [(B*F), (T*C), H, W] containing constants, prescribed, and prognostic/output variables
        """
        tensors = []
        if constants is not None: tensors.append(einops.rearrange(constants[:, 0], "b c f h w -> (b f) c h w"))
        if prescribed is not None: tensors.append(einops.rearrange(prescribed, "b t c f h w -> (b f) (t c) h w"))
        if prognostic is not None: tensors.append(einops.rearrange(prognostic, "b t c f h w -> (b f) (t c) h w"))
        return th.cat(tensors, dim=1)


class UNetEncoder(th.nn.Module):

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: list = [8, 16, 32],
        n_convolutions: int = 2,
        activation: th.nn.Module = th.nn.ReLU(),
        mesh: str = "equirectangular"
    ):
        super(UNetEncoder, self).__init__()
        self.layers = []

        channels = [in_channels] + hidden_channels

        for c_idx in range(len(channels[:-1])):
            layer = []
            c_in = channels[c_idx]
            c_out = channels[c_idx+1]

            # Apply downsampling prior to convolutions if not in top-most layer
            if c_idx > 0: layer.append(th.nn.AvgPool2d(kernel_size=2, stride=2, padding=0))

            # Perform n convolutions (only half as many in bottom-most layer, since other half is done in decoder)
            n_convs = n_convolutions//2 if c_idx == len(hidden_channels)-1 else n_convolutions
            for n_conv in range(n_convs):
                if mesh == "equirectangular":
                    layer.append(CylinderPad(padding=1))
                    layer.append(th.nn.Conv2d(
                        in_channels=c_in if n_conv == 0 else c_out,
                        out_channels=c_out, 
                        kernel_size=3, 
                        padding=0
                    ))
                elif mesh == "healpix":
                    layer.append(HEALPixLayer(
                        layer=th.nn.Conv2d,
                        in_channels=c_in if n_conv == 0 else c_out,
                        out_channels=c_out,
                        kernel_size=3,
                        padding=1
                    ))
                
                # Activation function
                layer.append(activation)

            self.layers.append(th.nn.Sequential(*layer))

        self.layers = th.nn.ModuleList(self.layers)

    def forward(self, x: th.Tensor) -> list:
        # Store intermediate model outputs (per layer) for skip connections
        outs = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x)
        return outs


class UNetDecoder(th.nn.Module):

    def __init__(
        self,
        hidden_channels: list = [8, 16, 32],
        out_channels: int = 2,
        n_convolutions: int = 2,
        activation: th.nn.Module = th.nn.ReLU(),
        mesh: str = "equirectangular"
    ):
        super(UNetDecoder, self).__init__()
        self.layers = []
        hidden_channels = hidden_channels[::-1]  # Invert as we go up in decoder, i.e., from bottom to top layers

        for c_idx in range(len(hidden_channels)):
            layer = []
            c_in = hidden_channels[c_idx]
            c_out = hidden_channels[c_idx]

            # Perform n convolutions (only half as many in bottom-most layer, since other half is done in encoder)
            n_convs = n_convolutions//2 if c_idx == 0 else n_convolutions
            for n_conv in range(n_convs):
                c_in_ = c_in if c_idx == 0 else 2*hidden_channels[c_idx]  # Skip connection from encoder
                if mesh == "equirectangular":
                    layer.append(CylinderPad(padding=1))
                    layer.append(th.nn.Conv2d(
                        in_channels=c_in_ if n_conv == 0 else c_out,
                        out_channels=c_out,
                        kernel_size=3, 
                        padding=0
                    ))
                elif mesh == "healpix":
                    layer.append(HEALPixLayer(
                        layer=th.nn.Conv2d,
                        in_channels=c_in_ if n_conv == 0 else c_out,
                        out_channels=c_out,
                        kernel_size=3,
                        padding=1
                    ))

                # Activation function
                layer.append(activation)

            # Apply upsampling if not in top-most layer
            if c_idx < len(hidden_channels)-1:
                layer.append(th.nn.ConvTranspose2d(
                    in_channels=c_out,
                    out_channels=hidden_channels[c_idx+1],
                    kernel_size=2,
                    stride=2
                ))

            self.layers.append(th.nn.Sequential(*layer))

        self.layers = th.nn.ModuleList(self.layers)
        
        # Add linear output layer
        self.output_layer = th.nn.Conv2d(
            in_channels=c_out,
            out_channels=out_channels,
            kernel_size=1
        )

    def forward(self, x: th.Tensor, skips: list) -> th.Tensor:
        for l_idx, layer in enumerate(self.layers):
            x = th.cat([skips[l_idx], x], dim=1) if l_idx > 0 else x
            x = layer(x)
        return self.output_layer(x)


if __name__ == "__main__":

    # Demo
    in_channels = 1
    hidden_channels = [8, 16, 32]
    out_channels = 1
    n_convolutions = 2
    activation = th.nn.ReLU()
    context_size = 2
    mesh = "equirectangular"
    teacher_forcing_steps = 15

    model = UNet(
        name="model_name",
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        n_convolutions=n_convolutions,
        activation=activation,
        context_size=context_size,
        mesh=mesh
    )

    x = th.randn(4, 25, in_channels, 32, 64)  # B, T, C, H, W
    y_hat = model(x=x, teacher_forcing_steps=teacher_forcing_steps)
    print(y_hat.shape)
