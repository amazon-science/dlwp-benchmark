# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch as th
import einops


class UNet(th.nn.Module):
    """
    A UNet implementation.
    """

    def __init__(
        self, 
        in_channels: int = 2,
        hidden_channels: list = [8, 16, 32],
        out_channels: int = 1,
        n_convolutions: int = 2,
        activation: th.nn.Module = th.nn.ReLU(),
        padding_mode: str = "zeros",
        context_size: int = 1,
        **kwargs
    ):
        super(UNet, self).__init__()
        if isinstance(activation, str): activation = eval(activation)

        self.context_size = context_size

        self.encoder = UNetEncoder(
            in_channels=in_channels*context_size,
            hidden_channels=hidden_channels,
            n_convolutions=n_convolutions,
            activation=activation,
            padding_mode=padding_mode
        )
        self.decoder = UNetDecoder(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            n_convolutions=n_convolutions,
            activation=activation,
            padding_mode=padding_mode
        )

    def forward(self, x: th.Tensor, teacher_forcing_steps: int=50) -> th.Tensor:
        """
        ...
        """
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
                    x_t = th.cat([x_obs, x_out], dim=1)

            # Forward input through model if context size has been reached
            if t < self.context_size - 1:
                # Not forwarding through model because context size not yet reached in output. Instead, consider the
                # most recent input as output.
                out = x_t[:, -1]
            else:
                enc = self.encoder(einops.rearrange(x_t, "b t d h w -> b (t d) h w"))
                out = x_t[:, -1] + self.decoder(x=enc[-1], skips=enc[::-1])

            outs.append(out)
        
        return th.stack(outs, dim=1)


class UNetEncoder(th.nn.Module):

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: list = [8, 16, 32],
        n_convolutions: int = 2,
        activation: th.nn.Module = th.nn.ReLU(),
        padding_mode: str = "zeros"
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
                layer.append(th.nn.Conv2d(
                    in_channels=c_in if n_conv == 0 else c_out,
                    out_channels=c_out, 
                    kernel_size=3, 
                    padding=1, 
                    padding_mode=padding_mode
                ))

                # Activation function
                layer.append(activation)

            self.layers.append(th.nn.Sequential(*layer))

        self.layers = th.nn.ModuleList(self.layers)

    def forward(self, x: th.Tensor) -> list():
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
        padding_mode: str = "zeros"
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
                layer.append(th.nn.Conv2d(
                    in_channels=c_in_ if n_conv == 0 else c_out,
                    out_channels=c_out,
                    kernel_size=3, 
                    padding=1, 
                    padding_mode=padding_mode
                ))

                # Activation function
                layer.append(activation)

            # Apply upsampling if not in top-most layer
            if c_idx < len(hidden_channels)-1:
                layer.append(th.nn.ConvTranspose2d(
                    in_channels=c_out,
                    out_channels=hidden_channels[c_idx+1],
                    kernel_size=2,
                    stride=2,
                    # padding_mode=padding_mode  # Support only 'zero' padding
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
    padding_mode = "circular"
    context_size = 2
    teacher_forcing_steps = 15

    model = UNet(
        name="model_name",
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        n_convolutions=n_convolutions,
        activation=activation,
        padding_mode=padding_mode,
        context_size=context_size
    )

    x = th.randn(4, 25, in_channels, 64, 64)  # B, T, C, H, W
    y_hat = model(x=x, teacher_forcing_steps=teacher_forcing_steps)
    print(y_hat.shape)
