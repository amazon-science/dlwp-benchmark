# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Parts of the code in this file have been adapted from ConvLSTM_pytorch repo Copyright (c) Andrea Palazzi

import torch as th
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    A ConvLSTM implementation using Conv1d instead of Conv2d operations.
    """

    def __init__(self, batch_size, input_size, hidden_size, height, width, device, bias=True):
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
        self.conv = nn.Conv2d(
            in_channels=input_size + hidden_size,
            out_channels=hidden_size*4,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=bias
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

    def __init__(self, batch_size, input_size, hidden_sizes, height, width, device, bias=True, **kwargs):
        super(ConvLSTM, self).__init__()

        # Parameters
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.height = height
        self.width = width
        self.bias = bias

        self.encoder = th.nn.Sequential(
            th.nn.Conv2d(in_channels=1, out_channels=hidden_sizes[0], kernel_size=3, padding=1, padding_mode="circular"),
            th.nn.Tanh(),
            th.nn.Conv2d(in_channels=hidden_sizes[0], out_channels=hidden_sizes[0], kernel_size=3, padding=1, padding_mode="circular"),
            th.nn.Tanh(),
            th.nn.Conv2d(in_channels=hidden_sizes[0], out_channels=hidden_sizes[0], kernel_size=3, padding=1, padding_mode="circular"),
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
                bias=bias
                ))
        self.clstm = th.nn.Sequential(*clstm)

        self.decoder = th.nn.Sequential(
            th.nn.Conv2d(in_channels=hidden_sizes[-1], out_channels=input_size, kernel_size=3, padding=1, padding_mode="circular"),
            )

    def forward(self, x: float, teacher_forcing_steps: int=50):
        """
        ...
        """

        # Set hidden and cell states of LSTM to zero
        self.reset(batch_size=x.shape[0])
        outs = []

        # Iterate over sequence
        for t in range(x.shape[1]):
            x_t = x[:, t] if t < teacher_forcing_steps else x_t
            x_t = self.encoder(x_t)
            for clstm_cell in self.clstm:
                x_t, _ = clstm_cell(x_t)
            x_t = self.decoder(x_t)
            outs.append(x_t)

        return th.stack(outs, dim=1)

    def reset(self, batch_size):
        for clstm_cell in self.clstm:
            clstm_cell.reset_states(batch_size=batch_size)




