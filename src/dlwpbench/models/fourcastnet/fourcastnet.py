#! /usr/bin/env python3

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Parts of the code in this file have been adapted from FourCastNet repo Copyright (c) FourCastNet authors

# references:
# https://github.com/NVlabs/AFNO-transformer
# https://github.com/NVlabs/FourCastNet/blob/master/networks/afnonet.py

from functools import partial
from numpy.lib.arraypad import pad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO
from torch_harmonics.examples.sfno import SphericalFourierNeuralOperatorNet as SFNO
from timm.models.layers import DropPath, trunc_normal_
import torch.fft
from einops import rearrange


class PeriodicPad2d(nn.Module):
    """ 
        pad longitudinal (left-right) circular 
        and pad latitude (top-bottom) with zeros
    """
    def __init__(self, pad_width):
       super(PeriodicPad2d, self).__init__()
       self.pad_width = pad_width

    def forward(self, x):
        # pad left and right circular
        out = F.pad(x, (self.pad_width, self.pad_width, 0, 0), mode="circular") 
        # pad top and bottom zeros
        out = F.pad(out, (0, 0, self.pad_width, self.pad_width), mode="constant", value=0) 
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AFNO2D(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes]  = (
            torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes]  = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, H, W // 2 + 1, C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1,2), norm="ortho")
        x = x.type(dtype)

        return x + bias


class FNO2D(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1,
                 n_modes=(12,12), max_n_modes=None, n_layers=1):
        super().__init__()
        self.fno = FNO(
            n_modes=n_modes,
            max_n_modes=max_n_modes,
            in_channels=hidden_size,
            hidden_channels=hidden_size,
            lifting_channels=hidden_size,
            projection_channels=hidden_size,
            out_channels=hidden_size,
            n_layers=n_layers
        )

    def forward(self, x):
        bias = x

        x = x.float()
        #B, H, W, C = x.shape

        x = rearrange(self.fno(rearrange(x, "b h w c -> b c h w")), "b c h w -> b h w c")

        return x + bias


class Block(nn.Module):
    def __init__(
            self,
            dim,
            filter=AFNO2D,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            double_skip=True,
            num_blocks=8,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = filter(dim, num_blocks, sparsity_threshold, hard_thresholding_fraction) 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x

class PrecipNet(nn.Module):
    def __init__(self, params, backbone):
        super().__init__()
        self.params = params
        self.patch_size = (params.patch_size, params.patch_size)
        self.in_chans = params.N_in_channels
        self.out_chans = params.N_out_channels
        self.backbone = backbone
        self.ppad = PeriodicPad2d(1)
        self.conv = nn.Conv2d(self.out_chans, self.out_chans, kernel_size=3, stride=1, padding=0, bias=True)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.backbone(x)
        x = self.ppad(x)
        x = self.conv(x)
        x = self.act(x)
        return x

class AFNONet(nn.Module):
    def __init__(
        self,
        img_height=720,
        img_width=1440,
        patch_size=(16, 16),
        constant_channels: int = 4,
        prescribed_channels: int = 0,
        prognostic_channels: int = 1,
        filter="AFNO2D",
        embed_dim=768,
        depth=12,
        mlp_ratio=4.,
        drop_rate=0.,
        drop_path_rate=0.,
        num_blocks=16,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
        context_size: int = 1,
        use_pos_embed: bool = True,
        **kwargs
    ):
        super().__init__()
        self.img_size = (img_height, img_width)
        self.patch_size = patch_size
        self.in_chans = constant_channels + (prescribed_channels+prognostic_channels)*context_size
        self.out_chans = prognostic_channels
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.context_size = context_size
        self.use_pos_embed = use_pos_embed
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        if use_pos_embed: self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.h = self.img_size[0] // self.patch_size[0]
        self.w = self.img_size[1] // self.patch_size[1]

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, filter=eval(filter), mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
            num_blocks=self.num_blocks, sparsity_threshold=sparsity_threshold, hard_thresholding_fraction=hard_thresholding_fraction) 
        for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, self.out_chans*self.patch_size[0]*self.patch_size[1], bias=False)

        if use_pos_embed: trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        if self.use_pos_embed: x = x + self.pos_embed
        x = self.pos_drop(x)
        
        x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)

        return x
    def _prepare_inputs(
        self,
        constants: torch.Tensor = None,
        prescribed: torch.Tensor = None,
        prognostic: torch.Tensor = None
    ) -> torch.Tensor:
        """
        return: Tensor of shape [B, (T*C), H, W] containing constants, prescribed, and prognostic/output variables
        """
        tensors = []
        if constants is not None: tensors.append(constants[:, 0])
        if prescribed is not None: tensors.append(rearrange(prescribed, "b t c h w -> b (t c) h w"))
        if prognostic is not None: tensors.append(rearrange(prognostic, "b t c h w -> b (t c) h w"))
        return torch.cat(tensors, dim=1)

    def forward(
        self,
        constants: torch.Tensor = None,
        prescribed: torch.Tensor = None,
        prognostic: torch.Tensor = None
    ) -> torch.Tensor:
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
                prognostic_t = torch.cat(
                    tensors=[prognostic[:, t_start:self.context_size],        # Prognostic input before context_size
                             torch.stack(outs, dim=1)[:, -self.context_size:]].to(device=prognostic.device),  # Outputs since context_size
                    dim=1
                )
                x_t = self._prepare_inputs(
                    constants=constants,
                    prescribed=prescribed[:, t-self.context_size:t] if prescribed is not None else None,
                    prognostic=prognostic_t
                )

            # Forward input through model if context size has been reached
            x_t = self.forward_features(x_t)
            x_t = self.head(x_t)
            x_t = rearrange(
                x_t,
                "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                h=self.img_size[0] // self.patch_size[0],
                w=self.img_size[1] // self.patch_size[1],
            )
            out = prognostic_t[:, -1] + x_t
            outs.append(out.cpu())
        
        return torch.stack(outs, dim=1)


class SFNONet(nn.Module):
    def __init__(
        self,
        img_height=720,
        img_width=1440,
        patch_size=(16, 16),
        constant_channels: int = 4,
        prescribed_channels: int = 0,
        prognostic_channels: int = 1,
        spectral_transform='sht',
        grid="legendre-gauss",
        num_layers=4,
        scale_factor=3,
        embed_dim=768,
        operator_type='driscoll-healy',
        drop_rate=0.,
        num_blocks=16,
        hard_thresholding_fraction=1.0,
        factorization: str = None,
        rank: float = 1.0,
        big_skip: bool = False,
        use_pos_embed: bool = True,
        use_mlp: bool = False,
        normalization_layer: str = None,
        context_size: int = 1,
        **kwargs
    ):
        super().__init__()
        self.img_size = (img_height, img_width)
        self.patch_size = patch_size
        self.in_chans = constant_channels + (prescribed_channels+prognostic_channels)*context_size
        self.out_chans = prognostic_channels
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.context_size = context_size
        self.use_pos_embed = use_pos_embed
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        if use_pos_embed: self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.h = self.img_size[0] // self.patch_size[0]
        self.w = self.img_size[1] // self.patch_size[1]

        self.sfno = SFNO(
            in_chans=embed_dim,
            out_chans=embed_dim,
            spectral_transform=spectral_transform,
            img_size=(img_height, img_width),
            grid=grid,
            num_layers=num_layers,
            scale_factor=scale_factor,
            embed_dim=embed_dim,
            operator_type=operator_type,
            hard_thresholding_fraction=hard_thresholding_fraction,
            factorization=factorization,
            rank=rank,
            big_skip=big_skip,
            pos_embed=use_pos_embed,
            use_mlp=use_mlp,
            normalization_layer=normalization_layer
        )

        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, self.out_chans*self.patch_size[0]*self.patch_size[1], bias=False)

        if use_pos_embed: trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        if self.use_pos_embed: x = x + self.pos_embed
        x = self.pos_drop(x)
        x = rearrange(x.reshape(B, self.h, self.w, self.embed_dim), "b h w c -> b c h w")
        x = self.sfno(x)
        x = rearrange(x, "b c h w -> b h w c")
        return x
    
    def _prepare_inputs(
        self,
        constants: torch.Tensor = None,
        prescribed: torch.Tensor = None,
        prognostic: torch.Tensor = None
    ) -> torch.Tensor:
        """
        return: Tensor of shape [B, (T*C), H, W] containing constants, prescribed, and prognostic/output variables
        """
        tensors = []
        if constants is not None: tensors.append(constants[:, 0])
        if prescribed is not None: tensors.append(rearrange(prescribed, "b t c h w -> b (t c) h w"))
        if prognostic is not None: tensors.append(rearrange(prognostic, "b t c h w -> b (t c) h w"))
        return torch.cat(tensors, dim=1)

    def forward(
        self,
        constants: torch.Tensor = None,
        prescribed: torch.Tensor = None,
        prognostic: torch.Tensor = None
    ) -> torch.Tensor:
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
                prognostic_t = torch.cat(
                    tensors=[prognostic[:, t_start:self.context_size],        # Prognostic input before context_size
                             torch.stack(outs, dim=1)[:, -self.context_size:]],  # Outputs since context_size
                    dim=1
                )
                x_t = self._prepare_inputs(
                    constants=constants,
                    prescribed=prescribed[:, t-self.context_size:t] if prescribed is not None else None,
                    prognostic=prognostic_t
                )

            # Forward input through model if context size has been reached
            x_t = self.forward_features(x_t)
            x_t = self.head(x_t)
            x_t = rearrange(
                x_t,
                "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                h=self.img_size[0] // self.patch_size[0],
                w=self.img_size[1] // self.patch_size[1],
            )
            out = prognostic_t[:, -1] + x_t
            outs.append(out)
        
        return torch.stack(outs, dim=1)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


if __name__ == "__main__":
    model = AFNONet(img_size=(720, 1440), patch_size=(4,4), in_chans=3, out_chans=10)
    sample = torch.randn(1, 3, 720, 1440)
    result = model(sample)
    print(result.shape)
    print(torch.norm(result))

