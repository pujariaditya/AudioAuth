# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Encodec SEANet-based encoder and decoder implementation."""

import typing as tp
import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from . import (
    SConv1d,
    SConvTranspose1d,
    CausalSTFT
)

from ..functional import STFT


def dws_conv_block(
    act: nn.Module, activation_params: dict, in_chs: int, out_chs: int, kernel_size: int,
    stride: int = 1, dilation: int = 1, norm: str = "weight_norm", norm_params: dict = {},
    causal: bool = False, pad_mode: str = 'constant', act_all: bool = False,
    transposed: bool = False, expansion: int = 1, groups: int = -1, bias: bool = True,
) -> tp.List[nn.Module]:
    """Build a depth-wise separable convolution block.

    Constructs a sequence of modules consisting of a pointwise (1x1) convolution
    followed by a depthwise convolution, with activation and normalization.

    Args:
        act (nn.Module): Activation function class (e.g., nn.ELU).
        activation_params (dict): Parameters passed to the activation function.
        in_chs (int): Number of input channels.
        out_chs (int): Number of output channels.
        kernel_size (int): Kernel size for the depthwise convolution.
        stride (int): Stride for the depthwise convolution.
        dilation (int): Dilation for the depthwise convolution.
        norm (str): Normalization type (e.g., 'weight_norm', 'none').
        norm_params (dict): Parameters for the normalization layer.
        causal (bool): Whether to use causal convolution.
        pad_mode (str): Padding mode for convolutions.
        act_all (bool): If True, insert activation before both the pointwise
            and depthwise convolutions; otherwise only before the pointwise.
        transposed (bool): If True, use transposed convolution for the depthwise layer.
        expansion (int): Expansion factor for computing groups.
        groups (int): Number of groups for the depthwise convolution. If -1,
            computed as ``out_chs // expansion``.
        bias (bool): Whether to include bias in the convolution layers.

    Returns:
        tp.List[nn.Module]: List of modules forming the depth-wise separable
            convolution block.
    """
    block = [
        act(**activation_params),
        SConv1d(in_chs, out_chs, kernel_size=1, norm=norm, norm_kwargs=norm_params,
                bias=bias if act_all else False,
                nonlinearity='relu'),
    ]
    if act_all:
        block.append(act(**activation_params))
    
    Conv = SConvTranspose1d if transposed else SConv1d
    if groups == -1:
        groups = out_chs // expansion
    block.append(
        Conv(
            out_chs, out_chs, kernel_size=kernel_size, stride=stride, dilation=dilation,
            groups=groups, norm=norm, norm_kwargs=norm_params, causal=causal,
            pad_mode=pad_mode, bias=bias,
            nonlinearity='relu' if act_all else 'linear'
        )
    )
    return block


class SEANetResnetBlock(nn.Module):
    """Residual block from SEANet model.

    Args:
        dim (int): Dimension of the input/output.
        kernel_size (int): Kernel size for the depthwise convolutions.
        dilations (list): List of dilations for the convolutions.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        skip (str): Skip connection type (``'identity'``, ``'1x1'``, ``'scale'``,
            ``'exp_scale'``, or ``'channelwise_scale'``).
        act_all (bool): (Used only when dws=True) Whether to insert activation
            before SepConv & DWConv or before a SepConv only.
        expansion (int): Expansion factor for grouped convolutions.
        groups (int): Number of groups for the depthwise convolution.
        bias (bool): Whether to include bias in the convolution layers.
        res_scale (float or None): Residual scaling factor for depth-dependent
            scaling.
        idx (int): Block index within the encoder/decoder stack.
        zero_init (bool): Whether to zero-initialize the residual scale parameter.

    Attributes:
        block: Sequential stack of depthwise-separable convolution layers.
        shortcut: Skip connection module.
        pre_scale: Input scaling factor or ``None``.
        scale: Learnable scale parameter or ``None``.
        exp_scale: Whether ``scale`` is exponentiated before use.
        res_scale: Fixed residual scaling factor.
        res_scale_param: Learnable residual scaling multiplier or ``None``.
    """
    def __init__(
        self, dim: int, kernel_size: int = 3,
        dilations: tp.List[int] = [1, 1], activation: str = 'ELU',
        activation_params: dict = {'alpha': 1.0},  norm: str = 'weight_norm',
        norm_params: tp.Dict[str, tp.Any] = {}, causal: bool = True,
        pad_mode: str = 'constant', skip: str = '1x1',
        act_all : bool = False, expansion: int = 1, groups: int = -1, bias: bool = True,
        res_scale: tp.Optional[float] = None, idx: int = 0, zero_init: bool = True,
    ):
        super().__init__()
        act = getattr(nn, activation)
        block = []
        inplace_act_params = activation_params.copy()
        inplace_act_params["inplace"] = True
        # Depth-dependent input scaling: deeper blocks accumulate more residuals,
        # so we attenuate the input to keep the signal magnitude stable.
        self.pre_scale = (1 + idx * res_scale**2)**-0.5 if res_scale is not None else None
        for i, dilation in enumerate(dilations):
            _activation_params = activation_params if i == 0 else inplace_act_params
            block += dws_conv_block(
                act,
                inplace_act_params, # _activation_params,
                dim,
                dim,
                kernel_size,
                dilation=dilation,
                norm=norm,
                norm_params=norm_params,
                causal=causal,
                pad_mode=pad_mode,
                act_all=act_all,
                expansion=expansion,
                groups=groups,
                bias=bias,
            )
        self.block = nn.Sequential(*block)
        self.shortcut: nn.Module
        
        self.scale = None
        self.exp_scale = False
        if skip == "identity":
            self.shortcut = nn.Identity()
        elif skip == "1x1":
            self.shortcut = SConv1d(dim, dim, kernel_size=1, norm=norm, norm_kwargs=norm_params,
                                    bias=bias)
        elif skip == "scale":
            self.scale = nn.Parameter(torch.ones(1, 1, 1))
        elif skip == "exp_scale":
            self.scale = nn.Parameter(torch.zeros(1, 1, 1))
            self.exp_scale = True
        elif skip == "channelwise_scale":
            self.scale = nn.Parameter(torch.ones(1, dim, 1))
        
        self.res_scale = res_scale
        if zero_init:
            self.res_scale_param = nn.Parameter(torch.zeros(1))
        else:
            self.res_scale_param = None

    def forward(self, x: Tensor) -> Tensor:
        # shortcut
        if self.scale is not None:
            scale = self.scale
            if self.exp_scale:
                scale = scale.exp()
            shortcut = scale * x
        else:
            shortcut = self.shortcut(x)
        
        # block
        if self.pre_scale is not None:
            x = x * self.pre_scale
        y: Tensor = self.block(x)
        
        # residual connection
        scale = 1.0 if self.res_scale is None else self.res_scale
        if self.res_scale_param is not None:
            scale = scale * self.res_scale_param
        return y.mul_(scale).add_(shortcut)


class L2Norm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-12, inout_norm: bool = True):
        super().__init__()
        self.scale = channels ** 0.5
        self.eps = eps
        self.inout_norm = inout_norm
    
    def forward(self, x: Tensor) -> Tensor:
        y = nn.functional.normalize(x, p=2.0, dim=1, eps=self.eps)
        if self.inout_norm:
            return y.mul_(self.scale)
        return y


class Scale(nn.Module):
    def __init__(self, dim: int, value: float = 1.0,
                 learnable: bool = True, inplace: bool = False):
        super().__init__()
        if learnable:
            self.scale = nn.Parameter(torch.ones(1, dim, 1) * value)
        else:
            self.scale = value
        self.inplace = inplace
    
    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            return x.mul_(self.scale)
        return self.scale * x

class SpecBlock(nn.Module):
    def __init__(
        self, spec: str, spec_compression: str,
        n_fft: int, channels: int, stride: int, norm: str, norm_params: tp.Dict[str, tp.Any],
        bias: bool, pad_mode: str, learnable: bool, causal: bool = True,
        mean: float = 0.0, std: float = 1.0, res_scale: tp.Optional[float] = 1.0,
        zero_init: bool = True, inout_norm: bool = True,
    ) -> None:
        super().__init__()
        self.learnable = learnable

        
        if spec == "stft":
            if causal:
                self.spec = CausalSTFT(n_fft=n_fft, hop_size=stride, pad_mode=pad_mode, learnable=learnable)
            else:
                self.spec = STFT(n_fft=n_fft, hop_size=stride, center=False, magnitude=True)
        elif spec == "":
            self.spec = None
            return
        else:
            raise ValueError(f"Unknown spec: {spec}")
        
 
        if spec_compression == "log":
            self.compression = "log"
        elif spec_compression == "":
            self.compression = ""
        else:
            self.compression = float(spec_compression)
        
 

        self.inout_norm = inout_norm
        self.mean, self.std = mean, std
        self.scale = res_scale
        self.scale_param = None
        self.layer = SConv1d(n_fft//2+1, channels, 1, norm=norm, norm_kwargs=norm_params, bias=bias, pad_mode=pad_mode)
        
        if zero_init:
            self.scale_param = nn.Parameter(torch.zeros(1))
     

    def forward(self, x: Tensor, wav: Tensor) -> Tensor:
        
        if self.spec is None:

            return x
        
        # Spectrogram
        y: Tensor = self.spec(wav)

        
        
        # Compression
        if self.compression == "log":
            y = y.clamp_min(1e-5).log_()
           
        elif self.compression == "":
            pass
        else:
            y = y.sign() * y.abs().pow(self.compression)

        
        # Normalize
        if self.inout_norm:
            y.sub_(self.mean).div_(self.std)
      
        
        # Layer
        y = self.layer(y)
  
        
        # Scaling
        scale = 1.0 if self.scale is None else self.scale
        if self.scale_param is not None:
            scale = self.scale_param * scale
            
        
        x.add_(y.mul_(scale))
     
        return x


    
class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer
    """
    def __init__(self, condition_dim: int):
        super().__init__()
        self.gamma_layer = nn.Linear(condition_dim, 1)
        self.beta_layer = nn.Linear(condition_dim, 1)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """
        Applies FiLM modulation to the input tensor.
        """

        gamma = self.gamma_layer(condition).unsqueeze(-1)
        beta = self.beta_layer(condition).unsqueeze(-1)
        return (x * gamma) + beta


class ResidualFiLM(nn.Module):
    """
    Residual Feature-wise Linear Modulation (ResidualFiLM) layer

    Applies gentle residual modulation using delta predictions:
    gamma = 1 + ε * tanh(delta_gamma)
    beta = ε * tanh(delta_beta)
    """
    def __init__(self, condition_dim: int, epsilon: float = 0.1):
        super().__init__()
        self.epsilon = epsilon
        self.delta_gamma_layer = nn.Linear(condition_dim, 1)
        self.delta_beta_layer = nn.Linear(condition_dim, 1)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """
        Applies residual FiLM modulation to the input tensor.

        Args:
            x: Input tensor [B, C, T]
            condition: Conditioning tensor [B, condition_dim]

        Returns:
            Modulated tensor with gentle residual changes
        """
        delta_gamma = self.delta_gamma_layer(condition).unsqueeze(-1)
        delta_beta = self.delta_beta_layer(condition).unsqueeze(-1)

        # Apply tanh to constrain and scale with epsilon
        gamma = 1 + self.epsilon * torch.tanh(delta_gamma)
        beta = self.epsilon * torch.tanh(delta_beta)

        # Residual modulation: start with identity and add gentle changes
        return x * gamma + beta
    
class SEANetEncoder(nn.Module):
    """SEANet-based encoder with dual watermark embedding via FiLM modulation.

    Attributes:
        dimension: Latent embedding dimension.
        n_filters: Base number of convolutional filters (doubled per stage).
        ratios: Downsampling ratios per encoder stage (stored in reversed order).
        n_residual_layers: Number of residual blocks per stage.
        hop_length: Total temporal downsampling factor (product of ratios).
        freq_bands: Number of frequency bands for FiLM modulation.
        model_msg_bits: Bit count for the model watermark.
        data_msg_bits: Bit count for the data watermark.
        conv_pre: Initial convolution (with optional input normalization).
        blocks: Residual blocks per stage.
        spec_blocks: Spectrogram side-chain blocks per stage.
        downsample: Strided downsampling layers per stage.
        conv_post: Final projection to latent dimension.
        model_embedding: MLP mapping model-watermark bits to embeddings.
        data_embedding: MLP mapping data-watermark bits to embeddings.
        model_film_layers: Per-stage, per-band FiLM layers for model watermark.
        data_film_layers: Per-stage, per-band FiLM layers for data watermark.
    """

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        msg_dimension: int = 16, 
        n_filters: int = 32,
        n_fft_base: int = 64,
        n_residual_layers: int = 1, 
        ratios: tp.List[int] = [8, 5, 4, 2],
        activation: str = 'ELU', 
        activation_params: dict = {'alpha': 1.0},
        norm: str = 'weight_norm',
        norm_params: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7, 
        last_kernel_size: int = 7, 
        residual_kernel_size: int = 3,
        dilation_base: int = 2, 
        skip: str = '1x1',
        causal: bool = False, 
        pad_mode: str = 'constant',
        act_all: bool = False, 
        expansion: int = 1, 
        groups: int = -1,
        l2norm: bool = False, 
        bias: bool = True, 
        spec: str = "stft",
        spec_compression: str = "",
        spec_learnable: bool = False,
        res_scale: tp.Optional[float] = None,
        wav_std: float = 0.1122080159,
        spec_means: tp.List[float] = [-4.554, -4.315, -4.021, -3.726, -3.477],
        spec_stds: tp.List[float] = [2.830, 2.837, 2.817, 2.796, 2.871],
        zero_init: bool = True,
        inout_norm: bool = True,
        
        embedding_dim: int = 64,  # Increased embedding dimension
        embedding_layers: int = 2, # Number of layers in MLP
        freq_bands: int = 4, # Number of frequency bands
        freq_band_primary_weight: float = 0.7,  # Weight for primary watermark
        freq_band_secondary_weight: float = 0.3,  # Weight for secondary watermark
        freq_band_attenuation: float = 0.3,  # Attenuation factor for secondary
        model_msg_bits: int = 16,  # Number of bits for model watermark
        data_msg_bits: int = 16,  # Number of bits for data watermark

        # New gentle modulation parameters
        film_epsilon: float = 0.1,  # Modulation strength for ResidualFiLM
        film_residual_alpha: float = 0.3,  # Residual blend factor
        film_start_block: int = 1,  # Block to start modulation
        embedding_gate_layers: int = 2,  # Gating MLP depth
        enable_layer_norm_embedding: bool = True,  # Enable LayerNorm
    ):
        super().__init__()
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)
        self.freq_bands = freq_bands
        self.freq_band_primary_weight = freq_band_primary_weight
        self.freq_band_secondary_weight = freq_band_secondary_weight
        self.freq_band_attenuation = freq_band_attenuation
        self.model_msg_bits = model_msg_bits
        self.data_msg_bits = data_msg_bits

        # Store gentle modulation parameters
        self.film_epsilon = film_epsilon
        self.film_residual_alpha = film_residual_alpha
        self.film_start_block = film_start_block
        self.embedding_gate_layers = embedding_gate_layers
        self.enable_layer_norm_embedding = enable_layer_norm_embedding

        # Validate that the sum of bits equals the expected total message size
        assert model_msg_bits + data_msg_bits == msg_dimension, (
            f"model_msg_bits ({model_msg_bits}) + data_msg_bits ({data_msg_bits}) "
            f"must equal msg_dimension ({msg_dimension})"
        )
        act = getattr(nn, activation)
        mult = 1
        self.conv_pre = nn.Sequential(
            Scale(1, value=1/wav_std, learnable=False, inplace=False) if inout_norm else nn.Identity(),
            SConv1d(
                channels, mult * n_filters, kernel_size, norm=norm, norm_kwargs=norm_params,
                causal=causal, pad_mode=pad_mode, bias=bias
            )
        )
        
        self.blocks = nn.ModuleList()
        self.spec_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        stride = 1
        for block_idx, ratio in enumerate(self.ratios):
            # Add residual layers
            block = []
            for j in range(1, n_residual_layers + 1):
                idx = j - 1 if spec == "" else j
                block += [
                    SEANetResnetBlock(mult * n_filters, kernel_size=residual_kernel_size,
                                    dilations=[dilation_base ** j, 1],
                                    norm=norm, norm_params=norm_params,
                                    activation=activation, activation_params=activation_params,
                                    causal=causal, pad_mode=pad_mode,
                                    skip=skip, act_all=act_all,
                                    expansion=expansion, groups=groups, bias=bias,
                                    res_scale=res_scale, idx=idx,
                                    zero_init=zero_init,)]
            self.blocks.append(nn.Sequential(*block))
            
            # add spectrogram layer
            spec_block = SpecBlock(
                spec, spec_compression, mult*n_fft_base, mult*n_filters, stride,
                norm, norm_params, bias=False, pad_mode=pad_mode,
                learnable=spec_learnable, causal=causal,
                mean=spec_means[block_idx], std=spec_stds[block_idx], res_scale=res_scale,
                zero_init=zero_init, inout_norm=inout_norm,
            )
            self.spec_blocks.append(spec_block)
            stride *= ratio

            # Add downsampling layers
            if res_scale is None:
                scale_layer = nn.Identity()
            else:
                scale_layer = Scale(
                    1, value=(1+n_residual_layers*res_scale**2)**-0.5,
                    learnable=False, inplace=True
                )
            downsample = nn.Sequential(
                scale_layer,
                act(inplace=True, **activation_params),
                SConv1d(mult * n_filters, mult * n_filters * 2, 1,
                        norm=norm, norm_kwargs=norm_params, bias=False,
                        nonlinearity='relu'),
                SConv1d(mult * n_filters * 2, mult * n_filters * 2,
                        kernel_size=ratio * 2, stride=ratio, groups=mult*n_filters*2,
                        norm=norm, norm_kwargs=norm_params,
                        causal=causal, pad_mode=pad_mode, bias=bias),
            )
            self.downsample.append(downsample)

            mult *= 2

        self.spec_post = SpecBlock(
            spec, spec_compression, mult*n_fft_base, mult*n_filters,
            stride, norm, norm_params, bias=False, pad_mode=pad_mode,
            learnable=spec_learnable, causal=causal,
            mean=spec_means[-1], std=spec_stds[-1], res_scale=res_scale, zero_init=zero_init,
            inout_norm=inout_norm,
        )
        self.conv_post = nn.Sequential(
            act(inplace=False, **activation_params),
            SConv1d(mult * n_filters, mult * n_filters, last_kernel_size, groups=mult*n_filters,
                    norm=norm, norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode,
                    bias=False, nonlinearity='relu'),
            SConv1d(mult * n_filters, dimension, 1, norm=norm,
                    norm_kwargs=norm_params, bias=bias),
            L2Norm(dimension, inout_norm=inout_norm) if l2norm else nn.Identity(),
        )
        if l2norm:
            # If the input audio has silent frames,
            # the encoder output will be near-zero vector sequence (right after initialization).
            # With l2norm, this will result in big scaling at the forward & backward,
            # meaning huge gradient will be encountered.
            # To avoid this, we initialize the last conv layer with
            # big non-zero bias.
            self.conv_post[-2].conv.conv.bias.data.normal_()

        # Model watermark embedding (first 16 bits)
        model_embedding_mlp = []
        for _ in range(embedding_layers):
            model_embedding_mlp.append(nn.Linear(embedding_dim, embedding_dim))
            model_embedding_mlp.append(nn.ReLU())

        self.model_embedding = nn.Sequential(
            nn.Linear(self.model_msg_bits, embedding_dim),  # Configurable bits for model
            *model_embedding_mlp
        )

        # Data watermark embedding (last 16 bits)
        data_embedding_mlp = []
        for _ in range(embedding_layers):
            data_embedding_mlp.append(nn.Linear(embedding_dim, embedding_dim))
            data_embedding_mlp.append(nn.ReLU())
        self.data_embedding = nn.Sequential(
            nn.Linear(self.data_msg_bits, embedding_dim),  # Configurable bits for data
            *data_embedding_mlp
        )

        # Message embedding normalization to prevent large swings (conditional)
        if enable_layer_norm_embedding:
            self.model_embedding_norm = nn.LayerNorm(embedding_dim)
            self.data_embedding_norm = nn.LayerNorm(embedding_dim)
        else:
            # Use identity modules to maintain compatibility
            self.model_embedding_norm = nn.Identity()
            self.data_embedding_norm = nn.Identity()

        # Gating MLP to control embedding magnitude (dynamic layers)
        self.model_embedding_gate = self._create_embedding_gate(embedding_dim, embedding_gate_layers)
        self.data_embedding_gate = self._create_embedding_gate(embedding_dim, embedding_gate_layers)

        # Frequency-specific FiLM layers for dual watermarks
        # Use ResidualFiLM for gentle modulation with configurable epsilon
        # Create one set of layers per encoder block to avoid out-of-bounds errors
        self.model_film_layers = nn.ModuleList([
            nn.ModuleList([ResidualFiLM(embedding_dim, epsilon=film_epsilon) for _ in range(freq_bands)])
            for _ in range(len(self.ratios))  # One per encoder block
        ])
        self.data_film_layers = nn.ModuleList([
            nn.ModuleList([ResidualFiLM(embedding_dim, epsilon=film_epsilon) for _ in range(freq_bands)])
            for _ in range(len(self.ratios))  # One per encoder block
        ])

    def _create_embedding_gate(self, embedding_dim: int, num_layers: int) -> nn.Sequential:
        """Create a dynamic embedding gate MLP with configurable layers.

        Args:
            embedding_dim: Input embedding dimension
            num_layers: Number of layers in the gate MLP

        Returns:
            nn.Sequential: The gate MLP
        """
        layers = []

        if num_layers == 1:
            # Single layer: directly to gate value
            layers.append(nn.Linear(embedding_dim, 1))
        else:
            # Multiple layers: progressively reduce to embedding_dim // 2 then to 1
            for i in range(num_layers):
                if i == 0:
                    # First layer: embedding_dim -> embedding_dim // 2
                    layers.append(nn.Linear(embedding_dim, embedding_dim // 2))
                elif i < num_layers - 1:
                    # Intermediate layers: embedding_dim // 2 -> embedding_dim // 2
                    layers.append(nn.Linear(embedding_dim // 2, embedding_dim // 2))
                else:
                    # Last layer: embedding_dim // 2 -> 1
                    layers.append(nn.Linear(embedding_dim // 2, 1))

                # Add activation (except for last layer)
                if i < num_layers - 1:
                    layers.append(nn.ReLU())

        # Add final sigmoid activation
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def forward(self, x: Tensor, msg: Tensor) -> Tensor:
        
        wav = x
        x = self.conv_pre(x)

        if msg is not None:
            msg = msg.float()
            # Split message into model and data watermarks
            model_bits = msg[:, :self.model_msg_bits]  # First model_msg_bits
            data_bits = msg[:, self.model_msg_bits:]   # Remaining data_msg_bits

            # Get initial embeddings
            model_embedded_raw = self.model_embedding(model_bits)
            data_embedded_raw = self.data_embedding(data_bits)

            # Apply LayerNorm to prevent large swings
            model_embedded_norm = self.model_embedding_norm(model_embedded_raw)
            data_embedded_norm = self.data_embedding_norm(data_embedded_raw)

            # Apply gating to control magnitude
            model_gate = self.model_embedding_gate(model_embedded_norm)
            data_gate = self.data_embedding_gate(data_embedded_norm)

            # Scale embeddings with gates
            model_embedded = model_embedded_norm * model_gate
            data_embedded = data_embedded_norm * data_gate

        for block_idx in range(len(self.blocks)):
            block = self.blocks[block_idx]
            spec_block = self.spec_blocks[block_idx]
            downsample = self.downsample[block_idx]

            x = spec_block(x, wav)
            x = block(x)
            x = downsample(x)

            if msg is not None:
                # Block-specific modulation strategy
                # Block 0: No modulation (preserve near-wave domain)
                # Block 1: Residual only (gentle)
                # Block 2: Residual + selective bands
                # Block 3: Full modulation (deepest block)

                if block_idx < self.film_start_block:
                    # Skip modulation in early blocks as configured
                    continue

                band_width = x.shape[1] // self.freq_bands
                x_bands = []  # Collect modified bands

                model_film_layer = self.model_film_layers[block_idx]
                data_film_layer = self.data_film_layers[block_idx]

                # Residual alpha values (blend factor for residual modulation)
                # Use configurable alpha for early blocks, higher for later blocks
                residual_alpha = self.film_residual_alpha if block_idx == self.film_start_block else 0.5

                for band_idx in range(self.freq_bands):
                    start = band_idx * band_width
                    end = (band_idx + 1) * band_width
                    x_band = x[:, start:end]

                    # Frequency-selective modulation
                    # Skip more bands in early blocks, focus on mid/high frequencies
                    should_modulate = False
                    if block_idx == 1:
                        # Block 1: Only modulate every 3rd band
                        should_modulate = (band_idx % 3 == 0)
                    elif block_idx == 2:
                        # Block 2: Modulate alternate bands, skipping lowest
                        should_modulate = (band_idx % 2 == 1) or (band_idx >= 2)
                    else:
                        # Block 3: Modulate all bands (deepest block)
                        should_modulate = True

                    if not should_modulate:
                        # Keep original band
                        x_bands.append(x_band)
                        continue

                    # Apply frequency-specific dual modulation
                    if block_idx < 3:
                        # Blocks 1-2: Use residual modulation (gentle)
                        if band_idx % 2 == 0:  # Even bands: Model-primary
                            x_band_model = model_film_layer[band_idx](x_band, model_embedded)
                            x_band_data = data_film_layer[band_idx](x_band, data_embedded * self.freq_band_attenuation)
                            modulated = self.freq_band_primary_weight * x_band_model + self.freq_band_secondary_weight * x_band_data
                        else:  # Odd bands: Data-primary
                            x_band_data = data_film_layer[band_idx](x_band, data_embedded)
                            x_band_model = model_film_layer[band_idx](x_band, model_embedded * self.freq_band_attenuation)
                            modulated = self.freq_band_secondary_weight * x_band_model + self.freq_band_primary_weight * x_band_data

                        # Residual blending
                        x_band = x_band + residual_alpha * (modulated - x_band)
                    else:
                        # Block 3: Full modulation (current approach)
                        if band_idx % 2 == 0:  # Even bands: Model-primary
                            x_band_model = model_film_layer[band_idx](x_band, model_embedded)
                            x_band_data = data_film_layer[band_idx](x_band, data_embedded * self.freq_band_attenuation)
                            x_band = self.freq_band_primary_weight * x_band_model + self.freq_band_secondary_weight * x_band_data
                        else:  # Odd bands: Data-primary
                            x_band_data = data_film_layer[band_idx](x_band, data_embedded)
                            x_band_model = model_film_layer[band_idx](x_band, model_embedded * self.freq_band_attenuation)
                            x_band = self.freq_band_secondary_weight * x_band_model + self.freq_band_primary_weight * x_band_data

                    x_bands.append(x_band)

                # Concatenate modified bands
                x = torch.cat(x_bands, dim=1)

        x = self.spec_post(x, wav)
        x = self.conv_post(x)
        return x


class SEANetDecoder(nn.Module):
    """SEANet-based decoder that reconstructs audio from latent representations.

    Attributes:
        dimension: Latent embedding dimension.
        channels: Number of output audio channels.
        n_filters: Base number of convolutional filters.
        ratios: Upsampling ratios per decoder stage.
        n_residual_layers: Number of residual blocks per stage.
        hop_length: Total temporal upsampling factor (product of ratios).
        model: Sequential stack of upsampling, residual, and output layers.
    """

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 1, 
        ratios: tp.List[int] = [8, 5, 4, 2],
        activation: str = 'ELU', 
        activation_params: dict = {'alpha': 1.0},
        norm: str = 'weight_norm', 
        norm_params: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7, 
        last_kernel_size: int = 7, 
        residual_kernel_size: int = 3,
        dilation_base: int = 2, 
        skip: str = '1x1',
        causal: bool = False, 
        pad_mode: str = 'constant', 
        trim_right_ratio: float = 1.0,
        final_activation: tp.Optional[str] = None,
        final_activation_params: tp.Optional[dict] = None,
        act_all: bool = False,
        expansion: int = 1, 
        groups: int = -1, 
        bias: bool = True,
        res_scale: tp.Optional[float] = None,
        wav_std: float = 0.1122080159,
        zero_init: bool = True,
        inout_norm: bool = True,
    ):
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)

        act = getattr(nn, activation)
        mult = int(2 ** len(self.ratios))
        model: tp.List[nn.Module] = []
        model += [
            SConv1d(dimension, mult * n_filters, 1, norm=norm, norm_kwargs=norm_params, bias=False),
            SConv1d(mult * n_filters, mult * n_filters, kernel_size, groups=mult*n_filters,
                    norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode, bias=bias)
        ]

        j = 0
        # Upsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            # Add upsampling layers
            if i > 0:
                if res_scale is None:
                    scale_layer = nn.Identity()
                else:
                    scale_layer = Scale(
                        1, value=(1+n_residual_layers*res_scale**2)**-0.5,
                        learnable=False, inplace=True
                    )
            else:
                scale_layer = nn.Identity()
            model += [
                scale_layer,
                act(inplace=True, **activation_params),
                SConvTranspose1d(mult * n_filters, mult * n_filters,
                                 kernel_size=ratio * 2, stride=ratio, groups=mult*n_filters,
                                norm=norm, norm_kwargs=norm_params,
                                causal=causal, trim_right_ratio=trim_right_ratio, bias=False,
                                nonlinearity='relu'),
                SConv1d(mult * n_filters, mult * n_filters // 2, 1,
                        norm=norm, norm_kwargs=norm_params, bias=bias)
            ]
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(mult * n_filters // 2, kernel_size=residual_kernel_size,
                                      dilations=[dilation_base ** j, 1],
                                    activation=activation, activation_params=activation_params,
                                    norm=norm, norm_params=norm_params, causal=causal,
                                    pad_mode=pad_mode, skip=skip,
                                    act_all=act_all, expansion=expansion, groups=groups,
                                    bias=bias, res_scale=res_scale, idx=j, zero_init=zero_init,)]
            mult //= 2

        # Add final layers
        if res_scale is None:
            scale_layer = nn.Identity()
        else:
            scale_layer = Scale(
                1, value=(1+n_residual_layers*res_scale**2)**-0.5, learnable=False, inplace=True
            )
        model += [
            scale_layer,
            act(inplace=True, **activation_params),
            SConv1d(n_filters, channels, last_kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode, bias=bias, nonlinearity='relu'),
            Scale(1, value=wav_std, learnable=False, inplace=True) if inout_norm else nn.Identity(),
        ]
        # Add optional final activation to decoder (eg. tanh)
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [
                final_act(**final_activation_params)
            ]
        self.model = nn.Sequential(*model)

    def forward(self, z):
        y = self.model(z)
        return y