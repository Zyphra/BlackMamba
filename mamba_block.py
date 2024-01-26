
import math
from typing import Optional, Union
import re
from contextlib import nullcontext
from abc import ABC, abstractmethod
from dataclasses import dataclass
import functools
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
except ImportError:
    selective_scan_fn, mamba_inner_fn = None, None

try:
    from ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from mamba_layer import MambaLayer
from mamba_config import MambaConfig
from mlp import MLP
from switch_mlp import SwitchMLP


class MambaBlock(nn.Module):
    def __init__(
        self, config, mixer_cls, moe_cls=None, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        super().__init__()
        self.config = config
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(config)
        
        if not config.rms_norm:
            self.norm = norm_cls
        else:
            self.norm = norm_cls(config.hidden_size)
            
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
        if moe_cls is not None:
            self.moe = moe_cls(config)
        else:
            self.moe = None

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states , residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

class MambaBlockParallelMoe(nn.Module):
    def __init__(
        self, config, mixer_cls, moe_cls=None, norm_cls=nn.LayerNorm, norm_moe=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):

        super().__init__()
        self.config = config
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(config)
        if not config.rms_norm:
            self.norm = norm_cls
            self.norm_moe = norm_moe
        else:
            self.norm = norm_cls(config.hidden_size)
            self.norm_moe = norm_moe(config.hidden_size)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
            assert isinstance(
                self.norm_moe, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
        if moe_cls is not None:
            self.moe = moe_cls(config)
        else:
            self.moe = None

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            hidden_states_moe = self.norm_moe(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
            hidden_states_moe, _ = fused_add_norm_fn(
                hidden_states,
                self.norm_moe.weight,
                self.norm_moe.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm_moe.eps,
            )
        
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        
        hidden_states_moe = self.moe(hidden_states_moe)
        hidden_states += hidden_states_moe
        return hidden_states , residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class MoEBlock(nn.Module):
    def __init__(
        self, config, mixer_cls, moe_cls=None, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):

        super().__init__()
        self.config = config
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(config)
        if not config.rms_norm:
            self.norm = norm_cls
        else:
            self.norm = norm_cls(config.hidden_size)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
        if moe_cls is not None:
            self.moe = moe_cls(config)
        else:
            self.moe = None

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states)
        return hidden_states , residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
        

def create_block(config, layer_idx):

    if config.rms_norm:
        norm_cls = partial(RMSNorm, eps=config.layernorm_epsilon)
    else:
        norm_cls = partial(nn.LayerNorm if not config.rms_norm else RMSNorm, eps=config.layernorm_epsilon)
    
    if (not config.mamba_moe_layers) or config.mamba_moe_layers[layer_idx-1][0] == 'r':
        if (not config.mamba_moe_layers) or len(config.mamba_moe_layers[layer_idx-1]) == 1:
            mixer_cls = partial(MambaLayer, layer_idx=layer_idx)
            block = MambaBlock(
                config,
                mixer_cls=mixer_cls,
                norm_cls=norm_cls,
                fused_add_norm=config.fused_add_norm,
                residual_in_fp32=config.residual_in_fp32,
            )
        else:
            if config.mamba_moe_layers[layer_idx-1][1] == '1':
                if config.rms_norm:
                    norm_moe = partial(RMSNorm, eps=config.layernorm_epsilon)
                else:
                    norm_moe = partial(
                        nn.LayerNorm if not config.rms_norm else RMSNorm, eps=config.layernorm_epsilon
                    )
                mixer_cls = partial(MambaLayer, layer_idx=layer_idx)
                moe_cls = partial(MLP, layer_idx=layer_idx)
                block = MambaBlockParallelMoe(
                config,
                mixer_cls=mixer_cls,
                moe_cls=moe_cls,
                norm_cls=norm_cls,
                norm_moe=norm_moe,
                fused_add_norm=config.fused_add_norm,
                residual_in_fp32=config.residual_in_fp32,
            )
            else:
                if config.rms_norm:
                    norm_moe = partial(RMSNorm, eps=config.layernorm_epsilon)
                else:
                    norm_moe = partial(
                        nn.LayerNorm if not config.rms_norm else RMSNorm, eps=config.layernorm_epsilon
                    )
                mixer_cls = partial(MambaLayer, layer_idx=layer_idx)
                moe_cls = partial(SwitchMLP, layer_idx=layer_idx)
                block = MambaBlockParallelMoe(
                config,
                mixer_cls=mixer_cls,
                moe_cls=moe_cls,
                norm_cls=norm_cls,
                norm_moe=norm_moe,
                fused_add_norm=config.fused_add_norm,
                residual_in_fp32=config.residual_in_fp32,
            )
    else:
        if config.mamba_moe_layers[layer_idx-1][0] == '1':
            mixer_cls = partial(MLP, layer_idx=layer_idx)
            block = MoEBlock(
                config,
                mixer_cls=mixer_cls,
                norm_cls=norm_cls,
                fused_add_norm=config.fused_add_norm,
                residual_in_fp32=config.residual_in_fp32,
            )
        else:
            mixer_cls = partial(SwitchMLP, layer_idx=layer_idx)
            block = MoEBlock(
                config,
                mixer_cls=mixer_cls,
                norm_cls=norm_cls,
                fused_add_norm=config.fused_add_norm,
                residual_in_fp32=config.residual_in_fp32,
            )
    block.layer_idx = layer_idx
    return block

class MambaDecoder(nn.Module):
    """Class wrapping a decoder stack of mamba blocks."""

    def __init__(
        self,
        config: MambaConfig,
        post_layer_norm=True,
        pre_process=True,
        post_process=True,
    ):
        super().__init__()

        self.config: MambaConfig = config
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.norm_cls = partial(nn.LayerNorm, eps=self.config.layernorm_epsilon)
        
        self._build_layers()

    def _build_layers(self):

        num_layers_to_build = self.config.num_layers
        # build the actual mamba layers
        self.layers = torch.nn.ModuleList([create_block(self.config, i + 1) for i in range(num_layers_to_build)])

        if self.post_process and self.post_layer_norm:
            # Final layer norm before output.
            self.final_layernorm = self.norm_cls(self.config.hidden_size, bias = True)

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def forward(self, hidden_states, residual = None, inference_params=None):

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor
            
        residual = None
        for i,layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states=hidden_states,
                residual = residual,
                inference_params=inference_params,
            )
            
        # Final layer norm.
        if self.post_process and self.post_layer_norm:
            if not self.config.fused_add_norm:
                residual = (hidden_states + residual) if residual is not None else hidden_states
                hidden_states = self.final_layernorm(residual.to(dtype=self.final_layernorm.weight.dtype))
            else:
                # Set prenorm=False here since we don't need the residual
                fused_add_norm_fn = rms_norm_fn if isinstance(self.final_layernorm, RMSNorm) else layer_norm_fn
                hidden_states = fused_add_norm_fn(
                    hidden_states,
                    self.final_layernorm.weight,
                    self.final_layernorm.bias,
                    eps=self.final_layernorm.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
        return hidden_states