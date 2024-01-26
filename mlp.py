from dataclasses import dataclass
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import bias_gelu_impl
from mamba_config import MambaConfig

class MLP(nn.Module):
    def __init__(
        self, config: MambaConfig, is_expert: bool = False, layer_idx=None
    ):
        super().__init__()
        
        self.config: MambaConfig = config
        self.layer = layer_idx
        ffn_hidden_size_1 = self.config.ffn_hidden_size
        ffn_hidden_size_2 = self.config.ffn_hidden_size
        
        # If this is a gated linear unit we double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        if self.config.gated_linear_unit:
            ffn_hidden_size_1 *= 2
            
        self.linear_fc1 = nn.Linear(self.config.hidden_size, ffn_hidden_size_1, bias = self.config.add_bias_linear, device = self.config.device)
        self.linear_fc1.is_expert = is_expert

        if self.config.gated_linear_unit:

            def glu(x):
                x = torch.chunk(x, 2, dim=-1)
                return self.config.activation_func(x[0]) * x[1]

            self.activation_func = glu
        else:
            self.activation_func = self.config.activation_func

        self.linear_fc2 = nn.Linear(ffn_hidden_size_2, self.config.hidden_size, bias = self.config.add_bias_linear, device = self.config.device)

    def forward(self, hidden_states, inference_params=None):
        intermediate = self.linear_fc1(hidden_states)
        intermediate = self.activation_func(intermediate)
        output = self.linear_fc2(intermediate)
        return output