from dataclasses import dataclass
from typing import Callable
import torch
import torch.nn.functional as F
from utils import init_method_normal, scaled_init_method_normal


@dataclass
class MambaConfig():
    base_model_type: str = "mamba"
    num_layers: int = 0
    hidden_size: int = 0
    state_size: int = 0
    vocab_size: int = 50000
    expansion_factor: int = 2
    conv_dimension: int = 0
    conv_bias: bool = True
    bias: bool = True
    use_fast_path: bool = True
    dt_rank: str = "auto"
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    rms_norm: bool = True
    fused_add_norm: bool = False  
    residual_in_fp32: bool = True
    hidden_dropout: float = 0.0
    ffn_hidden_size: int = None
    gated_linear_unit: bool = False
    mamba_moe_layers: str = ""
    routing_mode: str = "sinkhorn"
    device: str = "cuda"
    fp32_residual_connection: bool = False
    layernorm_epsilon: float = 1e-5
    layernorm_zero_centered_gamma: bool = False
    add_bias_linear: bool = True
    activation_func: Callable = F.gelu
    num_moe_experts: int = None

    # initialization
    init_method: Callable = None
    output_layer_init_method: Callable = None
    init_method_std: float = 0.02

    # mixed-precision
    apply_query_key_layer_scaling: bool = True
    attention_softmax_in_fp32: bool = True

    # fusion
    gated_linear_unit: bool = False
    bias_gelu_fusion: bool = False  
    persist_layer_norm: bool = False
    bias_dropout_fusion: bool = False 


    def __post_init__(self):
        """ Python dataclass method that is used to modify attributes after initialization.
            See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more details.
        """
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = 4 * self.hidden_size

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        if self.bias_gelu_fusion:
            if not self.add_bias_linear:
                raise ValueError(
                    "When bias_gelu_fusion is True, add_bias_linear must also be True."
                )

            if self.activation_func != F.gelu:
                raise ValueError(f'When bias_gelu_fusion is True, activation_func must be F.gelu.')

        if self.init_method is None:
            self.init_method = init_method_normal(self.init_method_std)

        if self.output_layer_init_method is None:
            self.output_layer_init_method = scaled_init_method_normal(
                self.init_method_std, self.num_layers
            )
