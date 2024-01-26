import logging
from typing import Literal, Optional, Union
import functools
from functools import partial
import torch
import torch.nn as nn
from torch import Tensor
import math
import os
from mamba_block import MambaBlock, MambaDecoder
from mamba_config import MambaConfig
from hf_utils import *
import os


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MambaModel(nn.Module):
    def __init__(
        self,
        config: MambaConfig,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = True,
        initializer_cfg = None,
    ) -> None:
        super().__init__()

        self.config: MambaConfig = config
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        
        if self.pre_process:
            self.embedding = nn.Embedding(self.config.vocab_size, self.config.hidden_size)


        self.decoder = MambaDecoder(
            config = self.config,
            pre_process = self.pre_process,
            post_process = self.post_process,
        )
        
        if post_process:
            self.output_layer = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias = self.config.add_bias_linear)
            if self.share_embeddings_and_output_weights and (self.pre_process or self.post_process):
                self.initialize_last_stage_with_word_embeddings()
            
        # apply weight initialization
        self.apply(
            partial(
                _init_weights,
                n_layer=self.config.num_layers,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        
    def initialize_last_stage_with_word_embeddings(self):
        with torch.no_grad():
            self.output_layer.weight = self.embedding.weight

    def forward(
        self,
        input_ids,
        position_ids = None,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params=None,
    ) -> Tensor:
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids)
        else:
            decoder_input = None

        hidden_states = self.decoder(
            hidden_states=decoder_input,
            residual=None,
            inference_params=inference_params,
        )
        
        if not self.post_process:
            return hidden_states
        
        logits = self.output_layer(hidden_states)

        return logits.contiguous()
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f)
