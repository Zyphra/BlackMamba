![blackmamba-wider](https://github.com/Zyphra/BlackMamba/assets/10281105/02641d76-b4ca-4056-a7d1-2d2d2b08692e)


# BlackMamba

![image](https://github.com/Zyphra/BlackMamba/assets/10281105/045516bd-1e72-4413-a1da-7e9a4e1372d0)

> **BlackMamba: Mixture of Experts for State-space models**\
> Quentin Anthony*, Yury Tokpanov*, Paolo Glorioso*, Beren Millidge*\
> Paper: https://arxiv.org/abs/2402.01771

## About
In this repository we provide inference code for our BlackMamba model. 

BlackMamba is an novel architecture which combines state-space models (SSMs) with mixture of experts (MoE). It uses [Mamba](https://arxiv.org/abs/2312.00752) as its SSM block and [switch transformer](https://arxiv.org/abs/2101.03961) as its MoE block base. BlackMamba is extremely low latency for generation and inference, providing significant speedups over all of classical transformers, MoEs, and Mamba SSM models. Additionally, due to its SSM sequence mixer, BlackMamba retains linear computational complexity in the sequence length. 

## Requirements
`pip install causal-conv1d>=1.1.0`: required for Mamba. The rest of the kernels should be built locally.

Other requirements:

Linux
NVIDIA GPU
PyTorch 1.12+
CUDA 11.6+

## Quick installation in a fresh Python environment
- `pip install torch packaging`
- `pip install .` to install from source from this repository

## Pretrained Models

Our pretrained models are uploaded to [our HuggingFace](https://huggingface.co/Zyphra): 
- [BlackMamba 340M/1.5B](https://huggingface.co/Zyphra/BlackMamba-1.5B)*
- [BlackMamba 630M/2.8B](https://huggingface.co/Zyphra/BlackMamba-2.8B)*

*Since models are MoE, they're named according to `(Forward Pass Parameters) / (Total Parameters)` for clarity.

## Usage

```
from mamba_model import MambaModel
import torch

model = MambaModel.from_pretrained(pretrained_model_name="Zyphra/BlackMamba-2.8B")
model = model.cuda().half()
inputs = torch.tensor([1, 2]).cuda().long().unsqueeze(0)
out = model(inputs)
```


