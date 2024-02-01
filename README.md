# BlackMamba: 


> **BlackMamba: Mixture of Experts for State-space models**\
> Quentin Anthony*, Yury Tokpanov*, Paolo Glorioso*, Beren Millidge*\
> Paper: 

## About
In this repository we provide inference and generation code for our BlackMamba model. 

BlackMamba is an novel architecture which combines state-space models (SSMs) with mixture of experts (MoE). It uses [Mamba](https://arxiv.org/abs/2312.00752) as its SSM block and [switch transformer](https://arxiv.org/abs/2101.03961) as its MoE block base. BlackMamba is extremely fast and low latency for generation and inference, providing significant speedups over all of classical transformers, MoEs, and Mamba SSM models. Additionally, due to its SSM sequence mixer, BlackMamba retains linear compuational complexity in the sequence length. 

## Installation



## Pretrained Models

Our pretrained models are uploaded to [huggingface](LINK): `Zyphra/BlackMamba350m` and `Zyphra/BlackMamba630m`

## Usage

