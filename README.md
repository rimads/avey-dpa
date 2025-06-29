# Don't Pay Attention (DPA)

This repository includes the code used to train models, run benchmarks, and create plots for the paper [Don't Pay Attention](https://arxiv.org/abs/2506.11305).

> [!NOTE]
> Licenses in subdirectories take precedence over the repository license for their respective subdirectories, and licenses in individual files take precedence over the subdirectory license for their respective files.

All the model checkpoints trained during this work are available on [HuggingFace](https://huggingface.co/collections/avey-ai/dont-pay-attention-68454fb98666af07b6adf429). Base and instruct-tuned preview versions of Avey-1 which you can try out are available [here](https://huggingface.co/collections/avey-ai/avey-1-research-preview-684594455dc52c40017b3ffa).

## Requirements

This code was tested on Ubuntu 22.04, python 3.12, and A100, H100 and H200 GPUs. It is recommended to run the setup and code in a clean python environment. Please make sure CUDA toolkit is installed correctly.

To clone the repo and install the dependencies, run
```bash
git clone https://github.avey.ai/avey-dpa
cd avey-dpa
source setup.sh
```

Either login to weights & biases if you want to log training metrics:
```bash
wandb login
```
or disable it:
```bash
wandb disabled
```

Set the model path for either one of the following models you want to train/test:

> [!WARNING]
> Running training and benchmarks will download the sample-10BT split of FinWeb dataset and model checkpoints, make sure you have enough available disk space (at least ~120GB)

The names of the model checkpoins are in the format `<model name>-dpa-<model size>-<checkpoint>`, where:

- Model name can be one of `avey`, `mamba`, `rwkv7`, or `tpp`
- Model size can be one of `0.1B`, `0.5B` or `1.5B`
- Available checkpoints are `100BT`, `95BT`, or `90BT`

All checkpoints are available under `avey-ai` on HuggingFace over [here](https://huggingface.co/collections/avey-ai/dont-pay-attention-68454fb98666af07b6adf429).

### Avey

```bash
export MODEL_NAME=avey
export MODEL_PATH=avey-ai/avey1-dpa-1.5B-100BT
```

### Mamba

Install dependencies:
```bash
sh mamba/setup.sh
```

set the model name and path:
```bash
export MODEL_NAME=mamba
export MODEL_PATH=avey-ai/mamba-dpa-1.5B-100BT
```

### RWKV-7

```bash
source rwkv7/env.sh
export MODEL_NAME=rwkv7
export MODEL_PATH=avey-ai/rwkv7-dpa-1.5B-100BT
```

### Transformers

```bash
export MODEL_NAME=tpp
export MODEL_PATH=avey-ai/tpp-dpa-1.5B-100BT
```

## Training

Adjust `NUMBER_OF_GPUS` (on a single node) and `BATCH_SIZE` (start at 1, increase until your GPU runs out of memory) in `train.sh`, and then run:

```bash
sh train.sh
```

## Benchmarks

For standard benchmarks reported in the paper, run:

```bash
sh eval.sh
```

For RULER S-NIAH, run:
```bash
sh eval-long.sh
```

## Plots

To plot the NIAH heatmap (figure 1 from the paper) run:
```bash
sh plot-niah.sh
```

To plot TTFT vs context length (figure 4 from the paper) run:

> [!IMPORTANT]
> Make sure you've already run the setup steps for mamba (run mamba/setup.sh) and rwkv7 (source rwkv7/env.sh)

```bash
python3 plot_ttft.py
```
