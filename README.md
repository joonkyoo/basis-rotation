# Mitigating Staleness in Asynchronous Pipeline Parallelism via Basis Rotation

Hyunji Jung*, Sungbin Shin*, Namhoon Lee &nbsp; (\*equal contribution)  
POSTECH

**ICML 2026** &nbsp;·&nbsp; [Paper](https://arxiv.org/abs/2602.03515)

---

## Overview

We identify the root cause as basis misalignment, which induces oscillation in Adam's update directions by breaking its coordinate-wise adaptivity. Under gradient delay, these oscillations cause the updates from delayed gradients to be misaligned with — or even opposite to — the non-delayed counterparts, harming optimization. We propose Basis Rotation, which realigns the optimization space with the Hessian eigenbasis, restoring Adam's adaptivity and eliminating delay sensitivity.

## Installation

Experiments are tested on Python 3.12. We use [uv](https://github.com/astral-sh/uv) for environment management.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python pin 3.12
uv sync
```

### Dataset Cache

Datasets are downloaded from Hugging Face on first use. Set `HF_HOME` to control where they are cached:

```bash
export HF_HOME=/scratch/data   # or any directory with sufficient space
```

## Usage

See `run.bash` for a complete example. The script can be configured for any number of pipeline stages by adjusting `nstages`.

### Pipeline-Parallel Training

```bash
# Example: 32-stage pipeline on 8 GPUs (4 stages per GPU)
ngpus=8
nstages=32
master_port=12345

for rank in $(seq 0 $(($nstages - 1))); do
    local_rank=$(($rank % $ngpus))
    .venv/bin/python main_with_runtime.py \
      --module models.gptn \
      --n_layer 32 --n_embd 384 --n_head 6 --block_size 512 \
      --config_path models/gptn/layers=32/stage${nstages}.json \
      -d openwebtext \
      --optimizer basisrotation \
      --rotation_geometry bi --approx_source 2nd \
      --subspace_update_frequency 10 \
      --lr 1e-3 --lr_warmup --lr_policy cosine --epochs 250 \
      --clip_grad 1 --recompute \
      --rank $rank --local_rank $local_rank \
      --master_addr localhost --master_port $master_port \
      --distributed_backend gloo > rank${rank}.log 2>&1 &
done
wait
```

## Key Arguments

### Optimizer

| Argument | Values | Description |
|---|---|---|
| `--optimizer` | `basisrotation`, `adamw`, `nadamw` | Optimizer choice. `basisrotation` is Adam with Basis Rotation (Algorithm 1). |
| `--rotation_geometry` | `bi`, `uni` | Rotation geometry: bilateral (two-sided) or unilateral (one-sided). Corresponds to $\mathcal{G}$ in Algorithm 2. |
| `--approx_source` | `2nd`, `1st` | Approximation source for eigenbasis estimation: second-order covariance ($\mathcal{S} = 2^\text{nd}$) or first-order gradient ($\mathcal{S} = 1^\text{st}$). Corresponds to $\mathcal{S}$ in Algorithm 2. |
| `--subspace_update_frequency` | int | How often to refresh the rotation basis (in steps). |

### Pipeline Parallelism

| Argument | Description |
|---|---|
| `--config_path` | Path to stage configuration JSON (e.g., `models/gptn/layers=32/stage4.json`). Maps model submodules to pipeline stages. |
| `--rank` | Rank of this process. |
| `--local_rank` | Local GPU index within the node. |
| `--stash_to_cpu` | Offload stashed weight versions to CPU to reduce GPU memory usage. |
| `--recompute` | Recompute activations in the backward pass (saves memory at the cost of extra compute). |

## Model Architecture

The GPT model is defined in `models/gptn/` and split into pipeline stages via JSON configs in `models/gptn/layers=<N>/stage<K>.json`. Available configs: `stage1` (single GPU), `stage2`, `stage4`, `stage8`, `stage16`, `stage32`.

To add a new model size, define the architecture in `models/` and create corresponding stage JSON files.

## Acknowledgements

This codebase builds upon the following open-source projects:

- [PipeDream](https://github.com/msr-fiddle/pipedream)
- [NanoGPT](https://github.com/karpathy/nanoGPT)
- [AsyncPP](https://github.com/PluralisResearch/AsyncPP)
- [SOAP](https://github.com/nikhilvyas/SOAP)

## Citation

```bibtex
@inproceedings{jung2026mitigating,
  title={Mitigating Staleness in Asynchronous Pipeline Parallelism via Basis Rotation},
  author={Jung, Hyunji and Shin, Sungbin and Lee, Namhoon},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```
