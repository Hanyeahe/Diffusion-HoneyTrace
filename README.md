# Diffusion HoneyTrace

Official research code for **Diffusion HoneyTrace (DHT)**, a behavioral
watermark for tracing diffusion models trained from protected black-box
outputs.

DHT applies a secret score-conditioned bias along a fixed DCT direction during
the final denoising steps. A student trained on protected generations can
inherit the resulting keyed relation, which is tested through output-only
queries.

## Repository layout

- `dht32/scripts/`: CIFAR-10 experiments at 32x32, including protected sampling,
  clean and constant-DCT controls, student training, seed-addressable audits,
  output-feature audits, post-processing tests, and FID/KID evaluation.
- `dht64/scripts/`: FFHQ experiments at 64x64 using the NVIDIA EDM codebase,
  including protected sampling, transfer training, matched controls, robustness,
  and quality evaluation.
- `results/cifar10/`: CSV and JSON summaries used for the CIFAR-10 results.
- `results/ffhq64/`: CSV and JSON summaries used for the FFHQ-64 results.

The paper-facing CIFAR-10 implementation is
`dht32/scripts/run_fixed_bias_honeytrace.py`. The file
`run_honeytrace_sampling.py` records an earlier pilot implementation. For the
post-processing experiment, use `evaluate_wrongkey_postprocess_axisfix.py`,
which follows the final horizontal/vertical DCT-axis convention.

## Environment

The experiments were run with the following main packages:

```text
Python
PyTorch 2.4.1 + CUDA 12.4
torchvision 0.19.1
diffusers 0.30.3
NumPy 1.26.4
SciPy 1.14.1
Pillow 10.4.0
Matplotlib 3.9.2
```

The original EDM environment file is included at
`dht64/edm-environment.yml`.

## External assets

Datasets, generated image collections, model checkpoints, and secret watermark
states are intentionally excluded from this repository.

- CIFAR-10 teacher: [`google/ddpm-cifar10-32`](https://huggingface.co/google/ddpm-cifar10-32)
- NVIDIA EDM code: [`NVlabs/edm`](https://github.com/NVlabs/edm)
- FFHQ-64 teacher: [`edm-ffhq-64x64-uncond-vp.pkl`](https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl)

The EDM dependency is distributed under its original license. A copy is kept at
`dht64/EDM-LICENSE.txt`.

## Main reported settings

- CIFAR-10: 32x32 DDPM, 1000 sampling steps, final 50 protected steps,
  `lambda=0.03`, `alpha=2`, and DCT direction `(4,5)`.
- FFHQ: 64x64 EDM, 40 sampling steps, final 5 protected steps, `lambda=3`,
  `alpha=2`, and DCT direction `(20,20)`.
- Black-box decisions use a one-sided test with `p < 0.005` under the evaluated
  protocol.

## Included results

The `results/` directory contains compact metrics and aggregate summaries used
for query-budget, matched-margin, wrong-key, post-processing, ablation, FID,
and KID analyses. Per-sample scores, retained secret seeds, large image
directories, and trained student weights are not included.
