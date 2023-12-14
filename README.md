# Deprecation note
We are moving away from maintaining this repository. 

# 🦆 ~~Uberduck Synthetic Speech~~
![](https://img.shields.io/github/forks/uberduck-ai/uberduck-ml-dev)
![](https://img.shields.io/github/stars/uberduck-ai/uberduck-ml-dev)
![](https://img.shields.io/github/issues/uberduck-ai/uberduck-ml-dev)
![GithubActions](https://github.com/uberduck-ai/uberduck-ml-dev/actions/workflows/main.yml/badge.svg)
[![Discord](https://img.shields.io/discord/1037326658807533628?color=%239B59B6&label=chat%20on%20discord)](https://discord.com/invite/ATYWnMu)

This repository includes
<ul>
  <li>Production model code for Tacotron2, so-ViTS-svc, zero-shot RADTTS, HiFi-GAN, and RVC.</li>
  <li>Production training code for Tacotron2, RADTTS, HiFi-GAN, and RVC.</li>
  <li>Fill-populating inference, additional languages, and Torchmoji emotional encoding.</li>
</ul>

Notebooks are available [**here**](https://app.uberduck.ai/), and models to fine-tune from are available here [**here**](https://huggingface.co/Uberduck).

## Summary

![Summary](https://github.com/uberduck-ai/uberduck-ml-dev/blob/master/analytics/dependencies/details.png)

## Installation

If you want to install on your own machine, create a virtual environment and install like 

```bash
conda create -n 'test-env' python=3.10 -y
source activate test-env
pip install git+https://github.com/uberduck-ai/uberduck-ml-dev
```

## Training

Train a radtts on LJ Speech as follows

```
cd tutorials/radtts
bash download.sh
bash train.sh
```

You will need to adjust the paths and potentially other training settings in `tutorials/radtts/demo_config.json`.
This code has been tested on a single T4 as well as 2 A6000s.

For processing of new datasets, see `tutorials/radtts/radtts_data_processing.ipynb`.

# Development

We love contributions!

## Installation

To install in development mode, run

```bash
pip install pre-commit black # format your code on commit by installing black!
git clone git@github.com:uberduck-ai/uberduck-ml-dev.git
cd uberduck-ml-dev
pre-commit install # Install required Git hooks
python setup.py develop # Install the library
```

## 🚩 Testing

In an environment or image with uberduck-ml-dev installed, run 

```bash
cd uberduck-ml-dev
python -m pytest
```
