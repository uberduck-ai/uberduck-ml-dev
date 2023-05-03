# 🦆 Uberduck Synthetic Speech
![](https://img.shields.io/github/forks/uberduck-ai/uberduck-ml-dev)
![](https://img.shields.io/github/stars/uberduck-ai/uberduck-ml-dev)
![](https://img.shields.io/github/issues/uberduck-ai/uberduck-ml-dev)
![GithubActions](https://github.com/uberduck-ai/uberduck-ml-dev/actions/workflows/main.yml/badge.svg)
[![Discord](https://img.shields.io/discord/1037326658807533628?color=%239B59B6&label=chat%20on%20discord)](https://discord.com/invite/ATYWnMu)

[**Uberduck**](https://uberduck.ai/) is a tool for fun and creativity with neural text-to-speech. This repository will get you creating your own speech synthesis models. Please see our [**training**](https://colab.research.google.com/drive/1jF-Otw2_ssEcus4ISaIZu3QDmtifUvyY) and [**synthesis**](https://colab.research.google.com/drive/1wXWuhnw2pdfFy1L-pUzHfopW10W2GiJS) notebooks. Please reach out for help and contribute!

## Overview

The models in this repository used in production are the Tacotron2, SO-VITS-SVC, zero-shot RADTTS, and HiFi-GAN.  Training code is included for Tacotron2, RADTTS, and HiFi-GAN.  Other goodies include fill-populating inference, additional languages, and Torchmoji emotional encoding. 

![Summary](https://github.com/uberduck-ai/uberduck-ml-dev/blob/master/details.png)

## Usage

Download models to fine-tune from [**here**](https://huggingface.co/Uberduck).  The [**notebooks**](https://app.uberduck.ai/) are the easiest ways to try these out.

### Installation

If you want to install on your own machine, create a virtual environment and install like 

```bash
pip install git+https://github.com/uberduck-ai/uberduck-ml-dev.git
```

### Training

Please see the tests subfolder for examples of up to date training and inference invocation.

## Development

We love contributions! Feel free to reach out to discuss contribution.

### Installation

To install in development mode, run

```bash
pip install pre-commit black # format your code on commit by installing black!
git clone git@github.com:uberduck-ai/uberduck-ml-dev.git
cd uberduck-ml-dev
pre-commit install # Install required Git hooks
python setup.py develop # Install the library
```

### 🚩 Testing

In an environment or image with uberduck-ml-dev installed in the uberduck-ml-dev root folder, run 

```bash
python -m pytest
```
