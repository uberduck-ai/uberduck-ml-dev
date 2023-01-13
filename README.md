# 🦆 Uberduck Text-to-speech ![](https://img.shields.io/github/forks/uberduck-ai/uberduck-ml-dev) ![](https://img.shields.io/github/stars/uberduck-ai/uberduck-ml-dev) ![](https://img.shields.io/github/issues/uberduck-ai/uberduck-ml-dev)

[**Uberduck**](https://uberduck.ai/) is a tool for fun and creativity with voice cloning with neural text-to-speech. This repository will get you creating your own speech synthesis model. Please see our [**training**](https://colab.research.google.com/drive/1jF-Otw2_ssEcus4ISaIZu3QDmtifUvyY) and [**synthesis**](https://colab.research.google.com/drive/1wXWuhnw2pdfFy1L-pUzHfopW10W2GiJS) notebooks, and the [**Wiki**](https://github.com/uberduck-ai/uberduck-ml-dev/wiki).

## Overview

The main "Tacotron2" model in this repository is based on the NVIDIA Mellotron.  The states of the various latent space features are

- Multispeaker training (functioning, beneficial) <br>
- Torchmoji conditioning (functioning)  <br>
- Audio/speaker embedding (functioning) <br>
- Pitch conditioning (non functioning) <br>
- SRMR and MOSNet conditioning (non functioning) <br>

It also includes teacher forcing type methods for prosody matching, as well as language, tensorboard, and torchscript support, and improvements to the learning rate scheduling.

## Usage

The easiest ways to try us out are the colab notebooks, but if you want to install, run 

### Installation

```
conda create -n 'uberduck-ml-dev' python=3.8
source activate uberduck-ml-dev
pip install git+https://github.com/uberduck-ai/uberduck-ml-dev.git
```

### Training

1. Download torchmoji models if training with Torchmoji GST.

   ```bash
   wget "https://github.com/johnpaulbin/torchMoji/releases/download/files/pytorch_model.bin" -O pytorch_model.bin
   wget "https://raw.githubusercontent.com/johnpaulbin/torchMoji/master/model/vocabulary.json" -O vocabulary.json
   ```
2. Create your training config. Use the training configs in the `configs` directory as a starting point, e.g. [this one](https://github.com/uberduck-ai/uberduck-ml-dev/blob/master/configs/tacotron2_config.json).
3. Start training. Example invocation for Tacotron2 training:
   ```bash
   python -m uberduck_ml_dev.exec.train_tacotron2 --config tacotron2_config.json
   ```

## Development


```bash
pip install pre-commit black # install the required development dependencies in a virtual environment
git clone git@github.com:uberduck-ai/uberduck-ml-dev.git # clone the repository:
pre-commit install # Install required Git hooks:
python setup.py develop # Install the library
```

### 🚩 Testing

In an environment with uberduck-ml-dev installed, run 

```bash
python -m pytest
```
