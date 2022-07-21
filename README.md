# 🦆 Uberduck TTS ![](https://img.shields.io/github/forks/uberduck-ai/uberduck-ml-dev) ![](https://img.shields.io/github/stars/uberduck-ai/uberduck-ml-dev) ![](https://img.shields.io/github/issues/uberduck-ai/uberduck-ml-dev)

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#🦆-Uberduck-TTS---" data-toc-modified-id="🦆-Uberduck-TTS----1"><span class="toc-item-num">1&nbsp;&nbsp;</span>🦆 Uberduck TTS <img src="https://img.shields.io/github/forks/uberduck-ai/uberduck-ml-dev" alt=""> <img src="https://img.shields.io/github/stars/uberduck-ai/uberduck-ml-dev" alt=""> <img src="https://img.shields.io/github/issues/uberduck-ai/uberduck-ml-dev" alt=""></a></span><ul class="toc-item"><li><span><a href="#Installation" data-toc-modified-id="Installation-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Installation</a></span></li><li><span><a href="#Development" data-toc-modified-id="Development-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Development</a></span><ul class="toc-item"><li><span><a href="#🚩-Testing" data-toc-modified-id="🚩-Testing-1.2.1"><span class="toc-item-num">1.2.1&nbsp;&nbsp;</span>🚩 Testing</a></span></li></ul></li><li><span><a href="#📦️-nbdev" data-toc-modified-id="📦️-nbdev-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>📦️ nbdev</a></span><ul class="toc-item"><li><span><a href="#🔧-Troubleshooting-Tips" data-toc-modified-id="🔧-Troubleshooting-Tips-1.3.1"><span class="toc-item-num">1.3.1&nbsp;&nbsp;</span>🔧 Troubleshooting Tips</a></span></li></ul></li><li><span><a href="#Overview" data-toc-modified-id="Overview-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Overview</a></span></li></ul></li></ul></div>

[**Uberduck**](https://uberduck.ai/) is a tool for fun and creativity with audio machine learning, currently focused on voice cloning and neural text-to-speech. This repository includes development tools to get started with creating your own speech synthesis model. For more information on the state of this repo, please see the [**Wiki**](https://github.com/uberduck-ai/uberduck-ml-dev/wiki).

## Overview

An overview of the subpackages in this library:

`models`: TTS model implementations. All models descend from `models.base.TTSModel`.

`trainer`: A trainer has logic for training a model.

`exec`: Contains entrypoint scripts for running training jobs. Executed via a command like
`python -m uberduck_ml_dev.exec.train_tacotron2 --your-args here`

## Installation

```
conda create -n 'uberduck-ml-dev' python=3.8
source activate uberduck-ml-dev
pip install git+https://github.com/uberduck-ai/uberduck-ml-dev.git
```

## Development

To start contributing, install the required development dependencies in a virtual environment:

```bash
pip install pre-commit black
```

Clone the repository:

```bash
git clone git@github.com:uberduck-ai/uberduck-ml-dev.git
```

Install required Git hooks:

```bash
pre-commit install
```

Install the library:

```bash
python setup.py develop
```

### 🚩 Testing

```bash
python -m pytest
```

### 🔧 Troubleshooting

- It is important for you to spell the name of your user and repo correctly in `settings.ini`. If you change the name of the repo, you have to make the appropriate changes in `settings.ini`
