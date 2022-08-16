import torch
import tempfile
import sys
import pytest
import gdown
from scipy.io import wavfile

from uberduck_ml_dev.models.tacotron2 import DEFAULTS as TACOTRON2_DEFAULTS
from uberduck_ml_dev.models.tacotron2 import Tacotron2
from uberduck_ml_dev.vendor.tfcompat.hparam import HParams

import numpy as np


# NOTE (Sam): move to Tacotron2 model and remove from Uberduck repo
def _load_tacotron_uninitialized(overrides=None):
    overrides = overrides or {}
    defaults = dict(**TACOTRON2_DEFAULTS.values())
    defaults.update(overrides)
    hparams = HParams(**defaults)
    return Tacotron2(hparams)


@pytest.fixture
def lj_speech_tacotron2():

    device = "cpu"
    # NOTE (Sam): A canonical LJ statedict used in our warm starting notebook
    url = "https://drive.google.com/uc?id=1qgEwtL53oFsdllM14FRZncgnARnAGInO"
    output_file = tempfile.NamedTemporaryFile()
    gdown.download(url, output_file.name, quiet=False)
    config_overrides = {}
    config_overrides["cudnn_enabled"] = device != "cpu"
    _model = _load_tacotron_uninitialized(config_overrides)
    checkpoint = torch.load(output_file.name, map_location=device)
    _model.from_pretrained(model_dict=checkpoint["state_dict"], device=device)

    return _model


@pytest.fixture
def sample_inference_spectrogram():
    url = "https://drive.google.com/uc?id=1IF8gXYg5x5zh9lkeKBqUg_Fc2pcNu05_"
    output_file = tempfile.NamedTemporaryFile()
    gdown.download(url, output_file.name, quiet=False)
    inference_spectrogram = torch.load(output_file.name)
    return inference_spectrogram


@pytest.fixture
def sample_inference_tf_spectrogram():
    # NOTE (Sam): made with abov at timestep 111,
    # text = "I, Sam, am a very bad boy."
    url = "https://drive.google.com/uc?id=1uJkp915fF4N3ozJHZcwPLLDk8k0sEC1E"
    output_file = tempfile.NamedTemporaryFile()
    gdown.download(url, output_file.name, quiet=False)
    inference_spectrogram = torch.load(output_file.name)
    return inference_spectrogram
