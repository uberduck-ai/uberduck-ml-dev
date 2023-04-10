from io import BytesIO
import requests

import torch

from TTS.encoder.models.resnet import ResNetSpeakerEncoder

DEFAULT_AUDIO_CONFIG = {
    "fft_size": 512,
    "win_length": 400,
    "hop_length": 160,
    "frame_shift_ms": None,
    "frame_length_ms": None,
    "stft_pad_mode": "reflect",
    "sample_rate": 22050,
    "resample": False,
    "preemphasis": 0.97,
    "ref_level_db": 20,
    "do_sound_norm": False,
    "do_trim_silence": False,
    "trim_db": 60,
    "power": 1.5,
    "griffin_lim_iters": 60,
    "num_mels": 64,
    "mel_fmin": 0.0,
    "mel_fmax": 8000.0,
    "spec_gain": 20,
    "signal_norm": False,
    "min_level_db": -100,
    "symmetric_norm": False,
    "max_norm": 4.0,
    "clip_norm": False,
    "stats_path": None,
    "do_rms_norm": True,
    "db_level": -27.0,
}

RESNET_SE_MODEL_URL = "https://uberduck-models-us-west-2.s3.us-west-2.amazonaws.com/coqui/resnet_se.pth.tar"
RESNET_SE_CONFIG_URL = "https://uberduck-models-us-west-2.s3.us-west-2.amazonaws.com/coqui/resnet_se_config.json"


def get_pretrained_model():
    print("Getting model config...")
    response = requests.get(RESNET_SE_CONFIG_URL)
    resnet_config = response.json()
    model_params = resnet_config["model_params"]
    if "model_name" in model_params:
        del model_params["model_name"]
    audio_config = dict(resnet_config["audio"])
    audio_config["sample_rate"] = 22050
    model = ResNetSpeakerEncoder(**model_params, audio_config=audio_config)
    print("Loading pretrained model...")
    load_pretrained(model)
    print("Got pretrained model...")
    model.eval()
    return model


def load_pretrained(model):
    response = requests.get(RESNET_SE_MODEL_URL, stream=True)
    bio = BytesIO(response.content)
    loaded = torch.load(bio)
    model.load_state_dict(loaded["model"])
