from io import BytesIO
import os
import requests
import json

from scipy.io.wavfile import read
import torch

# TODO (Sam): eliminate redundancy.
from .speaker.resnet import ResNetSpeakerEncoder

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


def get_pretrained_model(
    config_url=None, model_url=None, config_path=None, model_path=None
):
    assert not ((config_url is not None) and (config_path is not None))
    assert not ((model_url is not None) and (model_path is not None))

    if config_path is None:
        print("Getting model config...")
        if config_url is None:
            config_url = os.environ["RESNET_SE_CONFIG_URL"]
        response = requests.get(config_url)
        resnet_config = response.json()
    else:
        with open(config_path) as f:
            resnet_config = json.load(f)
    model_params = resnet_config["model_params"]
    if "model_name" in model_params:
        del model_params["model_name"]

    audio_config = dict(resnet_config["audio"])
    audio_config["sample_rate"] = 22050
    model = ResNetSpeakerEncoder(**model_params, audio_config=audio_config)
    print("Loading pretrained model...")
    load_pretrained(model, model_url=model_url, model_path=model_path)
    print("Got pretrained model...")
    model.eval()
    return model


def load_pretrained(model, model_url=None, model_path=None):
    assert not ((model_url is not None) and (model_path is not None))
    if model_path is not None:
        loaded = torch.load(model_path)
    else:
        if model_url is None:
            model_url = os.environ["RESNET_SE_MODEL_URL"]
        response = requests.get(model_url, stream=True)
        bio = BytesIO(response.content)
        loaded = torch.load(bio)
    model.load_state_dict(loaded["model"])


class ResNetSpeakerEncoderCallable:
    def __init__(self, model_path: str, config_path: str):
        print("initializing resnet speaker encoder")
        with open(config_path) as f:
            resnet_config = json.load(f)

        state_dict = torch.load(model_path)["model"]
        audio_config = dict(resnet_config["audio"])
        model_params = resnet_config["model_params"]
        if "model_name" in model_params:
            del model_params["model_name"]

        self.device = "cuda"
        self.model = ResNetSpeakerEncoder(**model_params, audio_config=audio_config)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.cuda()

    # NOTE (Sam): might have to accept bytes input for anyscale distributed data loading?
    def __call__(self, audiopaths):
        print("calling resnet speaker encoder")
        for audiopath in audiopaths:
            audio_data = read(audiopath)[1]
            datum = torch.FloatTensor(audio_data).unsqueeze(-1).t().cuda()
            # datum = torch.FloatTensor(audio_data).unsqueeze(-1).t()
            emb = self.model(datum)
            emb = emb.cpu().detach().numpy()
            yield {"audio_embedding": emb}
